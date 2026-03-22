# main.py
# ==============================================================
# Orquestrador do pipeline completo.
#
# Modos de execução:
#   python main.py --train      coleta, processa e treina
#   python main.py --predict    gera previsões com modelos salvos
#   python main.py --evaluate   exibe métricas e importância de features
#   python main.py --full       executa train + evaluate + predict
#
# Pré-requisitos antes de executar:
#   1. Planilhas CEPEA salvas em data/raw/ (ver config/settings.py)
#   2. Arquivo ~/.cdsapirc configurado (Copernicus)
#   3. pip install -r requirements.txt
# ==============================================================

import argparse
import sys
import pandas as pd
from pathlib import Path

# Garante que o diretório raiz está no path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import DATA_PROCESSED, HORIZONS
from src.processing.merger    import build_dataset
from src.processing.cleaner   import clean
from src.features.engineering import build_features
from src.models.train         import train_all
from src.models.evaluate      import metrics_mean, feature_importance, print_report
from src.models.predict       import predict_latest


# ── Caminhos de cache ──────────────────────────────────────────
DATASET_CACHE = DATA_PROCESSED / "dataset_features.parquet"


def step_collect_and_process(use_cache: bool = False) -> pd.DataFrame:
    """
    Etapa 1: coleta, limpeza e feature engineering.
    Salva o resultado em parquet para evitar re-coleta desnecessária.
    """
    if use_cache and DATASET_CACHE.exists():
        print(f"[main] Carregando dataset do cache: {DATASET_CACHE}")
        return pd.read_parquet(DATASET_CACHE)

    print("[main] Coletando dados de todas as fontes...")
    df_raw = build_dataset()

    print("[main] Limpando dados...")
    df_clean = clean(df_raw)

    print("[main] Construindo features...")
    df_features = build_features(df_clean)

    df_features.to_parquet(DATASET_CACHE)
    print(f"[main] Dataset salvo em cache: {DATASET_CACHE}")

    return df_features


def step_train(df: pd.DataFrame) -> dict:
    """Etapa 2: treinamento walk-forward de todos os horizontes."""
    print("\n[main] Iniciando treinamento...")
    results = train_all(df)
    print("[main] Treinamento concluído.")
    return results


def step_evaluate(train_results: dict) -> None:
    """Etapa 3: relatório de métricas e importância de features."""
    print_report(train_results)

    print("\n[main] Gerando gráficos de feature importance...")
    for h in HORIZONS:
        for model_type in ["xgb", "rf"]:
            try:
                feature_importance(h, model_type=model_type, save_plot=True)
            except FileNotFoundError as e:
                print(f"  [aviso] {e}")


def step_predict(df: pd.DataFrame) -> None:
    """Etapa 4: previsões a partir do último registro disponível."""
    print("\n[main] Gerando previsões...")
    preds = predict_latest(df)

    print("\n" + "=" * 60)
    print("PREVISÕES — PREÇO DO BOI GORDO (R$/arroba, preço real)")
    print("=" * 60)
    print(preds.to_string(index=False))
    print("=" * 60)

    out_path = DATA_PROCESSED / "previsoes_latest.csv"
    preds.to_csv(out_path, index=False)
    print(f"\n[main] Previsões salvas em: {out_path}")


def run_train(use_cache: bool = False) -> dict:
    df = step_collect_and_process(use_cache=use_cache)
    return step_train(df), df


def run_predict(use_cache: bool = True) -> None:
    df = step_collect_and_process(use_cache=use_cache)
    step_predict(df)


def run_evaluate(use_cache: bool = True) -> None:
    df = step_collect_and_process(use_cache=use_cache)
    results = step_train(df)
    step_evaluate(results)


def run_full(use_cache: bool = False) -> None:
    df = step_collect_and_process(use_cache=use_cache)
    results = step_train(df)
    step_evaluate(results)
    step_predict(df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de previsão do preço do Boi Gordo (SP)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Coleta dados, processa e treina os modelos.",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Gera previsões usando modelos já treinados.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Treina e exibe métricas de validação walk-forward.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Executa coleta, treino, avaliação e previsão.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Força re-coleta dos dados mesmo se cache existir.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    use_cache = not args.no_cache

    if args.full:
        run_full(use_cache=use_cache)

    elif args.train:
        results, _ = run_train(use_cache=use_cache)
        print("\n[main] Use --evaluate para ver métricas detalhadas.")

    elif args.evaluate:
        run_evaluate(use_cache=use_cache)

    elif args.predict:
        run_predict(use_cache=use_cache)

    else:
        print(
            "Nenhum modo especificado.\n"
            "Use: python main.py --train | --predict | --evaluate | --full\n"
            "     adicione --no-cache para forçar re-coleta dos dados."
        )
        sys.exit(1)