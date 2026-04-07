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

import matplotlib
matplotlib.use("Agg")  # backend sem GUI — obrigatorio para uso em scripts no Windows
import argparse
import sys
import pandas as pd
import joblib
from pathlib import Path

# Garante que o diretório raiz está no path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import DATA_PROCESSED, HORIZONS
from src.collectors.cepea import save_cepea_xlsx
from src.processing.merger    import build_dataset
from src.processing.cleaner   import clean
from src.features.engineering import build_features
from src.models.train         import train_all
from src.models.evaluate      import (
    metrics_mean,
    feature_importance,
    print_report,
    plot_previsao_vs_real,
    plot_metricas_por_horizonte,
    plot_walk_forward_folds,
    plot_analise_residuos,
    plot_erro_mensal,
    add_baseline_to_results,
    export_metricas_csv,
)
from src.models.predict       import predict_latest


# ── Caminhos de cache ──────────────────────────────────────────
DATASET_CACHE = DATA_PROCESSED / "dataset_features.parquet"
TRAIN_RESULTS_CACHE = DATA_PROCESSED / "train_results.joblib"


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

    print("[main] Salvando Excel de inspecao do CEPEA...")
    save_cepea_xlsx()

    print("[main] Limpando dados...")
    df_clean = clean(df_raw)

    print("[main] Construindo features...")
    df_features = build_features(df_clean)

    # Remove attrs (ex.: holdout metadata com Timestamp) para evitar falha de serialização no parquet
    df_features.attrs.clear()
    df_features.to_parquet(DATASET_CACHE)
    print(f"[main] Dataset salvo em cache: {DATASET_CACHE}")

    return df_features


def step_train(df: pd.DataFrame) -> dict:
    """Etapa 2: treinamento walk-forward de todos os horizontes."""
    print("\n[main] Iniciando treinamento...")
    results = train_all(df)
    
    # Salva o dicionário de validação em disco
    joblib.dump(results, TRAIN_RESULTS_CACHE)
    print(f"[main] Treinamento concluído e cache salvo em: {TRAIN_RESULTS_CACHE}")
    
    return results


def step_evaluate(resultados_treinamento: dict, dataframe_features: pd.DataFrame = None) -> None:
    """Etapa 3: relatório de métricas e importância de features."""
    resultados_treinamento = add_baseline_to_results(resultados_treinamento, dataframe_features)
    horizontes_avaliados = sorted(resultados_treinamento.keys())

    print_report(resultados_treinamento)
    export_metricas_csv(resultados_treinamento)

    print("\n[main] Gerando gráficos de feature importance...")
    for horizonte_dias in horizontes_avaliados:
        for tipo_modelo in ["xgboost", "random_forest"]:
            try:
                feature_importance(horizonte_dias, tipo_modelo=tipo_modelo, save_plot=True)
            except FileNotFoundError as e:
                print(f"  [aviso] {e}")

    print("\n[main] Gerando gráficos de previsão vs. real por horizonte e modelo...")
    for tipo_modelo in ["xgboost", "random_forest", "baseline"]:
        for horizonte_dias in horizontes_avaliados:
            try:
                plot_previsao_vs_real(
                    dataframe_features,
                    resultados_treinamento,
                    tipo_modelo=tipo_modelo,
                    horizonte_dias=horizonte_dias,
                    data_inicio="2025-11-01",
                )
            except Exception as e:
                print(f"  [aviso] {e}")

    print("\n[main] Gerando gráfico de comparação de MAPE...")
    plot_metricas_por_horizonte(resultados_treinamento)

    print("\n[main] Gerando gráfico de estrutura walk-forward...")
    try:
        plot_walk_forward_folds(dataframe_features)
    except Exception as e:
        print(f"  [aviso] {e}")

    print("\n[main] Gerando gráficos de análise de resíduos...")
    for horizonte_dias in horizontes_avaliados:
        for tipo_modelo in ["xgboost", "random_forest", "baseline"]:
            try:
                # Usa dados a partir de 2020 para maior volume estatístico no scatter
                plot_analise_residuos(resultados_treinamento, horizonte_dias=horizonte_dias, tipo_modelo=tipo_modelo, data_inicio="2020-01-01")
            except Exception as e:
                print(f"  [aviso] Falha na análise de resíduos (h={horizonte_dias}, {tipo_modelo}): {e}")

    print("\n[main] Gerando gráficos mensais de sazonalidade de erro (MAE/MAPE)...")
    for horizonte_dias in horizontes_avaliados:
        for tipo_modelo in ["xgboost", "random_forest", "baseline"]:
            try:
                plot_erro_mensal(resultados_treinamento, horizonte_dias=horizonte_dias, tipo_modelo=tipo_modelo, data_inicio="2022-01-01")
            except Exception as e:
                print(f"  [aviso] Falha na análise mensal MAE/MAPE (h={horizonte_dias}, {tipo_modelo}): {e}")


def step_predict(df: pd.DataFrame) -> None:
    """Etapa 4: previsões a partir do último registro disponível."""
    print("\n[main] Gerando previsões...")
    preds = predict_latest(df)

    print("\n" + "=" * 60)
    print("PREVISOES -- PRECO DO BOI GORDO (R$/arroba, preco real)")
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
    
    if use_cache and TRAIN_RESULTS_CACHE.exists():
        print(f"[main] Carregando métricas de treinamento histórico salvas: {TRAIN_RESULTS_CACHE}")
        results = joblib.load(TRAIN_RESULTS_CACHE)
    else:
        results = step_train(df)
        
    step_evaluate(results, df)


def run_full(use_cache: bool = False) -> None:
    df = step_collect_and_process(use_cache=use_cache)
    results = step_train(df)
    step_evaluate(results, df)
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
