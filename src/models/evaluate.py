# src/models/evaluate.py
# ==============================================================
# Avaliação dos modelos treinados.
# Gera relatório de métricas e análise de importância de features.
# ==============================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from config.settings import HORIZONS, MODELS_DIR, DATA_PROCESSED
from src.features.engineering import get_feature_columns


def _load_model(model_type: str, horizon: int):
    path = MODELS_DIR / f"{model_type}_h{horizon}d.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {path}\n"
            "Execute main.py --train primeiro."
        )
    return joblib.load(path)


def _load_feature_cols(horizon: int) -> list[str]:
    path = MODELS_DIR / f"feature_cols_h{horizon}d.joblib"
    return joblib.load(path)


def metrics_summary(train_results: dict) -> pd.DataFrame:
    """
    Consolida métricas walk-forward de todos os horizontes em um DataFrame.

    Parâmetros
    ----------
    train_results : saída de train_all()

    Retorna
    -------
    pd.DataFrame com colunas: horizonte, modelo, fold, RMSE, MAE, MAPE
    """
    rows = []
    for h, result in train_results.items():
        for model_name, cv_key in [("XGBoost", "xgb_cv_metrics"), ("RandomForest", "rf_cv_metrics")]:
            for fold_idx, metrics in enumerate(result[cv_key]):
                rows.append({
                    "horizonte_dias": h,
                    "modelo":         model_name,
                    "fold":           fold_idx + 1,
                    **metrics,
                })

    df = pd.DataFrame(rows)
    return df


def metrics_mean(train_results: dict) -> pd.DataFrame:
    """
    Média das métricas walk-forward por horizonte e modelo.
    Esta é a métrica de comparação principal entre modelos.
    """
    df = metrics_summary(train_results)
    summary = (
        df.groupby(["horizonte_dias", "modelo"])[["RMSE", "MAE", "MAPE"]]
        .mean()
        .round(4)
        .reset_index()
    )
    return summary


def feature_importance(
    horizon: int,
    model_type: str = "xgb",
    top_n: int = 20,
    save_plot: bool = True,
) -> pd.DataFrame:
    """
    Retorna e opcionalmente plota a importância de features.

    Parâmetros
    ----------
    horizon    : horizonte em dias
    model_type : 'xgb' ou 'rf'
    top_n      : número de features exibidas
    save_plot  : salva gráfico em data/processed/

    Retorna
    -------
    pd.DataFrame com colunas: feature, importance (ordenado decrescente)
    """
    model = _load_model(model_type, horizon)
    feature_cols = _load_feature_cols(horizon)

    importances = model.feature_importances_
    df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    if save_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(df["feature"][::-1], df["importance"][::-1])
        ax.set_title(f"Feature Importance — {model_type.upper()} h{horizon}d")
        ax.set_xlabel("Importance")
        ax.tick_params(axis="y", labelsize=8)
        plt.tight_layout()
        plot_path = DATA_PROCESSED / f"feature_importance_{model_type}_h{horizon}d.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[evaluate] Gráfico salvo: {plot_path}")

    return df


def print_report(train_results: dict) -> None:
    """Imprime relatório consolidado de métricas no console."""
    df = metrics_mean(train_results)
    print("\n" + "=" * 60)
    print("MÉTRICAS WALK-FORWARD (médias por horizonte)")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)

    # Indica o melhor modelo por horizonte com base em MAPE
    print("\nMelhor modelo por horizonte (menor MAPE médio):")
    for h in df["horizonte_dias"].unique():
        sub = df[df["horizonte_dias"] == h]
        best = sub.loc[sub["MAPE"].idxmin()]
        print(f"  h{h}d → {best['modelo']} (MAPE={best['MAPE']:.2f}%)")