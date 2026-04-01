# notebooks/shap_analysis.py
# ==============================================================
# Analise de interpretabilidade via SHAP values.
#
# SHAP (SHapley Additive exPlanations) decompoe cada previsao
# individualmente, mostrando quanto cada feature contribuiu
# para aquele resultado especifico.
#
# Instalacao da dependencia (apenas uma vez):
#   pip install shap
#
# Execucao:
#   python notebooks/shap_analysis.py
#   python notebooks/shap_analysis.py --horizonte 15 --modelo rf
# ==============================================================

import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import MODELS_DIR, DATA_PROCESSED

DATASET_CACHE = Path("data/processed/dataset_features.parquet")


def carregar_dados(horizonte: int):
    df           = pd.read_parquet(DATASET_CACHE)
    feature_cols = joblib.load(MODELS_DIR / f"feature_cols_h{horizonte}d.joblib")
    target_col   = f"target_h{horizonte}d"
    df_valid     = df.dropna(subset=[target_col]).copy()
    X_df         = df_valid[feature_cols].fillna(df_valid[feature_cols].median())
    return df_valid, X_df, feature_cols, target_col


def gerar_shap(horizonte: int, model_type: str, top_n: int, n_amostras: int):
    try:
        import shap
    except ImportError:
        print("[shap] Biblioteca nao instalada. Execute: pip install shap")
        return

    model_label = {"xgb": "XGBoost", "rf": "Random Forest"}.get(model_type, model_type)

    print(f"\n[shap] Carregando modelo {model_label} — horizonte {horizonte}d...")
    df_valid, X_df, feature_cols, target_col = carregar_dados(horizonte)
    model = joblib.load(MODELS_DIR / f"{model_type}_h{horizonte}d.joblib")

    # Usa amostra para nao sobrecarregar memoria (RF e mais lento)
    if len(X_df) > n_amostras:
        X_sample = X_df.sample(n=n_amostras, random_state=42)
        print(f"[shap] Usando amostra de {n_amostras} observacoes de {len(X_df)} totais.")
    else:
        X_sample = X_df

    print("[shap] Calculando SHAP values (pode levar alguns minutos para Random Forest)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # ── Grafico 1: Summary plot (beeswarm) ────────────────────
    print("[shap] Gerando summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_cols,
        max_display=top_n,
        show=False,
    )
    plt.title(
        f"SHAP Values — {model_label} | Horizonte: {horizonte} dias\n"
        "Impacto de cada variavel sobre a previsao",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out_dir = DATA_PROCESSED / "shap_importancia"
    out_dir.mkdir(exist_ok=True)
    path1 = out_dir / f"shap_summary_{model_type}_h{horizonte}d.png"
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shap] Grafico salvo: {path1}")

    # ── Grafico 2: Bar plot (importancia media absoluta) ──────
    print("[shap] Gerando bar plot de importancia media...")
    shap_mean = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame({
        "feature":    feature_cols,
        "shap_mean":  shap_mean,
    }).sort_values("shap_mean", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(imp_df["feature"][::-1], imp_df["shap_mean"][::-1], color="#9b2b9b", alpha=0.85)
    ax.set_title(
        f"Importancia Media (|SHAP|) — {model_label}\nHorizonte: {horizonte} dias | Top {top_n} features",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Valor SHAP medio absoluto (R$/arroba)", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    out_dir = DATA_PROCESSED / "shap_importancia"
    out_dir.mkdir(exist_ok=True)
    path2 = out_dir / f"shap_importancia_{model_type}_h{horizonte}d.png"
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"[shap] Grafico salvo: {path2}")

    # Exibe top 10 no terminal
    print(f"\nTop 10 features por |SHAP| medio — {model_label} h{horizonte}d:")
    print(f"{'Feature':<45} {'|SHAP| medio':>14}")
    print("-" * 62)
    for _, row in imp_df.head(10).iterrows():
        print(f"{row['feature']:<45} {row['shap_mean']:>14.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analise SHAP de interpretabilidade")
    parser.add_argument("--horizonte", type=int, default=1,
                        choices=[1, 15, 30, 60],
                        help="Horizonte de previsao em dias (default: 1)")
    parser.add_argument("--modelo", type=str, default="xgb",
                        choices=["xgb", "rf"],
                        help="Modelo a analisar: xgb ou rf (default: xgb)")
    parser.add_argument("--top-n", type=int, default=15,
                        help="Numero de features exibidas (default: 15)")
    parser.add_argument("--amostras", type=int, default=500,
                        help="Numero de amostras para calculo SHAP (default: 500)")
    args = parser.parse_args()

    gerar_shap(args.horizonte, args.modelo, args.top_n, args.amostras)

    print(f"\n[shap] Analise concluida. Graficos em: {DATA_PROCESSED}")


if __name__ == "__main__":
    main()