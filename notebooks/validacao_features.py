# notebooks/validacao_features.py
# ==============================================================
# Validacao estatistica da importancia das features.
#
# Execucao (com ambiente virtual ativado):
#   python notebooks/validacao_features.py
#
# Ou para um horizonte especifico:
#   python notebooks/validacao_features.py --horizonte 15
# ==============================================================

import sys
import argparse
import warnings
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

# Garante que o diretorio raiz esta no path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import MODELS_DIR, DATA_PROCESSED

DATASET_CACHE = Path("data/processed/dataset_features.parquet")


def carregar_dados(horizonte: int):
    if not DATASET_CACHE.exists():
        raise FileNotFoundError(
            f"Cache nao encontrado: {DATASET_CACHE}\n"
            "Execute python main.py --train primeiro."
        )

    df           = pd.read_parquet(DATASET_CACHE)
    feature_cols = joblib.load(MODELS_DIR / f"feature_cols_h{horizonte}d.joblib")
    target_col   = f"target_h{horizonte}d"

    df_valid = df.dropna(subset=[target_col]).copy()
    X_df     = df_valid[feature_cols].fillna(df_valid[feature_cols].median())

    return df_valid, X_df, feature_cols, target_col


def analise_spearman(df_valid: pd.DataFrame, feature_cols: list, target_col: str, horizonte: int, top_n: int = 20):
    """
    Correlacao de Spearman entre cada feature e o target.
    Valida se a hierarquia de importancia do modelo e consistente
    com a correlacao estatistica direta.
    """
    print(f"\n{'='*60}")
    print(f"CORRELACAO DE SPEARMAN — Horizonte: {horizonte} dias")
    print(f"{'='*60}")
    print(f"{'Feature':<45} {'rho':>8}  {'p-valor':>12}  {'Significativo':>13}")
    print("-" * 82)

    resultados = []
    for col in feature_cols:
        serie = df_valid[col].fillna(df_valid[col].median())
        corr, pval = spearmanr(serie, df_valid[target_col])
        resultados.append({
            "feature":       col,
            "rho":           corr,
            "p_valor":       pval,
            "abs_rho":       abs(corr),
            "significativo": "Sim" if pval < 0.05 else "Nao",
        })

    df_corr = (
        pd.DataFrame(resultados)
        .sort_values("abs_rho", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    for _, row in df_corr.iterrows():
        print(f"{row['feature']:<45} {row['rho']:>8.4f}  {row['p_valor']:>12.2e}  {row['significativo']:>13}")

    # Grafico
    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = ["#2b9957" if r > 0 else "#c0392b" for r in df_corr["rho"]]
    ax.barh(df_corr["feature"][::-1], df_corr["rho"][::-1], color=colors[::-1], alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(
        f"Correlacao de Spearman com o Target\nHorizonte: {horizonte} dias | Top {top_n} features",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Coeficiente rho de Spearman", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    out_dir = DATA_PROCESSED / "spearman"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"spearman_h{horizonte}d.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n[validacao] Grafico salvo: {path}")

    return df_corr


def analise_permutation(X_df: pd.DataFrame, feature_cols: list, df_valid: pd.DataFrame,
                         target_col: str, horizonte: int, top_n: int = 20):
    """
    Permutation Importance: embaralha uma feature por vez e mede
    a queda no MAE. Mais confiavel que importancia nativa do XGBoost
    pois nao e influenciada por multicolinearidade.
    """
    print(f"\n{'='*60}")
    print(f"PERMUTATION IMPORTANCE — Horizonte: {horizonte} dias")
    print(f"{'='*60}")

    for model_type, label in [("xgb", "XGBoost"), ("rf", "Random Forest")]:
        model = joblib.load(MODELS_DIR / f"{model_type}_h{horizonte}d.joblib")
        y     = df_valid[target_col].values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = permutation_importance(
                model, X_df.values, y,
                n_repeats=10,
                random_state=42,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )

        imp_df = pd.DataFrame({
            "feature":    feature_cols,
            "importance": result.importances_mean,
            "std":        result.importances_std,
        }).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

        print(f"\n{label}:")
        print(f"{'Feature':<45} {'Importancia':>12}  {'Desvio Pad':>11}")
        print("-" * 72)
        for _, row in imp_df.iterrows():
            print(f"{row['feature']:<45} {row['importance']:>12.4f}  {row['std']:>11.4f}")

        # Grafico
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(
            imp_df["feature"][::-1],
            imp_df["importance"][::-1],
            xerr=imp_df["std"][::-1],
            color="#1a6fad", alpha=0.85, capsize=3,
        )
        ax.set_title(
            f"Permutation Importance — {label}\nHorizonte: {horizonte} dias | Top {top_n} features",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Queda no MAE (media de 10 repeticoes)", fontsize=11)
        ax.tick_params(axis="y", labelsize=8)
        plt.tight_layout()
        out_dir = DATA_PROCESSED / "permutation_importance"
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"permutation_importance_{model_type}_h{horizonte}d.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[validacao] Grafico salvo: {path}")

    return imp_df


def main():
    parser = argparse.ArgumentParser(description="Validacao estatistica das features")
    parser.add_argument("--horizonte", type=int, default=1,
                        choices=[1, 15, 30, 60],
                        help="Horizonte de previsao em dias (default: 1)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Numero de features exibidas (default: 20)")
    args = parser.parse_args()

    print(f"[validacao] Carregando dados para horizonte {args.horizonte}d...")
    df_valid, X_df, feature_cols, target_col = carregar_dados(args.horizonte)
    print(f"[validacao] {len(df_valid)} observacoes | {len(feature_cols)} features")

    analise_spearman(df_valid, feature_cols, target_col, args.horizonte, args.top_n)
    analise_permutation(X_df, feature_cols, df_valid, target_col, args.horizonte, args.top_n)

    print(f"\n[validacao] Analise concluida. Graficos em: {DATA_PROCESSED}")


if __name__ == "__main__":
    main()