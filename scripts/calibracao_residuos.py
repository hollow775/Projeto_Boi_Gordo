"""
Gera diagnósticos e pós-calibração (isotônica) das previsões OOF,
corrigindo acentuação (UTF-8) e salvando gráficos/relatório.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# Configuração de fonte para acentos corretos
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
})

ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = ROOT / "data" / "processed" / "analise_residuos"
DIAG_DIR = OUT_BASE / "diagnostico"
CAL_DIR = OUT_BASE / "calibracao"
REPORT_PATH = OUT_BASE / "relatorio_calibracao.md"
DIAG_DIR.mkdir(parents=True, exist_ok=True)
CAL_DIR.mkdir(parents=True, exist_ok=True)

results = joblib.load(ROOT / "data" / "processed" / "train_results.joblib")

# Coluna-alvo (mantém compatibilidade) e rótulos
TARGET_COL = "y_true"
TARGET_LABEL = "arroba_boi_gordo (real)"
PRED_LABEL = "previsão"

bins = [0, 240, 255, 270, 285, 300, 315, 330, 400]
models = ["xgboost", "random_forest"]

metrics_before: list[dict] = []
metrics_after: list[dict] = []
bins_before: list[dict] = []
bins_after: list[dict] = []
notes: list[dict] = []


def mape(y, p) -> float:
    y, p = np.array(y), np.array(p)
    mask = (y != 0) & ~np.isnan(y) & ~np.isnan(p)
    return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])) * 100)


def mae(y, p) -> float:
    y, p = np.array(y), np.array(p)
    mask = ~np.isnan(y) & ~np.isnan(p)
    return float(np.mean(np.abs(y[mask] - p[mask])))


def rmse(y, p) -> float:
    y, p = np.array(y), np.array(p)
    mask = ~np.isnan(y) & ~np.isnan(p)
    return float(np.sqrt(np.mean((y[mask] - p[mask]) ** 2)))


for h, res in results.items():
    df = res["out_of_fold_dataframe"].copy()
    for model in models:
        y_true = df[TARGET_COL].values
        y_pred = df[f"previsao_{model}"].values

        # métricas antes
        m_b = dict(
            h=h,
            model=model,
            MAE=mae(y_true, y_pred),
            MAPE=mape(y_true, y_pred),
            RMSE=rmse(y_true, y_pred),
        )
        metrics_before.append(m_b)

        counts_true = pd.cut(df[TARGET_COL], bins=bins, right=False).value_counts().sort_index()
        counts_pred = pd.cut(pd.Series(y_pred), bins=bins, right=False).value_counts().sort_index()
        bins_before.append({"h": h, "model": model, "type": "true", **counts_true.to_dict()})
        bins_before.append({"h": h, "model": model, "type": "pred", **counts_pred.to_dict()})

        # gráficos diagnósticos
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df[TARGET_COL], bins=40, alpha=0.5, label=TARGET_LABEL, density=True, color="#555")
        ax.hist(y_pred, bins=40, alpha=0.5, label=f"{PRED_LABEL}_{model}", density=True, color="#1a6fad")
        ax.set_title(f"Histograma {TARGET_LABEL} vs previsão ({model}) h={h}")
        ax.set_xlabel("Preço (R$/@)")
        ax.set_ylabel("Densidade")
        ax.legend()
        fig.tight_layout()
        fig.savefig(DIAG_DIR / f"hist_{model}_h{h}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        hb = ax.hexbin(df[TARGET_COL], y_pred, gridsize=40, cmap="viridis", bins="log")
        ax.plot([df[TARGET_COL].min(), df[TARGET_COL].max()], [df[TARGET_COL].min(), df[TARGET_COL].max()], "r--", lw=1.5)
        ax.set_title(f"Dispersão (hexbin) {TARGET_LABEL} vs previsão ({model}) h={h}")
        ax.set_xlabel(f"{TARGET_LABEL} (R$/@)")
        ax.set_ylabel("Valor previsto (R$/@)")
        fig.colorbar(hb, ax=ax, label="log10(contagem)")
        fig.tight_layout()
        fig.savefig(DIAG_DIR / f"hexbin_{model}_h{h}.png", dpi=150)
        plt.close(fig)

        nbins = 12
        df_cal = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
        df_cal["bin"] = pd.qcut(df_cal["y_pred"], q=nbins, duplicates="drop")
        cal = df_cal.groupby("bin").agg(y_pred_media=("y_pred", "mean"), y_true_media=("y_true", "mean"))
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(cal["y_pred_media"], cal["y_true_media"], "o-", label="média por bin")
        lims = [
            min(cal.min().min(), df_cal["y_pred"].min(), df_cal["y_true"].min()),
            max(cal.max().max(), df_cal["y_pred"].max(), df_cal["y_true"].max()),
        ]
        ax.plot(lims, lims, "r--", label="y=x")
        ax.set_title(f"Curva de calibração (antes) {model} h={h}")
        ax.set_xlabel("Previsão média (bin)")
        ax.set_ylabel(f"{TARGET_LABEL} médio (bin)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(DIAG_DIR / f"calibracao_{model}_h{h}.png", dpi=150)
        plt.close(fig)

        # Calibração isotônica
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(y_pred, y_true)
        y_cal = iso.predict(y_pred)

        m_a = dict(
            h=h,
            model=model,
            MAE=mae(y_true, y_cal),
            MAPE=mape(y_true, y_cal),
            RMSE=rmse(y_true, y_cal),
        )
        metrics_after.append(m_a)

        counts_pred_after = pd.cut(pd.Series(y_cal), bins=bins, right=False).value_counts().sort_index()
        bins_after.append({"h": h, "model": model, "type": "pred_calibrated", **counts_pred_after.to_dict()})

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(y_pred, bins=40, alpha=0.4, label="pred_original", density=True, color="#1a6fad")
        ax.hist(y_cal, bins=40, alpha=0.4, label="pred_calibrado", density=True, color="#e06f00")
        ax.hist(df[TARGET_COL], bins=40, alpha=0.3, label=TARGET_LABEL, density=True, color="#555")
        ax.set_title(f"Histograma (pós calibração) {model} h={h}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(CAL_DIR / f"hist_cal_{model}_h{h}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        hb = ax.hexbin(df[TARGET_COL], y_cal, gridsize=40, cmap="magma", bins="log")
        ax.plot([df[TARGET_COL].min(), df[TARGET_COL].max()], [df[TARGET_COL].min(), df[TARGET_COL].max()], "r--", lw=1.5)
        ax.set_title(f"Dispersão pós-calibração (hexbin) {model} h={h}")
        ax.set_xlabel(f"{TARGET_LABEL} (R$/@)")
        ax.set_ylabel("Valor previsto calibrado (R$/@)")
        fig.colorbar(hb, ax=ax, label="log10(contagem)")
        fig.tight_layout()
        fig.savefig(CAL_DIR / f"hexbin_cal_{model}_h{h}.png", dpi=150)
        plt.close(fig)

        df_cal2 = pd.DataFrame({"y_true": y_true, "y_pred": y_cal}).dropna()
        df_cal2["bin"] = pd.qcut(df_cal2["y_pred"], q=nbins, duplicates="drop")
        cal2 = df_cal2.groupby("bin").agg(y_pred_media=("y_pred", "mean"), y_true_media=("y_true", "mean"))
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(cal2["y_pred_media"], cal2["y_true_media"], "o-", label="média por bin (pós)")
        ax.plot(lims, lims, "r--", label="y=x")
        ax.set_title(f"Curva de calibração (pós) {model} h={h}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(CAL_DIR / f"calibracao_pos_{model}_h{h}.png", dpi=150)
        plt.close(fig)

        tol = 2 if h in [1, 7] else 5
        delta = m_a["MAPE"] - m_b["MAPE"]
        status = "ok" if delta <= tol else "degradou"
        notes.append(
            {
                "h": h,
                "model": model,
                "MAPE_before": m_b["MAPE"],
                "MAPE_after": m_a["MAPE"],
                "delta": delta,
                "tolerance": tol,
                "status": status,
            }
        )

# salvar tabelas (UTF-8)
pd.DataFrame(metrics_before).to_csv(CAL_DIR / "metrics_before.csv", index=False, encoding="utf-8")
pd.DataFrame(metrics_after).to_csv(CAL_DIR / "metrics_after.csv", index=False, encoding="utf-8")
pd.DataFrame(bins_before).to_csv(CAL_DIR / "bins_before.csv", index=False, encoding="utf-8")
pd.DataFrame(bins_after).to_csv(CAL_DIR / "bins_after.csv", index=False, encoding="utf-8")
pd.DataFrame(notes).to_csv(CAL_DIR / "tolerance_notes.csv", index=False, encoding="utf-8")

mb = pd.DataFrame(metrics_before)
ma = pd.DataFrame(metrics_after)
nt = pd.DataFrame(notes)
lines: list[str] = []
lines.append("# Relatório de calibração (solo)\n")
lines.append("## Métricas antes (MAPE por horizonte/modelo)")
lines.append(mb.pivot(index="h", columns="model", values="MAPE").round(2).to_string())
lines.append("\n## Métricas depois (MAPE pós-calibração)")
lines.append(ma.pivot(index="h", columns="model", values="MAPE").round(2).to_string())
lines.append("\n## Notas de tolerância")
lines.append(nt.round(3).to_string(index=False))
lines.append("\nGráficos gerados em:")
lines.append(f"- {DIAG_DIR}")
lines.append(f"- {CAL_DIR}")
REPORT_PATH.write_text("\n\n".join(lines), encoding="utf-8")
print("Done. Report at", REPORT_PATH)
