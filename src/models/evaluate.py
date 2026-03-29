# src/models/evaluate.py
# ==============================================================
# Avaliação dos modelos treinados.
# Gera relatório de métricas e análise de importância de features.
# ==============================================================

import matplotlib
matplotlib.use("Agg")  # backend sem GUI
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

    MODEL_NAMES = {"xgb": "XGBoost", "rf": "Random Forest"}
    model_label = MODEL_NAMES.get(model_type, model_type.upper())

    if save_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(df["feature"][::-1], df["importance"][::-1])
        ax.set_title(
            f"Importância das Variáveis — {model_label} | Horizonte: {horizon} dias",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Importância", fontsize=11)
        ax.set_ylabel("Variável", fontsize=11)
        ax.tick_params(axis="y", labelsize=8)
        plt.tight_layout()
        plot_path = DATA_PROCESSED / f"importancia_variaveis_{model_type}_h{horizon}d.png"
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


# ── Paleta de cores do projeto ─────────────────────────────────
HORIZON_COLORS = {
    1:  "#2b9957",   # verde
    15: "#e06f00",   # laranja
    30: "#1a6fad",   # azul
    60: "#9b2b9b",   # roxo
}

HORIZON_LABELS = {
    1:  "1 dia",
    15: "15 dias",
    30: "30 dias",
    60: "60 dias",
}


def plot_previsao_vs_real(
    df: pd.DataFrame,
    model_type: str = "xgb",
    data_inicio: str = "2025-11-01",
    save_plot: bool = True,
) -> None:
    """
    Gera grafico de previsao vs. valor real para todos os horizontes.

    Cada horizonte e plotado como uma linha tracejada colorida.
    O valor real do boi gordo e plotado em preto como referencia.

    Parametros
    ----------
    df          : DataFrame com features (saida de build_features)
    model_type  : 'xgb' ou 'rf'
    data_inicio : data de inicio do grafico (formato YYYY-MM-DD)
    save_plot   : salva grafico em data/processed/
    """
    from src.models.predict import _load_model, _load_feature_cols

    MODEL_NAMES = {"xgb": "XGBoost", "rf": "Random Forest"}
    model_label = MODEL_NAMES.get(model_type, model_type.upper())

    # Seleciona a partir da data de inicio
    df_plot = df.loc[data_inicio:].copy()

    if df_plot.empty:
        print(f"[evaluate] Sem dados a partir de {data_inicio}. Verifique DATE_RANGE em settings.py.")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    # Linha do valor real
    ax.plot(
        df_plot.index,
        df_plot["preco_boi_gordo"],
        color="#1a1a1a",
        linewidth=2,
        label="Valor Real",
        zorder=5,
    )

    # Uma linha por horizonte
    for h in HORIZONS:
        feature_cols = _load_feature_cols(h)
        model        = _load_model(model_type, h)

        X_df = df_plot[feature_cols].copy()
        X_df = X_df.fillna(X_df.median())
        preds = model.predict(X_df.values)

        # Desloca as previsoes para o instante correto:
        # a previsao feita em t para t+h deve ser plotada em t+h
        pred_series = pd.Series(preds, index=df_plot.index)
        pred_series.index = pred_series.index + pd.Timedelta(days=h)

        # Alinha com o periodo do grafico
        pred_series = pred_series[pred_series.index <= df_plot.index[-1] + pd.Timedelta(days=h)]

        ax.plot(
            pred_series.index,
            pred_series.values,
            color=HORIZON_COLORS[h],
            linewidth=1.4,
            linestyle="--",
            alpha=0.85,
            label=f"Previsão {HORIZON_LABELS[h]}",
        )

    ax.set_title(
        f"Previsão vs. Valor Real — {model_label}\n"
        f"Período: {data_inicio} a {df_plot.index[-1].strftime('%d/%m/%Y')}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Data", fontsize=11)
    ax.set_ylabel("Preço (R$/arroba, preço real)", fontsize=11)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()

    if save_plot:
        plot_path = DATA_PROCESSED / f"previsao_vs_real_{model_type}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[evaluate] Gráfico salvo: {plot_path}")
    else:
        plt.show()


def plot_metricas_por_horizonte(
    train_results: dict,
    save_plot: bool = True,
) -> None:
    """
    Grafico de barras comparando MAPE medio de XGBoost e Random Forest
    por horizonte de previsao.
    """
    df = metrics_mean(train_results)

    horizontes  = sorted(df["horizonte_dias"].unique())
    x           = np.arange(len(horizontes))
    largura     = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    xgb_mapes = [
        df[(df["horizonte_dias"] == h) & (df["modelo"] == "XGBoost")]["MAPE"].values[0]
        for h in horizontes
    ]
    rf_mapes = [
        df[(df["horizonte_dias"] == h) & (df["modelo"] == "RandomForest")]["MAPE"].values[0]
        for h in horizontes
    ]

    bars_xgb = ax.bar(x - largura/2, xgb_mapes, largura, label="XGBoost",      color="#2b9957", alpha=0.9)
    bars_rf  = ax.bar(x + largura/2, rf_mapes,  largura, label="Random Forest", color="#e06f00", alpha=0.9)

    # Rotulos de valor nas barras
    for bar in bars_xgb:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{bar.get_height():.2f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    for bar in bars_rf:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{bar.get_height():.2f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_title(
        "MAPE Médio por Horizonte de Previsão\nValidação Walk-Forward (5 folds)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Horizonte de Previsão (dias)", fontsize=11)
    ax.set_ylabel("MAPE Médio (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} dias" for h in horizontes], fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max(max(xgb_mapes), max(rf_mapes)) * 1.2)

    plt.tight_layout()

    if save_plot:
        plot_path = DATA_PROCESSED / "comparacao_mape_modelos.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[evaluate] Gráfico salvo: {plot_path}")
    else:
        plt.show()


def plot_walk_forward_folds(
    df: pd.DataFrame,
    save_plot: bool = True,
) -> None:
    """
    Gera grafico visual dos 5 folds do walk-forward com datas reais.

    Cada fold e representado por uma faixa horizontal:
        - Azul escuro : periodo de treino (expanding window)
        - Cor unica   : periodo de teste de cada fold

    A legenda exibe as datas exatas de inicio e fim de cada periodo.
    """
    from src.models.train import MIN_TRAIN_DAYS, N_FOLDS

    # Usa apenas linhas sem NaN na variavel alvo principal
    df_valid = df.dropna(subset=["preco_boi_gordo"])
    datas    = df_valid.index
    n        = len(datas)

    test_size = (n - MIN_TRAIN_DAYS) // N_FOLDS

    # Cores distintas para cada fold de teste
    FOLD_COLORS = ["#2b9957", "#e06f00", "#1a6fad", "#9b2b9b", "#c0392b"]

    fig, ax = plt.subplots(figsize=(14, 5))

    for i in range(N_FOLDS):
        train_end = MIN_TRAIN_DAYS + i * test_size
        test_end  = min(train_end + test_size, n)
        fold_num  = i + 1

        # Datas reais
        treino_inicio = datas[0]
        treino_fim    = datas[train_end - 1]
        teste_inicio  = datas[train_end]
        teste_fim     = datas[test_end - 1]

        y_pos = N_FOLDS - i  # fold 1 no topo

        # Barra de treino
        ax.barh(
            y=y_pos,
            width=(treino_fim - treino_inicio).days,
            left=treino_inicio,
            height=0.5,
            color="#1a3a5c",
            alpha=0.75,
            label="Treino" if i == 0 else "",
        )

        # Barra de teste
        ax.barh(
            y=y_pos,
            width=(teste_fim - teste_inicio).days,
            left=teste_inicio,
            height=0.5,
            color=FOLD_COLORS[i],
            alpha=0.9,
            label=f"Fold {fold_num} — Teste: {teste_inicio.strftime('%d/%m/%Y')} a {teste_fim.strftime('%d/%m/%Y')}",
        )

        # Rotulo do fold no eixo y
        ax.text(
            treino_inicio - pd.Timedelta(days=30),
            y_pos,
            f"Fold {fold_num}",
            va="center", ha="right",
            fontsize=10, fontweight="bold",
        )

        # Rotulo de data de inicio do treino (apenas fold 1)
        if i == 0:
            ax.text(
                treino_inicio + pd.Timedelta(days=15),
                y_pos + 0.32,
                f"Treino inicia: {treino_inicio.strftime('%d/%m/%Y')}",
                va="bottom", ha="left",
                fontsize=8, color="#1a3a5c",
            )

        # Rotulo de data dentro da barra de teste
        ax.text(
            teste_inicio + (teste_fim - teste_inicio) / 2,
            y_pos,
            f"{teste_inicio.strftime('%b/%Y')} – {teste_fim.strftime('%b/%Y')}",
            va="center", ha="center",
            fontsize=8.5, color="white", fontweight="bold",
        )

    ax.set_title(
        "Estrutura de Validação Walk-Forward — Expanding Window"
        f"{N_FOLDS} Folds | Treino mínimo: {MIN_TRAIN_DAYS} dias",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Data", fontsize=11)
    ax.set_yticks([])
    ax.xaxis_date()
    fig.autofmt_xdate(rotation=30)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_xlim(
        datas[0] - pd.Timedelta(days=60),
        datas[-1] + pd.Timedelta(days=60),
    )
    ax.set_ylim(0.5, N_FOLDS + 0.8)

    plt.tight_layout()

    if save_plot:
        plot_path = DATA_PROCESSED / "walk_forward_folds.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[evaluate] Gráfico salvo: {plot_path}")
    else:
        plt.show()