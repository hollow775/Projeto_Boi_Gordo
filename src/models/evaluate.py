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


def metrics_summary(resultados_treinamento: dict) -> pd.DataFrame:
    """
    Consolida as métricas walk-forward num único DataFrame tidy.

    Retorna
    -------
    pd.DataFrame com colunas: horizonte, modelo, fold, RMSE, MAE, MAPE
    """
    rows = []
    for horizonte_dias, result in resultados_treinamento.items():
        for model_name, cv_key in [("XGBoost", "metricas_cv_xgboost"), ("RandomForest", "metricas_cv_random_forest")]:
            for fold_idx, metrics in enumerate(result[cv_key]):
                rows.append({
                    "horizonte_dias": horizonte_dias,
                    "modelo":         model_name,
                    "fold":           fold_idx + 1,
                    **metrics,
                })

    dataframe_metricas = pd.DataFrame(rows)
    return dataframe_metricas


def metrics_mean(resultados_treinamento: dict) -> pd.DataFrame:
    """
    Média das métricas walk-forward por horizonte e modelo.
    Esta é a métrica de comparação principal entre modelos.
    """
    dataframe_metricas = metrics_summary(resultados_treinamento)
    summary = (
        dataframe_metricas.groupby(["horizonte_dias", "modelo"])[["RMSE", "MAE", "MAPE"]]
        .mean()
        .round(4)
        .reset_index()
    )
    return summary


def feature_importance(
    horizonte_dias: int,
    tipo_modelo: str = "xgboost",
    top_n: int = 20,
    save_plot: bool = True,
) -> pd.DataFrame:
    """
    Retorna e opcionalmente plota a importância de features.

    Parâmetros
    ----------
    horizonte_dias    : horizonte em dias
    tipo_modelo : 'xgboost' ou 'random_forest'
    top_n      : número de features exibidas
    save_plot  : salva gráfico em data/processed/

    Retorna
    -------
    pd.DataFrame com colunas: feature, importance (ordenado decrescente)
    """
    model = _load_model(tipo_modelo, horizonte_dias)
    feature_cols = _load_feature_cols(horizonte_dias)

    importances = model.feature_importances_
    dataframe_importances = pd.DataFrame({
        "feature":    feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    MODEL_NAMES = {"xgboost": "XGBoost", "random_forest": "Random Forest"}
    model_label = MODEL_NAMES.get(tipo_modelo, tipo_modelo.upper())

    if save_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(dataframe_importances["feature"][::-1], dataframe_importances["importance"][::-1])
        ax.set_title(
            f"Importância das Variáveis — {model_label} | Horizonte: {horizonte_dias} dias",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Importância", fontsize=11)
        ax.set_ylabel("Variável", fontsize=11)
        ax.tick_params(axis="y", labelsize=8)
        plt.tight_layout()
        out_dir = DATA_PROCESSED / "importancia_variaveis"
        out_dir.mkdir(exist_ok=True)
        plot_path = out_dir / f"importancia_variaveis_{tipo_modelo}_h{horizonte_dias}d.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[evaluate] Gráfico salvo: {plot_path}")

    return dataframe_importances


def print_report(resultados_treinamento: dict) -> None:
    """Imprime relatório consolidado de métricas no console."""
    dataframe_medias = metrics_mean(resultados_treinamento)
    print("\n" + "=" * 60)
    print("MÉTRICAS WALK-FORWARD (médias por horizonte)")
    print("=" * 60)
    print(dataframe_medias.to_string(index=False))
    print("=" * 60)

    # Indica o melhor modelo por horizonte com base em MAPE
    print("\nMelhor modelo por horizonte (menor MAPE médio):")
    for horizonte_dias in dataframe_medias["horizonte_dias"].unique():
        sub = dataframe_medias[dataframe_medias["horizonte_dias"] == horizonte_dias]
        best = sub.loc[sub["MAPE"].idxmin()]
        print(f"  h{horizonte_dias}d -> {best['modelo']} (MAPE={best['MAPE']:.2f}%)")


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
    dataframe_dados: pd.DataFrame,
    resultados_treinamento: dict,
    tipo_modelo: str = "xgboost",
    data_inicio: str = "2025-11-01",
    save_plot: bool = True,
) -> None:
    """
    Gera grafico de previsao vs. valor real para todos os horizontes.

    Cada horizonte e plotado como uma linha tracejada colorida.
    O valor real do boi gordo e plotado em preto como referencia.

    Parametros
    ----------
    dataframe_dados : DataFrame original
    tipo_modelo     : 'xgboost' ou 'random_forest'
    data_inicio     : data de inicio do grafico (formato YYYY-MM-DD)
    save_plot       : salva grafico em data/processed/
    """
    from src.models.predict import _load_model, _load_feature_cols

    MODEL_NAMES = {"xgboost": "XGBoost", "random_forest": "Random Forest"}
    model_label = MODEL_NAMES.get(tipo_modelo, tipo_modelo.upper())

    # Seleciona a partir da data de inicio
    df_plot = dataframe_dados.loc[data_inicio:].copy()

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
    for horizonte_dias in HORIZONS:
        out_of_fold_dataframe = resultados_treinamento[horizonte_dias]["out_of_fold_dataframe"]
        
        coluna_previsao = f"previsao_{tipo_modelo}"
        previsoes = out_of_fold_dataframe[coluna_previsao].copy()

        # Desloca as previsoes para o instante correto:
        # a previsao feita em t para t+horizonte_dias deve ser plotada em t+horizonte_dias
        serie_previsoes = pd.Series(previsoes.values, index=out_of_fold_dataframe.index)
        serie_previsoes.index = serie_previsoes.index + pd.Timedelta(days=horizonte_dias)

        # Alinha com o periodo do grafico: Cortar datas OOF antigas
        serie_previsoes = serie_previsoes[serie_previsoes.index >= pd.to_datetime(data_inicio)]
        serie_previsoes = serie_previsoes[serie_previsoes.index <= df_plot.index[-1] + pd.Timedelta(days=horizonte_dias)]

        ax.plot(
            serie_previsoes.index,
            serie_previsoes.values,
            color=HORIZON_COLORS[horizonte_dias],
            linewidth=1.4,
            linestyle="--",
            alpha=0.85,
            label=f"Previsão {HORIZON_LABELS[horizonte_dias]}",
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
        out_dir = DATA_PROCESSED / "previsao_vs_real"
        out_dir.mkdir(exist_ok=True)
        plot_path = out_dir / f"previsao_vs_real_{tipo_modelo}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[evaluate] Gráfico salvo: {plot_path}")
    else:
        plt.show()


def plot_metricas_por_horizonte(
    resultados_treinamento: dict,
    save_plot: bool = True,
) -> None:
    """
    Grafico de barras comparando MAPE medio de XGBoost e Random Forest
    por horizonte de previsao.
    """
    dataframe_medias = metrics_mean(resultados_treinamento)

    horizontes  = sorted(dataframe_medias["horizonte_dias"].unique())
    x           = np.arange(len(horizontes))
    largura     = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    mapes_xgboost = [
        dataframe_medias[(dataframe_medias["horizonte_dias"] == horizonte_dias) & (dataframe_medias["modelo"] == "XGBoost")]["MAPE"].values[0]
        for horizonte_dias in horizontes
    ]
    mapes_random_forest = [
        dataframe_medias[(dataframe_medias["horizonte_dias"] == horizonte_dias) & (dataframe_medias["modelo"] == "RandomForest")]["MAPE"].values[0]
        for horizonte_dias in horizontes
    ]

    bars_xgb = ax.bar(x - largura/2, mapes_xgboost, largura, label="XGBoost",      color="#2b9957", alpha=0.9)
    bars_rf  = ax.bar(x + largura/2, mapes_random_forest,  largura, label="Random Forest", color="#e06f00", alpha=0.9)

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
    ax.set_ylim(0, max(max(mapes_xgboost), max(mapes_random_forest)) * 1.2)

    plt.tight_layout()

    if save_plot:
        out_dir = DATA_PROCESSED / "metricas"
        out_dir.mkdir(exist_ok=True)
        plot_path = out_dir / "comparacao_mape_modelos.png"
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

    y_ticks = []
    y_ticklabels = []

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
        y_ticks.append(y_pos)
        y_ticklabels.append(f"Fold {fold_num}")

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
        MESES_PT = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun", 7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"}
        lbl_inicio = f"{MESES_PT[teste_inicio.month]}/{teste_inicio.year}"
        lbl_fim    = f"{MESES_PT[teste_fim.month]}/{teste_fim.year}"
        ax.text(
            teste_inicio + (teste_fim - teste_inicio) / 2,
            y_pos,
            f"{lbl_inicio} – {lbl_fim}",
            va="center", ha="center",
            fontsize=8.5, color="white", fontweight="bold",
        )

    ax.set_title(
        "Estrutura de validação walk-forward",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Data", fontsize=11)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels, fontsize=10, fontweight="bold")
    ax.xaxis_date()
    fig.autofmt_xdate(rotation=30)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_xlim(
        datas[0] - pd.Timedelta(days=60),
        datas[-1] + pd.Timedelta(days=60),
    )
    ax.set_ylim(0.5, N_FOLDS + 0.8)

    plt.tight_layout()

    if save_plot:
        out_dir = DATA_PROCESSED / "walk_forward"
        out_dir.mkdir(exist_ok=True)
        plot_path = out_dir / "walk_forward_folds.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[evaluate] Gráfico salvo: {plot_path}")
    else:
        plt.show()


def plot_analise_residuos(
    resultados_treinamento: dict,
    horizonte_dias: int = 1,
    tipo_modelo: str = "xgboost",
    data_inicio: str = "2024-01-01",
    save_plot: bool = True,
) -> None:
    """
    Gera um painel com análise gráfica de resíduos para um modelo e horizonte específicos.
    
    O painel inclui:
    1. Gráfico de dispersão Valor Real vs. Previsão
    2. Distribuição (Histograma) dos resíduos com curva normal
    3. Comportamento dos resíduos ao longo do tempo
    Gera as visualizações de análise de comportamento residual do modelo.
    """
    import scipy.stats as stats

    MODEL_NAMES = {"xgboost": "XGBoost", "random_forest": "Random Forest"}
    model_label = MODEL_NAMES.get(tipo_modelo, tipo_modelo.upper())

    # Carrega base OOF e seleciona a partir da data de inicio
    out_of_fold_dataframe = resultados_treinamento[horizonte_dias]["out_of_fold_dataframe"]
    try:
        df_valid = out_of_fold_dataframe.loc[data_inicio:].copy()
    except KeyError:
        df_valid = out_of_fold_dataframe.copy()

    if df_valid.empty:
        print(f"[evaluate] Sem dados válidos (OOF) para a data a partir de {data_inicio}.")
        return

    y_true = df_valid["y_true"].values
    y_pred = df_valid[f"previsao_{tipo_modelo}"].values

    residuos = y_true - y_pred

    # Criação do diretório específico para despoluir a pasta processed
    out_dir = DATA_PROCESSED / "analise_residuos"
    if save_plot:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # 1. Scatter Real vs Previsto
    # ==========================================
    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
    v_min = min(y_true.min(), y_pred.min())
    v_max = max(y_true.max(), y_pred.max())

    ax_scatter.scatter(y_true, y_pred, alpha=0.6, color="#1a6fad", edgecolors="white", s=45)
    ax_scatter.plot([v_min, v_max], [v_min, v_max], "r--", lw=2, label="Previsão Perfeita (y=x)")
    ax_scatter.set_title(f"Valor Real vs. Previsão ({model_label})\nHorizonte: {horizonte_dias} dias", fontsize=12, fontweight="bold")
    ax_scatter.set_xlabel("Valor Real Observado (R$/arroba)", fontsize=10)
    ax_scatter.set_ylabel("Valor Previsto pelo Modelo (R$/arroba)", fontsize=10)
    ax_scatter.legend()
    ax_scatter.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_plot:
        scatter_path = out_dir / f"scatter_previsao_vs_real_{tipo_modelo}_h{horizonte_dias}d.png"
        plt.savefig(scatter_path, dpi=150)
        plt.close(fig_scatter)
    else:
        plt.show()

    # ==========================================
    # 2. Histograma dos Resíduos
    # ==========================================
    fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
    ax_hist.hist(residuos, bins=30, density=True, alpha=0.7, color="#e06f00", edgecolor="black")
    
    xmin, xmax = ax_hist.get_xlim()
    x_pdf = np.linspace(xmin, xmax, 100)
    p_pdf = stats.norm.pdf(x_pdf, np.mean(residuos), np.std(residuos))
    ax_hist.plot(x_pdf, p_pdf, "k", linewidth=2, label="Curva Normal Teórica")
    
    ax_hist.axvline(x=0, color='r', linestyle='--', lw=2, label='Erro Zero')
    ax_hist.set_title(f"Distribuição dos Resíduos ({model_label})\nHorizonte: {horizonte_dias} dias", fontsize=12, fontweight="bold")
    ax_hist.set_xlabel("Resíduo (Real - Previsto)", fontsize=10)
    ax_hist.set_ylabel("Densidade", fontsize=10)
    ax_hist.legend()
    ax_hist.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_plot:
        hist_path = out_dir / f"distribuicao_residuos_{tipo_modelo}_h{horizonte_dias}d.png"
        plt.savefig(hist_path, dpi=150)
        plt.close(fig_hist)
    else:
        plt.show()

    # ==========================================
    # 3. Resíduos ao longo do tempo
    # ==========================================
    fig_time, ax_time = plt.subplots(figsize=(10, 5))
    ax_time.plot(df_valid.index, residuos, color="#2b9957", lw=1.5, alpha=0.85)
    ax_time.axhline(y=0, color="r", linestyle="--", lw=2, label="Erro Linha Base (Zero)")
    ax_time.set_title(f"Comportamento Residual Histórico ({model_label})\nHorizonte: {horizonte_dias} dias", fontsize=12, fontweight="bold")
    ax_time.set_xlabel("Data", fontsize=10)
    ax_time.set_ylabel("Erro (R$ na arroba)", fontsize=10)
    ax_time.legend(loc="upper right")
    ax_time.grid(True, linestyle="--", alpha=0.4)
    fig_time.autofmt_xdate(rotation=30)
    plt.tight_layout()

    if save_plot:
        time_path = out_dir / f"historico_residuos_{tipo_modelo}_h{horizonte_dias}d.png"
        plt.savefig(time_path, dpi=150)
        plt.close(fig_time)
        print(f"[evaluate] 3 Gráficos de resíduos isolados salvos na pasta: {out_dir.name}/")
    else:
        plt.show()


def plot_erro_mensal(
    resultados_treinamento: dict,
    horizonte_dias: int = 1,
    tipo_modelo: str = "xgboost",
    data_inicio: str = "2022-01-01",
    save_plot: bool = True,
) -> None:
    """
    Constrói um gráfico de barras que demonstra a oscilação do MAPE e MAE 
    historicamente de acordo com os Mêses do Ano. 
    Ideal para identificar a perda de previsibilidade na entressafra e fatores macro.
    """
    MODEL_NAMES = {"xgboost": "XGBoost", "random_forest": "Random Forest"}
    model_label = MODEL_NAMES.get(tipo_modelo, tipo_modelo.upper())

    out_of_fold_dataframe = resultados_treinamento[horizonte_dias]["out_of_fold_dataframe"]
    try:
        df_valid = out_of_fold_dataframe.loc[data_inicio:].copy()
    except KeyError:
        df_valid = out_of_fold_dataframe.copy()

    if df_valid.empty:
        print(f"[evaluate] Sem dados válidos (OOF) ao agrupar meses a partir de {data_inicio}.")
        return

    # Certifica-se de que o index é data para extrair o mês
    if not isinstance(df_valid.index, pd.DatetimeIndex):
        print("[evaluate] O índice do DataFrame não é datetime. Faltou indexar a data para esta análise mensal.")
        return

    y_true = df_valid["y_true"].values
    y_pred = df_valid[f"previsao_{tipo_modelo}"].values

    # Cria DataFrame com as métricas individuais por linha
    df_residuos = pd.DataFrame({
        "erro_absoluto": np.abs(y_true - y_pred),
        "mape": (np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8)) * 100
    }, index=df_valid.index)

    # Extrai o número do mês de 1 a 12
    df_residuos["mes"] = df_residuos.index.month
    
    # Agrupa tirando a média das métricas para aquele mês historicamente
    agrupado = df_residuos.groupby("mes")[["erro_absoluto", "mape"]].mean().reset_index()

    # Mapeamento e alinhamento de 1 a 12
    meses_str = {
        1:"Jan", 2:"Fev", 3:"Mar", 4:"Abr", 5:"Mai", 6:"Jun",
        7:"Jul", 8:"Ago", 9:"Set", 10:"Out", 11:"Nov", 12:"Dez"
    }
    
    todos_meses = pd.DataFrame({"mes": list(range(1, 13))})
    agrupado = pd.merge(todos_meses, agrupado, on="mes", how="left")
    
    x_labels = [meses_str[m] for m in agrupado["mes"]]
    mae_vals = agrupado["erro_absoluto"].values
    mape_vals = agrupado["mape"].values

    out_dir = DATA_PROCESSED / "erro_sazonal"
    if save_plot:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # Gráfico A: MAE Mensal (Gráfico Separado)
    # ==========================================
    fig_mae, ax1 = plt.subplots(figsize=(10, 5))
    bars1 = ax1.bar(x_labels, mae_vals, color="#a37638", alpha=0.9, edgecolor="black")
    ax1.set_title(f"MAE por Mês — {model_label}\nHorizonte: {horizonte_dias} dias", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Mês do Ano", fontsize=11)
    ax1.set_ylabel("MAE (R$/arroba)", fontsize=11)
    ax1.set_ylim(0, max(mae_vals) * 1.15)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars1:
        h_val = bar.get_height()
        if not np.isnan(h_val) and h_val > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, h_val + (max(mae_vals)*0.02), 
                     f"R${h_val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()

    if save_plot:
        mae_path = out_dir / f"erro_sazonal_mae_{tipo_modelo}_h{horizonte_dias}d.png"
        plt.savefig(mae_path, dpi=150)
        plt.close(fig_mae)
    else:
        plt.show()

    # ==========================================
    # Gráfico B: MAPE Mensal (Gráfico Separado)
    # ==========================================
    fig_mape, ax2 = plt.subplots(figsize=(10, 5))
    bars2 = ax2.bar(x_labels, mape_vals, color="#3864a3", alpha=0.9, edgecolor="black")
    ax2.set_title(f"MAPE por Mês — {model_label}\nHorizonte: {horizonte_dias} dias", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Mês do Ano", fontsize=11)
    ax2.set_ylabel("MAPE (%)", fontsize=11)
    ax2.set_ylim(0, max(mape_vals) * 1.15)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars2:
        h_val = bar.get_height()
        if not np.isnan(h_val) and h_val > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, h_val + (max(mape_vals)*0.02), 
                     f"{h_val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()

    if save_plot:
        mape_path = out_dir / f"erro_sazonal_mape_{tipo_modelo}_h{horizonte_dias}d.png"
        plt.savefig(mape_path, dpi=150)
        plt.close(fig_mape)
        print(f"[evaluate] 2 Gráficos de Sazonalidade (MAE/MAPE) isolados salvos na pasta: {out_dir.name}/")
    else:
        plt.show()