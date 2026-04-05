# src/features/engineering.py
# ==============================================================
# Engenharia de features para os modelos XGBoost e Random Forest.
#
# Features geradas:
#   - Lags temporais das variáveis principais
#   - Médias móveis (rolling means)
#   - Desvio padrão móvel (volatilidade)
#   - Variação percentual
#   - Features de calendário (sazonalidade)
#   - Targets deslocados para cada horizonte de previsão
#
# Referências metodológicas:
#   Mion (2022) — Random Forest para previsão de boi em SP
#   Souza (2019) — Redes neurais com lags para arroba
# ==============================================================

import pandas as pd
import numpy as np
from config.settings import HORIZONS


# ── Configuração de lags e janelas ─────────────────────────────
# Lags em dias corridos aplicados às variáveis de preço e abate
LAG_DAYS = [1, 7, 14, 21, 30, 60, 90]

# Janelas para médias e desvios móveis (em dias)
ROLLING_WINDOWS = [7, 14, 30, 60, 90]

# Variáveis que recebem lags e rolling features
LAG_FEATURES = [
    "preco_boi_gordo",
    "preco_bezerro",
    "preco_milho",
    "abate_peso_ton",
    "export_usd_fob",
    "precipitacao_mm",
    "cotacao_dolar_venda",
]


def _lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gera colunas de lags para cada variável em LAG_FEATURES."""
    for col in LAG_FEATURES:
        if col not in df.columns:
            continue
        for lag in LAG_DAYS:
            df[f"{col}_lag{lag}d"] = df[col].shift(lag)
    return df


def _rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera médias móveis e desvio padrão para cada variável.
    min_periods=1 evita NaN nas primeiras linhas.
    """
    for col in LAG_FEATURES:
        if col not in df.columns:
            continue
        for window in ROLLING_WINDOWS:
            df[f"{col}_ma{window}d"] = (
                df[col].rolling(window=window, min_periods=1).mean()
            )
            df[f"{col}_std{window}d"] = (
                df[col].rolling(window=window, min_periods=1).std()
            )
    return df


def _pct_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variação percentual em relação a 1, 7 e 30 dias anteriores."""
    for col in LAG_FEATURES:
        if col not in df.columns:
            continue
        for period in [1, 7, 30]:
            df[f"{col}_pct{period}d"] = df[col].pct_change(periods=period)
    return df


def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features de calendário para capturar sazonalidade.
    O ciclo pecuário brasileiro tem forte componente sazonal
    (Lampert et al., 2017; Renesto, 2008).
    """
    df["mes"]         = df.index.month
    df["trimestre"]   = df.index.quarter
    df["dia_semana"]  = df.index.dayofweek
    df["dia_ano"]     = df.index.dayofyear

    # Codificação cíclica do mês (evita descontinuidade dez→jan)
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)

    return df


def _ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratios entre variáveis — capturam relações de custo/benefício.
    Ex: relação boi/milho é indicador clássico de rentabilidade
    do confinamento (Sartorello, 2016).
    """
    if "preco_boi_gordo" in df.columns and "preco_milho" in df.columns:
        df["ratio_boi_milho"] = df["preco_boi_gordo"] / df["preco_milho"].replace(0, np.nan)

    if "preco_boi_gordo" in df.columns and "preco_bezerro" in df.columns:
        df["ratio_boi_bezerro"] = df["preco_boi_gordo"] / df["preco_bezerro"].replace(0, np.nan)

    return df


def _build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria variável-alvo para cada horizonte de previsão.
    target_hXd = preço do boi daqui a X dias.

    Utiliza Direct Multi-Step Forecasting:
    um modelo independente por horizonte.
    """
    if "preco_boi_gordo" not in df.columns:
        raise KeyError("Coluna 'preco_boi_gordo' ausente. Verifique o merger.")

    for h in HORIZONS:
        df[f"target_h{h}d"] = df["preco_boi_gordo"].shift(-h)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo de feature engineering.

    Ordem das etapas:
        1. Lags
        2. Rolling statistics
        3. Variação percentual
        4. Features de calendário
        5. Ratios entre variáveis
        6. Targets para cada horizonte

    Parâmetros
    ----------
    df : DataFrame limpo e integrado (saída de cleaner.clean)

    Retorna
    -------
    pd.DataFrame com features e targets.
    Linhas com NaN em target são inevitáveis no final da série
    (horizonte futuro além do período disponível).
    """
    print(f"[features] Iniciando feature engineering. Shape: {df.shape}")

    df = _lag_features(df)
    df = _rolling_features(df)
    df = _pct_change_features(df)
    df = _calendar_features(df)
    df = _ratio_features(df)
    df = _build_targets(df)

    print(f"[features] Feature engineering concluída. Shape: {df.shape}")
    print(f"[features] Total de features: {df.shape[1] - len(HORIZONS)}")

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Retorna lista de colunas de features (exclui targets e séries originais
    que causariam data leakage se incluídas sem lag).

    Colunas excluídas como features diretas:
        - preco_boi_gordo (variável-alvo sem lag = leakage)
        - colunas _nominal (redundantes após deflação)
        - colunas target_h*
    """
    exclude_patterns = ["target_h", "_nominal", "ipca_index"]
    # preco_boi_gordo sem lag seria leakage — excluir a coluna original
    direct_exclude = ["preco_boi_gordo"]

    feature_cols = [
        col for col in df.columns
        if col not in direct_exclude
        and not any(col.startswith(p) or p in col for p in exclude_patterns)
    ]

    return feature_cols