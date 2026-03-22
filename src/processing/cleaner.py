# src/processing/cleaner.py
# ==============================================================
# Limpeza e validação do DataFrame integrado.
# Aplicado após o merger, antes da engenharia de features.
# ==============================================================

import pandas as pd
import numpy as np


# Limites físicos razoáveis para validação de domínio
# Ajustar se necessário conforme a série histórica
DOMAIN_BOUNDS = {
    "preco_boi_gordo":  (50,   600),   # R$/arroba
    "preco_bezerro":    (200,  5000),  # R$/cabeça
    "preco_milho":      (10,   200),   # R$/saca 60kg
    "abate_cabecas":    (1e3,  1e7),   # cabeças/trimestre
    "abate_peso_ton":   (1e2,  1e6),   # toneladas
    "export_usd_fob":   (0,    1e10),  # USD
    "export_kg":        (0,    1e9),   # kg
    "precipitacao_mm":  (0,    2000),  # mm/mês
    "ipca_index":       (1,    500),   # índice encadeado
}


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"[cleaner] Removidas {n_removed} linhas duplicadas.")
    return df


def _validate_domain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Substitui por NaN valores fora dos limites físicos definidos.
    Outliers extremos são provavelmente erros de coleta.
    """
    for col, (low, high) in DOMAIN_BOUNDS.items():
        if col not in df.columns:
            continue
        mask = (df[col] < low) | (df[col] > high)
        n_invalid = mask.sum()
        if n_invalid > 0:
            print(f"[cleaner] {col}: {n_invalid} valores fora do domínio [{low}, {high}] → NaN.")
            df.loc[mask, col] = np.nan
    return df


def _fill_missing(df: pd.DataFrame, max_gap: int = 7) -> pd.DataFrame:
    """
    Preenche NaN via interpolação linear para gaps <= max_gap dias.
    Gaps maiores permanecem como NaN (serão tratados no treinamento).

    Parâmetros
    ----------
    max_gap : número máximo de dias consecutivos a interpolar
    """
    for col in df.columns:
        n_missing_before = df[col].isna().sum()
        if n_missing_before == 0:
            continue

        df[col] = df[col].interpolate(
            method="linear",
            limit=max_gap,
            limit_direction="forward",
        )

        n_missing_after = df[col].isna().sum()
        filled = n_missing_before - n_missing_after
        if filled > 0:
            print(f"[cleaner] {col}: {filled} NaN preenchidos por interpolação (limite={max_gap}d).")
        if n_missing_after > 0:
            print(f"[cleaner] {col}: {n_missing_after} NaN remanescentes (gaps > {max_gap}d).")

    return df


def _report_missing(df: pd.DataFrame) -> None:
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("[cleaner] Sem valores ausentes após limpeza.")
    else:
        total = len(df)
        print("\n[cleaner] Valores ausentes remanescentes:")
        for col, n in missing.items():
            print(f"  {col}: {n} ({n/total*100:.2f}%)")


def clean(df: pd.DataFrame, max_gap: int = 7) -> pd.DataFrame:
    """
    Pipeline completo de limpeza.

    Etapas:
        1. Remove duplicatas de índice
        2. Valida limites de domínio
        3. Interpola gaps pequenos

    Parâmetros
    ----------
    df      : DataFrame com índice DatetimeIndex diário
    max_gap : limite de dias para interpolação de missing values

    Retorna
    -------
    pd.DataFrame limpo
    """
    print(f"[cleaner] Iniciando limpeza. Shape: {df.shape}")

    df = _remove_duplicates(df)
    df = _validate_domain(df)
    df = _fill_missing(df, max_gap=max_gap)
    _report_missing(df)

    print(f"[cleaner] Limpeza concluída. Shape final: {df.shape}")
    return df