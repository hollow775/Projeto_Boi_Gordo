# src/processing/cleaner.py
import pandas as pd
import numpy as np


# Limites de dominio apenas para variaveis nao-preco.
# Precos CEPEA sao validados separadamente apos inspecao da serie.
DOMAIN_BOUNDS = {
    "abate_cabecas":    (1e3,   1e7),
    "abate_peso_ton":   (1e2,   1e9),
    "export_usd_fob":   (0,     1e11),
    "export_kg":        (0,     1e10),
    "precipitacao_mm":  (0,     3000),
    "ipca_index":       (1,     500),
}


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas com indice duplicado, mantendo a primeira ocorrencia."""
    n_before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"[cleaner] Removidas {n_removed} linhas com indice duplicado.")
    return df


def _validate_domain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Substitui por NaN valores fora dos limites fisicos definidos.
    Opera apenas sobre colunas listadas em DOMAIN_BOUNDS.
    """
    for col, (low, high) in DOMAIN_BOUNDS.items():
        if col not in df.columns:
            continue

        series = df[col]

        # Constroi mascara booleana sem operacoes entre Series de indices distintos
        below = (series < low).fillna(False).values   # numpy array
        above = (series > high).fillna(False).values
        mask  = below | above

        n_invalid = int(mask.sum())
        if n_invalid > 0:
            print(f"[cleaner] {col}: {n_invalid} valores fora do dominio [{low}, {high}] -> NaN.")
            df[col] = np.where(mask, np.nan, df[col].values)

    return df


def _report_prices(df: pd.DataFrame) -> None:
    """Exibe estatisticas das series de preco para inspecao manual."""
    price_cols = [c for c in df.columns if "preco" in c and "nominal" not in c]
    if not price_cols:
        return
    print("\n[cleaner] Estatisticas das series de preco (para inspecao):")
    for col in price_cols:
        s = df[col].dropna()
        if s.empty:
            print(f"  {col}: VAZIA")
        else:
            print(f"  {col}: min={s.min():.2f} | max={s.max():.2f} | "
                  f"media={s.mean():.2f} | nulos={int(df[col].isna().sum())}")


def _fill_missing(df: pd.DataFrame, max_gap: int = 7) -> pd.DataFrame:
    """
    Preenche NaN via interpolacao linear para gaps <= max_gap dias.
    """
    # Remove colunas duplicadas antes de processar
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    for col in df.columns:
        n_missing_before = int(df[col].isna().sum())
        if n_missing_before == 0:
            continue

        df[col] = df[col].interpolate(
            method="linear",
            limit=max_gap,
            limit_direction="forward",
        )

        n_missing_after  = int(df[col].isna().sum())
        filled           = n_missing_before - n_missing_after

        if filled > 0:
            print(f"[cleaner] {col}: {filled} NaN preenchidos (limite={max_gap}d).")
        if n_missing_after > 0:
            print(f"[cleaner] {col}: {n_missing_after} NaN remanescentes (gaps > {max_gap}d).")

    return df


def _report_missing(df: pd.DataFrame) -> None:
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("[cleaner] Sem valores ausentes apos limpeza.")
    else:
        total = len(df)
        print("\n[cleaner] Valores ausentes remanescentes:")
        for col, n in missing.items():
            print(f"  {col}: {int(n)} ({int(n)/total*100:.2f}%)")


def clean(df: pd.DataFrame, max_gap: int = 7) -> pd.DataFrame:
    """
    Pipeline de limpeza:
        1. Remove indice duplicado
        2. Valida limites de dominio (variaveis nao-preco)
        3. Exibe estatisticas de preco para inspecao
        4. Interpola gaps pequenos
    """
    print(f"[cleaner] Iniciando limpeza. Shape: {df.shape}")

    # Elimina colunas duplicadas — ocorre quando fontes compartilham nomes
    n_dup_cols = df.columns.duplicated().sum()
    if n_dup_cols > 0:
        print(f"[cleaner] Removidas {n_dup_cols} colunas duplicadas: {list(df.columns[df.columns.duplicated()])}")
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    df = _remove_duplicates(df)
    df = _validate_domain(df)
    _report_prices(df)
    df = _fill_missing(df, max_gap=max_gap)
    _report_missing(df)

    print(f"[cleaner] Limpeza concluida. Shape final: {df.shape}")
    return df