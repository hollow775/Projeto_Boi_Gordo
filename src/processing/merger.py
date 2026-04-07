# src/processing/merger.py
# ==============================================================
# Integra todas as fontes de dados em um único DataFrame diário.
# Aplica deflação dos preços nominais pelo índice de inflação configurado.
# ==============================================================

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from config.settings import DEFLATION_BASE

from src.collectors.cepea      import load_cepea
from src.collectors.ibge_sidra import load_sidra
from src.collectors.comexstat  import load_comexstat
from src.collectors.copernicus import load_copernicus
from src.collectors.bcb_ptax   import load_ptax
from src.collectors.base_deflacionaria import load_inflation_deflator


# Colunas de preço que devem ser deflacionadas
PRICE_COLUMNS = ["preco_boi_gordo", "preco_bezerro", "preco_milho"]
# Séries de frequência mensal (ou inferior) que só ficam disponíveis
# ao fim do mês → aplicar lag mensal antes de expandir para frequência diária
# para evitar vazamento de informação (leakage).
MONTHLY_COLUMNS = [
    "inflation_index",  # IPCA/IGP encadeado – divulgado mensalmente
    "precipitacao_mm",  # ERA5 mensal
    "export_usd_fob",   # ComexStat mensal
    "export_kg",
    "abate_cabecas",    # SIDRA trimestral → tratado como mensal para lag
    "abate_peso_ton",
]
MONTHLY_LAG_MONTHS = 1
HOLDOUT_CUTOFF = pd.Timestamp("2025-12-31")


def _lag_monthly_then_ffill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica lag mensal (+1m) nas séries mensais/trimestrais e só então
    expande para frequência diária via forward fill, evitando leakage
    intra-mês.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    touched = []
    for col in MONTHLY_COLUMNS:
        if col not in df.columns:
            continue
        monthly = df[col].resample("MS").first()
        lagged  = monthly.shift(MONTHLY_LAG_MONTHS)
        df[col] = lagged.reindex(df.index, method="ffill")
        touched.append(col)

    if touched:
        print(f"[merger] Lag mensal (+{MONTHLY_LAG_MONTHS}m) aplicado e forward fill diário em: {touched}")
    return df


def _annotate_holdout(df: pd.DataFrame, cutoff: pd.Timestamp = HOLDOUT_CUTOFF) -> pd.DataFrame:
    """
    Armazena o trecho de holdout (datas > cutoff) em attrs para uso
    posterior (plots), sem removê-lo aqui.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    df.attrs["holdout_cutoff"] = cutoff
    if df.index.max() > cutoff:
        holdout_tail = df.loc[df.index > cutoff].copy()
        df.attrs["holdout_tail"] = holdout_tail
        print(f"[merger] Holdout identificado: {len(holdout_tail)} linhas após {cutoff.date()} armazenadas em attrs['holdout_tail'].")
    else:
        df.attrs["holdout_tail"] = pd.DataFrame(columns=df.columns)
    return df


def _deflate_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte preços nominais em preços reais (base = DEFLATION_BASE).

    Fórmula:
        preco_real = preco_nominal / (inflation_index / 100)

    Após deflação, mantém o índice de inflação no DataFrame para uso
    como feature (pode capturar contexto inflacionário).
    """
    if "inflation_index" not in df.columns:
        raise KeyError("Coluna 'inflation_index' ausente. Verifique o collector BCB.")

    for col in PRICE_COLUMNS:
        if col not in df.columns:
            continue
        nominal_col = f"{col}_nominal"
        df[nominal_col] = df[col].copy()  # mantém série nominal original
        df[col] = df[col] / (df["inflation_index"] / 100)

    print(f"[merger] Deflação aplicada. Base: {DEFLATION_BASE} = 100.")
    return df


def build_dataset(
    include_cepea:      bool = True,
    include_sidra:      bool = True,
    include_comexstat:  bool = True,
    include_copernicus: bool = True,
    include_ptax:       bool = True,
    deflate:            bool = True,
) -> pd.DataFrame:
    """
    Coleta, integra e opcionalmente deflaciona todas as fontes.

    Parâmetros
    ----------
    include_*  : flags para incluir/excluir fontes (útil em testes)
    deflate    : se True, aplica deflação pelo índice configurado nos preços

    Retorna
    -------
    pd.DataFrame com índice DatetimeIndex diário e todas as variáveis.
    """
    frames = []

    if include_cepea:
        print("[merger] Carregando CEPEA...")
        frames.append(load_cepea())

    if include_sidra:
        print("[merger] Carregando SIDRA (IBGE)...")
        frames.append(load_sidra())

    if include_comexstat:
        print("[merger] Carregando ComexStat (MDIC)...")
        frames.append(load_comexstat())

    if include_copernicus:
        print("[merger] Carregando ERA5 (Copernicus)...")
        frames.append(load_copernicus())

    if include_ptax:
        print("[merger] Carregando Dólar PTAX (BCB)...")
        frames.append(load_ptax())

    if deflate:
        print("[merger] Carregando índice de inflação (BCB)...")
        frames.append(load_inflation_deflator())

    if not frames:
        raise ValueError("Nenhuma fonte selecionada para o dataset.")

    # Join pelo índice de data — inner join garante que só ficam
    # dias com dados em todas as fontes
    df = frames[0]
    for f in frames[1:]:
        df = df.join(f, how="outer")

    df = df.sort_index()

    if deflate:
        df = _deflate_prices(df)

    # Após deflação, aplica lag mensal e forward fill diário nas séries
    # de baixa frequência para evitar leakage intra-mês nos modelos.
    df = _lag_monthly_then_ffill(df)

    df = _annotate_holdout(df, cutoff=HOLDOUT_CUTOFF)

    print(f"[merger] Dataset integrado. Shape: {df.shape}")
    # Evita caractere Unicode não suportado em consoles cp1252
    print(f"[merger] Periodo: {df.index.min().date()} -> {df.index.max().date()}")

    return df


if __name__ == "__main__":
    df = build_dataset()
    print("\nColunas:", df.columns.tolist())
    print(df.head())
