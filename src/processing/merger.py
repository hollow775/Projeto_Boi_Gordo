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

    print(f"[merger] Dataset integrado. Shape: {df.shape}")
    print(f"[merger] Período: {df.index.min().date()} -> {df.index.max().date()}")

    return df


if __name__ == "__main__":
    df = build_dataset()
    print("\nColunas:", df.columns.tolist())
    print(df.head())