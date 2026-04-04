# src/collectors/base_deflacionaria.py
# ==============================================================
# Coleta o índice de inflação (IPCA ou IGP-DI) via API do Banco Central do Brasil (SGS).
#
# Usado para deflacionar os preços nominais do CEPEA.
# Documentação SGS: https://www.bcb.gov.br/estatisticas/tabelasespeciais
# ==============================================================

import requests
import pandas as pd
from config.settings import DATE_RANGE, DEFLATION_BASE, INFLATION_SGS_CODE, BCB_SGS_URL


def _fetch_inflation_series() -> pd.Series:
    url = BCB_SGS_URL.format(code=INFLATION_SGS_CODE)
    params = {
        "formato":    "json",
        "dataInicial": pd.Timestamp(DATE_RANGE["start"]).strftime("%d/%m/%Y"),
        "dataFinal":   pd.Timestamp(DATE_RANGE["end"]).strftime("%d/%m/%Y"),
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Falha ao acessar API BCB: {e}")

    data = response.json()
    df = pd.DataFrame(data)
    df["data"]  = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = df["valor"].astype(float)
    df = df.set_index("data").sort_index()

    return df["valor"]  # variação % mensal


def _build_price_index(series: pd.Series, base_period: str) -> pd.Series:
    """
    Converte variação % mensal em índice de preços encadeado.
    Base = 100 no período informado em DEFLATION_BASE.
    """
    # Índice encadeado: produto acumulado de (1 + variação/100)
    index = (1 + series / 100).cumprod()

    # Rebase para o período-base
    base_date = pd.Timestamp(base_period + "-01")
    if base_date not in index.index:
        raise ValueError(
            f"Período base '{base_period}' não encontrado na série de inflação. "
            "Verifique DEFLATION_BASE em config/settings.py."
        )

    base_value = index.loc[base_date]
    index = index / base_value * 100

    return index


def _expand_to_daily(series: pd.Series, start: str, end: str) -> pd.Series:
    daily_index = pd.date_range(start=start, end=end, freq="D")
    series = series.reindex(daily_index, method="ffill")
    series.index.name = "data"
    return series


def load_inflation_deflator() -> pd.DataFrame:
    """
    Retorna o índice de inflação diário para deflação de séries de preços.

    Retorna
    -------
    pd.DataFrame com coluna:
        inflation_index : índice encadeado (base = 100 em DEFLATION_BASE)
    """
    series = _fetch_inflation_series()
    index  = _build_price_index(series, DEFLATION_BASE)

    start = DATE_RANGE["start"]
    end   = DATE_RANGE["end"]

    index = index.loc[start:end]
    index = _expand_to_daily(index, start, end)

    return index.rename("inflation_index").to_frame()


if __name__ == "__main__":
    df = load_inflation_deflator()
    print(df.head(10))
    print(f"\nÍndice no período base: {df.loc[DEFLATION_BASE + '-01', 'inflation_index']:.2f}")
