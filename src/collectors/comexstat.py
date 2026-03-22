# src/collectors/comexstat.py
# ==============================================================
# Coleta dados de exportação de carne bovina via API ComexStat
# do MDIC (Ministério do Desenvolvimento, Indústria e Comércio).
#
# Endpoint: https://api-comexstat.mdic.gov.br/general
# Documentação: http://comexstat.mdic.gov.br/
#
# NCMs utilizados:
#   0201 → Carnes bovinas frescas ou refrigeradas
#   0202 → Carnes bovinas congeladas
#
# Métricas retornadas:
#   metricFOB  → valor exportado em USD FOB
#   metricKG   → peso líquido em kg
# ==============================================================

import requests
import pandas as pd
from config.settings import DATE_RANGE, COMEX_NCM_CODES, COMEX_API_URL


def _fetch_comex_ncm(ncm: str, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Busca dados mensais de exportação para um NCM específico.
    """
    payload = {
        "flow": "export",
        "monthStart": f"{start_year}01",
        "monthEnd": f"{end_year}12",
        "filters": [
            {"filter": "ncm", "values": [ncm]}
        ],
        "details": ["ncm"],
        "metrics": ["metricFOB", "metricKG"],
        "lang": "pt",
    }

    try:
        response = requests.post(
            f"{COMEX_API_URL}",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(
            f"Falha ao conectar com ComexStat para NCM {ncm}: {e}"
        )

    data = response.json()

    # A API retorna { "data": { "list": [...] } }
    records = data.get("data", {}).get("list", [])
    if not records:
        raise ValueError(f"Resposta ComexStat vazia para NCM {ncm}.")

    df = pd.DataFrame(records)
    return df


def _parse_comex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza o DataFrame retornado pela API ComexStat.
    """
    # A API retorna colunas: coAno, coMes, vlFob, kgLiquido, etc.
    df = df.rename(columns={
        "coAno":     "ano",
        "coMes":     "mes",
        "vlFob":     "export_usd_fob",
        "kgLiquido": "export_kg",
    })

    df["data"] = pd.to_datetime(
        df["ano"].astype(str) + "-" + df["mes"].astype(str).str.zfill(2) + "-01"
    )

    df = df[["data", "export_usd_fob", "export_kg"]].copy()
    df["export_usd_fob"] = pd.to_numeric(df["export_usd_fob"], errors="coerce")
    df["export_kg"]      = pd.to_numeric(df["export_kg"],      errors="coerce")

    df = df.groupby("data").sum()
    df.index.name = "data"

    return df


def _expand_to_daily(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Expande dados mensais para frequência diária via forward fill.
    """
    daily_index = pd.date_range(start=start, end=end, freq="D")
    df = df.reindex(daily_index, method="ffill")
    df.index.name = "data"
    return df


def load_comexstat() -> pd.DataFrame:
    """
    Coleta e retorna exportações de carne bovina em frequência diária.

    Retorna
    -------
    pd.DataFrame com colunas:
        export_usd_fob : valor exportado em USD FOB (soma NCM 0201 + 0202)
        export_kg      : peso exportado em kg       (soma NCM 0201 + 0202)
    """
    start_year = int(DATE_RANGE["start"][:4])
    end_year   = int(DATE_RANGE["end"][:4])
    start      = DATE_RANGE["start"]
    end        = DATE_RANGE["end"]

    frames = []
    for ncm in COMEX_NCM_CODES:
        raw = _fetch_comex_ncm(ncm, start_year, end_year)
        parsed = _parse_comex(raw)
        frames.append(parsed)

    df = pd.concat(frames).groupby("data").sum()
    df = df.sort_index()
    df = df.loc[start:end]
    df = _expand_to_daily(df, start, end)

    return df


if __name__ == "__main__":
    df = load_comexstat()
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"Missing values:\n{df.isna().sum()}")