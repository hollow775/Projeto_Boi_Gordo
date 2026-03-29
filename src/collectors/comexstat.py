# src/collectors/comexstat.py
import time
import requests
import pandas as pd
from pathlib import Path
from config.settings import DATE_RANGE, COMEX_API_URL, DATA_RAW

BEEF_HEADINGS = ["0201", "0202"]
WAIT_SECONDS  = 12
MAX_RETRIES   = 3
CACHE_FILE    = DATA_RAW / "comexstat_cache.parquet"


def _fetch_year(year: int) -> pd.DataFrame:
    payload = {
        "flow":        "export",
        "monthDetail": True,
        "period":      {"from": f"{year}-01", "to": f"{year}-12"},
        "filters":     [{"filter": "heading", "values": BEEF_HEADINGS}],
        "details":     ["heading"],
        "metrics":     ["metricFOB", "metricKG"],
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                COMEX_API_URL,
                json=payload,
                timeout=60,
                headers={"Content-Type": "application/json"},
            )
            if response.status_code == 429:
                wait = WAIT_SECONDS * attempt
                print(f"    [429] Rate limit. Aguardando {wait}s...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            records = response.json().get("data", {}).get("list", [])
            return pd.DataFrame(records) if records else pd.DataFrame()
        except requests.exceptions.HTTPError as e:
            if attempt == MAX_RETRIES:
                raise ConnectionError(
                    f"Falha na API ComexStat para {year} "
                    f"apos {MAX_RETRIES} tentativas: {e}"
                )
            time.sleep(WAIT_SECONDS * attempt)
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Falha ao conectar com ComexStat ({year}): {e}")

    return pd.DataFrame()


def _parse_comex(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "year":        "ano",
        "monthNumber": "mes",
        "metricFOB":   "export_usd_fob",
        "metricKG":    "export_kg",
    })
    df["export_usd_fob"] = pd.to_numeric(df["export_usd_fob"], errors="coerce").fillna(0)
    df["export_kg"]      = pd.to_numeric(df["export_kg"],      errors="coerce").fillna(0)
    df["data"] = pd.to_datetime(
        df["ano"].astype(str) + "-" + df["mes"].astype(str).str.zfill(2) + "-01"
    )
    df = df.groupby("data")[["export_usd_fob", "export_kg"]].sum()
    df.index.name = "data"
    return df.sort_index()


def _expand_to_daily(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    daily_index = pd.date_range(start=start, end=end, freq="D")
    df = df.reindex(daily_index, method="ffill")
    df.index.name = "data"
    return df


def _fetch_all(start_year: int, end_year: int) -> pd.DataFrame:
    """Coleta todos os anos via API com espera entre requisicoes."""
    total  = end_year - start_year + 1
    frames = []

    for i, year in enumerate(range(start_year, end_year + 1), 1):
        print(f"[comexstat] Coletando {year} ({i}/{total})...", end=" ", flush=True)
        df_year = _fetch_year(year)
        if df_year.empty:
            print("sem dados.")
        else:
            frames.append(_parse_comex(df_year))
            print("OK")
        if i < total:
            time.sleep(WAIT_SECONDS)

    if not frames:
        raise ValueError("Nenhum dado retornado pelo ComexStat.")

    return pd.concat(frames).groupby("data").sum().sort_index()


def load_comexstat(force: bool = False) -> pd.DataFrame:
    """
    Coleta exportacoes de carne bovina e retorna em frequencia diaria.

    Na primeira execucao, busca ano a ano via API e salva cache em
    data/raw/comexstat_cache.parquet.

    Nas execucoes seguintes, carrega diretamente do cache.

    Parametros
    ----------
    force : se True, ignora cache e re-coleta via API.

    Retorna
    -------
    pd.DataFrame com colunas: export_usd_fob, export_kg
    """
    start      = DATE_RANGE["start"]
    end        = DATE_RANGE["end"]
    start_year = int(start[:4])
    end_year   = int(end[:4])

    # Carrega do cache se disponivel
    if CACHE_FILE.exists() and not force:
        print(f"[comexstat] Carregando do cache: {CACHE_FILE}")
        df = pd.read_parquet(CACHE_FILE)
    else:
        print(f"[comexstat] Iniciando coleta {start_year}-{end_year} via API...")
        print(f"[comexstat] Estimativa: ~{(end_year - start_year + 1) * WAIT_SECONDS // 60 + 1} minutos.")
        df = _fetch_all(start_year, end_year)

        # Salva cache
        df.to_parquet(CACHE_FILE)
        print(f"[comexstat] Cache salvo em: {CACHE_FILE}")

    df = df.loc[start:end]
    df = _expand_to_daily(df, start, end)

    print(f"[comexstat] {len(df)} registros diarios carregados.")
    return df


if __name__ == "__main__":
    # Use force=True para forcar re-coleta
    df = load_comexstat(force=False)
    print(df.head(10))
    print(f"\nShape: {df.shape}")