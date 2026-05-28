# src/collectors/comexstat.py
import json
import time
import requests
import pandas as pd
from pathlib import Path
from config.settings import DATE_RANGE, COMEX_API_URL, COMEX_NCM_CODES, DATA_RAW

WAIT_SECONDS  = 12
MAX_RETRIES   = 3
CACHE_FILE    = DATA_RAW / "comexstat_cache.parquet"
CACHE_META_FILE = DATA_RAW / "comexstat_cache.meta.json"


def _fetch_year(year: int) -> pd.DataFrame:
    payload = {
        "flow":        "export",
        "monthDetail": True,
        "period":      {"from": f"{year}-01", "to": f"{year}-12"},
        "filters":     [{"filter": "heading", "values": COMEX_NCM_CODES}],
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


def _cache_metadata(start: str, end: str) -> dict:
    return {
        "start": start,
        "end": end,
        "start_year": int(start[:4]),
        "end_year": int(end[:4]),
        "ncm_codes": COMEX_NCM_CODES,
        "api_url": COMEX_API_URL,
    }


def _read_cache_metadata() -> dict | None:
    if not CACHE_META_FILE.exists():
        return None
    try:
        return json.loads(CACHE_META_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _cache_matches_config(df: pd.DataFrame, expected_metadata: dict) -> bool:
    metadata = _read_cache_metadata()
    if metadata != expected_metadata:
        return False
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return False
    start_year = expected_metadata["start_year"]
    end_year = expected_metadata["end_year"]
    return df.index.min().year <= start_year and df.index.max().year >= end_year


def _write_cache(df: pd.DataFrame, metadata: dict) -> None:
    df.to_parquet(CACHE_FILE)
    CACHE_META_FILE.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


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
    expected_metadata = _cache_metadata(start, end)

    # Carrega do cache se disponivel
    if CACHE_FILE.exists() and not force:
        cached_df = pd.read_parquet(CACHE_FILE)
        if _cache_matches_config(cached_df, expected_metadata):
            print(f"[comexstat] Carregando do cache: {CACHE_FILE}")
            df = cached_df
        else:
            print("[comexstat] Cache incompatível com config atual; re-coletando via API.")
            df = _fetch_all(start_year, end_year)
            _write_cache(df, expected_metadata)
            print(f"[comexstat] Cache salvo em: {CACHE_FILE}")
    else:
        print(f"[comexstat] Iniciando coleta {start_year}-{end_year} via API...")
        print(f"[comexstat] Estimativa: ~{(end_year - start_year + 1) * WAIT_SECONDS // 60 + 1} minutos.")
        df = _fetch_all(start_year, end_year)

        # Salva cache
        _write_cache(df, expected_metadata)
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
