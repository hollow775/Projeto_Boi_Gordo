# src/collectors/pipeline_runner.py
# ==============================================================
# Incremental data collection wrapper.
# Calls each existing collector from start_date to today,
# appending only new rows. Persists last successful date.
# ==============================================================
from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pandas as pd

from config.settings import (
    DATA_RAW,
    DATE_RANGE,
    LAST_RUN_PATH,
)


def _today_str() -> str:
    return date.today().isoformat()


def read_last_run() -> dict:
    """Read last_run.json or return defaults."""
    if LAST_RUN_PATH.exists():
        return json.loads(LAST_RUN_PATH.read_text(encoding="utf-8"))
    return {"last_date": DATE_RANGE["start"], "timestamp": None}


def _save_last_run(last_date: str) -> None:
    LAST_RUN_PATH.write_text(
        json.dumps(
            {"last_date": last_date, "timestamp": datetime.now(timezone.utc).isoformat()},
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_cepea(start_date: str) -> pd.DataFrame:
    """CEPEA is file-based (manual XLS). Just reload full file — no API call."""
    from src.collectors.cepea import load_cepea

    df = load_cepea()
    return df.loc[start_date:]


def _run_sidra(start_date: str) -> pd.DataFrame:
    """SIDRA: reload full (API returns last 64 quarters). Filter to new rows."""
    from src.collectors.ibge_sidra import load_sidra

    df = load_sidra()
    return df.loc[start_date:]


def _run_comexstat(start_date: str) -> pd.DataFrame:
    """ComexStat: fetch only years >= start_date year, merge with cache."""
    from src.collectors.comexstat import load_comexstat

    df = load_comexstat(force=False)
    return df.loc[start_date:]


def _run_copernicus(start_date: str) -> pd.DataFrame:
    """Copernicus ERA5: reload from cached NetCDF. Filter to new rows."""
    from src.collectors.copernicus import load_copernicus

    df = load_copernicus(force_download=False)
    return df.loc[start_date:]


def _run_ptax(start_date: str) -> pd.DataFrame:
    """BCB PTAX: full API call (fast), filter to new rows."""
    from src.collectors.bcb_ptax import load_ptax

    df = load_ptax()
    return df.loc[start_date:]


def _run_inflation(start_date: str) -> pd.DataFrame:
    """BCB inflation index: full API call, filter to new rows."""
    from src.collectors.base_deflacionaria import load_inflation_deflator

    df = load_inflation_deflator()
    return df.loc[start_date:]


COLLECTORS = [
    ("cepea", _run_cepea),
    ("sidra", _run_sidra),
    ("comexstat", _run_comexstat),
    ("copernicus", _run_copernicus),
    ("ptax", _run_ptax),
    ("inflation", _run_inflation),
]


def run_collectors(start_date: str | None = None) -> dict[str, pd.DataFrame]:
    """
    Run all collectors incrementally from start_date to today.

    Parameters
    ----------
    start_date : ISO date string. If None, reads from last_run.json.

    Returns
    -------
    dict mapping collector name to its DataFrame (new rows only).
    """
    if start_date is None:
        start_date = read_last_run()["last_date"]

    end_date = min(_today_str(), DATE_RANGE["end"])
    print(f"[pipeline_runner] Collecting from {start_date} to {end_date}")

    results: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}

    for name, fn in COLLECTORS:
        try:
            print(f"[pipeline_runner] Running {name}...")
            df = fn(start_date)
            results[name] = df
            print(f"[pipeline_runner] {name}: {len(df)} rows")
        except Exception as e:
            errors[name] = str(e)
            print(f"[pipeline_runner] {name} FAILED: {e}")

    _save_last_run(end_date)
    print(f"[pipeline_runner] Done. Successes: {len(results)}, Failures: {len(errors)}")

    if errors:
        print(f"[pipeline_runner] Failed collectors: {list(errors.keys())}")

    return results


if __name__ == "__main__":
    results = run_collectors()
    for name, df in results.items():
        print(f"  {name}: {df.shape}")
