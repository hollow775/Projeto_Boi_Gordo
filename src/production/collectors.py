from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from typing import Callable

import pandas as pd

from config import settings
from src.production.db import (
    finish_collector_run,
    managed_connection,
    start_collector_run,
    upsert_source_dataframe,
)


@dataclass(frozen=True)
class ProductionSourceSpec:
    source_name: str
    table_name: str
    loader: Callable[[], pd.DataFrame]


def _load_ptax() -> pd.DataFrame:
    from src.collectors.bcb_ptax import load_ptax

    return load_ptax()


def _load_comexstat() -> pd.DataFrame:
    from src.collectors.comexstat import load_comexstat

    return load_comexstat(force=False)


def _load_sidra() -> pd.DataFrame:
    from src.collectors.ibge_sidra import load_sidra

    return load_sidra()


def _load_copernicus() -> pd.DataFrame:
    from src.collectors.copernicus import load_copernicus

    return load_copernicus(force_download=False)


def _load_inflation() -> pd.DataFrame:
    from src.collectors.base_deflacionaria import load_inflation_deflator

    return load_inflation_deflator()


API_SOURCE_SPECS: tuple[ProductionSourceSpec, ...] = (
    ProductionSourceSpec("ptax", "source_ptax_daily", _load_ptax),
    ProductionSourceSpec("comexstat", "source_comexstat_monthly", _load_comexstat),
    ProductionSourceSpec("sidra", "source_sidra_periodic", _load_sidra),
    ProductionSourceSpec("copernicus", "source_copernicus_monthly", _load_copernicus),
    ProductionSourceSpec("inflation", "source_inflation_index", _load_inflation),
)


@contextmanager
def production_date_range(start_date: str, end_date: str):
    """Temporarily scope legacy collectors to a production date range.

    Several brownfield collectors read config.settings.DATE_RANGE directly. The
    production lane must not inherit the fixed academic 2025 end date, so this
    adapter patches the shared mapping only for the collector call.
    """

    original = settings.DATE_RANGE.copy()
    settings.DATE_RANGE["start"] = start_date
    settings.DATE_RANGE["end"] = end_date
    try:
        yield
    finally:
        settings.DATE_RANGE.clear()
        settings.DATE_RANGE.update(original)


def _today_str() -> str:
    return date.today().isoformat()


def _filter_window(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if df.empty:
        return df
    frame = df.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        if "date" in frame.columns:
            frame.index = pd.to_datetime(frame.pop("date"))
        else:
            raise ValueError("Collector output must have a DatetimeIndex or a 'date' column.")
    return frame.sort_index().loc[start_date:end_date]


def refresh_api_sources(
    conn,
    start_date: str,
    end_date: str | None = None,
    sources: tuple[ProductionSourceSpec, ...] = API_SOURCE_SPECS,
) -> dict[str, dict[str, object]]:
    end_date = end_date or _today_str()
    results: dict[str, dict[str, object]] = {}

    for spec in sources:
        run_id = start_collector_run(conn, spec.source_name, start_date=start_date, end_date=end_date)
        try:
            with production_date_range(start_date, end_date):
                df = _filter_window(spec.loader(), start_date, end_date)
            rows_upserted = upsert_source_dataframe(conn, spec.table_name, df, run_id=run_id)
            finish_collector_run(
                conn,
                run_id,
                status="success",
                rows_fetched=len(df),
                rows_upserted=rows_upserted,
            )
            results[spec.source_name] = {
                "status": "success",
                "run_id": run_id,
                "rows_fetched": len(df),
                "rows_upserted": rows_upserted,
                "table_name": spec.table_name,
            }
        except Exception as exc:
            finish_collector_run(
                conn,
                run_id,
                status="failed",
                error_message=str(exc),
            )
            results[spec.source_name] = {
                "status": "failed",
                "run_id": run_id,
                "rows_fetched": 0,
                "rows_upserted": 0,
                "table_name": spec.table_name,
                "error": str(exc),
            }
    return results


def refresh_api_sources_with_managed_connection(
    start_date: str,
    end_date: str | None = None,
) -> dict[str, dict[str, object]]:
    with managed_connection() as conn:
        return refresh_api_sources(conn, start_date=start_date, end_date=end_date)
