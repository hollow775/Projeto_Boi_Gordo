from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from config.settings import CEPEA_FILES
from src.production.db import record_manual_source_file


@dataclass(frozen=True)
class ManualSourceRecord:
    source_name: str
    file_path: Path
    file_hash_sha256: str
    rows_loaded: int
    effective_start_date: pd.Timestamp | None
    effective_end_date: pd.Timestamp | None
    manifest_id: int | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _date_range_from_frame(df: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return None, None
    return pd.Timestamp(df.index.min()), pd.Timestamp(df.index.max())


def record_manual_cepea_file(
    conn,
    source_name: str,
    file_path: str | Path,
    loader: Callable[[Path], pd.DataFrame],
    run_id: int | None = None,
) -> ManualSourceRecord:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CEPEA manual file not found: {path}")

    df = loader(path)
    start_date, end_date = _date_range_from_frame(df)
    file_hash = sha256_file(path)
    manifest_id = record_manual_source_file(
        conn,
        source_name=source_name,
        file_path=str(path),
        file_hash_sha256=file_hash,
        rows_loaded=len(df),
        effective_start_date=start_date,
        effective_end_date=end_date,
        run_id=run_id,
    )
    return ManualSourceRecord(
        source_name=source_name,
        file_path=path,
        file_hash_sha256=file_hash,
        rows_loaded=len(df),
        effective_start_date=start_date,
        effective_end_date=end_date,
        manifest_id=manifest_id,
    )


def configured_cepea_manual_files() -> dict[str, Path]:
    return {
        "cepea_boi_gordo": CEPEA_FILES["boi_gordo"],
        "cepea_bezerro": CEPEA_FILES["bezerro"],
        "cepea_milho": CEPEA_FILES["milho"],
    }
