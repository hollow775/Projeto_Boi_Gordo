from __future__ import annotations

import os
import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class MySQLConfig:
    host: str
    port: int
    database: str
    user: str
    password: str


REQUIRED_ENV = {
    "host": "BOI_DB_HOST",
    "port": "BOI_DB_PORT",
    "database": "BOI_DB_NAME",
    "user": "BOI_DB_USER",
    "password": "BOI_DB_PASSWORD",
}

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS collector_runs (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        source_name VARCHAR(80) NOT NULL,
        started_at_utc DATETIME NOT NULL,
        finished_at_utc DATETIME NULL,
        status VARCHAR(20) NOT NULL,
        start_date DATE NULL,
        end_date DATE NULL,
        rows_fetched INT NOT NULL DEFAULT 0,
        rows_upserted INT NOT NULL DEFAULT 0,
        error_message TEXT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS source_ptax_daily (
        date DATE PRIMARY KEY,
        cotacao_dolar_venda DOUBLE NULL,
        collected_at_utc DATETIME NOT NULL,
        run_id BIGINT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS source_inflation_index (
        date DATE PRIMARY KEY,
        inflation_index DOUBLE NULL,
        collected_at_utc DATETIME NOT NULL,
        run_id BIGINT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS source_comexstat_monthly (
        date DATE PRIMARY KEY,
        export_usd_fob DOUBLE NULL,
        export_kg DOUBLE NULL,
        collected_at_utc DATETIME NOT NULL,
        run_id BIGINT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS source_sidra_periodic (
        date DATE PRIMARY KEY,
        abate_cabecas DOUBLE NULL,
        abate_peso_ton DOUBLE NULL,
        collected_at_utc DATETIME NOT NULL,
        run_id BIGINT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS source_copernicus_monthly (
        date DATE PRIMARY KEY,
        precipitacao_mm DOUBLE NULL,
        collected_at_utc DATETIME NOT NULL,
        run_id BIGINT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS manual_source_files (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        source_name VARCHAR(80) NOT NULL,
        file_path TEXT NOT NULL,
        file_hash_sha256 CHAR(64) NOT NULL,
        ingested_at_utc DATETIME NOT NULL,
        effective_start_date DATE NULL,
        effective_end_date DATE NULL,
        rows_loaded INT NOT NULL DEFAULT 0,
        run_id BIGINT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS production_model_versions (
        version VARCHAR(32) PRIMARY KEY,
        trained_at_utc DATETIME NOT NULL,
        train_start_date DATE NULL,
        train_end_date DATE NULL,
        data_max_date_by_source JSON NULL,
        metrics_path TEXT NULL,
        models_path TEXT NULL,
        promoted BOOLEAN NOT NULL DEFAULT FALSE,
        promotion_reason TEXT NULL
    )
    """,
]

SOURCE_TABLE_COLUMNS = {
    "source_ptax_daily": ["cotacao_dolar_venda"],
    "source_inflation_index": ["inflation_index"],
    "source_comexstat_monthly": ["export_usd_fob", "export_kg"],
    "source_sidra_periodic": ["abate_cabecas", "abate_peso_ton"],
    "source_copernicus_monthly": ["precipitacao_mm"],
}


def load_mysql_config(env: dict[str, str] | None = None) -> MySQLConfig:
    env = env or os.environ
    missing = [env_name for env_name in REQUIRED_ENV.values() if not env.get(env_name)]
    if missing:
        raise RuntimeError(
            "Missing MySQL environment variables for production lane: "
            + ", ".join(sorted(missing))
        )

    return MySQLConfig(
        host=env[REQUIRED_ENV["host"]],
        port=int(env[REQUIRED_ENV["port"]]),
        database=env[REQUIRED_ENV["database"]],
        user=env[REQUIRED_ENV["user"]],
        password=env[REQUIRED_ENV["password"]],
    )


def connect(config: MySQLConfig | None = None):
    config = config or load_mysql_config()
    try:
        import mysql.connector
    except ImportError as exc:  # pragma: no cover - exercised when dependency missing locally
        raise RuntimeError(
            "mysql-connector-python is required for live MySQL production operations. "
            "Install project requirements before running --init-db/--refresh-data."
        ) from exc

    return mysql.connector.connect(
        host=config.host,
        port=config.port,
        database=config.database,
        user=config.user,
        password=config.password,
    )


@contextmanager
def managed_connection(config: MySQLConfig | None = None):
    conn = connect(config)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_schema(conn) -> None:
    cursor = conn.cursor()
    try:
        for statement in SCHEMA_STATEMENTS:
            cursor.execute(statement)
    finally:
        cursor.close()


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _date_value(value: Any):
    if pd.isna(value):
        return None
    return pd.Timestamp(value).date()


def _clean_value(value: Any):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _frame_rows(df: pd.DataFrame, value_columns: Iterable[str], run_id: int | None) -> list[tuple]:
    if df.empty:
        return []
    frame = df.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        if "date" not in frame.columns:
            raise ValueError("DataFrame must have a DatetimeIndex or a 'date' column.")
        frame.index = pd.to_datetime(frame.pop("date"))

    collected_at = _utc_now_naive()
    rows = []
    for date_index, row in frame.iterrows():
        rows.append(
            (
                _date_value(date_index),
                *[_clean_value(row.get(column)) for column in value_columns],
                collected_at,
                run_id,
            )
        )
    return rows


def upsert_source_dataframe(
    conn,
    table_name: str,
    df: pd.DataFrame,
    run_id: int | None = None,
) -> int:
    if table_name not in SOURCE_TABLE_COLUMNS:
        raise ValueError(f"Unsupported production source table: {table_name}")

    value_columns = SOURCE_TABLE_COLUMNS[table_name]
    rows = _frame_rows(df, value_columns, run_id)
    if not rows:
        return 0

    columns = ["date", *value_columns, "collected_at_utc", "run_id"]
    placeholders = ", ".join(["%s"] * len(columns))
    updates = ", ".join(
        f"{column}=VALUES({column})" for column in [*value_columns, "collected_at_utc", "run_id"]
    )
    sql = (
        f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON DUPLICATE KEY UPDATE {updates}"
    )

    cursor = conn.cursor()
    try:
        cursor.executemany(sql, rows)
    finally:
        cursor.close()
    return len(rows)



def start_collector_run(
    conn,
    source_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> int | None:
    started_at_utc = _utc_now_naive()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO collector_runs (
                source_name, started_at_utc, finished_at_utc, status,
                start_date, end_date, rows_fetched, rows_upserted, error_message
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                source_name,
                started_at_utc,
                None,
                "running",
                _date_value(start_date) if start_date else None,
                _date_value(end_date) if end_date else None,
                0,
                0,
                None,
            ),
        )
        return getattr(cursor, "lastrowid", None)
    finally:
        cursor.close()


def finish_collector_run(
    conn,
    run_id: int | None,
    status: str,
    rows_fetched: int = 0,
    rows_upserted: int = 0,
    error_message: str | None = None,
) -> None:
    if run_id is None:
        return
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            UPDATE collector_runs
               SET finished_at_utc = %s,
                   status = %s,
                   rows_fetched = %s,
                   rows_upserted = %s,
                   error_message = %s
             WHERE id = %s
            """,
            (_utc_now_naive(), status, rows_fetched, rows_upserted, error_message, run_id),
        )
    finally:
        cursor.close()


def record_manual_source_file(
    conn,
    source_name: str,
    file_path: str,
    file_hash_sha256: str,
    rows_loaded: int,
    effective_start_date: str | pd.Timestamp | None = None,
    effective_end_date: str | pd.Timestamp | None = None,
    run_id: int | None = None,
) -> int | None:
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO manual_source_files (
                source_name, file_path, file_hash_sha256, ingested_at_utc,
                effective_start_date, effective_end_date, rows_loaded, run_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                source_name,
                file_path,
                file_hash_sha256,
                _utc_now_naive(),
                _date_value(effective_start_date) if effective_start_date is not None else None,
                _date_value(effective_end_date) if effective_end_date is not None else None,
                rows_loaded,
                run_id,
            ),
        )
        return getattr(cursor, "lastrowid", None)
    finally:
        cursor.close()


def record_production_model_version(
    conn,
    version: str,
    trained_at_utc: datetime,
    train_start_date: str | pd.Timestamp | None = None,
    train_end_date: str | pd.Timestamp | None = None,
    data_max_date_by_source: dict[str, str] | None = None,
    metrics_path: str | None = None,
    models_path: str | None = None,
    promoted: bool = False,
    promotion_reason: str | None = None,
) -> None:
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO production_model_versions (
                version, trained_at_utc, train_start_date, train_end_date,
                data_max_date_by_source, metrics_path, models_path,
                promoted, promotion_reason
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                trained_at_utc=VALUES(trained_at_utc),
                train_start_date=VALUES(train_start_date),
                train_end_date=VALUES(train_end_date),
                data_max_date_by_source=VALUES(data_max_date_by_source),
                metrics_path=VALUES(metrics_path),
                models_path=VALUES(models_path),
                promoted=VALUES(promoted),
                promotion_reason=VALUES(promotion_reason)
            """,
            (
                version,
                trained_at_utc.replace(tzinfo=None),
                _date_value(train_start_date) if train_start_date is not None else None,
                _date_value(train_end_date) if train_end_date is not None else None,
                json.dumps(data_max_date_by_source or {}, ensure_ascii=False),
                metrics_path,
                models_path,
                promoted,
                promotion_reason,
            ),
        )
    finally:
        cursor.close()


def record_collector_run(
    conn,
    source_name: str,
    status: str,
    started_at_utc: datetime,
    start_date: str | None = None,
    end_date: str | None = None,
    rows_fetched: int = 0,
    rows_upserted: int = 0,
    error_message: str | None = None,
) -> int | None:
    finished_at_utc = _utc_now_naive()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO collector_runs (
                source_name, started_at_utc, finished_at_utc, status,
                start_date, end_date, rows_fetched, rows_upserted, error_message
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                source_name,
                started_at_utc,
                finished_at_utc,
                status,
                _date_value(start_date) if start_date else None,
                _date_value(end_date) if end_date else None,
                rows_fetched,
                rows_upserted,
                error_message,
            ),
        )
        return getattr(cursor, "lastrowid", None)
    finally:
        cursor.close()
