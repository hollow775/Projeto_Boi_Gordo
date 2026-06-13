from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import DATE_RANGE
from src.collectors.cepea import PRODUCT_MAP, _read_cepea_sheet
from src.production.cepea_manual import configured_cepea_manual_files, record_manual_cepea_file
from src.production.collectors import production_date_range, refresh_api_sources
from src.production.db import create_schema, managed_connection
from src.production.export import export_production_outputs
from src.production.policy import get_production_policy
from src.production.retrain import train_production_models


def _today_iso() -> str:
    return date.today().isoformat()


def write_production_last_run(policy, payload: dict) -> None:
    policy.last_run_path.parent.mkdir(parents=True, exist_ok=True)
    policy.last_run_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_production_features(start_date: str, end_date: str):
    from src.features.engineering import build_features
    from src.processing.cleaner import clean
    from src.processing.merger import build_dataset

    with production_date_range(start_date, end_date):
        raw_df = build_dataset()
    if raw_df.empty:
        raise ValueError("Dataset vazio após coleta/build_dataset.")
    clean_df = clean(raw_df, exclude_holdout=False)
    features_df = build_features(clean_df.copy())
    return features_df, clean_df


def _cepea_loader_for(source_name: str):
    product = source_name.removeprefix("cepea_")
    column_name = PRODUCT_MAP[product]

    def load_one_file(path):
        return _read_cepea_sheet(path, column_name).to_frame()

    return load_one_file


def record_configured_cepea_manual_files(conn) -> list[dict]:
    records = []
    for source_name, file_path in configured_cepea_manual_files().items():
        record = record_manual_cepea_file(
            conn,
            source_name=source_name,
            file_path=file_path,
            loader=_cepea_loader_for(source_name),
        )
        records.append(
            {
                "source_name": record.source_name,
                "file_path": str(record.file_path),
                "file_hash_sha256": record.file_hash_sha256,
                "rows_loaded": record.rows_loaded,
                "effective_start_date": None
                if record.effective_start_date is None
                else record.effective_start_date.strftime("%Y-%m-%d"),
                "effective_end_date": None
                if record.effective_end_date is None
                else record.effective_end_date.strftime("%Y-%m-%d"),
            }
        )
    return records


def assert_refresh_succeeded(refresh_results: dict) -> None:
    failed = {
        source_name: result.get("error", "unknown error")
        for source_name, result in refresh_results.items()
        if result.get("status") != "success"
    }
    if failed:
        details = "; ".join(f"{source}: {error}" for source, error in failed.items())
        raise RuntimeError(f"Production API refresh failed; aborting train/export: {details}")


def run_production_daily(
    start_date: str,
    end_date: str,
    *,
    init_db: bool = False,
    skip_refresh: bool = False,
    skip_train: bool = False,
    skip_export: bool = False,
) -> dict:
    policy = get_production_policy()
    refresh_results = {}
    cepea_manual_records = []
    training_run = None

    with managed_connection() as conn:
        if init_db:
            create_schema(conn)
        if not skip_refresh:
            refresh_results = refresh_api_sources(conn, start_date=start_date, end_date=end_date)
            assert_refresh_succeeded(refresh_results)
        cepea_manual_records = record_configured_cepea_manual_files(conn)

        features_df, clean_df = build_production_features(start_date, end_date)
        if not skip_train:
            training_run = train_production_models(
                features_df,
                conn=conn,
                policy=policy,
                data_max_date_by_source={
                    "features": features_df.index.max().strftime("%Y-%m-%d"),
                    **{
                        source_name: end_date
                        for source_name, result in refresh_results.items()
                        if result.get("status") == "success"
                    },
                },
            )
        if not skip_export:
            export_production_outputs(features_df, clean_df, policy=policy)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "start_date": start_date,
        "end_date": end_date,
        "refresh_results": refresh_results,
        "cepea_manual_records": cepea_manual_records,
        "training_version": None if training_run is None else training_run.version,
        "metrics_path": None if training_run is None else str(training_run.metrics_path),
        "models_path": None if training_run is None else str(training_run.version_path),
    }
    write_production_last_run(policy, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local-first production update: refresh API sources in MySQL, retrain production models, export website outputs.",
    )
    parser.add_argument("--start-date", default=DATE_RANGE["start"], help="Initial collection date, YYYY-MM-DD.")
    parser.add_argument("--end-date", default=_today_iso(), help="Final collection date, YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--init-db", action="store_true", help="Create/update production MySQL schema before running.")
    parser.add_argument("--skip-refresh", action="store_true", help="Skip API collectors/MySQL refresh.")
    parser.add_argument("--skip-train", action="store_true", help="Skip production model training/versioning.")
    parser.add_argument("--skip-export", action="store_true", help="Skip website output export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_production_daily(
        start_date=args.start_date,
        end_date=args.end_date,
        init_db=args.init_db,
        skip_refresh=args.skip_refresh,
        skip_train=args.skip_train,
        skip_export=args.skip_export,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
