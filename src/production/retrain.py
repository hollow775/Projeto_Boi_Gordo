from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.evaluate import metrics_mean, metrics_summary
from src.production.db import record_production_model_version
from src.production.policy import ProductionPolicy, get_production_policy


@dataclass(frozen=True)
class ProductionTrainingRun:
    version: str
    trained_at_utc: datetime
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    version_path: Path
    metrics_path: Path
    metrics_summary_path: Path
    promoted: bool


def version_stamp(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%S")


def versioned_models_dir(policy: ProductionPolicy) -> Path:
    return policy.models_dir / "versioned"


def production_version_dir(policy: ProductionPolicy, version: str) -> Path:
    return versioned_models_dir(policy) / version


def latest_pointer_path(policy: ProductionPolicy) -> Path:
    return versioned_models_dir(policy) / "latest.json"


def _serialisable_results(results: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Strip fitted model objects/dataframes while keeping metrics for audit JSON."""
    payload: dict[str, Any] = {}
    for horizon, result in results.items():
        payload[str(horizon)] = {
            "metricas_cv_xgboost": result.get("metricas_cv_xgboost", []),
            "metricas_cv_random_forest": result.get("metricas_cv_random_forest", []),
            "metricas_cv_baseline": result.get("metricas_cv_baseline", []),
            "feature_cols": result.get("feature_cols", []),
            "tuning_fold": result.get("tuning_fold"),
        }
    return payload


def write_production_metrics(
    policy: ProductionPolicy,
    version: str,
    results: dict[int, dict[str, Any]],
) -> tuple[Path, Path]:
    policy.processed_dir.mkdir(parents=True, exist_ok=True)
    mean_path = policy.processed_dir / f"metricas_producao_{version}.csv"
    summary_path = policy.processed_dir / f"metricas_producao_cv_{version}.csv"
    latest_mean_path = policy.processed_dir / "metricas_producao.csv"
    latest_summary_path = policy.processed_dir / "metricas_producao_cv.csv"
    json_path = policy.processed_dir / f"metricas_producao_{version}.json"

    mean_df = metrics_mean(results)
    summary_df = metrics_summary(results)
    mean_df.to_csv(mean_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    mean_df.to_csv(latest_mean_path, index=False)
    summary_df.to_csv(latest_summary_path, index=False)
    json_path.write_text(
        json.dumps(_serialisable_results(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return mean_path, summary_path


def promote_version(policy: ProductionPolicy, version_path: Path) -> None:
    policy.models_dir.mkdir(parents=True, exist_ok=True)
    for path in version_path.iterdir():
        if path.is_file():
            shutil.copy2(path, policy.models_dir / path.name)


def write_latest_pointer(
    policy: ProductionPolicy,
    run: ProductionTrainingRun,
    metrics_path: Path,
) -> None:
    pointer_path = latest_pointer_path(policy)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(
        json.dumps(
            {
                "version": run.version,
                "trained_at_utc": run.trained_at_utc.isoformat(),
                "train_start_date": run.train_start_date.strftime("%Y-%m-%d"),
                "train_end_date": run.train_end_date.strftime("%Y-%m-%d"),
                "metrics_path": str(metrics_path),
                "models_path": str(run.version_path),
                "promoted": run.promoted,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def train_production_models(
    features_df: pd.DataFrame,
    conn=None,
    policy: ProductionPolicy | None = None,
    version: str | None = None,
    data_max_date_by_source: dict[str, str] | None = None,
    promote: bool = True,
) -> ProductionTrainingRun:
    """Train/version the production model on all available rows.

    This function is intentionally separate from the TCC/academic scripts: it
    accepts the already-built production features, removes the academic 2025
    cutoff through explicit train_all parameters, stores models under
    models_saved/production, and writes production metrics separately under
    data/processed/production.
    """
    if features_df.empty:
        raise ValueError("features_df is empty; production training needs data.")
    if not isinstance(features_df.index, pd.DatetimeIndex):
        raise ValueError("features_df must use a DatetimeIndex.")

    policy = policy or get_production_policy()
    policy.ensure_directories()
    version = version or version_stamp()
    trained_at_utc = datetime.now(timezone.utc)
    version_path = production_version_dir(policy, version)
    version_path.mkdir(parents=True, exist_ok=True)

    train_start_date = pd.Timestamp(features_df.index.min())
    train_end_date = pd.Timestamp(features_df.index.max())

    from src.models.train import train_all

    results = train_all(
        features_df.sort_index(),
        models_dir=version_path,
        data_processed_dir=policy.processed_dir,
        cutoff_date=None,
    )

    metrics_path, metrics_summary_path = write_production_metrics(policy, version, results)

    run = ProductionTrainingRun(
        version=version,
        trained_at_utc=trained_at_utc,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        version_path=version_path,
        metrics_path=metrics_path,
        metrics_summary_path=metrics_summary_path,
        promoted=promote,
    )

    if promote:
        promote_version(policy, version_path)
    write_latest_pointer(policy, run, metrics_path)

    if conn is not None:
        record_production_model_version(
            conn,
            version=version,
            trained_at_utc=trained_at_utc,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            data_max_date_by_source=data_max_date_by_source or {"features": train_end_date.strftime("%Y-%m-%d")},
            metrics_path=str(metrics_path),
            models_path=str(version_path),
            promoted=promote,
            promotion_reason="production retrain completed",
        )

    return run
