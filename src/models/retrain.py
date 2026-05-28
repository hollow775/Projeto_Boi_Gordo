# src/models/retrain.py
# ==============================================================
# Retrain models on full updated dataset with versioning.
# Saves timestamped models, promotes to "latest" only if RMSE
# on most recent fold is within 10% of previous best.
# Keeps only the two most recent versions.
# ==============================================================
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config.settings import HORIZONS, MODELS_DIR, MODELS_VERSIONED_DIR


def _version_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _latest_pointer_path() -> Path:
    return MODELS_VERSIONED_DIR / "latest.json"


def _read_latest_pointer() -> dict | None:
    p = _latest_pointer_path()
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _write_latest_pointer(version: str, metrics: dict) -> None:
    _latest_pointer_path().write_text(
        json.dumps({"version": version, "metrics": metrics}, indent=2),
        encoding="utf-8",
    )


def _version_dir(version: str) -> Path:
    return MODELS_VERSIONED_DIR / version


def _cleanup_old_versions(keep: int = 2) -> None:
    """Delete all but the `keep` most recent version directories."""
    dirs = sorted(
        [d for d in MODELS_VERSIONED_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    for old_dir in dirs[keep:]:
        shutil.rmtree(old_dir)
        print(f"[retrain] Removed old version: {old_dir.name}")


def _last_fold_rmse(results: dict, horizon: int) -> float:
    """Extract RMSE from the last walk-forward fold for a given horizon."""
    cv_metrics = results[horizon]["metricas_cv_xgboost"]
    # Last fold that wasn't used for tuning
    non_tuning = [m for m in cv_metrics if not m.get("used_for_tuning", False)]
    if non_tuning:
        return non_tuning[-1]["RMSE"]
    return cv_metrics[-1]["RMSE"]


def retrain_with_versioning(features_df: pd.DataFrame) -> dict:
    """
    Retrain all horizons from scratch on the full dataset.
    Save versioned models and promote to latest if quality check passes.

    Returns
    -------
    dict with train results keyed by horizon.
    """
    from src.models.train import train_all

    version = _version_stamp()
    version_path = _version_dir(version)
    version_path.mkdir(parents=True, exist_ok=True)

    # Temporarily redirect MODELS_DIR in train module
    import src.models.train as train_module

    original_models_dir = train_module.MODELS_DIR
    train_module.MODELS_DIR = version_path
    try:
        results = train_all(features_df)
    finally:
        train_module.MODELS_DIR = original_models_dir

    # Compute quality metrics for this version
    new_metrics = {
        h: _last_fold_rmse(results, h) for h in HORIZONS if h in results
    }

    # Check against previous latest
    latest = _read_latest_pointer()
    promote = True

    if latest and latest.get("metrics"):
        prev_metrics = latest["metrics"]
        for h_str, prev_rmse in prev_metrics.items():
            h = int(h_str)
            if h in new_metrics:
                threshold = prev_rmse * 1.10
                if new_metrics[h] > threshold:
                    print(
                        f"[retrain] h{h}d RMSE {new_metrics[h]:.4f} > "
                        f"110% of previous {prev_rmse:.4f}. NOT promoting."
                    )
                    promote = False
                    break

    if promote:
        # Copy to main MODELS_DIR as the active models
        for f in version_path.iterdir():
            shutil.copy2(f, MODELS_DIR / f.name)
        _write_latest_pointer(version, {str(h): v for h, v in new_metrics.items()})
        print(f"[retrain] Promoted version {version} to latest.")
    else:
        print(f"[retrain] Version {version} saved but NOT promoted.")

    _cleanup_old_versions(keep=2)
    return results


if __name__ == "__main__":
    from src.processing.merger import build_dataset
    from src.processing.cleaner import clean
    from src.features.engineering import build_features

    raw_df = build_dataset()
    clean_df = clean(raw_df, exclude_holdout=False)
    features_df = build_features(clean_df.copy())
    retrain_with_versioning(features_df)
