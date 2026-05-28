# daily_cron.py
# ==============================================================
# Standalone scheduler: collect → features → retrain → export.
# Logs each step to logs/pipeline.log.
# Aborts on critical failures (empty dataset, all-NaN targets).
# ==============================================================
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import LOGS_DIR

# ── Logging setup ──────────────────────────────────────────────
LOG_PATH = LOGS_DIR / "pipeline.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("daily_cron")


def _step(name: str, fn, *, critical: bool = False):
    """Run a pipeline step, log timing and errors."""
    log.info(f"STEP START: {name}")
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        log.info(f"STEP OK: {name} ({elapsed:.1f}s)")
        return result
    except Exception as e:
        elapsed = time.time() - t0
        log.error(f"STEP FAILED: {name} ({elapsed:.1f}s) — {e}")
        if critical:
            log.critical(f"CRITICAL FAILURE in {name}. Aborting pipeline.")
            raise SystemExit(1)
        return None


def step_collect():
    from src.collectors.pipeline_runner import run_collectors
    return run_collectors()


def step_build_features():
    from src.processing.merger import build_dataset
    from src.processing.cleaner import clean
    from src.features.engineering import build_features

    raw_df = build_dataset()
    if raw_df.empty:
        raise ValueError("Dataset vazio após coleta — abortando.")

    clean_df = clean(raw_df, exclude_holdout=False)
    features_df = build_features(clean_df.copy())

    # Critical check: all targets NaN means no usable data
    target_cols = [c for c in features_df.columns if c.startswith("target_h")]
    if features_df[target_cols].notna().sum().sum() == 0:
        raise ValueError("Todos os targets são NaN — dados insuficientes.")

    return features_df, clean_df


def step_retrain(features_df):
    from src.models.retrain import retrain_with_versioning
    return retrain_with_versioning(features_df)


def step_export(features_df, clean_df):
    from src.export.export_outputs import export_all
    export_all(features_df, clean_df)


def main():
    log.info("=" * 60)
    log.info(f"PIPELINE START: {datetime.utcnow().isoformat()}")
    log.info("=" * 60)

    _step("collect", step_collect, critical=False)

    result = _step(
        "build_features",
        step_build_features,
        critical=True,
    )
    features_df, clean_df = result

    _step("retrain", lambda: step_retrain(features_df), critical=False)
    _step("export", lambda: step_export(features_df, clean_df), critical=False)

    log.info(f"PIPELINE END: {datetime.utcnow().isoformat()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
