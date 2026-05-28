# Tests/test_pipeline_components.py
# ==============================================================
# Pytest tests for each production pipeline component.
# Uses minimal synthetic data — no external API calls.
# ==============================================================
from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_dirs(tmp_path):
    """Create temporary directory structure mimicking the project."""
    (tmp_path / "data" / "outputs").mkdir(parents=True)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "logs").mkdir(parents=True)
    (tmp_path / "models_saved" / "versioned").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def synthetic_features_df():
    """Minimal synthetic features DataFrame with targets."""
    dates = pd.date_range("2020-01-01", periods=1000, freq="D")
    np.random.seed(42)
    price = np.cumsum(np.random.randn(1000)) + 300
    df = pd.DataFrame(
        {
            "preco_boi_gordo": price,
            "preco_bezerro": price * 0.4,
            "preco_milho": price * 0.1,
            "preco_boi_gordo_lag1d": np.roll(price, 1),
            "preco_boi_gordo_lag7d": np.roll(price, 7),
            "preco_boi_gordo_ma7d": pd.Series(price).rolling(7, min_periods=1).mean().values,
            "preco_bezerro_lag1d": np.roll(price * 0.4, 1),
            "preco_milho_lag1d": np.roll(price * 0.1, 1),
            "mes": dates.month,
            "mes_sin": np.sin(2 * np.pi * dates.month / 12),
            "mes_cos": np.cos(2 * np.pi * dates.month / 12),
        },
        index=dates,
    )
    for h in [1, 7, 15, 30, 60]:
        df[f"target_h{h}d"] = df["preco_boi_gordo"].shift(-h)
    return df


@pytest.fixture
def synthetic_clean_df():
    """Minimal clean DataFrame."""
    dates = pd.date_range("2020-01-01", periods=1000, freq="D")
    np.random.seed(42)
    price = np.cumsum(np.random.randn(1000)) + 300
    return pd.DataFrame({"preco_boi_gordo": price}, index=dates)


# ── Test 1: pipeline_runner ────────────────────────────────────
class TestPipelineRunner:
    def test_run_collectors_with_mock(self, tmp_dirs):
        """run_collectors runs without error when collectors are mocked."""
        mock_df = pd.DataFrame(
            {"preco_boi_gordo": [300.0]},
            index=pd.DatetimeIndex(["2025-01-01"]),
        )

        with patch("config.settings.LAST_RUN_PATH", tmp_dirs / "data" / "last_run.json"):
            from src.collectors.pipeline_runner import run_collectors, COLLECTORS

            # Mock all collectors to return a simple df
            mock_collectors = [(name, lambda sd, d=mock_df: d) for name, _ in COLLECTORS]
            with patch("src.collectors.pipeline_runner.COLLECTORS", mock_collectors):
                results = run_collectors("2025-01-01")

        assert len(results) == len(mock_collectors)
        for name, df in results.items():
            assert not df.empty

    def test_read_last_run_missing(self, tmp_dirs):
        """read_last_run returns defaults when file doesn't exist."""
        with patch("config.settings.LAST_RUN_PATH", tmp_dirs / "nonexistent.json"):
            from src.collectors.pipeline_runner import read_last_run

            result = read_last_run()
            assert "last_date" in result


# ── Test 2: daily_cron ─────────────────────────────────────────
class TestDailyCron:
    def test_step_helper_logs_success(self, tmp_dirs):
        """_step runs a function and returns its result."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

        with patch("config.settings.LOGS_DIR", tmp_dirs / "logs"):
            from daily_cron import _step

            result = _step("test_step", lambda: 42)
            assert result == 42

    def test_step_helper_handles_failure(self, tmp_dirs):
        """_step returns None on non-critical failure."""
        with patch("config.settings.LOGS_DIR", tmp_dirs / "logs"):
            from daily_cron import _step

            result = _step("failing_step", lambda: 1 / 0, critical=False)
            assert result is None


# ── Test 3: retrain ────────────────────────────────────────────
class TestRetrain:
    def test_retrain_versioning(self, tmp_dirs, synthetic_features_df):
        """retrain_with_versioning creates versioned directory and models."""
        models_dir = tmp_dirs / "models_saved"
        versioned_dir = models_dir / "versioned"

        with (
            patch("config.settings.MODELS_DIR", models_dir),
            patch("config.settings.MODELS_VERSIONED_DIR", versioned_dir),
            patch("src.models.retrain.MODELS_DIR", models_dir),
            patch("src.models.retrain.MODELS_VERSIONED_DIR", versioned_dir),
            patch("src.models.train.MODELS_DIR", models_dir),
            patch("src.models.train.DATA_PROCESSED", tmp_dirs / "data"),
        ):
            from src.models.retrain import retrain_with_versioning

            results = retrain_with_versioning(synthetic_features_df)

        assert isinstance(results, dict)
        assert len(results) > 0
        # Check that at least one version directory was created
        version_dirs = [d for d in versioned_dir.iterdir() if d.is_dir()]
        assert len(version_dirs) >= 1


# ── Test 4: export ─────────────────────────────────────────────
class TestExport:
    def test_export_predictions(self, tmp_dirs, synthetic_features_df):
        """export_predictions creates predictions.csv with correct columns."""
        import joblib

        models_dir = tmp_dirs / "models_saved"
        outputs_dir = tmp_dirs / "data" / "outputs"

        # Create minimal mock models
        from sklearn.linear_model import LinearRegression

        for h in [1, 7, 15, 30, 60]:
            feat_cols = [c for c in synthetic_features_df.columns if not c.startswith("target_")]
            model = LinearRegression()
            X = synthetic_features_df[feat_cols].fillna(0).values[:100]
            y = np.random.randn(100)
            model.fit(X, y)
            joblib.dump(model, models_dir / f"xgboost_h{h}d.joblib")
            joblib.dump(model, models_dir / f"random_forest_h{h}d.joblib")
            joblib.dump(feat_cols, models_dir / f"feature_cols_h{h}d.joblib")
            joblib.dump(np.zeros(len(feat_cols)), models_dir / f"feature_medians_h{h}d.joblib")

        with (
            patch("src.export.export_outputs.MODELS_DIR", models_dir),
            patch("src.export.export_outputs.DATA_OUTPUTS", outputs_dir),
        ):
            from src.export.export_outputs import export_predictions

            df = export_predictions(synthetic_features_df)

        assert not df.empty
        assert set(df.columns) == {"date", "horizon", "predicted_value", "model"}
        assert (outputs_dir / "predictions.csv").exists()

    def test_export_price_history(self, tmp_dirs, synthetic_clean_df):
        """export_price_history creates price_history.csv with correct columns."""
        outputs_dir = tmp_dirs / "data" / "outputs"

        with patch("src.export.export_outputs.DATA_OUTPUTS", outputs_dir):
            from src.export.export_outputs import export_price_history

            df = export_price_history(synthetic_clean_df)

        assert not df.empty
        assert "real_price_deflated" in df.columns
        assert (outputs_dir / "price_history.csv").exists()


# ── Test 5: Streamlit app reads CSVs ──────────────────────────
class TestStreamlitApp:
    def test_load_predictions_from_csv(self, tmp_dirs):
        """App loads predictions from CSV correctly."""
        outputs_dir = tmp_dirs / "data" / "outputs"
        pred_df = pd.DataFrame({
            "date": ["2025-06-01", "2025-06-07"],
            "horizon": [1, 7],
            "predicted_value": [310.5, 312.0],
            "model": ["xgboost", "xgboost"],
        })
        pred_df.to_csv(outputs_dir / "predictions.csv", index=False)

        with patch("app_split_2024_holdout_2025.DATA_OUTPUTS", outputs_dir):
            from app_split_2024_holdout_2025 import _load_predictions

            loaded = _load_predictions()

        assert len(loaded) == 2
        assert list(loaded.columns) == ["date", "horizon", "predicted_value", "model"]

    def test_load_price_history_from_csv(self, tmp_dirs):
        """App loads price history from CSV correctly."""
        outputs_dir = tmp_dirs / "data" / "outputs"
        hist_df = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=5),
            "real_price_deflated": [300, 301, 302, 303, 304],
        })
        hist_df.to_csv(outputs_dir / "price_history.csv", index=False)

        with patch("app_split_2024_holdout_2025.DATA_OUTPUTS", outputs_dir):
            from app_split_2024_holdout_2025 import _load_price_history

            loaded = _load_price_history()

        assert len(loaded) == 5
        assert "real_price_deflated" in loaded.columns

    def test_last_run_timestamp(self, tmp_dirs):
        """App reads last run timestamp from JSON."""
        last_run_path = tmp_dirs / "data" / "last_run.json"
        last_run_path.write_text(
            json.dumps({"last_date": "2025-05-27", "timestamp": "2025-05-27T10:00:00"}),
            encoding="utf-8",
        )

        with patch("app_split_2024_holdout_2025.LAST_RUN_PATH", last_run_path):
            from app_split_2024_holdout_2025 import _read_last_run_timestamp

            ts = _read_last_run_timestamp()

        assert ts == "2025-05-27T10:00:00"
