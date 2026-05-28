import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.models.evaluate import metrics_summary
import src.models.train as train_module
from src.models.train import (
    _apply_imputation_medians,
    _fit_imputation_medians,
    _purge_train_rows_with_targets_in_test_window,
)


class TrainingLeakageGuardTests(unittest.TestCase):
    def test_purge_drops_training_rows_whose_target_date_falls_in_test_window(self):
        dates = pd.date_range("2025-01-01", periods=10, freq="D")
        train_idx = np.arange(0, 5)
        test_idx = np.arange(5, 8)

        purged = _purge_train_rows_with_targets_in_test_window(
            train_idx=train_idx,
            test_idx=test_idx,
            dates=dates,
            horizon=3,
        )

        self.assertEqual(purged.tolist(), [0, 1])
        self.assertEqual(test_idx.tolist(), [5, 6, 7])

    def test_imputation_medians_are_fit_from_training_values_only(self):
        X_train = np.array([[1.0, np.nan], [3.0, np.nan]])
        X_test = np.array([[np.nan, np.nan]])

        medians = _fit_imputation_medians(X_train)
        X_test_imputed = _apply_imputation_medians(X_test, medians)

        self.assertEqual(medians.tolist(), [2.0, 0.0])
        self.assertEqual(X_test_imputed.tolist(), [[2.0, 0.0]])


class EvaluationSummaryTests(unittest.TestCase):
    def test_metrics_summary_excludes_tuning_fold(self):
        results = {
            1: {
                "tuning_fold": 1,
                "metricas_cv_xgboost": [
                    {"fold": 1, "used_for_tuning": True, "RMSE": 100.0, "MAE": 100.0, "MAPE": 100.0},
                    {"fold": 2, "used_for_tuning": False, "RMSE": 2.0, "MAE": 2.0, "MAPE": 2.0},
                ],
                "metricas_cv_random_forest": [
                    {"fold": 1, "used_for_tuning": True, "RMSE": 50.0, "MAE": 50.0, "MAPE": 50.0},
                    {"fold": 2, "used_for_tuning": False, "RMSE": 3.0, "MAE": 3.0, "MAPE": 3.0},
                ],
            }
        }

        summary = metrics_summary(results)

        self.assertEqual(summary["fold"].tolist(), [2, 2])
        self.assertNotIn(100.0, summary["MAPE"].tolist())
        self.assertNotIn(50.0, summary["MAPE"].tolist())


class BaselineSelectionTests(unittest.TestCase):
    def test_baseline_choice_uses_training_error_not_test_error(self):
        class FakeRegressor:
            def __init__(self, **_kwargs):
                pass

            def fit(self, _X, _y):
                return self

            def predict(self, X):
                return np.zeros(len(X))

        index = pd.date_range("2024-01-01", periods=8, freq="D")
        df = pd.DataFrame(
            {
                "preco_boi_gordo": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
                "feature_a": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0],
                "target_h1d": [10.0, 20.0, 30.0, 40.0, 30.0, 35.0, np.nan, np.nan],
            },
            index=index,
        )

        with patch.object(train_module, "_purged_walk_forward_splits", return_value=[(np.arange(0, 4), np.arange(4, 6))]), patch.object(
            train_module,
            "_tune_with_budget",
            return_value={
                "xgboost": train_module.XGBOOST_PARAMS.copy(),
                "random_forest": train_module.RANDOM_FOREST_PARAMS.copy(),
            },
        ), patch.object(train_module, "XGBRegressor", FakeRegressor), patch.object(
            train_module, "RandomForestRegressor", FakeRegressor
        ), patch.object(train_module.joblib, "dump"):
            result = train_module.train_horizon(df, 1)

        selected = result["out_of_fold_dataframe"]["previsao_baseline"].tolist()
        self.assertEqual(selected, [50.0, 60.0])


if __name__ == "__main__":
    unittest.main()
