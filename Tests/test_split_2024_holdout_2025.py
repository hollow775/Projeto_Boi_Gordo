import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import app_split_2024_holdout_2025 as ui_app
from src.experiments.split_2024_holdout_2025 import (
    TRAIN_END,
    build_feature_row_for_manual_inputs,
    compose_daily_forecast,
    get_experiment_paths,
    predict_manual_curve,
    save_ui_reference_artifacts,
)

TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "Tests" / "_tmp"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _fresh_tmp_dir(name: str) -> Path:
    path = TEST_TMP_ROOT / name
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


class ComposeDailyForecastTests(unittest.TestCase):
    def test_compose_daily_forecast_uses_expected_horizon_blocks(self):
        forecast = compose_daily_forecast(
            anchor_predictions={1: 100.0, 7: 110.0, 15: 120.0},
            forecast_start=pd.Timestamp("2026-01-01"),
        )

        self.assertEqual(len(forecast), 15)
        self.assertEqual(forecast.loc[0, "valor_previsto"], 100.0)
        self.assertTrue((forecast.loc[1:6, "valor_previsto"] == 110.0).all())
        self.assertTrue((forecast.loc[7:, "valor_previsto"] == 120.0).all())
        self.assertEqual(forecast.loc[0, "modelo_origem"], "h1")
        self.assertEqual(forecast.loc[1, "modelo_origem"], "h7")
        self.assertEqual(forecast.loc[7, "modelo_origem"], "h15")


class BuildFeatureRowForManualInputsTests(unittest.TestCase):
    def test_manual_row_is_appended_on_requested_date(self):
        history_index = pd.date_range("2024-12-25", periods=10, freq="D")
        history_df = pd.DataFrame(
            {
                "preco_boi_gordo": [300 + i for i in range(10)],
                "preco_bezerro": [2200 + i for i in range(10)],
                "preco_milho": [70 + i for i in range(10)],
                "abate_cabecas": [1000 + i for i in range(10)],
                "abate_peso_ton": [500 + i for i in range(10)],
                "export_usd_fob": [2000 + i for i in range(10)],
                "export_kg": [3000 + i for i in range(10)],
                "precipitacao_mm": [40 + i for i in range(10)],
                "inflation_index": [100 + i for i in range(10)],
                "cotacao_dolar_venda": [5 + i / 100 for i in range(10)],
            },
            index=history_index,
        )

        feature_row = build_feature_row_for_manual_inputs(
            clean_history_df=history_df,
            manual_inputs={
                "preco_boi_gordo": 350.0,
                "preco_bezerro": 2300.0,
                "preco_milho": 80.0,
                "abate_cabecas": 1100.0,
                "abate_peso_ton": 550.0,
                "export_usd_fob": 2100.0,
                "export_kg": 3100.0,
                "precipitacao_mm": 55.0,
                "inflation_index": 105.0,
                "cotacao_dolar_venda": 5.5,
            },
            forecast_base_date=TRAIN_END,
        )

        self.assertEqual(len(feature_row), 1)
        self.assertEqual(feature_row.index[0], TRAIN_END)
        self.assertEqual(feature_row.loc[TRAIN_END, "preco_bezerro"], 2300.0)
        self.assertIn("target_h1d", feature_row.columns)


class ExperimentPathsAndArtifactsTests(unittest.TestCase):
    def test_get_experiment_paths_creates_separate_directories(self):
        tmp_dir = _fresh_tmp_dir("paths")
        paths = get_experiment_paths(tmp_dir)

        self.assertTrue(paths.processed_dir.exists())
        self.assertTrue(paths.models_dir.exists())
        self.assertIn("train_split_2024_holdout_2025", str(paths.processed_dir))
        self.assertIn("train_split_2024_holdout_2025", str(paths.models_dir))

    def test_save_ui_reference_artifacts_exports_example_and_recent_history(self):
        tmp_dir = _fresh_tmp_dir("artifacts")
        paths = get_experiment_paths(tmp_dir)
        index = pd.date_range("2024-09-05", periods=120, freq="D")
        clean_df = pd.DataFrame(
            {
                "preco_boi_gordo": range(120),
                "preco_bezerro": range(120, 240),
                "preco_milho": range(240, 360),
                "abate_cabecas": range(360, 480),
                "abate_peso_ton": range(480, 600),
                "export_usd_fob": range(600, 720),
                "export_kg": range(720, 840),
                "precipitacao_mm": range(840, 960),
                "inflation_index": range(960, 1080),
                "cotacao_dolar_venda": range(1080, 1200),
            },
            index=index,
        )

        save_ui_reference_artifacts(clean_df, paths=paths)

        self.assertTrue(paths.example_values_path.exists())
        self.assertTrue(paths.ui_history_path.exists())

        example_df = pd.read_csv(paths.example_values_path, index_col="data", parse_dates=True)
        history_df = pd.read_csv(paths.ui_history_path, index_col="data", parse_dates=True)

        self.assertEqual(example_df.index[0], TRAIN_END)
        self.assertEqual(len(history_df), 120)
        self.assertIn("preco_boi_gordo", history_df.columns)


class PredictManualCurveTests(unittest.TestCase):
    def test_predict_manual_curve_returns_three_anchor_models_and_daily_curve(self):
        feature_row = pd.DataFrame({"f1": [1.0]}, index=[TRAIN_END])
        clean_history_df = pd.DataFrame({"preco_boi_gordo": [300.0]}, index=[TRAIN_END - pd.Timedelta(days=1)])

        class FakeModel:
            def __init__(self, value):
                self.value = value

            def predict(self, _):
                return [self.value]

        def fake_load_model(_paths, model_type, horizon):
            base = {1: 10.0, 7: 20.0, 15: 30.0}[horizon]
            if model_type == "random_forest":
                base += 2.0
            return FakeModel(base)

        with patch(
            "src.experiments.split_2024_holdout_2025.build_feature_row_for_manual_inputs",
            return_value=feature_row,
        ), patch(
            "src.experiments.split_2024_holdout_2025._load_feature_columns",
            return_value=["f1"],
        ), patch(
            "src.experiments.split_2024_holdout_2025._load_model",
            side_effect=fake_load_model,
        ):
            curve_df, anchors_df = predict_manual_curve(
                clean_history_df=clean_history_df,
                manual_inputs={"preco_boi_gordo": 310.0},
                forecast_base_date=TRAIN_END,
                model_type="media_modelos",
            )

        self.assertEqual(len(anchors_df), 3)
        self.assertEqual(len(curve_df), 15)
        self.assertEqual(curve_df.loc[0, "valor_previsto"], 11.0)
        self.assertEqual(curve_df.loc[1, "valor_previsto"], 21.0)
        self.assertEqual(curve_df.loc[7, "valor_previsto"], 31.0)


class UiManualValueParsingTests(unittest.TestCase):
    def test_parse_manual_values_accepts_comma_decimal(self):
        parsed = ui_app._parse_manual_values({"preco_boi_gordo": "123,45"})
        self.assertEqual(parsed["preco_boi_gordo"], 123.45)

    def test_parse_manual_values_rejects_missing_and_invalid_fields(self):
        with self.assertRaises(ValueError) as exc:
            ui_app._parse_manual_values(
                {
                    "preco_boi_gordo": "",
                    "preco_milho": "abc",
                }
            )

        self.assertIn("Campos obrigatorios vazios", str(exc.exception))
        self.assertIn("Campos invalidos", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
