import unittest

import pandas as pd

from src.experiments.split_2024_holdout_2025 import (
    TRAIN_END,
    build_feature_row_for_manual_inputs,
    compose_daily_forecast,
)


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


if __name__ == "__main__":
    unittest.main()
