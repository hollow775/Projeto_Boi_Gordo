import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.experiments.raw_data_charts import save_raw_charts
from src.experiments.split_2024_holdout_2025 import MANUAL_INPUT_COLUMNS


class RawDataChartsGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_tmp = Path("Tests/_tmp/raw_charts")
        cls.repo_tmp.mkdir(parents=True, exist_ok=True)

    def test_save_raw_charts_creates_pngs_for_all_variables_plus_consolidated(self) -> None:
        index = pd.date_range("2024-01-01", periods=12, freq="MS")
        raw_df = pd.DataFrame(index=index)
        for idx, column in enumerate(MANUAL_INPUT_COLUMNS):
            raw_df[column] = [float((idx + 1) * 10 + step) for step in range(len(index))]

        output_dir = self.repo_tmp / f"case_{uuid4().hex}" / "data" / "raw" / "graficos_dados_brutos"
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = save_raw_charts(raw_df=raw_df, output_dir=output_dir)
        self.assertEqual(len(paths), len(MANUAL_INPUT_COLUMNS) + 1)
        self.assertTrue((output_dir / "todas_variaveis_dados_brutos.png").exists())
        for column in MANUAL_INPUT_COLUMNS:
            expected = output_dir / f"{column}_dados_brutos.png"
            self.assertTrue(expected.exists(), f"Grafico ausente para {column}")


if __name__ == "__main__":
    unittest.main()
