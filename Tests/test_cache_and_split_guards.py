import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import src.collectors.comexstat as comexstat
import src.collectors.copernicus as copernicus
from src.models.train import _walk_forward_splits

TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "Tests" / "_tmp"


def _fresh_tmp_dir(name: str) -> Path:
    path = TEST_TMP_ROOT / name
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


class WalkForwardSplitGuardTests(unittest.TestCase):
    def test_last_fold_includes_remainder_rows(self):
        splits = _walk_forward_splits(n=13, min_train=5, n_folds=3)

        self.assertEqual(splits[0][1].tolist(), [5, 6])
        self.assertEqual(splits[1][1].tolist(), [7, 8])
        self.assertEqual(splits[2][1].tolist(), [9, 10, 11, 12])

    def test_raises_when_not_enough_test_rows_for_minimum_fold_size(self):
        with self.assertRaisesRegex(ValueError, "Dados insuficientes"):
            _walk_forward_splits(n=7, min_train=5, n_folds=3)


class ComexStatCacheValidationTests(unittest.TestCase):
    def test_cache_metadata_must_match_config_and_cover_years(self):
        tmp = _fresh_tmp_dir("comex_cache_validation")
        metadata_path = tmp / "comex.meta.json"
        expected = {
            "start": "2020-01-01",
            "end": "2021-12-31",
            "start_year": 2020,
            "end_year": 2021,
                "ncm_codes": ["0201", "0202"],
            "api_url": "https://example.test",
        }
        df = pd.DataFrame(
            {"export_usd_fob": [1, 2], "export_kg": [3, 4]},
            index=pd.to_datetime(["2020-01-01", "2021-12-01"]),
        )

        with patch.object(comexstat, "CACHE_META_FILE", metadata_path):
            metadata_path.write_text(json.dumps(expected), encoding="utf-8")
            self.assertTrue(comexstat._cache_matches_config(df, expected))

            mismatched = {**expected, "end": "2022-12-31", "end_year": 2022}
            self.assertFalse(comexstat._cache_matches_config(df, mismatched))


class CopernicusCacheValidationTests(unittest.TestCase):
    def test_era5_cache_metadata_must_match_scope(self):
        tmp = _fresh_tmp_dir("era5_cache_validation")
        era5_path = tmp / "era5_sp.nc"
        era5_path.write_bytes(b"CDF")
        expected_scope = {
            "start": "2020-01-01",
            "end": "2021-12-31",
            "dataset": "dataset",
            "variable": "total_precipitation",
            "bbox": [-53.1, -25.3, -44.0, -19.8],
            "area": [-19.8, -53.1, -25.3, -44.0],
            "years": ["2020", "2021"],
            "months": [str(month).zfill(2) for month in range(1, 13)],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

        copernicus._metadata_path(era5_path).write_text(
            json.dumps(expected_scope),
            encoding="utf-8",
        )

        self.assertTrue(copernicus._cache_matches_config(era5_path, expected_scope))
        self.assertFalse(
            copernicus._cache_matches_config(
                era5_path,
                {**expected_scope, "variable": "other_variable"},
            )
        )


if __name__ == "__main__":
    unittest.main()
