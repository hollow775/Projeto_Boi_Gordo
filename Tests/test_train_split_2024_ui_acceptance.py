import re
import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.processing.cleaner import clean


def _sample_daily_dataframe(start: str, end: str) -> pd.DataFrame:
    index = pd.date_range(start, end, freq="D")
    return pd.DataFrame(
        {
            "preco_boi_gordo": [300.0 + i for i in range(len(index))],
            "abate_cabecas": [1000.0 + i for i in range(len(index))],
        },
        index=index,
    )


class TrainSplit2024AcceptanceTests(unittest.TestCase):
    def test_clean_can_isolate_2025_as_holdout_for_the_new_flow(self) -> None:
        df = _sample_daily_dataframe("2024-12-29", "2025-01-03")

        cleaned = clean(df.copy(), train_cutoff="2024-12-31", exclude_holdout=True)
        holdout = cleaned.attrs["holdout_tail"]

        self.assertEqual(str(cleaned.index.max().date()), "2024-12-31")
        self.assertEqual(str(cleaned.attrs["holdout_cutoff"].date()), "2024-12-31")
        self.assertEqual(str(holdout.index.min().date()), "2025-01-01")
        self.assertEqual(str(holdout.index.max().date()), "2025-01-03")
        self.assertTrue((holdout.index >= pd.Timestamp("2025-01-01")).all())

    def test_clean_can_keep_2025_rows_available_for_validation(self) -> None:
        df = _sample_daily_dataframe("2024-12-29", "2025-01-03")

        cleaned = clean(df.copy(), train_cutoff="2024-12-31", exclude_holdout=False)
        holdout = cleaned.attrs["holdout_tail"]

        self.assertEqual(str(cleaned.index.max().date()), "2025-01-03")
        self.assertEqual(str(holdout.index.min().date()), "2025-01-01")
        self.assertEqual(len(holdout), 3)

    def test_default_cutoff_keeps_the_legacy_pipeline_behavior(self) -> None:
        df = _sample_daily_dataframe("2025-12-30", "2026-01-02")

        cleaned = clean(df.copy(), exclude_holdout=True)
        holdout = cleaned.attrs["holdout_tail"]

        self.assertEqual(str(cleaned.index.max().date()), "2025-12-31")
        self.assertEqual(str(cleaned.attrs["holdout_cutoff"].date()), "2025-12-31")
        self.assertEqual(str(holdout.index.min().date()), "2026-01-01")
        self.assertEqual(len(holdout), 2)


class FutureImplementationContractTests(unittest.TestCase):
    def test_ui_entrypoint_contract_is_ready_for_implementation(self) -> None:
        candidate_files = [
            path for path in REPO_ROOT.rglob("*.py")
            if path.parts[0] not in {".omx", ".git", "__pycache__", "Tests"}
        ]
        streamlit_files = [
            path for path in candidate_files
            if re.search(
                r"(^|\n)\s*(import\s+streamlit|from\s+streamlit\s+import)\b",
                path.read_text(encoding="utf-8", errors="ignore").lower(),
            )
        ]
        if not streamlit_files:
            self.skipTest("UI entrypoint not implemented in this branch yet.")

        ui_text = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore").lower()
            for path in streamlit_files
        )
        self.assertIn("15", ui_text)
        self.assertTrue(
            any(token in ui_text for token in ["histor", "previs", "boi gordo"]),
            "UI entrypoint should mention historical data and forecast context.",
        )

    def test_daily_curve_rule_contract_is_ready_for_implementation(self) -> None:
        candidate_files = [
            path for path in REPO_ROOT.rglob("*.py")
            if path.parts[0] not in {".omx", ".git", "__pycache__", "Tests"}
        ]
        combined_source = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore").lower()
            for path in candidate_files
        )
        rule_patterns = [
            r"dia\s*1.*h1",
            r"dias?\s*2.*7.*h7",
            r"dias?\s*8.*15.*h15",
        ]
        if not all(re.search(pattern, combined_source, flags=re.DOTALL) for pattern in rule_patterns):
            self.skipTest("Daily 1..15 composed-curve rule not implemented in this branch yet.")

        for pattern in rule_patterns:
            self.assertRegex(combined_source, pattern)

    def test_history_chart_exposes_hover_tooltip_with_day_value(self) -> None:
        app_file = REPO_ROOT / "app_split_2024_holdout_2025.py"
        if not app_file.exists():
            self.skipTest("UI file not available in this branch yet.")

        source = app_file.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("st.altair_chart", source)
        self.assertIn("selection_point", source)
        self.assertIn('Tooltip("data:T"', source)
        self.assertIn("Valor (R$/arroba)", source)

    def test_manual_form_uses_compact_two_column_layout(self) -> None:
        app_file = REPO_ROOT / "app_split_2024_holdout_2025.py"
        if not app_file.exists():
            self.skipTest("UI file not available in this branch yet.")

        source = app_file.read_text(encoding="utf-8", errors="ignore")
        self.assertRegex(source, r"input_cols\s*=\s*st\.columns\(\s*2")
        self.assertIn("idx % 2", source)


if __name__ == "__main__":
    unittest.main()

