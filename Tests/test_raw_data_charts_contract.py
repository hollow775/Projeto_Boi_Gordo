import re
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.split_2024_holdout_2025 import MANUAL_INPUT_COLUMNS


def _candidate_python_files() -> list[Path]:
    return [
        path
        for path in REPO_ROOT.rglob("*.py")
        if not {".omx", ".git", "__pycache__", "Tests"}.intersection(path.parts)
    ]


def _raw_chart_sources() -> list[tuple[Path, str]]:
    matches: list[tuple[Path, str]] = []
    for path in _candidate_python_files():
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if "graficos_dados_brutos" in text:
            matches.append((path, text))
    return matches


class RawDataChartsContractTests(unittest.TestCase):
    def test_raw_chart_entrypoint_contract_is_ready_for_implementation(self) -> None:
        raw_sources = _raw_chart_sources()
        if not raw_sources:
            self.skipTest("Raw-data chart entrypoint not implemented in this branch yet.")

        combined_source = "\n".join(text for _, text in raw_sources)
        self.assertIn("data/raw/graficos_dados_brutos", combined_source)
        self.assertIn(".png", combined_source)

    def test_raw_chart_flow_references_all_current_prediction_variables(self) -> None:
        raw_sources = _raw_chart_sources()
        if not raw_sources:
            self.skipTest("Raw-data chart generation flow not implemented in this branch yet.")

        combined_source = "\n".join(text for _, text in raw_sources)
        missing_variables = [
            variable for variable in MANUAL_INPUT_COLUMNS if variable not in combined_source
        ]
        self.assertEqual(
            missing_variables,
            [],
            f"Raw-data chart flow should cover all prediction variables: {missing_variables}",
        )

    def test_raw_chart_flow_avoids_cleaning_and_deflation_calls(self) -> None:
        raw_sources = _raw_chart_sources()
        if not raw_sources:
            self.skipTest("Raw-data chart generation flow not implemented in this branch yet.")

        combined_source = "\n".join(text for _, text in raw_sources)
        forbidden_patterns = [
            r"\bclean\s*\(",
            r"defla",
            r"winsor",
            r"imput",
        ]
        for pattern in forbidden_patterns:
            self.assertIsNone(
                re.search(pattern, combined_source),
                f"Raw-data chart generation must not call cleaning/deflation logic: {pattern}",
            )

    def test_raw_chart_flow_documents_a_reproducible_command(self) -> None:
        raw_sources = _raw_chart_sources()
        if not raw_sources:
            self.skipTest("Raw-data chart generation flow not implemented in this branch yet.")

        combined_source = "\n".join(text for _, text in raw_sources)
        self.assertTrue(
            "__main__" in combined_source or "argparse" in combined_source,
            "Raw-data chart flow should expose a reproducible script/command entrypoint.",
        )

    def test_raw_and_holdout_charts_use_two_year_date_ticks(self) -> None:
        raw_source = (REPO_ROOT / "src" / "experiments" / "raw_data_charts.py").read_text(
            encoding="utf-8",
            errors="ignore",
        )
        split_source = (REPO_ROOT / "src" / "experiments" / "split_2024_holdout_2025.py").read_text(
            encoding="utf-8",
            errors="ignore",
        )
        for source in (raw_source, split_source):
            self.assertIn("YearLocator(base=2)", source)
            self.assertIn('DateFormatter("%Y")', source)


if __name__ == "__main__":
    unittest.main()
