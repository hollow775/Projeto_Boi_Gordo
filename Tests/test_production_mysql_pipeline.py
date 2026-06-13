import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.production.db import (
    SCHEMA_STATEMENTS,
    create_schema,
    finish_collector_run,
    load_mysql_config,
    record_collector_run,
    record_manual_source_file,
    record_production_model_version,
    start_collector_run,
    upsert_source_dataframe,
)
from src.production.cepea_manual import (
    configured_cepea_manual_files,
    record_manual_cepea_file,
    sha256_file,
)
from src.production.collectors import ProductionSourceSpec, production_date_range, refresh_api_sources
from src.production.export import export_production_outputs
from src.production.policy import get_production_policy
from src.production.retrain import train_production_models
from config import settings


class FakeCursor:
    def __init__(self):
        self.statements = []
        self.executemany_calls = []
        self.closed = False
        self.lastrowid = 42

    def execute(self, sql, params=None):
        self.statements.append((" ".join(sql.split()), params))

    def executemany(self, sql, rows):
        self.executemany_calls.append((" ".join(sql.split()), list(rows)))

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self):
        self.cursor_obj = FakeCursor()

    def cursor(self):
        return self.cursor_obj


class ProductionPolicyTests(unittest.TestCase):
    def test_production_policy_uses_separate_namespace_and_no_training_cutoff(self):
        root = Path("/tmp/boi-gordo-test")
        policy = get_production_policy(root)

        self.assertEqual(policy.slug, "production")
        self.assertIsNone(policy.training_cutoff)
        self.assertFalse(policy.allow_future_holdout_tail)
        self.assertEqual(policy.processed_dir, root / "data" / "processed" / "production")
        self.assertEqual(policy.models_dir, root / "models_saved" / "production")
        self.assertEqual(policy.outputs_dir, root / "data" / "outputs" / "production")
        self.assertEqual(policy.last_run_path, root / "data" / "production_last_run.json")


class MySQLConfigTests(unittest.TestCase):
    def test_load_mysql_config_requires_all_env_vars(self):
        with self.assertRaises(RuntimeError) as exc:
            load_mysql_config({})

        self.assertIn("BOI_DB_HOST", str(exc.exception))
        self.assertIn("BOI_DB_PASSWORD", str(exc.exception))

    def test_load_mysql_config_from_env_mapping(self):
        config = load_mysql_config(
            {
                "BOI_DB_HOST": "localhost",
                "BOI_DB_PORT": "3306",
                "BOI_DB_NAME": "boi_gordo",
                "BOI_DB_USER": "user",
                "BOI_DB_PASSWORD": "secret",
            }
        )

        self.assertEqual(config.host, "localhost")
        self.assertEqual(config.port, 3306)
        self.assertEqual(config.database, "boi_gordo")
        self.assertEqual(config.user, "user")
        self.assertEqual(config.password, "secret")


class SchemaTests(unittest.TestCase):
    def test_create_schema_executes_expected_tables(self):
        conn = FakeConnection()
        create_schema(conn)

        executed_sql = "\n".join(sql for sql, _params in conn.cursor_obj.statements)
        self.assertEqual(len(conn.cursor_obj.statements), len(SCHEMA_STATEMENTS))
        self.assertIn("CREATE TABLE IF NOT EXISTS collector_runs", executed_sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS source_ptax_daily", executed_sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS manual_source_files", executed_sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS production_model_versions", executed_sql)
        self.assertTrue(conn.cursor_obj.closed)


class UpsertTests(unittest.TestCase):
    def test_upsert_source_dataframe_uses_duplicate_key_update(self):
        conn = FakeConnection()
        df = pd.DataFrame(
            {"cotacao_dolar_venda": [5.1, 5.2]},
            index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
        )

        rows = upsert_source_dataframe(conn, "source_ptax_daily", df, run_id=7)

        self.assertEqual(rows, 2)
        sql, values = conn.cursor_obj.executemany_calls[0]
        self.assertIn("INSERT INTO source_ptax_daily", sql)
        self.assertIn("ON DUPLICATE KEY UPDATE", sql)
        self.assertEqual(len(values), 2)
        self.assertEqual(values[0][0], pd.Timestamp("2026-01-01").date())
        self.assertEqual(values[0][1], 5.1)
        self.assertEqual(values[0][-1], 7)
        self.assertTrue(conn.cursor_obj.closed)

    def test_upsert_empty_dataframe_is_noop(self):
        conn = FakeConnection()
        rows = upsert_source_dataframe(conn, "source_ptax_daily", pd.DataFrame(), run_id=1)

        self.assertEqual(rows, 0)
        self.assertEqual(conn.cursor_obj.executemany_calls, [])

    def test_upsert_rejects_unknown_table(self):
        with self.assertRaises(ValueError):
            upsert_source_dataframe(FakeConnection(), "source_cepea_scraping", pd.DataFrame())


class CollectorRunTests(unittest.TestCase):
    def test_start_and_finish_collector_run_record_running_then_terminal_status(self):
        conn = FakeConnection()
        run_id = start_collector_run(
            conn,
            source_name="ptax",
            start_date="2026-01-01",
            end_date="2026-01-02",
        )
        finish_collector_run(
            conn,
            run_id,
            status="success",
            rows_fetched=2,
            rows_upserted=2,
        )

        self.assertEqual(run_id, 42)
        insert_sql, insert_params = conn.cursor_obj.statements[0]
        update_sql, update_params = conn.cursor_obj.statements[1]
        self.assertIn("INSERT INTO collector_runs", insert_sql)
        self.assertEqual(insert_params[3], "running")
        self.assertIn("UPDATE collector_runs", update_sql)
        self.assertEqual(update_params[1], "success")
        self.assertEqual(update_params[2], 2)
        self.assertEqual(update_params[3], 2)
        self.assertEqual(update_params[-1], 42)

    def test_record_collector_run_records_audit_metadata(self):
        conn = FakeConnection()
        run_id = record_collector_run(
            conn,
            source_name="ptax",
            status="success",
            started_at_utc=datetime(2026, 1, 2, 3, 4, 5),
            start_date="2026-01-01",
            end_date="2026-01-02",
            rows_fetched=2,
            rows_upserted=2,
        )

        self.assertEqual(run_id, 42)
        sql, params = conn.cursor_obj.statements[0]
        self.assertIn("INSERT INTO collector_runs", sql)
        self.assertEqual(params[0], "ptax")
        self.assertEqual(params[3], "success")
        self.assertEqual(params[6], 2)
        self.assertEqual(params[7], 2)
        self.assertTrue(conn.cursor_obj.closed)


class ProductionModelVersionTests(unittest.TestCase):
    def test_record_production_model_version_upserts_metadata(self):
        conn = FakeConnection()

        record_production_model_version(
            conn,
            version="20260102T030405",
            trained_at_utc=datetime(2026, 1, 2, 3, 4, 5),
            train_start_date="2010-01-01",
            train_end_date="2026-01-02",
            data_max_date_by_source={"ptax": "2026-01-02"},
            metrics_path="data/processed/production/metricas_producao.csv",
            models_path="models_saved/production/versioned/20260102T030405",
            promoted=True,
            promotion_reason="ok",
        )

        sql, params = conn.cursor_obj.statements[0]
        self.assertIn("INSERT INTO production_model_versions", sql)
        self.assertIn("ON DUPLICATE KEY UPDATE", sql)
        self.assertEqual(params[0], "20260102T030405")
        self.assertEqual(params[3], pd.Timestamp("2026-01-02").date())
        self.assertIn('"ptax": "2026-01-02"', params[4])
        self.assertTrue(params[7])


class ManualCepeaProvenanceTests(unittest.TestCase):
    def test_record_manual_source_file_writes_manifest_row(self):
        conn = FakeConnection()
        manifest_id = record_manual_source_file(
            conn,
            source_name="cepea_boi_gordo",
            file_path="data/raw/cepea_boi_gordo.xlsx",
            file_hash_sha256="a" * 64,
            rows_loaded=10,
            effective_start_date="2024-01-01",
            effective_end_date="2024-01-10",
            run_id=5,
        )

        self.assertEqual(manifest_id, 42)
        sql, params = conn.cursor_obj.statements[0]
        self.assertIn("INSERT INTO manual_source_files", sql)
        self.assertEqual(params[0], "cepea_boi_gordo")
        self.assertEqual(params[2], "a" * 64)
        self.assertEqual(params[6], 10)
        self.assertEqual(params[7], 5)

    def test_record_manual_cepea_file_hashes_and_records_effective_range(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cepea.csv"
            path.write_text("data,valor\n2026-01-01,300\n", encoding="utf-8")
            expected_hash = sha256_file(path)

            def loader(_path):
                return pd.DataFrame(
                    {"preco_boi_gordo": [300.0, 301.0]},
                    index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
                )

            conn = FakeConnection()
            record = record_manual_cepea_file(
                conn,
                source_name="cepea_boi_gordo",
                file_path=path,
                loader=loader,
                run_id=9,
            )

        self.assertEqual(record.source_name, "cepea_boi_gordo")
        self.assertEqual(record.file_hash_sha256, expected_hash)
        self.assertEqual(record.rows_loaded, 2)
        self.assertEqual(record.effective_start_date, pd.Timestamp("2026-01-01"))
        self.assertEqual(record.effective_end_date, pd.Timestamp("2026-01-02"))
        self.assertEqual(record.manifest_id, 42)
        sql, params = conn.cursor_obj.statements[0]
        self.assertIn("INSERT INTO manual_source_files", sql)
        self.assertEqual(params[7], 9)

    def test_configured_cepea_manual_files_declares_three_manual_sources(self):
        files = configured_cepea_manual_files()

        self.assertEqual(
            set(files),
            {"cepea_boi_gordo", "cepea_bezerro", "cepea_milho"},
        )


class ProductionCollectorRefreshTests(unittest.TestCase):
    def test_production_date_range_temporarily_overrides_shared_date_range(self):
        original = settings.DATE_RANGE.copy()

        with production_date_range("2026-01-01", "2026-01-03"):
            self.assertEqual(settings.DATE_RANGE["start"], "2026-01-01")
            self.assertEqual(settings.DATE_RANGE["end"], "2026-01-03")

        self.assertEqual(settings.DATE_RANGE, original)

    def test_refresh_api_sources_upserts_configured_api_sources_without_cepea(self):
        conn = FakeConnection()

        def load_ptax():
            return pd.DataFrame(
                {"cotacao_dolar_venda": [5.1, 5.2]},
                index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
            )

        def load_inflation():
            return pd.DataFrame(
                {"inflation_index": [100.0]},
                index=pd.to_datetime(["2026-01-01"]),
            )

        sources = (
            ProductionSourceSpec("ptax", "source_ptax_daily", load_ptax),
            ProductionSourceSpec("inflation", "source_inflation_index", load_inflation),
        )

        results = refresh_api_sources(
            conn,
            start_date="2026-01-01",
            end_date="2026-01-02",
            sources=sources,
        )

        self.assertEqual(set(results), {"ptax", "inflation"})
        self.assertNotIn("cepea", results)
        self.assertEqual(results["ptax"]["status"], "success")
        self.assertEqual(results["ptax"]["rows_upserted"], 2)
        self.assertEqual(results["inflation"]["rows_upserted"], 1)

        executed_sql = "\n".join(sql for sql, _params in conn.cursor_obj.statements)
        self.assertIn("INSERT INTO collector_runs", executed_sql)
        self.assertIn("UPDATE collector_runs", executed_sql)
        self.assertEqual(len(conn.cursor_obj.executemany_calls), 2)

    def test_refresh_api_sources_records_failed_source_without_raising(self):
        conn = FakeConnection()

        def broken_loader():
            raise RuntimeError("api unavailable")

        sources = (
            ProductionSourceSpec("ptax", "source_ptax_daily", broken_loader),
        )

        results = refresh_api_sources(
            conn,
            start_date="2026-01-01",
            end_date="2026-01-02",
            sources=sources,
        )

        self.assertEqual(results["ptax"]["status"], "failed")
        self.assertIn("api unavailable", results["ptax"]["error"])
        update_sql, update_params = conn.cursor_obj.statements[1]
        self.assertIn("UPDATE collector_runs", update_sql)
        self.assertEqual(update_params[1], "failed")
        self.assertIn("api unavailable", update_params[4])


class ProductionRetrainTests(unittest.TestCase):
    def test_train_all_accepts_explicit_production_paths_without_mutating_globals(self):
        import tempfile
        import src.models.train as train_module

        original_models_dir = train_module.MODELS_DIR
        original_data_processed = train_module.DATA_PROCESSED
        original_cutoff_date = train_module.CUTOFF_DATE

        with tempfile.TemporaryDirectory() as tmpdir:
            policy = get_production_policy(Path(tmpdir))
            version_path = policy.models_dir / "versioned" / "v1"
            version_path.mkdir(parents=True)

            self.assertIsNone(train_module._assert_training_cutoff_for(
                pd.DataFrame(index=pd.to_datetime(["2026-01-01"])),
                cutoff_date=None,
            ))

        self.assertEqual(train_module.MODELS_DIR, original_models_dir)
        self.assertEqual(train_module.DATA_PROCESSED, original_data_processed)
        self.assertEqual(train_module.CUTOFF_DATE, original_cutoff_date)

    def test_train_production_models_versions_models_and_writes_separate_metrics(self):
        import tempfile
        from unittest.mock import patch
        import src.models.train as train_module

        with tempfile.TemporaryDirectory() as tmpdir:
            policy = get_production_policy(Path(tmpdir))
            features = pd.DataFrame(
                {
                    "preco_boi_gordo": [300.0, 301.0, 302.0],
                    "target_h1d": [301.0, 302.0, None],
                },
                index=pd.to_datetime(["2025-12-31", "2026-01-01", "2026-01-02"]),
            )

            def fake_train_all(received_df, **_kwargs):
                self.assertEqual(received_df.index.max(), pd.Timestamp("2026-01-02"))
                (policy.models_dir / "versioned" / "vtest" / "xgboost_h1d.joblib").write_text("model", encoding="utf-8")
                return {
                    1: {
                        "metricas_cv_xgboost": [
                            {"fold": 1, "used_for_tuning": True, "RMSE": 3.0, "MAE": 2.0, "MAPE": 1.0},
                            {"fold": 2, "used_for_tuning": False, "RMSE": 2.0, "MAE": 1.0, "MAPE": 0.5},
                        ],
                        "metricas_cv_random_forest": [
                            {"fold": 1, "used_for_tuning": True, "RMSE": 4.0, "MAE": 3.0, "MAPE": 2.0},
                            {"fold": 2, "used_for_tuning": False, "RMSE": 3.0, "MAE": 2.0, "MAPE": 1.5},
                        ],
                        "feature_cols": ["preco_boi_gordo"],
                        "tuning_fold": 1,
                    }
                }

            conn = FakeConnection()
            with patch("src.models.train.train_all", side_effect=fake_train_all) as train_all_mock:
                run = train_production_models(
                    features,
                    conn=conn,
                    policy=policy,
                    version="vtest",
                    data_max_date_by_source={"features": "2026-01-02"},
                )

            _, kwargs = train_all_mock.call_args
            self.assertEqual(kwargs["models_dir"], policy.models_dir / "versioned" / "vtest")
            self.assertEqual(kwargs["data_processed_dir"], policy.processed_dir)
            self.assertIsNone(kwargs["cutoff_date"])
            self.assertEqual(run.train_end_date, pd.Timestamp("2026-01-02"))
            self.assertTrue((policy.models_dir / "versioned" / "vtest" / "xgboost_h1d.joblib").exists())
            self.assertTrue((policy.models_dir / "xgboost_h1d.joblib").exists())
            self.assertTrue((policy.processed_dir / "metricas_producao_vtest.csv").exists())
            self.assertTrue((policy.processed_dir / "metricas_producao.csv").exists())
            self.assertTrue((policy.models_dir / "versioned" / "latest.json").exists())
            sql, params = conn.cursor_obj.statements[0]
            self.assertIn("INSERT INTO production_model_versions", sql)
            self.assertEqual(params[0], "vtest")
            self.assertEqual(params[3], pd.Timestamp("2026-01-02").date())


class ProductionWebsiteOutputTests(unittest.TestCase):
    def test_app_prefers_production_outputs_and_computes_change_signals(self):
        import tempfile
        from unittest.mock import patch
        from app_split_2024_holdout_2025 import (
            _compute_history_change_signals,
            _load_predictions,
            _load_price_history,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir) / "data" / "outputs"
            production_dir = outputs_dir / "production"
            production_dir.mkdir(parents=True)
            outputs_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                {
                    "date": ["2026-01-03"],
                    "horizon": [1],
                    "predicted_value": [321.0],
                    "model": ["xgboost"],
                }
            ).to_csv(production_dir / "predictions.csv", index=False)
            pd.DataFrame(
                {
                    "date": pd.date_range("2025-12-27", periods=8),
                    "real_price_deflated": [300, 301, 302, 303, 304, 305, 306, 308],
                }
            ).to_csv(production_dir / "price_history.csv", index=False)

            with patch("app_split_2024_holdout_2025.DATA_OUTPUTS", outputs_dir):
                predictions = _load_predictions()
                history = _load_price_history()
                signals = _compute_history_change_signals(history)

        self.assertEqual(predictions["date"].tolist(), ["2026-01-03"])
        self.assertEqual(str(history.index.max().date()), "2026-01-03")
        self.assertEqual(signals["latest_date"], "2026-01-03")
        self.assertEqual(signals["delta_1d"], 2.0)
        self.assertEqual(signals["delta_7d"], 8.0)

    def test_change_signals_handle_zero_week_base_without_percentage(self):
        from app_split_2024_holdout_2025 import _compute_history_change_signals

        history = pd.DataFrame(
            {"real_price_deflated": [0, 1, 2, 3, 4, 5, 6, 8]},
            index=pd.date_range("2026-01-01", periods=8),
        )

        signals = _compute_history_change_signals(history)

        self.assertEqual(signals["delta_7d"], 8.0)
        self.assertIsNone(signals["pct_7d"])

    def test_export_production_outputs_uses_explicit_production_dirs(self):
        import tempfile
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            policy = get_production_policy(Path(tmpdir))
            with patch("src.export.export_outputs.export_all") as export_all_mock:
                export_production_outputs(pd.DataFrame(), pd.DataFrame(), policy=policy)

            _, kwargs = export_all_mock.call_args
            self.assertEqual(kwargs["models_dir"], policy.models_dir)
            self.assertEqual(kwargs["data_outputs"], policy.outputs_dir)


class ProductionDailyCommandTests(unittest.TestCase):
    def test_write_production_last_run_writes_scheduler_state_file(self):
        import json
        import tempfile
        from production_daily import write_production_last_run

        with tempfile.TemporaryDirectory() as tmpdir:
            policy = get_production_policy(Path(tmpdir))
            write_production_last_run(
                policy,
                {
                    "timestamp": "2026-01-02T03:04:05+00:00",
                    "training_version": "vtest",
                },
            )

            payload = json.loads(policy.last_run_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["training_version"], "vtest")
        self.assertEqual(payload["timestamp"], "2026-01-02T03:04:05+00:00")

    def test_assert_refresh_succeeded_fails_on_failed_source(self):
        from production_daily import assert_refresh_succeeded

        with self.assertRaises(RuntimeError) as exc:
            assert_refresh_succeeded({"ptax": {"status": "failed", "error": "api down"}})

        self.assertIn("aborting train/export", str(exc.exception))
        self.assertIn("ptax", str(exc.exception))

    def test_build_production_features_applies_requested_date_range(self):
        from unittest.mock import patch
        from production_daily import build_production_features

        raw = pd.DataFrame(
            {"preco_boi_gordo": [300.0], "inflation_index": [100.0]},
            index=pd.to_datetime(["2026-01-02"]),
        )
        with patch("src.processing.merger.build_dataset", return_value=raw) as build_dataset_mock, patch(
            "src.processing.cleaner.clean", side_effect=lambda df, **_kwargs: df
        ), patch("src.features.engineering.build_features", side_effect=lambda df: df):
            features_df, clean_df = build_production_features("2026-01-01", "2026-01-02")

        self.assertEqual(settings.DATE_RANGE["end"], "2025-12-31")
        self.assertEqual(build_dataset_mock.call_count, 1)
        self.assertEqual(features_df.index.max(), pd.Timestamp("2026-01-02"))
        self.assertEqual(clean_df.index.max(), pd.Timestamp("2026-01-02"))

    def test_record_configured_cepea_manual_files_records_all_configured_sources(self):
        from unittest.mock import patch
        from production_daily import record_configured_cepea_manual_files

        with patch(
            "production_daily.configured_cepea_manual_files",
            return_value={
                "cepea_boi_gordo": Path("boi.xlsx"),
                "cepea_bezerro": Path("bezerro.xlsx"),
                "cepea_milho": Path("milho.xlsx"),
            },
        ), patch("production_daily.record_manual_cepea_file") as record_mock:
            record_mock.side_effect = [
                type("Record", (), {
                    "source_name": "cepea_boi_gordo",
                    "file_path": Path("boi.xlsx"),
                    "file_hash_sha256": "a" * 64,
                    "rows_loaded": 1,
                    "effective_start_date": pd.Timestamp("2026-01-01"),
                    "effective_end_date": pd.Timestamp("2026-01-01"),
                })(),
                type("Record", (), {
                    "source_name": "cepea_bezerro",
                    "file_path": Path("bezerro.xlsx"),
                    "file_hash_sha256": "b" * 64,
                    "rows_loaded": 1,
                    "effective_start_date": pd.Timestamp("2026-01-01"),
                    "effective_end_date": pd.Timestamp("2026-01-01"),
                })(),
                type("Record", (), {
                    "source_name": "cepea_milho",
                    "file_path": Path("milho.xlsx"),
                    "file_hash_sha256": "c" * 64,
                    "rows_loaded": 1,
                    "effective_start_date": pd.Timestamp("2026-01-01"),
                    "effective_end_date": pd.Timestamp("2026-01-01"),
                })(),
            ]

            records = record_configured_cepea_manual_files(FakeConnection())

        self.assertEqual([record["source_name"] for record in records], [
            "cepea_boi_gordo",
            "cepea_bezerro",
            "cepea_milho",
        ])
        self.assertEqual(record_mock.call_count, 3)


if __name__ == "__main__":
    unittest.main()
