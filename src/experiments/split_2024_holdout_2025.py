from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import HORIZONS, ROOT_DIR
from src.features.engineering import build_features


EXPERIMENT_SLUG = "train_split_2024_holdout_2025"
TRAIN_END = pd.Timestamp("2024-12-31")
HOLDOUT_START = pd.Timestamp("2025-01-01")
HOLDOUT_END = pd.Timestamp("2025-12-31")
UI_HORIZON_SOURCES = {1: 1, 7: 7, 15: 15}
MANUAL_INPUT_COLUMNS = [
    "preco_boi_gordo",
    "preco_bezerro",
    "preco_milho",
    "abate_cabecas",
    "abate_peso_ton",
    "export_usd_fob",
    "export_kg",
    "precipitacao_mm",
    "inflation_index",
    "cotacao_dolar_venda",
]
SERIES_LABELS = {
    "preco_boi_gordo": "Preço do boi gordo (R$/arroba)",
    "preco_bezerro": "Preço do bezerro (R$/cabeça)",
    "preco_milho": "Preço do milho (R$/saca)",
    "abate_cabecas": "Abate (cabeças)",
    "abate_peso_ton": "Abate (toneladas)",
    "export_usd_fob": "Exportação FOB (USD)",
    "export_kg": "Exportação (kg)",
    "precipitacao_mm": "Precipitação (mm)",
    "inflation_index": "Índice de inflação",
    "cotacao_dolar_venda": "Dólar PTAX venda",
}
MODEL_OUTPUT_COLUMNS = {
    "xgboost": "previsao_xgboost",
    "random_forest": "previsao_random_forest",
    "media_modelos": "media_modelos",
}


def _joblib():
    import joblib

    return joblib


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]

    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    mae = float(np.mean(np.abs(y_t - y_p)))
    mape = float(np.mean(np.abs((y_t - y_p) / np.where(y_t == 0, np.nan, y_t))) * 100)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


@dataclass(frozen=True)
class ExperimentPaths:
    root_dir: Path
    processed_dir: Path
    models_dir: Path
    charts_dir: Path
    holdout_dir: Path
    cache_train_path: Path
    cache_full_path: Path
    cache_clean_path: Path
    train_results_path: Path
    holdout_metrics_path: Path
    example_values_path: Path
    ui_history_path: Path


def get_experiment_paths(root_dir: Path = ROOT_DIR) -> ExperimentPaths:
    processed_dir = root_dir / "data" / "processed" / EXPERIMENT_SLUG
    models_dir = root_dir / "models_saved" / EXPERIMENT_SLUG
    charts_dir = processed_dir / "graficos"
    holdout_dir = processed_dir / "holdout_2025"

    for directory in (processed_dir, models_dir, charts_dir, holdout_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return ExperimentPaths(
        root_dir=root_dir,
        processed_dir=processed_dir,
        models_dir=models_dir,
        charts_dir=charts_dir,
        holdout_dir=holdout_dir,
        cache_train_path=processed_dir / "dataset_train_features.joblib",
        cache_full_path=processed_dir / "dataset_full_features.joblib",
        cache_clean_path=processed_dir / "dataset_clean_full.joblib",
        train_results_path=processed_dir / "train_results.joblib",
        holdout_metrics_path=processed_dir / "metricas_holdout_2025.csv",
        example_values_path=processed_dir / "exemplo_ultimo_dia_treino.csv",
        ui_history_path=processed_dir / "historico_recente_boi_gordo.csv",
    )


def _existing_manual_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in MANUAL_INPUT_COLUMNS if column in df.columns]


@contextmanager
def configured_training_runtime(paths: ExperimentPaths):
    import src.models.train as train_module

    old_models_dir = train_module.MODELS_DIR
    old_processed_dir = train_module.DATA_PROCESSED
    old_cutoff = train_module.CUTOFF_DATE
    try:
        train_module.MODELS_DIR = paths.models_dir
        train_module.DATA_PROCESSED = paths.processed_dir
        train_module.CUTOFF_DATE = TRAIN_END
        yield train_module
    finally:
        train_module.MODELS_DIR = old_models_dir
        train_module.DATA_PROCESSED = old_processed_dir
        train_module.CUTOFF_DATE = old_cutoff


def _build_feature_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from src.processing.cleaner import clean
    from src.processing.merger import build_dataset

    raw_df = build_dataset()
    clean_full_df = clean(raw_df, train_cutoff=TRAIN_END, exclude_holdout=False)
    train_base_df = clean_full_df.loc[:TRAIN_END].copy()

    train_features_df = build_features(train_base_df)
    full_features_df = build_features(clean_full_df.copy())

    return train_features_df, full_features_df, clean_full_df


def load_or_build_feature_datasets(
    use_cache: bool = True,
    paths: ExperimentPaths | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = paths or get_experiment_paths()
    if (
        use_cache
        and paths.cache_train_path.exists()
        and paths.cache_full_path.exists()
        and paths.cache_clean_path.exists()
    ):
        return (
            _joblib().load(paths.cache_train_path),
            _joblib().load(paths.cache_full_path),
            _joblib().load(paths.cache_clean_path),
        )

    train_features_df, full_features_df, clean_full_df = _build_feature_datasets()
    _joblib().dump(train_features_df, paths.cache_train_path)
    _joblib().dump(full_features_df, paths.cache_full_path)
    _joblib().dump(clean_full_df, paths.cache_clean_path)

    save_ui_reference_artifacts(clean_full_df, paths=paths)
    return train_features_df, full_features_df, clean_full_df


def save_ui_reference_artifacts(
    clean_full_df: pd.DataFrame,
    paths: ExperimentPaths | None = None,
) -> None:
    paths = paths or get_experiment_paths()
    example_columns = _existing_manual_columns(clean_full_df)
    training_slice = clean_full_df.loc[:TRAIN_END]
    if training_slice.empty:
        raise ValueError(
            "Nao foi possivel gerar exemplo da UI: sem dados ate a data final de treino (2024-12-31)."
        )
    example_df = training_slice.tail(1).loc[:, example_columns].copy()
    example_df.index.name = "data"
    example_df.to_csv(paths.example_values_path)

    history_df = clean_full_df.loc[:, ["preco_boi_gordo"]].tail(120).copy()
    history_df.index.name = "data"
    history_df.to_csv(paths.ui_history_path)


def train_experiment(
    train_features_df: pd.DataFrame,
    paths: ExperimentPaths | None = None,
) -> dict[int, dict[str, Any]]:
    paths = paths or get_experiment_paths()
    with configured_training_runtime(paths) as train_module:
        results = train_module.train_all(train_features_df)
    _joblib().dump(results, paths.train_results_path)
    return results


def _load_model(paths: ExperimentPaths, model_type: str, horizon: int):
    return _joblib().load(paths.models_dir / f"{model_type}_h{horizon}d.joblib")


def _load_feature_columns(paths: ExperimentPaths, horizon: int) -> list[str]:
    return _joblib().load(paths.models_dir / f"feature_cols_h{horizon}d.joblib")


def _load_feature_medians(paths: ExperimentPaths, horizon: int) -> np.ndarray:
    return _joblib().load(paths.models_dir / f"feature_medians_h{horizon}d.joblib")


def _fill_with_training_medians(X_df: pd.DataFrame, medians: np.ndarray) -> pd.DataFrame:
    return X_df.fillna(pd.Series(medians, index=X_df.columns))


def evaluate_holdout(
    full_features_df: pd.DataFrame,
    paths: ExperimentPaths | None = None,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    paths = paths or get_experiment_paths()
    summary_rows: list[dict[str, Any]] = []
    prediction_frames: dict[int, pd.DataFrame] = {}

    for horizon in HORIZONS:
        feature_columns = _load_feature_columns(paths, horizon)
        feature_medians = _load_feature_medians(paths, horizon)
        target_column = f"target_h{horizon}d"
        holdout_slice = (
            full_features_df.loc[HOLDOUT_START:HOLDOUT_END]
            .dropna(subset=[target_column])
            .copy()
        )
        holdout_slice = holdout_slice[
            holdout_slice.index + pd.Timedelta(days=horizon) <= HOLDOUT_END
        ]
        if holdout_slice.empty:
            continue

        X_df = holdout_slice[feature_columns].copy()
        X_df = _fill_with_training_medians(X_df, feature_medians)
        X = X_df.values
        y_true = holdout_slice[target_column].values

        prediction_frame = pd.DataFrame(
            {
                "data_base": holdout_slice.index,
                "data_alvo": holdout_slice.index + pd.Timedelta(days=horizon),
                "valor_real": y_true,
                "previsao_xgboost": _load_model(paths, "xgboost", horizon).predict(X),
                "previsao_random_forest": _load_model(paths, "random_forest", horizon).predict(X),
            }
        )
        prediction_frame["media_modelos"] = (
            prediction_frame["previsao_xgboost"] + prediction_frame["previsao_random_forest"]
        ) / 2
        prediction_frame["horizonte_dias"] = horizon
        prediction_frames[horizon] = prediction_frame

        for model_name, column in MODEL_OUTPUT_COLUMNS.items():
            metrics = _compute_metrics(y_true, prediction_frame[column].values)
            summary_rows.append(
                {
                    "periodo": "holdout_2025",
                    "horizonte_dias": horizon,
                    "modelo": model_name,
                    **metrics,
                }
            )

        prediction_frame.to_csv(
            paths.holdout_dir / f"predicoes_holdout_2025_h{horizon}d.csv",
            index=False,
        )
        _plot_holdout_predictions(prediction_frame, horizon, paths)

    metrics_df = pd.DataFrame(summary_rows).round(4)
    metrics_df.to_csv(paths.holdout_metrics_path, index=False)
    return metrics_df, prediction_frames


def _plot_holdout_predictions(
    prediction_frame: pd.DataFrame,
    horizon: int,
    paths: ExperimentPaths,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        prediction_frame["data_alvo"],
        prediction_frame["valor_real"],
        color="#1f1f1f",
        linewidth=2,
        label="Valor real",
    )
    axis.plot(
        prediction_frame["data_alvo"],
        prediction_frame["previsao_xgboost"],
        color="#2b9957",
        linewidth=1.4,
        linestyle="--",
        label="XGBoost",
    )
    axis.plot(
        prediction_frame["data_alvo"],
        prediction_frame["previsao_random_forest"],
        color="#e06f00",
        linewidth=1.4,
        linestyle="--",
        label="Random Forest",
    )
    axis.plot(
        prediction_frame["data_alvo"],
        prediction_frame["media_modelos"],
        color="#6d7a32",
        linewidth=1.6,
        alpha=0.9,
        label="Média dos modelos",
    )
    axis.set_title(f"Holdout 2025 - previsão vs. real (h={horizon}d)")
    axis.set_xlabel("Data alvo")
    axis.set_ylabel("Preço real (R$/arroba)")
    axis.xaxis.set_major_locator(mdates.YearLocator(base=2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(paths.holdout_dir / f"previsao_vs_real_holdout_2025_h{horizon}d.png", dpi=150)
    plt.close(figure)


def compose_daily_forecast(
    anchor_predictions: dict[int, float],
    forecast_start: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    for day_offset in range(1, 16):
        source_horizon = 1 if day_offset == 1 else 7 if day_offset <= 7 else 15
        rows.append(
            {
                "dia": day_offset,
                "data_previsao": forecast_start + pd.Timedelta(days=day_offset - 1),
                "valor_previsto": float(anchor_predictions[source_horizon]),
                "modelo_origem": f"h{source_horizon}",
            }
        )
    return pd.DataFrame(rows)


def build_feature_row_for_manual_inputs(
    clean_history_df: pd.DataFrame,
    manual_inputs: dict[str, float],
    forecast_base_date: str | pd.Timestamp,
) -> pd.DataFrame:
    forecast_base_date = pd.Timestamp(forecast_base_date)
    history_df = clean_history_df.copy()
    if forecast_base_date in history_df.index:
        history_df = history_df.drop(index=forecast_base_date)

    manual_columns = _existing_manual_columns(history_df)
    new_row = {column: np.nan for column in history_df.columns}
    for column in manual_columns:
        new_row[column] = manual_inputs[column]

    augmented_df = pd.concat(
        [history_df, pd.DataFrame([new_row], index=[forecast_base_date])],
        axis=0,
    ).sort_index()
    feature_df = build_features(augmented_df)
    return feature_df.loc[[forecast_base_date]]


def predict_manual_curve(
    clean_history_df: pd.DataFrame,
    manual_inputs: dict[str, float],
    forecast_base_date: str | pd.Timestamp,
    model_type: str = "media_modelos",
    paths: ExperimentPaths | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = paths or get_experiment_paths()
    feature_row = build_feature_row_for_manual_inputs(
        clean_history_df=clean_history_df,
        manual_inputs=manual_inputs,
        forecast_base_date=forecast_base_date,
    )
    anchor_predictions: dict[int, float] = {}
    anchor_rows: list[dict[str, Any]] = []

    for source_horizon in sorted(set(UI_HORIZON_SOURCES.values())):
        feature_columns = _load_feature_columns(paths, source_horizon)
        feature_medians = _load_feature_medians(paths, source_horizon)
        X_df = feature_row[feature_columns].copy()
        X_df = _fill_with_training_medians(X_df, feature_medians)
        X = X_df.values

        prediction_xgboost = float(_load_model(paths, "xgboost", source_horizon).predict(X)[0])
        prediction_random_forest = float(_load_model(paths, "random_forest", source_horizon).predict(X)[0])
        prediction_mean = (prediction_xgboost + prediction_random_forest) / 2

        anchor_rows.append(
            {
                "horizonte_modelo": source_horizon,
                "previsao_xgboost": prediction_xgboost,
                "previsao_random_forest": prediction_random_forest,
                "media_modelos": prediction_mean,
            }
        )
        anchor_predictions[source_horizon] = {
            "xgboost": prediction_xgboost,
            "random_forest": prediction_random_forest,
            "media_modelos": prediction_mean,
        }[model_type]

    curve_df = compose_daily_forecast(
        anchor_predictions=anchor_predictions,
        forecast_start=pd.Timestamp(forecast_base_date) + pd.Timedelta(days=1),
    )
    anchors_df = pd.DataFrame(anchor_rows)
    return curve_df, anchors_df


def run_full_experiment(
    use_cache: bool = True,
    paths: ExperimentPaths | None = None,
) -> dict[str, Any]:
    paths = paths or get_experiment_paths()
    train_features_df, full_features_df, clean_full_df = load_or_build_feature_datasets(
        use_cache=use_cache,
        paths=paths,
    )
    save_ui_reference_artifacts(clean_full_df, paths=paths)
    train_results = train_experiment(train_features_df, paths=paths)
    holdout_metrics_df, prediction_frames = evaluate_holdout(full_features_df, paths=paths)
    return {
        "train_features_df": train_features_df,
        "full_features_df": full_features_df,
        "clean_full_df": clean_full_df,
        "train_results": train_results,
        "holdout_metrics_df": holdout_metrics_df,
        "prediction_frames": prediction_frames,
        "paths": paths,
    }
