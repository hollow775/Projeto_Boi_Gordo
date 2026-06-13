# src/export/export_outputs.py
# ==============================================================
# Export pipeline outputs as CSV for Streamlit consumption.
# Produces predictions.csv and price_history.csv in data/outputs/.
# ==============================================================
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from config.settings import DATA_OUTPUTS, HORIZONS, MODELS_DIR


def _load_model(model_type: str, horizon: int, models_dir=None):
    models_dir = models_dir or MODELS_DIR
    path = models_dir / f"{model_type}_h{horizon}d.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def _load_feature_cols(horizon: int, models_dir=None) -> list[str]:
    models_dir = models_dir or MODELS_DIR
    return joblib.load(models_dir / f"feature_cols_h{horizon}d.joblib")


def _load_medians(horizon: int, models_dir=None) -> np.ndarray:
    models_dir = models_dir or MODELS_DIR
    path = models_dir / f"feature_medians_h{horizon}d.joblib"
    if path.exists():
        return joblib.load(path)
    return None


def export_predictions(
    features_df: pd.DataFrame,
    models_dir=None,
    data_outputs=None,
) -> pd.DataFrame:
    """
    Generate predictions for the latest available date per horizon.
    Exports to data/outputs/predictions.csv.
    """
    models_dir = models_dir or MODELS_DIR
    data_outputs = data_outputs or DATA_OUTPUTS
    rows = []
    last_date = features_df.index.max()

    for h in HORIZONS:
        try:
            feat_cols = _load_feature_cols(h, models_dir=models_dir)
            medians = _load_medians(h, models_dir=models_dir)
            X_row = features_df.loc[[last_date], feat_cols].copy()
            if medians is not None:
                X_row = X_row.fillna(pd.Series(medians, index=feat_cols))
            X = X_row.values

            for model_type in ("xgboost", "random_forest"):
                model = _load_model(model_type, h, models_dir=models_dir)
                pred = float(model.predict(X)[0])
                rows.append({
                    "date": (last_date + pd.Timedelta(days=h)).strftime("%Y-%m-%d"),
                    "horizon": h,
                    "predicted_value": round(pred, 2),
                    "model": model_type,
                })
        except Exception as e:
            print(f"[export] Skipping h{h}d: {e}")

    df = pd.DataFrame(rows)
    out_path = data_outputs / "predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[export] predictions.csv: {len(df)} rows → {out_path}")
    return df


def export_price_history(clean_df: pd.DataFrame, data_outputs=None) -> pd.DataFrame:
    """
    Export deflated price history to data/outputs/price_history.csv.
    """
    if "preco_boi_gordo" not in clean_df.columns:
        raise KeyError("preco_boi_gordo not found in clean_df")
    data_outputs = data_outputs or DATA_OUTPUTS

    df = clean_df[["preco_boi_gordo"]].dropna().copy()
    df = df.rename(columns={"preco_boi_gordo": "real_price_deflated"})
    df.index.name = "date"

    out_path = data_outputs / "price_history.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    print(f"[export] price_history.csv: {len(df)} rows → {out_path}")
    return df


def export_all(
    features_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    models_dir=None,
    data_outputs=None,
) -> None:
    """Run all exports."""
    export_predictions(features_df, models_dir=models_dir, data_outputs=data_outputs)
    export_price_history(clean_df, data_outputs=data_outputs)


if __name__ == "__main__":
    from src.processing.merger import build_dataset
    from src.processing.cleaner import clean
    from src.features.engineering import build_features

    raw_df = build_dataset()
    clean_df = clean(raw_df, exclude_holdout=False)
    features_df = build_features(clean_df.copy())
    export_all(features_df, clean_df)
