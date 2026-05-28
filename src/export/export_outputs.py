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


def _load_model(model_type: str, horizon: int):
    path = MODELS_DIR / f"{model_type}_h{horizon}d.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def _load_feature_cols(horizon: int) -> list[str]:
    return joblib.load(MODELS_DIR / f"feature_cols_h{horizon}d.joblib")


def _load_medians(horizon: int) -> np.ndarray:
    path = MODELS_DIR / f"feature_medians_h{horizon}d.joblib"
    if path.exists():
        return joblib.load(path)
    return None


def export_predictions(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions for the latest available date per horizon.
    Exports to data/outputs/predictions.csv.
    """
    rows = []
    last_date = features_df.index.max()

    for h in HORIZONS:
        try:
            feat_cols = _load_feature_cols(h)
            medians = _load_medians(h)
            X_row = features_df.loc[[last_date], feat_cols].copy()
            if medians is not None:
                X_row = X_row.fillna(pd.Series(medians, index=feat_cols))
            X = X_row.values

            for model_type in ("xgboost", "random_forest"):
                model = _load_model(model_type, h)
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
    out_path = DATA_OUTPUTS / "predictions.csv"
    df.to_csv(out_path, index=False)
    print(f"[export] predictions.csv: {len(df)} rows → {out_path}")
    return df


def export_price_history(clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Export deflated price history to data/outputs/price_history.csv.
    """
    if "preco_boi_gordo" not in clean_df.columns:
        raise KeyError("preco_boi_gordo not found in clean_df")

    df = clean_df[["preco_boi_gordo"]].dropna().copy()
    df = df.rename(columns={"preco_boi_gordo": "real_price_deflated"})
    df.index.name = "date"

    out_path = DATA_OUTPUTS / "price_history.csv"
    df.to_csv(out_path)
    print(f"[export] price_history.csv: {len(df)} rows → {out_path}")
    return df


def export_all(features_df: pd.DataFrame, clean_df: pd.DataFrame) -> None:
    """Run all exports."""
    export_predictions(features_df)
    export_price_history(clean_df)


if __name__ == "__main__":
    from src.processing.merger import build_dataset
    from src.processing.cleaner import clean
    from src.features.engineering import build_features

    raw_df = build_dataset()
    clean_df = clean(raw_df, exclude_holdout=False)
    features_df = build_features(clean_df.copy())
    export_all(features_df, clean_df)
