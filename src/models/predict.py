# src/models/predict.py
# ==============================================================
# Inferência com os modelos treinados.
# Recebe o DataFrame de features e retorna previsões para
# todos os horizontes, de ambos os modelos.
# ==============================================================

import pandas as pd
import numpy as np
import joblib
from config.settings import HORIZONS, MODELS_DIR


def _load_model(model_type: str, horizon: int):
    path = MODELS_DIR / f"{model_type}_h{horizon}d.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {path}\n"
            "Execute main.py --train primeiro."
        )
    return joblib.load(path)


def _load_feature_cols(horizon: int) -> list[str]:
    return joblib.load(MODELS_DIR / f"feature_cols_h{horizon}d.joblib")


def predict_latest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera previsões a partir do último registro disponível no DataFrame.

    Usa a última linha (hoje) como vetor de features e retorna
    previsões para 1, 15, 30 e 60 dias à frente, por modelo.

    Parâmetros
    ----------
    df : DataFrame com features (saída de build_features)

    Retorna
    -------
    pd.DataFrame com colunas:
        horizonte_dias, xgb_pred, rf_pred, data_previsao
    """
    last_date = df.index[-1]
    rows = []

    for h in HORIZONS:
        feature_cols = _load_feature_cols(h)

        # Última linha com todas as features preenchidas
        row = df[feature_cols].dropna().iloc[[-1]]

        X = row.values

        xgb_pred = _load_model("xgb", h).predict(X)[0]
        rf_pred  = _load_model("rf",  h).predict(X)[0]

        rows.append({
            "horizonte_dias": h,
            "data_base":      last_date.date(),
            "data_previsao":  (last_date + pd.Timedelta(days=h)).date(),
            "xgb_pred":       round(xgb_pred, 2),
            "rf_pred":        round(rf_pred,  2),
            "media_modelos":  round((xgb_pred + rf_pred) / 2, 2),
        })

    result = pd.DataFrame(rows)
    return result


def predict_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Gera previsões retroativas para um período (backtesting visual).
    Útil para comparar previsão vs. real em gráficos.

    Parâmetros
    ----------
    df    : DataFrame com features
    start : data inicial (YYYY-MM-DD)
    end   : data final   (YYYY-MM-DD)

    Retorna
    -------
    pd.DataFrame com previsões de cada modelo por horizonte e data.
    """
    sub = df.loc[start:end]
    records = []

    for h in HORIZONS:
        feature_cols = _load_feature_cols(h)
        target_col   = f"target_h{h}d"

        X_df = sub[feature_cols].copy()
        col_medians = X_df.median()
        X_df = X_df.fillna(col_medians)

        X = X_df.values

        xgb_preds = _load_model("xgb", h).predict(X)
        rf_preds  = _load_model("rf",  h).predict(X)

        for i, date in enumerate(sub.index):
            real = sub[target_col].iloc[i] if target_col in sub.columns else np.nan
            records.append({
                "data":           date.date(),
                "horizonte_dias": h,
                "real":           round(real, 2) if not np.isnan(real) else None,
                "xgb_pred":       round(xgb_preds[i], 2),
                "rf_pred":        round(rf_preds[i],  2),
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    # Exemplo de uso rápido após treinamento
    import sys
    sys.path.insert(0, ".")
    from src.processing.merger  import build_dataset
    from src.processing.cleaner import clean
    from src.features.engineering import build_features

    df = build_dataset()
    df = clean(df)
    df = build_features(df)

    preds = predict_latest(df)
    print("\nPrevisões a partir de hoje:")
    print(preds.to_string(index=False))