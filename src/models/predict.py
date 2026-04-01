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


def _load_model(tipo_modelo: str, horizonte_dias: int):
    path = MODELS_DIR / f"{tipo_modelo}_h{horizonte_dias}d.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {path}\n"
            "Execute main.py --train primeiro."
        )
    return joblib.load(path)


def _load_feature_cols(horizonte_dias: int) -> list[str]:
    return joblib.load(MODELS_DIR / f"feature_cols_h{horizonte_dias}d.joblib")


def predict_latest(dataframe_features: pd.DataFrame) -> pd.DataFrame:
    """
    Gera previsões a partir do último registro disponível no DataFrame.

    Usa a última linha (hoje) como vetor de features e retorna
    previsões para 1, 15, 30 e 60 dias à frente, por modelo.

    Parâmetros
    ----------
    dataframe_features : DataFrame com features (saída de build_features)

    Retorna
    -------
    pd.DataFrame com colunas:
        horizonte_dias, previsao_xgboost, previsao_random_forest, data_previsao
    """
    ultima_data = dataframe_features.index[-1]
    rows = []

    for horizonte_dias in HORIZONS:
        feature_cols = _load_feature_cols(horizonte_dias)

        # Última linha com todas as features preenchidas
        row = dataframe_features[feature_cols].dropna().iloc[[-1]]

        X = row.values

        previsao_xgboost = _load_model("xgboost", horizonte_dias).predict(X)[0]
        previsao_random_forest  = _load_model("random_forest",  horizonte_dias).predict(X)[0]

        rows.append({
            "horizonte_dias": horizonte_dias,
            "data_base":      ultima_data.date(),
            "data_previsao":  (ultima_data + pd.Timedelta(days=horizonte_dias)).date(),
            "previsao_xgboost":       round(previsao_xgboost, 2),
            "previsao_random_forest":        round(previsao_random_forest,  2),
            "media_modelos":  round((previsao_xgboost + previsao_random_forest) / 2, 2),
        })

    result = pd.DataFrame(rows)
    return result


def predict_period(dataframe_features: pd.DataFrame, data_inicial: str, data_final: str) -> pd.DataFrame:
    """
    Gera previsões retroativas para um período (backtesting visual).
    Útil para comparar previsão vs. real em gráficos.

    Parâmetros
    ----------
    dataframe_features : DataFrame com features
    data_inicial : data inicial (YYYY-MM-DD)
    data_final   : data final   (YYYY-MM-DD)

    Retorna
    -------
    pd.DataFrame com previsões de cada modelo por horizonte e data.
    """
    sub = dataframe_features.loc[data_inicial:data_final]
    records = []

    for horizonte_dias in HORIZONS:
        feature_cols = _load_feature_cols(horizonte_dias)
        target_col   = f"target_h{horizonte_dias}d"

        X_df = sub[feature_cols].copy()
        col_medians = X_df.median()
        X_df = X_df.fillna(col_medians)

        X = X_df.values

        previsoes_xgboost = _load_model("xgboost", horizonte_dias).predict(X)
        previsoes_random_forest  = _load_model("random_forest",  horizonte_dias).predict(X)

        for i, data in enumerate(sub.index):
            valor_real = sub[target_col].iloc[i] if target_col in sub.columns else np.nan
            records.append({
                "data":           data.date(),
                "horizonte_dias": horizonte_dias,
                "valor_real":           round(valor_real, 2) if not np.isnan(valor_real) else None,
                "previsao_xgboost":       round(previsoes_xgboost[i], 2),
                "previsao_random_forest":        round(previsoes_random_forest[i],  2),
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