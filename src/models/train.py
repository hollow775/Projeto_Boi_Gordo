# src/models/train.py
# ==============================================================
# Treinamento de XGBoost e Random Forest para cada horizonte.
#
# Esquema de validação: Walk-Forward (expanding window).
# K-fold convencional é INADEQUADO para séries temporais pois
# permite contaminação de informação futura no treino.
#
# Para cada horizonte h em HORIZONS:
#   - Treina XGBoost e Random Forest independentemente
#   - Salva modelos em MODELS_DIR
#   - Retorna métricas por fold e por modelo
# ==============================================================

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from config.settings import HORIZONS, MODELS_DIR
from src.features.engineering import get_feature_columns


# ── Hiperparâmetros padrão ─────────────────────────────────────
# Ponto de partida conservador — ajuste via grid search posterior.
XGBOOST_PARAMS = {
    "n_estimators":     500,
    "learning_rate":    0.05,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth":    None,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs":       -1,
}

# Mínimo de observações para o primeiro fold de treino (2 anos)
MIN_TRAIN_DAYS = 730

# Número de folds walk-forward
N_FOLDS = 5


def _walk_forward_splits(
    n: int,
    min_train: int = MIN_TRAIN_DAYS,
    n_folds: int = N_FOLDS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Gera índices de treino/teste em expanding window.

    Para n=2000, min_train=730, n_folds=5:
        fold 1: treino=[0:730],   teste=[730:1000]
        fold 2: treino=[0:1000],  teste=[1000:1270]
        ...
    """
    test_size = (n - min_train) // n_folds
    splits = []
    for i in range(n_folds):
        train_end = min_train + i * test_size
        test_end  = train_end + test_size
        if test_end > n:
            test_end = n
        splits.append((
            np.arange(0, train_end),
            np.arange(train_end, test_end),
        ))
    return splits


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula RMSE, MAE e MAPE."""
    mask  = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t   = y_true[mask]
    y_p   = y_pred[mask]

    rmse = np.sqrt(np.mean((y_t - y_p) ** 2))
    mae  = np.mean(np.abs(y_t - y_p))
    mape = np.mean(np.abs((y_t - y_p) / np.where(y_t == 0, np.nan, y_t))) * 100

    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def train_horizon(
    df: pd.DataFrame,
    horizon: int,
) -> dict:
    """
    Treina XGBoost e Random Forest para um horizonte específico.

    Parâmetros
    ----------
    df      : DataFrame com features e target já construídos
    horizon : horizonte em dias (ex: 1, 15, 30, 60)

    Retorna
    -------
    dict com:
        "xgb_model"     : modelo XGBoost treinado no dataset completo
        "rf_model"      : modelo RF treinado no dataset completo
        "xgb_cv_metrics": métricas walk-forward do XGBoost
        "rf_cv_metrics" : métricas walk-forward do RF
        "feature_cols"  : colunas de features usadas
    """
    target_col = f"target_h{horizon}d"
    if target_col not in df.columns:
        raise KeyError(f"Target '{target_col}' não encontrado.")

    feature_cols = get_feature_columns(df)

    # Remove linhas onde target é NaN (fim da série)
    df_valid = df.dropna(subset=[target_col])

    X = df_valid[feature_cols].values
    y = df_valid[target_col].values

    # Substitui NaN em features por mediana (XGBoost tolera, RF não)
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    n = len(X)
    splits = _walk_forward_splits(n)

    xgb_cv, rf_cv = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # XGBoost
        xgb = XGBRegressor(**XGBOOST_PARAMS)
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        xgb_cv.append(_compute_metrics(y_test, xgb_pred))

        # Random Forest
        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_cv.append(_compute_metrics(y_test, rf_pred))

        print(
            f"  [h{horizon}d | fold {fold_idx+1}/{len(splits)}] "
            f"XGB MAPE={xgb_cv[-1]['MAPE']:.2f}% | "
            f"RF  MAPE={rf_cv[-1]['MAPE']:.2f}%"
        )

    # Treina modelo final em todo o dataset (sem split)
    print(f"  [h{horizon}d] Treinando modelo final em {n} observações...")

    xgb_final = XGBRegressor(**XGBOOST_PARAMS)
    xgb_final.fit(X, y)

    rf_final = RandomForestRegressor(**RF_PARAMS)
    rf_final.fit(X, y)

    # Salva modelos
    xgb_path = MODELS_DIR / f"xgb_h{horizon}d.joblib"
    rf_path  = MODELS_DIR / f"rf_h{horizon}d.joblib"
    joblib.dump(xgb_final, xgb_path)
    joblib.dump(rf_final,  rf_path)

    # Salva nomes das features junto com o modelo
    feat_path = MODELS_DIR / f"feature_cols_h{horizon}d.joblib"
    joblib.dump(feature_cols, feat_path)

    print(f"  [h{horizon}d] Modelos salvos em {MODELS_DIR}")

    return {
        "xgb_model":      xgb_final,
        "rf_model":       rf_final,
        "xgb_cv_metrics": xgb_cv,
        "rf_cv_metrics":  rf_cv,
        "feature_cols":   feature_cols,
    }


def train_all(df: pd.DataFrame) -> dict:
    """
    Executa o treinamento para todos os horizontes definidos em HORIZONS.

    Retorna
    -------
    dict keyed por horizonte (int): resultados de train_horizon
    """
    results = {}
    for h in HORIZONS:
        print(f"\n[train] Horizonte: {h} dias")
        results[h] = train_horizon(df, h)

    return results