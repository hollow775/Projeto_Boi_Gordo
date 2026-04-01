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
        "xgboost"     : modelo XGBoost treinado no dataset completo
        "random_forest"      : modelo RF treinado no dataset completo
        "metricas_cv_xgboost": métricas walk-forward do XGBoost
        "metricas_cv_random_forest" : métricas walk-forward do RF
        "feature_cols"  : colunas de features usadas
        "out_of_fold_dataframe" : DataFrame com previsões OOF e valores reais
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

    metricas_cv_xgboost, metricas_cv_random_forest = [], []
    
    datas_out_of_fold = []
    y_verdadeiro_out_of_fold = []
    previsoes_xgboost_out_of_fold = []
    previsoes_random_forest_out_of_fold = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # XGBoost
        xgboost = XGBRegressor(**XGBOOST_PARAMS)
        xgboost.fit(X_train, y_train)
        previsao_xgboost = xgboost.predict(X_test)
        metricas_cv_xgboost.append(_compute_metrics(y_test, previsao_xgboost))

        # Random Forest
        random_forest = RandomForestRegressor(**RF_PARAMS)
        random_forest.fit(X_train, y_train)
        previsao_random_forest = random_forest.predict(X_test)
        metricas_cv_random_forest.append(_compute_metrics(y_test, previsao_random_forest))

        # Salva o dataset de log de previsoes out-of-fold para analise cega graficamente
        test_dates = df_valid.index[test_idx]
        datas_out_of_fold.extend(test_dates)
        y_verdadeiro_out_of_fold.extend(y_test)
        previsoes_xgboost_out_of_fold.extend(previsao_xgboost)
        previsoes_random_forest_out_of_fold.extend(previsao_random_forest)

        print(
            f"  [h{horizon}d | fold {fold_idx+1}/{len(splits)}] "
            f"XGB MAPE={metricas_cv_xgboost[-1]['MAPE']:.2f}% | "
            f"RF  MAPE={metricas_cv_random_forest[-1]['MAPE']:.2f}%"
        )

    # Treina modelo final em todo o dataset (sem split)
    print(f"  [h{horizon}d] Treinando modelo final em {n} observações...")

    xgboost_final = XGBRegressor(**XGBOOST_PARAMS)
    xgboost_final.fit(X, y)

    random_forest_final = RandomForestRegressor(**RF_PARAMS)
    random_forest_final.fit(X, y)

    # Salva modelos
    caminho_xgboost = MODELS_DIR / f"xgboost_h{horizon}d.joblib"
    caminho_random_forest  = MODELS_DIR / f"random_forest_h{horizon}d.joblib"
    joblib.dump(xgboost_final, caminho_xgboost)
    joblib.dump(random_forest_final,  caminho_random_forest)

    # Salva nomes das features junto com o modelo
    feat_path = MODELS_DIR / f"feature_cols_h{horizon}d.joblib"
    joblib.dump(feature_cols, feat_path)

    print(f"  [h{horizon}d] Modelos salvos em {MODELS_DIR}")

    out_of_fold_dataframe = pd.DataFrame({
        "y_true": y_verdadeiro_out_of_fold,
        "previsao_xgboost": previsoes_xgboost_out_of_fold,
        "previsao_random_forest": previsoes_random_forest_out_of_fold
    }, index=datas_out_of_fold)

    return {
        "xgboost":      xgboost_final,
        "random_forest":       random_forest_final,
        "metricas_cv_xgboost": metricas_cv_xgboost,
        "metricas_cv_random_forest":  metricas_cv_random_forest,
        "feature_cols":   feature_cols,
        "out_of_fold_dataframe": out_of_fold_dataframe,
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