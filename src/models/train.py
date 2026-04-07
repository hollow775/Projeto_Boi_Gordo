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

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from config.settings import DATA_PROCESSED, HORIZONS, MODELS_DIR
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

# Limites de segurança e orçamento de tuning
CUTOFF_DATE = pd.Timestamp("2025-12-31")
TUNING_BUDGET_MIN = 10


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


def _assert_training_cutoff(df: pd.DataFrame) -> None:
    """Garante que dados de treino/tuning não ultrapassem 2025-12-31."""
    max_date = pd.to_datetime(df.index.max())
    if max_date > CUTOFF_DATE:
        raise AssertionError(
            f"Dataset de treino/tuning com data máxima {max_date.date()} "
            f"ultrapassa o limite permitido de {CUTOFF_DATE.date()}."
        )


def _tune_with_budget(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    horizon: int,
    time_budget_min: int = TUNING_BUDGET_MIN,
) -> dict:
    """
    Executa uma busca reduzida com orçamento de tempo por horizonte
    e salva um log em data/processed/tuning_log_h{h}d.json.
    Retorna os melhores hiperparâmetros encontrados para XGBoost e RF.
    """
    start = time.time()
    deadline = start + time_budget_min * 60

    # Usa o primeiro fold para avaliação rápida
    train_idx, test_idx = splits[0]
    if len(test_idx) == 0:
        log_path = DATA_PROCESSED / f"tuning_log_h{horizon}d.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return {"xgboost": XGBOOST_PARAMS.copy(), "random_forest": RF_PARAMS.copy()}

    X_train, X_val = X[train_idx], X[test_idx]
    y_train, y_val = y[train_idx], y[test_idx]

    xgb_candidates = [
        {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 4, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 450, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 600, "learning_rate": 0.03, "max_depth": 8, "subsample": 0.7, "colsample_bytree": 0.7},
    ]

    rf_candidates = [
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt"},
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 5, "max_features": "sqrt"},
        {"n_estimators": 700, "max_depth": 20, "min_samples_leaf": 3, "max_features": 0.6},
    ]

    log_entries = []
    best_params = {"xgboost": XGBOOST_PARAMS.copy(), "random_forest": RF_PARAMS.copy()}
    best_mape_xgb = np.inf
    best_mape_rf = np.inf

    def _remaining_time_ok() -> bool:
        return time.time() < deadline

    for params in xgb_candidates:
        if not _remaining_time_ok():
            break
        merged = {**XGBOOST_PARAMS, **params}
        model = XGBRegressor(**merged)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        metrics = _compute_metrics(y_val, preds)
        log_entries.append({
            "model": "xgboost",
            "params": merged,
            "metrics": metrics,
            "elapsed_seconds": round(time.time() - start, 2),
        })
        if metrics["MAPE"] < best_mape_xgb:
            best_mape_xgb = metrics["MAPE"]
            best_params["xgboost"] = merged

    for params in rf_candidates:
        if not _remaining_time_ok():
            break
        merged = {**RF_PARAMS, **params}
        model = RandomForestRegressor(**merged)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        metrics = _compute_metrics(y_val, preds)
        log_entries.append({
            "model": "random_forest",
            "params": merged,
            "metrics": metrics,
            "elapsed_seconds": round(time.time() - start, 2),
        })
        if metrics["MAPE"] < best_mape_rf:
            best_mape_rf = metrics["MAPE"]
            best_params["random_forest"] = merged

    log_path = DATA_PROCESSED / f"tuning_log_h{horizon}d.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log_entries, f, ensure_ascii=False, indent=2)

    return best_params


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula RMSE, MAE e MAPE."""
    mask  = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t   = y_true[mask]
    y_p   = y_pred[mask]

    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    mae  = float(np.mean(np.abs(y_t - y_p)))
    mape = float(np.mean(np.abs((y_t - y_p) / np.where(y_t == 0, np.nan, y_t))) * 100)

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

    _assert_training_cutoff(df)
    if "preco_boi_gordo" not in df.columns:
        raise KeyError("Coluna 'preco_boi_gordo' é necessária para baseline ingênuo.")

    df = df.sort_index()
    df["baseline_last"] = df["preco_boi_gordo"]
    df["baseline_ma7"] = df["preco_boi_gordo"].rolling(window=7, min_periods=1).mean()

    feature_cols = get_feature_columns(df)

    # Remove linhas onde target é NaN (fim da série)
    df_valid = df.dropna(subset=[target_col])

    X = df_valid[feature_cols].values
    y = df_valid[target_col].values
    baseline_last_array = df_valid["baseline_last"].values
    baseline_ma7_array = df_valid["baseline_ma7"].values

    # Substitui NaN em features por mediana (XGBoost tolera, RF não)
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    n = len(X)
    splits = _walk_forward_splits(n)
    best_params = _tune_with_budget(X, y, splits, horizon)

    metricas_cv_xgboost, metricas_cv_random_forest = [], []
    metricas_cv_baseline = []
    
    datas_out_of_fold = []
    y_verdadeiro_out_of_fold = []
    previsoes_xgboost_out_of_fold = []
    previsoes_random_forest_out_of_fold = []
    previsoes_baseline_last_out_of_fold = []
    previsoes_baseline_ma7_out_of_fold = []
    previsao_baseline_escolhida_out_of_fold = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # XGBoost
        xgboost = XGBRegressor(**best_params["xgboost"])
        xgboost.fit(X_train, y_train)
        previsao_xgboost = xgboost.predict(X_test)
        metricas_cv_xgboost.append(_compute_metrics(y_test, previsao_xgboost))

        # Random Forest
        random_forest = RandomForestRegressor(**best_params["random_forest"])
        random_forest.fit(X_train, y_train)
        previsao_random_forest = random_forest.predict(X_test)
        metricas_cv_random_forest.append(_compute_metrics(y_test, previsao_random_forest))

        # Baselines ingênuos (último valor vs média 7 dias)
        baseline_last_pred = baseline_last_array[test_idx]
        baseline_ma7_pred = baseline_ma7_array[test_idx]
        metrics_baseline_last = _compute_metrics(y_test, baseline_last_pred)
        metrics_baseline_ma7 = _compute_metrics(y_test, baseline_ma7_pred)
        use_ma7 = metrics_baseline_ma7["MAPE"] <= metrics_baseline_last["MAPE"]
        baseline_pred = baseline_ma7_pred if use_ma7 else baseline_last_pred
        metricas_cv_baseline.append(metrics_baseline_ma7 if use_ma7 else metrics_baseline_last)

        # Salva o dataset de log de previsoes out-of-fold para analise cega graficamente
        test_dates = df_valid.index[test_idx]
        datas_out_of_fold.extend(test_dates)
        y_verdadeiro_out_of_fold.extend(y_test)
        previsoes_xgboost_out_of_fold.extend(previsao_xgboost)
        previsoes_random_forest_out_of_fold.extend(previsao_random_forest)
        previsoes_baseline_last_out_of_fold.extend(baseline_last_pred)
        previsoes_baseline_ma7_out_of_fold.extend(baseline_ma7_pred)
        previsao_baseline_escolhida_out_of_fold.extend(baseline_pred)

        print(
            f"  [h{horizon}d | fold {fold_idx+1}/{len(splits)}] "
            f"XGB MAPE={metricas_cv_xgboost[-1]['MAPE']:.2f}% | "
            f"RF  MAPE={metricas_cv_random_forest[-1]['MAPE']:.2f}% | "
            f"Baseline({'MA7' if use_ma7 else 'ultimo'}) MAPE={metricas_cv_baseline[-1]['MAPE']:.2f}%"
        )

    # Treina modelo final em todo o dataset (sem split)
    print(f"  [h{horizon}d] Treinando modelo final em {n} observações...")

    xgboost_final = XGBRegressor(**best_params["xgboost"])
    xgboost_final.fit(X, y)

    random_forest_final = RandomForestRegressor(**best_params["random_forest"])
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
        "previsao_random_forest": previsoes_random_forest_out_of_fold,
        "baseline_last": previsoes_baseline_last_out_of_fold,
        "baseline_ma7": previsoes_baseline_ma7_out_of_fold,
        "baseline_escolhida": previsao_baseline_escolhida_out_of_fold,
    }, index=datas_out_of_fold)

    return {
        "xgboost":      xgboost_final,
        "random_forest":       random_forest_final,
        "metricas_cv_xgboost": metricas_cv_xgboost,
        "metricas_cv_random_forest":  metricas_cv_random_forest,
        "metricas_cv_baseline": metricas_cv_baseline,
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
