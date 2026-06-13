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

RANDOM_FOREST_PARAMS = {
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
_DEFAULT_CUTOFF = object()


def _walk_forward_splits(
    n: int,
    min_train: int = MIN_TRAIN_DAYS,
    n_folds: int = N_FOLDS,
    min_test_size: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Gera índices de treino/teste em expanding window.

    Para n=2000, min_train=730, n_folds=5:
        fold 1: treino=[0:730],   teste=[730:1000]
        fold 2: treino=[0:1000],  teste=[1000:1270]
        ...
    """
    if n_folds < 1:
        raise ValueError("n_folds deve ser >= 1.")
    if min_train < 1:
        raise ValueError("min_train deve ser >= 1.")
    if min_test_size < 1:
        raise ValueError("min_test_size deve ser >= 1.")
    if n <= min_train:
        raise ValueError(
            f"Dados insuficientes para walk-forward: n={n}, min_train={min_train}."
        )

    available_test = n - min_train
    if available_test < n_folds * min_test_size:
        raise ValueError(
            "Dados insuficientes para gerar folds de teste mínimos: "
            f"disponível={available_test}, necessário={n_folds * min_test_size}."
        )

    test_size = available_test // n_folds
    splits = []
    for i in range(n_folds):
        train_end = min_train + i * test_size
        test_end = n if i == n_folds - 1 else train_end + test_size
        splits.append((
            np.arange(0, train_end),
            np.arange(train_end, test_end),
        ))
    return splits


def _purge_train_rows_with_targets_in_test_window(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    dates: pd.DatetimeIndex,
    horizon: int,
) -> np.ndarray:
    """Remove training rows whose forecast target date falls inside the test window."""
    if len(train_idx) == 0 or len(test_idx) == 0:
        return train_idx

    test_start = dates[test_idx[0]]
    test_end = dates[test_idx[-1]]
    target_dates = dates[train_idx] + pd.Timedelta(days=horizon)
    keep_mask = (target_dates < test_start) | (target_dates > test_end)
    return train_idx[keep_mask]


def _purged_walk_forward_splits(
    n: int,
    dates: pd.DatetimeIndex,
    horizon: int,
    min_train: int = MIN_TRAIN_DAYS,
    n_folds: int = N_FOLDS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build walk-forward splits and purge boundary rows with targets in test windows."""
    base_splits = _walk_forward_splits(n, min_train=min_train, n_folds=n_folds)
    return [
        (
            _purge_train_rows_with_targets_in_test_window(train_idx, test_idx, dates, horizon),
            test_idx,
        )
        for train_idx, test_idx in base_splits
    ]


def _fit_imputation_medians(X_train: np.ndarray) -> np.ndarray:
    X_train = np.asarray(X_train, dtype=float)
    medians = []
    for col_idx in range(X_train.shape[1]):
        column = X_train[:, col_idx]
        medians.append(0.0 if np.isnan(column).all() else float(np.nanmedian(column)))
    return np.array(medians)


def _apply_imputation_medians(X: np.ndarray, medians: np.ndarray) -> np.ndarray:
    X_imputed = np.asarray(X, dtype=float).copy()
    nan_mask = np.isnan(X_imputed)
    if nan_mask.any():
        X_imputed[nan_mask] = np.take(medians, np.where(nan_mask)[1])
    return X_imputed


def _assert_training_cutoff(df: pd.DataFrame) -> None:
    """Garante que dados de treino/tuning não ultrapassem 2025-12-31."""
    _assert_training_cutoff_for(df, CUTOFF_DATE)


def _assert_training_cutoff_for(df: pd.DataFrame, cutoff_date: pd.Timestamp | None) -> None:
    """Validate a lane-specific training cutoff when one is configured."""
    if cutoff_date is None:
        return
    max_date = pd.to_datetime(df.index.max())
    if max_date > cutoff_date:
        raise AssertionError(
            f"Dataset de treino/tuning com data máxima {max_date.date()} "
            f"ultrapassa o limite permitido de {cutoff_date.date()}."
        )


def _tune_with_budget(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    horizon: int,
    time_budget_min: int = TUNING_BUDGET_MIN,
    data_processed_dir=None,
) -> dict:
    """
    Executa uma busca reduzida com orçamento de tempo por horizonte
    e salva um log em data/processed/tuning_log_h{h}d.json.
    Retorna os melhores hiperparâmetros encontrados para XGBoost e Random Forest.
    """
    start = time.time()
    data_processed_dir = data_processed_dir or DATA_PROCESSED
    deadline = start + time_budget_min * 60

    # Usa o primeiro fold para avaliação rápida
    train_idx, test_idx = splits[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        log_path = data_processed_dir / f"tuning_log_h{horizon}d.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return {"xgboost": XGBOOST_PARAMS.copy(), "random_forest": RANDOM_FOREST_PARAMS.copy()}

    X_train_raw, X_val_raw = X[train_idx], X[test_idx]
    imputation_medians = _fit_imputation_medians(X_train_raw)
    X_train = _apply_imputation_medians(X_train_raw, imputation_medians)
    X_val = _apply_imputation_medians(X_val_raw, imputation_medians)
    y_train, y_val = y[train_idx], y[test_idx]

    xgboost_candidates = [
        {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 4, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 450, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 600, "learning_rate": 0.03, "max_depth": 8, "subsample": 0.7, "colsample_bytree": 0.7},
    ]

    random_forest_candidates = [
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt"},
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 5, "max_features": "sqrt"},
        {"n_estimators": 700, "max_depth": 20, "min_samples_leaf": 3, "max_features": 0.6},
    ]

    log_entries = []
    best_params = {"xgboost": XGBOOST_PARAMS.copy(), "random_forest": RANDOM_FOREST_PARAMS.copy()}
    best_mape_xgboost = np.inf
    best_mape_random_forest = np.inf

    def _remaining_time_ok() -> bool:
        return time.time() < deadline

    for params in xgboost_candidates:
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
        if metrics["MAPE"] < best_mape_xgboost:
            best_mape_xgboost = metrics["MAPE"]
            best_params["xgboost"] = merged

    for params in random_forest_candidates:
        if not _remaining_time_ok():
            break
        merged = {**RANDOM_FOREST_PARAMS, **params}
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
        if metrics["MAPE"] < best_mape_random_forest:
            best_mape_random_forest = metrics["MAPE"]
            best_params["random_forest"] = merged

    log_path = data_processed_dir / f"tuning_log_h{horizon}d.json"
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


def _tag_fold_metrics(metrics: dict, fold_number: int, used_for_tuning: bool = False) -> dict:
    return {
        **metrics,
        "fold": fold_number,
        "used_for_tuning": used_for_tuning,
    }


def train_horizon(
    df: pd.DataFrame,
    horizon: int,
    models_dir=None,
    data_processed_dir=None,
    cutoff_date=_DEFAULT_CUTOFF,
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
        "random_forest"      : modelo Random Forest treinado no dataset completo
        "metricas_cv_xgboost": métricas walk-forward do XGBoost
        "metricas_cv_random_forest" : métricas walk-forward do Random Forest
        "feature_cols"  : colunas de features usadas
        "out_of_fold_dataframe" : DataFrame com previsões OOF e valores reais
    """
    target_col = f"target_h{horizon}d"
    if target_col not in df.columns:
        raise KeyError(f"Target '{target_col}' não encontrado.")

    models_dir = models_dir or MODELS_DIR
    data_processed_dir = data_processed_dir or DATA_PROCESSED
    cutoff_date = CUTOFF_DATE if cutoff_date is _DEFAULT_CUTOFF else cutoff_date
    _assert_training_cutoff_for(df, cutoff_date)
    if "preco_boi_gordo" not in df.columns:
        raise KeyError("Coluna 'preco_boi_gordo' é necessária para baseline ingênuo.")

    df = df.sort_index()
    df["baseline_last"] = df["preco_boi_gordo"]
    df["baseline_ma7"] = df["preco_boi_gordo"].rolling(window=7, min_periods=1).mean()

    feature_cols = get_feature_columns(df)

    # Remove linhas onde target é NaN (fim da série)
    df_valid = df.dropna(subset=[target_col])

    X = df_valid[feature_cols].to_numpy(dtype=float)
    y = df_valid[target_col].values
    baseline_last_array = df_valid["baseline_last"].values
    baseline_ma7_array = df_valid["baseline_ma7"].values

    n = len(X)
    splits = _purged_walk_forward_splits(n, pd.DatetimeIndex(df_valid.index), horizon)
    best_params = _tune_with_budget(X, y, splits, horizon, data_processed_dir=data_processed_dir)

    metricas_cv_xgboost, metricas_cv_random_forest = [], []
    metricas_cv_baseline = []
    
    datas_out_of_fold = []
    y_verdadeiro_out_of_fold = []
    previsoes_xgboost_out_of_fold = []
    previsoes_random_forest_out_of_fold = []
    previsoes_baseline_last_out_of_fold = []
    previsoes_baseline_ma7_out_of_fold = []
    previsoes_baseline_out_of_fold = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        if len(train_idx) == 0 or len(test_idx) == 0:
            print(
                f"  [h{horizon}d | fold {fold_idx+1}/{len(splits)}] "
                "fold ignorado: treino ou teste vazio apos purga temporal."
            )
            continue

        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        imputation_medians = _fit_imputation_medians(X_train_raw)
        X_train = _apply_imputation_medians(X_train_raw, imputation_medians)
        X_test = _apply_imputation_medians(X_test_raw, imputation_medians)
        y_train, y_test = y[train_idx], y[test_idx]

        # XGBoost
        xgboost = XGBRegressor(**best_params["xgboost"])
        xgboost.fit(X_train, y_train)
        previsao_xgboost = xgboost.predict(X_test)
        metricas_cv_xgboost.append(
            _tag_fold_metrics(
                _compute_metrics(y_test, previsao_xgboost),
                fold_number=fold_idx + 1,
                used_for_tuning=(fold_idx == 0),
            )
        )

        # Random Forest
        random_forest = RandomForestRegressor(**best_params["random_forest"])
        random_forest.fit(X_train, y_train)
        previsao_random_forest = random_forest.predict(X_test)
        metricas_cv_random_forest.append(
            _tag_fold_metrics(
                _compute_metrics(y_test, previsao_random_forest),
                fold_number=fold_idx + 1,
                used_for_tuning=(fold_idx == 0),
            )
        )

        # Baselines ingênuos (último valor vs média 7 dias).
        # A escolha entre eles usa apenas erro no periodo de treino do fold.
        baseline_last_train_pred = baseline_last_array[train_idx]
        baseline_ma7_train_pred = baseline_ma7_array[train_idx]
        metrics_baseline_last_train = _compute_metrics(y_train, baseline_last_train_pred)
        metrics_baseline_ma7_train = _compute_metrics(y_train, baseline_ma7_train_pred)
        use_ma7 = metrics_baseline_ma7_train["MAPE"] <= metrics_baseline_last_train["MAPE"]

        baseline_last_pred = baseline_last_array[test_idx]
        baseline_ma7_pred = baseline_ma7_array[test_idx]
        metrics_baseline_last = _compute_metrics(y_test, baseline_last_pred)
        metrics_baseline_ma7 = _compute_metrics(y_test, baseline_ma7_pred)
        baseline_pred = baseline_ma7_pred if use_ma7 else baseline_last_pred
        metricas_cv_baseline.append(
            _tag_fold_metrics(
                metrics_baseline_ma7 if use_ma7 else metrics_baseline_last,
                fold_number=fold_idx + 1,
                used_for_tuning=(fold_idx == 0),
            )
        )

        # Salva o dataset de log de previsoes out-of-fold para analise cega graficamente
        test_dates = df_valid.index[test_idx]
        datas_out_of_fold.extend(test_dates)
        y_verdadeiro_out_of_fold.extend(y_test)
        previsoes_xgboost_out_of_fold.extend(previsao_xgboost)
        previsoes_random_forest_out_of_fold.extend(previsao_random_forest)
        previsoes_baseline_last_out_of_fold.extend(baseline_last_pred)
        previsoes_baseline_ma7_out_of_fold.extend(baseline_ma7_pred)
        previsoes_baseline_out_of_fold.extend(baseline_pred)

        print(
            f"  [h{horizon}d | fold {fold_idx+1}/{len(splits)}] "
            f"XGBoost MAPE={metricas_cv_xgboost[-1]['MAPE']:.2f}% | "
            f"Random Forest MAPE={metricas_cv_random_forest[-1]['MAPE']:.2f}% | "
            f"Baseline({'MA7' if use_ma7 else 'ultimo'}) MAPE={metricas_cv_baseline[-1]['MAPE']:.2f}%"
        )

    # Treina modelo final em todo o dataset (sem split)
    print(f"  [h{horizon}d] Treinando modelo final em {n} observações...")

    final_imputation_medians = _fit_imputation_medians(X)
    X_final = _apply_imputation_medians(X, final_imputation_medians)

    xgboost_final = XGBRegressor(**best_params["xgboost"])
    xgboost_final.fit(X_final, y)

    random_forest_final = RandomForestRegressor(**best_params["random_forest"])
    random_forest_final.fit(X_final, y)

    # Salva modelos
    caminho_xgboost = models_dir / f"xgboost_h{horizon}d.joblib"
    caminho_random_forest  = models_dir / f"random_forest_h{horizon}d.joblib"
    joblib.dump(xgboost_final, caminho_xgboost)
    joblib.dump(random_forest_final,  caminho_random_forest)

    # Salva nomes das features junto com o modelo
    feat_path = models_dir / f"feature_cols_h{horizon}d.joblib"
    joblib.dump(feature_cols, feat_path)
    medians_path = models_dir / f"feature_medians_h{horizon}d.joblib"
    joblib.dump(final_imputation_medians, medians_path)

    print(f"  [h{horizon}d] Modelos salvos em {models_dir}")

    out_of_fold_dataframe = pd.DataFrame({
        "y_true": y_verdadeiro_out_of_fold,
        "previsao_xgboost": previsoes_xgboost_out_of_fold,
        "previsao_random_forest": previsoes_random_forest_out_of_fold,
        "baseline_last": previsoes_baseline_last_out_of_fold,
        "baseline_ma7": previsoes_baseline_ma7_out_of_fold,
        "previsao_baseline": previsoes_baseline_out_of_fold,
    }, index=datas_out_of_fold)

    return {
        "xgboost":      xgboost_final,
        "random_forest":       random_forest_final,
        "metricas_cv_xgboost": metricas_cv_xgboost,
        "metricas_cv_random_forest":  metricas_cv_random_forest,
        "metricas_cv_baseline": metricas_cv_baseline,
        "feature_cols":   feature_cols,
        "feature_medians": final_imputation_medians,
        "tuning_fold": 1,
        "out_of_fold_dataframe": out_of_fold_dataframe,
    }


def train_all(
    df: pd.DataFrame,
    models_dir=None,
    data_processed_dir=None,
    cutoff_date=_DEFAULT_CUTOFF,
) -> dict:
    """
    Executa o treinamento para todos os horizontes definidos em HORIZONS.

    Retorna
    -------
    dict keyed por horizonte (int): resultados de train_horizon
    """
    results = {}
    models_dir = models_dir or MODELS_DIR
    data_processed_dir = data_processed_dir or DATA_PROCESSED
    cutoff_date = CUTOFF_DATE if cutoff_date is _DEFAULT_CUTOFF else cutoff_date
    for h in HORIZONS:
        print(f"\n[train] Horizonte: {h} dias")
        results[h] = train_horizon(
            df,
            h,
            models_dir=models_dir,
            data_processed_dir=data_processed_dir,
            cutoff_date=cutoff_date,
        )

    return results
