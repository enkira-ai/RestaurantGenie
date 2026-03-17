import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURE_COLS = [
    "restaurants_250m", "restaurants_500m", "restaurants_1000m",
    "restaurants_same_cuisine_250m", "restaurants_same_cuisine_500m", "restaurants_same_cuisine_1000m",
    "bars_250m", "bars_500m", "bars_1000m",
    "offices_250m", "offices_500m", "offices_1000m",
    "hotels_250m", "hotels_500m", "hotels_1000m",
    "transit_stops_250m", "transit_stops_500m", "transit_stops_1000m",
    "schools_250m", "schools_500m", "schools_1000m",
    "median_income", "total_population", "median_age",
    "cuisine_encoded", "price_level",
]
TARGET_COL = "is_successful"


def make_splits(
    df: pd.DataFrame, test_city_fraction: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (train_search_set, calibration_set, test_df).
    test_df contains held-out cities. calibration_set is 20% of remaining rows.
    """
    rng = np.random.default_rng(random_state)
    cities = sorted(df["city"].unique())
    n_test = max(1, round(len(cities) * test_city_fraction))
    test_cities = set(rng.choice(cities, size=n_test, replace=False).tolist())

    test_df = df[df["city"].isin(test_cities)].copy()
    train_df = df[~df["city"].isin(test_cities)].copy()

    cal_mask = rng.random(len(train_df)) < 0.20
    calibration_set = train_df[cal_mask].copy()
    train_search_set = train_df[~cal_mask].copy()

    return train_search_set, calibration_set, test_df


def encode_cuisine_column(
    df: pd.DataFrame, params_path: str | Path
) -> tuple[pd.DataFrame, dict]:
    """Add cuisine_encoded column. Returns (df_with_encoding, label_map)."""
    with open(params_path) as f:
        params = json.load(f)
    label_map: dict = params["cuisine_label_map"]
    df = df.copy()
    df["cuisine_encoded"] = df["cuisine"].map(
        lambda c: label_map.get(c, 0)
    )
    return df, label_map


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_folds: int = 5,
    random_state: int = 42,
) -> list[str]:
    """5-fold CV feature selection using SHAP + permutation importance.
    Returns list of surviving feature names.
    """
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    shap_means = np.zeros(len(feature_names))
    perm_means = np.zeros(len(feature_names))

    for train_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = lgb.LGBMClassifier(n_estimators=100, random_state=random_state, verbose=-1)
        model.fit(X_tr, y_tr)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # binary: take positive class
        elif hasattr(shap_values, 'shape') and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]  # (samples, features, classes) -> positive class
        shap_means += np.abs(shap_values).mean(axis=0)

        pi_result = permutation_importance(
            model, X_val, y_val, n_repeats=5, random_state=random_state, scoring="roc_auc"
        )
        perm_means += pi_result.importances_mean

    shap_means /= n_folds
    perm_means /= n_folds

    top_shap = shap_means.max()
    threshold = 0.01 * top_shap

    selected = [
        name for i, name in enumerate(feature_names)
        if shap_means[i] >= threshold and perm_means[i] >= 0
    ]
    print(f"Feature selection: {len(selected)} of {len(feature_names)} features retained")
    return selected


def _geographic_cv_folds(cities: np.ndarray, n_folds: int = 5):
    """Yield (train_idx, val_idx) splitting by city groups."""
    unique_cities = sorted(set(cities))
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(unique_cities).tolist()
    city_folds = [shuffled[i::n_folds] for i in range(n_folds)]
    for fold_cities in city_folds:
        val_mask = np.isin(cities, fold_cities)
        yield np.where(~val_mask)[0], np.where(val_mask)[0]


def search_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    cities: np.ndarray,
    n_trials: int = 50,
    random_state: int = 42,
) -> tuple[dict, float, float]:
    """Optuna search. Returns (best_params, mean_cv_n_estimators, best_cv_score)."""

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "bagging_freq": 1,
            "verbose": -1,
            "random_state": random_state,
        }
        scores = []
        n_estimators_list = []
        for train_idx, val_idx in _geographic_cv_folds(cities, n_folds=5):
            if len(val_idx) == 0 or len(train_idx) == 0:
                continue
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            model = lgb.LGBMClassifier(n_estimators=500, **params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
            n_estimators_list.append(model.best_iteration_ or model.n_estimators)
            proba = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, proba))

        trial.set_user_attr("mean_n_estimators", float(np.mean(n_estimators_list)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    best_params = best_trial.params
    mean_n_est = best_trial.user_attrs.get("mean_n_estimators", 100.0)
    best_cv_score = float(best_trial.value)
    print(f"Best CV ROC-AUC: {best_cv_score:.4f} | mean n_estimators: {mean_n_est:.0f}")
    return best_params, mean_n_est, best_cv_score
