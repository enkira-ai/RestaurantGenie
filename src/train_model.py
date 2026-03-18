import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import shap
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

optuna.logging.set_verbosity(optuna.logging.WARNING)


class _PlattWrapper:
    """Platt-scaling wrapper for a pre-fitted classifier.

    Replaces CalibratedClassifierCV(estimator, cv='prefit', method='sigmoid'),
    which was removed in scikit-learn 1.8.
    """

    def __init__(self, estimator):
        self.estimator = estimator
        self._sigmoid = LogisticRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_PlattWrapper":
        raw = self.estimator.predict_proba(X)[:, 1].reshape(-1, 1)
        self._sigmoid.fit(raw, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.estimator.predict_proba(X)[:, 1].reshape(-1, 1)
        return self._sigmoid.predict_proba(raw)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

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
    # Derived features
    "same_cuisine_saturation_250m", "same_cuisine_saturation_500m", "same_cuisine_saturation_1000m",
    "restaurant_bar_ratio_500m",
    "foot_traffic_proxy_500m",
    "demand_per_restaurant_500m",
    "income_office_interaction",
    "income_per_capita_proxy",
    "poi_diversity_500m",
    "total_pois_500m",
    # Spatial census features
    "median_income_500m_avg", "median_income_1000m_avg",
    "total_population_500m_avg", "total_population_1000m_avg",
    "median_age_500m_avg", "median_age_1000m_avg",
    "income_variance_1000m",
    # Price tier features
    "price_tier_success_rate",
    "price_tier_count_log",
    # Yelp spatial features
    "avg_price_1km", "median_price_1km",
    "avg_rating_1km", "avg_reviews_1km", "total_reviews_1km",
    "same_price_1km", "cuisine_entropy_1km",
    "restaurants_2km",
    "price_mismatch_1km",
    "cuisine_gap",
    "cluster_score",
    "distance_city_center",
    "same_cuisine_price_1km",
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

        if not scores:
            trial.set_user_attr("mean_n_estimators", 100.0)
            return 0.5
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


def fit_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    best_params: dict,
    n_estimators: int,
) -> tuple:
    """Refit LightGBM on train_search_set, calibrate on calibration_set.
    Returns (calibrated_model, base_lgbm).
    """
    params = {**best_params, "verbose": -1, "random_state": 42}
    base_lgbm = lgb.LGBMClassifier(n_estimators=n_estimators, **params)
    base_lgbm.fit(X_train, y_train)

    calibrated = _PlattWrapper(base_lgbm)
    calibrated.fit(X_cal, y_cal)
    return calibrated, base_lgbm


def save_artifacts(
    calibrated_model,
    base_lgbm,
    selected_features: list[str],
    best_params: dict,
    params_path: str | Path,
    model_dir: str | Path,
) -> None:
    """Save model.pkl, shap_explainer.pkl, updated normalization_params.json."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(calibrated_model, f)

    explainer = shap.TreeExplainer(base_lgbm)
    with open(model_dir / "shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)

    # Update normalization_params.json with selected_features and best_params
    with open(params_path) as f:
        params = json.load(f)
    params["selected_features"] = selected_features
    params["best_hyperparameters"] = best_params
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"Saved model artifacts to {model_dir}/")


def evaluate_on_test(
    calibrated_model, base_lgbm, X_test: np.ndarray, y_test: np.ndarray
) -> dict:
    """Returns test metrics: uncalibrated ROC-AUC (base model) and calibrated Brier score."""
    proba_cal = calibrated_model.predict_proba(X_test)[:, 1]
    proba_raw = base_lgbm.predict_proba(X_test)[:, 1]
    return {
        "test_roc_auc_uncalibrated": roc_auc_score(y_test, proba_raw),
        "test_brier_calibrated": brier_score_loss(y_test, proba_cal),
    }


def train_model(
    parquet_path: str | Path,
    params_path: str | Path,
    models_dir: str | Path,
    n_trials: int = 50,
    random_state: int = 42,
) -> None:
    """Full Stage 2 pipeline."""
    models_dir = Path(models_dir)

    print("Loading dataset...")
    df = pd.read_parquet(parquet_path)
    df, label_map = encode_cuisine_column(df, params_path)

    # Ensure all feature columns exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    print("Making data splits...")
    train_search, calibration_set, test_df = make_splits(df, random_state=random_state)

    X_ts = train_search[FEATURE_COLS].values.astype(float)
    y_ts = train_search[TARGET_COL].values
    cities_ts = train_search["city"].values
    X_cal = calibration_set[FEATURE_COLS].values.astype(float)
    y_cal = calibration_set[TARGET_COL].values
    X_test = test_df[FEATURE_COLS].values.astype(float)
    y_test = test_df[TARGET_COL].values

    print("Running feature selection...")
    selected_features = select_features(X_ts, y_ts, feature_names=FEATURE_COLS)
    if not selected_features:
        print("Warning: no features survived selection, falling back to all features.")
        selected_features = FEATURE_COLS
    feat_idx = [FEATURE_COLS.index(f) for f in selected_features]
    X_ts_sel = X_ts[:, feat_idx]
    X_cal_sel = X_cal[:, feat_idx]
    X_test_sel = X_test[:, feat_idx]

    print(f"Running Optuna search ({n_trials} trials)...")
    best_params, mean_cv_n_est, best_cv_score = search_hyperparameters(
        X_ts_sel, y_ts, cities=cities_ts, n_trials=n_trials, random_state=random_state
    )

    n_estimators = round(mean_cv_n_est * 1.25)
    print(f"Refitting with n_estimators={n_estimators}...")
    calibrated_model, base_lgbm = fit_final_model(
        X_ts_sel, y_ts, X_cal_sel, y_cal, best_params, n_estimators
    )

    print("Evaluating on test cities...")
    metrics = evaluate_on_test(calibrated_model, base_lgbm, X_test_sel, y_test)

    save_artifacts(
        calibrated_model=calibrated_model,
        base_lgbm=base_lgbm,
        selected_features=selected_features,
        best_params=best_params,
        params_path=params_path,
        model_dir=models_dir,
    )

    # Score all rows and write predicted_probability back to parquet
    X_all = df[FEATURE_COLS].values.astype(float)[:, feat_idx]
    df["predicted_probability"] = calibrated_model.predict_proba(X_all)[:, 1]
    df.to_parquet(parquet_path, index=False)
    print("Wrote predicted_probability to parquet.")

    # Performance report
    report_lines = [
        f"Feature selection: {len(selected_features)} of {len(FEATURE_COLS)} features retained",
        f"Selected features: {selected_features}",
        f"Best hyperparameters: {best_params}",
        f"n_estimators (refit): {n_estimators}",
        f"Geographic CV ROC-AUC (train_search_set, 5-fold): {best_cv_score:.4f}",
        f"Test-city ROC-AUC (held-out, uncalibrated): {metrics['test_roc_auc_uncalibrated']:.4f}",
        f"Test-city Brier score (calibrated): {metrics['test_brier_calibrated']:.4f}",
    ]
    report = "\n".join(report_lines)
    print("\n=== Performance Report ===")
    print(report)
    (models_dir / "performance_report.txt").write_text(report)


if __name__ == "__main__":
    train_model(
        parquet_path="data/processed/restaurant_features.parquet",
        params_path="models/normalization_params.json",
        models_dir="models",
    )
