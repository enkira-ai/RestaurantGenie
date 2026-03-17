# tests/test_train_model.py

import pandas as pd


def test_make_splits_test_cities_are_disjoint(synthetic_model_df):
    from src.train_model import make_splits
    train_search, calibration, test_df = make_splits(synthetic_model_df, random_state=42)
    train_cities = set(train_search["city"]) | set(calibration["city"])
    test_cities = set(test_df["city"])
    assert train_cities.isdisjoint(test_cities), "Test cities leaked into train"


def test_make_splits_sizes(synthetic_model_df):
    from src.train_model import make_splits
    train_search, calibration, test_df = make_splits(synthetic_model_df, random_state=42)
    total = len(train_search) + len(calibration) + len(test_df)
    assert total == len(synthetic_model_df)
    # test_city_fraction=0.2 on 2 cities → exactly 1 city held out
    assert len(set(test_df["city"])) == 1


def test_encode_cuisine_known_and_unknown(tmp_path):
    from src.train_model import encode_cuisine_column
    params_path = tmp_path / "normalization_params.json"
    import json
    params_path.write_text(json.dumps({
        "p95_log_reviews": 5.3,
        "cuisine_label_map": {"other": 0, "italian": 1, "mexican": 2},
    }))
    import pandas as pd
    df = pd.DataFrame({"cuisine": ["italian", "mexican", "sushi", None]})
    result, label_map = encode_cuisine_column(df, params_path)
    assert result["cuisine_encoded"].tolist() == [1, 2, 0, 0]


def test_select_features_drops_noise_feature(synthetic_model_df):
    from src.train_model import select_features, FEATURE_COLS, TARGET_COL
    import numpy as np
    rng = np.random.default_rng(0)
    # Build a dataset with genuine signal from several features
    df = synthetic_model_df.copy()
    # Target is a noisy combination of several real features so multiple survive
    signal = (
        df["restaurants_250m"] * 0.3
        + df["bars_250m"] * 0.2
        + df["median_income"] / 150000.0 * 0.3
        + df["transit_stops_250m"] * 0.2
    )
    prob = 1 / (1 + np.exp(-(signal - signal.mean()) / signal.std()))
    df[TARGET_COL] = (rng.random(len(df)) < prob).astype(int)
    df["pure_noise"] = rng.random(len(df))
    features = FEATURE_COLS + ["pure_noise"]
    X = df[features].values
    y = df[TARGET_COL].values
    selected = select_features(X, y, feature_names=features)
    # Noise column should not survive selection
    assert "pure_noise" not in selected
    # Real features should survive
    assert len(selected) >= 5


def test_search_hyperparameters_returns_params_and_n_estimators(synthetic_model_df):
    from src.train_model import search_hyperparameters, FEATURE_COLS, TARGET_COL
    import numpy as np
    X = synthetic_model_df[FEATURE_COLS].values
    y = synthetic_model_df[TARGET_COL].values
    cities = synthetic_model_df["city"].values
    best_params, mean_n_est, best_cv_score = search_hyperparameters(
        X, y, cities=cities, n_trials=2, random_state=42
    )
    assert "num_leaves" in best_params
    assert "learning_rate" in best_params
    assert mean_n_est > 0
    assert 0.0 <= best_cv_score <= 1.0


def test_fit_final_model_produces_calibrated_output(synthetic_model_df):
    from src.train_model import fit_final_model, FEATURE_COLS, TARGET_COL
    import numpy as np
    df = synthetic_model_df
    X_train = df[FEATURE_COLS].fillna(0).values[:160]
    y_train = df[TARGET_COL].values[:160]
    X_cal = df[FEATURE_COLS].fillna(0).values[160:]
    y_cal = df[TARGET_COL].values[160:]
    best_params = {"num_leaves": 31, "max_depth": 5, "min_child_samples": 20,
                   "lambda_l1": 0.1, "lambda_l2": 0.1, "feature_fraction": 0.8,
                   "bagging_fraction": 0.8, "learning_rate": 0.05}
    calibrated, base_lgbm = fit_final_model(
        X_train, y_train, X_cal, y_cal, best_params, n_estimators=50
    )
    proba = calibrated.predict_proba(X_cal)[:, 1]
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    assert calibrated.estimator is base_lgbm


def test_save_and_load_model_artifacts(synthetic_model_df, tmp_path):
    from src.train_model import fit_final_model, save_artifacts, FEATURE_COLS, TARGET_COL
    import pickle
    df = synthetic_model_df
    X = df[FEATURE_COLS].fillna(0).values
    y = df[TARGET_COL].values
    best_params = {"num_leaves": 15, "learning_rate": 0.1, "max_depth": 3,
                   "min_child_samples": 20, "lambda_l1": 0, "lambda_l2": 0,
                   "feature_fraction": 0.8, "bagging_fraction": 0.8}
    calibrated, base_lgbm = fit_final_model(X[:160], y[:160], X[160:], y[160:], best_params, n_estimators=20)

    params_path = tmp_path / "normalization_params.json"
    import json
    params_path.write_text(json.dumps({"p95_log_reviews": 5.3, "cuisine_label_map": {"other": 0}}))

    save_artifacts(
        calibrated_model=calibrated,
        base_lgbm=base_lgbm,
        selected_features=FEATURE_COLS,
        best_params=best_params,
        params_path=params_path,
        model_dir=tmp_path,
    )

    assert (tmp_path / "model.pkl").exists()
    assert (tmp_path / "shap_explainer.pkl").exists()
    loaded = pickle.loads((tmp_path / "model.pkl").read_bytes())
    proba = loaded.predict_proba(X[160:])[:, 1]
    assert proba.min() >= 0.0


def test_train_model_end_to_end(synthetic_model_df, tmp_path):
    """Smoke test: full pipeline runs without error and creates all artifacts."""
    from src.train_model import train_model
    import json

    parquet_path = tmp_path / "restaurant_features.parquet"
    params_path = tmp_path / "normalization_params.json"
    models_dir = tmp_path / "models"

    df = synthetic_model_df.copy()
    df.to_parquet(parquet_path)
    params_path.write_text(json.dumps({
        "p95_log_reviews": 5.3,
        "cuisine_label_map": {"other": 0, "italian": 1, "mexican": 2, "chinese": 3},
    }))

    train_model(
        parquet_path=parquet_path,
        params_path=params_path,
        models_dir=models_dir,
        n_trials=2,   # smoke test: 2 optuna trials
    )

    assert (models_dir / "model.pkl").exists()
    assert (models_dir / "shap_explainer.pkl").exists()
    assert (models_dir / "performance_report.txt").exists()
    df_out = pd.read_parquet(parquet_path)
    assert "predicted_probability" in df_out.columns
    # Verify performance_report.txt content
    report = (models_dir / "performance_report.txt").read_text()
    assert "Test-city ROC-AUC" in report
    assert "Test-city Brier score" in report
    assert "Selected features" in report
    # Verify predicted_probability values are valid probabilities
    proba = df_out["predicted_probability"]
    assert proba.min() >= 0.0 and proba.max() <= 1.0
