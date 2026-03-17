# tests/test_train_model.py

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
