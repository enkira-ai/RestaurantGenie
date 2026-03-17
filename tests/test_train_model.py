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
    # test_df ≈ 20% of cities; with 2 cities exactly 1 goes to test
    assert len(test_df) == 100  # CityB (100 rows)


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
