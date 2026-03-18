# tests/test_predict.py
import pytest
import pandas as pd
import numpy as np
import json


def test_geocode_address_success(mocker):
    from src.predict import geocode_address
    mock_location = mocker.MagicMock()
    mock_location.latitude = 36.17
    mock_location.longitude = -115.14
    mocker.patch("src.predict.Nominatim.geocode", return_value=mock_location)
    lat, lon = geocode_address("123 Main St, Las Vegas, NV")
    assert abs(lat - 36.17) < 0.001
    assert abs(lon + 115.14) < 0.001


def test_geocode_address_failure_raises(mocker):
    from src.predict import geocode_address
    mocker.patch("src.predict.Nominatim.geocode", return_value=None)
    with pytest.raises(SystemExit):
        geocode_address("not a real address ever")


def test_assemble_feature_vector_uses_selected_features(tmp_path, mocker):
    from src.predict import assemble_feature_vector
    params = {
        "p95_log_reviews": 5.3,
        "cuisine_label_map": {"other": 0, "italian": 1},
        "selected_features": ["restaurants_500m", "median_income", "cuisine_encoded", "price_level"],
    }
    (tmp_path / "normalization_params.json").write_text(json.dumps(params))

    mock_features = {
        "restaurants_250m": 5, "restaurants_500m": 12, "restaurants_1000m": 30,
        "restaurants_same_cuisine_250m": None, "restaurants_same_cuisine_500m": None,
        "restaurants_same_cuisine_1000m": None,
        "bars_250m": 2, "bars_500m": 4, "bars_1000m": 10,
        "offices_250m": 3, "offices_500m": 7, "offices_1000m": 15,
        "hotels_250m": 0, "hotels_500m": 1, "hotels_1000m": 3,
        "transit_stops_250m": 1, "transit_stops_500m": 2, "transit_stops_1000m": 5,
        "schools_250m": 0, "schools_500m": 1, "schools_1000m": 2,
        "median_income": 80000.0, "total_population": 40000.0, "median_age": 35.0,
    }
    mocker.patch("src.predict.generate_neighborhood_features", return_value=mock_features)

    vector, names = assemble_feature_vector(36.17, -115.14, "italian", 2,
                                            tmp_path / "normalization_params.json")
    assert names == ["restaurants_500m", "median_income", "cuisine_encoded", "price_level"]
    assert vector[2] == 1   # italian → 1
    assert vector[3] == 2   # price_level


def test_compute_percentile_rank_basic(tmp_path):
    from src.predict import compute_percentile_rank
    df = pd.DataFrame({
        "cuisine": ["italian"] * 10,
        "city": ["Las Vegas"] * 10,
        "predicted_probability": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    })
    # Score of 0.75 should be roughly in top 30%
    rank = compute_percentile_rank(0.75, df, cuisine="italian", city="Las Vegas")
    assert 0 <= rank <= 100
    assert rank < 35  # top 30%


def test_compute_percentile_rank_small_group_falls_back():
    from src.predict import compute_percentile_rank
    # Group has only 2 rows → falls back to all-rows distribution
    df = pd.DataFrame({
        "cuisine": ["rare"] * 2 + ["italian"] * 20,
        "city": ["NYC"] * 22,
        "predicted_probability": [0.5, 0.6] + [0.3] * 20,
    })
    rank = compute_percentile_rank(0.55, df, cuisine="rare", city="NYC")
    assert 0 <= rank <= 100


def test_get_shap_pros_cons():
    from src.predict import get_shap_pros_cons
    # Mock explainer output
    feature_names = ["restaurants_500m", "median_income", "offices_500m"]
    shap_values = np.array([-0.3, 0.5, 0.2])
    feature_values = {"restaurants_500m": 20, "median_income": 90000, "offices_500m": 15}
    pros, cons = get_shap_pros_cons(shap_values, feature_names, feature_values)
    assert len(pros) >= 1
    assert len(cons) >= 1
    assert any("median_income" in p["feature"] for p in pros)
    assert any("restaurants_500m" in c["feature"] for c in cons)


def test_find_comparable_restaurants_filters_by_cuisine_and_price():
    from src.predict import find_comparable_restaurants
    df = pd.DataFrame({
        "name": ["A", "B", "C", "D"],
        "cuisine": ["italian", "italian", "mexican", "italian"],
        "price_level": [2.0, 3.0, 2.0, 4.0],
        "rating": [4.5, 4.2, 3.9, 4.8],
        "lat": [36.17, 36.175, 36.16, 36.18],
        "lon": [-115.14, -115.143, -115.15, -115.13],
    })
    results = find_comparable_restaurants(36.17, -115.14, "italian", 2, df, max_distance_km=5)
    cuisines = [r["cuisine"] for r in results]
    assert "mexican" not in cuisines
    prices = [r["price_level"] for r in results]
    assert all(abs(p - 2) <= 1 for p in prices)   # ±1 price level


def test_format_output_contains_key_sections():
    from src.predict import format_output
    output = format_output(
        address="123 Main St, Austin TX",
        cuisine="italian",
        price_level=2,
        probability=0.71,
        percentile_rank=71.0,
        pros=[{"feature": "offices_500m", "label": "high office density", "value": 15, "shap": 0.3}],
        cons=[{"feature": "restaurants_500m", "label": "restaurant saturation", "value": 40, "shap": -0.2}],
        comparables=[{"name": "Olive Garden", "cuisine": "italian",
                      "price_level": 2, "rating": 4.1, "distance_km": 1.2}],
    )
    assert "GOOD LOCATION" in output or "POOR LOCATION" in output
    assert "SUMMARY" in output
    assert "KEY FACTORS" in output
    assert "Olive Garden" in output


def test_run_prediction_end_to_end(tmp_path, mocker):
    """Full predict pipeline with mocked external calls."""
    from src.predict import run_prediction
    import pickle
    import json

    # Build fake model + explainer
    from sklearn.dummy import DummyClassifier
    dummy = DummyClassifier(strategy="constant", constant=1)
    dummy.fit([[0, 0]], [1])
    cal = mocker.MagicMock()
    cal.predict_proba = lambda X: np.array([[0.3, 0.7]] * len(X))
    cal.estimator = dummy

    params = {
        "p95_log_reviews": 5.3,
        "cuisine_label_map": {"other": 0, "italian": 1},
        "selected_features": ["restaurants_500m", "median_income"],
        "best_hyperparameters": {},
    }
    (tmp_path / "normalization_params.json").write_text(json.dumps(params))

    # Minimal reference parquet
    ref_df = pd.DataFrame({
        "name": ["Test Bistro"],
        "cuisine": ["italian"],
        "city": ["Austin"],
        "price_level": [2.0],
        "rating": [4.2],
        "lat": [30.27],
        "lon": [-97.74],
        "predicted_probability": [0.65],
    })
    ref_df.to_parquet(tmp_path / "restaurant_features.parquet")

    explainer_mock = mocker.MagicMock()
    explainer_mock.model.num_feature.return_value = 2
    explainer_mock.shap_values.return_value = np.array([[0.3, -0.1]])

    mocker.patch("src.predict.load_artifacts", return_value=(cal, explainer_mock))
    mocker.patch("src.predict.geocode_address", return_value=(30.27, -97.74))
    mocker.patch("src.predict.generate_neighborhood_features", return_value={
        "restaurants_250m": 5, "restaurants_500m": 10, "restaurants_1000m": 20,
        "restaurants_same_cuisine_250m": None, "restaurants_same_cuisine_500m": None,
        "restaurants_same_cuisine_1000m": None,
        "bars_250m": 2, "bars_500m": 4, "bars_1000m": 8,
        "offices_250m": 3, "offices_500m": 6, "offices_1000m": 12,
        "hotels_250m": 0, "hotels_500m": 1, "hotels_1000m": 2,
        "transit_stops_250m": 1, "transit_stops_500m": 2, "transit_stops_1000m": 4,
        "schools_250m": 0, "schools_500m": 1, "schools_1000m": 2,
        "median_income": 80000.0, "total_population": 40000.0, "median_age": 35.0,
    })

    output = run_prediction(
        address="123 Main St, Austin TX",
        cuisine="italian",
        price_level=2,
        models_dir=tmp_path,
        parquet_path=tmp_path / "restaurant_features.parquet",
        params_path=tmp_path / "normalization_params.json",
    )
    assert "GOOD LOCATION" in output or "POOR LOCATION" in output
    assert "SUMMARY" in output
    assert "KEY FACTORS" in output
