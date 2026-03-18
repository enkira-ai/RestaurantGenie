import json
import pytest
import pandas as pd
import numpy as np

FAKE_OSM_RESPONSE = {
    "elements": [
        {"type": "node", "lat": 36.170, "lon": -115.140, "tags": {"amenity": "bar"}},
        {"type": "node", "lat": 36.151, "lon": -115.150, "tags": {"amenity": "restaurant", "cuisine": "pizza"}},
        {"type": "node", "lat": 36.160, "lon": -115.130, "tags": {"office": "company"}},
        {"type": "node", "lat": 36.140, "lon": -115.160, "tags": {"tourism": "hotel"}},
        {"type": "node", "lat": 36.180, "lon": -115.120, "tags": {"public_transport": "stop_position"}},
        {"type": "node", "lat": 36.150, "lon": -115.141, "tags": {"amenity": "school"}},
    ]
}

FAKE_CENSUS_ACS_RESPONSE = [
    ["B19013_001E", "B01003_001E", "B01002_001E", "state", "county", "tract"],
    ["75000", "50000", "34", "32", "003", "001000"],
]

YELP_FIXTURE_RECORDS = [
    {"business_id": "biz1", "name": "Joe's Pizza", "city": "Las Vegas", "state": "NV",
     "latitude": 36.17, "longitude": -115.14, "categories": "Restaurants, Pizza, Italian",
     "stars": 4.5, "review_count": 200, "is_open": 1,
     "attributes": {"RestaurantsPriceRange2": "2"}},
    {"business_id": "biz2", "name": "Taco Place", "city": "Las Vegas", "state": "NV",
     "latitude": 36.15, "longitude": -115.15, "categories": "Restaurants, Mexican",
     "stars": 3.8, "review_count": 80, "is_open": 1,
     "attributes": {"RestaurantsPriceRange2": "1"}},
    {"business_id": "biz3", "name": "Closed Diner", "city": "Las Vegas", "state": "NV",
     "latitude": 36.16, "longitude": -115.16, "categories": "Restaurants, American (Traditional)",
     "stars": 2.5, "review_count": 30, "is_open": 0,
     "attributes": {"RestaurantsPriceRange2": "1"}},
    {"business_id": "biz4", "name": "Fine Dining", "city": "Phoenix", "state": "AZ",
     "latitude": 33.45, "longitude": -112.07, "categories": "Restaurants, French",
     "stars": 4.8, "review_count": 500, "is_open": 1,
     "attributes": {"RestaurantsPriceRange2": "4"}},
    # Non-restaurant — must be filtered out
    {"business_id": "biz5", "name": "Hair Salon", "city": "Las Vegas", "state": "NV",
     "latitude": 36.17, "longitude": -115.13, "categories": "Beauty & Spas",
     "stars": 4.0, "review_count": 60, "is_open": 1, "attributes": {}},
]


@pytest.fixture
def yelp_jsonl_file(tmp_path):
    path = tmp_path / "yelp_academic_dataset_business.json"
    with open(path, "w") as f:
        for record in YELP_FIXTURE_RECORDS:
            f.write(json.dumps(record) + "\n")
    return path


@pytest.fixture
def small_restaurant_df():
    return pd.DataFrame({
        "business_id": ["biz1", "biz2", "biz3", "biz4"],
        "name": ["Joe's Pizza", "Taco Place", "Closed Diner", "Fine Dining"],
        "city": ["Las Vegas", "Las Vegas", "Las Vegas", "Phoenix"],
        "state": ["NV", "NV", "NV", "AZ"],
        "lat": [36.17, 36.15, 36.16, 33.45],
        "lon": [-115.14, -115.15, -115.16, -112.07],
        "cuisine": ["italian", "mexican", None, "french"],
        "price_level": [2.0, 1.0, 1.0, 4.0],
        "rating": [4.5, 3.8, 2.5, 4.8],
        "review_count": [200, 80, 30, 500],
        "is_open": [1, 1, 0, 1],
    })


@pytest.fixture
def synthetic_model_df():
    """200-row synthetic dataset across 2 cities for model pipeline tests."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "city": ["CityA"] * 100 + ["CityB"] * 100,
        "state": ["PA"] * 100 + ["TX"] * 100,
        "cuisine": rng.choice(["italian", "mexican", "chinese"], n).tolist(),
        "price_level": rng.integers(1, 5, n).tolist(),
        "restaurants_250m": rng.integers(0, 20, n).tolist(),
        "restaurants_500m": rng.integers(0, 40, n).tolist(),
        "restaurants_1000m": rng.integers(0, 80, n).tolist(),
        "restaurants_same_cuisine_250m": rng.integers(0, 5, n).tolist(),
        "restaurants_same_cuisine_500m": rng.integers(0, 10, n).tolist(),
        "restaurants_same_cuisine_1000m": rng.integers(0, 20, n).tolist(),
        "bars_250m": rng.integers(0, 10, n).tolist(),
        "bars_500m": rng.integers(0, 20, n).tolist(),
        "bars_1000m": rng.integers(0, 40, n).tolist(),
        "offices_250m": rng.integers(0, 10, n).tolist(),
        "offices_500m": rng.integers(0, 20, n).tolist(),
        "offices_1000m": rng.integers(0, 40, n).tolist(),
        "hotels_250m": rng.integers(0, 5, n).tolist(),
        "hotels_500m": rng.integers(0, 10, n).tolist(),
        "hotels_1000m": rng.integers(0, 20, n).tolist(),
        "transit_stops_250m": rng.integers(0, 5, n).tolist(),
        "transit_stops_500m": rng.integers(0, 10, n).tolist(),
        "transit_stops_1000m": rng.integers(0, 20, n).tolist(),
        "schools_250m": rng.integers(0, 3, n).tolist(),
        "schools_500m": rng.integers(0, 5, n).tolist(),
        "schools_1000m": rng.integers(0, 10, n).tolist(),
        "median_income": rng.integers(30000, 150000, n).tolist(),
        "total_population": rng.integers(1000, 50000, n).tolist(),
        "median_age": rng.uniform(25.0, 55.0, n).tolist(),
        "cuisine_encoded": rng.integers(0, 3, n).tolist(),
        "review_count": rng.integers(5, 500, n).tolist(),
        "is_successful": rng.integers(0, 2, n).tolist(),
        # Derived features
        "same_cuisine_saturation_250m": rng.uniform(0, 1, n).tolist(),
        "same_cuisine_saturation_500m": rng.uniform(0, 1, n).tolist(),
        "same_cuisine_saturation_1000m": rng.uniform(0, 1, n).tolist(),
        "restaurant_bar_ratio_500m": rng.uniform(0, 10, n).tolist(),
        "foot_traffic_proxy_500m": rng.uniform(0, 20, n).tolist(),
        "demand_per_restaurant_500m": rng.uniform(0, 5, n).tolist(),
        "income_office_interaction": rng.uniform(0, 5, n).tolist(),
        "income_per_capita_proxy": rng.uniform(0, 2000, n).tolist(),
        "poi_diversity_500m": rng.integers(0, 7, n).tolist(),
        "total_pois_500m": rng.integers(0, 100, n).tolist(),
    })
    # Spatial census features (use same values as median_income etc.)
    for radius in [500, 1000]:
        df[f"median_income_{radius}m_avg"] = df["median_income"]
        df[f"total_population_{radius}m_avg"] = df["total_population"]
        df[f"median_age_{radius}m_avg"] = df["median_age"]
    df["income_variance_1000m"] = rng.random(200) * 10000

    df["business_id"] = [f"biz_{i}" for i in range(200)]

    # Price tier and income×price features
    df["price_tier_success_rate"] = 0.5
    df["price_tier_count_log"] = np.log1p(20)
    df["median_income_x_price"] = df["median_income"] / 100000.0 * df["price_level"]

    # State-relative income features (mirrors train_model computation)
    _fixture_state_medians = {"PA": 67_587, "TX": 67_321}
    _us_fallback = 74_580
    df["_state_median"] = df["state"].map(_fixture_state_medians).fillna(_us_fallback)
    df["income_relative_to_state"] = (df["median_income"] / df["_state_median"]).clip(0.1, 5.0)
    df["income_level_state_cat"] = pd.cut(
        df["income_relative_to_state"].fillna(1.0),
        bins=[0.0, 0.75, 1.25, 6.0], labels=[0, 1, 2],
    ).astype("float64")
    _fixture_state_map = {"PA": 0, "TX": 1}
    df["state_encoded"] = df["state"].map(_fixture_state_map).fillna(-1).astype("float64")
    df.drop(columns=["_state_median"], inplace=True)

    # Yelp spatial features
    df["avg_price_1km"] = rng.uniform(1, 4, 200)
    df["median_price_1km"] = rng.uniform(1, 4, 200)
    df["avg_rating_1km"] = rng.uniform(3.0, 4.5, 200)
    df["avg_reviews_1km"] = rng.uniform(50, 500, 200)
    df["total_reviews_1km"] = rng.uniform(500, 5000, 200)
    df["same_price_1km"] = rng.integers(0, 10, 200)
    df["cuisine_entropy_1km"] = rng.uniform(0.5, 2.0, 200)
    df["restaurants_2km"] = rng.integers(5, 50, 200)
    df["price_mismatch_1km"] = rng.uniform(0, 2, 200)
    df["cuisine_gap"] = rng.uniform(100, 10000, 200)
    df["cluster_score"] = rng.uniform(0, 5, 200)
    df["distance_city_center"] = rng.uniform(0.1, 10.0, 200)
    df["same_cuisine_price_1km"] = rng.integers(0, 5, 200)
    return df
