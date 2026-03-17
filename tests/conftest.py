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
    return pd.DataFrame({
        "city": ["CityA"] * 100 + ["CityB"] * 100,
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
        "is_successful": rng.integers(0, 2, n).tolist(),
    })
