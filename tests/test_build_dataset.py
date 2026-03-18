import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from tests.conftest import FAKE_OSM_RESPONSE, FAKE_CENSUS_ACS_RESPONSE


def test_load_yelp_businesses_filters_to_restaurants(yelp_jsonl_file):
    from src.build_dataset import load_yelp_businesses
    df = load_yelp_businesses(yelp_jsonl_file)
    assert len(df) == 4  # 5 records minus 1 non-restaurant
    assert "business_id" in df.columns
    assert set(df.columns) >= {"name", "city", "state", "lat", "lon",
                                "cuisine", "price_level", "rating", "review_count", "is_open"}


def test_load_yelp_businesses_cuisine_extraction(yelp_jsonl_file):
    from src.build_dataset import load_yelp_businesses
    df = load_yelp_businesses(yelp_jsonl_file)
    row = df[df["business_id"] == "biz1"].iloc[0]
    assert row["cuisine"] in {"pizza", "italian"}  # either is valid
    row2 = df[df["business_id"] == "biz3"].iloc[0]
    assert row2["cuisine"] in {"american", None}


def test_load_yelp_businesses_price_level_parsed(yelp_jsonl_file):
    from src.build_dataset import load_yelp_businesses
    df = load_yelp_businesses(yelp_jsonl_file)
    row = df[df["business_id"] == "biz1"].iloc[0]
    assert row["price_level"] == 2.0
    row4 = df[df["business_id"] == "biz4"].iloc[0]
    assert row4["price_level"] == 4.0


def test_enrich_with_osm_features_adds_poi_columns(small_restaurant_df, mocker):
    from src.build_dataset import enrich_with_osm_features
    mock_resp = MagicMock()
    mock_resp.json.return_value = FAKE_OSM_RESPONSE
    mock_resp.raise_for_status.return_value = None
    mocker.patch("src.features.requests.post", return_value=mock_resp)
    mocker.patch("src.build_dataset.time.sleep")  # don't actually sleep in tests

    result = enrich_with_osm_features(small_restaurant_df)

    assert "restaurants_500m" in result.columns
    assert "bars_250m" in result.columns
    assert "offices_1000m" in result.columns
    assert "transit_stops_500m" in result.columns
    assert "schools_250m" in result.columns
    assert len(result) == len(small_restaurant_df)


def test_enrich_with_census_adds_demographic_columns(small_restaurant_df, mocker):
    from src.build_dataset import enrich_with_census
    mocker.patch(
        "censusgeocode.CensusGeocode.coordinates",
        return_value=[{"geographies": {"Census Tracts": [
            {"TRACT": "001000", "COUNTY": "003", "STATE": "32"}
        ]}}],
    )
    mock_resp = MagicMock()
    mock_resp.json.return_value = FAKE_CENSUS_ACS_RESPONSE
    mock_resp.raise_for_status.return_value = None
    mocker.patch("src.build_dataset.requests.get", return_value=mock_resp)

    result = enrich_with_census(small_restaurant_df)

    assert "median_income" in result.columns
    assert "total_population" in result.columns
    assert "median_age" in result.columns
    assert len(result) == len(small_restaurant_df)
    assert result["median_income"].iloc[0] == 75000.0
    assert result["total_population"].iloc[0] == 50000.0
    assert result["median_age"].iloc[0] == 34.0


def test_enrich_with_census_caches_fips(small_restaurant_df, mocker):
    """Census ACS API called only once per unique FIPS, not once per restaurant."""
    from src.build_dataset import enrich_with_census
    mocker.patch(
        "censusgeocode.CensusGeocode.coordinates",
        return_value=[{"geographies": {"Census Tracts": [
            {"TRACT": "001000", "COUNTY": "003", "STATE": "32"}
        ]}}],
    )
    mock_resp = MagicMock()
    mock_resp.json.return_value = FAKE_CENSUS_ACS_RESPONSE
    mock_resp.raise_for_status.return_value = None
    # Patch the requests.get used inside build_dataset (imported at module level there)
    acs_mock = mocker.patch("src.build_dataset.requests.get", return_value=mock_resp)

    enrich_with_census(small_restaurant_df)

    # 4 restaurants but only 1 unique FIPS → ACS should be called once
    assert acs_mock.call_count == 1


def test_compute_success_labels_creates_column(small_restaurant_df):
    from src.build_dataset import compute_success_labels
    p95 = float(np.log1p(small_restaurant_df["review_count"]).quantile(0.95))
    result = compute_success_labels(small_restaurant_df, p95_log_reviews=p95, review_stats_path=None)
    assert "success_score" in result.columns
    assert "is_successful" in result.columns
    assert result["is_successful"].isin([0, 1]).all()
    # Within Las Vegas, biz1 (rating=4.5, 200 reviews) should outscore biz3 (rating=2.5, 30 reviews)
    biz1_score = result.loc[result["business_id"] == "biz1", "success_score"].values[0]
    biz3_score = result.loc[result["business_id"] == "biz3", "success_score"].values[0]
    assert biz1_score > biz3_score


def test_compute_success_labels_small_groups_merged():
    """Groups <10 restaurants → cuisine='other' for percentile calculation."""
    from src.build_dataset import compute_success_labels
    df = pd.DataFrame({
        "business_id": [f"biz_{i}" for i in range(20)],
        "city": ["NYC"] * 5 + ["NYC"] * 15,
        "cuisine": ["rare_cuisine"] * 5 + ["italian"] * 15,
        "rating": [4.0] * 20,
        "review_count": [100] * 20,
        "is_open": [1] * 20,
        "price_level": [2.0] * 20,
    })
    p95 = float(np.log1p(df["review_count"]).quantile(0.95))
    result = compute_success_labels(df, p95_log_reviews=p95, review_stats_path=None)
    # Rare cuisine group (5 rows) merged into 'other' — no crash
    assert "is_successful" in result.columns
    assert len(result) == 20


def test_build_dataset_produces_parquet(yelp_jsonl_file, tmp_path, mocker):
    from src.build_dataset import build_dataset
    mock_osm = MagicMock(); mock_osm.json.return_value = FAKE_OSM_RESPONSE
    mock_osm.raise_for_status.return_value = None
    mocker.patch("src.features.requests.post", return_value=mock_osm)
    mocker.patch("src.build_dataset.time.sleep")
    mocker.patch(
        "censusgeocode.CensusGeocode.coordinates",
        return_value=[{"geographies": {"Census Tracts": [
            {"TRACT": "001000", "COUNTY": "003", "STATE": "32"}
        ]}}],
    )
    mock_census = MagicMock(); mock_census.json.return_value = FAKE_CENSUS_ACS_RESPONSE
    mock_census.raise_for_status.return_value = None
    mocker.patch("src.build_dataset.requests.get", return_value=mock_census)

    out_parquet = tmp_path / "restaurant_features.parquet"
    out_params = tmp_path / "normalization_params.json"

    build_dataset(yelp_jsonl_file, out_parquet, out_params)

    import json
    assert out_parquet.exists()
    df = pd.read_parquet(out_parquet)
    assert len(df) == 4  # 4 restaurants
    assert "is_successful" in df.columns
    assert "restaurants_500m" in df.columns
    assert out_params.exists()
    params = json.loads(out_params.read_text())
    assert "p95_log_reviews" in params
    assert "cuisine_label_map" in params
