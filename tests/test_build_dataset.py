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
