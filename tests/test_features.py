from unittest.mock import MagicMock, patch
from tests.conftest import FAKE_OSM_RESPONSE


def test_fetch_pois_for_bbox_returns_typed_list():
    from src.features import fetch_pois_for_bbox
    mock_resp = MagicMock()
    mock_resp.json.return_value = FAKE_OSM_RESPONSE
    mock_resp.raise_for_status.return_value = None
    with patch("src.features.requests.post", return_value=mock_resp):
        pois = fetch_pois_for_bbox(36.10, -115.20, 36.25, -115.10)
    assert len(pois) == 6
    types = {p["type"] for p in pois}
    assert types == {"bar", "restaurant", "office", "hotel", "transit", "school"}
    for p in pois:
        assert "lat" in p and "lon" in p and "type" in p
    pizza = next(p for p in pois if p["type"] == "restaurant")
    assert pizza["cuisine"] == "pizza"


def test_count_pois_by_type_respects_radius():
    from src.features import count_pois_by_type
    # 1 degree lat ≈ 111 km → 300m ≈ 0.0027 deg, 600m ≈ 0.0054 deg
    center = (40.0, -74.0)
    pois = [
        {"lat": 40.0027, "lon": -74.0, "type": "bar", "cuisine": None},   # ~300m
        {"lat": 40.0054, "lon": -74.0, "type": "bar", "cuisine": None},   # ~600m
        {"lat": 40.008,  "lon": -74.0, "type": "office", "cuisine": None}, # ~890m
    ]
    result = count_pois_by_type(*center, pois, radii_m=[250, 500, 1000])
    assert result["bars_250m"] == 0
    assert result["bars_500m"] == 1
    assert result["bars_1000m"] == 2
    assert result["offices_1000m"] == 1
    assert result["offices_500m"] == 0


def test_count_pois_by_type_same_cuisine():
    from src.features import count_pois_by_type
    pois = [
        {"lat": 40.001, "lon": -74.0, "type": "restaurant", "cuisine": "pizza"},
        {"lat": 40.002, "lon": -74.0, "type": "restaurant", "cuisine": "sushi"},
    ]
    result = count_pois_by_type(40.0, -74.0, pois, radii_m=[500], target_cuisine="pizza")
    assert result["restaurants_500m"] == 2
    assert result["restaurants_same_cuisine_500m"] == 1


def test_count_pois_by_type_empty_returns_zeros():
    from src.features import count_pois_by_type
    result = count_pois_by_type(40.0, -74.0, [], radii_m=[250, 500, 1000])
    assert result["bars_250m"] == 0
    assert result["restaurants_500m"] == 0
    assert result["transit_stops_1000m"] == 0


def test_fetch_census_demographics_returns_expected_fields(mocker):
    from src.features import fetch_census_demographics
    from tests.conftest import FAKE_CENSUS_ACS_RESPONSE

    FAKE_GEOCODER_RESPONSE = {
        "result": {
            "geographies": {
                "Census Tracts": [{"TRACT": "001000", "COUNTY": "003", "STATE": "32"}]
            }
        }
    }

    geocoder_resp = MagicMock()
    geocoder_resp.json.return_value = FAKE_GEOCODER_RESPONSE
    geocoder_resp.raise_for_status.return_value = None

    acs_resp = MagicMock()
    acs_resp.json.return_value = FAKE_CENSUS_ACS_RESPONSE
    acs_resp.raise_for_status.return_value = None

    mocker.patch("src.features.requests.get", side_effect=[geocoder_resp, acs_resp])
    result = fetch_census_demographics(36.17, -115.14)
    assert result["median_income"] == 75000.0
    assert result["total_population"] == 50000.0
    assert result["median_age"] == 34.0


def test_fetch_census_demographics_handles_missing_tract(mocker):
    from src.features import fetch_census_demographics

    empty_resp = MagicMock()
    empty_resp.json.return_value = {"result": {"geographies": {"Census Tracts": []}}}
    empty_resp.raise_for_status.return_value = None
    mocker.patch("src.features.requests.get", return_value=empty_resp)

    result = fetch_census_demographics(0.0, 0.0)
    assert result["median_income"] is None
    assert result["total_population"] is None
    assert result["median_age"] is None


def test_generate_neighborhood_features_shape(mocker):
    from src.features import generate_neighborhood_features
    from tests.conftest import FAKE_OSM_RESPONSE, FAKE_CENSUS_ACS_RESPONSE
    from unittest.mock import MagicMock
    mock_osm = MagicMock()
    mock_osm.json.return_value = FAKE_OSM_RESPONSE
    mock_osm.raise_for_status.return_value = None
    mocker.patch("src.features.requests.post", return_value=mock_osm)
    mocker.patch(
        "censusgeocode.CensusGeocode.coordinates",
        return_value=[{"geographies": {"Census Tracts": [
            {"TRACT": "001000", "COUNTY": "003", "STATE": "32"}
        ]}}],
    )
    mock_census = MagicMock()
    mock_census.json.return_value = FAKE_CENSUS_ACS_RESPONSE
    mock_census.raise_for_status.return_value = None
    mocker.patch("src.features.requests.get", return_value=mock_census)

    result = generate_neighborhood_features(36.17, -115.14)

    # Spot-check keys
    assert "restaurants_500m" in result
    assert "bars_1000m" in result
    assert "offices_250m" in result
    assert "transit_stops_500m" in result
    assert "schools_1000m" in result
    assert "median_income" in result
    assert "total_population" in result
    assert "median_age" in result
    # cuisine and price_level are NOT in the result (caller appends them)
    assert "cuisine_encoded" not in result
    assert "price_level" not in result
