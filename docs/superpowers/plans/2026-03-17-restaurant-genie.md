# RestaurantGenie Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CLI tool predicting restaurant success probability for any US address using Yelp Open Dataset + OSM + US Census, with LightGBM model, Optuna hyperparameter search, and SHAP explanations.

**Architecture:** Three sequential scripts share `features.py`. `build_dataset.py` runs once to produce `data/processed/restaurant_features.parquet`; `train_model.py` runs once to produce model artifacts in `models/`; `predict.py` answers queries in seconds using those artifacts.

**Tech Stack:** Python 3.10+, LightGBM, Optuna, SHAP, scikit-learn, pandas/numpy, geopy (Nominatim), requests, censusgeocode, pyarrow, tqdm, pytest, pytest-mock

---

## File Map

| File | Responsibility |
|------|----------------|
| `src/features.py` | `fetch_pois_for_bbox()`, `count_pois_by_type()`, `fetch_census_demographics()`, `generate_neighborhood_features()` |
| `src/build_dataset.py` | Parse Yelp JSONL, batch OSM enrichment by city, Census enrichment, success labels, save parquet + `normalization_params.json` |
| `src/train_model.py` | Data splits, feature selection (SHAP + permutation), Optuna search, refit, calibration, save all artifacts |
| `src/predict.py` | Geocode → features → model inference → percentile rank → SHAP pros/cons → comparable restaurants → CLI output |
| `tests/conftest.py` | Shared fixtures: fake OSM/Census responses, Yelp JSONL fixture, synthetic DataFrame |
| `tests/test_features.py` | Unit tests for all `features.py` functions with mocked HTTP |
| `tests/test_build_dataset.py` | Unit tests for Yelp loading, OSM batch enrichment, Census enrichment, label computation |
| `tests/test_train_model.py` | Unit tests for splits, feature selection, Optuna smoke test, refit + calibration, artifact saving |
| `tests/test_predict.py` | Unit tests for geocoding error handling, feature assembly, percentile ranking, output formatting |
| `requirements.txt` | All dependencies |
| `README.md` | Setup and usage |

---

### Task 1: Project scaffold

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`, `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `data/raw/.gitkeep`, `data/processed/.gitkeep`, `models/.gitkeep`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src tests data/raw data/processed models
touch src/__init__.py tests/__init__.py
```

- [ ] **Step 2: Write `requirements.txt`**

```
lightgbm>=4.0.0
optuna>=3.6.0
shap>=0.44.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
requests>=2.31.0
censusgeocode>=0.5.4
pyarrow>=14.0.0
geopy>=2.4.0
tqdm>=4.66.0
pytest>=7.4.0
pytest-mock>=3.12.0
```

- [ ] **Step 3: Write `tests/conftest.py`**

```python
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
```

- [ ] **Step 4: Install dependencies and verify pytest works**

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Expected: 0 tests collected, no errors.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/ tests/ data/ models/
git commit -m "feat: project scaffold with requirements and shared test fixtures"
```

---

### Task 2: `features.py` — OSM POI fetching

**Files:**
- Create: `src/features.py`
- Create: `tests/test_features.py`

The Overpass API call fetches all relevant POI types in one bounding-box query. Tag-to-type mapping:
- `amenity ∈ {bar, pub, nightclub}` → `bar`
- `amenity = restaurant` → `restaurant`
- key `office` exists (any value) → `office`
- `tourism = hotel` → `hotel`
- `public_transport = stop_position` OR `highway = bus_stop` OR `railway ∈ {station, halt, tram_stop}` → `transit`
- `amenity ∈ {school, university, college}` → `school`

- [ ] **Step 1: Write failing test**

```python
# tests/test_features.py
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
```

- [ ] **Step 2: Run test — confirm FAIL**

```bash
pytest tests/test_features.py::test_fetch_pois_for_bbox_returns_typed_list -v
```

- [ ] **Step 3: Write `src/features.py`**

```python
import time
import requests
import numpy as np
from sklearn.neighbors import BallTree

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
EARTH_RADIUS_M = 6_371_000
RADII_M = [250, 500, 1000]

_TAG_RULES = [
    (lambda t: t.get("amenity") in {"bar", "pub", "nightclub"}, "bar"),
    (lambda t: t.get("amenity") == "restaurant", "restaurant"),
    (lambda t: any(k == "office" for k in t), "office"),
    (lambda t: t.get("tourism") == "hotel", "hotel"),
    (lambda t: t.get("public_transport") == "stop_position"
               or t.get("highway") == "bus_stop"
               or t.get("railway") in {"station", "halt", "tram_stop"}, "transit"),
    (lambda t: t.get("amenity") in {"school", "university", "college"}, "school"),
]


def _classify_tags(tags: dict) -> str | None:
    for predicate, poi_type in _TAG_RULES:
        if predicate(tags):
            return poi_type
    return None


def fetch_pois_for_bbox(
    south: float, west: float, north: float, east: float
) -> list[dict]:
    """Query OSM Overpass for all relevant POIs in the bounding box.
    Returns list of {"lat", "lon", "type", "cuisine"}.
    """
    bbox = f"{south},{west},{north},{east}"
    query = f"""[out:json][timeout:90];
(
  node["amenity"~"bar|pub|nightclub|restaurant|school|university|college"]({bbox});
  node["office"]({bbox});
  node["tourism"="hotel"]({bbox});
  node["public_transport"="stop_position"]({bbox});
  node["highway"="bus_stop"]({bbox});
  node["railway"~"station|halt|tram_stop"]({bbox});
);
out body;
"""
    response = requests.post(OVERPASS_URL, data=query, timeout=120)
    response.raise_for_status()
    pois = []
    for el in response.json().get("elements", []):
        tags = el.get("tags", {})
        poi_type = _classify_tags(tags)
        if poi_type is None:
            continue
        lat, lon = el.get("lat"), el.get("lon")
        if lat is None or lon is None:
            continue
        pois.append({"lat": lat, "lon": lon, "type": poi_type,
                     "cuisine": tags.get("cuisine")})
    return pois
```

- [ ] **Step 4: Run test — confirm PASS**

```bash
pytest tests/test_features.py::test_fetch_pois_for_bbox_returns_typed_list -v
```

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: OSM POI fetching with bounding-box query and tag classification"
```

---

### Task 3: `features.py` — POI radius counting

**Files:**
- Modify: `src/features.py`
- Modify: `tests/test_features.py`

- [ ] **Step 1: Write failing tests**

```python
def test_count_pois_by_type_respects_radius():
    from src.features import count_pois_by_type
    # 1 degree lat ≈ 111 km → 300m ≈ 0.0027 deg, 600m ≈ 0.0054 deg
    center = (40.0, -74.0)
    pois = [
        {"lat": 40.0027, "lon": -74.0, "type": "bar", "cuisine": None},   # ~300m
        {"lat": 40.0054, "lon": -74.0, "type": "bar", "cuisine": None},   # ~600m
        {"lat": 40.011,  "lon": -74.0, "type": "office", "cuisine": None}, # ~1200m
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
```

- [ ] **Step 2: Run tests — confirm FAIL**

```bash
pytest tests/test_features.py -k "count_pois" -v
```

- [ ] **Step 3: Add `count_pois_by_type` to `src/features.py`**

```python
_TYPE_KEY = {
    "restaurant": "restaurants",
    "bar": "bars",
    "office": "offices",
    "hotel": "hotels",
    "transit": "transit_stops",
    "school": "schools",
}


def count_pois_by_type(
    lat: float,
    lon: float,
    pois: list[dict],
    radii_m: list[int] = RADII_M,
    target_cuisine: str | None = None,
) -> dict:
    """Count POIs of each type within each radius. Returns flat dict.

    Keys: bars_250m, bars_500m, bars_1000m, restaurants_250m, ...,
          restaurants_same_cuisine_250m, ... (None when target_cuisine is None)
    """
    result: dict = {}
    center_rad = np.radians([[lat, lon]])

    for poi_type, key_prefix in _TYPE_KEY.items():
        subset = [p for p in pois if p["type"] == poi_type]
        if not subset:
            for r in radii_m:
                result[f"{key_prefix}_{r}m"] = 0
            if poi_type == "restaurant":
                for r in radii_m:
                    result[f"restaurants_same_cuisine_{r}m"] = None
            continue

        coords_rad = np.radians([[p["lat"], p["lon"]] for p in subset])
        tree = BallTree(coords_rad, metric="haversine")
        for r in radii_m:
            count = tree.query_radius(center_rad, r=r / EARTH_RADIUS_M, count_only=True)[0]
            result[f"{key_prefix}_{r}m"] = int(count)

        if poi_type == "restaurant":
            if target_cuisine:
                same = [p for p in subset if p.get("cuisine") == target_cuisine]
                if not same:
                    for r in radii_m:
                        result[f"restaurants_same_cuisine_{r}m"] = 0
                else:
                    same_rad = np.radians([[p["lat"], p["lon"]] for p in same])
                    same_tree = BallTree(same_rad, metric="haversine")
                    for r in radii_m:
                        count = same_tree.query_radius(
                            center_rad, r=r / EARTH_RADIUS_M, count_only=True
                        )[0]
                        result[f"restaurants_same_cuisine_{r}m"] = int(count)
            else:
                for r in radii_m:
                    result[f"restaurants_same_cuisine_{r}m"] = None

    return result
```

- [ ] **Step 4: Run tests — confirm PASS**

```bash
pytest tests/test_features.py -k "count_pois" -v
```

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: POI radius counting with BallTree haversine distances"
```

---

### Task 4: `features.py` — Census demographics fetching

**Files:**
- Modify: `src/features.py`
- Modify: `tests/test_features.py`

Two-step process:
1. `censusgeocode.CensusGeocode().coordinates(x=lon, y=lat)` → FIPS (state, county, tract)
2. GET `https://api.census.gov/data/2022/acs/acs5?get=B19013_001E,B01003_001E,B01002_001E&for=tract:{tract}&in=state:{state}+county:{county}&key={key}`

Variables: `B19013_001E` = median income, `B01003_001E` = total population, `B01002_001E` = median age.

Note: `pct_age_25_44` from the spec is approximated here by `median_age` (a single variable rather than summing 12 age-bucket variables). The feature name in the dataset is `median_age`.

The `CENSUS_API_KEY` environment variable is read at import time; if absent, no key is sent (works up to ~500 req/day).

- [ ] **Step 1: Write failing test**

```python
def test_fetch_census_demographics_returns_expected_fields(mocker):
    from src.features import fetch_census_demographics
    mocker.patch(
        "censusgeocode.CensusGeocode.coordinates",
        return_value=[{
            "geographies": {
                "Census Tracts": [{"TRACT": "001000", "COUNTY": "003", "STATE": "32"}]
            }
        }],
    )
    mock_resp = MagicMock()
    mock_resp.json.return_value = FAKE_CENSUS_ACS_RESPONSE
    mock_resp.raise_for_status.return_value = None
    mocker.patch("src.features.requests.get", return_value=mock_resp)
    result = fetch_census_demographics(36.17, -115.14)
    assert result["median_income"] == 75000.0
    assert result["total_population"] == 50000.0
    assert result["median_age"] == 34.0


def test_fetch_census_demographics_handles_missing_tract(mocker):
    from src.features import fetch_census_demographics
    mocker.patch(
        "censusgeocode.CensusGeocode.coordinates",
        return_value=[{"geographies": {"Census Tracts": []}}],
    )
    result = fetch_census_demographics(0.0, 0.0)
    assert result["median_income"] is None
    assert result["total_population"] is None
    assert result["median_age"] is None
```

- [ ] **Step 2: Run tests — confirm FAIL**

```bash
pytest tests/test_features.py -k "census" -v
```

- [ ] **Step 3: Add `fetch_census_demographics` to `src/features.py`**

```python
import os
import censusgeocode as cg

CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", "")
CENSUS_ACS_URL = "https://api.census.gov/data/2022/acs/acs5"
_CENSUS_VARS = "B19013_001E,B01003_001E,B01002_001E"


def fetch_census_demographics(lat: float, lon: float) -> dict:
    """Return median_income, total_population, median_age for the census tract
    containing (lat, lon). Returns None values if lookup fails.
    """
    null_result = {"median_income": None, "total_population": None, "median_age": None}
    try:
        geocode_result = cg.CensusGeocode().coordinates(x=lon, y=lat)
        tracts = geocode_result[0]["geographies"].get("Census Tracts", [])
        if not tracts:
            return null_result
        tract_info = tracts[0]
        state = tract_info["STATE"]
        county = tract_info["COUNTY"]
        tract = tract_info["TRACT"]
    except Exception:
        return null_result

    params = {
        "get": _CENSUS_VARS,
        "for": f"tract:{tract}",
        "in": f"state:{state}+county:{county}",
    }
    if CENSUS_API_KEY:
        params["key"] = CENSUS_API_KEY

    try:
        resp = requests.get(CENSUS_ACS_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        header, values = data[0], data[1]
        row = dict(zip(header, values))
        return {
            "median_income": float(row["B19013_001E"]) if row["B19013_001E"] else None,
            "total_population": float(row["B01003_001E"]) if row["B01003_001E"] else None,
            "median_age": float(row["B01002_001E"]) if row["B01002_001E"] else None,
        }
    except Exception:
        return null_result
```

- [ ] **Step 4: Run tests — confirm PASS**

```bash
pytest tests/test_features.py -k "census" -v
```

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: Census ACS demographics via censusgeocode + direct ACS API call"
```

---

### Task 5: `features.py` — `generate_neighborhood_features`

**Files:**
- Modify: `src/features.py`
- Modify: `tests/test_features.py`

This is the function called by `predict.py` for live single-location queries. It issues a live Overpass query (1km bounding box around the point) + Census lookup and returns the full feature dict. `cuisine` and `price_level` are appended by the caller.

- [ ] **Step 1: Write failing test**

```python
def test_generate_neighborhood_features_shape(mocker):
    from src.features import generate_neighborhood_features
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
```

- [ ] **Step 2: Run test — confirm FAIL**

```bash
pytest tests/test_features.py::test_generate_neighborhood_features_shape -v
```

- [ ] **Step 3: Add `generate_neighborhood_features` to `src/features.py`**

```python
_PADDING_DEG = 0.012   # ~1.3km padding each side


def generate_neighborhood_features(
    lat: float,
    lon: float,
    target_cuisine: str | None = None,
) -> dict:
    """Live OSM + Census feature generation for a single location.
    Caller must append cuisine_encoded and price_level before model input.
    """
    pois = fetch_pois_for_bbox(
        lat - _PADDING_DEG, lon - _PADDING_DEG,
        lat + _PADDING_DEG, lon + _PADDING_DEG,
    )
    poi_counts = count_pois_by_type(lat, lon, pois, target_cuisine=target_cuisine)
    demographics = fetch_census_demographics(lat, lon)
    return {**poi_counts, **demographics}
```

- [ ] **Step 4: Run test — confirm PASS**

```bash
pytest tests/test_features.py::test_generate_neighborhood_features_shape -v
```

- [ ] **Step 5: Run all feature tests**

```bash
pytest tests/test_features.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: generate_neighborhood_features combining OSM + Census for single location"
```

---

### Task 6: `build_dataset.py` — Yelp JSON loader and cuisine mapper

**Files:**
- Create: `src/build_dataset.py`
- Create: `tests/test_build_dataset.py`

The Yelp file `yelp_academic_dataset_business.json` is JSONL (one JSON object per line). Filter to businesses where `categories` contains "Restaurants". Cuisine is extracted from `categories` using a keyword map.

Cuisine keyword map (partial — add more as needed):

```python
_CUISINE_MAP = {
    "pizza": "pizza", "italian": "italian", "mexican": "mexican",
    "chinese": "chinese", "japanese": "japanese", "sushi": "japanese",
    "american (traditional)": "american", "american (new)": "american",
    "burgers": "burgers", "sandwiches": "sandwiches", "thai": "thai",
    "indian": "indian", "mediterranean": "mediterranean", "greek": "greek",
    "french": "french", "korean": "korean", "vietnamese": "vietnamese",
    "seafood": "seafood", "steakhouses": "steakhouses",
}
```

- [ ] **Step 1: Write failing tests**

```python
# tests/test_build_dataset.py
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
```

- [ ] **Step 2: Run tests — confirm FAIL**

```bash
pytest tests/test_build_dataset.py -v
```

- [ ] **Step 3: Implement `load_yelp_businesses` in `src/build_dataset.py`**

```python
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

_CUISINE_MAP = {
    "pizza": "pizza", "italian": "italian", "mexican": "mexican",
    "chinese": "chinese", "japanese": "japanese", "sushi": "japanese",
    "american (traditional)": "american", "american (new)": "american",
    "burgers": "burgers", "sandwiches": "sandwiches", "thai": "thai",
    "indian": "indian", "mediterranean": "mediterranean", "greek": "greek",
    "french": "french", "korean": "korean", "vietnamese": "vietnamese",
    "seafood": "seafood", "steakhouses": "steakhouses",
}


def _extract_cuisine(categories_str: str | None) -> str | None:
    if not categories_str:
        return None
    categories_lower = categories_str.lower()
    for keyword, cuisine in _CUISINE_MAP.items():
        if keyword in categories_lower:
            return cuisine
    return None


def _extract_price_level(attributes: dict | None) -> float | None:
    if not attributes:
        return None
    val = attributes.get("RestaurantsPriceRange2")
    if val is None:
        return None
    try:
        return float(str(val).strip("'"))
    except (ValueError, TypeError):
        return None


def load_yelp_businesses(jsonl_path: str | Path) -> pd.DataFrame:
    """Load and filter Yelp JSONL to US restaurant businesses only."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                biz = json.loads(line)
            except json.JSONDecodeError:
                continue
            cats = biz.get("categories") or ""
            if "Restaurants" not in cats:
                continue
            lat = biz.get("latitude")
            lon = biz.get("longitude")
            if lat is None or lon is None:
                continue
            records.append({
                "business_id": biz.get("business_id"),
                "name": biz.get("name"),
                "city": biz.get("city"),
                "state": biz.get("state"),
                "lat": float(lat),
                "lon": float(lon),
                "cuisine": _extract_cuisine(cats),
                "price_level": _extract_price_level(biz.get("attributes")),
                "rating": biz.get("stars"),
                "review_count": biz.get("review_count"),
                "is_open": biz.get("is_open"),
            })
    return pd.DataFrame(records)
```

- [ ] **Step 4: Run tests — confirm PASS**

```bash
pytest tests/test_build_dataset.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/build_dataset.py tests/test_build_dataset.py
git commit -m "feat: Yelp JSONL loader with cuisine extraction and price level parsing"
```

---

### Task 7: `build_dataset.py` — Batch OSM enrichment by city

**Files:**
- Modify: `src/build_dataset.py`
- Modify: `tests/test_build_dataset.py`

For each unique city in the dataset, compute the bounding box of all restaurants in that city (padded by 1.5km), issue ONE Overpass query, then use `count_pois_by_type` for every restaurant in that city. Sleep 2 seconds between city queries.

- [ ] **Step 1: Write failing test**

```python
def test_enrich_with_osm_features_adds_poi_columns(small_restaurant_df, mocker):
    from src.build_dataset import enrich_with_osm_features
    mock_resp = MagicMock()
    mock_resp.json.return_value = FAKE_OSM_RESPONSE
    mock_resp.raise_for_status.return_value = None
    mocker.patch("src.features.requests.post", return_value=mock_resp)
    mocker.patch("time.sleep")  # don't actually sleep in tests

    result = enrich_with_osm_features(small_restaurant_df)

    assert "restaurants_500m" in result.columns
    assert "bars_250m" in result.columns
    assert "offices_1000m" in result.columns
    assert "transit_stops_500m" in result.columns
    assert "schools_250m" in result.columns
    assert len(result) == len(small_restaurant_df)
```

- [ ] **Step 2: Run test — confirm FAIL**

```bash
pytest tests/test_build_dataset.py::test_enrich_with_osm_features_adds_poi_columns -v
```

- [ ] **Step 3: Implement `enrich_with_osm_features` in `src/build_dataset.py`**

```python
from src.features import fetch_pois_for_bbox, count_pois_by_type

_OSM_PADDING_DEG = 0.014  # ~1.5km padding around city bounding box


def enrich_with_osm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Batch OSM POI enrichment: one Overpass query per city."""
    rows_with_features = []
    for city, group in tqdm(df.groupby("city"), desc="OSM enrichment by city"):
        south = group["lat"].min() - _OSM_PADDING_DEG
        north = group["lat"].max() + _OSM_PADDING_DEG
        west = group["lon"].min() - _OSM_PADDING_DEG
        east = group["lon"].max() + _OSM_PADDING_DEG
        try:
            pois = fetch_pois_for_bbox(south, west, north, east)
        except Exception as e:
            print(f"OSM query failed for {city}: {e}. Filling zeros.")
            pois = []

        for _, row in group.iterrows():
            counts = count_pois_by_type(
                row["lat"], row["lon"], pois, target_cuisine=row.get("cuisine")
            )
            rows_with_features.append({**row.to_dict(), **counts})

        time.sleep(2)  # rate-limit OSM Overpass

    return pd.DataFrame(rows_with_features)
```

- [ ] **Step 4: Run test — confirm PASS**

```bash
pytest tests/test_build_dataset.py::test_enrich_with_osm_features_adds_poi_columns -v
```

- [ ] **Step 5: Commit**

```bash
git add src/build_dataset.py tests/test_build_dataset.py
git commit -m "feat: batch OSM POI enrichment grouped by city with rate limiting"
```

---

### Task 8: `build_dataset.py` — Census enrichment with FIPS caching

**Files:**
- Modify: `src/build_dataset.py`
- Modify: `tests/test_build_dataset.py`

Cache by census tract FIPS (state+county+tract) to avoid redundant API calls. Many restaurants in the same neighborhood share a tract.

- [ ] **Step 1: Write failing test**

```python
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
    mocker.patch("src.features.requests.get", return_value=mock_resp)

    result = enrich_with_census(small_restaurant_df)

    assert "median_income" in result.columns
    assert "total_population" in result.columns
    assert "median_age" in result.columns
    assert len(result) == len(small_restaurant_df)


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
```

- [ ] **Step 2: Run tests — confirm FAIL**

```bash
pytest tests/test_build_dataset.py -k "census" -v
```

- [ ] **Step 3: Implement `enrich_with_census` in `src/build_dataset.py`**

```python
import censusgeocode as cg
from src.features import fetch_census_demographics, CENSUS_ACS_URL, _CENSUS_VARS, CENSUS_API_KEY

# Re-implement with FIPS caching
def enrich_with_census(df: pd.DataFrame) -> pd.DataFrame:
    """Add median_income, total_population, median_age via Census ACS.
    Caches ACS lookup by FIPS (state+county+tract) to minimise API calls.
    Uses module-level `requests` so tests can patch `src.build_dataset.requests.get`.
    """
    geocoder = cg.CensusGeocode()
    fips_cache: dict[tuple, dict] = {}  # (state, county, tract) → demographics

    def _lookup_fips(state, county, tract):
        key = (state, county, tract)
        if key in fips_cache:
            return fips_cache[key]
        params = {
            "get": _CENSUS_VARS,
            "for": f"tract:{tract}",
            "in": f"state:{state}+county:{county}",
        }
        if CENSUS_API_KEY:
            params["key"] = CENSUS_API_KEY
        try:
            resp = requests.get(CENSUS_ACS_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            header, values = data[0], data[1]
            row = dict(zip(header, values))
            result = {
                "median_income": float(row["B19013_001E"]) if row["B19013_001E"] else None,
                "total_population": float(row["B01003_001E"]) if row["B01003_001E"] else None,
                "median_age": float(row["B01002_001E"]) if row["B01002_001E"] else None,
            }
        except Exception:
            result = {"median_income": None, "total_population": None, "median_age": None}
        fips_cache[key] = result
        return result

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Census enrichment"):
        demo = {"median_income": None, "total_population": None, "median_age": None}
        try:
            geocode_result = geocoder.coordinates(x=row["lon"], y=row["lat"])
            tracts = geocode_result[0]["geographies"].get("Census Tracts", [])
            if tracts:
                t = tracts[0]
                demo = _lookup_fips(t["STATE"], t["COUNTY"], t["TRACT"])
        except Exception:
            pass
        rows.append({**row.to_dict(), **demo})

    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests — confirm PASS**

```bash
pytest tests/test_build_dataset.py -k "census" -v
```

- [ ] **Step 5: Commit**

```bash
git add src/build_dataset.py tests/test_build_dataset.py
git commit -m "feat: Census enrichment with FIPS-based caching to minimise API calls"
```

---

### Task 9: `build_dataset.py` — Success labels and normalization params

**Files:**
- Modify: `src/build_dataset.py`
- Modify: `tests/test_build_dataset.py`

Formula from spec:
```
normalized_rating = (rating - 1.0) / 4.0
normalized_log_reviews = log1p(review_count) / p95_log_reviews
success_score = 0.4 * normalized_rating + 0.4 * normalized_log_reviews + 0.2 * is_open
is_successful = success_score > 70th percentile within (cuisine, city)
```
Groups with fewer than 10 restaurants → merge into `other` cuisine label before computing percentile.

- [ ] **Step 1: Write failing tests**

```python
def test_compute_success_labels_creates_column(small_restaurant_df):
    from src.build_dataset import compute_success_labels
    p95 = float(np.log1p(small_restaurant_df["review_count"]).quantile(0.95))
    result = compute_success_labels(small_restaurant_df, p95_log_reviews=p95)
    assert "success_score" in result.columns
    assert "is_successful" in result.columns
    assert result["is_successful"].isin([0, 1]).all()


def test_compute_success_labels_small_groups_merged():
    """Groups <10 restaurants → cuisine='other' for percentile calculation."""
    from src.build_dataset import compute_success_labels
    df = pd.DataFrame({
        "city": ["NYC"] * 5 + ["NYC"] * 15,
        "cuisine": ["rare_cuisine"] * 5 + ["italian"] * 15,
        "rating": [4.0] * 20,
        "review_count": [100] * 20,
        "is_open": [1] * 20,
    })
    p95 = float(np.log1p(df["review_count"]).quantile(0.95))
    result = compute_success_labels(df, p95_log_reviews=p95)
    # Rare cuisine group (5 rows) merged into 'other' — no crash
    assert "is_successful" in result.columns
    assert len(result) == 20
```

- [ ] **Step 2: Run tests — confirm FAIL**

```bash
pytest tests/test_build_dataset.py -k "success" -v
```

- [ ] **Step 3: Implement `compute_success_labels`**

```python
def compute_success_labels(df: pd.DataFrame, p95_log_reviews: float) -> pd.DataFrame:
    """Add success_score and is_successful columns."""
    df = df.copy()
    df["_norm_rating"] = (df["rating"].fillna(3.0) - 1.0) / 4.0
    df["_norm_log_reviews"] = (
        np.log1p(df["review_count"].fillna(0)) / max(p95_log_reviews, 1e-6)
    )
    df["success_score"] = (
        0.4 * df["_norm_rating"]
        + 0.4 * df["_norm_log_reviews"].clip(upper=1.0)
        + 0.2 * df["is_open"].fillna(0)
    )

    # Merge small cuisine groups into 'other'
    cuisine_counts = df.groupby(["city", "cuisine"])["business_id"].transform("count")
    df["_cuisine_group"] = df["cuisine"].where(cuisine_counts >= 10, other="other")

    # Compute 70th percentile threshold per (cuisine_group, city)
    def _threshold(group):
        return group["success_score"].quantile(0.70)

    thresholds = df.groupby(["city", "_cuisine_group"]).apply(_threshold).rename("_threshold")
    df = df.join(thresholds, on=["city", "_cuisine_group"])
    df["is_successful"] = (df["success_score"] > df["_threshold"]).astype(int)
    df = df.drop(columns=["_norm_rating", "_norm_log_reviews", "_cuisine_group", "_threshold"])
    return df
```

- [ ] **Step 4: Run tests — confirm PASS**

```bash
pytest tests/test_build_dataset.py -k "success" -v
```

- [ ] **Step 5: Commit**

```bash
git add src/build_dataset.py tests/test_build_dataset.py
git commit -m "feat: success score computation with within-group percentile labels"
```

---

### Task 10: `build_dataset.py` — Orchestration, normalization params, save parquet

**Files:**
- Modify: `src/build_dataset.py`
- Modify: `tests/test_build_dataset.py`

Main function ties everything together. Also computes cuisine label map and saves `normalization_params.json`. Output: `data/processed/restaurant_features.parquet`.

- [ ] **Step 1: Write failing test**

```python
def test_build_dataset_produces_parquet(yelp_jsonl_file, tmp_path, mocker):
    from src.build_dataset import build_dataset
    mock_osm = MagicMock(); mock_osm.json.return_value = FAKE_OSM_RESPONSE
    mock_osm.raise_for_status.return_value = None
    mocker.patch("src.features.requests.post", return_value=mock_osm)
    mocker.patch("time.sleep")
    mocker.patch(
        "censusgeocode.CensusGeocode.coordinates",
        return_value=[{"geographies": {"Census Tracts": [
            {"TRACT": "001000", "COUNTY": "003", "STATE": "32"}
        ]}}],
    )
    mock_census = MagicMock(); mock_census.json.return_value = FAKE_CENSUS_ACS_RESPONSE
    mock_census.raise_for_status.return_value = None
    mocker.patch("src.features.requests.get", return_value=mock_census)

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
```

- [ ] **Step 2: Run test — confirm FAIL**

```bash
pytest tests/test_build_dataset.py::test_build_dataset_produces_parquet -v
```

- [ ] **Step 3: Implement `build_dataset` main function**

```python
import requests  # needed so tests can patch src.build_dataset.requests.get
import json as _json


def build_dataset(
    yelp_jsonl_path: str | Path,
    output_parquet: str | Path,
    output_params: str | Path,
) -> None:
    """Full Stage 1 pipeline. Saves parquet + normalization_params.json."""
    print("Loading Yelp data...")
    df = load_yelp_businesses(yelp_jsonl_path)
    print(f"  {len(df)} restaurants loaded.")

    print("Computing normalization params...")
    p95_log_reviews = float(np.log1p(df["review_count"].fillna(0)).quantile(0.95))

    # Cuisine label map: sorted unique cuisines → int; None/'other' → 0
    cuisines = sorted({c for c in df["cuisine"].dropna().unique()})
    cuisine_label_map = {"other": 0, None: 0}
    for i, c in enumerate(cuisines, start=1):
        cuisine_label_map[c] = i

    params = {"p95_log_reviews": p95_log_reviews, "cuisine_label_map": cuisine_label_map}
    Path(output_params).parent.mkdir(parents=True, exist_ok=True)
    with open(output_params, "w") as f:
        _json.dump(params, f)
    print(f"  Saved normalization params to {output_params}")

    print("Enriching with OSM features (batched by city)...")
    df = enrich_with_osm_features(df)

    print("Enriching with Census demographics...")
    df = enrich_with_census(df)

    print("Computing success labels...")
    df = compute_success_labels(df, p95_log_reviews=p95_log_reviews)

    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f"  Saved {len(df)} rows to {output_parquet}")


if __name__ == "__main__":
    build_dataset(
        yelp_jsonl_path="data/raw/yelp_academic_dataset_business.json",
        output_parquet="data/processed/restaurant_features.parquet",
        output_params="models/normalization_params.json",
    )
```

- [ ] **Step 4: Run test — confirm PASS**

```bash
pytest tests/test_build_dataset.py::test_build_dataset_produces_parquet -v
```

- [ ] **Step 5: Run all dataset tests**

```bash
pytest tests/test_build_dataset.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/build_dataset.py tests/test_build_dataset.py
git commit -m "feat: build_dataset orchestration saving parquet and normalization params"
```

---

### Task 11: `train_model.py` — Data splits

**Files:**
- Create: `src/train_model.py`
- Create: `tests/test_train_model.py`

Splits (all upfront before any fitting):
1. **test_cities** — 20% of unique cities, held out entirely
2. Within remaining cities: **calibration_set** — random 20% of rows, **train_search_set** — remaining 80%

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests — confirm FAIL**

```bash
pytest tests/test_train_model.py -v
```

- [ ] **Step 3: Implement `make_splits` in `src/train_model.py`**

```python
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

optuna.logging.set_verbosity(optuna.logging.WARNING)

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
```

- [ ] **Step 4: Run tests — confirm PASS**

```bash
pytest tests/test_train_model.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/train_model.py tests/test_train_model.py
git commit -m "feat: geographic data splits with held-out test cities and calibration set"
```

---

### Task 12: `train_model.py` — cuisine encoding

**Files:**
- Modify: `src/train_model.py`
- Modify: `tests/test_train_model.py`

Encode cuisine using label map from `normalization_params.json`. Unseen cuisines → 0 (`other`).

- [ ] **Step 1: Write failing test**

```python
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
```

- [ ] **Step 2: Run test — confirm FAIL**

```bash
pytest tests/test_train_model.py::test_encode_cuisine_known_and_unknown -v
```

- [ ] **Step 3: Implement `encode_cuisine_column`**

```python
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
```

- [ ] **Step 4: Run test — confirm PASS**

```bash
pytest tests/test_train_model.py::test_encode_cuisine_known_and_unknown -v
```

- [ ] **Step 5: Commit**

```bash
git add src/train_model.py tests/test_train_model.py
git commit -m "feat: cuisine label encoding with OOV→0 fallback"
```

---

### Task 13: `train_model.py` — Feature selection (SHAP + permutation importance)

**Files:**
- Modify: `src/train_model.py`
- Modify: `tests/test_train_model.py`

5-fold internal CV on `train_search_set`. Per fold: compute SHAP + permutation importance on held-out fold. Average across folds. Drop features below 1% threshold (SHAP) or negative permutation importance.

- [ ] **Step 1: Write failing test**

```python
def test_select_features_drops_noise_feature(synthetic_model_df):
    from src.train_model import select_features
    import numpy as np
    rng = np.random.default_rng(0)
    # Add pure noise column
    df = synthetic_model_df.copy()
    df["pure_noise"] = rng.random(len(df))
    features = FEATURE_COLS + ["pure_noise"]
    X = df[features].values
    y = df[TARGET_COL].values
    selected = select_features(X, y, feature_names=features)
    # Noise column should not survive selection
    assert "pure_noise" not in selected
    # Real features should survive
    assert len(selected) >= 5
```

Import `FEATURE_COLS` and `TARGET_COL` from `src.train_model` in the test.

- [ ] **Step 2: Run test — confirm FAIL**

```bash
pytest tests/test_train_model.py::test_select_features_drops_noise_feature -v
```

- [ ] **Step 3: Implement `select_features`**

```python
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
```

- [ ] **Step 4: Run test — confirm PASS**

```bash
pytest tests/test_train_model.py::test_select_features_drops_noise_feature -v
```

Note: this test runs actual LightGBM + SHAP on the synthetic dataset — expect ~10–20 seconds.

- [ ] **Step 5: Commit**

```bash
git add src/train_model.py tests/test_train_model.py
git commit -m "feat: feature selection via 5-fold SHAP and permutation importance"
```

---

### Task 14: `train_model.py` — Optuna hyperparameter search

**Files:**
- Modify: `src/train_model.py`
- Modify: `tests/test_train_model.py`

5-fold geographic CV (split by city). Early stopping per fold. Track `n_estimators` per fold. Return best params and `mean_cv_n_estimators`.

- [ ] **Step 1: Write failing test (smoke test with 2 trials)**

```python
def test_search_hyperparameters_returns_params_and_n_estimators(synthetic_model_df):
    from src.train_model import search_hyperparameters
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
```

- [ ] **Step 2: Run test — confirm FAIL**

```bash
pytest tests/test_train_model.py::test_search_hyperparameters_returns_params_and_n_estimators -v
```

- [ ] **Step 3: Implement `search_hyperparameters`**

```python
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
) -> tuple[dict, float]:
    """Optuna search. Returns (best_params, mean_cv_n_estimators)."""

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
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            model = lgb.LGBMClassifier(n_estimators=500, **params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
            n_estimators_list.append(model.best_iteration_)
            proba = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, proba))

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
```

- [ ] **Step 4: Run test — confirm PASS**

```bash
pytest tests/test_train_model.py::test_search_hyperparameters_returns_params_and_n_estimators -v
```

- [ ] **Step 5: Commit**

```bash
git add src/train_model.py tests/test_train_model.py
git commit -m "feat: Optuna hyperparameter search with geographic CV and early stopping"
```

---

### Task 15: `train_model.py` — Refit, calibration, and save artifacts

**Files:**
- Modify: `src/train_model.py`
- Modify: `tests/test_train_model.py`

Final refit on all of `train_search_set`. `n_estimators = round(mean_cv_n_estimators * 1.25)`. Wrap in `CalibratedClassifierCV(cv='prefit')` fitted on `calibration_set`. Save `model.pkl`, `shap_explainer.pkl`, `normalization_params.json`.

- [ ] **Step 1: Write failing tests**

```python
def test_fit_final_model_produces_calibrated_output(synthetic_model_df):
    from src.train_model import fit_final_model
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
    from src.train_model import fit_final_model, save_artifacts
    import pickle
    df = synthetic_model_df
    X = df[FEATURE_COLS].fillna(0).values
    y = df[TARGET_COL].values
    best_params = {"num_leaves": 15, "learning_rate": 0.1, "max_depth": 3,
                   "min_child_samples": 20, "lambda_l1": 0, "lambda_l2": 0,
                   "feature_fraction": 1.0, "bagging_fraction": 1.0}
    calibrated, base_lgbm = fit_final_model(X[:160], y[:160], X[160:], y[160:],
                                            best_params, n_estimators=20)
    save_artifacts(calibrated, base_lgbm, tmp_path)
    assert (tmp_path / "model.pkl").exists()
    assert (tmp_path / "shap_explainer.pkl").exists()
```

- [ ] **Step 2: Run tests — confirm FAIL**

```bash
pytest tests/test_train_model.py -k "fit_final or save_artifacts" -v
```

- [ ] **Step 3: Implement `fit_final_model` and `save_artifacts`**

```python
def fit_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    best_params: dict,
    n_estimators: int,
    random_state: int = 42,
) -> tuple:
    """Refit base LightGBM then wrap with Platt calibration.
    Returns (calibrated_model, base_lgbm).
    n_estimators = round(mean_cv_n_estimators * 1.25); early stopping disabled.
    """
    params = {**best_params, "n_estimators": n_estimators,
              "verbose": -1, "random_state": random_state,
              "bagging_freq": 1}
    base_lgbm = lgb.LGBMClassifier(**params)
    base_lgbm.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(base_lgbm, cv="prefit", method="sigmoid")
    calibrated.fit(X_cal, y_cal)
    return calibrated, base_lgbm


def save_artifacts(
    calibrated_model,
    base_lgbm,
    models_dir: str | Path,
) -> None:
    """Save model.pkl and shap_explainer.pkl to models_dir."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "model.pkl", "wb") as f:
        pickle.dump(calibrated_model, f)
    explainer = shap.TreeExplainer(base_lgbm)
    with open(models_dir / "shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)
    print(f"Saved model.pkl and shap_explainer.pkl to {models_dir}")
```

- [ ] **Step 4: Run tests — confirm PASS**

```bash
pytest tests/test_train_model.py -k "fit_final or save_artifacts" -v
```

- [ ] **Step 5: Commit**

```bash
git add src/train_model.py tests/test_train_model.py
git commit -m "feat: final model refit with calibration (cv=prefit) and artifact saving"
```

---

### Task 16: `train_model.py` — Evaluation, score all rows, performance report, orchestration

**Files:**
- Modify: `src/train_model.py`
- Modify: `tests/test_train_model.py`

After saving artifacts: score test set → ROC-AUC + Brier. Score all rows in parquet → add `predicted_probability` column. Save `performance_report.txt`. Main `train_model` function ties it all together.

- [ ] **Step 1: Write failing test**

```python
def test_train_model_end_to_end(synthetic_model_df, tmp_path, mocker):
    """Smoke test: full pipeline runs without error and creates all artifacts."""
    from src.train_model import train_model
    import json

    parquet_path = tmp_path / "restaurant_features.parquet"
    params_path = tmp_path / "normalization_params.json"
    models_dir = tmp_path / "models"

    # Write cuisine_encoded and other required columns to synthetic df
    df = synthetic_model_df.copy()
    # Parquet needs a city column already (it does) and cuisine_encoded (it does)
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
    # parquet should now have predicted_probability column
    df_out = pd.read_parquet(parquet_path)
    assert "predicted_probability" in df_out.columns
```

- [ ] **Step 2: Run test — confirm FAIL**

```bash
pytest tests/test_train_model.py::test_train_model_end_to_end -v
```

- [ ] **Step 3: Implement evaluation helpers and `train_model` main function**

```python
def evaluate_on_test(
    calibrated_model, base_lgbm, X_test: np.ndarray, y_test: np.ndarray
) -> dict:
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

    # Ensure all feature columns exist (fill missing with 0/NaN as appropriate)
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
        X_ts_sel, y_ts, X_cal_sel, y_cal, best_params, n_estimators, random_state
    )

    print("Evaluating on test cities...")
    metrics = evaluate_on_test(calibrated_model, base_lgbm, X_test_sel, y_test)

    save_artifacts(calibrated_model, base_lgbm, models_dir)

    # Write selected_features into normalization_params.json
    with open(params_path) as f:
        params = json.load(f)
    params["selected_features"] = selected_features
    params["best_hyperparameters"] = best_params
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

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
```

- [ ] **Step 4: Run test — confirm PASS**

```bash
pytest tests/test_train_model.py::test_train_model_end_to_end -v
```

- [ ] **Step 5: Run all model tests**

```bash
pytest tests/test_train_model.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/train_model.py tests/test_train_model.py
git commit -m "feat: model evaluation, predicted_probability scoring, performance report, orchestration"
```

---

### Task 17: `predict.py` — Geocoding and feature assembly

**Files:**
- Create: `src/predict.py`
- Create: `tests/test_predict.py`

Geocode address with Nominatim. Call `generate_neighborhood_features`. Append `cuisine_encoded` and `price_level`. Load `normalization_params.json` for label map and selected features.

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests — confirm FAIL**

```bash
pytest tests/test_predict.py -v
```

- [ ] **Step 3: Implement geocoding and feature assembly in `src/predict.py`**

```python
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim

from src.features import generate_neighborhood_features

_GEOLOCATOR = Nominatim(user_agent="restaurantgenie/1.0")


def geocode_address(address: str) -> tuple[float, float]:
    """Geocode US address to (lat, lon) using Nominatim. Exits with message on failure."""
    location = _GEOLOCATOR.geocode(address, country_codes="us", timeout=10)
    if location is None:
        print(
            f'Error: Could not geocode address "{address}". '
            "Try a more specific address including city and state."
        )
        sys.exit(1)
    return location.latitude, location.longitude


def assemble_feature_vector(
    lat: float,
    lon: float,
    cuisine: str,
    price_level: int,
    params_path: str | Path,
) -> tuple[np.ndarray, list[str]]:
    """Generate full feature vector ordered by selected_features from params."""
    with open(params_path) as f:
        params = json.load(f)
    label_map: dict = params["cuisine_label_map"]
    selected: list[str] = params["selected_features"]

    neighborhood = generate_neighborhood_features(lat, lon, target_cuisine=cuisine)
    neighborhood["cuisine_encoded"] = label_map.get(cuisine, 0)
    neighborhood["price_level"] = price_level

    vector = np.array(
        [float(neighborhood.get(f, 0) or 0) for f in selected], dtype=float
    )
    return vector, selected
```

- [ ] **Step 4: Run tests — confirm PASS**

```bash
pytest tests/test_predict.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/predict.py tests/test_predict.py
git commit -m "feat: geocoding with Nominatim and feature vector assembly for prediction"
```

---

### Task 18: `predict.py` — Model inference, percentile rank, SHAP explanation

**Files:**
- Modify: `src/predict.py`
- Modify: `tests/test_predict.py`

- [ ] **Step 1: Write failing tests**

```python
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
    import shap
    # Mock explainer output
    feature_names = ["restaurants_500m", "median_income", "offices_500m"]
    shap_values = np.array([-0.3, 0.5, 0.2])
    feature_values = {"restaurants_500m": 20, "median_income": 90000, "offices_500m": 15}
    pros, cons = get_shap_pros_cons(shap_values, feature_names, feature_values)
    assert len(pros) >= 1
    assert len(cons) >= 1
    assert any("median_income" in p["feature"] for p in pros)
    assert any("restaurants_500m" in c["feature"] for c in cons)
```

- [ ] **Step 2: Run tests — confirm FAIL**

```bash
pytest tests/test_predict.py -k "percentile or shap" -v
```

- [ ] **Step 3: Implement inference, percentile rank, and SHAP**

```python
def load_artifacts(models_dir: str | Path):
    """Load model.pkl and shap_explainer.pkl. Returns (calibrated_model, explainer)."""
    models_dir = Path(models_dir)
    with open(models_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(models_dir / "shap_explainer.pkl", "rb") as f:
        explainer = pickle.load(f)
    return model, explainer


def compute_percentile_rank(
    score: float,
    reference_df: pd.DataFrame,
    cuisine: str,
    city: str,
    min_group_size: int = 5,
) -> float:
    """Return percentile rank (0–100) of score vs. comparable restaurants.
    Falls back to all rows if group is too small.
    """
    group = reference_df[
        (reference_df["cuisine"] == cuisine) & (reference_df["city"] == city)
    ]["predicted_probability"]
    if len(group) < min_group_size:
        group = reference_df["predicted_probability"]
    rank = float(np.mean(group < score) * 100)
    return rank


_FEATURE_LABELS = {
    "restaurants_500m": "restaurant density (500m)",
    "restaurants_same_cuisine_500m": "same-cuisine competition (500m)",
    "bars_500m": "nightlife density (500m)",
    "offices_500m": "office density (500m)",
    "hotels_500m": "hotel density (500m)",
    "transit_stops_500m": "transit access (500m)",
    "schools_500m": "schools nearby (500m)",
    "median_income": "neighborhood median income",
    "total_population": "neighborhood population",
    "median_age": "neighborhood median age",
    "cuisine_encoded": "cuisine type fit",
    "price_level": "price level fit",
}


def get_shap_pros_cons(
    shap_values: np.ndarray,
    feature_names: list[str],
    feature_values: dict,
    top_n: int = 3,
) -> tuple[list[dict], list[dict]]:
    """Return top_n pros (positive SHAP) and cons (negative SHAP)."""
    pairs = list(zip(shap_values, feature_names))
    pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
    pros = [
        {"feature": f, "label": _FEATURE_LABELS.get(f, f), "value": feature_values.get(f), "shap": s}
        for s, f in pairs_sorted if s > 0
    ][:top_n]
    cons = [
        {"feature": f, "label": _FEATURE_LABELS.get(f, f), "value": feature_values.get(f), "shap": s}
        for s, f in reversed(pairs_sorted) if s < 0
    ][:top_n]
    return pros, cons
```

- [ ] **Step 4: Run tests — confirm PASS**

```bash
pytest tests/test_predict.py -k "percentile or shap" -v
```

- [ ] **Step 5: Commit**

```bash
git add src/predict.py tests/test_predict.py
git commit -m "feat: model inference, percentile rank computation, and SHAP pros/cons"
```

---

### Task 19: `predict.py` — Comparable restaurants and CLI output

**Files:**
- Modify: `src/predict.py`
- Modify: `tests/test_predict.py`

- [ ] **Step 1: Write failing tests**

```python
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
        pros=[{"label": "high office density", "value": 15, "shap": 0.3}],
        cons=[{"label": "restaurant saturation", "value": 40, "shap": -0.2}],
        comparables=[{"name": "Olive Garden", "cuisine": "italian",
                      "price_level": 2, "rating": 4.1, "distance_km": 1.2}],
    )
    assert "0.71" in output
    assert "LIKELY GOOD" in output or "UNLIKELY" in output
    assert "PROS" in output
    assert "CONS" in output
    assert "Olive Garden" in output
```

- [ ] **Step 2: Run tests — confirm FAIL**

```bash
pytest tests/test_predict.py -k "comparable or format_output" -v
```

- [ ] **Step 3: Implement comparable restaurants and output formatting**

```python
from math import radians, sin, cos, sqrt, atan2


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def find_comparable_restaurants(
    lat: float,
    lon: float,
    cuisine: str,
    price_level: int,
    reference_df: pd.DataFrame,
    max_distance_km: float = 5.0,
    top_n: int = 5,
) -> list[dict]:
    """Find nearby restaurants of similar cuisine and price level."""
    df = reference_df.copy()
    df["_dist"] = df.apply(
        lambda r: _haversine_km(lat, lon, r["lat"], r["lon"]), axis=1
    )
    mask = (
        (df["cuisine"] == cuisine)
        & (df["price_level"].notna())
        & ((df["price_level"] - price_level).abs() <= 1)
        & (df["_dist"] <= max_distance_km)
    )
    nearby = df[mask].sort_values("_dist").head(top_n)
    return [
        {
            "name": row["name"],
            "cuisine": row["cuisine"],
            "price_level": row["price_level"],
            "rating": row.get("rating"),
            "distance_km": round(row["_dist"], 1),
        }
        for _, row in nearby.iterrows()
    ]


_PRICE_SYMBOLS = {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}


def format_output(
    address: str,
    cuisine: str,
    price_level: int,
    probability: float,
    percentile_rank: float,
    pros: list[dict],
    cons: list[dict],
    comparables: list[dict],
) -> str:
    verdict = "LIKELY GOOD LOCATION ✓" if probability >= 0.5 else "UNLIKELY GOOD LOCATION ✗"
    price_sym = _PRICE_SYMBOLS.get(price_level, str(price_level))
    top_pct = round(100 - percentile_rank)

    lines = [
        "",
        "RestaurantGenie — Location Analysis",
        "━" * 38,
        f"Address:   {address}",
        f"Cuisine:   {cuisine.title()}",
        f"Price:     {price_sym}",
        "",
        f"Success probability:  {probability:.2f}  (top {top_pct}% of comparable restaurants)",
        f"Verdict:              {verdict}",
        "",
        "PROS",
    ]
    for p in pros:
        lines.append(f"  + {p['label']}")
    if not pros:
        lines.append("  (no strong positive signals)")
    lines += ["", "CONS"]
    for c in cons:
        lines.append(f"  - {c['label']}")
    if not cons:
        lines.append("  (no strong negative signals)")
    lines += ["", "Comparable restaurants nearby:"]
    for r in comparables:
        stars = f"{r['rating']:.1f}★" if r.get("rating") else "n/a"
        p_sym = _PRICE_SYMBOLS.get(int(r["price_level"]), "?")
        lines.append(f"  {r['name'][:30]:<30} ({r['cuisine']}, {p_sym})  {stars}  {r['distance_km']}km")
    if not comparables:
        lines.append("  No comparable restaurants found within 5km.")
    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests — confirm PASS**

```bash
pytest tests/test_predict.py -k "comparable or format_output" -v
```

- [ ] **Step 5: Commit**

```bash
git add src/predict.py tests/test_predict.py
git commit -m "feat: comparable restaurant lookup and formatted CLI output"
```

---

### Task 20: `predict.py` — CLI entrypoint (end-to-end wiring)

**Files:**
- Modify: `src/predict.py`
- Modify: `tests/test_predict.py`

- [ ] **Step 1: Write failing end-to-end smoke test**

```python
def test_run_prediction_end_to_end(tmp_path, mocker):
    """Full predict pipeline with mocked external calls."""
    from src.predict import run_prediction
    import pickle
    import json
    import shap

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
    assert "0.70" in output
    assert "PROS" in output
```

- [ ] **Step 2: Run test — confirm FAIL**

```bash
pytest tests/test_predict.py::test_run_prediction_end_to_end -v
```

- [ ] **Step 3: Implement `run_prediction` and `main` CLI**

```python
import argparse


def run_prediction(
    address: str,
    cuisine: str,
    price_level: int,
    models_dir: str | Path = "models",
    parquet_path: str | Path = "data/processed/restaurant_features.parquet",
    params_path: str | Path = "models/normalization_params.json",
) -> str:
    model, explainer = load_artifacts(models_dir)
    reference_df = pd.read_parquet(parquet_path)
    lat, lon = geocode_address(address)

    feature_vector, feature_names = assemble_feature_vector(
        lat, lon, cuisine, price_level, params_path
    )

    probability = float(model.predict_proba([feature_vector])[0, 1])
    percentile_rank = compute_percentile_rank(
        probability, reference_df,
        cuisine=cuisine,
        city=_infer_city(lat, lon, reference_df),
    )

    shap_values = explainer.shap_values(feature_vector.reshape(1, -1))
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_vals_1d = shap_values[0]

    feature_values = {n: feature_vector[i] for i, n in enumerate(feature_names)}
    pros, cons = get_shap_pros_cons(shap_vals_1d, feature_names, feature_values)
    comparables = find_comparable_restaurants(lat, lon, cuisine, price_level, reference_df)

    return format_output(address, cuisine, price_level, probability,
                         percentile_rank, pros, cons, comparables)


def _infer_city(lat: float, lon: float, reference_df: pd.DataFrame) -> str:
    """Approximate city by finding the nearest restaurant in the reference set."""
    dists = reference_df.apply(
        lambda r: _haversine_km(lat, lon, r["lat"], r["lon"]), axis=1
    )
    return reference_df.loc[dists.idxmin(), "city"]


def main():
    parser = argparse.ArgumentParser(description="RestaurantGenie — location success predictor")
    parser.add_argument("--address", required=True, help='Full US address e.g. "123 Main St, Austin TX"')
    parser.add_argument("--cuisine", required=True, help='Cuisine type e.g. italian, mexican, pizza')
    parser.add_argument("--price", type=int, choices=[1, 2, 3, 4], required=True,
                        help="Price level: 1=$ 2=$$ 3=$$$ 4=$$$$")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--parquet", default="data/processed/restaurant_features.parquet")
    parser.add_argument("--params", default="models/normalization_params.json")
    args = parser.parse_args()

    output = run_prediction(
        address=args.address,
        cuisine=args.cuisine,
        price_level=args.price,
        models_dir=args.models_dir,
        parquet_path=args.parquet,
        params_path=args.params,
    )
    print(output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test — confirm PASS**

```bash
pytest tests/test_predict.py::test_run_prediction_end_to_end -v
```

- [ ] **Step 5: Run all tests**

```bash
pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/predict.py tests/test_predict.py
git commit -m "feat: predict CLI entrypoint wiring all components end-to-end"
```

---

### Task 21: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# RestaurantGenie

Predicts whether a US address is a good location for a new restaurant.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the [Yelp Open Dataset](https://www.yelp.com/dataset) (free, requires registration).
   Place `yelp_academic_dataset_business.json` in `data/raw/`.

3. (Optional) Get a free Census API key at https://api.census.gov/data/key_signup.html.
   Set it as an environment variable:
   ```bash
   export CENSUS_API_KEY=your_key_here
   ```

## Stage 1: Build Dataset (~30–60 min)

```bash
python src/build_dataset.py
```

Produces `data/processed/restaurant_features.parquet` (~150k rows) and
`models/normalization_params.json`.

## Stage 2: Train Model (~5–15 min)

```bash
python src/train_model.py
```

Produces `models/model.pkl`, `models/shap_explainer.pkl`, and
`models/performance_report.txt` (ROC-AUC, Brier score).

## Stage 3: Predict

```bash
python src/predict.py \
  --address "123 Main St, Austin TX" \
  --cuisine italian \
  --price 2
```

### Example output

```
RestaurantGenie — Location Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Address:   123 Main St, Austin TX
Cuisine:   Italian
Price:     $$

Success probability:  0.71  (top 29% of comparable restaurants)
Verdict:              LIKELY GOOD LOCATION ✓

PROS
  + neighborhood median income
  + office density (500m)

CONS
  - restaurant density (500m)

Comparable restaurants nearby:
  Olive & Vine                   (italian, $$)  4.2★  2.1km
```

## Run Tests

```bash
pytest tests/ -v
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add setup and usage README"
```

---

## Final check

- [ ] **Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests PASS, no errors.
