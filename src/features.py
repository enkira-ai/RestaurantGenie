import os
import time
import requests
import numpy as np
from sklearn.neighbors import BallTree

CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", "")
CENSUS_ACS_URL = "https://api.census.gov/data/2022/acs/acs5"
_CENSUS_GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
_CENSUS_VARS = "B19013_001E,B01003_001E,B01002_001E"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def fetch_census_demographics(lat: float, lon: float) -> dict:
    """Return median_income, total_population, median_age for the census tract
    containing (lat, lon). Returns None values if lookup fails.
    Uses the Census REST geocoder directly (censusgeocode library is unreliable).
    """
    null_result = {"median_income": None, "total_population": None, "median_age": None}
    try:
        resp = requests.get(
            _CENSUS_GEOCODER_URL,
            params={
                "x": lon, "y": lat,
                "benchmark": "Public_AR_Current",
                "vintage": "Current_Current",
                "format": "json",
            },
            timeout=20,
        )
        resp.raise_for_status()
        tracts = resp.json().get("result", {}).get("geographies", {}).get("Census Tracts", [])
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


def fetch_restaurants_nearby(
    lat: float,
    lon: float,
    cuisine: str | None = None,
    radius_km: float = 5.0,
    top_n: int = 5,
) -> list[dict]:
    """Query OSM for nearby restaurants, sorted by distance.

    Returns up to top_n entries: {"name", "lat", "lon", "cuisine", "distance_km"}.
    Tries matching cuisine first; falls back to all restaurants if fewer than top_n found.
    Expands radius up to 20km if needed.
    """
    from math import atan2, cos, radians, sin, sqrt

    def _hav(lat1, lon1, lat2, lon2):
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        return 6371.0 * 2 * atan2(sqrt(a), sqrt(1 - a))

    for search_radius in [radius_km, min(radius_km * 3, 20.0)]:
        delta = search_radius / 111.0
        bbox = f"{lat - delta},{lon - delta},{lat + delta},{lon + delta}"
        query = f"""[out:json][timeout:60];
(
  node["amenity"="restaurant"]({bbox});
  way["amenity"="restaurant"]({bbox});
);
out center tags;
"""
        last_exc = None
        data = None
        for url in _OVERPASS_MIRRORS:
            try:
                resp = requests.post(url, data=query, timeout=90)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                last_exc = e
                time.sleep(1)
        if data is None:
            raise last_exc

        rows = []
        for el in data.get("elements", []):
            tags = el.get("tags", {})
            r_lat = el.get("lat") or el.get("center", {}).get("lat")
            r_lon = el.get("lon") or el.get("center", {}).get("lon")
            if r_lat is None or r_lon is None:
                continue
            name = tags.get("name")
            if not name:
                continue
            rows.append({
                "name": name,
                "lat": r_lat,
                "lon": r_lon,
                "cuisine": tags.get("cuisine"),
                "distance_km": round(_hav(lat, lon, r_lat, r_lon), 2),
            })

        rows.sort(key=lambda r: r["distance_km"])

        if cuisine:
            # Match on exact OSM cuisine tokens (tags like "italian;pizza" are split on ";")
            same = [
                r for r in rows
                if r.get("cuisine")
                and any(cuisine == tok.strip() for tok in r["cuisine"].lower().split(";"))
            ]
            if same:
                return same[:top_n]
            # No same-cuisine at this radius — try expanding before giving up
            continue

        # No cuisine filter requested — return nearest restaurants
        if rows:
            return rows[:top_n]

    return []


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


_OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]


def fetch_pois_for_bbox(
    south: float, west: float, north: float, east: float
) -> list[dict]:
    """Query OSM Overpass for all relevant POIs in the bounding box.
    Returns list of {"lat", "lon", "type", "cuisine"}.
    Retries across mirror servers on timeout or 5xx errors.
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
    last_exc = None
    for url in _OVERPASS_MIRRORS:
        try:
            response = requests.post(url, data=query, timeout=120)
            response.raise_for_status()
            break
        except Exception as e:
            last_exc = e
            time.sleep(2)
    else:
        raise last_exc
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
