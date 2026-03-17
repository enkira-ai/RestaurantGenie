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
