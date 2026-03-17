import json
import os
import time
from pathlib import Path

import censusgeocode as cg
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.features import fetch_pois_for_bbox, count_pois_by_type, CENSUS_ACS_URL, _CENSUS_VARS, CENSUS_API_KEY

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


_OSM_PADDING_DEG = 0.014  # ~1.5km padding around city bounding box


def compute_success_labels(df: pd.DataFrame, p95_log_reviews: float) -> pd.DataFrame:
    """Add success_score and is_successful columns."""
    df = df.copy()
    df["_norm_rating"] = (df["rating"].fillna(3.0) - 1.0) / 4.0
    df["_norm_log_reviews"] = (
        np.log1p(df["review_count"].fillna(0)) / max(p95_log_reviews, 1e-6)
    )
    df["success_score"] = (
        0.4 * df["_norm_rating"]
        + 0.4 * df["_norm_log_reviews"]
        + 0.2 * df["is_open"].fillna(0)
    )

    # Fill cuisine NaN with 'other' to avoid silent NaN key drops in groupby
    df["_cuisine_filled"] = df["cuisine"].fillna("other")

    # Merge small cuisine groups into 'other'
    cuisine_counts = df.groupby(["city", "_cuisine_filled"]).transform("size")
    df["_cuisine_group"] = df["_cuisine_filled"].where(cuisine_counts >= 10, other="other")

    # Compute 70th percentile threshold per (city, cuisine_group) using merge
    thresholds = (
        df.groupby(["city", "_cuisine_group"])["success_score"]
        .quantile(0.70)
        .reset_index()
        .rename(columns={"success_score": "_threshold"})
    )
    df = df.merge(thresholds, on=["city", "_cuisine_group"], how="left")
    df["is_successful"] = (df["success_score"] > df["_threshold"]).astype(int)
    df = df.drop(columns=["_norm_rating", "_norm_log_reviews", "_cuisine_filled", "_cuisine_group", "_threshold"])
    return df


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
    cuisine_label_map = {"other": 0}
    for i, c in enumerate(cuisines, start=1):
        cuisine_label_map[c] = i

    params = {"p95_log_reviews": p95_log_reviews, "cuisine_label_map": cuisine_label_map}
    Path(output_params).parent.mkdir(parents=True, exist_ok=True)
    with open(output_params, "w") as f:
        json.dump(params, f)
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
