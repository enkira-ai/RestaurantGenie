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


def compute_success_labels(
    df: pd.DataFrame,
    p95_log_reviews: float,
    review_stats_path: str | "Path" | None = "data/processed/review_stats.parquet",
) -> pd.DataFrame:
    """Compute success score and binary label using improved methodology.

    Improvements over v1:
    - Review velocity (age-adjusted) instead of raw review count
    - Bayesian-smoothed rating per peer group
    - Peer group = (city, cuisine_group, price_bucket) with hierarchical backoff
    - Minimum evidence threshold (>= 15 reviews)
    - Removes is_open from target (circular/noisy)
    - Recent activity signal from review timestamps
    """
    df = df.copy()

    # ── Load review stats if available ──────────────────────────────────────
    has_review_stats = False
    if review_stats_path is not None:
        try:
            rs = pd.read_parquet(review_stats_path)
            df = df.merge(rs[["business_id", "first_review_date", "last_review_date",
                               "review_count_total", "reviews_last_12m"]],
                          on="business_id", how="left")
            has_review_stats = True
        except Exception as e:
            print(f"  Warning: could not load review stats ({e}), falling back to basic labels")

    # ── Review velocity ──────────────────────────────────────────────────────
    if has_review_stats:
        # Age in years from first to last review date
        first = pd.to_datetime(df["first_review_date"])
        last  = pd.to_datetime(df["last_review_date"])
        age_years = ((last - first).dt.days / 365.25).clip(lower=0.25)  # min 3 months
        df["restaurant_age_years"] = age_years

        review_count = df["review_count_total"].fillna(df["review_count"].fillna(0))
        df["review_velocity"] = review_count / age_years.clip(lower=0.25)
        df["recent_activity"] = df["reviews_last_12m"].fillna(0)
        # Minimum evidence: at least 15 reviews to be eligible for labelling
        df["_eligible"] = review_count >= 15
    else:
        df["review_velocity"] = df["review_count"].fillna(0)
        df["recent_activity"] = 0.0
        df["_eligible"] = df["review_count"].fillna(0) >= 5

    # ── Price bucket ─────────────────────────────────────────────────────────
    # 1-2 = "affordable", 3-4 = "premium", null = "unknown"
    df["_price_bucket"] = pd.cut(
        df["price_level"].fillna(0),
        bins=[-0.1, 0.5, 2.5, 4.5],
        labels=["unknown", "affordable", "premium"],
    ).astype(str)

    # ── Peer group with hierarchical backoff ──────────────────────────────────
    df["_cuisine_filled"] = df["cuisine"].fillna("other")

    # Merge small cuisine groups into 'other' (< 10 per city)
    cuisine_counts = df.groupby(["city", "_cuisine_filled"]).transform("size")
    df["_cuisine_group"] = df["_cuisine_filled"].where(cuisine_counts >= 10, other="other")

    # Primary peer group: (city, cuisine_group, price_bucket)
    df["_peer_group"] = (
        df["city"] + "||" + df["_cuisine_group"] + "||" + df["_price_bucket"]
    )
    peer_sizes = df.groupby("_peer_group")["_peer_group"].transform("count")

    # Backoff to (city, cuisine_group) if < 10 in primary group
    df["_peer_group_fallback"] = df["city"] + "||" + df["_cuisine_group"]
    df["_peer_group_final"] = df["_peer_group"].where(
        peer_sizes >= 10, other=df["_peer_group_fallback"]
    )

    # ── Bayesian-smoothed rating ──────────────────────────────────────────────
    # smoothed = (v/(v+m))*R + (m/(v+m))*C
    # where C = peer group mean rating, m = 25 (prior strength)
    m = 25
    peer_mean_rating = (
        df.groupby("_peer_group_final")["rating"]
        .transform("mean")
        .fillna(df["rating"].mean())
    )
    v = df["review_count"].fillna(0).clip(lower=0)
    R = df["rating"].fillna(3.0)
    df["smoothed_rating"] = (v / (v + m)) * R + (m / (v + m)) * peer_mean_rating

    # ── Compute composite score with z-score standardisation within peer group ──
    def _z_within_group(series: pd.Series, group: pd.Series) -> pd.Series:
        """Z-score within each peer group. Fill NaN groups with 0."""
        mu = series.groupby(group).transform("mean")
        sd = series.groupby(group).transform("std").fillna(1.0).clip(lower=1e-6)
        return (series - mu) / sd

    log_velocity = np.log1p(df["review_velocity"].fillna(0))
    log_activity = np.log1p(df["recent_activity"].fillna(0))

    z_rating   = _z_within_group(df["smoothed_rating"], df["_peer_group_final"])
    z_velocity = _z_within_group(log_velocity,          df["_peer_group_final"])
    z_activity = _z_within_group(log_activity,          df["_peer_group_final"])

    df["success_score"] = (
        0.50 * z_rating
        + 0.35 * z_velocity
        + 0.15 * z_activity
    )

    # ── Binary label: top 25% within peer group, among eligible restaurants ──
    # Ineligible restaurants get label 0 (insufficient evidence)
    thresholds = (
        df[df["_eligible"]]
        .groupby("_peer_group_final")["success_score"]
        .quantile(0.75)
        .reset_index()
        .rename(columns={"success_score": "_threshold"})
    )
    df = df.merge(thresholds, on="_peer_group_final", how="left")

    df["is_successful"] = (
        (df["_eligible"]) &
        (df["success_score"] > df["_threshold"].fillna(float("inf")))
    ).astype(int)

    # ── Clean up temp columns ─────────────────────────────────────────────────
    drop_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=drop_cols, errors="ignore")

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


def add_spatial_census_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add radius-averaged census features using other restaurants as tract proxies.

    For each restaurant, averages the census demographics of all other restaurants
    within 500m and 1000m as a proxy for nearby-tract demographics.
    """
    from sklearn.neighbors import BallTree

    df = df.copy()

    # Only use rows with valid census data as reference points
    census_cols = ["median_income", "total_population", "median_age"]
    valid = df[census_cols].notna().all(axis=1)
    ref = df[valid].copy()

    if len(ref) < 10:
        for col in census_cols:
            for radius in [500, 1000]:
                df[f"{col}_{radius}m_avg"] = df[col]
        df["income_variance_1000m"] = 0.0
        return df

    coords_rad = np.radians(ref[["lat", "lon"]].values)
    tree = BallTree(coords_rad, metric="haversine")

    all_coords_rad = np.radians(df[["lat", "lon"]].values)

    for radius_m in [500, 1000]:
        r = radius_m / 6_371_000
        indices = tree.query_radius(all_coords_rad, r=r)

        for col in census_cols:
            ref_vals = ref[col].values
            avgs = []
            for idx_list in indices:
                if len(idx_list) > 0:
                    avgs.append(float(np.nanmean(ref_vals[idx_list])))
                else:
                    avgs.append(np.nan)
            df[f"{col}_{radius_m}m_avg"] = avgs

    # Income variance within 1km (neighbourhood income diversity)
    ref_income = ref["median_income"].values
    variances = []
    r1000 = 1000 / 6_371_000
    indices_1000 = tree.query_radius(all_coords_rad, r=r1000)
    for idx_list in indices_1000:
        if len(idx_list) > 1:
            variances.append(float(np.nanstd(ref_income[idx_list])))
        else:
            variances.append(0.0)
    df["income_variance_1000m"] = variances

    return df


def add_price_tier_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price-tier success rate features per city.

    For expensive/rare price tiers with few examples, tells the model
    how viable that price point is in this city.
    """
    df = df.copy()

    # Success rate per (city, price_level) — how often does this price tier succeed here
    tier_success = (
        df.groupby(["city", "price_level"])["is_successful"]
        .mean()
        .reset_index()
        .rename(columns={"is_successful": "price_tier_success_rate"})
    )
    df = df.merge(tier_success, on=["city", "price_level"], how="left")

    # Count of restaurants at this price tier in this city (rarity signal)
    tier_count = (
        df.groupby(["city", "price_level"])["is_successful"]
        .count()
        .reset_index()
        .rename(columns={"is_successful": "price_tier_count"})
    )
    df = df.merge(tier_count, on=["city", "price_level"], how="left")

    # Log-transform count to dampen outliers
    df["price_tier_count_log"] = np.log1p(df["price_tier_count"].fillna(0))

    return df


def add_yelp_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute spatial features from the Yelp dataset itself using BallTree.

    All features are derivable at inference time by querying the reference parquet.
    """
    from sklearn.neighbors import BallTree
    import numpy as np
    from scipy.stats import entropy as scipy_entropy

    df = df.copy()
    coords_rad = np.radians(df[["lat", "lon"]].values)
    tree = BallTree(coords_rad, metric="haversine")

    EARTH_RADIUS_M = 6_371_000

    # ── Radius indices (precompute once) ────────────────────────────────────
    idx_500m  = tree.query_radius(coords_rad, r=500  / EARTH_RADIUS_M)
    idx_1000m = tree.query_radius(coords_rad, r=1000 / EARTH_RADIUS_M)
    idx_2000m = tree.query_radius(coords_rad, r=2000 / EARTH_RADIUS_M)

    price_vals   = df["price_level"].values
    rating_vals  = df["rating"].fillna(0).values
    reviews_vals = df["review_count"].fillna(0).values
    cuisine_vals = df["cuisine"].fillna("other").values

    n = len(df)

    avg_price_1km      = np.zeros(n)
    median_price_1km   = np.zeros(n)
    avg_rating_1km     = np.zeros(n)
    avg_reviews_1km    = np.zeros(n)
    total_reviews_1km  = np.zeros(n)
    same_price_1km     = np.zeros(n)
    cuisine_entropy_1km = np.zeros(n)
    restaurant_count_2km = np.zeros(n)

    cuisine_col = df["cuisine"].fillna("other").values
    price_col   = df["price_level"].fillna(0).values

    for i in range(n):
        # 1km neighbours (exclude self)
        nbrs = idx_1000m[i]
        nbrs = nbrs[nbrs != i]

        if len(nbrs) > 0:
            p = price_vals[nbrs]
            valid_p = p[~np.isnan(p)]
            avg_price_1km[i]    = np.mean(valid_p) if len(valid_p) > 0 else 0
            median_price_1km[i] = np.median(valid_p) if len(valid_p) > 0 else 0
            avg_rating_1km[i]   = np.mean(rating_vals[nbrs])
            avg_reviews_1km[i]  = np.mean(reviews_vals[nbrs])
            total_reviews_1km[i]= np.sum(reviews_vals[nbrs])

            # Same price tier count within 1km
            own_price = price_col[i]
            same_price_1km[i] = np.sum(np.abs(valid_p - own_price) <= 0) if len(valid_p) > 0 else 0

            # Cuisine Shannon entropy
            cuisines_nearby = cuisine_vals[nbrs]
            _, counts = np.unique(cuisines_nearby, return_counts=True)
            probs = counts / counts.sum()
            cuisine_entropy_1km[i] = float(scipy_entropy(probs))

        # 2km restaurant count
        nbrs_2km = idx_2000m[i]
        restaurant_count_2km[i] = len(nbrs_2km[nbrs_2km != i])

    df["avg_price_1km"]       = avg_price_1km
    df["median_price_1km"]    = median_price_1km
    df["avg_rating_1km"]      = avg_rating_1km
    df["avg_reviews_1km"]     = avg_reviews_1km
    df["total_reviews_1km"]   = total_reviews_1km
    df["same_price_1km"]      = same_price_1km
    df["cuisine_entropy_1km"] = cuisine_entropy_1km
    df["restaurants_2km"]     = restaurant_count_2km

    # Price mismatch: |proposed_price - avg local price|
    df["price_mismatch_1km"] = (
        (df["price_level"].fillna(0) - df["avg_price_1km"]).abs()
    )

    # Cuisine gap: population / max(1, same_cuisine within 1km)
    # Uses restaurants_same_cuisine_1000m already computed
    same_cuisine_1km = df.get("restaurants_same_cuisine_1000m", pd.Series(0, index=df.index)).fillna(0)
    pop = df.get("total_population", pd.Series(1, index=df.index)).fillna(1).clip(lower=1)
    df["cuisine_gap"] = pop / (same_cuisine_1km + 1)

    # Cluster score: restaurants per unit population
    restaurants_500m = df.get("restaurants_500m", pd.Series(0, index=df.index)).fillna(0)
    df["cluster_score"] = restaurants_500m / pop.clip(lower=1) * 1000  # per 1000 people

    # Distance to city center (centroid of all restaurants in same city)
    city_centers = df.groupby("city")[["lat", "lon"]].mean().rename(
        columns={"lat": "city_center_lat", "lon": "city_center_lon"}
    )
    df = df.merge(city_centers, on="city", how="left")
    from math import radians as _rad, sin, cos, sqrt, atan2
    def _hav(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = _rad(lat2 - lat1); dlon = _rad(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(_rad(lat1))*cos(_rad(lat2))*sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1-a))
    df["distance_city_center"] = df.apply(
        lambda r: _hav(r["lat"], r["lon"], r["city_center_lat"], r["city_center_lon"]), axis=1
    )
    df = df.drop(columns=["city_center_lat", "city_center_lon"])

    # Same cuisine + same price within 1km
    same_cuisine_price = np.zeros(n)
    for i in range(n):
        nbrs = idx_1000m[i]
        nbrs = nbrs[nbrs != i]
        if len(nbrs) > 0:
            own_cuisine = cuisine_col[i]
            own_price   = price_col[i]
            match = (cuisine_vals[nbrs] == own_cuisine) & (np.abs(price_vals[nbrs] - own_price) <= 0)
            same_cuisine_price[i] = np.sum(match)
    df["same_cuisine_price_1km"] = same_cuisine_price

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features from OSM counts and Census demographics."""
    df = df.copy()

    # --- Competition / saturation ratios ---
    # Same-cuisine saturation: fraction of nearby restaurants that are same cuisine
    for r in [250, 500, 1000]:
        total_col = f"restaurants_{r}m"
        same_col = f"restaurants_same_cuisine_{r}m"
        new_col = f"same_cuisine_saturation_{r}m"
        df[new_col] = (
            df[same_col].fillna(0) / df[total_col].replace(0, np.nan)
        ).fillna(0)

    # Restaurant-to-bar ratio at 500m (entertainment vs residential signal)
    df["restaurant_bar_ratio_500m"] = (
        df["restaurants_500m"] / (df["bars_500m"] + 1)
    )

    # --- Demand density composites ---
    # Foot traffic proxy: offices + transit + hotels weighted sum at 500m
    df["foot_traffic_proxy_500m"] = (
        df["offices_500m"].fillna(0) * 0.5
        + df["transit_stops_500m"].fillna(0) * 1.0
        + df["hotels_500m"].fillna(0) * 0.3
    )

    # Demand per restaurant (how many demand sources per competing restaurant)
    df["demand_per_restaurant_500m"] = (
        df["foot_traffic_proxy_500m"] / (df["restaurants_500m"] + 1)
    )

    # --- Census interactions ---
    # Income × office density: wealthy daytime workers nearby
    df["income_office_interaction"] = (
        df["median_income"].fillna(0) / 100000.0
        * df["offices_500m"].fillna(0)
    )

    # Income per capita proxy: income / sqrt(population) normalises for city size
    df["income_per_capita_proxy"] = (
        df["median_income"].fillna(0).astype(float)
        / (np.sqrt(df["total_population"].fillna(1).astype(float).clip(lower=1)))
    )

    # --- Neighbourhood character ---
    # Diversity index: count of non-zero POI types within 500m
    poi_cols_500m = ["restaurants_500m", "bars_500m", "offices_500m",
                     "hotels_500m", "transit_stops_500m", "schools_500m"]
    df["poi_diversity_500m"] = (df[poi_cols_500m].fillna(0) > 0).sum(axis=1)

    # Total POI density at 500m (raw activity level)
    df["total_pois_500m"] = df[poi_cols_500m].fillna(0).sum(axis=1)

    return df


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
    df = compute_success_labels(df, p95_log_reviews=p95_log_reviews,
                                 review_stats_path="data/processed/review_stats.parquet")

    print("Computing derived features...")
    df = add_derived_features(df)

    print("Computing spatial census features...")
    df = add_spatial_census_features(df)

    print("Computing price tier features...")
    df = add_price_tier_features(df)

    print("Computing Yelp spatial features...")
    df = add_yelp_spatial_features(df)

    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f"  Saved {len(df)} rows to {output_parquet}")


if __name__ == "__main__":
    build_dataset(
        yelp_jsonl_path="data/raw/yelp_academic_dataset_business.json",
        output_parquet="data/processed/restaurant_features.parquet",
        output_params="models/normalization_params.json",
    )
