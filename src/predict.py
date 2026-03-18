import argparse
import json
import pickle
import sys
from math import atan2, cos, radians, sin, sqrt
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
    parquet_path: str | Path = "data/processed/restaurant_features.parquet",  # unused, kept for API compat
    city: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Generate full feature vector ordered by selected_features from params.
    All features come from free sources: OSM Overpass and US Census ACS.
    """
    with open(params_path) as f:
        params = json.load(f)
    label_map: dict = params["cuisine_label_map"]
    selected: list[str] = params["selected_features"]

    neighborhood = generate_neighborhood_features(lat, lon, target_cuisine=cuisine)
    neighborhood["cuisine_encoded"] = label_map.get(cuisine, 0)
    neighborhood["price_level"] = price_level

    # Derived features (mirrors add_derived_features in build_dataset.py)
    for r in [250, 500, 1000]:
        total = neighborhood.get(f"restaurants_{r}m") or 0
        same = neighborhood.get(f"restaurants_same_cuisine_{r}m") or 0
        neighborhood[f"same_cuisine_saturation_{r}m"] = (same / total) if total > 0 else 0

    neighborhood["restaurant_bar_ratio_500m"] = (
        (neighborhood.get("restaurants_500m") or 0)
        / ((neighborhood.get("bars_500m") or 0) + 1)
    )
    foot = (
        (neighborhood.get("offices_500m") or 0) * 0.5
        + (neighborhood.get("transit_stops_500m") or 0) * 1.0
        + (neighborhood.get("hotels_500m") or 0) * 0.3
    )
    neighborhood["foot_traffic_proxy_500m"] = foot
    neighborhood["demand_per_restaurant_500m"] = foot / ((neighborhood.get("restaurants_500m") or 0) + 1)
    neighborhood["income_office_interaction"] = (
        (neighborhood.get("median_income") or 0) / 100000.0
        * (neighborhood.get("offices_500m") or 0)
    )
    import math
    pop = max(neighborhood.get("total_population") or 1, 1)
    neighborhood["income_per_capita_proxy"] = (neighborhood.get("median_income") or 0) / math.sqrt(pop)
    poi_cols = ["restaurants_500m", "bars_500m", "offices_500m", "hotels_500m", "transit_stops_500m", "schools_500m"]
    neighborhood["poi_diversity_500m"] = sum(1 for c in poi_cols if (neighborhood.get(c) or 0) > 0)
    neighborhood["total_pois_500m"] = sum((neighborhood.get(c) or 0) for c in poi_cols)

    # Spatial census approximation: use single-tract values as fallback for predict
    for radius in [500, 1000]:
        neighborhood[f"median_income_{radius}m_avg"] = neighborhood.get("median_income") or 0
        neighborhood[f"total_population_{radius}m_avg"] = neighborhood.get("total_population") or 0
        neighborhood[f"median_age_{radius}m_avg"] = neighborhood.get("median_age") or 0
    neighborhood["income_variance_1000m"] = 0.0

    # OSM+Census derived — no external API needed
    same_c_1km = neighborhood.get("restaurants_same_cuisine_1000m") or 0
    r500m = neighborhood.get("restaurants_500m") or 0
    neighborhood["cuisine_gap"] = pop / (same_c_1km + 1)
    neighborhood["cluster_score"] = r500m / pop * 1000
    neighborhood["median_income_x_price"] = (
        (neighborhood.get("median_income") or 0) / 100000.0 * price_level
    )

    # Price tier success rate — lookup table stored at train time, free at inference
    price_tier_rates = params.get("price_tier_rates", {})
    global_rates = price_tier_rates.get("global", {})
    city_rates = price_tier_rates.get("by_city", {})
    p_key = str(price_level)
    city_specific = city_rates.get(city, {}).get(p_key) if city else None
    global_fallback = global_rates.get(p_key, 0.2)
    neighborhood["price_tier_success_rate"] = city_specific if city_specific is not None else global_fallback

    vector = np.array(
        [float(neighborhood.get(f, 0) or 0) for f in selected], dtype=float
    )
    return vector, selected


class _Unpickler(pickle.Unpickler):
    """Resolve _PlattWrapper regardless of whether it was pickled from __main__ or src.train_model."""
    def find_class(self, module, name):
        if name == "_PlattWrapper":
            from src.train_model import _PlattWrapper
            return _PlattWrapper
        return super().find_class(module, name)


def load_artifacts(models_dir: str | Path):
    """Load model.pkl and shap_explainer.pkl. Returns (calibrated_model, explainer)."""
    models_dir = Path(models_dir)
    model_path = models_dir / "model.pkl"
    explainer_path = models_dir / "shap_explainer.pkl"
    if not model_path.exists() or not explainer_path.exists():
        print(f"Error: Model artifacts not found in {models_dir}. Run 'python src/train_model.py' first.")
        sys.exit(1)
    with open(model_path, "rb") as f:
        model = _Unpickler(f).load()
    with open(explainer_path, "rb") as f:
        explainer = pickle.load(f)
    return model, explainer


def compute_percentile_rank(
    score: float,
    reference_df: pd.DataFrame,
    cuisine: str,
    city: str,
    min_group_size: int = 5,
) -> float:
    """Return percentile rank (0-100) of score vs. comparable restaurants.
    Falls back to all rows if group is too small.
    """
    group = reference_df[
        (reference_df["cuisine"] == cuisine) & (reference_df["city"] == city)
    ]["predicted_probability"].dropna()
    if len(group) < min_group_size:
        group = reference_df["predicted_probability"].dropna()
    # `rank` is the "top X%" value — e.g., rank=30 means the score is in the
    # top 30% of comparable restaurants.
    pct_below = float(np.mean(group < score) * 100)
    rank = round(100 - pct_below, 1)
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


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _haversine_km_vec(lat1: float, lon1: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Vectorized haversine from a single point to an array of points."""
    R = 6371.0
    dlat = np.radians(lats - lat1)
    dlon = np.radians(lons - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lats)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


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
    df["_dist"] = _haversine_km_vec(lat, lon, df["lat"].to_numpy(), df["lon"].to_numpy())
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
            "rating": row["rating"] if pd.notna(row["rating"]) else None,
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
    # percentile_rank = 100 - pct_below (top 10% means better than 90%)
    # Verdict: top 35% or better is "LIKELY GOOD LOCATION"
    verdict = "LIKELY GOOD LOCATION" if percentile_rank <= 35 else "UNLIKELY GOOD LOCATION"
    price_sym = _PRICE_SYMBOLS.get(price_level, str(price_level))
    # Express as "better than X%" for clarity; cap at 99 to avoid "better than 100%"
    beats_pct = min(99, round(100 - percentile_rank))

    lines = [
        "",
        "RestaurantGenie -- Location Analysis",
        "-" * 38,
        f"Address:   {address}",
        f"Cuisine:   {cuisine.title()}",
        f"Price:     {price_sym}",
        "",
        f"Success probability:  {probability:.2f}  (better than {beats_pct}% of comparable restaurants)",
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
        stars = f"{r['rating']:.1f}" if r.get("rating") is not None else "n/a"
        p_sym = _PRICE_SYMBOLS.get(int(r["price_level"]), "?")
        lines.append(f"  {r['name'][:30]:<30} ({r['cuisine']}, {p_sym})  {stars}  {r['distance_km']}km")
    if not comparables:
        lines.append("  No comparable restaurants found within 5km.")
    lines.append("")
    return "\n".join(lines)


def _infer_city(lat: float, lon: float, reference_df: pd.DataFrame) -> str:
    """Approximate city by finding the nearest restaurant in the reference set."""
    if reference_df.empty:
        return "unknown"
    dists = _haversine_km_vec(lat, lon, reference_df["lat"].to_numpy(), reference_df["lon"].to_numpy())
    dists = pd.Series(dists, index=reference_df.index)
    return reference_df.loc[dists.idxmin(), "city"]


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
    city = _infer_city(lat, lon, reference_df)

    feature_vector, feature_names = assemble_feature_vector(
        lat, lon, cuisine, price_level, params_path, parquet_path, city=city
    )

    probability = float(model.predict_proba([feature_vector])[0, 1])

    # Score the reference set so percentile_rank compares on the same probability scale.
    if "predicted_probability" not in reference_df.columns:
        ref_features = reference_df[feature_names].fillna(0).values
        reference_df = reference_df.copy()
        reference_df["predicted_probability"] = model.predict_proba(ref_features)[:, 1]

    percentile_rank = compute_percentile_rank(
        probability, reference_df,
        cuisine=cuisine,
        city=city,
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


def main():
    parser = argparse.ArgumentParser(description="RestaurantGenie -- location success predictor")
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
