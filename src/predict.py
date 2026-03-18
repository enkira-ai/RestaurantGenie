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


def geocode_address(address: str) -> tuple[float, float, str | None, str | None]:
    """Geocode US address to (lat, lon, city, state). Exits with message on failure."""
    location = _GEOLOCATOR.geocode(address, country_codes="us", timeout=10, addressdetails=True)
    if location is None:
        print(
            f'Error: Could not geocode address "{address}". '
            "Try a more specific address including city and state."
        )
        sys.exit(1)
    addr_raw = location.raw.get("address", {})
    city = (
        addr_raw.get("city")
        or addr_raw.get("town")
        or addr_raw.get("village")
        or addr_raw.get("county")
    )
    state = addr_raw.get("state")
    return location.latitude, location.longitude, city, state


def assemble_feature_vector(
    lat: float,
    lon: float,
    cuisine: str,
    price_level: int,
    params_path: str | Path,
    city: str | None = None,
    state: str | None = None,
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

    # State-relative income features
    state_abbrev = _STATE_NAME_TO_ABBREV.get(state, "") if state else ""
    state_median = _STATE_ABBREV_TO_MEDIAN.get(state_abbrev, _US_MEDIAN_INCOME)
    inc = neighborhood.get("median_income") or 0
    ratio = (inc / state_median) if (inc > 0 and state_median > 0) else 1.0
    ratio = max(0.1, min(5.0, ratio))
    neighborhood["income_relative_to_state"] = ratio
    neighborhood["income_level_state_cat"] = float(0 if ratio < 0.75 else (1 if ratio < 1.25 else 2))
    state_label_map: dict = params.get("state_label_map", {})
    neighborhood["state_encoded"] = float(state_label_map.get(state_abbrev, -1))

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
    # Competition
    "restaurants_250m": "restaurant competition (250m)",
    "restaurants_500m": "restaurant competition (500m)",
    "restaurants_1000m": "restaurant competition (1km)",
    "restaurants_same_cuisine_250m": "same-cuisine competition (250m)",
    "restaurants_same_cuisine_500m": "same-cuisine competition (500m)",
    "restaurants_same_cuisine_1000m": "same-cuisine competition (1km)",
    "same_cuisine_saturation_250m": "same-cuisine saturation (250m)",
    "same_cuisine_saturation_500m": "same-cuisine saturation (500m)",
    "same_cuisine_saturation_1000m": "same-cuisine saturation (1km)",
    # Foot traffic demand
    "bars_250m": "nightlife density (250m)",
    "bars_500m": "nightlife density (500m)",
    "bars_1000m": "nightlife density (1km)",
    "offices_250m": "office / daytime workers (250m)",
    "offices_500m": "office / daytime workers (500m)",
    "offices_1000m": "office / daytime workers (1km)",
    "hotels_250m": "hotels nearby (250m)",
    "hotels_500m": "hotels nearby (500m)",
    "hotels_1000m": "hotels nearby (1km)",
    "transit_stops_250m": "transit access (250m)",
    "transit_stops_500m": "transit access (500m)",
    "transit_stops_1000m": "transit access (1km)",
    "schools_250m": "schools nearby (250m)",
    "schools_500m": "schools nearby (500m)",
    "schools_1000m": "schools nearby (1km)",
    "foot_traffic_proxy_500m": "estimated foot traffic (offices + transit + hotels)",
    "demand_per_restaurant_500m": "foot traffic per competing restaurant",
    "restaurant_bar_ratio_500m": "restaurant-to-bar ratio (entertainment vs residential)",
    "poi_diversity_500m": "neighbourhood activity variety",
    "total_pois_500m": "total nearby points of interest",
    # Demographics
    "median_income": "neighbourhood median household income",
    "total_population": "neighbourhood population",
    "median_age": "neighbourhood median age",
    "median_income_500m_avg": "average income within 500m",
    "median_income_1000m_avg": "average income within 1km",
    "total_population_500m_avg": "population density within 500m",
    "total_population_1000m_avg": "population density within 1km",
    "median_age_500m_avg": "average age within 500m",
    "income_office_interaction": "wealthy daytime workers nearby",
    "income_per_capita_proxy": "neighbourhood income and population",
    "median_income_x_price": "area income matched to price tier",
    # Price & cuisine fit
    "cuisine_encoded": "cuisine type fit for area",
    "price_level": "price level fit",
    "price_tier_success_rate": "historical success rate for this price tier in this city",
    # Spatial
    "cuisine_gap": "unmet demand for this cuisine",
    "cluster_score": "restaurant cluster density",
    "distance_city_center": "distance from city centre",
    # State-relative income
    "income_relative_to_state": "neighbourhood income relative to state median",
    "income_level_state_cat": "income tier vs state (low / avg / high)",
    "state_encoded": "state",
}


_FEATURE_SUMMARIES = {
    "restaurants_250m":          ("low nearby competition", "high nearby competition"),
    "restaurants_500m":          ("low nearby competition", "high nearby competition"),
    "restaurants_1000m":         ("low competition in the area", "heavy competition in the area"),
    "restaurants_same_cuisine_250m": ("few same-cuisine rivals nearby", "many same-cuisine rivals nearby"),
    "restaurants_same_cuisine_500m": ("few same-cuisine rivals nearby", "many same-cuisine rivals nearby"),
    "restaurants_same_cuisine_1000m": ("few same-cuisine rivals in the area", "many same-cuisine rivals in the area"),
    "same_cuisine_saturation_250m":  ("unders­aturated cuisine niche nearby", "over-saturated cuisine niche nearby"),
    "same_cuisine_saturation_1000m": ("undersaturated cuisine niche in the area", "over-saturated cuisine niche in the area"),
    "bars_500m":                 ("strong nightlife / evening foot traffic", "low nightlife foot traffic"),
    "bars_1000m":                ("strong nightlife / evening foot traffic nearby", "low nightlife foot traffic nearby"),
    "offices_250m":              ("strong lunch demand from nearby offices", "few offices nearby for lunch trade"),
    "offices_500m":              ("strong lunch demand from nearby offices", "few offices nearby for lunch trade"),
    "offices_1000m":             ("good office lunch demand in the area", "limited office lunch demand"),
    "transit_stops_250m":        ("excellent transit access", "poor transit access"),
    "transit_stops_500m":        ("good transit access", "limited transit access"),
    "transit_stops_1000m":       ("transit access nearby", "limited transit access in the area"),
    "schools_1000m":             ("schools nearby drive family dining demand", "few schools nearby"),
    "hotels_250m":               ("hotels nearby drive tourist dining demand", "few hotels nearby"),
    "foot_traffic_proxy_500m":   ("high estimated foot traffic", "low estimated foot traffic"),
    "demand_per_restaurant_500m": ("high demand relative to competition", "low demand relative to competition"),
    "poi_diversity_500m":        ("diverse, active neighbourhood", "low neighbourhood activity"),
    "median_income":             ("affluent neighbourhood supports dining spend", "lower-income area may limit spend"),
    "total_population":          ("large local population", "small local population"),
    "median_age":                ("demographic profile suits this cuisine", "demographic profile may not suit this cuisine"),
    "income_office_interaction": ("wealthy daytime workers nearby", "few wealthy daytime workers nearby"),
    "income_per_capita_proxy":   ("high income per capita", "low income per capita"),
    "median_income_x_price":     ("neighbourhood income well-matched to this price level", "income-to-price fit is challenging for this market"),
    "cuisine_encoded":           ("cuisine type suits local demand", "cuisine type may not suit local demand"),
    "price_level":               ("price level suits local market", "price level may not suit local market"),
    "price_tier_success_rate":   ("this price tier performs well in this city", "this price tier has a poor track record in this city"),
    "cuisine_gap":               ("underserved demand for this cuisine", "plenty of this cuisine already available"),
    "cluster_score":             ("strong restaurant cluster — destination dining area", "sparse restaurant area"),
    "income_relative_to_state":  ("neighbourhood income above state average", "neighbourhood income below state average"),
    "income_level_state_cat":    ("affluent area by state standards", "lower-income area by state standards"),
    "state_encoded":             ("state-level dining market is favourable", "state-level dining market is challenging"),
}


def build_summary(
    cuisine: str,
    price_level: int,
    score: int,
    pros: list[dict],
    cons: list[dict],
) -> str:
    """Build a plain-English paragraph explaining the score."""
    price_sym = _PRICE_SYMBOLS.get(price_level, str(price_level))
    if score >= 70:
        opening = f"This looks like a strong location for a {price_sym} {cuisine.title()} restaurant."
    elif score >= 50:
        opening = f"This is a reasonable location for a {price_sym} {cuisine.title()} restaurant with some caveats."
    elif score >= 30:
        opening = f"This location has meaningful challenges for a {price_sym} {cuisine.title()} restaurant."
    else:
        opening = f"This location is a poor fit for a {price_sym} {cuisine.title()} restaurant."

    pro_phrases = []
    for p in pros:
        f = p["feature"]
        if f in _FEATURE_SUMMARIES:
            pro_phrases.append(_FEATURE_SUMMARIES[f][0])

    con_phrases = []
    for c in cons:
        f = c["feature"]
        if f in _FEATURE_SUMMARIES:
            con_phrases.append(_FEATURE_SUMMARIES[f][1])

    parts = [opening]
    if pro_phrases:
        parts.append("The main strengths are: " + "; ".join(pro_phrases) + ".")
    if con_phrases:
        parts.append("The main risks are: " + "; ".join(con_phrases) + ".")

    return " ".join(parts)


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
    top_n: int = 5,
) -> tuple[list[dict], bool]:
    """Find nearby restaurants from live OSM data, sorted by distance.

    Returns (results, same_cuisine_found). When same_cuisine_found is False
    the results are restaurants of any cuisine (used as context only).
    """
    from src.features import fetch_restaurants_nearby

    # Try same cuisine first
    same = fetch_restaurants_nearby(lat, lon, cuisine=cuisine, top_n=top_n)
    if same and any(
        r.get("cuisine") and any(cuisine == tok.strip() for tok in r["cuisine"].lower().split(";"))
        for r in same
    ):
        return same, True

    # Fall back to all restaurants nearby as context
    all_nearby = fetch_restaurants_nearby(lat, lon, cuisine=None, top_n=top_n)
    return all_nearby, False


_PRICE_SYMBOLS = {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}

# Full state name (as returned by Nominatim) → 2-letter USPS abbreviation
_STATE_NAME_TO_ABBREV: dict[str, str] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN",
    "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
    "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}

# 2-letter → median income (mirrors train_model._STATE_ABBREV_TO_MEDIAN)
_STATE_ABBREV_TO_MEDIAN: dict[str, int] = {
    "AL": 54_943, "AK": 77_640, "AZ": 65_913, "AR": 52_528,
    "CA": 84_097, "CO": 82_254, "CT": 83_771, "DE": 76_048,
    "DC": 101_027, "FL": 63_062, "GA": 65_030,
    "HI": 88_005, "ID": 67_473, "IL": 72_205, "IN": 62_743,
    "IA": 66_054, "KS": 65_700, "KY": 57_834, "LA": 54_216,
    "ME": 68_251, "MD": 94_384, "MA": 89_645, "MI": 65_459,
    "MN": 80_441, "MS": 48_610, "MO": 61_847, "MT": 63_249,
    "NE": 68_116, "NV": 67_510, "NH": 90_845, "NJ": 96_346,
    "NM": 54_020, "NY": 75_157, "NC": 62_891, "ND": 70_651,
    "OH": 62_689, "OK": 57_057, "OR": 72_258, "PA": 67_587,
    "RI": 74_489, "SC": 60_965, "SD": 65_030, "TN": 60_560,
    "TX": 67_321, "UT": 82_646, "VT": 72_431, "VA": 82_450,
    "WA": 87_648, "WV": 51_615, "WI": 69_943, "WY": 68_002,
}

# State median household incomes — 2022 ACS 5-year estimates
# Keys are full state names as returned by Nominatim
_STATE_MEDIAN_INCOME: dict[str, int] = {
    "Alabama": 54_943, "Alaska": 77_640, "Arizona": 65_913, "Arkansas": 52_528,
    "California": 84_097, "Colorado": 82_254, "Connecticut": 83_771, "Delaware": 76_048,
    "District of Columbia": 101_027, "Florida": 63_062, "Georgia": 65_030,
    "Hawaii": 88_005, "Idaho": 67_473, "Illinois": 72_205, "Indiana": 62_743,
    "Iowa": 66_054, "Kansas": 65_700, "Kentucky": 57_834, "Louisiana": 54_216,
    "Maine": 68_251, "Maryland": 94_384, "Massachusetts": 89_645, "Michigan": 65_459,
    "Minnesota": 80_441, "Mississippi": 48_610, "Missouri": 61_847, "Montana": 63_249,
    "Nebraska": 68_116, "Nevada": 67_510, "New Hampshire": 90_845, "New Jersey": 96_346,
    "New Mexico": 54_020, "New York": 75_157, "North Carolina": 62_891,
    "North Dakota": 70_651, "Ohio": 62_689, "Oklahoma": 57_057, "Oregon": 72_258,
    "Pennsylvania": 67_587, "Rhode Island": 74_489, "South Carolina": 60_965,
    "South Dakota": 65_030, "Tennessee": 60_560, "Texas": 67_321, "Utah": 82_646,
    "Vermont": 72_431, "Virginia": 82_450, "Washington": 87_648,
    "West Virginia": 51_615, "Wisconsin": 69_943, "Wyoming": 68_002,
}

# US national benchmarks (2022 ACS)
_US_MEDIAN_INCOME = 74_580
_US_MEDIAN_AGE = 38.8
_US_MEDIAN_TRACT_POP = 4_100   # typical census tract


def _format_feature_value(
    feature: str,
    value,
    state: str | None = None,
    feature_values: dict | None = None,
) -> str | None:
    """Return a formatted value string with units and qualitative level.

    Returns None if no sensible formatting is defined for the feature.
    state (full name, e.g. "Connecticut") is used to compare income against
    the state median rather than the US national median.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None

    if feature in ("median_income", "median_income_500m_avg", "median_income_1000m_avg"):
        if v < 0:
            return None
        ref = _STATE_MEDIAN_INCOME.get(state, _US_MEDIAN_INCOME) if state else _US_MEDIAN_INCOME
        ref_label = f"{state} median ${ref:,}" if state and state in _STATE_MEDIAN_INCOME else f"US median ${ref:,}"
        if v > ref * 1.35:
            level = "high"
        elif v < ref * 0.65:
            level = "low"
        else:
            level = "near state avg" if state and state in _STATE_MEDIAN_INCOME else "near avg"
        return f"${v:,.0f}/yr  ({level}, {ref_label})"

    if feature == "median_income_x_price":
        price = float((feature_values or {}).get("price_level") or 0)
        if price > 0:
            implied_income = v / price * 100_000
            ref = _STATE_ABBREV_TO_MEDIAN.get(
                _STATE_NAME_TO_ABBREV.get(state, ""), _US_MEDIAN_INCOME
            ) if state else _US_MEDIAN_INCOME
            ref_label = f"{state} median" if state and state in _STATE_MEDIAN_INCOME else "state median"
            if implied_income > ref * 2.0:
                note = f"ultra-affluent — high-end dining market already saturated ({ref_label} ${ref:,})"
            elif implied_income > ref * 1.35:
                note = f"affluent area suits this price tier ({ref_label} ${ref:,})"
            elif implied_income < ref * 0.65:
                note = f"area income may be low for this price tier ({ref_label} ${ref:,})"
            else:
                note = f"income well-matched to price tier ({ref_label} ${ref:,})"
            return f"tract median ~${implied_income:,.0f}  — {note}"
        return None

    if feature in ("median_age", "median_age_500m_avg", "median_age_1000m_avg"):
        if v < 30:
            level = "young area"
        elif v > 50:
            level = "older area"
        else:
            level = "near US avg"
        return f"{v:.1f} yrs  ({level}, US median {_US_MEDIAN_AGE})"

    if feature in ("total_population", "total_population_500m_avg", "total_population_1000m_avg"):
        if v < 0:
            return None
        if v > _US_MEDIAN_TRACT_POP * 1.5:
            level = "dense tract"
        elif v < _US_MEDIAN_TRACT_POP * 0.5:
            level = "sparse tract"
        else:
            level = "typical density"
        return f"{v:,.0f} people  ({level})"

    if feature.startswith("restaurants_") and "same_cuisine" not in feature and "saturation" not in feature:
        radius = feature.split("_")[-1]  # e.g. "500m"
        thresholds = {"250m": (5, 12), "500m": (10, 25), "1000m": (25, 60)}
        lo, hi = thresholds.get(radius, (10, 25))
        level = "high competition" if v > hi else ("moderate" if v > lo else "low competition")
        return f"{int(v)} nearby  ({level})"

    if feature.startswith("restaurants_same_cuisine_"):
        radius = feature.split("_")[-1]
        thresholds = {"250m": (2, 6), "500m": (4, 12), "1000m": (8, 25)}
        lo, hi = thresholds.get(radius, (4, 12))
        level = "heavily contested" if v > hi else ("some competition" if v > lo else "low same-cuisine competition")
        return f"{int(v)} same-cuisine  ({level})"

    if feature.startswith("bars_"):
        radius = feature.split("_")[-1]
        thresholds = {"250m": (1, 4), "500m": (3, 10), "1000m": (6, 20)}
        lo, hi = thresholds.get(radius, (3, 10))
        level = "strong nightlife" if v > hi else ("some nightlife" if v > lo else "quiet area")
        return f"{int(v)} bars  ({level})"

    if feature.startswith("offices_"):
        radius = feature.split("_")[-1]
        thresholds = {"250m": (2, 6), "500m": (5, 15), "1000m": (10, 30)}
        lo, hi = thresholds.get(radius, (5, 15))
        level = "strong office district" if v > hi else ("some offices" if v > lo else "few offices")
        return f"{int(v)} offices  ({level})"

    if feature.startswith("transit_stops_"):
        radius = feature.split("_")[-1]
        level = "excellent access" if v >= 4 else ("good access" if v >= 2 else ("limited" if v == 1 else "none"))
        return f"{int(v)} stops  ({level})"

    if feature.startswith("hotels_"):
        level = "tourist area" if v >= 3 else ("some tourist traffic" if v >= 1 else "few tourists")
        return f"{int(v)} hotels  ({level})"

    if feature.startswith("schools_"):
        level = "family area" if v >= 3 else ("some families" if v >= 1 else "few schools")
        return f"{int(v)} schools  ({level})"

    if feature == "foot_traffic_proxy_500m":
        level = "high" if v > 8 else ("moderate" if v > 3 else "low")
        return f"{v:.1f}  ({level} foot traffic)"

    if feature == "poi_diversity_500m":
        level = "very mixed" if v >= 5 else ("varied" if v >= 3 else "limited variety")
        return f"{int(v)}/6 POI types  ({level})"

    if feature == "price_tier_success_rate":
        level = "good track record" if v >= 0.25 else ("average" if v >= 0.15 else "poor track record")
        return f"{v * 100:.0f}%  ({level} for this price tier)"

    if feature == "income_office_interaction":
        level = "strong" if v > 0.5 else ("moderate" if v > 0.15 else "weak")
        return f"{v:.2f}  ({level} wealthy-office signal)"

    if feature == "income_per_capita_proxy":
        # Back-calculate the underlying components: proxy = median_income / sqrt(population)
        # Show median income and population separately — the raw index is not human-readable
        income = (feature_values or {}).get("median_income") or (feature_values or {}).get("median_income_500m_avg")
        pop = (feature_values or {}).get("total_population") or (feature_values or {}).get("total_population_500m_avg")
        if income and pop:
            ref = _STATE_ABBREV_TO_MEDIAN.get(
                _STATE_NAME_TO_ABBREV.get(state, ""), _US_MEDIAN_INCOME
            ) if state else _US_MEDIAN_INCOME
            inc_level = "high" if income > ref * 1.35 else ("low" if income < ref * 0.65 else "avg")
            return f"tract income ${income:,.0f}  ({inc_level} for state), {int(pop):,} residents"
        level = "high" if v > 1200 else ("moderate" if v > 500 else "low")
        return f"{level} income relative to local population"

    if feature == "income_relative_to_state":
        pct = round((v - 1.0) * 100)
        direction = f"+{pct}%" if pct >= 0 else f"{pct}%"
        level = "above" if v >= 1.25 else ("below" if v <= 0.75 else "near")
        ref_label = f"{state} median" if state and state in _STATE_MEDIAN_INCOME else "state median"
        return f"{direction} {ref_label}  ({level} average)"

    if feature == "income_level_state_cat":
        labels = {0.0: "low (bottom third vs state)", 1.0: "average vs state", 2.0: "high (top third vs state)"}
        return labels.get(v)

    return None


def format_output(
    address: str,
    cuisine: str,
    price_level: int,
    probability: float,
    percentile_rank: float,
    pros: list[dict],
    cons: list[dict],
    comparables: list[dict],
    state: str | None = None,
    all_feature_values: dict | None = None,
    same_cuisine_found: bool = True,
) -> str:
    # percentile_rank = 100 - pct_below (top 10% means better than 90%)
    # Verdict: top 35% or better is "LIKELY GOOD LOCATION"
    is_good = percentile_rank <= 35
    verdict = "GOOD LOCATION" if is_good else "POOR LOCATION"
    verdict_icon = "✓" if is_good else "✗"
    price_sym = _PRICE_SYMBOLS.get(price_level, str(price_level))

    # Location score: 0–100 where 100 = best possible vs comparable restaurants
    beats_pct = min(99, round(100 - percentile_rank))
    score = beats_pct  # "beats X% of comparable restaurants" == score of X/100

    # Star rating 1–5 mapped from score
    if score >= 80:
        stars = "★★★★★"
    elif score >= 60:
        stars = "★★★★☆"
    elif score >= 40:
        stars = "★★★☆☆"
    elif score >= 20:
        stars = "★★☆☆☆"
    else:
        stars = "★☆☆☆☆"

    summary = build_summary(cuisine, price_level, score, pros, cons)
    # Word-wrap summary to 60 chars
    import textwrap
    summary_lines = textwrap.wrap(summary, width=60)

    lines = [
        "",
        "RestaurantGenie -- Location Analysis",
        "=" * 38,
        f"  Address : {address}",
        f"  Cuisine : {cuisine.title()}",
        f"  Price   : {price_sym}",
        "=" * 38,
        f"  Score   : {score}/100  {stars}",
        f"  Verdict : {verdict_icon}  {verdict}",
        "",
        f"  This location scores better than {beats_pct}% of {cuisine.title()} {price_sym}",
        f"  restaurants in comparable cities.",
        "",
        "-" * 38,
        "  SUMMARY",
        "-" * 38,
    ] + [f"  {line}" for line in summary_lines] + [
        "",
        "-" * 38,
        "  KEY FACTORS",
        "-" * 38,
        "  Positive factors:",
    ]
    fv = all_feature_values or {item["feature"]: item.get("value") for item in pros + cons}
    for p in pros:
        val_str = _format_feature_value(p["feature"], p.get("value"), state=state, feature_values=fv)
        if val_str:
            lines.append(f"    + {p['label']}")
            lines.append(f"        {val_str}")
        else:
            lines.append(f"    + {p['label']}")
    if not pros:
        lines.append("    (none identified)")
    lines += ["", "  Negative factors:"]
    for c in cons:
        val_str = _format_feature_value(c["feature"], c.get("value"), state=state, feature_values=fv)
        if val_str:
            lines.append(f"    - {c['label']}")
            lines.append(f"        {val_str}")
        else:
            lines.append(f"    - {c['label']}")
    if not cons:
        lines.append("    (none identified)")
    if same_cuisine_found:
        comp_header = f"  NEARBY {cuisine.upper()} RESTAURANTS  (live OSM)"
    else:
        comp_header = f"  NEARBY RESTAURANTS  (no {cuisine.title()} found in OSM — showing all)"
    lines += [
        "",
        "-" * 38,
        comp_header,
        "-" * 38,
    ]
    for r in comparables:
        cuisine_str = (r.get("cuisine") or "?")[:14]
        lines.append(f"  {r['name'][:30]:<30}  {cuisine_str:<14}  {r['distance_km']}km")
    if not comparables:
        lines.append(f"  No restaurants found nearby in OSM.")
    lines += ["=" * 38, ""]
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
    parquet_path: str | Path = "models/reference_scores.parquet",
    params_path: str | Path = "models/normalization_params.json",
) -> str:
    model, explainer = load_artifacts(models_dir)
    # Slim reference: only name/lat/lon/city/cuisine/price_level/rating/predicted_probability.
    # No stale Yelp feature columns are loaded or used at inference time.
    reference_df = pd.read_parquet(parquet_path)
    lat, lon, city, state = geocode_address(address)
    # If Nominatim didn't return a city component, fall back to nearest entry in reference set.
    if city is None:
        city = _infer_city(lat, lon, reference_df)

    feature_vector, feature_names = assemble_feature_vector(
        lat, lon, cuisine, price_level, params_path, city=city, state=state
    )

    probability = float(model.predict_proba([feature_vector])[0, 1])

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
    comparables, same_cuisine_found = find_comparable_restaurants(lat, lon, cuisine)

    return format_output(address, cuisine, price_level, probability,
                         percentile_rank, pros, cons, comparables,
                         state=state, all_feature_values=feature_values,
                         same_cuisine_found=same_cuisine_found)


_VALID_CUISINES = {
    "american", "burgers", "chinese", "french", "greek", "indian", "italian",
    "japanese", "korean", "mediterranean", "mexican", "other", "pizza",
    "sandwiches", "seafood", "steakhouses", "thai", "vietnamese",
}


def main():
    import difflib

    parser = argparse.ArgumentParser(description="RestaurantGenie -- location success predictor")
    parser.add_argument("--address", required=True, help='Full US address e.g. "123 Main St, Austin TX"')
    parser.add_argument("--cuisine", required=True, help='Cuisine type e.g. italian, mexican, pizza')
    parser.add_argument("--price", type=int, choices=[1, 2, 3, 4], required=True,
                        help="Price level: 1=$ 2=$$ 3=$$$ 4=$$$$")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--parquet", default="models/reference_scores.parquet")
    parser.add_argument("--params", default="models/normalization_params.json")
    args = parser.parse_args()

    cuisine = args.cuisine.lower().strip()
    if cuisine not in _VALID_CUISINES:
        matches = difflib.get_close_matches(cuisine, _VALID_CUISINES, n=3, cutoff=0.6)
        msg = f"Unknown cuisine '{cuisine}'."
        if matches:
            msg += f" Did you mean: {', '.join(matches)}?"
        msg += f"\nValid cuisines: {', '.join(sorted(_VALID_CUISINES))}"
        print(msg)
        sys.exit(1)

    output = run_prediction(
        address=args.address,
        cuisine=cuisine,
        price_level=args.price,
        models_dir=args.models_dir,
        parquet_path=args.parquet,
        params_path=args.params,
    )
    print(output)


if __name__ == "__main__":
    main()
