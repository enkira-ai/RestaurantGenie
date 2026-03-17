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
