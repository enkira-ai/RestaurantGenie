import sys
sys.path.insert(0, "/Users/erhanbilal/Work/Projects/RestaurantGenie")
import pandas as pd
import numpy as np
from src.build_dataset import (
    add_derived_features,
    add_spatial_census_features,
    add_price_tier_features,
    add_yelp_spatial_features,
    compute_success_labels,
)

# Load the parquet that has OSM+Census features but old labels
df = pd.read_parquet("data/processed/restaurant_features.parquet")
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# Drop old label/score columns so they get recomputed cleanly
drop = ["success_score", "is_successful", "predicted_probability",
        "price_tier_success_rate", "price_tier_count", "price_tier_count_log",
        "smoothed_rating", "review_velocity", "recent_activity", "restaurant_age_years"]
df = df.drop(columns=[c for c in drop if c in df.columns])

# Get p95_log_reviews from normalization_params.json
import json
with open("models/normalization_params.json") as f:
    p95 = json.load(f)["p95_log_reviews"]

# Recompute success labels with new methodology
df = compute_success_labels(df, p95_log_reviews=p95,
                             review_stats_path="data/processed/review_stats.parquet")
print(f"New label distribution: {df['is_successful'].value_counts().to_dict()}")

# Recompute price_tier features (depend on is_successful)
df = add_price_tier_features(df)

# Add Yelp spatial features (in case they need recomputing)
if "avg_price_1km" not in df.columns:
    df = add_yelp_spatial_features(df)

print(f"Final: {len(df)} rows, {len(df.columns)} columns")
df.to_parquet("data/processed/restaurant_features.parquet", index=False)
print("Saved.")
