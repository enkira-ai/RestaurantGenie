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
