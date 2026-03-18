"""Backfill Census demographics for all restaurants in the parquet.

Strategy:
1. Group restaurants into ~2km cells (0.02° rounding) → ~3,300 unique locations
2. Geocode each cell centroid once → (state, county, tract)
3. Fetch ACS data once per unique census tract
4. Apply results back to all restaurants in each cell
"""

import time
import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CENSUS_ACS_URL = "https://api.census.gov/data/2022/acs/acs5"
CENSUS_GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
CENSUS_API_KEY = ""  # set CENSUS_API_KEY env var for higher rate limits

PARQUET_PATH = Path("data/processed/restaurant_features.parquet")
RESOLUTION = 0.02  # degrees (~2km)


def geocode_to_tract(lat: float, lon: float) -> tuple[str, str, str] | None:
    """Return (state, county, tract) or None on failure."""
    try:
        resp = requests.get(
            CENSUS_GEOCODER_URL,
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
            return None
        t = tracts[0]
        return t["STATE"], t["COUNTY"], t["TRACT"]
    except Exception:
        return None


def fetch_acs(state: str, county: str, tract: str) -> dict:
    """Fetch median_income, total_population, median_age for one census tract."""
    params = {
        "get": "B19013_001E,B01003_001E,B01002_001E",
        "for": f"tract:{tract}",
        "in": f"state:{state}+county:{county}",
    }
    if CENSUS_API_KEY:
        params["key"] = CENSUS_API_KEY
    try:
        resp = requests.get(CENSUS_ACS_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        row = dict(zip(data[0], data[1]))
        return {
            "median_income": float(row["B19013_001E"]) if row.get("B19013_001E") else None,
            "total_population": float(row["B01003_001E"]) if row.get("B01003_001E") else None,
            "median_age": float(row["B01002_001E"]) if row.get("B01002_001E") else None,
        }
    except Exception:
        return {"median_income": None, "total_population": None, "median_age": None}


def main():
    print("Loading parquet...")
    df = pd.read_parquet(PARQUET_PATH)

    # Round to grid cells
    lat_r = (df["lat"] / RESOLUTION).round() * RESOLUTION
    lon_r = (df["lon"] / RESOLUTION).round() * RESOLUTION
    df["_cell"] = list(zip(lat_r.round(6), lon_r.round(6)))

    unique_cells = list({(round(la / RESOLUTION) * RESOLUTION,
                          round(lo / RESOLUTION) * RESOLUTION)
                         for la, lo in zip(df["lat"], df["lon"])})
    print(f"Geocoding {len(unique_cells)} unique ~2km cells...")

    # Step 1: geocode cells → tracts (parallel, 8 workers)
    cell_to_tract: dict[tuple, tuple | None] = {}

    def _geocode(cell):
        lat, lon = cell
        result = geocode_to_tract(lat, lon)
        time.sleep(0.05)  # ~20 req/s to be polite
        return cell, result

    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_geocode, c): c for c in unique_cells}
        for fut in as_completed(futures):
            cell, tract = fut.result()
            cell_to_tract[cell] = tract
            done += 1
            if done % 200 == 0:
                print(f"  geocoded {done}/{len(unique_cells)}")

    print("Geocoding done.")

    # Step 2: fetch ACS for unique tracts
    unique_tracts = {v for v in cell_to_tract.values() if v is not None}
    print(f"Fetching ACS data for {len(unique_tracts)} unique census tracts...")

    tract_to_demo: dict[tuple, dict] = {}
    for i, tract_key in enumerate(unique_tracts):
        state, county, tract = tract_key
        tract_to_demo[tract_key] = fetch_acs(state, county, tract)
        time.sleep(0.05)
        if (i + 1) % 200 == 0:
            print(f"  fetched {i+1}/{len(unique_tracts)} tracts")

    print("ACS fetch done.")

    # Step 3: map back to restaurants
    cell_to_demo: dict[tuple, dict] = {
        cell: tract_to_demo.get(tract, {"median_income": None, "total_population": None, "median_age": None})
        for cell, tract in cell_to_tract.items()
    }

    null_demo = {"median_income": None, "total_population": None, "median_age": None}
    demos = df["_cell"].map(lambda c: cell_to_demo.get(c, null_demo))

    df["median_income"] = [d["median_income"] for d in demos]
    df["total_population"] = [d["total_population"] for d in demos]
    df["median_age"] = [d["median_age"] for d in demos]
    df.drop(columns=["_cell"], inplace=True)

    filled = df["median_income"].notna().sum()
    print(f"Filled {filled}/{len(df)} rows ({filled/len(df)*100:.1f}%) with Census data")

    # Recompute Census-derived features
    df["income_office_interaction"] = (
        df["median_income"].fillna(0) / 100000.0 * df["offices_500m"].fillna(0)
    )
    df["income_per_capita_proxy"] = (
        df["median_income"].fillna(0) / np.sqrt(df["total_population"].fillna(1).clip(lower=1))
    )
    for radius in [500, 1000]:
        df[f"median_income_{radius}m_avg"] = df["median_income"].fillna(0)
        df[f"total_population_{radius}m_avg"] = df["total_population"].fillna(0)
        df[f"median_age_{radius}m_avg"] = df["median_age"].fillna(0)
    df["income_variance_1000m"] = 0.0  # single-tract; variance requires multi-restaurant BallTree

    # Add income × price interaction explicitly
    df["median_income_x_price"] = df["median_income"].fillna(0) / 100000.0 * df["price_level"].fillna(1)

    df.to_parquet(PARQUET_PATH, index=False)
    print(f"Saved updated parquet to {PARQUET_PATH}")

    # Summary stats
    print("\nCensus columns summary:")
    for c in ["median_income", "total_population", "median_age"]:
        print(f"  {c}: mean={df[c].mean():.0f}, null={df[c].isna().sum()}")


if __name__ == "__main__":
    main()
