# RestaurantGenie — Design Spec
_Date: 2026-03-17_

## Overview

RestaurantGenie is a command-line tool that tells an entrepreneur whether a given US address is a good location for a new restaurant, given the cuisine type and price level. It outputs a probability score, a binary verdict, and a list of pros and cons explaining the key drivers.

---

## System Goal

**Input:**
- Street address (US only)
- Cuisine type (e.g., italian, mexican, pizza)
- Price level (1–4, matching Yelp's $ to $$$$)

**Output:**
- Success probability (0–1) and percentile rank vs. comparable restaurants
- Binary verdict: LIKELY GOOD LOCATION / UNLIKELY GOOD LOCATION
- Pros: top positive location factors (SHAP-driven)
- Cons: top negative location factors (SHAP-driven)
- List of comparable restaurants nearby (same cuisine + price level, from OSM)

---

## Architecture

Three sequential stages:

```
Stage 1: build_dataset.py     (run once, ~30–60 min)
  ├── Yelp Open Dataset        → restaurant base (ratings, reviews, status)
  ├── OSM Overpass API         → POI features (bars, hotels, offices, transit)
  └── US Census ACS 5-year API → demographics (income, population density, age)
      ↓
  data/processed/restaurant_features.parquet  (~150k rows)

Stage 2: train_model.py        (run once, ~5 min)
  ├── Compute success_score from rating + review_count + is_open
  ├── Define is_successful = top 30% within same cuisine + city bucket
  ├── Train LightGBM binary classifier
  ├── Geographic cross-validation (80/20 city split)
  └── Save models/model.pkl + models/shap_explainer.pkl

Stage 3: predict.py            (use anytime)
  ├── Input: address, cuisine, price_level
  ├── Geocode via Nominatim (free, no API key)
  ├── Fetch live neighborhood features from OSM Overpass + Census API
  ├── Run model → probability score
  └── Output: score + verdict + pros/cons + comparable restaurants
```

---

## Data Sources

| Source | What it provides | Access |
|--------|-----------------|--------|
| Yelp Open Dataset | name, city, lat/lon, cuisine, rating, review_count, is_open | Manual download from yelp.com/dataset (free, ~1GB JSON) |
| OSM Overpass API | nearby bars, hotels, offices, transit stops, schools, parking | HTTP requests (free, no key) |
| US Census ACS 5-year | median household income, population density, age distribution | Census API (free, no key required for basic use) |

The Yelp Open Dataset must be downloaded manually and placed in `data/raw/` before running Stage 1.

---

## Features

### Input features (available at any US address at prediction time)

**OSM POI counts** (computed at 250m, 500m, and 1000m radii):
- `restaurants_Xm` — total restaurant competition
- `restaurants_same_cuisine_Xm` — direct cuisine competition (sparse; ~40% OSM coverage, handled as nullable)
- `bars_Xm` — nightlife density
- `offices_Xm` — daytime foot traffic proxy
- `hotels_Xm` — tourist demand
- `transit_stops_Xm` — accessibility
- `schools_Xm` — family demand

**Census demographics** (attached by census tract via lat/lon):
- `median_income`
- `population_density`
- `pct_age_25_44` — prime dining-out demographic

**User-provided:**
- `cuisine` — label-encoded
- `price_level` — integer 1–4

### Target variable (training only, never used as a feature)

```
success_score = 0.4 * normalized_rating
              + 0.4 * log1p(review_count)
              + 0.2 * is_open

is_successful = success_score > 70th percentile within (cuisine_bucket, city)
```

This within-group comparison ensures a mid-range Italian restaurant is judged against other mid-range Italian restaurants in the same city, not against fine dining in a different market.

---

## Shared Feature Module

`src/features.py` exposes a single function:

```python
def generate_neighborhood_features(lat: float, lon: float) -> dict:
    ...
```

Both `build_dataset.py` (training) and `predict.py` (inference) call this same function, guaranteeing identical feature computation at train and test time.

---

## Model

- **Algorithm:** LightGBM binary classifier
- **Validation:** Geographic cross-validation — hold out 20% of cities entirely; the model never sees any restaurant from test cities during training
- **Evaluation metrics:** ROC-AUC, calibration curve (calibration ensures probability outputs are meaningful, not just rankings)
- **Explainability:** SHAP TreeExplainer; top positive SHAP features → Pros; top negative → Cons

---

## Prediction Output Format

```
RestaurantGenie — Location Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Address:   123 Main St, Austin TX
Cuisine:   Italian
Price:     $$

Success probability:  0.71  (top 29% of comparable restaurants)
Verdict:              LIKELY GOOD LOCATION ✓

PROS
  + High office density nearby — strong lunch demand
  + Above-average median income ($87k) — supports $$ price point
  + Low direct competition — only 3 Italian restaurants within 500m

CONS
  - High overall restaurant density — saturated dining market
  - Low transit access — car-dependent area

Comparable restaurants nearby:
  Olive & Vine (Italian, $$)     4.2★  2.1km
  Trattoria Roma (Italian, $$$)  4.5★  3.8km
```

---

## Project Structure

```
RestaurantGenie/
├── data/
│   ├── raw/                    # Yelp JSON files placed here manually
│   └── processed/
│       └── restaurant_features.parquet
├── models/
│   ├── model.pkl
│   └── shap_explainer.pkl
├── src/
│   ├── build_dataset.py        # Stage 1
│   ├── train_model.py          # Stage 2
│   ├── predict.py              # Stage 3 (CLI)
│   └── features.py             # Shared feature generation
├── requirements.txt
└── README.md
```

---

## Dependencies

```
lightgbm
shap
pandas
numpy
scikit-learn
requests
censusgeocode
pyarrow
geopy
tqdm
```

No database required. The processed dataset fits comfortably in memory as a parquet file.

---

## Known Limitations & Risks

| Risk | Mitigation |
|------|-----------|
| Yelp dataset is ~2019-era snapshot | Acceptable for prototyping; success signals are relative, not time-sensitive |
| OSM cuisine tags sparse (~40% coverage) | LightGBM handles nulls natively; feature kept but flagged as noisy |
| Survivorship bias (closed restaurants underrepresented) | `is_open` field in Yelp partially captures closures; acknowledged limitation |
| City-specific behavior | Within-group success labeling (per city) reduces cross-city bias |
| Review manipulation | Ratings are one signal among many; review_count partially corrects for this |

---

## Scope Boundaries

- US addresses only
- Open data only (no Google Places API, no Yelp Fusion API)
- Command-line interface only (no web UI, no hosting)
- No real-time foot traffic data (SafeGraph/Placer.ai are paid; proxied by POI density)
