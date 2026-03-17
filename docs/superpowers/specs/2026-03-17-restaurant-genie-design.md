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
- List of comparable restaurants nearby (same cuisine + price level, sourced from `restaurant_features.parquet`)

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
  ├── Define is_successful = top 30% within (cuisine, city) group
  ├── Train LightGBM binary classifier wrapped in CalibratedClassifierCV
  ├── Geographic cross-validation (80/20 city split)
  └── Save models/model.pkl + models/shap_explainer.pkl

Stage 3: predict.py            (use anytime)
  ├── Input: address, cuisine, price_level
  ├── Geocode via Nominatim (free, no API key); exit with clear error if geocoding fails
  ├── Fetch live neighborhood features from OSM Overpass + Census API
  ├── Build full feature vector (neighborhood features + cuisine + price_level)
  ├── Run model → calibrated probability score
  ├── Compute percentile rank against restaurant_features.parquet (loaded at predict time)
  └── Output: score + percentile + verdict + pros/cons + comparable restaurants
```

---

## Data Sources

| Source | What it provides | Access |
|--------|-----------------|--------|
| Yelp Open Dataset | name, city, lat/lon, cuisine, rating, review_count, is_open | Manual download from yelp.com/dataset (free, ~1GB JSON) |
| OSM Overpass API | nearby bars, hotels, offices, transit stops, schools | HTTP requests (free, no key); 1–2 second sleep between requests |
| US Census ACS 5-year | median household income, population density, age distribution | Census API (free; register for a free API key at api.census.gov to avoid the ~500 req/day unauthenticated limit) |

The Yelp Open Dataset must be downloaded manually and placed in `data/raw/` before running Stage 1.

**Census data access — two-step process:**
1. `censusgeocode` library: converts lat/lon → census tract FIPS code
2. Direct `requests` call to `api.census.gov/data/2022/acs/acs5` with the FIPS code: fetches `B19013_001E` (median income), `B01003_001E` (population), and `B09021_001E` (age 25–44 share)

`censusgeocode` alone does not fetch ACS variables — it only returns the FIPS identifier.

**OSM rate limiting:** Stage 1 queries POI counts for each restaurant via the Overpass API. To avoid being blocked, the implementation must either:
- Batch restaurants into bounding-box queries grouped by city/region, or
- Sleep 1–2 seconds between per-restaurant queries

Batching by city is strongly preferred and reduces total request count from ~150k to ~hundreds.

---

## Features

### Feature vector construction

`features.py` exposes:

```python
def generate_neighborhood_features(lat: float, lon: float) -> dict:
    """Returns POI counts and demographic features for the given location."""
    ...
```

The caller (both `build_dataset.py` and `predict.py`) appends `cuisine` and `price_level` to the returned dict before constructing the final feature vector:

```python
features = generate_neighborhood_features(lat, lon)
features["cuisine_encoded"] = encode_cuisine(cuisine)
features["price_level"] = price_level
```

This keeps `generate_neighborhood_features` focused on spatial/demographic signals while ensuring cuisine and price level enter the feature vector identically at train and predict time.

### Input features (available at any US address at prediction time)

**OSM POI counts** (computed at 250m, 500m, and 1000m radii):
- `restaurants_Xm` — total restaurant competition
- `restaurants_same_cuisine_Xm` — direct cuisine competition (sparse; ~40% OSM coverage, handled as nullable by LightGBM)
- `bars_Xm` — nightlife density
- `offices_Xm` — daytime foot traffic proxy
- `hotels_Xm` — tourist demand
- `transit_stops_Xm` — accessibility
- `schools_Xm` — family demand

**Census demographics** (attached by census tract via lat/lon → FIPS → ACS API):
- `median_income`
- `population_density`
- `pct_age_25_44` — prime dining-out demographic

**User-provided (appended by caller after `generate_neighborhood_features`):**
- `cuisine_encoded` — label-encoded integer; trained on training-set cuisines and saved in `models/normalization_params.json`; unseen cuisines at prediction time map to a reserved `other` integer (not an error)
- `price_level` — integer 1–4

### Target variable (training only, never used as a feature)

```
normalized_rating    = (rating - 1.0) / 4.0                    # scale 1–5 → 0–1, dataset-wide
normalized_log_reviews = log1p(review_count) / p95_log_reviews  # divide by 95th-percentile value
                                                                 # so the term stays in ~[0,1]

success_score = 0.4 * normalized_rating
              + 0.4 * normalized_log_reviews
              + 0.2 * is_open

is_successful = success_score > 70th percentile within (cuisine, city) group
```

`p95_log_reviews` is the 95th-percentile value of `log1p(review_count)` across the full dataset, computed once during Stage 1 and stored as a scalar in `models/normalization_params.json` so Stage 2 and `predict.py` use the same value.

The grouping key is the raw `cuisine` field and the raw `city` field from the Yelp dataset — no additional bucketing. Groups with fewer than 10 restaurants are merged into an `other` bucket to avoid unstable percentile estimates.

"Top 30%" and "above the 70th percentile" are equivalent phrasings of the same threshold.

---

## Percentile Rank at Prediction Time

During Stage 2, after training, `train_model.py` scores all rows in `restaurant_features.parquet` using the calibrated model and writes the result back as a new column `predicted_probability`, saving the enriched file in place. This means `restaurant_features.parquet` permanently contains one pre-computed probability per row after Stage 2 runs.

At prediction time, `predict.py` loads `restaurant_features.parquet`, filters to the same `(cuisine, city)` group as the input, and computes the percentile rank of the new location's predicted probability against those pre-computed values. If the group has fewer than 5 rows, the percentile is computed against all rows regardless of group to avoid unreliable estimates.

Runtime artifacts required by `predict.py`:
- `models/model.pkl` — calibrated LightGBM model
- `models/shap_explainer.pkl` — SHAP TreeExplainer on the base LightGBM estimator (see Model section)
- `models/normalization_params.json` — `p95_log_reviews` scalar and cuisine label map
- `data/processed/restaurant_features.parquet` — enriched with `predicted_probability` column

---

## Comparable Restaurants Output

The "comparable restaurants nearby" list is sourced from `restaurant_features.parquet` (Yelp data), not from OSM. At prediction time, `predict.py` filters `restaurant_features.parquet` for rows where:
- `cuisine` matches the input cuisine (or is `None` if no match exists)
- `price_level` is within ±1 of the input price level
- haversine distance from the input lat/lon is ≤ 5km

The top 5 closest matches are shown with name, rating, price level, and distance.

---

## Model

- **Algorithm:** LightGBM binary classifier

### Data splits

All splits are done before any model fitting begins:

```
All cities
├── test_cities (20%, geographic split)    — held out until final evaluation only;
│                                            never used for training, feature selection,
│                                            hyperparameter search, or calibration
└── train_cities (80%, geographic split)
    ├── calibration_set (20% of train-city restaurants, random)
    │                   — set aside before any model fitting;
    │                     used ONLY to fit Platt scaling
    └── train_search_set (80% of train-city restaurants)
                        — used for feature selection, hyperparameter search,
                          and final model refit
```

### Feature selection

Performed entirely within `train_search_set` using an internal 5-fold CV to avoid overfitting the feature set to a single split:

1. **SHAP-based elimination:** Compute mean absolute SHAP values across all 5 CV folds of a default LightGBM trained on `train_search_set`. Drop any feature whose mean |SHAP| across folds is below 1% of the top feature's value.
2. **Permutation importance check:** For each of the 5 folds, compute `permutation_importance` on the fold's held-out data. Drop any feature whose permutation importance is negative on average across folds.

The surviving feature set is fixed before the hyperparameter search begins.

### Hyperparameter search

Use `optuna` to search 50 trials over the following LightGBM parameters, evaluated by 5-fold geographic CV ROC-AUC on `train_search_set`:

| Parameter | Search range | Purpose |
|-----------|-------------|---------|
| `num_leaves` | 15 – 127 | Controls tree complexity |
| `max_depth` | 3 – 10 | Limits depth; key overfitting control |
| `min_child_samples` | 20 – 200 | Min samples per leaf; prevents small splits |
| `lambda_l1` | 0.0 – 5.0 | L1 regularization |
| `lambda_l2` | 0.0 – 5.0 | L2 regularization |
| `feature_fraction` | 0.5 – 1.0 | Column subsampling per tree |
| `bagging_fraction` | 0.5 – 1.0 | Row subsampling per tree |
| `learning_rate` | 0.01 – 0.2 | Step size; use early stopping per fold to set `n_estimators` |

### Model selection and refit

> The combination of (surviving features + best hyperparameters) = the selected model.

After the best hyperparameters are identified, **refit the base LightGBM on all of `train_search_set`** using those hyperparameters and the selected features. This is `base_lgbm`. Refitting on all available train data (rather than keeping one CV fold's model) ensures the production model uses the full training signal.

`n_estimators` for the refit is set to `round(mean_cv_n_estimators * 1.25)`, where `mean_cv_n_estimators` is the average number of trees selected by early stopping across the 5 hyperparameter-search CV folds for the winning trial. The 1.25 multiplier accounts for training on ~25% more data in the refit. Early stopping is **disabled** for the refit itself — `calibration_set` must not be used as a validation set here.

### Calibration

```python
calibrated_model = CalibratedClassifierCV(base_lgbm, cv='prefit', method='sigmoid')
calibrated_model.fit(calibration_set_X, calibration_set_y)
```

`cv='prefit'` means `base_lgbm` is already fitted; only the Platt sigmoid layer is fitted on `calibration_set`. Because `calibration_set` is never used for training or evaluation, there is no leakage.

`calibrated_model` is saved as `models/model.pkl`.

### Explainability

`base_lgbm` (the pre-fitted LightGBM) is passed directly to `shap.TreeExplainer(base_lgbm)`. With `cv='prefit'`, `calibrated_model.estimator` is `base_lgbm`. The TreeExplainer object is saved as `models/shap_explainer.pkl`. Top positive SHAP features → Pros; top negative → Cons.

### Reported performance

`train_model.py` prints and saves `models/performance_report.txt`. Test-city evaluation is the first and only time `test_cities` data is touched:

```
Feature selection: 18 of 23 features retained
Best hyperparameters: {num_leaves: 47, max_depth: 6, ...}
Geographic CV ROC-AUC (train_search_set, 5-fold): 0.72 ± 0.04
Test-city ROC-AUC (held-out, uncalibrated):       0.69
Test-city Brier score (calibrated model):          0.21
```

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

**Geocoding failure:** If Nominatim cannot resolve the input address, `predict.py` prints a clear error and exits:
```
Error: Could not geocode address "...". Try a more specific address including city and state.
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
│   ├── shap_explainer.pkl
│   ├── normalization_params.json   # p95_log_reviews + cuisine label map
│   └── performance_report.txt      # CV score, test-city score, calibration
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
optuna              # hyperparameter search (RandomizedSearchCV as fallback)
shap
pandas
numpy
scikit-learn        # CalibratedClassifierCV, BallTree, geographic CV
requests            # OSM Overpass + Census ACS API calls
censusgeocode       # lat/lon → census tract FIPS code
pyarrow             # parquet read/write
geopy               # Nominatim geocoding at predict time
tqdm                # progress bars during dataset build
```

No database required. The processed dataset fits comfortably in memory as a parquet file.

---

## Known Limitations & Risks

| Risk | Mitigation |
|------|-----------|
| Yelp dataset is ~2019-era snapshot | Acceptable for prototyping; success signals are relative, not time-sensitive |
| OSM cuisine tags sparse (~40% coverage) | LightGBM handles nulls natively; feature kept but flagged as noisy |
| Survivorship bias (closed restaurants underrepresented) | `is_open` field in Yelp partially captures closures; acknowledged limitation |
| City-specific behavior | Within-group success labeling (per cuisine + city) reduces cross-city bias |
| Review manipulation | Ratings are one signal among many; review_count partially corrects for this |
| Census API rate limit (500 req/day unauthenticated) | Register for free Census API key; configure in environment variable `CENSUS_API_KEY` |

---

## Scope Boundaries

- US addresses only
- Open data only (no Google Places API, no Yelp Fusion API)
- Command-line interface only (no web UI, no hosting)
- No real-time foot traffic data (SafeGraph/Placer.ai are paid; proxied by POI density)
