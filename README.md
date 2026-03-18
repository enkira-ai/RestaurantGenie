# RestaurantGenie

Predicts whether a US address is a good location for a new restaurant, given cuisine type and price level. Returns a success probability, verdict, key drivers (via SHAP), and a list of comparable nearby restaurants.

---

## Installation

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/enkira-ai/RestaurantGenie.git
cd RestaurantGenie
uv sync
```

### Optional: Census API key

Demographic lookups work without a key but are rate-limited. For faster dataset building, get a free key at https://api.census.gov/data/key_signup.html and export it:

```bash
export CENSUS_API_KEY=your_key_here
```

---

## Quick start (model already trained)

The trained model is included in `models/`. Skip to [Predicting](#predicting) if you just want to run predictions.

---

## Building the dataset from scratch

### 1. Download the Yelp Open Dataset

Register at https://www.yelp.com/dataset (free) and download the academic dataset. Place these two files in `data/raw/`:

- `yelp_academic_dataset_business.json`
- `yelp_academic_dataset_review.json`

### 2. Extract review statistics

```bash
uv run python -m src.review_stats
```

Streams the 6.9M-review file to compute per-business review velocity and recency.
Output: `data/processed/review_stats.parquet`

### 3. Build the feature dataset

```bash
uv run python -m src.build_dataset
```

For each restaurant this:
- Queries OSM Overpass for nearby POIs (restaurants, bars, offices, hotels, transit, schools) at 250m / 500m / 1km radii
- Looks up US Census ACS demographics (median income, population, median age) for the census tract
- Computes derived features (saturation ratios, foot traffic proxy, demand density, POI diversity)
- Computes the success label (see [Success criteria](#success-criteria))

Output: `data/processed/restaurant_features.parquet` (~52k restaurants, ~80 columns)
Runtime: 30–60 minutes (OSM queries are the bottleneck)

### 4. Backfill Census demographics

```bash
uv run python scripts/backfill_census.py
```

Geocodes all restaurants to census tracts and fills in income / population / age. Groups by ~2km cells so only ~3,300 geocoder calls are needed for 52k restaurants.
Runtime: ~15 minutes

### 5. Train the model

```bash
uv run python -m src.train_model
```

Output: `models/model.pkl`, `models/shap_explainer.pkl`, `models/normalization_params.json`, `models/performance_report.txt`, `models/reference_scores.parquet`
Runtime: 5–15 minutes

`reference_scores.parquet` is a slim 5 MB file (8 columns: name, lat, lon, city, cuisine, price_level, rating, predicted_probability) used at inference time for percentile ranking and comparable restaurant display. It contains no stale feature columns — all feature computation at inference time uses live OSM + Census data.

---

## Predicting

### Shell script (recommended)

```bash
./predict.sh "<address>" <cuisine> <price>
```

**Arguments**

| Argument | Values |
|---|---|
| `address` | Any full US address in quotes, e.g. `"1600 N Broad St, Philadelphia PA"` |
| `cuisine` | See table below |
| `price` | See table below |

**Cuisine values**

| Value | Description |
|---|---|
| `american` | American / comfort food |
| `burgers` | Burger joints |
| `chinese` | Chinese |
| `french` | French |
| `greek` | Greek |
| `indian` | Indian |
| `italian` | Italian |
| `japanese` | Japanese (including sushi) |
| `korean` | Korean |
| `mediterranean` | Mediterranean |
| `mexican` | Mexican / Tex-Mex |
| `pizza` | Pizza |
| `sandwiches` | Sandwiches / delis |
| `seafood` | Seafood |
| `steakhouses` | Steakhouses |
| `thai` | Thai |
| `vietnamese` | Vietnamese |
| `other` | Any cuisine not listed above |

**Price values**

| Value | Symbol | Description |
|---|---|---|
| `1` | $ | Budget — most items under $15 |
| `2` | $$ | Mid-range — most items $15–$40 |
| `3` | $$$ | Upscale — most items $40–$80 |
| `4` | $$$$ | Fine dining — most items over $80 |

**Examples**

```bash
./predict.sh "1600 N Broad St, Philadelphia PA" american 2
./predict.sh "900 N Michigan Ave, Chicago IL" italian 3
./predict.sh "742 Evergreen Terrace, Springfield IL" pizza 1
```

### Python module

```bash
uv run python -W ignore -m src.predict \
  --address "1600 N Broad St, Philadelphia PA" \
  --cuisine american \
  --price 2
```

Optional flags: `--models-dir` (default: `models`), `--parquet` (default: `models/reference_scores.parquet`), `--params` (default: `models/normalization_params.json`)

### Example output

```
RestaurantGenie -- Location Analysis
--------------------------------------
Address:   1600 N Broad St, Philadelphia PA 19121
Cuisine:   American
Price:     $$

Success probability:  0.44  (better than 99% of comparable restaurants)
Verdict:              LIKELY GOOD LOCATION

KEY FACTORS — POSITIVE
  + price tier success rate          0.51 (51% success rate for $$ in this city)
  + restaurants within 250m          3 restaurants (low competition)
  + schools within 1km               4 schools (strong family demand)

KEY FACTORS — NEGATIVE
  - restaurant density (500m)        28 restaurants (high competition)
  - cuisine type fit                 encoded=0 (american)
  - transit stops within 250m        0 stops (low foot traffic)

--------------------------------------
  NEARBY AMERICAN RESTAURANTS  (live OSM)
--------------------------------------
  Masters Bar & Restaurant       american   0.1km
  honeygrow                      american   0.1km
  Draught Horse Pub & Grill      american   0.2km
```

**Interpreting the output**

- **Success probability** — model's estimated probability this location outperforms comparable restaurants. Higher is better.
- **Better than X%** — how this location ranks against restaurants of the same cuisine and price tier in the reference dataset.
- **Verdict** — LIKELY GOOD if the location scores in the top 35% of comparable restaurants.
- **PROS / CONS** — all SHAP drivers pushing the prediction up or down, with quantitative values and context.
- **Comparable restaurants** — real restaurants of the same cuisine queried live from OpenStreetMap within ~5km. Sorted by cuisine specificity (dedicated restaurants first) then distance.

---

## Success criteria

The model predicts a **proxy success score** since restaurant revenue is not publicly available. A restaurant is labelled successful if it ranks in the **top 25% of its peer group** on a composite score:

```
success_score = 0.50 × z_rating + 0.35 × z_velocity + 0.15 × z_activity
```

Where, within each (city, cuisine, price tier) peer group:

| Component | Definition |
|---|---|
| `z_rating` | Z-score of Bayesian-smoothed rating (smoothed toward peer mean, m=25 reviews) |
| `z_velocity` | Z-score of review velocity = total reviews ÷ restaurant age in years |
| `z_activity` | Z-score of reviews received in the last 12 months |

**Filters applied before labelling:**
- Minimum 15 reviews (restaurants with fewer are excluded from training)
- Peer group must have ≥ 10 members; falls back to (city, cuisine) if smaller

---

## Model details

**Algorithm:** LightGBM binary classifier with Platt calibration
**Training data:** ~52k US restaurants from the [Yelp Open Dataset](https://www.yelp.com/dataset) across 920 cities
**Positive rate:** ~18.8% (top 25% within peer group, after 15-review minimum filter)
**Validation:** Geographic cross-validation — entire cities held out, never split by row

| Metric | Value |
|---|---|
| Geographic CV ROC-AUC (5-fold) | 0.685 |
| Held-out city ROC-AUC | 0.683 |
| Calibrated Brier score | 0.135 |

### Surviving features (19 of 50 candidates)

All features are derivable for free at inference time from OSM Overpass + US Census ACS + a lookup table stored in `models/normalization_params.json`. No paid API is required.

| Feature | Source | What it captures |
|---|---|---|
| `restaurants_250m` | OSM | Immediate competition density |
| `bars_500m` | OSM | Nightlife / foot traffic nearby |
| `offices_1000m` | OSM | Daytime lunch demand |
| `schools_1000m` | OSM | Family / community demand |
| `median_income` | Census ACS | Neighbourhood wealth |
| `total_population` | Census ACS | Catchment size |
| `median_age` | Census ACS | Demographic profile |
| `cuisine_encoded` | Input | Cuisine type fit |
| `price_level` | Input | Price tier |
| `foot_traffic_proxy_500m` | OSM derived | Offices + transit + hotels weighted |
| `income_office_interaction` | Census × OSM | Wealthy daytime workers nearby |
| `income_per_capita_proxy` | Census derived | Income normalised for city size |
| `poi_diversity_500m` | OSM derived | Neighbourhood activity variety |
| `total_population_500m_avg` | Census derived | Local density |
| `price_tier_success_rate` | Training lookup | Historical success rate for this price tier in this city |
| `median_income_x_price` | Census × Input | Income × price level interaction (expensive restaurants need wealthier areas) |
| `income_relative_to_state` | Census derived | Household income as ratio of state median (normalises across states) |
| `income_level_state_cat` | Census derived | Categorical: below (< 0.75×), near (0.75–1.25×), or above (> 1.25×) state median |
| `state_encoded` | Input | State identifier (restaurant economics vary by state) |

### Hyperparameters

Found by Optuna (50 trials, geographic CV objective):

```
num_leaves: 126      max_depth: 5        min_child_samples: 133
lambda_l1: 2.58      lambda_l2: 4.19     feature_fraction: 0.748
bagging_fraction: 0.572                  learning_rate: 0.107
n_estimators: 56
```

---

## Running tests

```bash
uv run pytest tests/ -v
```

34 tests covering feature engineering, success label computation, model pipeline, and prediction.

---

## Data sources

| Source | Used for | Cost |
|---|---|---|
| [Yelp Open Dataset](https://www.yelp.com/dataset) | Training labels (ratings, review counts, business metadata) | Free (registration required) |
| [OSM Overpass API](https://overpass-api.de) | POI counts at inference time | Free |
| [US Census ACS](https://www.census.gov/data/developers/data-sets/acs-5year.html) | Demographics at inference time | Free |
| [Census Geocoder](https://geocoding.geo.census.gov) | Coordinate → census tract lookup | Free |
