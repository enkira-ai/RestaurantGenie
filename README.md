# RestaurantGenie

Predicts whether a US address is a good location for a new restaurant.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the [Yelp Open Dataset](https://www.yelp.com/dataset) (free, requires registration).
   Place `yelp_academic_dataset_business.json` in `data/raw/`.

3. (Optional) Get a free Census API key at https://api.census.gov/data/key_signup.html.
   Set it as an environment variable:
   ```bash
   export CENSUS_API_KEY=your_key_here
   ```

## Stage 1: Build Dataset (~30–60 min)

```bash
python src/build_dataset.py
```

Produces `data/processed/restaurant_features.parquet` (~150k rows) and
`models/normalization_params.json`.

## Stage 2: Train Model (~5–15 min)

```bash
python src/train_model.py
```

Produces `models/model.pkl`, `models/shap_explainer.pkl`, and
`models/performance_report.txt` (ROC-AUC, Brier score).

## Stage 3: Predict

```bash
python src/predict.py \
  --address "123 Main St, Austin TX" \
  --cuisine italian \
  --price 2
```

Optional flags: `--models-dir` (default: `models`), `--parquet` (default: `data/processed/restaurant_features.parquet`), `--params` (default: `models/normalization_params.json`).

### Example output

```
RestaurantGenie -- Location Analysis
--------------------------------------
Address:   123 Main St, Austin TX
Cuisine:   Italian
Price:     $$

Success probability:  0.71  (top 29% of comparable restaurants)
Verdict:              LIKELY GOOD LOCATION ✓

PROS
  + neighborhood median income
  + office density (500m)

CONS
  - restaurant density (500m)

Comparable restaurants nearby:
  Olive & Vine                   (italian, $$)  4.2★  2.1km
```

## Run Tests

```bash
pytest tests/ -v
```
