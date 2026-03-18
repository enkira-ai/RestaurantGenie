"""Stream yelp_academic_dataset_review.json and compute per-business review stats."""
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd


def extract_review_stats(
    review_jsonl_path: str | Path,
    output_path: str | Path,
    cutoff_days: int = 365,
) -> pd.DataFrame:
    """Stream review file and compute per-business stats.

    Returns DataFrame with columns:
        business_id, first_review_date, last_review_date,
        review_count_total, reviews_last_12m
    """
    stats: dict[str, dict] = {}

    with open(review_jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            bid = r.get("business_id")
            date_str = r.get("date", "")
            if not bid or not date_str:
                continue
            try:
                dt = datetime.fromisoformat(date_str[:10])
            except ValueError:
                continue

            if bid not in stats:
                stats[bid] = {"first": dt, "last": dt, "total": 0, "recent": 0}
            else:
                if dt < stats[bid]["first"]:
                    stats[bid]["first"] = dt
                if dt > stats[bid]["last"]:
                    stats[bid]["last"] = dt
            stats[bid]["total"] += 1

            if i % 500_000 == 0:
                print(f"  {i:,} reviews processed...")

    # Determine dataset cutoff date (latest review date across all businesses)
    if not stats:
        return pd.DataFrame()

    global_max_date = max(v["last"] for v in stats.values())
    cutoff = global_max_date - timedelta(days=cutoff_days)
    print(f"  Global max date: {global_max_date.date()}, cutoff for 'recent': {cutoff.date()}")

    # Second pass: count recent reviews
    # We already counted total; now re-stream to count recent
    print("  Second pass: counting recent reviews...")
    recent: dict[str, int] = {bid: 0 for bid in stats}
    with open(review_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            bid = r.get("business_id")
            date_str = r.get("date", "")
            if not bid or not date_str or bid not in recent:
                continue
            try:
                dt = datetime.fromisoformat(date_str[:10])
            except ValueError:
                continue
            if dt >= cutoff:
                recent[bid] += 1

    rows = []
    for bid, s in stats.items():
        rows.append({
            "business_id": bid,
            "first_review_date": s["first"],
            "last_review_date": s["last"],
            "review_count_total": s["total"],
            "reviews_last_12m": recent[bid],
        })

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  Saved {len(df):,} business review stats to {output_path}")
    return df


if __name__ == "__main__":
    extract_review_stats(
        review_jsonl_path="data/raw/yelp_academic_dataset_review.json",
        output_path="data/processed/review_stats.parquet",
    )
