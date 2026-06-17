"""
anomaly.py
─────────────────────────────────────────
Stage 5: Anomaly Detection
Mirrors notebook cells [25], [27], [29]:
- Z-score on ROAS (|Z| > 3)
- Isolation Forest on [ROAS, CTR, CPC, ad_spend, conversions, CPA, revenue]
- High-confidence anomalies = both methods agree
- Optional: re-run excluding TikTok Ads to check if it dominates

Run standalone:
    python src/anomaly.py
    python src/anomaly.py --exclude-tiktok
"""

import argparse
import logging
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.ensemble import IsolationForest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH = Path("data/processed/clean_ads.csv")  # raw df, not the encoded feature table
OUTPUT_PATH = Path("data/processed/anomalies.csv")

ISO_FEATURES = ["ROAS", "CTR", "CPC", "ad_spend", "conversions", "CPA", "revenue"]
Z_SCORE_THRESHOLD = 3
ISO_CONTAMINATION = 0.05


def flag_zscore_anomalies(df: pd.DataFrame) -> pd.Series:
    z = stats.zscore(df["ROAS"])
    is_anomaly = (abs(z) > Z_SCORE_THRESHOLD)
    logger.info(f"Z-score anomalies: {is_anomaly.sum()} / {len(df)} ({is_anomaly.mean()*100:.2f}%)")
    return is_anomaly


def flag_isolation_forest_anomalies(df: pd.DataFrame) -> pd.Series:
    iso = IsolationForest(contamination=ISO_CONTAMINATION, random_state=42, n_estimators=100)
    preds = iso.fit_predict(df[ISO_FEATURES])
    is_anomaly = pd.Series(preds == -1, index=df.index)
    logger.info(f"Isolation Forest anomalies: {is_anomaly.sum()} / {len(df)} ({is_anomaly.mean()*100:.2f}%)")
    return is_anomaly


def run(path: Path = INPUT_PATH, exclude_tiktok: bool = False, save: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)

    if exclude_tiktok:
        before = len(df)
        df = df[df["platform"] != "TikTok Ads"].copy()
        logger.info(f"Excluded TikTok Ads: {before} -> {len(df)} rows")

    df["is_anomaly_zscore"] = flag_zscore_anomalies(df)
    df["is_anomaly_iso"] = flag_isolation_forest_anomalies(df)
    df["both_anomaly"] = df["is_anomaly_zscore"] & df["is_anomaly_iso"]

    n_both = df["both_anomaly"].sum()
    logger.info(f"High-confidence anomalies (both methods): {n_both}")

    if "platform" in df.columns and n_both > 0:
        breakdown = df[df["both_anomaly"]].groupby("platform").size().sort_values(ascending=False)
        logger.info(f"Breakdown by platform:\n{breakdown}")

    if save:
        out_path = OUTPUT_PATH if not exclude_tiktok else Path("data/processed/anomalies_no_tiktok.csv")
        result = df[df["both_anomaly"]]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        logger.info(f"Saved {len(result)} flagged campaigns to {out_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-tiktok", action="store_true")
    args = parser.parse_args()
    run(exclude_tiktok=args.exclude_tiktok)
