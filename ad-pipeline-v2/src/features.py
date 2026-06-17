"""
features.py
─────────────────────────────────────────
Stage 2: Feature Engineering
Mirrors notebook cell [10] exactly:
- Derived features: conversion_rate, spend_efficiency, ctr_cpc_ratio,
  log_impressions, cost_per_impression
- Target: high_roas = 1 if ROAS > median, else 0
- Categorical encoding: platform, campaign_type, industry, country

Run standalone:
    python src/features.py
"""

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH = Path("data/processed/clean_ads.csv")
OUTPUT_PATH = Path("data/processed/features.csv")
ENCODERS_PATH = Path("models/label_encoders.pkl")

CAT_COLS = ["platform", "campaign_type", "industry", "country"]

# Full feature set used for model training (matches notebook FEATURES list).
# CPA is included here but train.py also runs a leakage check that drops it
# (see notebook cell [15] — "Data Leakage Check").
FEATURES = [
    "impressions", "clicks", "CTR", "CPC", "ad_spend", "conversions", "CPA",
    "conversion_rate", "ctr_cpc_ratio", "log_impressions", "cost_per_impression",
    "platform_enc", "campaign_type_enc", "industry_enc", "country_enc",
]
TARGET = "high_roas"


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """conversion_rate, spend_efficiency, ctr_cpc_ratio, log_impressions, cost_per_impression."""
    df = df.copy()
    df["conversion_rate"] = df["conversions"] / df["clicks"]
    df["spend_efficiency"] = df["revenue"] / df["ad_spend"]  # ≈ ROAS, excluded from FEATURES
    df["ctr_cpc_ratio"] = df["CTR"] / (df["CPC"] + 1e-6)
    df["log_impressions"] = np.log1p(df["impressions"])
    df["cost_per_impression"] = df["ad_spend"] / df["impressions"]
    logger.info("Added derived features: conversion_rate, ctr_cpc_ratio, log_impressions, cost_per_impression")
    return df


def add_target_label(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """high_roas = 1 if ROAS > median(ROAS), else 0."""
    df = df.copy()
    roas_threshold = df["ROAS"].median()
    df["high_roas"] = (df["ROAS"] > roas_threshold).astype(int)
    logger.info(f"ROAS threshold (median): {roas_threshold:.3f}")
    logger.info(f"Class distribution:\n{df['high_roas'].value_counts()}")
    return df, roas_threshold


def encode_categoricals(df: pd.DataFrame, save_encoders: bool = True) -> pd.DataFrame:
    """Label-encode platform, campaign_type, industry, country for XGBoost compatibility."""
    df = df.copy()
    le_dict = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        le_dict[col] = le
        logger.info(f"{col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    if save_encoders:
        ENCODERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(le_dict, ENCODERS_PATH)
        logger.info(f"Saved label encoders to {ENCODERS_PATH}")

    return df


def run(path: Path = INPUT_PATH, save: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = add_derived_features(df)
    df, threshold = add_target_label(df)
    df = encode_categoricals(df)

    if save:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        logger.info(f"Saved feature table to {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    run()
