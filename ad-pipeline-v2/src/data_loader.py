"""
data_loader.py
─────────────────────────────────────────
Stage 1: Ingestion
Downloads the Kaggle "Global Ads Performance" dataset via kagglehub
and validates the schema. Mirrors notebook cells [5]-[6].

Run standalone:
    python src/data_loader.py
"""

import pandas as pd
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_PATH = Path("data/processed/clean_ads.csv")

KAGGLE_DATASET = "nudratabbas/global-ads-performance-google-meta-tiktok"
CSV_FILENAME = "global_ads_performance_dataset.csv"

# Actual columns from the dataset (confirmed from notebook)
REQUIRED_COLUMNS = [
    "platform", "campaign_type", "industry", "country",
    "impressions", "clicks", "CTR", "CPC", "ad_spend",
    "conversions", "CPA", "revenue", "ROAS",
]


def download_dataset() -> str:
    """Downloads the dataset via kagglehub, returns local folder path."""
    import kagglehub
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    logger.info(f"Dataset downloaded to: {path}")
    logger.info(f"Files: {os.listdir(path)}")
    return path


def load_raw_data(csv_path: str = None) -> pd.DataFrame:
    """Load the raw CSV — either from a given path or by downloading via kagglehub."""
    if csv_path is None:
        folder = download_dataset()
        csv_path = f"{folder}/{CSV_FILENAME}"

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Missing values:\n{df.isnull().sum()}")
    logger.info(f"Duplicates: {df.duplicated().sum()}")
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info("Schema validation passed")


def run(csv_path: str = None, save: bool = True) -> pd.DataFrame:
    df = load_raw_data(csv_path)
    validate_schema(df)

    if save:
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        logger.info(f"Saved cleaned data to {PROCESSED_DATA_PATH}")

    return df


if __name__ == "__main__":
    run()
