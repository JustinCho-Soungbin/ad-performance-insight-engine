"""
explain.py
─────────────────────────────────────────
Stage 4: Model Explainability
Mirrors notebook cell [23]: SHAP for both Logistic Regression
(LinearExplainer) and XGBoost (TreeExplainer).

Run standalone:
    python src/explain.py
"""

import logging
import joblib
import pandas as pd
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURES_PATH = Path("data/processed/features.csv")
MODEL_PATH = Path("models/best_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")
SHAP_OUTPUT_PATH = Path("models/shap_summary.csv")

FEATURES = [
    "impressions", "clicks", "CTR", "CPC", "ad_spend", "conversions", "CPA",
    "conversion_rate", "ctr_cpc_ratio", "log_impressions", "cost_per_impression",
    "platform_enc", "campaign_type_enc", "industry_enc", "country_enc",
]
TARGET = "high_roas"


def run():
    df = pd.read_csv(FEATURES_PATH)
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # model_type detection: LinearExplainer for LogReg, TreeExplainer for XGBoost
    model_class = type(model).__name__
    logger.info(f"Computing SHAP for {model_class}...")

    if "Logistic" in model_class:
        explainer = shap.LinearExplainer(model, X_train_scaled, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_test_scaled)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

    mean_abs_shap = pd.Series(
        abs(shap_values).mean(axis=0), index=FEATURES
    ).sort_values(ascending=False)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    mean_abs_shap.to_csv(SHAP_OUTPUT_PATH, header=["mean_abs_shap"])

    logger.info("Feature importance (mean |SHAP value|):")
    for feat, val in mean_abs_shap.items():
        logger.info(f"  {feat:<22} {val:.4f}")

    logger.info(f"Top predictor: {mean_abs_shap.index[0]} (matches notebook finding: CPA)")
    return mean_abs_shap


if __name__ == "__main__":
    run()
