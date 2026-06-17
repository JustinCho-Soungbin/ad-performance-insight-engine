"""
service.py
─────────────────────────────────────────
Service Layer: takes raw campaign metrics, computes the same
derived features as src/features.py (conversion_rate, ctr_cpc_ratio,
log_impressions, cost_per_impression), encodes categoricals, then
calls the model. api/main.py never sees any of this logic.
"""

import logging
import numpy as np

from api.model_loader import model_bundle, MODEL_VERSION
from api.schemas import CampaignFeatures, PredictionResponse

logger = logging.getLogger(__name__)

HIGH_ROAS_THRESHOLD = 0.5

# Must match the exact column order used in src/train.py's FEATURES list
FEATURE_ORDER = [
    "impressions", "clicks", "CTR", "CPC", "ad_spend", "conversions", "CPA",
    "conversion_rate", "ctr_cpc_ratio", "log_impressions", "cost_per_impression",
    "platform_enc", "campaign_type_enc", "industry_enc", "country_enc",
]


def build_feature_row(f: CampaignFeatures) -> list[float]:
    """Recompute the same derived features as src/features.py, in the
    same order the model was trained on."""
    conversion_rate = f.conversions / f.clicks if f.clicks > 0 else 0.0
    ctr_cpc_ratio = f.CTR / (f.CPC + 1e-6)
    log_impressions = float(np.log1p(f.impressions))
    cost_per_impression = f.ad_spend / f.impressions

    platform_enc = model_bundle.encode_categorical("platform", f.platform)
    campaign_type_enc = model_bundle.encode_categorical("campaign_type", f.campaign_type)
    industry_enc = model_bundle.encode_categorical("industry", f.industry)
    country_enc = model_bundle.encode_categorical("country", f.country)

    row = {
        "impressions": f.impressions, "clicks": f.clicks, "CTR": f.CTR,
        "CPC": f.CPC, "ad_spend": f.ad_spend, "conversions": f.conversions,
        "CPA": f.CPA, "conversion_rate": conversion_rate,
        "ctr_cpc_ratio": ctr_cpc_ratio, "log_impressions": log_impressions,
        "cost_per_impression": cost_per_impression,
        "platform_enc": platform_enc, "campaign_type_enc": campaign_type_enc,
        "industry_enc": industry_enc, "country_enc": country_enc,
    }
    return [row[col] for col in FEATURE_ORDER]


def predict_roas_class(features: CampaignFeatures) -> PredictionResponse:
    feature_row = build_feature_row(features)
    probability = model_bundle.predict_proba(feature_row)

    logger.info(
        "prediction_made",
        extra={
            "platform": features.platform,
            "CTR": features.CTR,
            "CPA": features.CPA,
            "probability": round(probability, 4),
        },
    )

    return PredictionResponse(
        high_roas_probability=round(probability, 4),
        is_high_roas=probability >= HIGH_ROAS_THRESHOLD,
        model_version=MODEL_VERSION,
    )
