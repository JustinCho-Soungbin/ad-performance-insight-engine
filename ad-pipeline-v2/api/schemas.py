"""
schemas.py
─────────────────────────────────────────
Request/response models — field names match the notebook's
FEATURES list exactly (cell [10]/[12]).
"""

from pydantic import BaseModel, ConfigDict, Field


class CampaignFeatures(BaseModel):
    """Raw campaign metrics — derived features (CTR_CPC ratio etc.)
    and categorical encodings are computed server-side in service.py."""
    impressions: int = Field(..., gt=0)
    clicks: int = Field(..., ge=0)
    CTR: float = Field(..., ge=0, le=1)
    CPC: float = Field(..., ge=0)
    ad_spend: float = Field(..., gt=0)
    conversions: int = Field(..., ge=0)
    CPA: float = Field(..., ge=0)
    platform: str = Field(..., description="e.g. 'Google Ads', 'Meta Ads', 'TikTok Ads'")
    campaign_type: str
    industry: str
    country: str


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    high_roas_probability: float
    is_high_roas: bool
    model_version: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
