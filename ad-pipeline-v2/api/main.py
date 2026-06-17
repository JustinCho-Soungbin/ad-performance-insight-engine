"""
main.py
─────────────────────────────────────────
API layer ("front of house"): routing only.
No model logic, no preprocessing — everything is delegated to
service.py. This file should stay small forever.

Run:
    uvicorn api.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from api.logging_config import configure_logging
from api.model_loader import model_bundle
from api.schemas import CampaignFeatures, PredictionResponse, HealthResponse
from api import service

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not on every request
    model_bundle.load()
    yield


app = FastAPI(
    title="Ad Performance Insight Engine API",
    description="Serves the ROAS classification model trained in src/train.py",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=model_bundle.loaded)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CampaignFeatures):
    if not model_bundle.loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python src/train.py` to generate it.",
        )
    return service.predict_roas_class(features)
