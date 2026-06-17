"""
model_loader.py
─────────────────────────────────────────
Model abstraction layer. Holds the fitted model, scaler, AND
label encoders together so api/service.py never imports
sklearn/joblib directly — only this module does.
"""

import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/best_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")
ENCODERS_PATH = Path("models/label_encoders.pkl")
MODEL_VERSION = "logreg-v1"  # bump whenever train.py produces a new artifact


class ModelBundle:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoders = None
        self.loaded = False

    def load(self):
        required = [MODEL_PATH, SCALER_PATH, ENCODERS_PATH]
        if not all(p.exists() for p in required):
            missing = [str(p) for p in required if not p.exists()]
            logger.warning(f"Missing model artifacts: {missing}. Run the src/ pipeline first.")
            self.loaded = False
            return

        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.encoders = joblib.load(ENCODERS_PATH)
        self.loaded = True
        logger.info(f"Loaded model artifacts (version={MODEL_VERSION})")

    def encode_categorical(self, column: str, value: str) -> int:
        """Encode a raw categorical value using the saved LabelEncoder.
        Falls back to -1 for unseen categories instead of crashing."""
        le = self.encoders[column]
        if value not in le.classes_:
            logger.warning(f"Unseen category '{value}' for column '{column}' — encoding as -1")
            return -1
        return int(le.transform([value])[0])

    def predict_proba(self, feature_row: list[float]) -> float:
        if not self.loaded:
            raise RuntimeError("Model not loaded — call .load() first")
        scaled = self.scaler.transform([feature_row])
        return float(self.model.predict_proba(scaled)[0, 1])


model_bundle = ModelBundle()
