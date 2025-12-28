"""Helper class for loading the trained model pipeline and running predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd

from .config import FEATURE_COLUMNS, RISK_LEVELS, OBESITY_RISK_LEVELS, TARGET_COLUMN


@dataclass
class DiseaseRisk:
    """Single disease risk prediction."""
    risk_level: int
    probability: float
    label: str


@dataclass
class MultiOutputPrediction:
    """Multi-disease risk prediction."""
    diabetes: DiseaseRisk
    obesity: DiseaseRisk


class DiabetesPredictor:
    """Load the persisted pipeline and expose a convenient predict API."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.pipeline = joblib.load(model_path)

    def predict(self, payload: Dict[str, Any]) -> MultiOutputPrediction:
        """Run inference and return risk levels for Diabetes and Obesity."""
        features = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
        predictions = self.pipeline.predict(features)
        
        try:
            if predictions.ndim == 2:
                diabetes_pred, obesity_pred = predictions[0][0], predictions[0][1]
            else:
                diabetes_pred = int(predictions[0]) if hasattr(predictions[0], '__iter__') == False else predictions[0]
                obesity_pred = 0
        except (IndexError, TypeError):
            diabetes_pred = int(predictions[0]) if hasattr(predictions, '__len__') else int(predictions)
            obesity_pred = 0
        
        try:
            probabilities = self.pipeline.predict_proba(features)
            
            if isinstance(probabilities, list) and len(probabilities) >= 2:
                diabetes_probs = probabilities[0][0]  # [P(class 0), P(class 1), P(class 2)]
                diabetes_prob = float(max(diabetes_probs))  # Take max probability
                
                obesity_probs = probabilities[1][0]  # [P(class 0), P(class 1)]
                obesity_prob = float(max(obesity_probs))  # Take max probability
            else:
                # Single output probability
                diabetes_probs = probabilities[0][0]
                diabetes_prob = float(max(diabetes_probs)) if len(diabetes_probs) > 0 else 0.5
                obesity_prob = 0.5
        except Exception as e:
            diabetes_prob = 0.5
            obesity_prob = 0.5
        
        diabetes_risk_info = RISK_LEVELS[int(diabetes_pred)]
        obesity_risk_info = OBESITY_RISK_LEVELS[int(obesity_pred)]
        
        return MultiOutputPrediction(
            diabetes=DiseaseRisk(
                risk_level=int(diabetes_pred),
                probability=diabetes_prob,  # Keep as decimal 0.0-1.0
                label=diabetes_risk_info["label"]
            ),
            obesity=DiseaseRisk(
                risk_level=int(obesity_pred),
                probability=obesity_prob,  # Keep as decimal 0.0-1.0
                label=obesity_risk_info["label"]
            )
        )
