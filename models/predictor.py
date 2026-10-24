"""
Predictor module for Airbnb listing prices.
Loads trained scikit-learn pipeline and provides an easy-to-use prediction API.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
IMPORTANCE_PATH = os.path.join(MODELS_DIR, "feature_importance.json")

# Default coordinates for cities
CITY_COORDINATES = {
    "Berlin": {"lat": 52.5200, "lon": 13.4050},
    "Munich": {"lat": 48.1351, "lon": 11.5820},
    "Hamburg": {"lat": 53.5511, "lon": 9.9937},
    "Frankfurt": {"lat": 50.1109, "lon": 8.6821},
    "Cologne": {"lat": 50.9375, "lon": 6.9603},
    "Stuttgart": {"lat": 48.7758, "lon": 9.1829},
    "Dresden": {"lat": 51.0504, "lon": 13.7373},
    "Leipzig": {"lat": 51.3397, "lon": 12.3731},
}

class AirbnbPricePredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.pipeline = None
        self.metrics = None
        self.feature_importance = None
        self.load()

    def load(self):
        """Loads model and metadata artifacts if present."""
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
        
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                self.metrics = json.load(f)
                
        if os.path.exists(IMPORTANCE_PATH):
            with open(IMPORTANCE_PATH, "r") as f:
                self.feature_importance = json.load(f)

    @property
    def is_trained(self) -> bool:
        return self.pipeline is not None

    def predict(self, listing_data: dict) -> dict:
        """
        Takes property attributes dictionary, runs model inference,
        and returns structured pricing estimates and insights.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained or loaded. Run train.py first.")
            
        # Ensure DataFrame has expected columns
        df_input = pd.DataFrame([listing_data])
        
        # Predict price
        raw_pred = float(self.pipeline.predict(df_input)[0])
        pred_price = max(20.0, round(raw_pred, 2))
        
        # Uncertainty bounds based on test MAE
        mae = self.metrics.get("winning_metrics", {}).get("mae", 30.0) if self.metrics else 30.0
        low_bound = max(15.0, round(pred_price - (1.2 * mae), 2))
        high_bound = round(pred_price + (1.2 * mae), 2)
        
        accommodates = max(1, listing_data.get("accommodates", 1))
        price_per_guest = round(pred_price / accommodates, 2)
        
        # Approximate monthly revenue (assuming 70% occupancy)
        estimated_monthly_revenue = round(pred_price * 30 * 0.70, 2)
        
        return {
            "predicted_price": pred_price,
            "low_estimate": low_bound,
            "high_estimate": high_bound,
            "price_per_guest": price_per_guest,
            "estimated_monthly_revenue": estimated_monthly_revenue,
            "model_name": self.metrics.get("winning_model", "Selected Model") if self.metrics else "Best Model",
            "model_r2": self.metrics.get("winning_metrics", {}).get("r2_score", 0.78) if self.metrics else 0.78,
            "model_mae": mae,
            "currency": "EUR (€)"
        }

    def get_city_coords(self, city_name: str) -> dict:
        return CITY_COORDINATES.get(city_name, {"lat": 52.5200, "lon": 13.4050})
