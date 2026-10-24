"""
Machine Learning Training Pipeline for Airbnb Price Prediction.
Trains multiple regression models, compares them on validation metrics,
and saves the best model along with evaluation metrics and feature importances.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "airbnb_listings.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
FEATURE_IMPORTANCE_PATH = os.path.join(MODELS_DIR, "feature_importance.json")

# Features Definition
CATEGORICAL_FEATURES = ["city", "neighborhood_type", "room_type"]
NUMERICAL_FEATURES = [
    "accommodates",
    "bedrooms",
    "bathrooms",
    "beds",
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating"
]
BINARY_FEATURES = [
    "has_wifi",
    "has_kitchen",
    "has_air_conditioning",
    "has_free_parking",
    "has_washer",
    "has_workspace",
    "has_balcony",
    "has_elevator",
    "is_pet_friendly",
    "has_pool_or_hot_tub"
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + BINARY_FEATURES
TARGET = "price"

def build_preprocessor():
    """Builds a scikit-learn ColumnTransformer for preprocessing."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor

def get_candidate_models():
    """Returns a dictionary of candidate regression models to evaluate."""
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=180,
            max_depth=16,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.85,
            random_state=42
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=160,
            learning_rate=0.08,
            max_depth=7,
            random_state=42
        ),
        "Ridge Regression": Ridge(
            alpha=1.0,
            random_state=42
        )
    }

def train_and_evaluate_models(data_path=DATA_PATH, save_dir=MODELS_DIR):
    """
    Main training and model comparison workflow.
    Returns:
        dict: Summary of metrics, winning model name, and performance.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}. Run generate_dataset.py first.")
        
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Feature extraction
    X = df[ALL_FEATURES]
    y = df[TARGET]
    
    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Dataset split: {len(X_train)} training samples, {len(X_test)} testing samples.")
    
    candidates = get_candidate_models()
    results = {}
    fitted_pipelines = {}
    
    print("\nTraining and evaluating candidate models:")
    print("-" * 75)
    print(f"{'Model Name':<25} | {'R² Score':<10} | {'MAE (€)':<10} | {'RMSE (€)':<10} | {'MAPE':<8}")
    print("-" * 75)
    
    best_model_name = None
    best_score = -float("inf")
    
    for name, model in candidates.items():
        preprocessor = build_preprocessor()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Calculate evaluation metrics
        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mape = float(mean_absolute_percentage_error(y_test, y_pred))
        
        results[name] = {
            "r2_score": round(r2, 4),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 4)
        }
        fitted_pipelines[name] = pipeline
        
        print(f"{name:<25} | {r2:<10.4f} | €{mae:<9.2f} | €{rmse:<9.2f} | {mape * 100:.1f}%")
        
        # Pick the best model based on R2 score (or lowest MAE)
        if r2 > best_score:
            best_score = r2
            best_model_name = name
            
    print("-" * 75)
    print(f"--> Winning Model: {best_model_name} with R² = {best_score:.4f}")
    
    best_pipeline = fitted_pipelines[best_model_name]
    
    # Extract feature names & importances
    feature_names = []
    cat_encoder = best_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    feature_names.extend(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names.extend(NUMERICAL_FEATURES)
    feature_names.extend(BINARY_FEATURES)
    
    regressor = best_pipeline.named_steps["regressor"]
    feature_importances = {}
    
    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        for idx in sorted_indices[:15]:
            fname = feature_names[idx]
            feature_importances[fname] = round(float(importances[idx]), 4)
    elif hasattr(regressor, "coef_"):
        coefs = np.abs(regressor.coef_)
        sorted_indices = np.argsort(coefs)[::-1]
        for idx in sorted_indices[:15]:
            fname = feature_names[idx]
            feature_importances[fname] = round(float(coefs[idx]), 4)
            
    # Save artifacts
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    print(f"Saved best model pipeline to: {BEST_MODEL_PATH}")
    
    meta_summary = {
        "timestamp": datetime.now().isoformat(),
        "winning_model": best_model_name,
        "models_evaluated": results,
        "winning_metrics": results[best_model_name],
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "features_used": ALL_FEATURES
    }
    
    with open(METRICS_PATH, "w") as f:
        json.dump(meta_summary, f, indent=2)
    print(f"Saved metrics summary to: {METRICS_PATH}")
    
    with open(FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(feature_importances, f, indent=2)
    print(f"Saved top feature importances to: {FEATURE_IMPORTANCE_PATH}")
    
    return meta_summary

if __name__ == "__main__":
    train_and_evaluate_models()
