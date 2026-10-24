# 🏠 Airbnb Rental Price Predictor (ML + Local AI)

A polished, full-stack machine learning web application that predicts Airbnb nightly rental prices from listing features such as location, room type, guest capacity, bedrooms, bathrooms, and key amenities.

Includes an automated multi-model machine learning comparison pipeline, interactive map visualizer, model retraining dashboard, and private **Local Ollama AI advisory** (using your installed models like `llama3.2:latest`) with **zero API costs and full local privacy compliance**.

---

## ✨ Features

- **Instant Price Estimation:** Real-time pricing predictions with high/low fair market range and estimated monthly gross revenue.
- **Multi-Model Machine Learning:** Trains and cross-evaluates 4 regression models:
  - **Random Forest Regressor**
  - **Gradient Boosting Regressor**
  - **Hist Gradient Boosting Regressor**
  - **Ridge Regression**
  - Automatically selects, benchmarks, and deploys the highest-performing model based on $R^2$ score and MAE.
- **Feature Importance Analysis:** Explains the most influential price drivers (room type, prime city locations, amenities like AC/parking/pool).
- **1-Click Retraining:** Retrain models directly through the web UI or via `python train.py`.
- **Local Ollama AI Integration (100% Free & Private):**
  - Connects to your local Ollama daemon (`http://localhost:11434`).
  - Generates revenue management tips, weekend pricing strategies, and catchy listing copy.
  - Safe for Germany/EU privacy regulations: Zero data leaves your computer.
- **Interactive Geospatial Visualization:** Renders listing location on interactive 3D map views using PyDeck.
- **Docker Ready:** Includes `Dockerfile` and `docker-compose.yml` for containerized deployment.

---

## 🚀 Quick Start (Local Setup)

### 1. Requirements
- Python 3.10+ (Tested on Python 3.14)
- (Optional) [Ollama](https://ollama.com/) running locally with any model (e.g., `llama3.2:latest`, `llama3.1:latest`, `qwen2.5:3b`).

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Dataset & Train Models
```bash
python data/generate_dataset.py
python train.py
```
*(The best model pipeline is automatically saved to `models/best_model.joblib` alongside metrics in `models/metrics.json`)*.

### 4. Run the Web Application
#### On Windows:
Double-click `run.bat` or run:
```bash
python -m streamlit run app.py
```

Open your browser at **`http://localhost:8501`**.

---

## 🐳 Docker Option

To run the application inside an isolated Docker container:

```bash
docker-compose up --build
```

Then visit **`http://localhost:8501`**. The container is configured with `host.docker.internal` to seamlessly communicate with your host's local Ollama instance.

---

## 📁 Project Architecture

```
airbnb-price-predictor/
├── app.py                     # Streamlit web app with custom Airbnb-styled UI
├── train.py                   # Automated ML training, evaluation & model selection pipeline
├── ollama_client.py           # 100% local, offline Ollama API integration
├── requirements.txt           # Python dependencies
├── run.bat                    # 1-click launcher for Windows
├── Dockerfile                 # Container definition
├── docker-compose.yml         # Compose configuration
├── data/
│   ├── generate_dataset.py    # Realistic synthetic Airbnb dataset generator (German cities)
│   └── airbnb_listings.csv    # 4,000+ benchmark listings with coordinates and amenities
└── models/
    ├── predictor.py           # Clean prediction interface & uncertainty calculation
    ├── best_model.joblib      # Serialized scikit-learn pipeline (winning model)
    ├── metrics.json           # Model comparison benchmarks (R², MAE, RMSE, MAPE)
    └── feature_importance.json# Extracted top predictive coefficients/weights
```

---

## 🤖 Machine Learning Pipeline Details

1. **Preprocessing Pipeline (`ColumnTransformer`):**
   - **Categorical:** `city`, `neighborhood_type`, `room_type` encoded with `OneHotEncoder(handle_unknown='ignore')`.
   - **Numerical:** `accommodates`, `bedrooms`, `bathrooms`, `beds`, `minimum_nights`, `number_of_reviews`, `review_scores_rating` scaled with `StandardScaler()`.
   - **Binary Amenities:** Pass-through for 10 amenities (WiFi, Kitchen, AC, Parking, Washer, Workspace, Balcony, Elevator, Pets, Pool/Hot tub).

2. **Model Evaluation & Selection:**
   - Evaluated on a 20% holdout test set using:
     - **$R^2$ Score** (Goodness of fit)
     - **MAE** (Mean Absolute Error in Euros)
     - **RMSE** (Root Mean Squared Error in Euros)
     - **MAPE** (Mean Absolute Percentage Error)
   - The best model is automatically saved as the active serving pipeline.

---

## 🛡️ Privacy & Compliance Notice

- **Zero External Tracking:** No telemetry, tracking pixels, or paid APIs are used.
- **GDPR & German Data Privacy Compliant:** The application operates completely offline. No user data, property addresses, or prompt queries are transmitted over the internet.
- All benchmark data is generated locally without copyright scraping.
