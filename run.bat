@echo off
echo ========================================================
echo   Airbnb Rental Price Predictor (ML + Local Ollama)
echo ========================================================
echo.

:: Check if virtual environment or packages exist
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    pause
    exit /b 1
)

:: Check if models exist, if not train them
if not exist "models\best_model.joblib" (
    echo [INFO] Training ML models for the first time...
    python data\generate_dataset.py
    python train.py
)

echo [INFO] Starting Streamlit web application...
echo App will open in your browser at http://localhost:8501
python -m streamlit run app.py

pause
