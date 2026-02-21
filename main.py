from fastapi import FastAPI, HTTPException
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import boto3, os

from pipelines.inference_pipeline.inference import predict

# ----------------------------
# Config
# ----------------------------
# Updated to your specific weather bucket
S3_BUCKET = os.getenv("S3_BUCKET", "weather-forecast-data-mle")
REGION = os.getenv("AWS_REGION", "eu-central-1")
s3 = boto3.client("s3", region_name=REGION)

def load_from_s3(key, local_path):
    """Download artifacts from your S3 registry if not cached locally."""
    local_path = Path(local_path)
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {key} from S3...")
        try:
            s3.download_file(S3_BUCKET, key, str(local_path))
        except Exception as e:
            print(f"Failed to download {key}: {e}")
    return str(local_path)

# ----------------------------
# Paths & Artifacts
# ----------------------------
# Points to newly uploaded XGBoost model and encoded training data
MODEL_PATH = Path(load_from_s3("models/best_weather_xgb.pkl", "models/best_weather_xgb.pkl"))
SCALER_PATH = Path(load_from_s3("models/tuned_scaler.pkl", "models/tuned_scaler.pkl"))
TRAIN_FE_PATH = Path(load_from_s3("processed/train_encoded.csv", "data/processed/train_encoded.csv"))

# Load expected training features for alignment
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    # target column is 'temperature_2m_max'
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "temperature_2m_max"]
else:
    TRAIN_FEATURE_COLUMNS = None

# ----------------------------
# App Setup
# ----------------------------
app = FastAPI(title="Weather Forecast ML API")

@app.get("/")
def root():
    return {"message": "Weather Forecast API is live 🌤️"}

@app.get("/health")
def health():
    """Confirms model presence and feature alignment status."""
    status = {
        "status": "healthy" if MODEL_PATH.exists() else "unhealthy",
        "model": str(MODEL_PATH.name),
        "n_features_expected": len(TRAIN_FEATURE_COLUMNS) if TRAIN_FEATURE_COLUMNS else 0
    }
    return status

# ----------------------------
# Prediction Endpoint
# ----------------------------
@app.post("/predict")
def predict_weather(data: List[dict]):
    """
    Core Endpoint: Accepts a list of daily weather records.
    NOTE: For accurate results (MAE ~1.84°C), provide a continuous 
    7-day sequence for the city to enable lag/rolling features.
    """
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=500, detail="Model artifact missing.")

    df = pd.DataFrame(data)
    if df.empty:
        raise HTTPException(status_code=400, detail="No data provided.")

    try:
        # Calls inference logic which handles Lags and Scaling
        preds_df = predict(df, model_path=MODEL_PATH, scaler_path=SCALER_PATH)

        resp = {
            "city": preds_df["city"].tolist(),
            "time": preds_df["time"].astype(str).tolist(),
            "predicted_max_temp": preds_df["predicted_temp_max"].astype(float).tolist()
        }
        
        # Include actuals if they were passed in the request (for validation)
        if "actual_temp_max" in preds_df.columns:
            resp["actual_max_temp"] = preds_df["actual_temp_max"].astype(float).tolist()

        return resp

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))