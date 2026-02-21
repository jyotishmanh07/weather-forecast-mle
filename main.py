from fastapi import FastAPI, HTTPException
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import boto3, os

# Import your inference logic
from pipelines.inference_pipeline.inference import predict

# ----------------------------
# Config & AWS Setup
# ----------------------------
S3_BUCKET = os.getenv("S3_BUCKET", "weather-forecast-data-mle")
REGION = os.getenv("AWS_REGION", "eu-central-1")
s3 = boto3.client("s3", region_name=REGION)

def load_from_s3(key, local_path):
    local_path = Path(local_path)
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(S3_BUCKET, key, str(local_path))
    return str(local_path)

# ----------------------------
# Paths & Artifacts
# ----------------------------
MODEL_PATH = Path(load_from_s3("models/best_weather_xgb.pkl", "models/best_weather_xgb.pkl"))
SCALER_PATH = Path(load_from_s3("models/tuned_scaler.pkl", "models/tuned_scaler.pkl"))

# ----------------------------
# App Setup
# ----------------------------
app = FastAPI(title="Weather Forecast ML API")

@app.get("/health")
def health():
    return {"status": "healthy", "model": str(MODEL_PATH.name)}

# ----------------------------
# Prediction Endpoint
# ----------------------------
@app.post("/predict")
def predict_weather(data: List[dict]):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=500, detail="Model artifact missing.")

    df = pd.DataFrame(data)
    if df.empty:
        raise HTTPException(status_code=400, detail="No data provided.")

    try:
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])

        city_cols = [c for c in df.columns if c.startswith("city_")]
        if "city" not in df.columns and city_cols:
            df["city"] = df[city_cols].idxmax(axis=1).str.replace("city_", "").astype(int)
            
        cols_to_drop = city_cols + [c for c in ["year", "month"] if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Run full inference pipeline
        preds_df = predict(df, model_path=MODEL_PATH, scaler_path=SCALER_PATH)

        # Build Response for Streamlit
        resp = {
            "city": preds_df["city"].tolist() if "city" in preds_df.columns else [],
            "time": preds_df["time"].astype(str).tolist(),
            "predicted_max_temp": preds_df["predicted_temp_max"].astype(float).tolist()
        }
        
        # Safely map actuals
        if "actual_temp_max" in preds_df.columns:
            resp["actual_max_temp"] = preds_df["actual_temp_max"].astype(float).tolist()
        elif "temperature_2m_max" in preds_df.columns:
            resp["actual_max_temp"] = preds_df["temperature_2m_max"].astype(float).tolist()

        return resp

 

    except Exception as e:
        print(f"INFERENCE ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")