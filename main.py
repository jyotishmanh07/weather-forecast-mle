from fastapi import FastAPI, HTTPException
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import os
import joblib
import pandas as pd
import boto3
import mlflow
from pydantic import BaseModel, Field, field_validator, model_validator

from pipelines.inference_pipeline.inference import predict, _load_from_registry
from pipelines.monitoring_pipeline.drift import log_prediction_inputs, run_drift_check

S3_BUCKET = os.getenv("S3_BUCKET", "weather-forecast-data-mle")
REGION = os.getenv("AWS_REGION", "eu-central-1")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
s3 = boto3.client("s3", region_name=REGION)

_VALID_WMO_CODES = {0, 1, 2, 3, 45, 51, 53, 55, 61, 63, 65, 71, 73, 75, 80, 81, 82, 95, 96, 99}


class WeatherInput(BaseModel):
    time: datetime
    city: int = Field(..., ge=1, le=10, description="City ID (1–10)")
    weathercode: int = Field(..., description="WMO weather interpretation code")
    temperature_2m_min: float = Field(..., ge=-30.0, le=45.0, description="Daily min temperature (°C)")
    temperature_2m_max: Optional[float] = Field(
        None, ge=-30.0, le=45.0,
        description="Daily max temperature (°C). Optional — returned as actual_max_temp if provided."
    )
    precipitation_sum: float = Field(..., ge=0.0, le=60.0, description="Daily precipitation (mm)")

    @field_validator("weathercode")
    @classmethod
    def validate_wmo_code(cls, v: int) -> int:
        if v not in _VALID_WMO_CODES:
            raise ValueError(
                f"Invalid WMO weathercode {v}. Valid codes: {sorted(_VALID_WMO_CODES)}"
            )
        return v

    @model_validator(mode="after")
    def max_gte_min(self) -> "WeatherInput":
        if self.temperature_2m_max is not None and self.temperature_2m_max < self.temperature_2m_min:
            raise ValueError("temperature_2m_max must be >= temperature_2m_min")
        return self


class PredictionResponse(BaseModel):
    city: List[int]
    time: List[str]
    predicted_max_temp: List[float]
    actual_max_temp: Optional[List[float]] = None
    model_version: str


def _load_from_s3(key, local_path):
    local_path = Path(local_path)
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(S3_BUCKET, key, str(local_path))
    return local_path


# Loaded once at startup; reused on every request.
_model, _scaler, _model_version = _load_from_registry()

if _model is None:
    _model_path = _load_from_s3("models/best_weather_xgb.pkl", "models/best_weather_xgb.pkl")
    _scaler_path = _load_from_s3("models/tuned_scaler.pkl", "models/tuned_scaler.pkl")
    _model = joblib.load(_model_path)
    _scaler = joblib.load(_scaler_path)
    _model_version = _model_path.name


app = FastAPI(title="Weather Forecast ML API")


@app.get("/health")
def health():
    return {"status": "healthy", "model_version": _model_version}


@app.get("/metrics")
def get_metrics():
    result = run_drift_check()
    result["model_version"] = _model_version or "unknown"
    return result


@app.post("/predict", response_model=PredictionResponse)
def predict_weather(data: List[WeatherInput]):
    if _model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    if not data:
        raise HTTPException(status_code=400, detail="No data provided.")

    df = pd.DataFrame([row.model_dump() for row in data])

    try:
        log_prediction_inputs(df)
        preds_df = predict(df, model=_model, scaler=_scaler)

        resp = PredictionResponse(
            city=preds_df["city"].tolist() if "city" in preds_df.columns else [],
            time=preds_df["time"].astype(str).tolist(),
            predicted_max_temp=preds_df["predicted_temp_max"].astype(float).tolist(),
            model_version=_model_version or "unknown",
        )

        if "actual_temp_max" in preds_df.columns:
            resp.actual_max_temp = preds_df["actual_temp_max"].astype(float).tolist()
        elif "temperature_2m_max" in preds_df.columns:
            resp.actual_max_temp = preds_df["temperature_2m_max"].astype(float).tolist()

        return resp

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
