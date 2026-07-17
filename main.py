from fastapi import FastAPI, HTTPException
import logging
from pathlib import Path
from datetime import datetime
from typing import List
import os
import joblib
import pandas as pd
import mlflow
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Summary
from pydantic import BaseModel, Field, field_validator, model_validator

from pipelines.inference_pipeline.inference import predict, _load_from_registry
from pipelines.monitoring_pipeline.drift import log_prediction_inputs, run_drift_check

LOGGER = logging.getLogger(__name__)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

_VALID_WMO_CODES = {0, 1, 2, 3, 45, 51, 53, 55, 61, 63, 65, 71, 73, 75, 80, 81, 82, 95, 96, 99}


class WeatherInput(BaseModel):
    time: datetime
    city: int = Field(..., ge=1, le=10, description="City ID (1–10)")
    weathercode: int = Field(..., description="WMO weather interpretation code")
    temperature_2m_min: float = Field(..., ge=-30.0, le=45.0, description="Daily min temperature (°C)")
    temperature_2m_max: float = Field(
        ..., ge=-30.0, le=45.0,
        description="Today's observed max temperature (°C). A feature for the next-day forecast."
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
        if self.temperature_2m_max < self.temperature_2m_min:
            raise ValueError("temperature_2m_max must be >= temperature_2m_min")
        return self


class PredictionResponse(BaseModel):
    city: List[int]
    time: List[str]
    forecast_date: List[str]
    predicted_max_temp: List[float]
    model_version: str


# Loaded once at startup; reused on every request. Registry (@champion →
# @challenger) first, then the bundled local .pkl as an offline fallback.
_model, _scaler, _model_version = _load_from_registry()

if _model is None:
    LOGGER.warning("Registry unavailable — loading model from local .pkl files.")
    _model_path = Path("models/best_weather_xgb.pkl")
    _scaler_path = Path("models/tuned_scaler.pkl")
    _model = joblib.load(_model_path)
    _scaler = joblib.load(_scaler_path) if _scaler_path.exists() else None
    _model_version = _model_path.name


app = FastAPI(title="Weather Forecast ML API")

# Prometheus: default HTTP metrics (latency, request counts) at /metrics, plus
# two model-specific metrics recorded on each prediction.
Instrumentator().instrument(app).expose(app)
PREDICTIONS_TOTAL = Counter("predictions_total", "Predictions served", ["model_version"])
PREDICTED_TEMP = Summary(
    "predicted_temp_max_celsius", "Distribution of predicted next-day max temperature"
)


@app.get("/health")
def health():
    return {"status": "healthy", "model_version": _model_version}


@app.get("/drift")
def get_drift():
    result = run_drift_check()
    result["model_version"] = _model_version or "unknown"
    if result.get("dataset_drifted"):
        # A breach is the retraining signal. In this local-Airflow setup that is
        # a manual/REST trigger of the weather_retrain DAG; a hosted scheduler
        # would use a webhook. See airflow/README.md.
        LOGGER.warning("Input drift detected (%s columns)", result.get("drifted_column_count"))
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

        preds = preds_df["predicted_temp_max"].astype(float).tolist()
        for p in preds:
            PREDICTED_TEMP.observe(p)
        PREDICTIONS_TOTAL.labels(model_version=_model_version or "unknown").inc(len(preds))

        return PredictionResponse(
            city=preds_df["city"].tolist() if "city" in preds_df.columns else [],
            time=preds_df["time"].astype(str).tolist(),
            forecast_date=preds_df["forecast_date"].astype(str).tolist(),
            predicted_max_temp=preds,
            model_version=_model_version or "unknown",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
