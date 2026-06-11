"""
Inference pipeline for Weather Forecasting.
- Takes RAW weather input data.
- Applies preprocessing, validation, and feature engineering.
- Aligns features with the training schema using the saved scaler.
- Returns temperature predictions aligned with original metadata.
"""

from __future__ import annotations
import argparse
import os
import pandas as pd
import logging
from pathlib import Path
from joblib import load

import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

# Import internal modules
from pipelines.feature_pipeline.preprocess import clean_weather_data
from pipelines.feature_pipeline.feature_engineering import validate_weather_data, encode_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "best_weather_xgb.pkl"
DEFAULT_SCALER = PROJECT_ROOT / "models" / "tuned_scaler.pkl"
TRAIN_FE_PATH = PROJECT_ROOT / "data" / "processed" / "train_encoded.csv"

REGISTRY_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "weather-xgb")
REGISTRY_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")

# Load the exact feature columns the model expects
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [
        c for c in _train_cols.columns
        if c not in ["temperature_2m_max", "time"]
    ]
else:
    TRAIN_FEATURE_COLUMNS = None


def _load_from_registry():
    """Try loading model and scaler from MLflow Model Registry.

    Checks Production stage first, then Staging. Returns (model, scaler, version_str)
    or (None, None, None) if the registry is unavailable.
    """
    client = MlflowClient()
    for stage in (REGISTRY_STAGE, "Staging"):
        try:
            versions = client.get_latest_versions(REGISTRY_MODEL_NAME, stages=[stage])
            if not versions:
                continue
            version = versions[0]
            model = mlflow.xgboost.load_model(f"models:/{REGISTRY_MODEL_NAME}/{stage}")
            scaler_local = mlflow.artifacts.download_artifacts(
                run_id=version.run_id, artifact_path="tuned_scaler.pkl"
            )
            scaler = load(scaler_local)
            version_str = f"{REGISTRY_MODEL_NAME}/v{version.version} ({stage})"
            LOGGER.info(f"Loaded model from registry: {version_str}")
            return model, scaler, version_str
        except Exception as e:
            LOGGER.warning(f"Registry load failed (stage={stage}): {e}")
    return None, None, None


def predict(
    input_df: pd.DataFrame,
    model=None,
    scaler=None,
    model_path: Path | str | None = None,
    scaler_path: Path | str | None = None,
) -> pd.DataFrame:
    """Run full inference pipeline on raw weather data.

    Accepts pre-loaded model/scaler objects (preferred for servers that load
    once at startup). When neither objects nor paths are supplied, tries the
    MLflow Model Registry then falls back to local files.
    """
    if model is None:
        if model_path is not None:
            model_path = Path(model_path)
            scaler_path = Path(scaler_path) if scaler_path else DEFAULT_SCALER
            model = load(model_path)
            scaler = load(scaler_path) if scaler_path.exists() else None
        else:
            model, scaler, _ = _load_from_registry()
            if model is None:
                LOGGER.info("Registry unavailable — loading from local files.")
                model = load(DEFAULT_MODEL)
                scaler = load(DEFAULT_SCALER) if DEFAULT_SCALER.exists() else None

    # Standard cleaning (handles lags/rolling for cities)
    df = clean_weather_data(input_df.copy())

    # 'clean_weather_data' sets 'time' as index. Because multiple cities share
    # the same time, we must reset the index to avoid duplicate labels during reindex.
    df = df.reset_index(drop=True)

    # Validation
    validate_weather_data(df, "Inference")

    # Encoding (City One-Hot Encoding)
    df = encode_features(df)

    # Capture actuals for the final output
    if "temperature_2m_max" in df.columns:
        df = df.drop(columns=["temperature_2m_max"])

    # Feature alignment — forces exactly the columns the model was trained on
    if TRAIN_FEATURE_COLUMNS is not None:
        df = df.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)
    elif "time" in df.columns:
        df = df.drop(columns=["time"])

    # Scaling
    if scaler is not None:
        X_scaled = scaler.transform(df)
    else:
        LOGGER.warning("Scaler not available — proceeding without scaling.")
        X_scaled = df.values

    preds = model.predict(X_scaled)

    # Build output aligned to original input
    out = input_df.copy()
    out["predicted_temp_max"] = preds
    if "temperature_2m_max" in out.columns:
        out = out.rename(columns={"temperature_2m_max": "actual_temp_max"})

    return out

if __name__ == "__main__":
    # Standard CLI runner logic
    parser = argparse.ArgumentParser(description="Run inference on new weather data.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw weather CSV")
    parser.add_argument("--output", type=str, default="data/predictions.csv", help="Output path")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--scaler", type=str, default=str(DEFAULT_SCALER))

    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    preds_df = predict(raw_df, model_path=args.model, scaler_path=args.scaler)

    preds_df.to_csv(args.output, index=False)
    LOGGER.info(f"Predictions saved to {args.output}")