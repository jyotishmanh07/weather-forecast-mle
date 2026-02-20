"""
Inference pipeline for Weather Forecasting.
- Takes RAW weather input data.
- Applies preprocessing, validation, and feature engineering.
- Aligns features with the training schema using the saved scaler.
- Returns temperature predictions aligned with original metadata.
"""

from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from joblib import load

from pipelines.feature_pipeline.preprocess import clean_weather_data
from pipelines.feature_pipeline.feature_engineering import validate_weather_data, encode_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "best_weather_xgb.pkl"
DEFAULT_SCALER = PROJECT_ROOT / "models" / "tuned_scaler.pkl"
TRAIN_FE_PATH = PROJECT_ROOT / "data" / "processed" / "train_encoded.csv"

if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [
        c for c in _train_cols.columns 
        if c not in ["temperature_2m_max", "time"]
    ]
else:
    TRAIN_FEATURE_COLUMNS = None

def predict(
    input_df: pd.DataFrame,
    model_path: Path | str = DEFAULT_MODEL,
    scaler_path: Path | str = DEFAULT_SCALER,
) -> pd.DataFrame:
    """Run full inference pipeline on raw weather data."""

    df = clean_weather_data(input_df.copy())
    
    validate_weather_data(df, "Inference")
    
    df = encode_features(df)
    
    y_true = None
    if "temperature_2m_max" in df.columns:
        y_true = df["temperature_2m_max"].values
        df = df.drop(columns=["temperature_2m_max"])

    if TRAIN_FEATURE_COLUMNS is not None:
        df = df.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)
    elif "time" in df.columns:
        df = df.drop(columns=["time"])

    if Path(scaler_path).exists():
        scaler = load(scaler_path)
        X_scaled = scaler.transform(df)
    else:
        LOGGER.warning("Scaler not found. Proceeding without scaling.")
        X_scaled = df

    model = load(model_path)
    preds = model.predict(X_scaled)

    out = input_df.copy()
    out["predicted_temp_max"] = preds

    if "temperature_2m_max" in out.columns:
        out = out.rename(columns={"temperature_2m_max": "actual_temp_max"})

    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on new weather data.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw weather CSV")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output path")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--scaler", type=str, default=str(DEFAULT_SCALER))

    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    preds_df = predict(raw_df, model_path=args.model, scaler_path=args.scaler)

    preds_df.to_csv(args.output, index=False)
    LOGGER.info(f"Predictions saved to {args.output}")