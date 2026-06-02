import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from zenml import step
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LOGGER = logging.getLogger(__name__)

TARGET = "temperature_2m_max"


@step
def evaluate_weather_model(
    eval_df: pd.DataFrame,
    model_path: str = "models/best_weather_xgb.pkl",
    scaler_path: str = "models/tuned_scaler.pkl",
    mae_threshold: float = 3.0,
) -> bool:
    """Evaluates the trained XGBoost model on the eval split.

    Logs MAE, RMSE, and R² to stdout. Returns True if MAE is within
    mae_threshold — this boolean gates the deployment step.
    """
    model_file = Path(model_path)
    scaler_file = Path(scaler_path)

    if not model_file.exists() or not scaler_file.exists():
        raise FileNotFoundError(f"Artifacts not found: {model_path}, {scaler_path}")

    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)

    df = eval_df.copy()
    if "time" in df.columns:
        df = df.drop(columns=["time"])

    y_true = df[TARGET].values
    X = df.drop(columns=[TARGET])
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)

    LOGGER.info(f"Eval metrics — MAE: {mae:.4f}  RMSE: {rmse:.4f}  R²: {r2:.4f}")
    LOGGER.info(f"Deploy decision: MAE {mae:.4f} {'<=' if mae <= mae_threshold else '>'} threshold {mae_threshold}")

    return bool(mae <= mae_threshold)
