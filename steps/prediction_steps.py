import logging
import pandas as pd
from pathlib import Path
from zenml import step
from pipelines.inference_pipeline.inference import predict

LOGGER = logging.getLogger(__name__)


@step
def run_batch_inference(
    df: pd.DataFrame,
    model_path: str = "models/best_weather_xgb.pkl",
    scaler_path: str = "models/tuned_scaler.pkl",
    output_path: str = "predictions.csv",
) -> pd.DataFrame:
    """Runs the full inference pipeline on raw weather data.

    Applies preprocessing, validation, feature encoding, scaling, and
    XGBoost prediction. Saves results to output_path and returns the
    predictions DataFrame.
    """
    predictions = predict(df, model_path=model_path, scaler_path=scaler_path)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out, index=False)
    LOGGER.info(f"Saved {len(predictions)} predictions to {out}")

    return predictions
