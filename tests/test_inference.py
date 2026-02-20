import sys
import os
from pathlib import Path
import pandas as pd
import pytest

# Add project root to sys.path to ensure 'pipelines' is discoverable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Import the weather-specific predict function
from pipelines.inference_pipeline.inference import predict

@pytest.fixture(scope="session")
def sample_weather_df():
    """Load a small sample from RAW holdout data for inference testing."""
    # Use the raw file that still has the 'city' column
    sample_path = ROOT / "data/raw/holdout.csv"
    
    if not sample_path.exists():
        pytest.skip(f"Raw data not found at {sample_path}.")
        
    df = pd.read_csv(sample_path).sample(5, random_state=42).reset_index(drop=True)
    return df

def test_inference_runs_and_returns_temp_predictions(sample_weather_df):
    """Ensure weather inference pipeline runs and returns the correct prediction column."""
    # The predict function handles preprocessing, scaling, and XGBoost inference
    preds_df = predict(sample_weather_df)

    # 1. Check output is not empty
    assert not preds_df.empty

    # 2. Verify the specific output column for weather forecasting
    assert "predicted_temp_max" in preds_df.columns

    # 3. Predictions should be numeric floats
    assert pd.api.types.is_numeric_dtype(preds_df["predicted_temp_max"])

    # 4. Logical Check: Predicted temperatures should be within a reasonable range
    assert preds_df["predicted_temp_max"].between(-40, 50).all()

    print("Weather inference pipeline test passed. Sample Predictions:")
    print(preds_df[["predicted_temp_max"]].head())