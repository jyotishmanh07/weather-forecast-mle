"""
Feature Engineering Script for Weather Forecasting.
- Validates weather data logic (temp bounds, WMO codes).
- Performs One-Hot Encoding on City IDs.
- Saves engineered datasets to data/processed/.
"""

import pandas as pd
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")

def validate_weather_data(df: pd.DataFrame, dataset_name: str):
    # City Check (Must be 1-10)
    assert df['city'].between(1, 10).all(), f"[{dataset_name}] City ID out of range! Must be 1-10."
    
    # Temperature Checks (-30 to 45 degrees)
    assert df['temperature_2m_max'].between(-30, 45).all(), f"[{dataset_name}] Max temp out of logical bounds!"
    assert df['temperature_2m_min'].between(-30, 45).all(), f"[{dataset_name}] Min temp out of logical bounds!"
    
    # Logic Check: Max temp must always be >= Min temp
    assert (df['temperature_2m_max'] >= df['temperature_2m_min']).all(), f"[{dataset_name}] Found Min Temp > Max Temp!"
    
    # Precipitation Check (0.0 to 60.0)
    assert df['precipitation_sum'].between(0.0, 60.0).all(), f"[{dataset_name}] Precipitation out of bounds!"
    
    # Valid WMO Weathercodes Check
    valid_codes = [0, 1, 2, 3, 45, 51, 53, 55, 61, 63, 65, 71, 73, 75, 80, 81, 82, 95, 96, 99]
    assert df['weathercode'].isin(valid_codes).all(), f"[{dataset_name}] Found an invalid WMO weathercode!"
    
    LOGGER.info(f"{dataset_name} passed all validation checks.")

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies One-Hot Encoding to categorical features.
    Converts city IDs into separate binary columns.
    """
    df_encoded = df.copy()
    df_encoded = pd.get_dummies(df_encoded, columns=['city'], drop_first=False, dtype=int)
    return df_encoded

def run_feature_engineering(splits=("train", "eval", "holdout")):
    """Pipeline to load, validate, encode, and save weather features."""
    
    for split in splits:
        input_path = PROCESSED_DIR / f"{split}_processed.csv"
        
        if not input_path.exists():
            LOGGER.warning(f"File {input_path} not found. Skipping...")
            continue

        # Load processed data
        df = pd.read_csv(input_path, index_col="time", parse_dates=True)
        
        # Validation
        validate_weather_data(df, split.capitalize())
        
        # Encoding
        df_eng = encode_features(df)
        
        # Save
        output_path = PROCESSED_DIR / f"{split}_encoded.csv"
        df_eng.to_csv(output_path, index=True)
        
        LOGGER.info(f"Successfully engineered {split}. Columns: {df_eng.columns.tolist()}")

def go():
    """Main execution function."""
    try:
        run_feature_engineering()
        LOGGER.info("Feature engineering pipeline completed successfully.")
    except Exception as e:
        LOGGER.error(f"Feature engineering failed: {e}")
        raise

if __name__ == "__main__":
    go()