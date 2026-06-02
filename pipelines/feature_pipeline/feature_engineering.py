"""
Feature Engineering Script for Weather Forecasting.
- Validates weather data logic (temp bounds, WMO codes).
- Performs One-Hot Encoding on City IDs.
- Saves engineered datasets to data/processed/.
"""

import pandas as pd
import pandera.pandas as pa
import logging
from pathlib import Path

from pipelines.feature_pipeline.schemas import ProcessedWeatherSchema

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")

def validate_weather_data(df: pd.DataFrame, dataset_name: str) -> None:
    try:
        ProcessedWeatherSchema.validate(df, lazy=True)
        LOGGER.info(f"{dataset_name} passed all validation checks.")
    except pa.errors.SchemaErrors as exc:
        failure_cases = exc.failure_cases[["check", "column", "failure_case", "index"]]
        LOGGER.error(f"[{dataset_name}] Schema validation failed:\n{failure_cases.to_string()}")
        raise

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