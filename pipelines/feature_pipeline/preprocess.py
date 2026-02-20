"""
Preprocessing script for Weather Forecasting.
- Cleans weather data: date conversion and deduplication.
- Feature Engineering: creates 'month-day' column.
- Saves processed splits to data/processed/.
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    if 'time' not in df.columns:
        df = df.reset_index()
        if 'time' not in df.columns:
            df.rename(columns={'index': 'time'}, inplace=True)
    
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d')

    df['month-day'] = df['time'].dt.strftime('%m.%d').astype(float)

    df = df.dropna().drop_duplicates()

    df.set_index('time', inplace=True)
    df = df.sort_index()
    
    return df

def preprocess_splits(splits=("train", "eval", "holdout")):
    """Iterates through data splits, cleans them, and saves to processed directory."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for split in splits:
        input_path = RAW_DIR / f"{split}.csv"
        
        if not input_path.exists():
            LOGGER.warning(f"File {input_path} not found. Skipping...")
            continue

        LOGGER.info(f"Cleaning {split} data...")
        df = pd.read_csv(input_path)
        
        cleaned_df = clean_weather_data(df)

        output_path = PROCESSED_DIR / f"{split}_processed.csv"
        cleaned_df.to_csv(output_path, index=True)
        
        LOGGER.info(f"Successfully processed {split}. Shape: {cleaned_df.shape}")

def go():
    """Main execution function."""
    try:
        preprocess_splits()
        LOGGER.info("Preprocessing pipeline completed successfully.")
    except Exception as e:
        LOGGER.error(f"Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    go()