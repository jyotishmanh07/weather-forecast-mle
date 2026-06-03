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


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced features: multi-day lags and volatility measures."""
    df = df.sort_values(['city', 'time'])
    
    df['lag_temp_1d'] = df.groupby('city')['temperature_2m_max'].shift(1)
    df['lag_temp_3d'] = df.groupby('city')['temperature_2m_max'].shift(3)
    df['lag_temp_7d'] = df.groupby('city')['temperature_2m_max'].shift(7)
    
    group = df.groupby('city')['temperature_2m_max']

    df['rolling_temp_7d_mean'] = group.transform(lambda x: x.rolling(7).mean().shift(1))
    df['rolling_temp_7d_std'] = group.transform(lambda x: x.rolling(7).std().shift(1))

    # Only the engineered lag/rolling columns are legitimately NaN for the first
    # rows of each city; fill those with 0. Do NOT fill genuine NaNs (e.g. a
    # missing target) here — those rows are dropped in clean_weather_data.
    lag_cols = ['lag_temp_1d', 'lag_temp_3d', 'lag_temp_7d',
                'rolling_temp_7d_mean', 'rolling_temp_7d_std']
    df[lag_cols] = df[lag_cols].fillna(0)
    return df

def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    if 'time' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'time'})

    df['time'] = pd.to_datetime(df['time'])
    df['month-day'] = df['time'].dt.strftime('%m.%d').astype(float)

    # Deduplicate on the natural key and drop rows missing the target BEFORE
    # engineering features, so lag/rolling values aren't computed from dupes.
    df = df.drop_duplicates(subset=['time', 'city'])
    df = df.dropna(subset=['temperature_2m_max'])

    df = add_time_series_features(df)

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