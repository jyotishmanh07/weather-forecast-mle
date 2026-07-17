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

def add_forecast_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Create the next-day forecast target.

    The model forecasts a city's max temperature `horizon` day(s) ahead, so the
    target is temperature_2m_max shifted backwards within each city. Today's
    observed temperature_2m_max stays as a legitimate feature (it is known at
    prediction time). Rows whose future target falls outside this split (the
    last day per city) are dropped.

    This is applied only in the training feature pipeline — never in
    clean_weather_data, which the inference pipeline reuses on data that has no
    future target to learn from.
    """
    df = df.reset_index()  # 'time' index -> column so we can sort by [city, time]
    df = df.sort_values(['city', 'time'])
    df['target_temp_max'] = df.groupby('city')['temperature_2m_max'].shift(-horizon)
    df = df.dropna(subset=['target_temp_max'])
    return df.set_index('time').sort_index()

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
        cleaned_df = add_forecast_target(cleaned_df)

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