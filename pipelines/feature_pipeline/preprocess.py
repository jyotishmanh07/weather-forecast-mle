"""
Preprocessing script for Weather Forecasting.
- Cleans weather data: date conversion and deduplication.
- Feature Engineering: creates 'month-day' column.
- Saves processed splits to data/processed/.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


ATMOS_COLS = ['pressure_msl_mean', 'wind_speed_10m_max',
              'relative_humidity_2m_mean', 'cloud_cover_mean']

def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced features: multi-day lags and volatility measures."""
    df = df.sort_values(['city', 'time'])

    df['lag_temp_1d'] = df.groupby('city')['temperature_2m_max'].shift(1)
    df['lag_temp_3d'] = df.groupby('city')['temperature_2m_max'].shift(3)
    df['lag_temp_7d'] = df.groupby('city')['temperature_2m_max'].shift(7)

    group = df.groupby('city')['temperature_2m_max']

    df['rolling_temp_7d_mean'] = group.transform(lambda x: x.rolling(7).mean().shift(1))
    df['rolling_temp_7d_std'] = group.transform(lambda x: x.rolling(7).std().shift(1))

    # Atmospheric columns: fill occasional archive gaps within each city from
    # neighbouring days, then derive the 24h pressure change — a falling
    # pressure + wind shift is the classic signature of an approaching front.
    present_atmos = [c for c in ATMOS_COLS if c in df.columns]
    if present_atmos:
        # float cast: the API serialises humidity/cloud as integers, but the
        # schema (coerce=False) expects real floats.
        df[present_atmos] = df[present_atmos].astype(float)
        df[present_atmos] = df.groupby('city')[present_atmos].transform(
            lambda s: s.ffill().bfill()
        )
    if 'pressure_msl_mean' in df.columns:
        df['pressure_change_1d'] = df.groupby('city')['pressure_msl_mean'].diff()

    # Only the engineered lag/rolling columns are legitimately NaN for the first
    # rows of each city; fill those with 0. Do NOT fill genuine NaNs (e.g. a
    # missing target) here — those rows are dropped in clean_weather_data.
    lag_cols = ['lag_temp_1d', 'lag_temp_3d', 'lag_temp_7d',
                'rolling_temp_7d_mean', 'rolling_temp_7d_std']
    if 'pressure_change_1d' in df.columns:
        lag_cols.append('pressure_change_1d')
    df[lag_cols] = df[lag_cols].fillna(0)
    return df

FORECAST_HORIZONS = (1, 2, 3)

def add_forecast_target(df: pd.DataFrame, horizons: tuple[int, ...] = FORECAST_HORIZONS) -> pd.DataFrame:
    """Create multi-horizon forecast targets (t+1 .. t+3 by default).

    For each horizon h, a copy of the frame gets target_temp_max =
    temperature_2m_max shifted h days into the future within each city, plus a
    `horizon` feature column — so one observed day yields one training row per
    horizon and a single model learns all horizons. Today's observed
    temperature_2m_max stays a legitimate feature (it is known at prediction
    time). Rows whose future target falls outside this split are dropped.

    This is applied only in the training feature pipeline — never in
    clean_weather_data, which the inference pipeline reuses on data that has no
    future target to learn from.
    """
    df = df.reset_index()  # 'time' index -> column so we can sort by [city, time]
    df = df.sort_values(['city', 'time'])
    frames = []
    for h in horizons:
        fh = df.copy()
        fh['horizon'] = h
        fh['target_temp_max'] = fh.groupby('city')['temperature_2m_max'].shift(-h)
        frames.append(fh)
    out = pd.concat(frames, ignore_index=True).dropna(subset=['target_temp_max'])
    return out.set_index('time').sort_index()

def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    if 'time' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'time'})

    df['time'] = pd.to_datetime(df['time'])
    # Cyclical season encoding: Dec 31 and Jan 1 are neighbours on a circle,
    # not 12 units apart (which is what the old month.day float implied).
    day_of_year = df['time'].dt.dayofyear
    df['season_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df['season_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

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