import os
import sys
import pytest
import pandas as pd
import numpy as np
import pandera.pandas as pa
from pathlib import Path
from unittest.mock import patch, MagicMock

from pipelines.feature_pipeline.preprocess import clean_weather_data, add_forecast_target
from pipelines.feature_pipeline.feature_engineering import validate_weather_data, encode_features


def _valid_processed_row() -> pd.DataFrame:
    """A single row satisfying ProcessedWeatherSchema (floats are real floats
    since the schema uses coerce=False)."""
    return pd.DataFrame({
        'city': [1],
        'weathercode': [0],
        'temperature_2m_max': [10.0],
        'temperature_2m_min': [5.0],
        'precipitation_sum': [0.0],
        'pressure_msl_mean': [1013.0],
        'wind_speed_10m_max': [15.0],
        'relative_humidity_2m_mean': [70.0],
        'cloud_cover_mean': [50.0],
        'pressure_change_1d': [-2.0],
        'lag_temp_1d': [9.0],
        'lag_temp_3d': [8.0],
        'lag_temp_7d': [7.0],
        'rolling_temp_7d_mean': [8.0],
        'rolling_temp_7d_std': [1.0],
    })

# =========================
# Preprocessing Unit Tests
# =========================

def test_clean_weather_data_logic():
    """Confirms date conversion, month-day feature, and sorting."""
    df = pd.DataFrame({
        'time': ['2026-02-01', '2026-01-01'],
        'city': [1, 1],
        'temperature_2m_max': [10.5, 5.0]
    })
    
    cleaned = clean_weather_data(df)
    
    # Check if 'time' became the index and is sorted
    assert isinstance(cleaned.index, pd.DatetimeIndex)
    assert cleaned.index[0] == pd.Timestamp('2026-01-01')
    
    # Cyclical season encoding: both components present and on the unit circle.
    assert 'season_sin' in cleaned.columns and 'season_cos' in cleaned.columns
    assert abs(cleaned.loc['2026-01-01', 'season_sin']**2
               + cleaned.loc['2026-01-01', 'season_cos']**2 - 1.0) < 1e-9

def test_clean_weather_data_deduplication():
    """Confirms duplicates and NaNs are removed."""
    df = pd.DataFrame({
        'time': ['2026-01-01', '2026-01-01', '2026-01-02'],
        'city': [1, 1, 1],
        'temperature_2m_max': [10, 10, np.nan]
    })
    
    cleaned = clean_weather_data(df)
    # Should drop the duplicate (row 2) and the NaN (row 3)
    assert len(cleaned) == 1

def test_add_forecast_target_no_leakage():
    """target_temp_max must be the max at time + horizon days (strictly future)
    per city, never the same-day value — this is the anti-leakage contract."""
    dates = ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04', '2026-01-05']
    maxes = [10.0, 11.0, 12.0, 13.0, 14.0]
    df = pd.DataFrame({
        'time': dates,
        'city': [1] * 5,
        'weathercode': [0] * 5,
        'temperature_2m_max': maxes,
        'temperature_2m_min': [1.0] * 5,
        'precipitation_sum': [0.0] * 5,
    })

    out = add_forecast_target(clean_weather_data(df)).reset_index()

    # 5 days -> h1: 4 rows, h2: 3 rows, h3: 2 rows (future targets only).
    assert len(out) == 9
    assert sorted(out['horizon'].unique()) == [1, 2, 3]

    # Every row's target equals the observed max exactly `horizon` days later.
    by_date = dict(zip(pd.to_datetime(dates), maxes))
    for _, r in out.iterrows():
        future_day = r['time'] + pd.to_timedelta(r['horizon'], unit='D')
        assert r['target_temp_max'] == by_date[future_day]

    # Today's observed max is preserved as a feature (distinct from the target).
    assert (out['target_temp_max'] != out['temperature_2m_max']).all()

# ===============================
# Feature Engineering Unit Tests
# ===============================

def test_validate_weather_data_accepts_valid():
    """A fully-valid processed row passes ProcessedWeatherSchema validation."""
    validate_weather_data(_valid_processed_row(), "Test")  # should not raise


def test_validate_weather_data_bounds():
    """Confirms ProcessedWeatherSchema rejects out-of-bounds / inconsistent data."""
    # Invalid City ID (schema requires 1 <= city <= 10)
    df_bad_city = _valid_processed_row()
    df_bad_city['city'] = [99]
    with pytest.raises(pa.errors.SchemaErrors):
        validate_weather_data(df_bad_city, "Test")

    # Max Temp < Min Temp (dataframe-level max_gte_min check)
    df_bad_temp = _valid_processed_row()
    df_bad_temp['temperature_2m_max'] = [5.0]
    df_bad_temp['temperature_2m_min'] = [10.0]
    with pytest.raises(pa.errors.SchemaErrors):
        validate_weather_data(df_bad_temp, "Test")

    # Invalid WMO weathercode (not in the allowed set)
    df_bad_wmo = _valid_processed_row()
    df_bad_wmo['weathercode'] = [999]
    with pytest.raises(pa.errors.SchemaErrors):
        validate_weather_data(df_bad_wmo, "Test")

def test_encode_features_one_hot():
    """Confirms city IDs are converted to dummy columns."""
    df = pd.DataFrame({'city': [1, 2]})
    encoded = encode_features(df)
    
    assert 'city_1' in encoded.columns
    assert 'city_2' in encoded.columns
    assert encoded['city_1'].iloc[0] == 1
    assert encoded['city_1'].iloc[1] == 0

# =========================
# Integration Mocks (go())
# =========================

@patch('pipelines.feature_pipeline.preprocess.pd.read_csv')
def test_preprocess_go_integration(mock_read, tmp_path):
    """Mocks file I/O to test the full preprocess loop."""
    from pipelines.feature_pipeline.preprocess import preprocess_splits
    
    # Create mock data
    mock_df = pd.DataFrame({
        'time': ['2026-01-01'], 'city': [1], 'temperature_2m_max': [10]
    })
    mock_read.return_value = mock_df
    
    # Create a dummy raw file so the script doesn't skip
    (tmp_path / "train.csv").touch()
    
    with patch('pipelines.feature_pipeline.preprocess.RAW_DIR', tmp_path), \
         patch('pipelines.feature_pipeline.preprocess.PROCESSED_DIR', tmp_path):
        preprocess_splits(splits=["train"])
        
        # Check if output file was created
        assert (tmp_path / "train_processed.csv").exists()