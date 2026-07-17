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
    
    # Check feature engineering from notebook
    assert 'month-day' in cleaned.columns
    assert cleaned.loc['2026-01-01', 'month-day'] == 1.01

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
    """target_temp_max must be the NEXT day's max (strictly future) per city,
    never the same-day value — this is the anti-leakage contract."""
    df = pd.DataFrame({
        'time': ['2026-01-01', '2026-01-02', '2026-01-03'],
        'city': [1, 1, 1],
        'weathercode': [0, 0, 0],
        'temperature_2m_max': [10.0, 11.0, 12.0],
        'temperature_2m_min': [1.0, 2.0, 3.0],
        'precipitation_sum': [0.0, 0.0, 0.0],
    })

    out = add_forecast_target(clean_weather_data(df))

    # The last day per city has no next-day target and is dropped.
    assert len(out) == 2
    # Each row carries the FOLLOWING day's max as its target.
    assert out.loc['2026-01-01', 'target_temp_max'] == 11.0
    assert out.loc['2026-01-02', 'target_temp_max'] == 12.0
    # Today's observed max is preserved as a feature (distinct from the target).
    assert out.loc['2026-01-01', 'temperature_2m_max'] == 10.0
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