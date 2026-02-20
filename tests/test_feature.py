import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from pipelines.feature_pipeline.preprocess import clean_weather_data
from pipelines.feature_pipeline.feature_engineering import validate_weather_data, encode_features

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

# ===============================
# Feature Engineering Unit Tests
# ===============================

def test_validate_weather_data_bounds():
    """Confirms strict validation from 02_feature_engg.ipynb."""
    # Test invalid City ID
    df_bad_city = pd.DataFrame({'city': [99], 'temperature_2m_max': [10], 'temperature_2m_min': [5], 'precipitation_sum': [0], 'weathercode': [0]})
    with pytest.raises(AssertionError, match="City ID out of range"):
        validate_weather_data(df_bad_city, "Test")

    # Test Max Temp < Min Temp
    df_bad_temp = pd.DataFrame({'city': [1], 'temperature_2m_max': [5], 'temperature_2m_min': [10], 'precipitation_sum': [0], 'weathercode': [0]})
    with pytest.raises(AssertionError, match="Found Min Temp > Max Temp"):
        validate_weather_data(df_bad_temp, "Test")

    # Test Invalid WMO Code
    df_bad_wmo = pd.DataFrame({'city': [1], 'temperature_2m_max': [10], 'temperature_2m_min': [5], 'precipitation_sum': [0], 'weathercode': [999]})
    with pytest.raises(AssertionError, match="Found an invalid WMO weathercode"):
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