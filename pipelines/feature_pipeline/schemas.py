"""
Pandera schemas for weather feature validation.

ProcessedWeatherSchema validates data after clean_weather_data() and before
encode_features(). This is the contract between the feature pipeline and the
training / inference pipelines.
"""

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

_VALID_WMO_CODES = [0, 1, 2, 3, 45, 51, 53, 55, 61, 63, 65, 71, 73, 75, 80, 81, 82, 95, 96, 99]


class ProcessedWeatherSchema(pa.DataFrameModel):
    city: Series[int] = pa.Field(ge=1, le=10, description="City ID (1–10)")
    weathercode: Series[int] = pa.Field(
        isin=_VALID_WMO_CODES, description="WMO weather interpretation code"
    )
    temperature_2m_max: Series[float] = pa.Field(ge=-30.0, le=45.0)
    temperature_2m_min: Series[float] = pa.Field(ge=-30.0, le=45.0)
    precipitation_sum: Series[float] = pa.Field(ge=0.0, le=60.0)
    lag_temp_1d: Series[float] = pa.Field(nullable=False)
    lag_temp_3d: Series[float] = pa.Field(nullable=False)
    lag_temp_7d: Series[float] = pa.Field(nullable=False)
    rolling_temp_7d_mean: Series[float] = pa.Field(nullable=False)
    rolling_temp_7d_std: Series[float] = pa.Field(ge=0.0, nullable=False)

    @pa.dataframe_check
    @classmethod
    def max_gte_min(cls, df: pd.DataFrame) -> Series:
        return df["temperature_2m_max"] >= df["temperature_2m_min"]

    class Config:
        strict = False  # extra columns (month-day, etc.) are allowed
        coerce = False
