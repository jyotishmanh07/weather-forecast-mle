import pandas as pd
from zenml import step
from pipelines.feature_pipeline.preprocess import clean_weather_data


@step
def preprocess_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans weather data: date conversion, deduplication, lag and rolling features."""
    return clean_weather_data(df)
