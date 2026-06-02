import pandas as pd
from zenml import step
from pipelines.feature_pipeline.load import load_config, fetch_weather_data


@step
def ingest_weather_data() -> pd.DataFrame:
    """Fetches 360 days of weather data for all configured cities from Open-Meteo."""
    config = load_config()
    raw_df, _ = fetch_weather_data(config)
    return raw_df
