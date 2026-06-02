import pandas as pd
from zenml import step
from datetime import datetime, timedelta


@step
def split_weather_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits raw weather data chronologically to avoid look-ahead bias.

    Returns train, eval, holdout DataFrames using fixed time offsets from now:
    - holdout: last 5 days
    - eval:    days 6–10 from the end
    - train:   everything before eval
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(by=["city", "time"])

    end_date = datetime.now()
    cutoff_eval = end_date - timedelta(days=10)
    cutoff_holdout = end_date - timedelta(days=5)

    train_df = df[df["time"] < cutoff_eval].reset_index(drop=True)
    eval_df = df[(df["time"] >= cutoff_eval) & (df["time"] < cutoff_holdout)].reset_index(drop=True)
    holdout_df = df[df["time"] >= cutoff_holdout].reset_index(drop=True)

    return train_df, eval_df, holdout_df
