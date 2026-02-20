"""
Modular script to ingest weather data from Open-Meteo API 
and split it into Train, Eval, and Holdout sets.
"""
import http.client
import json
import yaml
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)

def fetch_weather_data(config):
    """Fetches raw data from the API based on the 180-day window in your notebook."""
    hostname = config['data_ingestion']['hostname']
    conn = http.client.HTTPSConnection(hostname)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=360)
    
    LOGGER.info(f"Ingesting data from {start_date.date()} to {end_date.date()}")
    
    raw_data = pd.DataFrame()
    
    for city_key, city_info in config['cities'].items():
        lat, lon = city_info['latitude'], city_info['longitude']
        city_id = city_info['id']
        
        LOGGER.info(f"Fetching data for City ID: {city_id}")
        
        req_url = (f"/v1/archive?latitude={lat}&longitude={lon}"
                   f"&start_date={start_date.strftime('%Y-%m-%d')}"
                   f"&end_date={end_date.strftime('%Y-%m-%d')}"
                   f"&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum"
                   f"&timezone=Europe%2FLondon")
        
        conn.request("GET", req_url)
        res = conn.getresponse()
        data = json.loads(res.read().decode("utf-8"))
        
        df_city = pd.DataFrame(data['daily'])
        df_city["city"] = city_id
        raw_data = pd.concat([raw_data, df_city])

    return raw_data.reset_index(drop=True), end_date

def split_and_save(df, end_date, output_dir="data/raw"):
    """Splits data chronologically to avoid leakage and saves to CSV."""
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by=['city', 'time'])

    cutoff_eval = end_date - timedelta(days=10)
    cutoff_holdout = end_date - timedelta(days=5)

    train_df = df[df["time"] < cutoff_eval]
    eval_df = df[(df["time"] >= cutoff_eval) & (df["time"] < cutoff_holdout)]
    holdout_df = df[df["time"] >= cutoff_holdout]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_path / "train.csv", index=False)
    eval_df.to_csv(out_path / "eval.csv", index=False)
    holdout_df.to_csv(out_path / "holdout.csv", index=False)

    LOGGER.info(f"Saved splits to {out_path}. Train: {train_df.shape}, Eval: {eval_df.shape}, Holdout: {holdout_df.shape}")
    return train_df, eval_df, holdout_df

def go():
    """Main execution function."""
    try:
        config = load_config()
        raw_df, end_date = fetch_weather_data(config)
        split_and_save(raw_df, end_date)
        LOGGER.info("Data load and split pipeline completed successfully.")
    except Exception as e:
        LOGGER.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    go()