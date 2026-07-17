import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import os
import yaml
from pathlib import Path

# ============================
# 1. Config
# ============================
# API_URL points to the FastAPI server (the api Space, or local uvicorn).
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")

# Holdout features are versioned with DVC — fetch them with `dvc pull`.
HOLDOUT_PATH = "data/processed/holdout_encoded.csv"

# Load City Names from config.yaml
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

@st.cache_data
def get_city_mapping():
    """Maps ID to Name based on your config.yaml."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        return {info['id']: name for name, info in config['cities'].items()}
    return {}

CITY_MAP = get_city_mapping()

# ============================
# 2. Data Loading & Cleaning
# ============================
@st.cache_data
def load_data():
    if not Path(HOLDOUT_PATH).exists():
        st.error(f"Holdout data not found at {HOLDOUT_PATH}. Run `dvc pull` or the feature pipeline first.")
        st.stop()
    # Load the encoded features
    data = pd.read_csv(HOLDOUT_PATH)
    
    # Ensure 'time' is a column and not an index
    data = data.reset_index()
    if "time" not in data.columns and "index" in data.columns:
        data = data.rename(columns={"index": "time"})
    
    data["time"] = pd.to_datetime(data["time"])
    
    # Reconstruct 'city' from One-Hot binary columns
    city_cols = [c for c in data.columns if c.startswith("city_")]
    if "city" not in data.columns and city_cols:
        data["city"] = data[city_cols].idxmax(axis=1).str.replace("city_", "").astype(int)
    
    data["year"] = data["time"].dt.year
    data["month"] = data["time"].dt.month
    
    return data

df = load_data()

# ============================
# 3. Streamlit UI
# ============================
st.set_page_config(page_title="Weather Forecast Explorer", page_icon="🌤️")
st.title("🌤️ Weather Forecast")

# Sidebar Diagnostics
st.sidebar.header("Data Inspector")
st.sidebar.write(f"**Date Range:** {df['time'].min().date()} to {df['time'].max().date()}")
if st.sidebar.checkbox("Show Raw Columns"):
    st.sidebar.write(df.columns.tolist())

# UI Filters with City Names
available_ids = sorted(df["city"].unique())
city_options = {city_id: f"{CITY_MAP.get(city_id, 'Unknown')} ({city_id})" for city_id in available_ids}

col1, col2, col3 = st.columns(3)
with col1:
    selected_city_id = st.selectbox(
        "Select City", 
        options=available_ids, 
        format_func=lambda x: city_options[x]
    )
with col2:
    selected_year = st.selectbox("Select Year", sorted(df["year"].unique()), index=0)
with col3:
    selected_month = st.selectbox("Select Month", sorted(df["month"].unique()), index=0)

# ============================
# 4. Prediction Execution
# ============================
if st.button("Generate Forecast 🚀"):
    # Filter for the specific city and month
    mask = (df["city"] == selected_city_id) & \
           (df["year"] == selected_year) & \
           (df["month"] == selected_month)
    
    # Sort by time to ensure lag features work in inference
    subset = df[mask].sort_values("time").copy()

    if subset.empty:
        st.warning(f"No data found for {city_options[selected_city_id]} in {selected_year}-{selected_month:02d}.")
        st.info("Check the sidebar for the available date range.")
    else:
        with st.spinner("Requesting forecast from API..."):
            # Serialize Timestamps to strings for JSON
            payload_df = subset.copy()
            payload_df['time'] = payload_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            payload = payload_df.to_dict(orient="records")

            try:
                resp = requests.post(API_URL, json=payload, timeout=60)
                resp.raise_for_status()
                out = resp.json()
                
                # Next-day forecast from the API.
                subset["prediction"] = out.get("predicted_max_temp")
                subset["forecast_date"] = pd.to_datetime(out.get("forecast_date"))
                # The holdout already contains the realised next-day max
                # (target_temp_max), so the forecast can be scored against it.
                if "target_temp_max" in subset.columns:
                    subset["actual"] = subset["target_temp_max"]

                subset = subset.dropna(subset=["prediction"])

                st.subheader(f"Next-day forecast for {CITY_MAP.get(selected_city_id, 'City ' + str(selected_city_id))}")

                has_actual = "actual" in subset.columns
                mae = (subset["prediction"] - subset["actual"]).abs().mean() if has_actual else None
                y_cols = ["actual", "prediction"] if has_actual else ["prediction"]

                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Mean Absolute Error", f"{mae:.2f}°C" if mae is not None else "n/a")
                with m2:
                    st.metric("Days Forecast", len(subset))

                # Plot the forecast (and realised actual when available) by forecast date.
                fig = px.line(
                    subset, x="forecast_date", y=y_cols, markers=True,
                    title=f"Next-day max temp — {CITY_MAP.get(selected_city_id)}",
                    labels={"value": "Temperature (°C)", "forecast_date": "Forecast date"},
                    color_discrete_map={"actual": "#1f77b4", "prediction": "#ff7f0e"}
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"API Error: {e}")
                st.info("Ensure FastAPI is running: `uvicorn main:app --reload`")