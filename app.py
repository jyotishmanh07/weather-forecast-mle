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
# API_URL points to the FastAPI server (local uvicorn or docker-compose).
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

OBS_FIELDS = [
    "weathercode", "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
    "pressure_msl_mean", "wind_speed_10m_max", "relative_humidity_2m_mean", "cloud_cover_mean",
]


@st.cache_data
def get_city_config():
    """Maps ID -> name and ID -> (lat, lon) based on your config.yaml."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        names = {info['id']: name for name, info in config['cities'].items()}
        coords = {info['id']: (info['latitude'], info['longitude'])
                  for info in config['cities'].values()}
        return names, coords
    return {}, {}

CITY_MAP, CITY_COORDS = get_city_config()


@st.cache_data(ttl=1800)
def fetch_recent_observations(lat: float, lon: float) -> pd.DataFrame:
    """Last ~14 fully-observed days for one location, fetched live.

    The dashboard doesn't read the training dataset — it pulls fresh
    observations at view time, so the forecast origin is always the latest
    complete day (usually yesterday: today's max doesn't exist until the day
    is over). Today's partial row is dropped.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily={','.join(OBS_FIELDS)}"
        "&past_days=14&forecast_days=1&timezone=Europe%2FBerlin"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    daily = r.json()["daily"]
    df = pd.DataFrame({"time": pd.to_datetime(daily["time"]),
                       **{f: daily[f] for f in OBS_FIELDS}})
    today = pd.Timestamp.now(tz="Europe/Berlin").normalize().tz_localize(None)
    df = df[df["time"] < today]          # complete days only
    return df.dropna(subset=OBS_FIELDS)  # a var can lag a day — keep clean rows


@st.cache_data(ttl=3600)
def fetch_reference_forecast(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Professional NWP forecast from Open-Meteo's free forecast API.

    Used as the reference to compare the project's model against — the same
    class of forecast weather sites (Google, etc.) display, but free and
    keyless. Cached for an hour.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&daily=temperature_2m_max"
        f"&timezone=Europe%2FBerlin&start_date={start_date}&end_date={end_date}"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    daily = r.json()["daily"]
    return pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "temp": daily["temperature_2m_max"],
    })


# ============================
# 2. Streamlit UI
# ============================
st.set_page_config(page_title="Weather Forecast Explorer", page_icon="🌤️")
st.title("🌤️ Weather Forecast")

available_ids = sorted(CITY_MAP.keys())
selected_city_id = st.selectbox(
    "Select City",
    options=available_ids,
    format_func=lambda x: CITY_MAP.get(x, f"City {x}"),
)
city_name = CITY_MAP.get(selected_city_id, f"City {selected_city_id}")

# ============================
# 3. 3-Day Forecast
# ============================
if st.button("Generate Forecast 🚀"):
    lat_lon = CITY_COORDS.get(selected_city_id)
    if lat_lon is None:
        st.error(f"No coordinates for {city_name} in config.yaml.")
        st.stop()

    with st.spinner("Fetching latest observations + requesting forecast..."):
        try:
            # Live recent history: the API recomputes lag/rolling features
            # from these rows, so the origin day gets real values.
            history = fetch_recent_observations(*lat_lon)
        except Exception as e:
            st.error(f"Could not fetch recent observations: {e}")
            st.stop()

        if history.empty:
            st.warning(f"No recent observations available for {city_name}.")
            st.stop()

        origin_ts = history["time"].iloc[-1]
        payload_df = history.copy()
        payload_df["city"] = selected_city_id
        payload_df["time"] = payload_df["time"].dt.strftime("%Y-%m-%d")
        payload = payload_df.to_dict(orient="records")

        try:
            resp = requests.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            out = resp.json()

            all_preds = pd.DataFrame({
                "time": pd.to_datetime(out["time"]),
                "forecast_date": pd.to_datetime(out["forecast_date"]),
                "horizon": out["horizon"],
                "predicted_max_temp": out["predicted_max_temp"],
            })
            # Keep only the forecasts issued from the latest observed day.
            forecast = all_preds[all_preds["time"] == origin_ts].sort_values("horizon")

            origin = origin_ts.date()

            # Reference: professional NWP forecast for the same dates.
            reference = None
            try:
                reference = fetch_reference_forecast(
                    lat_lon[0], lat_lon[1],
                    forecast["forecast_date"].min().strftime("%Y-%m-%d"),
                    forecast["forecast_date"].max().strftime("%Y-%m-%d"),
                )
            except Exception:
                st.info("Reference forecast (Open-Meteo) unavailable — showing model only.")

            st.subheader(f"3-day forecast for {city_name}")
            st.caption(
                f"Forecast origin: {origin} — the latest complete observed day, "
                "fetched live (today's max isn't known until the day ends). "
                "Delta on each tile = this model vs. Open-Meteo's professional "
                "forecast for the same date."
            )

            ref_by_date = (
                dict(zip(reference["date"], reference["temp"]))
                if reference is not None else {}
            )
            cols = st.columns(len(forecast))
            for col, (_, row) in zip(cols, forecast.iterrows()):
                ref_temp = ref_by_date.get(row["forecast_date"])
                with col:
                    st.metric(
                        row["forecast_date"].strftime("%a %d %b"),
                        f"{row['predicted_max_temp']:.1f}°C",
                        delta=(
                            f"{row['predicted_max_temp'] - ref_temp:+.1f}° vs Open-Meteo ({ref_temp:.1f}°)"
                            if ref_temp is not None else None
                        ),
                        delta_color="off",
                    )

            # Forecast comparison chart: this model vs the professional
            # reference for the same three dates.
            fc = forecast.rename(
                columns={"forecast_date": "date", "predicted_max_temp": "temp"}
            )[["date", "temp"]]
            fc["series"] = "this model"
            chart_df = fc

            if reference is not None and not reference.empty:
                ref = reference.copy()
                ref["series"] = "Open-Meteo forecast"
                chart_df = pd.concat([chart_df, ref], ignore_index=True)

            fig = px.line(
                chart_df, x="date", y="temp", color="series", markers=True,
                title=f"3-day max-temperature forecast — {city_name}",
                labels={"temp": "Temperature (°C)", "date": "Date"},
                color_discrete_map={
                    "this model": "#ff7f0e",
                    "Open-Meteo forecast": "#2ca02c",
                },
            )
            fig.update_xaxes(tickformat="%a %d %b", dtick="D1")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"API Error: {e}")
            st.info("Ensure FastAPI is running: `uvicorn main:app --reload`")
