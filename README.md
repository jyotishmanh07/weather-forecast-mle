# Weather Forecast MLE

An end-to-end MLOps project that forecasts the **next 3 days'** maximum temperature for ten German cities with XGBoost. The point of the project is the **operations**, not the model: a closed-loop retraining system with a no-regression promotion gate and rollback, experiment tracking + a model registry, data/Pandera validation, drift monitoring, a REST API, a dashboard, and CI/CD — all on a **free** stack you can run yourself.

> Built to be cloned and run. No paid cloud account required: experiment tracking + storage on **DagsHub**, images on **GHCR**, and orchestration on a local **Apache Airflow**. The full stack runs on your machine with docker-compose.

---

## The closed loop (the part that matters)

```
                         Apache Airflow  (weather_retrain DAG)
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ ingest → preprocess → feature_engineering → dvc push → train challenger     │
   │                                                              │              │
   │                                                     evaluate gate           │
   │                                       challenger MAE ≤ 4.0  AND  ≤ champion? │
   │                                          │ yes                 │ no         │
   │                                  promote → @champion    keep current champion│
   │                                  (snapshot last_known_good)                  │
   └───────────────────────────────────────│──────────────────────────────────┘
                                            ▼
   DagsHub MLflow Registry  ──(@champion)──▶  FastAPI  :8000  ──▶  Streamlit :8501
   (tracking + model registry)                /predict /health
                                              /metrics (Prometheus)
                                              /drift   (Evidently)  ──▶ drift breach
                                                                        signals retrain

   weather_rollback DAG (manual): re-point @champion → last_known_good
```

**Cities:** Berlin, Munich, Hamburg, Frankfurt, Cologne, Stuttgart, Düsseldorf, Leipzig, Dortmund, Essen
**Target:** `temperature_2m_max` (daily max, °C) at t+1, t+2 and t+3 — one model, `horizon` as a feature

---

## Stack

| Layer | Tool |
|---|---|
| Data ingestion | Open-Meteo Archive API |
| Feature engineering | pandas, scikit-learn |
| Data validation | Pandera (`ProcessedWeatherSchema`) |
| Model training | XGBoost + Lasso baseline |
| Hyperparameter tuning | Optuna |
| Experiment tracking + Model Registry | **MLflow on DagsHub** (champion/challenger **aliases**) |
| Data & artifact versioning | **DVC** → DagsHub remote |
| Orchestration + retraining | **Apache Airflow** (docker-compose) |
| API validation | Pydantic v2 (`WeatherInput`, `PredictionResponse`) |
| Drift monitoring | Evidently (K-S + chi-square, `/drift`) |
| App metrics | Prometheus (`/metrics`) |
| Model serving | FastAPI |
| UI | Streamlit + Plotly |
| Containerisation | Docker + docker-compose |
| Container registry | **GitHub Container Registry (GHCR)** |
| Local hosting | Docker + docker-compose |
| CI/CD | GitHub Actions → GHCR |

---

## Project Structure

```
.
├── pipelines/
│   ├── feature_pipeline/
│   │   ├── load.py               # Open-Meteo API ingestion + chronological split
│   │   ├── preprocess.py         # cleaning, lag/rolling features, 3-day targets
│   │   ├── feature_engineering.py# one-hot encoding, Pandera validation
│   │   └── schemas.py            # ProcessedWeatherSchema (Pandera)
│   ├── training_pipeline/
│   │   ├── train.py              # Lasso baseline → MLflow registry
│   │   ├── tune.py               # XGBoost + Optuna → registry (@challenger)
│   │   └── backtest.py           # rolling-origin backtesting (per-horizon MAE)
│   ├── inference_pipeline/
│   │   └── inference.py          # full preprocess → predict; loads @champion
│   └── monitoring_pipeline/
│       └── drift.py              # Evidently drift check, prediction logger
├── airflow/
│   ├── docker-compose.airflow.yml# local Airflow (Postgres + LocalExecutor)
│   ├── Dockerfile.airflow        # Airflow base + app runtime deps
│   ├── dags/
│   │   ├── weather_retrain_dag.py# closed-loop retrain + gated promotion
│   │   └── weather_rollback_dag.py# manual champion rollback
│   └── README.md                 # Airflow runbook
├── tests/                        # pytest unit + integration tests
├── data/                         # DVC-tracked (raw/ + processed/); fetch with `dvc pull`
├── models/                       # local .pkl fallback (offline dev)
├── main.py                       # FastAPI app
├── app.py                        # Streamlit dashboard
├── config.yaml                   # city coords, API config
├── requirements-pipeline.txt     # app deps for the Airflow image
├── Dockerfile                    # API image
├── Dockerfile.streamlit          # UI image
├── docker-compose.yml            # local API + UI (MLflow is hosted on DagsHub)
├── .env.example                  # required environment variables
└── .github/workflows/
    └── mlops.yml                 # CI: test → build + push images to GHCR
```

---

## Quickstart

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (`pip install uv`)
- A free [DagsHub](https://dagshub.com) account (hosted MLflow + DVC remote)

### Install

```bash
uv sync
```

### Configure

```bash
cp .env.example .env.local      # fill in your DagsHub MLflow URI + token
```

Point the DVC remote at your DagsHub repo and pull the datasets:

```bash
# one-time: replace USERNAME in .dvc/config, then add credentials locally
dvc remote modify origin --local auth basic
dvc remote modify origin --local user  <DAGSHUB_USER>
dvc remote modify origin --local password <DAGSHUB_TOKEN>
dvc pull                         # fetches data/raw + data/processed
```

### Run the pipeline locally

```bash
# 1. Ingest raw data and build features (writes t+1..t+3 targets)
python -m pipelines.feature_pipeline.load
python -m pipelines.feature_pipeline.preprocess
python -m pipelines.feature_pipeline.feature_engineering

# 2. Tune + register a challenger in the DagsHub MLflow registry
python -m pipelines.training_pipeline.tune --n-trials 30

# 3. View experiments in the DagsHub MLflow UI (your repo's .mlflow URL)
```

### Start the API and UI

```bash
# Terminal 1
uvicorn main:app --reload --port 8000

# Terminal 2
streamlit run app.py
```

---

## Local Full-Stack with Docker Compose

Brings up FastAPI and Streamlit (MLflow tracking is hosted on DagsHub, so there's no local MLflow container to run):

```bash
cp .env.example .env.local       # DagsHub MLflow creds
docker-compose up --build
```

| Service | URL |
|---|---|
| API (Swagger) | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

The API loads its model from the registry; `./models` is mounted read-only as an offline fallback.

---

## API Reference

### `GET /health`

```json
{ "status": "healthy", "model_version": "weather-xgb/v9 (@champion)" }
```

### `GET /metrics`

Prometheus exposition format — HTTP request latency/counts plus `predictions_total{model_version=...}` and a `predicted_temp_max_celsius` summary. Point Prometheus/Grafana here.

### `GET /drift`

Runs an Evidently drift check comparing the training distribution against recent prediction inputs. Returns per-column K-S / chi-square p-values; a breach (`dataset_drifted: true`) is logged as the signal to retrain.

```json
{
  "model_version": "weather-xgb/v9 (@champion)",
  "reference_rows": 1755,
  "current_rows": 320,
  "dataset_drifted": false,
  "drifted_column_count": 0,
  "columns": {
    "temperature_2m_min": { "drift_score": 0.42, "threshold": 0.05, "method": "ks", "drifted": false },
    "precipitation_sum":  { "drift_score": 0.71, "threshold": 0.05, "method": "ks", "drifted": false },
    "weathercode":        { "drift_score": 0.88, "threshold": 0.05, "method": "chisquare", "drifted": false }
  }
}
```

### `POST /predict`

Accepts a JSON array of `WeatherInput` objects (today's observations) and returns a **3-day** max-temperature forecast for each — one entry per horizon (1, 2, 3 days ahead) with the `forecast_date` it applies to. All fields are Pydantic-validated; invalid weathercodes, out-of-range temperatures, or `max < min` are rejected with `422` before reaching the model.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{
    "time": "2024-06-01",
    "city": 1,
    "weathercode": 51,
    "temperature_2m_min": 12.0,
    "temperature_2m_max": 22.5,
    "precipitation_sum": 1.4,
    "pressure_msl_mean": 1016.2,
    "wind_speed_10m_max": 14.8,
    "relative_humidity_2m_mean": 64.0,
    "cloud_cover_mean": 35.0
  }]'
```

**Response:**

```json
{
  "city": [1, 1, 1],
  "time": ["2024-06-01", "2024-06-01", "2024-06-01"],
  "horizon": [1, 2, 3],
  "forecast_date": ["2024-06-02", "2024-06-03", "2024-06-04"],
  "predicted_max_temp": [21.8, 22.4, 20.9],
  "model_version": "weather-xgb/v9 (@champion)"
}
```

**`WeatherInput` field constraints:**

| Field | Type | Constraints |
|---|---|---|
| `time` | datetime | ISO 8601 (the observation day) |
| `city` | int | 1–10 |
| `weathercode` | int | WMO codes only |
| `temperature_2m_min` | float | −30 to 45 °C |
| `temperature_2m_max` | float | −30 to 45 °C, must be ≥ min — today's observed max, used as a feature |
| `precipitation_sum` | float | 0–200 mm |
| `pressure_msl_mean` | float | 940–1080 hPa |
| `wind_speed_10m_max` | float | 0–250 km/h |
| `relative_humidity_2m_mean` | float | 0–100 % |
| `cloud_cover_mean` | float | 0–100 % |

Full schema at `/docs` (Swagger UI).

---

## Dashboard: model vs. professional forecast

The Streamlit UI fetches the selected city's recent observations **live** at
view time (so the forecast origin is always the latest complete day — today's
max doesn't exist until the day ends) and shows the model's 3-day forecast next
to **Open-Meteo's professional NWP forecast** for the same dates — one delta per
tile and both curves on the chart. This is a deliberate honesty check: a
lag-feature model can extrapolate recent conditions but cannot see incoming
weather systems, so the professional forecast usually wins at day 2–3.
(Google's weather API is paid; Open-Meteo provides the same class of forecast
keylessly.)

---

## Forecasting target (no leakage)

The model predicts max temperature **1, 2 and 3 days ahead**. For each horizon
*h*, the target is `temperature_2m_max` shifted *h* days into the future within
each city (`groupby('city').shift(-h)`), and the frame is stacked with `horizon`
as an explicit feature — one observed day yields three training rows and a
single model serves all horizons (`preprocess.add_forecast_target`). Today's
observed `temperature_2m_max` remains a *legitimate feature* (it is known at
prediction time) — only the **target** is in the future. The leakage assertion in
`tests/test_feature.py` verifies every row's target is exactly its horizon days
ahead.

Other leakage protections (unchanged):

- **Chronological splits** — train ≤ *T−10d*, eval *T−10…T−5*, holdout last 5d (`load.py`).
- **Per-city lag features** — `lag_temp_{1,3,7}d` use `groupby('city').shift()`.
- **Shifted rolling windows** — `rolling_temp_7d_*` use `.rolling(7)…shift(1)`.
- **Fit on train only** — `StandardScaler` is fit on train and persisted with the model.

**Features:** today's observations (temps, precipitation, weather code) +
atmospheric state (**mean sea-level pressure and its 24h change, max wind
speed, humidity, cloud cover** — the classic front-arrival signals), per-city
lag/rolling temperature history, cyclical season encoding (sin/cos of
day-of-year), one-hot city, and the forecast `horizon`. Trained on **5 years**
of history for all ten cities.

---

## Backtesting (how accuracy is actually measured)

A single eval split is a coin flip; `backtest.py` slides a 7-day test window
over the most recent weeks (rolling-origin), trains on everything strictly
before each window — excluding rows whose *target date* falls inside it — and
averages per-horizon MAE over the folds. Fixed model parameters make runs
comparable by construction.

```bash
python -m pipelines.training_pipeline.backtest --folds 6 --test-days 7
```

Measured impact of the feature/data upgrade (same 6 test windows, Jun–Jul 2026):

| Rolling-origin MAE | 360d history, temp-only features | 5y history + atmospheric + cyclical season | Δ |
|---|---|---|---|
| Overall | 4.19 °C | **3.62 °C** | −14% |
| t+1 | 3.24 | **2.75** | −15% |
| t+2 | 4.40 | **3.92** | −11% |
| t+3 | 5.09 | **4.33** | −15% |

Error still grows with horizon — that's physics: surface observations carry
less and less signal about days further out, which is why numerical weather
models (see the dashboard comparison) stay ahead at t+2/t+3.

---

## Model Registry — champion / challenger

The registry uses **aliases** (the modern replacement for deprecated stages):

- `tune.py` registers each new version and sets `@challenger` + tag `validation_status=pending`.
- The Airflow gate promotes a challenger to `@champion` only if it passes (below).
- The API loads `models:/weather-xgb@champion`, falling back to `@challenger`, then a local `.pkl`.

```python
# Manual promotion (normally done by the retrain DAG)
from mlflow.tracking import MlflowClient
client = MlflowClient()
v = client.get_model_version_by_alias("weather-xgb", "challenger")
client.set_registered_model_alias("weather-xgb", "champion", v.version)
```

---

## Retraining & the promotion gate (Airflow)

`airflow/dags/weather_retrain_dag.py` runs weekly (Sundays 02:00, while the local Airflow stack is up):

```
ingest → preprocess → feature_engineering → dvc_push → train_challenger
        → evaluate_gate ──(pass)→ promote_champion → trigger_redeploy
                        └─(fail)→ notify_no_promote   (current champion retained)
```

**No-regression gate:** the new `@challenger` is scored on the fresh holdout and
promoted **only if** its MAE ≤ 4.0 (bar calibrated to the multi-horizon metric) **and** (no champion yet **or** its MAE ≤ the
current `@champion`'s MAE on the same data). A worse model is never promoted.
Before re-pointing, the outgoing champion's version is saved as the
`last_known_good` registered-model tag.

Run it locally:

```bash
cp airflow/.env.example airflow/.env        # AIRFLOW_UID + DagsHub creds
docker compose -f airflow/docker-compose.airflow.yml up -d --build
# open http://localhost:8080, trigger the `weather_retrain` DAG
```

See [airflow/README.md](airflow/README.md) for details.

### Rollback runbook

If a promoted model misbehaves in production, trigger the **`weather_rollback`**
DAG from the Airflow UI. It re-points `@champion` back to the `last_known_good`
version and tags it `validation_status=rolled_back`. Restart the serving
container (`docker compose restart api`) so it reloads the reverted champion.

---

## Data versioning (DVC)

`data/raw` and `data/processed` are tracked with DVC and stored on the DagsHub
remote, not in git. `dvc pull` fetches them; the retrain DAG runs
`dvc add` + `dvc push` after each ingest, so every run's data snapshot is
reproducible and reversible (commit the updated `.dvc` pointers to pin it).
Credentials live in the gitignored `.dvc/config.local`.

---

## Drift Monitoring

Every `/predict` call logs raw input features (`weathercode`,
`temperature_2m_min`, `precipitation_sum`) to `data/prediction_log.csv`. `/drift`
compares the last 500 logged rows against the training distribution with Evidently:

- **Numerical** — Kolmogorov-Smirnov (p < 0.05 = drift)
- **Categorical** — chi-square (p < 0.05 = drift)
- **Dataset drifted** — true if ≥ 50% of monitored columns drift

A breach is logged as a warning and is the signal to trigger the retrain DAG
(locally a manual/REST trigger; a hosted scheduler would use a webhook). Full
HTML reports are written to `reports/` via `drift.save_drift_report_html()`.

---

## CI/CD

### On push to `main` (`mlops.yml`)

1. `test` — run `pytest tests/`; the build is blocked if tests fail.
2. `build` — build `weather-api` (`Dockerfile`) and `weather-ui`
   (`Dockerfile.streamlit`) and push to **GHCR**
   (`ghcr.io/<owner>/weather-{api,ui}`), tagged with the commit SHA + `latest`.

There is no hosted deploy step — the stack runs locally from these images (or
from source) via docker-compose, and a hosting provider can be pointed at the
GHCR images later without workflow changes.

**Scheduled retraining** (weekly) is owned by the Airflow scheduler, not GitHub Actions.

**Required GitHub config:** none — GHCR authenticates with the built-in
`GITHUB_TOKEN`; no cloud secrets of any kind.

---

## Common Commands

```bash
# Run the test suite (same gate CI uses)
pytest tests/ -v

# Train the Lasso baseline
python -m pipelines.training_pipeline.train

# Tune XGBoost with a custom trial budget
python -m pipelines.training_pipeline.tune --n-trials 50

# Rolling-origin backtest (per-horizon MAE over recent weeks)
python -m pipelines.training_pipeline.backtest --folds 6 --test-days 7

# Batch inference on a raw CSV
python -m pipelines.inference_pipeline.inference \
  --input data/raw/holdout.csv --output data/predictions.csv

# Local API + UI
docker-compose up --build

# Local Airflow (closed-loop retraining)
docker compose -f airflow/docker-compose.airflow.yml up -d --build
```

---

## Future work (deliberately out of scope)

Scoped, not gaps — the current build is intentionally lean:

- **Longer horizons** (h = 1..7) and prediction intervals (currently h = 1..3).
- **Spatial features** — western cities' conditions today predict eastern cities tomorrow.
- **NWP post-processing** — use the professional forecast as a model input and learn to correct its local bias (how forecasts are beaten in practice).
- **Grafana** dashboards over the Prometheus `/metrics` endpoint.
- **Feature store** and **streaming ingest** instead of batch CSVs.
- **Kubernetes / IaC** and **shadow / A-B** traffic splitting for safer rollouts.

---

## Key Design Patterns

- **Closed-loop ops** — retraining, a no-regression promotion gate, and rollback are automated, not manual.
- **Registry-first serving** — images contain code only; the model is resolved from the registry (`@champion`) at startup, decoupling model releases from image builds.
- **Validation gates at every boundary** — Pandera between pipeline stages, Pydantic at the API edge, an MAE gate before any promotion.
- **Train/serve symmetry** — the fitted scaler and encoding are persisted with the model and reapplied verbatim at inference time.
- **Reproducibility** — data and model artifacts are versioned (DVC + MLflow), so any run can be reproduced or rolled back.
```
