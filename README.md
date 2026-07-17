# Weather Forecast MLE

An end-to-end MLOps project that forecasts **next-day** maximum temperature for five German cities with XGBoost. The point of the project is the **operations**, not the model: a closed-loop retraining system with a no-regression promotion gate and rollback, experiment tracking + a model registry, data/Pandera validation, drift monitoring, a REST API, a dashboard, and CI/CD — all on a **free** stack you can run yourself.

> Built to be cloned and run. No paid cloud account required: experiment tracking + storage on **DagsHub**, images on **GHCR**, and orchestration on a local **Apache Airflow**. The full stack runs on your machine with docker-compose.

---

## The closed loop (the part that matters)

```
                         Apache Airflow  (weather_retrain DAG)
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ ingest → preprocess → feature_engineering → dvc push → train challenger     │
   │                                                              │              │
   │                                                     evaluate gate           │
   │                                       challenger MAE ≤ 3.0  AND  ≤ champion? │
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

**Cities:** Berlin, Munich, Hamburg, Frankfurt, Cologne
**Target:** next-day `temperature_2m_max` (daily maximum temperature, °C)

> The committed `mle_flowchart.png` shows the previous AWS topology and is being refreshed — the diagram above is the current source of truth.

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
│   │   ├── preprocess.py         # cleaning, lag/rolling features, next-day target
│   │   ├── feature_engineering.py# one-hot encoding, Pandera validation
│   │   └── schemas.py            # ProcessedWeatherSchema (Pandera)
│   ├── training_pipeline/
│   │   ├── train.py              # Lasso baseline → MLflow registry
│   │   └── tune.py               # XGBoost + Optuna → registry (@challenger)
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
# 1. Ingest raw data and build features (writes the next-day target)
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
{ "status": "healthy", "model_version": "weather-xgb/v3 (@champion)" }
```

### `GET /metrics`

Prometheus exposition format — HTTP request latency/counts plus `predictions_total{model_version=...}` and a `predicted_temp_max_celsius` summary. Point Prometheus/Grafana here.

### `GET /drift`

Runs an Evidently drift check comparing the training distribution against recent prediction inputs. Returns per-column K-S / chi-square p-values; a breach (`dataset_drifted: true`) is logged as the signal to retrain.

```json
{
  "model_version": "weather-xgb/v3 (@champion)",
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

Accepts a JSON array of `WeatherInput` objects (today's observations) and returns the **next-day** max-temperature forecast for each, with the `forecast_date` it applies to. All fields are Pydantic-validated; invalid weathercodes, out-of-range temperatures, or `max < min` are rejected with `422` before reaching the model.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{
    "time": "2024-06-01",
    "city": 1,
    "weathercode": 51,
    "temperature_2m_min": 12.0,
    "temperature_2m_max": 22.5,
    "precipitation_sum": 1.4
  }]'
```

**Response:**

```json
{
  "city": [1],
  "time": ["2024-06-01"],
  "forecast_date": ["2024-06-02"],
  "predicted_max_temp": [21.8],
  "model_version": "weather-xgb/v3 (@champion)"
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
| `precipitation_sum` | float | 0–60 mm |

Full schema at `/docs` (Swagger UI).

---

## Forecasting target (no leakage)

The model predicts the **next day's** max temperature. The target is
`temperature_2m_max` shifted backwards one day within each city
(`groupby('city').shift(-1)`), built in `preprocess.add_forecast_target`. Today's
observed `temperature_2m_max` remains a *legitimate feature* (it is known at
prediction time) — only the **target** is in the future. The leakage assertion in
`tests/test_feature.py` guards this contract.

Other leakage protections (unchanged):

- **Chronological splits** — train ≤ *T−10d*, eval *T−10…T−5*, holdout last 5d (`load.py`).
- **Per-city lag features** — `lag_temp_{1,3,7}d` use `groupby('city').shift()`.
- **Shifted rolling windows** — `rolling_temp_7d_*` use `.rolling(7)…shift(1)`.
- **Fit on train only** — `StandardScaler` is fit on train and persisted with the model.

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

`airflow/dags/weather_retrain_dag.py` (manual trigger while the stack is being
verified; designed to run nightly at 02:00):

```
ingest → preprocess → feature_engineering → dvc_push → train_challenger
        → evaluate_gate ──(pass)→ promote_champion → trigger_redeploy
                        └─(fail)→ notify_no_promote   (current champion retained)
```

**No-regression gate:** the new `@challenger` is scored on the fresh holdout and
promoted **only if** its MAE ≤ 3.0 **and** (no champion yet **or** its MAE ≤ the
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
remote, not in git. `dvc pull` fetches them; the retrain DAG runs `dvc push`
after each ingest, so every run's data snapshot is reproducible and reversible
(this replaces the old S3 dated-archive). Credentials live in the gitignored
`.dvc/config.local`.

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

**Scheduled retraining** is owned by the Airflow scheduler, not GitHub Actions.

**Required GitHub config:** none — GHCR authenticates with the built-in
`GITHUB_TOKEN`. (No more AWS secrets.)

---

## Common Commands

```bash
# Run the test suite (same gate CI uses)
pytest tests/ -v

# Train the Lasso baseline
python -m pipelines.training_pipeline.train

# Tune XGBoost with a custom trial budget
python -m pipelines.training_pipeline.tune --n-trials 50

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

- **Multi-horizon forecasting** (h = 1..7) with rolling-origin backtesting and prediction intervals.
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
