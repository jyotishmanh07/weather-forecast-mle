# Hugging Face Spaces (live demo)

The live demo runs as **two Docker SDK Spaces** that are deployed automatically
by `.github/workflows/mlops.yml` on every push to `main`.

| Space | Builds from | Port | Purpose |
|---|---|---|---|
| `weather-api` | repo root `Dockerfile` | 8000 | FastAPI — loads the `@champion` model from the MLflow registry |
| `weather-ui`  | repo root `Dockerfile.streamlit` | 8501 | Streamlit dashboard, calls the api Space |

## How deployment works

HF Docker Spaces always build from the **root `Dockerfile`** and read their
Space front-matter from the **root `README.md`**. Since both Spaces come from
this one repo, the `deploy` job in `mlops.yml` prepares those two root files
per Space before pushing:

- **weather-api** → root `Dockerfile` is already the API image; root `README.md`
  is replaced with [`spaces/weather-api/README.md`](weather-api/README.md).
- **weather-ui** → root `Dockerfile` is replaced with `Dockerfile.streamlit`;
  root `README.md` is replaced with [`spaces/weather-ui/README.md`](weather-ui/README.md).

## One-time setup (user)

1. Create two **Docker** SDK Spaces on Hugging Face:
   `huggingface.co/spaces/<HF_USERNAME>/weather-api` and `.../weather-ui`.
2. In the GitHub repo settings add:
   - secret **`HF_TOKEN`** — a write-scoped HF access token.
   - variable **`HF_USERNAME`** — your HF account name.
3. On the **weather-api** Space, add these Space secrets (so it can reach the
   MLflow registry at startup):
   `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`,
   `MLFLOW_MODEL_NAME`.
4. On the **weather-ui** Space, set `API_URL` to the api Space's public
   `/predict` URL (e.g. `https://<HF_USERNAME>-weather-api.hf.space/predict`).

Free Spaces idle-sleep when unused; the first request after a nap cold-starts.
