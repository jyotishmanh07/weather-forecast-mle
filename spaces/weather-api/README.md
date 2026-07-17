---
title: Weather Forecast API
emoji: 🌦️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# Weather Forecast API

FastAPI service serving **next-day** max-temperature forecasts for five German
cities. At startup it loads the `@champion` model from the MLflow Model Registry
(hosted on DagsHub), falling back to `@challenger` then a bundled `.pkl`.

- `POST /predict` — next-day forecast for a batch of daily observations
- `GET /health` — served model version
- `GET /metrics` — Prometheus metrics
- `GET /drift` — Evidently drift report

This Space is auto-deployed from the GitHub repo's CI; the build uses the repo's
root `Dockerfile`. Source: the `weather-forecast-mle` repository.
