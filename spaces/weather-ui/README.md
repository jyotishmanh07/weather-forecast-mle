---
title: Weather Forecast Dashboard
emoji: 📊
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8501
pinned: false
---

# Weather Forecast Dashboard

Streamlit dashboard for the weather-forecast service. It calls the `weather-api`
Space's `/predict` endpoint (set via the `API_URL` Space secret) and charts the
next-day max-temperature forecast per city.

This Space is auto-deployed from the GitHub repo's CI; the build uses the repo's
`Dockerfile.streamlit` (copied to the Space's root `Dockerfile` at deploy time).
