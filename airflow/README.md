# Airflow — closed-loop retraining

Local Apache Airflow (docker-compose) that orchestrates the weather pipelines as
a closed-loop retraining system with champion/challenger promotion and rollback.

## What's here

| File | Purpose |
|---|---|
| `Dockerfile.airflow` | Airflow base + the app's runtime deps (`../requirements-pipeline.txt`). |
| `docker-compose.airflow.yml` | Postgres + LocalExecutor Airflow; mounts the repo at `/opt/airflow/project`. |
| `dags/weather_retrain_dag.py` | Retrain + gated promotion, weekly Sundays @ 02:00 (runs while the stack is up). |
| `dags/weather_rollback_dag.py` | Manual rollback to `last_known_good`. |
| `.env.example` | Env the compose needs — copy to `.env`. |

## Start

```bash
# from the repo root
cp airflow/.env.example airflow/.env      # fill in DagsHub creds + AIRFLOW_UID (`id -u`)
docker compose -f airflow/docker-compose.airflow.yml up -d
```

Open http://localhost:8080 (login `airflow` / `airflow`). `weather_retrain`
runs weekly (Sundays 02:00) while the stack is up (unpause it in the UI); it
can also be triggered manually any time. `weather_rollback` is manual-only.

Stop / reset:

```bash
docker compose -f airflow/docker-compose.airflow.yml down        # stop
docker compose -f airflow/docker-compose.airflow.yml down -v      # + wipe the Airflow DB
```

## `weather_retrain` task graph

```
ingest -> preprocess -> feature_engineering -> dvc_push -> train_challenger -> evaluate_gate
                                                                                  |
                                                          +-----------------------+----------------+
                                                          v                                        v
                                                  promote_champion                          notify_no_promote
```

- **ingest / preprocess / feature_engineering** — the existing
  `python -m pipelines.feature_pipeline.*` entrypoints (BashOperator, cwd = repo).
- **dvc_push** — versions the fresh data snapshot; logs-and-continues if no DVC
  remote is configured yet.
- **train_challenger** — `python -m pipelines.training_pipeline.tune`; registers
  a new version and sets `@challenger`.
- **evaluate_gate** — scores `@challenger` (and `@champion`, if any) on
  `data/processed/holdout_encoded.csv`. Promotes **only if**
  `challenger MAE <= 4.0` **and** (`no champion` **or** `challenger MAE <= champion MAE`).
- **promote_champion** — records the outgoing champion version as the
  registered-model tag `last_known_good`, then re-points `@champion` to the
  challenger.
- **notify_no_promote** — keeps the current champion (regression protection).

After a promotion, restart the serving container so it reloads `@champion`
(`docker compose restart api`). A hosted provider would hook a redeploy call
here instead.

## Rollback runbook

A promoted champion is misbehaving in production. Roll back to the previous good
version:

1. Airflow UI -> DAGs -> `weather_rollback` -> Trigger DAG (it's manual,
   `schedule=None`).
2. It reads the `last_known_good` registered-model tag (set by the retrain DAG
   when it last promoted) and re-points `@champion` to that version, logging the
   before/after versions and tagging the version `validation_status=rolled_back`.
3. Restart the serving container so it reloads `@champion`
   (`docker compose restart api`).

If `last_known_good` is unset, the task fails fast (there is no prior champion to
roll back to — e.g. only one promotion has ever happened).

## TODOs for you

- **DagsHub creds** — fill `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`,
  `MLFLOW_TRACKING_PASSWORD`, `DAGSHUB_USER_TOKEN` in `airflow/.env`.
