"""Manual rollback DAG for the weather forecast model.

Re-points @champion to the version recorded as `last_known_good` by the retrain
DAG's promote step. Triggered by hand from the Airflow UI when a promoted model
turns out to be bad in production. schedule=None (never runs on a timer).
"""

from __future__ import annotations

import os

import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "weather-xgb")


def rollback_champion(**context):
    """Move @champion back to the `last_known_good` version, logging before/after."""
    import logging

    from mlflow.tracking import MlflowClient

    log = logging.getLogger("airflow.task")
    client = MlflowClient()

    rm = client.get_registered_model(MODEL_NAME)
    last_known_good = rm.tags.get("last_known_good")
    if not last_known_good:
        raise RuntimeError(
            f"No 'last_known_good' tag on model '{MODEL_NAME}' — nothing to roll back to. "
            "(The retrain DAG sets this tag only after it promotes over an existing champion.)"
        )

    try:
        before = client.get_model_version_by_alias(MODEL_NAME, "champion").version
    except Exception:
        before = "none"

    log.warning("ROLLBACK: @champion v%s -> v%s (last_known_good)", before, last_known_good)
    client.set_registered_model_alias(MODEL_NAME, "champion", last_known_good)
    client.set_model_version_tag(MODEL_NAME, last_known_good, "validation_status", "rolled_back")
    log.warning("ROLLBACK complete: @champion now v%s", last_known_good)


with DAG(
    dag_id="weather_rollback",
    description="Manually re-point @champion to the last_known_good version.",
    schedule=None,  # manual trigger only
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=["weather", "mlops", "rollback"],
) as dag:

    PythonOperator(
        task_id="rollback_champion",
        python_callable=rollback_champion,
    )
