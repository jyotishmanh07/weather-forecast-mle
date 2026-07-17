"""Closed-loop retraining DAG for the weather forecast model.

Flow (manual trigger for now; nightly 02:00 once verified):

    ingest -> preprocess -> feature_engineering -> dvc_push
        -> train_challenger -> evaluate_gate
            -> promote_champion -> trigger_redeploy   (gate passes)
            -> notify_no_promote                       (gate fails)

The data/training steps reuse the existing `python -m pipelines.*` entrypoints
via BashOperator (cwd=/opt/airflow/project, repo on PYTHONPATH). The promotion
gate and champion/challenger logic are PythonOperator callables that talk to the
DagsHub-hosted MLflow registry through MlflowClient.
"""

from __future__ import annotations

import os

import pendulum

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

# Repo is mounted here by docker-compose; all module/data paths are relative to it.
PROJECT_DIR = "/opt/airflow/project"

# Must match `target_temp_max` produced by the feature pipeline (next-day max temp).
TARGET_COL = "target_temp_max"
MAE_THRESHOLD = 3.0  # absolute MAE gate, mirrors the old retrain.yml

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "weather-xgb")


# --------------------------------------------------------------------------- #
# Gate / promotion helpers
# --------------------------------------------------------------------------- #
def _score_alias_on_holdout(client, alias: str) -> float | None:
    """Load the model+scaler behind `alias` and return its holdout MAE.

    Returns None if the alias does not resolve (e.g. no champion yet).
    """
    import mlflow.xgboost
    import pandas as pd
    from joblib import load
    from sklearn.metrics import mean_absolute_error

    try:
        version = client.get_model_version_by_alias(MODEL_NAME, alias)
    except Exception:
        return None  # alias not set (no champion on a first run)

    model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}@{alias}")
    scaler_path = mlflow.artifacts.download_artifacts(
        run_id=version.run_id, artifact_path="tuned_scaler.pkl"
    )
    scaler = load(scaler_path)

    # Fresh holdout written by the feature pipeline earlier in this DAG run.
    df = pd.read_csv(os.path.join(PROJECT_DIR, "data/processed/holdout_encoded.csv"))
    y_true = df[TARGET_COL].values
    X = scaler.transform(df.drop(columns=[c for c in ("time", TARGET_COL) if c in df.columns]))
    return float(mean_absolute_error(y_true, model.predict(X)))


def evaluate_gate(**context):
    """No-regression promotion gate.

    Promote ONLY IF:
      challenger MAE <= MAE_THRESHOLD
      AND (no champion exists OR challenger MAE <= champion MAE)

    Pushes both MAEs to XCom and branches to promote_champion / notify_no_promote.
    """
    import logging

    from mlflow.tracking import MlflowClient

    log = logging.getLogger("airflow.task")
    client = MlflowClient()

    challenger_mae = _score_alias_on_holdout(client, "challenger")
    if challenger_mae is None:
        raise RuntimeError("No @challenger alias found — did train_challenger run?")

    champion_mae = _score_alias_on_holdout(client, "champion")  # None on first run

    context["ti"].xcom_push(key="challenger_mae", value=challenger_mae)
    context["ti"].xcom_push(key="champion_mae", value=champion_mae)

    passes_threshold = challenger_mae <= MAE_THRESHOLD
    beats_champion = champion_mae is None or challenger_mae <= champion_mae
    promote = passes_threshold and beats_champion

    log.info(
        "Gate: challenger_mae=%.4f champion_mae=%s threshold=%.1f -> %s",
        challenger_mae,
        f"{champion_mae:.4f}" if champion_mae is not None else "none",
        MAE_THRESHOLD,
        "PROMOTE" if promote else "HOLD",
    )
    return "promote_champion" if promote else "notify_no_promote"


def promote_champion(**context):
    """Promote the current @challenger to @champion.

    Before re-pointing, record the OUTGOING champion version as the registered-
    model tag `last_known_good` — the rollback DAG reads it. Then move the
    @champion alias to the challenger version and tag it validation_status=passed.
    """
    import logging

    from mlflow.tracking import MlflowClient

    log = logging.getLogger("airflow.task")
    client = MlflowClient()

    challenger = client.get_model_version_by_alias(MODEL_NAME, "challenger")

    # Snapshot the outgoing champion (if any) for rollback.
    try:
        current_champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
        client.set_registered_model_tag(MODEL_NAME, "last_known_good", current_champion.version)
        log.info("Recorded last_known_good=v%s (outgoing champion)", current_champion.version)
    except Exception:
        log.info("No existing champion — nothing to record as last_known_good.")

    client.set_registered_model_alias(MODEL_NAME, "champion", challenger.version)
    client.set_model_version_tag(MODEL_NAME, challenger.version, "validation_status", "passed")
    log.info("Promoted v%s -> @champion", challenger.version)


def notify_no_promote(**context):
    """Gate failed: keep the current champion (the regression protection)."""
    import logging

    log = logging.getLogger("airflow.task")
    ti = context["ti"]
    challenger_mae = ti.xcom_pull(task_ids="evaluate_gate", key="challenger_mae")
    champion_mae = ti.xcom_pull(task_ids="evaluate_gate", key="champion_mae")
    log.warning(
        "Challenger NOT promoted (challenger_mae=%.4f, champion_mae=%s, threshold=%.1f). "
        "Current champion retained.",
        challenger_mae,
        f"{champion_mae:.4f}" if champion_mae is not None else "none",
        MAE_THRESHOLD,
    )


# --------------------------------------------------------------------------- #
# DAG
# --------------------------------------------------------------------------- #
with DAG(
    dag_id="weather_retrain",
    description="Closed-loop retraining: ingest -> train -> gate -> promote/rollback.",
    # Manual trigger only while the stack is being wired up. Once an end-to-end
    # run has been verified, restore the nightly schedule: "0 2 * * *"
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=["weather", "mlops", "retrain"],
    default_args={"retries": 1},
) as dag:

    bash_env = {
        "PYTHONPATH": PROJECT_DIR,
        # Threaded through so the BashOperator tasks reach DagsHub MLflow too.
        "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", ""),
        "MLFLOW_TRACKING_USERNAME": os.getenv("MLFLOW_TRACKING_USERNAME", ""),
        "MLFLOW_TRACKING_PASSWORD": os.getenv("MLFLOW_TRACKING_PASSWORD", ""),
        "DAGSHUB_USER_TOKEN": os.getenv("DAGSHUB_USER_TOKEN", ""),
        "MLFLOW_MODEL_NAME": MODEL_NAME,
    }

    ingest = BashOperator(
        task_id="ingest",
        bash_command=f"cd {PROJECT_DIR} && python -m pipelines.feature_pipeline.load",
        env=bash_env,
        append_env=True,
    )

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command=f"cd {PROJECT_DIR} && python -m pipelines.feature_pipeline.preprocess",
        env=bash_env,
        append_env=True,
    )

    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command=f"cd {PROJECT_DIR} && python -m pipelines.feature_pipeline.feature_engineering",
        env=bash_env,
        append_env=True,
    )

    # Version the fresh data snapshot. No-op-and-continue if no DVC remote is
    # configured yet (Workstream 4 wires the DagsHub remote).
    dvc_push = BashOperator(
        task_id="dvc_push",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "(dvc push && echo 'dvc push complete') || "
            "echo 'dvc push skipped (no remote configured) — continuing'"
        ),
        env=bash_env,
        append_env=True,
    )

    # Registers a new model version, sets @challenger + validation_status=pending.
    train_challenger = BashOperator(
        task_id="train_challenger",
        bash_command=f"cd {PROJECT_DIR} && python -m pipelines.training_pipeline.tune --n-trials 30",
        env=bash_env,
        append_env=True,
    )

    gate = BranchPythonOperator(
        task_id="evaluate_gate",
        python_callable=evaluate_gate,
    )

    promote = PythonOperator(
        task_id="promote_champion",
        python_callable=promote_champion,
    )

    no_promote = PythonOperator(
        task_id="notify_no_promote",
        python_callable=notify_no_promote,
    )

    # Placeholder: tell the live serving layer to reload @champion.
    # TODO(user): replace with a real trigger for your Hugging Face Space, e.g.
    #   curl -X POST -H "Authorization: Bearer $HF_TOKEN" \
    #     "https://huggingface.co/api/spaces/<user>/<space>/restart"
    # HF Space name/token are user-specific — keep them out of the repo (use an
    # Airflow Connection/Variable or an env var sourced from .env).
    trigger_redeploy = BashOperator(
        task_id="trigger_redeploy",
        bash_command='echo "TODO: trigger HF Space rebuild so it reloads models:/<name>@champion"',
    )

    ingest >> preprocess >> feature_engineering >> dvc_push >> train_challenger >> gate
    gate >> promote >> trigger_redeploy
    gate >> no_promote
