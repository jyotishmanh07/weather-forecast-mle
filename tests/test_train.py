import math
import joblib
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from pipelines.training_pipeline.train import load_and_preprocess, train_model
from pipelines.training_pipeline.tune import tune_weather_model


def _synthetic_encoded(n_rows: int, start: str, seed: int) -> pd.DataFrame:
    """A small frame matching the *_encoded.csv schema the pipelines produce.

    Real data is DVC-tracked and absent in CI, so tests generate their own.
    Some precipitation values are NaN on purpose to exercise the NaN handling.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_rows, freq="D")
    temp = 15 + 8 * np.sin(np.arange(n_rows) / 30) + rng.normal(0, 2, n_rows)
    return pd.DataFrame({
        "time": dates,
        "weathercode": rng.choice([0, 1, 2, 3, 61], n_rows),
        "temperature_2m_max": temp.round(1),
        "temperature_2m_min": (temp - rng.uniform(3, 8, n_rows)).round(1),
        "precipitation_sum": np.where(
            rng.random(n_rows) < 0.2, np.nan, rng.uniform(0, 10, n_rows).round(1)
        ),
        "month-day": dates.strftime("%m.%d").astype(float),
        "lag_temp_1d": np.roll(temp, 1).round(1),
        "lag_temp_3d": np.roll(temp, 3).round(1),
        "lag_temp_7d": np.roll(temp, 7).round(1),
        "rolling_temp_7d_mean": pd.Series(temp).rolling(7, min_periods=1).mean().round(1),
        "rolling_temp_7d_std": pd.Series(temp).rolling(7, min_periods=1).std().fillna(0).round(1),
        # next-day target: tomorrow's max, i.e. temp shifted back one day
        "target_temp_max": np.roll(temp, -1).round(1),
        "city_1": 1, "city_2": 0, "city_3": 0, "city_4": 0, "city_5": 0,
    })


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    """Writes synthetic train/eval encoded CSVs and returns their directory."""
    d = tmp_path_factory.mktemp("processed")
    _synthetic_encoded(120, "2026-01-01", seed=42).to_csv(d / "train_encoded.csv", index=False)
    _synthetic_encoded(20, "2026-05-01", seed=7).to_csv(d / "eval_encoded.csv", index=False)
    return d


def _assert_metrics(m):
    """Ensures metrics exist and are finite numbers."""
    assert set(m.keys()) >= {"mae", "rmse", "r2"}
    assert all(isinstance(v, (float, np.float64)) and math.isfinite(v) for v in m.values())


def test_load_and_preprocess_handles_nans(data_dir):
    """Confirms NaN filling for precipitation works."""
    X_train, y_train, X_eval, y_eval = load_and_preprocess(
        data_dir / "train_encoded.csv", data_dir / "eval_encoded.csv"
    )

    assert not X_train['precipitation_sum'].isnull().any()
    assert not X_eval['precipitation_sum'].isnull().any()
    assert len(X_train) > 0

def test_train_model_saves_artifacts(data_dir):
    """Verifies that both the Lasso model and the Scaler are created."""
    X_train, y_train, X_eval, y_eval = load_and_preprocess(
        data_dir / "train_encoded.csv", data_dir / "eval_encoded.csv"
    )

    model, scaler, metrics = train_model(X_train, y_train, X_eval, y_eval, alpha=0.1)

    _assert_metrics(metrics)
    assert model is not None
    assert scaler is not None

    X_train_scaled = scaler.transform(X_train)
    assert math.isclose(X_train_scaled.mean(), 0, abs_tol=1e-1)

def test_tune_weather_model_integration(data_dir, tmp_path):
    """Runs a 2-trial tune to ensure the Optuna-MLflow loop is solid."""
    import mlflow

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("pipelines.training_pipeline.tune.MODELS_DIR", tmp_path)
        mp.setattr("pipelines.training_pipeline.tune.PROCESSED_DIR", data_dir)

        # Isolate MLflow tracking + registry to a temp store so the test never
        # touches ambient state (e.g. a committed mlflow.db with absolute paths).
        tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        mp.setenv("MLFLOW_TRACKING_URI", tracking_uri)
        mp.setenv("MLFLOW_REGISTRY_URI", tracking_uri)
        prev_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        # New experiment writes artifacts under tmp_path instead of cwd.
        mlflow.create_experiment(
            "ci_test_tuning", artifact_location=(tmp_path / "artifacts").as_uri()
        )
        try:
            tune_weather_model(n_trials=2, experiment_name="ci_test_tuning")

            assert (tmp_path / "best_weather_xgb.pkl").exists()
            assert (tmp_path / "tuned_scaler.pkl").exists()

            # The run must register a version and point @challenger at it.
            client = mlflow.tracking.MlflowClient()
            v = client.get_model_version_by_alias("weather-xgb", "challenger")
            assert v.version is not None
        finally:
            mlflow.set_tracking_uri(prev_uri)

def test_model_prediction_range(data_dir):
    """Safety check: ensures the model isn't predicting impossible temperatures."""
    X_train, y_train, X_eval, y_eval = load_and_preprocess(
        data_dir / "train_encoded.csv", data_dir / "eval_encoded.csv"
    )
    model, scaler, _ = train_model(X_train, y_train, X_eval, y_eval)

    X_eval_scaled = scaler.transform(X_eval)
    preds = model.predict(X_eval_scaled)

    assert (preds > -40).all() and (preds < 55).all()
