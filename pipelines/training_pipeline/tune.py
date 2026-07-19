from __future__ import annotations
import logging
import os
import tempfile
import joblib
import mlflow
import mlflow.xgboost
import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REGISTRY_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "weather-xgb")

def _load_and_prepare(train_path, eval_path, target='target_temp_max'):
    train_df = pd.read_csv(train_path, index_col="time", parse_dates=True)
    eval_df = pd.read_csv(eval_path, index_col="time", parse_dates=True)

    cols_to_fix = [
        'precipitation_sum', 
        'lag_temp_1d', 'lag_temp_3d', 'lag_temp_7d', 
        'rolling_temp_7d_mean', 'rolling_temp_7d_std'
    ]
    
    for col in cols_to_fix:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna(0)
            eval_df[col] = eval_df[col].fillna(0)

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_eval = eval_df.drop(columns=[target])
    y_eval = eval_df[target]
    
    return X_train, y_train, X_eval, y_eval

def tune_weather_model(n_trials: int = 30, experiment_name: str = "weather_xgb_lags_tuning"):
    mlflow.set_experiment(experiment_name)
    
    X_train, y_train, X_eval, y_eval = _load_and_prepare(
        PROCESSED_DIR / "train_encoded.csv",
        PROCESSED_DIR / "eval_encoded.csv"
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    def objective(trial: optuna.Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
            "tree_method": "hist"
        }

        with mlflow.start_run(nested=True):
            model = XGBRegressor(**params)
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_eval_scaled)
            mae = mean_absolute_error(y_eval, preds)
            
            mlflow.log_params(params)
            mlflow.log_metric("mae", mae)
        return mae

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Retrain final best model and register in MLflow Model Registry
    with mlflow.start_run(run_name="best_xgb_with_lags") as run:
        best_model = XGBRegressor(**study.best_params, tree_method="hist")
        best_model.fit(X_train_scaled, y_train)

        preds = best_model.predict(X_eval_scaled)
        final_metrics = {
            "mae": mean_absolute_error(y_eval, preds),
            "rmse": np.sqrt(mean_squared_error(y_eval, preds)),
            "r2": r2_score(y_eval, preds),
        }
        # Per-horizon MAE (t+1 .. t+3) so degradation with lead time is visible.
        if "horizon" in X_eval.columns:
            for h in sorted(X_eval["horizon"].unique()):
                m = (X_eval["horizon"] == h).values
                final_metrics[f"mae_h{int(h)}"] = mean_absolute_error(y_eval[m], preds[m])
        mlflow.log_params({**study.best_params, "n_trials": n_trials})
        mlflow.log_metrics(final_metrics)

        # Save locally (dev fallback)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODELS_DIR / "best_weather_xgb.pkl")
        joblib.dump(scaler, MODELS_DIR / "tuned_scaler.pkl")

        # Log model + scaler as plain run artifacts. MLflow 3's log_model()
        # creates a LoggedModel entity that remote registries like DagsHub can't
        # resolve at registration time, so save the MLmodel directory locally
        # and attach it to the run instead.
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            mlflow.xgboost.save_model(best_model, path=str(model_dir))
            mlflow.log_artifacts(str(model_dir), artifact_path="model")
        mlflow.log_artifact(str(MODELS_DIR / "tuned_scaler.pkl"))

        # Register the new version and mark it as the challenger. Promotion to
        # @champion happens only after the evaluation gate passes (Airflow DAG).
        client = mlflow.tracking.MlflowClient()
        try:
            client.create_registered_model(REGISTRY_MODEL_NAME)
        except Exception:
            pass  # already exists
        mv = client.create_model_version(
            REGISTRY_MODEL_NAME,
            source=f"runs:/{run.info.run_id}/model",
            run_id=run.info.run_id,
        )
        client.set_registered_model_alias(REGISTRY_MODEL_NAME, "challenger", mv.version)
        client.set_model_version_tag(
            REGISTRY_MODEL_NAME, mv.version, "validation_status", "pending"
        )

        LOGGER.info(
            f"Registered {REGISTRY_MODEL_NAME} v{mv.version} → @challenger  "
            f"(MAE: {final_metrics['mae']:.4f}  RMSE: {final_metrics['rmse']:.4f}  "
            f"R²: {final_metrics['r2']:.4f})"
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()
    tune_weather_model(n_trials=args.n_trials)