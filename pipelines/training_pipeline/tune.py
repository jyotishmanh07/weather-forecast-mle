from __future__ import annotations
import logging
import joblib
import mlflow
import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

def _load_and_prepare(train_path, eval_path, target='temperature_2m_max'):
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

def tune_weather_model(n_trials: int = 30):
    mlflow.set_experiment("weather_xgb_lags_tuning")
    
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

    # Retrain final best model
    with mlflow.start_run(run_name="best_xgb_with_lags"):
        best_model = XGBRegressor(**study.best_params)
        best_model.fit(X_train_scaled, y_train)
        
        joblib.dump(best_model, MODELS_DIR / "best_weather_xgb.pkl")
        joblib.dump(scaler, MODELS_DIR / "tuned_scaler.pkl")
        
        LOGGER.info(f"Tuning Complete. Best MAE: {study.best_value:.4f}")

if __name__ == "__main__":
    tune_weather_model()