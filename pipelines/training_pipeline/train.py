import pandas as pd
import numpy as np
import logging
import joblib
import mlflow
from pathlib import Path
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_and_preprocess(train_path, eval_path, target='temperature_2m_max'):
    """Loads data and handles missing values for all advanced features."""
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

def train_model(X_train, y_train, X_eval, y_eval, alpha=0.1):
    """Scales features including lags and trains the Lasso model."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    model = Lasso(alpha=alpha)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_eval_scaled)
    metrics = {
        "mae": mean_absolute_error(y_eval, preds),
        "rmse": np.sqrt(mean_squared_error(y_eval, preds)),
        "r2": r2_score(y_eval, preds)
    }

    return model, scaler, metrics

def go(alpha=0.1):
    with mlflow.start_run(run_name="weather_training_with_lags"):
        X_train, y_train, X_eval, y_eval = load_and_preprocess(
            PROCESSED_DIR / "train_encoded.csv",
            PROCESSED_DIR / "eval_encoded.csv"
        )
        
        model, scaler, metrics = train_model(X_train, y_train, X_eval, y_eval, alpha=alpha)
        
        mlflow.log_params({"alpha": alpha, "model_type": "Lasso", "features": "lags_and_rolling"})
        mlflow.log_metrics(metrics)
        
        joblib.dump(model, MODELS_DIR / "weather_model.pkl")
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
        
        LOGGER.info(f"Training Complete. MAE: {metrics['mae']:.4f}")

if __name__ == "__main__":
    go()