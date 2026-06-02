from zenml import step
from pipelines.training_pipeline.tune import tune_weather_model


@step
def train_weather_model(n_trials: int = 30) -> str:
    """Tunes XGBoost with Optuna across n_trials, logs every trial to MLflow.

    Returns the path to the saved best model artifact.
    """
    tune_weather_model(n_trials=n_trials)
    return "models/best_weather_xgb.pkl"
