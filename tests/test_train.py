import math
import joblib
import pytest
import numpy as np
from pathlib import Path

from pipelines.training_pipeline.train import load_and_preprocess, train_model
from pipelines.training_pipeline.tune import tune_weather_model

PROCESSED_DIR = Path("data/processed")
TRAIN_PATH = PROCESSED_DIR / "train_encoded.csv"
EVAL_PATH = PROCESSED_DIR / "eval_encoded.csv"

def _assert_metrics(m):
    """Ensures metrics exist and are finite numbers."""
    assert set(m.keys()) >= {"mae", "rmse", "r2"}
    assert all(isinstance(v, (float, np.float64)) and math.isfinite(v) for v in m.values())


def test_load_and_preprocess_handles_nans():
    """Confirms NaN filling for precipitation works."""
    X_train, y_train, X_eval, y_eval = load_and_preprocess(TRAIN_PATH, EVAL_PATH)
    
    assert not X_train['precipitation_sum'].isnull().any()
    assert not X_eval['precipitation_sum'].isnull().any()
    assert len(X_train) > 0
    print("load_and_preprocess test passed")

def test_train_model_saves_artifacts(tmp_path):
    """Verifies that both the Lasso model and the Scaler are created."""
    X_train, y_train, X_eval, y_eval = load_and_preprocess(TRAIN_PATH, EVAL_PATH)
    
    model, scaler, metrics = train_model(X_train, y_train, X_eval, y_eval, alpha=0.1)
    
    _assert_metrics(metrics)
    assert model is not None
    assert scaler is not None
    
    X_train_scaled = scaler.transform(X_train)
    assert math.isclose(X_train_scaled.mean(), 0, abs_tol=1e-1)
    print("train_model artifacts test passed")

def test_tune_weather_model_integration(tmp_path):
    """Runs a 2-trial tune to ensure the Optuna-MLflow loop is solid."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("pipelines.training_pipeline.tune.MODELS_DIR", tmp_path)
        mp.setattr("pipelines.training_pipeline.tune.PROCESSED_DIR", PROCESSED_DIR)
        
        tune_weather_model(n_trials=2, experiment_name="ci_test_tuning")
        
        assert (tmp_path / "best_weather_xgb.pkl").exists()
        assert (tmp_path / "tuned_scaler.pkl").exists()
        
    print("tune_weather_model integration test passed")

def test_model_prediction_range():
    """Safety check: ensures the model isn't predicting impossible temperatures."""
    X_train, y_train, X_eval, y_eval = load_and_preprocess(TRAIN_PATH, EVAL_PATH)
    model, scaler, _ = train_model(X_train, y_train, X_eval, y_eval)
    
    X_eval_scaled = scaler.transform(X_eval)
    preds = model.predict(X_eval_scaled)
    
    assert (preds > -40).all() and (preds < 55).all()
    print("Prediction range safety test passed")