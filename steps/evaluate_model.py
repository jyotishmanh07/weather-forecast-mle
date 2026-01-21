import mlflow
from zenml import step
from sklearn.metrics import accuracy_score

@step
def evaluate_model(df: pd.DataFrame) -> bool:
    """Determines if the model meets the production threshold."""
    # In a real scenario, we'd load the model and run predictions on a test set
    # Here, we fetch the latest metric from the MLflow run
    run = mlflow.active_run()
    accuracy = run.data.metrics.get("eval_accuracy", 0)
    
    threshold = 0.80 # 80% Accuracy requirement
    
    if accuracy >= threshold:
        print(f"Model passed! Accuracy: {accuracy}")
        return True
    else:
        print(f"Model failed. Accuracy: {accuracy}")
        return False