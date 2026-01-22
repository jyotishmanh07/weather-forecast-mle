import mlflow
import pandas as pd
from zenml import step
from sklearn.metrics import accuracy_score

@step(experiment_tracker="mlflow_tracker")
def evaluate_model(df: pd.DataFrame) -> bool:
    """Evaluates the model and returns a native Python boolean."""
    # Load model and run predictions
    model_uri = "models:/NewsIntegrityModel/latest"
    model = mlflow.transformers.load_model(model_uri, task="text-classification")
    
    texts = df['content'].tolist()
    results = model(texts, truncation=True, max_length=512)
    
    # Map labels and calculate accuracy
    predictions = [1 if res['label'] == 'LABEL_1' else 0 for res in results]
    y_true = df['label'].tolist()
    acc = accuracy_score(y_true, predictions)
    
    print(f"Final Evaluation Accuracy: {acc:.2f}")
    passed_threshold = bool(acc >= 0.75) 
    
    return passed_threshold