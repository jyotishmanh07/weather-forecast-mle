import mlflow
import pandas as pd
from zenml import step
from sklearn.metrics import accuracy_score

@step(experiment_tracker="mlflow_tracker", enable_cache=False)
def evaluate_model(df: pd.DataFrame, model_uri: str) -> bool: # ADD 'model_uri' HERE 
    """Evaluates the model instance created in the current run."""
    
    texts = df['content'].tolist()
    y_true = df['label'].tolist()
    
    print(f"Evaluating model at: {model_uri}")
    model = mlflow.transformers.load_model(model_uri, task="text-classification")
    
    results = model(texts, truncation=True, max_length=512)
    predictions = [1 if res['label'] == 'LABEL_1' else 0 for res in results]
    
    acc = accuracy_score(y_true, predictions)
    print(f"Final Evaluation Accuracy: {acc:.2f}")
    
    return bool(acc >= 0.75)