import pandas as pd
import requests
import mlflow
from zenml import step

# Added the experiment tracker to capture batch results
@step(experiment_tracker="mlflow_tracker", enable_cache=False)
def predictor(data: pd.DataFrame) -> pd.DataFrame:
    """Sends cleaned news data and logs prediction results to MLflow."""
    url = "http://127.0.0.1:8000/invocations"

    articles = [{"text": str(val)[:2000]} for val in data['content'].tolist()]
    
    payload = {
        "instances": articles,
        "params": {"truncation": True, "max_length": 512}
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code != 200:
            raise RuntimeError(f"Server error: {response.text}")
            
        predictions = response.json().get("predictions")
        data["prediction"] = [p["label"] for p in predictions]
        data["confidence"] = [p["score"] for p in predictions]

        mlflow.log_table(data=data, artifact_file="inference/batch_predictions.json")
        mlflow.log_metric("inference_count", len(data))
        
        print("Batch inference successful and logged to MLflow!")
        return data

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Communication error: {e}")