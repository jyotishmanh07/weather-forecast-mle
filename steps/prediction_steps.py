import pandas as pd
import requests
from zenml import step

@step(enable_cache=False)
def predictor(data: pd.DataFrame) -> pd.DataFrame:
    """Sends cleaned news data with explicit truncation parameters."""
    url = "http://127.0.0.1:8000/invocations"

    # cut the text to roughly 2000 characters (safe for 512 tokens)
    articles = [{"text": str(val)[:2000]} for val in data['content'].tolist()]
    
    # add truncation parameters to the payload
    payload = {
        "instances": articles,
        "params": {
            "truncation": True,
            "max_length": 512
        }
    }
    
    print(f"Sending {len(data)} articles to the server (with truncation)...")
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code != 200:
            print(f"Server Error Log: {response.text}")
            raise RuntimeError(f"Server returned status {response.status_code}")
            
        predictions = response.json().get("predictions")
        data["prediction"] = [p["label"] for p in predictions]
        data["confidence"] = [p["score"] for p in predictions]
        
        print("Batch inference successful!")
        return data

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Communication error: {e}")