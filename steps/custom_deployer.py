import os
import subprocess
import time
import requests
from zenml import step

@step(enable_cache=False)
def custom_mlflow_deployer(model_uri: str, deploy_decision: bool):
    if not deploy_decision:
        print("Deployment decision was False. Skipping deployment.")
        return

    port = 8000
    cmd = [
        "mlflow", "models", "serve",
        "-m", model_uri,
        "-p", str(port),
        "--no-conda",
        "--host", "127.0.0.1"
    ]
    
    subprocess.run(["pkill", "-f", f"serve -m .* -p {port}"], stderr=subprocess.DEVNULL)
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print(f"Waiting for MLflow server to start at http://127.0.0.1:{port}...")
    for _ in range(30): # Wait up to 30 seconds
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health")
            if response.status_code == 200:
                print("Server is up and healthy!")
                return
        except:
            pass
        time.sleep(2)
    
    raise RuntimeError("MLflow server failed to start within 60 seconds.")