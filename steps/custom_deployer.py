import os
import subprocess
import time
from zenml import step

@step(enable_cache=False)
def custom_mlflow_deployer(model_uri: str, deploy_decision: bool):
    """Manually triggers the MLflow serve process if evaluation passes."""
    if not deploy_decision:
        print("Deployment decision was False. Skipping deployment.")
        return

    print(f"Deploying model from: {model_uri}")
    
    port = 8000

    cmd = [
        "mlflow", "models", "serve",
        "-m", model_uri,
        "-p", str(port),
        "--no-conda",
        "--host", "127.0.0.1"
    ]
    
    # Check if a server is already running on that port and kill it
    subprocess.run(["pkill", "-f", f"serve -m .* -p {port}"], stderr=subprocess.DEVNULL)
    
    # Start the new server
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print(f"MLflow model server started at http://127.0.0.1:{port}")
