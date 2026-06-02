import logging
import subprocess
import time
import requests
from zenml import step

LOGGER = logging.getLogger(__name__)


@step(enable_cache=False)
def deploy_weather_api(deploy_decision: bool, port: int = 8000):
    """Starts the FastAPI weather service if deploy_decision is True.

    Kills any existing process on the target port, launches uvicorn in the
    background, then polls /health until the server is ready (up to 30 s).
    Raises RuntimeError if the server does not come up in time.
    """
    if not deploy_decision:
        LOGGER.info("Deploy decision is False — skipping deployment.")
        return

    subprocess.run(
        ["pkill", "-f", f"uvicorn main:app.*--port {port}"],
        stderr=subprocess.DEVNULL,
    )

    subprocess.Popen(
        ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    LOGGER.info(f"Waiting for weather API on http://0.0.0.0:{port} ...")
    for _ in range(15):
        try:
            resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if resp.status_code == 200:
                LOGGER.info("Weather API is healthy.")
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)

    raise RuntimeError(f"Weather API did not start within 30 s on port {port}.")
