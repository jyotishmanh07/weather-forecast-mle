"""
Drift monitoring using Evidently.

Compares the training-data distribution (reference) against recent API
prediction inputs (current window) to detect feature drift.
Called by the /metrics endpoint in main.py.
"""

from __future__ import annotations

import logging
import threading
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
PREDICTION_LOG = PROJECT_ROOT / "data" / "prediction_log.csv"

MONITOR_NUMERICAL = ["temperature_2m_min", "precipitation_sum"]
MONITOR_CATEGORICAL = ["weathercode"]
MONITOR_COLS = MONITOR_NUMERICAL + MONITOR_CATEGORICAL

_log_lock = threading.Lock()


def log_prediction_inputs(df: pd.DataFrame) -> None:
    """Append raw API input rows to the rolling prediction log."""
    cols = [c for c in MONITOR_COLS + ["time", "city"] if c in df.columns]
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock:
        write_header = not PREDICTION_LOG.exists() or PREDICTION_LOG.stat().st_size == 0
        df[cols].to_csv(PREDICTION_LOG, mode="a", header=write_header, index=False)


def run_drift_check(
    window_rows: int = 500,
    drift_share_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compare training distribution against the last `window_rows` prediction inputs.

    Returns a JSON-serialisable dict with per-column drift p-values and an
    overall dataset_drifted flag. Used by GET /metrics.
    """
    warnings.filterwarnings("ignore")

    if not REFERENCE_PATH.exists():
        return {"error": f"Reference data not found: {REFERENCE_PATH}"}
    if not PREDICTION_LOG.exists():
        return {"error": "No prediction log yet — call /predict first."}

    try:
        from evidently import Dataset, DataDefinition, Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        return {"error": "evidently not installed"}

    ref = pd.read_csv(REFERENCE_PATH, usecols=lambda c: c in MONITOR_COLS)
    cur = pd.read_csv(PREDICTION_LOG, usecols=lambda c: c in MONITOR_COLS)
    cur = cur.tail(window_rows)

    ref = ref.dropna(subset=MONITOR_COLS)
    cur = cur.dropna(subset=MONITOR_COLS)

    result: dict[str, Any] = {
        "reference_rows": len(ref),
        "current_rows": len(cur),
    }

    if len(cur) < 30:
        result["warning"] = (
            f"Only {len(cur)} rows in current window — "
            "drift estimates are unreliable below 30 rows."
        )

    schema = DataDefinition(
        numerical_columns=MONITOR_NUMERICAL,
        categorical_columns=MONITOR_CATEGORICAL,
    )
    ref_ds = Dataset.from_pandas(ref, data_definition=schema)
    cur_ds = Dataset.from_pandas(cur, data_definition=schema)

    # Force p-value tests so the drift_score is always a p-value (< 0.05 = drift).
    # K-S for continuous, chi-square for categorical — standard and interview-explainable.
    report = Report([DataDriftPreset(num_method="ks", cat_method="chisquare", threshold=0.05)])
    run = report.run(reference_data=ref_ds, current_data=cur_ds)
    metrics = run.dict()["metrics"]

    columns: dict[str, dict] = {}
    drifted_share = 0.0

    for m in metrics:
        name: str = m["metric_name"]
        value = m["value"]

        if name.startswith("DriftedColumnsCount"):
            drifted_share = float(value["share"])
            result["drifted_column_count"] = int(value["count"])
            result["drifted_column_share"] = round(drifted_share, 3)

        elif name.startswith("ValueDrift"):
            col = m["config"]["column"]
            score = float(value)
            threshold = float(m["config"].get("threshold", 0.05))
            method = m["config"].get("method", "unknown")
            # K-S, chi-square, etc. return p-values: low score = drift.
            # Distance methods (Wasserstein, Jensen-Shannon): high score = drift.
            _p_value_tests = {"ks", "chisquare", "fisher_exact", "mannw", "t_test"}
            normalized = method.lower().replace("-", "").replace(" ", "")
            is_pvalue = any(t in normalized for t in _p_value_tests)
            drifted = score < threshold if is_pvalue else score > threshold
            columns[col] = {
                "drift_score": round(score, 4),
                "threshold": threshold,
                "method": method,
                "drifted": drifted,
            }

    result["dataset_drifted"] = drifted_share >= drift_share_threshold
    result["columns"] = columns
    return result


def save_drift_report_html(output_path: str | Path = "drift_report.html") -> Path:
    """Generate and save a full Evidently HTML drift report."""
    warnings.filterwarnings("ignore")

    from evidently import Dataset, DataDefinition, Report
    from evidently.presets import DataDriftPreset

    ref = pd.read_csv(REFERENCE_PATH, usecols=lambda c: c in MONITOR_COLS)
    cur = pd.read_csv(PREDICTION_LOG, usecols=lambda c: c in MONITOR_COLS)

    schema = DataDefinition(
        numerical_columns=MONITOR_NUMERICAL,
        categorical_columns=MONITOR_CATEGORICAL,
    )
    ref_ds = Dataset.from_pandas(ref.dropna(), data_definition=schema)
    cur_ds = Dataset.from_pandas(cur.dropna(), data_definition=schema)

    report = Report([DataDriftPreset(num_method="ks", cat_method="chisquare", threshold=0.05)])
    run = report.run(reference_data=ref_ds, current_data=cur_ds)

    output_path = Path(output_path)
    run.save_html(str(output_path))
    LOGGER.info(f"Drift report saved to {output_path}")
    return output_path
