"""Rolling-origin backtesting for the weather forecast model.

Instead of trusting one tiny eval split, slide a test window over the most
recent weeks: for each fold, train on everything strictly before the window
and score on the window, per horizon. The fold average is a far more stable
accuracy estimate, and running it before/after a change gives an honest
comparison on identical test dates.

Leakage guard: a training row at time t with horizon h has its target at
t + h. Rows whose target date falls inside (or after) the test window are
excluded from training, so no future information reaches the fit.

Model parameters are FIXED (not re-tuned per fold) so backtests measure the
data/features, not tuner luck — comparable across runs by construction.

Usage:
    python -m pipelines.training_pipeline.backtest [--folds 6] [--test-days 7]
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
SPLIT_FILES = ["train_encoded.csv", "eval_encoded.csv", "holdout_encoded.csv"]
TARGET = "target_temp_max"

# Fixed, moderate parameters — identical for every fold and every backtest run.
XGB_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
)


def _load_full_frame() -> pd.DataFrame:
    """All encoded splits stacked into one chronologically-contiguous frame."""
    frames = [
        pd.read_csv(PROCESSED_DIR / f, parse_dates=["time"])
        for f in SPLIT_FILES
        if (PROCESSED_DIR / f).exists()
    ]
    if not frames:
        raise FileNotFoundError(f"No encoded splits found in {PROCESSED_DIR}")
    df = pd.concat(frames, ignore_index=True)

    lag_cols = [c for c in df.columns if c.startswith(("lag_", "rolling_")) or c == "precipitation_sum"]
    df[lag_cols] = df[lag_cols].fillna(0)
    return df.sort_values("time").reset_index(drop=True)


def run_backtest(n_folds: int = 6, test_days: int = 7) -> pd.DataFrame:
    df = _load_full_frame()
    horizons = sorted(df["horizon"].unique()) if "horizon" in df.columns else [1]
    last_day = df["time"].max()

    results = []
    for fold in range(n_folds):
        test_end = last_day - pd.Timedelta(days=fold * test_days)
        test_start = test_end - pd.Timedelta(days=test_days - 1)

        test = df[(df["time"] >= test_start) & (df["time"] <= test_end)]
        # Leakage guard: training targets must be realised BEFORE the window.
        target_date = df["time"] + pd.to_timedelta(df.get("horizon", 1), unit="D")
        train = df[target_date < test_start]

        if train.empty or test.empty:
            LOGGER.warning(f"Fold {fold}: empty train/test — skipping.")
            continue

        X_tr = train.drop(columns=["time", TARGET])
        X_te = test.drop(columns=["time", TARGET])
        scaler = StandardScaler().fit(X_tr)

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(scaler.transform(X_tr), train[TARGET])
        preds = model.predict(scaler.transform(X_te))

        row = {
            "fold": fold,
            "test_start": test_start.date(),
            "test_end": test_end.date(),
            "train_rows": len(train),
            "mae": mean_absolute_error(test[TARGET], preds),
        }
        for h in horizons:
            m = (X_te["horizon"] == h).values if "horizon" in X_te.columns else np.ones(len(X_te), bool)
            if m.any():
                row[f"mae_h{int(h)}"] = mean_absolute_error(test[TARGET][m], preds[m])
        results.append(row)

    res = pd.DataFrame(results)
    if res.empty:
        raise RuntimeError("No folds produced results — not enough data?")

    LOGGER.info("\nPer-fold results:\n%s", res.round(3).to_string(index=False))
    mae_cols = [c for c in res.columns if c.startswith("mae")]
    summary = res[mae_cols].mean().round(3)
    LOGGER.info("\n=== Backtest average over %d folds (test window %dd) ===\n%s",
                len(res), test_days, summary.to_string())
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rolling-origin backtest.")
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--log-mlflow", action="store_true",
                        help="Log fold-average metrics to MLflow as a 'backtest' run.")
    args = parser.parse_args()

    res = run_backtest(n_folds=args.folds, test_days=args.test_days)

    if args.log_mlflow:
        import mlflow
        with mlflow.start_run(run_name="backtest"):
            mlflow.log_params({"folds": len(res), "test_days": args.test_days, **XGB_PARAMS})
            for col in [c for c in res.columns if c.startswith("mae")]:
                mlflow.log_metric(f"backtest_{col}", float(res[col].mean()))
        LOGGER.info("Backtest averages logged to MLflow.")
