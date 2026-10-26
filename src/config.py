"""
Configuration file for NIFTY 50 Stock Price Prediction Pipeline.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CSV_CACHE_PATH = DATA_DIR / "nifty50_data.csv"
METRICS_CSV_PATH = OUTPUTS_DIR / "metrics_summary.csv"
METRICS_JSON_PATH = OUTPUTS_DIR / "metrics_summary.json"

# Stock Data Parameters
TICKER = "^NSEI"
START_DATE = "2015-01-01"
END_DATE = "2025-07-31"

# ARIMA Parameters
# Order (p=1, d=0, q=0) — `d=0` because `FirstDifference` is already stationary (ADF p < 1e-25).
# Using d=1 on a stationary series causes over-differencing (d=2 on raw log price),
# degrading AIC by ~1,051 points vs. ARIMA(1,0,0).
ARIMA_ORDER = (1, 0, 0)

# Deep Learning Parameters
TIME_STEP = 60
TRAIN_RATIO = 0.70
BATCH_SIZE = 64
EPOCHS = 10
RANDOM_SEED = 42
