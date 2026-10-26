"""
Deep Learning time series models: LSTM, Bi-LSTM, GRU, and Deep GRU.
Includes sequence generation, training callbacks, evaluation metrics, and trading alpha backtesting.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from typing import Tuple, Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.config import TIME_STEP, TRAIN_RATIO, BATCH_SIZE, EPOCHS, RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def prepare_train_test_data(
    hybrid_series: pd.Series,
    train_ratio: float = TRAIN_RATIO,
    time_step: int = TIME_STEP
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, pd.Index]:
    """
    Splits hybrid series into train/test sets and fits scaler STRICTLY on training data
    to eliminate data leakage.
    
    Fix: Uses range(len(dataset) - time_step) to ensure the very latest trading day
    is included in both training and test sequences.
    
    Returns:
        X_train, y_train, X_test, y_test, scaler, test_dates
    """
    raw_values = np.array(hybrid_series).reshape(-1, 1)
    
    training_size = int(len(raw_values) * train_ratio)
    test_size = len(raw_values) - training_size
    
    train_raw = raw_values[:training_size]
    test_raw = raw_values[training_size:]
    
    # Fit scaler ONLY on training data (prevents lookahead bias / data leakage)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = scaler.fit_transform(train_raw)
    test_data = scaler.transform(test_raw)
    
    def create_dataset(dataset: np.ndarray, step: int):
        dataX, dataY = [], []
        # Correct boundary: range(len - step) ensures the latest trading day is included
        for i in range(len(dataset) - step):
            a = dataset[i:(i + step)]
            dataX.append(a)
            dataY.append(dataset[i + step])
        return np.array(dataX), np.array(dataY)
    
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Exact aligned test date indices
    test_dates = hybrid_series.index[training_size + time_step : training_size + time_step + len(y_test)]
    
    return X_train, y_train, X_test, y_test, scaler, test_dates


def get_model_callbacks(patience_es: int = 4, patience_lr: int = 2) -> List[Any]:
    """
    Returns EarlyStopping and Learning Rate Scheduler callbacks for deep learning models.
    Prevents overfitting and restores the optimal validation epoch weights.
    """
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-5,
            verbose=0
        )
    ]


def build_lstm_model(input_shape: Tuple[int, int] = (TIME_STEP, 1)) -> Sequential:
    """Standard single-layer LSTM network."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64),
        Dropout(0.3),
        Dense(25, activation="relu"),
        Dense(1)
    ], name="LSTM")
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae"])
    return model


def build_bilstm_model(input_shape: Tuple[int, int] = (TIME_STEP, 1)) -> Sequential:
    """Bidirectional LSTM network."""
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(32, activation="tanh"),
        Dropout(0.3),
        Dense(1)
    ], name="Bi-LSTM")
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae"])
    return model


def build_gru_model(input_shape: Tuple[int, int] = (TIME_STEP, 1)) -> Sequential:
    """Gated Recurrent Unit (GRU) network."""
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="tanh"),
        Dense(1)
    ], name="GRU")
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae"])
    return model


def build_deep_gru_model(input_shape: Tuple[int, int] = (TIME_STEP, 1)) -> Sequential:
    """Deep Multi-Layer GRU network."""
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=True),
        Dropout(0.3),
        GRU(64, return_sequences=True),
        Dropout(0.3),
        GRU(32, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="tanh"),
        Dense(1)
    ], name="Deep_GRU")
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae"])
    return model


def evaluate_forecast(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Computes comprehensive time series metrics: RMSE, MSE, MAE, MAPE, MASE, R2.
    """
    y_true = y_true_scaled.reshape(-1)
    y_pred = y_pred_scaled.reshape(-1)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MASE = MAE / Naive forecast error on training set
    naive_train_error = np.mean(np.abs(np.diff(y_train_scaled.reshape(-1))))
    mase = mae / (naive_train_error if naive_train_error != 0 else 1e-8)
    
    metrics = {
        "Model": model_name,
        "RMSE": float(rmse),
        "MSE": float(mse),
        "MAE": float(mae),
        "MAPE (%)": float(mape * 100),
        "MASE": float(mase),
        "R2": float(r2)
    }
    
    return metrics


def reconstruct_actual_vs_predicted_prices(
    df_log_close: pd.DataFrame,
    test_dates: pd.Index,
    test_predict_scaled: np.ndarray,
    scaler: MinMaxScaler
) -> pd.DataFrame:
    """
    Transforms scaled predictions back into original Rupee values (INR) with date indices.
    """
    df_eval = pd.DataFrame(index=test_dates)
    df_eval["Actual_Price"] = np.exp(df_log_close.loc[test_dates, "close"].values)

    # Invert scaling on prediction and exponentiate log price
    pred_unscaled = scaler.inverse_transform(test_predict_scaled)
    df_eval["Predicted_Price"] = np.exp(pred_unscaled.reshape(-1))

    return df_eval


def compute_trading_metrics(
    df_eval: pd.DataFrame,
    model_name: str = "Model",
    fee_bps: float = 0.0
) -> Dict[str, Any]:
    """
    Fix 4 & 5 — Directional Accuracy and Financial Trading Metrics.

    Strategy: Go LONG (+1) when predicted return > 0, else CASH (0).
    Benchmark: Buy-and-Hold over the same test period.

    Args:
        df_eval: DataFrame with columns ['Actual_Price', 'Predicted_Price'] and DatetimeIndex.
        model_name: Name label for reporting.
        fee_bps: Optional transaction cost in basis points per turnover (e.g., 5 bps = 0.05%).

    Returns:
        Dict with Directional_Accuracy, Sharpe_Ratio, Max_Drawdown_Pct,
        Strategy_Total_Return_Pct, BuyHold_Total_Return_Pct, etc.
    """
    actual = df_eval["Actual_Price"].values
    predicted = df_eval["Predicted_Price"].values

    # Log-returns from actual prices
    actual_returns = np.diff(np.log(actual))          # r_t = ln(P_t / P_{t-1})

    # Predicted log-returns from model's predicted prices
    predicted_returns = np.diff(np.log(predicted))    # r̂_t

    # ── Fix 4: Directional Accuracy (Hit Ratio) ───────────────────────────────
    actual_direction = np.sign(actual_returns)
    predicted_direction = np.sign(predicted_returns)
    hit_ratio = np.mean(actual_direction == predicted_direction) * 100

    # ── Fix 5: Strategy & Trading Metrics ────────────────────────────────────
    # Signal: go long (+1) when predicted return > 0, else cash (0)
    signal = (predicted_returns > 0).astype(float)

    # Calculate turnover and friction/transaction cost
    if fee_bps > 0:
        turnover = np.abs(np.diff(np.insert(signal, 0, 0)))
        cost = turnover * (fee_bps / 10000.0)
    else:
        cost = 0.0

    # Strategy returns: apply signal minus cost
    strategy_returns = (signal * actual_returns) - cost

    # Buy-and-Hold returns (always invested)
    bh_returns = actual_returns

    # Cumulative returns
    strategy_cumulative = np.cumprod(1 + strategy_returns)
    bh_cumulative = np.cumprod(1 + bh_returns)

    strategy_total_return = (strategy_cumulative[-1] - 1) * 100
    bh_total_return = (bh_cumulative[-1] - 1) * 100

    # Annualised Sharpe Ratio (252 trading days, risk-free = 0)
    trading_days = 252
    if strategy_returns.std() > 0:
        sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(trading_days)
    else:
        sharpe = 0.0

    # Max Drawdown
    running_max = np.maximum.accumulate(strategy_cumulative)
    drawdowns = (strategy_cumulative - running_max) / running_max
    max_drawdown = drawdowns.min() * 100   # as percentage

    return {
        "Model": model_name,
        "Directional_Accuracy (%)": round(hit_ratio, 2),
        "Sharpe_Ratio": round(sharpe, 4),
        "Max_Drawdown (%)": round(max_drawdown, 2),
        "Strategy_Return (%)": round(strategy_total_return, 2),
        "BuyHold_Return (%)": round(bh_total_return, 2),
        "_strategy_cumulative": strategy_cumulative,   # for plotting (underscore = internal)
        "_bh_cumulative": bh_cumulative,               # for plotting
        "_dates": df_eval.index[1:],                   # aligned dates (diff drops 1)
    }
