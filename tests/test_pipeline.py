"""
Automated Pytest Test Suite for NIFTY 50 Quantitative Pipeline.
Verifies data loading, stationarity tests, contamination-free ARIMA modeling,
deep learning architectures, sequence generation, and financial backtesting metrics.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import TICKER, START_DATE, END_DATE, ARIMA_ORDER, TIME_STEP, TRAIN_RATIO
from src.data_loader import load_nifty50_data, prepare_log_transformed_series
from src.stationarity import check_adf_stationarity, check_kpss_stationarity, compute_first_difference
from src.arima_hybrid import fit_arima_on_train, get_out_of_sample_residuals, build_hybrid_series
from src.models import (
    set_seed, prepare_train_test_data, get_model_callbacks,
    build_lstm_model, build_bilstm_model, build_gru_model, build_deep_gru_model,
    evaluate_forecast, reconstruct_actual_vs_predicted_prices, compute_trading_metrics
)


@pytest.fixture(scope="module")
def sample_data():
    """Loads historical dataset for testing."""
    df = load_nifty50_data(TICKER, START_DATE, END_DATE)
    df_log = prepare_log_transformed_series(df)
    df_diff = compute_first_difference(df_log)
    return {"df": df, "df_log": df_log, "df_diff": df_diff}


# ----------------------------------------------------------------------
# 1. Data Loader & Preprocessing Tests
# ----------------------------------------------------------------------
def test_data_loader_structure(sample_data):
    df = sample_data["df"]
    assert isinstance(df, pd.DataFrame)
    assert "date" in df.columns
    assert "close" in df.columns
    assert len(df) > 1000, "Historical dataset should contain at least 1000 trading days."
    assert not df["close"].isnull().any(), "Close prices must not contain nulls."


def test_log_transformation(sample_data):
    df_log = sample_data["df_log"]
    df = sample_data["df"]
    assert np.allclose(df_log["close"], np.log(df["close"]))


# ----------------------------------------------------------------------
# 2. Stationarity Tests
# ----------------------------------------------------------------------
def test_stationarity_differencing(sample_data):
    df_diff = sample_data["df_diff"]
    assert "FirstDifference" in df_diff.columns
    assert not df_diff["FirstDifference"].isnull().any()
    
    # ADF test on differenced series should confirm stationarity (p < 0.05)
    adf_res = check_adf_stationarity(df_diff["FirstDifference"], verbose=False)
    assert adf_res["is_stationary"] is True, "First-differenced series must be stationary."
    assert adf_res["p_value"] < 0.01


# ----------------------------------------------------------------------
# 3. Contamination-Free ARIMA Modeling Tests
# ----------------------------------------------------------------------
def test_arima_train_test_separation(sample_data):
    df_diff = sample_data["df_diff"]
    hybrid_series, arima_result = build_hybrid_series(
        df_diff, train_ratio=0.70, order=ARIMA_ORDER, verbose=False
    )
    
    assert isinstance(hybrid_series, pd.Series)
    assert len(hybrid_series) > 0
    assert not hybrid_series.isnull().any()
    assert arima_result.aic < 0, "ARIMA AIC must be negative for log difference."


def test_no_future_lookahead_in_hybrid(sample_data):
    df_diff = sample_data["df_diff"]
    hybrid_series, _ = build_hybrid_series(df_diff, train_ratio=0.70, order=(1, 0, 0), verbose=False)
    # The hybrid series size should equal df_diff size (no extra dropna from shift(-1))
    assert len(hybrid_series) == len(df_diff)


# ----------------------------------------------------------------------
# 4. Sequence Generator & Scaler Tests
# ----------------------------------------------------------------------
def test_sequence_generator_and_boundary(sample_data):
    df_diff = sample_data["df_diff"]
    hybrid_series, _ = build_hybrid_series(df_diff, train_ratio=TRAIN_RATIO, order=ARIMA_ORDER, verbose=False)
    
    X_train, y_train, X_test, y_test, scaler, test_dates = prepare_train_test_data(
        hybrid_series, train_ratio=TRAIN_RATIO, time_step=TIME_STEP
    )
    
    # Verify 3D tensor shapes for recurrent neural nets: (samples, time_steps, features)
    assert X_train.ndim == 3
    assert X_train.shape[1] == TIME_STEP
    assert X_train.shape[2] == 1
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(test_dates) == len(y_test)
    
    # Scaler must be strictly in [0, 1] on train data
    assert y_train.min() >= -0.01 and y_train.max() <= 1.01


# ----------------------------------------------------------------------
# 5. Deep Learning Model Architectures Tests
# ----------------------------------------------------------------------
@pytest.mark.parametrize("builder,model_name", [
    (build_lstm_model, "LSTM"),
    (build_bilstm_model, "Bi-LSTM"),
    (build_gru_model, "GRU"),
    (build_deep_gru_model, "Deep_GRU")
])
def test_model_build_and_forward_pass(builder, model_name):
    model = builder(input_shape=(TIME_STEP, 1))
    assert model.name == model_name
    
    # Dummy forward pass
    dummy_input = np.random.rand(4, TIME_STEP, 1).astype(np.float32)
    output = model(dummy_input)
    assert output.shape == (4, 1)


def test_training_callbacks():
    callbacks = get_model_callbacks(patience_es=3, patience_lr=2)
    assert len(callbacks) == 2


# ----------------------------------------------------------------------
# 6. Evaluation & Trading Metrics Tests
# ----------------------------------------------------------------------
def test_evaluate_forecast():
    y_true = np.array([0.5, 0.6, 0.7, 0.8])
    y_pred = np.array([0.52, 0.58, 0.71, 0.79])
    y_train = np.array([0.1, 0.2, 0.3, 0.4])
    
    metrics = evaluate_forecast(y_true, y_pred, y_train, model_name="TestModel")
    assert "RMSE" in metrics
    assert "MAE" in metrics
    assert "MAPE (%)" in metrics
    assert "MASE" in metrics
    assert "R2" in metrics
    assert metrics["RMSE"] >= 0.0
    assert metrics["R2"] <= 1.0


def test_compute_trading_metrics():
    # Construct synthetic upward trend evaluation df
    dates = pd.date_range(start="2024-01-01", periods=10, freq="B")
    actual_prices = np.array([100, 102, 101, 104, 106, 105, 108, 110, 109, 112])
    pred_prices = np.array([100, 101.5, 101.2, 103.8, 105.5, 105.2, 107.5, 109.8, 108.9, 111.5])
    
    df_eval = pd.DataFrame({
        "Actual_Price": actual_prices,
        "Predicted_Price": pred_prices
    }, index=dates)
    
    # 0 fee test
    res = compute_trading_metrics(df_eval, model_name="TestModel", fee_bps=0.0)
    assert "Directional_Accuracy (%)" in res
    assert "Sharpe_Ratio" in res
    assert "Max_Drawdown (%)" in res
    assert "Strategy_Return (%)" in res
    assert "BuyHold_Return (%)" in res
    assert 0.0 <= res["Directional_Accuracy (%)"] <= 100.0
    
    # 5 bps fee test (strategy return must be slightly lower due to transaction friction)
    res_fee = compute_trading_metrics(df_eval, model_name="TestModel", fee_bps=5.0)
    assert res_fee["Strategy_Return (%)"] <= res["Strategy_Return (%)"]
