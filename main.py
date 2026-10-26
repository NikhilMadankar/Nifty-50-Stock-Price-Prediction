"""
Main entry point for running the NIFTY 50 Stock Price Prediction Pipeline.
Executes data extraction, stationarity analysis, ARIMA hybrid modeling,
Deep Learning training (LSTM, Bi-LSTM, GRU, Deep GRU), evaluation, and visualization.
"""

import sys
import os
import warnings
import argparse
import json
import time
import pandas as pd
from tabulate import tabulate

# Configure utf-8 stdout encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from src.config import (
    TICKER, START_DATE, END_DATE, ARIMA_ORDER, TIME_STEP,
    TRAIN_RATIO, BATCH_SIZE, EPOCHS, OUTPUTS_DIR, MODELS_DIR,
    METRICS_CSV_PATH, METRICS_JSON_PATH
)
from src.data_loader import load_nifty50_data, prepare_log_transformed_series
from src.stationarity import check_adf_stationarity, check_kpss_stationarity, compute_first_difference
from src.arima_hybrid import build_hybrid_series
from src.models import (
    set_seed, prepare_train_test_data, get_model_callbacks,
    build_lstm_model, build_bilstm_model, build_gru_model, build_deep_gru_model,
    evaluate_forecast, reconstruct_actual_vs_predicted_prices, compute_trading_metrics
)
from src.visualization import (
    plot_historical_price, plot_stationarity_and_acf_pacf,
    plot_single_model_prediction, plot_all_models_comparison,
    plot_metrics_bar_chart, plot_strategy_equity_curves
)


def run_pipeline(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    fee_bps: float = 0.0,
    save_plots: bool = True
):
    start_total_time = time.time()
    set_seed()
    
    print("\n" + "=" * 80)
    print("       NIFTY 50 STOCK PRICE PREDICTION - HYBRID ARIMA & DEEP LEARNING")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # Step 1: Data Ingestion & Preprocessing
    # -------------------------------------------------------------
    print("\n>>> [STEP 1/5] Ingesting NIFTY 50 Historical Data...")
    df = load_nifty50_data(TICKER, START_DATE, END_DATE)
    df_log = prepare_log_transformed_series(df)
    print(f"Total trading days loaded: {len(df)}")
    print(f"Date Range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    print(f"Starting Price: INR {df['close'].iloc[0]:,.2f} | Ending Price: INR {df['close'].iloc[-1]:,.2f}")
    
    if save_plots:
        plot_historical_price(df)

    # -------------------------------------------------------------
    # Step 2: Stationarity & Time Series Diagnostics
    # -------------------------------------------------------------
    print("\n>>> [STEP 2/5] Performing Stationarity & Diagnostic Tests...")
    check_adf_stationarity(df_log["close"], name="Log(Close Price)")
    check_kpss_stationarity(df_log["close"], name="Log(Close Price)")
    
    df_diff = compute_first_difference(df_log)
    print("\nApplied First Differencing...")
    check_adf_stationarity(df_diff["FirstDifference"], name="First Difference of Log(Price)")
    check_kpss_stationarity(df_diff["FirstDifference"], name="First Difference of Log(Price)")
    
    if save_plots:
        plot_stationarity_and_acf_pacf(df_diff)

    # -------------------------------------------------------------
    # Step 3: ARIMA Modeling & Hybrid Feature Extraction (Contamination-Free)
    # -------------------------------------------------------------
    print(f"\n>>> [STEP 3/5] Fitting ARIMA{ARIMA_ORDER} on Training Partition & Reconstructing Hybrid Series...")
    print("    [Fix 3] ARIMA is fit ONLY on the training partition (70%).")
    print("    [Fix 3] Out-of-sample residuals for the test partition are obtained")
    print("            via model.apply() with FROZEN coefficients — no re-estimation.")
    hybrid_series, arima_result = build_hybrid_series(
        df_diff, train_ratio=TRAIN_RATIO, order=ARIMA_ORDER, verbose=False
    )
    print(f"ARIMA AIC: {arima_result.aic:.4f}, BIC: {arima_result.bic:.4f}")
    print(f"Engineered hybrid series size: {len(hybrid_series)}")

    # -------------------------------------------------------------
    # Step 4: Dataset Preparation (Zero Data Leakage)
    # -------------------------------------------------------------
    print("\n>>> [STEP 4/5] Preparing Train/Test Splits with Zero Data Leakage...")
    X_train, y_train, X_test, y_test, scaler, test_dates = prepare_train_test_data(
        hybrid_series, train_ratio=TRAIN_RATIO, time_step=TIME_STEP
    )
    print(f"Lookback Window (time_step): {TIME_STEP}")
    print(f"Training Sequences: {X_train.shape[0]} | Shape: {X_train.shape}")
    print(f"Testing Sequences:  {X_test.shape[0]} | Shape: {X_test.shape}")
    print(f"Test Date Range:    {test_dates[0]} to {test_dates[-1]}")

    # -------------------------------------------------------------
    # Step 5: Model Training, Evaluation & Comparison
    # -------------------------------------------------------------
    print(f"\n>>> [STEP 5/5] Training & Evaluating Deep Learning Models ({epochs} Epochs with Callbacks)...")
    
    models_to_train = [
        ("LSTM", build_lstm_model((TIME_STEP, 1))),
        ("Bi-LSTM", build_bilstm_model((TIME_STEP, 1))),
        ("GRU", build_gru_model((TIME_STEP, 1))),
        ("Deep_GRU", build_deep_gru_model((TIME_STEP, 1))),
    ]
    
    metrics_list = []
    eval_dfs = {}
    
    for model_name, model in models_to_train:
        print(f"\n--- Training {model_name} ---")
        callbacks = get_model_callbacks(patience_es=4, patience_lr=2)
        t0 = time.time()
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        elapsed = time.time() - t0
        
        # Save model weights/architecture
        model_save_path = MODELS_DIR / f"{model_name.lower()}_model.keras"
        model.save(model_save_path)
        
        # Predictions
        test_predict = model.predict(X_test, verbose=0)
        
        # Evaluation metrics
        metrics = evaluate_forecast(y_test, test_predict, y_train, model_name=model_name)
        metrics["Train_Time_Sec"] = round(elapsed, 2)
        metrics_list.append(metrics)
        
        # Reconstruct actual vs predicted INR price
        df_eval = reconstruct_actual_vs_predicted_prices(df_log, test_dates, test_predict, scaler)
        eval_dfs[model_name] = df_eval
        
        print(f"Finished {model_name} in {elapsed:.2f}s | RMSE: {metrics['RMSE']:.4f} | MAPE: {metrics['MAPE (%)']:.2f}% | MASE: {metrics['MASE']:.4f} | R2: {metrics['R2']:.4f}")
        
        if save_plots:
            plot_single_model_prediction(
                df_eval,
                model_name=model_name,
                metrics=metrics,
                save_path=OUTPUTS_DIR / f"{model_name.lower()}_prediction.png"
            )

    # -------------------------------------------------------------
    # Overall Summary & Visualizations
    # -------------------------------------------------------------
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)
    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(metrics_list, f, indent=4)

    if save_plots:
        plot_all_models_comparison(eval_dfs)
        plot_metrics_bar_chart(metrics_df)

    print("\n" + "=" * 80)
    print("                           MODEL PERFORMANCE SUMMARY")
    print("=" * 80)
    print(tabulate(metrics_df, headers="keys", tablefmt="grid", showindex=False))

    best_model = metrics_df.sort_values(by="RMSE").iloc[0]
    print(f"\nBest Performing Model by RMSE: {best_model['Model']} (RMSE: {best_model['RMSE']:.4f}, MAPE: {best_model['MAPE (%)']:.2f}%, MASE: {best_model['MASE']:.4f}, R2: {best_model['R2']:.4f})")

    # ------------------------------------------------------------------
    # Fix 4 & 5: Directional Accuracy + Financial Trading Metrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"         DIRECTIONAL ACCURACY & FINANCIAL TRADING METRICS (Fee: {fee_bps} bps)")
    print("=" * 80)
    trading_records = []
    trading_results_dict = {}
    for model_name, df_eval in eval_dfs.items():
        t = compute_trading_metrics(df_eval, model_name=model_name, fee_bps=fee_bps)
        trading_results_dict[model_name] = t
        trading_records.append({
            "Model": t["Model"],
            "Hit Ratio (%)": t["Directional_Accuracy (%)"],
            "Sharpe Ratio": t["Sharpe_Ratio"],
            "Max Drawdown (%)": t["Max_Drawdown (%)"],
            "Strategy Return (%)": t["Strategy_Return (%)"],
            "Buy & Hold (%)": t["BuyHold_Return (%)"],
        })
    trading_df = pd.DataFrame(trading_records)
    print(tabulate(trading_df, headers="keys", tablefmt="grid", showindex=False))
    print("\nStrategy: LONG when predicted log-return > 0, else CASH.")
    print("Benchmark: Buy-and-Hold for the same test period.")

    if save_plots:
        plot_strategy_equity_curves(trading_results_dict)

    print(f"\nTotal Pipeline Execution Time: {time.time() - start_total_time:.2f} seconds.")
    print(f"All plots and models saved in: {OUTPUTS_DIR} and {MODELS_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIFTY 50 Stock Price Prediction Pipeline")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs per model")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--fee-bps", type=float, default=0.0, help="Transaction fee in basis points per turnover")
    parser.add_argument("--no-plot", action="store_true", help="Disable saving plots to disk")
    args = parser.parse_args()
    
    run_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        fee_bps=args.fee_bps,
        save_plots=not args.no_plot
    )
