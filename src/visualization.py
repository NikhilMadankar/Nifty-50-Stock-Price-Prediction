"""
Visualization routines for time series analysis, diagnostics, and model predictions.
"""

from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from src.config import OUTPUTS_DIR

# Set clean aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.autolayout"] = True


def plot_historical_price(df: pd.DataFrame, save_path: Path = OUTPUTS_DIR / "historical_prices.png") -> None:
    """Plots original NIFTY 50 closing price and log-closing price."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    
    # 1. Raw Closing Price
    axes[0].plot(pd.to_datetime(df["date"]), df["close"], color="#1f77b4", linewidth=1.8, label="NIFTY 50 Closing Price")
    axes[0].set_title("NIFTY 50 Index - Historical Closing Price (2015 - 2025)", fontsize=14, fontweight="bold", pad=10)
    axes[0].set_ylabel("Price (INR)", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper left")
    
    # 2. Log-Transformed Price
    axes[1].plot(pd.to_datetime(df["date"]), np.log(df["close"]), color="#ff7f0e", linewidth=1.8, label="Log(Close Price)")
    axes[1].set_title("Log-Transformed Closing Price", fontsize=14, fontweight="bold", pad=10)
    axes[1].set_ylabel("Log(Price)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Date", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved historical price plot to {save_path}")


def plot_stationarity_and_acf_pacf(df_diff: pd.DataFrame, save_path: Path = OUTPUTS_DIR / "stationarity_acf_pacf.png") -> None:
    """Plots first differenced series alongside ACF and PACF plots."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)
    
    # 1. First Difference Time Series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_diff.index, df_diff["FirstDifference"], color="#2ca02c", linewidth=1.2, label="First Difference of Log(Price)")
    ax1.axhline(0, color="black", linestyle="--", alpha=0.7)
    ax1.set_title("First Differenced Series (Stationary)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Difference Value", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")
    
    # 2. Autocorrelation (ACF)
    ax2 = fig.add_subplot(gs[1, 0])
    plot_acf(df_diff["FirstDifference"], ax=ax2, lags=40, color="#d62728", title="Autocorrelation Function (ACF)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    # 3. Partial Autocorrelation (PACF)
    ax3 = fig.add_subplot(gs[1, 1])
    plot_pacf(df_diff["FirstDifference"], ax=ax3, lags=40, color="#9467bd", title="Partial Autocorrelation Function (PACF)")
    ax3.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved stationarity & ACF/PACF plot to {save_path}")


def plot_single_model_prediction(
    df_eval: pd.DataFrame,
    model_name: str,
    metrics: Dict[str, Any],
    save_path: Path
) -> None:
    """Plots individual model predictions vs actual prices with error subplot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    
    dates = pd.to_datetime(df_eval.index)
    actual = df_eval["Actual_Price"].values
    predicted = df_eval["Predicted_Price"].values
    errors = actual - predicted
    
    # Prediction vs Actual
    ax1.plot(dates, actual, label="Actual Price", color="#d62728", linewidth=2.0)
    ax1.plot(dates, predicted, label=f"{model_name} Predicted", color="#2ca02c", linewidth=2.0, linestyle="--")
    ax1.set_title(
        f"{model_name} Prediction vs Actual | RMSE: {metrics['RMSE']:.4f}, MAPE: {metrics['MAPE (%)']:.2f}%, MASE: {metrics['MASE']:.4f}",
        fontsize=13, fontweight="bold", pad=10
    )
    ax1.set_ylabel("Price (INR)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # Residual Error Plot
    ax2.bar(dates, errors, color="#1f77b4", width=1.5, alpha=0.7, label="Residual Error (Actual - Predicted)")
    ax2.axhline(0, color="black", linestyle="--", alpha=0.8)
    ax2.set_ylabel("Error (INR)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved {model_name} plot to {save_path}")


def plot_all_models_comparison(
    eval_dict: Dict[str, pd.DataFrame],
    save_path: Path = OUTPUTS_DIR / "model_comparison.png"
) -> None:
    """Plots all model predictions against the actual price on a single canvas."""
    plt.figure(figsize=(15, 8))
    
    first_key = list(eval_dict.keys())[0]
    dates = pd.to_datetime(eval_dict[first_key].index)
    actual = eval_dict[first_key]["Actual_Price"].values
    
    plt.plot(dates, actual, label="Actual NIFTY 50 Price", color="black", linewidth=2.5, zorder=5)
    
    palette = {"LSTM": "#1f77b4", "Bi-LSTM": "#ff7f0e", "GRU": "#2ca02c", "Deep_GRU": "#9467bd"}
    for name, df_eval in eval_dict.items():
        color = palette.get(name, None)
        plt.plot(dates, df_eval["Predicted_Price"].values, label=f"Predicted ({name})", linewidth=1.8, linestyle="--", color=color)
        
    plt.title("NIFTY 50 Test Period Forecast: All Deep Learning Architectures", fontsize=15, fontweight="bold", pad=12)
    plt.ylabel("Price (INR)", fontsize=12, fontweight="bold")
    plt.xlabel("Date", fontsize=12, fontweight="bold")
    plt.legend(loc="upper left", frameon=True, fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved overall model comparison plot to {save_path}")


def plot_metrics_bar_chart(
    metrics_df: pd.DataFrame,
    save_path: Path = OUTPUTS_DIR / "metrics_comparison_barchart.png"
) -> None:
    """Plots comparative bar charts across all evaluation metrics."""
    metrics_to_plot = ["RMSE", "MAE", "MAPE (%)", "MASE"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    
    for i, metric in enumerate(metrics_to_plot):
        sns.barplot(data=metrics_df, x="Model", y=metric, ax=axes[i], palette="viridis", hue="Model", legend=False)
        axes[i].set_title(f"Model Comparison - {metric}", fontsize=13, fontweight="bold")
        axes[i].set_ylabel(metric, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].grid(True, linestyle="--", alpha=0.6)
        
        # Annotate bar values
        for p in axes[i].patches:
            h = p.get_height()
            if not np.isnan(h):
                axes[i].annotate(f"{h:.4f}" if metric != "MAPE (%)" else f"{h:.2f}%",
                                (p.get_x() + p.get_width() / 2., h),
                                ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 3),
                                textcoords='offset points')
                
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved metrics comparison bar chart to {save_path}")


def plot_strategy_equity_curves(
    trading_results: Dict[str, Dict[str, Any]],
    save_path: Path = OUTPUTS_DIR / "strategy_equity_curves.png"
) -> None:
    """
    Plots a 2x2 grid of strategy equity curves (Long/Cash vs Buy & Hold) for all models.
    """
    palette = {"LSTM": "#1f77b4", "Bi-LSTM": "#ff7f0e", "GRU": "#2ca02c", "Deep_GRU": "#9467bd"}
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, (mname, t) in enumerate(trading_results.items()):
        ax = axes[idx]
        dates = pd.to_datetime(t["_dates"])
        ax.plot(dates, t["_bh_cumulative"], label="Buy & Hold", color="#d62728", linewidth=2.0, linestyle="--")
        ax.plot(dates, t["_strategy_cumulative"], label=f"{mname} Strategy", color=palette.get(mname, "#1f77b4"), linewidth=2.0)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
        ax.fill_between(
            dates, 1.0, t["_strategy_cumulative"],
            where=(t["_strategy_cumulative"] >= 1.0), alpha=0.12, color=palette.get(mname, "#1f77b4"), label="Profit zone"
        )
        ax.fill_between(
            dates, 1.0, t["_strategy_cumulative"],
            where=(t["_strategy_cumulative"] < 1.0), alpha=0.12, color="red", label="Loss zone"
        )
        ax.set_title(
            f"{mname} | Hit: {t['Directional_Accuracy (%)']:.1f}% | Sharpe: {t['Sharpe_Ratio']:.2f} | DD: {t['Max_Drawdown (%)']:.1f}%",
            fontsize=11, fontweight="bold"
        )
        ax.set_ylabel("Portfolio Value (INR 1 start)", fontsize=10)
        ax.set_xlabel("Date", fontsize=10)
        ax.legend(fontsize=9, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)
        
    plt.suptitle(
        "Quantitative Strategy Equity Curves: Long/Cash vs Buy & Hold (NIFTY 50 Test Period)",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved strategy equity curves plot to {save_path}")

