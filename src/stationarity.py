"""
Statistical tests for stationarity analysis: Augmented Dickey-Fuller (ADF) & KPSS tests.
"""

from typing import Dict, Any
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def check_adf_stationarity(series: pd.Series, name: str = "Series", verbose: bool = True) -> Dict[str, Any]:
    """
    Performs the Augmented Dickey-Fuller (ADF) test for stationarity.
    H0: The series has a unit root (non-stationary).
    H1: The series is stationary.
    """
    clean_series = series.dropna()
    result = adfuller(clean_series)
    adf_stat, p_value, lags, nobs, crit_values, icbest = result
    is_stationary = p_value <= 0.05
    
    res = {
        "test": "ADF",
        "series_name": name,
        "statistic": adf_stat,
        "p_value": p_value,
        "used_lags": lags,
        "n_obs": nobs,
        "critical_values": crit_values,
        "is_stationary": is_stationary
    }
    
    if verbose:
        print(f"\n--- Augmented Dickey-Fuller (ADF) Test: {name} ---")
        print(f"ADF Statistic: {adf_stat:.6f}")
        print(f"p-value: {p_value:.6e}")
        print(f"Used Lags: {lags}, Observations: {nobs}")
        print("Critical Values:")
        for k, v in crit_values.items():
            print(f"   {k}: {v:.4f}")
        if is_stationary:
            print("=> Result: The series IS stationary (Reject H0 at alpha=0.05).")
        else:
            print("=> Result: The series is NOT stationary (Fail to reject H0).")
            
    return res


def check_kpss_stationarity(series: pd.Series, name: str = "Series", verbose: bool = True) -> Dict[str, Any]:
    """
    Performs the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for stationarity.
    H0: The series is level or trend stationary.
    H1: The series has a unit root (non-stationary).
    """
    clean_series = series.dropna()
    # statsmodels kpss returns: statistic, p_value, n_lags, critical_values
    stat, p_val, lags, crit_vals = kpss(clean_series, regression="c", nlags="auto")
    is_stationary = p_val >= 0.05
    
    res = {
        "test": "KPSS",
        "series_name": name,
        "statistic": stat,
        "p_value": p_val,
        "used_lags": lags,
        "critical_values": crit_vals,
        "is_stationary": is_stationary
    }
    
    if verbose:
        print(f"\n--- KPSS Test: {name} ---")
        print(f"KPSS Statistic: {stat:.6f}")
        print(f"p-value: {p_val:.6f}")
        print(f"Number of Lags: {lags}")
        print("Critical Values:")
        for k, v in crit_vals.items():
            print(f"   {k}: {v:.4f}")
        if is_stationary:
            print("=> Result: The series IS stationary (Fail to reject H0).")
        else:
            print("=> Result: The series is NOT stationary (Reject H0 at alpha=0.05).")
            
    return res


def compute_first_difference(df_log: pd.DataFrame) -> pd.DataFrame:
    """
    Computes first order difference for the log-price series.
    """
    df_diff = df_log.copy()
    df_diff["FirstDifference"] = df_diff["close"] - df_diff["close"].shift(1)
    df_diff.dropna(inplace=True)
    return df_diff
