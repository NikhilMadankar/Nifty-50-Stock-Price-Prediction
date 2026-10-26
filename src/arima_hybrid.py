"""
ARIMA model fitting and hybrid feature engineering module.

Fix 3 — Train/Test Contamination:
    Previously, ARIMA was fit on the ENTIRE dataset (2015-2025) before splitting.
    This contaminated the test-period residuals with future training data.

    Correct approach (implemented here):
        1. Split df_diff into train (70%) / test (30%) FIRST.
        2. Fit ARIMA strictly on the TRAIN partition only.
        3. Obtain in-sample residuals from the TRAIN fit.
        4. Apply the FROZEN ARIMA coefficients to the TEST partition via model.apply()
           to generate genuine out-of-sample one-step-ahead residuals.
        5. Reconstruct causal hybrid series separately for train and test.
"""

from typing import Tuple, Any
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from src.config import ARIMA_ORDER, TRAIN_RATIO


def fit_arima_on_train(
    train_diff: pd.Series,
    order: Tuple[int, int, int] = ARIMA_ORDER,
    verbose: bool = True
) -> Any:
    """
    Fits ARIMA strictly on the training partition of the first-differenced series.

    Correct order is (p=1, d=0, q=0):
        - `FirstDifference` = ln(P_t) - ln(P_{t-1}) is confirmed stationary by ADF (p < 1e-25).
        - Setting d=1 on an already-stationary series causes *over-differencing* (equivalent
          to d=2 on the raw log price), introducing artificial MA roots and degrading AIC
          by ~1,051 points relative to ARIMA(1,0,0).

    Args:
        train_diff: Training partition of the FirstDifference series.
        order: ARIMA (p, d, q) order. Default from config.
        verbose: Whether to print the model summary.

    Returns:
        Fitted ARIMA result object (frozen coefficients).
    """
    model = ARIMA(train_diff.dropna(), order=order)
    result = model.fit()

    if verbose:
        print(f"\n--- ARIMA{order} Model Fit Summary (Train Only) ---")
        print(result.summary())

    return result


def get_out_of_sample_residuals(arima_result: Any, test_diff: pd.Series) -> pd.Series:
    """
    Generates genuine out-of-sample one-step-ahead residuals on the test partition.

    Uses `model.apply()` to freeze the ARIMA coefficients from the training fit
    and apply them to the test series — NO re-estimation on test data.

    Args:
        arima_result: ARIMA result fitted on training data.
        test_diff: Test partition of the FirstDifference series.

    Returns:
        pd.Series: One-step-ahead residuals for the test period.
    """
    # Apply frozen ARIMA parameters to test data (no re-fitting)
    test_model = arima_result.apply(test_diff.dropna(), refit=False)
    return test_model.resid


def build_hybrid_series(
    df_diff: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    order: Tuple[int, int, int] = ARIMA_ORDER,
    verbose: bool = True
) -> Tuple[pd.Series, Any]:
    """
    Full pipeline: splits df_diff, fits ARIMA on train only, applies to test,
    and reconstructs the causal hybrid series across the full index.

    Pipeline (Contamination-Free):
        1. Split df_diff at train_ratio boundary.
        2. Fit ARIMA(p, 0, q) on train_diff ONLY.
        3. Get in-sample residuals (train) and out-of-sample residuals (test).
        4. Concatenate residuals across full index and cumsum.
        5. Reconstruct: hybrid_t = cumsum(e)_t + ln(P_{t-1})

    Args:
        df_diff: DataFrame with 'FirstDifference' and 'close' columns.
        train_ratio: Fraction of data to use for ARIMA training.
        order: ARIMA order tuple.
        verbose: Print ARIMA summary.

    Returns:
        Tuple of:
            - hybrid_series (pd.Series): Causal hybrid target across full index.
            - arima_result: Fitted ARIMA result (for reporting AIC/BIC).
    """
    series = df_diff["FirstDifference"].dropna()
    split_idx = int(len(series) * train_ratio)

    train_diff = series.iloc[:split_idx]
    test_diff = series.iloc[split_idx:]

    # Fit ARIMA strictly on training partition
    arima_result = fit_arima_on_train(train_diff, order=order, verbose=verbose)

    # In-sample residuals (train)
    train_resid = arima_result.resid

    # Out-of-sample residuals (test) — frozen coefficients, no re-fitting
    test_resid = get_out_of_sample_residuals(arima_result, test_diff)

    # Concatenate residuals across full timeline
    all_resid = pd.concat([train_resid, test_resid]).sort_index()

    # Causal cumulative residuals: cumsum(e)_t uses only e_1..e_t (no lookahead)
    cumulative_resid = all_resid.cumsum()

    # a_t = ln(P_{t-1}) = ln(P_t) - FirstDifference_t
    a = df_diff["close"] - df_diff["FirstDifference"]

    # Hybrid series: strictly causal (no shift(-1) lookahead)
    hybrid_series = cumulative_resid + a
    hybrid_series = hybrid_series.dropna()

    if verbose:
        print(f"\nARIMA AIC: {arima_result.aic:.4f} | BIC: {arima_result.bic:.4f}")
        print(f"Train residuals: {len(train_resid)} | Test residuals: {len(test_resid)}")
        print(f"Hybrid series size: {len(hybrid_series)}")

    return hybrid_series, arima_result
