"""
Data loader module for fetching, caching, and preprocessing NIFTY 50 time series data.
"""

import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from src.config import TICKER, START_DATE, END_DATE, CSV_CACHE_PATH


def load_nifty50_data(
    ticker: str = TICKER,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetches historical stock data from Yahoo Finance and caches it locally as CSV.
    
    Returns:
        pd.DataFrame: Cleaned dataframe with 'date' and 'close' columns.
    """
    df = None
    
    # Try fetching fresh data via yfinance
    try:
        session = requests.Session()
        session.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        
        print(f"[DataLoader] Fetching ticker '{ticker}' from {start_date} to {end_date} via yfinance...")
        raw_df = yf.download(ticker, start=start_date, end=end_date, session=session, timeout=15)
        
        if raw_df is not None and not raw_df.empty:
            if isinstance(raw_df.columns, pd.MultiIndex):
                raw_df.columns = raw_df.columns.get_level_values(0)
            
            raw_df.reset_index(inplace=True)
            df = raw_df[["Date", "Close"]].copy()
            df.rename(columns={"Date": "date", "Close": "close"}, inplace=True)
            df["close"] = df["close"].astype(float)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df.dropna(inplace=True)
            
            # Cache locally
            df.to_csv(CSV_CACHE_PATH, index=False)
            print(f"[DataLoader] Successfully downloaded and cached {len(df)} records to {CSV_CACHE_PATH}")
    except Exception as e:
        print(f"[DataLoader] Warning: yfinance download encountered an issue ({e}).")

    # Fallback to cache if download didn't produce data
    if df is None or df.empty:
        if use_cache and os.path.exists(CSV_CACHE_PATH):
            print(f"[DataLoader] Loading cached historical dataset from {CSV_CACHE_PATH}...")
            df = pd.read_csv(CSV_CACHE_PATH)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df["close"] = df["close"].astype(float)
        else:
            raise RuntimeError("Failed to fetch NIFTY 50 data from yfinance and no local cache was found.")
            
    return df


def prepare_log_transformed_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes natural log of the closing price and sets the date as the index.
    
    Returns:
        pd.DataFrame: Dataframe indexed by date with 'close' as log(price).
    """
    df_log = pd.DataFrame(index=df["date"])
    df_log["close"] = np.log(df["close"].values)
    return df_log
