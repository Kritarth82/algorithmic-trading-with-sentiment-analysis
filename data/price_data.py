"""
Phase 1 — Market Dataset Construction

Builds the market-side feature matrix used to test the incremental effect of
news-derived sentiment proxies on forward returns.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import StandardScaler

from config import (
    TICKERS, TRAIN_START, TRAIN_END, VAL_START, VAL_END,
    TEST_START, TEST_END, PRICE_FEATURES, TARGET_COL,
    FORECAST_HORIZON, PRICE_CACHE_DIR, RANDOM_SEED,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Technical indicator computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators and add to dataframe."""
    df = df.copy()

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    def sma(s, n): return s.rolling(n).mean()
    def ema(s, n): return s.ewm(span=n, adjust=False).mean()

    df["sma_20"]  = sma(close, 20)
    df["sma_50"]  = sma(close, 50)
    df["ema_12"]  = ema(close, 12)
    df["ema_26"]  = ema(close, 26)

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["rsi_14"]  = 100 - 100 / (1 + rs)

    # MACD
    df["macd"]        = ema(close, 12) - ema(close, 26)
    df["macd_signal"] = ema(df["macd"], 9)
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    mid = sma(close, 20)
    std = close.rolling(20).std()
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / close

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # OBV
    direction       = np.sign(close.diff()).fillna(0)
    df["obv"]       = (direction * vol).cumsum()

    # Stochastic
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["stoch_k"] = 100 * (close - low14) / (high14 - low14 + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # Returns
    df["returns"]     = close.pct_change()
    df["log_returns"] = np.log(close / close.shift(1))
    df["vol_20"]      = df["returns"].rolling(20).std()

    # Target: 5-day forward return
    df[TARGET_COL]    = close.shift(-FORECAST_HORIZON) / close - 1

    return df


def _download_ticker(ticker: str) -> pd.DataFrame:
    """Download OHLCV from yfinance and normalise column names."""
    logger.info(f"Downloading {ticker} from yfinance ...")
    raw = yf.download(ticker, start=TRAIN_START, end=TEST_END,
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    # yfinance may return MultiIndex columns when downloading single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
    raw.index   = pd.to_datetime(raw.index)
    raw.index.name = "date"
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_price_dataset(
    tickers: list[str] | None = None,
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV data and compute technical indicators for every ticker.
    Returns dict[ticker -> DataFrame] with DatetimeIndex (business days).
    """
    tickers = tickers or TICKERS
    datasets: Dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        cache_path = PRICE_CACHE_DIR / f"{ticker}_price_raw.csv"

        if use_cache and cache_path.exists():
            logger.info(f"Loading {ticker} from cache: {cache_path}")
            df = pd.read_csv(cache_path, index_col="date", parse_dates=True)
        else:
            raw = _download_ticker(ticker)
            df  = _compute_indicators(raw)
            df.dropna(subset=["close"], inplace=True)
            PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path)
            logger.info(f"Saved {ticker} price data → {cache_path}")

        datasets[ticker] = df

    return datasets


def chronological_split(
    df: pd.DataFrame,
    val_start: str  = VAL_START,
    test_start: str = TEST_START,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-indexed DataFrame chronologically — no shuffling, no leakage.
    Returns (df_train, df_val, df_test).
    """
    idx = df.index
    df_train = df[idx <  val_start]
    df_val   = df[(idx >= val_start) & (idx < test_start)]
    df_test  = df[idx >= test_start]
    return df_train, df_val, df_test


def normalize_features(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    feature_cols: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Fit StandardScaler on training set, transform val/test with the same params.
    Returns transformed DataFrames and scaler_params dict for inverse-transform.
    """
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].fillna(0))

    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[feature_cols] = scaler.transform(df[feature_cols].fillna(0))
        return out

    scaler_params = {
        "mean_":  scaler.mean_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "cols":   feature_cols,
    }

    return _transform(df_train), _transform(df_val), _transform(df_test), scaler_params


def inverse_transform_target(
    values: np.ndarray,
    scaler_params: dict,
) -> np.ndarray:
    """Inverse-transform a 1-D array of scaled target values."""
    cols  = scaler_params["cols"]
    if TARGET_COL not in cols:
        return values  # target was not scaled
    idx   = cols.index(TARGET_COL)
    mean_ = scaler_params["mean_"][idx]
    std_  = scaler_params["scale_"][idx]
    return values * std_ + mean_
