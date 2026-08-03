"""Shared statistical and technical indicators for both trading bots."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis


def get_hurst_exponent(price_series: np.ndarray | pd.Series, min_lag: int = 8, max_lag: int = 100) -> float:
    """Computes the Hurst Exponent (H) using classical Rescaled Range (R/S) analysis.

    H > 0.5 indicates persistence (trending memory).
    H ~ 0.5 indicates random walk.
    H < 0.5 indicates mean reversion.
    """
    price_series = np.asarray(price_series, dtype=float)
    n = len(price_series)
    if n < 20:
        return 0.5

    max_lag = min(max_lag, n // 2)
    if max_lag <= min_lag:
        return 0.5

    log_ret = np.diff(np.log(price_series))
    if len(log_ret) < max_lag:
        return 0.5

    n_ret = len(log_ret)
    valid_lags = []
    rs_values = []

    for lag in range(min_lag, max_lag):
        num_blocks = n_ret // lag
        if num_blocks < 1:
            continue

        # Vectorized block processing: reshape into (num_blocks, lag)
        blocks = log_ret[:num_blocks * lag].reshape(num_blocks, lag)
        means = blocks.mean(axis=1, keepdims=True)
        deviations = np.cumsum(blocks - means, axis=1)
        R = deviations.max(axis=1) - deviations.min(axis=1)
        S = blocks.std(axis=1, ddof=1)
        mask = S > 1e-12
        if mask.any():
            rs_values.append(np.mean(R[mask] / S[mask]))
            valid_lags.append(lag)

    if len(valid_lags) < 2:
        return 0.5

    poly = np.polyfit(np.log(valid_lags), np.log(rs_values), 1)
    return float(poly[0])


def compute_clv(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Computes Close Location Value (CLV).

    CLV <= 0.15: Close near extreme low (selling climax / capitulation).
    CLV >= 0.85: Close near extreme high (bullish absorption / reversal).
    """
    rng = high - low
    rng_safe = rng.replace(0, 1e-8)
    clv = (close - low) / rng_safe
    return clv.clip(0.0, 1.0)


def compute_true_range(df: pd.DataFrame) -> pd.Series:
    """Computes True Range (TR) from DataFrame with high, low, close."""
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def compute_tr_zscore(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """Computes rolling Z-Score of True Range over specified window (default 30 days).

    Z_TR > 3.0 indicates extreme volatility explosion (3-sigma phase transition).
    """
    tr = compute_true_range(df)
    tr_mean = tr.rolling(window=window, min_periods=max(1, window // 2)).mean()
    tr_std = tr.rolling(window=window, min_periods=max(1, window // 2)).std().replace(0, 1e-8)
    return (tr - tr_mean) / tr_std


def compute_excess_kurtosis(returns_series: pd.Series, window: int = 365) -> pd.Series:
    """Computes rolling excess kurtosis of logarithmic returns.

    K > 5.0 indicates significant heavy tail properties (Levy/Pareto distribution).
    """
    return returns_series.rolling(window=window, min_periods=min(180, window)).apply(
        lambda x: kurtosis(x, fisher=True, bias=True), raw=True
    )
