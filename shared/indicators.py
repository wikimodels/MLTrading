"""Shared statistical and technical indicators for both trading bots."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis


def get_hurst_exponent(time_series: np.ndarray | pd.Series, max_lag: int = 20) -> float:
    """Computes the Hurst Exponent (H) to evaluate trend persistence vs random walk.

    H > 0.5 indicates persistence (trending memory).
    H ~ 0.5 indicates random walk.
    H < 0.5 indicates mean reversion.
    """
    ts = np.asarray(time_series)
    if len(ts) < max_lag * 2:
        return 0.5  # default to random walk if too short

    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        diff = np.subtract(ts[lag:], ts[:-lag])
        std_diff = np.std(diff)
        tau.append(np.sqrt(std_diff) if std_diff > 0 else 1e-8)

    if not tau or any(t == 0 for t in tau):
        return 0.5

    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return float(poly[0] * 2.0)


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
