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


def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothed moving average (RMA). Identical to ta.atr() in TradingView."""
    alpha = 1.0 / period
    result = np.full(len(series), np.nan)
    first_valid = series.first_valid_index()
    if first_valid is None:
        return pd.Series(result, index=series.index)
    start_idx = series.index.get_loc(first_valid)
    if start_idx + period - 1 >= len(series):
        return pd.Series(result, index=series.index)
    result[start_idx + period - 1] = series.iloc[start_idx : start_idx + period].mean()
    for i in range(start_idx + period, len(series)):
        result[i] = alpha * series.iloc[i] + (1 - alpha) * result[i - 1]
    return pd.Series(result, index=series.index)


def compute_chandelier_exit(df: pd.DataFrame, period: int = 22, mult: float = 3.0, use_close: bool = True) -> pd.DataFrame:
    """
    Computes Pine Script Chandelier Exit (Alex Orekhov, GPL-3.0) with trailing logic.
    Returns DataFrame with new columns: atr, ce_long, ce_short, ce_dir, ce_signal
    """
    df = df.copy()
    n = len(df)

    tr = compute_true_range(df)
    atr_raw = _wilder_rma(tr, period)
    atr = mult * atr_raw
    df["atr"] = atr_raw

    if use_close:
        hh = df["close"].rolling(period).max()
        ll = df["close"].rolling(period).min()
    else:
        hh = df["high"].rolling(period).max()
        ll = df["low"].rolling(period).min()

    long_raw = (hh - atr).values
    short_raw = (ll + atr).values
    close_arr = df["close"].values

    long_stop = np.full(n, np.nan)
    short_stop = np.full(n, np.nan)
    direction = np.ones(n, dtype=int)

    first_valid = 0
    while first_valid < n and np.isnan(long_raw[first_valid]):
        first_valid += 1

    if first_valid < n:
        long_stop[first_valid] = long_raw[first_valid]
        short_stop[first_valid] = short_raw[first_valid]

    for i in range(first_valid + 1, n):
        if np.isnan(long_raw[i]):
            continue

        prev_ls = long_stop[i - 1]
        prev_ss = short_stop[i - 1]
        prev_c = close_arr[i - 1]
        cur_c = close_arr[i]

        if not np.isnan(prev_ls):
            if prev_c > prev_ls:
                long_stop[i] = max(long_raw[i], prev_ls)
            else:
                long_stop[i] = long_raw[i]
        else:
            long_stop[i] = long_raw[i]

        if not np.isnan(prev_ss):
            if prev_c < prev_ss:
                short_stop[i] = min(short_raw[i], prev_ss)
            else:
                short_stop[i] = short_raw[i]
        else:
            short_stop[i] = short_raw[i]

        prev_dir = direction[i - 1]
        if not np.isnan(prev_ss) and cur_c > prev_ss:
            direction[i] = 1
        elif not np.isnan(prev_ls) and cur_c < prev_ls:
            direction[i] = -1
        else:
            direction[i] = prev_dir

    df["ce_long"] = long_stop
    df["ce_short"] = short_stop
    df["ce_dir"] = direction
    df["ce_signal"] = (df["ce_dir"] != df["ce_dir"].shift(1)) & df["atr"].notna()

    return df


def compute_donchian_channel(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Computes Donchian Channel max, min, and width percentage."""
    df = df.copy()
    df["donchian_max"] = df["high"].rolling(window=period).max()
    df["donchian_min"] = df["low"].rolling(window=period).min()
    df["donchian_width"] = df["donchian_max"] - df["donchian_min"]
    
    # Avoid div by zero, use mid price
    mid = (df["donchian_max"] + df["donchian_min"]) / 2
    mid = mid.replace(0, 1e-8)
    df["donchian_width_pct"] = df["donchian_width"] / mid
    return df
