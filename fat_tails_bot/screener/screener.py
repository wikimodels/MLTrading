"""Quantitative asset selection screener based on Hurst exponent and Excess Kurtosis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import kurtosis

from shared.config_loader import get_config
from shared.data.storage import DataStorage
from shared.indicators import get_hurst_exponent


def load_daily_ohlcv(storage: DataStorage, symbol: str) -> pd.DataFrame | None:
    """Loads 1D OHLCV data from storage, with automatic fallback to resampled 4H data."""
    df = storage.load_ohlcv(symbol, "1d")
    
    if df is None or len(df) < 100:
        df_4h = storage.load_ohlcv(symbol, "4h")
        if df_4h is not None and len(df_4h) >= 400:
            logger.debug(f"Resampling 4H data to 1D for {symbol}")
            df_4h = df_4h.copy()
            df_4h["date"] = df_4h["timestamp"].dt.floor("D")
            df = df_4h.groupby("date").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "timestamp": "last"
            }).reset_index(drop=True)
        else:
            df = None

    if df is None:
        return None

    # Clean data and enforce constraints
    df = df.dropna(subset=["close", "volume"])
    
    if len(df) < 365:
        return None
        
    # Max history 3 years (1095 days)
    if len(df) > 1095:
        df = df.tail(1095).reset_index(drop=True)
        
    return df


class FatTailsScreener:
    """Screening universe of symbols using trend persistence (Hurst) and heavy tails (Kurtosis)."""

    def __init__(self):
        self.cfg = get_config()
        self.storage = DataStorage()
        self.hurst_min = self.cfg.get("strategy", {}).get("hurst_min", 0.25)
        self.kurtosis_min = self.cfg.get("strategy", {}).get("kurtosis_min", 5.0)
        self.exclude = set(self.cfg.get("global_exclude_symbols", []))

    def screen_universe(self) -> pd.DataFrame:
        """Evaluates all stored symbols and returns filtered candidates sorted by excess kurtosis."""
        symbols = self.storage.list_symbols("4h")
        results = []

        for sym in symbols:
            if sym in self.exclude:
                continue

            df = load_daily_ohlcv(self.storage, sym)
            if df is None or len(df) < 150:
                continue

            # Calculate log returns
            returns = np.log(df["close"] / df["close"].shift(1)).dropna()
            if len(returns) < 100:
                continue

            # Calculate metrics
            hurst = get_hurst_exponent(df["close"].tail(180).values, max_lag=20)
            excess_kurt = float(kurtosis(returns.tail(180), fisher=True, bias=True))

            passed = (hurst >= self.hurst_min) and (excess_kurt >= self.kurtosis_min)
            results.append({
                "symbol": sym,
                "hurst_exponent": round(hurst, 4),
                "excess_kurtosis": round(excess_kurt, 2),
                "bars_1d": len(df),
                "selected": passed,
            })

        if not results:
            logger.warning("No qualifying symbols evaluated in screener.")
            return pd.DataFrame()

        df_res = pd.DataFrame(results).sort_values("excess_kurtosis", ascending=False)
        selected_count = df_res["selected"].sum()
        logger.info(f"Screener evaluation completed: {selected_count}/{len(df_res)} symbols meet quantitative thresholds.")
        return df_res
