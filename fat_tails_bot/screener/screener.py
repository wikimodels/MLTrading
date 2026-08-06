"""Quantitative asset selection screener based on Hurst exponent and Excess Kurtosis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import kurtosis

from shared.config_loader import get_config
from shared.data.storage import DataStorage
from shared.indicators import get_hurst_exponent


def load_daily_ohlcv(storage: DataStorage, symbol: str, timeframe: str = "4h") -> pd.DataFrame | None:
    """Loads 4H OHLCV data from storage and resamples to 1D for structural metrics."""
    df_4h = storage.load_ohlcv(symbol, timeframe)
    
    if df_4h is None or len(df_4h) < 200:
        return None
        
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

    df = df.dropna(subset=["close", "volume"])
    
    if len(df) < 180:
        return None
        
    return df


class FatTailsScreener:
    """Screening universe of symbols using trend persistence (Hurst) and heavy tails (Kurtosis)."""

    def __init__(self):
        self.cfg = get_config()
        self.storage = DataStorage()
        self.lookback_days = self.cfg.get("strategy", {}).get("screener_lookback_days", 180)
        self.top_n = self.cfg.get("symbol_universe", {}).get("groups", {}).get("main", {}).get("top_n", 20)
        self.exclude = set(self.cfg.get("global_exclude_symbols", []))

    def screen_universe(self) -> pd.DataFrame:
        """Evaluates all stored symbols over the last 6 months and returns the Top N trending coins."""
        symbols = self.storage.list_symbols("4h")
        results = []

        logger.info(f"Running Top {self.top_n} Dynamic Screener (Lookback: {self.lookback_days} days)")

        for sym in symbols:
            if sym in self.exclude:
                continue

            df = load_daily_ohlcv(self.storage, sym)
            if df is None:
                continue
                
            # Filter to the lookback window (e.g., last 180 days)
            df = df.tail(self.lookback_days).copy()
            if len(df) < 60:
                continue

            # Calculate log returns
            returns = np.log(df["close"] / df["close"].shift(1)).dropna()
            if len(returns) < 30:
                continue

            hurst = get_hurst_exponent(df["close"].values, min_lag=10, max_lag=min(60, len(df)//3))
            excess_kurt = float(kurtosis(returns, fisher=True, bias=True))

            results.append({
                "symbol": sym,
                "hurst": hurst,
                "kurtosis": excess_kurt,
            })

        if not results:
            logger.warning("No qualifying symbols evaluated in screener.")
            return pd.DataFrame()

        df_res = pd.DataFrame(results)
        
        # Rank them
        df_res["hurst_rank"] = df_res["hurst"].rank(ascending=False)
        df_res["kurt_rank"] = df_res["kurtosis"].rank(ascending=False)
        df_res["score"] = df_res["hurst_rank"] + df_res["kurt_rank"]
        
        # Sort by best score (lowest sum of ranks)
        df_res = df_res.sort_values("score", ascending=True)
        df_res["selected"] = False
        
        # Select top N
        top_indices = df_res.head(self.top_n).index
        df_res.loc[top_indices, "selected"] = True
        
        logger.info(f"Screener evaluation completed: {self.top_n} symbols selected out of {len(df_res)}.")
        logger.info(f"Top 5: {df_res.head(5)['symbol'].tolist()}")
        return df_res
