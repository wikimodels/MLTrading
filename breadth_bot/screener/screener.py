"""Universe selection for breadth bot: coins whose 1h returns are correlated with BTC."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from shared.config_loader import get_config
from shared.data.storage import DataStorage


def load_hourly_ohlcv(storage: DataStorage, symbol: str) -> pd.DataFrame | None:
    """Loads 1H OHLCV data, cleaned and indexed by timestamp."""
    df = storage.load_ohlcv(symbol, "1h")
    if df is None or len(df) < 1000:
        return None
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df.dropna(subset=["close", "high", "low", "open"])
    if len(df) < 1000:
        return None
    return df


def returns_series(df: pd.DataFrame) -> pd.Series:
    return np.log(df["close"] / df["close"].shift(1))


class BreadthScreener:
    """Selects coins correlated with BTC (log-return correlation over trailing window)."""

    def __init__(self):
        self.cfg = get_config()
        self.storage = DataStorage()
        sc = self.cfg.get("screener", {})
        self.timeframe = sc.get("timeframe", "1h")
        self.btc_symbol = sc.get("btc_symbol", "BTC/USDT:USDT")
        self.corr_min = sc.get("correlation_min", 0.4)
        self.corr_window_days = sc.get("correlation_window_days", 365)

    def select_universe(self) -> list[str]:
        """Returns symbols (excluding BTC) with hourly correlation to BTC > threshold."""
        btc = load_hourly_ohlcv(self.storage, self.btc_symbol)
        if btc is None:
            logger.error("BTC data not found; cannot compute correlations.")
            return []
        btc_ret = returns_series(btc)
        n_hours = self.corr_window_days * 24
        btc_ret = btc_ret.tail(n_hours)

        symbols = [s for s in self.storage.list_symbols("1h") if s != self.btc_symbol]
        selected = []
        for sym in symbols:
            df = load_hourly_ohlcv(self.storage, sym)
            if df is None:
                continue
            r = returns_series(df).tail(n_hours)
            joined = pd.concat([r, btc_ret], axis=1, join="inner").dropna()
            if len(joined) < 500:
                continue
            c = joined.iloc[:, 0].corr(joined.iloc[:, 1])
            if np.isfinite(c) and c > self.corr_min:
                selected.append(sym)

        logger.info(f"Universe: {len(selected)}/{len(symbols)} symbols pass correlation {self.corr_min}.")
        return selected
