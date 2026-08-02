"""Signal generation based on Volatility Phase Transitions and Close Location Value."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from shared.config_loader import get_config
from shared.indicators import compute_clv, compute_true_range, compute_tr_zscore


class SignalGenerator:
    """Generates trading signals for Fat Tails D1 breakout strategy."""

    def __init__(self):
        self.cfg = get_config()
        strat_cfg = self.cfg.get("strategy", {})
        self.tr_zscore_min = strat_cfg.get("tr_zscore_min", 2.5)
        self.clv_short_max = strat_cfg.get("clv_short_max", 0.25)
        self.clv_long_min = strat_cfg.get("clv_long_min", 0.75)
        self.allow_short = strat_cfg.get("allow_short", True)
        self.allow_long = strat_cfg.get("allow_long", False)
        self.sl_mult = self.cfg.get("risk", {}).get("stop_loss_atr_mult", 1.5)

    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Appends technical indicators, ATR, signals, and dynamic stop loss bounds to DataFrame."""
        if df is None or len(df) < 30:
            return pd.DataFrame()

        df = df.copy()

        # Compute Core Indicators
        tr = compute_true_range(df)
        df["atr_14"] = tr.rolling(14, min_periods=7).mean().replace(0, 1e-8)
        df["tr_zscore"] = compute_tr_zscore(df, window=30)
        df["clv"] = compute_clv(df["high"], df["low"], df["close"])
        df["q01"] = df["close"].rolling(window=90, min_periods=30).apply(lambda x: np.quantile(x, 0.01), raw=True)
        df["ema5"] = df["close"].ewm(span=5, adjust=False).mean()

        # Initialize signal columns
        df["signal_short"] = 0
        df["signal_long"] = 0
        df["initial_sl"] = np.nan

        # Short Trigger per theory: Z-Score of TR >= threshold, CLV <= max (capitulation), and Close <= 90d 1% Quantile
        if self.allow_short:
            cond_short = (
                (df["tr_zscore"] >= self.tr_zscore_min) &
                (df["clv"] <= self.clv_short_max) &
                (df["close"] <= df["q01"])
            )
            df.loc[cond_short, "signal_short"] = 1
            df.loc[cond_short, "initial_sl"] = df.loc[cond_short, "close"] + self.sl_mult * df.loc[cond_short, "atr_14"]

        # Long Trigger (optional reversal)
        if self.allow_long:
            cond_long = (
                (df["tr_zscore"] >= self.tr_zscore_min) &
                (df["clv"] >= self.clv_long_min) &
                (df["close"] >= df["close"].rolling(window=90, min_periods=30).apply(lambda x: np.quantile(x, 0.99), raw=True))
            )
            df.loc[cond_long, "signal_long"] = 1
            df.loc[cond_long, "initial_sl"] = df.loc[cond_long, "close"] - self.sl_mult * df.loc[cond_long, "atr_14"]

        return df
