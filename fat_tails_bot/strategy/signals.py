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

        # Lagged variables for Day D-1 and confirmation Day D
        tr_z_prev = df["tr_zscore"].shift(1)
        clv_prev = df["clv"].shift(1)
        close_prev = df["close"].shift(1)

        # Initialize signal columns
        df["signal_short"] = 0
        df["signal_long"] = 0
        df["initial_sl"] = np.nan

        # Short Trigger: Day D-1 extreme volatility explosion & close near bottom; Day D continuation
        if self.allow_short:
            cond_short = (
                (tr_z_prev >= self.tr_zscore_min) &
                (clv_prev <= self.clv_short_max) &
                (df["close"] < close_prev) &
                (df["clv"] <= 0.40)
            )
            df.loc[cond_short, "signal_short"] = 1
            df.loc[cond_short, "initial_sl"] = df.loc[cond_short, "close"] + self.sl_mult * df.loc[cond_short, "atr_14"]

        # Long Trigger (optional)
        if self.allow_long:
            cond_long = (
                (tr_z_prev >= self.tr_zscore_min) &
                (clv_prev >= self.clv_long_min) &
                (df["close"] > close_prev) &
                (df["clv"] >= 0.60)
            )
            df.loc[cond_long, "signal_long"] = 1
            df.loc[cond_long, "initial_sl"] = df.loc[cond_long, "close"] - self.sl_mult * df.loc[cond_long, "atr_14"]

        return df
