"""Signal generation based on Chandelier Exit Flip and Donchian Compression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from shared.config_loader import get_config
from shared.indicators import (
    compute_chandelier_exit,
    compute_donchian_channel
)


class SignalGenerator:
    """Generates trading signals for Fat Tails 4H breakout strategy."""

    def __init__(self):
        self.cfg = get_config()
        strat_cfg = self.cfg.get("strategy", {})
        
        self.donchian_period = strat_cfg.get("donchian_period", 20)
        self.donchian_threshold = strat_cfg.get("donchian_threshold", 0.05)
        self.donchian_pct_window = strat_cfg.get("donchian_percentile_window", 100)
        
        self.chandelier_period = strat_cfg.get("chandelier_period", 22)
        self.chandelier_mult = strat_cfg.get("chandelier_atr_mult", 3.0)
        
        self.allow_short = strat_cfg.get("allow_short", True)
        self.allow_long = strat_cfg.get("allow_long", True)
        
        # Initial SL distance is strictly 1 ATR for risk sizing
        self.sl_mult = 1.0 
        
        # Look-ahead bias prevention: conditions checked on shifted bars
        self.screen_lag = 1

    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Appends technical indicators, signals, and SL/TP bounds to DataFrame."""
        if df is None or len(df) < max(self.chandelier_period, self.donchian_pct_window) + 10:
            if df is not None:
                df = df.copy()
                df["entry_signal"] = False
                df["signal_short"] = 0
            return df if df is not None else pd.DataFrame()

        df = df.copy()

        # 1. Compute Donchian Channel
        df = compute_donchian_channel(df, period=self.donchian_period)
        
        # Compute Donchian compression percentile
        df["donchian_width_percentile"] = df["donchian_width_pct"].rolling(
            window=self.donchian_pct_window, min_periods=self.donchian_pct_window // 2
        ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        
        # The compression condition must be met on the PREVIOUS closed bar (lag = 1)
        df["is_compressed"] = df["donchian_width_percentile"].shift(self.screen_lag) <= self.donchian_threshold
        
        # 2. Compute Chandelier Exit
        df = compute_chandelier_exit(df, period=self.chandelier_period, mult=self.chandelier_mult)
        
        # Initialize signal columns
        df["signal_short"] = 0
        df["signal_long"] = 0
        df["initial_sl"] = np.nan
        df["entry_signal"] = False

        # 3. Triggers: Chandelier flipped on THIS bar + Compression on PREV bar
        # In live trading, we check this at the CLOSE of the 4H bar, and enter on NEXT open.
        # ce_signal is True if ce_dir != ce_dir.shift(1)
        
        c_flip = df["ce_signal"] == True
        c_comp = df["is_compressed"] == True
        
        if self.allow_short:
            cond_short = c_flip & c_comp & (df["ce_dir"] == -1)
            df.loc[cond_short, "signal_short"] = 1
            df.loc[cond_short, "entry_signal"] = True
            # SL is 1 ATR above entry (close of flip bar)
            df.loc[cond_short, "initial_sl"] = df.loc[cond_short, "close"] + df.loc[cond_short, "atr"]

        if self.allow_long:
            cond_long = c_flip & c_comp & (df["ce_dir"] == 1)
            df.loc[cond_long, "signal_long"] = 1
            df.loc[cond_long, "entry_signal"] = True
            # SL is 1 ATR below entry
            df.loc[cond_long, "initial_sl"] = df.loc[cond_long, "close"] - df.loc[cond_long, "atr"]

        return df
