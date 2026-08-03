"""Signal generation based on Volatility Phase Transitions and Close Location Value."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from shared.config_loader import get_config
from shared.indicators import (
    compute_clv, 
    compute_true_range, 
    compute_tr_zscore,
    compute_excess_kurtosis,
    get_hurst_exponent
)


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
        
        self.kurtosis_min = strat_cfg.get("kurtosis_min", 5.0)
        self.hurst_min = strat_cfg.get("hurst_min", 0.25)
        self.turnover_min = strat_cfg.get("turnover_min", 3_000_000)
        
        # Look-ahead bias prevention
        self.screen_lag = 1
        self.min_history_days = max(365, strat_cfg.get("min_history_days", 365))
        self.kurt_window = 365
        self.hurst_window = 180

    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Appends technical indicators, ATR, signals, and dynamic stop loss bounds to DataFrame."""
        if df is None or len(df) < self.min_history_days:
            if df is not None:
                df = df.copy()
                df["entry_signal"] = False
                df["signal_short"] = 0
            return df if df is not None else pd.DataFrame()

        df = df.copy()

        # 1. Rolling Screen (Kurtosis & Hurst)
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['kurtosis'] = compute_excess_kurtosis(df['log_ret'], window=self.kurt_window)
        df['hurst'] = df['close'].rolling(window=self.hurst_window).apply(
            lambda x: get_hurst_exponent(x), raw=True
        )
        
        df['passes_filter_raw'] = (df['kurtosis'] >= self.kurtosis_min) & (df['hurst'] >= self.hurst_min)
        
        # Anti-look-ahead: shift the filter result
        df['passes_filter'] = df['passes_filter_raw'].shift(self.screen_lag).fillna(False)

        # 2. Compute Core Indicators for Trigger
        tr = compute_true_range(df)
        df["tr"] = tr
        df["atr_14"] = tr.rolling(14, min_periods=7).mean().replace(0, 1e-8)
        df["tr_zscore"] = compute_tr_zscore(df, window=30)
        df["clv"] = compute_clv(df["high"], df["low"], df["close"])
        df["q01"] = df["close"].rolling(window=90, min_periods=30).apply(lambda x: np.quantile(x, 0.01), raw=True)
        df["ema5"] = df["close"].ewm(span=5, adjust=False).mean()

        # Initialize signal columns
        df["signal_short"] = 0
        df["signal_long"] = 0
        df["initial_sl"] = np.nan
        df["entry_signal"] = False
        df["exit_signal"] = False

        # 3. Triggers
        c1 = df['passes_filter']
        c2 = df['tr_zscore'] >= self.tr_zscore_min
        c3 = df['clv'] <= self.clv_short_max
        c4 = df['close'] < df['q01']
        
        if self.turnover_min is not None and 'volume' in df.columns:
            # Turnover = base volume * price
            c5 = (df['volume'] * df['close']) >= self.turnover_min
        else:
            c5 = True

        if self.allow_short:
            cond_short = c1 & c2 & c3 & c4 & c5
            df.loc[cond_short, "signal_short"] = 1
            df.loc[cond_short, "entry_signal"] = True
            df.loc[cond_short, "initial_sl"] = df.loc[cond_short, "close"] + self.sl_mult * df.loc[cond_short, "atr_14"]

        # Long Trigger (optional reversal - logic not heavily defined in theory, left for future if allow_long is True)
        if self.allow_long:
            c3_long = df['clv'] >= self.clv_long_min
            c4_long = df['close'] >= df['close'].rolling(window=90, min_periods=30).apply(lambda x: np.quantile(x, 0.99), raw=True)
            cond_long = c1 & c2 & c3_long & c4_long & c5
            df.loc[cond_long, "signal_long"] = 1
            df.loc[cond_long, "entry_signal"] = True
            df.loc[cond_long, "initial_sl"] = df.loc[cond_long, "close"] - self.sl_mult * df.loc[cond_long, "atr_14"]

        # Exit signal
        df['exit_signal'] = (df['close'] > df['ema5']) | (df['clv'] >= self.clv_long_min)

        return df
