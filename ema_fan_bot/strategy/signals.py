"""EMA Fan Strategy — signal generation module.

Logic:
  - Bullish Fan:  EMA50 > EMA100 > EMA150
  - Bearish Fan:  EMA50 < EMA100 < EMA150

Entry signals:
  - LONG:  Bullish Fan + candle closes BELOW EMA150 (first close below, prev was above or equal)
  - SHORT: Bearish Fan + candle closes ABOVE EMA150 (first close above, prev was below or equal)

Stop-loss: set at entry ± stop_loss_atr_mult × ATR14
Trailing exit: Chandelier Exit (highest_high / lowest_low ± trailing_atr_mult × ATR14)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from shared.config_loader import get_config


class EmaFanSignalGenerator:
    """Computes EMA fan indicators and generates entry signals."""

    def __init__(self):
        cfg = get_config()
        strat = cfg.get("strategy", {})
        self.ema_fast = strat.get("ema_fast", 50)
        self.ema_mid = strat.get("ema_mid", 100)
        self.ema_slow = strat.get("ema_slow", 150)
        self.atr_period = strat.get("atr_period", 14)
        self.fan_gap_min_pct = strat.get("fan_gap_min_pct", 0.0)

    # ─────────────────────────────────────────────────────────────────
    # Indicator calculation
    # ─────────────────────────────────────────────────────────────────

    def _compute_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA columns to DataFrame."""
        df = df.copy()
        df[f"ema_{self.ema_fast}"] = df["close"].ewm(span=self.ema_fast, adjust=False).mean()
        df[f"ema_{self.ema_mid}"] = df["close"].ewm(span=self.ema_mid, adjust=False).mean()
        df[f"ema_{self.ema_slow}"] = df["close"].ewm(span=self.ema_slow, adjust=False).mean()
        return df

    def _compute_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ATR column using Wilder's smoothing."""
        df = df.copy()
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1.0 / self.atr_period, adjust=False).mean()
        return df

    # ─────────────────────────────────────────────────────────────────
    # Fan detection
    # ─────────────────────────────────────────────────────────────────

    def _detect_fan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify each bar as bull_fan, bear_fan, or no_fan."""
        ema_f = df[f"ema_{self.ema_fast}"]
        ema_m = df[f"ema_{self.ema_mid}"]
        ema_s = df[f"ema_{self.ema_slow}"]
        price = df["close"]

        if self.fan_gap_min_pct > 0:
            # Требуем минимального зазора между EMA (в % от цены)
            gap = self.fan_gap_min_pct / 100.0
            bull = (ema_f > ema_m * (1 + gap)) & (ema_m > ema_s * (1 + gap))
            bear = (ema_f < ema_m * (1 - gap)) & (ema_m < ema_s * (1 - gap))
        else:
            bull = (ema_f > ema_m) & (ema_m > ema_s)
            bear = (ema_f < ema_m) & (ema_m < ema_s)

        df = df.copy()
        df["fan_bull"] = bull
        df["fan_bear"] = bear
        return df

    # ─────────────────────────────────────────────────────────────────
    # Signal generation
    # ─────────────────────────────────────────────────────────────────

    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full pipeline: add EMA, ATR, fan classification, and entry signals.

        Returns DataFrame with added columns:
          ema_50, ema_100, ema_150, atr,
          fan_bull, fan_bear,
          entry_long, entry_short
        """
        if len(df) < self.ema_slow + 10:
            logger.debug(f"Insufficient bars ({len(df)}) for EMA{self.ema_slow} calculation.")
            df["entry_long"] = False
            df["entry_short"] = False
            return df

        df = self._compute_emas(df)
        df = self._compute_atr(df)
        df = self._detect_fan(df)

        ema_f = df[f"ema_{self.ema_fast}"]
        ema_m = df[f"ema_{self.ema_mid}"]
        ema_s = df[f"ema_{self.ema_slow}"]
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        # ── Fan width filter ──────────────────────────────────────────
        # Only trade when EMA fan is sufficiently wide (strong trend).
        # Gaps measured as % of price to be scale-independent.
        if self.fan_gap_min_pct > 0:
            gap = self.fan_gap_min_pct / 100.0
            fan_wide = (
                ((ema_m - ema_f).abs() / ema_m > gap) &
                ((ema_s - ema_m).abs() / ema_s > gap)
            )
        else:
            fan_wide = pd.Series(True, index=df.index)

        # ── Bull / Bear fan ───────────────────────────────────────────
        bull = (ema_f > ema_m) & (ema_m > ema_s) & fan_wide
        bear = (ema_f < ema_m) & (ema_m < ema_s) & fan_wide

        df["fan_bull"] = bull
        df["fan_bear"] = bear

        # ── Entry signals: WICK REJECTION at EMA ─────────────────────
        #
        # SHORT (bearish fan):
        #   High wicks INTO EMA150 or EMA100, close stays BELOW → rejection
        #   Price rallied to EMA resistance and got pushed back down → short
        #
        # LONG (bullish fan):
        #   Low wicks INTO EMA150 or EMA100, close stays ABOVE → rejection
        #   Price dipped to EMA support and bounced back up → long

        # SHORT: high wicks into EMA150 or EMA100, close stays BELOW → wick rejection at MA
        wick_ema150_short = (high >= ema_s) & (close < ema_s)
        wick_ema100_short = (high >= ema_m) & (close < ema_m)

        # LONG: low wicks into EMA150 or EMA100, close stays ABOVE → wick rejection at MA
        wick_ema150_long = (low <= ema_s) & (close > ema_s)
        wick_ema100_long = (low <= ema_m) & (close > ema_m)

        df["entry_short"] = df["fan_bear"] & (wick_ema150_short | wick_ema100_short)
        df["entry_long"]  = df["fan_bull"] & (wick_ema150_long  | wick_ema100_long)

        # Drop warmup rows (EMA not yet stable)
        warmup = self.ema_slow + self.atr_period
        df.iloc[:warmup, df.columns.get_loc("entry_long")]  = False
        df.iloc[:warmup, df.columns.get_loc("entry_short")] = False

        # Apply direction filter from config
        cfg = get_config()
        direction = cfg.get("strategy", {}).get("direction", "both")
        if direction == "short_only":
            df["entry_long"] = False
        elif direction == "long_only":
            df["entry_short"] = False

        n_long = df["entry_long"].sum()
        n_short = df["entry_short"].sum()
        logger.debug(f"Signals computed: {n_long} LONG, {n_short} SHORT entries")

        return df
