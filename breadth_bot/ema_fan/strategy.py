"""EMA fan breakdown strategy: short coins whose bullish EMA21/50/100 fan
fully breaks (EMA21 < EMA50 < EMA100 for the first time after bullish order).

Bull fan anatomy:
  S1: EMA21 < EMA50            first crack (noisy, weak)
  S2: EMA21 < EMA100           short under long (medium)
  S3: EMA21 < EMA50 < EMA100   full breakdown -> flush follows (strong)
Entry on S3 only. Exit: close after N bars (10d optimal on 1d TF).
"""

from __future__ import annotations

import pandas as pd


class EmaFanBreakdownSignal:
    """Detects the first full fan breakdown (21<50<100) after bullish order.

    Params:
        fast/mid/slow  EMA spans (default 21/50/100)
        hold_bars      exit bars after entry (10d = 10 bars on 1d, 60 on 4h)
    """

    def __init__(
        self,
        fast: int = 21,
        mid: int = 50,
        slow: int = 100,
        hold_bars: int = 10,
    ) -> None:
        self.fast = fast
        self.mid = mid
        self.slow = slow
        self.hold_bars = hold_bars

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds 'ema_fast','ema_mid','ema_slow','bull_fan' (bool), 'entry' (bool)."""
        df = df.copy()
        c = df["close"]
        df["ema_fast"] = c.ewm(span=self.fast, adjust=False).mean()
        df["ema_mid"] = c.ewm(span=self.mid, adjust=False).mean()
        df["ema_slow"] = c.ewm(span=self.slow, adjust=False).mean()

        f, m, s = df["ema_fast"], df["ema_mid"], df["ema_slow"]
        df["bull_fan"] = (f > m) & (m > s)
        bear_full = (f < m) & (m < s)

        # Fresh S3: first bar with full bear order after fan was bullish
        entry = pd.Series(False, index=df.index)
        was_bull = False
        for i in range(len(df)):
            if pd.isna(f.iloc[i]) or pd.isna(m.iloc[i]) or pd.isna(s.iloc[i]):
                continue
            if df["bull_fan"].iloc[i]:
                was_bull = True
                continue
            if was_bull and bool(bear_full.iloc[i]):
                entry.iloc[i] = True
                was_bull = False  # one entry per breakdown episode
        df["entry"] = entry
        return df
