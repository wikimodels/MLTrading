"""RVWAP Rebound strategy: long coins that dropped X% below their 7d Rolling VWAP.

Indicator: dev = close / rvwap - 1, where rvwap is the volume-weighted average
execution price of the last 7 days (excl. current bar). Entry on the FIRST day
dev <= -threshold (fresh event, no re-entry until dev recovers above 0).
Exit: close after N bars (1-2 days optimal).
"""

from __future__ import annotations

import pandas as pd


class RvwapReboundSignal:
    """Generates RVWAP deviation series and rebound entry events for a symbol.

    Parameters match the winning backtest config:
        period        7 bars (7 days on 1d TF, 42 bars on 4h TF)
        threshold     20.0  (entry when dev <= -20%)
        hold_bars     2     (exit 2 bars after entry)
    """

    def __init__(
        self,
        period: int = 7,
        threshold: float = 20.0,
        hold_bars: int = 2,
        shift: int = 1,
    ) -> None:
        self.period = period
        self.threshold = threshold
        self.hold_bars = hold_bars
        self.shift = shift

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds 'rvwap_dev' (percent), 'entry' (bool, first bar below -thr),
        'exit' (bar index relative to entry, pre-computed for backtest).
        """
        df = df.copy()
        src = (df["high"] + df["low"] + df["close"]) / 3.0
        vol = df["volume"]

        pv = (src * vol).rolling(self.period, min_periods=self.period).sum().shift(self.shift)
        vs = vol.rolling(self.period, min_periods=self.period).sum().shift(self.shift)
        vwap = pv / vs
        df["rvwap"] = vwap
        df["rvwap_dev"] = (df["close"] / vwap - 1.0) * 100.0

        dev = df["rvwap_dev"]
        below = dev <= -self.threshold

        # Fresh event: first bar below threshold after being above zero
        entry = pd.Series(False, index=df.index)
        active = False
        for i in range(len(df)):
            dv = dev.iloc[i]
            if pd.isna(dv):
                continue
            if below.iloc[i] and not active:
                entry.iloc[i] = True
                active = True
            elif not below.iloc[i] and active and dv > 0:
                active = False
        df["entry"] = entry
        return df
