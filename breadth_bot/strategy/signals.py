"""Chandelier exit and market breadth signal generation for breadth_bot."""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared.config_loader import get_config


def compute_true_range(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr


def _rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's RMA smoothing (equivalent of ta.atr / ta.rma in Pine)."""
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


class BreadthSignalGenerator:
    """Chandelier Exit (everget/TradingView) state machine + directional triggers.

    Port of the Pine script:
        atr       = mult * ta.atr(length)
        longStop  = highest(close, length) - atr   (ratchet: max with prev if close[1] > longStop[1])
        shortStop = lowest(close, length) + atr    (ratchet: min with prev if close[1] < shortStop[1])
        dir = close > shortStop[1] ? 1 : close < longStop[1] ? -1 : dir[1]
        buySignal  = dir == 1 and dir[1] == -1
        sellSignal = dir == -1 and dir[1] == 1
    """

    def __init__(self):
        self.cfg = get_config()
        strat_cfg = self.cfg.get("strategy", {})
        self.atr_period = strat_cfg.get("atr_period", 21)
        self.chandelier_mult = strat_cfg.get("chandelier_mult", 2.5)
        self.breadth_min = strat_cfg.get("breadth_min", 0.5)
        self.allow_short = strat_cfg.get("allow_short", True)
        self.allow_long = strat_cfg.get("allow_long", False)

    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Appends ATR, Chandelier stop lines, direction state and buy/sell triggers."""
        df = df.copy()
        length = self.atr_period
        mult = self.chandelier_mult

        tr = compute_true_range(df)
        atr = mult * _rma(tr, length)
        df["atr"] = atr

        close = df["close"]
        long_stop_raw = close.rolling(length, min_periods=length).max() - atr
        short_stop_raw = close.rolling(length, min_periods=length).min() + atr

        # Ratchet the stops (only tighten), exactly like the Pine script
        n = len(df)
        long_stop = np.full(n, np.nan, dtype=float)
        short_stop = np.full(n, np.nan, dtype=float)
        for i in range(n):
            raw_l = long_stop_raw.iloc[i]
            raw_s = short_stop_raw.iloc[i]
            if np.isnan(raw_l):
                continue
            if i == 0:
                long_stop[i] = raw_l
                short_stop[i] = raw_s
                continue
            prev_l = long_stop[i - 1]
            prev_s = short_stop[i - 1]
            # close[1] vs prev stop
            if close.iloc[i - 1] > prev_l:
                long_stop[i] = max(raw_l, prev_l)
            else:
                long_stop[i] = raw_l
            if close.iloc[i - 1] < prev_s:
                short_stop[i] = min(raw_s, prev_s)
            else:
                short_stop[i] = raw_s

        df["long_stop"] = long_stop
        df["short_stop"] = short_stop

        # Direction state machine
        long_prev = pd.Series(long_stop).shift(1)
        short_prev = pd.Series(short_stop).shift(1)
        dir_arr = np.ones(n, dtype=int)
        for i in range(1, n):
            if pd.isna(short_prev.iloc[i]) or pd.isna(long_prev.iloc[i]):
                dir_arr[i] = dir_arr[i - 1]
                continue
            if close.iloc[i] > short_prev.iloc[i]:
                dir_arr[i] = 1
            elif close.iloc[i] < long_prev.iloc[i]:
                dir_arr[i] = -1
            else:
                dir_arr[i] = dir_arr[i - 1]
        df["ce_dir"] = dir_arr
        df["dir_prev"] = pd.Series(dir_arr).shift(1)

        # NB: keep index-agnostic (numpy) — assigning a RangeIndex Series to a
        # DatetimeIndex frame would align to NaN, making bool(NaN)=True everywhere.
        dir_arr_s = pd.Series(dir_arr)
        df["short_sig"] = ((dir_arr == -1) & (dir_arr_s.shift(1) == 1)).to_numpy()
        df["long_sig"] = ((dir_arr == 1) & (dir_arr_s.shift(1) == -1)).to_numpy()
        return df

    def compute_breadth(self, data: dict[str, pd.DataFrame]) -> tuple[pd.Series, pd.Series]:
        """Hourly fraction of coins in short/long state (CE direction) across the whole universe."""
        dates = sorted(set().union(*[set(df.index) for df in data.values()]))
        short_breadth = pd.Series(np.nan, index=dates, dtype=float)
        long_breadth = pd.Series(np.nan, index=dates, dtype=float)
        for t in dates:
            ss = 0; ls = 0; n = 0
            for sym, df in data.items():
                if t not in df.index:
                    continue
                n += 1
                if df.at[t, "ce_dir"] == -1: ss += 1
                if df.at[t, "ce_dir"] == 1: ls += 1
            if n:
                short_breadth[t] = ss / n
                long_breadth[t] = ls / n
        return short_breadth, long_breadth

    def breadth_triggered(self, breadth: pd.Series, direction: str) -> pd.Series:
        return breadth >= self.breadth_min
