"""Reproducible backtest for the RVWAP Rebound strategy.

Run from repo root:
    python breadth_bot/rvwap_rebound/backtest.py

Outputs per threshold/hold combination: OOS/IS totals, mean, median, winrate,
alpha vs BTC on the same days. Fees: taker 0.05% + slippage 0.03% per side.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
os.chdir(_root)

from pandas import Timestamp
import pandas as pd

from shared.config_loader import set_active_bot
set_active_bot("breadth_bot")

from shared.data.storage import DataStorage

TF = "1d"
PERIOD = 7          # 7 bars = 7 days on 1d TF
IS_START = Timestamp("2025-08-03", tz="UTC")
TAKER_FEE = 0.0005
SLIPPAGE = 0.0003
THRESHOLDS = (10.0, 15.0, 20.0, 25.0, 30.0)
HOLDS = {"1d": 1, "2d": 2, "5d": 5}


def load_tf(storage: DataStorage, sym: str):
    df = storage.load_ohlcv(sym, TF)
    if df is None:
        return None
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp").sort_index().dropna(
        subset=["close", "high", "low", "open", "volume"])


def rvwap_dev(df: pd.DataFrame, period: int) -> pd.Series:
    src = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"]
    pv = (src * vol).rolling(period, min_periods=period).sum().shift(1)
    vs = vol.rolling(period, min_periods=period).sum().shift(1)
    vwap = pv / vs
    return (df["close"] / vwap - 1.0) * 100.0


def run_threshold(frames, thr: float, hold: int, btc_ret: pd.Series):
    events = []
    for sym, (df, d) in frames.items():
        idx = df.index
        active = False
        for i in range(len(idx)):
            dv = d.iloc[i]
            is_below = pd.notna(dv) and dv <= -thr
            if is_below and not active:
                ei = i + 1
                if ei + hold >= len(idx):
                    break
                entry = float(df.iloc[ei]["open"]) * (1 + SLIPPAGE)
                exit_px = float(df.iloc[ei + hold]["close"]) * (1 - SLIPPAGE)
                pnl = (exit_px / entry - 1) * 100 - 2 * TAKER_FEE * 100
                events.append((idx[ei], sym, pnl))
                active = True
            elif not is_below and active and pd.notna(dv) and dv > 0:
                active = False
    if not events:
        return None
    ev = pd.DataFrame(events, columns=["t", "sym", "pnl"])
    oos = ev[ev["t"] < IS_START]
    isd = ev[ev["t"] >= IS_START]

    def fmt(g):
        if len(g) == 0:
            return "none"
        s = g["pnl"]
        days = sorted(g["t"].unique())
        bm = [float(btc_ret.at[t]) for t in days if t in btc_ret.index]
        bm_m = pd.Series(bm).mean() if bm else float("nan")
        return (f"n={len(g):>4} sum={s.sum():+8.1f}% mean={s.mean():+5.2f}% "
                f"med={s.median():+5.2f}% win={int((s > 0).sum()):>3}/{len(g)} "
                f"alpha={s.mean() - bm_m:+5.2f}%")

    print(f"  thr={thr:>2.0f}% hold={hold}d: OOS {fmt(oos)} | IS {fmt(isd)}", flush=True)


def main() -> None:
    storage = DataStorage()
    symbols = storage.list_symbols(TF)
    frames = {}
    for sym in symbols:
        df = load_tf(storage, sym)
        if df is None:
            continue
        frames[sym] = (df, rvwap_dev(df, PERIOD))
    btc_df = frames["BTC/USDT:USDT"][0]
    print(f"TF={TF}, period={PERIOD} bars (7d), fees={TAKER_FEE*100:.2f}% taker + {SLIPPAGE*100:.2f}% slippage/side",
          flush=True)
    for hold in sorted(set(HOLDS.values())):
        btc_ret = btc_df["close"].pct_change(hold) * 100
        print(f"hold={hold}d", flush=True)
        for thr in THRESHOLDS:
            run_threshold(frames, thr, hold, btc_ret)


if __name__ == "__main__":
    main()
