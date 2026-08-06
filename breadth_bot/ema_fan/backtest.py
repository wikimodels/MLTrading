"""Reproducible backtest for the EMA fan breakdown strategy (short).

Usage:
    python breadth_bot/ema_fan/backtest.py [1d|4h]

Outputs per hold: OOS/IS totals, mean, median, winrate, alpha vs BTC.
Fees: taker 0.05% + slippage 0.03% per side. Entry: next bar open.
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
import os
os.chdir(_root)

from pandas import Timestamp
import pandas as pd

from shared.config_loader import set_active_bot
set_active_bot("breadth_bot")

from shared.data.storage import DataStorage

IS_START = Timestamp("2025-08-03", tz="UTC")
TAKER_FEE = 0.0005
SLIPPAGE = 0.0003
HOLDS_1D = {"1d": 1, "2d": 2, "5d": 5, "10d": 10}
HOLDS_4H = {"6h": 1, "1d": 6, "2d": 12, "5d": 30, "10d": 60}


def load_tf(storage: DataStorage, sym: str, tf: str):
    df = storage.load_ohlcv(sym, tf)
    if df is None:
        return None
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp").sort_index().dropna(
        subset=["close", "high", "low", "open", "volume"])


def main() -> None:
    tf = sys.argv[1] if len(sys.argv) > 1 else "1d"
    holds = HOLDS_1D if tf == "1d" else HOLDS_4H
    storage = DataStorage()
    symbols = storage.list_symbols(tf)
    frames = {}
    for sym in symbols:
        df = load_tf(storage, sym, tf)
        if df is None:
            continue
        c = df["close"]
        f = c.ewm(span=21, adjust=False).mean()
        m = c.ewm(span=50, adjust=False).mean()
        s = c.ewm(span=100, adjust=False).mean()
        frames[sym] = (df, f, m, s)
    btc_df = frames["BTC/USDT:USDT"][0]
    print(f"TF={tf}, EMA 21/50/100, S3 full breakdown, fees {TAKER_FEE*100:.2f}+{SLIPPAGE*100:.2f}%/side", flush=True)

    for hname, hb in holds.items():
        events = []
        for sym, (df, f, m, s) in frames.items():
            idx = df.index
            was_bull = False
            for i in range(len(idx)):
                if pd.isna(f.iloc[i]) or pd.isna(m.iloc[i]) or pd.isna(s.iloc[i]):
                    continue
                if f.iloc[i] > m.iloc[i] > s.iloc[i]:
                    was_bull = True
                    continue
                if was_bull and f.iloc[i] < m.iloc[i] < s.iloc[i]:
                    ei = i + 1
                    if ei + hb >= len(idx):
                        continue
                    entry = float(df.iloc[ei]["open"]) * (1 - SLIPPAGE)
                    exit_px = float(df.iloc[ei + hb]["close"]) * (1 + SLIPPAGE)
                    pnl = (entry / exit_px - 1) * 100 - 2 * TAKER_FEE * 100
                    events.append((idx[ei], sym, pnl))
                    was_bull = False
        if not events:
            print(f"  hold={hname:>3}: no events", flush=True)
            continue
        ev = pd.DataFrame(events, columns=["t", "sym", "pnl"])
        oos = ev[ev["t"] < IS_START]
        isd = ev[ev["t"] >= IS_START]
        btc_ret = btc_df["close"].pct_change(hb) * 100

        def fmt(g):
            if len(g) == 0:
                return "none"
            p = g["pnl"]
            days = sorted(g["t"].unique())
            bm = [float(btc_ret.at[t]) for t in days if t in btc_ret.index]
            bm_m = pd.Series(bm).mean() if bm else float("nan")
            return (f"n={len(g):>3} sum={p.sum():+8.1f}% mean={p.mean():+5.2f}% "
                    f"med={p.median():+5.2f}% win={int((p > 0).sum()):>3}/{len(g)} "
                    f"alpha={p.mean() - bm_m:+5.2f}%")

        print(f"  SHORT hold={hname:>3}: OOS {fmt(oos)} | IS {fmt(isd)}", flush=True)


if __name__ == "__main__":
    main()
