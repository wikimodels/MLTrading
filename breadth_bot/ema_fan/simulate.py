"""Portfolio simulator for EMA fan strategy (and any event list).

Rules:
  - starting capital $1000 (or CLI arg)
  - no position limits: every fresh signal opens a position
  - position margin = pos_pct of current equity at entry (default 10%)
  - equity updated at exit: pnl = margin * pnl_pct
  - equity curve per day, max drawdown, $ pnl, WR
Usage:
    python breadth_bot/ema_fan/simulate.py 1d [1000]
    python breadth_bot/ema_fan/simulate.py 4h [1000]
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
HOLD_BARS = 10


def load_tf(storage: DataStorage, sym: str, tf: str):
    df = storage.load_ohlcv(sym, tf)
    if df is None:
        return None
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp").sort_index().dropna(
        subset=["close", "high", "low", "open", "volume"])


def collect_events(tf: str, hold_bars: int):
    storage = DataStorage()
    symbols = storage.list_symbols(tf)
    events = []
    for sym in symbols:
        df = load_tf(storage, sym, tf)
        if df is None:
            continue
        c = df["close"]
        f = c.ewm(span=21, adjust=False).mean()
        m = c.ewm(span=50, adjust=False).mean()
        s = c.ewm(span=100, adjust=False).mean()
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
                if ei + hold_bars >= len(idx):
                    continue
                entry_t = idx[ei]
                entry = float(df.iloc[ei]["open"]) * (1 - SLIPPAGE)
                exit_t = idx[ei + hold_bars]
                exit_px = float(df.iloc[ei + hold_bars]["close"]) * (1 + SLIPPAGE)
                pnl = (entry / exit_px - 1) * 100 - 2 * TAKER_FEE * 100
                events.append({"sym": sym, "entry_t": entry_t, "exit_t": exit_t,
                               "entry": entry, "pnl": pnl})
                was_bull = False
    return events


def simulate(tf: str, hold_bars: int, init: float, pos_pct: float):
    events = collect_events(tf, hold_bars)
    events.sort(key=lambda e: e["entry_t"])

    equity = init
    curve = []          # (date, equity, open_positions)
    open_pos = {}       # sym -> {margin}
    pnls = []

    timeline = sorted({e["entry_t"] for e in events} | {e["exit_t"] for e in events})
    by_ts = {}
    for e in events:
        by_ts.setdefault(e["entry_t"], []).append(("open", e))
        by_ts.setdefault(e["exit_t"], []).append(("close", e))

    for t in timeline:
        for kind, e in by_ts.get(t, []):
            if kind == "open":
                margin = equity * pos_pct
                open_pos[e["sym"]] = margin
            else:
                margin = open_pos.pop(e["sym"], equity * pos_pct)
                pnl_d = margin * e["pnl"] / 100
                equity += pnl_d
                pnls.append(e["pnl"])
        curve.append((t, equity, len(open_pos)))

    eq = pd.Series({t: v for t, v, _ in curve}).sort_index()
    peak = eq.cummax()
    dd = (eq / peak - 1) * 100
    max_dd = dd.min()

    p = pd.Series(pnls)
    oos_p = [e["pnl"] for e in events if e["entry_t"] < IS_START]
    is_p = [e["pnl"] for e in events if e["entry_t"] >= IS_START]

    print(f"== {tf} hold={hold_bars}d pos_pct={pos_pct*100:.0f}% init=${init:,.0f} ==")
    print(f"trades: {len(p)}  WR: {int((p > 0).sum())}/{len(p)} ({int((p > 0).mean() * 100)}%)")
    print(f"final equity: ${equity:,.2f}  pnl: ${equity - init:+,.2f}  ({((equity/init)-1)*100:+,.1f}%)")
    print(f"max drawdown: {max_dd:.1f}%")
    print(f"OOS trades: {len(oos_p)} WR {int(sum(1 for x in oos_p if x > 0))}/{len(oos_p)} | "
          f"IS trades: {len(is_p)} WR {int(sum(1 for x in is_p if x > 0))}/{len(is_p)}")
    eq.to_csv(Path(__file__).parent / f"equity_{tf}.csv")
    print(f"equity curve saved: equity_{tf}.csv ({len(eq)} days)")


if __name__ == "__main__":
    tf = sys.argv[1] if len(sys.argv) > 1 else "1d"
    init = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0
    simulate(tf, HOLD_BARS, init, 0.10)
