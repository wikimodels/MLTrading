"""Stage 2: backtest each breadth event from breadth_events.json into individual trades.

Rules (transparent, no hidden logic):
- Entry: for every coin listed in the event's short/long participant list, enter on the
  open of the next 1h bar AFTER the event hour (t+1). Short coins -> SHORT, long -> LONG.
- Exit: reverse Chandelier break. For a SHORT, exit when ce_dir flips back to +1 (the
  ratcheted Chandelier stop was violated upward); for a LONG, exit when ce_dir flips to -1.
  Execution at the open of the first bar where the flip is observed.
- Costs: taker_fee 0.05% each side + slippage 0.03% each side (same as settings).
- Per trade we record MAE / MFE (max adverse / favorable excursion vs entry) so quality
  is visible even when the trade ends green/red.

Output: breadth_bot/backtest/trades_from_events.json
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
os.chdir(_root)

from shared.config_loader import set_active_bot
set_active_bot("breadth_bot")

from shared.data.storage import DataStorage
from breadth_bot.screener.screener import load_hourly_ohlcv
from breadth_bot.strategy.signals import BreadthSignalGenerator
from pandas import Timestamp

HOURS_1Y = 24 * 365
EVENTS = Path(r"D:\GitHub\MLTrading\breadth_bot\backtest\breadth_events.json")
OUT = Path(r"D:\GitHub\MLTrading\breadth_bot\backtest\trades_from_events.json")
TAKER_FEE = 0.0005
SLIPPAGE = 0.0003
BREADTH_MIN = 0.50
SIDES = ("short",)
INVERSED = False  # test: where the algorithm says SHORT, go LONG, and vice versa
SL_GRID = [0.015, 0.02, 0.03, 0.04]
TP_GRID = [0.03, 0.05, 0.08, 0.12]


def main() -> None:
    doc = json.loads(EVENTS.read_text(encoding="utf-8"))
    events = doc["events"]

    storage = DataStorage()
    gen = BreadthSignalGenerator()
    symbols = storage.list_symbols("1h")

    # Precompute signals once per symbol, keep full history so exits can extend beyond 1y window.
    sig = {}
    for sym in symbols:
        df = load_hourly_ohlcv(storage, sym)
        if df is None:
            continue
        sig[sym] = gen.compute_signals(df)

    trades = []
    for e in events:
        event_time = e["time"]
        for side_key in SIDES:
            part = e[side_key]
            if part["count"] == 0:
                continue
            if part["breadth"] < BREADTH_MIN:
                continue
            if side_key == "short":
                direction = "long" if INVERSED else "short"
            else:
                direction = "short" if INVERSED else "long"
            for sym in part["coins"]:
                df = sig.get(sym)
                if df is None:
                    continue
                # find the event hour exactly
                try:
                    i = df.index.get_loc(Timestamp(event_time, tz="UTC"))
                except KeyError:
                    # event time may be missing for this symbol (delisted gap); skip coin
                    continue
                if i + 1 >= len(df.index):
                    continue
                t_entry = df.index[i + 1]
                entry_px = float(df.at[t_entry, "open"])
                entry_px_cost = entry_px * (1 + SLIPPAGE) if direction == "long" else entry_px * (1 - SLIPPAGE)

                # walk forward, track MAE/MFE, exit on reverse flip
                exit_px = None
                exit_time = None
                reason = "end_of_data"
                bars = 0
                mae = 0.0  # max adverse excursion (in return pct vs entry)
                mfe = 0.0
                for j in range(i + 1, len(df.index)):
                    bars += 1
                    row = df.iloc[j]
                    high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
                    if direction == "short":
                        adverse = (high - entry_px) / entry_px
                        favorable = (entry_px - low) / entry_px
                    else:
                        adverse = (entry_px - low) / entry_px
                        favorable = (high - entry_px) / entry_px
                    mae = max(mae, adverse)
                    mfe = max(mfe, favorable)
                    if bool(row["ce_dir"] == 1) and direction == "short":
                        exit_px = float(df.iloc[j + 1]["open"]) if j + 1 < len(df.index) else close
                        exit_time = df.index[j + 1] if j + 1 < len(df.index) else df.index[j]
                        reason = "chand_flip"
                        break
                    if bool(row["ce_dir"] == -1) and direction == "long":
                        exit_px = float(df.iloc[j + 1]["open"]) if j + 1 < len(df.index) else close
                        exit_time = df.index[j + 1] if j + 1 < len(df.index) else df.index[j]
                        reason = "chand_flip"
                        break
                if exit_px is None:
                    last = df.iloc[-1]
                    exit_px = float(last["close"])
                    exit_time = last.name
                    reason = "end_of_data"

                gross = (exit_px - entry_px) / entry_px if direction == "long" else (entry_px - exit_px) / entry_px
                costs = 2 * TAKER_FEE + 2 * SLIPPAGE
                ret = gross - costs
                trades.append({
                    "symbol": sym,
                    "direction": direction,
                    "event_time": event_time,
                    "entry_time": str(t_entry),
                    "entry_price": round(entry_px, 6),
                    "exit_time": str(exit_time),
                    "exit_price": round(exit_px, 6),
                    "bars_held": bars,
                    "gross_return_pct": round(gross * 100, 4),
                    "net_return_pct": round(ret * 100, 4),
                    "mae_pct": round(mae * 100, 4),
                    "mfe_pct": round(mfe * 100, 4),
                    "exit_reason": reason,
                })

    OUT.write_text(json.dumps(trades, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {OUT} ({OUT.stat().st_size/1024:.0f} KB, {len(trades)} trades)")

    # transparent summary
    for side in ("short", "long"):
        tside = [t for t in trades if t["direction"] == side]
        if not tside:
            print(f"{side.upper()}: no trades")
            continue
        wins = [t for t in tside if t["net_return_pct"] > 0]
        loss = [t for t in tside if t["net_return_pct"] <= 0]
        gross_w = sum(t["net_return_pct"] for t in wins)
        gross_l = abs(sum(t["net_return_pct"] for t in loss))
        pf = gross_w / gross_l if gross_l else float("inf")
        avg_hold = sum(t["bars_held"] for t in tside) / len(tside)
        avg_mae = sum(t["mae_pct"] for t in tside) / len(tside)
        avg_mfe = sum(t["mfe_pct"] for t in tside) / len(tside)
        print(f"\n=== {side.upper()} ({len(tside)} trades, {len(wins)} wins, {len(loss)} losses) ===")
        print(f"win rate: {len(wins)/len(tside)*100:.1f}%")
        print(f"avg net return: {sum(t['net_return_pct'] for t in tside)/len(tside):.3f}%")
        print(f"avg gross win / avg gross loss: {sum(t['net_return_pct'] for t in wins)/max(len(wins),1):.3f}% / {sum(t['net_return_pct'] for t in loss)/max(len(loss),1):.3f}%")
        print(f"profit factor: {pf:.3f}")
        print(f"avg hold: {avg_hold:.1f} bars, avg MAE: {avg_mae:.3f}%, avg MFE: {avg_mfe:.3f}%")
        print(f"total net: {sum(t['net_return_pct'] for t in tside):.2f}%")


if __name__ == "__main__":
    main()
