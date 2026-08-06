"""Stage 2b: grid search over SL/TP on direct (non-inverted) short trades from breadth events.

Entry: every coin listed in the event's short participant list, on the open of the next
1h bar AFTER the event hour (t+1), as a SHORT.
Exit: first of (a) reverse Chandelier flip, (b) fixed stop-loss, (c) fixed take-profit.
Bar-level honesty: if both SL and TP are hit inside the same bar, the SL (worst case) wins.
Costs: taker_fee 0.05% each side + slippage 0.03% each side.

Output: breadth_bot/backtest/trades_sl_tp_grid.json (all trades per combination)
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
OUT = Path(r"D:\GitHub\MLTrading\breadth_bot\backtest\trades_sl_tp_grid.json")
TAKER_FEE = 0.0005
SLIPPAGE = 0.0003
BREADTH_MIN = 0.50
SL_GRID = [0.0, 0.015, 0.02, 0.03, 0.04]   # 0.0 = no stop (pure chandelier flip exit)
TP_GRID = [0.0, 0.03, 0.05, 0.08, 0.12]    # 0.0 = no take (pure chandelier flip exit)


def simulate(sl: float, tp: float, sig: dict, events: list) -> list[dict]:
    """Run one SL/TP combination over all short breadth events."""
    trades = []
    for e in events:
        part = e["short"]
        if part["count"] == 0 or part["breadth"] < BREADTH_MIN:
            continue
        for sym in part["coins"]:
            df = sig.get(sym)
            if df is None:
                continue
            try:
                i = df.index.get_loc(Timestamp(e["time"], tz="UTC"))
            except KeyError:
                continue
            if i + 1 >= len(df.index):
                continue
            t_entry = df.index[i + 1]
            entry_px = float(df.at[t_entry, "open"])
            entry_cost = entry_px * (1 - SLIPPAGE)  # short entry is a sell

            exit_px = None
            exit_time = None
            reason = "end_of_data"
            bars = 0
            mae = 0.0
            mfe = 0.0
            for j in range(i + 1, len(df.index)):
                bars += 1
                row = df.iloc[j]
                high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
                adverse = (high - entry_px) / entry_px          # for short
                favorable = (entry_px - low) / entry_px
                mae = max(mae, adverse)
                mfe = max(mfe, favorable)

                hit_sl = sl > 0 and adverse >= sl
                hit_tp = tp > 0 and favorable >= tp

                if hit_sl and hit_tp:
                    exit_px = entry_px * (1 + sl) * (1 - SLIPPAGE)
                    exit_time = df.index[j]
                    reason = "sl_tp_same_bar_sl_wins"
                    break
                if hit_sl:
                    exit_px = entry_px * (1 + sl) * (1 - SLIPPAGE)
                    exit_time = df.index[j]
                    reason = "stop_loss"
                    break
                if hit_tp:
                    exit_px = entry_px * (1 - tp) * (1 - SLIPPAGE)
                    exit_time = df.index[j]
                    reason = "take_profit"
                    break
                if bool(row["ce_dir"] == 1):  # reverse flip -> exit short
                    exit_px = float(df.iloc[j + 1]["open"]) * (1 - SLIPPAGE) if j + 1 < len(df.index) else close * (1 - SLIPPAGE)
                    exit_time = df.index[j + 1] if j + 1 < len(df.index) else df.index[j]
                    reason = "chand_flip"
                    break
            if exit_px is None:
                last = df.iloc[-1]
                exit_px = float(last["close"]) * (1 - SLIPPAGE)
                exit_time = last.name
                reason = "end_of_data"

            gross = (entry_cost - exit_px) / entry_cost  # short: sell high, buy back low
            ret = gross - 2 * TAKER_FEE
            trades.append({
                "symbol": sym,
                "direction": "short",
                "event_time": e["time"],
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
    return trades


def summarize(ts: list[dict]) -> dict:
    if not ts:
        return {"trades": 0}
    wins = [t for t in ts if t["net_return_pct"] > 0]
    loss = [t for t in ts if t["net_return_pct"] <= 0]
    gw = sum(t["net_return_pct"] for t in wins)
    gl = abs(sum(t["net_return_pct"] for t in loss))
    rets = [t["net_return_pct"] for t in ts]
    return {
        "trades": len(ts),
        "win_rate": len(wins) / len(ts),
        "avg_ret": sum(rets) / len(rets),
        "pf": gw / gl if gl else float("inf"),
        "total_net": sum(rets),
        "avg_hold_bars": sum(t["bars_held"] for t in ts) / len(ts),
        "avg_mae": sum(t["mae_pct"] for t in ts) / len(ts),
        "avg_mfe": sum(t["mfe_pct"] for t in ts) / len(ts),
    }


def main() -> None:
    doc = json.loads(EVENTS.read_text(encoding="utf-8"))
    events = doc["events"]

    storage = DataStorage()
    gen = BreadthSignalGenerator()
    symbols = storage.list_symbols("1h")
    sig = {}
    for sym in symbols:
        df = load_hourly_ohlcv(storage, sym)
        if df is None:
            continue
        sig[sym] = gen.compute_signals(df)

    results = []
    for sl in SL_GRID:
        for tp in TP_GRID:
            trades = simulate(sl, tp, sig, events)
            s = summarize(trades)
            results.append({"sl": sl, "tp": tp, **s})
            print(f"SL {sl:.3f} TP {tp:.3f}: trades={s['trades']:4d}  WR={s['win_rate']*100:5.1f}%  "
                  f"avg={s['avg_ret']:+.4f}%  PF={s['pf']:5.2f}  total={s['total_net']:+8.2f}%  "
                  f"hold={s['avg_hold_bars']:5.1f}b  MAE={s['avg_mae']:.2f}%  MFE={s['avg_mfe']:.2f}%")

    OUT.write_text(json.dumps({"grid": results, "notes": [
        "SL/TP=0 means no stop/take; exit = reverse Chandelier flip only",
        "same-bar SL+TP -> counted as SL (worst case)",
    ]}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
