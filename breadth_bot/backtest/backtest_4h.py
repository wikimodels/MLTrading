"""4h test: same Chandelier Exit(21, 2.5) logic, events >= BREADTH_MIN, SL/TP grid.

Mirrors the 1h pipeline: extract breadth events (short entries only), simulate trades
with SL/TP grid, dump per-combination stats. Compare with 1h results.
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
from breadth_bot.strategy.signals import BreadthSignalGenerator
from pandas import Timestamp

TIMEFRAME = "4h"
BARS_1Y = 6 * 365
BREADTH_MIN = 0.50
SL_GRID = [0.0, 0.02, 0.03, 0.04]
TP_GRID = [0.0, 0.05, 0.08, 0.12]
TAKER_FEE = 0.0005
SLIPPAGE = 0.0003
OUT_EVENTS = Path(r"D:\GitHub\MLTrading\breadth_bot\backtest\breadth_events_4h.json")
OUT_GRID = Path(r"D:\GitHub\MLTrading\breadth_bot\backtest\trades_sl_tp_grid_4h.json")


def load_tf(storage, sym):
    df = storage.load_ohlcv(sym, TIMEFRAME)
    if df is None or len(df) < 1000:
        return None
    df = df.copy()
    df["timestamp"] = pd_to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df.dropna(subset=["close", "high", "low", "open"])
    return df if len(df) >= 1000 else None


def pd_to_datetime(x):
    from pandas import to_datetime
    return to_datetime(x)


def main() -> None:
    storage = DataStorage()
    gen = BreadthSignalGenerator()
    symbols = storage.list_symbols(TIMEFRAME)

    data = {}
    for sym in symbols:
        df = load_tf(storage, sym)
        if df is None:
            continue
        df = df.tail(BARS_1Y)
        sig = gen.compute_signals(df)
        data[sym] = sig

    universe = sorted(data.keys())
    dates = sorted(set().union(*[set(df.index) for df in data.values()]))
    n = len(universe)

    events = []
    for t in dates:
        short_coins, long_coins = [], []
        for sym in universe:
            df = data[sym]
            if t not in df.index:
                continue
            if bool(df.at[t, "short_sig"]):
                short_coins.append(sym)
            if bool(df.at[t, "long_sig"]):
                long_coins.append(sym)
        sb = len(short_coins) / n
        lb = len(long_coins) / n
        if sb >= BREADTH_MIN or lb >= BREADTH_MIN:
            events.append({
                "time": t.strftime("%Y-%m-%d %H:%M"),
                "short": {"breadth": round(sb, 4), "count": len(short_coins), "coins": short_coins},
                "long": {"breadth": round(lb, 4), "count": len(long_coins), "coins": long_coins},
            })

    OUT_EVENTS.write_text(json.dumps({"meta": {
        "timeframe": TIMEFRAME, "breadth_min": BREADTH_MIN,
        "universe_size": n, "hours": len(dates),
        "range": f"{dates[0]:%Y-%m-%d} -> {dates[-1]:%Y-%m-%d}",
    }, "events": events}, indent=2, ensure_ascii=False), encoding="utf-8")
    n_s = sum(1 for e in events if e["short"]["count"] > 0)
    n_l = sum(1 for e in events if e["long"]["count"] > 0)
    print(f"[4h] events: {len(events)} (short {n_s}, long {n_l}); bars: {len(dates)}; universe: {n}")

    def simulate(sl, tp):
        trades = []
        for e in events:
            part = e["short"]
            if part["count"] == 0 or part["breadth"] < BREADTH_MIN:
                continue
            for sym in part["coins"]:
                df = data.get(sym)
                if df is None:
                    continue
                try:
                    i = df.index.get_loc(Timestamp(e["time"], tz="UTC"))
                except KeyError:
                    continue
                if i + 1 >= len(df.index):
                    continue
                entry_px = float(df.iloc[i + 1]["open"])
                entry_cost = entry_px * (1 - SLIPPAGE)
                exit_px = None; exit_time = None; reason = "end_of_data"
                bars = 0; mae = 0.0; mfe = 0.0
                for j in range(i + 1, len(df.index)):
                    bars += 1
                    row = df.iloc[j]
                    high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
                    adverse = (high - entry_px) / entry_px
                    favorable = (entry_px - low) / entry_px
                    mae = max(mae, adverse); mfe = max(mfe, favorable)
                    hit_sl = sl > 0 and adverse >= sl
                    hit_tp = tp > 0 and favorable >= tp
                    if hit_sl and hit_tp:
                        exit_px = entry_px * (1 + sl) * (1 - SLIPPAGE); exit_time = df.index[j]
                        reason = "sl_tp_same_bar_sl_wins"; break
                    if hit_sl:
                        exit_px = entry_px * (1 + sl) * (1 - SLIPPAGE); exit_time = df.index[j]
                        reason = "stop_loss"; break
                    if hit_tp:
                        exit_px = entry_px * (1 - tp) * (1 - SLIPPAGE); exit_time = df.index[j]
                        reason = "take_profit"; break
                    if bool(row["ce_dir"] == 1):
                        exit_px = float(df.iloc[j + 1]["open"]) * (1 - SLIPPAGE) if j + 1 < len(df.index) else close * (1 - SLIPPAGE)
                        exit_time = df.index[j + 1] if j + 1 < len(df.index) else df.index[j]
                        reason = "chand_flip"; break
                if exit_px is None:
                    last = df.iloc[-1]
                    exit_px = float(last["close"]) * (1 - SLIPPAGE)
                    exit_time = last.name
                gross = (entry_cost - exit_px) / entry_cost
                ret = gross - 2 * TAKER_FEE
                trades.append({"net_return_pct": ret * 100, "bars_held": bars,
                               "mae_pct": mae * 100, "mfe_pct": mfe * 100,
                               "exit_reason": reason, "symbol": sym,
                               "entry_time": str(df.index[i + 1])})
        return trades

    results = []
    for sl in SL_GRID:
        for tp in TP_GRID:
            ts = simulate(sl, tp)
            if not ts:
                print(f"SL {sl:.3f} TP {tp:.3f}: no trades"); continue
            wins = [t for t in ts if t["net_return_pct"] > 0]
            loss = [t for t in ts if t["net_return_pct"] <= 0]
            gw = sum(t["net_return_pct"] for t in wins)
            gl = abs(sum(t["net_return_pct"] for t in loss))
            s = {
                "sl": sl, "tp": tp, "trades": len(ts),
                "win_rate": len(wins) / len(ts),
                "avg_ret": sum(t["net_return_pct"] for t in ts) / len(ts),
                "pf": gw / gl if gl else float("inf"),
                "total_net": sum(t["net_return_pct"] for t in ts),
                "avg_hold_bars": sum(t["bars_held"] for t in ts) / len(ts),
                "avg_mae": sum(t["mae_pct"] for t in ts) / len(ts),
                "avg_mfe": sum(t["mfe_pct"] for t in ts) / len(ts),
            }
            results.append(s)
            print(f"SL {sl:.3f} TP {tp:.3f}: trades={s['trades']:4d}  WR={s['win_rate']*100:5.1f}%  "
                  f"avg={s['avg_ret']:+.4f}%  PF={s['pf']:5.2f}  total={s['total_net']:+8.2f}%  "
                  f"hold={s['avg_hold_bars']:5.1f}b  MAE={s['avg_mae']:.2f}%  MFE={s['avg_mfe']:.2f}%")

    OUT_GRID.write_text(json.dumps({"timeframe": TIMEFRAME, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved events: {OUT_EVENTS} ({OUT_EVENTS.stat().st_size/1024:.0f} KB)")
    print(f"Saved grid: {OUT_GRID} ({OUT_GRID.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
