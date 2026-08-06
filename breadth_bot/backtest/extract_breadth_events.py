"""Stage 1: extract all market-breadth events (breadth >= BREADTH_MIN) into JSON.

For every hour of the last 1y, when the fraction of the universe that JUST entered a
Chandelier Exit(21, 2.5) SHORT signal (or LONG signal) reaches >= BREADTH_MIN, the event
is recorded with the exact list of participating coins and market-state context.

Output: breadth_bot/backtest/breadth_events.json
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

from loguru import logger
from shared.config_loader import set_active_bot
set_active_bot("breadth_bot")

from shared.data.storage import DataStorage
from breadth_bot.screener.screener import load_hourly_ohlcv
from breadth_bot.strategy.signals import BreadthSignalGenerator

HOURS_1Y = 24 * 365
BREADTH_MIN = 0.50
OUT = Path(r"D:\GitHub\MLTrading\breadth_bot\backtest\breadth_events.json")


def main() -> None:
    storage = DataStorage()
    gen = BreadthSignalGenerator()
    symbols = storage.list_symbols("1h")

    data = {}
    for sym in symbols:
        df = load_hourly_ohlcv(storage, sym)
        if df is None:
            continue
        df = df.tail(HOURS_1Y)
        if len(df) < HOURS_1Y:
            logger.warning(f"{sym}: only {len(df)}h, skipped")
            continue
        sig = gen.compute_signals(df)
        data[sym] = sig[["ce_dir", "short_sig", "long_sig", "close"]]

    universe = sorted(data.keys())
    dates = sorted(set().union(*[set(df.index) for df in data.values()]))
    n = len(universe)

    events = []
    for t in dates:
        short_coins, long_coins = [], []
        state_short, state_long = 0, 0
        for sym in universe:
            df = data[sym]
            if t not in df.index:
                continue
            if bool(df.at[t, "short_sig"]):
                short_coins.append(sym)
            if bool(df.at[t, "long_sig"]):
                long_coins.append(sym)
            d = int(df.at[t, "ce_dir"])
            if d == -1:
                state_short += 1
            elif d == 1:
                state_long += 1

        short_breath = len(short_coins) / n
        long_breath = len(long_coins) / n
        if short_breath >= BREADTH_MIN or long_breath >= BREADTH_MIN:
            events.append({
                "time": t.strftime("%Y-%m-%d %H:%M"),
                "timestamp_utc": t.timestamp(),
                "short": {
                    "breadth": round(short_breath, 4),
                    "count": len(short_coins),
                    "coins": short_coins,
                },
                "long": {
                    "breadth": round(long_breath, 4),
                    "count": len(long_coins),
                    "coins": long_coins,
                },
                "market_state": {
                    "in_short": state_short,
                    "in_long": state_long,
                    "neutral": n - state_short - state_long,
                    "universe": n,
                },
            })

    doc = {
        "meta": {
            "generator": "extract_breadth_events.py",
            "strategy": "Chandelier Exit(21, 2.5), 1h, events of direction flip",
            "breadth_min": BREADTH_MIN,
            "universe": universe,
            "universe_size": n,
            "hours": len(dates),
            "range": f"{dates[0]:%Y-%m-%d} -> {dates[-1]:%Y-%m-%d}",
            "note": "breadth = fraction of universe that JUST entered SHORT/LONG in this hour (entry event, not held state)",
        },
        "events": events,
    }

    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
    n_short = sum(1 for e in events if e["short"]["count"] > 0)
    n_long = sum(1 for e in events if e["long"]["count"] > 0)
    n_both = sum(1 for e in events if e["short"]["count"] > 0 and e["long"]["count"] > 0)
    print(f"events: {len(events)} total  ({n_short} with short-entry, {n_long} with long-entry, {n_both} with both)")
    max_s = max((e["short"]["count"] for e in events), default=0)
    max_l = max((e["long"]["count"] for e in events), default=0)
    print(f"max short count: {max_s}/{n}, max long count: {max_l}/{n}")
    for e in events[:5]:
        print(f'  {e["time"]}  short={e["short"]["count"]} long={e["long"]["count"]} state(short/long)={e["market_state"]["in_short"]}/{e["market_state"]["in_long"]}')


if __name__ == "__main__":
    main()
