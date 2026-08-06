"""Extract Chandelier(21, 2.5) short/long entry signals for ALL symbols (incl. BTC)
over the last 1 year of 1h data. Writes a single common CSV file."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
os.chdir(_root)

import pandas as pd
from loguru import logger
from shared.config_loader import set_active_bot
set_active_bot("breadth_bot")

from shared.data.storage import DataStorage
from breadth_bot.screener.screener import load_hourly_ohlcv
from breadth_bot.strategy.signals import BreadthSignalGenerator

HOURS_1Y = 24 * 365  # 8760


def main() -> None:
    storage = DataStorage()
    gen = BreadthSignalGenerator()
    symbols = storage.list_symbols("1h")  # includes BTC/USDT:USDT
    logger.info(f"Processing {len(symbols)} symbols (incl BTC) over last 1y of 1h data...")

    all_rows = []
    for sym in symbols:
        df = load_hourly_ohlcv(storage, sym)
        if df is None:
            logger.warning(f"No data for {sym}")
            continue
        df = df.tail(HOURS_1Y)
        if len(df) < HOURS_1Y:
            logger.warning(f"{sym}: only {len(df)}h of history, skipped (need {HOURS_1Y})")
            continue
        sig = gen.compute_signals(df)

        # Fresh entries are the CE buy/sell signals (dir flips); execution on next open
        rows = []
        idxs = sig.index
        for t in range(len(sig)):
            se = bool(sig["short_sig"].iloc[t])
            le = bool(sig["long_sig"].iloc[t])
            if se or le:
                if t + 1 >= len(sig):
                    continue
                rows.append({
                    "symbol": sym,
                    "direction": "short" if se else "long",
                    "signal_time": idxs[t],
                    "exec_open": idxs[t + 1],
                    "signal_close": round(sig["close"].iloc[t], 8),
                    "entry_open": round(sig["open"].iloc[t + 1], 8),
                    "stop_price": round(sig["short_stop"].iloc[t], 8) if se else round(sig["long_stop"].iloc[t], 8),
                    "atr21": round(sig["atr"].iloc[t], 8),
                    "ce_dir_after": int(sig["ce_dir"].iloc[t]),
                })
        all_rows.extend(rows)
        logger.info(f"{sym}: {len(rows)} entries (short={sum(r['direction']=='short' for r in rows)}, long={sum(r['direction']=='long' for r in rows)})")

    out = pd.DataFrame(all_rows).sort_values(["symbol", "signal_time"])
    out_path = _root / "breadth_bot" / "backtest" / "chandelier_entries_1y.csv"
    out.to_csv(out_path, index=False)
    total = len(out)
    short = int((out["direction"] == "short").sum())
    long = int((out["direction"] == "long").sum())
    print(f"\nSaved {total} entries to {out_path}")
    print(f"  short: {short}, long: {long}")
    print(f"  symbols with entries: {out['symbol'].nunique()} / {len(symbols)}")


if __name__ == "__main__":
    main()
