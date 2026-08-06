"""Breadth bot CLI: screener and backtest."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_bot_dir = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_bot_dir) not in sys.path:
    sys.path.insert(0, str(_bot_dir))
os.chdir(_bot_dir)

from loguru import logger
from shared.config_loader import set_active_bot

set_active_bot("breadth_bot")

logger.remove()
logger.add(sys.stdout, level="INFO")


def cmd_screen() -> None:
    from breadth_bot.screener.screener import BreadthScreener
    sc = BreadthScreener()
    universe = sc.select_universe()
    print(f"\nUniverse ({len(universe)} coins):")
    for sym in universe:
        print(f"  {sym}")


def cmd_backtest() -> None:
    from breadth_bot.backtest.engine import BreadthBacktestEngine
    engine = BreadthBacktestEngine()
    result = engine.run_backtest()
    engine.log_summary(result)

    if not result.trades:
        print("\nNo trades.")
        return

    import pandas as pd
    df = pd.DataFrame([{
        "symbol": t.symbol, "entry_time": t.entry_time, "exit_time": t.exit_time,
        "entry": round(t.entry_price, 6), "exit": round(t.exit_price, 6) if t.exit_price else None,
        "bars": t.bars_held, "ret": round(t.return_pct, 4),
        "pnl": round(t.pnl_usdt, 4), "exit_reason": t.exit_reason,
    } for t in result.trades])
    df["year"] = pd.to_datetime(df["entry_time"]).dt.year

    print("\n=== BY YEAR ===")
    print(df.groupby("year").agg(trades=("pnl", "count"), pnl=("pnl", "sum")).to_string())
    print("\n=== BY EXIT REASON ===")
    print(df.groupby("exit_reason").agg(trades=("pnl", "count"), pnl=("pnl", "sum"),
                                        avg_ret=("ret", "mean"), avg_bars=("bars", "mean")).to_string())
    print("\n=== BY SYMBOL ===")
    g = df.groupby("symbol").agg(trades=("pnl", "count"), pnl=("pnl", "sum"))
    g["wr"] = df.groupby("symbol")["pnl"].apply(lambda x: (x > 0).mean() * 100).round(0)
    print(g.sort_values("trades", ascending=False).head(20).to_string())
    print("\n=== SAMPLE TRADES (worst & best) ===")
    sample = df.sort_values("pnl")
    print(pd.concat([sample.head(5), sample.tail(5)])[["symbol", "entry_time", "exit_time", "entry", "exit", "bars", "ret", "pnl", "exit_reason"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Market breadth short-on-dump bot")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("screen")
    sub.add_parser("backtest")
    args = parser.parse_args()

    if args.command == "screen":
        cmd_screen()
    elif args.command == "backtest":
        cmd_backtest()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
