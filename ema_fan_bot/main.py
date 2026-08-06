"""EMA Fan Bot — CLI entry point.

Usage:
    poetry run python ema_fan_bot/main.py backtest          # Both timeframes
    poetry run python ema_fan_bot/main.py backtest --tf 4h  # 4H only
    poetry run python ema_fan_bot/main.py backtest --tf 1d  # 1D only
    poetry run python ema_fan_bot/main.py status            # Data summary
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
os.chdir(_root)

from shared.config_loader import set_active_bot
set_active_bot("ema_fan_bot")

import argparse
from loguru import logger

from shared.data.storage import DataStorage
from ema_fan_bot.backtest.engine import EmaFanBacktestEngine


# ─────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────

LOG_FILE = Path(__file__).parent / "logs" / "ema_fan.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logger.add(LOG_FILE, rotation="10 MB", retention="30 days", level="DEBUG")


# ─────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────

def cmd_backtest(tf: str | None) -> None:
    engine = EmaFanBacktestEngine()
    from shared.config_loader import get_config
    cfg = get_config()
    timeframes = cfg["data"]["timeframes"] if tf is None else [tf]

    for timeframe in timeframes:
        df = engine.run(timeframe=timeframe)
        if df.empty:
            logger.warning(f"No trades generated for [{timeframe}]")
        else:
            logger.info(
                f"[{timeframe.upper()}] Done: {len(df)} trades, "
                f"PnL ${df['pnl_usdt'].sum():.2f}, "
                f"WR {(df['pnl_usdt'] > 0).mean() * 100:.1f}%"
            )


def cmd_status() -> None:
    storage = DataStorage()
    from shared.config_loader import get_config
    cfg = get_config()
    timeframes = cfg["data"]["timeframes"]

    print("\n=== EMA Fan Bot — Data Status ===")
    for tf in timeframes:
        symbols = storage.list_symbols(tf)
        print(f"\n  [{tf.upper()}]  {len(symbols)} symbols available")
        for sym in symbols[:10]:
            df = storage.load_ohlcv(sym, tf)
            if df is not None and len(df) > 0:
                print(
                    f"    {sym:30s}  {len(df):5d} bars  "
                    f"{df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}"
                )
        if len(symbols) > 10:
            print(f"    ... and {len(symbols) - 10} more")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ema_fan_bot",
        description="EMA Fan Mean-Reversion Strategy — Backtest CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # backtest command
    p_bt = subparsers.add_parser("backtest", help="Run backtest (default: all timeframes)")
    p_bt.add_argument(
        "--tf",
        choices=["4h", "1d"],
        default=None,
        help="Timeframe to backtest (default: both)",
    )

    # status command
    subparsers.add_parser("status", help="Show data availability summary")

    args = parser.parse_args()

    if args.command == "backtest":
        cmd_backtest(args.tf)
    elif args.command == "status":
        cmd_status()


if __name__ == "__main__":
    main()
