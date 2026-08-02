"""Fat Tails D1 Bot — entry point."""

from __future__ import annotations

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
from shared.config_loader import set_active_bot
set_active_bot("fat_tails_bot")

from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | {message}",
)
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logger.add(
    "logs/fat_tails.log",
    level="DEBUG",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
    encoding="utf-8",
)


def cmd_screener():
    """Run quantitative screening for heavy tails and trend persistence."""
    from fat_tails_bot.screener.screener import FatTailsScreener
    logger.info("Starting quantitative screening...")
    screener = FatTailsScreener()
    df = screener.screen_universe()
    if df.empty:
        print("No qualifying symbols evaluated.")
    else:
        print("\n=== FAT TAILS SCREENER RESULTS ===")
        print(df.to_string(index=False))


def cmd_backtest():
    """Run walk-forward / sequential backtest on D1 timeframe."""
    from fat_tails_bot.backtest.engine import BacktestEngine
    logger.info("Starting D1 strategy backtest...")
    engine = BacktestEngine()
    df = engine.run_backtest()
    if not df.empty:
        print("\n=== RECENT TRADES SAMPLE ===")
        print(df.tail(15).to_string(index=False))


def cmd_run():
    """Start live execution loop for Fat Tails bot (placeholder / future scheduler)."""
    logger.info("Live trading execution loop for Fat Tails D1 Bot is ready for deployment.")
    print("Live execution loop will be connected to Bybit executor in next release.")


def cmd_status():
    """Show status of shared database from Fat Tails perspective."""
    from shared.data.storage import DataStorage
    storage = DataStorage()
    df = storage.status()
    if df.empty:
        print("No data found in shared storage.")
    else:
        print(df.to_string(index=False))


COMMANDS = {
    "screener": cmd_screener,
    "backtest": cmd_backtest,
    "run": cmd_run,
    "status": cmd_status,
}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python -m fat_tails_bot.main <command>")
        print(f"Commands: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd not in COMMANDS:
        print(f"Unknown command: '{cmd}'")
        print(f"Commands: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    COMMANDS[cmd]()


if __name__ == "__main__":
    main()
