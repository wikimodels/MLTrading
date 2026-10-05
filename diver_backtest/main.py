import sys
import argparse
from pathlib import Path
from loguru import logger

# Setup paths
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from diver_backtest.config import Config
from diver_backtest.engine import run_backtest
from diver_backtest.grid_search import run_grid_search

def run():
    parser = argparse.ArgumentParser(description="Multi-coin Divergence Backtester")
    parser.add_argument("mode", nargs="?", default="run", help="Mode: run, grid")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()

    cfg = Config()
    if args.start:
        cfg.start_date = args.start
    if args.end:
        cfg.end_date = args.end

    if args.mode == "grid":
        logger.info(f"Запуск Grid Search... Период: {cfg.start_date} -> {cfg.end_date}")
        report = run_grid_search(cfg)
        print("\n=== Отчет Grid Search (Топ 20 по Net Return) ===")
        # handle cp1252 print issue gracefully
        try:
            print(report.head(20).to_string())
        except UnicodeEncodeError:
            pass
        logger.info(f"Отчет сохранен в {cfg.output_dir}/grid_search_report.csv")
        return

    logger.info(f"Запуск бэктеста мульти-валютных дивергенций... Период: {cfg.start_date} -> {cfg.end_date}")
    results = run_backtest(cfg)
    
    if not results["trades"].empty:
        logger.info(f"Сводка успешно построена. Всего сделок: {len(results['trades'])}")

if __name__ == "__main__":
    run()
