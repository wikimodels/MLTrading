"""ML Swing Short Bot — entry point."""

from __future__ import annotations

import sys
from loguru import logger
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | {message}",
)
logger.add(
    "logs/mltrading.log",
    level="DEBUG",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
    encoding="utf-8",
)


def cmd_collect():
    """Collect OHLCV + funding data from Bybit for the selected universe."""
    from data.collector import BybitCollector
    logger.info("Starting data collection...")
    collector = BybitCollector(testnet=False)
    symbols = collector.collect_all(incremental=False)
    logger.info(f"Collection done: {len(symbols)} symbols collected")


def cmd_train():
    """Train the walk-forward LightGBM model on all collected data."""
    import pandas as pd
    from data.storage import DataStorage
    from features.engineer import FeatureEngineer
    from features.market_breadth import MarketBreadthCalculator
    from labeling.triple_barrier import TripleBarrierLabeler
    from models.trainer import WalkForwardTrainer

    logger.info("Starting model training...")

    storage = DataStorage()
    engineer = FeatureEngineer()
    breadth_calc = MarketBreadthCalculator()
    labeler = TripleBarrierLabeler()
    trainer = WalkForwardTrainer()

    from data.collector import BybitCollector
    collector = BybitCollector()
    active_symbols = set(collector.get_top_symbols("all"))

    symbols = storage.list_symbols("4h")
    symbols = [s for s in symbols if s in active_symbols]
    if not symbols:
        logger.error("No symbols found in storage. Run 'collect' first.")
        sys.exit(1)

    logger.info(f"Building dataset for {len(symbols)} symbols...")
    all_symbols_ohlcv = {}
    for sym in symbols:
        df = storage.load_ohlcv(sym, "4h")
        if df is not None and len(df) > 200:
            all_symbols_ohlcv[sym] = df

    logger.info("Computing market breadth features...")
    breadth_df = breadth_calc.compute_all(all_symbols_ohlcv)
    all_labeled = []
    for sym_id, (sym, df_4h) in enumerate(all_symbols_ohlcv.items()):
        logger.info(f"  [{sym_id+1}/{len(all_symbols_ohlcv)}] Engineering features for {sym}...")

        df_1h = storage.load_ohlcv(sym, "1h")
        df_1d = storage.load_ohlcv(sym, "1d")
        funding_df = storage.load_funding(sym)

        df_feat = engineer.compute(df_4h, df_1h, df_1d, funding_df, symbol_id=sym_id)
        df_feat = breadth_calc.add_symbol_specific(df_feat, breadth_df, all_symbols_ohlcv, sym)
        df_labeled = labeler.label(df_feat)
        df_labeled["symbol"] = sym
        if sym != "BTC/USDT:USDT":
            all_labeled.append(df_labeled)

    combined = pd.concat(all_labeled, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined)} rows, training model...")
    model = trainer.train(combined, verbose=True)
    from features.engineer import FeatureEngineer as FE
    feature_cols = FE.get_feature_columns(combined)
    path = trainer.save_model(model, feature_cols)
    logger.info(f"Model saved to {path}")


def cmd_backtest():
    """Run walk-forward backtest and generate PDF report."""
    import pandas as pd
    from datetime import datetime, timezone, timedelta
    from data.storage import DataStorage
    from features.engineer import FeatureEngineer
    from features.market_breadth import MarketBreadthCalculator
    from labeling.triple_barrier import TripleBarrierLabeler
    from models.trainer import WalkForwardBacktestTrainer
    from backtest.engine import BacktestEngine

    logger.info("Starting backtest...")

    storage = DataStorage()
    engineer = FeatureEngineer()
    breadth_calc = MarketBreadthCalculator()
    labeler = TripleBarrierLabeler()

    from config_loader import get_config
    cfg = get_config()
    exclude_symbols = set(cfg["trading"].get("exclude_symbols", []))

    from data.collector import BybitCollector
    collector = BybitCollector()
    active_symbols = set(collector.get_top_symbols("all"))

    symbols = storage.list_symbols("4h")
    symbols = [s for s in symbols if s in active_symbols and s not in exclude_symbols]

    if not symbols:
        logger.error("No symbols found in storage. Run 'collect' first.")
        sys.exit(1)

    logger.info(f"Loading data for {len(symbols)} symbols...")
    all_symbols_ohlcv = {}
    for sym in symbols:
        df = storage.load_ohlcv(sym, "4h")
        if df is not None and len(df) > 200:
            all_symbols_ohlcv[sym] = df

    breadth_df = breadth_calc.compute_all(all_symbols_ohlcv)

    all_labeled = []
    for sym_id, (sym, df_4h) in enumerate(all_symbols_ohlcv.items()):
        logger.info(f"  [{sym_id+1}/{len(all_symbols_ohlcv)}] Processing {sym}...")
        df_1h = storage.load_ohlcv(sym, "1h")
        df_1d = storage.load_ohlcv(sym, "1d")
        funding_df = storage.load_funding(sym)

        df_feat = engineer.compute(df_4h, df_1h, df_1d, funding_df, symbol_id=sym_id)
        df_feat = breadth_calc.add_symbol_specific(df_feat, breadth_df, all_symbols_ohlcv, sym)
        df_labeled = labeler.label(df_feat)
        df_labeled["symbol"] = sym
        if sym != "BTC/USDT:USDT":
            all_labeled.append(df_labeled)

    combined = pd.concat(all_labeled, ignore_index=True)
    logger.info(f"Dataset ready: {len(combined)} rows across {len(all_labeled)} symbols")

    wf_trainer = WalkForwardBacktestTrainer()
    # Фиксируем дату окончания бэктеста до июня 2026
    end_date = datetime(2026, 6, 1, tzinfo=timezone.utc)
    start_date = end_date - timedelta(days=int(365 * 2.5))

    logger.info(f"Walk-forward: {start_date.date()} -> {end_date.date()}")
    wf_results = wf_trainer.run(combined, start_date, end_date)

    engine = BacktestEngine()
    results = engine.run(wf_results)
    engine.print_report(results)
    engine.save_pdf_report(results, "backtest/backtest_report.pdf")

    if not results.equity_curve.empty:
        results.equity_curve.to_csv("backtest/equity_curve.csv", index=False)
        logger.info("Equity curve saved to backtest/equity_curve.csv")
    
    # Сохраняем полный лог всех сделок для дашборда
    if results.trades:
        trades_data = []
        for t in results.trades:
            trades_data.append({
                "symbol": t.symbol,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "qty": t.qty,
                "side": t.side,
                "outcome": t.outcome,
                "pnl_usdt": t.pnl_usdt,
                "pnl_pct": t.pnl_pct,
                "commission_usdt": t.commission_usdt,
                "funding_usdt": t.funding_usdt,
                "net_pnl_usdt": t.net_pnl_usdt
            })
        import pandas as pd
        pd.DataFrame(trades_data).to_csv("backtest/trades_history.csv", index=False)
        logger.info(f"Saved {len(trades_data)} trades to backtest/trades_history.csv")


def cmd_run():
    """Start the live trading scheduler."""
    from scheduler import BotScheduler
    logger.info("Starting live trading bot...")
    scheduler = BotScheduler()
    scheduler.start()


def cmd_status():
    """Show summary of collected data."""
    from data.storage import DataStorage
    storage = DataStorage()
    df = storage.status()
    if df.empty:
        print("No data found. Run 'collect' first.")
    else:
        print(df.to_string(index=False))


COMMANDS = {
    "collect": cmd_collect,
    "train": cmd_train,
    "backtest": cmd_backtest,
    "run": cmd_run,
    "status": cmd_status,
}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: python main.py [{' | '.join(COMMANDS)}]")
        sys.exit(1)

    cmd = sys.argv[1]
    logger.info(f"Command: {cmd}")
    COMMANDS[cmd]()
