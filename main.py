"""[Translated]"""

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
)


def cmd_collect():
    """[Translated]"""
    from data.collector import BybitCollector
    logger.info("")
    collector = BybitCollector(testnet=False)
    symbols = collector.collect_all(incremental=False)
    logger.info(f"")


def cmd_train():
    """[Translated]"""
    import pandas as pd
    from data.storage import DataStorage
    from features.engineer import FeatureEngineer
    from features.market_breadth import MarketBreadthCalculator
    from labeling.triple_barrier import TripleBarrierLabeler
    from models.trainer import WalkForwardTrainer

    logger.info("")

    storage = DataStorage()
    engineer = FeatureEngineer()
    breadth_calc = MarketBreadthCalculator()
    labeler = TripleBarrierLabeler()
    trainer = WalkForwardTrainer()

    symbols = storage.list_symbols("4h")
    if not symbols:
        logger.error("")
        sys.exit(1)

    logger.info(f"")
    all_symbols_ohlcv = {}
    for sym in symbols:
        df = storage.load_ohlcv(sym, "4h")
        if df is not None and len(df) > 200:
            all_symbols_ohlcv[sym] = df

    # Market Breadth
    logger.info("")
    breadth_df = breadth_calc.compute_all(all_symbols_ohlcv)
    all_labeled = []
    for sym_id, (sym, df_4h) in enumerate(all_symbols_ohlcv.items()):
        logger.info(f"")

        df_1h = storage.load_ohlcv(sym, "1h")
        df_1d = storage.load_ohlcv(sym, "1d")
        funding_df = storage.load_funding(sym)

        df_feat = engineer.compute(df_4h, df_1h, df_1d, funding_df, symbol_id=sym_id)
        df_feat = breadth_calc.add_symbol_specific(df_feat, breadth_df, all_symbols_ohlcv, sym)
        df_labeled = labeler.label(df_feat)
        df_labeled["symbol"] = sym
        all_labeled.append(df_labeled)

    combined = pd.concat(all_labeled, ignore_index=True)
    logger.info(f"")
    model = trainer.train(combined, verbose=True)
    from features.engineer import FeatureEngineer as FE
    feature_cols = FE.get_feature_columns(combined)
    path = trainer.save_model(model, feature_cols)
    logger.info(f"")


def cmd_backtest():
    """[Translated]"""
    import pandas as pd
    from datetime import datetime, timezone, timedelta
    from data.storage import DataStorage
    from features.engineer import FeatureEngineer
    from features.market_breadth import MarketBreadthCalculator
    from labeling.triple_barrier import TripleBarrierLabeler
    from models.trainer import WalkForwardBacktestTrainer
    from backtest.engine import BacktestEngine

    logger.info("")

    storage = DataStorage()
    engineer = FeatureEngineer()
    breadth_calc = MarketBreadthCalculator()
    labeler = TripleBarrierLabeler()

    symbols = storage.list_symbols("4h")
    if not symbols:
        logger.error("")
        sys.exit(1)
    all_symbols_ohlcv = {}
    for sym in symbols:
        df = storage.load_ohlcv(sym, "4h")
        if df is not None and len(df) > 200:
            all_symbols_ohlcv[sym] = df

    breadth_df = breadth_calc.compute_all(all_symbols_ohlcv)

    all_labeled = []
    for sym_id, (sym, df_4h) in enumerate(all_symbols_ohlcv.items()):
        df_1h = storage.load_ohlcv(sym, "1h")
        df_1d = storage.load_ohlcv(sym, "1d")
        funding_df = storage.load_funding(sym)

        df_feat = engineer.compute(df_4h, df_1h, df_1d, funding_df, symbol_id=sym_id)
        df_feat = breadth_calc.add_symbol_specific(df_feat, breadth_df, all_symbols_ohlcv, sym)
        df_labeled = labeler.label(df_feat)
        df_labeled["symbol"] = sym
        all_labeled.append(df_labeled)

    combined = pd.concat(all_labeled, ignore_index=True)
    wf_trainer = WalkForwardBacktestTrainer()
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=365 * 2)

    wf_results = wf_trainer.run(combined, start_date, end_date)
    engine = BacktestEngine()
    results = engine.run(wf_results)
    engine.print_report(results)
    engine.save_pdf_report(results, "backtest/backtest_report.pdf")
    if not results.equity_curve.empty:
        results.equity_curve.to_csv("backtest/equity_curve.csv", index=False)
        logger.info("")


def cmd_run():
    """[Translated]"""
    from scheduler import BotScheduler
    logger.info("")
    scheduler = BotScheduler()
    scheduler.start()


def cmd_status():
    """[Translated]"""
    from data.storage import DataStorage
    storage = DataStorage()
    df = storage.status()
    if df.empty:
        print("")
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
        print(f"")
        sys.exit(1)

    cmd = sys.argv[1]
    logger.info(f"")
    COMMANDS[cmd]()
