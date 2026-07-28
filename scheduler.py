"""[Translated]"""

from __future__ import annotations

from datetime import datetime, timezone

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from config_loader import get_config


class BotScheduler:
    """[Translated]"""

    def __init__(self):
        self.cfg = get_config()
        self.scheduler = BlockingScheduler(timezone="UTC")
        self._setup_jobs()

    def _setup_jobs(self) -> None:
        self.scheduler.add_job(
            self._trade_loop,
            CronTrigger(hour="0,4,8,12,16,20", minute=5),
            id="trade_loop",
            name="Trade Loop (4H)",
            misfire_grace_time=300,
        )
        self.scheduler.add_job(
            self._sync_data,
            CronTrigger(hour="23,3,7,11,15,19", minute=30),
            id="data_sync",
            name="Data Sync",
        )
        self.scheduler.add_job(
            self._retrain_and_update,
            CronTrigger(day_of_week="sun", hour=2, minute=0),
            id="retrain",
            name="Weekly Retrain",
        )
        self.scheduler.add_job(
            self._check_drift,
            CronTrigger(hour=1, minute=0),
            id="drift_check",
            name="Drift Check",
        )

        logger.info("")
        for job in self.scheduler.get_jobs():
            logger.info(f"  {job.name}: {job.trigger}")

    # ──────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────

    def _trade_loop(self) -> None:
        """[Translated]"""
        logger.info(f"=== TRADE LOOP START: {datetime.now(timezone.utc)} ===")

        try:
            from trading.bot import TradingBot
            bot = TradingBot()
            bot.run_cycle()
        except Exception as e:
            logger.error(f"", exc_info=True)

    # ──────────────────────────────────────────────────────────────
    # Data Sync
    # ──────────────────────────────────────────────────────────────

    def _sync_data(self) -> None:
        """[Translated]"""
        logger.info("")
        try:
            from data.collector import BybitCollector
            from data.storage import DataStorage

            storage = DataStorage()
            symbols = storage.list_symbols("4h")
            global_exclude = set(self.cfg.get("global_exclude_symbols", []))
            symbols = [s for s in symbols if s not in global_exclude]

            if symbols:
                collector = BybitCollector(testnet=False)
                collector.collect_all(symbols=symbols, incremental=True)
                logger.info(f"")
        except Exception as e:
            logger.error(f"", exc_info=True)

    # ──────────────────────────────────────────────────────────────
    # Weekly Retrain + Symbol Update
    # ──────────────────────────────────────────────────────────────

    def _retrain_and_update(self) -> None:
        """[Translated]"""
        logger.info("")
        try:
            from data.collector import BybitCollector
            collector = BybitCollector(testnet=False)
            # get_top_symbols() reads active_group from symbol_universe config internally
            top_symbols = collector.get_top_symbols()
            logger.info(f"Retrain: collected {len(top_symbols)} symbols from active group")
            collector.collect_all(symbols=top_symbols, incremental=True)
            from models.trainer import WalkForwardTrainer
            from features.engineer import FeatureEngineer
            from features.market_breadth import MarketBreadthCalculator
            from labeling.triple_barrier import TripleBarrierLabeler
            from data.storage import DataStorage

            storage = DataStorage()
            engineer = FeatureEngineer()
            breadth_calc = MarketBreadthCalculator()
            labeler = TripleBarrierLabeler()
            trainer = WalkForwardTrainer()
            all_symbols_ohlcv = {}
            for sym in top_symbols:
                df = storage.load_ohlcv(sym, "4h")
                if df is not None:
                    all_symbols_ohlcv[sym] = df

            if len(all_symbols_ohlcv) < 5:
                logger.warning("")
                return

            # Breadth
            breadth_df = breadth_calc.compute_all(all_symbols_ohlcv)
            all_labeled = []
            for sym_id, (sym, df_4h) in enumerate(all_symbols_ohlcv.items()):
                df_1h = storage.load_ohlcv(sym, "1h")
                df_1d = storage.load_ohlcv(sym, "1d")
                df_1w = storage.load_ohlcv(sym, "1w")
                funding_df = storage.load_funding(sym)

                df_feat = engineer.compute(df_4h, df_1h, df_1d, funding_df, df_1w=df_1w, symbol_id=sym_id)
                df_feat = breadth_calc.add_symbol_specific(df_feat, breadth_df, all_symbols_ohlcv, sym)
                df_labeled = labeler.label(df_feat)
                df_labeled["symbol"] = sym
                all_labeled.append(df_labeled)

            combined = __import__("pandas").concat(all_labeled, ignore_index=True)
            logger.info(f"")
            model = trainer.train(combined)
            feature_cols = FeatureEngineer.get_feature_columns(combined)
            trainer.save_model(model, feature_cols)
            from models.predictor import Predictor
            predictor = Predictor()
            predictor.reload()

            logger.info("")

        except Exception as e:
            logger.error(f"", exc_info=True)

    # ──────────────────────────────────────────────────────────────
    # Drift Check
    # ──────────────────────────────────────────────────────────────

    def _check_drift(self) -> None:
        """[Translated]"""
        logger.info("")
        try:
            logger.debug("")
        except Exception as e:
            logger.error(f"")

    # ──────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """[Translated]"""
        logger.info("")
        try:
            self.scheduler.start()
        except KeyboardInterrupt:
            logger.info("")
            self.scheduler.shutdown()
