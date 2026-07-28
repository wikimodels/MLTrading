"""[Translated]"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from loguru import logger

from config_loader import get_config
from data.storage import DataStorage
from features.engineer import FeatureEngineer
from features.market_breadth import MarketBreadthCalculator
from models.predictor import Predictor
from trading.risk_manager import RiskManager
from trading.executor import OrderExecutor


class TradingBot:
    """[Translated]"""

    def __init__(self, testnet: bool = True):
        cfg = get_config()
        self.cfg = cfg
        self.testnet = testnet

        self.storage = DataStorage()
        self.engineer = FeatureEngineer()
        self.breadth_calc = MarketBreadthCalculator()
        self.predictor = Predictor()
        self.risk_manager = RiskManager()
        self.executor = OrderExecutor(testnet=testnet)

        logger.info(f"")

    def run_cycle(self) -> None:
        """[Translated]"""
        logger.info(f"")
        capital = self.executor.get_balance()
        if capital <= 0:
            logger.warning("")
            return

        if not self.risk_manager.update_capital(capital):
            logger.critical("")
            return

        logger.info(f"")
        global_exclude = set(self.cfg.get("global_exclude_symbols", []))
        symbols = self.storage.list_symbols("4h")
        symbols = [s for s in symbols if s not in global_exclude]
        if not symbols:
            logger.warning("")
            return
        all_symbols_ohlcv = {}
        for sym in symbols:
            df = self.storage.load_ohlcv(sym, "4h")
            if df is not None and len(df) > 50:
                all_symbols_ohlcv[sym] = df

        if len(all_symbols_ohlcv) < 5:
            logger.warning("")
            return

        breadth_df = self.breadth_calc.compute_all(all_symbols_ohlcv)
        for sym in symbols:
            try:
                self._process_symbol(sym, all_symbols_ohlcv, breadth_df, capital)
            except Exception as e:
                logger.error(f"")

        logger.info("")

    def _process_symbol(
        self,
        symbol: str,
        all_symbols_ohlcv: dict,
        breadth_df: pd.DataFrame,
        capital: float,
    ) -> None:
        """[Translated]"""

        df_4h = all_symbols_ohlcv.get(symbol)
        if df_4h is None or len(df_4h) < 200:
            return
        df_1h = self.storage.load_ohlcv(symbol, "1h")
        df_1d = self.storage.load_ohlcv(symbol, "1d")
        df_1w = self.storage.load_ohlcv(symbol, "1w")
        funding_df = self.storage.load_funding(symbol)

        sym_id = list(all_symbols_ohlcv.keys()).index(symbol)
        df_feat = self.engineer.compute(
            df_4h.tail(300),
            df_1h, df_1d, funding_df, df_1w=df_1w, symbol_id=sym_id
        )
        df_feat = self.breadth_calc.add_symbol_specific(
            df_feat, breadth_df, all_symbols_ohlcv, symbol
        )

        if df_feat.empty:
            return
        signal = self.predictor.predict(df_feat)

        if not signal["trade"]:
            return
        last_bar = df_feat.iloc[-1]
        entry_price = self.executor.get_current_price(symbol)
        atr = float(last_bar["atr_14"])

        sizing = self.risk_manager.calculate_position(
            capital=capital,
            entry_price=entry_price,
            atr=atr,
            symbol=symbol,
            confidence=signal["confidence"],
        )

        if not sizing["allowed"]:
            logger.info(f"")
            return
        order = self.executor.open_short(
            symbol=symbol,
            qty=sizing["qty"],
            sl_price=sizing["sl_price"],
            tp_price=sizing["tp_price"],
        )

        if "error" not in order:
            self.risk_manager.register_open(symbol, sizing)
            logger.info(
                f""
                f"qty={sizing['qty']:.4f} | "
                f"entry={entry_price:.2f} | "
                f"SL={sizing['sl_price']:.2f} | "
                f"TP={sizing['tp_price']:.2f} | "
                f"confidence={signal['confidence']:.3f}"
            )
