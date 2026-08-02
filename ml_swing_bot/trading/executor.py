"""[Translated]"""

from __future__ import annotations

import os
import time
from typing import Optional

import ccxt
from dotenv import load_dotenv
from loguru import logger

from shared.config_loader import get_config


class OrderExecutor:
    """[Translated]"""

    def __init__(self, testnet: bool = True):
        load_dotenv()
        cfg = get_config()
        self.cfg = cfg
        self.leverage = cfg["trading"]["leverage"]

        params = {
            "enableRateLimit": True,
            "options": {"defaultType": "linear"},
        }

        if testnet:
            params["apiKey"] = os.getenv("BYBIT_API_KEY", "")
            params["secret"] = os.getenv("BYBIT_API_SECRET", "")
        else:
            params["apiKey"] = os.getenv("BYBIT_MAINNET_API_KEY", "")
            params["secret"] = os.getenv("BYBIT_MAINNET_API_SECRET", "")

        self.exchange = ccxt.bybit(params)

        if testnet:
            self.exchange.set_sandbox_mode(True)

        logger.info(f"")

    # ──────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────

    def open_short(
        self,
        symbol: str,
        qty: float,
        sl_price: float,
        tp_price: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> dict:
        """[Translated]"""
        try:
            self._set_leverage(symbol)
            logger.info(
                f""
                f"SL={sl_price:.2f} | TP={tp_price:.2f}"
            )

            params = {
                "stopLoss": {
                    "triggerPrice": sl_price,
                    "price": sl_price,
                    "type": "market",
                },
                "takeProfit": {
                    "triggerPrice": tp_price,
                    "price": tp_price,
                    "type": "market",
                },
                "positionIdx": 2,  # Bybit: 2 = short side (hedge mode)
            }

            if order_type == "market":
                order = self.exchange.create_market_sell_order(
                    symbol, qty, params=params
                )
            else:
                assert limit_price is not None
                order = self.exchange.create_limit_sell_order(
                    symbol, qty, limit_price, params=params
                )

            logger.info(
                f""
                f"side=short | price={order.get('price')} | qty={order['amount']}"
            )
            return order

        except ccxt.InsufficientFunds as e:
            logger.error(f"")
            return {"error": str(e)}
        except ccxt.InvalidOrder as e:
            logger.error(f"")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────

    def close_position(self, symbol: str, qty: float) -> dict:
        """[Translated]"""
        try:
            params = {"positionIdx": 2}  # hedge mode short
            order = self.exchange.create_market_buy_order(
                symbol, qty, params=params
            )
            logger.info(f"")
            return order
        except Exception as e:
            logger.error(f"")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────

    def get_open_positions(self) -> list[dict]:
        """[Translated]"""
        try:
            positions = self.exchange.fetch_positions()
            active = [
                p for p in positions
                if p.get("contracts") and float(p.get("contracts", 0)) > 0
            ]
            return active
        except Exception as e:
            logger.error(f"")
            return []

    def get_balance(self) -> float:
        """[Translated]"""
        try:
            balance = self.exchange.fetch_balance()
            usdt = balance.get("USDT", {})
            total = usdt.get("total", 0) or 0
            return float(total)
        except Exception as e:
            logger.error(f"")
            return 0.0

    def get_current_price(self, symbol: str) -> float:
        """[Translated]"""
        ticker = self.exchange.fetch_ticker(symbol)
        return float(ticker["last"])

    # ──────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────

    def _set_leverage(self, symbol: str) -> None:
        """[Translated]"""
        try:
            self.exchange.set_leverage(self.leverage, symbol)
            logger.debug(f"")
        except Exception as e:
            logger.debug(f"")
