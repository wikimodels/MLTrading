"""[Translated]"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from dotenv import load_dotenv
from loguru import logger


class TelegramMonitor:
    """[Translated]"""

    def __init__(self):
        load_dotenv()
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)

        if not self.enabled:
            logger.info("")

    def send(self, message: str) -> None:
        """[Translated]"""
        if not self.enabled:
            return
        try:
            asyncio.run(self._send_async(message))
        except Exception as e:
            logger.error(f"")

    async def _send_async(self, message: str) -> None:
        from telegram import Bot
        bot = Bot(token=self.token)
        await bot.send_message(
            chat_id=self.chat_id,
            text=message,
            parse_mode="HTML",
        )

    def notify_short_opened(self, symbol: str, sizing: dict, confidence: float) -> None:
        msg = (
            f""
            f"Symbol: <code>{symbol}</code>\n"
            f"Entry: <code>{sizing['entry_price']:.2f}</code>\n"
            f"SL: <code>{sizing['sl_price']:.2f}</code>\n"
            f"TP: <code>{sizing['tp_price']:.2f}</code>\n"
            f"R:R: <code>{sizing['rr']:.2f}</code>\n"
            f"Risk: <code>${sizing['risk_usdt']:.2f}</code>\n"
            f"Confidence: <code>{confidence:.1%}</code>"
        )
        self.send(msg)

    def notify_position_closed(self, symbol: str, pnl: float, outcome: str) -> None:
        emoji = "✅" if pnl > 0 else "❌"
        msg = (
            f""
            f"Symbol: <code>{symbol}</code>\n"
            f"Outcome: <code>{outcome}</code>\n"
            f"PnL: <code>${pnl:+.2f}</code>"
        )
        self.send(msg)

    def notify_retrain(self, metrics: Optional[dict] = None) -> None:
        msg = ""
        if metrics:
            msg += f"\nPrecision: <code>{metrics.get('precision', 0):.3f}</code>"
        self.send(msg)

    def notify_drawdown_alert(self, drawdown_pct: float, capital: float) -> None:
        msg = (
            f""
            f"Drawdown: <code>{drawdown_pct:.1f}%</code>\n"
            f""
        )
        self.send(msg)
