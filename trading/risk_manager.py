"""[Translated]"""

from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger

from config_loader import get_config
from labeling.triple_barrier import TripleBarrierLabeler


class RiskManager:
    """[Translated]"""

    def __init__(self):
        cfg = get_config()["risk"]
        self.risk_pct = cfg["risk_per_trade_pct"] / 100       # 1% → 0.01
        self.max_positions = cfg["max_open_positions"]         # 3
        self.max_drawdown = cfg["max_drawdown_pct"] / 100     # 15% → 0.15
        self.sl_atr_mult = cfg["atr_sl_mult"]                 # 1.5
        self.funding_blackout = cfg["funding_blackout_minutes""""[Translated]"""
        check = self._pre_checks(symbol)
        if not check["allowed"]:
            return check

        # ── ATR levels ────────────────────────────────────────────
        levels = self.labeler.label_realtime(entry_price, atr)
        sl_price = levels["sl_price"]
        tp_price = levels["tp_price"]
        sl_dist = sl_price - entry_price
        confidence_factor = min(1.0, max(0.5, confidence))
        risk_usdt = capital * self.risk_pct * confidence_factor
        qty = risk_usdt / sl_dist
        notional = qty * entry_price

        logger.info(
            f""
            f"entry={entry_price:.2f} | "
            f"SL={sl_price:.2f} (+{levels['sl_dist_pct']*100:.1f}%) | "
            f"TP={tp_price:.2f} (-{levels['tp_dist_pct']*100:.1f}%) | "
            f"R:R={levels['rr']:.2f} | "
            f"Qty={qty:.4f} | "
            f"Risk=${risk_usdt:.2f} | "
            f"Notional=${notional:.2f}"
        )

        return {
            "allowed": True,
            "reason": "",
            "symbol": symbol,
            "qty": qty,
            "notional": notional,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "risk_usdt": risk_usdt,
            "rr": levels["rr"],
            "entry_price": entry_price,
        }

    # ──────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────

    def _pre_checks(self, symbol: str) -> dict:
        """[Translated]"""
        if symbol in self.open_positions:
            return {"allowed": False, "reason": f""}
        if len(self.open_positions) >= self.max_positions:
            return {
                "allowed": False,
                "reason": f""
            }
        if self._is_funding_blackout():
            return {
                "allowed": False,
                "reason": ""
            }

        return {"allowed": True, "reason": ""}

    def _is_funding_blackout(self) -> bool:
        """[Translated]"""
        now = datetime.now(timezone.utc)
        minutes_since_funding = (now.hour % 8) * 60 + now.minute
        minutes_to_funding = (8 * 60) - minutes_since_funding

        if minutes_to_funding <= self.funding_blackout:
            logger.debug(f"")
            return True
        return False

    # ──────────────────────────────────────────────────────────────
    # Drawdown Guard
    # ──────────────────────────────────────────────────────────────

    def update_capital(self, current_capital: float) -> bool:
        """[Translated]"""
        if self.peak_capital is None or current_capital > self.peak_capital:
            self.peak_capital = current_capital

        drawdown = (self.peak_capital - current_capital) / self.peak_capital

        if drawdown >= self.max_drawdown:
            logger.critical(
                f""
                f""
                f""
            )
            return False

        logger.debug(f"")
        return True

    # ──────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────

    def register_open(self, symbol: str, position: dict) -> None:
        """[Translated]"""
        self.open_positions[symbol] = position
        logger.info(f"")

    def register_close(self, symbol: str, pnl: float) -> None:
        """[Translated]"""
        if symbol in self.open_positions:
            del self.open_positions[symbol]
        logger.info(f""
                    f"")

    def get_status(self) -> dict:
        """[Translated]"""
        return {
            "open_positions": len(self.open_positions),
            "positions": list(self.open_positions.keys()),
            "peak_capital": self.peak_capital,
        }
