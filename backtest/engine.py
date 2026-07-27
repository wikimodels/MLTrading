"""[Translated]"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config_loader import get_config


@dataclass
class Trade:
    """[Translated]"""
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: float
    qty: float
    side: str  # 'short'
    outcome: str  # 'tp', 'sl', 'timeout', 'forced'
    pnl_usdt: float
    pnl_pct: float
    commission_usdt: float
    funding_usdt: float
    net_pnl_usdt: float
    bars_held: int
    confidence: float


@dataclass
class BacktestResults:
    """[Translated]"""
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.net_pnl_usdt > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.net_pnl_usdt <= 0)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0

    @property
    def total_pnl(self) -> float:
        return sum(t.net_pnl_usdt for t in self.trades)

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.net_pnl_usdt for t in self.trades if t.net_pnl_usdt > 0)
        gross_loss = abs(sum(t.net_pnl_usdt for t in self.trades if t.net_pnl_usdt < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    @property
    def avg_win(self) -> float:
        wins = [t.net_pnl_usdt for t in self.trades if t.net_pnl_usdt > 0]
        return np.mean(wins) if wins else 0

    @property
    def avg_loss(self) -> float:
        losses = [t.net_pnl_usdt for t in self.trades if t.net_pnl_usdt <= 0]
        return np.mean(losses) if losses else 0


class BacktestEngine:
    """[Translated]"""

    def __init__(self, initial_capital: Optional[float] = None):
        cfg = get_config()
        bcfg = cfg["backtest"]
        rcfg = cfg["risk"]

        self.initial_capital = initial_capital or bcfg["initial_capital"]
        self.commission_taker = bcfg["commission_taker"]   # 0.00055
        self.commission_maker = bcfg["commission_maker"]   # 0.00020
        self.slippage = bcfg["slippage"]                   # 0.0005
        self.leverage = cfg["trading"]["leverage"]          # 3

        self.risk_pct = rcfg["risk_per_trade_pct"] / 100
        self.sl_atr_mult = rcfg["atr_sl_mult"]
        self.max_positions = rcfg["max_open_positions"]
        self.min_confidence = cfg["model"]["min_confidence"]

    def run(self, wf_results: list[dict]) -> BacktestResults:
        """[Translated]"""
        results = BacktestResults()
        capital = self.initial_capital
        equity_rows = []

        logger.info(f"")

        for window in wf_results:
            val_df = window["val_df"].copy()
            val_df = val_df.sort_values("timestamp").reset_index(drop=True)
            window_trades = self._simulate_window(val_df, capital)

            for trade in window_trades:
                capital += trade.net_pnl_usdt
                results.trades.append(trade)
                equity_rows.append({
                    "timestamp": trade.exit_time,
                    "capital": capital,
                    "trade_pnl": trade.net_pnl_usdt,
                    "symbol": trade.symbol,
                })

        # Equity curve
        if equity_rows:
            results.equity_curve = pd.DataFrame(equity_rows)
            results.equity_curve["drawdown"] = self._compute_drawdown(
                results.equity_curve["capital"]
            )

        return results

    def _simulate_window(self, df: pd.DataFrame, capital: float) -> list[Trade]:
        """[Translated]"""
        trades = []
        open_positions: dict[str, dict] = {}

        for i, row in df.iterrows():
            symbol = row.get("symbol", "UNKNOWN")
            ts = row["timestamp"]
            for sym in list(open_positions.keys()):
                pos = open_positions[sym]
                sym_rows = df[(df.get("symbol", "UNKNOWN") == sym) if "symbol" in df.columns else df.index >= 0]
                trade = self._check_exit(pos, row, sym)
                if trade:
                    trades.append(trade)
                    capital += trade.net_pnl_usdt
                    del open_positions[sym]
            if (
                row.get("prediction", 0) == 1
                and row.get("confidence", 0) >= self.min_confidence
                and symbol not in open_positions
                and len(open_positions) < self.max_positions
                and "atr_14" in row.index
            ):
                entry_price = row["close"] * (1 + self.slippage)  # slippage
                atr = row["atr_14"]

                sl_price = entry_price + atr * self.sl_atr_mult
                tp_dist = atr * get_config()["labeling"]["tp_atr_mult"]
                tp_price = entry_price - tp_dist

                sl_dist = sl_price - entry_price
                risk_usdt = capital * self.risk_pct * min(1.0, row.get("confidence", 1.0))
                qty = risk_usdt / sl_dist if sl_dist > 0 else 0

                if qty > 0:
                    open_positions[symbol] = {
                        "symbol": symbol,
                        "entry_time": ts,
                        "entry_price": entry_price,
                        "qty": qty,
                        "sl_price": sl_price,
                        "tp_price": tp_price,
                        "bars_held": 0,
                        "confidence": row.get("confidence", 1.0),
                        "funding_rate": row.get("funding_rate", 0.0),
                        "max_bars": get_config()["labeling"]["max_bars"],
                    }
        for sym, pos in open_positions.items():
            last_row = df.iloc[-1]
            exit_price = last_row["close"] * (1 - self.slippage)
            trade = self._close_trade(pos, exit_price, df.iloc[-1]["timestamp"], "forced")
            trades.append(trade)

        return trades

    def _check_exit(self, pos: dict, row: pd.Series, symbol: str) -> Optional[Trade]:
        """[Translated]"""
        pos["bars_held"] += 1
        low = row.get("low", row["close"])
        high = row.get("high", row["close"])
        if low <= pos["tp_price"]:
            exit_price = pos["tp_price"] * (1 - self.slippage)
            return self._close_trade(pos, exit_price, row["timestamp"], "tp")
        if high >= pos["sl_price"]:
            exit_price = pos["sl_price"] * (1 + self.slippage)
            return self._close_trade(pos, exit_price, row["timestamp"], "sl")
        if pos["bars_held"] >= pos["max_bars"]:
            exit_price = row["close"] * (1 - self.slippage)
            return self._close_trade(pos, exit_price, row["timestamp"], "timeout")

        return None

    def _close_trade(
        self, pos: dict, exit_price: float, exit_time, outcome: str
    ) -> Trade:
        """[Translated]"""
        entry_price = pos["entry_price"]
        qty = pos["qty"]
        bars_held = pos["bars_held"]
        pnl_usdt = (entry_price - exit_price) * qty
        pnl_pct = (entry_price - exit_price) / entry_price
        notional = qty * entry_price
        commission = notional * self.commission_taker * 2
        funding_payments = bars_held // 2
        funding_usdt = pos.get("funding_rate", 0) * notional * funding_payments

        net_pnl = pnl_usdt - commission + funding_usdt

        return Trade(
            symbol=pos["symbol"],
            entry_time=pos["entry_time"],
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            qty=qty,
            side="short",
            outcome=outcome,
            pnl_usdt=pnl_usdt,
            pnl_pct=pnl_pct,
            commission_usdt=commission,
            funding_usdt=funding_usdt,
            net_pnl_usdt=net_pnl,
            bars_held=bars_held,
            confidence=pos.get("confidence", 1.0),
        )

    @staticmethod
    def _compute_drawdown(equity: pd.Series) -> pd.Series:
        """[Translated]"""
        rolling_max = equity.cummax()
        return (equity - rolling_max) / rolling_max

    # ──────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────

    def print_report(self, results: BacktestResults) -> None:
        """[Translated]"""
        t = results

        if not t.trades:
            logger.warning("")
            return

        final_capital = self.initial_capital + t.total_pnl
        total_return = t.total_pnl / self.initial_capital * 100

        # Max Drawdown
        if not results.equity_curve.empty:
            max_dd = results.equity_curve["drawdown"].min() * 100
        else:
            max_dd = 0.0
        if not results.equity_curve.empty and len(results.equity_curve) > 1:
            daily_returns = results.equity_curve["capital"].pct_change().dropna()
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        else:
            sharpe = 0.0
        tp_trades = [tr for tr in t.trades if tr.outcome == "tp"]
        sl_trades = [tr for tr in t.trades if tr.outcome == "sl"]
        timeout_trades = [tr for tr in t.trades if tr.outcome == "timeout"]

        report = f"""[Translated]"""
        try:
            print(report)
        except Exception:
            print("")
            
        logger.info("\n" + report)

    def save_pdf_report(self, results: BacktestResults, path: str = "backtest_report.pdf") -> None:
        """[Translated]"""
        try:
            from fpdf import FPDF
        except ImportError:
            logger.error("")
            return

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, "BACKTEST REPORT - ML Swing Short Bot", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(5)
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 8, "1. Overall Performance", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 11)
        
        t = results
        final_capital = self.initial_capital + t.total_pnl
        total_return = (t.total_pnl / self.initial_capital) * 100
        max_dd = results.equity_curve["drawdown"].min() * 100 if not results.equity_curve.empty else 0.0

        lines = [
            f"Initial Capital: ${self.initial_capital:,.2f}",
            f"Final Capital:   ${final_capital:,.2f}",
            f"Net PnL:         ${t.total_pnl:,.2f} ({total_return:+.1f}%)",
            f"Max Drawdown:    {max_dd:.2f}%",
            f"Profit Factor:   {t.profit_factor:.3f}",
            f"Total Trades:    {t.total_trades}",
            f"Win Rate:        {t.win_rate*100:.1f}%",
            f"Wins / Losses:   {t.wins} / {t.losses}",
        ]
        for line in lines:
            pdf.cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
            
        pdf.ln(8)
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 8, "2. Per-Symbol Performance", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "B", 9)
        
        col_w = [35, 20, 20, 30, 20, 40]
        headers = ["Symbol", "Trades", "Win %", "PnL ($)", "PF", "Avg Win/Loss ($)"]
        for i, h in enumerate(headers):
            pdf.cell(col_w[i], 7, h, border=1, align="C")
        pdf.ln()
        
        pdf.set_font("helvetica", "", 9)
        
        from collections import defaultdict
        sym_trades = defaultdict(list)
        for tr in results.trades:
            sym_trades[tr.symbol].append(tr)
            
        for sym, trs in sorted(sym_trades.items()):
            wins = [tr.net_pnl_usdt for tr in trs if tr.net_pnl_usdt > 0]
            losses = [tr.net_pnl_usdt for tr in trs if tr.net_pnl_usdt <= 0]
            total = len(trs)
            wr = len(wins) / total * 100 if total > 0 else 0
            pnl = sum(tr.net_pnl_usdt for tr in trs)
            gl = abs(sum(losses))
            pf = sum(wins) / gl if gl > 0 else float('inf')
            aw = np.mean(wins) if wins else 0
            al = np.mean(losses) if losses else 0
            
            pdf.cell(col_w[0], 7, sym, border=1)
            pdf.cell(col_w[1], 7, str(total), border=1, align="C")
            pdf.cell(col_w[2], 7, f"{wr:.1f}%", border=1, align="R")
            pdf.cell(col_w[3], 7, f"${pnl:.2f}", border=1, align="R")
            pdf.cell(col_w[4], 7, f"{pf:.2f}", border=1, align="R")
            pdf.cell(col_w[5], 7, f"{aw:.1f} / {al:.1f}", border=1, align="C")
            pdf.ln()
            
        try:
            pdf.output(path)
            logger.info(f"")
        except Exception as e:
            logger.error(f"")
