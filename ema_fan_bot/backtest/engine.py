"""Backtest engine for EMA Fan Strategy (Bullish/Bearish fan, LONG + SHORT).

Strategy:
  - Bullish Fan (EMA50 > EMA100 > EMA150): LONG when close drops below EMA150
  - Bearish Fan (EMA50 < EMA100 < EMA150): SHORT when close pops above EMA150
  - Stop-loss:  2.0 × ATR from entry price
  - Exit:       Chandelier trailing stop (dynamic ATR-based trailing)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from shared.config_loader import get_config
from shared.data.storage import DataStorage
from ema_fan_bot.strategy.signals import EmaFanSignalGenerator


# ─────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol: str
    direction: str              # "long" | "short"
    timeframe: str
    entry_time: pd.Timestamp
    entry_price: float
    initial_sl: float           # Initial stop-loss price
    stop_price: float           # Current stop-loss (trails over time)
    size: float                 # Position size in asset units
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl_usdt: float = 0.0
    return_pct: float = 0.0
    exit_reason: str = "open"
    bars_held: int = 0
    # Chandelier tracking
    highest_high: float = 0.0
    lowest_low: float = 1e12


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)

    def summary(self) -> dict:
        if not self.trades:
            return {"total_trades": 0}
        pnls = [t.pnl_usdt for t in self.trades if t.exit_time is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        eq = [e for _, e in self.equity_curve]
        max_dd = self._max_drawdown(eq) if len(eq) > 1 else np.nan
        long_trades = [t for t in self.trades if t.direction == "long"]
        short_trades = [t for t in self.trades if t.direction == "short"]
        long_wins = [t for t in long_trades if t.pnl_usdt > 0]
        short_wins = [t for t in short_trades if t.pnl_usdt > 0]
        return {
            "total_trades": len(pnls),
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "win_rate": len(wins) / len(pnls) if pnls else 0,
            "long_win_rate": len(long_wins) / len(long_trades) if long_trades else 0,
            "short_win_rate": len(short_wins) / len(short_trades) if short_trades else 0,
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "total_pnl": sum(pnls),
            "profit_factor": (
                sum(wins) / abs(sum(losses)) if losses and wins else (np.inf if wins else 0)
            ),
            "max_drawdown_pct": max_dd,
            "final_equity": eq[-1] if eq else np.nan,
        }

    @staticmethod
    def _max_drawdown(equity_curve: list[float]) -> float:
        eq = np.array(equity_curve, dtype=float)
        running_max = np.maximum.accumulate(eq)
        drawdown = (eq - running_max) / running_max
        return float(drawdown.min() * 100)


# ─────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────

class EmaFanBacktestEngine:
    """Multi-asset, multi-timeframe backtest for EMA Fan mean-reversion strategy."""

    def __init__(self):
        self.cfg = get_config()
        self.storage = DataStorage()
        self.generator = EmaFanSignalGenerator()

        risk = self.cfg.get("risk", {})
        self.trade_size_usdt = risk.get("trade_size_usdt", 10.0)
        self.stop_loss_mult = risk.get("stop_loss_atr_mult", 2.0)
        self.trailing_mult = risk.get("trailing_atr_mult", 2.5)

        bt = self.cfg.get("backtest", {})
        self.taker_fee = bt.get("taker_fee", 0.0005)
        self.slippage = bt.get("slippage", 0.0003)
        self.max_concurrent = bt.get("max_concurrent_positions", 20)

        self.exclude = (
            set(self.cfg.get("global_exclude_symbols", [])) |
            set(self.cfg.get("exclude_symbols", []))
        )

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def run(self, timeframe: str = "4h") -> pd.DataFrame:
        """Full pipeline: load data → compute signals → backtest → save CSV.

        Args:
            timeframe: "4h" or "1d"

        Returns:
            DataFrame of closed trades.
        """
        logger.info(f"=== EMA Fan Backtest [{timeframe.upper()}] ===")

        symbols = self.storage.list_symbols(timeframe)
        symbols = [s for s in symbols if s not in self.exclude]
        logger.info(f"Found {len(symbols)} symbols for [{timeframe}]")

        if not symbols:
            logger.warning("No symbols available.")
            return pd.DataFrame()

        data: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = self.storage.load_ohlcv(sym, timeframe)
            if df is None or len(df) < 200:
                continue
            df = df.sort_values("timestamp").reset_index(drop=True)
            df = self.generator.compute_signals(df)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            data[sym] = df

        logger.info(f"Loaded and processed {len(data)} symbols")

        result = self._run_simulation(data, timeframe)
        df_trades = self._to_dataframe(result.trades)
        self._save(df_trades, timeframe)
        self._log_summary(result.summary(), timeframe)

        return df_trades

    # ─────────────────────────────────────────────────────────────────
    # Simulation core
    # ─────────────────────────────────────────────────────────────────

    def _run_simulation(
        self,
        data: dict[str, pd.DataFrame],
        timeframe: str,
        initial_capital: float = 10_000.0,
    ) -> BacktestResult:
        all_dates = sorted(set().union(*[set(df.index) for df in data.values()]))
        if not all_dates:
            return BacktestResult()

        capital = initial_capital
        open_trades: dict[str, Trade] = {}    # symbol → Trade
        pending: dict[str, tuple[str, str]] = {}  # symbol → (direction, signal_date)
        result = BacktestResult()

        logger.info(f"Simulating {len(all_dates)} bars across {len(data)} symbols...")

        for current_date in all_dates:

            # ── 1. Execute pending entries (signal on prev bar → enter on this bar's open) ──
            to_remove = []
            for sym, (direction, _) in pending.items():
                df = data[sym]
                if current_date not in df.index:
                    continue
                if len(open_trades) >= self.max_concurrent:
                    to_remove.append(sym)
                    continue

                row = df.loc[current_date]
                if pd.isna(row.get("atr", np.nan)) or pd.isna(row.get("open", np.nan)):
                    to_remove.append(sym)
                    continue

                atr = float(row["atr"])
                if direction == "long":
                    entry_price = float(row["open"]) * (1 - self.slippage)
                    initial_sl = entry_price - self.stop_loss_mult * atr
                else:
                    entry_price = float(row["open"]) * (1 + self.slippage)
                    initial_sl = entry_price + self.stop_loss_mult * atr

                size = self.trade_size_usdt / entry_price
                trade = Trade(
                    symbol=sym,
                    direction=direction,
                    timeframe=timeframe,
                    entry_time=current_date,
                    entry_price=entry_price,
                    initial_sl=initial_sl,
                    stop_price=initial_sl,
                    size=size,
                    highest_high=float(row["high"]),
                    lowest_low=float(row["low"]),
                )
                capital -= entry_price * size * self.taker_fee
                open_trades[sym] = trade
                to_remove.append(sym)

            for s in to_remove:
                pending.pop(s, None)

            # ── 2. Update open positions and check exits ──
            for sym in list(open_trades.keys()):
                df = data[sym]
                if current_date not in df.index:
                    continue
                row = df.loc[current_date]
                trade = open_trades[sym]
                atr_t = float(row.get("atr", np.nan))

                if trade.direction == "long":
                    # Update highest_high for Chandelier (long)
                    trade.highest_high = max(trade.highest_high, float(row["high"]))
                    if not np.isnan(atr_t):
                        trailing = trade.highest_high - self.trailing_mult * atr_t
                        # Trailing never goes below initial_sl direction for long
                        trade.stop_price = max(trade.initial_sl, trailing)

                    # Exit: stop hit → check if LOW fell below stop
                    if float(row["low"]) <= trade.stop_price:
                        exit_price = trade.stop_price * (1 - self.slippage)
                        raw_ret = (exit_price - trade.entry_price) / trade.entry_price
                        self._close_trade(trade, current_date, exit_price, raw_ret, "stop", result, capital)
                        capital += trade.pnl_usdt
                        capital -= exit_price * trade.size * self.taker_fee
                        del open_trades[sym]
                        continue

                else:  # short
                    # Update lowest_low for Chandelier (short)
                    trade.lowest_low = min(trade.lowest_low, float(row["low"]))
                    if not np.isnan(atr_t):
                        trailing = trade.lowest_low + self.trailing_mult * atr_t
                        # Trailing never goes above initial_sl for short
                        trade.stop_price = min(trade.initial_sl, trailing)

                    # Exit: stop hit → check if HIGH rose above stop
                    if float(row["high"]) >= trade.stop_price:
                        exit_price = trade.stop_price * (1 + self.slippage)
                        raw_ret = (trade.entry_price - exit_price) / trade.entry_price
                        self._close_trade(trade, current_date, exit_price, raw_ret, "stop", result, capital)
                        capital += trade.pnl_usdt
                        capital -= exit_price * trade.size * self.taker_fee
                        del open_trades[sym]
                        continue

                trade.bars_held += 1

            # ── 3. Detect new signals on current bar ──
            for sym, df in data.items():
                if sym in open_trades or sym in pending:
                    continue
                if current_date not in df.index:
                    continue
                row = df.loc[current_date]

                if row.get("entry_long", False):
                    pending[sym] = ("long", str(current_date))
                elif row.get("entry_short", False):
                    pending[sym] = ("short", str(current_date))

            result.equity_curve.append((current_date, capital))

        # ── 4. Force-close remaining positions at end of data ──
        for sym, trade in open_trades.items():
            df = data[sym]
            last_row = df.iloc[-1]
            last_date = df.index[-1]
            if trade.direction == "long":
                exit_price = float(last_row["close"]) * (1 - self.slippage)
                raw_ret = (exit_price - trade.entry_price) / trade.entry_price
            else:
                exit_price = float(last_row["close"]) * (1 + self.slippage)
                raw_ret = (trade.entry_price - exit_price) / trade.entry_price
            self._close_trade(trade, last_date, exit_price, raw_ret, "end_of_data", result, capital)

        logger.info(f"Simulation complete: {len(result.trades)} closed trades")
        return result

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def _close_trade(
        self,
        trade: Trade,
        exit_time: pd.Timestamp,
        exit_price: float,
        raw_return: float,
        reason: str,
        result: BacktestResult,
        capital: float,
    ) -> None:
        net_return = raw_return - 2 * self.taker_fee
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.return_pct = net_return
        trade.pnl_usdt = net_return * self.trade_size_usdt
        trade.exit_reason = reason
        result.trades.append(trade)

    def _to_dataframe(self, trades: list[Trade]) -> pd.DataFrame:
        if not trades:
            return pd.DataFrame()
        rows = []
        for t in trades:
            if t.exit_time is None:
                continue
            rows.append({
                "symbol": t.symbol,
                "direction": t.direction,
                "timeframe": t.timeframe,
                "entry_time": t.entry_time,
                "entry_price": round(t.entry_price, 6),
                "initial_sl": round(t.initial_sl, 6),
                "exit_time": t.exit_time,
                "exit_price": round(t.exit_price, 6) if t.exit_price else None,
                "bars_held": t.bars_held,
                "pnl_usdt": round(t.pnl_usdt, 4),
                "return_pct": round(t.return_pct, 4),
                "exit_reason": t.exit_reason,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("exit_time").reset_index(drop=True)
        return df

    def _save(self, df_trades: pd.DataFrame, timeframe: str) -> None:
        out_dir = Path(__file__).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"trades_history_{timeframe}.csv"
        df_trades.to_csv(out_file, index=False)
        logger.info(f"Saved {len(df_trades)} trades → {out_file.name}")

    def _log_summary(self, summary: dict, timeframe: str) -> None:
        logger.info(f"=== [{timeframe.upper()}] BACKTEST SUMMARY ===")
        logger.info(f"  Total trades:  {summary.get('total_trades', 0)}")
        logger.info(f"  Long / Short:  {summary.get('long_trades', 0)} / {summary.get('short_trades', 0)}")
        wr = summary.get("win_rate", 0) * 100
        lwr = summary.get("long_win_rate", 0) * 100
        swr = summary.get("short_win_rate", 0) * 100
        logger.info(f"  Win rate:      {wr:.1f}%  (L: {lwr:.1f}%  S: {swr:.1f}%)")
        logger.info(f"  Total PnL:     ${summary.get('total_pnl', 0):.2f}")
        pf = summary.get("profit_factor", 0)
        logger.info(f"  Profit factor: {pf:.2f}" if not np.isnan(pf) and not np.isinf(pf) else "  Profit factor: ∞")
        dd = summary.get("max_drawdown_pct", np.nan)
        logger.info(f"  Max drawdown:  {dd:.2f}%" if not np.isnan(dd) else "  Max drawdown: N/A")
