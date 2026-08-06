"""Breadth-confirmed short backtest engine (1h, Chandelier stops)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from loguru import logger

from shared.config_loader import get_config
from shared.data.storage import DataStorage
from breadth_bot.screener.screener import load_hourly_ohlcv, BreadthScreener
from breadth_bot.strategy.signals import BreadthSignalGenerator


@dataclass
class Trade:
    symbol: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    stop: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    pnl_usdt: float = 0.0
    return_pct: float = 0.0
    exit_reason: str = "open"
    bars_held: int = 0


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)

    def summary(self) -> dict:
        if not self.trades:
            return {"total_trades": 0}
        pnls = [t.pnl_usdt for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        eq = [e for _, e in self.equity_curve]
        dd = self._max_drawdown(eq) if eq else np.nan
        return {
            "total_trades": len(pnls),
            "win_rate": len(wins) / len(pnls),
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "total_pnl": sum(pnls),
            "profit_factor": (sum(wins) / abs(sum(losses))) if losses else np.nan,
            "max_drawdown_pct": dd,
            "final_equity": eq[-1] if eq else np.nan,
        }

    @staticmethod
    def _max_drawdown(eq: list[float]) -> float:
        arr = np.array(eq, dtype=float)
        running_max = np.maximum.accumulate(arr)
        dd = (arr - running_max) / running_max
        return dd.min() * 100


class BreadthBacktestEngine:
    """Short-on-dump simulation: enter a coin when its own Chandelier break is
    confirmed by market breadth >= threshold; exit on reverse Chandelier break."""

    def __init__(self):
        self.cfg = get_config()
        self.storage = DataStorage()
        self.screener = BreadthScreener()
        self.generator = BreadthSignalGenerator()

        risk_cfg = self.cfg.get("risk", {})
        self.trade_size_usdt = risk_cfg.get("trade_size_usdt", 10.0)
        bt_cfg = self.cfg.get("backtest", {})
        self.initial_capital = bt_cfg.get("initial_capital", 10000.0)
        self.taker_fee = bt_cfg.get("taker_fee", 0.0005)
        self.slippage = bt_cfg.get("slippage", 0.0003)

    def run_backtest(self, symbols: list[str] | None = None) -> BacktestResult:
        if symbols is None:
            symbols = self.screener.select_universe()
        if not symbols:
            logger.warning("Empty universe; nothing to backtest.")
            return BacktestResult()

        data = {}
        for sym in symbols:
            df = load_hourly_ohlcv(self.storage, sym)
            if df is None:
                continue
            data[sym] = self.generator.compute_signals(df)
        if not data:
            return BacktestResult()

        short_breadth, long_breadth = self.generator.compute_breadth(data)
        dates = sorted(set().union(*[set(df.index) for df in data.values()]))

        capital = self.initial_capital
        result = BacktestResult()
        open_pos: dict[str, dict] = {}
        cooldown_until: dict[str, pd.Timestamp] = {}

        for i, t in enumerate(dates):
            # Exits: Chandelier short stop (ratcheted). Exit when price crosses it upward.
            for sym in list(open_pos.keys()):
                pos = open_pos[sym]
                if t not in data[sym].index:
                    continue
                row = data[sym].loc[t]
                stop = pos["stop"]
                if row["high"] >= stop:
                    exit_px = stop if row["open"] < stop else row["open"]
                    reason = "chand_break"
                else:
                    # ratchet is already baked into short_stop series; adopt current value
                    new_stop = row["short_stop"]
                    if not np.isnan(new_stop) and new_stop < stop:
                        pos["stop"] = new_stop
                    continue
                ret = (pos["entry"] - exit_px) / pos["entry"] - 2 * self.taker_fee - 2 * self.slippage
                trade = Trade(
                    symbol=sym, direction="short",
                    entry_time=pos["entry_t"], entry_price=pos["entry"],
                    stop=pos["stop"], exit_time=t, exit_price=exit_px,
                    return_pct=ret, pnl_usdt=ret * self.trade_size_usdt,
                    exit_reason=reason, bars_held=i - pos["i"],
                )
                result.trades.append(trade)
                cooldown_until[sym] = t + pd.Timedelta(hours=48)
                del open_pos[sym]

            # Entries: sellSignal (dir flips 1->-1) + short breadth confirmed,
            # execute on bar t+1 open. Cooldown prevents instant re-entry.
            for sym, df in data.items():
                if sym in open_pos:
                    continue
                if cooldown_until.get(sym) is not None and t < cooldown_until[sym]:
                    continue
                if t not in df.index or t not in short_breadth.index:
                    continue
                if not df.at[t, "short_sig"]:
                    continue
                sb = short_breadth.loc[t]
                if not (np.isfinite(sb) and sb >= self.generator.breadth_min):
                    continue
                if idx + 1 >= len(df.index):
                    continue
                t_next = df.index[idx + 1]
                if t_next not in dates:
                    continue
                entry_px = df.loc[t_next, "open"] * (1 + self.slippage)
                # Short stop known at signal time (ratcheted chandelier short stop on signal bar)
                stop = df.loc[t, "short_stop"]
                if not np.isfinite(stop):
                    continue
                open_pos[sym] = {"entry": entry_px, "entry_t": t_next, "stop": stop, "i": i}

            result.equity_curve.append((t, capital))

        # Force-close remaining positions at end of data
        for sym, pos in open_pos.items():
            df = data[sym]
            last = df.iloc[-1]
            ret = (pos["entry"] - last["close"]) / pos["entry"] - 2 * self.taker_fee - 2 * self.slippage
            result.trades.append(Trade(
                symbol=sym, direction="short",
                entry_time=pos["entry_t"], entry_price=pos["entry"], stop=pos["stop"],
                exit_time=last.name, exit_price=last["close"],
                return_pct=ret, pnl_usdt=ret * self.trade_size_usdt,
                exit_reason="end_of_data", bars_held=len(dates) - pos["i"],
            ))

        return result

    def log_summary(self, result: BacktestResult) -> None:
        s = result.summary()
        logger.info(f"=== BREADTH SHORT BACKTEST SUMMARY ===")
        logger.info(f"Total trades: {s['total_trades']}")
        logger.info(f"Win rate: {s['win_rate']*100:.1f}%")
        logger.info(f"Total PnL: ${s['total_pnl']:.2f}")
        logger.info(f"Profit factor: {s['profit_factor']:.2f}")
        logger.info(f"Max drawdown: {s['max_drawdown_pct']:.1f}%")
