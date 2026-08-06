"""Backtest engine for Fat Tails 4H Grail architecture."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from shared.config_loader import get_config
from shared.data.storage import DataStorage
from fat_tails_bot.screener.screener import FatTailsScreener
from fat_tails_bot.strategy.signals import SignalGenerator


@dataclass
class Trade:
    symbol: str
    direction: str
    entry_time: datetime | pd.Timestamp
    entry_price: float
    initial_sl: float = 0.0
    exit_time: datetime | pd.Timestamp | None = None
    exit_price: float | None = None
    pnl_usdt: float = 0.0
    return_pct: float = 0.0
    exit_reason: str = "open"
    bars_held: int = 0
    size: float = 0.0
    stop_price: float = 0.0
    highest_high: float = -1e12
    lowest_low: float = 1e12

    @property
    def entry_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.entry_time)

    @property
    def exit_date(self) -> pd.Timestamp | None:
        return pd.Timestamp(self.exit_time) if self.exit_time else None

    @property
    def pnl(self) -> float:
        return self.pnl_usdt


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)

    def summary(self) -> dict:
        if not self.trades:
            return {'total_trades': 0}
        pnls = [t.pnl_usdt for t in self.trades if t.pnl_usdt is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        eq = [e for _, e in self.equity_curve]
        max_dd = self._max_drawdown(eq) if eq else np.nan
        return {
            'total_trades': len(pnls),
            'win_rate': len(wins) / len(pnls) if pnls else 0,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'total_pnl': sum(pnls),
            'profit_factor': (sum(wins) / abs(sum(losses))) if losses else np.nan,
            'max_drawdown_pct': max_dd,
            'final_equity': eq[-1] if eq else np.nan,
        }

    @staticmethod
    def _max_drawdown(equity_curve: list[float]) -> float:
        eq = np.array(equity_curve, dtype=float)
        running_max = np.maximum.accumulate(eq)
        drawdown = (eq - running_max) / running_max
        return drawdown.min() * 100


class BacktestEngine:
    """Simulates multi-symbol trading on 4H timeframe using dynamic trailing exit."""

    def __init__(self):
        self.cfg = get_config()
        self.storage = DataStorage()
        self.screener = FatTailsScreener()
        self.generator = SignalGenerator()
        
        risk_cfg = self.cfg.get("risk", {})
        self.trade_size_usdt = risk_cfg.get("trade_size_usdt", 10.0)
        self.trailing_mult = risk_cfg.get("trailing_atr_mult", 2.5)
        
        bt_cfg = self.cfg.get("backtest", {})
        self.taker_fee = bt_cfg.get("taker_fee", 0.0015)
        self.slippage = bt_cfg.get("slippage", 0.0000)

    def run_backtest(self, max_concurrent_positions: int = 100) -> pd.DataFrame:
        logger.info("Running Screener to select Top 20 Universe...")
        top_coins_df = self.screener.screen_universe()
        if top_coins_df.empty:
            logger.warning("No symbols selected by screener.")
            return pd.DataFrame()
            
        symbols = top_coins_df[top_coins_df["selected"]]["symbol"].tolist()
        logger.info(f"Selected {len(symbols)} symbols. Loading 4H data for simulation...")
        
        data = {}
        for sym in symbols:
            df = self.storage.load_ohlcv(sym, "4h")
            if df is not None and len(df) >= 200:
                df = self.generator.compute_signals(df)
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df.set_index('timestamp', inplace=True)
                data[sym] = df
                
        if not data:
            return pd.DataFrame()
            
        result = self.multi_asset_backtest(
            data,
            max_concurrent_positions=max_concurrent_positions,
            execute_next_open=True
        )
        
        return self._trades_to_dataframe(result.trades)

    def multi_asset_backtest(
        self,
        data: dict[str, pd.DataFrame],
        initial_capital: float = 500.0,
        max_concurrent_positions: int = 100,
        execute_next_open: bool = True,
    ) -> BacktestResult:
        logger.info(f"Running multi-asset backtest on {len(data)} symbols...")
        
        all_dates = sorted(set().union(*[set(df.index) for df in data.values()]))
        if not all_dates:
            return BacktestResult()
        
        capital = initial_capital
        open_trades: dict[str, Trade] = {}
        pending_entries: dict[str, pd.Timestamp] = {}
        result = BacktestResult()
        
        for current_date in all_dates:
            # 1. Execute pending entries (signal was on previous bar close)
            if execute_next_open:
                to_remove = []
                for symbol, signal_date in pending_entries.items():
                    df = data[symbol]
                    if current_date not in df.index:
                        continue
                        
                    row = df.loc[current_date]
                    if 'open' not in row:
                        to_remove.append(symbol)
                        continue

                    # Get the atr from the previous bar (where the signal triggered)
                    prev_row = df.loc[signal_date]
                    atr = prev_row.get("atr", np.nan)
                    if pd.isna(atr):
                        to_remove.append(symbol)
                        continue
                        
                    direction = "long" if prev_row.get("signal_long", 0) == 1 else "short"
                    entry_price = row['open'] * (1 + (self.slippage if direction == "long" else -self.slippage))
                    
                    if direction == "long":
                        stop_price = entry_price - atr
                    else:
                        stop_price = entry_price + atr
                        
                    size = self.trade_size_usdt / entry_price
                    
                    trade = Trade(
                        symbol=symbol,
                        direction=direction,
                        entry_time=current_date,
                        entry_price=entry_price,
                        initial_sl=stop_price,
                        stop_price=stop_price,
                        size=size,
                        highest_high=entry_price,
                        lowest_low=entry_price
                    )
                    capital -= entry_price * size * self.taker_fee
                    open_trades[symbol] = trade
                    to_remove.append(symbol)
                
                for s in to_remove:
                    pending_entries.pop(s, None)

            # 2. Check open positions for exits
            for symbol in list(open_trades.keys()):
                df = data[symbol]
                if current_date not in df.index:
                    continue
                row = df.loc[current_date]
                trade = open_trades[symbol]
                
                atr_t = row.get("atr", np.nan)
                if pd.isna(atr_t):
                    continue

                exit_price = None

                if trade.direction == "short":
                    # Update lowest low
                    trade.lowest_low = min(trade.lowest_low, row["low"])
                    # Calculate trailing stop
                    trailing = trade.lowest_low + self.trailing_mult * atr_t
                    trade.stop_price = min(trade.initial_sl, trailing)

                    if row["high"] >= trade.stop_price:
                        # Exit at worst price between open (gap) and stop_price
                        exit_price = max(row["open"], trade.stop_price) * (1.0 + self.slippage)
                        raw_return = (trade.entry_price - exit_price) / trade.entry_price
                
                elif trade.direction == "long":
                    # Update highest high
                    trade.highest_high = max(trade.highest_high, row["high"])
                    # Calculate trailing stop
                    trailing = trade.highest_high - self.trailing_mult * atr_t
                    trade.stop_price = max(trade.initial_sl, trailing)

                    if row["low"] <= trade.stop_price:
                        exit_price = min(row["open"], trade.stop_price) * (1.0 - self.slippage)
                        raw_return = (exit_price - trade.entry_price) / trade.entry_price

                if exit_price is not None:
                    net_return = raw_return - (2 * self.taker_fee)
                    trade.exit_time = current_date
                    trade.exit_price = exit_price
                    trade.return_pct = net_return
                    trade.pnl_usdt = net_return * self.trade_size_usdt
                    trade.exit_reason = "trailing_stop"
                    trade.bars_held = (current_date - pd.to_datetime(trade.entry_time)).total_seconds() // 3600 // 4
                    
                    capital += trade.pnl_usdt
                    capital -= exit_price * trade.size * self.taker_fee
                    result.trades.append(trade)
                    del open_trades[symbol]
                    continue

            # 3. Detect new signals on current bar close
            for symbol, df in data.items():
                if symbol in open_trades or symbol in pending_entries:
                    continue
                if current_date not in df.index:
                    continue
                row = df.loc[current_date]
                
                if row.get("entry_signal", False):
                    pending_entries[symbol] = current_date

            result.equity_curve.append((current_date, capital))

        # Force close remaining positions at end
        if open_trades:
            last_date = all_dates[-1]
            for symbol, trade in open_trades.items():
                df = data[symbol]
                last_row = df.iloc[-1]
                
                if trade.direction == "short":
                    exit_price = last_row["close"] * (1.0 + self.slippage)
                    raw_return = (trade.entry_price - exit_price) / trade.entry_price
                else:
                    exit_price = last_row["close"] * (1.0 - self.slippage)
                    raw_return = (exit_price - trade.entry_price) / trade.entry_price
                    
                net_return = raw_return - (2 * self.taker_fee)
                
                trade.exit_time = last_row.name
                trade.exit_price = exit_price
                trade.return_pct = net_return
                trade.pnl_usdt = net_return * self.trade_size_usdt
                trade.exit_reason = "end_of_data"
                trade.bars_held = (last_row.name - pd.to_datetime(trade.entry_time)).total_seconds() // 3600 // 4
                result.trades.append(trade)

        if not result.trades:
            logger.info("Backtest completed: 0 trades triggered.")
            return result
            
        df_trades = self._trades_to_dataframe(result.trades)
        self._save_trades(df_trades)
        self._log_summary_metrics(df_trades)
        
        return result

    def _trades_to_dataframe(self, trades: list[Trade]) -> pd.DataFrame:
        columns = [
            "symbol", "direction", "entry_time", "entry_price", "initial_sl",
            "exit_time", "exit_price", "bars_held", "pnl_usdt", "return_pct", "exit_reason",
        ]
        trades_data = [
            {
                "symbol": t.symbol,
                "direction": t.direction,
                "entry_time": t.entry_time,
                "entry_price": round(t.entry_price, 6),
                "initial_sl": round(t.initial_sl, 6),
                "exit_time": t.exit_time,
                "exit_price": round(t.exit_price, 6) if t.exit_price else None,
                "bars_held": t.bars_held,
                "pnl_usdt": round(t.pnl_usdt, 4),
                "return_pct": round(t.return_pct, 4),
                "exit_reason": t.exit_reason,
            }
            for t in trades if t.exit_time is not None
        ]
        if not trades_data:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(trades_data).sort_values("exit_time").reset_index(drop=True)

    def _save_trades(self, df_trades: pd.DataFrame) -> None:
        out_dir = Path(__file__).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "trades_history.csv"
        df_trades.to_csv(out_file, index=False)
        logger.info(f"Saved {len(df_trades)} closed trades to {out_file.name}")

    def _log_summary_metrics(self, df_trades: pd.DataFrame) -> None:
        if df_trades.empty:
            return
        total = len(df_trades)
        wins = df_trades[df_trades["pnl_usdt"] > 0]
        win_rate = len(wins) / total * 100
        total_pnl = df_trades["pnl_usdt"].sum()
        profit_factor = wins["pnl_usdt"].sum() / abs(df_trades[df_trades["pnl_usdt"] <= 0]["pnl_usdt"].sum()) if len(wins) < total else np.inf
        
        logger.info(f"=== BACKTEST SUMMARY ===")
        logger.info(f"Total trades: {total}")
        logger.info(f"Win rate: {win_rate:.1f}%")
        logger.info(f"Total PnL: ${total_pnl:.2f}")
        logger.info(f"Profit factor: {profit_factor:.2f}")