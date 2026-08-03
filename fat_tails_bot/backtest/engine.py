"""Backtest engine for Fat Tails D1 dynamic strategy without static Target Profit."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from shared.config_loader import get_config
from shared.data.storage import DataStorage
from fat_tails_bot.screener.screener import FatTailsScreener, load_daily_ohlcv
from fat_tails_bot.strategy.signals import SignalGenerator


@dataclass
class Trade:
    symbol: str
    direction: str
    entry_time: datetime | pd.Timestamp
    entry_price: float
    entry_tr_zscore: float = 0.0
    entry_clv: float = 0.0
    initial_sl: float = 0.0
    exit_time: datetime | pd.Timestamp | None = None
    exit_price: float | None = None
    pnl_usdt: float = 0.0
    return_pct: float = 0.0
    exit_reason: str = "open"
    bars_held: int = 0
    size: float = 0.0
    stop_price: float = 0.0

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
    """Simulates multi-symbol trading on D1 timeframe using dynamic trailing exit."""

    def __init__(self):
        self.cfg = get_config()
        self.storage = DataStorage()
        self.screener = FatTailsScreener()
        self.generator = SignalGenerator()
        
        risk_cfg = self.cfg.get("risk", {})
        self.trade_size_usdt = risk_cfg.get("trade_size_usdt", 10.0)
        self.trailing_mult = risk_cfg.get("trailing_atr_mult", 2.0)
        self.stop_loss_mult = risk_cfg.get("stop_loss_atr_mult", 1.5)
        
        bt_cfg = self.cfg.get("backtest", {})
        self.taker_fee = bt_cfg.get("taker_fee", 0.0005)
        self.slippage = bt_cfg.get("slippage", 0.0003)
        
        strat_cfg = self.cfg.get("strategy", {})
        self.exclude = set(self.cfg.get("global_exclude_symbols", []))
        self.clv_long_min = strat_cfg.get("clv_long_min", 0.75)

    def run_backtest(self, max_concurrent_positions: int = 100) -> pd.DataFrame:
        """Legacy API: runs full pipeline from screener to trades_history.csv"""
        logger.info("Starting Bot Farm time-series backtest...")
        
        screen_df = self.screener.screen_universe()
        if screen_df is None or screen_df.empty:
            logger.warning("Screener returned no symbols.")
            return pd.DataFrame()
            
        qualifying = screen_df[screen_df["selected"] == True]
        if qualifying.empty:
            logger.warning("No symbols met strict criteria. Testing top ones.")
            symbols_to_test = screen_df.head(10)["symbol"].tolist()
        else:
            symbols_to_test = qualifying.head(max_concurrent_positions)["symbol"].tolist()
            
        logger.info(f"Loading {len(symbols_to_test)} symbols for Bot Farm simulation...")
        data = {}
        for sym in symbols_to_test:
            df = load_daily_ohlcv(self.storage, sym)
            if df is not None and len(df) >= 50:
                df = self.generator.compute_signals(df)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                data[sym] = df
                
        if not data:
            return pd.DataFrame()
            
        result = self.multi_asset_backtest(
            data,
            max_concurrent_positions=max_concurrent_positions,
            execute_next_open=False
        )
        
        return self._trades_to_dataframe(result.trades)

    def multi_asset_backtest(
        self,
        data: dict[str, pd.DataFrame],
        initial_capital: float = 10000.0,
        max_concurrent_positions: int = 5,
        execute_next_open: bool = True,
    ) -> BacktestResult:
        """
        Multi-asset backtest on pre-computed signal data.
        
        Args:
            data: dict {symbol: DataFrame} with signals already computed (index = datetime)
            initial_capital: starting capital
            max_concurrent_positions: max simultaneous open positions
            execute_next_open: if True, enter on next bar open; if False, enter on signal bar close
        """
        logger.info(f"Running multi-asset backtest on {len(data)} symbols...")
        
        all_dates = sorted(set().union(*[set(df.index) for df in data.values()]))
        if not all_dates:
            return BacktestResult()
        
        capital = initial_capital
        open_trades: dict[str, Trade] = {}
        pending_entries: dict[str, pd.Timestamp] = {}
        result = BacktestResult()
        
        logger.info(f"Simulating time-series across {len(all_dates)} days...")
        
        for current_date in all_dates:
            # 1. Execute pending entries (signal was on previous bar)
            if execute_next_open:
                to_remove = []
                for symbol, signal_date in pending_entries.items():
                    df = data[symbol]
                    if current_date not in df.index:
                        continue
                    row = df.loc[current_date]
                    if 'open' not in row or pd.isna(row.get('atr_14', np.nan)):
                        to_remove.append(symbol)
                        continue

                    if len(open_trades) >= max_concurrent_positions:
                        to_remove.append(symbol)
                        continue

                    entry_price = row['open'] * (1 + self.slippage)
                    atr = row['atr_14']
                    stop_price = entry_price + self.stop_loss_mult * atr
                    size = self.trade_size_usdt / entry_price
                    
                    trade = Trade(
                        symbol=symbol,
                        direction="short",
                        entry_time=current_date,
                        entry_price=entry_price,
                        initial_sl=stop_price,
                        stop_price=stop_price,
                        size=size,
                        entry_tr_zscore=row.get("tr_zscore", 0.0),
                        entry_clv=row.get("clv", 0.0)
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

                # Stop loss hit (check HIGH for short)
                if row["high"] >= trade.stop_price:
                    exit_price = trade.stop_price * (1.0 + self.slippage)
                    raw_return = (trade.entry_price - exit_price) / trade.entry_price
                    net_return = raw_return - (2 * self.taker_fee)
                    
                    trade.exit_time = current_date
                    trade.exit_price = exit_price
                    trade.return_pct = net_return
                    trade.pnl_usdt = net_return * self.trade_size_usdt
                    trade.exit_reason = "stop_loss_hit"
                    trade.bars_held = (current_date - pd.to_datetime(trade.entry_time)).days
                    
                    capital += trade.pnl_usdt
                    capital -= exit_price * trade.size * self.taker_fee
                    result.trades.append(trade)
                    del open_trades[symbol]
                    continue

                # Theory exit: EMA5 revert or CLV revert
                ema5_val = row.get("ema5", 1e9)
                clv_val = row.get("clv", 0.5)
                if row["close"] > ema5_val or clv_val >= self.clv_long_min:
                    reason = "ema5_reverted" if row["close"] > ema5_val else "clv_reverted"
                    exit_price = row["close"] * (1.0 + self.slippage)
                    raw_return = (trade.entry_price - exit_price) / trade.entry_price
                    net_return = raw_return - (2 * self.taker_fee)
                    
                    trade.exit_time = current_date
                    trade.exit_price = exit_price
                    trade.return_pct = net_return
                    trade.pnl_usdt = net_return * self.trade_size_usdt
                    trade.exit_reason = reason
                    trade.bars_held = (current_date - pd.to_datetime(trade.entry_time)).days
                    
                    capital += trade.pnl_usdt
                    capital -= exit_price * trade.size * self.taker_fee
                    result.trades.append(trade)
                    del open_trades[symbol]
                    continue

            # 3. Detect new signals on current bar
            for symbol, df in data.items():
                if symbol in open_trades or symbol in pending_entries:
                    continue
                if current_date not in df.index:
                    continue
                row = df.loc[current_date]
                
                if row.get("entry_signal", False):
                    if execute_next_open:
                        pending_entries[symbol] = current_date
                    else:
                        if len(open_trades) >= max_concurrent_positions:
                            continue
                        entry_price = row["close"] * (1 + self.slippage)
                        atr = row.get("atr_14", np.nan)
                        if pd.isna(atr):
                            continue
                        
                        stop_price = entry_price + self.stop_loss_mult * atr
                        size = self.trade_size_usdt / entry_price
                        
                        trade = Trade(
                            symbol=symbol,
                            direction="short",
                            entry_time=current_date,
                            entry_price=entry_price,
                            initial_sl=stop_price,
                            stop_price=stop_price,
                            size=size,
                            entry_tr_zscore=row.get("tr_zscore", 0.0),
                            entry_clv=row.get("clv", 0.0)
                        )
                        capital -= entry_price * size * self.taker_fee
                        open_trades[symbol] = trade

            result.equity_curve.append((current_date, capital))

        # Force close remaining positions at end
        if open_trades:
            last_date = all_dates[-1]
            for symbol, trade in open_trades.items():
                df = data[symbol]
                last_row = df.iloc[-1]
                exit_price = last_row["close"] * (1.0 + self.slippage)
                raw_return = (trade.entry_price - exit_price) / trade.entry_price
                net_return = raw_return - (2 * self.taker_fee)
                
                trade.exit_time = last_row.name
                trade.exit_price = exit_price
                trade.return_pct = net_return
                trade.pnl_usdt = net_return * self.trade_size_usdt
                trade.exit_reason = "end_of_data"
                trade.bars_held = (last_row.name - pd.to_datetime(trade.entry_time)).days
                result.trades.append(trade)

        if not result.trades:
            logger.info("Backtest completed: 0 trades triggered.")
            return result
            
        df_trades = self._trades_to_dataframe(result.trades)
        self._save_trades(df_trades)
        self._log_summary_metrics(df_trades)
        
        return result

    def _trades_to_dataframe(self, trades: list[Trade]) -> pd.DataFrame:
        trades_data = [
            {
                "symbol": t.symbol,
                "direction": t.direction,
                "entry_time": t.entry_time,
                "entry_price": round(t.entry_price, 6),
                "initial_sl": round(t.initial_sl, 6),
                "entry_tr_zscore": round(t.entry_tr_zscore, 2),
                "entry_clv": round(t.entry_clv, 3),
                "exit_time": t.exit_time,
                "exit_price": round(t.exit_price, 6) if t.exit_price else None,
                "bars_held": t.bars_held,
                "pnl_usdt": round(t.pnl_usdt, 4),
                "return_pct": round(t.return_pct, 4),
                "exit_reason": t.exit_reason,
            }
            for t in trades if t.exit_time is not None
        ]
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