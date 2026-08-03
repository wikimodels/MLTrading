"""Backtest engine for Fat Tails D1 dynamic strategy without static Target Profit."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from shared.config_loader import get_config
from shared.data.storage import DataStorage
from fat_tails_bot.screener.screener import load_daily_ohlcv
from fat_tails_bot.strategy.signals import SignalGenerator


@dataclass
class Trade:
symbol: str
entry_date: pd.Timestamp
entry_price: float
stop_price: float
size: float
exit_date: Optional[pd.Timestamp] = None
exit_price: Optional[float] = None
exit_reason: Optional[str] = None
pnl: Optional[float] = None

def close(self, date, price, reason):
self.exit_date = date
self.exit_price = price
self.exit_reason = reason
# SHORT: (entry - exit) * size
self.pnl = (self.entry_price - self.exit_price) * self.size


@dataclass
class MultiAssetBacktestResult:
trades: List[Trade] = field(default_factory=list)
equity_curve: List[Tuple[pd.Timestamp, float]] = field(default_factory=list)

def summary(self) -> dict:
if not self.trades:
return {'total_trades': 0}
pnls = [t.pnl for t in self.trades if t.pnl is not None]
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
'profit_factor': (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float('inf'),
'expected_value': np.mean(pnls) if pnls else 0,
'max_drawdown_pct': max_dd,
'final_equity': eq[-1] if eq else np.nan,
}

@staticmethod
def _max_drawdown(equity_curve: List[float]) -> float:
eq = np.array(equity_curve, dtype=float)
running_max = np.maximum.accumulate(eq)
# Avoid division by zero
running_max[running_max == 0] = 1e-8
drawdown = (eq - running_max) / running_max
return drawdown.min() * 100  # in percentage, negative number


class BacktestEngine:
"""Simulates multi-symbol trading on D1 timeframe using dynamic trailing exit."""

def __init__(self):
self.cfg = get_config()
self.storage = DataStorage()
self.generator = SignalGenerator()

risk_cfg = self.cfg.get("risk", {})
self.trade_size_usdt = risk_cfg.get("trade_size_usdt", 10.0)
self.sl_mult = risk_cfg.get("stop_loss_atr_mult", 1.5)

bt_cfg = self.cfg.get("backtest", {})
self.taker_fee = bt_cfg.get("taker_fee", 0.0005)
self.slippage = bt_cfg.get("slippage", 0.001)
self.funding_daily = bt_cfg.get("funding_daily_pct", 0.0001)

self.exclude = set(self.cfg.get("global_exclude_symbols", ["BTC/USDT:USDT"]))

def run_backtest(self, max_concurrent_positions: int = 100, execute_next_open: bool = True) -> pd.DataFrame:
"""Runs the multi-asset bot farm backtest over all available coins."""
logger.info("Loading all available 4h/1d symbols from storage for the Bot Farm...")
symbols = [s for s in self.storage.list_symbols("4h") if s not in self.exclude]

if not symbols:
logger.error("No valid symbols available in storage for backtest.")
return pd.DataFrame()

logger.info(f"Generating rolling signals for {len(symbols)} symbols. This may take a minute...")

prepared_data = {}
for sym in symbols:
df = load_daily_ohlcv(self.storage, sym)
if df is None or len(df) < 50:
continue

# Compute signals with rolling K/H and lag to prevent look-ahead bias
sig_df = self.generator.compute_signals(df)
if not sig_df.empty:
prepared_data[sym] = sig_df

if not prepared_data:
logger.info("No data generated after signal computation.")
return pd.DataFrame()

logger.info(f"Starting Multi-Asset Bot Farm simulation across {len(prepared_data)} viable symbols.")
result = self.multi_asset_backtest(
data=prepared_data,
initial_capital=10_000.0,
max_concurrent_positions=max_concurrent_positions,
execute_next_open=execute_next_open
)

summary = result.summary()
self._log_summary_metrics(summary)

if not result.trades:
return pd.DataFrame()

# Compile trades DataFrame
trades_data = [
{
"symbol": t.symbol,
"entry_time": t.entry_date,
"entry_price": round(t.entry_price, 6),
"stop_price": round(t.stop_price, 6),
"size": round(t.size, 4),
"exit_time": t.exit_date,
"exit_price": round(t.exit_price, 6) if t.exit_price else None,
"pnl_usdt": round(t.pnl, 4) if t.pnl else 0.0,
"exit_reason": t.exit_reason,
}
for t in result.trades if t.exit_date is not None
]

df_trades = pd.DataFrame(trades_data).sort_values("exit_time").reset_index(drop=True)

# Save results
out_dir = Path(__file__).parent
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "trades_history.csv"
df_trades.to_csv(out_file, index=False)
logger.info(f"Saved {len(df_trades)} closed trades to {out_file.name}")

return df_trades

def compute_stop_price(self, entry_price: float, atr14: float) -> float:
"""SHORT: stop above entry price."""
return entry_price + self.sl_mult * atr14

def multi_asset_backtest(self, 
data: Dict[str, pd.DataFrame],
initial_capital: float = 10_000.0,
max_concurrent_positions: int = 100,
execute_next_open: bool = True) -> MultiAssetBacktestResult:

result = MultiAssetBacktestResult()

# Build unified date calendar
all_dates = sorted(set().union(*[set(df.index) for df in data.values()]))

capital = initial_capital
open_trades: Dict[str, Trade] = {}   # symbol -> Trade
pending_entries: Dict[str, pd.Timestamp] = {}  # symbol -> signal_date

for date_idx, date in enumerate(all_dates):
# 1. Process pending entries (from previous bar's signal)
if execute_next_open:
to_remove = []
for symbol, signal_date in pending_entries.items():
df = data[symbol]
if date not in df.index:
continue
row = df.loc[date]
active_trade.exit_time = last_row["timestamp"]
active_trade.exit_price = exit_price
active_trade.return_pct = raw_return
active_trade.pnl_usdt = raw_return * self.trade_size_usdt
active_trade.exit_reason = "end_of_data"
try:
active_trade.bars_held = (pd.to_datetime(last_row["timestamp"]) - pd.to_datetime(active_trade.entry_time)).days
except Exception:
active_trade.bars_held = 0
trades.append(active_trade)

return trades

def _log_summary_metrics(self, df: pd.DataFrame) -> None:
total_pnl = df["pnl_usdt"].sum()
win_rate = (df["pnl_usdt"] > 0).mean() * 100
n_trades = len(df)
avg_trade = df["pnl_usdt"].mean()
logger.info("=" * 50)
logger.info(f"FAT TAILS BACKTEST SUMMARY ({n_trades} trades)")
logger.info(f"Total PnL:    ${total_pnl:+.2f} USDT")
logger.info(f"Win Rate:     {win_rate:.1f}%")
logger.info(f"Avg Trade:    ${avg_trade:+.4f} USDT")
logger.info("=" * 50)

# 2. Check open positions: stop loss / regime exit
for symbol in list(open_trades.keys()):
df = data[symbol]
if date not in df.index:
continue
row = df.loc[date]
trade = open_trades[symbol]

# Chandelier Trailing Exit (Dynamic Stop Loss)
if not pd.isna(row.get('atr_14')):
chandelier_stop = row['low'] + self.sl_mult * row['atr_14']
if chandelier_stop < trade.stop_price:
trade.stop_price = chandelier_stop

# Stop loss hit
if row['high'] >= trade.stop_price:
trade.close(row['timestamp'], trade.stop_price, 'stop_loss')
capital += trade.pnl
capital -= trade.stop_price * trade.size * self.taker_fee # Pay exit fee
result.trades.append(trade)
del open_trades[symbol]
continue

# Regime exit
if row.get('exit_signal', False):
exit_price = row['close'] * (1 + self.slippage) # short covering
trade.close(row['timestamp'], exit_price, 'regime_exit')
capital += trade.pnl
capital -= exit_price * trade.size * self.taker_fee # Pay exit fee
result.trades.append(trade)
del open_trades[symbol]
continue

# Funding rate for holding short (positive rate = longs pay shorts)
if self.funding_daily != 0:
capital += trade.entry_price * trade.size * self.funding_daily
continue
row = df.loc[date]

if row.get('entry_signal', False):
if execute_next_open:
pending_entries[symbol] = date
else:
if len(open_trades) >= max_concurrent_positions:
continue
entry_price = row['close'] * (1 - self.slippage)
if pd.isna(row.get('atr_14', np.nan)):
continue

stop_price = self.compute_stop_price(entry_price, row['atr_14'])

# AGENTS.md Rule: STRICT $10 USDT position size
if capital >= self.trade_size_usdt:
size = self.trade_size_usdt / entry_price
trade = Trade(symbol=symbol, entry_date=date, entry_price=entry_price,
stop_price=stop_price, size=size)
capital -= entry_price * size * self.taker_fee # Pay entry fee
open_trades[symbol] = trade

result.equity_curve.append((date, capital))

return result

def _log_summary_metrics(self, summary: dict) -> None:
logger.info("=" * 50)
logger.info(f"FAT TAILS BOT FARM BACKTEST SUMMARY")
logger.info(f"Total Trades: {summary.get('total_trades', 0)}")
logger.info(f"Win Rate:     {summary.get('win_rate', 0) * 100:.1f}%")
logger.info(f"Avg Win:      ${summary.get('avg_win', 0):+.4f} USDT")
logger.info(f"Avg Loss:     ${summary.get('avg_loss', 0):+.4f} USDT")
logger.info(f"Total PnL:    ${summary.get('total_pnl', 0):+.2f} USDT")
logger.info(f"Exp. Value:   ${summary.get('expected_value', 0):+.4f} USDT/trade")
pf = summary.get('profit_factor', np.nan)
logger.info(f"Profit Factr: {pf:.2f}")
logger.info(f"Max Drawdown: {summary.get('max_drawdown_pct', 0):.2f}%")
logger.info(f"Final Equity: ${summary.get('final_equity', 0):.2f} USDT")
logger.info("=" * 50)

