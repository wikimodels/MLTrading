"""Backtest engine for Fat Tails D1 dynamic strategy without static Target Profit."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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

bt_cfg = self.cfg.get("backtest", {})
self.taker_fee = bt_cfg.get("taker_fee", 0.0005)
self.slippage = bt_cfg.get("slippage", 0.0003)

def run_backtest(self, max_coins: int = 20) -> pd.DataFrame:
"""Runs screening and simulates trades across qualifying symbols."""
logger.info("Running universe screening before backtest...")
screen_df = self.screener.screen_universe()
if screen_df.empty:
logger.error("No symbols available in storage for backtest.")
return pd.DataFrame()

qualifying = screen_df[screen_df["selected"] == True]
if qualifying.empty:
logger.warning("No symbols met strict screener criteria. Taking top 10 by Kurtosis.")
symbols_to_test = screen_df.head(10)["symbol"].tolist()
else:
symbols_to_test = qualifying.head(max_coins)["symbol"].tolist()

logger.info(f"Starting simulation across {len(symbols_to_test)} symbols on D1 timeframe.")
all_trades: list[Trade] = []

for sym in symbols_to_test:
df = load_daily_ohlcv(self.storage, sym)
if df is None or len(df) < 50:
continue

df = self.generator.compute_signals(df)
trades = self._simulate_symbol(sym, df)
all_trades.extend(trades)

if not all_trades:
logger.info("Backtest completed: 0 trades triggered.")
return pd.DataFrame()

# Compile trades DataFrame
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
for t in all_trades if t.exit_time is not None
]

df_trades = pd.DataFrame(trades_data).sort_values("exit_time").reset_index(drop=True)

# Save results
out_dir = Path(__file__).parent
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "trades_history.csv"
df_trades.to_csv(out_file, index=False)
logger.info(f"Saved {len(df_trades)} closed trades to {out_file.name}")

self._log_summary_metrics(df_trades)
return df_trades

def _simulate_symbol(self, symbol: str, df: pd.DataFrame) -> list[Trade]:
trades: list[Trade] = []
active_trade: Trade | None = None
current_sl: float = 0.0

for idx, row in df.iterrows():
# If in an active short trade, check stop loss / theory exit
if active_trade is not None and active_trade.direction == "short":
# 1. Check if price hit Initial Stop Loss intraday
if row["high"] >= current_sl:
exit_price = current_sl * (1.0 + self.slippage)
raw_return = (active_trade.entry_price - exit_price) / active_trade.entry_price
fee_impact = 2 * self.taker_fee
net_return = raw_return - fee_impact

active_trade.exit_time = row["timestamp"]
active_trade.exit_price = exit_price
active_trade.return_pct = net_return
active_trade.pnl_usdt = net_return * self.trade_size_usdt
active_trade.exit_reason = "stop_loss_hit"
try:
active_trade.bars_held = (pd.to_datetime(row["timestamp"]) - pd.to_datetime(active_trade.entry_time)).days
except Exception:
active_trade.bars_held = 0
trades.append(active_trade)
active_trade = None
continue

# 2. Check Theory Exits (Markov State Change on daily close):
# Return above short-term average (C > EMA5) OR Bullish absorption CLV >= 0.85
ema5_val = row.get("ema5", 1e9)
clv_val = row.get("clv", 0.5)
if row["close"] > ema5_val or clv_val >= 0.85:
reason = "ema5_reverted" if row["close"] > ema5_val else "clv_reverted"
exit_price = row["close"] * (1.0 + self.slippage)
raw_return = (active_trade.entry_price - exit_price) / active_trade.entry_price
net_return = raw_return - (2 * self.taker_fee)

active_trade.exit_time = row["timestamp"]
active_trade.exit_price = exit_price
active_trade.return_pct = net_return
active_trade.pnl_usdt = net_return * self.trade_size_usdt
active_trade.exit_reason = reason
try:
active_trade.bars_held = (pd.to_datetime(row["timestamp"]) - pd.to_datetime(active_trade.entry_time)).days
except Exception:
active_trade.bars_held = 0
trades.append(active_trade)
active_trade = None
continue
continue

# If no active trade, check entry signal
if active_trade is None and row["signal_short"] == 1:
entry_price = row["close"] * (1.0 - self.slippage)
current_sl = row["initial_sl"]
entry_tr = float(row.get("tr_zscore", 0.0))
entry_clv_val = float(row.get("clv", 0.0))
active_trade = Trade(
symbol=symbol,
direction="short",
entry_time=row["timestamp"],
entry_price=entry_price,
entry_tr_zscore=entry_tr,
entry_clv=entry_clv_val,
initial_sl=current_sl,
)

# Close open trade at end of data
if active_trade is not None:
last_row = df.iloc[-1]
exit_price = last_row["close"]
raw_return = (active_trade.entry_price - exit_price) / active_trade.entry_price - 2 * self.taker_fee
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

