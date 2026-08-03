import re
with open("fat_tails_bot/backtest/engine.py", "r", encoding="utf-8") as f:
    code = f.read()

new_run_backtest = """
    def run_backtest(self, max_concurrent_positions: int = 100) -> pd.DataFrame:
        logger.info("Starting Bot Farm time-series backtest...")
        from fat_tails_bot.screener.screener import FatTailsScreener, load_daily_ohlcv
        import numpy as np
        
        screen_df = FatTailsScreener().screen()
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
            
        all_dates = sorted(set().union(*[set(df.index) for df in data.values()]))
        
        all_trades = []
        open_trades = {}
        
        logger.info(f"Simulating time-series across {len(all_dates)} days...")
        for current_date in all_dates:
            # 1. Process exits
            for sym in list(open_trades.keys()):
                df = data[sym]
                if current_date not in df.index:
                    continue
                row = df.loc[current_date]
                trade = open_trades[sym]
                
                # Check stop loss hit
                if row["high"] >= trade.initial_sl:
                    exit_price = trade.initial_sl * (1.0 + self.slippage)
                    raw_return = (trade.entry_price - exit_price) / trade.entry_price
                    net_return = raw_return - (2 * self.taker_fee)
                    
                    trade.exit_time = str(current_date)
                    trade.exit_price = exit_price
                    trade.return_pct = net_return
                    trade.pnl_usdt = net_return * self.trade_size_usdt
                    trade.exit_reason = "stop_loss_hit"
                    trade.bars_held = (current_date - pd.to_datetime(trade.entry_time)).days
                    all_trades.append(trade)
                    del open_trades[sym]
                    continue
                    
                # Check theory exit
                ema5_val = row.get("ema5", 1e9)
                clv_val = row.get("clv", 0.5)
                if row["close"] > ema5_val or clv_val >= 0.85:
                    reason = "ema5_reverted" if row["close"] > ema5_val else "clv_reverted"
                    exit_price = row["close"] * (1.0 + self.slippage)
                    raw_return = (trade.entry_price - exit_price) / trade.entry_price
                    net_return = raw_return - (2 * self.taker_fee)
                    
                    trade.exit_time = str(current_date)
                    trade.exit_price = exit_price
                    trade.return_pct = net_return
                    trade.pnl_usdt = net_return * self.trade_size_usdt
                    trade.exit_reason = reason
                    trade.bars_held = (current_date - pd.to_datetime(trade.entry_time)).days
                    all_trades.append(trade)
                    del open_trades[sym]
                    continue
                    
            # 2. Process new entries
            for sym in symbols_to_test:
                if sym in open_trades:
                    continue
                if len(open_trades) >= max_concurrent_positions:
                    break
                    
                df = data.get(sym)
                if df is None or current_date not in df.index:
                    continue
                row = df.loc[current_date]
                
                if row.get("entry_signal", False):
                    entry_price = row["close"] * (1 - self.slippage)
                    atr = row.get("atr_14", np.nan)
                    if pd.isna(atr):
                        continue
                        
                    current_sl = entry_price + (2.0 * atr)
                    
                    trade = Trade(
                        symbol=sym,
                        direction="short",
                        entry_time=str(current_date),
                        entry_price=entry_price,
                        initial_sl=current_sl,
                        entry_tr_zscore=row.get("tr_zscore", 0.0),
                        entry_clv=row.get("clv", 0.0)
                    )
                    open_trades[sym] = trade
                    
        # Force close remaining
        if open_trades:
            last_date = all_dates[-1]
            for sym, trade in open_trades.items():
                df = data[sym]
                last_row = df.iloc[-1]
                exit_price = last_row["close"] * (1.0 + self.slippage)
                raw_return = (trade.entry_price - exit_price) / trade.entry_price
                net_return = raw_return - (2 * self.taker_fee)
                
                trade.exit_time = str(last_row.name)
                trade.exit_price = exit_price
                trade.return_pct = net_return
                trade.pnl_usdt = net_return * self.trade_size_usdt
                trade.exit_reason = "end_of_data"
                trade.bars_held = (last_row.name - pd.to_datetime(trade.entry_time)).days
                all_trades.append(trade)
                
        if not all_trades:
            logger.info("Backtest completed: 0 trades triggered.")
            return pd.DataFrame()
            
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
        
        from pathlib import Path
        out_dir = Path(__file__).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "trades_history.csv"
        df_trades.to_csv(out_file, index=False)
        logger.info(f"Saved {len(df_trades)} closed trades to {out_file.name}")
        
        self._log_summary_metrics(df_trades)
        return df_trades
"""

# Replace the run_backtest and everything below it
code = re.sub(r'    def run_backtest\(self.*', new_run_backtest, code, flags=re.DOTALL)

with open("fat_tails_bot/backtest/engine.py", "w", encoding="utf-8") as f:
    f.write(code)
