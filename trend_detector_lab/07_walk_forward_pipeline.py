"""
07_walk_forward_pipeline.py — Adaptive Walk-Forward Engine
Dynamic Universe Selection (Hurst + Kurtosis) + Dynamic Trailing Stop Optimization.
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from datetime import timedelta

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.indicators import get_hurst_exponent
from scipy.stats import kurtosis

DATA_DIR = Path("trend_detector_lab/data")
OHLCV_DIR = DATA_DIR / "ohlcv"
FEATURES_PATH = DATA_DIR / "features_eq.parquet"

COMM_PCT = 0.30

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {level} | {message}", level="INFO")

def load_all_ohlcv():
    ohlcv_dict = {}
    for file in OHLCV_DIR.glob("*_4h.parquet"):
        symbol = file.stem.split("_")[0]
        df = pd.read_parquet(file)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        ohlcv_dict[symbol] = df
    return ohlcv_dict

def get_top_coins(ohlcv_dict, train_start, train_end):
    results = []
    for symbol, df in ohlcv_dict.items():
        mask = (df["timestamp"] >= train_start) & (df["timestamp"] < train_end)
        train_df = df[mask].copy()
        
        if len(train_df) < 200:
            continue
            
        train_df.set_index("timestamp", inplace=True)
        d1 = train_df.resample("1D").agg({"close": "last"}).dropna()
        if len(d1) < 60:
            continue
            
        h = get_hurst_exponent(d1["close"].values, min_lag=10, max_lag=60)
        
        log_ret = np.diff(np.log(d1["close"].values))
        kurt = kurtosis(log_ret, fisher=True, bias=True)
        
        results.append({"coin": symbol, "hurst": h, "kurtosis": kurt})
        
    if not results:
        return []
        
    res_df = pd.DataFrame(results)
    res_df["hurst_rank"] = res_df["hurst"].rank(ascending=False)
    res_df["kurt_rank"] = res_df["kurtosis"].rank(ascending=False)
    res_df["score"] = res_df["hurst_rank"] + res_df["kurt_rank"]
    
    top_20 = res_df.sort_values("score").head(20)["coin"].tolist()
    return top_20

def simulate_trades(trades, ohlcv_dict, mults_to_test):
    results = []
    for symbol in trades["coin_base"].unique():
        coin_trades = trades[trades["coin_base"] == symbol]
        if symbol not in ohlcv_dict: continue
        
        ohlcv = ohlcv_dict[symbol]
        ts_to_idx = {ts: idx for idx, ts in enumerate(ohlcv["timestamp"])}
        
        highs = ohlcv["high"].values
        lows = ohlcv["low"].values
        opens = ohlcv["open"].values
        closes = ohlcv["close"].values
        
        for _, trade in coin_trades.iterrows():
            sig_ts = trade["signal_ts"]
            if sig_ts not in ts_to_idx: continue
            
            idx = ts_to_idx[sig_ts]
            if idx + 1 >= len(ohlcv): continue
            
            entry_price = opens[idx + 1]
            atr = trade["atr"]
            atr_pct = trade["atr_pct"]
            direction = 1 if trade["signal"] == "long" else -1
            
            fwd_highs = highs[idx+1:]
            fwd_lows = lows[idx+1:]
            fwd_opens = opens[idx+1:]
            
            trade_res = {"signal_ts": sig_ts, "coin": symbol, "direction": direction}
            
            trailing_states = {m: {"peak": entry_price, "active": True, "exit_price": 0} for m in mults_to_test}
            
            for i in range(len(fwd_highs)):
                h, l, o = fwd_highs[i], fwd_lows[i], fwd_opens[i]
                for m, state in trailing_states.items():
                    if not state["active"]: continue
                    
                    if direction == 1:
                        stop_price = state["peak"] - m * atr
                        if l <= stop_price:
                            state["exit_price"] = min(o, stop_price)
                            state["active"] = False
                        else:
                            state["peak"] = max(state["peak"], h)
                    else:
                        stop_price = state["peak"] + m * atr
                        if h >= stop_price:
                            state["exit_price"] = max(o, stop_price)
                            state["active"] = False
                        else:
                            state["peak"] = min(state["peak"], l)
                            
                if all(not s["active"] for s in trailing_states.values()):
                    break
                    
            last_price = closes[-1]
            for m, state in trailing_states.items():
                if state["active"]: state["exit_price"] = last_price
                
                r = ((state["exit_price"] - entry_price) / entry_price * direction * 100 - COMM_PCT) / atr_pct
                trade_res[f"trail_{m}"] = r
                
            results.append(trade_res)
            
    return pd.DataFrame(results) if results else pd.DataFrame()


def main():
    logger.info("Loading Data...")
    ohlcv_dict = load_all_ohlcv()
    
    df = pd.read_parquet(FEATURES_PATH)
    df["signal_ts"] = pd.to_datetime(df["signal_ts"], utc=True)
    df["coin_base"] = df["coin"].str.split("/").str[0]
    
    # Base filter
    valid_trades = df[df["donchian_width_percentile_100"] <= 0.05].copy()
    
    start_dt = pd.to_datetime("2023-01-01", utc=True)
    end_dt = pd.to_datetime("2026-07-01", utc=True)
    months = pd.date_range(start=start_dt, end=end_dt, freq='MS')
    
    # Grid of trailing ATRs to test dynamically
    atr_mults = [1.5, 2.0, 2.5, 3.0, 4.0]
    
    all_test_trades = []
    
    # We need 6 months of data to start trading
    logger.info(f"Starting Walk-Forward. Total months: {len(months)}")
    
    last_best_mult = 2.5 # Default fallback
    
    for i in range(6, len(months) - 1):
        train_start = months[i - 6]
        train_end = months[i]
        test_end = months[i + 1]
        
        logger.info(f"--- WFA Step: Train [{train_start.date()} to {train_end.date()}] | Test [{train_end.date()} to {test_end.date()}] ---")
        
        # 1. Screen Universe
        top_20 = get_top_coins(ohlcv_dict, train_start, train_end)
        if not top_20:
            logger.warning("Not enough data to screen coins.")
            continue
            
        # 2. Train Exits
        train_mask = (valid_trades["signal_ts"] >= train_start) & (valid_trades["signal_ts"] < train_end)
        train_pool = valid_trades[train_mask & valid_trades["coin_base"].isin(top_20)]
        
        best_mult = last_best_mult
        
        if len(train_pool) >= 5:
            train_res = simulate_trades(train_pool, ohlcv_dict, atr_mults)
            if not train_res.empty:
                means = {m: train_res[f"trail_{m}"].mean() for m in atr_mults}
                best_mult = max(means, key=means.get)
                last_best_mult = best_mult
                logger.info(f"Train Trades: {len(train_pool)} | Best ATR Trailing: {best_mult} (Expected R: {means[best_mult]:.2f})")
            else:
                logger.info(f"Train Trades: 0 | Using fallback ATR: {best_mult}")
        else:
            logger.info(f"Train Trades: {len(train_pool)} | Too few to optimize, using fallback ATR: {best_mult}")
            
        # 3. Test
        test_mask = (valid_trades["signal_ts"] >= train_end) & (valid_trades["signal_ts"] < test_end)
        test_pool = valid_trades[test_mask & valid_trades["coin_base"].isin(top_20)]
        
        if not test_pool.empty:
            test_res = simulate_trades(test_pool, ohlcv_dict, [best_mult])
            if not test_res.empty:
                # Rename the chosen column to 'realized_r'
                test_res["realized_r"] = test_res[f"trail_{best_mult}"]
                test_res["selected_mult"] = best_mult
                all_test_trades.append(test_res)
                
                win_pct = (test_res['realized_r'] > 0).mean() * 100
                logger.info(f"Test Trades: {len(test_res)} | Realized EES: {test_res['realized_r'].mean():.2f} R | WinRate: {win_pct:.1f}%")
        else:
            logger.info(f"Test Trades: 0 | No setups found for Top 20 coins.")
            
    if all_test_trades:
        final_df = pd.concat(all_test_trades, ignore_index=True)
        total_ees = final_df["realized_r"].mean()
        win_rate = (final_df["realized_r"] > 0).mean() * 100
        
        logger.info("\n" + "="*50)
        logger.info("FINAL ADAPTIVE WALK-FORWARD RESULTS")
        logger.info("="*50)
        logger.info(f"Total OOS Trades: {len(final_df)}")
        logger.info(f"Total OOS Expected R: {total_ees:.3f} R")
        logger.info(f"Total OOS Win Rate:   {win_rate:.1f}%")
        logger.info("="*50)
        
        # Save report
        report_path = Path("trend_detector_lab/exit_report/wfa_results.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Adaptive Walk-Forward Results\n\n")
            f.write("- **Lookback (Train):** 6 months\n")
            f.write("- **Step (Test):** 1 month\n")
            f.write("- **Universe Selection:** Dynamic Top 20 by Hurst & Kurtosis\n")
            f.write("- **Parameter Selection:** Dynamic ATR Trailing (1.5 - 4.0)\n\n")
            f.write(f"### Performance\n")
            f.write(f"- **Total OOS Trades:** {len(final_df)}\n")
            f.write(f"- **Total OOS Expected R:** `{total_ees:.3f} R`\n")
            f.write(f"- **Total OOS Win Rate:** `{win_rate:.1f}%`\n")
            
        logger.info(f"Saved report to {report_path}")

if __name__ == "__main__":
    main()
