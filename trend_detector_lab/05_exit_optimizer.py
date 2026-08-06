"""
05_exit_optimizer.py — Подбор оптимального выхода для подтверждённого входа.

Тестирует 3 парадигмы:
1. Reverse Chandelier
2. ATR Trailing Stop (разные мультипликаторы)
3. Fixed TP/SL (разные мультипликаторы)
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
LAB  = Path(__file__).resolve().parent

with open(LAB / "config" / "settings.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

DATA_DIR    = LAB / CFG["data"]["output_dir"]
OHLCV_DIR   = DATA_DIR / "ohlcv"
SIGNALS_PATH = DATA_DIR / "raw_signals.parquet"
FEATURES_PATH = DATA_DIR / "features_eq.parquet"
REPORT_DIR = Path("exit_report")
REPORT_DIR.mkdir(exist_ok=True)

COMM_PCT = 0.30 # 0.15% Taker * 2 (Round trip)

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {level} | {message}", level="INFO")

def load_ohlcv(symbol: str) -> pd.DataFrame | None:
    sym_safe = symbol.replace("/", "_").replace(":", "_")
    path = OHLCV_DIR / f"{sym_safe}_4h.parquet"
    if not path.exists(): return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)

def simulate_exits(trades: pd.DataFrame, ohlcv: pd.DataFrame, all_signals: pd.DataFrame, symbol: str):
    ohlcv = ohlcv.reset_index(drop=True)
    ts_to_idx = {ts: idx for idx, ts in enumerate(ohlcv["timestamp"])}
    
    sym_signals = all_signals[all_signals["coin"] == symbol].sort_values("timestamp").reset_index(drop=True)
    
    highs = ohlcv["high"].values
    lows = ohlcv["low"].values
    opens = ohlcv["open"].values
    closes = ohlcv["close"].values
    timestamps = ohlcv["timestamp"].values
    
    results = []
    
    for _, trade in trades.iterrows():
        sig_ts = trade["signal_ts"]
        if sig_ts not in ts_to_idx: continue
        
        idx = ts_to_idx[sig_ts]
        if idx + 1 >= len(ohlcv): continue
        
        entry_price = opens[idx + 1]
        atr = trade["atr"]
        atr_pct = trade["atr_pct"]
        direction = 1 if trade["signal"] == "long" else -1
        
        trade_res = {
            "coin": symbol,
            "signal_ts": sig_ts,
            "direction": direction,
            "year": sig_ts.year
        }
        
        # 1. Reverse Chandelier
        future_sigs = sym_signals[(sym_signals["timestamp"] > sig_ts) & (sym_signals["signal"] != trade["signal"])]
        if not future_sigs.empty:
            rev_ts = future_sigs.iloc[0]["timestamp"]
            if rev_ts in ts_to_idx:
                rev_idx = ts_to_idx[rev_ts]
                if rev_idx + 1 < len(ohlcv):
                    rev_price = opens[rev_idx + 1]
                    raw_r = ((rev_price - entry_price) / entry_price * direction * 100 - COMM_PCT) / atr_pct
                    trade_res["rev_chandelier_r"] = raw_r
                    trade_res["rev_chandelier_bars"] = rev_idx + 1 - (idx + 1)
        
        if "rev_chandelier_r" not in trade_res:
            trade_res["rev_chandelier_r"] = -COMM_PCT / atr_pct
            trade_res["rev_chandelier_bars"] = 0
            
        # 2. ATR Trailing Stops [1.5, 2.0, 2.5, 3.0, 4.0]
        # Simulate bar by bar
        fwd_highs = highs[idx+1:]
        fwd_lows = lows[idx+1:]
        fwd_opens = opens[idx+1:]
        
        trailing_mults = [1.5, 2.0, 2.5, 3.0, 4.0]
        trailing_states = {m: {"peak": entry_price, "active": True, "exit_price": 0, "bars": 0} for m in trailing_mults}
        
        # 3. Fixed TP/SL
        tp_mults = [3, 4, 5, 10]
        sl_mults = [1, 1.5, 2]
        fixed_states = {}
        for tp in tp_mults:
            for sl in sl_mults:
                fixed_states[(tp, sl)] = {"active": True, "exit_price": 0, "bars": 0}
                
        # Forward simulation
        for i in range(len(fwd_highs)):
            h, l, o = fwd_highs[i], fwd_lows[i], fwd_opens[i]
            
            # Trailing stops
            for m, state in trailing_states.items():
                if not state["active"]: continue
                
                if direction == 1:
                    stop_price = state["peak"] - m * atr
                    if l <= stop_price:
                        state["exit_price"] = min(o, stop_price)
                        state["active"] = False
                        state["bars"] = i + 1
                    else:
                        state["peak"] = max(state["peak"], h)
                else:
                    stop_price = state["peak"] + m * atr
                    if h >= stop_price:
                        state["exit_price"] = max(o, stop_price)
                        state["active"] = False
                        state["bars"] = i + 1
                    else:
                        state["peak"] = min(state["peak"], l)
                        
            # Fixed TP/SL
            for (tp, sl), state in fixed_states.items():
                if not state["active"]: continue
                
                if direction == 1:
                    tp_price = entry_price + tp * atr
                    sl_price = entry_price - sl * atr
                    
                    if l <= sl_price:
                        state["exit_price"] = min(o, sl_price)
                        state["active"] = False
                        state["bars"] = i + 1
                    elif h >= tp_price:
                        state["exit_price"] = max(o, tp_price)
                        state["active"] = False
                        state["bars"] = i + 1
                else:
                    tp_price = entry_price - tp * atr
                    sl_price = entry_price + sl * atr
                    
                    if h >= sl_price:
                        state["exit_price"] = max(o, sl_price)
                        state["active"] = False
                        state["bars"] = i + 1
                    elif l <= tp_price:
                        state["exit_price"] = min(o, tp_price)
                        state["active"] = False
                        state["bars"] = i + 1
                        
            # Early break if all inactive
            if all(not s["active"] for s in trailing_states.values()) and all(not s["active"] for s in fixed_states.values()):
                break
                
        # Finalize inactive states (end of data)
        last_price = closes[-1]
        for m, state in trailing_states.items():
            if state["active"]:
                state["exit_price"] = last_price
                state["bars"] = len(fwd_highs)
            
            r = ((state["exit_price"] - entry_price) / entry_price * direction * 100 - COMM_PCT) / atr_pct
            trade_res[f"trail_{m}_r"] = r
            
        for (tp, sl), state in fixed_states.items():
            if state["active"]:
                state["exit_price"] = last_price
                state["bars"] = len(fwd_highs)
                
            r = ((state["exit_price"] - entry_price) / entry_price * direction * 100 - COMM_PCT) / atr_pct
            trade_res[f"fixed_{tp}_{sl}_r"] = r
            
        results.append(trade_res)
        
    return pd.DataFrame(results)

def main():
    logger.info("Loading features and raw signals...")
    df = pd.read_parquet(FEATURES_PATH)
    all_signals = pd.read_parquet(SIGNALS_PATH)
    
    all_signals["timestamp"] = pd.to_datetime(all_signals["timestamp"], utc=True)
    df["signal_ts"] = pd.to_datetime(df["signal_ts"], utc=True)
    
    # FILTER THE PROVEN EDGE ONLY
    mask = (df["donchian_width_percentile_100"] <= 0.05)
    valid_trades = df[mask].copy()
    
    top_coins_path = Path("trend_detector_lab/top_20_coins.txt")
    if top_coins_path.exists():
        top_coins = top_coins_path.read_text().splitlines()
        valid_trades = valid_trades[valid_trades["coin"].str.split("/").str[0].isin(top_coins)].copy()
        logger.info(f"Filtered to {len(top_coins)} top trending coins.")

    
    logger.info(f"Total verified entries: {len(valid_trades)}")
    
    coins = valid_trades["coin"].unique()
    all_res = []
    
    for i, symbol in enumerate(sorted(coins), 1):
        ohlcv = load_ohlcv(symbol)
        if ohlcv is None: continue
        
        coin_trades = valid_trades[valid_trades["coin"] == symbol]
        res = simulate_exits(coin_trades, ohlcv, all_signals, symbol)
        all_res.append(res)
        
        logger.info(f"[{i}/{len(coins)}] {symbol}: simulated {len(res)} exits")
        
    final_df = pd.concat(all_res, ignore_index=True)
    
    # Evaluation
    train = final_df[final_df["year"] <= 2024]
    test = final_df[final_df["year"] >= 2025]
    
    logger.info(f"Train set: {len(train)} trades. Test set: {len(test)} trades.")
    
    # Calculate performance
    strategies = [c for c in final_df.columns if c.endswith("_r")]
    
    perf = []
    for s in strategies:
        tr_mean = train[s].mean()
        tr_win = (train[s] > 0).mean()
        
        te_mean = test[s].mean()
        te_win = (test[s] > 0).mean()
        
        perf.append({
            "Strategy": s.replace("_r", ""),
            "Train_EES": tr_mean,
            "Train_Win%": tr_win * 100,
            "Test_EES": te_mean,
            "Test_Win%": te_win * 100,
            "Total_EES": final_df[s].mean()
        })
        
    perf_df = pd.DataFrame(perf).sort_values("Test_EES", ascending=False).reset_index(drop=True)
    
    md = "# Exit Optimization Results\n\n"
    md += f"**Validated Entries:** {len(final_df)}\n"
    md += f"**Train (23-24):** {len(train)} trades | **Test (25-26):** {len(test)} trades\n"
    md += f"**Commission Setup:** 0.30% Round-trip (Taker/Taker)\n\n"
    
    md += "### Strategy Leaderboard (Sorted by Test OOS Expected R)\n\n"
    md += perf_df.to_markdown(index=False, floatfmt=".2f")
    
    with open(REPORT_DIR / "optimization.md", "w", encoding="utf-8") as f:
        f.write(md)
        
    logger.info(f"\n{perf_df.head(10).to_string(index=False)}")
    logger.info(f"\nOptimization complete. Report saved to {REPORT_DIR / 'optimization.md'}")

if __name__ == "__main__":
    main()
