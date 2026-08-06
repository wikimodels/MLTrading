import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

LAB = Path("trend_detector_lab")
OHLCV_DIR = LAB / "data/ohlcv"

def compute_atr(df, period=21):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    alpha = 1.0 / period
    atr = np.full(len(tr), np.nan)
    first_valid = tr.first_valid_index()
    if first_valid is None: return tr
    start_idx = tr.index.get_loc(first_valid)
    if start_idx + period - 1 >= len(tr): return tr
    atr[start_idx + period - 1] = tr.iloc[start_idx : start_idx + period].mean()
    for i in range(start_idx + period, len(tr)):
        atr[i] = alpha * tr.iloc[i] + (1 - alpha) * atr[i - 1]
    return pd.Series(atr, index=tr.index)

def test_donchian_breakout(symbol, ohlcv_path):
    df = pd.read_parquet(ohlcv_path)
    if len(df) < 200: return []
    
    df = df.sort_values("timestamp").reset_index(drop=True)
    df['atr'] = compute_atr(df, 21)
    df['atr_pct'] = df['atr'] / df['close']
    
    # Donchian 20
    hh20 = df["high"].rolling(20).max()
    ll20 = df["low"].rolling(20).min()
    
    width_pct = (hh20 - ll20) / df["close"] * 100
    dw_min = width_pct.rolling(100).min()
    dw_max = width_pct.rolling(100).max()
    df["dw_pct_100"] = (width_pct - dw_min) / (dw_max - dw_min + 1e-8)
    
    # Breakout logic (price crosses previous 20-bar high/low)
    # Richard Dennis style: if high > prev_hh20 -> Long
    prev_hh = hh20.shift(1)
    prev_ll = ll20.shift(1)
    prev_dw = df["dw_pct_100"].shift(1)
    
    # To compare fairly, we need to enter on the open of the next bar AFTER the breakout candle closes.
    # OR we enter intraday (which is harder to model without tick data).
    # Let's use the exact same methodology as Chandelier:
    # "Close crosses above upper Donchian band" -> Signal generated. Enter on next Open.
    
    long_signal = (df["close"] > prev_hh) & (prev_dw <= 0.05)
    short_signal = (df["close"] < prev_ll) & (prev_dw <= 0.05)
    
    # We also want to compare pure Chandelier (from our previous data) vs pure Donchian.
    # Let's just gather the Donchian trades here.
    trades = []
    
    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    
    signal_idx = np.where(long_signal | short_signal)[0]
    
    for i in signal_idx:
        if i + 1 >= len(df): continue
        if df['atr'].iloc[i] <= 0: continue
            
        direction = 1 if long_signal.iloc[i] else -1
        atr = df['atr'].iloc[i]
        entry_price = opens[i + 1]
        
        tp_price = entry_price + direction * 3.0 * atr
        sl_price = entry_price - direction * 1.0 * atr
        
        # Forward window
        w = 60
        end_idx = min(i + 1 + w, len(df))
        fwd_highs = highs[i + 1 : end_idx]
        fwd_lows  = lows[i + 1 : end_idx]
        
        if len(fwd_highs) == 0: continue
        
        if direction == 1:
            tp_hits = fwd_highs >= tp_price
            sl_hits = fwd_lows <= sl_price
        else:
            tp_hits = fwd_lows <= tp_price
            sl_hits = fwd_highs >= sl_price
            
        tp_idx_arr = np.where(tp_hits)[0]
        sl_idx_arr = np.where(sl_hits)[0]
        
        first_tp = tp_idx_arr[0] if len(tp_idx_arr) > 0 else 9999
        first_sl = sl_idx_arr[0] if len(sl_idx_arr) > 0 else 9999
        
        if first_tp == first_sl and first_tp != 9999:
            first_tp = 9999
            
        if first_tp < first_sl:
            exit_reason = "tp"
        elif first_sl < first_tp:
            exit_reason = "sl"
        else:
            exit_reason = "timeout"
            
        comm_atr = (0.15 * 2) / (df['atr_pct'].iloc[i] * 100) # 0.15% taker
        
        r = 3.0 - comm_atr if exit_reason == 'tp' else (-1.0 - comm_atr if exit_reason == 'sl' else 0.0 - comm_atr)
        
        trades.append({
            "coin": symbol,
            "ts": df['timestamp'].iloc[i],
            "year": pd.to_datetime(df['timestamp'].iloc[i]).year,
            "direction": direction,
            "exit_reason": exit_reason,
            "r": r
        })
        
    return trades

def main():
    print("Running Donchian Breakout test...")
    all_trades = []
    
    for file in OHLCV_DIR.glob("*_4h.parquet"):
        symbol = file.stem.split("_")[0]
        trades = test_donchian_breakout(symbol, file)
        all_trades.extend(trades)
        
    df = pd.DataFrame(all_trades)
    if len(df) == 0:
        print("No trades found.")
        return
        
    # Stats
    n = len(df)
    p_tp = (df['exit_reason'] == 'tp').mean()
    ees = df['r'].mean()
    
    print(f"\n--- PURE DONCHIAN BREAKOUT + COMPRESSION (<=5%) ---")
    print(f"Total Trades: {n}")
    print(f"Win Rate P(3ATR): {p_tp:.1%}")
    print(f"EES (Taker 0.15%): {ees:.2f}R")
    
    # By year
    print("\n--- By Year ---")
    for y in sorted(df['year'].unique()):
        y_df = df[df['year'] == y]
        p_tp_y = (y_df['exit_reason'] == 'tp').mean()
        ees_y = y_df['r'].mean()
        print(f"{y}: N={len(y_df):<4} | P(3ATR)={p_tp_y:.1%} | EES={ees_y:.2f}R")
        
if __name__ == "__main__":
    main()
