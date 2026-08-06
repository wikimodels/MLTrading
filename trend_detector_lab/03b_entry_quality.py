"""
03b_entry_quality.py — Оценка чистого качества входа (Entry Quality).

Для каждого сигнала из raw_signals.parquet симулирует вход на Open следующей свечи
и проверяет достижение статических уровней (например, TP = +3 ATR, SL = -1 ATR)
на горизонтах 30, 60 и 90 баров.

Выход: data/entry_quality.parquet
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
OUTPUT_PATH  = DATA_DIR / "entry_quality.parquet"

TP_MULT = 3.0
SL_MULT = 1.0
FORWARD_WINDOWS = [30, 60, 90]

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {level} | {message}", level="INFO")

def _load_ohlcv(symbol: str) -> pd.DataFrame | None:
    sym_safe = symbol.replace("/", "_").replace(":", "_")
    path = OHLCV_DIR / f"{sym_safe}_4h.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)

def build_entry_quality(signals: pd.DataFrame, ohlcv: pd.DataFrame, symbol: str) -> pd.DataFrame:
    ohlcv = ohlcv.copy().reset_index(drop=True)
    ts_to_idx = {ts: idx for idx, ts in enumerate(ohlcv["timestamp"])}

    records = []
    sym_sigs = signals[signals["coin"] == symbol].reset_index(drop=True)

    highs = ohlcv["high"].values
    lows = ohlcv["low"].values
    opens = ohlcv["open"].values

    for i in range(len(sym_sigs)):
        sig = sym_sigs.iloc[i]
        signal_ts = sig["timestamp"]
        direction = 1 if sig["signal"] == "long" else -1
        atr = float(sig["atr"])

        if atr <= 0 or signal_ts not in ts_to_idx:
            continue
            
        sig_idx = ts_to_idx[signal_ts]
        if sig_idx + 1 >= len(ohlcv):
            continue
            
        entry_price = float(opens[sig_idx + 1])
        
        # Calculate TP and SL price levels
        if direction == 1:
            tp_price = entry_price + TP_MULT * atr
            sl_price = entry_price - SL_MULT * atr
        else:
            tp_price = entry_price - TP_MULT * atr
            sl_price = entry_price + SL_MULT * atr

        max_window = max(FORWARD_WINDOWS)
        end_idx = min(sig_idx + 1 + max_window, len(ohlcv))
        
        # Extract forward window prices
        fwd_highs = highs[sig_idx + 1 : end_idx]
        fwd_lows  = lows[sig_idx + 1 : end_idx]
        actual_len = len(fwd_highs)
        
        if actual_len == 0:
            continue
            
        # Calculate MFE / MAE arrays
        if direction == 1:
            favorable_arr = (fwd_highs - entry_price) / atr
            adverse_arr   = (entry_price - fwd_lows) / atr
            
            # Hit masks
            tp_hits = fwd_highs >= tp_price
            sl_hits = fwd_lows <= sl_price
        else:
            favorable_arr = (entry_price - fwd_lows) / atr
            adverse_arr   = (fwd_highs - entry_price) / atr
            
            # Hit masks
            tp_hits = fwd_lows <= tp_price
            sl_hits = fwd_highs >= sl_price

        # Find first hit indices
        tp_idx_arr = np.where(tp_hits)[0]
        sl_idx_arr = np.where(sl_hits)[0]
        
        first_tp = tp_idx_arr[0] if len(tp_idx_arr) > 0 else 9999
        first_sl = sl_idx_arr[0] if len(sl_idx_arr) > 0 else 9999
        
        # If both hit on the exact same bar, assume SL hit first (conservative)
        if first_tp == first_sl and first_tp != 9999:
            first_tp = 9999
            
        record = {
            "coin": symbol,
            "tf": sig["tf"],
            "signal_ts": signal_ts,
            "signal": sig["signal"],
            "entry_price": entry_price,
            "atr": atr,
            "atr_pct": sig["atr_pct"]
        }
        
        # Evaluate for each window
        for w in FORWARD_WINDOWS:
            # Mask hits within window
            w_tp = first_tp if first_tp < min(w, actual_len) else 9999
            w_sl = first_sl if first_sl < min(w, actual_len) else 9999
            
            if w_tp < w_sl:
                exit_reason = "tp"
            elif w_sl < w_tp:
                exit_reason = "sl"
            elif w_sl == 9999 and w_tp == 9999:
                exit_reason = "timeout"
            else:
                exit_reason = "sl" # safety fallback
                
            record[f"exit_reason_{w}"] = exit_reason
            
            # MFE and MAE for the window
            w_len = min(w, actual_len)
            record[f"mfe_{w}"] = float(np.max(favorable_arr[:w_len]))
            record[f"mae_{w}"] = float(np.max(adverse_arr[:w_len]))
            
        records.append(record)

    return pd.DataFrame(records)

def main() -> None:
    if not SIGNALS_PATH.exists():
        logger.error(f"Not found: {SIGNALS_PATH}")
        sys.exit(1)

    signals = pd.read_parquet(SIGNALS_PATH)
    signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True)
    logger.info(f"Loaded {len(signals):,} raw signals, {signals['coin'].nunique()} coins")

    all_trades = []
    coins = signals["coin"].unique()

    for i, symbol in enumerate(sorted(coins), 1):
        ohlcv = _load_ohlcv(symbol)
        if ohlcv is None or ohlcv.empty:
            continue
        
        trades = build_entry_quality(signals, ohlcv, symbol)
        if trades.empty:
            continue

        all_trades.append(trades)
        logger.info(f"[{i}/{len(coins)}] {symbol}: {len(trades)} entry events")

    result = pd.concat(all_trades, ignore_index=True)
    result.to_parquet(OUTPUT_PATH, index=False)
    
    logger.info(f"\nTotal entry events: {len(result):,}")
    
    # Baseline stats
    w = 60
    tp_rate = (result[f"exit_reason_{w}"] == "tp").mean()
    sl_rate = (result[f"exit_reason_{w}"] == "sl").mean()
    to_rate = (result[f"exit_reason_{w}"] == "timeout").mean()
    
    logger.info(f"Baseline (W={w}): TP={tp_rate:.1%} | SL={sl_rate:.1%} | TO={to_rate:.1%}")
    logger.info(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
