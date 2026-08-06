"""
04b_features_eq.py — Добавление фичей физического сжатия к таблице entry_quality.parquet.

Выход: data/features_eq.parquet
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
LAB  = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared.indicators import compute_true_range, compute_tr_zscore

with open(LAB / "config" / "settings.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

DATA_DIR     = LAB / CFG["data"]["output_dir"]
OHLCV_DIR    = DATA_DIR / "ohlcv"
TRADES_PATH  = DATA_DIR / "entry_quality.parquet"
OUTPUT_PATH  = DATA_DIR / "features_eq.parquet"

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {level} | {message}", level="INFO")

def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
    alpha = 1.0 / period
    result = np.full(len(series), np.nan)
    first_valid = series.first_valid_index()
    if first_valid is None: return pd.Series(result, index=series.index)
    start_idx = series.index.get_loc(first_valid)
    if start_idx + period - 1 >= len(series): return pd.Series(result, index=series.index)
    result[start_idx + period - 1] = series.iloc[start_idx : start_idx + period].mean()
    for i in range(start_idx + period, len(series)):
        result[i] = alpha * series.iloc[i] + (1 - alpha) * result[i - 1]
    return pd.Series(result, index=series.index)

def add_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    ohlcv_path = OHLCV_DIR / f"{symbol.replace('/', '_').replace(':', '_')}_4h.parquet"
    if ohlcv_path.exists():
        ohlcv = pd.read_parquet(ohlcv_path)
        ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], utc=True)
        ohlcv = ohlcv.sort_values("timestamp").reset_index(drop=True)
        
        ohlcv["zscore_tr_20"] = compute_tr_zscore(ohlcv, window=20)
        
        tr = compute_true_range(ohlcv)
        ohlcv["atr_21"] = _wilder_rma(tr, 21)
        atr_100_min = ohlcv["atr_21"].rolling(100).min()
        atr_100_max = ohlcv["atr_21"].rolling(100).max()
        ohlcv["atr_percentile_100"] = (ohlcv["atr_21"] - atr_100_min) / (atr_100_max - atr_100_min + 1e-8)
        
        hh20 = ohlcv["high"].rolling(20).max()
        ll20 = ohlcv["low"].rolling(20).min()
        ohlcv["donchian_width_20_pct"] = (hh20 - ll20) / ohlcv["close"] * 100
        dw_100_min = ohlcv["donchian_width_20_pct"].rolling(100).min()
        dw_100_max = ohlcv["donchian_width_20_pct"].rolling(100).max()
        ohlcv["donchian_width_percentile_100"] = (ohlcv["donchian_width_20_pct"] - dw_100_min) / (dw_100_max - dw_100_min + 1e-8)
        
        basis = ohlcv["close"].rolling(20).mean()
        dev = ohlcv["close"].rolling(20).std(ddof=0) * 2.0
        bb_width = 2 * dev
        kc_dev = _wilder_rma(tr, 20) * 1.5
        kc_width = 2 * kc_dev
        ohlcv["bb_kc_ratio_20"] = bb_width / kc_width
        
        up = ohlcv["high"].diff()
        down = -ohlcv["low"].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        atr_14 = _wilder_rma(tr, 14)
        plus_di = 100 * _wilder_rma(pd.Series(plus_dm), 14) / atr_14
        minus_di = 100 * _wilder_rma(pd.Series(minus_dm), 14) / atr_14
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8) * 100
        ohlcv["adx_14"] = _wilder_rma(dx, 14)
        
        target_cols = [
            "zscore_tr_20", "atr_percentile_100", 
            "donchian_width_20_pct", "donchian_width_percentile_100",
            "bb_kc_ratio_20", "adx_14"
        ]
        
        for col in target_cols:
            for L in [3, 5, 8, 13]:
                ohlcv[f"{col}_min_{L}"] = ohlcv[col].rolling(L).min()
                ohlcv[f"{col}_max_{L}"] = ohlcv[col].rolling(L).max()
                
        ohlcv = ohlcv.rename(columns={"timestamp": "signal_ts"})
        join_cols = ["signal_ts"] + target_cols + [f"{c}_{agg}_{L}" for c in target_cols for agg in ["min", "max"] for L in [3, 5, 8, 13]]
        df = df.merge(ohlcv[join_cols], on="signal_ts", how="left")
    else:
        logger.warning(f"OHLCV not found for {symbol}")

    return df

def main() -> None:
    if not TRADES_PATH.exists():
        logger.error(f"Not found: {TRADES_PATH}")
        sys.exit(1)

    trades = pd.read_parquet(TRADES_PATH)
    trades["signal_ts"] = pd.to_datetime(trades["signal_ts"], utc=True)
    logger.info(f"Loaded {len(trades):,} trades, {trades['coin'].nunique()} coins")

    all_parts = []
    coins = trades["coin"].unique()

    for i, symbol in enumerate(sorted(coins), 1):
        coin_df = trades[trades["coin"] == symbol].copy().reset_index(drop=True)
        if coin_df.empty:
            continue
        
        coin_df = add_features(coin_df, symbol)
        all_parts.append(coin_df)
        logger.info(f"[{i}/{len(coins)}] {symbol}: features added ({len(coin_df)} events)")

    result = pd.concat(all_parts, ignore_index=True)
    result.to_parquet(OUTPUT_PATH, index=False)
    
    logger.info(f"\nTotal entry events with features: {len(result):,}")
    logger.info(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
