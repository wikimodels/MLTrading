"""
04_features.py — Feature engineering для каждой сделки.

Добавляет контекст N предыдуSCHих сделок:
  - СкользяSCHая статистика bars_in_trade (avg, median, max, min, std)
  - Streak-фичи: сколько подряд сделок <= X свечей
  - Cumsum боковика: суммарных свечей за streak
  - distance_to_previous_big_trend: свечей с окончания последней большой сделки
  - ATR% контекст: средний ATR% за N сделок

Читает:  data/trades.parquet
Пишет:   data/features.parquet

Запуск:
    poetry run python trend_detector_lab/04_features.py
"""
from __future__ import annotations

import sys
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

# Добавляем путь к корню, чтобы импортировать shared
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared.indicators import compute_true_range, compute_tr_zscore

with open(LAB / "config" / "settings.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

DATA_DIR     = LAB / CFG["data"]["output_dir"]
OHLCV_DIR    = DATA_DIR / "ohlcv"
TRADES_PATH  = DATA_DIR / "trades.parquet"
OUTPUT_PATH  = DATA_DIR / "features.parquet"

LOOK_BACK_WINDOWS    = CFG["features"]["look_back_windows"]    # [3,4,5,6,7,8,10]
SHORT_THRESHOLDS     = CFG["features"]["short_thresholds"]     # [4,5,6,8,10,12]
BIG_TREND_THRESHOLDS = CFG["features"]["big_trend_thresholds"] # [15,25,50]

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {level} | {message}", level="INFO")


# ── Feature engineering ───────────────────────────────────────────────────────

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
    """
    Добавляет все feature-колонки для одной монеты.
    df должен быть уже отсортирован по signal_ts.
    """
    bars = df["bars_in_trade"].values
    atr_pct = df["atr_pct"].values
    n = len(df)

    # ── 1. СкользяSCHая статистика за N предыдуSCHих сделок ──────────────────────
    for N in LOOK_BACK_WINDOWS:
        avg = np.full(n, np.nan)
        med = np.full(n, np.nan)
        mx  = np.full(n, np.nan)
        mn  = np.full(n, np.nan)
        std = np.full(n, np.nan)
        for i in range(N, n):
            window = bars[i - N : i]  # N сделок ДО текуSCHей
            avg[i] = window.mean()
            med[i] = np.median(window)
            mx[i]  = window.max()
            mn[i]  = window.min()
            std[i] = window.std(ddof=0)
        df[f"bars_last{N}_avg"]    = avg
        df[f"bars_last{N}_median"] = med
        df[f"bars_last{N}_max"]    = mx
        df[f"bars_last{N}_min"]    = mn
        df[f"bars_last{N}_std"]    = std

    # ── 2. Streak-фичи (сколько подряд <= X) ─────────────────────────────────
    for X in SHORT_THRESHOLDS:
        streak  = np.zeros(n, dtype=int)
        cumsum  = np.zeros(n, dtype=int)
        cur_streak = 0
        cur_cumsum = 0
        for i in range(n):
            if i == 0:
                # Для первой сделки нет предыдуSCHего контекста
                streak[i] = 0
                cumsum[i] = 0
                continue
            prev_bars = bars[i - 1]
            if prev_bars <= X:
                cur_streak += 1
                cur_cumsum += prev_bars
            else:
                cur_streak = 0
                cur_cumsum = 0
            streak[i] = cur_streak
            cumsum[i] = cur_cumsum
        df[f"streak_le{X}"]  = streak
        df[f"cumsum_le{X}"]  = cumsum

    # ── 3. distance_to_previous_big_trend ────────────────────────────────────
    # Для каждого порога B: сколько свечей прошло с момента окончания
    # последней сделки с bars_in_trade > B
    # "прошло свечей" = сумма bars_in_trade всех сделок после большой
    for B in BIG_TREND_THRESHOLDS:
        dist = np.full(n, np.nan)
        bars_since = 0
        found_big = False
        for i in range(n):
            if i == 0:
                dist[i] = np.nan
                continue
            prev_bars = bars[i - 1]
            if prev_bars > B:
                found_big  = True
                bars_since = 0   # сбрасываем счётчик ПОСЛЕ большой сделки
            else:
                if found_big:
                    bars_since += prev_bars
            dist[i] = bars_since if found_big else np.nan
        df[f"bars_since_big{B}"] = dist

    # ── 4. ATR% контекст ─────────────────────────────────────────────────────
    for N in [5, 10]:
        atr_avg = np.full(n, np.nan)
        for i in range(N, n):
            atr_avg[i] = atr_pct[i - N : i].mean()
        df[f"atr_pct_last{N}_avg"] = atr_avg

    # ── 5. Чередование сигналов ───────────────────────────────────────────────
    # bool: последние 4 сигнала чередовались L/S/L/S
    sigs = df["signal"].values
    alternating = np.zeros(n, dtype=bool)
    for i in range(4, n):
        window = sigs[i - 4 : i]
        alt = all(window[j] != window[j + 1] for j in range(3))
        alternating[i] = alt
    df["signal_alternating_4"] = alternating

    # ── 6. Непрерывные фичи из OHLCV (Сжатие волатильности) ────────────────
    ohlcv_path = OHLCV_DIR / f"{symbol.replace('/', '_').replace(':', '_')}_4h.parquet"
    if ohlcv_path.exists():
        ohlcv = pd.read_parquet(ohlcv_path)
        ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], utc=True)
        ohlcv = ohlcv.sort_values("timestamp").reset_index(drop=True)
        
        # 1. Z-Score True Range
        ohlcv["zscore_tr_20"] = compute_tr_zscore(ohlcv, window=20)
        
        # 2. ATR & ATR Percentile 100
        tr = compute_true_range(ohlcv)
        ohlcv["atr_21"] = _wilder_rma(tr, 21)
        # Min-Max нормализация за 100 баров (эквивалент перцентиля)
        atr_100_min = ohlcv["atr_21"].rolling(100).min()
        atr_100_max = ohlcv["atr_21"].rolling(100).max()
        ohlcv["atr_percentile_100"] = (ohlcv["atr_21"] - atr_100_min) / (atr_100_max - atr_100_min + 1e-8)
        
        # 3. Donchian Width 20 & Percentile 100
        hh20 = ohlcv["high"].rolling(20).max()
        ll20 = ohlcv["low"].rolling(20).min()
        ohlcv["donchian_width_20_pct"] = (hh20 - ll20) / ohlcv["close"] * 100
        dw_100_min = ohlcv["donchian_width_20_pct"].rolling(100).min()
        dw_100_max = ohlcv["donchian_width_20_pct"].rolling(100).max()
        ohlcv["donchian_width_percentile_100"] = (ohlcv["donchian_width_20_pct"] - dw_100_min) / (dw_100_max - dw_100_min + 1e-8)
        
        # 4. TTM Squeeze (BB & KC ratio)
        basis = ohlcv["close"].rolling(20).mean()
        dev = ohlcv["close"].rolling(20).std(ddof=0) * 2.0
        bb_width = 2 * dev
        kc_dev = _wilder_rma(tr, 20) * 1.5
        kc_width = 2 * kc_dev
        ohlcv["bb_kc_ratio_20"] = bb_width / kc_width
        
        # 5. ADX 14
        up = ohlcv["high"].diff()
        down = -ohlcv["low"].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        atr_14 = _wilder_rma(tr, 14)
        plus_di = 100 * _wilder_rma(pd.Series(plus_dm), 14) / atr_14
        minus_di = 100 * _wilder_rma(pd.Series(minus_dm), 14) / atr_14
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8) * 100
        ohlcv["adx_14"] = _wilder_rma(dx, 14)
        
        # 6. Lookback агрегации для фазовых переходов
        target_cols = [
            "zscore_tr_20", "atr_percentile_100", 
            "donchian_width_20_pct", "donchian_width_percentile_100",
            "bb_kc_ratio_20", "adx_14"
        ]
        
        # Для cross_up нужно знать минимум за последние N свечей, 
        # а для cross_down - максимум.
        for col in target_cols:
            for L in [3, 5, 8, 13]:
                ohlcv[f"{col}_min_{L}"] = ohlcv[col].rolling(L).min()
                ohlcv[f"{col}_max_{L}"] = ohlcv[col].rolling(L).max()
                
        # Merge by timestamp (signal_ts)
        ohlcv = ohlcv.rename(columns={"timestamp": "signal_ts"})
        join_cols = ["signal_ts"] + target_cols + [f"{c}_{agg}_{L}" for c in target_cols for agg in ["min", "max"] for L in [3, 5, 8, 13]]
        df = df.merge(ohlcv[join_cols], on="signal_ts", how="left")
    else:
        logger.warning(f"OHLCV file not found for {symbol}")

    return df


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not TRADES_PATH.exists():
        logger.error(f"Not found: {TRADES_PATH}. Run 03_trades.py first.")
        sys.exit(1)

    trades = pd.read_parquet(TRADES_PATH)
    trades["signal_ts"] = pd.to_datetime(trades["signal_ts"], utc=True)
    trades["exit_ts"]   = pd.to_datetime(trades["exit_ts"],   utc=True)
    logger.info(f"Loaded {len(trades):,} trades, {trades['coin'].nunique()} coins")

    all_parts = []
    coins = trades["coin"].unique()

    for i, symbol in enumerate(sorted(coins), 1):
        coin_df = trades[trades["coin"] == symbol].sort_values("signal_ts").copy()
        if len(coin_df) < 5:
            logger.warning(f"[{i}] {symbol}: too few trades ({len(coin_df)}), skip")
            continue

        coin_df = add_features(coin_df, symbol)
        all_parts.append(coin_df)
        logger.info(f"[{i}/{len(coins)}] {symbol}: features added ({len(coin_df)} trades)")

    if not all_parts:
        logger.error("No features computed. Exiting.")
        sys.exit(1)

    result = pd.concat(all_parts, ignore_index=True)
    result.to_parquet(OUTPUT_PATH, index=False)

    logger.info(f"\nTotal: {len(result):,} trades with features")
    logger.info(f"Columns: {len(result.columns)}")
    logger.info(f"Saved: {OUTPUT_PATH}")

    # Быстрая сводка по streak фичам
    logger.info("\n=== Baseline распределение bars_in_trade ===")
    b = result["bars_in_trade"]
    logger.info(f"  mean={b.mean():.1f}  median={b.median():.1f}  max={b.max()}  std={b.std():.1f}")
    for threshold in [10, 15, 25]:
        pct = (b > threshold).mean() * 100
        logger.info(f"  P(bars>{threshold}) = {pct:.1f}%")


if __name__ == "__main__":
    main()
