"""
02_chandelier.py — Вычисление Chandelier Exit (Wilder RMA, ATR-21, mult 2.5).

Читает data/ohlcv/<symbol>_4h.parquet (из 01_download.py).
Для каждой монеты вычисляет CE и записывает RAW SIGNALS (переворот = 1 строка).
Сохраняет: data/raw_signals.parquet

Запуск:
    poetry run python trend_detector_lab/02_chandelier.py
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

OHLCV_DIR   = LAB / CFG["data"]["output_dir"] / "ohlcv"
OUTPUT_DIR  = LAB / CFG["data"]["output_dir"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ATR_PERIOD  = CFG["chandelier"]["atr_period"]     # 21
MULTIPLIER  = CFG["chandelier"]["multiplier"]      # 2.5
TIMEFRAME   = CFG["chandelier"]["timeframe"]       # "4h"

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {level} | {message}", level="INFO")


# ── Chandelier Exit — точная копия Pine Script v6 ─────────────────────────────
#
# Pine Script source (Alex Orekhov, GPL-3.0):
#   atr = mult * ta.atr(length)
#   longStop  = highest(close, length) - atr          # useClose=true (default)
#   longStop  := close[1] > longStopPrev  ? max(longStop, longStopPrev)  : longStop
#   shortStop = lowest(close, length) + atr
#   shortStop := close[1] < shortStopPrev ? min(shortStop, shortStopPrev) : shortStop
#   dir := close > shortStopPrev ? 1 : close < longStopPrev ? -1 : dir

def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothed moving average (RMA). Идентично ta.atr() в TradingView."""
    alpha = 1.0 / period
    result = np.full(len(series), np.nan)
    first_valid = series.first_valid_index()
    if first_valid is None:
        return pd.Series(result, index=series.index)
    start_idx = series.index.get_loc(first_valid)
    if start_idx + period - 1 >= len(series):
        return pd.Series(result, index=series.index)
    result[start_idx + period - 1] = series.iloc[start_idx : start_idx + period].mean()
    for i in range(start_idx + period, len(series)):
        result[i] = alpha * series.iloc[i] + (1 - alpha) * result[i - 1]
    return pd.Series(result, index=series.index)


def compute_chandelier(df: pd.DataFrame, period: int = 22, mult: float = 3.0,
                       use_close: bool = True) -> pd.DataFrame:
    """
    Точное воспроизведение Pine Script Chandelier Exit (Alex Orekhov, GPL-3.0).

    Параметры (default = TradingView defaults):
        period    = 22   (ATR Period)
        mult      = 3.0  (ATR Multiplier)
        use_close = True (Use Close Price for Extremums)

    Возвращает df с колонками:
        atr, ce_long, ce_short, ce_dir (+1 long / -1 short), ce_signal (bool переворота)
    """
    df = df.copy()
    n = len(df)

    # ── ATR (Wilder RMA, как ta.atr() в Pine) ─────────────────────────────────
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_raw = _wilder_rma(tr, period)  # bare ATR
    atr     = mult * atr_raw           # atr = mult * ta.atr(length)
    df["atr"] = atr_raw                # сохраняем без множителя для метрик

    # ── Raw CE levels (до trailing) ────────────────────────────────────────────
    if use_close:
        hh = df["close"].rolling(period).max()   # highest(close, length)
        ll = df["close"].rolling(period).min()   # lowest(close, length)
    else:
        hh = df["high"].rolling(period).max()    # highest(high, length)
        ll = df["low"].rolling(period).min()     # lowest(low, length)

    long_raw  = (hh - atr).values
    short_raw = (ll + atr).values
    close_arr = df["close"].values

    # ── Trailing + Direction (строго по Pine Script) ───────────────────────────
    long_stop  = np.full(n, np.nan)
    short_stop = np.full(n, np.nan)
    direction  = np.ones(n, dtype=int)

    # Найдём первый валидный индекс
    first_valid = 0
    while first_valid < n and np.isnan(long_raw[first_valid]):
        first_valid += 1

    if first_valid < n:
        long_stop[first_valid]  = long_raw[first_valid]
        short_stop[first_valid] = short_raw[first_valid]
        # direction[first_valid]  = 1  (уже)

    for i in range(first_valid + 1, n):
        if np.isnan(long_raw[i]):
            continue

        prev_ls = long_stop[i - 1]
        prev_ss = short_stop[i - 1]
        prev_c  = close_arr[i - 1]
        cur_c   = close_arr[i]

        # longStop trailing (Pine: close[1] > longStopPrev ? max(...) : longStop)
        if not np.isnan(prev_ls):
            if prev_c > prev_ls:
                long_stop[i] = max(long_raw[i], prev_ls)
            else:
                long_stop[i] = long_raw[i]
        else:
            long_stop[i] = long_raw[i]

        # shortStop trailing (Pine: close[1] < shortStopPrev ? min(...) : shortStop)
        if not np.isnan(prev_ss):
            if prev_c < prev_ss:
                short_stop[i] = min(short_raw[i], prev_ss)
            else:
                short_stop[i] = short_raw[i]
        else:
            short_stop[i] = short_raw[i]

        # Direction (Pine: close > shortStopPrev ? 1 : close < longStopPrev ? -1 : dir)
        prev_dir = direction[i - 1]
        if not np.isnan(prev_ss) and cur_c > prev_ss:
            direction[i] = 1
        elif not np.isnan(prev_ls) and cur_c < prev_ls:
            direction[i] = -1
        else:
            direction[i] = prev_dir

    df["ce_long"]  = long_stop
    df["ce_short"] = short_stop
    df["ce_dir"]   = direction
    df["ce_signal"] = (df["ce_dir"] != df["ce_dir"].shift(1)) & df["atr"].notna()

    # Вспомогательные для rolling extremums (для отчётов)
    df["highest_high"] = hh
    df["lowest_low"]   = ll

    return df


# ── Извлечение сигналов ───────────────────────────────────────────────────────

def extract_signals(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Извлекает строки переворота CE. Добавляет вспомогательные поля."""
    df["atr_pct"]      = df["atr"] / df["close"] * 100
    df["range_21_pct"] = (df["highest_high"] - df["lowest_low"]) / df["close"] * 100
    df["ce_level"]     = np.where(df["ce_dir"] == 1, df["ce_long"], df["ce_short"])
    df["ce_dist_pct"]  = (df["close"] - df["ce_level"]) / df["close"] * 100

    sig_mask = df["ce_signal"] & df["atr"].notna()
    signals  = df[sig_mask].copy()

    signals["coin"]            = symbol
    signals["tf"]              = TIMEFRAME
    signals["signal"]          = np.where(signals["ce_dir"] == 1, "long", "short")
    signals["highest_high_21"] = signals["highest_high"]
    signals["lowest_low_21"]   = signals["lowest_low"]

    cols = [
        "coin", "tf", "timestamp", "signal",
        "close", "atr", "atr_pct",
        "highest_high_21", "lowest_low_21", "range_21_pct",
        "ce_level", "ce_dist_pct",
        "ce_long", "ce_short",
    ]
    return signals[cols].reset_index(drop=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parquet_files = sorted(OHLCV_DIR.glob("*_4h.parquet"))
    if not parquet_files:
        logger.error(f"No parquet files found in {OHLCV_DIR}. Run 01_download.py first.")
        sys.exit(1)

    logger.info(f"Processing {len(parquet_files)} symbols (ATR={ATR_PERIOD}, mult={MULTIPLIER}, use_close=True)")
    logger.info("NOTE: Using highest(CLOSE)/lowest(CLOSE) -- matches TradingView default")

    all_signals = []

    for i, path in enumerate(parquet_files, 1):
        symbol_raw = path.stem.replace("_4h", "")
        parts = symbol_raw.split("_")
        symbol = f"{parts[0]}/{parts[1]}:{parts[2]}" if len(parts) >= 3 else symbol_raw

        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        if len(df) < ATR_PERIOD * 2:
            logger.warning(f"[{i}] {symbol}: too few bars ({len(df)}), skip")
            continue

        try:
            df_ce = compute_chandelier(df, period=ATR_PERIOD, mult=MULTIPLIER, use_close=True)
            sigs  = extract_signals(df_ce, symbol)
            all_signals.append(sigs)
            logger.info(f"[{i}/{len(parquet_files)}] {symbol}: {len(df)} bars -> {len(sigs)} signals")
        except Exception as e:
            logger.error(f"[{i}] {symbol}: ERROR -- {e}")
            import traceback; traceback.print_exc()

    if not all_signals:
        logger.error("No signals computed. Exiting.")
        sys.exit(1)

    result = pd.concat(all_signals, ignore_index=True)
    out_path = OUTPUT_DIR / "raw_signals.parquet"
    result.to_parquet(out_path, index=False)

    logger.info(f"\nTotal signals: {len(result):,}")
    logger.info(f"Coins: {result['coin'].nunique()}")
    logger.info(f"Date range: {result['timestamp'].min().date()} -> {result['timestamp'].max().date()}")
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
