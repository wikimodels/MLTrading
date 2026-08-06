"""
03_trades.py — Построение таблицы сделок из raw signals.

Каждая сделка: от момента CE-сигнала до следуюSCHего CE-сигнала.
Вход: open следуюSCHей свечи после сигнала.
Выход: зависит от stop_mode (chandelier | fixed).

Дополнительно считает:
  - max_favorable_atr / max_adverse_atr за forward_bars_window свечей вперёд
  - reached_2atr / 3atr / 5atr / 8atr (bool)
  - stop_hit, stop_level_at_entry

Читает:  data/raw_signals.parquet + data/ohlcv/*.parquet
Пишет:   data/trades.parquet

Запуск:
    poetry run python trend_detector_lab/03_trades.py
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

ROOT = Path(__file__).resolve().parent.parent
LAB  = Path(__file__).resolve().parent

with open(LAB / "config" / "settings.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

DATA_DIR    = LAB / CFG["data"]["output_dir"]
OHLCV_DIR   = DATA_DIR / "ohlcv"
SIGNALS_PATH = DATA_DIR / "raw_signals.parquet"
OUTPUT_PATH  = DATA_DIR / "trades.parquet"

STOP_MODE       = CFG["stop"]["mode"]          # "chandelier" | "fixed"
SL_ATR_MULT     = CFG["stop"]["sl_atr_mult"]  # для fixed mode
FWD_WINDOW      = CFG["data"]["forward_bars_window"]  # 60 свечей вперёд

TARGET_ATR_MULT = CFG["analysis"]["target_atr_mult"]  # [2, 3, 5, 8]
TARGET_BARS     = CFG["analysis"]["target_bars"]       # [10, 15, 25, 50]

COMMISSION = CFG["backtest"]["commission"]
SLIPPAGE   = CFG["backtest"]["slippage"]

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {level} | {message}", level="INFO")


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_ohlcv(symbol: str) -> pd.DataFrame | None:
    """Загружает OHLCV для символа."""
    sym_safe = symbol.replace("/", "_").replace(":", "_")
    path = OHLCV_DIR / f"{sym_safe}_4h.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _compute_forward_stats(
    ohlcv: pd.DataFrame,
    entry_idx: int,
    direction: int,        # +1 long, -1 short
    entry_price: float,
    atr_at_entry: float,
    fwd_window: int,
) -> dict:
    """
    Вычисляет max_favorable_atr, max_adverse_atr и reached_NxATR
    за следуюSCHие fwd_window свечей от entry_idx.
    """
    end_idx = min(entry_idx + fwd_window + 1, len(ohlcv))
    fwd = ohlcv.iloc[entry_idx + 1 : end_idx]

    if fwd.empty or atr_at_entry <= 0:
        return {
            "max_favorable_atr": 0.0,
            "max_adverse_atr": 0.0,
            **{f"reached_{m}atr": False for m in TARGET_ATR_MULT},
        }

    if direction == 1:  # long
        favorable = (fwd["high"] - entry_price) / atr_at_entry
        adverse   = (entry_price - fwd["low"])  / atr_at_entry
    else:               # short
        favorable = (entry_price - fwd["low"])  / atr_at_entry
        adverse   = (fwd["high"] - entry_price) / atr_at_entry

    max_fav = float(favorable.max())
    max_adv = float(adverse.max())

    result: dict = {
        "max_favorable_atr": round(max_fav, 3),
        "max_adverse_atr":   round(max_adv, 3),
    }
    for m in TARGET_ATR_MULT:
        result[f"reached_{m}atr"] = max_fav >= m

    return result


# ── основная логика ───────────────────────────────────────────────────────────

def build_trades(signals: pd.DataFrame, ohlcv: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Строит таблицу сделок для одной монеты.
    Каждая сделка: от i-го сигнала до (i+1)-го сигнала.
    """
    ohlcv = ohlcv.copy().reset_index(drop=True)
    # Индекс по timestamp для быстрого поиска
    ts_to_idx = {ts: idx for idx, ts in enumerate(ohlcv["timestamp"])}

    records = []
    sym_sigs = signals[signals["coin"] == symbol].reset_index(drop=True)

    if len(sym_sigs) < 2:
        return pd.DataFrame()

    for i in range(len(sym_sigs) - 1):
        sig       = sym_sigs.iloc[i]
        next_sig  = sym_sigs.iloc[i + 1]

        signal_ts = sig["timestamp"]
        exit_ts   = next_sig["timestamp"]
        direction = 1 if sig["signal"] == "long" else -1

        # Находим индекс сигнальной свечи в OHLCV
        if signal_ts not in ts_to_idx:
            continue
        sig_idx = ts_to_idx[signal_ts]

        # Вход: open следуюSCHей свечи после сигнала
        if sig_idx + 1 >= len(ohlcv):
            continue
        entry_bar   = ohlcv.iloc[sig_idx + 1]
        entry_price = float(entry_bar["open"]) * (1 + SLIPPAGE * direction)

        # Выход: определяем в зависимости от stop_mode
        if exit_ts not in ts_to_idx:
            continue
        exit_sig_idx = ts_to_idx[exit_ts]

        # Свечи внутри сделки (после входа, до сигнала выхода)
        trade_bars = ohlcv.iloc[sig_idx + 1 : exit_sig_idx + 1]
        bars_in_trade = len(trade_bars)

        atr_at_entry = float(sig["atr"])
        atr_pct      = float(sig["atr_pct"])

        # ── Stop level ────────────────────────────────────────────────────────
        if STOP_MODE == "chandelier":
            # Стоп = CE level (трейляSCHийся) — выход происходит при переходе к next_sig
            stop_level = float(sig["ce_long"] if direction == 1 else sig["ce_short"])
            stop_hit   = False   # при chandelier стоп = сам следуюSCHий сигнал
        else:  # fixed
            if direction == 1:
                stop_level = entry_price - SL_ATR_MULT * atr_at_entry
            else:
                stop_level = entry_price + SL_ATR_MULT * atr_at_entry

            # Проверяем пробой внутри trade_bars
            stop_hit = False
            if direction == 1:
                stop_hit = bool((trade_bars["low"] < stop_level).any())
            else:
                stop_hit = bool((trade_bars["high"] > stop_level).any())

        # ── Exit price ────────────────────────────────────────────────────────
        if STOP_MODE == "fixed" and stop_hit:
            # Выход по стопу: open свечи после пробоя
            if direction == 1:
                stop_bar_mask = trade_bars["low"] < stop_level
            else:
                stop_bar_mask = trade_bars["high"] > stop_level
            stop_bar_idx_in_ohlcv = trade_bars.index[stop_bar_mask][0]
            exit_bar_idx = min(stop_bar_idx_in_ohlcv + 1, len(ohlcv) - 1)
            exit_price = float(ohlcv.iloc[exit_bar_idx]["open"]) * (1 - SLIPPAGE * direction)
            bars_in_trade = stop_bar_idx_in_ohlcv - (sig_idx + 1) + 1
        else:
            # Выход по CE-сигналу: open свечи после следуюSCHего сигнала
            if exit_sig_idx + 1 < len(ohlcv):
                exit_price = float(ohlcv.iloc[exit_sig_idx + 1]["open"]) * (1 - SLIPPAGE * direction)
            else:
                exit_price = float(ohlcv.iloc[exit_sig_idx]["close"])

        # ── PnL ───────────────────────────────────────────────────────────────
        if direction == 1:
            profit_pct = (exit_price - entry_price) / entry_price * 100
        else:
            profit_pct = (entry_price - exit_price) / entry_price * 100

        # Комиссия (taker in + out)
        profit_pct -= (COMMISSION * 2) * 100

        profit_atr = profit_pct / atr_pct if atr_pct > 0 else 0.0

        # ── Целевые метрики (по bars_in_trade) ───────────────────────────────
        bars_flags = {f"bars_gt_{b}": bars_in_trade > b for b in TARGET_BARS}

        # ── Forward stats (max_favorable_atr и т.д.) ─────────────────────────
        fwd_stats = _compute_forward_stats(
            ohlcv, sig_idx, direction, entry_price, atr_at_entry, FWD_WINDOW
        )

        record = {
            "coin":              symbol,
            "tf":                sig["tf"],
            "signal_ts":         signal_ts,
            "exit_ts":           exit_ts,
            "signal":            sig["signal"],
            "entry_price":       round(entry_price, 6),
            "exit_price":        round(exit_price, 6),
            "bars_in_trade":     bars_in_trade,
            "profit_pct":        round(profit_pct, 4),
            "profit_atr":        round(profit_atr, 4),
            "atr":               round(atr_at_entry, 6),
            "atr_pct":           round(atr_pct, 4),
            "stop_mode":         STOP_MODE,
            "stop_level_at_entry": round(stop_level, 6),
            "stop_hit":          stop_hit,
            **bars_flags,
            **fwd_stats,
        }
        records.append(record)

    return pd.DataFrame(records)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not SIGNALS_PATH.exists():
        logger.error(f"Not found: {SIGNALS_PATH}. Run 02_chandelier.py first.")
        sys.exit(1)

    signals = pd.read_parquet(SIGNALS_PATH)
    signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True)
    logger.info(f"Loaded {len(signals):,} raw signals, {signals['coin'].nunique()} coins")
    logger.info(f"Stop mode: {STOP_MODE}")

    all_trades = []
    coins = signals["coin"].unique()

    for i, symbol in enumerate(sorted(coins), 1):
        ohlcv = _load_ohlcv(symbol)
        if ohlcv is None or ohlcv.empty:
            logger.warning(f"[{i}] {symbol}: no OHLCV, skip")
            continue

        trades = build_trades(signals, ohlcv, symbol)
        if trades.empty:
            logger.warning(f"[{i}] {symbol}: no trades built")
            continue

        all_trades.append(trades)
        logger.info(f"[{i}/{len(coins)}] {symbol}: {len(trades)} trades")

    if not all_trades:
        logger.error("No trades built. Exiting.")
        sys.exit(1)

    result = pd.concat(all_trades, ignore_index=True)
    result.to_parquet(OUTPUT_PATH, index=False)

    logger.info(f"\nTotal trades: {len(result):,}")
    logger.info(f"Coins: {result['coin'].nunique()}")
    logger.info(f"Date range: {result['signal_ts'].min().date()} -> {result['signal_ts'].max().date()}")
    logger.info(f"Avg bars_in_trade: {result['bars_in_trade'].mean():.1f}")
    logger.info(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
