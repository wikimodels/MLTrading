"""
01_download.py — Инкрементальное обновление 4H OHLCV данных до сегодняшнего дня.

Читает суSCHествуюSCHие parquet из ../data/storage/raw/ (уже 48 монет).
Дополняет только недостаюSCHие свечи через ccxt (Bybit).
Сохраняет каждую монету в data/ohlcv/<symbol>_4h.parquet.

Запуск:
    poetry run python trend_detector_lab/01_download.py
"""
from __future__ import annotations

import sys
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd
import yaml
from loguru import logger

# ── пути ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # D:\GitHub\MLTrading
LAB  = Path(__file__).resolve().parent                 # trend_detector_lab\

sys.path.insert(0, str(ROOT))

# ── конфиг ────────────────────────────────────────────────────────────────────
with open(LAB / "config" / "settings.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

SOURCE_DIR = (LAB / CFG["data"]["source_dir"]).resolve()   # ../data/storage/raw
OUTPUT_DIR = LAB / CFG["data"]["output_dir"] / "ohlcv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAME   = CFG["chandelier"]["timeframe"]          # "4h"
TF_MS       = 14_400_000                              # 4h в миллисекундах
RATE_LIMIT  = 0.25                                    # сек между запросами

# ── логирование ───────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {level} | {message}", level="INFO")
logger.add(LAB / "data" / "download.log", rotation="5 MB", level="DEBUG")


# ── helpers ───────────────────────────────────────────────────────────────────

def _symbol_to_dirname(symbol: str) -> str:
    """BTC/USDT:USDT -> BTC_USDT_USDT"""
    return symbol.replace("/", "_").replace(":", "_")


def _dirname_to_symbol(dirname: str) -> str:
    """BTC_USDT_USDT -> BTC/USDT:USDT"""
    parts = dirname.split("_")
    if len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}:{parts[2]}"
    return dirname.replace("_", "/", 1)


def _load_existing(symbol: str) -> pd.DataFrame | None:
    """Загружает суSCHествуюSCHий parquet из source_dir (data/storage/raw/)."""
    sym_dir = SOURCE_DIR / _symbol_to_dirname(symbol)
    path = sym_dir / f"{TIMEFRAME}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _load_output(symbol: str) -> pd.DataFrame | None:
    """Загружает уже обновлённый parquet из output_dir."""
    path = OUTPUT_DIR / f"{_symbol_to_dirname(symbol)}_4h.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _save_output(df: pd.DataFrame, symbol: str) -> None:
    path = OUTPUT_DIR / f"{_symbol_to_dirname(symbol)}_4h.parquet"
    df.to_parquet(path, index=False)


def _fetch_since(exchange: ccxt.Exchange, symbol: str, since_ts: int) -> pd.DataFrame:
    """Подгружает новые свечи начиная с since_ts (ms)."""
    all_ohlcv = []
    limit = 1000
    now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    while since_ts < now_ts:
        try:
            batch = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since_ts, limit=limit)
        except ccxt.RateLimitExceeded:
            logger.warning("Rate limit — sleep 10s")
            time.sleep(10)
            continue
        except ccxt.NetworkError as e:
            logger.error(f"Network error: {e}")
            time.sleep(5)
            continue
        except Exception as e:
            logger.error(f"Unexpected error fetching {symbol}: {e}")
            break

        if not batch:
            break
        all_ohlcv.extend(batch)
        last_ts = batch[-1][0]
        if len(batch) < limit:
            break
        since_ts = last_ts + TF_MS
        time.sleep(RATE_LIMIT)

    if not all_ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Инициализация биржи (без API-ключей, только публичные данные)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    import os

    exchange_params: dict = {
        "enableRateLimit": True,
        "options": {"defaultType": "linear", "fetchMarkets": ["linear"]},
    }
    proxy = os.getenv("BYBIT_PROXY", "")
    if proxy:
        exchange_params["proxies"] = {"http": proxy, "https": proxy}
        exchange_params["httpsProxy"] = proxy
        logger.info(f"Proxy: {proxy}")

    exchange = ccxt.bybit(exchange_params)
    logger.info("Exchange initialized (public data only)")

    # Список монет из source_dir
    symbols = []
    for d in sorted(SOURCE_DIR.iterdir()):
        if d.is_dir() and (d / f"{TIMEFRAME}.parquet").exists():
            symbols.append(_dirname_to_symbol(d.name))
    logger.info(f"Found {len(symbols)} symbols in source_dir")

    now_dt = datetime.now(timezone.utc)
    updated = 0
    skipped = 0

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] {symbol}")

        # Сначала смотрим output (уже обновлённый), иначе source (исходник)
        df_existing = _load_output(symbol) or _load_existing(symbol)

        if df_existing is None or df_existing.empty:
            logger.warning(f"  No existing data for {symbol}, skipping")
            skipped += 1
            continue

        last_ts = df_existing["timestamp"].iloc[-1]
        hours_behind = (now_dt - last_ts.to_pydatetime().replace(tzinfo=timezone.utc)).total_seconds() / 3600

        if hours_behind < 4:
            logger.info(f"  Already up to date ({last_ts.date()})")
            # Всё равно сохраняем в output если там еSCHё нет
            if not (OUTPUT_DIR / f"{_symbol_to_dirname(symbol)}_4h.parquet").exists():
                _save_output(df_existing, symbol)
            skipped += 1
            continue

        logger.info(f"  Last bar: {last_ts.date()} ({hours_behind:.0f}h behind), fetching new...")

        since_ts = int(last_ts.timestamp() * 1000) + TF_MS
        df_new = _fetch_since(exchange, symbol, since_ts)

        if df_new.empty:
            logger.warning(f"  No new data returned for {symbol}")
            _save_output(df_existing, symbol)
            skipped += 1
            continue

        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        _save_output(df_combined, symbol)

        new_bars = len(df_combined) - len(df_existing)
        logger.info(f"  +{new_bars} new bars -> {len(df_combined)} total [{df_combined['timestamp'].iloc[0].date()} -> {df_combined['timestamp'].iloc[-1].date()}]")
        updated += 1
        time.sleep(RATE_LIMIT)

    logger.info(f"\nDone. Updated: {updated}, skipped/up-to-date: {skipped}")
    logger.info(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
