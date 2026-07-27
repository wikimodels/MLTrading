"""[Translated]"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd
from loguru import logger

from data.storage import DataStorage
from config_loader import get_config


class BybitCollector:
    """[Translated]"""

    TIMEFRAME_MS = {
        "1m": 60_000,
        "5m": 300_000,
        "15m": 900_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }

    def __init__(self, testnet: bool = True):
        cfg = get_config()
        self.cfg = cfg
        self.storage = DataStorage()

        import os
        from dotenv import load_dotenv
        load_dotenv()

        exchange_params = {
            "enableRateLimit": True,
            "options": {
                "defaultType": "linear",     
                "adjustForTimeDifference": True,
                "fetchMarkets": ["linear"],
            },
        }
        proxy = os.getenv("BYBIT_PROXY", "")
        if proxy:
            exchange_params["proxies"] = {"http": proxy, "https": proxy}
            exchange_params["httpsProxy"] = proxy
            logger.info(f"")

        if testnet:
            exchange_params["apiKey"] = os.getenv("BYBIT_API_KEY", "")
            exchange_params["secret"] = os.getenv("BYBIT_API_SECRET", "")
            exchange_params["options"]["sandboxMode"] = True
        else:
            exchange_params["apiKey"] = os.getenv("BYBIT_MAINNET_API_KEY", "")
            exchange_params["secret"] = os.getenv("BYBIT_MAINNET_API_SECRET", "")

        self.exchange = ccxt.bybit(exchange_params)

        if testnet:
            self.exchange.set_sandbox_mode(True)

        logger.info(f"")

    # ──────────────────────────────────────────────
    # ──────────────────────────────────────────────

    def get_top_symbols(self, n: int = 20) -> list[str]:
        """[Translated]"""
        logger.info("")

        EXCLUDE = {"USDC", "USDT", "BUSD", "DAI", "TUSD", "USDP", "FDUSD",
                   "BTC3L", "BTC3S", "ETH3L", "ETH3S", "BTCDOM", "DEFI",
                   "SOXL", "NVDA", "TSLA", "MSTR", "SKHYNIX", "XAU", "PAXG"}

        btc_sym = "BTC/USDT:USDT"
        tickers = self.exchange.fetch_tickers()
        valid_symbols = []
        for symbol, ticker in tickers.items():
            if not symbol.endswith("/USDT:USDT"): continue
            base = symbol.split("/")[0]
            if base in EXCLUDE: continue
            volume_usdt = ticker.get("quoteVolume") or 0
            valid_symbols.append((symbol, volume_usdt))
        valid_symbols.sort(key=lambda x: x[1], reverse=True)
        logger.info("")
        min_history_days = self.cfg["data"]["history_days"]
        check_since = int((datetime.now(timezone.utc).timestamp() - min_history_days * 86400) * 1000)
        
        history_ok = []
        if btc_sym in [s for s, _ in valid_symbols]:
            history_ok.append((btc_sym, tickers[btc_sym].get("quoteVolume", 0)))
            
        for symbol, vol in valid_symbols:
            if symbol == btc_sym: continue
            if len(history_ok) >= 100:
                break
                
            try:
                probe = self.exchange.fetch_ohlcv(symbol, "4h", since=check_since, limit=3)
                if probe and len(probe) >= 2:
                    first_bar_ts = probe[0][0]
                    if first_bar_ts <= check_since + 30 * 86400 * 1000:
                        history_ok.append((symbol, vol))
                        logger.debug(f"")
                time.sleep(self.cfg["exchange"]["rate_limit_ms"] / 1000)
            except Exception as e:
                continue
                
        logger.info(f"")
        logger.info("")
        history_ok.sort(key=lambda x: x[1], reverse=True)
        top_50_vol = [s for s, _ in history_ok[:50]]
        
        if btc_sym not in top_50_vol:
            top_50_vol.insert(0, btc_sym)
            
        logger.info(f"")
        logger.info("")
        corr_days = 90
        corr_since = int((datetime.now(timezone.utc).timestamp() - corr_days * 86400) * 1000)

        btc_df = self.fetch_ohlcv(btc_sym, "4h", since_ts=corr_since)
        if btc_df.empty:
            return top_50_vol[:n]

        btc_returns = btc_df.set_index("timestamp")["close"].pct_change().dropna()
        correlations = {btc_sym: 1.0}

        for symbol in top_50_vol:
            if symbol == btc_sym: continue
            try:
                df = self.fetch_ohlcv(symbol, "4h", since_ts=corr_since)
                if df.empty or len(df) < 50: continue

                sym_returns = df.set_index("timestamp")["close"].pct_change().dropna()
                aligned = pd.concat([btc_returns.rename("btc"), sym_returns.rename("sym")], axis=1).dropna()
                if len(aligned) < 30: continue

                corr = aligned["btc"].corr(aligned["sym"])
                if not pd.isna(corr):
                    correlations[symbol] = corr
                time.sleep(self.cfg["exchange"]["rate_limit_ms"] / 1000)
            except Exception:
                continue

        ranked = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        top = [btc_sym]
        for sym, corr in ranked:
            if sym == btc_sym: continue
            if len(top) >= n: break
            top.append(sym)

        logger.info(f"")
        for sym, corr in ranked[:n]:
            logger.info(f"  {sym:30s}  corr={corr:+.3f}")

        return top



    # ──────────────────────────────────────────────
    # ──────────────────────────────────────────────

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "4h",
        since_days: Optional[int] = None,
        since_ts: Optional[int] = None,
    ) -> pd.DataFrame:
        """[Translated]"""
        if since_ts is None:
            days = since_days or self.cfg["data"]["history_days"]
            since_ts = int(
                (datetime.now(timezone.utc).timestamp() - days * 86400) * 1000
            )

        tf_ms = self.TIMEFRAME_MS.get(timeframe, 14_400_000)
        limit = 1000  # Bybit max per request
        all_ohlcv = []

        logger.info(f"")

        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=since_ts, limit=limit
                )
            except ccxt.RateLimitExceeded:
                logger.warning("")
                time.sleep(10)
                continue
            except ccxt.NetworkError as e:
                logger.error(f"")
                time.sleep(5)
                continue

            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            if len(ohlcv) < limit:
                break
            since_ts = last_ts + tf_ms
            time.sleep(self.cfg["exchange"]["rate_limit_ms"] / 1000)
            now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
            if since_ts >= now_ts:
                break

        if not all_ohlcv:
            logger.warning(f"")
            return pd.DataFrame()

        df = pd.DataFrame(
            all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

        logger.info(f""
                    f"[{df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}]")
        return df

    # ──────────────────────────────────────────────
    # Funding Rate
    # ──────────────────────────────────────────────

    def fetch_funding_rate_history(
        self, symbol: str, since_days: int = 730
    ) -> pd.DataFrame:
        """[Translated]"""
        since_ts = int(
            (datetime.now(timezone.utc).timestamp() - since_days * 86400) * 1000
        )

        all_records = []
        limit = 200

        logger.info(f"")

        while True:
            try:
                records = self.exchange.fetch_funding_rate_history(
                    symbol, since=since_ts, limit=limit
                )
            except Exception as e:
                logger.error(f"")
                break

            if not records:
                break

            all_records.extend(records)

            if len(records) < limit:
                break

            since_ts = records[-1]["timestamp"] + 1
            time.sleep(self.cfg["exchange"]["rate_limit_ms"] / 1000)

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df[["timestamp", "fundingRate"]].drop_duplicates("timestamp").sort_values("timestamp")
        df = df.rename(columns={"fundingRate": "funding_rate"})
        df = df.reset_index(drop=True)

        logger.info(f"")
        return df

    # ──────────────────────────────────────────────
    # ──────────────────────────────────────────────

    def collect_symbol(
        self,
        symbol: str,
        timeframes: Optional[list[str]] = None,
        incremental: bool = True,
    ) -> None:
        """[Translated]"""
        if timeframes is None:
            timeframes = ["4h", "1h", "1d"]

        for tf in timeframes:
            since_ts = None

            if incremental:
                existing = self.storage.load_ohlcv(symbol, tf)
                if existing is not None and len(existing) > 0:
                    last_ts = existing["timestamp"].iloc[-1]
                    since_ts = int(last_ts.timestamp() * 1000) + self.TIMEFRAME_MS[tf]
                    logger.info(f"")

            df_new = self.fetch_ohlcv(symbol, tf, since_ts=since_ts)

            if df_new.empty:
                continue

            if incremental:
                existing = self.storage.load_ohlcv(symbol, tf)
                if existing is not None and len(existing) > 0:
                    df_new = pd.concat([existing, df_new], ignore_index=True)
                    df_new = df_new.drop_duplicates("timestamp").sort_values("timestamp")

            self.storage.save_ohlcv(df_new, symbol, tf)
        df_funding = self.fetch_funding_rate_history(symbol)
        if not df_funding.empty:
            self.storage.save_funding(df_funding, symbol)

    # ──────────────────────────────────────────────
    # ──────────────────────────────────────────────

    def collect_all(
        self,
        symbols: Optional[list[str]] = None,
        incremental: bool = True,
    ) -> list[str]:
        """[Translated]"""
        if symbols is None:
            n = self.cfg["trading"]["top_n_symbols"]
            symbols = self.get_top_symbols(n)

        success = []
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"")
            try:
                self.collect_symbol(symbol, incremental=incremental)
                success.append(symbol)
            except Exception as e:
                logger.error(f"")

        logger.info(f"")
        return success
