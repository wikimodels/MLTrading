"""Data storage helpers — read/write parquet files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from shared.config_loader import get_config, get_project_root


class DataStorage:
    """Manages local parquet data for OHLCV and funding."""

    def __init__(self):
        cfg = get_config()
        self.raw_dir = get_project_root() / cfg["data"]["raw_dir"]
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _symbol_dir(self, symbol: str) -> Path:
        """Returns (and creates) the directory for a given symbol."""
        symbol_safe = symbol.replace("/", "_").replace(":", "_")
        path = self.raw_dir / symbol_safe
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ──────────────────────────────────────────────────────────────
    # OHLCV
    # ──────────────────────────────────────────────────────────────

    def save_ohlcv(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Saves OHLCV data to parquet."""
        path = self._symbol_dir(symbol) / f"{timeframe}.parquet"
        df.to_parquet(path, index=False)
        logger.debug(f"Saved {symbol} {timeframe}: {len(df)} rows -> {path}")

    def load_ohlcv(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Loads OHLCV data from parquet. Returns None if not found."""
        path = self._symbol_dir(symbol) / f"{timeframe}.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    def save_funding(self, df: pd.DataFrame, symbol: str) -> None:
        """Saves funding rate data."""
        path = self._symbol_dir(symbol) / "funding.parquet"
        df.to_parquet(path, index=False)

    def load_funding(self, symbol: str) -> Optional[pd.DataFrame]:
        """Loads funding rate data. Returns None if not found."""
        path = self._symbol_dir(symbol) / "funding.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    # ──────────────────────────────────────────────────────────────
    # Listing / Status
    # ──────────────────────────────────────────────────────────────

    def list_symbols(self, timeframe: str = "4h") -> list[str]:
        """Returns a list of symbols that have OHLCV data for the given timeframe."""
        symbols = []
        for d in sorted(self.raw_dir.iterdir()):
            if d.is_dir() and (d / f"{timeframe}.parquet").exists():
                # Directory names are like BTC_USDT_USDT -> BTC/USDT:USDT
                parts = d.name.split("_")
                if len(parts) >= 3:
                    sym = f"{parts[0]}/{parts[1]}:{parts[2]}"
                else:
                    sym = d.name.replace("_", "/", 1)
                symbols.append(sym)
        return symbols

    def status(self) -> pd.DataFrame:
        """Returns a summary DataFrame of all stored data."""
        rows = []
        for symbol in self.list_symbols("4h"):
            df = self.load_ohlcv(symbol, "4h")
            if df is not None and len(df) > 0:
                rows.append({
                    "symbol": symbol,
                    "bars_4h": len(df),
                    "from": df["timestamp"].iloc[0].date(),
                    "to": df["timestamp"].iloc[-1].date(),
                    "has_funding": (
                        self._symbol_dir(symbol) / "funding.parquet"
                    ).exists(),
                })
        return pd.DataFrame(rows)
