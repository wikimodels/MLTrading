"""[Translated]"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from config_loader import get_config


class DataStorage:
    """[Translated]"""

    def __init__(self):
        cfg = get_config()
        self.raw_dir = Path(cfg["data"]["raw_dir"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _symbol_dir(self, symbol: str) -> Path:
        """[Translated]""""""[Translated]""""""
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
