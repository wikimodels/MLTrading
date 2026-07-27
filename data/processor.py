"""[Translated]"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class DataProcessor:
    """[Translated]"""

    @staticmethod
    def clean(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
        """[Translated]"""
        n_before = len(df)
        df = df.copy()
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        df = df[(df["close"] > 0) & (df["volume"] >= 0)]

        # OHLC consistency
        df = df[df["high"] >= df["low"]]
        df = df[df["high"] >= df["close"]]
        df = df[df["low"] <= df["close"]]
        df = df.reset_index(drop=True)

        n_after = len(df)
        if n_before != n_after:
            logger.debug(f"")

        return df

    @staticmethod
    def validate(df: pd.DataFrame, symbol: str = "", min_bars: int = 200) -> bool:
        """[Translated]"""
        if df is None or len(df) < min_bars:
            logger.warning(f"")
            return False

        null_pct = df[["open", "high", "low", "close", "volume"]].isnull().mean().max()
        if null_pct > 0.05:
            logger.warning(f"")
            return False

        return True
