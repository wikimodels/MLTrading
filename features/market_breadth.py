"""[Translated]"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta
from loguru import logger

from config_loader import get_config


class MarketBreadthCalculator:
    """[Translated]"""

    def __init__(self):
        cfg = get_config()
        self.corr_window = cfg["features"]["breadth_rolling_corr_window"]  # 30

    def compute_all(
        self,
        all_symbols_ohlcv: dict[str, pd.DataFrame],
        btc_symbol: str = "BTC/USDT:USDT",
    ) -> pd.DataFrame:
        """[Translated]"""
        if btc_symbol not in all_symbols_ohlcv:
            logger.warning(f"")
            return pd.DataFrame()
        logger.info("")

        btc_df = all_symbols_ohlcv[btc_symbol].copy().sort_values("timestamp")
        btc_df["timestamp"] = pd.to_datetime(btc_df["timestamp"], utc=True)
        all_timestamps = pd.DatetimeIndex(btc_df["timestamp"])
        symbols = list(all_symbols_ohlcv.keys())
        n_symbols = len(symbols)

        closes_matrix = pd.DataFrame(index=all_timestamps)
        rsi_matrix = pd.DataFrame(index=all_timestamps)
        ema50_matrix = pd.DataFrame(index=all_timestamps)

        for sym in symbols:
            df = all_symbols_ohlcv[sym].copy().sort_values("timestamp")
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
            df_reindexed = df.reindex(all_timestamps, method="ffill")


            closes_matrix[sym] = df_reindexed["close"]
            if "rsi_14" in df.columns:
                rsi_matrix[sym] = df_reindexed["rsi_14"]
            else:
                rsi_vals = ta.momentum.rsi(df["close"], window=14)
                rsi_matrix[sym] = rsi_vals.reindex(all_timestamps, method="ffill")

            # EMA50
            ema50_vals = ta.trend.ema_indicator(df["close"], window=50)
            ema50_matrix[sym] = ema50_vals.reindex(all_timestamps, method="ffill")
        result = pd.DataFrame(index=all_timestamps)
        returns_matrix = closes_matrix.pct_change()
        result["breadth_down_ratio"] = (returns_matrix < 0).sum(axis=1) / n_symbols
        result["avg_rsi_top20"] = rsi_matrix.mean(axis=1)
        result["breadth_momentum"] = returns_matrix.mean(axis=1)
        result["breadth_trend_5"] = result["breadth_down_ratio"].diff(5)
        above_ema50 = closes_matrix > ema50_matrix
        result["pct_above_ema50"] = above_ema50.sum(axis=1) / n_symbols
        btc_close = closes_matrix[btc_symbol] if btc_symbol in closes_matrix else closes_matrix.iloc[:, 0]
        btc_ema200 = ta.trend.ema_indicator(btc_close, window=200)
        btc_rsi = rsi_matrix[btc_symbol] if btc_symbol in rsi_matrix else rsi_matrix.iloc[:, 0]

        result["btc_vs_ema200"] = (btc_close > btc_ema200).astype(int)
        result["btc_rsi"] = btc_rsi

        # 7. Market Regime
        result["market_regime"] = self._compute_regime(result)
        result["market_overbought"] = (
            (result["avg_rsi_top20"] > 70) &
            (result["breadth_down_ratio"] < 0.3)
        ).astype(int)

        result["market_oversold"] = (
            (result["avg_rsi_top20"] < 35) &
            (result["breadth_down_ratio"] > 0.7)
        ).astype(int)

        result = result.reset_index().rename(columns={"index": "timestamp"})
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)

        logger.info(f"")
        return result

    def add_symbol_specific(
        self,
        df_4h: pd.DataFrame,
        breadth_df: pd.DataFrame,
        all_symbols_ohlcv: dict[str, pd.DataFrame],
        symbol: str,
        btc_symbol: str = "BTC/USDT:USDT",
    ) -> pd.DataFrame:
        """[Translated]"""
        df = df_4h.copy().sort_values("timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        breadth_local = breadth_df.copy()
        breadth_local["timestamp"] = pd.to_datetime(breadth_local["timestamp"], utc=True)
        breadth_cols = [c for c in breadth_local.columns if c != "timestamp"]
        df = df.merge(
            breadth_local[["timestamp"] + breadth_cols],
            on="timestamp",
            how="left"
        )
        df[breadth_cols] = df[breadth_cols].ffill()

        # Relative Strength vs BTC
        sym_returns = df["close"].pct_change()
        if btc_symbol in all_symbols_ohlcv and symbol != btc_symbol:
            btc_df = all_symbols_ohlcv[btc_symbol].copy().sort_values("timestamp")
            btc_df["timestamp"] = pd.to_datetime(btc_df["timestamp"], utc=True)
            btc_df = btc_df.set_index("timestamp")
            
            ts_index = pd.DatetimeIndex(df["timestamp"])
            btc_returns_aligned = (
                btc_df["close"].pct_change()
                .reindex(ts_index, method="ffill")
                .values
            )
            df["rs_vs_btc"] = sym_returns.values / (btc_returns_aligned + 1e-8)
            df["btc_correlation_30"] = sym_returns.rolling(self.corr_window).corr(
                pd.Series(btc_returns_aligned, index=sym_returns.index)
            )
        else:
            df["rs_vs_btc"] = 1.0
            df["btc_correlation_30"] = 1.0
        df["relative_strength_vs_breadth"] = sym_returns / (
            (-df["breadth_momentum"]).replace(0, np.nan)
        )
        df["relative_strength_vs_breadth"] = df["relative_strength_vs_breadth"].clip(-10, 10).fillna(0)

        return df

    # ──────────────────────────────────────────────────────────────
    # Market Regime
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_regime(result: pd.DataFrame) -> pd.Series:
        """[Translated]"""
        regime = pd.Series(0, index=result.index)

        bull_mask = (
            (result["btc_vs_ema200"] == 1) &
            (result["pct_above_ema50"] > 0.6)
        )
        bear_mask = (
            (result["btc_vs_ema200"] == 0) &
            (result["breadth_down_ratio"] > 0.55)
        )

        regime[bull_mask] = 1
        regime[bear_mask] = -1

        return regime
