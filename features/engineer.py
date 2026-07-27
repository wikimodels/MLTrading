"""[Translated]"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta

from config_loader import get_config


class FeatureEngineer:
    """[Translated]"""

    def __init__(self):
        cfg = get_config()["features"]
        self.atr_period = cfg["atr_period"]
        self.rsi_period = cfg["rsi_period"]
        self.ema_periods = cfg["ema_periods"]   # [21, 50, 200]
        self.bb_period = cfg["bb_period"]
        self.macd_fast = cfg["macd_fast"]
        self.macd_slow = cfg["macd_slow"]
        self.macd_signal = cfg["macd_signal"]

    # ──────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────

    def compute(
        self,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame | None = None,
        df_1d: pd.DataFrame | None = None,
        funding_df: pd.DataFrame | None = None,
        symbol_id: int = 0,
    ) -> pd.DataFrame:
        """[Translated]"""
        df = df_4h.copy()
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        assert required.issubset(df.columns), f""

        df = df.sort_values("timestamp").reset_index(drop=True)

        # ── 1. Trend ──────────────────────────────────────────────
        df = self._add_trend(df)

        # ── 2. Momentum ───────────────────────────────────────────
        df = self._add_momentum(df)

        # ── 3. Volatility ─────────────────────────────────────────
        df = self._add_volatility(df)

        # ── 4. Volume ─────────────────────────────────────────────
        df = self._add_volume(df)

        # ── 5. Market Structure ───────────────────────────────────
        df = self._add_structure(df)

        # ── 6. Funding Rate ───────────────────────────────────────
        if funding_df is not None:
            df = self._add_funding(df, funding_df)
        else:
            df["funding_rate"] = 0.0
            df["funding_cumsum_24h"] = 0.0

        # ── 7. Time Features ──────────────────────────────────────
        df = self._add_time_features(df)

        # ── 8. Multi-TF Features ──────────────────────────────────
        if df_1h is not None:
            df = self._add_multitf_1h(df, df_1h)
        else:
            df["trend_1h"] = 0
            df["rsi_1h"] = 50.0

        if df_1d is not None:
            df = self._add_multitf_1d(df, df_1d)
        else:
            df["trend_1d"] = 0
            df["rsi_1d"] = 50.0
            
        df["symbol_id"] = symbol_id
        feat_cols = [c for c in df.columns if c not in ("timestamp", "open", "high", "low", "close", "volume")]
        df = df.dropna(subset=feat_cols).reset_index(drop=True)

        return df

    # ──────────────────────────────────────────────────────────────
    # 1. TREND
    # ──────────────────────────────────────────────────────────────

    def _add_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        self.ema_periods = [21, 50, 200]
        for period in self.ema_periods:
            col = f"ema_{period}"
            df[col] = ta.trend.ema_indicator(close, window=period)
            df[f"price_vs_ema_{period}"] = (close - df[col]) / df[col]
            df[f"ema_{period}_slope"] = df[col].pct_change(3)

        # EMA crossovers
        df["ema21_vs_ema50"] = (df["ema_21"] > df["ema_50"]).astype(int)
        df["ema50_vs_ema200"] = (df["ema_50"] > df["ema_200"]).astype(int)

        return df

    # ──────────────────────────────────────────────────────────────
    # 2. MOMENTUM
    # ──────────────────────────────────────────────────────────────

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # ADX & DMI
        adx = ta.trend.ADXIndicator(high, low, close, window=14)
        df["adx"] = adx.adx()
        df["adx_pos"] = adx.adx_pos()
        df["adx_neg"] = adx.adx_neg()
        df["adx_strong"] = (df["adx"] > 25).astype(int)
        df["adx_chop"] = (df["adx"] < 20).astype(int)

        # RSI
        df["rsi_14"] = ta.momentum.rsi(close, window=14)
        df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
        df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)
        price_chg = close.pct_change(5)
        rsi_chg = df["rsi_14"].diff(5)
        df["rsi_bearish_div"] = ((price_chg > 0) & (rsi_chg < 0)).astype(int)

        # MACD
        macd = ta.trend.MACD(close,
                             window_slow=self.macd_slow,
                             window_fast=self.macd_fast,
                             window_sign=self.macd_signal)
        df["macd"] = macd.macd()
        df["macd_signal_line"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()
        df["macd_hist_slope"] = df["macd_hist"].diff()

        # Stochastic RSI
        stoch = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
        df["stoch_rsi_k"] = stoch.stochrsi_k()
        df["stoch_rsi_d"] = stoch.stochrsi_d()
        df["stoch_overbought"] = (df["stoch_rsi_k"] > 0.8).astype(int)

        # Rate of Change
        df["roc_4"] = ta.momentum.ROCIndicator(close, window=4).roc()
        df["roc_12"] = ta.momentum.ROCIndicator(close, window=12).roc()

        # Williams %R
        df["williams_r"] = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()

        return df

    # ──────────────────────────────────────────────────────────────
    # 3. VOLATILITY
    # ──────────────────────────────────────────────────────────────

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        df["atr_14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
        df["atr_pct"] = df["atr_14"] / close
        df["atr_ratio"] = df["atr_14"] / df["atr_14"].rolling(50).mean()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=self.bb_period, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_pct"] = bb.bollinger_pband()        
        df["bb_squeeze"] = (df["bb_width"] < df["bb_width"].rolling(50).mean() * 0.7).astype(int)
        log_ret = np.log(close / close.shift(1))
        df["hist_vol_20"] = log_ret.rolling(20).std() * np.sqrt(20)
        kc = ta.volatility.KeltnerChannel(high, low, close, window=20)
        df["kc_upper"] = kc.keltner_channel_hband()
        df["kc_lower"] = kc.keltner_channel_lband()
        
        # TTM Squeeze (BB inside KC)
        df["ttm_squeeze"] = ((df["bb_upper"] < df["kc_upper"]) & (df["bb_lower"] > df["kc_lower"])).astype(int)

        return df

    # ──────────────────────────────────────────────────────────────
    # 4. VOLUME
    # ──────────────────────────────────────────────────────────────

    def _add_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        volume = df["volume"]
        df["volume_ratio"] = volume / volume.rolling(20).mean()
        
        # VZO (Volume Zone Oscillator)
        sign = np.sign(close - close.shift(1))
        r = sign * volume
        vp = ta.trend.ema_indicator(r, window=14)
        tv = ta.trend.ema_indicator(volume, window=14)
        df["vzo"] = (100 * (vp / tv)).fillna(0)
        
        # Cumulative Volume Delta (CVD) - Восстановлено!
        body = df["close"] - df["open"]
        candle_range = (df["high"] - df["low"]).replace(0, np.nan)
        df["cvd_proxy"] = (body / candle_range * volume).fillna(0)
        df["cvd_cumsum"] = df["cvd_proxy"].rolling(20).sum()

        return df

    # ──────────────────────────────────────────────────────────────
    # 5. MARKET STRUCTURE
    # ──────────────────────────────────────────────────────────────

    def _add_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        open_ = df["open"]

        # ── Wick ratios ──────────────────────────────────────────
        candle_range = (high - low).replace(0, np.nan)
        body = (close - open_).abs()
        upper_wick = high - np.maximum(open_, close)
        lower_wick = np.minimum(open_, close) - low

        df["upper_wick_ratio"] = (upper_wick / candle_range).fillna(0)
        df["lower_wick_ratio"] = (lower_wick / candle_range).fillna(0)
        df["body_ratio"] = (body / candle_range).fillna(0)
        df["is_shooting_star"] = (
            (df["upper_wick_ratio"] > 0.6) &
            (df["body_ratio"] < 0.3)
        ).astype(int)
        df["is_bearish"] = (close < open_).astype(int)
        df["bearish_sequence"] = (
            df["is_bearish"].rolling(3).sum() >= 2
        ).astype(int)

        # ── Swing High/Low ───────────────────────────────────────
        n = 2
        df["swing_high"] = (
            (high == high.rolling(2 * n + 1, center=True).max())
        ).astype(int)
        df["swing_low"] = (
            (low == low.rolling(2 * n + 1, center=True).min())
        ).astype(int)
        last_swing_high = high.where(df["swing_high"] == 1).ffill()
        df["dist_from_swing_high"] = (last_swing_high - close) / (df["atr_14"] + 1e-8)
        last_swing_low = low.where(df["swing_low"] == 1).ffill()
        df["dist_from_swing_low"] = (close - last_swing_low) / (df["atr_14"] + 1e-8)

        # ── Higher High / Lower Low ───────────────────────────────
        df["higher_high"] = (high > high.shift(1)).astype(int)
        df["lower_low"] = (low < low.shift(1)).astype(int)
        roll_high = high.rolling(20).max()
        roll_low = low.rolling(20).min()
        roll_range = (roll_high - roll_low).replace(0, np.nan)
        df["price_position_20"] = ((close - roll_low) / roll_range).fillna(0.5)

        # ── Gaps ──────────────────────────────────────────────────
        df["gap_up"] = ((open_ > high.shift(1)) * 1).astype(int)
        df["gap_down"] = ((open_ < low.shift(1)) * 1).astype(int)

        return df

    # ──────────────────────────────────────────────────────────────
    # 6. FUNDING RATE
    # ──────────────────────────────────────────────────────────────

    def _add_funding(self, df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
        """[Translated]"""
        funding_df = funding_df.sort_values("timestamp").copy()
        funding_df["timestamp"] = pd.to_datetime(funding_df["timestamp"], utc=True)
        funding_df = funding_df.set_index("timestamp")
        df = df.sort_values("timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        ts_index = pd.DatetimeIndex(df["timestamp"])
        df["funding_rate"] = (
            funding_df["funding_rate"]
            .reindex(ts_index, method="ffill")
            .values
        )
        df["funding_rate"] = df["funding_rate"].fillna(0.0)
        df["funding_cumsum_24h"] = df["funding_rate"].rolling(6).sum()
        df["funding_extreme"] = (df["funding_rate"] > df["funding_rate"].rolling(100).mean()
                                 + 2 * df["funding_rate"].rolling(100).std()).astype(int)

        return df

    # ──────────────────────────────────────────────────────────────
    # 7. TIME FEATURES
    # ──────────────────────────────────────────────────────────────

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = df["timestamp"]

        df["hour"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek  # 0=Monday
        df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
        # Asia:   00–08
        # Europe: 08–16
        # US:     16–24
        hour = ts.dt.hour
        df["session_asia"] = ((hour >= 0) & (hour < 8)).astype(int)
        df["session_europe"] = ((hour >= 8) & (hour < 16)).astype(int)
        df["session_us"] = ((hour >= 16) & (hour < 24)).astype(int)
        hours_since_funding = hour % 8
        df["hours_to_funding"] = 8 - hours_since_funding

        return df

    # ──────────────────────────────────────────────────────────────
    # 8. MULTI-TIMEFRAME
    # ──────────────────────────────────────────────────────────────

    def _add_multitf_1h(self, df_4h: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
        df_1h = df_1h.copy().sort_values("timestamp")
        df_1h["timestamp"] = pd.to_datetime(df_1h["timestamp"], utc=True)
        ema50_1h = ta.trend.ema_indicator(df_1h["close"], window=50)
        rsi_1h = ta.momentum.rsi(df_1h["close"], window=14)
        df_1h["trend_1h"] = np.where(df_1h["close"] > ema50_1h, 1, -1)
        df_1h["rsi_1h"] = rsi_1h
        df_1h = df_1h.set_index("timestamp")[["trend_1h", "rsi_1h"]]
        df_4h = df_4h.sort_values("timestamp")
        df_4h["timestamp"] = pd.to_datetime(df_4h["timestamp"], utc=True)
        ts_index_4h = pd.DatetimeIndex(df_4h["timestamp"])
        for col in ["trend_1h", "rsi_1h"]:
            df_4h[col] = df_1h[col].reindex(ts_index_4h, method="ffill").values
            df_4h[col] = df_4h[col].fillna(0)
        return df_4h

    def _add_multitf_1d(self, df_4h: pd.DataFrame, df_1d: pd.DataFrame) -> pd.DataFrame:
        df_1d = df_1d.copy().sort_values("timestamp")
        df_1d["timestamp"] = pd.to_datetime(df_1d["timestamp"], utc=True)
        ema200_1d = ta.trend.ema_indicator(df_1d["close"], window=200)
        rsi_1d = ta.momentum.rsi(df_1d["close"], window=14)
        df_1d["trend_1d"] = np.where(df_1d["close"] > ema200_1d, 1, -1)
        df_1d["rsi_1d"] = rsi_1d
        df_1d = df_1d.set_index("timestamp")[["trend_1d", "rsi_1d"]]
        df_4h = df_4h.sort_values("timestamp")
        df_4h["timestamp"] = pd.to_datetime(df_4h["timestamp"], utc=True)
        ts_index_4h = pd.DatetimeIndex(df_4h["timestamp"])
        for col in ["trend_1d", "rsi_1d"]:
            df_4h[col] = df_1d[col].reindex(ts_index_4h, method="ffill").values
            df_4h[col] = df_4h[col].fillna(0)
        return df_4h

    # ──────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> list[str]:
        exclude = {"timestamp", "open", "high", "low", "close", "volume",
                   "label", "label_return", "label_outcome", "label_bars",
                   "tp_price", "sl_price"}
        return [c for c in df.columns if c not in exclude]
