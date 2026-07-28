"""Feature engineering for ML Trading Bot."""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta

from config_loader import get_config


class FeatureEngineer:
    """Computes all features for the ML model."""

    def __init__(self):
        cfg = get_config()
        feat_cfg = cfg["features"]
        self.atr_period = feat_cfg["atr_period"]
        self.rsi_period = feat_cfg["rsi_period"]
        self.ema_periods = feat_cfg["ema_periods"]   # [21, 50, 200]
        self.vwma_periods = feat_cfg.get("vwma_periods", [21, 50, 200])
        self.bb_period = feat_cfg["bb_period"]
        self.macd_fast = feat_cfg["macd_fast"]
        self.macd_slow = feat_cfg["macd_slow"]
        self.macd_signal = feat_cfg["macd_signal"]
        self.use_weekly_tf = feat_cfg.get("use_weekly_tf", False)

    # ──────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────

    def compute(
        self,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame | None = None,
        df_1d: pd.DataFrame | None = None,
        funding_df: pd.DataFrame | None = None,
        df_1w: pd.DataFrame | None = None,
        symbol_id: int = 0,
    ) -> pd.DataFrame:
        """
        Compute all features for one symbol.

        Weekly TF note: df_1w uses only COMPLETED weekly candles (shift=1).
        Any 4H bar inside week N will see week N-1 features, never the
        current forming week. See _add_multitf_1w() for details.
        """
        df = df_4h.copy()
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"

        df = df.sort_values("timestamp").reset_index(drop=True)

        # ── 1. Trend ──────────────────────────────────────────────
        df = self._add_trend(df)

        # ── 2. Momentum ───────────────────────────────────────────
        df = self._add_momentum(df)
        df = self._add_volatility(df)
        df = self._add_volume(df)
        df = self._add_candlesticks(df)
        df = self._add_time_features(df)

        # ── 5. Market Structure ───────────────────────────────────
        df = self._add_structure(df)

        # ── 6. Funding Rate ───────────────────────────────────────
        if funding_df is not None:
            df = self._add_funding(df, funding_df)
        else:
            df["funding_rate"] = 0.0
            df["funding_cumsum_24h"] = 0.0

        # ── 8. Multi-TF: 1H and 1D ────────────────────────────────
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

        # ── 9. Weekly TF (only COMPLETED weeks) ───────────────────
        if df_1w is not None and self.use_weekly_tf:
            df = self._add_multitf_1w(df, df_1w)
        else:
            # Neutral defaults when weekly data is absent or disabled
            df["week_candle_bearish"] = 0
            df["week_rsi"] = 50.0
            df["week_trend"] = 0
            df["week_price_vs_ema21"] = 0.0
            df["week_body_ratio"] = 0.5

        df["symbol_id"] = symbol_id
        feat_cols = [c for c in df.columns if c not in ("timestamp", "open", "high", "low", "close", "volume")]
        df = df.dropna(subset=feat_cols).reset_index(drop=True)

        return df

    # ──────────────────────────────────────────────────────────────
    # 1. TREND
    # ──────────────────────────────────────────────────────────────

    def _add_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        for period in self.ema_periods:
            col = f"ema_{period}"
            df[col] = ta.trend.ema_indicator(close, window=period)
            df[f"price_vs_ema_{period}"] = (close - df[col]) / df[col]
            df[f"ema_{period}_slope"] = df[col].pct_change(3)

        # EMA crossovers
        if 21 in self.ema_periods and 50 in self.ema_periods:
            df["ema21_vs_ema50"] = (df["ema_21"] > df["ema_50"]).astype(int)
        if 50 in self.ema_periods and 200 in self.ema_periods:
            df["ema50_vs_ema200"] = (df["ema_50"] > df["ema_200"]).astype(int)

        # VWMA (Volume Weighted Moving Average)
        volume = df["volume"]
        price_vol = close * volume
        for period in getattr(self, "vwma_periods", []):
            vwma_col = f"vwma_{period}"
            df[vwma_col] = price_vol.rolling(window=period).sum() / volume.rolling(window=period).sum()
            df[f"price_vs_vwma_{period}"] = (close - df[vwma_col]) / df[vwma_col]
            
            # Разница между ценовой средней и объемной (показывает, подкреплен ли тренд объемами)
            if period in getattr(self, "ema_periods", []):
                ema_col = f"ema_{period}"
                df[f"ema_vs_vwma_{period}"] = (df[ema_col] - df[vwma_col]) / df[vwma_col]

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
        
        # Bearish Divergence (5-candle window)
        # Price is higher than 5 candles ago, but RSI is lower, AND RSI is in overbought territory (>60)
        price_higher = close > close.shift(5)
        rsi_lower = df["rsi_14"] < df["rsi_14"].shift(5)
        df["rsi_bear_div_5"] = (price_higher & rsi_lower & (df["rsi_14"] > 60)).astype(int)
        
        # Remove old rsi_bearish_div if it existed to avoid confusion
        if "rsi_bearish_div" in df.columns:
            df.drop(columns=["rsi_bearish_div"], inplace=True)

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
        
        # BB Squeeze Breakdown: Was in a squeeze recently (last 3 candles) but just broke lower band
        recent_squeeze = df["bb_squeeze"].rolling(3).max() == 1
        df["bb_squeeze_breakdown"] = (recent_squeeze & (close < df["bb_lower"])).astype(int)
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

        # Cumulative Volume Delta (CVD)
        body = df["close"] - df["open"]
        candle_range = (df["high"] - df["low"]).replace(0, np.nan)
        df["cvd_proxy"] = (body / candle_range * volume).fillna(0)
        df["cvd_cumsum"] = df["cvd_proxy"].rolling(20).sum()

        return df

    # ──────────────────────────────────────────────────────────────
    # 5. CANDLESTICK PATTERNS
    # ──────────────────────────────────────────────────────────────

    def _add_candlesticks(self, df: pd.DataFrame) -> pd.DataFrame:
        open_ = df["open"]
        close = df["close"]
        high = df["high"]
        low = df["low"]
        
        body_size = (close - open_).abs()
        candle_size = (high - low)
        upper_wick = high - df[["open", "close"]].max(axis=1)
        lower_wick = df[["open", "close"]].min(axis=1) - low
        
        # 1. Bearish Engulfing
        prev_open = open_.shift(1)
        prev_close = close.shift(1)
        prev_is_green = prev_close > prev_open
        curr_is_red = close < open_
        engulfing = prev_is_green & curr_is_red & (open_ > prev_close) & (close < prev_open)
        df["bearish_engulfing"] = engulfing.astype(int)
        
        # 2. Shooting Star / Pin Bar (Bearish)
        # Small body (<= 50% of total), long upper wick (>= 2x body), very short lower wick (<= 10% of total)
        small_body = body_size <= (candle_size * 0.5)
        long_upper = upper_wick >= (body_size * 2)
        short_lower = lower_wick <= (candle_size * 0.1)
        df["shooting_star"] = (small_body & long_upper & short_lower & (candle_size > 0)).astype(int)
        
        # 3. Doji
        # Body is extremely small (<= 10% of total candle size)
        df["doji"] = ((body_size <= (candle_size * 0.1)) & (candle_size > 0)).astype(int)
        
        return df

    # ──────────────────────────────────────────────────────────────
    # 6. MARKET BREADTH / BTC
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
        """Merge funding rate history into OHLCV using forward-fill."""
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
        
        # Funding Rate Spike (Derivative over 3 candles)
        df["funding_spike_3"] = df["funding_rate"].diff(3).fillna(0.0)

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
    # 8. MULTI-TIMEFRAME: 1H and 1D
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
    # 9. WEEKLY TF — only COMPLETED candles
    #
    # Key insight: shift(1) ensures we always use the PREVIOUS
    # completed week, NEVER the current forming week.
    #
    # Timeline example:
    #   Week 1: Mon Jan 01 → Sun Jan 07 (closes Sunday)
    #   Week 2: Mon Jan 08 → Sun Jan 14
    #
    #   4H bars during week 2 see ONLY week 1 features:
    #   - Mon Jan 08 00:00 → week 1 features  ✓
    #   - Thu Jan 11 12:00 → week 1 features  ✓
    #   - Sun Jan 14 20:00 → week 1 features  ✓
    #   - Mon Jan 15 00:00 → week 2 features  ✓ (new week started)
    #
    # Implementation:
    #   1. Compute indicators on raw 1W OHLCV
    #   2. shift(1) — each weekly bar now carries PREV week's values
    #   3. reindex(4H timestamps, ffill) — spreads prev week's values
    #      across all 4H bars until the next weekly bar appears
    # ──────────────────────────────────────────────────────────────

    def _add_multitf_1w(self, df_4h: pd.DataFrame, df_1w: pd.DataFrame) -> pd.DataFrame:
        """
        Merge weekly features into 4H bars using only completed weekly candles.
        Uses shift(1) to prevent lookahead bias on the weekly timeframe.
        """
        WEEKLY_FEAT_COLS = [
            "week_candle_bearish",   # 1 = last closed week was red (strong short signal)
            "week_rsi",              # RSI(14) on weekly closes
            "week_trend",            # 1 = above EMA21(W), -1 = below
            "week_price_vs_ema21",   # (close - EMA21) / EMA21 — magnitude of deviation
            "week_body_ratio",       # |close - open| / (high - low) — candle strength
        ]

        df_1w = df_1w.copy().sort_values("timestamp").reset_index(drop=True)
        df_1w["timestamp"] = pd.to_datetime(df_1w["timestamp"], utc=True)

        # ── Compute weekly indicators on actual price data ─────────
        ema21_1w = ta.trend.ema_indicator(df_1w["close"], window=21)
        rsi_1w   = ta.momentum.rsi(df_1w["close"], window=14)
        candle_range = (df_1w["high"] - df_1w["low"]).replace(0, np.nan)

        df_1w["week_candle_bearish"] = (df_1w["close"] < df_1w["open"]).astype(int)
        df_1w["week_rsi"]            = rsi_1w
        df_1w["week_trend"]          = np.where(df_1w["close"] > ema21_1w, 1, -1)
        df_1w["week_price_vs_ema21"] = ((df_1w["close"] - ema21_1w) / ema21_1w).fillna(0)
        df_1w["week_body_ratio"]     = ((df_1w["close"] - df_1w["open"]).abs() / candle_range).fillna(0)

        # ── SHIFT(1): push features forward by one weekly bar ──────
        # After shift, the row at timestamp T (start of week N) contains
        # the features of week N-1 (the last COMPLETED week).
        # The current forming week's data is NEVER used.
        df_1w[WEEKLY_FEAT_COLS] = df_1w[WEEKLY_FEAT_COLS].shift(1)

        # ── Reindex into 4H resolution via forward-fill ────────────
        # ffill propagates week N-1 features across all 4H bars of week N
        # until the weekly bar for week N+1 appears (at which point
        # week N features take over — but again shifted, so they are week N).
        df_1w_indexed = df_1w.set_index("timestamp")[WEEKLY_FEAT_COLS]

        df_4h = df_4h.sort_values("timestamp")
        df_4h["timestamp"] = pd.to_datetime(df_4h["timestamp"], utc=True)
        ts_index_4h = pd.DatetimeIndex(df_4h["timestamp"])

        for col in WEEKLY_FEAT_COLS:
            values = df_1w_indexed[col].reindex(ts_index_4h, method="ffill").values
            # Fill NaN at the very beginning (before first completed week)
            default = 50.0 if col == "week_rsi" else 0.0
            df_4h[col] = pd.Series(values, index=df_4h.index).fillna(default)

        return df_4h

    # ──────────────────────────────────────────────────────────────
    # Feature column list
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> list[str]:
        exclude = {"timestamp", "open", "high", "low", "close", "volume",
                   "label", "label_return", "label_outcome", "label_bars",
                   "tp_price", "sl_price", "symbol"}
        return [c for c in df.columns if c not in exclude]
