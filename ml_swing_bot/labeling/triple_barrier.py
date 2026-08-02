"""[Translated]"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from shared.config_loader import get_config


class TripleBarrierLabeler:
    """[Translated]"""

    def __init__(self):
        cfg = get_config()["labeling"]
        self.atr_period = cfg["atr_period"]       # 14
        self.tp_atr_mult = cfg["tp_atr_mult"]   
        self.sl_atr_mult = cfg["sl_atr_mult"]   
        self.max_bars = cfg["max_bars"]          
        self.min_rr = cfg["min_rr"]             

    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        """[Translated]"""
        assert "atr_14" in df.columns, ""
        assert "close" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns

        df = df.copy().reset_index(drop=True)

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        atrs = df["atr_14"].values

        n = len(df)
        labels = np.zeros(n, dtype=int)
        outcomes = np.full(n, "timeout", dtype=object)
        bars_to_outcome = np.full(n, self.max_bars, dtype=int)
        returns = np.zeros(n, dtype=float)
        tp_prices = np.zeros(n, dtype=float)
        sl_prices = np.zeros(n, dtype=float)
        valid_end = n - self.max_bars

        for i in range(valid_end):
            entry = closes[i]
            atr = atrs[i]

            if atr <= 0 or np.isnan(atr):
                continue

            tp_dist = atr * self.tp_atr_mult 
            sl_dist = atr * self.sl_atr_mult 

            tp_price = entry - tp_dist        
            sl_price = entry + sl_dist        

            tp_prices[i] = tp_price
            sl_prices[i] = sl_price
            rr = tp_dist / sl_dist
            if rr < self.min_rr:
                outcomes[i] = "low_rr"
                continue
            outcome = "timeout"
            bars = self.max_bars

            for j in range(1, self.max_bars + 1):
                if i + j >= n:
                    break

                future_low = lows[i + j]
                future_high = highs[i + j]
                if future_low <= tp_price:
                    outcome = "tp"
                    bars = j
                    break
                if future_high >= sl_price:
                    outcome = "sl"
                    bars = j
                    break

            labels[i] = 1 if outcome == "tp" else 0
            outcomes[i] = outcome
            bars_to_outcome[i] = bars
            exit_close = closes[min(i + bars, n - 1)]
            returns[i] = (entry - exit_close) / entry

        df["label"] = labels
        df["label_outcome"] = outcomes
        df["label_bars"] = bars_to_outcome
        df["label_return"] = returns
        df["tp_price"] = tp_prices
        df["sl_price"] = sl_prices
        df = df.iloc[:valid_end].copy()
        total = len(df)
        labeled_1 = df["label"].sum()
        labeled_0 = total - labeled_1
        tp_count = (df["label_outcome"] == "tp").sum()
        sl_count = (df["label_outcome"] == "sl").sum()
        timeout_count = (df["label_outcome"] == "timeout").sum()

        logger.info(
            f""
            f""
            f""
            f"TP: {tp_count}, SL: {sl_count}, Timeout: {timeout_count}"
        )

        return df

    def label_realtime(
        self,
        entry_price: float,
        atr: float,
    ) -> dict:
        """[Translated]"""
        tp_price = entry_price - atr * self.tp_atr_mult
        sl_price = entry_price + atr * self.sl_atr_mult
        rr = (entry_price - tp_price) / (sl_price - entry_price)

        return {
            "tp_price": tp_price,
            "sl_price": sl_price,
            "rr": rr,
            "tp_dist_pct": (entry_price - tp_price) / entry_price,
            "sl_dist_pct": (sl_price - entry_price) / entry_price,
        }

    def get_class_weights(self, df: pd.DataFrame) -> dict:
        """[Translated]"""
        from sklearn.utils.class_weight import compute_class_weight
        weights = compute_class_weight(
            "balanced",
            classes=np.array([0, 1]),
            y=df["label"].values
        )
        return {0: weights[0], 1: weights[1]}
