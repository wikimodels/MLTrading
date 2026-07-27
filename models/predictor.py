"""[Translated]"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger

from models.trainer import WalkForwardTrainer
from config_loader import get_config


class Predictor:
    """[Translated]"""

    def __init__(self):
        self.cfg = get_config()["model"]
        self.trainer = WalkForwardTrainer()
        self.model: lgb.LGBMClassifier | None = None
        self.feature_cols: list[str] | None = None
        self._load()

    def _load(self) -> None:
        """[Translated]"""
        try:
            self.model, self.feature_cols = self.trainer.load_latest_model()
        except FileNotFoundError:
            logger.warning("")

    def reload(self) -> None:
        """[Translated]"""
        self.model, self.feature_cols = self.trainer.load_latest_model()
        logger.info("")

    def predict(self, df_bar: pd.DataFrame) -> dict:
        """[Translated]"""
        if self.model is None:
            logger.error("")
            return {"trade": False, "confidence": 0.0, "label": 0}

        X = df_bar[self.feature_cols].values
        proba = self.model.predict_proba(X)[:, 1]
        label = (proba >= self.cfg["min_confidence"]).astype(int)
        idx = -1
        confidence = float(proba[idx])
        trade = bool(label[idx])
        key_features = {}
        for feat in ["rsi_14", "market_regime", "breadth_down_ratio",
                     "avg_rsi_top20", "atr_pct", "upper_wick_ratio",
                     "rs_vs_btc", "funding_rate"]:
            if feat in df_bar.columns:
                key_features[feat] = float(df_bar[feat].iloc[idx])

        result = {
            "trade": trade,
            "confidence": confidence,
            "label": int(label[idx]),
            "features": key_features,
        }

        if trade:
            logger.info(
                f""
                + " | ".join(f"{k}={v:.3f}" for k, v in key_features.items())
            )
        else:
            logger.debug(f"")

        return result

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """[Translated]"""
        if self.model is None:
            raise RuntimeError("")

        X = df[self.feature_cols].values
        proba = self.model.predict_proba(X)[:, 1]
        label = (proba >= self.cfg["min_confidence"]).astype(int)

        df = df.copy()
        df["prediction"] = label
        df["confidence"] = proba
        return df
