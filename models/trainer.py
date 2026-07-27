"""[Translated]"""

from __future__ import annotations

import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config_loader import get_config
from features.engineer import FeatureEngineer


class WalkForwardTrainer:
    """[Translated]"""

    def __init__(self):
        cfg = get_config()
        self.mcfg = cfg["model"]
        self.saved_dir = Path("models/saved")
        self.saved_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────

    def train(
        self,
        df: pd.DataFrame,
        cutoff_date: Optional[datetime] = None,
        verbose: bool = True,
    ) -> lgb.LGBMClassifier:
        """[Translated]"""
        if cutoff_date is None:
            cutoff_date = datetime.now(timezone.utc)

        window_days = self.mcfg["train_window_days"]
        start_date = cutoff_date - timedelta(days=window_days)
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        mask = (df["timestamp"] >= start_date) & (df["timestamp"] < cutoff_date)
        train_df = df[mask].copy()

        if len(train_df) < self.mcfg["min_samples"]:
            raise ValueError(
                f""
                f""
            )

        logger.info(
            f""
            f""
            f""
            f""
        )
        feature_cols = FeatureEngineer.get_feature_columns(train_df)
        X = train_df[feature_cols].copy()
        cat_cols = X.select_dtypes(include=['object', 'string']).columns
        for col in cat_cols:
            X[col] = X[col].astype('category')
                
        y = train_df["label"].values
        sample_weights = self._compute_sample_weights(train_df["timestamp"])
        split_idx = int(len(train_df) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        w_train = sample_weights[:split_idx]

        # ── LightGBM ─────────────────────────────────────────────
        params = dict(self.mcfg["lgbm_params"])

        model = lgb.LGBMClassifier(**params)

        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        if verbose:
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]

            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_val, y_proba)
            except ValueError:
                auc = 0.5

            logger.info(
                f"Validation: Precision={precision:.3f} | "
                f"Recall={recall:.3f} | AUC={auc:.3f} | "
                f"Best iteration: {model.best_iteration_}"
            )
            if verbose:
                logger.info("\n" + classification_report(y_val, y_pred, target_names=["No Short", "Short"]))
            self._log_feature_importance(model, feature_cols)

        return model

    # ──────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────

    def _compute_sample_weights(self, timestamps: pd.Series) -> np.ndarray:
        """[Translated]"""
        halflife = self.mcfg["sample_weight_halflife_days"]

        latest = timestamps.max()
        days_ago = (latest - timestamps).dt.total_seconds() / 86400

        # w = 2^(-days_ago / halflife)
        weights = np.power(2.0, -days_ago.values / halflife)
        weights = weights / weights.sum() * len(weights)

        return weights

    # ──────────────────────────────────────────────────────────────
    # Feature Importance
    # ──────────────────────────────────────────────────────────────

    def _log_feature_importance(
        self, model: lgb.LGBMClassifier, feature_names: list[str]
    ) -> None:
        importance = pd.Series(
            model.feature_importances_, index=feature_names
        ).sort_values(ascending=False)

        logger.info("")
        for feat, imp in importance.head(20).items():
            logger.info(f"  {feat:40s} {imp:6.0f}")

    # ──────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────

    def save_model(
        self, model: lgb.LGBMClassifier, feature_cols: list[str]
    ) -> Path:
        """[Translated]"""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self.saved_dir / f"model_{ts}.pkl"

        payload = {
            "model": model,
            "feature_cols": feature_cols,
            "trained_at": datetime.now(timezone.utc),
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f)
        latest_path = self.saved_dir / "model_latest.pkl"
        with open(latest_path, "wb") as f:
            pickle.dump(payload, f)

        logger.info(f"")
        return path

    def load_latest_model(self) -> tuple[lgb.LGBMClassifier, list[str]]:
        """[Translated]"""
        path = self.saved_dir / "model_latest.pkl"

        if not path.exists():
            raise FileNotFoundError(
                ""
                "poetry run python -m scripts.initial_train"
            )

        with open(path, "rb") as f:
            payload = pickle.load(f)

        trained_at = payload.get("trained_at", "unknown")
        logger.info(f"")

        return payload["model"], payload["feature_cols"]

    # ──────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────

    def check_drift(self, recent_trades: pd.DataFrame) -> bool:
        """[Translated]"""
        n = self.mcfg["drift_check_last_n"]
        threshold = self.mcfg["drift_precision_threshold"]

        if len(recent_trades) < n:
            return False

        recent = recent_trades.tail(n)
        precision = precision_score(
            recent["label"], recent["prediction"], zero_division=0
        )

        if precision < threshold:
            logger.warning(
                f""
                f""
            )
            return True

        logger.debug(f"")
        return False


class WalkForwardBacktestTrainer:
    """[Translated]"""

    def __init__(self):
        self.trainer = WalkForwardTrainer()
        cfg = get_config()["model"]
        self.train_window = cfg["train_window_days"]
        self.step_days = cfg["retrain_every_days"]

    def run(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict]:
        """[Translated]"""
        results = []
        current = start_date + timedelta(days=self.train_window)

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        while current <= end_date:
            logger.info(f"")

            try:
                model = self.trainer.train(df, cutoff_date=current, verbose=False)
                feature_cols = FeatureEngineer.get_feature_columns(df)
                next_date = current + timedelta(days=self.step_days)
                val_mask = (df["timestamp"] >= current) & (df["timestamp"] < next_date)
                val_df = df[val_mask].copy()

                if len(val_df) > 0:
                    X_val = val_df[feature_cols].copy()
                    
                    cat_cols = X_val.select_dtypes(include=['object', 'string']).columns
                    for col in cat_cols:
                        X_val[col] = X_val[col].astype('category')
                        
                    val_df["prediction"] = model.predict(X_val)
                    val_df["confidence"] = model.predict_proba(X_val)[:, 1]

                    results.append({
                        "cutoff": current,
                        "model": model,
                        "feature_cols": feature_cols,
                        "val_df": val_df,
                        "n_train": len(df[df["timestamp"] < current]),
                        "n_val": len(val_df),
                    })

            except ValueError as e:
                logger.warning(f"")

            current += timedelta(days=self.step_days)

        logger.info(f"")
        return results
