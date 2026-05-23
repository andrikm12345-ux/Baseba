from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from loguru import logger

from src.config import MODELS_DIR
from src.data.features import FEATURE_COLUMNS


class Predictor:
    def __init__(self) -> None:
        self.m_ml = self._load(MODELS_DIR / "model_ml.joblib")
        self.m_total = self._load(MODELS_DIR / "model_total.joblib")
        self.m_rl = self._load(MODELS_DIR / "model_rl.joblib")

    @staticmethod
    def _load(path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        try:
            d = joblib.load(path)
            if d.get("features") != FEATURE_COLUMNS:
                logger.warning(f"{path.name}: feature mismatch — retraining required")
                return None
            return d
        except Exception as e:
            logger.error(f"Cannot load {path}: {e}")
            return None

    @property
    def ready(self) -> bool:
        return all([self.m_ml, self.m_total, self.m_rl])

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        if not self.ready:
            raise RuntimeError("Models not trained yet. Run training first.")
        X = features_df[FEATURE_COLUMNS].astype(float).values
        p_ml = self.m_ml["model"].predict_proba(X)[:, 1]      # P(home wins)
        p_total = self.m_total["model"].predict_proba(X)[:, 1] # P(over 8.5 runs)
        p_rl = self.m_rl["model"].predict_proba(X)[:, 1]      # P(home covers -1.5)
        out = features_df[["match_id"]].copy()
        out["p_home"] = p_ml
        out["p_away"] = 1.0 - p_ml
        out["p_over85"] = p_total
        out["p_rl_home"] = p_rl
        return out
