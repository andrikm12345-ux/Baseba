from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from loguru import logger

from src.config import MODELS_DIR
from src.data.features import FEATURE_COLUMNS

# Pairs of (home_col, away_col) that must be swapped to mirror the game perspective.
_SWAP_PAIRS = [
    ("home_elo", "away_elo"),
    ("home_win_rate", "away_win_rate"),
    ("home_rs_avg", "away_rs_avg"),
    ("home_ra_avg", "away_ra_avg"),
    ("home_rs_home_avg", "away_rs_away_avg"),
    ("home_ra_home_avg", "away_ra_away_avg"),
    ("h2h_home_avg_runs", "h2h_away_avg_runs"),
    ("home_pitcher_era", "away_pitcher_era"),
    ("home_pitcher_whip", "away_pitcher_whip"),
    ("home_pitcher_k9", "away_pitcher_k9"),
    ("home_pitcher_bb9", "away_pitcher_bb9"),
]
# Columns whose sign must be flipped (perspective reversal)
_NEGATE_COLS = ["elo_diff", "era_diff", "whip_diff", "rest_diff"]
# Columns that represent home win-rate fractions: mirror = 1 - value
_FLIP_RATE_COLS = ["h2h_home_winrate", "h2h_recency_winrate"]


def _mirror_features(X: np.ndarray) -> np.ndarray:
    """Return feature matrix with home/away swapped to predict P(away wins by 2+)."""
    col = {f: i for i, f in enumerate(FEATURE_COLUMNS)}
    X2 = X.copy()
    for a, b in _SWAP_PAIRS:
        if a in col and b in col:
            X2[:, col[a]] = X[:, col[b]]
            X2[:, col[b]] = X[:, col[a]]
    for c in _NEGATE_COLS:
        if c in col:
            X2[:, col[c]] = -X[:, col[c]]
    for c in _FLIP_RATE_COLS:
        if c in col:
            X2[:, col[c]] = 1.0 - X[:, col[c]]
    return X2


async def restore_models_from_db() -> bool:
    """Download model blobs from DB to disk. Returns True if the 3 core models restored."""
    try:
        from src.data.database import ModelBlob, SessionLocal
        restored = 0
        async with SessionLocal() as session:
            for name, filename in [
                ("model_ml", "model_ml.joblib"),
                ("model_total", "model_total.joblib"),
                ("model_rl", "model_rl.joblib"),
                ("model_itb", "model_itb.joblib"),
            ]:
                path = MODELS_DIR / filename
                if path.exists():
                    restored += 1
                    continue
                blob = await session.get(ModelBlob, name)
                if blob and blob.data:
                    MODELS_DIR.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(blob.data)
                    logger.info(f"Model '{name}' restored from DB ({len(blob.data)//1024} KB)")
                    restored += 1
        return restored >= 3
    except Exception as e:
        logger.warning(f"restore_models_from_db failed: {e}")
        return False


class Predictor:
    def __init__(self) -> None:
        self.m_ml = self._load(MODELS_DIR / "model_ml.joblib")
        self.m_total = self._load(MODELS_DIR / "model_total.joblib")
        self.m_rl = self._load(MODELS_DIR / "model_rl.joblib")
        self.m_itb = self._load(MODELS_DIR / "model_itb.joblib")

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
        X_mirror = _mirror_features(X)
        p_rl_away = self.m_rl["model"].predict_proba(X_mirror)[:, 1]  # P(away covers -1.5)
        out = features_df[["match_id"]].copy()
        out["p_home"] = p_ml
        out["p_away"] = 1.0 - p_ml
        out["p_over85"] = p_total
        out["p_rl_home"] = p_rl
        out["p_rl_away"] = p_rl_away
        if self.m_itb:
            p_itb_home = self.m_itb["model"].predict_proba(X)[:, 1]
            p_itb_away = self.m_itb["model"].predict_proba(_mirror_features(X))[:, 1]
            out["p_itb_home"] = p_itb_home
            out["p_itb_away"] = p_itb_away
        else:
            out["p_itb_home"] = 0.0
            out["p_itb_away"] = 0.0
        return out
