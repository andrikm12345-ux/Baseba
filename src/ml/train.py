"""Train three calibrated XGBoost models for MLB baseball: ML, TOTAL, RL."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from src.config import MODELS_DIR
from src.data.features import FEATURE_COLUMNS


async def _save_model_to_db(name: str, path: Path) -> None:
    """Save a joblib model file as binary blob in the database."""
    try:
        from src.data.database import ModelBlob, SessionLocal
        data = path.read_bytes()
        async with SessionLocal() as session:
            existing = await session.get(ModelBlob, name)
            if existing is None:
                session.add(ModelBlob(name=name, data=data))
            else:
                existing.data = data
                from datetime import datetime
                existing.updated_at = datetime.utcnow()
            await session.commit()
        logger.info(f"Model '{name}' saved to database ({len(data)//1024} KB)")
    except Exception as e:
        logger.warning(f"Could not save model '{name}' to DB: {e}")


def _make_estimator() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
    )


def _train_one(X: pd.DataFrame, y: np.ndarray, name: str) -> CalibratedClassifierCV:
    base = _make_estimator()
    cal = CalibratedClassifierCV(base, method="isotonic", cv=TimeSeriesSplit(n_splits=4))
    cal.fit(X, y)
    proba = cal.predict_proba(X)[:, 1]
    bs = brier_score_loss(y, proba)
    logger.info(f"[{name}] in-sample Brier={bs:.4f}")
    return cal


async def save_all_to_db() -> None:
    """Save all model files to DB (call after train_all)."""
    for name, filename in [
        ("model_ml", "model_ml.joblib"),
        ("model_total", "model_total.joblib"),
        ("model_rl", "model_rl.joblib"),
        ("model_itb", "model_itb.joblib"),
    ]:
        p = MODELS_DIR / filename
        if p.exists():
            await _save_model_to_db(name, p)


def train_all(features_df: pd.DataFrame) -> Dict[str, Any]:
    if features_df.empty or len(features_df) < 200:
        raise RuntimeError(
            f"Not enough training data ({len(features_df)} rows). "
            "Run the history ingest first."
        )
    features_df = features_df.dropna(subset=["ml_home", "over85", "rl_home", "itb_home", "itb_away"]).reset_index(drop=True)
    X = features_df[FEATURE_COLUMNS].astype(float)
    y_ml = features_df["ml_home"].astype(int).values
    y_total = features_df["over85"].astype(int).values
    y_rl = features_df["rl_home"].astype(int).values
    y_itb_home = features_df["itb_home"].astype(int).values

    paths: Dict[str, Path] = {}

    logger.info(f"Training ML (moneyline) on {len(X)} rows")
    m_ml = _train_one(X, y_ml, "ML")
    p = MODELS_DIR / "model_ml.joblib"
    joblib.dump({"model": m_ml, "features": FEATURE_COLUMNS}, p)
    paths["ML"] = p

    logger.info(f"Training TOTAL (over {len(X)} rows)")
    m_total = _train_one(X, y_total, "TOTAL")
    p = MODELS_DIR / "model_total.joblib"
    joblib.dump({"model": m_total, "features": FEATURE_COLUMNS}, p)
    paths["TOTAL"] = p

    logger.info(f"Training RL (run line) on {len(X)} rows")
    m_rl = _train_one(X, y_rl, "RL")
    p = MODELS_DIR / "model_rl.joblib"
    joblib.dump({"model": m_rl, "features": FEATURE_COLUMNS}, p)
    paths["RL"] = p

    logger.info(f"Training ITB (individual team total) on {len(X)} rows")
    m_itb = _train_one(X, y_itb_home, "ITB")
    p = MODELS_DIR / "model_itb.joblib"
    joblib.dump({"model": m_itb, "features": FEATURE_COLUMNS}, p)
    paths["ITB"] = p

    metrics_inn = {
        "n_train": len(X),
        "ml_brier": float(brier_score_loss(y_ml, m_ml.predict_proba(X)[:, 1])),
        "total_brier": float(brier_score_loss(y_total, m_total.predict_proba(X)[:, 1])),
        "rl_brier": float(brier_score_loss(y_rl, m_rl.predict_proba(X)[:, 1])),
        "itb_brier": float(brier_score_loss(y_itb_home, m_itb.predict_proba(X)[:, 1])),
    }
    walk = evaluate_walk_forward(features_df)

    top_features: list[str] = []
    try:
        base = m_ml.calibrated_classifiers_[0].estimator
        imp = sorted(
            zip(FEATURE_COLUMNS, base.feature_importances_),
            key=lambda x: -x[1],
        )[:5]
        top_features = [f"{name} ({score:.2f})" for name, score in imp]
    except Exception as e:
        logger.warning(f"feature_importance extract failed: {e}")

    last_path = MODELS_DIR / "_last_metrics.json"
    prev: Dict[str, float] = {}
    if last_path.exists():
        try:
            prev = json.loads(last_path.read_text())
        except Exception:
            prev = {}
    diff = {
        k: metrics_inn[k] - prev.get(k, metrics_inn[k])
        for k in metrics_inn
        if k != "n_train" and isinstance(metrics_inn[k], float)
    }
    try:
        last_path.write_text(json.dumps(metrics_inn, indent=2))
    except Exception as e:
        logger.warning(f"could not save _last_metrics.json: {e}")

    return {
        "paths": paths,
        "metrics": {
            **metrics_inn,
            "walk_forward": walk,
            "top_features": top_features,
            "diff_vs_prev": diff,
        },
    }


def evaluate_walk_forward(features_df: pd.DataFrame) -> Dict[str, float]:
    """Out-of-sample evaluation via expanding-window CV."""
    features_df = features_df.dropna(subset=["ml_home", "over85", "rl_home", "itb_home", "itb_away"]).reset_index(drop=True)
    if len(features_df) < 400:
        logger.warning("Not enough rows for walk-forward eval")
        return {}
    X = features_df[FEATURE_COLUMNS].astype(float).values
    y_ml = features_df["ml_home"].astype(int).values
    y_total = features_df["over85"].astype(int).values
    y_rl = features_df["rl_home"].astype(int).values
    y_itb_home = features_df["itb_home"].astype(int).values
    tscv = TimeSeriesSplit(n_splits=5)
    metrics: Dict[str, list] = {"ml_brier": [], "total_brier": [], "rl_brier": [], "itb_brier": []}
    for tr, te in tscv.split(X):
        m = _make_estimator().fit(X[tr], y_ml[tr])
        metrics["ml_brier"].append(brier_score_loss(y_ml[te], m.predict_proba(X[te])[:, 1]))
        m = _make_estimator().fit(X[tr], y_total[tr])
        metrics["total_brier"].append(brier_score_loss(y_total[te], m.predict_proba(X[te])[:, 1]))
        m = _make_estimator().fit(X[tr], y_rl[tr])
        metrics["rl_brier"].append(brier_score_loss(y_rl[te], m.predict_proba(X[te])[:, 1]))
        m = _make_estimator().fit(X[tr], y_itb_home[tr])
        metrics["itb_brier"].append(brier_score_loss(y_itb_home[te], m.predict_proba(X[te])[:, 1]))
    return {k: float(np.mean(v)) for k, v in metrics.items()}
