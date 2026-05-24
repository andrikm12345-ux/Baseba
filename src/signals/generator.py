"""Turn model probabilities into baseball bet signals.

Markets:
  ML    — Moneyline: HOME or AWAY wins
  TOTAL — Over/Under total runs (line = settings.total_line, default 8.5)
  RL    — Run Line: COVER (home −1.5) / LAY (away +1.5)
                    AWAY_COVER (away −1.5) / HOME_LAY (home +1.5)

A signal is emitted when:
  - model confidence >= MIN_CONFIDENCE
  - book odds available AND in [MIN_ODDS, MAX_ODDS]
  - edge (model_prob * book_odds - 1) >= MIN_EDGE

Without bookmaker odds a "model signal" is published at higher confidence threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from src.config import settings


@dataclass
class Signal:
    match_id: int
    market: str       # "ML" | "TOTAL" | "RL"
    pick: str         # "HOME"/"AWAY", "OVER"/"UNDER", "COVER"/"LAY"
    model_prob: float
    fair_odds: float
    book_odds: float  # 0.0 if unknown
    edge: float
    confidence: float
    stake_units: float
    is_value: bool


def _kelly(p: float, odds: float, fraction: float = 0.25, cap: float = 2.0) -> float:
    b = odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p
    k = (b * p - q) / b
    if k <= 0:
        return 0.0
    return min(k * fraction * 10, cap)


def _best_ml(row: pd.Series) -> tuple[str, float]:
    if row["p_home"] >= row["p_away"]:
        return "HOME", float(row["p_home"])
    return "AWAY", float(row["p_away"])


def _book_odds_for(row: pd.Series, market: str, pick: str) -> Optional[float]:
    col = {
        ("ML", "HOME"): "odds_ml_home",
        ("ML", "AWAY"): "odds_ml_away",
        ("TOTAL", "OVER"): "odds_over85",
        ("TOTAL", "UNDER"): "odds_under85",
        ("RL", "COVER"): "odds_rl_home",
        ("RL", "LAY"): "odds_rl_away",
        ("RL", "AWAY_COVER"): "odds_rl_away_cover",
        ("RL", "HOME_LAY"): "odds_rl_home_lay",
    }.get((market, pick))
    if col and col in row and pd.notna(row[col]) and row[col] > 1.0:
        return float(row[col])
    return None


def generate(predictions_with_odds: pd.DataFrame) -> List[Signal]:
    """predictions_with_odds expects: match_id, p_home, p_away, p_over85,
    plus optional odds_ml_home, odds_ml_away, odds_over85, odds_under85 columns."""
    out: List[Signal] = []
    for _, row in predictions_with_odds.iterrows():
        # Moneyline
        ml_pick, ml_prob = _best_ml(row)
        out.extend(_make_signal(row, "ML", ml_pick, ml_prob))

        # Total runs
        if row["p_over85"] >= 0.5:
            total_pick, total_prob = "OVER", float(row["p_over85"])
        else:
            total_pick, total_prob = "UNDER", float(1.0 - row["p_over85"])
        out.extend(_make_signal(row, "TOTAL", total_pick, total_prob))

    return out


MAX_EDGE = 0.40  # cap unrealistic edges from exchange lay prices


def _make_signal(row: pd.Series, market: str, pick: str, prob: float) -> List[Signal]:
    ai_applied = bool(row.get("_ai_applied", False))
    floor = 0.40 if ai_applied else settings.min_confidence
    if prob < floor:
        return []
    fair = 1.0 / max(prob, 1e-6)
    book = _book_odds_for(row, market, pick)
    if book is not None:
        if book < settings.min_odds or book > settings.max_odds:
            return []
        edge = prob * book - 1.0
        if edge < settings.min_edge or edge > MAX_EDGE:
            return []
        stake = _kelly(prob, book)
        return [Signal(
            match_id=int(row["match_id"]),
            market=market, pick=pick,
            model_prob=float(prob), fair_odds=float(fair),
            book_odds=float(book), edge=float(edge),
            confidence=float(prob), stake_units=float(round(stake, 2)),
            is_value=True,
        )]
    # RL без реальных коэффициентов не публикуем — слишком много шума (away +1.5 всегда ~60%)
    if market == "RL":
        return []
    model_floor = floor if ai_applied else max(settings.min_confidence, 0.60)
    if prob >= model_floor:
        return [Signal(
            match_id=int(row["match_id"]),
            market=market, pick=pick,
            model_prob=float(prob), fair_odds=float(fair),
            book_odds=0.0, edge=0.0,
            confidence=float(prob), stake_units=1.0,
            is_value=False,
        )]
    return []
