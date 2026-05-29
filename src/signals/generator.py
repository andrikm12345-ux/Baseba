"""Signal dataclass + staking helpers for the Claude-driven baseball bot.

Markets: ML (HOME/AWAY), TOTAL (OVER/UNDER), RL (COVER/AWAY_COVER).
Signals are produced by Claude in pipeline._ai_to_signal — this module only
holds the Signal container and the Kelly-stake helper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


MAX_EDGE = 0.40  # cap unrealistic edges from exchange/lay prices


@dataclass
class Signal:
    match_id: int
    market: str       # "ML" | "TOTAL" | "RL"
    pick: str         # HOME/AWAY, OVER/UNDER, COVER/AWAY_COVER
    model_prob: float
    fair_odds: float
    book_odds: float  # 0.0 if unknown
    edge: float
    confidence: float
    stake_units: float
    is_value: bool
    line: Optional[float] = None  # total/spread line; None for ML


def _kelly(p: float, odds: float, fraction: float = 0.25, cap: float = 2.0) -> float:
    b = odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p
    k = (b * p - q) / b
    if k <= 0:
        return 0.0
    return min(k * fraction * 10, cap)
