"""Settle signals against final scores and compute ROI — baseball edition."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from loguru import logger
from sqlalchemy import select

from src.config import settings
from src.data.database import Match, SessionLocal, Signal


def _did_win(market: str, pick: str, home_runs: int, away_runs: int) -> bool:
    total = home_runs + away_runs
    diff = home_runs - away_runs
    if market == "ML":
        return (home_runs > away_runs) if pick == "HOME" else (away_runs > home_runs)
    if market == "TOTAL":
        return (total > settings.total_line) if pick == "OVER" else (total < settings.total_line)
    if market == "RL":
        if pick == "COVER":
            return diff > settings.rl_line
        if pick == "LAY":
            return diff < settings.rl_line
        if pick == "AWAY_COVER":
            return (-diff) > settings.rl_line
        if pick == "HOME_LAY":
            return (-diff) < settings.rl_line
    if market == "ITB":
        if pick == "HOME_OVER":
            return home_runs > settings.itb_line
        if pick == "AWAY_OVER":
            return away_runs > settings.itb_line
    return False


@dataclass
class RoiStats:
    n_settled: int
    n_won: int
    staked: float
    returned: float
    profit: float
    roi: float
    hit_rate: float


async def settle_pending() -> int:
    """Mark every unsettled signal whose game is FINISHED."""
    settled = 0
    async with SessionLocal() as session:
        q = await session.execute(
            select(Signal, Match).join(Match, Match.id == Signal.match_id).where(
                Signal.settled.is_(False),
                Match.status == "FINISHED",
            )
        )
        for sig, match in q.all():
            if match.home_runs is None or match.away_runs is None:
                continue
            won = _did_win(sig.market, sig.pick, match.home_runs, match.away_runs)
            sig.won = won
            sig.settled = True
            if sig.book_odds and sig.book_odds > 1.0:
                sig.profit_units = (sig.stake_units * (sig.book_odds - 1.0)) if won else -sig.stake_units
            else:
                sig.profit_units = sig.stake_units if won else -sig.stake_units
            settled += 1
        await session.commit()
    if settled:
        logger.info(f"Settled {settled} signals")
    return settled


async def roi_stats(
    last_n: int | None = None,
    only_value: bool | None = True,
    ai_only: bool | None = None,
) -> RoiStats:
    async with SessionLocal() as session:
        q = select(Signal).where(Signal.settled.is_(True)).order_by(Signal.created_at.desc())
        if last_n:
            q = q.limit(last_n)
        rows: List[Signal] = list((await session.execute(q)).scalars())
    if only_value is True:
        rows = [r for r in rows if r.book_odds and r.book_odds > 1.0]
    elif only_value is False:
        rows = [r for r in rows if not r.book_odds or r.book_odds <= 1.0]
    if ai_only:
        rows = [r for r in rows if getattr(r, "is_ai_ensemble", False)]
    n = len(rows)
    if n == 0:
        return RoiStats(0, 0, 0, 0, 0, 0.0, 0.0)
    staked = sum(r.stake_units for r in rows)
    returned = sum(
        (r.stake_units * r.book_odds) if (r.won and r.book_odds > 1.0) else 0.0 for r in rows
    )
    profit = sum(r.profit_units or 0.0 for r in rows)
    won = sum(1 for r in rows if r.won)
    return RoiStats(
        n_settled=n, n_won=won, staked=staked, returned=returned,
        profit=profit,
        roi=(profit / staked * 100.0) if staked > 0 else 0.0,
        hit_rate=(won / n * 100.0) if n > 0 else 0.0,
    )
