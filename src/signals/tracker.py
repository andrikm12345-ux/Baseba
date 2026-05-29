"""Settle signals against final scores and compute ROI — baseball edition."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import List

from loguru import logger
from sqlalchemy import select

from src.config import settings
from src.data.database import Match, SessionLocal, Signal


def _did_win(market, pick, home_runs, away_runs, line=None):
    """Returns True/False, or None for a push (stake returned).

    `line` is the line the signal was actually placed at (total runs or |handicap|).
    """
    total = home_runs + away_runs
    diff = home_runs - away_runs
    if market == "ML":
        return (home_runs > away_runs) if pick == "HOME" else (away_runs > home_runs)
    if market == "TOTAL":
        ln = line if line is not None else settings.total_line
        if total == ln:
            return None  # push (integer line landed exactly)
        return (total > ln) if pick == "OVER" else (total < ln)
    if market == "RL":
        ln = line if line is not None else settings.rl_line
        if pick == "COVER":
            return diff > ln          # home favored -ln
        if pick == "AWAY_COVER":
            return (-diff) > ln        # away favored -ln
        # legacy underdog picks
        if pick == "LAY":
            return diff < ln
        if pick == "HOME_LAY":
            return (-diff) < ln
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
            won = _did_win(sig.market, sig.pick, match.home_runs, match.away_runs, sig.line)
            sig.settled = True
            if won is None:
                # push — stake returned, neutral
                sig.won = None
                sig.profit_units = 0.0
            else:
                sig.won = won
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
    # Exclude pushes (won is None) — stake returned, neutral for ROI/hit-rate
    rows = [r for r in rows if r.won is not None]
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


async def enrich_scores_from_odds_api(client) -> int:
    """Update home_runs/away_runs for matches that are missing scores via Odds API /scores."""
    from src.data.database import Team
    from src.data.odds_api import _canonical, _sim

    scores = await client.fetch_mlb_scores()
    if not scores:
        return 0

    updated = 0
    async with SessionLocal() as session:
        for ev in scores:
            if not ev.get("completed"):
                continue
            scores_data = ev.get("scores")
            if not scores_data:
                continue
            home_name = ev.get("home_team", "")
            away_name = ev.get("away_team", "")
            home_score = None
            away_score = None
            for team_key, score_val in scores_data.items():
                # scores format: {"home_team_name": {"score": "5"}, "away_team_name": {"score": "3"}}
                try:
                    score_int = int(score_val.get("score", ""))
                except (ValueError, AttributeError):
                    continue
                if _sim(team_key, home_name) > 0.8:
                    home_score = score_int
                elif _sim(team_key, away_name) > 0.8:
                    away_score = score_int

            if home_score is None or away_score is None:
                continue

            # Find matching match in DB by team names + date
            commence = ev.get("commence_time")
            from datetime import datetime
            try:
                game_dt = datetime.fromisoformat(commence.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                continue

            # Find by home team name + date window
            home_teams = (await session.execute(
                select(Team).where(Team.name.ilike(f"%{home_name.split()[-1]}%"))
            )).scalars().all()

            for ht in home_teams:
                matches = (await session.execute(
                    select(Match).where(
                        Match.home_team_id == ht.id,
                        Match.utc_date >= game_dt - timedelta(hours=4),
                        Match.utc_date <= game_dt + timedelta(hours=4),
                    )
                )).scalars().all()
                for m in matches:
                    if m.home_runs is None:
                        m.home_runs = home_score
                        m.away_runs = away_score
                        m.status = "FINISHED"
                        updated += 1
        await session.commit()

    if updated:
        logger.info(f"enrich_scores_from_odds_api: updated {updated} matches")
    return updated
