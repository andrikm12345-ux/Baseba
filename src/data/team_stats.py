"""Team statistics from our finished matches DB: H2H and last-10-games form."""
from __future__ import annotations

from typing import List


async def get_team_context(home_team_id: int, away_team_id: int, season: int) -> dict:
    """Compute H2H record and last-10-games form from our finished matches DB."""
    from src.data.database import Match, SessionLocal
    from sqlalchemy import select, and_, or_

    async with SessionLocal() as session:
        # H2H this season between these two teams
        h2h_q = await session.execute(
            select(Match).where(
                Match.status == "FINISHED",
                Match.season == season,
                or_(
                    and_(Match.home_team_id == home_team_id, Match.away_team_id == away_team_id),
                    and_(Match.home_team_id == away_team_id, Match.away_team_id == home_team_id),
                )
            ).order_by(Match.utc_date.desc()).limit(10)
        )
        h2h_matches = h2h_q.scalars().all()

        # Last 10 games for home team
        home_last10_q = await session.execute(
            select(Match).where(
                Match.status == "FINISHED",
                Match.season == season,
                or_(Match.home_team_id == home_team_id, Match.away_team_id == home_team_id),
            ).order_by(Match.utc_date.desc()).limit(10)
        )
        home_last10 = home_last10_q.scalars().all()

        # Last 10 games for away team
        away_last10_q = await session.execute(
            select(Match).where(
                Match.status == "FINISHED",
                Match.season == season,
                or_(Match.home_team_id == away_team_id, Match.away_team_id == away_team_id),
            ).order_by(Match.utc_date.desc()).limit(10)
        )
        away_last10 = away_last10_q.scalars().all()

    def _wins(matches: list, team_id: int) -> int:
        w = 0
        for m in matches:
            if m.home_runs is None or m.away_runs is None:
                continue
            if m.home_team_id == team_id:
                if m.home_runs > m.away_runs:
                    w += 1
            else:
                if m.away_runs > m.home_runs:
                    w += 1
        return w

    def _avg_runs(matches: list, team_id: int) -> float:
        runs: List[int] = []
        for m in matches:
            if m.home_runs is None:
                continue
            runs.append(m.home_runs if m.home_team_id == team_id else m.away_runs)
        return round(sum(runs) / len(runs), 1) if runs else 0.0

    h2h_home_wins = _wins(h2h_matches, home_team_id)
    h2h_away_wins = _wins(h2h_matches, away_team_id)

    return {
        "h2h_games": len(h2h_matches),
        "h2h_home_wins": h2h_home_wins,
        "h2h_away_wins": h2h_away_wins,
        "home_last10_wins": _wins(home_last10, home_team_id),
        "home_last10_games": len(home_last10),
        "home_avg_runs": _avg_runs(home_last10, home_team_id),
        "away_last10_wins": _wins(away_last10, away_team_id),
        "away_last10_games": len(away_last10),
        "away_avg_runs": _avg_runs(away_last10, away_team_id),
    }
