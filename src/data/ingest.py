from __future__ import annotations

from typing import Iterable, List

from loguru import logger
from sqlalchemy import select

from src.data.database import Match, SessionLocal, Team
from src.data.mlb_api import MlbApiClient, parse_game


async def _upsert_team(session, team_id: int, name: str, short_name: str | None) -> int:
    existing = await session.get(Team, team_id)
    if existing is None:
        session.add(Team(id=team_id, name=name, short_name=short_name))
    return team_id


async def store_games(games_raw: Iterable[dict]) -> int:
    """Parse and upsert MLB games into DB. Returns count written/updated."""
    count = 0
    async with SessionLocal() as session:
        for raw in games_raw:
            game = parse_game(raw)
            if game is None:
                continue
            try:
                await _upsert_team(
                    session,
                    game["home_team_id"],
                    game["home_team_name"],
                    game.get("home_team_short"),
                )
                await _upsert_team(
                    session,
                    game["away_team_id"],
                    game["away_team_name"],
                    game.get("away_team_short"),
                )

                match_id = game["id"]
                existing = await session.get(Match, match_id)
                if existing is None:
                    existing = Match(id=match_id)
                    session.add(existing)

                existing.competition = game["competition"]
                existing.season = game["season"]
                existing.utc_date = game["utc_date"]
                existing.status = game["status"]
                existing.home_team_id = game["home_team_id"]
                existing.away_team_id = game["away_team_id"]
                existing.home_runs = game["home_runs"]
                existing.away_runs = game["away_runs"]
                count += 1
            except Exception as e:
                logger.warning(f"Skip game {raw.get('gamePk')}: {e}")
        await session.commit()
    return count


async def ingest_history(client: MlbApiClient, seasons: List[int]) -> int:
    raw_games = await client.fetch_finished_history(seasons)
    n = await store_games(raw_games)
    logger.info(f"Stored {n} historical MLB games across seasons {seasons}")
    return n


async def ingest_upcoming(client: MlbApiClient, days_ahead: int = 7) -> int:
    raw_games = await client.fetch_upcoming(days_ahead=days_ahead)
    n = await store_games(raw_games)
    logger.info(f"Stored/updated {n} upcoming MLB games")
    return n
