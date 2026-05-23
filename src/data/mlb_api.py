"""Async client for the free MLB Stats API (statsapi.mlb.com).

No API key required. Rate limit is generous (~300 req/min).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


BASE_URL = "https://statsapi.mlb.com/api/v1"

GAME_TYPE_REGULAR = "R"


class MlbApiClient:
    """Thin async wrapper for statsapi.mlb.com."""

    def __init__(self, sport_id: int = 1) -> None:
        self.sport_id = sport_id
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Semaphore(4)

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=1, max=15))
    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        async with self._lock:
            session = await self._session_get()
            url = f"{BASE_URL}{path}"
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as r:
                if r.status == 429:
                    logger.warning("MLB API rate-limited, backing off 10s")
                    await asyncio.sleep(10)
                    raise aiohttp.ClientResponseError(
                        r.request_info, r.history, status=429, message="rate limited"
                    )
                r.raise_for_status()
                return await r.json()

    async def schedule(
        self,
        start_date: str,
        end_date: str,
        game_type: str = GAME_TYPE_REGULAR,
        hydrate: str = "team,linescore",
    ) -> List[Dict[str, Any]]:
        """Fetch games in the given date range. Returns flat list of game dicts."""
        params: Dict[str, Any] = {
            "sportId": self.sport_id,
            "startDate": start_date,
            "endDate": end_date,
            "gameType": game_type,
            "hydrate": hydrate,
        }
        try:
            data = await self._get("/schedule", params=params)
        except Exception as e:
            logger.warning(f"MLB schedule fetch failed ({start_date}→{end_date}): {e}")
            return []
        games: List[Dict[str, Any]] = []
        for date_entry in data.get("dates", []):
            games.extend(date_entry.get("games", []))
        return games

    async def fetch_finished_history(self, seasons: List[int]) -> List[Dict[str, Any]]:
        """Fetch all finished regular-season games for the given seasons."""
        out: List[Dict[str, Any]] = []
        for season in seasons:
            start = f"{season}-03-01"
            end = f"{season}-11-30"
            try:
                games = await self.schedule(start, end)
                finished = [g for g in games if _is_final(g)]
                logger.info(f"MLB {season}: {len(finished)} finished games (of {len(games)} total)")
                out.extend(finished)
            except Exception as e:
                logger.warning(f"MLB history season {season} failed: {e}")
        return out

    async def fetch_upcoming(self, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """Fetch scheduled games for the next N days."""
        today = datetime.now(timezone.utc).date()
        start = today.isoformat()
        end = (today + timedelta(days=days_ahead)).isoformat()
        try:
            games = await self.schedule(start, end)
            upcoming = [g for g in games if not _is_final(g)]
            logger.info(f"MLB upcoming: {len(upcoming)} games in next {days_ahead} days")
            return upcoming
        except Exception as e:
            logger.warning(f"MLB upcoming fetch failed: {e}")
            return []


def _is_final(game: Dict[str, Any]) -> bool:
    state = (game.get("status") or {}).get("detailedState", "")
    return state.lower() in {"final", "completed", "game over"}


def parse_game(game: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract a flat dict from an MLB schedule game entry.

    Returns None if the game entry is malformed.
    """
    try:
        game_pk = int(game["gamePk"])
        game_date_str = game.get("gameDate") or game.get("officialDate") or ""
        if not game_date_str:
            return None
        utc_date = datetime.fromisoformat(game_date_str.replace("Z", "+00:00")).replace(tzinfo=None)

        teams = game.get("teams", {})
        home_info = teams.get("home", {})
        away_info = teams.get("away", {})

        home_team = home_info.get("team", {})
        away_team = away_info.get("team", {})

        if not home_team.get("id") or not away_team.get("id"):
            return None

        status = (game.get("status") or {}).get("detailedState", "SCHEDULED")
        normalized_status = "FINISHED" if _is_final(game) else "SCHEDULED"

        home_runs = home_info.get("score")
        away_runs = away_info.get("score")

        # season from gameDate year
        season = utc_date.year

        return {
            "id": game_pk,
            "utc_date": utc_date,
            "season": season,
            "status": normalized_status,
            "competition": "mlb",
            "home_team_id": int(home_team["id"]),
            "home_team_name": home_team.get("name") or home_team.get("clubName") or f"Team {home_team['id']}",
            "home_team_short": home_team.get("abbreviation") or home_team.get("teamCode"),
            "away_team_id": int(away_team["id"]),
            "away_team_name": away_team.get("name") or away_team.get("clubName") or f"Team {away_team['id']}",
            "away_team_short": away_team.get("abbreviation") or away_team.get("teamCode"),
            "home_runs": int(home_runs) if home_runs is not None else None,
            "away_runs": int(away_runs) if away_runs is not None else None,
        }
    except Exception as e:
        logger.debug(f"parse_game failed for gamePk={game.get('gamePk')}: {e}")
        return None
