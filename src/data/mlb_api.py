"""Async client for the free MLB Stats API (statsapi.mlb.com).

No API key required. Rate limit is generous (~300 req/min).
Fetches games, probable starting pitchers, and pitcher season stats.
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
        self._pitcher_stats_cache: Dict[int, Dict[str, Any]] = {}

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
        hydrate: str = "team,linescore,probablePitcher",
    ) -> List[Dict[str, Any]]:
        """Fetch games in the given date range with probable pitchers."""
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

    async def fetch_pitcher_stats(self, pitcher_id: int, season: int) -> Dict[str, Any]:
        """Fetch season pitching stats for a player. Returns dict with ERA, WHIP, K9, BB9."""
        if pitcher_id in self._pitcher_stats_cache:
            return self._pitcher_stats_cache[pitcher_id]
        try:
            data = await self._get(
                f"/people/{pitcher_id}/stats",
                params={"stats": "season", "group": "pitching", "season": season},
            )
            stats = {}
            splits = (data.get("stats") or [{}])[0].get("splits", [])
            if splits:
                s = splits[0].get("stat", {})
                ip = float(s.get("inningsPitched") or 0)
                era = float(s.get("era") or 99.0)
                whip = float(s.get("whip") or 9.99)
                so = int(s.get("strikeOuts") or 0)
                bb = int(s.get("baseOnBalls") or 0)
                k9 = round((so / ip * 9), 2) if ip > 0 else 0.0
                bb9 = round((bb / ip * 9), 2) if ip > 0 else 0.0
                stats = {
                    "era": era if era < 99 else None,
                    "whip": whip if whip < 9 else None,
                    "k9": k9 if ip > 0 else None,
                    "bb9": bb9 if ip > 0 else None,
                    "ip": ip,
                }
            self._pitcher_stats_cache[pitcher_id] = stats
            return stats
        except Exception as e:
            logger.debug(f"pitcher stats fetch failed for {pitcher_id}: {e}")
            return {}

    async def enrich_with_pitcher_stats(self, games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """For each game, fetch season stats for both probable pitchers concurrently."""
        pitcher_ids: set[int] = set()
        for g in games:
            for side in ("home", "away"):
                pid = _extract_pitcher_id(g, side)
                if pid:
                    pitcher_ids.add(pid)

        if not pitcher_ids:
            return games

        season = datetime.utcnow().year
        sem = asyncio.Semaphore(5)

        async def _fetch(pid: int) -> tuple[int, Dict]:
            async with sem:
                stats = await self.fetch_pitcher_stats(pid, season)
                return pid, stats

        results = await asyncio.gather(*[_fetch(pid) for pid in pitcher_ids])
        stats_map: Dict[int, Dict] = dict(results)

        for g in games:
            g["_pitcher_stats"] = stats_map

        return games

    async def fetch_finished_history(self, seasons: List[int]) -> List[Dict[str, Any]]:
        """Fetch all finished regular-season games for the given seasons."""
        out: List[Dict[str, Any]] = []
        for season in seasons:
            start = f"{season}-03-01"
            end = f"{season}-11-30"
            try:
                # Historical games don't need pitcher stats (already finished)
                games = await self.schedule(
                    start, end,
                    hydrate="team,linescore",  # skip pitcher hydration for history (speed)
                )
                finished = [g for g in games if _is_final(g)]
                logger.info(f"MLB {season}: {len(finished)} finished games (of {len(games)} total)")
                out.extend(finished)
            except Exception as e:
                logger.warning(f"MLB history season {season} failed: {e}")
        return out

    async def fetch_upcoming(self, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """Fetch scheduled games with probable pitchers, then enrich with their stats."""
        today = datetime.now(timezone.utc).date()
        start = today.isoformat()
        end = (today + timedelta(days=days_ahead)).isoformat()
        try:
            games = await self.schedule(
                start, end,
                hydrate="team,linescore,probablePitcher",
            )
            upcoming = [g for g in games if not _is_final(g)]
            logger.info(f"MLB upcoming: {len(upcoming)} games in next {days_ahead} days")
            # Enrich with pitcher season stats
            upcoming = await self.enrich_with_pitcher_stats(upcoming)
            return upcoming
        except Exception as e:
            logger.warning(f"MLB upcoming fetch failed: {e}")
            return []


def _is_final(game: Dict[str, Any]) -> bool:
    state = (game.get("status") or {}).get("detailedState", "")
    return state.lower() in {"final", "completed", "game over"}


def _extract_pitcher_id(game: Dict[str, Any], side: str) -> Optional[int]:
    try:
        pitcher = game["teams"][side].get("probablePitcher") or {}
        pid = pitcher.get("id")
        return int(pid) if pid else None
    except Exception:
        return None


def _extract_pitcher_name(game: Dict[str, Any], side: str) -> Optional[str]:
    try:
        pitcher = game["teams"][side].get("probablePitcher") or {}
        return pitcher.get("fullName") or pitcher.get("lastName")
    except Exception:
        return None


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

        normalized_status = "FINISHED" if _is_final(game) else "SCHEDULED"
        home_runs = home_info.get("score")
        away_runs = away_info.get("score")
        season = utc_date.year

        # Probable pitchers
        stats_map: Dict[int, Dict] = game.get("_pitcher_stats") or {}
        home_pid = _extract_pitcher_id(game, "home")
        away_pid = _extract_pitcher_id(game, "away")
        home_pstats = stats_map.get(home_pid, {}) if home_pid else {}
        away_pstats = stats_map.get(away_pid, {}) if away_pid else {}

        return {
            "id": game_pk,
            "utc_date": utc_date,
            "season": season,
            "status": normalized_status,
            "competition": "mlb",
            "home_team_id": int(home_team["id"]),
            "home_team_name": home_team.get("name") or f"Team {home_team['id']}",
            "home_team_short": home_team.get("abbreviation") or home_team.get("teamCode"),
            "away_team_id": int(away_team["id"]),
            "away_team_name": away_team.get("name") or f"Team {away_team['id']}",
            "away_team_short": away_team.get("abbreviation") or away_team.get("teamCode"),
            "home_runs": int(home_runs) if home_runs is not None else None,
            "away_runs": int(away_runs) if away_runs is not None else None,
            # Pitchers
            "home_pitcher_id": home_pid,
            "home_pitcher_name": _extract_pitcher_name(game, "home"),
            "home_pitcher_era": home_pstats.get("era"),
            "home_pitcher_whip": home_pstats.get("whip"),
            "home_pitcher_k9": home_pstats.get("k9"),
            "home_pitcher_bb9": home_pstats.get("bb9"),
            "away_pitcher_id": away_pid,
            "away_pitcher_name": _extract_pitcher_name(game, "away"),
            "away_pitcher_era": away_pstats.get("era"),
            "away_pitcher_whip": away_pstats.get("whip"),
            "away_pitcher_k9": away_pstats.get("k9"),
            "away_pitcher_bb9": away_pstats.get("bb9"),
        }
    except Exception as e:
        logger.debug(f"parse_game failed for gamePk={game.get('gamePk')}: {e}")
        return None
