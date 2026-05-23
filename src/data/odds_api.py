"""Client for odds-api.io v3 — baseball (MLB) edition.

Markets we track:
  - ML  (Moneyline: home / away)
  - TOTAL (Over/Under total runs, line = settings.total_line default 8.5)
  - RL  (Run Line: home -1.5 / away +1.5)
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import settings


BASE_URL = "https://api.odds-api.io/v3"
DEFAULT_BOOKMAKERS = "Bet365,Betfair Exchange"

SPORT = "baseball"

LEAGUE_TO_SLUG: Dict[str, str] = {
    "mlb": "baseball_mlb",
}

ML_NAMES = {"ml", "moneyline", "h2h", "match winner", "1x2"}
TOTAL_NAMES = {
    "totals", "total", "over/under", "run totals", "game totals",
    "total runs", "runs over/under", "ou",
}
RL_NAMES = {
    "run line", "runline", "rl", "spread", "puck line",
    "handicap", "asian handicap",
}


class OddsApiError(Exception):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"odds-api.io {status}: {body[:200]}")
        self.status = status
        self.body = body


class OddsApiClient:
    def __init__(self, api_key: str, cache_ttl_seconds: float = 3600.0) -> None:
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_ttl = cache_ttl_seconds
        self._first_odds_logged = False
        self._first_event_logged = False

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        retry=retry_if_exception_type((aiohttp.ClientConnectionError, asyncio.TimeoutError)),
    )
    async def _get(self, path: str, params: Dict[str, Any]) -> Any:
        cache_key = f"{path}?{sorted(params.items())}"
        now = asyncio.get_event_loop().time()
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if now - ts < self._cache_ttl:
                return data
        session = await self._session_get()
        params = {**params, "apiKey": self.api_key}
        async with session.get(f"{BASE_URL}{path}", params=params, timeout=aiohttp.ClientTimeout(total=20)) as r:
            if r.status == 429:
                logger.warning("odds-api.io rate-limited, backing off 30s")
                await asyncio.sleep(30)
            if r.status >= 400:
                text = await r.text()
                logger.warning(f"odds-api.io {r.status} on {path}: {text[:200]}")
                raise OddsApiError(r.status, text)
            remaining = r.headers.get("x-requests-remaining") or r.headers.get("X-RateLimit-Remaining")
            if remaining:
                logger.info(f"odds-api.io quota remaining: {remaining}")
            data = await r.json()
            self._cache[cache_key] = (now, data)
            return data

    async def fetch_events(
        self,
        sport: str = SPORT,
        league: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"sport": sport, "limit": limit}
        if league:
            params["league"] = league
        try:
            data = await self._get("/events", params)
        except OddsApiError:
            raise
        except Exception as e:
            logger.warning(f"fetch_events failed: {e}")
            return []
        events = data if isinstance(data, list) else data.get("events", [])
        if not self._first_event_logged:
            self._first_event_logged = True
            if events:
                try:
                    logger.info(
                        f"odds-api.io sample event (keys={list(events[0].keys())}): "
                        f"{json.dumps(events[0])[:800]}"
                    )
                except Exception:
                    pass
            else:
                logger.warning(
                    f"odds-api.io returned 0 events. "
                    f"Raw response type={type(data).__name__} "
                    f"keys={list(data.keys()) if isinstance(data, dict) else 'list'} "
                    f"preview={json.dumps(data)[:400]}"
                )
        return events

    async def fetch_event_odds(
        self,
        event_id: int | str,
        bookmakers: str = DEFAULT_BOOKMAKERS,
    ) -> Optional[Dict[str, Any]]:
        try:
            data = await self._get("/odds", {"eventId": event_id, "bookmakers": bookmakers})
        except Exception as e:
            logger.warning(f"fetch_event_odds {event_id} failed: {e}")
            return None
        if not self._first_odds_logged:
            self._first_odds_logged = True
            try:
                logger.info(f"odds-api.io sample /odds payload: {json.dumps(data)[:1000]}")
            except Exception:
                pass
        return data


# ─── matching ───────────────────────────────────────────────────────────────

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.utcfromtimestamp(value / 1000.0 if value > 1e12 else value)
        except Exception:
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            return None
    return None


def _event_kickoff(ev: Dict[str, Any]) -> Optional[datetime]:
    for key in ("date", "commenceTime", "commence_time", "startTime", "start_time", "kickoff", "scheduled"):
        if key in ev:
            dt = _parse_dt(ev[key])
            if dt:
                return dt
    return None


def _event_teams(ev: Dict[str, Any]) -> Tuple[str, str]:
    # odds-api.io returns "home"/"away" as plain strings
    home = ev.get("home") or ev.get("homeTeam") or ev.get("home_team")
    away = ev.get("away") or ev.get("awayTeam") or ev.get("away_team")
    if isinstance(home, dict):
        home = home.get("name") or home.get("title") or ""
    if isinstance(away, dict):
        away = away.get("name") or away.get("title") or ""
    if not home or not away:
        teams = ev.get("teams") or ev.get("participants") or []
        if isinstance(teams, list) and len(teams) >= 2:
            t0 = teams[0].get("name") if isinstance(teams[0], dict) else teams[0]
            t1 = teams[1].get("name") if isinstance(teams[1], dict) else teams[1]
            home = home or t0
            away = away or t1
    return str(home or ""), str(away or "")


def _best_match(
    home_name: str, away_name: str, kickoff: datetime, events: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    best = None
    best_score = 0.0
    for ev in events:
        ev_time = _event_kickoff(ev)
        if ev_time and abs((ev_time - kickoff).total_seconds()) > 6 * 3600:
            continue
        h, a = _event_teams(ev)
        score = (_sim(home_name, h) + _sim(away_name, a)) / 2
        if score > best_score and score > 0.50:
            best_score = score
            best = ev
    return best


# ─── odds parsing ───────────────────────────────────────────────────────────

def _as_float(x: Any) -> Optional[float]:
    try:
        f = float(x)
        return f if f > 1.0 else None
    except (TypeError, ValueError):
        return None


def _market_name(market: Dict[str, Any]) -> str:
    return str(market.get("name") or market.get("key") or market.get("market") or "").lower().strip()


def extract_odds(odds_payload: Dict[str, Any]) -> Dict[str, float]:
    """Parse odds-api.io payload for ML / TOTAL / RL markets.

    Returns dict with keys: odds_ml_home, odds_ml_away,
    odds_over85, odds_under85, odds_rl_home, odds_rl_away.
    """
    total_line = settings.total_line
    rl_line = settings.rl_line

    aggregated: Dict[str, List[float]] = {
        "odds_ml_home": [],
        "odds_ml_away": [],
        "odds_over85": [],
        "odds_under85": [],
        "odds_rl_home": [],
        "odds_rl_away": [],
    }

    books = odds_payload.get("bookmakers") or {}
    if isinstance(books, list):
        books = {str(b.get("name") or i): b.get("markets", b) for i, b in enumerate(books)}
    if not isinstance(books, dict):
        return {k: 0.0 for k in aggregated}

    for _bk_name, markets in books.items():
        if not isinstance(markets, list):
            markets = markets.get("markets", []) if isinstance(markets, dict) else []
        for market in markets:
            if not isinstance(market, dict):
                continue
            name = _market_name(market)
            odds_list = market.get("odds")
            if not isinstance(odds_list, list):
                odds_list = [odds_list] if isinstance(odds_list, dict) else []

            if name in ML_NAMES:
                for entry in odds_list:
                    if not isinstance(entry, dict):
                        continue
                    h = _as_float(entry.get("home"))
                    a = _as_float(entry.get("away"))
                    if h:
                        aggregated["odds_ml_home"].append(h)
                    if a:
                        aggregated["odds_ml_away"].append(a)

            elif name in TOTAL_NAMES:
                for entry in odds_list:
                    if not isinstance(entry, dict):
                        continue
                    hdp = entry.get("hdp") or entry.get("line") or entry.get("total")
                    try:
                        hdp_f = float(hdp) if hdp is not None else None
                    except (TypeError, ValueError):
                        hdp_f = None
                    if hdp_f is None or abs(hdp_f - total_line) > 0.26:
                        continue
                    over = _as_float(entry.get("over"))
                    under = _as_float(entry.get("under"))
                    if over:
                        aggregated["odds_over85"].append(over)
                    if under:
                        aggregated["odds_under85"].append(under)

            elif name in RL_NAMES:
                for entry in odds_list:
                    if not isinstance(entry, dict):
                        continue
                    hdp = entry.get("hdp") or entry.get("line") or entry.get("handicap")
                    try:
                        hdp_f = float(hdp) if hdp is not None else None
                    except (TypeError, ValueError):
                        hdp_f = None
                    if hdp_f is None or abs(abs(hdp_f) - rl_line) > 0.26:
                        continue
                    home_lay = _as_float(entry.get("home"))
                    away_lay = _as_float(entry.get("away"))
                    if home_lay:
                        aggregated["odds_rl_home"].append(home_lay)
                    if away_lay:
                        aggregated["odds_rl_away"].append(away_lay)

    return {k: (max(v) if v else 0.0) for k, v in aggregated.items()}


def _is_upcoming(ev: Dict[str, Any], now: datetime) -> bool:
    status = str(ev.get("status", "")).lower()
    if status in {"settled", "finished", "ended", "cancelled", "canceled", "postponed"}:
        return False
    kickoff = _event_kickoff(ev)
    if kickoff is None:
        return True
    return kickoff >= now - timedelta(hours=2)


async def fetch_odds_for_matches(
    client: OddsApiClient,
    upcoming: List[Tuple[int, str, str, str, datetime]],
) -> Dict[int, Dict[str, float]]:
    """upcoming: (match_id, league, home_name, away_name, utc_date)."""
    if not upcoming:
        return {}

    now = datetime.utcnow()
    events_cache: Dict[str, List[Dict[str, Any]]] = {}
    bad_slugs: set[str] = set()

    leagues_needed = {league for _, league, _, _, _ in upcoming}
    for league in leagues_needed:
        if league in events_cache:
            continue
        slug = LEAGUE_TO_SLUG.get(league)
        # Fetch all baseball events and filter by league slug client-side
        # (odds-api.io league param causes 404 for unknown slugs)
        try:
            evs = await client.fetch_events(sport=SPORT, limit=500)
        except OddsApiError:
            evs = []
        if slug:
            evs = [
                ev for ev in evs
                if (ev.get("league") or {}).get("slug", "") == slug
            ]
        evs = [ev for ev in evs if _is_upcoming(ev, now)]
        events_cache[league] = evs
        logger.info(f"odds-api.io: {len(evs)} upcoming events for league={league} (slug={slug})")

    out: Dict[int, Dict[str, float]] = {}
    for match_id, league, home, away, kickoff in upcoming:
        events = events_cache.get(league) or []
        if not events:
            continue
        ev = _best_match(home, away, kickoff, events)
        if not ev:
            continue
        ev_id = ev.get("id") or ev.get("eventId") or ev.get("event_id")
        if not ev_id:
            continue
        payload = await client.fetch_event_odds(ev_id)
        if not payload:
            continue
        odds = extract_odds(payload)
        if any(v > 0 for v in odds.values()):
            out[match_id] = odds

    return out
