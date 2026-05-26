"""Client for The Odds API v4 (api.the-odds-api.com) — MLB edition.

Free tier budget: 500 req/month.
Strategy: cache 2 hours → ~12 calls/day → ~360/month (buffer for retries).
One call to /v4/sports/baseball_mlb/odds returns ALL upcoming games with odds.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from loguru import logger

from src.config import settings


BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"
# us = FanDuel/DraftKings/BetMGM; eu = Pinnacle (sharpest lines)
REGIONS = "us,eu"
MARKETS = "h2h,totals,spreads,h2h_1st_5_innings"

CACHE_TTL = 7200.0  # 2 hours

# Module-level cache: (timestamp, events_list)
_cache: Tuple[float, List[Dict]] = (0.0, [])


# ─── team name normalisation ────────────────────────────────────────────────

_MLB_CANONICAL: Dict[str, str] = {
    # Arizona
    "arizona diamondbacks": "arizona diamondbacks", "diamondbacks": "arizona diamondbacks",
    "d-backs": "arizona diamondbacks", "az diamondbacks": "arizona diamondbacks",
    # Atlanta
    "atlanta braves": "atlanta braves", "braves": "atlanta braves",
    # Baltimore
    "baltimore orioles": "baltimore orioles", "orioles": "baltimore orioles",
    # Boston
    "boston red sox": "boston red sox", "red sox": "boston red sox",
    # Chicago Cubs
    "chicago cubs": "chicago cubs", "cubs": "chicago cubs",
    # Chicago White Sox
    "chicago white sox": "chicago white sox", "white sox": "chicago white sox",
    # Cincinnati
    "cincinnati reds": "cincinnati reds", "reds": "cincinnati reds",
    # Cleveland
    "cleveland guardians": "cleveland guardians", "guardians": "cleveland guardians",
    # Colorado
    "colorado rockies": "colorado rockies", "rockies": "colorado rockies",
    # Detroit
    "detroit tigers": "detroit tigers", "tigers": "detroit tigers",
    # Houston
    "houston astros": "houston astros", "astros": "houston astros",
    # Kansas City
    "kansas city royals": "kansas city royals", "royals": "kansas city royals",
    "kc royals": "kansas city royals",
    # LA Angels
    "los angeles angels": "los angeles angels", "angels": "los angeles angels",
    "la angels": "los angeles angels", "anaheim angels": "los angeles angels",
    # LA Dodgers
    "los angeles dodgers": "los angeles dodgers", "dodgers": "los angeles dodgers",
    "la dodgers": "los angeles dodgers",
    # Miami
    "miami marlins": "miami marlins", "marlins": "miami marlins",
    # Milwaukee
    "milwaukee brewers": "milwaukee brewers", "brewers": "milwaukee brewers",
    # Minnesota
    "minnesota twins": "minnesota twins", "twins": "minnesota twins",
    # NY Mets
    "new york mets": "new york mets", "mets": "new york mets", "ny mets": "new york mets",
    # NY Yankees
    "new york yankees": "new york yankees", "yankees": "new york yankees",
    "ny yankees": "new york yankees",
    # Oakland / Athletics
    "oakland athletics": "oakland athletics", "athletics": "oakland athletics",
    "a's": "oakland athletics", "as": "oakland athletics",
    "oakland a's": "oakland athletics", "sacramento athletics": "oakland athletics",
    # Philadelphia
    "philadelphia phillies": "philadelphia phillies", "phillies": "philadelphia phillies",
    # Pittsburgh
    "pittsburgh pirates": "pittsburgh pirates", "pirates": "pittsburgh pirates",
    # San Diego
    "san diego padres": "san diego padres", "padres": "san diego padres",
    # San Francisco
    "san francisco giants": "san francisco giants", "giants": "san francisco giants",
    "sf giants": "san francisco giants",
    # Seattle
    "seattle mariners": "seattle mariners", "mariners": "seattle mariners",
    # St. Louis
    "st. louis cardinals": "st. louis cardinals", "st louis cardinals": "st. louis cardinals",
    "cardinals": "st. louis cardinals",
    # Tampa Bay
    "tampa bay rays": "tampa bay rays", "rays": "tampa bay rays",
    # Texas
    "texas rangers": "texas rangers", "rangers": "texas rangers",
    # Toronto
    "toronto blue jays": "toronto blue jays", "blue jays": "toronto blue jays",
    # Washington
    "washington nationals": "washington nationals", "nationals": "washington nationals",
    "nats": "washington nationals",
}


def _canonical(name: str) -> str:
    n = name.lower().strip()
    return _MLB_CANONICAL.get(n, n)


def _sim(a: str, b: str) -> float:
    ca, cb = _canonical(a), _canonical(b)
    if ca == cb:
        return 1.0
    return SequenceMatcher(None, ca, cb).ratio()


# ─── API client ─────────────────────────────────────────────────────────────

class OddsApiClient:
    """Thin wrapper for the-odds-api.com v4."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_mlb_scores(self) -> List[Dict[str, Any]]:
        """Fetch recent MLB scores for settling signals. daysFrom=3 covers last 3 days."""
        params = {
            "apiKey": self.api_key,
            "daysFrom": 3,
            "dateFormat": "iso",
        }
        try:
            session = await self._get_session()
            url = f"{BASE_URL}/sports/{SPORT}/scores"
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as r:
                remaining = r.headers.get("x-requests-remaining", "?")
                logger.debug(f"the-odds-api scores: remaining={remaining}")
                if r.status >= 400:
                    text = await r.text()
                    logger.warning(f"the-odds-api scores {r.status}: {text[:200]}")
                    return []
                data = await r.json()
                return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"the-odds-api scores fetch failed: {e}")
            return []

    async def fetch_mlb_odds(self) -> List[Dict[str, Any]]:
        """Fetch all upcoming MLB games with ML/totals/spreads odds.

        Two-layer cache: memory (fast) + DB (survives container restarts).
        2-hour TTL → ~12 calls/day → ~360/month, within 500 free tier.
        """
        import json as _json
        global _cache
        now_mono = time.monotonic()
        now_wall = time.time()

        # 1. Memory cache
        ts, data = _cache
        if now_mono - ts < CACHE_TTL and data:
            logger.debug(f"odds memory-cache hit ({len(data)} events)")
            return data

        # 2. DB cache (survives restarts)
        try:
            from src.data.settings_store import get_str
            db_raw = await get_str("odds_cache_payload", "")
            db_ts_str = await get_str("odds_cache_ts", "0")
            db_ts = float(db_ts_str)
            if db_raw and (now_wall - db_ts) < CACHE_TTL:
                db_data = _json.loads(db_raw)
                if db_data:
                    _cache = (now_mono, db_data)
                    logger.debug(f"odds db-cache hit ({len(db_data)} events, age={(now_wall-db_ts)/60:.0f}m)")
                    return db_data
        except Exception as e:
            logger.debug(f"odds db-cache read failed: {e}")

        params = {
            "apiKey": self.api_key,
            "regions": REGIONS,
            "markets": MARKETS,
            "dateFormat": "iso",
            "oddsFormat": "decimal",
        }
        try:
            session = await self._get_session()
            url = f"{BASE_URL}/sports/{SPORT}/odds"
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as r:
                remaining = r.headers.get("x-requests-remaining", "?")
                used = r.headers.get("x-requests-used", "?")
                logger.info(f"the-odds-api: used={used} remaining={remaining}")
                if remaining != "?" and int(remaining) < 50:
                    logger.warning(f"the-odds-api: LOW QUOTA — only {remaining} requests left this month!")
                if r.status == 401:
                    logger.error("the-odds-api: invalid API key")
                    return []
                if r.status == 422:
                    logger.error(f"the-odds-api: 422 — check SPORT/MARKETS params")
                    return []
                if r.status >= 400:
                    text = await r.text()
                    logger.warning(f"the-odds-api {r.status}: {text[:200]}")
                    return []
                events = await r.json()
                if not isinstance(events, list):
                    logger.warning(f"the-odds-api: unexpected response type {type(events)}")
                    return []
                logger.info(f"the-odds-api: fetched {len(events)} MLB events")
                _cache = (time.monotonic(), events)
                # Persist to DB so cache survives container restarts
                try:
                    import json as _json
                    from src.data.settings_store import set_str
                    await set_str("odds_cache_payload", _json.dumps(events))
                    await set_str("odds_cache_ts", str(now_wall))
                except Exception as ce:
                    logger.debug(f"odds db-cache write failed: {ce}")
                return events
        except Exception as e:
            logger.warning(f"the-odds-api fetch failed: {e}")
            return []


# ─── odds extraction ─────────────────────────────────────────────────────────

def _as_float(x: Any) -> Optional[float]:
    try:
        f = float(x)
        return f if 1.01 < f < 30.0 else None
    except (TypeError, ValueError):
        return None


def extract_odds_v4(event: Dict[str, Any]) -> Dict[str, float]:
    """Extract ML / TOTAL / RL odds from a the-odds-api.com v4 event."""
    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")
    total_line = settings.total_line
    rl_line = settings.rl_line

    ml_home: List[float] = []
    ml_away: List[float] = []
    over_list: List[float] = []
    under_list: List[float] = []
    rl_home: List[float] = []   # home -1.5 (home favored)
    rl_away: List[float] = []   # away +1.5
    rl_away_c: List[float] = [] # away -1.5 (away favored)
    rl_home_l: List[float] = [] # home +1.5
    f5_home: List[float] = []   # first 5 innings home
    f5_away: List[float] = []   # first 5 innings away

    for bk in event.get("bookmakers", []):
        for market in bk.get("markets", []):
            key = market.get("key", "")
            outcomes = market.get("outcomes", [])

            if key == "h2h_1st_5_innings":
                for o in outcomes:
                    price = _as_float(o.get("price"))
                    if price is None:
                        continue
                    name = o.get("name", "")
                    if _canonical(name) == _canonical(home_team):
                        f5_home.append(price)
                    elif _canonical(name) == _canonical(away_team):
                        f5_away.append(price)

            elif key == "h2h":
                for o in outcomes:
                    price = _as_float(o.get("price"))
                    if price is None:
                        continue
                    name = o.get("name", "")
                    if _canonical(name) == _canonical(home_team):
                        ml_home.append(price)
                    elif _canonical(name) == _canonical(away_team):
                        ml_away.append(price)

            elif key == "totals":
                for o in outcomes:
                    price = _as_float(o.get("price"))
                    if price is None:
                        continue
                    point = o.get("point")
                    try:
                        pt = float(point) if point is not None else None
                    except (TypeError, ValueError):
                        pt = None
                    if pt is not None and abs(pt - total_line) > 0.3:
                        continue
                    name = o.get("name", "").lower()
                    if name == "over":
                        over_list.append(price)
                    elif name == "under":
                        under_list.append(price)

            elif key == "spreads":
                for o in outcomes:
                    price = _as_float(o.get("price"))
                    if price is None:
                        continue
                    point = o.get("point")
                    try:
                        pt = float(point) if point is not None else None
                    except (TypeError, ValueError):
                        pt = None
                    if pt is None or abs(abs(pt) - rl_line) > 0.3:
                        continue
                    name = o.get("name", "")
                    can = _canonical(name)
                    if can == _canonical(home_team):
                        if pt < 0:
                            rl_home.append(price)   # home -1.5
                        else:
                            rl_home_l.append(price)  # home +1.5
                    elif can == _canonical(away_team):
                        if pt < 0:
                            rl_away_c.append(price)  # away -1.5
                        else:
                            rl_away.append(price)    # away +1.5

    def _best(lst: List[float]) -> float:
        return max(lst) if lst else 0.0

    return {
        "odds_ml_home": _best(ml_home),
        "odds_ml_away": _best(ml_away),
        "odds_over85": _best(over_list),
        "odds_under85": _best(under_list),
        "odds_rl_home": _best(rl_home),
        "odds_rl_away": _best(rl_away),
        "odds_rl_away_cover": _best(rl_away_c),
        "odds_rl_home_lay": _best(rl_home_l),
        "odds_itb_home": 0.0,
        "odds_itb_away": 0.0,
        "odds_f5_home": _best(f5_home),
        "odds_f5_away": _best(f5_away),
    }


# ─── matching ────────────────────────────────────────────────────────────────

def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            return None
    return None


def _find_event(
    home_name: str, away_name: str, kickoff: datetime, events: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Find the best matching event by team names + time."""
    best_ev = None
    best_score = 0.0

    for ev in events:
        ev_time = _parse_dt(ev.get("commence_time"))
        if ev_time and abs((ev_time - kickoff).total_seconds()) > 18 * 3600:
            continue

        h = ev.get("home_team", "")
        a = ev.get("away_team", "")

        score = (_sim(home_name, h) + _sim(away_name, a)) / 2
        if score > best_score and score > 0.75:
            best_score = score
            best_ev = ev

    if best_ev:
        logger.info(
            f"odds match [{home_name} vs {away_name}] → "
            f"[{best_ev.get('home_team')} vs {best_ev.get('away_team')}] "
            f"score={best_score:.2f}"
        )
    return best_ev


# ─── public interface (same as before) ──────────────────────────────────────

async def fetch_odds_for_matches(
    client: OddsApiClient,
    upcoming: List[Tuple[int, str, str, str, datetime]],
) -> Dict[int, Dict[str, float]]:
    """upcoming: (match_id, league, home_name, away_name, utc_date).

    Returns {match_id: odds_dict}.
    """
    if not upcoming:
        return {}

    events = await client.fetch_mlb_odds()
    if not events:
        logger.warning("fetch_odds_for_matches: no events from the-odds-api")
        return {}

    out: Dict[int, Dict[str, float]] = {}
    for match_id, _league, home, away, kickoff in upcoming:
        ev = _find_event(home, away, kickoff, events)
        if ev is None:
            continue

        odds = extract_odds_v4(ev)
        h_ml = odds.get("odds_ml_home", 0.0)
        a_ml = odds.get("odds_ml_away", 0.0)

        if h_ml > 0 and a_ml > 0:
            vig = 1 / h_ml + 1 / a_ml
            if not (0.98 <= vig <= 1.25):
                logger.warning(
                    f"odds sanity FAIL match_id={match_id} [{home} vs {away}]: "
                    f"ml_home={h_ml:.2f} ml_away={a_ml:.2f} vig={vig:.3f} — discarding"
                )
                continue
            logger.info(
                f"odds OK match_id={match_id} [{home} vs {away}]: "
                f"ml_home={h_ml:.2f} ml_away={a_ml:.2f} "
                f"over={odds['odds_over85']:.2f} under={odds['odds_under85']:.2f}"
            )

        if any(v > 0 for v in odds.values()):
            out[match_id] = odds

    return out
