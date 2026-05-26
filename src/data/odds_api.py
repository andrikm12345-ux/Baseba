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
    "mlb": "usa-mlb",
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
TEAM_TOTAL_NAMES = {
    "team totals", "team total", "home team total", "away team total",
    "team total runs", "alternate team totals", "batter totals",
}


class OddsApiError(Exception):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"odds-api.io {status}: {body[:200]}")
        self.status = status
        self.body = body


class OddsApiClient:
    def __init__(self, api_key: str, cache_ttl_seconds: float = 300.0) -> None:
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
    "angel stadium": "los angeles angels",
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
    "new york mets": "new york mets", "mets": "new york mets",
    "ny mets": "new york mets",
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
    """Normalize MLB team name to canonical form for exact matching."""
    return _MLB_CANONICAL.get(name.lower().strip(), name.lower().strip())


# ─── matching ───────────────────────────────────────────────────────────────

def _sim(a: str, b: str) -> float:
    ca, cb = _canonical(a), _canonical(b)
    if ca == cb:
        return 1.0
    return SequenceMatcher(None, ca, cb).ratio()


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
) -> Optional[Tuple[Dict[str, Any], bool]]:
    """Return (event, is_swapped) or None.

    is_swapped=True means the Odds API has home/away in reversed order —
    the caller must swap odds_ml_home ↔ odds_ml_away etc.
    """
    best_ev = None
    best_score = 0.0
    best_swapped = False

    for ev in events:
        ev_time = _event_kickoff(ev)
        if ev_time and abs((ev_time - kickoff).total_seconds()) > 6 * 3600:
            continue
        h, a = _event_teams(ev)
        if not h or not a:
            continue

        # Direct order: home→h, away→a
        s_direct = (_sim(home_name, h) + _sim(away_name, a)) / 2
        # Reversed order: home→a, away→h (some providers list teams differently)
        s_rev = (_sim(home_name, a) + _sim(away_name, h)) / 2

        score = max(s_direct, s_rev)
        swapped = s_rev > s_direct

        if score > best_score and score > 0.75:
            best_score = score
            best_ev = ev
            best_swapped = swapped

    if best_ev is not None:
        h, a = _event_teams(best_ev)
        logger.info(
            f"odds match [{home_name} vs {away_name}] → "
            f"[{h} vs {a}] score={best_score:.2f} swapped={best_swapped}"
        )
    else:
        logger.debug(f"odds match: no event found for [{home_name} vs {away_name}]")

    return (best_ev, best_swapped) if best_ev is not None else None


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
        "odds_rl_home": [],       # home -1.5 (hdp < 0)
        "odds_rl_away": [],       # away +1.5 (hdp < 0)
        "odds_rl_away_cover": [], # away -1.5 (hdp > 0)
        "odds_rl_home_lay": [],   # home +1.5 (hdp > 0)
        "odds_itb_home": [],
        "odds_itb_away": [],
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
                    # Only back odds (не lay): lay-коэффициенты на Betfair биржевые
                    # и могут быть 50-110 на явного проигрывающего — они нерелевантны
                    h = _as_float(entry.get("home"))
                    a = _as_float(entry.get("away"))
                    if h and h < 15.0:
                        aggregated["odds_ml_home"].append(h)
                    if a and a < 15.0:
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
                    if over and over < 15.0:
                        aggregated["odds_over85"].append(over)
                    if under and under < 15.0:
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
                    h = _as_float(entry.get("home"))
                    a = _as_float(entry.get("away"))
                    if hdp_f < 0:
                        # home -1.5 / away +1.5 (standard: home gives runs)
                        if h and h < 15.0:
                            aggregated["odds_rl_home"].append(h)
                        if a and a < 15.0:
                            aggregated["odds_rl_away"].append(a)
                    else:
                        # home +1.5 / away -1.5 (away is favourite, gives runs)
                        if h and h < 15.0:
                            aggregated["odds_rl_home_lay"].append(h)
                        if a and a < 15.0:
                            aggregated["odds_rl_away_cover"].append(a)

            elif name in TEAM_TOTAL_NAMES:
                for entry in odds_list:
                    if not isinstance(entry, dict):
                        continue
                    hdp = entry.get("hdp") or entry.get("line") or entry.get("total")
                    try:
                        hdp_f = float(hdp) if hdp is not None else None
                    except (TypeError, ValueError):
                        hdp_f = None
                    if hdp_f is not None and abs(hdp_f - settings.itb_line) > 0.3:
                        continue

                    # Format 1: home/away nested dicts
                    home_data = entry.get("home")
                    away_data = entry.get("away")
                    if isinstance(home_data, dict):
                        h_over = _as_float(home_data.get("over"))
                        if h_over and h_over < 15.0:
                            aggregated["odds_itb_home"].append(h_over)
                    if isinstance(away_data, dict):
                        a_over = _as_float(away_data.get("over"))
                        if a_over and a_over < 15.0:
                            aggregated["odds_itb_away"].append(a_over)

                    # Format 2: flat entry with team indicator
                    team = str(entry.get("team") or entry.get("name") or "").lower()
                    over_val = _as_float(entry.get("over"))
                    if over_val and over_val < 15.0:
                        if "home" in team:
                            aggregated["odds_itb_home"].append(over_val)
                        elif "away" in team:
                            aggregated["odds_itb_away"].append(over_val)

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
        result = _best_match(home, away, kickoff, events)
        if result is None:
            continue
        ev, is_swapped = result
        ev_id = ev.get("id") or ev.get("eventId") or ev.get("event_id")
        if not ev_id:
            continue
        payload = await client.fetch_event_odds(ev_id)
        if not payload:
            continue
        odds = extract_odds(payload)

        # If teams were matched in reversed order, swap home↔away odds
        if is_swapped:
            odds["odds_ml_home"], odds["odds_ml_away"] = odds["odds_ml_away"], odds["odds_ml_home"]
            odds["odds_rl_home"], odds["odds_rl_away"] = odds["odds_rl_away"], odds["odds_rl_home"]
            odds["odds_rl_home_lay"], odds["odds_rl_away_cover"] = (
                odds["odds_rl_away_cover"], odds["odds_rl_home_lay"]
            )
            odds["odds_itb_home"], odds["odds_itb_away"] = odds["odds_itb_away"], odds["odds_itb_home"]
            logger.info(
                f"odds swap applied for match_id={match_id} [{home} vs {away}]: "
                f"ml_home={odds['odds_ml_home']:.2f} ml_away={odds['odds_ml_away']:.2f}"
            )

        # Sanity check: ML vigorish should be 2–15% (implied probs sum 1.02–1.15)
        h_ml, a_ml = odds.get("odds_ml_home", 0.0), odds.get("odds_ml_away", 0.0)
        if h_ml > 0 and a_ml > 0:
            vig = 1 / h_ml + 1 / a_ml
            if not (1.02 <= vig <= 1.20):
                logger.warning(
                    f"odds sanity FAIL match_id={match_id} [{home} vs {away}]: "
                    f"ml_home={h_ml:.2f} ml_away={a_ml:.2f} vig={vig:.3f} — discarding"
                )
                continue

        if any(v > 0 for v in odds.values()):
            out[match_id] = odds
            logger.info(
                f"odds OK match_id={match_id} [{home} vs {away}]: "
                f"ml_home={h_ml:.2f} ml_away={a_ml:.2f}"
            )

    return out
