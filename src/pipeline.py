"""Pipeline: ingest → Claude AI analysis → emit signals.

XGBoost model removed. Claude analyzes every game with real odds.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import List

from loguru import logger
from sqlalchemy import select

from src.ai.predictor import ai_predict
from src.bot.formatters import format_signal
from src.bot.handlers import broadcast_signal
from src.config import settings
from src.data.database import Match, SessionLocal, Signal as SignalRow, Team
from src.data.mlb_api import MlbApiClient
from src.data.ingest import ingest_upcoming
from src.signals.generator import Signal, _kelly, MAX_EDGE
from src.data.odds_api import OddsApiClient, fetch_odds_for_matches
from src.signals.tracker import settle_pending, enrich_scores_from_odds_api


async def refresh_upcoming(days: int = 7) -> int:
    client = MlbApiClient(sport_id=settings.mlb_sport_id)
    try:
        return await ingest_upcoming(client, days_ahead=days)
    finally:
        await client.close()


async def _store_signals(signals: List[Signal]) -> List[SignalRow]:
    """Persist signals. One game = one signal (dedup by match_id)."""
    stored: List[SignalRow] = []
    async with SessionLocal() as session:
        for s in signals:
            exists = (await session.execute(
                select(SignalRow).where(SignalRow.match_id == s.match_id)
            )).scalar_one_or_none()
            if exists is not None:
                continue
            row = SignalRow(
                match_id=s.match_id, market=s.market, pick=s.pick,
                line=s.line,
                is_ai_ensemble=True,
                is_value=s.is_value,
                model_prob=s.model_prob, fair_odds=s.fair_odds,
                book_odds=s.book_odds, edge=s.edge, confidence=s.confidence,
                stake_units=s.stake_units,
            )
            session.add(row)
            stored.append(row)
        await session.commit()
    return stored


def _find_line(lines: list, point) -> dict | None:
    """Find the quote entry matching the line Claude picked (tolerance 0.01)."""
    try:
        pt = float(point)
    except (TypeError, ValueError):
        return None
    for entry in lines or []:
        if abs(entry.get("point", -999) - pt) < 0.01:
            return entry
    return None


def _ai_to_signal(ai_result: dict, match_id: int, odds: dict) -> Signal | None:
    """Convert Claude output to Signal.

    Edge is measured vs the market's no-vig probability (real divergence),
    not vs the raw confidence. Publishes only at odds >= min_odds and
    edge >= min_edge.
    """
    market = ai_result.get("market")
    pick = ai_result.get("pick")
    confidence = float(ai_result.get("confidence", 0.0))
    line = ai_result.get("line")
    if not market or not pick or confidence < 0.56:
        return None

    book: float = 0.0
    novig: float | None = None
    sig_line: float | None = None

    if market == "ML":
        book = float(odds.get("odds_ml_home" if pick == "HOME" else "odds_ml_away", 0.0))
        from src.data.odds_api import novig_two_way
        nv = novig_two_way(odds.get("odds_ml_home"), odds.get("odds_ml_away"))
        if nv:
            novig = nv[0] if pick == "HOME" else nv[1]

    elif market == "TOTAL":
        entry = _find_line(odds.get("totals_lines"), line)
        if entry is None:
            return None
        sig_line = entry["point"]
        if pick == "OVER":
            book, novig = entry["over"], entry.get("over_novig")
        else:
            book, novig = entry["under"], entry.get("under_novig")

    elif market == "RL":
        entry = _find_line(odds.get("spread_lines"), line)
        if entry is None:
            return None
        sig_line = entry["point"]
        if pick == "COVER":
            book, novig = entry["home"], entry.get("home_novig")
        else:  # AWAY_COVER
            book, novig = entry["away"], entry.get("away_novig")

    if not book or book < settings.min_odds:
        return None

    # Real edge = Claude's probability − market no-vig probability
    if novig is not None:
        edge = confidence - float(novig)
    else:
        edge = confidence * book - 1.0  # fallback if no-vig unavailable
    if edge < settings.min_edge:
        return None
    edge = min(edge, MAX_EDGE)

    stake = _kelly(confidence, book)
    return Signal(
        match_id=match_id,
        market=market, pick=pick,
        model_prob=confidence, fair_odds=1.0 / max(confidence, 1e-6),
        book_odds=float(book), edge=float(edge),
        confidence=confidence,
        stake_units=float(round(stake, 2)),
        is_value=True,
        line=sig_line,
    )


async def generate_and_broadcast(bot) -> int:
    """Claude analyzes every upcoming game with real odds and broadcasts signals."""
    now = datetime.utcnow()
    horizon = now + timedelta(hours=5)

    # Load upcoming games in 5h window
    async with SessionLocal() as session:
        upcoming_matches = (await session.execute(
            select(Match).where(
                Match.status != "FINISHED",
                Match.utc_date >= now,
                Match.utc_date <= horizon,
            ).order_by(Match.utc_date.asc())
        )).scalars().all()

        if not upcoming_matches:
            logger.info("No games in 5h window")
            return 0

        teams_cache: dict[int, Team] = {}
        for m in upcoming_matches:
            for tid in (m.home_team_id, m.away_team_id):
                if tid not in teams_cache:
                    t = await session.get(Team, tid)
                    if t:
                        teams_cache[tid] = t

        # Filter out already-signaled games
        existing_ids = set((await session.execute(
            select(SignalRow.match_id).where(
                SignalRow.match_id.in_([m.id for m in upcoming_matches])
            )
        )).scalars().all())

    to_analyze = [m for m in upcoming_matches if m.id not in existing_ids]
    if not to_analyze:
        logger.info("All games in window already have signals")
        return 0

    logger.info(
        f"generate_and_broadcast: {len(upcoming_matches)} in window, "
        f"{len(to_analyze)} need signals"
    )

    # Fetch odds
    odds_map: dict[int, dict] = {}
    if settings.odds_api_key:
        odds_client = OddsApiClient(settings.odds_api_key)
        try:
            tuples = []
            for m in to_analyze:
                ht = teams_cache.get(m.home_team_id)
                at = teams_cache.get(m.away_team_id)
                if ht and at:
                    tuples.append((m.id, m.competition, ht.name, at.name, m.utc_date))
            odds_map = await fetch_odds_for_matches(odds_client, tuples)
        finally:
            await odds_client.close()
        logger.info(f"Got odds for {len(odds_map)}/{len(to_analyze)} games")

    # Claude analyzes each game with odds
    sem = asyncio.Semaphore(3)

    async def _analyze_one(match: Match) -> tuple[int, dict | None, dict]:
        odds = odds_map.get(match.id, {})
        ml_h = odds.get("odds_ml_home", 0.0)
        ml_a = odds.get("odds_ml_away", 0.0)
        # Skip games without minimum viable odds
        if not ml_h or not ml_a or (ml_h < settings.min_odds and ml_a < settings.min_odds):
            logger.debug(f"Skipping {match.id}: no valid odds (ml_h={ml_h} ml_a={ml_a})")
            return match.id, None, odds

        ht = teams_cache.get(match.home_team_id)
        at = teams_cache.get(match.away_team_id)
        home_name = ht.name if ht else f"Team#{match.home_team_id}"
        away_name = at.name if at else f"Team#{match.away_team_id}"

        feat_dict = {
            "home_pitcher_name": match.home_pitcher_name,
            "away_pitcher_name": match.away_pitcher_name,
            "home_pitcher_era": match.home_pitcher_era,
            "home_pitcher_whip": match.home_pitcher_whip,
            "home_pitcher_k9": match.home_pitcher_k9,
            "home_pitcher_bb9": match.home_pitcher_bb9,
            "away_pitcher_era": match.away_pitcher_era,
            "away_pitcher_whip": match.away_pitcher_whip,
            "away_pitcher_k9": match.away_pitcher_k9,
            "away_pitcher_bb9": match.away_pitcher_bb9,
            "odds_ml_home": ml_h,
            "odds_ml_away": ml_a,
            "totals_lines": odds.get("totals_lines", []),
            "spread_lines": odds.get("spread_lines", []),
        }

        # H2H + team form from DB
        try:
            from src.data.team_stats import get_team_context
            ctx = await get_team_context(
                match.home_team_id, match.away_team_id, match.season or 2026
            )
            feat_dict.update(ctx)
        except Exception as e:
            logger.debug(f"get_team_context failed for {match.id}: {e}")

        async with sem:
            ai = await ai_predict(
                match_id=match.id,
                home=home_name, away=away_name,
                competition=match.competition or "MLB",
                features=feat_dict,
            )
        return match.id, ai, odds

    results = await asyncio.gather(*[_analyze_one(m) for m in to_analyze])

    signals: List[Signal] = []
    ai_results_cache: dict[int, dict] = {}
    for mid, ai, odds in results:
        if not ai:
            continue
        sig = _ai_to_signal(ai, mid, odds)
        if sig is None:
            continue
        signals.append(sig)
        ai_results_cache[mid] = ai

    new_rows = await _store_signals(signals)

    # Attach Claude reasoning and broadcast
    sent = 0
    if new_rows and bot:
        async with SessionLocal() as session:
            for row in new_rows:
                match = await session.get(Match, row.match_id)
                if not match:
                    continue
                home = await session.get(Team, match.home_team_id)
                away = await session.get(Team, match.away_team_id)

                ai_comment = None
                ai_res = ai_results_cache.get(row.match_id)
                if ai_res:
                    ai_comment = ai_res.get("reasoning")
                    if ai_comment:
                        stored = await session.get(SignalRow, row.id)
                        if stored:
                            stored.commentary = ai_comment
                            await session.commit()

                text = format_signal(row, match, home, away, ai_comment)
                sent += await broadcast_signal(bot, text)

    logger.info(f"Claude generated {len(new_rows)} signals, broadcast {sent}")
    return len(new_rows)


async def daily_cycle(bot) -> None:
    logger.info("Daily cycle start")
    await refresh_upcoming(days=7)
    if settings.odds_api_key:
        _settle_client = OddsApiClient(settings.odds_api_key)
        try:
            await enrich_scores_from_odds_api(_settle_client)
        finally:
            await _settle_client.close()
    await settle_pending()
    await generate_and_broadcast(bot)
    logger.info("Daily cycle done")
