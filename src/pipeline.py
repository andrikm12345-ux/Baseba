"""End-to-end pipeline: ingest → features → train → predict → emit signals."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import List

import pandas as pd
from loguru import logger
from sqlalchemy import select

from src.ai.predictor import ai_predict
from src.bot.formatters import format_signal, format_training_report
from src.bot.handlers import broadcast_signal
from src.config import settings
from src.data.database import AiPrediction, Match, SessionLocal, Signal as SignalRow, Team
from src.data.features import build_features, build_inference_features
from src.data.mlb_api import MlbApiClient
from src.data.ingest import ingest_history, ingest_upcoming
from src.data.odds_api import OddsApiClient, fetch_odds_for_matches
from src.data.settings_store import get_bool
from src.ml.predict import Predictor
from src.ml.train import train_all
from src.signals.generator import Signal, generate
from src.signals.tracker import settle_pending


async def _load_games_df() -> pd.DataFrame:
    async with SessionLocal() as session:
        rows = (await session.execute(select(Match))).scalars().all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "id": m.id,
            "utc_date": m.utc_date,
            "home_team_id": m.home_team_id,
            "away_team_id": m.away_team_id,
            "home_runs": m.home_runs,
            "away_runs": m.away_runs,
            "competition": m.competition,
            "status": m.status,
            "home_pitcher_era": m.home_pitcher_era,
            "home_pitcher_whip": m.home_pitcher_whip,
            "home_pitcher_k9": m.home_pitcher_k9,
            "home_pitcher_bb9": m.home_pitcher_bb9,
            "away_pitcher_era": m.away_pitcher_era,
            "away_pitcher_whip": m.away_pitcher_whip,
            "away_pitcher_k9": m.away_pitcher_k9,
            "away_pitcher_bb9": m.away_pitcher_bb9,
        }
        for m in rows
    ])


async def bootstrap_history(seasons: List[int] | None = None) -> int:
    """Initial load of historical MLB data — call once after first deploy."""
    if seasons is None:
        this_year = datetime.utcnow().year
        seasons = [this_year - 3, this_year - 2, this_year - 1]
    client = MlbApiClient(sport_id=settings.mlb_sport_id)
    try:
        n = await ingest_history(client, seasons)
        return n
    finally:
        await client.close()


async def refresh_upcoming(days: int = 7) -> int:
    client = MlbApiClient(sport_id=settings.mlb_sport_id)
    try:
        return await ingest_upcoming(client, days_ahead=days)
    finally:
        await client.close()


async def train_models(bot=None) -> None:
    df = await _load_games_df()
    if df.empty:
        logger.warning("No games in DB — skip training")
        return
    finished = df[df["status"] == "FINISHED"].copy()
    # Build features uses home_runs/away_runs columns
    finished = finished.rename(columns={"home_runs": "home_runs", "away_runs": "away_runs"})
    if len(finished) < 200:
        logger.warning(f"Only {len(finished)} finished games — not training yet")
        return
    features = build_features(finished)
    result = train_all(features)
    logger.info(f"Models saved: {result['paths']}")
    if bot:
        text = format_training_report(result["metrics"])
        for admin_id in settings.admin_ids:
            try:
                await bot.send_message(admin_id, text, parse_mode="HTML")
            except Exception as e:
                logger.warning(f"train notify to {admin_id} failed: {e}")


async def _store_signals(
    signals: List[Signal], ai_match_ids: set[int] | None = None
) -> List[SignalRow]:
    """Persist signals, de-duplicating by (match, market, pick)."""
    stored: List[SignalRow] = []
    ai_ids = ai_match_ids or set()
    async with SessionLocal() as session:
        for s in signals:
            exists = (await session.execute(
                select(SignalRow).where(
                    SignalRow.match_id == s.match_id,
                    SignalRow.market == s.market,
                    SignalRow.pick == s.pick,
                )
            )).scalar_one_or_none()
            if exists is not None:
                continue
            row = SignalRow(
                match_id=s.match_id, market=s.market, pick=s.pick,
                is_ai_ensemble=(s.match_id in ai_ids),
                is_value=s.is_value,
                model_prob=s.model_prob, fair_odds=s.fair_odds,
                book_odds=s.book_odds, edge=s.edge, confidence=s.confidence,
                stake_units=s.stake_units,
            )
            session.add(row)
            stored.append(row)
        await session.commit()
    return stored


async def generate_and_broadcast(bot) -> int:
    """Generate signals for upcoming games and broadcast new ones."""
    predictor = Predictor()
    if not predictor.ready:
        logger.warning("Models not ready — skip signal generation")
        return 0
    df = await _load_games_df()
    if df.empty:
        return 0
    finished = df[df["status"] == "FINISHED"].copy()
    now = datetime.utcnow()
    horizon = now + timedelta(hours=3)
    # Только pre-game в окне до 3 часов до старта
    upcoming = df[
        (df["status"] != "FINISHED")
        & (df["utc_date"] >= now - timedelta(minutes=15))
        & (df["utc_date"] <= horizon)
    ].copy()
    if upcoming.empty:
        logger.info("No upcoming games in the next 24 hours")
        return 0
    feats = build_inference_features(upcoming, finished)
    if feats.empty:
        return 0
    preds = predictor.predict(feats)

    if settings.odds_api_key:
        odds_client = OddsApiClient(settings.odds_api_key)
        try:
            async with SessionLocal() as session:
                tuples = []
                for mid in preds["match_id"].tolist():
                    match = await session.get(Match, int(mid))
                    if not match:
                        continue
                    home = await session.get(Team, match.home_team_id)
                    away = await session.get(Team, match.away_team_id)
                    if home and away:
                        tuples.append((match.id, match.competition, home.name, away.name, match.utc_date))
            odds_map = await fetch_odds_for_matches(odds_client, tuples)
        finally:
            await odds_client.close()
        for col in ["odds_ml_home", "odds_ml_away", "odds_over85", "odds_under85", "odds_rl_home", "odds_rl_away"]:
            preds[col] = preds["match_id"].map(lambda m: (odds_map.get(int(m)) or {}).get(col, 0.0))
        logger.info(f"Attached odds to {sum(1 for v in odds_map.values() if v)}/{len(preds)} games")

    ai_match_ids: set[int] = set()
    if await get_bool("ai_ensemble_enabled", False):
        preds, ai_match_ids = await _apply_ai_ensemble(preds, feats)

    preds["_ai_applied"] = preds["match_id"].isin(ai_match_ids)
    signals = generate(preds)
    new_rows = await _store_signals(signals, ai_match_ids=ai_match_ids)
    sent = 0
    ai_on = await get_bool("ai_ensemble_enabled", False)
    if new_rows and bot:
        async with SessionLocal() as session:
            for row in new_rows:
                match = await session.get(Match, row.match_id)
                if not match:
                    continue
                home = await session.get(Team, match.home_team_id)
                away = await session.get(Team, match.away_team_id)
                ai_comment = None
                if ai_on and row.match_id in ai_match_ids:
                    cached_ai = await session.get(AiPrediction, row.match_id)
                    if cached_ai:
                        try:
                            ai_comment = json.loads(cached_ai.payload).get("reasoning")
                        except Exception:
                            ai_comment = None
                    if ai_comment:
                        stored = await session.get(SignalRow, row.id)
                        if stored:
                            stored.commentary = ai_comment
                            await session.commit()
                text = format_signal(row, match, home, away, ai_comment)
                sent += await broadcast_signal(bot, text)
    logger.info(
        f"Generated {len(new_rows)} new signals, broadcast {sent} messages "
        f"(AI={'on' if ai_on else 'off'})"
    )
    return len(new_rows)


async def _apply_ai_ensemble(
    preds: pd.DataFrame, feats: pd.DataFrame
) -> tuple[pd.DataFrame, set[int]]:
    import asyncio

    weight = settings.ai_ensemble_weight
    top_n = settings.ai_ensemble_top_n
    threshold = settings.ai_ensemble_min_prob

    feats_by_id = {int(r["match_id"]): r.to_dict() for _, r in feats.iterrows()}
    preds = preds.copy()
    preds["_max_prob"] = preds[["p_home", "p_away"]].max(axis=1)
    candidates = (
        preds[preds["_max_prob"] >= threshold]
        .sort_values("_max_prob", ascending=False)
        .head(top_n)
    )
    preds.drop(columns=["_max_prob"], inplace=True)

    if candidates.empty:
        return preds, set()

    async with SessionLocal() as session:
        existing_rows = (await session.execute(
            select(SignalRow.match_id)
            .where(SignalRow.match_id.in_([int(m) for m in candidates["match_id"]]))
            .distinct()
        )).scalars().all()
        already_signaled = {int(m) for m in existing_rows}
        tasks = []
        match_meta = {}
        for _, row in candidates.iterrows():
            mid = int(row["match_id"])
            if mid in already_signaled:
                continue
            match = await session.get(Match, mid)
            if not match:
                continue
            home = await session.get(Team, match.home_team_id)
            away = await session.get(Team, match.away_team_id)
            if not home or not away:
                continue
            match_meta[mid] = (home.name, away.name, match.competition, match)
            tasks.append((mid, row))

    sem = asyncio.Semaphore(3)

    async def _one(mid: int, row) -> tuple[int, dict | None]:
        async with sem:
            ml_probs = {
                "p_home": float(row["p_home"]),
                "p_away": float(row["p_away"]),
                "p_over85": float(row["p_over85"]),
                "p_rl_home": float(row["p_rl_home"]),
            }
            home, away, comp, match_obj = match_meta[mid]
            feat_dict = feats_by_id.get(mid, {})
            # Enrich features with pitcher names from Match row
            feat_dict["home_pitcher_name"] = match_obj.home_pitcher_name
            feat_dict["away_pitcher_name"] = match_obj.away_pitcher_name
            ai = await ai_predict(
                match_id=mid,
                home=home, away=away, competition=comp,
                ml_probs=ml_probs,
                features=feat_dict,
            )
            return mid, ai

    results = await asyncio.gather(*[_one(mid, r) for mid, r in tasks])

    applied: set[int] = set()
    diffs = []
    for mid, ai in results:
        if not ai:
            continue
        mask = preds["match_id"] == mid
        for col_pred, col_ai in [
            ("p_home", "p_home"),
            ("p_away", "p_away"),
            ("p_over85", "p_over85"),
            ("p_rl_home", "p_rl_home"),
        ]:
            ml_v = float(preds.loc[mask, col_pred].iloc[0])
            ai_v = float(ai.get(col_ai, ml_v))
            diffs.append(abs(ml_v - ai_v))
            preds.loc[mask, col_pred] = (1 - weight) * ml_v + weight * ai_v
        applied.add(mid)

    if applied:
        avg_diff = sum(diffs) / len(diffs) if diffs else 0
        logger.info(
            f"AI ensemble applied to {len(applied)}/{len(tasks)} games "
            f"(weight={weight}, avg |ml-ai| = {avg_diff:.3f})"
        )
    return preds, applied


async def daily_cycle(bot) -> None:
    logger.info("Daily cycle start")
    await refresh_upcoming(days=7)
    await settle_pending()
    await train_models(bot=bot)
    await generate_and_broadcast(bot)
    logger.info("Daily cycle done")
