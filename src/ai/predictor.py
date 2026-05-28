"""Claude AI analysis for MLB games. JSON-only output enforced via system prompt."""
from __future__ import annotations

import json
import re
import time
from typing import Dict, Optional

from loguru import logger

from src.ai.commentary import call_llm, call_llm_with_search


_CACHE_TTL_SEC = 12 * 3600
_cache: Dict[int, tuple[float, dict]] = {}


async def _db_cache_get(match_id: int) -> Optional[dict]:
    from datetime import datetime, timedelta
    from src.data.database import AiPrediction, SessionLocal
    cutoff = datetime.utcnow() - timedelta(seconds=_CACHE_TTL_SEC)
    try:
        async with SessionLocal() as session:
            row = await session.get(AiPrediction, match_id)
            if row is None or row.created_at < cutoff:
                return None
            return json.loads(row.payload)
    except Exception as e:
        logger.warning(f"ai db cache read failed for {match_id}: {e}")
        return None


async def _db_cache_put(match_id: int, payload: dict) -> None:
    from datetime import datetime
    from src.data.database import AiPrediction, SessionLocal
    try:
        async with SessionLocal() as session:
            existing = await session.get(AiPrediction, match_id)
            data = json.dumps(payload)
            if existing is None:
                session.add(AiPrediction(
                    match_id=match_id, created_at=datetime.utcnow(), payload=data
                ))
            else:
                existing.created_at = datetime.utcnow()
                existing.payload = data
            await session.commit()
    except Exception as e:
        logger.warning(f"ai db cache write failed for {match_id}: {e}")


SYSTEM_PROMPT = (
    "Ты — профессиональный бейсбольный аналитик-квант. "
    "Отвечай ТОЛЬКО валидным JSON без markdown, без пояснений вне JSON. "
    "Формат ответа строго: "
    '{"market": "...", "pick": "...", "confidence": 0.XX, "reasoning": "..."}'
)

_PROMPT = """\
## МАТЧ
{home} (хозяева) vs {away} (гости) | MLB

## РЕАЛЬНЫЕ КОЭФФИЦИЕНТЫ БУКМЕКЕРОВ
Мани-лайн: {home} {ml_home} | {away} {ml_away}
Тотал {total_line}: Б {odds_over} | М {odds_under}
Ран-лайн (-1.5): {home} {rl_home} | {away} {rl_away}

## СТАРТОВЫЕ ПИТЧЕРЫ (из официальной БД MLB)
{home}: {home_pitcher_name} — ERA {home_era}, WHIP {home_whip}, K/9 {home_k9}, BB/9 {home_bb9}
{away}: {away_pitcher_name} — ERA {away_era}, WHIP {away_whip}, K/9 {away_k9}, BB/9 {away_bb9}

## СТАТИСТИКА СЕЗОНА (из нашей БД)
{home} последние 10 игр: {home_w10}/{home_g10} побед, среднее {home_avg_runs} ранов/игру
{away} последние 10 игр: {away_w10}/{away_g10} побед, среднее {away_avg_runs} ранов/игру
H2H в этом сезоне ({h2h_games} игр): {home} {h2h_home_wins}–{h2h_away_wins} {away}

## ДАННЫЕ ИЗ ВЕБ-ПОИСКА

### Питчеры / состав / травмы:
{web_pitchers}

### Погода / стадион:
{web_weather}

## ПРАВИЛА АНАЛИЗА

**Питчеры — главный фактор.**
ERA diff > 0.75 — сильный сигнал. ERA < 3.50 элита, > 4.50 слабый.
WHIP < 1.20 — доминирующий. WHIP > 1.40 — уязвим.

**Тотал.**
Оба питчера ERA < 3.50 → UNDER. Хотя бы один ERA > 4.50 → OVER.
Coors Field (Колорадо) +1.5 рана к тоталу. Petco Park (Сан-Диего) -1.0.
Ветер > 15 mph к центру → тотал вверх. Дождь/холод → тотал вниз.

**ML.**
ERA diff > 0.75 в пользу одного питчера — ставить на его команду.

**ПРАВИЛО ПРОПУСКА.** Если нет явного преимущества — верни confidence 0.50 и НЕ СТАВЬ.
Минимальный confidence для сигнала: 0.56.

## ВЫХОДНОЙ ФОРМАТ — ТОЛЬКО JSON

Выбери ОДНУ лучшую ставку:

{{"market": "TOTAL", "pick": "UNDER", "confidence": 0.63, "reasoning": "Chandler ERA 4.60 против Taillon ERA 5.20 — оба слабые стартеры, ожидаем высокий тотал"}}

market: "ML" | "TOTAL"
pick для ML: "HOME" | "AWAY"
pick для TOTAL: "OVER" | "UNDER"
confidence: 0.50–0.90 (если < 0.56 — не ставим, верни 0.50)
reasoning: 2–3 предложения на русском, конкретные факты\
"""

_PROMPT_NATIVE = """\
## МАТЧ
{home} (хозяева) vs {away} (гости) | MLB

## РЕАЛЬНЫЕ КОЭФФИЦИЕНТЫ БУКМЕКЕРОВ
Мани-лайн: {home} {ml_home} | {away} {ml_away}
Тотал {total_line}: Б {odds_over} | М {odds_under}

## СТАРТОВЫЕ ПИТЧЕРЫ (из официальной БД MLB)
{home}: {home_pitcher_name} — ERA {home_era}, WHIP {home_whip}
{away}: {away_pitcher_name} — ERA {away_era}, WHIP {away_whip}

## ЧТО ИСКАТЬ В ИНТЕРНЕТЕ
1. Актуальные ERA, WHIP, последние 3 выхода каждого питчера
2. Травмы ключевых хиттеров (IL/DL)
3. Усталость буллпена (3+ игры подряд)
4. Погода: ветер (направление, скорость), температура, осадки

## ПРАВИЛА АНАЛИЗА

**Питчеры — главный фактор.**
ERA diff > 0.75 — сильный сигнал. ERA < 3.50 элита, > 4.50 слабый.

**Тотал.**
Оба ERA < 3.50 → UNDER. Хотя бы один ERA > 4.50 → OVER.
Coors Field +1.5. Petco -1.0. Ветер > 15 mph к центру → тотал вверх.

**ПРАВИЛО ПРОПУСКА.** Нет явного преимущества → confidence 0.50, не ставим.

## ВЫХОДНОЙ ФОРМАТ — ТОЛЬКО JSON

{{"market": "TOTAL", "pick": "UNDER", "confidence": 0.63, "reasoning": "2-3 предложения"}}

market: "ML" | "TOTAL"
pick для ML: "HOME" | "AWAY"
pick для TOTAL: "OVER" | "UNDER"
confidence: 0.50–0.90\
"""


def _fmt(v, fmt=".2f") -> str:
    try:
        f = float(v)
        return format(f, fmt) if f > 0 else "н/д"
    except (TypeError, ValueError):
        return "н/д"


def _parse_json_strict(raw: str) -> Optional[dict]:
    if not raw:
        return None
    raw = raw.strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    # Strip markdown code blocks
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Find first JSON object
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


_VALID_PICKS = {
    "ML": {"HOME", "AWAY"},
    "TOTAL": {"OVER", "UNDER"},
}


def _validate_pick(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    market = d.get("market")
    pick = d.get("pick")
    confidence = d.get("confidence")
    if market not in _VALID_PICKS:
        return False
    if pick not in _VALID_PICKS[market]:
        return False
    if not isinstance(confidence, (int, float)):
        return False
    d["confidence"] = max(0.50, min(1.0, float(confidence)))
    return True


async def _build_proxy_prompt(home: str, away: str, cfg, features: dict) -> str:
    from src.data.web_search import tavily_search
    import asyncio

    web_pitchers, web_weather = await asyncio.gather(
        tavily_search(f"{home} vs {away} starting pitcher ERA WHIP lineup injury MLB today", days=3),
        tavily_search(f"MLB {home} {away} weather wind temperature stadium", days=2),
    )

    def _format_web(results: list, max_items: int = 4) -> str:
        if not results:
            return "(данных нет)"
        lines = []
        for r in results[:max_items]:
            title = r.get("title", "").strip()
            content = r.get("content", "").strip()
            lines.append(f"• {title}: {content[:300]}")
        return "\n".join(lines)

    return _PROMPT.format(
        home=home, away=away,
        total_line=cfg.total_line,
        ml_home=_fmt(features.get("odds_ml_home")),
        ml_away=_fmt(features.get("odds_ml_away")),
        odds_over=_fmt(features.get("odds_over85")),
        odds_under=_fmt(features.get("odds_under85")),
        rl_home=_fmt(features.get("odds_rl_home")),
        rl_away=_fmt(features.get("odds_rl_away")),
        home_pitcher_name=features.get("home_pitcher_name") or "неизвестен",
        home_era=_fmt(features.get("home_pitcher_era")),
        home_whip=_fmt(features.get("home_pitcher_whip")),
        home_k9=_fmt(features.get("home_pitcher_k9")),
        home_bb9=_fmt(features.get("home_pitcher_bb9")),
        away_pitcher_name=features.get("away_pitcher_name") or "неизвестен",
        away_era=_fmt(features.get("away_pitcher_era")),
        away_whip=_fmt(features.get("away_pitcher_whip")),
        away_k9=_fmt(features.get("away_pitcher_k9")),
        away_bb9=_fmt(features.get("away_pitcher_bb9")),
        home_w10=features.get("home_last10_wins", 0),
        home_g10=features.get("home_last10_games", 0),
        home_avg_runs=features.get("home_avg_runs", 0.0),
        away_w10=features.get("away_last10_wins", 0),
        away_g10=features.get("away_last10_games", 0),
        away_avg_runs=features.get("away_avg_runs", 0.0),
        h2h_games=features.get("h2h_games", 0),
        h2h_home_wins=features.get("h2h_home_wins", 0),
        h2h_away_wins=features.get("h2h_away_wins", 0),
        web_pitchers=_format_web(web_pitchers),
        web_weather=_format_web(web_weather),
    )


async def ai_predict(
    *,
    match_id: int,
    home: str,
    away: str,
    competition: str,
    ml_probs: dict,
    features: dict,
) -> Optional[dict]:
    """Returns {market, pick, confidence, reasoning} or None."""
    from src.config import settings as cfg

    now = time.time()
    cached = _cache.get(match_id)
    if cached and now - cached[0] < _CACHE_TTL_SEC:
        if _validate_pick(cached[1]):
            return cached[1]

    db_cached = await _db_cache_get(match_id)
    if db_cached is not None and _validate_pick(db_cached):
        _cache[match_id] = (now, db_cached)
        return db_cached

    if cfg.llm_base_url:
        prompt = await _build_proxy_prompt(home, away, cfg, features)
        raw = await call_llm(prompt, system=SYSTEM_PROMPT, max_tokens=600)
        mode = "proxy+tavily"
    else:
        prompt = _PROMPT_NATIVE.format(
            home=home, away=away,
            total_line=cfg.total_line,
            ml_home=_fmt(features.get("odds_ml_home")),
            ml_away=_fmt(features.get("odds_ml_away")),
            odds_over=_fmt(features.get("odds_over85")),
            odds_under=_fmt(features.get("odds_under85")),
            home_pitcher_name=features.get("home_pitcher_name") or "неизвестен",
            home_era=_fmt(features.get("home_pitcher_era")),
            home_whip=_fmt(features.get("home_pitcher_whip")),
            away_pitcher_name=features.get("away_pitcher_name") or "неизвестен",
            away_era=_fmt(features.get("away_pitcher_era")),
            away_whip=_fmt(features.get("away_pitcher_whip")),
        )
        raw = await call_llm_with_search(prompt, system=SYSTEM_PROMPT, max_tokens=800)
        mode = "anthropic+websearch"

    if not raw:
        logger.warning(f"ai_predict({match_id}): empty response [{mode}]")
        return None

    parsed = _parse_json_strict(raw)
    if not parsed or not _validate_pick(parsed):
        logger.warning(f"ai_predict({match_id}): invalid JSON [{mode}]: {raw[:300]}")
        return None

    # Enforce min confidence threshold
    if parsed["confidence"] < 0.56:
        logger.info(f"ai_predict({match_id}): confidence {parsed['confidence']:.2f} < 0.56 — skip")
        return None

    _cache[match_id] = (now, parsed)
    await _db_cache_put(match_id, parsed)
    logger.info(
        f"ai_predict({match_id}) {home} vs {away} [{mode}]: "
        f"{parsed['market']} {parsed['pick']} conf={parsed['confidence']:.2f}"
    )
    return parsed
