"""AI ансамбль: Claude самостоятельно ищет данные в интернете и выбирает ставку."""
from __future__ import annotations

import json
import re
import time
from typing import Dict, Optional

from loguru import logger

from src.ai.commentary import call_llm_with_search


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


_PROMPT = """Ты — Elite Baseball Quant Analyst. Твоя задача: проанализировать игру MLB и выбрать ОДНУ лучшую ставку.

## ИГРА
{home} (хозяева) vs {away} (гости) | MLB | Тотал-линия: {total_line} | ИТБ-линия: {itb_line}

## СПРАВКА ОТ ML-МОДЕЛИ (вспомогательно, не главное)
P(победа {home}) ≈ {p_home:.0%} | P(победа {away}) ≈ {p_away:.0%}
P(тотал > {total_line}) ≈ {p_over85:.0%}
P(ИТБ хоз > {itb_line}) ≈ {p_itb_home:.0%} | P(ИТБ гост > {itb_line}) ≈ {p_itb_away:.0%}
Из БД: {home} — {home_pitcher_name} (ERA {home_era}, WHIP {home_whip}) | {away} — {away_pitcher_name} (ERA {away_era}, WHIP {away_whip})

## ЧТО ИСКАТЬ В ИНТЕРНЕТЕ
Используй веб-поиск чтобы найти свежие данные об этой конкретной игре:
1. Стартовые питчеры: актуальные ERA, WHIP, K/9, последние 3 выхода, дни отдыха
2. Буллпен: усталость (сколько игр подряд работали ключевые питчеры), Relief ERA
3. Состав: травмы ключевых хиттеров, DL, IL
4. Коэффициенты букмекеров на эту игру (мани-лайн, тотал)
5. Погода на стадионе: ветер (скорость и направление), температура, осадки

## КАК АНАЛИЗИРОВАТЬ

**1. Питчеры** — главный фактор MLB.
ERA diff > 0.75 между стартерами — сильный сигнал. ERA < 3.50 элита, > 4.50 слабый.
Питчер на 4-й день отдыха после 100+ питчей — сниженная эффективность.

**2. Буллпен**
Усталость (3+ игры подряд) резко повышает тотал. Если команда вела в 8-м иннинге последние 3 игры — буллпен на износе.

**3. Park Factor**
Coors Field (Колорадо) ≈ +1.5 рана к тоталу.
Petco Park (Сан-Диего) ≈ -1.0, Oracle Park (Сан-Франциско) ≈ -0.8.

**4. Погода**
Ветер > 15 mph от питчера к хиттеру (to center): +0.5–1.0 рана к тоталу.
Ветер к питчеру (from center): -0.5–1.0 рана. Дождь / холод < 10°C: тотал вниз.

**5. Форма**
Win rate последних 10 игр. H2H встречи в этом сезоне.

**6. Выбор рынка**
- **ML**: кто победит? Лучший при явном преимуществе одного питчера (ERA diff > 0.75).
- **TOTAL** (> / < {total_line} ранов): лучший при доминирующих стартерах ИЛИ выраженной погоде.
- **ITB** (> {itb_line} ранов одной командой): лучший при явно слабом питчере соперника.

Выбери рынок с МАКСИМАЛЬНЫМ edge. Если неопределённость высока — выбирай TOTAL.
НЕ ставь если confidence < 0.55.

Верни СТРОГО JSON без markdown, рассуждения строго на русском:
{{"market": "TOTAL", "pick": "UNDER", "confidence": 0.63, "reasoning": "2-3 предложения: главный аргумент"}}

market: "ML" | "TOTAL" | "ITB"
pick для ML: "HOME" | "AWAY"
pick для TOTAL: "OVER" | "UNDER"
pick для ITB: "HOME_OVER" | "AWAY_OVER"
confidence: 0.50–0.90"""


def _parse_json_strict(raw: str) -> Optional[dict]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
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
    "ITB": {"HOME_OVER", "AWAY_OVER"},
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


async def ai_predict(
    *,
    match_id: int,
    home: str,
    away: str,
    competition: str,
    ml_probs: dict,
    features: dict,
) -> Optional[dict]:
    """Возвращает {market, pick, confidence, reasoning} или None."""
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

    def _fmt(v, fmt=".2f"):
        return format(v, fmt) if v is not None else "н/д"

    prompt = _PROMPT.format(
        home=home,
        away=away,
        total_line=cfg.total_line,
        itb_line=cfg.itb_line,
        p_home=ml_probs.get("p_home", 0.54),
        p_away=ml_probs.get("p_away", 0.46),
        p_over85=ml_probs.get("p_over85", 0.5),
        p_itb_home=ml_probs.get("p_itb_home", 0.5),
        p_itb_away=ml_probs.get("p_itb_away", 0.5),
        home_pitcher_name=features.get("home_pitcher_name") or "неизвестен",
        home_era=_fmt(features.get("home_pitcher_era")),
        home_whip=_fmt(features.get("home_pitcher_whip")),
        away_pitcher_name=features.get("away_pitcher_name") or "неизвестен",
        away_era=_fmt(features.get("away_pitcher_era")),
        away_whip=_fmt(features.get("away_pitcher_whip")),
    )

    raw = await call_llm_with_search(prompt, max_tokens=1200)
    if not raw:
        logger.warning(f"ai_predict({match_id}): пустой ответ")
        return None

    parsed = _parse_json_strict(raw)
    if not parsed or not _validate_pick(parsed):
        logger.warning(f"ai_predict({match_id}): невалидный JSON: {raw[:200]}")
        return None

    _cache[match_id] = (now, parsed)
    await _db_cache_put(match_id, parsed)
    logger.info(
        f"ai_predict({match_id}) {home} vs {away}: "
        f"{parsed['market']} {parsed['pick']} confidence={parsed['confidence']:.2f}"
    )
    return parsed
