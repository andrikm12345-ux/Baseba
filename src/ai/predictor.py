"""AI ансамбль: Claude независимо анализирует игру MLB через веб-поиск
и выбирает один рынок + направление без зависимости от XGBoost.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Dict, Optional

from loguru import logger

from src.ai.commentary import call_llm
from src.data.web_search import tavily_search


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
{home} (хозяева) vs {away} (гости)
Лига: MLB | Лайн тотала: {total_line} ранов | ИТБ линия: {itb_line} ранов

## СПРАВКА ОТ МОДЕЛИ (второстепенная информация)
P(победа {home}) ≈ {p_home:.0%} | P(победа {away}) ≈ {p_away:.0%}
P(тотал > {total_line}) ≈ {p_over85:.0%}
P(хозяева > {itb_line}) ≈ {p_itb_home:.0%} | P(гости > {itb_line}) ≈ {p_itb_away:.0%}
Питчеры из БД: {home} — {home_pitcher_name} (ERA {home_era}, WHIP {home_whip}) | {away} — {away_pitcher_name} (ERA {away_era}, WHIP {away_whip})

## ДАННЫЕ ИЗ ВЕБ-ПОИСКА

### Стартовые питчеры / составы / травмы:
{web_pitchers}

### Коэффициенты / линии / прогнозы экспертов:
{web_odds}

### Погода / стадион:
{web_weather}

## ИНСТРУКЦИЯ ПО АНАЛИЗУ

Проведи полный независимый анализ. Данные из веб-поиска приоритетнее данных модели.

**1. Стартовые питчеры** — главный фактор в MLB.
Найди актуальные ERA, WHIP, K/9, последние выходы. ERA < 3.50 — элита, > 4.50 — слабый. Если ERA diff > 0.75 — сильный сигнал на победителя и низкий тотал. Проверь количество дней отдыха.

**2. Буллпен** — второй по важности фактор.
Усталость буллпена (3+ игры подряд ключевых питчеров) резко повышает тотал. Посмотри статистику Relief ERA команд.

**3. Коэффициенты и линии** — что говорит рынок?
Если видишь коэффициенты в веб-данных — используй их. Сравни с вероятностями модели. Большое расхождение = потенциальный edge.

**4. Park Factor + Weather**
Стадион: Coors Field ≈ +1.5 рана к тоталу, Petco Park ≈ -1.0, Oracle Park ≈ -0.8.
Ветер > 15 mph к центру поля: +0.5–1.0 рана. Температура < 10°C: -0.5–1.0 рана.
Дождь / мокрая погода: снижает пауэр-хиттинг, тотал вниз.

**5. Форма и H2H**
Серия результатов последних 7–10 игр. Очные встречи в этом сезоне.

**6. Выбор рынка**
Оцени три рынка:
- **ML** (Мани-лайн): кто победит? Лучший выбор при явном преимуществе одного питчера.
- **TOTAL** (тотал > / < {total_line}): сумма ранов обеих команд. Лучший выбор при ярко выраженных погодных условиях или доминирующих стартовых питчерах.
- **ITB** (индивидуальный тотал > {itb_line}): одна команда наберёт больше {itb_line} ранов? Лучший выбор при явно слабом питчере противника.

Выбери рынок с МАКСИМАЛЬНЫМ edge и МИНИМАЛЬНОЙ неопределённостью. Если данных не хватает — выбирай TOTAL как наиболее предсказуемый. Не ставь если confidence < 0.55.

Верни СТРОГО JSON без markdown, рассуждения строго на русском:
{{"market": "TOTAL", "pick": "UNDER", "confidence": 0.63, "reasoning": "2-3 предложения: главный аргумент + почему этот рынок"}}

Допустимые значения market: "ML" | "TOTAL" | "ITB"
pick для ML: "HOME" | "AWAY"
pick для TOTAL: "OVER" | "UNDER"
pick для ITB: "HOME_OVER" | "AWAY_OVER"
confidence: 0.50–0.90"""


def _format_web(results: list[dict], max_items: int = 4) -> str:
    if not results:
        return "(данных нет)"
    lines = []
    for r in results[:max_items]:
        title = r.get("title", "").strip()
        content = r.get("content", "").strip()
        lines.append(f"• {title}: {content[:300]}")
    return "\n".join(lines)


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

    # Три параллельных веб-поиска: питчеры, коэффициенты, погода
    web_pitchers_task = tavily_search(
        f"{home} vs {away} starting pitcher ERA WHIP lineup bullpen injury MLB today", days=3
    )
    web_odds_task = tavily_search(
        f"{home} vs {away} MLB odds betting lines prediction picks", days=2
    )
    web_weather_task = tavily_search(
        f"MLB {home} {away} weather forecast wind temperature stadium", days=2
    )
    web_pitchers, web_odds, web_weather = await asyncio.gather(
        web_pitchers_task, web_odds_task, web_weather_task
    )

    def _fmt_stat(v, fmt=".2f", unknown="н/д"):
        return format(v, fmt) if v is not None else unknown

    home_era = features.get("home_pitcher_era")
    away_era = features.get("away_pitcher_era")

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
        home_era=_fmt_stat(home_era),
        home_whip=_fmt_stat(features.get("home_pitcher_whip")),
        away_pitcher_name=features.get("away_pitcher_name") or "неизвестен",
        away_era=_fmt_stat(away_era),
        away_whip=_fmt_stat(features.get("away_pitcher_whip")),
        web_pitchers=_format_web(web_pitchers),
        web_odds=_format_web(web_odds),
        web_weather=_format_web(web_weather),
    )

    raw = await call_llm(prompt, max_tokens=700)
    if not raw:
        logger.warning(f"ai_predict({match_id}): пустой ответ LLM")
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
