"""AI ансамбль: выбирает ОДНУ ставку на игру MLB.

Два режима:
- Прямой Anthropic API: Claude ищет данные сам через web_search_20250305
- NeuroAPI-прокси: Tavily выполняет 3 поиска, результаты передаются в промпт
"""
from __future__ import annotations

import asyncio
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


# --- Промпт для прямого Anthropic API (Claude ищет сам) ---
_PROMPT_NATIVE = """Ты — Elite Baseball Quant Analyst. Твоя задача: проанализировать игру MLB и выбрать ОДНУ лучшую ставку.

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

**2. Буллпен** — усталость (3+ игры подряд) резко повышает тотал.

**3. Park Factor**
Coors Field (Колорадо) ≈ +1.5 рана. Petco Park (Сан-Диего) ≈ -1.0. Oracle Park (Сан-Франциско) ≈ -0.8.

**4. Погода**
Ветер > 15 mph к центру поля: +0.5–1.0 рана к тоталу. Дождь / холод < 10°C: тотал вниз.

**5. Форма**  Win rate последних 10 игр. H2H встречи в этом сезоне.

**6. Выбор рынка**
- **ML**: кто победит? Лучший при явном преимуществе одного питчера (ERA diff > 0.75).
- **TOTAL** (> / < {total_line}): лучший при доминирующих стартерах ИЛИ выраженной погоде.
- **ITB** (> {itb_line} ранов одной командой): лучший при явно слабом питчере соперника.
- **F5** (первые 5 иннингов): лучший когда оба стартера сильные (ERA < 3.5) — нет зависимости от буллпена.

Выбери рынок с максимальным edge. Если неопределённость высока — TOTAL. НЕ ставь если confidence < 0.55.

Верни СТРОГО JSON без markdown, рассуждения строго на русском:
{{"market": "TOTAL", "pick": "UNDER", "confidence": 0.63, "reasoning": "2-3 предложения"}}

market: "ML" | "TOTAL" | "ITB" | "F5"
pick для ML: "HOME" | "AWAY"
pick для TOTAL: "OVER" | "UNDER"
pick для ITB: "HOME_OVER" | "AWAY_OVER"
pick для F5: "HOME" | "AWAY"
confidence: 0.50–0.90"""


# --- Промпт для NeuroAPI-прокси (Tavily передаёт данные заранее) ---
_PROMPT_PROXY = """Ты — Elite Baseball Quant Analyst. Твоя задача: проанализировать игру MLB и выбрать ОДНУ лучшую ставку.

## ИГРА
{home} (хозяева) vs {away} (гости) | MLB | Тотал-линия: {total_line} | ИТБ-линия: {itb_line}

## РЕАЛЬНЫЕ КОЭФФИЦИЕНТЫ (Pinnacle / FanDuel / DraftKings, актуальные)
Мани-лайн: {home} {ml_home} | {away} {ml_away}
Тотал {total_line}: Б {odds_over} | М {odds_under}
Ран-лайн (-1.5): {home} {rl_home} | {away} {rl_away}

## СПРАВКА ОТ ML-МОДЕЛИ (вспомогательно)
P(победа {home}) ≈ {p_home:.0%} | P(победа {away}) ≈ {p_away:.0%}
P(тотал > {total_line}) ≈ {p_over85:.0%}
P(ИТБ хоз > {itb_line}) ≈ {p_itb_home:.0%} | P(ИТБ гост > {itb_line}) ≈ {p_itb_away:.0%}
Из БД: {home} — {home_pitcher_name} (ERA {home_era}, WHIP {home_whip}) | {away} — {away_pitcher_name} (ERA {away_era}, WHIP {away_whip})

## СТАТИСТИКА СЕЗОНА

Последние 10 игр: {home} — {home_w10}/{home_g10} (ср. {home_avg_runs} ранов/игру)
Последние 10 игр: {away} — {away_w10}/{away_g10} (ср. {away_avg_runs} ранов/игру)
H2H в этом сезоне ({h2h_games} игр): {home} {h2h_home_wins}–{h2h_away_wins} {away}

## ДАННЫЕ ИЗ ВЕБ-ПОИСКА

### Стартовые питчеры / составы / травмы:
{web_pitchers}

### Погода / стадион:
{web_weather}

## КАК АНАЛИЗИРОВАТЬ

**1. Питчеры** — главный фактор MLB.
ERA diff > 0.75 между стартерами — сильный сигнал. ERA < 3.50 элита, > 4.50 слабый.

**2. Буллпен** — усталость (3+ игры подряд) резко повышает тотал.

**3. Park Factor**
Coors Field (Колорадо) ≈ +1.5 рана. Petco Park (Сан-Диего) ≈ -1.0. Oracle Park (Сан-Франциско) ≈ -0.8.

**4. Погода**
Ветер > 15 mph к центру поля: +0.5–1.0 рана к тоталу. Дождь / холод < 10°C: тотал вниз.

**5. Форма** — Win rate последних 10 игр. H2H встречи в этом сезоне.

**6. Выбор рынка**
- **ML**: кто победит? Лучший при явном преимуществе одного питчера.
- **TOTAL** (> / < {total_line}): лучший при доминирующих стартерах ИЛИ выраженной погоде.
- **ITB** (> {itb_line}): лучший при явно слабом питчере соперника.
- **F5** (первые 5 иннингов): лучший когда оба стартера сильные (ERA < 3.5) — нет зависимости от буллпена.

Выбери рынок с максимальным edge. Если неопределённость высока — TOTAL. НЕ ставь если confidence < 0.55.

Верни СТРОГО JSON без markdown, рассуждения строго на русском:
{{"market": "TOTAL", "pick": "UNDER", "confidence": 0.63, "reasoning": "2-3 предложения"}}

market: "ML" | "TOTAL" | "ITB" | "F5"
pick для ML: "HOME" | "AWAY"
pick для TOTAL: "OVER" | "UNDER"
pick для ITB: "HOME_OVER" | "AWAY_OVER"
pick для F5: "HOME" | "AWAY"
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
    "F5": {"HOME", "AWAY"},
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


async def _build_proxy_prompt(home: str, away: str, cfg, ml_probs: dict, features: dict) -> str:
    """Запускает 2 Tavily-поиска параллельно (питчеры + погода).
    Коэффициенты берём из Odds API (features), не из веба — исключает конфликт данных."""
    from src.data.web_search import tavily_search

    web_pitchers, web_weather = await asyncio.gather(
        tavily_search(f"{home} vs {away} starting pitcher ERA WHIP lineup bullpen injury MLB today", days=3),
        tavily_search(f"MLB {home} {away} weather forecast wind temperature stadium", days=2),
    )

    def _fmt(v, fmt=".2f"):
        try:
            f = float(v)
            return format(f, fmt) if f > 0 else "н/д"
        except (TypeError, ValueError):
            return "н/д"

    return _PROMPT_PROXY.format(
        home=home, away=away,
        total_line=cfg.total_line, itb_line=cfg.itb_line,
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
        ml_home=_fmt(features.get("odds_ml_home")),
        ml_away=_fmt(features.get("odds_ml_away")),
        odds_over=_fmt(features.get("odds_over85")),
        odds_under=_fmt(features.get("odds_under85")),
        rl_home=_fmt(features.get("odds_rl_home")),
        rl_away=_fmt(features.get("odds_rl_away")),
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

    if cfg.llm_base_url:
        # NeuroAPI-прокси: Tavily выполняет поиски, результаты в промпте
        prompt = await _build_proxy_prompt(home, away, cfg, ml_probs, features)
        raw = await call_llm(prompt, max_tokens=900)
        mode = "proxy+tavily"
    else:
        # Прямой Anthropic API: Claude ищет сам через web_search_20250305
        prompt = _PROMPT_NATIVE.format(
            home=home, away=away,
            total_line=cfg.total_line, itb_line=cfg.itb_line,
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
        mode = "anthropic+websearch"

    if not raw:
        logger.warning(f"ai_predict({match_id}): пустой ответ [{mode}]")
        return None

    parsed = _parse_json_strict(raw)
    if not parsed or not _validate_pick(parsed):
        logger.warning(f"ai_predict({match_id}): невалидный JSON [{mode}]: {raw[:200]}")
        return None

    _cache[match_id] = (now, parsed)
    await _db_cache_put(match_id, parsed)
    logger.info(
        f"ai_predict({match_id}) {home} vs {away} [{mode}]: "
        f"{parsed['market']} {parsed['pick']} confidence={parsed['confidence']:.2f}"
    )
    return parsed
