"""AI ensemble: Claude analyses an MLB game using web data and returns
independent probabilities blended with XGBoost predictions.
"""
from __future__ import annotations

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


_PROMPT = """Ты — элитный бейсбольный аналитик MLB с 15+ годами опыта в анализе матчей и ставках. Глубокое понимание sabermetrics (ERA, WHIP, FIP, wRC+, OPS+), математики линий и рынков.

ЦЕЛЬ: найти РАСХОЖДЕНИЕ между истинной вероятностью и рыночной линией в игре {home} vs {away}.

АЛГОРИТМ (7 ступеней):
1. СТАРТОВЫЕ ПИТЧЕРЫ: ERA, WHIP, K/9, BB/9, FIP за сезон и последние 5 стартов. Против текущего оппонента H2H.
2. БУЛЛПЕН: ERA/WHIP последних 7 дней. Количество дней отдыха ключевых бросающих. Стрейк использования.
3. АТАКА: OPS, wRC+, K%, BB% обеих линий. Home/Away splits (для хозяев vs гостей).
4. ФАКТОР ПАРКА И ПОГОДА: Park Factor стадиона (нейтральный=100). Ветер (скорость, направление) влияет на тотал ±1-2 рана. Температура.
5. H2H И ФОРМА: последние 2 сезона H2H. Текущий стрейк (W/L серия). Wins Last 10 games.
6. РЫНОЧНЫЙ АНАЛИЗ: убери маржу букмекера. Сравни с XGBoost. Найди расхождение ≥5%.
7. ИТОГ: определи лучший рынок (ML/TOTAL/RL) с наибольшим edge.

ДАННЫЕ ОТ ML-МОДЕЛИ (XGBoost, Elo-based):
P(победа хозяев) = {p_home:.0%}, P(победа гостей) = {p_away:.0%}
P(тотал > {total_line}) = {p_over85:.0%}
P(хозяева закроют ран-лайн -{rl_line}) = {p_rl_home:.0%}

ИСТОРИЧЕСКИЙ КОНТЕКСТ:
Elo {home} = {home_elo:.0f}, Elo {away} = {away_elo:.0f}
Win Rate (10 игр): {home} {home_win_rate:.0%}, {away} {away_win_rate:.0%}
Средние раны (атака/питчинг): {home} {home_rs_avg:.1f}/{home_ra_avg:.1f}, {away} {away_rs_avg:.1f}/{away_ra_avg:.1f}

СВЕЖИЕ ДАННЫЕ ИЗ СЕТИ:
{web_block}

ЗАДАЧА: пройди все 7 ступеней. Сформируй СВОИ независимые вероятности. Если данных нет — используй базовые распределения MLB (P(home win)≈0.54).

Верни СТРОГО JSON без markdown:
{{"p_home": float, "p_away": float, "p_over85": float, "p_rl_home": float, "reasoning": "1-2 предложения с главным аргументом и расхождением"}}"""


def _format_web(results: list[dict]) -> str:
    if not results:
        return "(нет свежих данных)"
    lines = []
    for r in results[:5]:
        title = r.get("title", "").strip()
        content = r.get("content", "").strip()
        lines.append(f"• {title}: {content[:250]}")
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


def _validate_probs(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    for key in ("p_home", "p_away", "p_over85", "p_rl_home"):
        v = d.get(key)
        if not isinstance(v, (int, float)):
            return False
        if v < 0 or v > 1.0:
            return False
    s = d["p_home"] + d["p_away"]
    if s <= 0:
        return False
    d["p_home"] = d["p_home"] / s
    d["p_away"] = d["p_away"] / s
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
    """Returns {p_home, p_away, p_over85, p_rl_home, reasoning} or None."""
    from src.config import settings as cfg
    now = time.time()
    cached = _cache.get(match_id)
    if cached and now - cached[0] < _CACHE_TTL_SEC:
        return cached[1]

    db_cached = await _db_cache_get(match_id)
    if db_cached is not None:
        _cache[match_id] = (now, db_cached)
        return db_cached

    web_results = await tavily_search(
        f"{home} vs {away} pitcher lineup injury MLB", days=5
    )
    prompt = _PROMPT.format(
        home=home,
        away=away,
        total_line=cfg.total_line,
        rl_line=cfg.rl_line,
        p_home=ml_probs.get("p_home", 0.54),
        p_away=ml_probs.get("p_away", 0.46),
        p_over85=ml_probs.get("p_over85", 0.5),
        p_rl_home=ml_probs.get("p_rl_home", 0.4),
        home_elo=features.get("home_elo", 1500),
        away_elo=features.get("away_elo", 1500),
        home_win_rate=features.get("home_win_rate", 0.5),
        away_win_rate=features.get("away_win_rate", 0.5),
        home_rs_avg=features.get("home_rs_avg", 4.3),
        home_ra_avg=features.get("home_ra_avg", 4.3),
        away_rs_avg=features.get("away_rs_avg", 4.3),
        away_ra_avg=features.get("away_ra_avg", 4.3),
        web_block=_format_web(web_results),
    )

    raw = await call_llm(prompt, max_tokens=600)
    if not raw:
        logger.warning(f"ai_predict({match_id}): empty LLM response")
        return None

    parsed = _parse_json_strict(raw)
    if not parsed or not _validate_probs(parsed):
        logger.warning(f"ai_predict({match_id}): invalid JSON: {raw[:200]}")
        return None

    _cache[match_id] = (now, parsed)
    await _db_cache_put(match_id, parsed)
    logger.info(
        f"ai_predict({match_id}): home={parsed['p_home']:.2f} "
        f"away={parsed['p_away']:.2f} "
        f"over85={parsed['p_over85']:.2f} rl={parsed['p_rl_home']:.2f}"
    )
    return parsed
