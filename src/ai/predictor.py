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


_PROMPT = """Ты — Elite Baseball Big Markets Quant Analyst.
Специализация: крупные ликвидные рынки MLB — Moneyline (ML), Run Line (RL, ±1.5), Total Runs (Over/Under {total_line}).
Стиль: исключительно холодный расчёт. Вероятности, Expected Value, edge, Bayesian uncertainty. Никаких эмоций.

## ВХОДНЫЕ ДАННЫЕ

### Prior от XGBoost-модели (твоя отправная точка):
P(победа {home}) = {p_home:.0%} | P(победа {away}) = {p_away:.0%}
P(тотал > {total_line}) = {p_over85:.0%}
P({home} закроет Run Line −1.5) = {p_rl_home:.0%}

### Стартовые питчеры (сезонная статистика):
{home}: {home_pitcher_name} | ERA {home_era} | WHIP {home_whip} | K/9 {home_k9} | BB/9 {home_bb9}
{away}: {away_pitcher_name} | ERA {away_era} | WHIP {away_whip} | K/9 {away_k9} | BB/9 {away_bb9}
ERA diff (хозяева−гости): {era_diff:+.2f} | Статус данных: {pitcher_known}

### Форма команд (скользящие 10 игр + Elo):
Elo: {home} {home_elo:.0f} vs {away} {away_elo:.0f}
Win Rate: {home} {home_win_rate:.0%} vs {away} {away_win_rate:.0%}
Раны: {home} {home_rs_avg:.1f} scored / {home_ra_avg:.1f} allowed
       {away} {away_rs_avg:.1f} scored / {away_ra_avg:.1f} allowed

### Свежие данные (веб-поиск: питчеры, буллпен, травмы, погода):
{web_block}

## АНАЛИЗ (рассуждай как Bayesian Hierarchical + Monte Carlo аналитик)

**1. Park Factor + Weather:**
По web_block определи стадион. Оцени Park Factor (нейтральный ≈1.0, Coors Field ≈1.15, petco park ≈0.85).
Ветер >15 mph к центру поля добавляет +0.5–1.0 рана к тоталу, от поля — вычитает. Температура <10°C снижает тотал на 0.5–1.0.

**2. Стартовые питчеры (главный фактор):**
ERA < 3.50 — элита. ERA > 4.50 — слабый. WHIP < 1.15 — топ контроль.
Если ERA diff > 0.75 в пользу одного питчера — это сильный сигнал.
Проверь свежесть: если питчер не выходил последние 5+ дней — возможна накопленная нагрузка.

**3. Bullpen Fatigue + Tail Risk:**
По web_block оцени усталость буллпенов (3+ игры подряд = повышенный риск).
Tail Risk: вероятность коллапса (≥5 ER за ≤3 IP) — если риск высок, увеличь дисперсию тотала.

**4. Bayesian Update:**
Обнови prior от XGBoost:
- Если ERA diff > 0.75 в пользу хозяев → сдвинь p_home на +3–7%
- Если оба командных тотала высокие (rs_avg > 5.0) + открытый парк + ветер к CF → p_over85 вверх
- Если буллпен соперника усталый → p_rl_home меняется соответственно
- Если данных мало — минимальное отклонение от prior (±2–3%)

**5. EV и рекомендация:**
edge = posterior_prob × book_odds − 1 (≥5% для валуя)
k = 0.25–0.35 стандарт | 0.10–0.18 при высокой неопределённости (погода, усталость) | ≤0.08 пропустить

Верни СТРОГО JSON без markdown:
{{"p_home": float, "p_away": float, "p_over85": float, "p_rl_home": float, "reasoning": "2-3 предложения: главный аргумент (ERA питчеров + буллпен/парк), в чём расхождение с XGBoost-prior, лучший рынок"}}"""


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
    def _fmt_stat(v, fmt=".2f", unknown="н/д"):
        return format(v, fmt) if v is not None else unknown

    home_era = features.get("home_pitcher_era")
    away_era = features.get("away_pitcher_era")
    era_diff = (home_era - away_era) if (home_era is not None and away_era is not None) else 0.0
    pitcher_known_str = "✅ оба известны" if features.get("pitcher_known", 0) == 1.0 else "⚠️ частично/неизвестны"

    prompt = _PROMPT.format(
        home=home,
        away=away,
        total_line=cfg.total_line,
        rl_line=cfg.rl_line,
        p_home=ml_probs.get("p_home", 0.54),
        p_away=ml_probs.get("p_away", 0.46),
        p_over85=ml_probs.get("p_over85", 0.5),
        p_rl_home=ml_probs.get("p_rl_home", 0.4),
        home_pitcher_name=features.get("home_pitcher_name") or "неизвестен",
        home_era=_fmt_stat(home_era),
        home_whip=_fmt_stat(features.get("home_pitcher_whip")),
        home_k9=_fmt_stat(features.get("home_pitcher_k9")),
        home_bb9=_fmt_stat(features.get("home_pitcher_bb9")),
        away_pitcher_name=features.get("away_pitcher_name") or "неизвестен",
        away_era=_fmt_stat(away_era),
        away_whip=_fmt_stat(features.get("away_pitcher_whip")),
        away_k9=_fmt_stat(features.get("away_pitcher_k9")),
        away_bb9=_fmt_stat(features.get("away_pitcher_bb9")),
        era_diff=era_diff,
        pitcher_known=pitcher_known_str,
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
