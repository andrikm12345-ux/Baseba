"""AI ensemble: Claude analyses an MLB game and independently picks the best market + direction."""
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
Специализация: крупные ликвидные рынки MLB — Moneyline (ML), Total Runs (Over/Under {total_line}), Individual Team Total Over (ITB {itb_line}).
Стиль: исключительно холодный расчёт. Байесовская вероятность, Expected Value, edge. Никаких эмоций.
Рассуждения СТРОГО на русском языке.

## ВХОДНЫЕ ДАННЫЕ

### Справочные вероятности XGBoost-модели:
P(победа {home}) = {p_home:.0%} | P(победа {away}) = {p_away:.0%}
P(тотал > {total_line}) = {p_over85:.0%}
P(хозяева > {itb_line} ранов) = {p_itb_home:.0%} | P(гости > {itb_line} ранов) = {p_itb_away:.0%}

### Стартовые питчеры:
{home}: {home_pitcher_name} | ERA {home_era} | WHIP {home_whip} | K/9 {home_k9} | BB/9 {home_bb9}
{away}: {away_pitcher_name} | ERA {away_era} | WHIP {away_whip} | K/9 {away_k9} | BB/9 {away_bb9}
ERA diff (хозяева−гости): {era_diff:+.2f} | Статус: {pitcher_known}

### Форма команд:
Elo: {home} {home_elo:.0f} vs {away} {away_elo:.0f}
Win Rate: {home} {home_win_rate:.0%} vs {away} {away_win_rate:.0%}
Раны: {home} {home_rs_avg:.1f} scored / {home_ra_avg:.1f} allowed
       {away} {away_rs_avg:.1f} scored / {away_ra_avg:.1f} allowed

### Свежие данные (веб-поиск: питчеры, буллпен, травмы, погода):
{web_block}

## АНАЛИЗ

1. **Park Factor + Weather:** По web_block определи стадион. Park Factor (нейтральный ≈1.0, Coors Field ≈1.15, Petco Park ≈0.85). Ветер >15 mph к центру поля +0.5–1.0 рана к тоталу, от поля — вычитает. Температура <10°C снижает тотал.

2. **Стартовые питчеры (ключевой фактор):** ERA < 3.50 — элита. ERA > 4.50 — слабый. WHIP < 1.15 — топ контроль. ERA diff > 0.75 в пользу одной команды — сильный сигнал. Проверь свежесть по web_block.

3. **Буллпен + Tail Risk:** Усталость буллпена (3+ игры подряд). Риск коллапса (≥5 ER за ≤3 IP) — если высок, увеличь дисперсию тотала. По web_block проверь травмы ключевых игроков.

4. **H2H + Home Advantage:** Очные встречи. Домашнее поле: стандарт +3–5% к вероятности.

5. **Выбор рынка:** Оцени все три рынка (ML, TOTAL, ITB). Выбери ОДИН рынок и направление с максимальным edge и наименьшей неопределённостью. Предпочитай рынок, где расхождение с моделью наиболее обосновано реальными данными (питчер, погода, буллпен). Если несколько рынков равнозначны — выбирай TOTAL как наиболее предсказуемый. Пороговый confidence для выбора: ≥ 0.55.

Допустимые значения:
- market: "ML" | "TOTAL" | "ITB"
- pick для ML: "HOME" или "AWAY"
- pick для TOTAL: "OVER" или "UNDER"
- pick для ITB: "HOME_OVER" или "AWAY_OVER"
- confidence: 0.50–0.90 (твоя итоговая вероятность этого исхода)

Верни СТРОГО JSON без markdown:
{{"market": "TOTAL", "pick": "UNDER", "confidence": 0.63, "reasoning": "2-3 предложения на русском: главный аргумент (ERA/буллпен/парк/погода), почему именно этот рынок"}}"""


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
    if confidence < 0.50 or confidence > 1.0:
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

    web_results = await tavily_search(
        f"{home} vs {away} pitcher lineup injury bullpen MLB", days=5
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
        itb_line=cfg.itb_line,
        p_home=ml_probs.get("p_home", 0.54),
        p_away=ml_probs.get("p_away", 0.46),
        p_over85=ml_probs.get("p_over85", 0.5),
        p_itb_home=ml_probs.get("p_itb_home", 0.5),
        p_itb_away=ml_probs.get("p_itb_away", 0.5),
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
    if not parsed or not _validate_pick(parsed):
        logger.warning(f"ai_predict({match_id}): invalid JSON or pick: {raw[:200]}")
        return None

    _cache[match_id] = (now, parsed)
    await _db_cache_put(match_id, parsed)
    logger.info(
        f"ai_predict({match_id}): {parsed['market']} {parsed['pick']} "
        f"confidence={parsed['confidence']:.2f}"
    )
    return parsed
