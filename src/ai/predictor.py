"""Claude AI analysis for MLB games.

Claude sees ALL bookmaker quotes (moneyline, every total line, every run-line)
with the market's no-vig probability for each, and picks the ONE quote with the
largest positive divergence (his probability − market no-vig probability).
JSON-only output enforced via system prompt.
"""
from __future__ import annotations

import json
import re
import time
from typing import Dict, Optional

from loguru import logger

from src.ai.commentary import call_llm, call_llm_with_search


_CACHE_TTL_SEC = 4 * 3600  # 4h — matches odds cache TTL, allows re-analysis same day
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
    "Твоя задача — не «найти ставку», а оценить наличие реального перевеса и решить: ставить или пропускать. "
    "В большинстве игр перевеса нет — это нормальный ответ. Если сигнал слабый/неопределённый — всегда пропуск. "
    "Отвечай ТОЛЬКО валидным JSON без markdown, без пояснений вне JSON. "
    "Формат строго: "
    '{"market":"...","pick":"...","line":X.X,"confidence":0.XX,"reasoning":"..."}'
)


_PROMPT = """\
ДАННЫЕ:
МАТЧ: {home} vs {away} | MLB
СТАРТОВЫЕ ПИТЧЕРЫ (ERA/WHIP/K9/BB9):
- {home}: {home_pitcher_name} — ERA {home_era}, WHIP {home_whip}, K/9 {home_k9}, BB/9 {home_bb9}
- {away}: {away_pitcher_name} — ERA {away_era}, WHIP {away_whip}, K/9 {away_k9}, BB/9 {away_bb9}
ФОРМА (последние игры в БД):
- {home}: {home_w10}/{home_g10} побед, {home_avg_runs} ранов/игру
- {away}: {away_w10}/{away_g10} побед, {away_avg_runs} ранов/игру
- H2H сезона ({h2h_games} игр): {home} {h2h_home_wins}–{h2h_away_wins} {away}
{web_block}КОТИРОВКИ (no-vig вероятности рынка):
{quotes_table}

ЗАДАЧА — оценить наличие реального перевеса, затем решить: сигнал или пропуск.

Шаг 1. Проверь наличие хотя бы одного фактора:
  A. ERA-разница стартеров |ERA_home − ERA_away| ≥ 1.0 — есть реальный перевес.
  B. Один питчер ERA < 3.50 и WHIP < 1.20, другой ERA > 4.50 — элита vs слабый.
  C. Команда выиграла ≥ 7 из последних 10 при кэфе ≥ 1.95 (рынок недооценил форму).
  D. Coors Field и ERA ≥ 4.00 у хотя бы одного — TOTAL OVER.
  E. Данные питчеров н/д, но команда выиграла ≥ 8/10 при кэфе ≥ 2.00 — форма говорит сама.

Шаг 2. Если хотя бы один фактор есть:
  - Оцени P_exp (твоя вероятность выигрыша) исходя из факторов.
  - Возьми P_mkt (no-vig) для нужного рынка/стороны из таблицы котировок.
  - Условие сигнала: P_exp ≥ 0.63 И (P_exp − P_mkt) ≥ 0.06.
  - При ERA-разнице от 2.0+ или критерии B — приоритет ML или RUN_LINE.
  - При факторе C/E — приоритет ML (выбрать ту сторону, у которой форма).
  - При факторе D — TOTAL OVER.

Шаг 3. Если ни одного фактора нет → confidence 0.50 (пропуск).
  Что НЕ является фактором: ERA-разница < 1.0; «команда дома»; интуиция без цифр.

ПРИОРИТЕТ РЫНКОВ при равной уверенности: ML > RUN_LINE > TOTAL.

CONFIDENCE:
- P_exp ≥ 0.72 → confidence 0.75–0.90
- 0.63 ≤ P_exp < 0.72 → confidence 0.63–0.74
- Нет сигнала → confidence 0.50

ФОРМАТ ОТВЕТА (ТОЛЬКО валидный JSON без markdown):
- ML: {{"market":"ML","pick":"HOME" или "AWAY","line":null,"confidence":0.XX,"reasoning":"..."}}
- Тотал: {{"market":"TOTAL","pick":"OVER" или "UNDER","line":X.X,"confidence":0.XX,"reasoning":"..."}}
- Фора: {{"market":"RUN_LINE","pick":"HOME" или "AWAY","line":Y.Y,"confidence":0.XX,"reasoning":"..."}}
reasoning = 1 строка: фактор(ы) + числа + P_exp vs P_mkt.

Пример сигнала: {{"market":"ML","pick":"HOME","line":null,"confidence":0.68,"reasoning":"ERA хозяев 2.8 vs гостей 4.9, разница 2.1 (факт A+B), P_exp=0.68 vs рынок 0.55, расхождение +13%"}}
Пример пропуска: {{"market":"ML","pick":"HOME","line":null,"confidence":0.50,"reasoning":"ERA разница 0.3, нет факторов A–E"}}\
"""


def _fmt(v, fmt=".2f") -> str:
    try:
        f = float(v)
        return format(f, fmt) if f > 0 else "н/д"
    except (TypeError, ValueError):
        return "н/д"


def _pct(p) -> str:
    try:
        return f"{float(p)*100:.0f}%"
    except (TypeError, ValueError):
        return "—"


def _build_quotes_table(home: str, away: str, features: dict) -> str:
    """Render every available quote (ML, all totals, all run lines) with no-vig %."""
    from src.data.odds_api import novig_two_way

    lines: list[str] = []

    # Moneyline
    ml_h = features.get("odds_ml_home", 0.0)
    ml_a = features.get("odds_ml_away", 0.0)
    nv = novig_two_way(ml_h, ml_a)
    if nv:
        lines.append("МАНИ-ЛАЙН (ML):")
        lines.append(f"  {home}: кэф {_fmt(ml_h)} | рынок {_pct(nv[0])}")
        lines.append(f"  {away}: кэф {_fmt(ml_a)} | рынок {_pct(nv[1])}")

    # Totals
    totals = features.get("totals_lines") or []
    if totals:
        lines.append("ТОТАЛ (по линиям):")
        for t in totals:
            lines.append(
                f"  Линия {t['point']}: "
                f"Б кэф {_fmt(t['over'])} (рынок {_pct(t.get('over_novig'))}) | "
                f"М кэф {_fmt(t['under'])} (рынок {_pct(t.get('under_novig'))})"
            )

    # Run lines (spreads) — only show sides with valid odds to avoid confusing the model
    spreads = features.get("spread_lines") or []
    if spreads:
        lines.append("ФОРА / РАН-ЛАЙН (−линия = тот кто даёт фору; только доступные ставки):")
        for s in spreads:
            parts = []
            if s.get("home", 0) > 1.0:
                parts.append(
                    f"{home} −{s['point']} кэф {_fmt(s['home'])} (рынок {_pct(s.get('home_novig'))}) → пик COVER"
                )
            if s.get("away", 0) > 1.0:
                parts.append(
                    f"{away} −{s['point']} кэф {_fmt(s['away'])} (рынок {_pct(s.get('away_novig'))}) → пик AWAY_COVER"
                )
            if parts:
                lines.append(f"  Линия {s['point']}: " + " | ".join(parts))

    return "\n".join(lines) if lines else "(котировок нет)"


async def _build_web_block(home: str, away: str) -> str:
    """Tavily searches for pitchers/weather (proxy mode only)."""
    from src.data.web_search import tavily_search
    import asyncio

    web_pitchers, web_weather = await asyncio.gather(
        tavily_search(f"{home} vs {away} starting pitcher ERA WHIP lineup injury MLB today", days=3),
        tavily_search(f"MLB {home} {away} weather wind temperature stadium", days=2),
    )

    def _fmt_web(results: list, max_items: int = 4) -> str:
        if not results:
            return "(данных нет)"
        out = []
        for r in results[:max_items]:
            title = r.get("title", "").strip()
            content = r.get("content", "").strip()
            out.append(f"• {title}: {content[:300]}")
        return "\n".join(out)

    return (
        "\n## ВЕБ-ПОИСК\n"
        "### Питчеры / состав / травмы:\n"
        f"{_fmt_web(web_pitchers)}\n\n"
        "### Погода / стадион:\n"
        f"{_fmt_web(web_weather)}\n"
    )


def _parse_json_strict(raw: str) -> Optional[dict]:
    if not raw:
        return None
    raw = raw.strip()
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
    "RL": {"COVER", "AWAY_COVER"},
}


def _normalize_market(d: dict) -> None:
    """Normalize Claude's output format to internal format.

    Claude prompt uses RUN_LINE/HOME/AWAY for run line;
    internal code uses RL/COVER/AWAY_COVER.
    """
    if d.get("market") == "RUN_LINE":
        d["market"] = "RL"
        if d.get("pick") == "HOME":
            d["pick"] = "COVER"
        elif d.get("pick") == "AWAY":
            d["pick"] = "AWAY_COVER"


def _validate_pick(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    _normalize_market(d)
    market = d.get("market")
    pick = d.get("pick")
    confidence = d.get("confidence")
    if market not in _VALID_PICKS or pick not in _VALID_PICKS[market]:
        return False
    if not isinstance(confidence, (int, float)):
        return False
    # line: required (number) for TOTAL/RL, ignored for ML
    if market in ("TOTAL", "RL"):
        line = d.get("line")
        if not isinstance(line, (int, float)):
            return False
        d["line"] = float(line)
    else:
        d["line"] = None
    d["confidence"] = max(0.50, min(1.0, float(confidence)))
    return True


async def _build_prompt(home: str, away: str, features: dict, web: bool) -> str:
    web_block = await _build_web_block(home, away) if web else "\n"
    return _PROMPT.format(
        home=home, away=away,
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
        web_block=web_block,
        quotes_table=_build_quotes_table(home, away, features),
    )


async def ai_predict(
    *,
    match_id: int,
    home: str,
    away: str,
    competition: str,
    features: dict,
) -> Optional[dict]:
    """Returns {market, pick, line, confidence, reasoning} or None."""
    from src.config import settings as cfg

    now = time.time()
    cached = _cache.get(match_id)
    if cached and now - cached[0] < _CACHE_TTL_SEC and _validate_pick(cached[1]):
        return cached[1]

    db_cached = await _db_cache_get(match_id)
    if db_cached is not None and _validate_pick(db_cached):
        _cache[match_id] = (now, db_cached)
        return db_cached

    if cfg.llm_base_url:
        prompt = await _build_prompt(home, away, features, web=True)
        raw = await call_llm(prompt, system=SYSTEM_PROMPT, max_tokens=700)
        mode = "proxy+tavily"
    else:
        prompt = await _build_prompt(home, away, features, web=False)
        raw = await call_llm_with_search(prompt, system=SYSTEM_PROMPT, max_tokens=900)
        mode = "anthropic+websearch"

    if not raw:
        logger.warning(f"ai_predict({match_id}): empty response [{mode}]")
        return None

    parsed = _parse_json_strict(raw)
    if not parsed or not _validate_pick(parsed):
        logger.warning(f"ai_predict({match_id}): invalid JSON [{mode}]: {raw[:300]}")
        return None

    if parsed["confidence"] < 0.56:
        logger.info(f"ai_predict({match_id}): confidence {parsed['confidence']:.2f} < 0.56 — skip")
        return None

    _cache[match_id] = (now, parsed)
    await _db_cache_put(match_id, parsed)
    logger.info(
        f"ai_predict({match_id}) {home} vs {away} [{mode}]: "
        f"{parsed['market']} {parsed['pick']} line={parsed.get('line')} "
        f"conf={parsed['confidence']:.2f}"
    )
    return parsed
