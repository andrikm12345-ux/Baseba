"""LLM commentary generation for baseball signals."""
from __future__ import annotations

import asyncio
from typing import Optional

from loguru import logger

from src.config import settings

_cache: dict[tuple, str] = {}
_lock = asyncio.Lock()


async def call_llm(prompt: str, max_tokens: int = 900) -> Optional[str]:
    """Call LLM (Anthropic or OpenAI-compatible proxy)."""
    try:
        if settings.llm_base_url:
            return await _call_openai_compat(prompt, max_tokens)
        elif settings.anthropic_api_key:
            return await _call_anthropic(prompt, max_tokens)
        else:
            logger.warning("No LLM API key configured")
            return None
    except Exception as e:
        logger.warning(f"call_llm failed: {e}")
        return None


async def _call_anthropic(prompt: str, max_tokens: int) -> Optional[str]:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    msg = await client.messages.create(
        model=settings.llm_model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text if msg.content else None


async def _call_openai_compat(prompt: str, max_tokens: int) -> Optional[str]:
    import aiohttp
    headers = {"Authorization": f"Bearer {settings.llm_api_key or settings.anthropic_api_key}"}
    payload = {
        "model": settings.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as r:
            r.raise_for_status()
            data = await r.json()
            return data["choices"][0]["message"]["content"]


async def generate_commentary(
    match_id: int,
    home: str,
    away: str,
    market: str,
    pick: str,
    prob: float,
) -> Optional[str]:
    """Generate a short 2-sentence commentary for a signal."""
    key = (match_id, market, pick)
    async with _lock:
        if key in _cache:
            return _cache[key]

    pick_label = {
        "HOME": f"{home} победят",
        "AWAY": f"{away} победят",
        "OVER": f"тотал больше {settings.total_line}",
        "UNDER": f"тотал меньше {settings.total_line}",
        "COVER": f"{home} выиграют с форой -{settings.rl_line}",
        "LAY": f"{away} выиграют с форой +{settings.rl_line}",
    }.get(pick, pick)

    prompt = (
        f"Ты бейсбольный аналитик. Игра MLB: {home} vs {away}. "
        f"Прогноз: {pick_label} (уверенность {prob:.0%}). "
        f"Напиши 2 предложения с ключевым аргументом в пользу этого прогноза. "
        f"Только факты о форме, питчерах или статистике. Без воды."
    )
    result = await call_llm(prompt, max_tokens=150)
    if result:
        async with _lock:
            _cache[key] = result
    return result
