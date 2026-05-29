"""LLM call wrappers (Anthropic direct + OpenAI-compatible proxy)."""
from __future__ import annotations

from typing import Optional

from loguru import logger

from src.config import settings


async def call_llm(prompt: str, system: str = "", max_tokens: int = 900) -> Optional[str]:
    """Call LLM (Anthropic or OpenAI-compatible proxy)."""
    try:
        if settings.llm_base_url:
            return await _call_openai_compat(prompt, system, max_tokens)
        elif settings.anthropic_api_key:
            return await _call_anthropic(prompt, system, max_tokens)
        else:
            logger.warning("No LLM API key configured")
            return None
    except Exception as e:
        logger.warning(f"call_llm failed: {e}")
        return None


async def call_llm_with_search(prompt: str, system: str = "", max_tokens: int = 1200) -> Optional[str]:
    """Call Anthropic Claude with built-in web_search_20250305 tool."""
    if not settings.anthropic_api_key:
        return await call_llm(prompt, system, max_tokens)
    try:
        return await _call_anthropic_with_search(prompt, system, max_tokens)
    except Exception as e:
        logger.warning(f"call_llm_with_search failed ({e}), falling back to call_llm")
        return await call_llm(prompt, system, max_tokens)


async def _call_anthropic_with_search(prompt: str, system: str, max_tokens: int) -> Optional[str]:
    """Tool-use loop with Anthropic's server-side web_search_20250305 tool."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    messages: list = [{"role": "user", "content": prompt}]
    tools = [{"type": "web_search_20250305", "name": "web_search"}]
    beta_headers = {"anthropic-beta": "web-search-2025-03-05"}
    kwargs = dict(
        model=settings.llm_model,
        max_tokens=max_tokens,
        tools=tools,
        messages=messages,
        extra_headers=beta_headers,
    )
    if system:
        kwargs["system"] = system

    for _ in range(10):
        response = await client.messages.create(**kwargs)

        text_blocks = [
            b.text
            for b in response.content
            if hasattr(b, "type") and b.type == "text" and hasattr(b, "text")
        ]

        if response.stop_reason != "tool_use":
            return "\n".join(text_blocks) or None

        # Append assistant turn and continue loop
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if not (hasattr(block, "type") and block.type == "tool_use"):
                continue
            # For server-side web_search the API may embed results in block.content
            raw = getattr(block, "content", "") or ""
            content_str = raw if isinstance(raw, str) else str(raw)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content_str,
            })

        if not tool_results:
            return "\n".join(text_blocks) or None
        messages.append({"role": "user", "content": tool_results})

    return None


async def _call_anthropic(prompt: str, system: str, max_tokens: int) -> Optional[str]:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    kwargs = dict(
        model=settings.llm_model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system
    msg = await client.messages.create(**kwargs)
    return msg.content[0].text if msg.content else None


async def _call_openai_compat(prompt: str, system: str, max_tokens: int) -> Optional[str]:
    import aiohttp
    headers = {"Authorization": f"Bearer {settings.llm_api_key or settings.anthropic_api_key}"}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as r:
            r.raise_for_status()
            data = await r.json()
            return data["choices"][0]["message"]["content"]
