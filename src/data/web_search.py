from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from src.config import settings


async def tavily_search(query: str, days: int = 7) -> List[Dict[str, Any]]:
    """Search the web via Tavily API. Returns list of result dicts."""
    if not settings.tavily_api_key:
        return []
    try:
        import aiohttp
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": settings.tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": 5,
            "days": days,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status != 200:
                    logger.warning(f"Tavily {r.status} for query: {query[:60]}")
                    return []
                data = await r.json()
                return data.get("results", [])
    except Exception as e:
        logger.warning(f"tavily_search failed: {e}")
        return []
