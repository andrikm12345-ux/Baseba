from __future__ import annotations

import os

from loguru import logger
from sqlalchemy import select

from src.data.database import SessionLocal, Setting


async def get_str(key: str, default: str = "") -> str:
    try:
        async with SessionLocal() as session:
            row = await session.get(Setting, key)
            return row.value if row else default
    except Exception as e:
        logger.warning(f"settings_store get_str({key}) failed: {e}")
        return default


async def set_str(key: str, value: str) -> None:
    try:
        async with SessionLocal() as session:
            row = await session.get(Setting, key)
            if row is None:
                session.add(Setting(key=key, value=value))
            else:
                row.value = value
            await session.commit()
    except Exception as e:
        logger.warning(f"settings_store set_str({key}) failed: {e}")


_ENV_DEFAULTS = {
    "ai_ensemble_enabled": os.getenv("AI_ENSEMBLE", "").lower() in {"1", "true", "yes", "on"},
}


async def get_bool(key: str, default: bool = False) -> bool:
    # Env var takes precedence as persistent default across restarts
    env_default = _ENV_DEFAULTS.get(key, default)
    val = await get_str(key, "")
    if val == "":
        return env_default
    return val.lower() in {"1", "true", "yes", "on"}


async def set_bool(key: str, value: bool) -> None:
    await set_str(key, "true" if value else "false")
