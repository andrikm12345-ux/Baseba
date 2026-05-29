"""Entry point for the MLB Baseball Signals Telegram bot."""
from __future__ import annotations

import asyncio
import os

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import (
    BotCommand,
    BotCommandScopeChat,
    BotCommandScopeDefault,
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

from src.bot.access import AccessMiddleware
from src.bot.handlers import broadcast_morning_digest, broadcast_results_summary, router
from src.config import settings
from src.data.database import init_db
from src.pipeline import daily_cycle, generate_and_broadcast, refresh_upcoming
from src.signals.tracker import settle_pending


_USER_COMMANDS = [
    BotCommand(command="signals", description="⚾ Сигналы"),
    BotCommand(command="today", description="📅 Игры сегодня"),
    BotCommand(command="stats", description="📊 Статистика"),
    BotCommand(command="chart", description="📈 График ROI"),
    BotCommand(command="history", description="📜 История ставок"),
    BotCommand(command="notifications", description="🔔 Уведомления"),
    BotCommand(command="help", description="ℹ️ Помощь"),
    BotCommand(command="start", description="🔄 Перезапустить меню"),
]

_ADMIN_COMMANDS = _USER_COMMANDS + [
    BotCommand(command="admin", description="🛠 Админ-панель"),
    BotCommand(command="check_signals", description="🔍 Диагностика сигналов"),
    BotCommand(command="test_odds", description="📡 Тест Odds API"),
    BotCommand(command="digest_now", description="📤 Разослать дайджест"),
    BotCommand(command="purge_signals", description="🗑 Очистить сигналы"),
    BotCommand(command="debugodds", description="🐞 Отладка кэфов"),
    BotCommand(command="allow", description="✅ Выдать доступ по ID"),
    BotCommand(command="deny", description="⛔ Забрать доступ по ID"),
]


async def _setup_commands(bot: Bot) -> None:
    """Register the slash-command list shown when the user types '/'."""
    try:
        await bot.set_my_commands(_USER_COMMANDS, scope=BotCommandScopeDefault())
        for admin_id in settings.admin_ids:
            try:
                await bot.set_my_commands(
                    _ADMIN_COMMANDS, scope=BotCommandScopeChat(chat_id=admin_id)
                )
            except Exception as e:
                logger.warning(f"set admin commands for {admin_id} failed: {e}")
        logger.info("Bot command list registered")
    except Exception as e:
        logger.warning(f"set_my_commands failed: {e}")


async def _warmup(bot: Bot) -> None:
    try:
        await refresh_upcoming(days=7)
        await settle_pending()
        await generate_and_broadcast(bot)
    except Exception as e:
        logger.error(f"Warmup failed (bot will still work): {e}")


async def _notify_startup(bot: Bot) -> None:
    if not settings.admin_ids:
        return
    from datetime import datetime, timezone, timedelta
    commit = os.environ.get("RAILWAY_GIT_COMMIT_SHA", "")[:7] or "dev"
    branch = os.environ.get("RAILWAY_GIT_BRANCH", "")
    now_msk = datetime.now(timezone(timedelta(hours=3))).strftime("%d.%m %H:%M МСК")
    version_line = f"<code>{commit}</code>"
    if branch:
        version_line += f" ({branch})"
    text = (
        f"🚀 <b>Бот запущен</b>\n"
        f"🕐 {now_msk}\n"
        f"📦 Версия: {version_line}"
    )
    for admin_id in settings.admin_ids:
        try:
            await bot.send_message(admin_id, text, parse_mode="HTML")
        except Exception:
            pass


async def main() -> None:
    if not settings.telegram_bot_token:
        logger.critical("TELEGRAM_BOT_TOKEN not set!")
        raise SystemExit(1)

    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    await bot.delete_webhook(drop_pending_updates=True)

    await init_db()
    logger.info("DB initialized")

    # Persist AI_ENSEMBLE env var to DB on first start
    from src.data.settings_store import get_str, set_bool
    existing = await get_str("ai_ensemble_enabled", "")
    if existing == "" and os.environ.get("AI_ENSEMBLE", "").lower() in {"1", "true", "yes", "on"}:
        await set_bool("ai_ensemble_enabled", True)
        logger.info("AI_ENSEMBLE env var persisted to DB")

    dp = Dispatcher(storage=MemoryStorage())
    dp.message.middleware(AccessMiddleware())
    dp.callback_query.middleware(AccessMiddleware())
    dp.include_router(router)

    scheduler = AsyncIOScheduler(timezone=settings.tz)
    scheduler.add_job(daily_cycle, "cron", hour=4, minute=0, args=[bot], id="daily")
    scheduler.add_job(generate_and_broadcast, "interval", minutes=15, args=[bot], id="signals_loop")
    scheduler.add_job(settle_pending, "interval", minutes=20, id="settle")
    scheduler.add_job(refresh_upcoming, "interval", minutes=30, kwargs={"days": 3}, id="refresh_upcoming")
    scheduler.add_job(broadcast_morning_digest, "cron", hour=9, minute=0, args=[bot], id="digest")
    scheduler.add_job(broadcast_results_summary, "cron", hour=7, minute=0, args=[bot], id="results")
    scheduler.start()

    await _setup_commands(bot)
    asyncio.create_task(_warmup(bot))
    asyncio.create_task(_notify_startup(bot))
    logger.info("Scheduler started — MLB Baseball Signals bot is running")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
