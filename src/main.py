"""Entry point for the MLB Baseball Signals Telegram bot."""
from __future__ import annotations

import asyncio
import os

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

from src.bot.access import AccessMiddleware
from src.bot.handlers import broadcast_morning_digest, router
from src.config import settings
from src.data.database import init_db
from src.ml.predict import Predictor, restore_models_from_db
from src.pipeline import (
    bootstrap_history,
    daily_cycle,
    generate_and_broadcast,
    refresh_upcoming,
    train_models,
)
from src.signals.tracker import settle_pending


async def _warmup(bot: Bot) -> None:
    try:
        predictor = Predictor()
        if predictor.ready:
            logger.info("Models found on disk — running incremental refresh")
            await refresh_upcoming(days=7)
            await settle_pending()
            await generate_and_broadcast(bot)
            return

        # Попытка восстановить модели из БД
        logger.info("Models not on disk — trying to restore from database...")
        restored = await restore_models_from_db()
        if restored and Predictor().ready:
            logger.info("Models restored from DB successfully")
            await refresh_upcoming(days=7)
            await settle_pending()
            await generate_and_broadcast(bot)
            return

        # Модели в БД есть, но feature mismatch → нужно переобучение
        if restored:
            logger.warning("Models from DB have feature mismatch — retraining on existing data")
        else:
            # Холодный старт: нет ни моделей, ни данных
            logger.info("No models in DB — cold start: bootstrapping MLB history...")
            await bootstrap_history()

        await refresh_upcoming(days=7)
        await train_models(bot=bot)

        # Сразу генерируем сигналы после обучения, не ждём расписание
        if Predictor().ready:
            await generate_and_broadcast(bot)
        else:
            logger.warning("Warmup: models still not ready after training — check training logs")
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
        logger.critical(
            "TELEGRAM_BOT_TOKEN not set! "
            "Go to Railway → Variables and add TELEGRAM_BOT_TOKEN=<your bot token>"
        )
        raise SystemExit(1)

    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    # Clear any active webhook before polling starts
    await bot.delete_webhook(drop_pending_updates=True)
    logger.info("Webhook cleared")

    # Init DB tables before bot handles any messages
    await init_db()
    logger.info("DB initialized")

    # Persist AI_ENSEMBLE env var to DB on first start so it survives future restarts
    # even if the env var is later removed.
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
    # Каждые 30 мин обновляем статусы матчей (FINISHED + финальный счёт).
    # Игры MLB идут ~3 часа — нужно быстро получать результаты для settle.
    scheduler.add_job(refresh_upcoming, "interval", minutes=30, kwargs={"days": 3}, id="refresh_upcoming")
    scheduler.add_job(broadcast_morning_digest, "cron", hour=9, minute=0, args=[bot], id="digest")
    scheduler.start()

    asyncio.create_task(_warmup(bot))
    asyncio.create_task(_notify_startup(bot))
    logger.info("Scheduler started — MLB Baseball Signals bot is running")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
