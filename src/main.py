"""Entry point for the MLB Baseball Signals Telegram bot."""
from __future__ import annotations

import asyncio

from aiogram import Bot, Dispatcher
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
        # Try to restore model files from database (persisted from previous run)
        if not Predictor().ready:
            logger.info("Models not on disk — trying to restore from database...")
            restored = await restore_models_from_db()
            if restored:
                logger.info("Models restored from DB successfully")
            else:
                logger.info("No models in DB — cold start: bootstrapping MLB history...")
                await bootstrap_history()
                await train_models(bot=bot)
        else:
            logger.info("Models found on disk — running incremental refresh")
            await refresh_upcoming(days=7)
    except Exception as e:
        logger.error(f"Warmup failed (bot will still work): {e}")


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

    dp = Dispatcher()
    dp.message.middleware(AccessMiddleware())
    dp.callback_query.middleware(AccessMiddleware())
    dp.include_router(router)

    scheduler = AsyncIOScheduler(timezone=settings.tz)
    scheduler.add_job(daily_cycle, "cron", hour=4, minute=0, args=[bot], id="daily")
    scheduler.add_job(generate_and_broadcast, "interval", hours=1, args=[bot], id="signals_loop")
    scheduler.add_job(settle_pending, "interval", minutes=20, id="settle")
    # Каждые 30 мин обновляем статусы матчей (FINISHED + финальный счёт).
    # Игры MLB идут ~3 часа — нужно быстро получать результаты для settle.
    scheduler.add_job(refresh_upcoming, "interval", minutes=30, kwargs={"days": 3}, id="refresh_upcoming")
    scheduler.add_job(broadcast_morning_digest, "cron", hour=9, minute=0, args=[bot], id="digest")
    scheduler.start()

    asyncio.create_task(_warmup(bot))
    logger.info("Scheduler started — MLB Baseball Signals bot is running")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
