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
from src.ml.predict import Predictor
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
        if not predictor.ready:
            logger.info("Cold start — bootstrapping MLB history...")
            await bootstrap_history()
            await train_models(bot=bot)
        else:
            logger.info("Models found — running incremental refresh")
            await refresh_upcoming(days=7)
    except Exception as e:
        logger.error(f"Warmup failed (bot will still work): {e}")


async def main() -> None:
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
    scheduler.add_job(settle_pending, "interval", minutes=30, id="settle")
    scheduler.add_job(refresh_upcoming, "interval", hours=6, kwargs={"days": 7}, id="refresh_upcoming")
    scheduler.add_job(broadcast_morning_digest, "cron", hour=9, minute=0, args=[bot], id="digest")
    scheduler.start()

    asyncio.create_task(_warmup(bot))
    logger.info("Scheduler started — MLB Baseball Signals bot is running")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
