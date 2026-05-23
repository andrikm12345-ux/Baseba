from __future__ import annotations

from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject

from src.config import settings
from src.data.database import PendingUser, SessionLocal, Subscriber


class AccessMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if isinstance(event, (Message, CallbackQuery)):
            user = event.from_user
            if user is None:
                return
            uid = user.id
            if uid in settings.admin_ids:
                return await handler(event, data)
            async with SessionLocal() as session:
                sub = await session.get(Subscriber, uid)
                if sub and sub.active:
                    return await handler(event, data)
                # track as pending lead
                pending = await session.get(PendingUser, uid)
                if pending is None:
                    session.add(PendingUser(
                        chat_id=uid,
                        username=user.username,
                        first_name=user.first_name,
                        last_name=user.last_name,
                    ))
                else:
                    pending.last_seen_at = __import__("datetime").datetime.utcnow()
                    pending.start_count += 1
                await session.commit()
            if isinstance(event, Message):
                await event.answer(
                    "⚾ <b>Бейсбол Сигналы</b>\n\n"
                    "Доступ закрыт. Ожидайте подтверждения от администратора.",
                    parse_mode="HTML",
                )
        return
