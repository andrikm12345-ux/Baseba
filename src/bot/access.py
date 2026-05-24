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
                # Track as pending lead
                pending = await session.get(PendingUser, uid)
                if pending is None:
                    session.add(PendingUser(
                        chat_id=uid,
                        username=user.username,
                        first_name=user.first_name,
                        last_name=user.last_name,
                    ))
                    # Notify admins about new lead
                    await session.commit()
                    try:
                        from aiogram import Bot
                        bot = event.bot if hasattr(event, "bot") else None
                        if bot is None and isinstance(event, (Message, CallbackQuery)):
                            bot = event.bot
                        if bot:
                            name = user.first_name or user.username or str(uid)
                            uname = f" (@{user.username})" if user.username else ""
                            for admin_id in settings.admin_ids:
                                await bot.send_message(
                                    admin_id,
                                    f"🔔 <b>Новый лид!</b>\n"
                                    f"👤 {name}{uname}\n"
                                    f"🆔 <code>{uid}</code>\n\n"
                                    f"Открой /admin → Лиды чтобы одобрить.",
                                    parse_mode="HTML",
                                )
                    except Exception:
                        pass
                else:
                    pending.last_seen_at = __import__("datetime").datetime.utcnow()
                    pending.start_count += 1
                    await session.commit()
            if isinstance(event, Message):
                await event.answer(
                    f"⚾ <b>Бейсбол Сигналы</b>\n\n"
                    f"Доступ только по приглашению администратора.\n\n"
                    f"Ваш Telegram ID:\n<code>{uid}</code>\n\n"
                    f"Скопируйте ID и отправьте администратору — он одобрит вас.",
                    parse_mode="HTML",
                )
            elif isinstance(event, CallbackQuery):
                await event.answer("Нет доступа. Обратитесь к администратору.", show_alert=True)
        return
