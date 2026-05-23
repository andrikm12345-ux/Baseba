"""Telegram command handlers for the MLB Baseball bot."""
from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from aiogram import Bot, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import BufferedInputFile, CallbackQuery, Message
from loguru import logger
from sqlalchemy import select

from src.bot.formatters import (
    HELP_TEXT,
    WELCOME_TEXT,
    format_roi_stats,
    format_signal,
    format_signal_short,
    format_training_report,
)
from src.bot.keyboards import admin_menu, main_menu, notifications_kb, signals_filter_kb
from src.config import settings
from src.data.database import (
    AiPrediction,
    Match,
    PendingUser,
    SessionLocal,
    Signal,
    Subscriber,
    Team,
)
from src.data.settings_store import get_bool, set_bool
from src.signals.tracker import roi_stats

router = Router()


def is_admin(user_id: int) -> bool:
    return user_id in settings.admin_ids


# ─── User commands ───────────────────────────────────────────────────────────

@router.message(Command("start"))
async def cmd_start(msg: Message):
    async with SessionLocal() as session:
        sub = await session.get(Subscriber, msg.from_user.id)
        if sub is None:
            session.add(Subscriber(
                chat_id=msg.from_user.id,
                username=msg.from_user.username,
            ))
            await session.commit()
    await msg.answer(WELCOME_TEXT, parse_mode="HTML", reply_markup=main_menu())


@router.message(Command("help"))
async def cmd_help(msg: Message):
    await msg.answer(HELP_TEXT, parse_mode="HTML")


@router.message(Command("signals"))
@router.message(lambda m: m.text == "⚾ Сигналы")
async def cmd_signals(msg: Message):
    await msg.answer("Фильтр сигналов:", reply_markup=signals_filter_kb())


@router.callback_query(lambda c: c.data and c.data.startswith("filter:"))
async def cb_filter(cb: CallbackQuery):
    flt = cb.data.split(":", 1)[1]
    horizon = datetime.utcnow() - timedelta(days=3)
    async with SessionLocal() as session:
        q = select(Signal).join(Match).where(
            Signal.created_at >= horizon,
            Match.utc_date >= datetime.utcnow() - timedelta(hours=2),
        ).order_by(Signal.created_at.desc()).limit(20)
        if flt == "ML":
            q = q.where(Signal.market == "ML")
        elif flt == "TOTAL":
            q = q.where(Signal.market == "TOTAL")
        elif flt == "RL":
            q = q.where(Signal.market == "RL")
        elif flt == "value":
            q = q.where(Signal.book_odds > 1.0)
        signals: List[Signal] = list((await session.execute(q)).scalars())
        if not signals:
            await cb.message.edit_text("Нет сигналов за последние 3 дня.")
            return
        ai_on = await get_bool("ai_ensemble_enabled", False)
        texts = []
        for s in signals:
            match = await session.get(Match, s.match_id)
            if not match:
                continue
            home = await session.get(Team, match.home_team_id)
            away = await session.get(Team, match.away_team_id)
            ai_comment = None
            if ai_on and s.is_ai_ensemble and s.commentary:
                ai_comment = s.commentary
            texts.append(format_signal(s, match, home, away, ai_comment))
    if not texts:
        await cb.message.edit_text("Нет сигналов.")
        return
    for text in texts[:5]:
        await cb.message.answer(text, parse_mode="HTML")
    await cb.answer()


@router.message(Command("today"))
@router.message(lambda m: m.text == "📅 Сегодня")
async def cmd_today(msg: Message):
    today = datetime.utcnow().date()
    start = datetime.combine(today, datetime.min.time())
    end = start + timedelta(days=1)
    async with SessionLocal() as session:
        q = select(Match, Team, Team).where(
            Match.utc_date >= start,
            Match.utc_date < end,
        ).limit(15)
        rows = (await session.execute(
            select(Match).where(Match.utc_date >= start, Match.utc_date < end).limit(15)
        )).scalars().all()
        if not rows:
            await msg.answer("Нет игр MLB на сегодня.")
            return
        lines = ["📅 <b>Игры MLB сегодня:</b>\n"]
        for m in rows:
            home = await session.get(Team, m.home_team_id)
            away = await session.get(Team, m.away_team_id)
            h = home.short_name or home.name if home else "?"
            a = away.short_name or away.name if away else "?"
            dt_msk = m.utc_date.replace(tzinfo=timezone.utc).astimezone(
                timezone(timedelta(hours=3))
            ).strftime("%H:%M")
            status = ""
            if m.status == "FINISHED":
                status = f" | {m.home_runs}:{m.away_runs}"
            lines.append(f"⚾ {h} vs {a}  {dt_msk} МСК{status}")
    await msg.answer("\n".join(lines), parse_mode="HTML")


@router.message(Command("stats"))
@router.message(lambda m: m.text == "📊 Статистика")
async def cmd_stats(msg: Message):
    model_s = await roi_stats(last_n=200, only_value=None)
    value_s = await roi_stats(last_n=200, only_value=True)
    ai_s = await roi_stats(last_n=200, ai_only=True, only_value=None)
    text = format_roi_stats(model_s, value_s, ai_s)
    await msg.answer(text, parse_mode="HTML")


@router.message(Command("chart"))
@router.message(lambda m: m.text == "📈 Кривая ROI")
async def cmd_chart(msg: Message):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        async with SessionLocal() as session:
            q = select(Signal).where(
                Signal.settled.is_(True),
                Signal.book_odds > 1.0,
            ).order_by(Signal.created_at.asc())
            rows: List[Signal] = list((await session.execute(q)).scalars())

        if not rows:
            await msg.answer("Нет данных для построения графика.")
            return

        profits = [r.profit_units or 0.0 for r in rows]
        cumulative = list(np.cumsum(profits))
        x = list(range(1, len(cumulative) + 1))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, cumulative, linewidth=2, color="#1f77b4")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.fill_between(x, cumulative, 0, alpha=0.15, color="#1f77b4")
        ax.set_title("Кривая ROI — MLB Baseball Signals", fontsize=14, pad=12)
        ax.set_xlabel("Сигнал #")
        ax.set_ylabel("Прибыль (единицы)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        await msg.answer_photo(
            BufferedInputFile(buf.read(), filename="roi.png"),
            caption=f"Всего VALUE ставок: {len(rows)} | Итог: {cumulative[-1]:+.1f} ед."
        )
    except Exception as e:
        logger.error(f"chart error: {e}")
        await msg.answer("Не удалось построить график.")


@router.message(Command("notifications"))
@router.message(lambda m: m.text == "🔔 Уведомления")
async def cmd_notifications(msg: Message):
    async with SessionLocal() as session:
        sub = await session.get(Subscriber, msg.from_user.id)
        enabled = sub.notifications_enabled if sub else True
    await msg.answer(
        f"Уведомления о новых сигналах: {'🔔 включены' if enabled else '🔕 выключены'}",
        reply_markup=notifications_kb(enabled),
    )


@router.callback_query(lambda c: c.data == "notif:toggle")
async def cb_notif_toggle(cb: CallbackQuery):
    async with SessionLocal() as session:
        sub = await session.get(Subscriber, cb.from_user.id)
        if sub:
            sub.notifications_enabled = not sub.notifications_enabled
            enabled = sub.notifications_enabled
            await session.commit()
        else:
            enabled = True
    await cb.message.edit_reply_markup(reply_markup=notifications_kb(enabled))
    await cb.answer("Уведомления " + ("включены" if enabled else "выключены"))


# ─── Admin commands ───────────────────────────────────────────────────────────

@router.message(Command("admin"))
async def cmd_admin(msg: Message):
    if not is_admin(msg.from_user.id):
        return
    await msg.answer("🔧 Админ-панель", reply_markup=admin_menu())


@router.callback_query(lambda c: c.data and c.data.startswith("admin:"))
async def cb_admin(cb: CallbackQuery):
    if not is_admin(cb.from_user.id):
        await cb.answer("Нет доступа")
        return
    action = cb.data.split(":", 1)[1]
    if action == "users":
        await _admin_users(cb)
    elif action == "ai_toggle":
        await _admin_ai_toggle(cb)
    elif action == "train":
        await cb.message.answer("Запустите /train вручную")
        await cb.answer()
    elif action == "leads":
        await _admin_leads(cb)


async def _admin_users(cb: CallbackQuery):
    async with SessionLocal() as session:
        subs = (await session.execute(
            select(Subscriber).where(Subscriber.active.is_(True))
        )).scalars().all()
    text = f"👥 Активных подписчиков: <b>{len(subs)}</b>\n\n"
    for s in subs[:20]:
        uname = f"@{s.username}" if s.username else str(s.chat_id)
        notif = "🔔" if s.notifications_enabled else "🔕"
        text += f"{notif} {uname}\n"
    await cb.message.answer(text, parse_mode="HTML")
    await cb.answer()


async def _admin_ai_toggle(cb: CallbackQuery):
    current = await get_bool("ai_ensemble_enabled", False)
    new_val = not current
    await set_bool("ai_ensemble_enabled", new_val)
    state = "включён" if new_val else "выключен"
    await cb.answer(f"AI ансамбль {state}", show_alert=True)


async def _admin_leads(cb: CallbackQuery):
    async with SessionLocal() as session:
        leads = (await session.execute(
            select(PendingUser).order_by(PendingUser.last_seen_at.desc()).limit(15)
        )).scalars().all()
    if not leads:
        await cb.answer("Нет лидов")
        return
    lines = [f"📋 <b>Лиды ({len(leads)}):</b>\n"]
    for p in leads:
        name = p.first_name or ""
        uname = f"@{p.username}" if p.username else str(p.chat_id)
        lines.append(f"• {uname} {name} (посещений: {p.start_count})")
    await cb.message.answer("\n".join(lines), parse_mode="HTML")
    await cb.answer()


@router.message(Command("allow"))
async def cmd_allow(msg: Message):
    if not is_admin(msg.from_user.id):
        return
    parts = msg.text.split()
    if len(parts) < 2:
        await msg.answer("Использование: /allow <user_id>")
        return
    try:
        uid = int(parts[1])
    except ValueError:
        await msg.answer("Неверный ID")
        return
    async with SessionLocal() as session:
        sub = await session.get(Subscriber, uid)
        if sub is None:
            session.add(Subscriber(chat_id=uid, active=True, notifications_enabled=True))
        else:
            sub.active = True
            sub.notifications_enabled = True
        await session.commit()
    await msg.answer(f"✅ Доступ предоставлен: {uid}")
    try:
        from aiogram import Bot
        bot = msg.bot
        await bot.send_message(
            uid,
            "✅ Ваш доступ к боту подтверждён. Используйте /start",
        )
    except Exception:
        pass


@router.message(Command("deny"))
async def cmd_deny(msg: Message):
    if not is_admin(msg.from_user.id):
        return
    parts = msg.text.split()
    if len(parts) < 2:
        await msg.answer("Использование: /deny <user_id>")
        return
    try:
        uid = int(parts[1])
    except ValueError:
        await msg.answer("Неверный ID")
        return
    async with SessionLocal() as session:
        sub = await session.get(Subscriber, uid)
        if sub:
            sub.active = False
            await session.commit()
    await msg.answer(f"🚫 Доступ закрыт: {uid}")


# ─── Broadcasting ─────────────────────────────────────────────────────────────

async def broadcast_signal(bot: Bot, text: str) -> int:
    """Send signal to all active subscribers with notifications enabled."""
    sent = 0
    async with SessionLocal() as session:
        subs = (await session.execute(
            select(Subscriber).where(
                Subscriber.active.is_(True),
                Subscriber.notifications_enabled.is_(True),
            )
        )).scalars().all()
        deactivated = []
        for sub in subs:
            try:
                await bot.send_message(sub.chat_id, text, parse_mode="HTML")
                sent += 1
            except Exception as e:
                err_str = str(e).lower()
                if "forbidden" in err_str or "blocked" in err_str or "deactivated" in err_str:
                    deactivated.append(sub.chat_id)
                    logger.info(f"Deactivating subscriber {sub.chat_id}: {e}")
        if deactivated:
            for cid in deactivated:
                sub = await session.get(Subscriber, cid)
                if sub:
                    sub.notifications_enabled = False
            await session.commit()
    return sent


async def broadcast_morning_digest(bot: Bot) -> None:
    """Send a morning digest of today's game signals."""
    today = datetime.utcnow().date()
    start = datetime.combine(today, datetime.min.time())
    end = start + timedelta(days=1)
    async with SessionLocal() as session:
        signals = (await session.execute(
            select(Signal).join(Match).where(
                Match.utc_date >= start,
                Match.utc_date < end,
            ).order_by(Signal.created_at.desc())
        )).scalars().all()
        matches = {}
        teams = {}
        for s in signals:
            m = await session.get(Match, s.match_id)
            if m:
                matches[s.match_id] = m
                teams[m.home_team_id] = await session.get(Team, m.home_team_id)
                teams[m.away_team_id] = await session.get(Team, m.away_team_id)
    if not signals:
        return
    text = "☀️ <b>Утренний дайджест — MLB</b>\n\n" + format_signal_short(signals, matches, teams)
    await broadcast_signal(bot, text)
