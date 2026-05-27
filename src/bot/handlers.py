"""Telegram command handlers for the MLB Baseball bot."""
from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from aiogram import Bot, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import BufferedInputFile, CallbackQuery, Message
from loguru import logger
from sqlalchemy import delete, select

from src.bot.formatters import (
    HELP_TEXT,
    MARKET_LABELS,
    PICK_LABELS,
    WELCOME_TEXT,
    _MONTHS_RU,
    format_history,
    format_roi_stats,
    format_signal,
    format_signal_short,
    format_training_report,
)
from src.bot.keyboards import admin_menu, history_nav_kb, lead_kb, main_menu, notifications_kb, user_remove_kb
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


class AdminStates(StatesGroup):
    waiting_user_id = State()


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


def _msk(dt: datetime) -> str:
    return dt.replace(tzinfo=timezone.utc).astimezone(
        timezone(timedelta(hours=3))
    ).strftime("%d.%m %H:%M МСК")


@router.message(Command("help"))
@router.message(lambda m: m.text and m.text.startswith("ℹ️"))
async def cmd_help(msg: Message):
    await msg.answer(HELP_TEXT, parse_mode="HTML")


@router.message(Command("signals"))
@router.message(lambda m: m.text == "⚾ Сигналы")
async def cmd_signals(msg: Message):
    await _show_signals(msg, flt="all")


async def _show_signals(event, flt: str = "all"):
    """Show signals — works with both Message and CallbackQuery."""
    is_cb = isinstance(event, CallbackQuery)
    send = event.message.answer if is_cb else event.answer

    # Только активные (не закрытые) сигналы для игр которые ещё впереди
    cutoff = datetime.utcnow() - timedelta(hours=4)  # игра закончилась не более 4ч назад
    async with SessionLocal() as session:
        q = (
            select(Signal)
            .join(Match, Match.id == Signal.match_id)
            .where(
                Signal.settled.is_(False),
                Match.utc_date >= cutoff,
            )
            .order_by(Match.utc_date.asc())
            .limit(20)
        )
        if flt == "ML":
            q = q.where(Signal.market == "ML")
        elif flt == "TOTAL":
            q = q.where(Signal.market == "TOTAL")
        elif flt == "ITB":
            q = q.where(Signal.market == "ITB")
        elif flt == "value":
            q = q.where(Signal.book_odds > 1.0)
        signals: List[Signal] = list((await session.execute(q)).scalars())
        if not signals:
            await send(
                "📭 <b>Активных сигналов нет.</b>\n\n"
                "Сигналы появляются за 3 часа до начала игры.\n"
                "Нажмите <b>🔄 Запустить анализ</b> для проверки ближайших игр.\n"
                "Прошедшие ставки — в разделе <b>📜 История ставок</b>.",
                parse_mode="HTML",
            )
            if is_cb:
                await event.answer()
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
        await send("Нет активных сигналов.")
        if is_cb:
            await event.answer()
        return
    for text in texts[:8]:
        await send(text, parse_mode="HTML")
    if is_cb:
        await event.answer()


@router.message(Command("today"))
@router.message(lambda m: m.text == "📅 Сегодня")
async def cmd_today(msg: Message):
    msk = timezone(timedelta(hours=3))
    now_msk = datetime.now(msk)
    msk_today_start = now_msk.replace(hour=0, minute=0, second=0, microsecond=0)
    start = (msk_today_start - timedelta(hours=18)).astimezone(timezone.utc).replace(tzinfo=None)
    end = (msk_today_start + timedelta(days=1)).astimezone(timezone.utc).replace(tzinfo=None)

    async with SessionLocal() as session:
        rows = (await session.execute(
            select(Match).where(Match.utc_date >= start, Match.utc_date < end).order_by(Match.utc_date)
        )).scalars().all()

    if not rows:
        await msg.answer("Нет игр MLB на сегодня.")
        return

    finished = [m for m in rows if m.status == "FINISHED"]
    upcoming = [m for m in rows if m.status != "FINISHED"]

    parts = []

    if finished:
        lines = [f"✅ <b>Результаты ({len(finished)}):</b>\n"]
        async with SessionLocal() as session:
            for m in finished:
                home = await session.get(Team, m.home_team_id)
                away = await session.get(Team, m.away_team_id)
                h = home.short_name or home.name if home else "?"
                a = away.short_name or away.name if away else "?"
                lines.append(f"⚾ {h} {m.home_runs}  —  {m.away_runs} {a}")
        parts.append("\n".join(lines))

    if upcoming:
        lines = [f"🕐 <b>Предстоящие ({len(upcoming)}):</b>\n"]
        async with SessionLocal() as session:
            for m in upcoming:
                home = await session.get(Team, m.home_team_id)
                away = await session.get(Team, m.away_team_id)
                h = home.short_name or home.name if home else "?"
                a = away.short_name or away.name if away else "?"
                dt_str = _msk(m.utc_date)
                pitcher = ""
                if m.home_pitcher_name or m.away_pitcher_name:
                    hp = m.home_pitcher_name or "?"
                    ap = m.away_pitcher_name or "?"
                    pitcher = f"\n   ↳ {hp} vs {ap}"
                lines.append(f"⚾ {h} vs {a}  {dt_str}{pitcher}")
        parts.append("\n".join(lines))

    for part in parts:
        await msg.answer(part, parse_mode="HTML")

    if upcoming:
        await msg.answer(
            "⚠️ <i>Стартовые питчеры могут измениться в последний момент — "
            "травма или ротация. Сверяйтесь ближе к первому питчу.</i>",
            parse_mode="HTML",
        )


@router.message(Command("stats"))
@router.message(lambda m: m.text == "📊 Статистика")
async def cmd_stats(msg: Message):
    model_s = await roi_stats(last_n=200, only_value=False)   # 📊 MODEL (без коэффа БК)
    value_s = await roi_stats(last_n=200, only_value=True)    # 🔥 VALUE (с edge)
    ai_s = await roi_stats(last_n=200, ai_only=True, only_value=None)  # 🤖 AI ансамбль
    text = format_roi_stats(model_s, value_s, ai_s)
    await msg.answer(text, parse_mode="HTML")


@router.message(Command("chart"))
@router.message(lambda m: m.text == "📈 График ROI")
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
@router.message(lambda m: m.text and m.text.startswith("🔔"))
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


# ─── New feature handlers ─────────────────────────────────────────────────────



@router.message(lambda m: m.text == "🔄 Запустить анализ")
async def cmd_refresh_odds(msg: Message):
    await msg.answer("🔄 Запускаю обновление расписания...")
    try:
        from src.pipeline import refresh_upcoming, generate_and_broadcast, train_models
        from src.signals.tracker import settle_pending
        from src.ml.predict import Predictor
        from sqlalchemy import select, func
        from src.data.database import Signal, Match

        n = await refresh_upcoming(days=3)

        # Статус моделей
        predictor = Predictor()
        models_ready = predictor.ready

        # Диагностика: считаем сигналы и завершённые матчи
        async with SessionLocal() as session:
            total_sigs = (await session.execute(select(func.count()).select_from(Signal))).scalar()
            unsettled = (await session.execute(
                select(func.count()).select_from(Signal).where(Signal.settled.is_(False))
            )).scalar()
            finished_matches = (await session.execute(
                select(func.count()).select_from(Match).where(Match.status == "FINISHED")
            )).scalar()
            # Next game without a signal yet (exclude already-signaled matches)
            now_utc = datetime.utcnow()
            signaled_ids_sq = select(Signal.match_id).distinct().scalar_subquery()
            next_game = (await session.execute(
                select(Match).where(
                    Match.status != "FINISHED",
                    Match.utc_date > now_utc,
                    Match.id.not_in(signaled_ids_sq),
                ).order_by(Match.utc_date).limit(1)
            )).scalar_one_or_none()

        model_status = "✅ готовы" if models_ready else "⏳ ещё обучаются"
        next_signal_hint = ""
        if next_game:
            signal_time = next_game.utc_date - timedelta(hours=5)
            game_msk = (next_game.utc_date.replace(tzinfo=timezone.utc)
                        .astimezone(timezone(timedelta(hours=3)))).strftime("%d.%m %H:%M МСК")
            if signal_time <= now_utc:
                next_signal_hint = f"\n⏰ Сигнал для игры {game_msk} будет при следующей проверке"
            else:
                msk_time = (signal_time.replace(tzinfo=timezone.utc)
                            .astimezone(timezone(timedelta(hours=3)))).strftime("%H:%M МСК")
                next_signal_hint = f"\n⏰ Следующий сигнал ожидается в ~{msk_time} (игра в {game_msk})"

        await msg.answer(
            f"✅ Загружено {n} матчей\n"
            f"🤖 Модели: {model_status}\n"
            f"📊 Сигналов в БД: {total_sigs} (незакрытых: {unsettled})\n"
            f"🏁 Завершённых матчей: {finished_matches}"
            f"{next_signal_hint}\n\n"
            f"Закрываю сыгранные..."
        )

        # Если модели не готовы — запускаем обучение в фоне
        if not models_ready and finished_matches >= 200:
            await msg.answer(
                "⚙️ Модели не найдены — запускаю обучение в фоне (~5 мин).\n"
                "Придёт уведомление когда готово. Сигналы появятся после обучения."
            )
            import asyncio as _aio
            _aio.create_task(train_models(bot=msg.bot))

        settled = await settle_pending()
        await msg.answer(
            f"✅ Закрыто сигналов: {settled}\n"
            f"Генерирую новые сигналы..."
        )
        new_count = await generate_and_broadcast(msg.bot)
        if new_count == 0:
            await msg.answer(
                f"📭 Новых сигналов нет.\n"
                f"{next_signal_hint}\n\n"
                "Возможные причины: сигналы уже были отправлены ранее, "
                "нет игр в ближайшие 5 часов или модели ещё обучаются.\n"
                "Сигналы появятся автоматически за 5 часов до начала игры.",
                reply_markup=main_menu(),
            )
        else:
            await msg.answer(f"✅ Отправлено {new_count} новых сигналов.", reply_markup=main_menu())
    except Exception as e:
        logger.error(f"refresh error: {e}")
        await msg.answer(f"❌ Ошибка: {e}")




# ─── History ─────────────────────────────────────────────────────────────────

@router.message(Command("history"))
@router.message(lambda m: m.text == "📜 История ставок")
async def cmd_history(msg: Message):
    await _show_history(msg, page=0)


@router.callback_query(lambda c: c.data and c.data.startswith("history:"))
async def cb_history(cb: CallbackQuery):
    page = int(cb.data.split(":")[1])
    await _show_history(cb, page=page)


async def _show_history(event, page: int = 0):
    is_cb = isinstance(event, CallbackQuery)
    send = event.message.answer if is_cb else event.answer

    msk = timezone(timedelta(hours=3))
    now_utc = datetime.utcnow()
    end_utc = now_utc - timedelta(days=page * 7)
    start_utc = end_utc - timedelta(days=7)

    async with SessionLocal() as session:
        rows = list((await session.execute(
            select(Signal)
            .join(Match, Match.id == Signal.match_id)
            .where(Match.utc_date >= start_utc, Match.utc_date < end_utc)
            .order_by(Match.utc_date.desc())
        )).scalars())

        # Группировка по MSK-дате
        day_map: dict = {}
        for s in rows:
            match = await session.get(Match, s.match_id)
            if not match:
                continue
            home_team = await session.get(Team, match.home_team_id)
            away_team = await session.get(Team, match.away_team_id)

            msk_date = match.utc_date.replace(tzinfo=timezone.utc).astimezone(msk).date()
            day_key = msk_date.isoformat()
            day_label = f"{msk_date.day} {_MONTHS_RU[msk_date.month]}"

            h_abbr = (home_team.short_name or home_team.name[:10]) if home_team else "?"
            a_abbr = (away_team.short_name or away_team.name[:10]) if away_team else "?"

            item = {
                "home_abbr": h_abbr,
                "away_abbr": a_abbr,
                "home_runs": match.home_runs,
                "away_runs": match.away_runs,
                "market": s.market,
                "pick": s.pick,
                "book_odds": s.book_odds,
                "profit": s.profit_units,
                "settled": s.settled,
                "won": s.won,
            }
            if day_key not in day_map:
                day_map[day_key] = (day_label, [])
            day_map[day_key][1].append(item)

    day_groups = [v for _, v in sorted(day_map.items(), reverse=True)]

    # Есть ли более старая страница (проверяем есть ли что-то ещё раньше)
    has_next = page < 3  # максимум 4 недели

    text = format_history(day_groups)
    kb = history_nav_kb(page, has_next)

    if is_cb:
        try:
            await event.message.edit_text(text, parse_mode="HTML", reply_markup=kb)
        except Exception:
            await send(text, parse_mode="HTML", reply_markup=kb)
        await event.answer()
    else:
        await send(text, parse_mode="HTML", reply_markup=kb)


# ─── Admin commands ───────────────────────────────────────────────────────────

async def _admin_menu_text_and_kb():
    from sqlalchemy import func
    ai_on = await get_bool("ai_ensemble_enabled", False)
    async with SessionLocal() as session:
        leads_count = (await session.execute(
            select(func.count()).select_from(PendingUser)
        )).scalar() or 0
        users_count = (await session.execute(
            select(func.count()).select_from(Subscriber).where(Subscriber.active.is_(True))
        )).scalar() or 0
    text = (
        f"🔧 <b>Админ-панель</b>\n\n"
        f"👥 Активных пользователей: <b>{users_count}</b>\n"
        f"📋 Лидов (не одобрено): <b>{leads_count}</b>\n"
        f"🤖 AI Ансамбль: {'✅ включён' if ai_on else '❌ выключен'}"
    )
    return text, admin_menu(leads_count=leads_count, ai_on=ai_on)


@router.message(Command("admin"))
async def cmd_admin(msg: Message):
    if not is_admin(msg.from_user.id):
        return
    text, kb = await _admin_menu_text_and_kb()
    await msg.answer(text, reply_markup=kb, parse_mode="HTML")


@router.message(Command("train"))
async def cmd_train(msg: Message):
    if not is_admin(msg.from_user.id):
        return
    await msg.answer("🔄 Обучение моделей запущено, подождите...")
    try:
        from src.pipeline import train_models
        await train_models(bot=msg.bot)
    except Exception as e:
        await msg.answer(f"❌ Ошибка обучения: {e}")


@router.message(Command("purge_signals"))
async def cmd_purge_signals(msg: Message):
    """Delete no-odds signals (book_odds=0) for unsettled future games.

    Usage:
      /purge_signals          — all future unsettled no-odds signals
      /purge_signals 2026-05-26  — only that date
    """
    if not is_admin(msg.from_user.id):
        return
    args = msg.text.split(maxsplit=1)

    async with SessionLocal() as session:
        if len(args) > 1:
            try:
                target_date = datetime.strptime(args[1].strip(), "%Y-%m-%d").date()
            except ValueError:
                await msg.answer("❌ Неверный формат даты. Используй: /purge_signals 2026-05-26")
                return
            day_start = datetime(target_date.year, target_date.month, target_date.day)
            day_end = day_start + timedelta(days=1)
            q = (
                select(Signal)
                .join(Match, Match.id == Signal.match_id)
                .where(
                    Signal.book_odds == 0.0,
                    Signal.settled.is_(False),
                    Match.utc_date >= day_start,
                    Match.utc_date < day_end,
                )
            )
            scope = str(target_date)
        else:
            # Все незакрытые сигналы без кэфов для будущих игр
            q = (
                select(Signal)
                .join(Match, Match.id == Signal.match_id)
                .where(
                    Signal.book_odds == 0.0,
                    Signal.settled.is_(False),
                    Match.utc_date > datetime.utcnow(),
                )
            )
            scope = "все будущие"

        rows = (await session.execute(q)).scalars().all()
        if not rows:
            await msg.answer(f"✅ Нет сигналов без коэффициентов ({scope})")
            return

        ids = [r.id for r in rows]
        await session.execute(delete(Signal).where(Signal.id.in_(ids)))
        await session.commit()

    logger.info(f"purge_signals: deleted {len(ids)} no-odds signals ({scope}) by admin {msg.from_user.id}")
    await msg.answer(
        f"🗑 Удалено <b>{len(ids)}</b> сигналов без коэффициентов ({scope}).\n"
        f"Эти матчи получат нормальные сигналы с кэфами когда войдут в 5-часовое окно.",
        parse_mode="HTML",
    )


@router.message(Command("debugodds"))
async def cmd_debug_odds(msg: Message):
    if not is_admin(msg.from_user.id):
        return
    from src.config import settings as cfg
    if not cfg.odds_api_key:
        await msg.answer("❌ ODDS_API_KEY не задан")
        return
    import json
    from src.data.odds_api import OddsApiClient
    client = OddsApiClient(cfg.odds_api_key)
    try:
        # Fetch all baseball events, find MLB league slug + check odds for first event
        await msg.answer("🔍 Ищу MLB события (все бейсбол, до 200)...")
        data = await client._get("/events", {"sport": "baseball", "limit": 200})
        events = data if isinstance(data, list) else data.get("events", [])

        # Collect unique league slugs
        leagues: dict[str, str] = {}
        mlb_events = []
        for ev in events:
            lg = ev.get("league") or {}
            slug = lg.get("slug", "") if isinstance(lg, dict) else ""
            name = lg.get("name", "") if isinstance(lg, dict) else ""
            leagues[slug] = name
            if "mlb" in slug.lower() or "major-league" in slug.lower():
                mlb_events.append(ev)

        league_list = "\n".join(f"  {s}: {n}" for s, n in sorted(leagues.items()))
        await msg.answer(f"<b>Лиги в бейсболе ({len(leagues)}):</b>\n<pre>{league_list[:1000]}</pre>", parse_mode="HTML")

        if mlb_events:
            ev = mlb_events[0]
            await msg.answer(f"<b>MLB событие найдено:</b>\n<pre>{json.dumps(ev)[:800]}</pre>", parse_mode="HTML")
            # Try /odds for this event
            ev_id = ev.get("id")
            if ev_id:
                await msg.answer(f"🔍 Запрашиваю /odds для event {ev_id}...")
                try:
                    odds = await client._get("/odds", {"eventId": ev_id, "bookmakers": "Bet365,Betfair Exchange"})
                    await msg.answer(f"<b>/odds ответ:</b>\n<pre>{json.dumps(odds)[:1200]}</pre>", parse_mode="HTML")
                except Exception as e:
                    await msg.answer(f"/odds ошибка: {e}")
        else:
            await msg.answer("⚠️ MLB событий не найдено среди бейсбол событий.\nВозможно лига называется иначе — смотри список выше.")
    except Exception as e:
        await msg.answer(f"❌ Ошибка: {e}")
    finally:
        await client.close()


@router.callback_query(lambda c: c.data and c.data.startswith("admin:"))
async def cb_admin(cb: CallbackQuery, state: FSMContext):
    if not is_admin(cb.from_user.id):
        await cb.answer("Нет доступа")
        return
    action = cb.data.split(":", 1)[1]
    if action == "users":
        await _admin_users(cb)
    elif action == "leads":
        await _admin_leads(cb)
    elif action == "ai_toggle":
        await _admin_ai_toggle(cb)
    elif action == "train":
        await cb.answer("Запускаю обучение...")
        await cb.message.answer("🔄 Обучение моделей запущено, подождите...")
        try:
            from src.pipeline import train_models
            await train_models(bot=cb.bot)
        except Exception as e:
            await cb.message.answer(f"❌ Ошибка обучения: {e}")
    elif action == "add_user":
        await state.set_state(AdminStates.waiting_user_id)
        await cb.message.answer(
            "Введите Telegram ID пользователя которого хотите добавить:\n"
            "(или /cancel чтобы отменить)"
        )
        await cb.answer()
    elif action == "close":
        await cb.message.delete()
        await cb.answer()


async def _admin_users(cb: CallbackQuery):
    async with SessionLocal() as session:
        subs = (await session.execute(
            select(Subscriber).where(Subscriber.active.is_(True)).order_by(Subscriber.subscribed_at.desc())
        )).scalars().all()
    if not subs:
        await cb.message.answer("👥 Нет активных пользователей.")
        await cb.answer()
        return
    await cb.message.answer(f"👥 <b>Активных пользователей: {len(subs)}</b>", parse_mode="HTML")
    for s in subs[:20]:
        uname = f"@{s.username}" if s.username else "—"
        notif = "🔔" if s.notifications_enabled else "🔕"
        text = f"{notif} {uname}\n🆔 <code>{s.chat_id}</code>"
        await cb.message.answer(text, parse_mode="HTML", reply_markup=user_remove_kb(s.chat_id))
    await cb.answer()


async def _admin_ai_toggle(cb: CallbackQuery):
    current = await get_bool("ai_ensemble_enabled", False)
    new_val = not current
    await set_bool("ai_ensemble_enabled", new_val)
    state = "включён ✅" if new_val else "выключен ❌"
    await cb.answer(f"AI ансамбль {state}", show_alert=True)
    # Refresh admin menu
    text, kb = await _admin_menu_text_and_kb()
    try:
        await cb.message.edit_text(text, reply_markup=kb, parse_mode="HTML")
    except Exception:
        pass


async def _admin_leads(cb: CallbackQuery):
    async with SessionLocal() as session:
        leads = (await session.execute(
            select(PendingUser).order_by(PendingUser.last_seen_at.desc()).limit(20)
        )).scalars().all()
    if not leads:
        await cb.answer("Нет лидов ✨", show_alert=True)
        return
    await cb.message.answer(f"📋 <b>Лидов: {len(leads)}</b>", parse_mode="HTML")
    for p in leads:
        name = p.first_name or ""
        last = p.last_name or ""
        uname = f"@{p.username}" if p.username else "—"
        from src.bot.formatters import _msk
        last_seen = _msk(p.last_seen_at) if p.last_seen_at else "?"
        text = (
            f"👤 {name} {last} {uname}\n"
            f"🆔 <code>{p.chat_id}</code>\n"
            f"🕐 {last_seen} | Попыток: {p.start_count}"
        )
        await cb.message.answer(text, parse_mode="HTML", reply_markup=lead_kb(p.chat_id))
    await cb.answer()


# ─── Lead approve / deny ─────────────────────────────────────────────────────

@router.callback_query(lambda c: c.data and c.data.startswith("lead:"))
async def cb_lead(cb: CallbackQuery):
    if not is_admin(cb.from_user.id):
        await cb.answer("Нет доступа")
        return
    parts = cb.data.split(":")
    action, uid = parts[1], int(parts[2])

    if action == "approve":
        async with SessionLocal() as session:
            pending = await session.get(PendingUser, uid)
            uname = pending.username if pending else None
            if pending:
                await session.delete(pending)
            sub = await session.get(Subscriber, uid)
            if sub is None:
                session.add(Subscriber(chat_id=uid, active=True, notifications_enabled=True, username=uname))
            else:
                sub.active = True
            await session.commit()
        await cb.answer("✅ Одобрен!", show_alert=True)
        await cb.message.edit_reply_markup(reply_markup=None)
        await cb.message.answer(f"✅ Пользователь <code>{uid}</code> одобрен.", parse_mode="HTML")
        try:
            await cb.bot.send_message(
                uid,
                "✅ <b>Доступ одобрен!</b>\n\n"
                "Нажмите /start чтобы начать пользоваться ботом.",
                parse_mode="HTML",
            )
        except Exception:
            pass

    elif action == "deny":
        async with SessionLocal() as session:
            pending = await session.get(PendingUser, uid)
            if pending:
                await session.delete(pending)
            await session.commit()
        await cb.answer("❌ Отклонён", show_alert=True)
        await cb.message.edit_reply_markup(reply_markup=None)
        await cb.message.answer(f"❌ Лид <code>{uid}</code> удалён.", parse_mode="HTML")
        try:
            await cb.bot.send_message(
                uid,
                "❌ Ваша заявка отклонена.\n"
                "Обратитесь к администратору если считаете это ошибкой.",
            )
        except Exception:
            pass


# ─── User remove ──────────────────────────────────────────────────────────────

@router.callback_query(lambda c: c.data and c.data.startswith("user:remove:"))
async def cb_user_remove(cb: CallbackQuery):
    if not is_admin(cb.from_user.id):
        await cb.answer("Нет доступа")
        return
    uid = int(cb.data.split(":")[-1])
    async with SessionLocal() as session:
        sub = await session.get(Subscriber, uid)
        if sub:
            sub.active = False
            await session.commit()
    await cb.answer("🗑 Удалён", show_alert=True)
    await cb.message.edit_reply_markup(reply_markup=None)
    await cb.message.answer(f"🗑 Доступ для <code>{uid}</code> закрыт.", parse_mode="HTML")
    try:
        await cb.bot.send_message(uid, "⛔ Ваш доступ к боту закрыт.")
    except Exception:
        pass


# ─── FSM: Add user by ID ──────────────────────────────────────────────────────

@router.message(AdminStates.waiting_user_id)
async def fsm_add_user(msg: Message, state: FSMContext):
    if not is_admin(msg.from_user.id):
        return
    text = msg.text.strip() if msg.text else ""
    if text == "/cancel":
        await state.clear()
        await msg.answer("Отменено.")
        return
    try:
        uid = int(text)
    except ValueError:
        await msg.answer("❌ Неверный формат. Введите числовой Telegram ID или /cancel.")
        return
    await state.clear()
    async with SessionLocal() as session:
        sub = await session.get(Subscriber, uid)
        if sub is None:
            session.add(Subscriber(chat_id=uid, active=True, notifications_enabled=True))
        else:
            sub.active = True
        await session.commit()
    await msg.answer(f"✅ Пользователь <code>{uid}</code> добавлен.", parse_mode="HTML")
    try:
        await msg.bot.send_message(
            uid,
            "✅ <b>Доступ одобрен!</b>\n\nНажмите /start чтобы начать.",
            parse_mode="HTML",
        )
    except Exception:
        pass


# /allow and /deny kept as text commands for backwards-compat
@router.message(Command("allow"))
async def cmd_allow(msg: Message):
    if not is_admin(msg.from_user.id):
        return
    parts = msg.text.split()
    if len(parts) < 2:
        await msg.answer("Использование: /allow <user_id>\nИли используй /admin → Лиды")
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
        await session.commit()
    await msg.answer(f"✅ Доступ предоставлен: <code>{uid}</code>", parse_mode="HTML")
    try:
        await msg.bot.send_message(uid, "✅ <b>Доступ одобрен!</b>\n\nНажмите /start.", parse_mode="HTML")
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
    await msg.answer(f"🚫 Доступ закрыт: <code>{uid}</code>", parse_mode="HTML")


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
    """Утренний дайджест = расписание сегодняшних игр с прогнозом модели.

    НЕ показывает сигналы (их ещё нет — они придут за 5ч до игры).
    Показывает ВСЕ сегодняшние игры с питчерами и вероятностями из ML-модели.
    """
    from src.ml.predict import Predictor
    from src.data.features import build_inference_features

    now_utc = datetime.utcnow()
    msk = timezone(timedelta(hours=3))

    # Конец сегодняшнего MSK-дня = 21:00 UTC (= 00:00 МСК следующего дня)
    msk_day_end_utc = datetime.combine(now_utc.date(), datetime.min.time()) + timedelta(hours=21)
    if now_utc >= msk_day_end_utc:
        msk_day_end_utc += timedelta(days=1)

    async with SessionLocal() as session:
        # Все сегодняшние игры которые ещё не начались
        today_matches = (await session.execute(
            select(Match).where(
                Match.utc_date > now_utc,
                Match.utc_date <= msk_day_end_utc,
                Match.status != "FINISHED",
            ).order_by(Match.utc_date.asc())
        )).scalars().all()

        if not today_matches:
            return

        teams_cache: dict[int, Team] = {}
        for m in today_matches:
            for tid in (m.home_team_id, m.away_team_id):
                if tid not in teams_cache:
                    t = await session.get(Team, tid)
                    if t:
                        teams_cache[tid] = t

    if not today_matches:
        return

    msk_now_str = now_utc.replace(tzinfo=timezone.utc).astimezone(msk).strftime("%d.%m")
    lines = [f"☀️ <b>Расписание MLB на {msk_now_str}</b> — {len(today_matches)} игр\n"]

    predictor = Predictor()

    for m in today_matches:
        home_t = teams_cache.get(m.home_team_id)
        away_t = teams_cache.get(m.away_team_id)
        home_name = home_t.name if home_t else f"Team#{m.home_team_id}"
        away_name = away_t.name if away_t else f"Team#{m.away_team_id}"

        # Аббревиатура команды (последние 3 буквы или первые 3 слова)
        def _abbr(name: str) -> str:
            parts = name.split()
            return parts[-1][:3].upper() if parts else name[:3].upper()

        home_abbr = _abbr(home_name)
        away_abbr = _abbr(away_name)

        game_msk = m.utc_date.replace(tzinfo=timezone.utc).astimezone(msk).strftime("%H:%M")
        signal_msk = (m.utc_date - timedelta(hours=5)).replace(tzinfo=timezone.utc).astimezone(msk).strftime("%H:%M")

        # Питчеры
        hp = f"{m.home_pitcher_name or '?'} ERA {m.home_pitcher_era or '?'}"
        ap = f"{m.away_pitcher_name or '?'} ERA {m.away_pitcher_era or '?'}"

        # ML-вероятности если модель готова
        prob_str = ""
        if predictor.ready:
            try:
                import pandas as pd
                from src.data.database import SessionLocal as SL
                # Быстрый прогноз через predict
                row_df = pd.DataFrame([{
                    "match_id": m.id,
                    "home_team_id": m.home_team_id,
                    "away_team_id": m.away_team_id,
                    "home_pitcher_era": m.home_pitcher_era or 4.5,
                    "home_pitcher_whip": m.home_pitcher_whip or 1.3,
                    "home_pitcher_k9": m.home_pitcher_k9 or 8.0,
                    "home_pitcher_bb9": m.home_pitcher_bb9 or 3.0,
                    "away_pitcher_era": m.away_pitcher_era or 4.5,
                    "away_pitcher_whip": m.away_pitcher_whip or 1.3,
                    "away_pitcher_k9": m.away_pitcher_k9 or 8.0,
                    "away_pitcher_bb9": m.away_pitcher_bb9 or 3.0,
                    "era_diff": (m.home_pitcher_era or 4.5) - (m.away_pitcher_era or 4.5),
                    "home_win_rate": 0.5, "away_win_rate": 0.5,
                    "home_run_rate": 4.5, "away_run_rate": 4.5,
                    "h2h_home_win_rate": 0.5,
                }])
                preds = predictor.predict(row_df)
                if not preds.empty:
                    p_h = float(preds.iloc[0].get("p_home", 0.5))
                    p_a = float(preds.iloc[0].get("p_away", 0.5))
                    fav = home_abbr if p_h >= p_a else away_abbr
                    fav_p = max(p_h, p_a)
                    prob_str = f" · {fav} {fav_p:.0%}"
            except Exception:
                pass

        # Уже есть сигнал?
        signal_tag = ""
        async with SessionLocal() as sess:
            existing_sig = (await sess.execute(
                select(Signal).where(Signal.match_id == m.id)
            )).scalar_one_or_none()
        if existing_sig:
            from src.bot.formatters import MARKET_LABELS, PICK_LABELS
            mkt = MARKET_LABELS.get(existing_sig.market, existing_sig.market)
            pick = PICK_LABELS.get(existing_sig.pick, existing_sig.pick)
            if existing_sig.book_odds and existing_sig.book_odds > 1.0:
                signal_tag = f"\n   ✅ Сигнал: {mkt} {pick} @ {existing_sig.book_odds:.2f}"
            else:
                signal_tag = f"\n   ⏰ Сигнал в ~{signal_msk} МСК"
        else:
            signal_tag = f"\n   ⏰ Сигнал в ~{signal_msk} МСК"

        lines.append(
            f"⚾ <b>{home_abbr}–{away_abbr}</b> {game_msk} МСК\n"
            f"   🎯 {hp} / {ap}{prob_str}"
            f"{signal_tag}"
        )

    await broadcast_signal(bot, "\n\n".join(lines))


# ─── Admin: check why signals are missing ─────────────────────────────────────

@router.message(Command("check_signals"))
async def cmd_check_signals(msg: Message):
    """Diagnostic: show upcoming games with model probs, odds, and signal status."""
    if not is_admin(msg.from_user.id):
        return
    await msg.answer("🔍 Диагностика сигналов...")

    from src.ml.predict import Predictor
    from src.data.odds_api import OddsApiClient, fetch_odds_for_matches
    from src.signals.generator import _book_odds_for, _kelly

    now_utc = datetime.utcnow()
    horizon = now_utc + timedelta(hours=12)

    async with SessionLocal() as session:
        upcoming = (await session.execute(
            select(Match).where(
                Match.status != "FINISHED",
                Match.utc_date >= now_utc,
                Match.utc_date <= horizon,
            ).order_by(Match.utc_date.asc())
        )).scalars().all()

        teams_cache: dict[int, Team] = {}
        for m in upcoming:
            for tid in (m.home_team_id, m.away_team_id):
                if tid not in teams_cache:
                    t = await session.get(Team, tid)
                    if t:
                        teams_cache[tid] = t

        # Already signaled?
        signaled = set((await session.execute(
            select(Signal.match_id).where(
                Signal.match_id.in_([m.id for m in upcoming])
            )
        )).scalars().all())

    if not upcoming:
        await msg.answer("❌ Нет игр в ближайшие 12 часов в БД")
        return

    msk = timezone(timedelta(hours=3))
    predictor = Predictor()

    # Fetch odds
    odds_map: dict = {}
    if settings.odds_api_key:
        client = OddsApiClient(settings.odds_api_key)
        try:
            tuples = []
            async with SessionLocal() as session:
                for m in upcoming:
                    ht = teams_cache.get(m.home_team_id)
                    at = teams_cache.get(m.away_team_id)
                    if ht and at:
                        tuples.append((m.id, m.competition, ht.name, at.name, m.utc_date))
            odds_map = await fetch_odds_for_matches(client, tuples)
        finally:
            await client.close()

    lines = [f"🔍 <b>Диагностика — {len(upcoming)} игр (ближайшие 12ч)</b>\n"]

    import pandas as pd

    for m in upcoming:
        ht = teams_cache.get(m.home_team_id)
        at = teams_cache.get(m.away_team_id)
        hn = ht.name.split()[-1] if ht else "?"
        an = at.name.split()[-1] if at else "?"
        game_msk = m.utc_date.replace(tzinfo=timezone.utc).astimezone(msk).strftime("%H:%M")

        already = "✅ сигнал есть" if m.id in signaled else ""

        # Model probs
        p_home = p_away = 0.5
        p_over = 0.5
        if predictor.ready:
            try:
                row_df = pd.DataFrame([{
                    "match_id": m.id,
                    "home_team_id": m.home_team_id, "away_team_id": m.away_team_id,
                    "home_pitcher_era": m.home_pitcher_era or 4.5,
                    "home_pitcher_whip": m.home_pitcher_whip or 1.3,
                    "home_pitcher_k9": m.home_pitcher_k9 or 8.0,
                    "home_pitcher_bb9": m.home_pitcher_bb9 or 3.0,
                    "away_pitcher_era": m.away_pitcher_era or 4.5,
                    "away_pitcher_whip": m.away_pitcher_whip or 1.3,
                    "away_pitcher_k9": m.away_pitcher_k9 or 8.0,
                    "away_pitcher_bb9": m.away_pitcher_bb9 or 3.0,
                    "era_diff": (m.home_pitcher_era or 4.5) - (m.away_pitcher_era or 4.5),
                    "home_win_rate": 0.5, "away_win_rate": 0.5,
                    "home_run_rate": 4.5, "away_run_rate": 4.5,
                    "h2h_home_win_rate": 0.5,
                }])
                preds = predictor.predict(row_df)
                if not preds.empty:
                    p_home = float(preds.iloc[0].get("p_home", 0.5))
                    p_away = float(preds.iloc[0].get("p_away", 0.5))
                    p_over = float(preds.iloc[0].get("p_over85", 0.5))
            except Exception as e:
                p_home = p_away = p_over = -1.0

        # Odds
        odds = odds_map.get(m.id) or {}
        ml_h = odds.get("odds_ml_home")
        ml_a = odds.get("odds_ml_away")
        tot_o = odds.get("odds_over85")
        tot_u = odds.get("odds_under85")

        # Check edge
        signals_possible = []
        if ml_h and p_home > 0:
            e = p_home * ml_h - 1.0
            flag = "✅" if e >= settings.min_edge and p_home >= settings.min_confidence else "❌"
            signals_possible.append(f"ML HOME {flag} p={p_home:.0%} @{ml_h:.2f} edge={e:+.1%}")
        if ml_a and p_away > 0:
            e = p_away * ml_a - 1.0
            flag = "✅" if e >= settings.min_edge and p_away >= settings.min_confidence else "❌"
            signals_possible.append(f"ML AWAY {flag} p={p_away:.0%} @{ml_a:.2f} edge={e:+.1%}")
        if tot_o and p_over > 0:
            e = p_over * tot_o - 1.0
            flag = "✅" if e >= settings.min_edge and p_over >= settings.min_confidence else "❌"
            signals_possible.append(f"TOT Б {flag} p={p_over:.0%} @{tot_o:.2f} edge={e:+.1%}")
        if tot_u and p_over > 0:
            p_u = 1.0 - p_over
            e = p_u * tot_u - 1.0
            flag = "✅" if e >= settings.min_edge and p_u >= settings.min_confidence else "❌"
            signals_possible.append(f"TOT М {flag} p={p_u:.0%} @{tot_u:.2f} edge={e:+.1%}")

        odds_status = "📊 Кэфы есть" if odds else "⚠️ НЕТ КЭФОВ от Odds API"
        model_status = "" if predictor.ready else "⚠️ модель не готова"

        block = [f"⚾ <b>{hn}–{an}</b> {game_msk} МСК {already}"]
        block.append(f"   {odds_status} {model_status}")
        if p_home > 0:
            block.append(f"   🧠 P(home)={p_home:.0%} P(away)={p_away:.0%} P(over)={p_over:.0%}")
        for s in signals_possible:
            block.append(f"   {s}")
        if not signals_possible and not odds:
            block.append(f"   → Нет кэфов = нет сигнала")
        elif not any("✅" in s for s in signals_possible):
            block.append(f"   → edge<{settings.min_edge:.0%} или prob<{settings.min_confidence:.0%}")

        lines.append("\n".join(block))

    lines.append(f"\n⚙️ Пороги: min_confidence={settings.min_confidence:.0%} min_edge={settings.min_edge:.0%}")

    # Split into chunks if too long
    full_text = "\n\n".join(lines)
    chunk_size = 3800
    for i in range(0, len(full_text), chunk_size):
        await msg.answer(full_text[i:i+chunk_size])



@router.message(Command("digest_now"))
async def cmd_digest_now(msg: Message):
    if not is_admin(msg.from_user.id):
        return
    await msg.answer("📤 Запускаю рассылку дайджеста всем подписчикам...")
    await broadcast_morning_digest(msg.bot)
    await msg.answer("✅ Дайджест отправлен.")


@router.message(Command("test_odds"))
async def cmd_test_odds(msg: Message):
    """Direct Odds API connectivity test — bypasses cache."""
    if not is_admin(msg.from_user.id):
        return

    import aiohttp
    from src.data.odds_api import BASE_URL, SPORT, REGIONS, MARKETS

    key = settings.odds_api_key
    if not key:
        await msg.answer("❌ ODDS_API_KEY не задан в переменных Railway!\nДобавь Variables → ODDS_API_KEY=твой_ключ")
        return

    await msg.answer(f"🔑 Ключ задан: ...{key[-6:]}\n⏳ Делаю прямой запрос к API (без кэша)...")

    params = {
        "apiKey": key,
        "regions": REGIONS,
        "markets": MARKETS,
        "dateFormat": "iso",
        "oddsFormat": "decimal",
    }
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{BASE_URL}/sports/{SPORT}/odds"
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as r:
                remaining = r.headers.get("x-requests-remaining", "?")
                used = r.headers.get("x-requests-used", "?")
                status = r.status

                if status == 401:
                    await msg.answer("❌ HTTP 401 — ключ неверный или просрочен\nПроверь ключ на https://the-odds-api.com/account/")
                    return
                if status == 422:
                    body = await r.text()
                    await msg.answer(f"❌ HTTP 422 — неверный параметр запроса\n<code>{body[:300]}</code>")
                    return
                if status == 429:
                    await msg.answer("❌ HTTP 429 — превышен лимит запросов (500/месяц на бесплатном тарифе)\nКупи план или жди следующего месяца")
                    return
                if status >= 400:
                    body = await r.text()
                    await msg.answer(f"❌ HTTP {status}\n<code>{body[:300]}</code>")
                    return

                events = await r.json()
                n = len(events) if isinstance(events, list) else 0

                lines = [
                    f"✅ <b>Odds API работает</b>",
                    f"📊 Использовано запросов: {used}",
                    f"📊 Осталось запросов: {remaining}",
                    f"⚾ Событий в ответе: {n}",
                ]
                if n == 0:
                    lines.append("⚠️ API вернул 0 событий — возможно нет активных рынков")
                else:
                    # Show first 3 events
                    lines.append("\n<b>Первые события из ответа:</b>")
                    for ev in events[:3]:
                        ht = ev.get("home_team", "?")
                        at = ev.get("away_team", "?")
                        ct = ev.get("commence_time", "?")
                        bk_count = len(ev.get("bookmakers", []))
                        lines.append(f"• {ht} vs {at} | {ct} | {bk_count} букмекеров")
                    if n > 3:
                        lines.append(f"... и ещё {n-3} событий")

                await msg.answer("\n".join(lines))

    except Exception as e:
        await msg.answer(f"❌ Ошибка соединения: <code>{e}</code>")


# ─── Catch-all ────────────────────────────────────────────────────────────────

@router.message()
async def catch_all(msg: Message):
    await msg.answer("Выберите действие:", reply_markup=main_menu())
