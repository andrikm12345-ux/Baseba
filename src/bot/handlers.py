"""Telegram command handlers for the MLB Baseball bot."""
from __future__ import annotations

import csv
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
from sqlalchemy import select

from src.bot.formatters import (
    HELP_TEXT,
    MARKET_LABELS,
    PICK_LABELS,
    WELCOME_TEXT,
    format_roi_stats,
    format_signal,
    format_signal_short,
    format_training_report,
)
from src.bot.keyboards import admin_menu, main_menu, matches_kb, notifications_kb, signals_filter_kb
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
    await msg.answer("Фильтр сигналов:", reply_markup=signals_filter_kb())


@router.callback_query(lambda c: c.data and c.data.startswith("filter:"))
async def cb_filter(cb: CallbackQuery):
    flt = cb.data.split(":", 1)[1]
    horizon = datetime.utcnow() - timedelta(days=7)
    async with SessionLocal() as session:
        q = select(Signal).join(Match).where(
            Signal.created_at >= horizon,
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
            await cb.message.edit_text(
                "Нет сигналов за последние 7 дней.\n\n"
                "Нажмите <b>🔄 Обновить коэффы</b> чтобы запустить генерацию.",
                parse_mode="HTML",
            )
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
@router.message(lambda m: m.text == "📅 Матчи сегодня")
async def cmd_today(msg: Message):
    # Use MSK date boundaries: MSK midnight = UTC -3h = yesterday 21:00 UTC
    msk = timezone(timedelta(hours=3))
    now_msk = datetime.now(msk)
    msk_today_start = now_msk.replace(hour=0, minute=0, second=0, microsecond=0)
    # Show from MSK yesterday 06:00 to cover last night's games too
    start = (msk_today_start - timedelta(hours=18)).astimezone(timezone.utc).replace(tzinfo=None)
    end = (msk_today_start + timedelta(days=1)).astimezone(timezone.utc).replace(tzinfo=None)
    async with SessionLocal() as session:
        rows = (await session.execute(
            select(Match).where(Match.utc_date >= start, Match.utc_date < end).order_by(Match.utc_date).limit(20)
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
            dt_str = _msk(m.utc_date)
            if m.status == "FINISHED":
                status = f" ✅ {m.home_runs}:{m.away_runs}"
            else:
                status = " 🕐"
            lines.append(f"⚾ {h} vs {a}  {dt_str}{status}")
    await msg.answer("\n".join(lines), parse_mode="HTML")


@router.message(Command("stats"))
@router.message(lambda m: m.text == "📊 Статистика")
async def cmd_stats(msg: Message):
    model_s = await roi_stats(last_n=200, only_value=False)   # 📊 MODEL (без коэффа БК)
    value_s = await roi_stats(last_n=200, only_value=True)    # 🔥 VALUE (с edge)
    ai_s = await roi_stats(last_n=200, ai_only=True, only_value=None)  # 🤖 AI ансамбль
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

@router.message(lambda m: m.text == "📋 Анализ матча")
async def cmd_match_analysis(msg: Message):
    today = datetime.utcnow().date()
    start = datetime.combine(today, datetime.min.time())
    end = start + timedelta(days=2)
    async with SessionLocal() as session:
        rows = (await session.execute(
            select(Match).where(
                Match.utc_date >= start,
                Match.utc_date < end,
                Match.status != "FINISHED",
            ).order_by(Match.utc_date).limit(15)
        )).scalars().all()
        if not rows:
            await msg.answer("Нет предстоящих игр MLB на сегодня/завтра.")
            return
        items = []
        for m in rows:
            home = await session.get(Team, m.home_team_id)
            away = await session.get(Team, m.away_team_id)
            h = (home.short_name or home.name)[:10] if home else "?"
            a = (away.short_name or away.name)[:10] if away else "?"
            items.append((m.id, f"⚾ {h} vs {a} | {_msk(m.utc_date)}"))
    await msg.answer("Выберите матч для анализа:", reply_markup=matches_kb(items))


@router.callback_query(lambda c: c.data and c.data.startswith("match_info:"))
async def cb_match_info(cb: CallbackQuery):
    mid = int(cb.data.split(":", 1)[1])
    async with SessionLocal() as session:
        match = await session.get(Match, mid)
        if not match:
            await cb.answer("Матч не найден")
            return
        home = await session.get(Team, match.home_team_id)
        away = await session.get(Team, match.away_team_id)
        ai_pred = await session.get(AiPrediction, mid)
        sigs: List[Signal] = list((await session.execute(
            select(Signal).where(Signal.match_id == mid)
        )).scalars())

    h_name = home.name if home else "?"
    a_name = away.name if away else "?"
    lines = [
        f"⚾ <b>{h_name} vs {a_name}</b>",
        f"📅 {_msk(match.utc_date)}\n",
    ]
    if match.home_pitcher_name:
        era = f"{match.home_pitcher_era:.2f}" if match.home_pitcher_era else "н/д"
        lines.append(f"🏠 {match.home_pitcher_name} ERA {era}")
    if match.away_pitcher_name:
        era = f"{match.away_pitcher_era:.2f}" if match.away_pitcher_era else "н/д"
        lines.append(f"✈️ {match.away_pitcher_name} ERA {era}")
    if sigs:
        lines.append(f"\n📊 Сигналы ({len(sigs)}):")
        for s in sigs:
            m_l = MARKET_LABELS.get(s.market, s.market)
            p_l = PICK_LABELS.get(s.pick, s.pick)
            edge_str = f" edge {s.edge:.1%}" if s.edge else ""
            lines.append(f"  • {m_l} → <b>{p_l}</b> ({s.confidence:.0%}){edge_str}")
    if ai_pred:
        try:
            reasoning = json.loads(ai_pred.payload).get("reasoning", "")
            if reasoning:
                lines.append(f"\n🤖 <i>{reasoning}</i>")
        except Exception:
            pass
    await cb.message.answer("\n".join(lines), parse_mode="HTML")
    await cb.answer()


@router.message(lambda m: m.text == "💰 Расчёт Kelly")
async def cmd_kelly(msg: Message):
    async with SessionLocal() as session:
        since = datetime.utcnow() - timedelta(days=7)
        sigs: List[Signal] = list((await session.execute(
            select(Signal).where(
                Signal.created_at >= since,
                Signal.book_odds > 1.0,
                Signal.settled.is_(False),
            ).order_by(Signal.created_at.desc()).limit(15)
        )).scalars())
    if not sigs:
        await msg.answer(
            "Нет активных VALUE-сигналов за 7 дней.\n\n"
            "VALUE появляются когда задан <b>ODDS_API_KEY</b> и найден edge ≥5%.",
            parse_mode="HTML",
        )
        return
    lines = ["💰 <b>Активные VALUE-сигналы (Kelly)</b>\n"]
    for s in sigs:
        m_l = MARKET_LABELS.get(s.market, s.market)
        p_l = PICK_LABELS.get(s.pick, s.pick)
        lines.append(
            f"• {m_l} <b>{p_l}</b> @ {s.book_odds:.2f} "
            f"| edge {s.edge:.1%} | {s.stake_units:.2f} ед."
        )
    await msg.answer("\n".join(lines), parse_mode="HTML")


@router.message(lambda m: m.text == "🔄 Обновить коэффы")
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
            # Ближайшая незавершённая игра
            now_utc = datetime.utcnow()
            next_game = (await session.execute(
                select(Match).where(
                    Match.status != "FINISHED",
                    Match.utc_date > now_utc,
                ).order_by(Match.utc_date).limit(1)
            )).scalar_one_or_none()

        model_status = "✅ готовы" if models_ready else "⏳ ещё обучаются"
        next_signal_hint = ""
        if next_game:
            signal_time = next_game.utc_date - timedelta(hours=3)
            msk_time = (signal_time.replace(tzinfo=timezone.utc)
                        .astimezone(timezone(timedelta(hours=3)))).strftime("%H:%M МСК")
            game_msk = (next_game.utc_date.replace(tzinfo=timezone.utc)
                        .astimezone(timezone(timedelta(hours=3)))).strftime("%d.%m %H:%M МСК")
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
                f"📭 Новых сигналов нет — нет игр в окне ±3 ч от текущего времени."
                f"{next_signal_hint}\n\n"
                "Сигналы появятся автоматически за 3 часа до начала игры.",
                reply_markup=main_menu(),
            )
        else:
            await msg.answer(f"✅ Отправлено {new_count} новых сигналов.", reply_markup=main_menu())
    except Exception as e:
        logger.error(f"refresh error: {e}")
        await msg.answer(f"❌ Ошибка: {e}")


@router.message(lambda m: m.text == "📥 Скачать CSV")
async def cmd_download_csv(msg: Message):
    async with SessionLocal() as session:
        since = datetime.utcnow() - timedelta(days=30)
        sigs: List[Signal] = list((await session.execute(
            select(Signal).where(Signal.created_at >= since).order_by(Signal.created_at.desc())
        )).scalars())
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Дата", "Матч", "Рынок", "Пик", "Вер-сть", "Коэф", "Edge", "Стейк", "Закрыт", "P/L"])
        for s in sigs:
            match = await session.get(Match, s.match_id)
            if match:
                home = await session.get(Team, match.home_team_id)
                away = await session.get(Team, match.away_team_id)
                mn = f"{home.name if home else '?'} vs {away.name if away else '?'}"
            else:
                mn = "?"
            writer.writerow([
                s.created_at.strftime("%Y-%m-%d %H:%M"), mn,
                s.market, s.pick,
                f"{s.model_prob:.3f}",
                f"{s.book_odds:.2f}" if s.book_odds else "-",
                f"{s.edge:.3f}" if s.edge else "-",
                f"{s.stake_units:.2f}",
                "Да" if s.settled else "Нет",
                f"{s.profit_units:+.2f}" if s.profit_units is not None else "-",
            ])
    content = buf.getvalue().encode("utf-8-sig")
    await msg.answer_document(
        BufferedInputFile(content, filename=f"signals_{datetime.utcnow().strftime('%Y%m%d')}.csv"),
        caption=f"Сигналы за 30 дней: {len(sigs)} записей",
    )


# ─── Admin commands ───────────────────────────────────────────────────────────

@router.message(Command("admin"))
async def cmd_admin(msg: Message):
    if not is_admin(msg.from_user.id):
        return
    from src.data.settings_store import get_bool
    ai_on = await get_bool("ai_ensemble_enabled", False)
    await msg.answer(
        f"🔧 Админ-панель\n🤖 AI Ансамбль: {'✅ включён' if ai_on else '❌ выключен'}",
        reply_markup=admin_menu(),
    )


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
        await cb.answer("Запускаю обучение...")
        await cb.message.answer("🔄 Обучение моделей запущено, подождите...")
        try:
            from src.pipeline import train_models
            await train_models(bot=cb.bot)
        except Exception as e:
            await cb.message.answer(f"❌ Ошибка обучения: {e}")
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


# ─── Catch-all ────────────────────────────────────────────────────────────────

@router.message()
async def catch_all(msg: Message):
    await msg.answer("Выберите действие:", reply_markup=main_menu())
