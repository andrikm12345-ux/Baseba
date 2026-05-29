"""Message formatters for the MLB Baseball Signals bot (Russian)."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

from src.config import settings
from src.data.database import Match, Signal, Team

MSK = timezone(timedelta(hours=3))

MARKET_LABELS: dict[str, str] = {
    "ML": "Мани-лайн",
    "TOTAL": "Тотал",
    "RL": "Ран-лайн",
}

PICK_LABELS: dict[str, str] = {
    "HOME": "П1",
    "AWAY": "П2",
    "OVER": "Тотал Б",
    "UNDER": "Тотал М",
    "COVER": f"−{settings.rl_line}",
    "AWAY_COVER": f"−{settings.rl_line}",
}


def _msk(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(MSK).strftime("%d.%m %H:%M МСК")


def _fmt_pitcher_line(match: Match) -> Optional[str]:
    """Returns a one-line pitcher matchup string if data is available."""
    hp = match.home_pitcher_name
    ap = match.away_pitcher_name
    if not hp and not ap:
        return None
    parts = []
    if hp:
        era = f" ERA {match.home_pitcher_era:.2f}" if match.home_pitcher_era else ""
        whip = f" WHIP {match.home_pitcher_whip:.2f}" if match.home_pitcher_whip else ""
        parts.append(f"{hp}{era}{whip}")
    else:
        parts.append("неизвестен")
    if ap:
        era = f" ERA {match.away_pitcher_era:.2f}" if match.away_pitcher_era else ""
        whip = f" WHIP {match.away_pitcher_whip:.2f}" if match.away_pitcher_whip else ""
        parts.append(f"{ap}{era}{whip}")
    else:
        parts.append("неизвестен")
    return f"⚾ Питчеры: {parts[0]} / {parts[1]}"


def format_signal(
    signal: Signal,
    match: Match,
    home: Optional[Team],
    away: Optional[Team],
    ai_comment: Optional[str] = None,
) -> str:
    home_name = home.name if home else "Хозяева"
    away_name = away.name if away else "Гости"
    kickoff = _msk(match.utc_date)
    market_label = MARKET_LABELS.get(signal.market, signal.market)

    ln = getattr(signal, "line", None)
    if signal.market == "TOTAL":
        tl = ln if ln is not None else settings.total_line
        market_label = f"Тотал ({tl:g})"
        pick_label = f"Тотал Б {tl:g}" if signal.pick == "OVER" else f"Тотал М {tl:g}"
    elif signal.market == "RL":
        rl = ln if ln is not None else settings.rl_line
        market_label = f"Ран-лайн (±{rl:g})"
        if signal.pick == "AWAY_COVER":
            pick_label = f"{away_name} −{rl:g}"
        else:  # COVER
            pick_label = f"{home_name} −{rl:g}"
    else:
        pick_label = PICK_LABELS.get(signal.pick, signal.pick)

    badge = "🔥 <b>VALUE</b>" if signal.is_value else "📊 <b>MODEL</b>"
    ai_badge = " 🤖" if signal.is_ai_ensemble else ""

    lines = [
        f"{badge}{ai_badge}",
        f"",
        f"⚾ <b>{home_name}</b> vs <b>{away_name}</b>",
        f"🕐 {kickoff}  |  MLB",
    ]
    pitcher_line = _fmt_pitcher_line(match)
    if pitcher_line:
        lines.append(pitcher_line)
    lines += [
        f"",
        f"<b>Рынок:</b> {market_label}",
        f"<b>Прогноз:</b> {pick_label}",
        f"<b>Уверенность:</b> {signal.confidence:.0%}",
        f"<b>Честный коэф.:</b> {signal.fair_odds:.2f}",
    ]
    if signal.book_odds > 1.0:
        lines.append(f"<b>Коэф. БК:</b> {signal.book_odds:.2f}")
    if signal.is_value and signal.edge > 0:
        lines += [
            f"<b>Edge:</b> +{signal.edge:.1%}",
            f"<b>Ставка:</b> {signal.stake_units:.2f} ед.",
        ]
    return "\n".join(lines)


def format_roi_stats(model_stats, value_stats, ai_stats) -> str:
    def _row(label: str, s) -> str:
        if s.n_settled == 0:
            return f"<b>{label}:</b> нет данных"
        sign = "+" if s.profit >= 0 else ""
        return (
            f"<b>{label}:</b> {s.n_settled} игр | "
            f"Попадания {s.hit_rate:.0f}% | "
            f"ROI {sign}{s.roi:.1f}% | "
            f"{sign}{s.profit:.1f} ед."
        )

    all_empty = model_stats.n_settled == 0 and value_stats.n_settled == 0 and ai_stats.n_settled == 0
    lines = [
        "📊 <b>Статистика сигналов</b>",
        "",
        _row("📊 MODEL (без коэффа)", model_stats),
        _row("🔥 VALUE (с edge)", value_stats),
        _row("🤖 AI-ансамбль", ai_stats),
    ]
    if all_empty:
        lines += [
            "",
            "<i>ℹ️ Статистика накапливается по мере работы бота.</i>",
            "<i>Данные появятся после того, как первые сигналы сыграют.</i>",
            "<i>Сигналы генерируются за 3 ч до игры, закрываются автоматически.</i>",
        ]
    return "\n".join(lines)


_MONTHS_RU = {
    1: "января", 2: "февраля", 3: "марта", 4: "апреля",
    5: "мая", 6: "июня", 7: "июля", 8: "августа",
    9: "сентября", 10: "октября", 11: "ноября", 12: "декабря",
}


def format_history(day_groups: list) -> str:
    """day_groups: список (day_label, [item_dict])
    item_dict: {home_abbr, away_abbr, home_runs, away_runs, market, pick,
                book_odds, profit, settled, won}
    """
    if not day_groups:
        return "📜 <b>История ставок пуста.</b>\n\nСигналы появятся после первых завершённых игр."

    lines = ["📜 <b>История ставок</b>\n"]
    for day_label, items in day_groups:
        settled = [x for x in items if x["settled"]]
        won_cnt = sum(1 for x in settled if x["won"])
        lost_cnt = sum(1 for x in settled if not x["won"])
        pending_cnt = sum(1 for x in items if not x["settled"])
        day_profit = sum(x.get("profit", 0.0) or 0.0 for x in settled)

        profit_str = f"{'+' if day_profit >= 0 else ''}{day_profit:.2f} ед."
        stats_parts = []
        if won_cnt or lost_cnt:
            stats_parts.append(f"✅ {won_cnt}  ❌ {lost_cnt}")
        if pending_cnt:
            stats_parts.append(f"⏳ {pending_cnt}")
        if settled:
            stats_parts.append(profit_str)
        stats = "  |  ".join(stats_parts)

        lines.append(f"\n<b>📅 {day_label}</b>  {stats}")

        for it in items:
            icon = "⏳" if not it["settled"] else ("✅" if it["won"] else "❌")
            h, a = it["home_abbr"], it["away_abbr"]
            hr, ar = it.get("home_runs"), it.get("away_runs")
            score = f"{h} {hr}:{ar} {a}" if (hr is not None and ar is not None) else f"{h} vs {a}"

            m_short = MARKET_LABELS.get(it["market"], it["market"])
            p_short = PICK_LABELS.get(it["pick"], it["pick"])
            odds = it.get("book_odds") or 0.0

            if it["settled"]:
                p = it.get("profit") or 0.0
                p_str = f"{'+' if p >= 0 else ''}{p:.2f} ед."
                if odds > 1.0:
                    row = f"{icon} {score} | {m_short} {p_short} @ {odds:.2f} → {p_str}"
                else:
                    row = f"{icon} {score} | {m_short} {p_short} → {p_str}"
            else:
                row = f"{icon} {score} | {m_short} {p_short}" + (f" @ {odds:.2f}" if odds > 1.0 else "")
            lines.append(row)

    return "\n".join(lines)


WELCOME_TEXT = """⚾ <b>Baseball Signals — MLB</b>

Пока другие продают «инсайды» — мы считаем.
Пока другие постят скрины — мы ведём учёт каждой ставки.

━━━━━━━━━━━━━━━━

🔍 <b>Мы находим то, что букмекер недооценил.</b>
Каждая игра MLB анализируется по десяткам параметров — форма команд, питчеры, усталость буллпена, домашнее поле, погода. Всё в одном сигнале.

📐 <b>Три рынка. Только с перевесом.</b>
Победитель · Тотал ранов · Фора ±1.5
Сигнал выходит когда математика на нашей стороне — иначе тишина.

💼 <b>Банкролл под защитой.</b>
Размер каждой ставки рассчитывается автоматически.
Никаких «ставь всё» — строгое управление капиталом.

━━━━━━━━━━━━━━━━

Сигналы приходят <b>за 3 часа до первого питча.</b>
Результаты фиксируются автоматически — ничего не скрываем.

👇 Нажми <b>⚾ Сигналы</b> и посмотри что работает прямо сейчас."""

HELP_TEXT = """⚾ <b>Команды бота</b>

/signals — активные сигналы (последние 3 дня)
/today — игры MLB сегодня
/stats — статистика ROI и попаданий
/chart — кривая ROI по времени
/help — эта справка

<b>Рынки:</b>
• <b>ML</b> — победитель (Мани-лайн). П1=хозяева, П2=гости.
• <b>TOTAL</b> — сумма ранов. Б=больше 8.5, М=меньше 8.5.
• <b>RL</b> — фора. −1.5=хозяева выигрывают с разницей 2+, +1.5=гости.

<b>Типы сигналов:</b>
🔥 VALUE — найден edge в коэффициентах букмекера
📊 MODEL — прогноз на основе модели (без odds)
🤖 — сигнал прошёл AI-верификацию (Claude)"""
