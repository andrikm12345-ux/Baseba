"""Message formatters for the MLB Baseball Signals bot (Russian)."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

from src.config import settings
from src.data.database import Match, Signal, Team

MSK = timezone(timedelta(hours=3))

MARKET_LABELS: dict[str, str] = {
    "ML": "Мани-лайн",
    "TOTAL": f"Тотал ({settings.total_line})",
    "RL": f"Ран-лайн (±{settings.rl_line})",
}

PICK_LABELS: dict[str, str] = {
    "HOME": "П1",
    "AWAY": "П2",
    "OVER": "Тотал Б",
    "UNDER": "Тотал М",
    "COVER": f"−{settings.rl_line}",
    "LAY": f"+{settings.rl_line}",
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

    # RL pick показываем с именем команды
    if signal.market == "RL":
        if signal.pick == "COVER":
            pick_label = f"{home_name} −{settings.rl_line}"
        else:
            pick_label = f"{away_name} +{settings.rl_line}"
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
    if ai_comment:
        lines += ["", f"💬 <i>{ai_comment}</i>"]
    return "\n".join(lines)


def format_signal_short(signals: list[Signal], matches: dict, teams: dict) -> str:
    """Compact one-liner per signal for digest messages."""
    if not signals:
        return "Нет активных сигналов."
    value_sigs = sorted([s for s in signals if s.is_value], key=lambda s: -s.edge)
    model_sigs = sorted([s for s in signals if not s.is_value], key=lambda s: -s.confidence)
    ordered = value_sigs + model_sigs
    lines = []
    for s in ordered[:10]:
        m = matches.get(s.match_id)
        if not m:
            continue
        home = teams.get(m.home_team_id)
        away = teams.get(m.away_team_id)
        h_name = (home.short_name or home.name)[:12] if home else "?"
        a_name = (away.short_name or away.name)[:12] if away else "?"
        market_label = MARKET_LABELS.get(s.market, s.market)
        if s.market == "RL":
            pick_label = f"{h_name} −{settings.rl_line}" if s.pick == "COVER" else f"{a_name} +{settings.rl_line}"
        else:
            pick_label = PICK_LABELS.get(s.pick, s.pick)
        badge = "🔥" if s.is_value else "📊"
        line = f"{badge} {h_name}–{a_name} | {market_label} → {pick_label} ({s.confidence:.0%})"
        if s.book_odds and s.book_odds > 1.0:
            line += f" @ {s.book_odds:.2f}"
        if s.is_value and s.edge > 0:
            line += f" | edge {s.edge:.1%}"
        lines.append(line)
    return "\n".join(lines)


def format_training_report(metrics: dict) -> str:
    n = metrics.get("n_train", 0)
    lines = [
        "🏋️ <b>Модели обновлены (MLB)</b>",
        f"Обучено на {n} играх",
        "",
        "<b>In-sample Brier:</b>",
        f"  ML (мани-лайн): {metrics.get('ml_brier', 0):.4f}",
        f"  TOTAL (тотал): {metrics.get('total_brier', 0):.4f}",
        f"  RL (ран-лайн): {metrics.get('rl_brier', 0):.4f}",
    ]
    wf = metrics.get("walk_forward", {})
    if wf:
        lines += [
            "",
            "<b>Walk-forward (out-of-sample):</b>",
            f"  ML: {wf.get('ml_brier', 0):.4f}",
            f"  TOTAL: {wf.get('total_brier', 0):.4f}",
            f"  RL: {wf.get('rl_brier', 0):.4f}",
        ]
    top = metrics.get("top_features", [])
    if top:
        lines += ["", "<b>Топ-5 признаков (ML):</b>"]
        lines += [f"  {i+1}. {f}" for i, f in enumerate(top)]
    diff = metrics.get("diff_vs_prev", {})
    if diff:
        lines.append("")
        lines.append("<b>Δ vs предыдущая версия:</b>")
        for k, v in diff.items():
            arrow = "↓" if v < 0 else "↑"
            lines.append(f"  {k}: {arrow}{abs(v):.4f}")
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

    return "\n".join([
        "📊 <b>Статистика сигналов</b>",
        "",
        _row("Все модели", model_stats),
        _row("VALUE ставки", value_stats),
        _row("AI-ансамбль", ai_stats),
    ])


WELCOME_TEXT = """⚾ <b>Бейсбол Сигналы — MLB Bot</b>

Привет! Я автономный бот для ставок на бейсбол MLB.

<b>Как это работает:</b>
1. 📥 Загружаю историю игр MLB через statsapi.mlb.com
2. 📊 Рассчитываю Elo-рейтинги, форму, средние раны
3. 🤖 Обучаю XGBoost на 3 рынках: ML / Тотал / Ран-лайн
4. 💹 Сравниваю с линиями букмекеров (Bet365, Betfair)
5. 🧠 Опционально — анализ Claude AI (ERA, WHIP, буллпен)
6. ✅ Веду учёт результатов и ROI

<b>Рынки:</b>
• <b>ML</b> — Мани-лайн (победитель игры)
• <b>TOTAL</b> — Тотал ранов (линия 8.5)
• <b>RL</b> — Ран-лайн (фора ±1.5 рана)

<b>Управление банкроллом:</b> ¼ Kelly, максимум 2 единицы за ставку.

Используйте /signals для актуальных прогнозов."""

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
