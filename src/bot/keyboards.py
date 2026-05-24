from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton


def main_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="⚾ Сигналы"), KeyboardButton(text="📊 Статистика")],
            [KeyboardButton(text="📅 Матчи сегодня"), KeyboardButton(text="📈 Кривая ROI")],
            [KeyboardButton(text="📋 Анализ матча"), KeyboardButton(text="💰 Расчёт Kelly")],
            [KeyboardButton(text="🔄 Обновить коэффы"), KeyboardButton(text="📥 Скачать CSV")],
            [KeyboardButton(text="ℹ️ Помощь"), KeyboardButton(text="🔔 Уведомления")],
        ],
        resize_keyboard=True,
    )


def signals_filter_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Все", callback_data="filter:all"),
            InlineKeyboardButton(text="Мани-лайн", callback_data="filter:ML"),
            InlineKeyboardButton(text="Тотал", callback_data="filter:TOTAL"),
        ],
        [
            InlineKeyboardButton(text="Ран-лайн", callback_data="filter:RL"),
            InlineKeyboardButton(text="Только VALUE", callback_data="filter:value"),
        ],
    ])


def admin_menu(leads_count: int = 0, ai_on: bool = False) -> InlineKeyboardMarkup:
    leads_label = f"📋 Лиды ({leads_count})" if leads_count else "📋 Лиды"
    ai_label = f"🤖 AI {'✅' if ai_on else '❌'}"
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="👥 Пользователи", callback_data="admin:users"),
            InlineKeyboardButton(text=leads_label, callback_data="admin:leads"),
        ],
        [
            InlineKeyboardButton(text="➕ Добавить по ID", callback_data="admin:add_user"),
            InlineKeyboardButton(text=ai_label, callback_data="admin:ai_toggle"),
        ],
        [
            InlineKeyboardButton(text="🔄 Обучить модели", callback_data="admin:train"),
            InlineKeyboardButton(text="🔙 Закрыть", callback_data="admin:close"),
        ],
    ])


def lead_kb(chat_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Одобрить", callback_data=f"lead:approve:{chat_id}"),
        InlineKeyboardButton(text="❌ Отклонить", callback_data=f"lead:deny:{chat_id}"),
    ]])


def user_remove_kb(chat_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="🗑 Удалить доступ", callback_data=f"user:remove:{chat_id}"),
    ]])


def notifications_kb(enabled: bool) -> InlineKeyboardMarkup:
    label = "🔕 Выключить уведомления" if enabled else "🔔 Включить уведомления"
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=label, callback_data="notif:toggle")]
    ])


def matches_kb(matches: list) -> InlineKeyboardMarkup:
    """Inline keyboard with today's matches for analysis."""
    buttons = []
    for mid, label in matches:
        buttons.append([InlineKeyboardButton(text=label, callback_data=f"match_info:{mid}")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)
