from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton


def main_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="⚾ Сигналы"),            KeyboardButton(text="📅 Сегодня")],
            [KeyboardButton(text="📊 Статистика"),          KeyboardButton(text="📈 График ROI")],
            [KeyboardButton(text="📜 История ставок"),      KeyboardButton(text="🔄 Запустить анализ")],
            [KeyboardButton(text="🔔 Уведомления"),         KeyboardButton(text="ℹ️ Помощь")],
        ],
        resize_keyboard=True,
    )


def history_nav_kb(page: int, has_next: bool) -> InlineKeyboardMarkup:
    buttons = []
    if page > 0:
        buttons.append(InlineKeyboardButton(text="⬅️ Новее", callback_data=f"history:{page - 1}"))
    if has_next:
        buttons.append(InlineKeyboardButton(text="Старее ➡️", callback_data=f"history:{page + 1}"))
    if not buttons:
        return InlineKeyboardMarkup(inline_keyboard=[])
    return InlineKeyboardMarkup(inline_keyboard=[buttons])


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
            InlineKeyboardButton(text="🗑 Очистить AI-кэш", callback_data="admin:clear_ai_cache"),
        ],
        [
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
