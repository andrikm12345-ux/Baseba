# ⚾ Baseball Signals Bot — MLB

Автономный Telegram-бот для ставок на бейсбол MLB с машинным обучением и AI-ансамблем.

## Возможности

- 📊 **3 рынка:** Мани-лайн (ML), Тотал ранов (8.5), Ран-лайн (±1.5)
- 🤖 **XGBoost** — 3 откалиброванных модели на Elo + форме + H2H
- 🧠 **AI ансамбль** — Claude анализирует питчеров, ERA/WHIP, буллпен
- 💹 **VALUE ставки** — сравнение с Bet365/Betfair через odds-api.io
- 📈 **Учёт ROI** — settlement результатов, кривая прибыли
- 🚀 **Деплой на Railway** с PostgreSQL

## Быстрый старт

```bash
cp .env.example .env
# заполните TELEGRAM_BOT_TOKEN и ADMIN_IDS в .env
pip install -r requirements.txt
python -m src.main
```

## Переменные окружения

| Переменная | Описание |
|-----------|----------|
| `TELEGRAM_BOT_TOKEN` | Токен бота от @BotFather |
| `ADMIN_IDS` | ID администраторов через запятую |
| `ODDS_API_KEY` | Ключ odds-api.io (опционально) |
| `ANTHROPIC_API_KEY` | Ключ Claude API (для AI ансамбля) |
| `DATABASE_URL` | PostgreSQL URL (Railway авто-инжектит) |
| `TOTAL_LINE` | Линия тотала (по умолчанию 8.5) |
| `RL_LINE` | Линия ран-лайна (по умолчанию 1.5) |

## Рынки

| Рынок | Описание | Варианты |
|-------|----------|----------|
| **ML** | Мани-лайн — победитель игры | HOME / AWAY |
| **TOTAL** | Тотал ранов | OVER / UNDER (линия 8.5) |
| **RL** | Ран-лайн | COVER (хозяева −1.5) / LAY (гости +1.5) |

## API

Используется бесплатный **MLB Stats API** (statsapi.mlb.com) — без регистрации и ключей.

## Деплой

```bash
# Railway
railway up
```

Переменные окружения устанавливаются в Railway dashboard. `DATABASE_URL` инжектируется автоматически при добавлении PostgreSQL плагина.
