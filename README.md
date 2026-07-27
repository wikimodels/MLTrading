# MLTrading Bot

ML-powered swing short trading bot for Bybit Futures.

## Стек

- **Python 3.12** + **Poetry**
- **CCXT** — подключение к Bybit
- **LightGBM** — ML модель
- **APScheduler** — планировщик задач
- **Parquet** — хранение данных

## Быстрый старт

```bash
# 1. Установка зависимостей
poetry install

# 2. Настройка API ключей
cp .env.example .env
# отредактируй .env — добавь Bybit API ключи

# 3. Сбор данных (топ-20 монет, 2 года истории)
poetry run python main.py collect

# 4. Обучение модели
poetry run python main.py train

# 5. Бэктест
poetry run python main.py backtest

# 6. Запуск бота (testnet)
poetry run python main.py run
```

## Статус данных

```bash
poetry run python main.py status
```

## Архитектура

```
data/           — сбор и хранение данных (CCXT → Parquet)
features/       — feature engineering (60+ фичей + Market Breadth)
labeling/       — ATR-based Triple Barrier для свинг-шортов
models/         — LightGBM тренер и предиктор (walk-forward)
trading/        — risk manager, executor, monitor
backtest/       — бэктест движок с учётом комиссий и funding
scheduler.py    — APScheduler: ретрейн каждые 7 дней, торговля каждые 4H
main.py         — точка входа
```

## Ключевые концепции

- **Свинг-шорты 4H** — целевой таймфрейм
- **ATR-based Triple Barrier** — автоматическая разметка (TP = 2 ATR, SL = 1.5 ATR)
- **Rolling Retrain** — модель переобучается каждые 7 дней на последних 365 днях
- **Market Breadth** — составной сигнал по топ-20 монетам (ключевая фича)
- **1% риска на сделку** — ATR-based position sizing

## Конфигурация

Все параметры в `config/settings.yaml`.
API ключи в `.env` (не коммитить!).
