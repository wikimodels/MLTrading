# AGENTS.md — Справочник для ИИ-ассистента (Dual-Bot Architecture)

> Этот документ описывает архитектуру двух независимых торговых ботов, общие модули, ключевые файлы, команды и соглашения проекта.
> Читай его **первым**, прежде чем редактировать любой код.

---

## Что это за проект

Репозиторий реализует двухнезависимую мульти-ботовую архитектуру для торговли на **Bybit (USDT Perpetual Futures)** со строго фиксированным размером позиции в **$10 USDT на сделку** и плечом **x1**.

В проекте существуют **два независимых бота**, использующих единое хранилище исторических данных Parquet и общие статистические утилиты в папке `shared/`:

1. **`ml_swing_bot` (ML Swing Short Bot)** — торгует только **шортами** на таймфрейме **4H**. Использует LightGBM-модель с walk-forward обучением для предсказания вероятности прибыльной сделки.
2. **`fat_tails_bot` (Fat Tails D1 Bot)** — количественная торговая система на таймфрейме **D1**, основанная на теории «тяжёлых хвостов» (Excess Kurtosis $K > 5$), устойчивости трендов (Hurst exponent $H$) и фазовых переходах волатильности ($Z_{TR} > 3$).

---

## Быстрый старт и команды запуска

```bash
# Установка зависимостей
poetry install

# ── ML Swing Short Bot (4H / LightGBM) ──
poetry run python ml_swing_bot/main.py collect     # Сбор OHLCV + funding с Bybit
poetry run python ml_swing_bot/main.py train       # Обучение модели на всех данных
poetry run python ml_swing_bot/main.py backtest    # Walk-forward бэктест + PDF-отчёт
poetry run python ml_swing_bot/main.py status      # Сводка по собранным данным
poetry run streamlit run ml_swing_bot/monitor.py   # ⭐ Streamlit-дашборд для ML бота
poetry run python ml_swing_bot/main.py run         # Запуск live-бота через APScheduler

# ── Fat Tails D1 Bot (D1 / Quant Strategy) ──
poetry run python fat_tails_bot/main.py screener   # Математический отбор монет по Hurst и Kurtosis
poetry run python fat_tails_bot/main.py backtest   # Бэктест стратегии на D1 с Chandelier Trailing
poetry run streamlit run fat_tails_bot/monitor.py  # ⭐ Streamlit-дашборд и визуальный аудит Fat Tails бота
poetry run python fat_tails_bot/main.py status     # Сводка по доступным данным из базы
poetry run python fat_tails_bot/main.py run        # Запуск live-цикла Fat Tails бота
```

---

## Структура проекта

```
MLTrading/
├── shared/                  # ⭐ Общий код для обоих ботов (ни с кем не смешивается)
│   ├── config_loader.py     # Загрузчик конфигов с динамическим выбором активного бота (ACTIVE_BOT)
│   ├── indicators.py        # Статистические индикаторы (Hurst, Excess Kurtosis, Z-Score TR, CLV)
│   └── data/
│       ├── collector.py     # Загрузка OHLCV + funding с Bybit через ccxt
│       ├── storage.py       # Чтение/запись parquet в data/storage/raw
│       └── processor.py     # Предобработка сырых данных
│
├── ml_swing_bot/            # ⭐ ML Swing Short Bot (4H)
│   ├── main.py              # Точка входа для команд ML бота
│   ├── monitor.py           # Streamlit-дашборд и мониторинг логов/результатов
│   ├── scheduler.py         # APScheduler: авто-ретрейн + авто-синк данных
│   ├── screener.py          # Ручной отбор слабых монет
│   ├── config/settings.yaml # Конфигурация ML-бота (университет монет, порог уверенности)
│   ├── features/            # Инженерный блок признаков (4H + 1H + 1D + 1W)
│   ├── labeling/            # Triple Barrier разметка (TP / SL / таймаут)
│   ├── models/              # WalkForwardTrainer и Predictor LightGBM
│   ├── backtest/            # Симуляция сделок, trades_history.csv, PDF-отчёт
│   ├── trading/             # Live-исполнение ордеров на Bybit
│   └── logs/mltrading.log   # Лог ML-бота
│
├── fat_tails_bot/           # ⭐ Fat Tails D1 Bot (Quant Strategy)
│   ├── main.py              # Точка входа: screener / backtest / run / status
│   ├── monitor.py           # ⭐ Streamlit-дашборд и интерактивный рентген сделок
│   ├── config/settings.yaml # Конфигурация порогов Hurst, Kurtosis, Z-Score TR, CLV
│   ├── screener/screener.py # Фильтрация активов по Hurst > 0.35 и Kurtosis > 5.0
│   ├── strategy/signals.py  # Генерация D1 сигналов и динамических стоп-лоссов
│   ├── backtest/engine.py   # Бэктестер с динамическим выходом без статического TP
│   └── logs/fat_tails.log   # Лог Fat Tails бота
│
├── data/storage/raw/        # ⭐ Единая тяжелая база Parquet (используется обоими ботами)
└── .env                     # Bybit API ключи (не в git)
```

---

## Конфигурация и разделение логики

Каждый бот изолирован в своей директории и содержит собственный `config/settings.yaml`. 
Загрузка конфигурации происходит через `shared.config_loader`. При вызове точек входа `main.py` или `monitor.py` автоматически устанавливается переменная активного бота (`set_active_bot("ml_swing_bot")` или `set_active_bot("fat_tails_bot")`), благодаря чему общие модули сбора данных (`shared.data.storage`) прозрачно обращаются к нужной конфигурации и единой корневой директории баз данных `data/storage/raw`.

### Ключевые принципы Fat Tails бота (`fat_tails_bot/`)
- **Асимметрия выигрыша:** отказ от фиксированного Target Profit (TP) в пользу динамического трейлинг-выхода (Chandelier Exit на базе ATR).
- **Квантитативный отбор:** монеты проходят отбор через `FatTailsScreener` по свойствам распределения доходностей (Excess Kurtosis $K > 5.0$, Hurst exponent $H > 0.35$).
- **Точка входа:** фазовый переход волатильности (Z-Score True Range $> 2.5$) при экстремальных значениях Close Location Value ($CLV \le 0.25$).

---

## Важные соглашения кода

1. **Изоляция проектов** — код `ml_swing_bot` и `fat_tails_bot` не должен зависеть друг от друга. Любой общий функционал выносится исключительно в `shared/`.
2. **Никакого хардкода параметров** — все пороги и настройки задаются через `config/settings.yaml` соответствующего бота.
3. **Логирование через loguru** — `from loguru import logger` во всех модулях. Каждый бот пишет в свою папку `logs/`.
4. **Данные в Parquet** — хранение свечей осуществляется в формате Parquet в корневом `data/storage/raw/`. Бэктестеры сохраняют результаты сделок в `backtest/trades_history.csv` внутри своих папок.
5. **Строгие правила мани-менеджмента** — размер позиции всегда фиксирован: `$10` USDT на сделку, плечо `x1`.
6. **BTC исключается из торговли альткоинами** — `BTC/USDT:USDT` находится в `global_exclude_symbols` и используется только как рыночный индикатор.

---

## Типичные сценарии

### Обновить данные с Bybit (для обоих ботов)
```bash
poetry run python ml_swing_bot/main.py collect
```

### Отфильтровать монеты с тяжелыми хвостами (Fat Tails Bot)
```bash
poetry run python fat_tails_bot/main.py screener
```

### Запустить бэктесты и проверить результаты
```bash
# Для ML-бота:
poetry run python ml_swing_bot/main.py backtest
poetry run streamlit run ml_swing_bot/monitor.py

# Для Fat Tails бота:
poetry run python fat_tails_bot/main.py backtest
```

### Добавить новый индикатор, полезный обоим ботам
→ Добавить функцию в `shared/indicators.py` и импортировать её в нужные стратегии/фичи.

---

## Файлы и папки, которые НЕ нужно редактировать вручную

| Файл / Папка | Причина |
|--------------|---------|
| `poetry.lock` | Управляется `poetry` автоматически |
| `data/storage/**` | Бинарные базы Parquet с Bybit |
| `ml_swing_bot/models/saved/**` | Сериализованные `.pkl` модели LightGBM |
| `*.csv`, `*.pdf`, `*.log` в `logs/` или `backtest/` | Автоматические артефакты бэктестов и монитора |
| `.env` | Секретные ключи API |
