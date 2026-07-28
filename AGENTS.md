# AGENTS.md — Справочник для ИИ-ассистента

> Этот документ описывает архитектуру, ключевые файлы, команды и соглашения проекта.
> Читай его **первым**, прежде чем редактировать любой код.

---

## Что это за проект

**ML Swing Short Bot** — автоматизированный торговый бот для Bybit Futures.
Торгует только **шортами** (short-only) на таймфрейме **4H**.
Использует LightGBM-модель с walk-forward обучением для предсказания вероятности прибыльной сделки.

**Биржа:** Bybit (USDT Perpetual Futures)
**Позиция:** строго $10 USDT на сделку, плечо x1

---

## Быстрый старт

```bash
# Зависимости
poetry install

# Сбор данных
poetry run python main.py collect

# Бэктест
poetry run python main.py backtest

# Живой мониторинг + дашборд результатов
poetry run streamlit run monitor.py

# Запуск бота (live trading)
poetry run python main.py run
```

---

## Структура проекта

```
MLTrading/
├── main.py                  # Entry point: collect / train / backtest / run / status
├── monitor.py               # ⭐ Streamlit-дашборд (единственный UI файл)
├── scheduler.py             # APScheduler: авто-ретрейн + авто-синк данных
├── config_loader.py         # Загрузка config/settings.yaml
│
├── config/
│   └── settings.yaml        # ⭐ Главный конфиг — все параметры здесь
│
├── data/
│   ├── collector.py         # Загрузка OHLCV + funding с Bybit через ccxt
│   ├── storage.py           # Чтение/запись parquet (data/storage/)
│   └── processor.py         # Предобработка сырых данных
│
├── features/
│   ├── engineer.py          # ⭐ Вся инженерия фич (4H + 1H + 1D + 1W)
│   └── market_breadth.py    # Market breadth фичи (корреляции, % монет выше EMA)
│
├── labeling/
│   └── triple_barrier.py    # Triple Barrier разметка: TP / SL / таймаут
│
├── models/
│   ├── trainer.py           # WalkForwardTrainer + WalkForwardBacktestTrainer
│   ├── predictor.py         # Загрузка модели и предсказание вероятности
│   └── saved/               # Сохранённые .pkl модели
│
├── backtest/
│   ├── engine.py            # BacktestEngine: симуляция сделок по предсказаниям
│   ├── trades_history.csv   # ⭐ Лог всех сделок бэктеста (читает дашборд)
│   ├── equity_curve.csv     # Кривая капитала
│   └── backtest_report.pdf  # PDF-отчёт
│
├── trading/
│   ├── bot.py               # Главный цикл бота: сигнал → вход
│   ├── executor.py          # Открытие/закрытие позиций на бирже
│   ├── risk_manager.py      # Проверки риска: drawdown, liquidity, funding
│   └── monitor.py           # Мониторинг открытых позиций (НЕ UI!)
│
├── logs/
│   └── mltrading.log        # ⭐ Лог всех процессов (читает monitor.py)
│
└── .env                     # Bybit API ключи (не в git)
```

---

## Главный конфиг: `config/settings.yaml`

Все параметры системы — только здесь. Никаких хардкодов в коде.

### Ключевые секции

| Секция | Что контролирует |
|--------|-----------------|
| `symbol_universe.active_group` | `"experimental"` или `"main"` — какой набор монет использовать |
| `exit_mode` | `"simple"` / `"partial_tp"` / `"trailing_only"` — стратегия выхода |
| `model.min_confidence` | Порог вероятности для входа в шорт (0.60 = 60%) |
| `risk.trade_size_usdt` | Размер позиции в $, сейчас $10 |
| `backtest.initial_capital` | Стартовый капитал для бэктеста |
| `features.use_weekly_tf` | Включить недельные фичи (1W таймфрейм) |
| `global_exclude_symbols` | Глобальный блэклист монет (везде: collect/train/backtest/live) |

### Группы монет

```yaml
symbol_universe:
  active_group: "experimental"  # переключи здесь

  groups:
    main:          # авто-отбор: объём → история 3г → корреляция с BTC
    experimental:  # ручной список ~31 "слабой" монеты
```

### Режимы выхода

```yaml
exit_mode: "simple"     # выбери один из трёх
exit_params:
  simple:               # TP + SL, без частичного закрытия
  partial_tp:           # 50% по TP, остаток — Chandelier Exit trailing
  trailing_only:        # только trailing SL с первой свечи
```

---

## Таймфреймы данных

| Таймфрейм | Роль |
|-----------|------|
| **4H** | ⭐ Основной торговый TF. Все решения принимаются здесь |
| **1H** | Дополнительные фичи (multi-TF) |
| **1D** | Дополнительные фичи (multi-TF) |
| **1W** | Дополнительные фичи (если `use_weekly_tf: true`) |

Данные хранятся в `data/storage/` в формате Parquet, разбиты по символу и таймфрейму.

---

## Pipeline: от данных до сделки

```
Bybit API
   ↓
data/collector.py         — загружает OHLCV + funding для каждого символа
   ↓
data/storage.py           — сохраняет в parquet (data/storage/)
   ↓
features/engineer.py      — считает ~100+ технических фич по 4H/1H/1D/1W
   ↓
features/market_breadth.py — добавляет рыночные фичи (breadth)
   ↓
labeling/triple_barrier.py — размечает: 1 (win) / 0 (loss/timeout)
   ↓
models/trainer.py         — walk-forward обучение LightGBM
   ↓
models/predictor.py       — предсказание P(win) для новой свечи
   ↓
trading/bot.py            — если P > min_confidence → открыть шорт
   ↓
trading/executor.py       — исполнение ордера на Bybit
   ↓
trading/risk_manager.py   — проверка drawdown / liquidity / funding
```

---

## Команды `main.py`

| Команда | Что делает |
|---------|-----------|
| `collect` | Скачивает OHLCV + funding с Bybit для активной группы монет |
| `train` | Обучает модель на всех доступных данных (не walk-forward) |
| `backtest` | Walk-forward бэктест → PDF-отчёт + `trades_history.csv` |
| `run` | Запускает live-бот через APScheduler |
| `status` | Показывает сводку по собранным данным |

---

## Дашборд: `monitor.py`

**Единственный UI-файл проекта.** Запуск: `poetry run streamlit run monitor.py`

Содержит два режима отображения:

### Режим 1: Живой монитор (по умолчанию)
- Читает `logs/mltrading.log` и парсит прогресс текущего процесса
- Показывает: фаза (engineering/training/simulation/done), прогресс-бары, ошибки, лог
- Сайдбар: авто-обновление, интервал, количество строк лога
- Кнопка "⬆ Наверх" — плавная прокрутка через JS (EaseOutExpo анимация)

### Режим 2: Дашборд результатов (кнопка "📊 Посмотреть итоги")
- Активируется через `st.session_state.show_results = True`
- Читает `backtest/trades_history.csv`
- Вкладки: General Performance + Pattern Analytics
- Внутренние функции: `_load_trades_data()`, `_render_results()`, `_render_patterns()`

> ⚠️ **`dashboard.py` удалён.** Весь код дашборда встроен в `monitor.py`.

---

## Вспомогательные скрипты в корне

| Файл | Назначение |
|------|-----------|
| `screener.py` | Отбор слабых монет для группы `experimental` |
| `compare_runs.py` | Сравнение нескольких прогонов бэктеста |
| `cleanup.py` | Очистка временных файлов и старых артефактов |
| `check_volume.py` | Проверка объёма торгов монет |
| `recalc.py` | Пересчёт метрик по существующему trades_history.csv |
| `get_list.py` | Получение списка активных символов |

---

## Окружение и зависимости

```toml
# pyproject.toml — основные зависимости
ccxt            # подключение к Bybit API
pandas          # работа с данными
lightgbm        # ML-модель
scikit-learn    # метрики, preprocessing
ta              # технические индикаторы
streamlit       # UI (monitor.py)
plotly          # графики в дашборде
apscheduler     # планировщик для live-бота
loguru          # логирование
fpdf2           # генерация PDF-отчётов
```

### `.env` файл (не в git!)

```env
BYBIT_API_KEY=...
BYBIT_API_SECRET=...
TELEGRAM_BOT_TOKEN=...   # опционально
TELEGRAM_CHAT_ID=...     # опционально
```

---

## Важные соглашения кода

1. **Никакого хардкода параметров** — всё через `config/settings.yaml`
2. **Логирование через loguru** — `from loguru import logger` во всех модулях
3. **Данные в Parquet** — не CSV, кроме финальных артефактов бэктеста
4. **Только шорты** — `direction: short` в конфиге, другие направления не реализованы
5. **Walk-forward** — никакого look-ahead bias: модель обучается только на прошлых данных
6. **$10 на сделку** — `trade_size_usdt: 10.0`, максимально фиксированный размер
7. **BTC исключается из обучения** — `BTC/USDT:USDT` используется только как рыночный индикатор

---

## Типичные сценарии

### Добавить новую монету в эксперимент
→ `config/settings.yaml` → `symbol_universe.groups.experimental.symbols`

### Поменять стратегию выхода
→ `config/settings.yaml` → `exit_mode: "trailing_only"`

### Запустить новый бэктест
```bash
poetry run python main.py backtest
poetry run streamlit run monitor.py   # смотреть прогресс и результаты
```

### Добавить новую фичу
→ `features/engineer.py` → добавить в `compute()` и `get_feature_columns()`

### Включить недельные фичи
→ `config/settings.yaml` → `features.use_weekly_tf: true`

---

## Файлы, которые НЕ нужно трогать

| Файл | Причина |
|------|---------|
| `poetry.lock` | Управляется `poetry` автоматически |
| `backtest/trades_history_1.csv` | Архивная копия предыдущего прогона |
| `backtest/backtest_report_1.pdf` | Архивная копия предыдущего прогона |
| `data/storage/**` | Бинарные данные, не редактировать вручную |
| `models/saved/**` | Сериализованные модели, не редактировать вручную |
| `.env` | Секреты, не коммитить |
