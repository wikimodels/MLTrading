# Trend Detector Lab

## Концепция

Исследовательский стенд для поиска рыночных режимов через анализ паттернов переворотов трендовых индикаторов.

В боковике Chandelier Exit часто переворачивается — образуются серии коротких трейдов:

```
Long  → 4 свечи
Short → 6 свечей
Long  → 5 свечей
Short → 7 свечей
Long  → 42 свечи  ← вот это и нужно поймать
```

Цель: найти условие, после которого **вероятность длинного тренда выше нормы**.

**Chandelier** здесь — первый плагин-детектор.
Архитектура универсальна: SuperTrend, Donchian, Keltner подключаются заменой одного модуля.

---

## Версионность

| версия | что делаем |
|---|---|
| **v1** | count, probability, lift, random baseline |
| **v2** | bootstrap CI, p-value |
| **v3** | walk-forward 9/12/18 мес, multi-indicator сравнение |

Сначала смотрим: есть ли вообще эффект (lift 1.7+ или 1.01?).
Потом занимаемся статистической значимостью.

---

## Данные

- Источник: `../data/storage/raw/` (48 монет, уже собраны)
- **01_download.py делает инкрементальное обновление до сегодняшнего дня** через ccxt
- Работаем с тем, что есть — не пересобираем всё заново

---

## Логика стопов (03_trades.py)

Два режима, переключаются через `settings.yaml`:

### Режим 1: `stop_mode: chandelier` (по умолчанию)

Стоп = уровень CE, который трейлится за ценой.
Выход происходит, когда цена пересекает CE-уровень (= следующий CE-сигнал).

```
Long trade:  стоп = CE_Long  (ниже цены, трейлится вверх)
Short trade: стоп = CE_Short (выше цены, трейлится вниз)

Выход: open свечи после переворота CE
```

### Режим 2: `stop_mode: fixed` (заложен, v1 не использует)

Стоп = фиксированное расстояние от цены входа, не двигается.

```
Long trade:  стоп = entry_price - sl_atr_mult × ATR_at_entry
Short trade: стоп = entry_price + sl_atr_mult × ATR_at_entry

Выход: open свечи после пробоя стопа
```

### В таблице trades.parquet сохраняем оба уровня:

| колонка | описание |
|---|---|
| `stop_mode` | `chandelier` / `fixed` |
| `stop_level_at_entry` | уровень CE или fixed-стоп на момент входа |
| `stop_hit` | bool: стоп был пробит (до следующего сигнала) |

---

## Пайплайн

```
01_download.py   →  data/ohlcv/<symbol>_4h.parquet
                              ↓
02_chandelier.py →  data/raw_signals.parquet   (CE переворот = 1 строка)
                              ↓
03_trades.py     →  data/trades.parquet        (сделка = время до след. сигнала)
                              ↓
04_features.py   →  data/features.parquet      (контекст последовательностей)
                              ↓
05_statistics.py →  output/statistics.xlsx     (вероятности + random baseline)
                              ↓
research.py      →  output/<experiment>.xlsx   (YAML-driven, любые параметры)
```

Каждый скрипт читает входной файл и пишет выходной.
Пересчитывать нужно только изменившийся этап и всё ниже.

---

## Структура проекта

```
trend_detector_lab/
├── PLAN.md
├── config/
│   └── settings.yaml
├── detectors/
│   └── chandelier.py          # первый плагин-детектор
├── research/
│   └── experiments/           # YAML-файлы экспериментов
│       ├── streak_basic.yaml
│       └── streak_with_distance.yaml
├── data/                       # промежуточные файлы (не в git)
│   ├── ohlcv/
│   ├── raw_signals.parquet
│   ├── trades.parquet
│   └── features.parquet
├── output/                     # результаты
│   ├── statistics.xlsx
│   └── <experiment_name>.xlsx
├── experiments/                # experiment_metadata.json для каждого запуска
├── 01_download.py
├── 02_chandelier.py
├── 03_trades.py
├── 04_features.py
├── 05_statistics.py
└── research.py
```

---

## 01_download.py → `data/ohlcv/`

- Читает существующие parquet из `../data/storage/raw/` (48 монет)
- Инкрементально дополняет через ccxt если нужно
- Берём всю доступную историю (3–5 лет, без ограничений)
- Сохраняет по монетам: `data/ohlcv/BTC_USDT_USDT_4h.parquet`

---

## 02_chandelier.py → `data/raw_signals.parquet`

Одна строка = **один CE-переворот**.

### Алгоритм (Wilder RMA, как в TradingView)

```
ATR_21   = RMA(TR, 21)          # RMA[i] = (prev × 20 + TR[i]) / 21
CE_Long  = Highest(high, 21) - 2.5 × ATR_21
CE_Short = Lowest(low, 21)  + 2.5 × ATR_21

direction = 1 (long) | -1 (short)
Signal    = смена direction
```

### Схема raw_signals.parquet

| колонка | описание |
|---|---|
| `coin` | символ |
| `tf` | таймфрейм (4h) |
| `timestamp` | время переворота |
| `signal` | `long` / `short` |
| `close` | цена закрытия |
| `atr` | ATR (абсолютный) |
| `atr_pct` | atr / close × 100 |
| `ce_level` | уровень CE |
| `ce_dist_pct` | (close − ce_level) / close × 100 |
| `highest_high_21` | max(high) за 21 бар |
| `lowest_low_21` | min(low) за 21 бар |
| `range_21_pct` | (highest − lowest) / close × 100 |

---

## 03_trades.py → `data/trades.parquet`

Одна строка = **одна сделка** (от сигнала до следующего сигнала).

### Схема trades.parquet

| колонка | описание |
|---|---|
| `coin` | символ |
| `tf` | таймфрейм |
| `signal_ts` | timestamp входа |
| `exit_ts` | timestamp выхода |
| `signal` | `long` / `short` |
| `entry_price` | open свечи после сигнала |
| `exit_price` | open свечи после следующего сигнала |
| `bars_in_trade` | свечей в сделке |
| `profit_pct` | (exit − entry) / entry × 100 × direction |
| `profit_atr` | profit_pct / atr_pct |
| `atr` | ATR на входе |
| `atr_pct` | atr / entry × 100 |
| `max_favorable_atr` | макс. движение В СТОРОНУ за след. 60 свечей |
| `max_adverse_atr` | макс. движение ПРОТИВ |
| `reached_2atr` | bool |
| `reached_3atr` | bool |
| `reached_5atr` | bool |
| `reached_8atr` | bool |
| `bars_gt_15` | bool |
| `bars_gt_25` | bool |
| `bars_gt_50` | bool |

---

## 04_features.py → `data/features.parquet`

Для каждой сделки добавляем контекст N предыдущих сделок по той же монете.

### Скользящая статистика (N в [3, 4, 5, 6, 7, 8, 10])

```
bars_last_N_avg      # среднее
bars_last_N_median   # медиана
bars_last_N_max      # максимум
bars_last_N_min      # минимум
bars_last_N_std      # стандартное отклонение
```

### Streak-фичи (порог X в [4, 5, 6, 8, 10, 12])

```
streak_le_X      # сколько подряд сделок ≤ X свечей
cumsum_le_X      # суммарных свечей за эту серию (время боковика)
```

**Пример:**
```
bars: 4, 7, 6, 5, [сигнал]
streak_le_8 = 4
cumsum_le_8 = 4+7+6+5 = 22 свечи боковика
```

### distance_to_previous_big_trend (ключевая фича)

```
bars_since_last_big_N   # свечей с момента окончания последней сделки
                         # с bars_in_trade > N  (для N в [15, 25, 50])
```

**Пример:**
```
Последний трейд > 25 свечей закончился 132 бара назад.
Рынок застрял надолго. Потенциально сильнее, чем streak.
```

### Дополнительно

```
atr_pct_last_N_avg    # средний ATR% за N сделок
signal_alternating    # bool: последние N чередовались L/S/L/S
```

---

## 05_statistics.py → `output/statistics.xlsx`

### v1: count + probability + lift + random baseline

Для каждого условия `(streak_le_X >= N)`:

| метрика | описание |
|---|---|
| `count` | кол-во сигналов с этим условием |
| `p_gt15bars` | P(bars_in_trade > 15) |
| `p_gt25bars` | P(bars_in_trade > 25) |
| `p_reach3atr` | P(reached_3atr) |
| `p_reach5atr` | P(reached_5atr) |
| `baseline_p_gt15bars` | P без фильтра (все сделки) |
| `lift` | p_gt15bars / baseline |
| `rand_baseline_mean` | средний P по 1000 случайным выборкам того же размера |
| `rand_baseline_p95` | 95-й перцентиль случайных выборок |
| `rand_above_p95` | bool: наш P выше 95% случайных — **главный критерий v1** |

### Random baseline (пермутационный подход)

```
Наш фильтр дал 273 сделки, P(>15) = 41%.

1000 раз:
  берём случайные 273 сделки из всей выборки
  → P(>15) = 22%, 24%, 20%, ...

Если наши 41% выше 95-го перцентиля случайных →
это уже не случайность.
```

Это интуитивно понятнее p-value: сразу видно, насколько эффект велик практически.

### Walk-forward (v1: только 6 мес)

Разбиваем историю на 6-месячные окна. Для каждого окна:
- применяем условие
- считаем lift

Итог: сколько окон lift > 1.0, сколько lift > 1.3

**v2+:** добавим 9/12/18 мес окна.

### Листы в Excel

1. **grid** — все комбинации N × X с lift и rand_above_p95
2. **per_symbol** — метрики по каждой монете
3. **by_atr_regime** — низкий / средний / высокий ATR%
4. **walk_forward_6m** — 6-месячные окна
5. **baseline** — безусловные распределения

---

## research.py — универсальный исследовательский скрипт

Простой движок: читает YAML → строит фильтр → считает статистику → сохраняет Excel.

```bash
poetry run python trend_detector_lab/research.py \
    --experiment research/experiments/streak_basic.yaml
```

### Формат YAML-эксперимента

```yaml
name: "streak_basic"
description: "Грид-поиск по N × X для streak_le_X"

grid:
  streak_threshold: [4, 5, 6, 8, 10, 12]
  streak_count: [2, 3, 4, 5, 6, 7]

filters:
  min_count: 50

targets:
  - bars_gt_15
  - reached_3atr
  - reached_5atr

random_baseline:
  iterations: 1000

walk_forward:
  window_months: 6

output: "output/streak_basic.xlsx"
```

```yaml
name: "streak_with_distance"
description: "Добавляем distance_to_previous_big_trend"

grid:
  streak_threshold: [5, 6, 8]
  streak_count: [3, 4, 5]
  bars_since_last_big: [50, 100, 150]

targets:
  - bars_gt_25
  - reached_5atr

random_baseline:
  iterations: 1000

output: "output/streak_with_distance.xlsx"
```

---

## experiment_metadata.json

Каждый запуск `research.py` сохраняет в `experiments/<name>_<timestamp>.json`:

```json
{
  "experiment": "streak_basic",
  "timestamp": "2026-08-06T22:45:00Z",
  "git_commit": "a3f9c12",
  "settings": {
    "atr_period": 21,
    "multiplier": 2.5,
    "timeframe": "4h",
    "commission": 0.00055
  },
  "data": {
    "coins": 48,
    "total_trades": 12847,
    "date_from": "2022-01-01",
    "date_to": "2026-08-06"
  },
  "runtime_seconds": 14.3
}
```

Через месяц позволяет точно воспроизвести любой эксперимент.

---

## config/settings.yaml

```yaml
detector: chandelier              # первый плагин

chandelier:
  atr_period: 21
  multiplier: 2.5
  timeframe: "4h"

data:
  source_dir: "../data/storage/raw"
  output_dir: "data"
  forward_bars_window: 60         # смотрим 60 свечей вперёд для max_favorable

features:
  look_back_windows: [3, 4, 5, 6, 7, 8, 10]
  short_thresholds: [4, 5, 6, 8, 10, 12]
  big_trend_thresholds: [15, 25, 50]

analysis:
  target_bars: [10, 15, 25, 50]
  target_atr_mult: [2, 3, 5, 8]
  random_baseline_iterations: 1000

backtest:
  commission: 0.00055
  slippage: 0.0005
```

---

## Ключевые принципы

- **Wilder RMA** — не EMA/SMA, иначе CE не совпадёт с TradingView
- **Разделение raw_signals / trades** — чистая архитектура, легко расширять
- **ATR%** = atr / close × 100 — нормализация BTC vs мемкоин
- **Метрика v1 = lift + rand_above_p95** — не p-value, а практический эффект
- **research.py простой** — читает YAML, фильтрует, считает, сохраняет
- **experiment_metadata.json** — воспроизводимость через месяц
- **Вся доступная история** — без ограничений, walk-forward покажет устойчивость

---

## Дорожная карта

```
v1  →  count, probability, lift, random baseline, walk-forward 6m
v2  →  bootstrap CI, p-value, walk-forward 9/12/18m
v3  →  SuperTrend как второй детектор, multi-indicator сравнение
```

---

## Запуск

```bash
cd D:\GitHub\MLTrading

poetry run python trend_detector_lab/01_download.py
poetry run python trend_detector_lab/02_chandelier.py
poetry run python trend_detector_lab/03_trades.py
poetry run python trend_detector_lab/04_features.py
poetry run python trend_detector_lab/05_statistics.py

poetry run python trend_detector_lab/research.py \
    --experiment research/experiments/streak_basic.yaml
```
