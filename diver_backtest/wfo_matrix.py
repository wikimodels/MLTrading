import sys
import pandas as pd
from pathlib import Path
from dateutil.relativedelta import relativedelta
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from diver_backtest.config import Config
from diver_backtest.grid_search import run_grid_search
from diver_backtest.data_loader import load_all
from diver_backtest.indicators import add_indicators, INDICATOR_MAP
from diver_backtest.swings import find_swing_highs, find_swing_lows
from diver_backtest.divergence import all_divergences
from diver_backtest.sync import align_divergences
from diver_backtest.signals import cluster_to_signal
from diver_backtest.backtest import simulate_trade

def run_wfo_matrix():
    logger.info("Запуск Walk-Forward Optimization (WFO)...")
    all_wfo_raw_trades = []
    
    # 1. Задаем режимы матрицы окон (Обучение / Торговля в месяцах)
    wfo_modes = [
        (6, 2),   # 6 months train, 2 months trade
        (12, 3),  # 12 months train, 3 months trade
    ]
    
    # Загружаем базу один раз за все 5 лет
    cfg = Config()
    cfg.start_date = "2021-01-01"
    cfg.end_date = "2026-10-06"
    logger.info("Подготовка полного кеша данных (индикаторы и т.д.)...")
    raw_data = load_all(cfg)
    
    full_prepped = {}
    for coin in cfg.coins:
        full_prepped[coin] = {}
        for tf in cfg.timeframes:
            if tf in raw_data[coin] and not raw_data[coin][tf].empty:
                full_prepped[coin][tf] = add_indicators(raw_data[coin][tf])
                
    master_start = pd.to_datetime("2021-01-01", utc=True)
    master_end = pd.to_datetime("2026-10-06", utc=True)
    
    # Для каждого режима считаем WFE
    results = []
    
    for train_m, test_m in wfo_modes:
        logger.info(f"\n=========================================\nЗапуск WFO режима: Обучение {train_m} мес / Торговля {test_m} мес")
        
        step = 0
        current_train_start = master_start
        all_oos_trades = []
        
        while True:
            current_train_end = current_train_start + relativedelta(months=train_m)
            current_test_start = current_train_end
            current_test_end = current_test_start + relativedelta(months=test_m)
            
            if current_test_start >= master_end:
                break # Дошли до конца истории
                
            if current_test_end > master_end:
                current_test_end = master_end
                
            step += 1
            logger.info(f"Шаг {step}: Train [{current_train_start.date()} - {current_train_end.date()}] | Test [{current_test_start.date()} - {current_test_end.date()}]")
            
            # --- 1. IN-SAMPLE OPTIMIZATION ---
            # Здесь мы запускаем мини-grid search
            step_cfg = Config()
            step_cfg.start_date = str(current_train_start.date())
            step_cfg.end_date = str(current_train_end.date())
            # Чтобы Colab не умер, ограничим перебор:
            # - vol_options = ['all', 'gt_50']
            # - sl_options = [1.2, 1.5, 2.0]
            # - tp_options = [2.0, 3.0]
            # (это захардкожено в grid_search, мы его вызовем напрямую или импортируем,
            # но чтобы скрипт был автономным и быстрым, напишем вызов тут)
            
            try:
                report = run_grid_search(step_cfg)
            except Exception as e:
                logger.error(f"Ошибка в Grid Search на шаге {step}: {e}")
                current_train_start += relativedelta(months=test_m)
                continue
                
            if report.empty:
                current_train_start += relativedelta(months=test_m)
                continue
                
            # Ищем ТОП-1 параметр для каждой монеты в этом окне
            report = report[report['trades'] >= 3] # хотя бы 3 сделки для стат. значимости
            if report.empty:
                current_train_start += relativedelta(months=test_m)
                continue
                
            best_params = report.loc[report.groupby('exec_coin')['total_net'].idxmax()]
            
            # --- 2. OUT-OF-SAMPLE TRADING (GRANDFATHERING) ---
            for _, row in best_params.iterrows():
                coin = row['exec_coin']
                tf = row['tf']
                best_lookback = int(row['lookback'])
                best_sl = float(row['sl_mult'])
                best_tp = float(row['tp_rr'])
                best_dir = row['direction']
                
                df_full = full_prepped.get(coin, {}).get(tf)
                if df_full is None or df_full.empty:
                    continue
                    
                # Ищем сигналы ТОЛЬКО внутри тестового окна
                # Оптимизация: мы ищем свинги только на куске df_test (плюс с запасом назад для расчета свингов)
                start_buffer = current_test_start - relativedelta(days=60)
                df_test_search = df_full[(df_full['timestamp'] >= start_buffer) & (df_full['timestamp'] <= current_test_end)].copy()
                
                if len(df_test_search) < 50:
                    continue
                    
                step_cfg.window_max_bars = best_lookback
                sh = find_swing_highs(df_test_search, step_cfg.swing_n.get(tf, 3), step_cfg.swing_atr_k.get(tf, 1.2))
                sl = find_swing_lows(df_test_search, step_cfg.swing_n.get(tf, 3), step_cfg.swing_atr_k.get(tf, 1.2))
                
                # Ищем дивергенции
                res = all_divergences(df_test_search, list(INDICATOR_MAP.keys()), sh, sl, step_cfg, tf=tf)
                
                # Собираем в список и фильтруем по дате тестирования
                for ind, divs in res.items():
                    for d in divs:
                        # Берем дату подтверждения сигнала
                        sig_time = df_test_search["timestamp"].iloc[d["confirm_idx"]]
                        # GRANDFATHERING ПРОВЕРКА: Сигнал должен быть строго внутри OOS окна
                        if current_test_start <= sig_time <= current_test_end:
                            # Проверяем направление (мы торгуем только лучшее для этой монеты)
                            d_dir = "Short" if d["direction"] == +1 else "Long"
                            if d_dir != best_dir:
                                continue
                                
                            # Симулируем сделку по всему датафрейму, начиная с sig_time (может закрыться далеко в будущем!)
                            step_cfg.stop_atr_mult = best_sl
                            step_cfg.take_rr = best_tp
                            
                            trade = simulate_trade(df_full, sig_time, +1 if d_dir == "Short" else -1, step_cfg)
                            if trade:
                                all_oos_trades.append({
                                    'coin': coin,
                                    'tf': tf,
                                    'direction': d_dir,
                                    'entry_time': sig_time,
                                    'exit_time': trade['exit_time'],
                                    'net_return': trade['net_return'],
                                    'win': trade['win'],
                                    'vol_filter': row['vol_filter'],
                                    'lookback': best_lookback,
                                    'sl_mult': best_sl,
                                    'tp_rr': best_tp,
                                    'indicator': ind
                                })
            
            # Сдвиг окна вперед
            current_train_start += relativedelta(months=test_m)
            
        # Агрегация результатов режима WFO
        if all_oos_trades:
            oos_df = pd.DataFrame(all_oos_trades)
            # Убираем дубликаты (одна свеча могла сгенерить дивер по 3 разным индикаторам)
            oos_df = oos_df.drop_duplicates(subset=['coin', 'tf', 'direction', 'entry_time'])
            
            # --- СОХРАНЕНИЕ ДЕТАЛЬНОГО ОТЧЕТА ---
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = cfg.output_dir / f"wfo_oos_trades_{train_m}_{test_m}.csv"
            oos_df.to_csv(csv_path, index=False, encoding="utf-8")
            all_wfo_raw_trades.extend(oos_df.to_dict("records"))
            logger.info(f"Детальный лог сделок для {train_m}/{test_m} сохранен в {csv_path}")
            
            total_oos_profit = oos_df['net_return'].sum()
            win_rate = oos_df['win'].mean() * 100
            trades_count = len(oos_df)
            
            results.append({
                'Mode': f"{train_m}/{test_m}",
                'OOS Trades': trades_count,
                'OOS WinRate': round(win_rate, 2),
                'OOS Total Net': round(total_oos_profit * 100, 2)
            })
            logger.info(f"Итог {train_m}/{test_m}: Сделок {trades_count}, Net Return = {total_oos_profit*100:.2f}%")
        else:
            logger.warning(f"Итог {train_m}/{test_m}: 0 сделок на OOS")
            
    print("\n\n=== ИТОГОВАЯ МАТРИЦА WFO ===")
    print(pd.DataFrame(results).to_string(index=False))


    if all_wfo_raw_trades:
        df_all_raw = pd.DataFrame(all_wfo_raw_trades)
        raw_path = cfg.output_dir / "wfo_raw_trades_all.csv"
        df_all_raw.to_csv(raw_path, index=False, encoding="utf-8")
        logger.info(f"==> WFO RAW ARRAYS: {len(df_all_raw)} trades saved to {raw_path}")

if __name__ == "__main__":

    run_wfo_matrix()
