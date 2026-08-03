"""Validation and sensitivity analysis tools for Fat Tails strategy."""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional
import pandas as pd
from loguru import logger

from fat_tails_bot.backtest.engine import BacktestEngine
from fat_tails_bot.strategy.signals import SignalGenerator
from shared.data.storage import DataStorage
from fat_tails_bot.screener.screener import load_daily_ohlcv


def walk_forward(train_window_days: int = 365,
                 test_window_days: int = 90,
                 step_days: int = 90,
                 max_concurrent_positions: int = 100) -> pd.DataFrame:
    """Runs a walk-forward validation without look-ahead bias across fixed thresholds."""
    logger.info(f"Starting Walk-Forward Validation: train={train_window_days}d, test={test_window_days}d, step={step_days}d")
    
    storage = DataStorage()
    generator = SignalGenerator()
    engine = BacktestEngine()
    
    symbols = [s for s in storage.list_symbols("4h") if s not in engine.exclude]
    if not symbols:
        return pd.DataFrame()
        
    prepared_data = {}
    for sym in symbols:
        df = load_daily_ohlcv(storage, sym)
        if df is None or len(df) < 50:
            continue
        sig_df = generator.compute_signals(df)
        if not sig_df.empty:
            prepared_data[sym] = sig_df

    all_dates = sorted(set().union(*[set(df.index) for df in prepared_data.values()]))
    if not all_dates:
        return pd.DataFrame()
        
    start_date = all_dates[0]
    end_date = all_dates[-1]

    results = []
    window_start = start_date

    while True:
        train_end = window_start + pd.Timedelta(days=train_window_days)
        test_end = train_end + pd.Timedelta(days=test_window_days)
        if test_end > end_date:
            break

        test_data = {
            symbol: df[(df.index > train_end) & (df.index <= test_end)]
            for symbol, df in prepared_data.items()
        }
        test_data = {s: df for s, df in test_data.items() if len(df) > 0}

        if test_data:
            res = engine.multi_asset_backtest(
                test_data, 
                max_concurrent_positions=max_concurrent_positions,
                execute_next_open=True
            )
            summary = res.summary()
            summary['train_end'] = train_end
            summary['test_start'] = train_end
            summary['test_end'] = test_end
            results.append(summary)

        window_start = window_start + pd.Timedelta(days=step_days)

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        logger.info("\n=== WALK FORWARD RESULTS ===")
        print(df_res[['test_start', 'test_end', 'total_trades', 'win_rate', 'total_pnl', 'profit_factor']].to_string(index=False))
    return df_res


def sensitivity_analysis(kurt_thresholds: List[float] = [3.0, 5.0, 7.0],
                         hurst_thresholds: List[float] = [0.55, 0.65, 0.70],
                         z_tr_thresholds: List[float] = [2.5, 3.0, 3.5],
                         clv_thresholds: List[float] = [0.10, 0.15, 0.20],
                         max_concurrent_positions: int = 100) -> pd.DataFrame:
    """Runs a parameter grid search to evaluate strategy robustness."""
    logger.info("Starting Sensitivity Analysis on all combinations of parameters...")
    
    storage = DataStorage()
    engine = BacktestEngine()
    
    symbols = [s for s in storage.list_symbols("4h") if s not in engine.exclude]
    if not symbols:
        return pd.DataFrame()

    raw_data = {}
    for sym in symbols:
        df = load_daily_ohlcv(storage, sym)
        if df is not None and len(df) >= 50:
            raw_data[sym] = df
            
    grid = list(itertools.product(kurt_thresholds, hurst_thresholds, z_tr_thresholds, clv_thresholds))
    logger.info(f"Total combinations to test: {len(grid)}")

    rows = []
    
    for i, (kurt_th, hurst_th, z_th, clv_th) in enumerate(grid, 1):
        logger.info(f"[{i}/{len(grid)}] Testing K={kurt_th}, H={hurst_th}, Z={z_th}, CLV={clv_th}")
        
        generator = SignalGenerator()
        generator.kurtosis_min = kurt_th
        generator.hurst_min = hurst_th
        generator.tr_zscore_min = z_th
        generator.clv_short_max = clv_th
        
        prepared = {}
        for symbol, df in raw_data.items():
            sig_df = generator.compute_signals(df)
            if not sig_df.empty:
                prepared[symbol] = sig_df

        res = engine.multi_asset_backtest(
            prepared, 
            max_concurrent_positions=max_concurrent_positions,
            execute_next_open=True
        )
        summary = res.summary()
        summary.update({
            'kurt_threshold': kurt_th,
            'hurst_threshold': hurst_th,
            'z_tr_threshold': z_th,
            'clv_threshold': clv_th,
        })
        rows.append(summary)

    df_res = pd.DataFrame(rows)
    if not df_res.empty:
        logger.info("\n=== SENSITIVITY RESULTS (TOP 10 BY PNL) ===")
        print(df_res.sort_values('total_pnl', ascending=False).head(10)[
            ['kurt_threshold', 'hurst_threshold', 'z_tr_threshold', 'clv_threshold', 'total_trades', 'total_pnl', 'profit_factor']
        ].to_string(index=False))
        
    return df_res
