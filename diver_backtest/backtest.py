import numpy as np
from diver_backtest.config import Config

def simulate_trade(df, signal_time, direction: int,
                   cfg: Config) -> dict:
    """
    Симулирует исполнение сделки по свечным данным.
    direction: +1 для Short (Bearish), -1 для Long (Bullish).
    """
    # Находим индекс свечи входа
    idx_candidates = df.index[df["timestamp"] >= signal_time]
    if len(idx_candidates) == 0:
        return None
    entry_idx = idx_candidates[0]

    if entry_idx >= len(df) - 1:
        return None

    entry_price = df["close"].iloc[entry_idx]
    atr_val = df["ATR"].iloc[entry_idx]
    if np.isnan(atr_val) or atr_val <= 0 or np.isnan(entry_price) or entry_price <= 0:
        return None

    lookback = 20
    # Stop Loss & Take Profit расчет
    if direction == +1:  # SHORT
        swing_extreme = df["high"].iloc[max(0, entry_idx - lookback):entry_idx + 1].max()
        risk_swing = swing_extreme - entry_price
        risk_atr = cfg.stop_atr_mult * atr_val
        
        # Защита от слишком мелких стопов на узковолатильном рынке
        risk = max(risk_swing, risk_atr)
        if risk <= 0:
            risk = risk_atr if risk_atr > 0 else 1.0 * atr_val
            
        stop = entry_price + risk
        target = entry_price - cfg.take_rr * risk
    else:  # LONG
        swing_extreme = df["low"].iloc[max(0, entry_idx - lookback):entry_idx + 1].min()
        risk_swing = entry_price - swing_extreme
        risk_atr = cfg.stop_atr_mult * atr_val
        
        # Защита от слишком мелких стопов на узковолатильном рынке
        risk = max(risk_swing, risk_atr)
        if risk <= 0:
            risk = risk_atr if risk_atr > 0 else 1.0 * atr_val
            
        stop = entry_price - risk
        target = entry_price + cfg.take_rr * risk

    fee = cfg.fee_bps / 10000.0
    slip = cfg.slippage_bps / 10000.0

    max_bars = cfg.time_stop_bars
    realized = 0.0
    remaining = 1.0
    bars_held = 0
    exit_reason = "time"
    exit_idx = entry_idx

    end_bar = min(entry_idx + 1 + max_bars, len(df))

    for k in range(entry_idx + 1, end_bar):
        bars_held += 1
        exit_idx = k
        h = df["high"].iloc[k]
        l = df["low"].iloc[k]

        if direction == +1:  # SHORT
            # 1. Stop Loss
            if h >= stop:
                realized += remaining * (entry_price - stop) / entry_price
                exit_reason = "stop"
                remaining = 0.0
                break

            # 2. Полный Take Profit
            if l <= target:
                realized += remaining * (entry_price - target) / entry_price
                exit_reason = "target"
                remaining = 0.0
                break

        else:  # LONG
            # 1. Stop Loss
            if l <= stop:
                realized += remaining * (stop - entry_price) / entry_price
                exit_reason = "stop"
                remaining = 0.0
                break

            # 2. Полный Take Profit
            if h >= target:
                realized += remaining * (target - entry_price) / entry_price
                exit_reason = "target"
                remaining = 0.0
                break

    # Выход по таймауту (time stop)
    if remaining > 0:
        last_price = df["close"].iloc[exit_idx]
        if direction == +1:
            realized += remaining * (entry_price - last_price) / entry_price
        else:
            realized += remaining * (last_price - entry_price) / entry_price

    # Комиссии и проскальзывание (вход + выход)
    total_costs = 2.0 * (fee + slip)
    net = realized - total_costs

    return {
        "entry_time": df["timestamp"].iloc[entry_idx],
        "exit_time": df["timestamp"].iloc[exit_idx],
        "entry_idx": entry_idx,
        "entry_price": entry_price,
        "exit_reason": exit_reason,
        "bars_held": bars_held,
        "gross_return": realized,
        "net_return": net,
        "win": 1 if net > 0 else 0,
        "direction": direction,
    }
