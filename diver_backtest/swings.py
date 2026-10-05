import numpy as np
import pandas as pd

def find_swing_highs(df: pd.DataFrame, n: int, atr_k: float) -> list:
    """
    Находит индексы i, где High[i] является локальным максимумом на интервале [i-n, i+n]
    и High[i] - min(Low[i-n:i+n]) > dynamic_k * ATR[i].
    Важно: подтверждение факта свинга происходит на баре i + n.
    """
    highs = df["high"].values
    lows = df["low"].values
    atr_vals = df["ATR"].values
    closes = df["close"].values
    
    # Расчет Z-score волатильности (rolling 100)
    # Используем относительную волатильность ATR/Close
    rel_vol = atr_vals / np.maximum(closes, 1e-8)
    vol_mean = pd.Series(rel_vol).rolling(100, min_periods=20).mean().values
    vol_std = pd.Series(rel_vol).rolling(100, min_periods=20).std().values
    
    out = []
    length = len(df)
    for i in range(n, length - n):
        window_h = highs[i-n:i+n+1]
        window_l = lows[i-n:i+n+1]
        if highs[i] == window_h.max() and np.sum(window_h == highs[i]) == 1:
            # Динамический порог (Z-score волатильности)
            z = 0.0
            if not np.isnan(vol_std[i]) and vol_std[i] > 0:
                z = (rel_vol[i] - vol_mean[i]) / vol_std[i]
            
            # Адаптируем atr_k: если волатильность высокая (z > 0), порог растёт
            dynamic_k = atr_k * (1.0 + max(0.0, z * 0.5))
            
            if (highs[i] - window_l.min()) > (dynamic_k * atr_vals[i]):
                out.append(i)
    return out

def find_swing_lows(df: pd.DataFrame, n: int, atr_k: float) -> list:
    """
    Находит индексы i, где Low[i] является локальным минимумом на интервале [i-n, i+n]
    и max(High[i-n:i+n]) - Low[i] > dynamic_k * ATR[i].
    """
    highs = df["high"].values
    lows = df["low"].values
    atr_vals = df["ATR"].values
    closes = df["close"].values
    
    # Расчет Z-score волатильности (rolling 100)
    # Используем относительную волатильность ATR/Close
    rel_vol = atr_vals / np.maximum(closes, 1e-8)
    vol_mean = pd.Series(rel_vol).rolling(100, min_periods=20).mean().values
    vol_std = pd.Series(rel_vol).rolling(100, min_periods=20).std().values
    
    out = []
    length = len(df)
    for i in range(n, length - n):
        window_h = highs[i-n:i+n+1]
        window_l = lows[i-n:i+n+1]
        if lows[i] == window_l.min() and np.sum(window_l == lows[i]) == 1:
            # Динамический порог (Z-score волатильности)
            z = 0.0
            if not np.isnan(vol_std[i]) and vol_std[i] > 0:
                z = (rel_vol[i] - vol_mean[i]) / vol_std[i]
            
            # Адаптируем atr_k: если волатильность высокая (z > 0), порог растёт
            dynamic_k = atr_k * (1.0 + max(0.0, z * 0.5))
            
            if (window_h.max() - lows[i]) > (dynamic_k * atr_vals[i]):
                out.append(i)
    return out
