import numpy as np
import pandas as pd
from diver_backtest.config import Config

def detect_divergence(df: pd.DataFrame, indicator: str,
                      swing_idx: list, kind: str,
                      min_bars: int, max_bars: int,
                      threshold: float, sigma_window: int = 100,
                      swing_n: int = 3) -> list:
    """
    kind: 'high' (bearish candidates) or 'low' (bullish candidates).
    direction: +1 для Short (Bearish), -1 для Long (Bullish).
    """
    if len(swing_idx) < 2:
        return []
    
    ind = df[indicator].values
    atr_vals = df["ATR"].values
    sigma = df[indicator].rolling(sigma_window, min_periods=20).std().values
    highs = df["high"].values
    lows = df["low"].values
    length = len(df)

    results = []
    for a in range(len(swing_idx) - 1):
        i1 = swing_idx[a]
        for b in range(a + 1, len(swing_idx)):
            i2 = swing_idx[b]
            gap = i2 - i1
            if gap < min_bars:
                continue
            if gap > max_bars:
                break

            if kind == "high":
                p1 = highs[i1]
                p2 = highs[i2]
            else:
                p1 = lows[i1]
                p2 = lows[i2]

            i1v, i2v = ind[i1], ind[i2]
            if np.isnan(i1v) or np.isnan(i2v):
                continue
            s = sigma[i2]
            if np.isnan(s) or s == 0:
                continue

            a_val = atr_vals[i2]
            if np.isnan(a_val) or a_val == 0:
                continue

            price_term = (p2 - p1) / a_val
            ind_term = (i2v - i1v) / s
            D = price_term - ind_term

            confirm_idx = min(i2 + swing_n + 1, length - 1)

            if kind == "high":
                # Regular Bearish: price HH (p2 > p1), indicator LH (i2v < i1v) => Short (+1)
                if p2 > p1 and i2v < i1v and D > threshold:
                    results.append({
                        "p1_idx": i1, "p2_idx": i2, "confirm_idx": confirm_idx,
                        "D": D, "direction": +1, "div_type": "regular_bear"
                    })
                # Hidden Bearish: price LH (p2 < p1), indicator HH (i2v > i1v) => Short (+1)
                elif p2 < p1 and i2v > i1v and D < -threshold:
                    results.append({
                        "p1_idx": i1, "p2_idx": i2, "confirm_idx": confirm_idx,
                        "D": D, "direction": +1, "div_type": "hidden_bear"
                    })
            else:
                # Regular Bullish: price LL (p2 < p1), indicator HL (i2v > i1v) => Long (-1)
                if p2 < p1 and i2v > i1v and D < -threshold:
                    results.append({
                        "p1_idx": i1, "p2_idx": i2, "confirm_idx": confirm_idx,
                        "D": D, "direction": -1, "div_type": "regular_bull"
                    })
                # Hidden Bullish: price HL (p2 > p1), indicator LL (i2v < i1v) => Long (-1)
                elif p2 > p1 and i2v < i1v and D > threshold:
                    results.append({
                        "p1_idx": i1, "p2_idx": i2, "confirm_idx": confirm_idx,
                        "D": D, "direction": -1, "div_type": "hidden_bull"
                    })
    return results

def all_divergences(df: pd.DataFrame, indicators: list,
                    swing_highs: list, swing_lows: list,
                    cfg: Config, tf: str = "4h") -> dict:
    """
    Возвращает {indicator: [ {'p1_idx','p2_idx','confirm_idx','D','direction','side','div_type'} ]}
    """
    out = {}
    sn = cfg.swing_n.get(tf, 3)
    for ind in indicators:
        if ind not in df.columns:
            continue
        res = []
        for d in detect_divergence(df, ind, swing_highs, "high",
                                   cfg.window_min_bars, cfg.window_max_bars,
                                   cfg.divergence_threshold, swing_n=sn):
            d["side"] = "high"
            d["indicator"] = ind
            res.append(d)
        for d in detect_divergence(df, ind, swing_lows, "low",
                                   cfg.window_min_bars, cfg.window_max_bars,
                                   cfg.divergence_threshold, swing_n=sn):
            d["side"] = "low"
            d["indicator"] = ind
            res.append(d)
        out[ind] = res
    return out
