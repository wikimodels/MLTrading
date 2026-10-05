import numpy as np

def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple:
    """
    Доверительный интервал Уилсона для биномиального распределения (винрейта).
    """
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))

def compute_max_dd(returns: np.ndarray) -> float:
    """
    Максимальная просадка от пика эквити.
    """
    if len(returns) == 0:
        return 0.0
    eq = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(dd.min())

def summarize(trades: list) -> dict:
    """
    Полная статистическая сводка серии сделок.
    """
    n = len(trades)
    if n == 0:
        return {
            "n": 0, "wins": 0, "winrate": 0.0, "ci_low": 0.0, "ci_high": 0.0,
            "avg_return": 0.0, "median_return": 0.0, "std_return": 0.0,
            "sharpe_per_trade": 0.0, "max_dd": 0.0, "profit_factor": 0.0
        }

    wins = sum(t["win"] for t in trades)
    p = wins / n
    lo, hi = wilson_ci(p, n)
    returns = np.array([t["net_return"] for t in trades])

    std_val = float(returns.std())
    sharpe = (float(returns.mean()) / std_val) if std_val > 1e-9 else 0.0

    pos_rets = returns[returns > 0]
    neg_rets = returns[returns < 0]
    if len(neg_rets) > 0 and neg_rets.sum() != 0:
        profit_factor = float(pos_rets.sum() / -neg_rets.sum())
    else:
        profit_factor = 999.0 if len(pos_rets) > 0 else 0.0

    return {
        "n": n,
        "wins": wins,
        "winrate": round(p, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "avg_return": round(float(returns.mean()), 5),
        "median_return": round(float(np.median(returns)), 5),
        "std_return": round(std_val, 5),
        "sharpe_per_trade": round(sharpe, 2),
        "max_dd": round(compute_max_dd(returns), 4),
        "profit_factor": round(profit_factor, 2),
    }
