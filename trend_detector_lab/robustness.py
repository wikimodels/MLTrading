import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_parquet("trend_detector_lab/data/features.parquet")
df["atr_expansion"] = df["atr"] > df.groupby("coin")["atr"].shift(1)

def calc_ci(p, n):
    if n < 5: return 0.0
    # 95% CI simple approximation
    return 1.96 * np.sqrt(p * (1 - p) / n)

print("=== ПРОВЕРКА НА УСТОЙЧИВОСТЬ (ROBUSTNESS CHECK) ===\n")

for X in [6, 8, 10, 12, 15]:
    col = f"streak_le{X}"
    if col not in df.columns: continue
    
    print(f"--- Порог короткой сделки: <= {X} свечей ---")
    
    # 1. Exact Streak (Точный)
    df["streak_bin"] = df[col].clip(upper=7)
    gb = df.groupby("streak_bin")
    res = gb.agg(n=("coin", "count"), p_3atr=("reached_3atr", "mean"))
    
    # 2. Cumulative Streak (Хвосты >= N)
    cum_res = []
    for N in range(8):
        sub = df[df[col] >= N]
        cum_res.append({
            "n": len(sub),
            "p_3atr": sub["reached_3atr"].mean() if len(sub) > 0 else 0
        })
        
    print(f"{'N':<4} | {'Exact P(3ATR) ± 95% CI':<30} | {'Cumulative (>=N) P(3ATR) ± 95% CI':<35}")
    for i in range(8):
        exact_n = int(res.loc[i, "n"]) if i in res.index else 0
        exact_p = res.loc[i, "p_3atr"] if i in res.index else 0
        exact_ci = calc_ci(exact_p, exact_n)
        
        cum_n = cum_res[i]["n"]
        cum_p = cum_res[i]["p_3atr"]
        cum_ci = calc_ci(cum_p, cum_n)
        
        label = f"{i}+" if i == 7 else str(i)
        
        # Форматирование
        s_exact = f"{exact_p:.1%} ± {exact_ci:.1%} (n={exact_n})" if exact_n > 0 else "N/A"
        s_cum   = f"{cum_p:.1%} ± {cum_ci:.1%} (n={cum_n})" if cum_n > 0 else "N/A"
        
        print(f"{label:<4} | {s_exact:<30} | {s_cum:<35}")
    print()
