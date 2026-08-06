import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from pathlib import Path

LAB = Path(__file__).resolve().parent
DATA_DIR = LAB / "data"

def calc_ci(p, n):
    if n < 5: return 0.0
    return 1.96 * np.sqrt(p * (1 - p) / n)

def evaluate_condition(df, mask, target_col="reached_3atr"):
    base_p = df[target_col].mean()
    n = mask.sum()
    if n == 0:
        return 0, 0, 0, 0
    p = df.loc[mask, target_col].mean()
    lift = p / base_p if base_p > 0 else 0
    ci = calc_ci(p, n)
    return n, p, lift, ci

def run_feature_ranking():
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    print(f"Loaded {len(df)} signals.")
    target = "reached_3atr"
    base_p = df[target].mean()
    print(f"Base P(3ATR) = {base_p:.1%}\n")
    
    # Define features and thresholds for transition testing
    configs = [
        # Z-Score Transitions (Cross Up)
        ("zscore_tr_20", "cross_up", -1.0, 1.0),
        ("zscore_tr_20", "cross_up", -1.0, 1.5),
        ("zscore_tr_20", "cross_up", -1.0, 2.0),
        ("zscore_tr_20", "cross_up", -0.5, 1.5),
        ("zscore_tr_20", "cross_up", 0.0, 2.0),
        
        # Squeeze Transitions (Cross Up out of squeeze)
        ("bb_kc_ratio_20", "cross_up", 1.0, 1.01), 
        ("bb_kc_ratio_20", "cross_up", 1.0, 1.05),
        ("bb_kc_ratio_20", "cross_up", 0.95, 1.05),
        
        # ATR Percentile Transitions (Cross Up from low volatility)
        ("atr_percentile_100", "cross_up", 0.10, 0.20),
        ("atr_percentile_100", "cross_up", 0.10, 0.30),
        ("atr_percentile_100", "cross_up", 0.20, 0.50),
        
        # Donchian Percentile
        ("donchian_width_percentile_100", "cross_up", 0.10, 0.30),
        ("donchian_width_percentile_100", "cross_up", 0.20, 0.40),
        
        # ADX (Cross Up)
        ("adx_14", "cross_up", 20, 25),
        ("adx_14", "cross_up", 15, 20),
    ]
    
    results = []
    
    for col, ctype, t1, t2 in configs:
        for L in [3, 5, 8, 13]:
            min_col = f"{col}_min_{L}"
            if min_col not in df.columns: continue
                
            mask = (df[min_col] <= t1) & (df[col] >= t2)
            n, p, lift, ci = evaluate_condition(df, mask, target)
            
            results.append({
                "Feature": col,
                "Type": f"{ctype} ({t1} -> {t2})",
                "Lookback": L,
                "N": n,
                "P(3ATR)": p,
                "Lift": lift,
                "CI": ci
            })
            
    # Also evaluate simple thresholds (ROC-style)
    thresholds = {
        "zscore_tr_20": [0.0, 1.0, 1.5, 2.0, 2.5],
        "atr_percentile_100": [0.05, 0.10, 0.20, 0.30, 0.50],
        "donchian_width_percentile_100": [0.05, 0.10, 0.20, 0.30, 0.50],
        "bb_kc_ratio_20": [0.9, 0.95, 1.0, 1.05, 1.1],
        "adx_14": [15, 20, 25, 30]
    }
    
    for col, ths in thresholds.items():
        if col not in df.columns: continue
        for th in ths:
            mask = df[col] >= th
            n, p, lift, ci = evaluate_condition(df, mask, target)
            results.append({
                "Feature": col,
                "Type": f"level >= {th}",
                "Lookback": 0,
                "N": n,
                "P(3ATR)": p,
                "Lift": lift,
                "CI": ci
            })
            
            mask_less = df[col] <= th
            n, p, lift, ci = evaluate_condition(df, mask_less, target)
            results.append({
                "Feature": col,
                "Type": f"level <= {th}",
                "Lookback": 0,
                "N": n,
                "P(3ATR)": p,
                "Lift": lift,
                "CI": ci
            })

    res_df = pd.DataFrame(results)
    if len(res_df) == 0:
        print("No results computed.")
        return
        
    valid = res_df[res_df["N"] >= 100].sort_values("Lift", ascending=False)
    
    print(f"{'Feature':<30} | {'Condition':<25} | {'L':<3} | {'N':<5} | {'P(3ATR)':<8} | {'Lift':<5} | {'CI'}")
    print("-" * 100)
    for _, row in valid.head(30).iterrows():
        s_p = f"{row['P(3ATR)']:.1%}"
        s_ci = f"± {row['CI']:.1%}"
        print(f"{row['Feature']:<30} | {row['Type']:<25} | {row['Lookback']:<3} | {row['N']:<5} | {s_p:<8} | {row['Lift']:<5.2f} | {s_ci}")
        
    print("\n\n=== ROC: ATR Percentile ===")
    roc_atr = res_df[(res_df["Feature"] == "atr_percentile_100") & (res_df["Type"].str.startswith("level <="))]
    for _, row in roc_atr.sort_values("Type").iterrows():
        print(f"{row['Type']:<20} N={row['N']:<5}  P(3ATR)={row['P(3ATR)']:.1%} (Lift {row['Lift']:.2f})")

if __name__ == "__main__":
    run_feature_ranking()
