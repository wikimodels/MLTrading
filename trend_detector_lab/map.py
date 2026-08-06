import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_parquet("trend_detector_lab/data/features.parquet")
df["atr_expansion"] = df["atr"] > df.groupby("coin")["atr"].shift(1)

for X in [8, 12, 15]:
    col = f"streak_le{X}"
    if col not in df.columns:
        continue
    
    print(f"=== Streak <= {X} bars map ===")
    
    df["streak_bin"] = df[col].clip(upper=7)
    
    gb = df.groupby("streak_bin")
    res = gb.agg(
        n=("coin", "count"),
        p_3atr=("reached_3atr", "mean"),
        p_5atr=("reached_5atr", "mean"),
        p_gt_15=("bars_gt_15", "mean"),
        atr_exp=("atr_expansion", "mean"),
        avg_bars=("bars_in_trade", "mean")
    )
    
    print(f"{'Streak':<8} {'N':<6} {'P(3ATR)':<8} {'P(5ATR)':<8} {'P(>15b)':<8} {'ATR Exp':<8} {'AvgBars':<8}")
    for i, row in res.iterrows():
        label = f"{i}+" if i == 7 else str(int(i))
        print(f"{label:<8} {int(row.n):<6} {row.p_3atr:.3f}   {row.p_5atr:.3f}   {row.p_gt_15:.3f}   {row.atr_exp:.3f}   {row.avg_bars:.1f}")
    print()
