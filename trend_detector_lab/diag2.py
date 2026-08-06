import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import pandas as pd
import numpy as np

df = pd.read_parquet("trend_detector_lab/data/features.parquet")

col = "bars_since_big15"
base3 = df["reached_3atr"].mean()
base5 = df["reached_5atr"].mean()
base8 = df["reached_8atr"].mean()
baseb15 = df["bars_gt_15"].mean()
baseb25 = df["bars_gt_25"].mean()

print("=== DEEP DIVE: bars_since_big15 ===")
print("baseline: P(3ATR)=%.3f  P(5ATR)=%.3f  P(8ATR)=%.3f  P(>15bars)=%.3f  P(>25bars)=%.3f" % (
    base3, base5, base8, baseb15, baseb25))
print()
print("%-20s %6s  P(3ATR) lift   P(5ATR) lift   P(8ATR) lift   P(>25) lift" % ("threshold","n"))
for C in [20, 30, 40, 50, 60, 80, 100, 120, 150, 200]:
    mask = df[col] >= C
    sub = df[mask]
    if len(sub) < 30:
        break
    p3 = sub["reached_3atr"].mean()
    p5 = sub["reached_5atr"].mean()
    p8 = sub["reached_8atr"].mean()
    pb25 = sub["bars_gt_25"].mean()
    print(">=%3d bars  %6d   %.3f  %.2f   %.3f  %.2f   %.3f  %.2f   %.3f  %.2f" % (
        C, len(sub), p3, p3/base3, p5, p5/base5, p8, p8/base8, pb25, pb25/baseb25))

print()
print("=== bars_since_big15 + streak_le8 combo ===")
print("%-30s %6s  P(3ATR) lift   P(5ATR) lift" % ("condition","n"))
for C in [60, 100, 150]:
    for N in [2, 3, 4]:
        mask = (df["bars_since_big15"] >= C) & (df["streak_le8"] >= N)
        sub = df[mask]
        if len(sub) < 30:
            continue
        p3 = sub["reached_3atr"].mean()
        p5 = sub["reached_5atr"].mean()
        label = "big15>=%d AND streak>=%d" % (C, N)
        print("%-30s %6d   %.3f  %.2f   %.3f  %.2f" % (label, len(sub), p3, p3/base3, p5, p5/base5))

print()
print("=== bars_since_big15 by signal direction ===")
for sig in ["long","short"]:
    sub_sig = df[df.signal == sig]
    mask = sub_sig["bars_since_big15"] >= 100
    sub = sub_sig[mask]
    if len(sub) < 10:
        continue
    p3 = sub["reached_3atr"].mean()
    base3_sig = sub_sig["reached_3atr"].mean()
    print("signal=%s  n_filtered=%d  P(3ATR)=%.3f  baseline=%.3f  lift=%.2f" % (
        sig, len(sub), p3, base3_sig, p3/base3_sig if base3_sig>0 else 0))
