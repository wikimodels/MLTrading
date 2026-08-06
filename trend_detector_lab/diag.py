import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
import pandas as pd
import numpy as np

df = pd.read_parquet("trend_detector_lab/data/features.parquet")

print("=== BASELINE ===")
print("Total trades:", len(df))
print("bars_in_trade: mean=%.1f, median=%.0f, std=%.1f, max=%d" % (
    df.bars_in_trade.mean(), df.bars_in_trade.median(),
    df.bars_in_trade.std(), df.bars_in_trade.max()))
print()
print("Distribution bars_in_trade:")
for t in [1, 2, 3, 5, 8, 10, 15, 25, 50]:
    eq = (df.bars_in_trade == t).mean()*100
    gt = (df.bars_in_trade > t).mean()*100
    print("  ==%2d: %5.1f%%    >%2d: %5.1f%%" % (t, eq, t, gt))

print()
print("=== ATR TARGETS (baseline) ===")
for col in ["reached_2atr","reached_3atr","reached_5atr","reached_8atr"]:
    if col in df.columns:
        print("  %-15s %.1f%%" % (col+":", df[col].mean()*100))

print()
print("=== STREAK vs ATR TARGETS ===")
print("  %-14s %6s  P(3ATR) lift3   P(5ATR) lift5" % ("condition","n"))
for X in [6, 8, 10, 12]:
    col = "streak_le%d" % X
    if col not in df.columns:
        continue
    for N in [3, 4, 5, 6]:
        mask = df[col] >= N
        sub = df[mask]
        if len(sub) < 50:
            continue
        base5 = df["reached_5atr"].mean()
        cond5 = sub["reached_5atr"].mean()
        lift5 = cond5/base5 if base5 > 0 else 0
        base3 = df["reached_3atr"].mean()
        cond3 = sub["reached_3atr"].mean()
        lift3 = cond3/base3 if base3 > 0 else 0
        print("  %dx<=%2d         %6d  %.3f   %.2f    %.3f   %.2f" % (N, X, len(sub), cond3, lift3, cond5, lift5))

print()
print("=== CUMSUM vs ATR (time in sideways) ===")
print("  %-20s %6s  P(3ATR) lift3   P(5ATR) lift5" % ("condition","n"))
for X in [6, 8, 10]:
    col = "cumsum_le%d" % X
    if col not in df.columns:
        continue
    for C in [20, 40, 60, 80]:
        mask = df[col] >= C
        sub = df[mask]
        if len(sub) < 50:
            continue
        base5 = df["reached_5atr"].mean()
        cond5 = sub["reached_5atr"].mean()
        lift5 = cond5/base5 if base5 > 0 else 0
        base3 = df["reached_3atr"].mean()
        cond3 = sub["reached_3atr"].mean()
        lift3 = cond3/base3 if base3 > 0 else 0
        print("  cumsum_le%d>=%3d  %6d  %.3f   %.2f    %.3f   %.2f" % (X, C, len(sub), cond3, lift3, cond5, lift5))

print()
print("=== BARS_SINCE_BIG vs ATR ===")
print("  %-22s %6s  P(3ATR) lift3   P(5ATR) lift5" % ("condition","n"))
for B in [15, 25, 50]:
    col = "bars_since_big%d" % B
    if col not in df.columns:
        continue
    for C in [30, 60, 100, 150]:
        mask = df[col] >= C
        sub = df[mask]
        if len(sub) < 50:
            continue
        base5 = df["reached_5atr"].mean()
        cond5 = sub["reached_5atr"].mean()
        lift5 = cond5/base5 if base5 > 0 else 0
        base3 = df["reached_3atr"].mean()
        cond3 = sub["reached_3atr"].mean()
        lift3 = cond3/base3 if base3 > 0 else 0
        print("  bars_since_big%2d>=%3d  %6d  %.3f   %.2f    %.3f   %.2f" % (B, C, len(sub), cond3, lift3, cond5, lift5))
