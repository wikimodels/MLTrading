import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_parquet("trend_detector_lab/data/features.parquet")
OUT = Path("trend_detector_lab/output")
OUT.mkdir(exist_ok=True)

TARGETS = ["reached_2atr","reached_3atr","reached_5atr","reached_8atr","bars_gt_15","bars_gt_25","bars_gt_50"]
BASELINES = {t: float(df[t].mean()) for t in TARGETS if t in df.columns}

def lift(p, base):
    return round(p / base, 3) if base > 0 else None

# ── 1. BASELINE DISTRIBUTION ──────────────────────────────────────────────────
rows = []
for t in [1,2,3,4,5,6,7,8,10,12,15,20,25,30,50,75,100]:
    rows.append({
        "bars": t,
        "pct_equal": round((df.bars_in_trade == t).mean()*100, 2),
        "pct_gt":    round((df.bars_in_trade >  t).mean()*100, 2),
        "count_gt":  int((df.bars_in_trade > t).sum()),
    })
pd.DataFrame(rows).to_csv(OUT/"01_baseline_distribution.csv", index=False)
print("Saved 01_baseline_distribution.csv")

# ── 2. BASELINE ATR TARGETS ───────────────────────────────────────────────────
rows = [{"target": t, "probability": round(v,4), "count": int((df[t]).sum())}
        for t,v in BASELINES.items()]
pd.DataFrame(rows).to_csv(OUT/"02_baseline_atr_targets.csv", index=False)
print("Saved 02_baseline_atr_targets.csv")

# ── 3. STREAK GRID (all targets) ──────────────────────────────────────────────
rows = []
for X in [4, 5, 6, 8, 10, 12]:
    col = "streak_le%d" % X
    if col not in df.columns: continue
    for N in [2, 3, 4, 5, 6, 7]:
        mask = df[col] >= N
        sub  = df[mask]
        if len(sub) < 20: continue
        row = {"streak_threshold": X, "streak_count": N,
               "condition": "%dx<=%d" % (N,X), "count": len(sub),
               "pct_of_total": round(len(sub)/len(df)*100,1)}
        for t in TARGETS:
            if t in sub.columns:
                p = float(sub[t].mean())
                row["p_"+t]    = round(p, 4)
                row["lift_"+t] = lift(p, BASELINES[t])
        rows.append(row)
pd.DataFrame(rows).to_csv(OUT/"03_streak_grid.csv", index=False)
print("Saved 03_streak_grid.csv")

# ── 4. CUMSUM GRID ────────────────────────────────────────────────────────────
rows = []
for X in [4, 5, 6, 8, 10, 12]:
    col = "cumsum_le%d" % X
    if col not in df.columns: continue
    for C in [10, 20, 30, 40, 50, 60, 80, 100, 120]:
        mask = df[col] >= C
        sub  = df[mask]
        if len(sub) < 20: continue
        row = {"cumsum_threshold": X, "cumsum_min_bars": C,
               "condition": "cumsum_le%d>=%d" % (X,C), "count": len(sub),
               "pct_of_total": round(len(sub)/len(df)*100,1)}
        for t in TARGETS:
            if t in sub.columns:
                p = float(sub[t].mean())
                row["p_"+t]    = round(p, 4)
                row["lift_"+t] = lift(p, BASELINES[t])
        rows.append(row)
pd.DataFrame(rows).to_csv(OUT/"04_cumsum_grid.csv", index=False)
print("Saved 04_cumsum_grid.csv")

# ── 5. BARS_SINCE_BIG GRID ────────────────────────────────────────────────────
rows = []
for B in [15, 25, 50]:
    col = "bars_since_big%d" % B
    if col not in df.columns: continue
    for C in [10,20,30,40,50,60,70,80,90,100,120,150,200]:
        mask = df[col] >= C
        sub  = df[mask]
        if len(sub) < 20: continue
        row = {"big_threshold": B, "since_min_bars": C,
               "condition": "bars_since_big%d>=%d" % (B,C),
               "count": len(sub),
               "pct_of_total": round(len(sub)/len(df)*100,1)}
        for t in TARGETS:
            if t in sub.columns:
                p = float(sub[t].mean())
                row["p_"+t]    = round(p, 4)
                row["lift_"+t] = lift(p, BASELINES[t])
        rows.append(row)
pd.DataFrame(rows).to_csv(OUT/"05_bars_since_big_grid.csv", index=False)
print("Saved 05_bars_since_big_grid.csv")

# ── 6. COMBO: bars_since_big15 + streak ───────────────────────────────────────
rows = []
for C in [60,80,100,120,150]:
    for X in [6,8,10,12]:
        for N in [2,3,4,5,6]:
            mask = (df["bars_since_big15"] >= C) & (df["streak_le%d"%X] >= N)
            sub  = df[mask]
            if len(sub) < 20: continue
            row = {"big15_min": C, "streak_threshold": X, "streak_count": N,
                   "condition": "big15>=%d AND %dx<=%d"%(C,N,X),
                   "count": len(sub),
                   "pct_of_total": round(len(sub)/len(df)*100,1)}
            for t in TARGETS:
                if t in sub.columns:
                    p = float(sub[t].mean())
                    row["p_"+t]    = round(p, 4)
                    row["lift_"+t] = lift(p, BASELINES[t])
            rows.append(row)
pd.DataFrame(rows).to_csv(OUT/"06_combo_big15_streak.csv", index=False)
print("Saved 06_combo_big15_streak.csv")

# ── 7. PER SYMBOL BASELINE ────────────────────────────────────────────────────
rows = []
for sym, g in df.groupby("coin"):
    row = {"coin": sym, "n_trades": len(g),
           "avg_bars": round(g.bars_in_trade.mean(),1),
           "median_bars": float(g.bars_in_trade.median()),
           "max_bars": int(g.bars_in_trade.max()),
           "pct_1bar": round((g.bars_in_trade==1).mean()*100,1),
           "avg_atr_pct": round(g.atr_pct.mean(),3)}
    for t in TARGETS:
        if t in g.columns:
            row["p_"+t] = round(float(g[t].mean()),4)
    rows.append(row)
pd.DataFrame(rows).sort_values("avg_bars", ascending=False).to_csv(OUT/"07_per_symbol_baseline.csv", index=False)
print("Saved 07_per_symbol_baseline.csv")

# ── 8. BARS_SINCE_BIG15 BY DIRECTION ─────────────────────────────────────────
rows = []
for sig in ["long","short"]:
    for C in [60,80,100,120,150]:
        sub_sig = df[df.signal==sig]
        mask = sub_sig["bars_since_big15"] >= C
        sub  = sub_sig[mask]
        if len(sub) < 15: continue
        row = {"signal": sig, "big15_min": C, "count": len(sub),
               "n_total_direction": len(sub_sig)}
        for t in TARGETS:
            if t in sub.columns and t in sub_sig.columns:
                p    = float(sub[t].mean())
                base = float(sub_sig[t].mean())
                row["p_"+t]    = round(p, 4)
                row["base_"+t] = round(base, 4)
                row["lift_"+t] = lift(p, base)
        rows.append(row)
pd.DataFrame(rows).to_csv(OUT/"08_big15_by_direction.csv", index=False)
print("Saved 08_big15_by_direction.csv")

# ── 9. FULL EXCEL (все листы) ─────────────────────────────────────────────────
sheets = {
    "01_baseline_dist":       pd.read_csv(OUT/"01_baseline_distribution.csv"),
    "02_baseline_atr":        pd.read_csv(OUT/"02_baseline_atr_targets.csv"),
    "03_streak_grid":         pd.read_csv(OUT/"03_streak_grid.csv"),
    "04_cumsum_grid":         pd.read_csv(OUT/"04_cumsum_grid.csv"),
    "05_bars_since_big":      pd.read_csv(OUT/"05_bars_since_big_grid.csv"),
    "06_combo_big15_streak":  pd.read_csv(OUT/"06_combo_big15_streak.csv"),
    "07_per_symbol":          pd.read_csv(OUT/"07_per_symbol_baseline.csv"),
    "08_big15_by_direction":  pd.read_csv(OUT/"08_big15_by_direction.csv"),
}
with pd.ExcelWriter(OUT/"research_v1.xlsx", engine="openpyxl") as w:
    for sheet, data in sheets.items():
        data.to_excel(w, sheet_name=sheet[:31], index=False)
print("Saved research_v1.xlsx")

print()
print("=== TOP FINDINGS ===")
big = pd.read_csv(OUT/"05_bars_since_big_grid.csv")
top = big[big["lift_reached_3atr"] > 1.0].sort_values("lift_reached_3atr", ascending=False)
print(top[["condition","count","p_reached_3atr","lift_reached_3atr","p_reached_5atr","lift_reached_5atr"]].to_string(index=False))
