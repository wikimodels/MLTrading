import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from matplotlib.backends.backend_pdf import PdfPages

# Ensure the output directory exists
REPORT_DIR = Path("edge_report")
REPORT_DIR.mkdir(exist_ok=True)

DATA_PATH = Path("trend_detector_lab/data/features_eq.parquet")

TP_MULT = 3.0
SL_MULT = 1.0
COMM_TAKER = 0.15  # % per side
COMM_MAKER = 0.07
COMM_STRESS = 0.20

WINDOW = 60 # Default robust window for analysis

def calc_ees(df, comm=COMM_TAKER):
    # EES: Entry Edge Score (Expected R)
    comm_atr = (comm * 2) / df["atr_pct"]
    
    r_tp = TP_MULT - comm_atr
    r_sl = -SL_MULT - comm_atr
    r_to = 0.0 - comm_atr
    
    res = np.where(df[f"exit_reason_{WINDOW}"] == "tp", r_tp,
          np.where(df[f"exit_reason_{WINDOW}"] == "sl", r_sl, r_to))
    
    return res

def calc_ci(data, alpha=0.95):
    if len(data) < 2: return 0.0
    se = np.std(data, ddof=1) / np.sqrt(len(data))
    return 1.96 * se

def evaluate_filter(df, mask, comm=COMM_TAKER):
    sub = df[mask]
    n = len(sub)
    if n == 0:
        return 0, 0, 0, 0, 0, 0
        
    p_tp = (sub[f"exit_reason_{WINDOW}"] == "tp").mean()
    mfe_med = sub[f"mfe_{WINDOW}"].median()
    mae_med = sub[f"mae_{WINDOW}"].median()
    
    ees_arr = calc_ees(sub, comm)
    ees_mean = ees_arr.mean()
    ees_ci = calc_ci(ees_arr)
    
    return n, p_tp, mfe_med, mae_med, ees_mean, ees_ci

def module_1_walk_forward(df, mask, name):
    df["half_year"] = df["signal_ts"].dt.year.astype(str) + "-H" + np.where(df["signal_ts"].dt.month <= 6, "1", "2")
    
    res = []
    periods = sorted(df["half_year"].unique())
    for p in periods:
        p_df = df[df["half_year"] == p]
        p_mask = mask & (df["half_year"] == p)
        
        base_ees = calc_ees(p_df).mean() if len(p_df) > 0 else 0
        
        n, p_tp, _, _, ees, ci = evaluate_filter(df, p_mask)
        lift_ees = ees - base_ees
        
        if n > 0:
            res.append({"Period": p, "N": n, "P(3ATR)": p_tp, "EES": ees, "EES_Lift": lift_ees, "CI": ci})
    
    res_df = pd.DataFrame(res)
    res_df.to_excel(REPORT_DIR / f"walk_forward_{name}.xlsx", index=False)
    print(f"\n--- Walk Forward: {name} ---")
    print(res_df.to_string())

def module_2_edge_decay(df, mask, name):
    train_mask = df["signal_ts"].dt.year <= 2024
    test_mask = df["signal_ts"].dt.year >= 2025
    
    n_tr, p_tr, _, _, ees_tr, _ = evaluate_filter(df, mask & train_mask)
    n_te, p_te, _, _, ees_te, _ = evaluate_filter(df, mask & test_mask)
    
    print(f"\n--- Edge Decay: {name} ---")
    print(f"Train (2023-24): N={n_tr}, P(3ATR)={p_tr:.1%}, EES={ees_tr:.2f}R")
    print(f"Test (2025-26):  N={n_te}, P(3ATR)={p_te:.1%}, EES={ees_te:.2f}R")

def module_3_per_coin(df, mask, name):
    res = []
    coins = sorted(df["coin"].unique())
    for c in coins:
        c_df = df[df["coin"] == c]
        c_mask = mask & (df["coin"] == c)
        
        base_ees = calc_ees(c_df).mean() if len(c_df) > 0 else 0
        n, p_tp, _, _, ees, ci = evaluate_filter(df, c_mask)
        
        if n > 0:
            res.append({"Coin": c, "N": n, "P(3ATR)": p_tp, "EES": ees, "Base_EES": base_ees, "EES_Lift": ees - base_ees})
            
    res_df = pd.DataFrame(res)
    res_df.to_excel(REPORT_DIR / f"per_coin_{name}.xlsx", index=False)
    
    positive_coins = (res_df["EES"] > 0).sum()
    mean_lift = res_df["EES_Lift"].mean()
    med_lift = res_df["EES_Lift"].median()
    
    print(f"\n--- Per-Coin Stability: {name} ---")
    print(f"Positive EES Coins (Sign Test): {positive_coins} / {len(res_df)}")
    print(f"Mean EES Lift: {mean_lift:.2f}R | Median EES Lift: {med_lift:.2f}R")

def module_4_mutual_info(df):
    print("\n--- Mutual Information ---")
    features = ["zscore_tr_20", "bb_kc_ratio_20", "atr_percentile_100", "donchian_width_percentile_100", "adx_14"]
    target = (df[f"exit_reason_{WINDOW}"] == "tp").astype(int)
    
    discretized = pd.DataFrame()
    for f in features:
        if f in df.columns:
            discretized[f] = pd.qcut(df[f].fillna(df[f].median()), q=10, duplicates='drop')
            discretized[f] = discretized[f].cat.codes
            
    mi_target = {}
    for f in discretized.columns:
        mi = mutual_info_score(discretized[f], target)
        mi_target[f] = mi
        
    print("MI with Target P(3ATR):")
    for f, mi in sorted(mi_target.items(), key=lambda x: x[1], reverse=True):
        print(f"  {f:<30}: {mi:.5f}")

def module_5_roc_curves(df):
    print("\n--- ROC Curves ---")
    thresholds = {
        "bb_kc_ratio_20": np.arange(0.9, 1.12, 0.02),
        "donchian_width_percentile_100": np.arange(0.05, 0.55, 0.05),
    }
    
    with PdfPages(REPORT_DIR / "roc_curves.pdf") as pdf:
        for f, ths in thresholds.items():
            if f not in df.columns: continue
            
            p_vals = []
            ees_vals = []
            x_vals = []
            for th in ths:
                mask = df[f] >= th if "bb" in f else df[f] <= th
                n, p_tp, _, _, ees, _ = evaluate_filter(df, mask)
                if n > 100:
                    p_vals.append(p_tp)
                    ees_vals.append(ees)
                    x_vals.append(th)
                    
            if len(x_vals) > 0:
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.plot(x_vals, p_vals, marker='o')
                plt.title(f"{f} -> P(3ATR)")
                plt.subplot(1, 2, 2)
                plt.plot(x_vals, ees_vals, marker='o', color='orange')
                plt.title(f"{f} -> EES")
                plt.tight_layout()
                pdf.savefig()
                plt.close()
    print("ROC curves saved to edge_report/roc_curves.pdf")

def module_6_bootstrap_monte_carlo(df, mask, name):
    sub = df[mask]
    n = len(sub)
    if n < 50: return
    
    ees_arr = calc_ees(sub)
    
    # Bootstrap
    np.random.seed(42)
    boot_means = []
    for _ in range(5000):
        sample = np.random.choice(ees_arr, size=n, replace=True)
        boot_means.append(np.mean(sample))
        
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)
    
    print(f"\n--- Bootstrap (5000) for {name} ---")
    print(f"EES: {np.mean(ees_arr):.2f}R  [95% CI: {ci_lower:.2f}R, {ci_upper:.2f}R]")
    
    pd.DataFrame({"EES_mean": boot_means}).to_excel(REPORT_DIR / f"bootstrap_{name}.xlsx", index=False)
    
    # Monte Carlo Shuffle
    with PdfPages(REPORT_DIR / f"equity_curves_{name}.pdf") as pdf:
        plt.figure(figsize=(10, 6))
        for _ in range(1000):
            np.random.shuffle(ees_arr)
            plt.plot(np.cumsum(ees_arr), color='blue', alpha=0.01)
            
        plt.plot(np.cumsum(calc_ees(sub)), color='red', linewidth=2, label="Actual Sequence")
        plt.title(f"Monte Carlo Shuffle (1000 paths) - {name}")
        plt.xlabel("Trade Number")
        plt.ylabel("Cumulative R")
        plt.legend()
        pdf.savefig()
        plt.close()

def module_7_commission_stress(df, mask, name):
    print(f"\n--- Commission Stress Test: {name} ---")
    for comm, label in [(COMM_MAKER, "Maker (0.07%)"), (COMM_TAKER, "Taker (0.15%)"), (COMM_STRESS, "Stress (0.20%)")]:
        _, p_tp, _, _, ees, _ = evaluate_filter(df, mask, comm=comm)
        print(f"  {label:<15} EES = {ees:.2f}R")

def run_all():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df)} signals.")
    
    base_ees = calc_ees(df).mean()
    base_tp = (df[f"exit_reason_{WINDOW}"] == "tp").mean()
    print(f"Baseline: P(3ATR) = {base_tp:.1%}, EES = {base_ees:.2f}R")
    
    module_4_mutual_info(df)
    module_5_roc_curves(df)
    
    filters = {
        "Squeeze_Breakout_L3": (df["bb_kc_ratio_20_min_3"] <= 0.95) & (df["bb_kc_ratio_20"] >= 1.05),
        "Donchian_Compression": (df["donchian_width_percentile_100"] <= 0.05),
        "ATR_Percentile_10_20": (df["atr_percentile_100_min_5"] <= 0.10) & (df["atr_percentile_100"] >= 0.20),
    }
    
    for name, mask in filters.items():
        if mask.sum() < 50:
            print(f"Skipping {name}, too few trades: {mask.sum()}")
            continue
            
        print(f"\n{'='*50}\nEvaluating Filter: {name} (N={mask.sum()})\n{'='*50}")
        
        module_1_walk_forward(df, mask, name)
        module_2_edge_decay(df, mask, name)
        module_3_per_coin(df, mask, name)
        module_6_bootstrap_monte_carlo(df, mask, name)
        module_7_commission_stress(df, mask, name)
        
    print("\nDone. Reports saved to edge_report/")

if __name__ == "__main__":
    run_all()
