"""
05_statistics.py — Подсчёт вероятностей и random baseline.

Для каждого условия (streak_le_X >= N) считает:
  - count, P(>15bars), P(>25bars), P(3ATR), P(5ATR)
  - baseline (безусловная вероятность)
  - lift = P(условие) / baseline
  - random baseline: 1000 случайных выборок того же размера
  - rand_above_p95: наш P выше 95-го перцентиля случайных

Дополнительно:
  - Walk-forward (6-мес окна): lift в каждом окне
  - По монете, по ATR-режиму

Читает:  data/features.parquet
Пишет:   output/statistics.xlsx

Запуск:
    poetry run python trend_detector_lab/05_statistics.py
"""
from __future__ import annotations

import sys
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
LAB  = Path(__file__).resolve().parent

with open(LAB / "config" / "settings.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

DATA_DIR      = LAB / CFG["data"]["output_dir"]
FEATURES_PATH = DATA_DIR / "features.parquet"
OUTPUT_DIR    = LAB / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH   = OUTPUT_DIR / "statistics.xlsx"

SHORT_THRESHOLDS = CFG["features"]["short_thresholds"]  # [4,5,6,8,10,12]
RAND_ITERS       = CFG["analysis"]["random_baseline_iterations"]  # 1000
WF_MONTHS        = CFG["analysis"]["walk_forward_months"]         # 6

TARGET_BARS      = CFG["analysis"]["target_bars"]       # [10,15,25,50]
TARGET_ATR_MULT  = CFG["analysis"]["target_atr_mult"]   # [2,3,5,8]

STREAK_COUNTS    = [2, 3, 4, 5, 6, 7]

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {level} | {message}", level="INFO")


# ── helpers ───────────────────────────────────────────────────────────────────

def random_baseline(
    series: pd.Series,
    sample_size: int,
    n_iter: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    ВозвраSCHает (mean, p95) для random baseline.
    series — булевый ряд целевой метрики.
    sample_size — размер выборки (сколько сделок в нашем фильтре).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    arr = series.values.astype(float)
    if sample_size >= len(arr) or sample_size < 2:
        return float(arr.mean()), float(arr.mean())
    samples = [rng.choice(arr, size=sample_size, replace=False).mean() for _ in range(n_iter)]
    return float(np.mean(samples)), float(np.percentile(samples, 95))


def compute_grid_stats(df: pd.DataFrame, primary_target: str = "bars_gt_15") -> pd.DataFrame:
    """
    Грид-поиск по всем комбинациям (streak_le_X >= N).
    Считает count, P(target), lift, random baseline.
    """
    rng = np.random.default_rng(42)
    rows = []

    baseline_val = float(df[primary_target].mean())
    total_n      = len(df)

    for X in SHORT_THRESHOLDS:
        col = f"streak_le{X}"
        if col not in df.columns:
            continue
        for N in STREAK_COUNTS:
            mask   = df[col] >= N
            subset = df[mask]
            count  = len(subset)
            if count < 20:
                continue

            p_target = float(subset[primary_target].mean())
            lift     = p_target / baseline_val if baseline_val > 0 else np.nan

            rand_mean, rand_p95 = random_baseline(df[primary_target], count, RAND_ITERS, rng)
            rand_above_p95 = p_target > rand_p95

            row: dict = {
                "condition": f"{N}×<={X}",
                "streak_threshold": X,
                "streak_count": N,
                "count": count,
                "pct_of_total": round(count / total_n * 100, 1),
                f"p_{primary_target}": round(p_target, 4),
                "baseline": round(baseline_val, 4),
                "lift": round(lift, 3),
                "rand_baseline_mean": round(rand_mean, 4),
                "rand_baseline_p95": round(rand_p95, 4),
                "rand_above_p95": rand_above_p95,
            }

            # Дополнительные целевые метрики
            for b in TARGET_BARS:
                col_b = f"bars_gt_{b}"
                if col_b in subset.columns:
                    row[f"p_bars_gt{b}"] = round(float(subset[col_b].mean()), 4)
            for m in TARGET_ATR_MULT:
                col_m = f"reached_{m}atr"
                if col_m in subset.columns:
                    row[f"p_reach{m}atr"] = round(float(subset[col_m].mean()), 4)

            rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("lift", ascending=False)
    return result


def compute_per_symbol(df: pd.DataFrame, primary_target: str = "bars_gt_15") -> pd.DataFrame:
    """Метрики по каждой монете (только baseline, без грид-поиска)."""
    rows = []
    for symbol, g in df.groupby("coin"):
        row = {
            "coin": symbol,
            "n_trades": len(g),
            f"baseline_{primary_target}": round(float(g[primary_target].mean()), 4),
            "avg_bars": round(float(g["bars_in_trade"].mean()), 1),
            "median_bars": float(g["bars_in_trade"].median()),
            "avg_atr_pct": round(float(g["atr_pct"].mean()), 3),
        }
        for m in TARGET_ATR_MULT:
            col = f"reached_{m}atr"
            if col in g.columns:
                row[f"baseline_reach{m}atr"] = round(float(g[col].mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_trades", ascending=False)


def compute_by_atr_regime(df: pd.DataFrame, primary_target: str = "bars_gt_15") -> pd.DataFrame:
    """Метрики разбитые по режимам волатильности (ATR%)."""
    q33 = df["atr_pct"].quantile(0.33)
    q66 = df["atr_pct"].quantile(0.66)
    df = df.copy()
    df["atr_regime"] = pd.cut(
        df["atr_pct"],
        bins=[-np.inf, q33, q66, np.inf],
        labels=["low", "mid", "high"]
    )
    rows = []
    for regime, g in df.groupby("atr_regime", observed=True):
        row = {
            "atr_regime": regime,
            "n_trades": len(g),
            "atr_pct_range": f"{g['atr_pct'].min():.2f}–{g['atr_pct'].max():.2f}",
            f"baseline_{primary_target}": round(float(g[primary_target].mean()), 4),
            "avg_bars": round(float(g["bars_in_trade"].mean()), 1),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def compute_walk_forward(df: pd.DataFrame, primary_target: str = "bars_gt_15") -> pd.DataFrame:
    """
    Walk-forward: разбиваем историю на 6-месячные окна.
    Для каждого окна считаем lift лучшего условия.
    """
    df = df.copy()
    df["signal_ts"] = pd.to_datetime(df["signal_ts"], utc=True)
    min_ts = df["signal_ts"].min()
    max_ts = df["signal_ts"].max()

    months = WF_MONTHS
    rows = []

    current = min_ts
    while current < max_ts:
        end = current + pd.DateOffset(months=months)
        window_df = df[(df["signal_ts"] >= current) & (df["signal_ts"] < end)]

        if len(window_df) < 50:
            current = end
            continue

        baseline = float(window_df[primary_target].mean())
        best_lift = 1.0
        best_cond = "none"

        for X in [5, 6, 8]:
            col = f"streak_le{X}"
            if col not in window_df.columns:
                continue
            for N in [3, 4, 5]:
                mask = window_df[col] >= N
                sub  = window_df[mask]
                if len(sub) < 10:
                    continue
                p = float(sub[primary_target].mean())
                lift = p / baseline if baseline > 0 else 1.0
                if lift > best_lift:
                    best_lift = lift
                    best_cond = f"{N}×<={X}"

        rows.append({
            "window_start": current.date(),
            "window_end": end.date(),
            "n_trades": len(window_df),
            "baseline": round(baseline, 4),
            "best_condition": best_cond,
            "best_lift": round(best_lift, 3),
            "lift_positive": best_lift > 1.0,
            "lift_strong": best_lift >= 1.3,
        })
        current = end

    result = pd.DataFrame(rows)
    if not result.empty:
        pos_fraction = result["lift_positive"].mean()
        strong_fraction = result["lift_strong"].mean()
        logger.info(f"Walk-forward ({months}m windows): {len(result)} windows, "
                    f"lift>1.0 in {pos_fraction:.0%}, lift>=1.3 in {strong_fraction:.0%}")
    return result


def compute_baseline_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Безусловное распределение bars_in_trade."""
    b = df["bars_in_trade"]
    rows = []
    for threshold in [5, 8, 10, 12, 15, 20, 25, 30, 50]:
        rows.append({
            "bars_threshold": f">{threshold}",
            "probability": round(float((b > threshold).mean()), 4),
            "count": int((b > threshold).sum()),
        })
    rows.append({
        "bars_threshold": "mean",
        "probability": round(float(b.mean()), 2),
        "count": int(len(b)),
    })
    rows.append({
        "bars_threshold": "median",
        "probability": round(float(b.median()), 1),
        "count": int(len(b)),
    })
    rows.append({
        "bars_threshold": "std",
        "probability": round(float(b.std()), 2),
        "count": int(len(b)),
    })
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not FEATURES_PATH.exists():
        logger.error(f"Not found: {FEATURES_PATH}. Run 04_features.py first.")
        sys.exit(1)

    df = pd.read_parquet(FEATURES_PATH)
    logger.info(f"Loaded {len(df):,} trades, {df['coin'].nunique()} coins")

    primary = "bars_gt_15"   # главная целевая метрика

    logger.info("Computing grid statistics...")
    grid = compute_grid_stats(df, primary)

    logger.info("Computing per-symbol stats...")
    per_sym = compute_per_symbol(df, primary)

    logger.info("Computing ATR regime stats...")
    by_atr = compute_by_atr_regime(df, primary)

    logger.info("Computing walk-forward...")
    wf = compute_walk_forward(df, primary)

    logger.info("Computing baseline distribution...")
    baseline = compute_baseline_distribution(df)

    # Вывод топ результатов
    logger.info("\n=== TOP-10 условий по lift ===")
    if not grid.empty:
        top = grid.head(10)[["condition", "count", f"p_{primary}", "baseline", "lift", "rand_above_p95"]]
        logger.info("\n" + top.to_string(index=False))

    # Сохранение в Excel
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        grid.to_excel(writer, sheet_name="grid",          index=False)
        per_sym.to_excel(writer, sheet_name="per_symbol", index=False)
        by_atr.to_excel(writer, sheet_name="by_atr_regime", index=False)
        wf.to_excel(writer, sheet_name=f"walk_forward_{WF_MONTHS}m", index=False)
        baseline.to_excel(writer, sheet_name="baseline",  index=False)

    logger.info(f"\nSaved: {OUTPUT_PATH}")

    # Итоговый вывод
    if not grid.empty:
        n_above_p95 = grid["rand_above_p95"].sum()
        logger.info(f"Conditions above random p95: {n_above_p95}/{len(grid)}")


if __name__ == "__main__":
    main()
