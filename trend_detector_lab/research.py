"""
research.py — Универсальный исследовательский скрипт.

Принимает YAML-файл эксперимента и выполняет грид-поиск по любым параметрам.
Сохраняет результаты в Excel + experiment_metadata.json.

Использование:
    poetry run python trend_detector_lab/research.py --experiment research/experiments/streak_basic.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
LAB  = Path(__file__).resolve().parent

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {level} | {message}", level="INFO")


def load_features(cfg_main: dict) -> pd.DataFrame:
    path = LAB / cfg_main["data"]["output_dir"] / "features.parquet"
    if not path.exists():
        logger.error(f"features.parquet not found: {path}. Run 04_features.py first.")
        sys.exit(1)
    df = pd.read_parquet(path)
    df["signal_ts"] = pd.to_datetime(df["signal_ts"], utc=True)
    return df


def random_baseline_stats(series: pd.Series, sample_size: int, n_iter: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    arr = series.values.astype(float)
    if sample_size >= len(arr) or sample_size < 2:
        return float(arr.mean()), float(arr.mean())
    samples = [rng.choice(arr, size=sample_size, replace=False).mean() for _ in range(n_iter)]
    return float(np.mean(samples)), float(np.percentile(samples, 95))


def build_filter(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Строит булевую маску из параметров эксперимента.
    Поддерживает streak_threshold + streak_count + bars_since_last_big.
    """
    mask = pd.Series(True, index=df.index)

    X = params.get("streak_threshold")
    N = params.get("streak_count")
    if X is not None and N is not None:
        col = f"streak_le{X}"
        if col in df.columns:
            mask &= df[col] >= N

    B = params.get("bars_since_last_big")
    threshold = params.get("bars_since_big_threshold", 25)
    if B is not None:
        col = f"bars_since_big{threshold}"
        if col in df.columns:
            mask &= df[col] >= B

    return mask


def run_experiment(exp_cfg: dict, df: pd.DataFrame, rand_iters: int) -> pd.DataFrame:
    """Выполняет грид-поиск по всем комбинациям параметров из exp_cfg['grid']."""
    grid_params: dict = exp_cfg.get("grid", {})
    targets: list     = exp_cfg.get("targets", ["bars_gt_15"])
    min_count: int    = exp_cfg.get("filters", {}).get("min_count", 30)
    primary: str      = targets[0]

    # Генерируем все комбинации параметров грида
    from itertools import product
    keys   = list(grid_params.keys())
    values = [grid_params[k] for k in keys]
    combos = list(product(*values))

    baseline_val = float(df[primary].mean()) if primary in df.columns else 0.0
    rows = []

    for combo in combos:
        params = dict(zip(keys, combo))
        mask   = build_filter(df, params)
        subset = df[mask]
        count  = len(subset)

        if count < min_count:
            continue

        row = {**params, "count": count, "pct_of_total": round(count / len(df) * 100, 1)}

        for target in targets:
            if target not in subset.columns:
                continue
            p = float(subset[target].mean())
            lift = p / baseline_val if baseline_val > 0 else np.nan
            rand_mean, rand_p95 = random_baseline_stats(df[target], count, rand_iters)

            row[f"p_{target}"]    = round(p, 4)
            row[f"lift_{target}"] = round(lift, 3)
            row[f"rand_p95_{target}"] = round(rand_p95, 4)
            row[f"above_p95_{target}"] = p > rand_p95

        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty and f"lift_{primary}" in result.columns:
        result = result.sort_values(f"lift_{primary}", ascending=False)
    return result


def walk_forward_exp(exp_cfg: dict, df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Walk-forward для одного набора параметров."""
    wf_months = exp_cfg.get("walk_forward", {}).get("window_months", 6)
    target    = exp_cfg.get("targets", ["bars_gt_15"])[0]

    min_ts = df["signal_ts"].min()
    max_ts = df["signal_ts"].max()
    rows   = []
    current = min_ts

    while current < max_ts:
        end    = current + pd.DateOffset(months=wf_months)
        window = df[(df["signal_ts"] >= current) & (df["signal_ts"] < end)]

        if len(window) < 30:
            current = end
            continue

        mask   = build_filter(window, params)
        subset = window[mask]
        n      = len(subset)

        baseline = float(window[target].mean()) if target in window.columns and len(window) > 0 else 0.0
        p        = float(subset[target].mean()) if target in subset.columns and n > 0 else 0.0
        lift     = p / baseline if baseline > 0 else np.nan

        rows.append({
            "window_start": current.date(),
            "window_end": end.date(),
            "n_all": len(window),
            "n_filtered": n,
            "baseline": round(baseline, 4),
            f"p_{target}": round(p, 4),
            "lift": round(lift, 3) if not np.isnan(lift) else np.nan,
            "lift_positive": bool(lift > 1.0) if not np.isnan(lift) else False,
        })
        current = end

    return pd.DataFrame(rows)


def save_metadata(exp_cfg: dict, cfg_main: dict, df: pd.DataFrame, runtime_s: float) -> None:
    """Сохраняет experiment_metadata.json."""
    experiments_dir = LAB / "experiments"
    experiments_dir.mkdir(exist_ok=True)

    name = exp_cfg.get("name", "unknown")
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    meta = {
        "experiment": name,
        "description": exp_cfg.get("description", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "settings": {
            "atr_period":  cfg_main["chandelier"]["atr_period"],
            "multiplier":  cfg_main["chandelier"]["multiplier"],
            "timeframe":   cfg_main["chandelier"]["timeframe"],
            "stop_mode":   cfg_main["stop"]["mode"],
            "commission":  cfg_main["backtest"]["commission"],
        },
        "data": {
            "coins":         int(df["coin"].nunique()),
            "total_trades":  int(len(df)),
            "date_from":     str(df["signal_ts"].min().date()),
            "date_to":       str(df["signal_ts"].max().date()),
        },
        "runtime_seconds": round(runtime_s, 1),
    }

    out = experiments_dir / f"{name}_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Trend Detector Lab — research runner")
    parser.add_argument("--experiment", required=True, help="Path to experiment YAML")
    args = parser.parse_args()

    exp_path = Path(args.experiment)
    if not exp_path.is_absolute():
        exp_path = LAB / exp_path
    if not exp_path.exists():
        logger.error(f"Experiment file not found: {exp_path}")
        sys.exit(1)

    with open(exp_path, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)

    with open(LAB / "config" / "settings.yaml", "r", encoding="utf-8") as f:
        cfg_main = yaml.safe_load(f)

    rand_iters = exp_cfg.get("random_baseline", {}).get("iterations", 1000)
    name       = exp_cfg.get("name", "experiment")
    output_rel = exp_cfg.get("output", f"output/{name}.xlsx")
    output_path = LAB / output_rel
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Experiment: {name}")
    logger.info(f"Description: {exp_cfg.get('description', '')}")

    t0 = time.time()
    df = load_features(cfg_main)
    logger.info(f"Loaded {len(df):,} trades, {df['coin'].nunique()} coins")

    # Грид-поиск
    logger.info("Running grid search...")
    grid = run_experiment(exp_cfg, df, rand_iters)
    logger.info(f"Grid results: {len(grid)} combinations")

    # Walk-forward для лучшего условия
    wf_dfs = {}
    if not grid.empty and "walk_forward" in exp_cfg:
        # Берём топ-1 условие для WF
        best_row = grid.iloc[0]
        best_params = {k: best_row[k] for k in exp_cfg["grid"] if k in best_row}
        logger.info(f"Walk-forward for best condition: {best_params}")
        wf = walk_forward_exp(exp_cfg, df, best_params)
        if not wf.empty:
            pos = wf["lift_positive"].mean()
            logger.info(f"Walk-forward: {len(wf)} windows, lift>1.0 in {pos:.0%}")
            wf_dfs["walk_forward"] = wf

    # Топ-10 в лог
    primary = exp_cfg.get("targets", ["bars_gt_15"])[0]
    logger.info(f"\n=== TOP-10 conditions (lift on {primary}) ===")
    if not grid.empty:
        show_cols = [c for c in grid.columns if c in [
            *exp_cfg["grid"].keys(), "count", f"p_{primary}", f"lift_{primary}", f"above_p95_{primary}"
        ]]
        logger.info("\n" + grid.head(10)[show_cols].to_string(index=False))

    # Сохранение
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        grid.to_excel(writer, sheet_name="grid", index=False)
        for sheet_name, wf_df in wf_dfs.items():
            wf_df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info(f"Saved: {output_path}")

    runtime = time.time() - t0
    save_metadata(exp_cfg, cfg_main, df, runtime)


if __name__ == "__main__":
    main()
