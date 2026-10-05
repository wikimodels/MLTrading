import sys
from pathlib import Path
from typing import Callable, Optional, Dict, Any
import pandas as pd
from loguru import logger

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from diver_backtest.config import Config
from diver_backtest.data_loader import load_all
from diver_backtest.indicators import add_indicators, INDICATOR_MAP
from diver_backtest.swings import find_swing_highs, find_swing_lows
from diver_backtest.divergence import all_divergences
from diver_backtest.sync import align_divergences
from diver_backtest.signals import cluster_to_signal
from diver_backtest.backtest import simulate_trade
from diver_backtest.stats import summarize

INDICATORS = list(INDICATOR_MAP.keys())

def run_backtest(
    cfg: Config,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Основной движок бэктеста мульти-валютных дивергенций.
    Поддерживает передачу progress_callback(pct, message) для Streamlit дашборда.
    """
    def notify(pct: float, msg: str):
        if progress_callback:
            progress_callback(pct, msg)
        logger.info(f"[{int(pct*100)}%] {msg}")

    notify(0.05, "Загрузка исторических данных (Parquet)...")
    data = load_all(cfg)

    # Кэшируем 4h данные для быстрого исполнения
    notify(0.15, "Подготовка свечей и расчет волатильности...")
    exec_4h_cache = {}
    for coin in cfg.coins:
        if "4h" in data.get(coin, {}) and not data[coin]["4h"].empty:
            exec_4h_cache[coin] = add_indicators(data[coin]["4h"])

    all_results = []
    trade_logs = []
    num_tfs = len(cfg.timeframes)

    for tf_idx, tf in enumerate(cfg.timeframes):
        base_pct = 0.20 + (tf_idx / num_tfs) * 0.65
        notify(base_pct, f"Таймфрейм {tf}: Расчет {len(INDICATORS)} индикаторов...")

        prepped = {}
        for coin in cfg.coins:
            df = data.get(coin, {}).get(tf)
            if df is None or df.empty:
                continue
            prepped[coin] = add_indicators(df)

        if not prepped:
            continue

        notify(base_pct + 0.05 / num_tfs, f"Таймфрейм {tf}: Поиск экстремумов (Свингов)...")
        swings = {}
        for coin, df in prepped.items():
            sh = find_swing_highs(df, cfg.swing_n.get(tf, 3), cfg.swing_atr_k.get(tf, 1.2))
            sl = find_swing_lows(df, cfg.swing_n.get(tf, 3), cfg.swing_atr_k.get(tf, 1.2))
            swings[coin] = (sh, sl)

        notify(base_pct + 0.10 / num_tfs, f"Таймфрейм {tf}: Детекция дивергенций...")
        divs_by_coin = {}
        for coin, df in prepped.items():
            sh, sl = swings[coin]
            res = all_divergences(df, INDICATORS, sh, sl, cfg, tf=tf)
            for ind, lst in res.items():
                for d in lst:
                    d["p1_time"] = df["timestamp"].iloc[d["p1_idx"]]
                    d["p2_time"] = df["timestamp"].iloc[d["p2_idx"]]
                    d["confirm_time"] = df["timestamp"].iloc[d["confirm_idx"]]
            divs_by_coin[coin] = res

        notify(base_pct + 0.15 / num_tfs, f"Таймфрейм {tf}: Синхронизация и кластеризация...")
        clusters = align_divergences(divs_by_coin, cfg, tf=tf)

        notify(base_pct + 0.20 / num_tfs, f"Таймфрейм {tf}: Симуляция сделок ({len(clusters)} кластеров)...")
        for cluster in clusters:
            sig = cluster_to_signal(cluster, cfg)
            if sig["direction"] == 0:
                continue

            exec_coin = sig["lead_coin"] if cfg.execution_mode == "coin" else "BTC"
            df_exec = exec_4h_cache.get(exec_coin, prepped.get(exec_coin))
            if df_exec is None or df_exec.empty:
                continue

            # signal_time strictly uses maximum confirm time from cluster avoiding look-ahead bias
            signal_time = max(evt["confirm_time"] for evt in cluster)

            trade = simulate_trade(df_exec, signal_time, sig["direction"], cfg)
            if trade is None:
                continue

            trade_record = {
                "tf": tf,
                "exec_coin": exec_coin,
                "direction": "Short" if sig["direction"] == +1 else "Long",
                "N_dir": sig["N_dir"],
                "coins": ",".join(sig["coins"]),
                "indicators": ",".join(sig["indicators"]),
                "entry_time": trade["entry_time"],
                "exit_time": trade["exit_time"],
                "entry_price": round(trade["entry_price"], 4),
                "net_return": round(trade["net_return"], 5),
                "gross_return": round(trade["gross_return"], 5),
                "win": trade["win"],
                "exit_reason": trade["exit_reason"],
                "bars_held": trade["bars_held"],
            }
            trade_logs.append(trade_record)

            for ind in sig["indicators"]:
                all_results.append({
                    "tf": tf,
                    "indicator": ind,
                    "N_dir": sig["N_dir"],
                    "direction": sig["direction"],
                    "n": 1,
                    "win": trade["win"],
                    "net_return": trade["net_return"],
                    "gross_return": trade["gross_return"],
                    "bars_held": trade["bars_held"],
                    "exit_reason": trade["exit_reason"],
                    "entry_time": trade["entry_time"],
                })

    notify(0.90, "Анализ и агрегация статистических метрик...")
    df_trades = pd.DataFrame(trade_logs)
    df_res = pd.DataFrame(all_results)

    if df_trades.empty:
        notify(1.0, "Бэктест завершен: сделок не найдено.")
        return {
            "trades": pd.DataFrame(),
            "cases": pd.DataFrame(),
            "summary": {"n": 0}
        }

    # Сортировка сделок по времени
    df_trades = df_trades.sort_values("entry_time").reset_index(drop=True)
    df_trades["cum_return"] = (1 + df_trades["net_return"]).cumprod() - 1

    # Агрегация по кейсам
    cases = []
    for (tf, ind, nd), g in df_res.groupby(["tf", "indicator", "N_dir"]):
        s = summarize(g.to_dict("records"))
        s.update({"tf": tf, "indicator": ind, "N_dir": nd})
        cases.append(s)

    df_cases = pd.DataFrame(cases)

    # Общий summary портфеля
    overall_summary = summarize(df_trades.to_dict("records"))

    # Сохранение артефактов на диск
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    df_trades.to_csv(cfg.output_dir / "all_trades.csv", index=False, encoding="utf-8")
    df_cases.to_csv(cfg.output_dir / "detail_cases.csv", index=False, encoding="utf-8")

    notify(1.0, f"Готово! Совершено {len(df_trades)} сделок.")

    return {
        "trades": df_trades,
        "cases": df_cases,
        "summary": overall_summary,
    }
