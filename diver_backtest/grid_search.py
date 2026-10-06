import sys
from pathlib import Path
from typing import Callable, Optional, List
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

INDICATORS = list(INDICATOR_MAP.keys())

def check_vol(vol, ma, v_opt):
    if v_opt == "all": return True
    if pd.isna(ma) or ma == 0: return False
    surge = (vol / ma) - 1.0
    if v_opt == "gt_30": return surge >= 0.30
    if v_opt == "30_50": return 0.30 <= surge <= 0.50
    if v_opt == "gt_50": return surge > 0.50
    return True

def run_grid_search(
    cfg: Config,
    sl_options: List[float] = [2.0, 2.5, 3.0],
    tp_options: List[float] = [2.0, 2.5, 3.0],
    lookback_options: List[int] = [15, 25, 35, 45],
    vol_options: List[str] = ["all"],
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> pd.DataFrame:
    """
    Выполняет поиск по сетке параметров входа/выхода, окну дивергенции и объемам.
    """
    def notify(pct: float, msg: str):
        if progress_callback:
            progress_callback(pct, msg)
        logger.info(f"[{int(pct*100)}%] {msg}")

    import copy
    cfg = copy.deepcopy(cfg)

    notify(0.05, "Grid Search: Загрузка данных...")
    data = load_all(cfg)

    min_time = None
    max_time = None
    exec_4h_cache = {}
    
    for coin in cfg.coins:
        if "4h" in data.get(coin, {}) and not data[coin]["4h"].empty:
            df_4h = data[coin]["4h"]
            exec_4h_cache[coin] = add_indicators(df_4h)
            if min_time is None or df_4h['timestamp'].min() < min_time:
                min_time = df_4h['timestamp'].min()
            if max_time is None or df_4h['timestamp'].max() > max_time:
                max_time = df_4h['timestamp'].max()

    total_months = 60.0
    if min_time is not None and max_time is not None:
        total_months = (pd.to_datetime(max_time) - pd.to_datetime(min_time)).days / 30.44

    num_tfs = len(cfg.timeframes)
    
    trade_logs = []
    case_logs = []
    
    total_combinations = len(sl_options) * len(tp_options) * len(lookback_options) * len(vol_options) * num_tfs
    comb_idx = 0
    
    for tf_idx, tf in enumerate(cfg.timeframes):
        base_pct = 0.10 + (tf_idx / num_tfs) * 0.30
        notify(base_pct, f"Grid Search ({tf}): Подготовка признаков и свингов...")
        
        prepped = {}
        for coin in cfg.coins:
            df = data.get(coin, {}).get(tf)
            if df is not None and not df.empty:
                prepped[coin] = add_indicators(df)
                
        if not prepped:
            continue
            
        swings = {}
        for coin, df in prepped.items():
            sh = find_swing_highs(df, cfg.swing_n.get(tf, 3), cfg.swing_atr_k.get(tf, 1.2))
            sl = find_swing_lows(df, cfg.swing_n.get(tf, 3), cfg.swing_atr_k.get(tf, 1.2))
            swings[coin] = (sh, sl)
            
        for lb in lookback_options:
            cfg.window_max_bars = lb
            notify(base_pct, f"Grid Search ({tf}): Lookback {lb}...")
            
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
                
            clusters = align_divergences(divs_by_coin, cfg, tf=tf)
            
            valid_signals = []
            for cluster in clusters:
                sig = cluster_to_signal(cluster, cfg)
                if sig["direction"] == 0:
                    continue
                valid_signals.append((cluster, sig))
                
            for v_opt in vol_options:
                for sl in sl_options:
                    for tp in tp_options:
                        comb_idx += 1
                        prog = 0.40 + (comb_idx / total_combinations) * 0.55
                        if comb_idx % max(1, (total_combinations // 20)) == 0:
                            notify(prog, f"Симуляция LB={lb} VOL={v_opt} SL={sl} TP={tp} на {tf}...")
                            
                        cfg.stop_atr_mult = sl
                        cfg.take_rr = tp
                        
                        for cluster, sig in valid_signals:
                            exec_coin = sig["lead_coin"] if cfg.execution_mode == "coin" else "BTC"
                            df_exec = exec_4h_cache.get(exec_coin, prepped.get(exec_coin))
                            if df_exec is None or df_exec.empty:
                                continue
                            
                            signal_time = max(evt["confirm_time"] for evt in cluster)
                            
                            # Проверка фильтра объема на сигнальном ТФ
                            df_sig = prepped.get(exec_coin)
                            if df_sig is None or df_sig.empty:
                                continue
                            row_sig = df_sig[df_sig['timestamp'] <= signal_time]
                            if row_sig.empty:
                                continue
                            vol_current = row_sig['volume'].values[-1]
                            vol_ma_val = row_sig['vol_ma'].values[-1]
                            if not check_vol(vol_current, vol_ma_val, v_opt):
                                continue
                                
                            trade = simulate_trade(df_exec, signal_time, sig["direction"], cfg)
                            if trade is None:
                                continue
                                
                            t_log = {
                                "tf": tf,
                                "exec_coin": exec_coin,
                                "direction": "Short" if sig["direction"] == +1 else "Long",
                                "lookback": lb,
                                "vol_filter": v_opt,
                                "sl_mult": sl,
                                "tp_rr": tp,
                                "N_dir": sig["N_dir"],
                                "win": trade["win"],
                                "net_return": trade["net_return"],
                                "exit_reason": trade["exit_reason"]
                            }
                            trade_logs.append(t_log)
                            
                            for ind in sig["indicators"]:
                                c_log = t_log.copy()
                                c_log["indicator"] = ind
                                case_logs.append(c_log)

    notify(0.95, "Агрегация отчета Grid Search...")
    df_trades = pd.DataFrame(trade_logs)
    df_cases = pd.DataFrame(case_logs)
    
    if df_trades.empty:
        notify(1.0, "Нет сделок в Grid Search.")
        return pd.DataFrame()
        
    report = df_trades.groupby(["exec_coin", "tf", "direction", "lookback", "vol_filter", "N_dir", "sl_mult", "tp_rr"]).agg(
        trades=("win", "count"),
        win_rate=("win", "mean"),
        avg_net=("net_return", "mean"),
        total_net=("net_return", "sum")
    ).reset_index()
    
    report["trades_per_month"] = (report["trades"] / total_months).round(1)
    report["win_rate"] = (report["win_rate"] * 100).round(2)
    report["avg_net"] = (report["avg_net"] * 100).round(2)
    report["total_net"] = (report["total_net"] * 100).round(2)
    report = report.sort_values(["exec_coin", "tf", "total_net"], ascending=[True, True, False])
    
    cases_report = df_cases.groupby(["indicator", "exec_coin", "tf", "direction", "lookback", "vol_filter", "sl_mult", "tp_rr"]).agg(
        trades=("win", "count"),
        win_rate=("win", "mean"),
        avg_net=("net_return", "mean")
    ).reset_index()
    
    cases_report["trades_per_month"] = (cases_report["trades"] / total_months).round(1)
    cases_report["win_rate"] = (cases_report["win_rate"] * 100).round(2)
    cases_report["avg_net"] = (cases_report["avg_net"] * 100).round(2)
    cases_report = cases_report.sort_values(["indicator", "tf", "avg_net"], ascending=[True, True, False])
    
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(cfg.output_dir / "grid_search_report.csv", index=False, encoding="utf-8")
    cases_report.to_csv(cfg.output_dir / "grid_cases_report.csv", index=False, encoding="utf-8")
    
    notify(1.0, "Grid Search завершен.")
    return report
