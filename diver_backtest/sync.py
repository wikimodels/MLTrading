from diver_backtest.config import Config

def align_divergences(divs_by_coin: dict, cfg: Config, tf: str = "4h") -> list:
    """
    Группирует дивергенции по разным монетам, чьи таймстемпы попадают в окно ±sync_window_bars.
    divs_by_coin: {coin: {indicator: [div, ...]}}
    Возвращает список кластеров: [ [event1, event2, ...], ... ]
    """
    events = []
    for coin, per_ind in divs_by_coin.items():
        for ind, divs in per_ind.items():
            for d in divs:
                events.append({
                    "coin": coin,
                    "indicator": ind,
                    "p2_time": d["p2_time"],
                    "confirm_time": d.get("confirm_time", d["p2_time"]),
                    "direction": d["direction"],
                    "D": d["D"],
                    "side": d["side"],
                    "div_type": d.get("div_type", ""),
                    "p1_idx": d["p1_idx"],
                    "p2_idx": d["p2_idx"],
                    "confirm_idx": d.get("confirm_idx", d["p2_idx"]),
                })
    if not events:
        return []

    # Сортировка по времени события
    time_key = "confirm_time" if cfg.look_ahead_fix else "p2_time"
    events.sort(key=lambda e: e[time_key])

    clusters = []
    used = [False] * len(events)
    tf_seconds = {"4h": 4*3600, "12h": 12*3600, "1d": 24*3600, "1w": 7*24*3600}
    window_sec = getattr(cfg, "sync_window_bars", 1) * tf_seconds.get(tf, 4*3600)

    for i, e in enumerate(events):
        if used[i]:
            continue
        group = [e]
        used[i] = True
        t0 = e[time_key]
        for j in range(i + 1, len(events)):
            if used[j]:
                continue
            dt = abs((events[j][time_key] - t0).total_seconds())
            if dt <= window_sec:
                group.append(events[j])
                used[j] = True
        clusters.append(group)
    return clusters
