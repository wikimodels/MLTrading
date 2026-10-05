from diver_backtest.config import Config

def cluster_to_signal(cluster: list, cfg: Config) -> dict:
    """
    Группирует данные кластера в сырой сигнал без искусственных весов.
    Цель: собрать максимальный объем данных для последующего анализа.
    """
    coins_bear = set()
    coins_bull = set()
    indicators_bear = set()
    indicators_bull = set()

    for evt in cluster:
        if evt["direction"] == +1:
            coins_bear.add(evt["coin"])
            indicators_bear.add(evt["indicator"])
        else:
            coins_bull.add(evt["coin"])
            indicators_bull.add(evt["indicator"])

    n_bear = len(coins_bear)
    n_bull = len(coins_bull)

    # Определение доминирующего направления (+1 short, -1 long)
    if n_bear > n_bull:
        direction = +1
        coins_dir = coins_bear
        indicators_set = indicators_bear
    elif n_bull > n_bear:
        direction = -1
        coins_dir = coins_bull
        indicators_set = indicators_bull
    else:
        return {"direction": 0, "N_dir": 0, "indicators": [], "coins": [], "lead_coin": "BTC"}

    N_dir = len(coins_dir)

    # Ведущая монета кластера - отдаем приоритет BTC, иначе просто берем первую
    lead_coin = "BTC" if "BTC" in coins_dir else (sorted(coins_dir)[0] if coins_dir else "BTC")

    return {
        "direction": direction,
        "N_dir": N_dir,
        "indicators": sorted(indicators_set),
        "coins": sorted(coins_dir),
        "lead_coin": lead_coin,
    }
