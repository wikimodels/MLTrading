from typing import Dict
import pandas as pd
from diver_backtest.config import Config

REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]

def load_ohlcv(coin: str, tf: str, cfg: Config) -> pd.DataFrame:
    """
    Загружает OHLCV данные напрямую из Parquet хранилища (data/storage/raw/).
    При необходимости поддерживает чтение и из CSV.
    """
    # 1. Поиск в parquet базе проекта: data/storage/raw/<coin>_USDT_USDT/<tf>.parquet
    parquet_path = cfg.raw_storage_dir / f"{coin}_USDT_USDT" / f"{tf}.parquet"
    
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        # Резервный поиск в CSV
        csv_path = cfg.raw_storage_dir / f"{coin}_{tf}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Данные не найдены ни в {parquet_path}, ни в {csv_path}")
        df = pd.read_csv(csv_path)

    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"{parquet_path} не содержит обязательную колонку {c}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Фильтрация по датам
    start_ts = pd.to_datetime(cfg.start_date, utc=True)
    end_ts = pd.to_datetime(cfg.end_date, utc=True)
    df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
    return df.reset_index(drop=True)

def load_all(cfg: Config) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Загружает исторические данные для всех монет и таймфреймов.
    Возвращает структуру: {coin: {tf: df}}
    """
    data = {}
    for coin in cfg.coins:
        data[coin] = {}
        for tf in cfg.timeframes:
            try:
                data[coin][tf] = load_ohlcv(coin, tf, cfg)
            except Exception as e:
                print(f"⚠️  Ошибка загрузки {coin} {tf}: {e}")
    return data
