from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path

# Project Root resolution
ROOT_DIR = Path(__file__).resolve().parent.parent

@dataclass
class Config:
    def __post_init__(self):
        if self.raw_storage_dir.exists():
            found = set()
            for p in self.raw_storage_dir.iterdir():
                if p.is_dir() and p.name.endswith("_USDT_USDT"):
                    found.add(p.name.replace("_USDT_USDT", ""))
                elif p.is_file() and p.name.endswith(".csv"):
                    name = p.name.split("_")[0]
                    found.add(name)
            if found:
                self.coins = sorted(list(found))

    # Target Coins (те самые 7 монет)
    coins: List[str] = field(default_factory=lambda: [
        "BTC", "ETH", "SOL", "XRP", "DOGE", "HYPE", "MNT"
    ])

    # Timeframes
    timeframes: List[str] = field(default_factory=lambda: ["4h", "12h", "1d"])

    # Data Source (единая база Parquet)
    raw_storage_dir: Path = ROOT_DIR / "data" / "storage" / "raw"
    start_date: str = "2021-10-01"
    end_date: str = "2026-10-06"

    # Swing detection (локальные экстремумы)
    swing_n: Dict[str, int] = field(default_factory=lambda: {
        "4h": 3, "12h": 2, "1d": 2, "1w": 2
    })
    swing_atr_k: Dict[str, float] = field(default_factory=lambda: {
        "4h": 1.5, "12h": 1.2, "1d": 1.0, "1w": 1.0
    })

    # Divergence parameters
    divergence_threshold: float = 0.5
    window_min_bars: int = 5
    window_max_bars: int = 30
    sync_window_bars: int = 1

    # Risk & Execution
    stop_atr_mult: float = 1.2
    take_rr: float = 2.0
    time_stop_bars: int = 60
    look_ahead_fix: bool = True

    # Режим исполнения сделок: 'coin' (на самой монете дивергенции) или 'BTC' (прокси через BTC)
    execution_mode: str = "coin"         # "coin" | "BTC"

    # Costs (Bybit standard taker/taker with slippage for altcoins)
    fee_bps: float = 8.0
    slippage_bps: float = 8.0

    # Output directory
    output_dir: Path = ROOT_DIR / "diver_backtest" / "output"
