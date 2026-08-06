import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

# Add shared to path
import sys
import os
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from shared.indicators import get_hurst_exponent
from scipy.stats import kurtosis

DATA_DIR = Path("trend_detector_lab/data")
OHLCV_DIR = DATA_DIR / "ohlcv"
EXIT_DF_PATH = Path("trend_detector_lab/exit_report/optimization.md") # Just for reference

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {level} | {message}", level="INFO")

def analyze_coin(symbol: str, path: Path):
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    # Use only Train data (2023-2024) to avoid lookahead bias
    train_df = df[df["timestamp"].dt.year <= 2024].copy()
    if len(train_df) < 500:
        return None
        
    train_df = train_df.sort_values("timestamp")
    
    # 1. Hurst Exponent on Daily timeframe (better for macro trend)
    # Resample to 1D
    train_df.set_index("timestamp", inplace=True)
    d1 = train_df.resample("1D").agg({"close": "last"}).dropna()
    if len(d1) < 100:
        return None
        
    h = get_hurst_exponent(d1["close"].values, min_lag=10, max_lag=100)
    
    # 2. Kurtosis
    log_ret = np.diff(np.log(d1["close"].values))
    kurt = kurtosis(log_ret, fisher=True, bias=True)
    
    return {
        "coin": symbol,
        "hurst": h,
        "kurtosis": kurt
    }

def main():
    logger.info("Starting Coin Screener Analysis (Train Data: 2023-2024)...")
    
    results = []
    for file in OHLCV_DIR.glob("*_4h.parquet"):
        symbol = file.stem.split("_")[0]
        res = analyze_coin(symbol, file)
        if res:
            results.append(res)
            
    res_df = pd.DataFrame(results)
    
    # Rank by Hurst and Kurtosis
    # We want High Hurst and High Kurtosis
    res_df["hurst_rank"] = res_df["hurst"].rank(ascending=False)
    res_df["kurt_rank"] = res_df["kurtosis"].rank(ascending=False)
    res_df["combined_score"] = res_df["hurst_rank"] + res_df["kurt_rank"]
    
    # Top 20 coins
    res_df = res_df.sort_values("combined_score").reset_index(drop=True)
    top_20 = res_df.head(20)
    
    logger.info("\n--- Top 20 Trending & Fat Tailed Coins ---")
    logger.info(f"\n{top_20[['coin', 'hurst', 'kurtosis']].to_string(index=False)}")
    
    # Now let's see how much they improved the OOS EES!
    logger.info("\nLoading Phase 4 Exit Optimizer data to compare OOS performance...")
    # We don't have the raw exit dataframe saved, so we will just print the coins.
    # To do a full OOS test, we'd need to re-run the `05_exit_optimizer.py` on just these 20 coins.
    
    top_coins = top_20["coin"].tolist()
    Path("trend_detector_lab/top_20_coins.txt").write_text("\n".join(top_coins))
    logger.info("Saved top 20 coins to top_20_coins.txt")
    
if __name__ == "__main__":
    main()
