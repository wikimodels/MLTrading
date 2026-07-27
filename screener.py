import ccxt
import pandas as pd
from datetime import datetime, timezone, timedelta
from loguru import logger
from config_loader import get_config
import os
from dotenv import load_dotenv

load_dotenv()

def get_bybit_client():
    proxy = os.getenv("BYBIT_PROXY", "")
    params = {
        "enableRateLimit": True,
        "options": {"defaultType": "linear", "adjustForTimeDifference": True},
    }
    if proxy:
        params["proxies"] = {"http": proxy, "https": proxy}
        params["httpsProxy"] = proxy
    return ccxt.bybit(params)

def fetch_daily_data(exchange, symbol, since_ms):
    try:
        # Fetch 1d candles
        all_ohlcv = []
        current_since = since_ms
        while True:
            ohlcv = exchange.fetch_ohlcv(symbol, '1d', since=current_since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            current_since = ohlcv[-1][0] + 86400000 # add one day in ms
            if len(ohlcv) < 1000:
                break
            time.sleep(exchange.rateLimit / 1000)
            
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.normalize()
        df = df.drop_duplicates(subset=['date']).set_index('date')
        df['return'] = df['close'].pct_change()
        return df
    except Exception as e:
        logger.debug(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

def run_screener():
    exchange = get_bybit_client()
    logger.info("Fetching Bybit markets...")
    exchange.load_markets()
    
    # Filter for USDT perpetual futures
    symbols = [s for s, m in exchange.markets.items() if m['linear'] and m['quote'] == 'USDT' and s.endswith('/USDT:USDT')]
    logger.info(f"Found {len(symbols)} USDT perpetual markets.")
    
    # 365 days ago
    start_date = datetime.now(timezone.utc) - timedelta(days=365)
    since_ms = int(start_date.timestamp() * 1000)
    
    logger.info(f"Fetching BTC benchmark data since {start_date.date()}...")
    btc_df = fetch_daily_data(exchange, 'BTC/USDT:USDT', since_ms)
    
    if btc_df.empty or len(btc_df) < 360:
        logger.error("Failed to fetch sufficient BTC data.")
        return
        
    # Find days where BTC dropped by more than 1%
    btc_down_days = btc_df[btc_df['return'] < -0.01].copy()
    avg_btc_drop = btc_down_days['return'].mean()
    
    logger.info(f"Found {len(btc_down_days)} days where BTC dropped > 1%. Average BTC drop on these days: {avg_btc_drop*100:.2f}%")
    
    results = []
    
    # Exclude already blacklisted coins if necessary, or just scan everything to see where they rank
    cfg = get_config()
    blacklist = set(cfg["trading"].get("exclude_symbols", []))
    
    for i, sym in enumerate(symbols):
        if i % 10 == 0:
            logger.info(f"Processing {i}/{len(symbols)}: {sym}...")
            
        # Optional: Skip BTC itself
        if sym == 'BTC/USDT:USDT':
            continue
            
        df = fetch_daily_data(exchange, sym, since_ms)
        
        # Require at least 360 days of history
        if len(df) < 360:
            continue
            
        # Align with BTC down days
        aligned = df.reindex(btc_down_days.index)
        
        # Calculate how much it dropped on those specific days
        avg_alt_drop = aligned['return'].mean()
        
        # If it actually went UP when BTC went down, the ratio will be negative (or we just handle it)
        # Downside Beta = alt_drop / btc_drop (both are negative, so we get a positive multiplier)
        # e.g., alt_drop = -0.06, btc_drop = -0.02 => beta = 3.0 (fell 3x harder)
        downside_beta = avg_alt_drop / avg_btc_drop if avg_btc_drop != 0 else 0
        
        is_blacklisted = sym in blacklist
        
        results.append({
            'symbol': sym,
            'days_history': len(df),
            'avg_drop_pct': avg_alt_drop * 100 if pd.notna(avg_alt_drop) else 0,
            'downside_beta': downside_beta if pd.notna(downside_beta) else 0,
            'is_blacklisted': is_blacklisted
        })
        
        import time
        time.sleep(exchange.rateLimit / 1000) # Respect rate limits
        
    res_df = pd.DataFrame(results)
    if res_df.empty:
        logger.error("No symbols met the 1-year history requirement.")
        return
        
    # Sort by Downside Beta descending (highest beta = weakest coin)
    res_df = res_df.sort_values('downside_beta', ascending=False).reset_index(drop=True)
    
    # Calculate cutoff for top 50%
    cutoff_idx = len(res_df) // 2
    top_50_pct = res_df.iloc[:cutoff_idx]
    
    res_df.to_csv("weak_coins_report.csv", index=False)
    top_50_pct.to_csv("weak_coins_top50_pct.csv", index=False)
    
    logger.info(f"Scan complete! {len(res_df)} coins met the 1-year requirement.")
    logger.info(f"Report saved to 'weak_coins_report.csv'.")
    logger.info(f"Top 50% weakest saved to 'weak_coins_top50_pct.csv'.")
    
    # Print top 15 for a quick peek
    logger.info("\nTOP 15 WEAKEST COINS (Highest Downside Beta):")
    print(res_df.head(15).to_string(index=False))

if __name__ == "__main__":
    run_screener()
