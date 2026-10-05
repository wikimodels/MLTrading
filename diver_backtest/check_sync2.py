import pandas as pd
df = pd.read_csv('diver_backtest/output/all_trades.csv')

# Group by the specific grid parameters AND entry time to see if multiple coins fired for the SAME grid setup
# Actually, the grid is per coin, but if we assume the same parameters (tf, vol_filter, direction, sl, tp, lookback) 
# were used, we can group by them.
# But lookback is missing.
# Let's just group by entry_time, tf, direction, vol_filter, sl_mult, tp_rr
group_cols = ['tf', 'direction', 'vol_filter', 'sl_mult', 'tp_rr', 'entry_time']

df['sync_count'] = df.groupby(group_cols)['exec_coin'].transform('nunique')
res = df.groupby('sync_count').agg(
    total_trades=('net_return', 'count'),
    win_rate=('net_return', lambda x: (x > 0).mean() * 100),
    avg_net=('net_return', 'mean')
).round(2).reset_index()

with open('diver_backtest/output/sync.txt', 'w') as f:
    f.write(res.to_markdown())
