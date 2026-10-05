import pandas as pd
df = pd.read_csv('diver_backtest/output/all_trades.csv')
# Keep only one record per coin-time (ignore grid variants)
unique = df.drop_duplicates(subset=['exec_coin', 'entry_time', 'tf']).copy()

# Group by time and tf to count coins
sync_counts = unique.groupby(['entry_time', 'tf'])['exec_coin'].nunique().reset_index(name='sync_count')

# Merge sync_count back to unique trades
unique = unique.merge(sync_counts, on=['entry_time', 'tf'])

# Stats by sync_count
res = unique.groupby('sync_count').agg(
    total_signals=('net_return', 'count'),
    win_rate=('win', 'mean'), # win is True/False boolean? Or string?
    avg_net=('net_return', 'mean')
).reset_index()
print(res)
