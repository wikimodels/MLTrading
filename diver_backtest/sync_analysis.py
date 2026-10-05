import pandas as pd

df = pd.read_csv('diver_backtest/output/all_trades.csv')
unique_entries = df.drop_duplicates(subset=['exec_coin', 'entry_time', 'tf', 'direction']).copy()
unique_entries['sync_count'] = unique_entries.groupby(['entry_time', 'tf', 'direction'])['exec_coin'].transform('nunique')

res = unique_entries.groupby('sync_count').agg(
    total_trades=('net_return', 'count'),
    win_rate=('net_return', lambda x: (x > 0).mean() * 100),
    avg_net=('net_return', 'mean')
).round(2).reset_index()

with open('diver_backtest/output/sync_analysis.txt', 'w', encoding='utf-8') as f:
    f.write('sync_count | total_trades | win_rate | avg_net\n')
    f.write(res.to_string(index=False))
print("Done")
