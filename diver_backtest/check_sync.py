import pandas as pd
df = pd.read_csv('diver_backtest/output/all_trades.csv')
unique_entries = df.drop_duplicates(subset=['exec_coin', 'entry_time', 'tf']).copy()

# Print top 5 entry times with most coins
counts = unique_entries.groupby('entry_time')['exec_coin'].nunique().sort_values(ascending=False)
print("Max coins at once:", counts.head(5))

