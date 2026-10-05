import pandas as pd
grid = pd.read_csv('diver_backtest/output/grid_search_report.csv')
grid = grid[grid['trades'] >= 10]
top = grid.loc[grid.groupby('exec_coin')['total_net'].idxmax()]
total_trades = top['trades'].sum()

with open('diver_backtest/output/freq.txt', 'w') as f:
    f.write(top[['exec_coin', 'tf', 'trades', 'total_net']].to_string(index=False) + '\n')
    f.write(f'\nTotal portfolio trades over 5 years: {total_trades}')
    f.write(f'\nAverage trades per month: {total_trades / 60:.2f}')
