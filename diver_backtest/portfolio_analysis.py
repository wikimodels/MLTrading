import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 1. Load best setups
grid = pd.read_csv('diver_backtest/output/grid_search_report.csv')
grid = grid[grid['trades'] >= 10]

# Get top setup per coin
top_per_coin = grid.loc[grid.groupby('exec_coin')['total_net'].idxmax()]

# 2. Extract trades
trades = pd.read_csv('diver_backtest/output/all_trades.csv')

portfolio_trades = pd.DataFrame()
for _, row in top_per_coin.iterrows():
    match = trades[
        (trades['exec_coin'] == row['exec_coin']) &
        (trades['tf'] == row['tf']) &
        (trades['direction'] == row['direction']) &
        (trades['vol_filter'] == row['vol_filter']) &
        (trades['sl_mult'] == row['sl_mult']) &
        (trades['tp_rr'] == row['tp_rr'])
    ]
    # To avoid duplicates if multiple lookbacks generated the exact same trade, drop dupes by entry_time
    match = match.drop_duplicates(subset=['entry_time'])
    portfolio_trades = pd.concat([portfolio_trades, match])
    
portfolio_trades['entry_time'] = pd.to_datetime(portfolio_trades['entry_time'], utc=True)
portfolio_trades = portfolio_trades.sort_values('entry_time')

with open('diver_backtest/output/portfolio_stats.txt', 'w', encoding='utf-8') as f:
    f.write("Топ-сетапы для каждой монеты в портфеле:\n")
    for _, row in top_per_coin.iterrows():
        f.write(f"{row['exec_coin']}: {row['tf']}, {row['direction']}, VOL: {row['vol_filter']}, SL: {row['sl_mult']}, TP: {row['tp_rr']} (Net: {row['total_net']:.2f}%)\n")

    f.write(f"\nВсего сделок в портфеле за весь период (почти 5 лет): {len(portfolio_trades)}\n")
    avg_per_month = len(portfolio_trades) / 60
    f.write(f"Среднее количество сделок в месяц (за всю историю): {avg_per_month:.1f}\n")
    
    max_date = portfolio_trades['entry_time'].max()
    last_year_start = max_date - pd.DateOffset(years=1)
    
    last_year_trades = portfolio_trades[portfolio_trades['entry_time'] >= last_year_start]
    f.write(f"\nСделок за последний год (с {last_year_start.date()} по {max_date.date()}): {len(last_year_trades)}\n")
    
    f.write("\nРазбивка портфеля по месяцам за последний год:\n")
    monthly = last_year_trades['entry_time'].dt.to_period('M').value_counts().sort_index()
    for month, count in monthly.items():
        coins_that_month = last_year_trades[last_year_trades['entry_time'].dt.to_period('M') == month]['exec_coin'].tolist()
        f.write(f"{month}: {count} сделок {coins_that_month}\n")
