import pandas as pd
df = pd.read_csv('diver_backtest/output/grid_cases_report.csv')

with open('diver_backtest/output/losers.txt', 'w', encoding='utf-8') as f:
    f.write('--- INDICATORS ---\n')
    res_ind = df.groupby('indicator').agg(trades=('trades', 'sum'), avg_net=('avg_net', 'mean')).sort_values('avg_net')
    f.write(res_ind.to_string())
    
    f.write('\n\n--- LOOKBACK PERIODS ---\n')
    res_lb = df.groupby('lookback').agg(trades=('trades', 'sum'), avg_net=('avg_net', 'mean')).sort_values('avg_net')
    f.write(res_lb.to_string())
