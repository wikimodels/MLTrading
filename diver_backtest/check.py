import pandas as pd
df = pd.read_csv('diver_backtest/output/grid_search_report.csv')
print('Unique coins:', df['exec_coin'].unique())
print('\nRows per coin:')
print(df['exec_coin'].value_counts())

df_f = df[df['trades'] >= 10]
print('\nAverage net per coin (>= 10 trades):')
print(df_f.groupby('exec_coin')['total_net'].mean().sort_values(ascending=False))
