import pandas as pd

df1 = pd.read_csv(r"D:\GitHub\MLTrading\backtest\trades_history_1.csv")
df2 = pd.read_csv(r"D:\GitHub\MLTrading\backtest\trades_history.csv")

common_coins = set(df1['symbol']).intersection(set(df2['symbol']))

res = []
for coin in sorted(common_coins):
    d1 = df1[df1['symbol'] == coin]
    d2 = df2[df2['symbol'] == coin]
    
    wr1 = len(d1[d1['net_pnl_usdt'] > 0]) / len(d1) * 100 if len(d1) > 0 else 0
    wr2 = len(d2[d2['net_pnl_usdt'] > 0]) / len(d2) * 100 if len(d2) > 0 else 0
    
    pnl1 = d1['net_pnl_usdt'].sum()
    pnl2 = d2['net_pnl_usdt'].sum()
    
    res.append({
        'Coin': coin.replace('/USDT:USDT', ''),
        'Trades (Old)': len(d1),
        'Trades (New)': len(d2),
        'WR Old (%)': round(wr1, 1),
        'WR New (%)': round(wr2, 1),
        'WR Diff': round(wr2 - wr1, 1),
        'PnL Old': round(pnl1, 2),
        'PnL New': round(pnl2, 2),
        'PnL Diff': round(pnl2 - pnl1, 2),
    })

res_df = pd.DataFrame(res)
print(res_df.to_string(index=False))

print("\n--- TOTALS FOR COMMON COINS ---")
d1_c = df1[df1['symbol'].isin(common_coins)]
d2_c = df2[df2['symbol'].isin(common_coins)]

wr1 = len(d1_c[d1_c['net_pnl_usdt'] > 0]) / len(d1_c) * 100 if len(d1_c) > 0 else 0
wr2 = len(d2_c[d2_c['net_pnl_usdt'] > 0]) / len(d2_c) * 100 if len(d2_c) > 0 else 0
p1 = d1_c['net_pnl_usdt'].sum()
p2 = d2_c['net_pnl_usdt'].sum()

print(f"WR Old: {wr1:.2f}% -> WR New: {wr2:.2f}% (Diff: {wr2-wr1:.2f}%)")
print(f"PnL Old: ${p1:.2f} -> PnL New: ${p2:.2f} (Diff: ${p2-p1:.2f})")
print(f"Trades Old: {len(d1_c)} -> Trades New: {len(d2_c)}")
