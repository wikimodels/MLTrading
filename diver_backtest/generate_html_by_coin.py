import pandas as pd
from pathlib import Path

grid_file = Path('diver_backtest/output/grid_search_report.csv')
cases_file = Path('diver_backtest/output/grid_cases_report.csv')
out_html = Path('diver_backtest/output/report_by_coin.html')

if not grid_file.exists() or not cases_file.exists():
    with open(out_html, 'w', encoding='utf-8') as f:
         f.write('<h1>Ошибка: Файлы отчета не найдены.</h1>')
    exit(1)

df_grid = pd.read_csv(grid_file)
df_cases = pd.read_csv(cases_file)

# Фильтруем (минимум 10 сделок)
df_grid = df_grid[df_grid['trades'] >= 10]
df_cases = df_cases[df_cases['trades'] >= 10]

def df_to_html(df):
    if df.empty: return '<p>Нет данных по этим фильтрам.</p>'
    formatted_df = df.copy()
    if 'win_rate' in formatted_df.columns:
        formatted_df['win_rate'] = formatted_df['win_rate'].apply(lambda x: f'{x:.1f}%')
    if 'total_net' in formatted_df.columns:
        formatted_df['total_net'] = formatted_df['total_net'].apply(lambda x: f'<span class="positive">{x:.2f}%</span>' if x > 0 else f'<span class="negative">{x:.2f}%</span>')
    if 'avg_net' in formatted_df.columns:
        formatted_df['avg_net'] = formatted_df['avg_net'].apply(lambda x: f'<span class="positive">{x:.2f}%</span>' if x > 0 else f'<span class="negative">{x:.2f}%</span>')
    return formatted_df.to_html(escape=False, index=False)

html_content = ""
coins = sorted(df_grid['exec_coin'].unique())

for coin in coins:
    coin_grid = df_grid[df_grid['exec_coin'] == coin].sort_values('total_net', ascending=False).head(20)
    if 'total_net' in df_cases.columns:
        coin_cases = df_cases[df_cases['exec_coin'] == coin].sort_values('total_net', ascending=False).head(20)
    else:
        coin_cases = df_cases[df_cases['exec_coin'] == coin].sort_values('avg_net', ascending=False).head(20)
        
    html_content += f"<div class='coin-section' id='coin-{coin}'>"
    html_content += f"<h2 class='coin-title'>🪙 Монета: {coin}</h2>"
    html_content += f"<h3>Топ-20 Стратегий по Кластерам</h3>"
    html_content += df_to_html(coin_grid)
    html_content += f"<h3>Топ-20 Одиночных Индикаторов</h3>"
    html_content += df_to_html(coin_cases)
    html_content += "</div><hr>"

if not df_grid.empty:
    best_tf = df_grid.groupby('tf')['total_net'].mean().reset_index().sort_values('total_net', ascending=False)
    best_vol = df_grid.groupby('vol_filter')['total_net'].mean().reset_index().sort_values('total_net', ascending=False)
else:
    best_tf = pd.DataFrame()
    best_vol = pd.DataFrame()

html = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>Квант-отчет: Разбор по Монетам</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #1e1e1e; color: #d4d4d4; }}
        h1, h2, h3 {{ color: #569cd6; }}
        h2.coin-title {{ color: #f39c12; border-bottom: 2px solid #f39c12; padding-bottom: 5px; margin-top: 40px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; font-size: 14px; }}
        th, td {{ border: 1px solid #444; padding: 8px 12px; text-align: right; }}
        th {{ background-color: #2d2d2d; color: #4ec9b0; font-weight: bold; text-align: center; }}
        tr:nth-child(even) {{ background-color: #252526; }}
        tr:hover {{ background-color: #333; }}
        .positive {{ color: #6a9955; font-weight: bold; }}
        .negative {{ color: #f44747; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .highlight {{ background: #2d2d2d; padding: 15px; border-left: 5px solid #569cd6; margin-bottom: 20px; }}
        hr {{ border: 1px solid #333; margin: 40px 0; }}
    </style>
</head>
<body>
<div class='container'>
    <h1>Полный Отчет: Разбор по Монетам (Топ-20)</h1>
    <div class='highlight'>
        <p>Отфильтровано: минимум 10 сделок за весь период.</p>
    </div>
    
    {html_content}
    
    <h2 style='color:#569cd6'>📊 Влияние Таймфрейма (Средний Return по всем сценариям)</h2>
    {df_to_html(best_tf)}
    
    <h2 style='color:#569cd6'>📊 Влияние Фильтра Объема (Средний Return по всем сценариям)</h2>
    {df_to_html(best_vol)}
</div>
</body>
</html>
'''

with open(out_html, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'HTML generated: {out_html.absolute()}')
