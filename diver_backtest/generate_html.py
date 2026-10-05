import pandas as pd
from pathlib import Path

grid_file = Path('diver_backtest/output/grid_search_report.csv')
cases_file = Path('diver_backtest/output/grid_cases_report.csv')
out_html = Path('diver_backtest/output/report.html')

if not grid_file.exists() or not cases_file.exists():
    with open(out_html, 'w', encoding='utf-8') as f:
         f.write('<h1>Ошибка: Файлы отчета не найдены.</h1>')
    print("Files not found.")
    exit(1)

df_grid = pd.read_csv(grid_file)
df_cases = pd.read_csv(cases_file)

# Фильтруем от мусора (минимум 10 сделок за всё время)
df_grid = df_grid[df_grid['trades'] >= 10]
df_cases = df_cases[df_cases['trades'] >= 10]

top_clusters = df_grid.sort_values('total_net', ascending=False).head(50)
if 'total_net' in df_cases.columns:
    top_indicators = df_cases.sort_values('total_net', ascending=False).head(50)
else:
    top_indicators = df_cases.sort_values('avg_net', ascending=False).head(50)

if not df_grid.empty:
    best_tf = df_grid.groupby('tf')['total_net'].mean().reset_index().sort_values('total_net', ascending=False)
    best_vol = df_grid.groupby('vol_filter')['total_net'].mean().reset_index().sort_values('total_net', ascending=False)
else:
    best_tf = pd.DataFrame()
    best_vol = pd.DataFrame()

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

html = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>Квант-отчет: Матрица Дивергенций</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #1e1e1e; color: #d4d4d4; }}
        h1, h2, h3 {{ color: #569cd6; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; font-size: 14px; }}
        th, td {{ border: 1px solid #444; padding: 8px 12px; text-align: right; }}
        th {{ background-color: #2d2d2d; color: #4ec9b0; font-weight: bold; text-align: center; }}
        tr:nth-child(even) {{ background-color: #252526; }}
        tr:hover {{ background-color: #333; }}
        .positive {{ color: #6a9955; font-weight: bold; }}
        .negative {{ color: #f44747; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .highlight {{ background: #2d2d2d; padding: 15px; border-left: 5px solid #569cd6; margin-bottom: 20px; }}
    </style>
</head>
<body>
<div class='container'>
    <h1>Полный Отчет: Оптимизация 1152 Сценариев</h1>
    <div class='highlight'>
        <p>Отфильтровано: минимум 10 сделок за весь период.</p>
    </div>
    
    <h2>🏆 Топ-50 Стратегий по Кластерам (Суммарный Net Return)</h2>
    {df_to_html(top_clusters)}
    
    <h2>🥇 Топ-50 Одиночных Индикаторов (По Avg Net / Винрейту)</h2>
    {df_to_html(top_indicators)}
    
    <h2>📊 Влияние Таймфрейма (Средний Return по всем сценариям)</h2>
    {df_to_html(best_tf)}
    
    <h2>📊 Влияние Фильтра Объема (Средний Return по всем сценариям)</h2>
    {df_to_html(best_vol)}
</div>
</body>
</html>
'''

with open(out_html, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'HTML generated: {out_html.absolute()}')
