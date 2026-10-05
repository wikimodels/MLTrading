from pathlib import Path
import pandas as pd
from diver_backtest.config import Config

def print_table(title: str, df: pd.DataFrame):
    """Печатает красивую ASCII таблицу в консоль."""
    print(f"\n{'='*75}")
    print(f"   {title.upper()}")
    print(f"{'='*75}")
    if df.empty:
        print("  (Нет данных)")
        return
    print(df.to_string(index=True))
    print(f"{'='*75}")

# Removed _agg function
def build_dashboard(results: dict, cfg: Config, out_dir: Path = None):
    """
    Формирует консольные сводки и сохраняет отчеты в формате CSV.
    Без сторонних зависимостей типа Excel.
    """
    output_dir = out_dir or cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_trades = results.get("trades", pd.DataFrame())
    df_cases = results.get("cases", pd.DataFrame())

    if df_trades.empty:
        print("\n❌  Результаты бэктеста пусты, строить дашборд не по чему.")
        return

    # Сохраняем детальные кейсы в CSV
    detail_path = output_dir / "detail_cases.csv"
    df_cases.to_csv(detail_path, index=False, encoding="utf-8")
    df_trades.to_csv(output_dir / "all_trades.csv", index=False, encoding="utf-8")

    # 1. По таймфреймам (по уникальным сделкам)
    by_tf = df_trades.groupby("tf").apply(lambda g: pd.Series({
        "Сделок (N)": len(g),
        "Побед": g["win"].sum(),
        "WinRate %": f"{(g['win'].mean() * 100):.1f}%",
        "Avg Net Return %": f"{(g['net_return'].mean() * 100):+.2f}%"
    }))
    print_table("Сводка по таймфреймам", by_tf)
    by_tf.to_csv(output_dir / "by_tf.csv", encoding="utf-8")

    # 2. По индикаторам (используем df_cases, т.к. там разделено по индикаторам)
    # df_cases содержит n, wins, avg_return, мы их агрегируем
    by_ind = df_cases.groupby("indicator").apply(lambda g: pd.Series({
        "Сделок (N)": g["n"].sum(),
        "Побед": g["wins"].sum(),
        "WinRate %": f"{(g['wins'].sum() / g['n'].sum() * 100) if g['n'].sum() else 0:.1f}%",
        "Avg Net Return %": f"{( (g['avg_return'] * g['n']).sum() / g['n'].sum() * 100) if g['n'].sum() else 0:+.2f}%"
    }))
    by_ind = by_ind.sort_values(by="WinRate %", ascending=False)
    print_table("Сводка по индикаторам (Рейтинг по WinRate)", by_ind)
    by_ind.to_csv(output_dir / "by_indicator.csv", encoding="utf-8")

    # 3. По количеству подтвердивших монет (N_dir)
    by_ndir = df_trades.groupby("N_dir").apply(lambda g: pd.Series({
        "Сделок (N)": len(g),
        "Побед": g["win"].sum(),
        "WinRate %": f"{(g['win'].mean() * 100):.1f}%",
        "Avg Net Return %": f"{(g['net_return'].mean() * 100):+.2f}%"
    }))
    print_table("Винрейт по количеству монет (N_dir: 1..7)", by_ndir)
    by_ndir.to_csv(output_dir / "by_N_dir.csv", encoding="utf-8")

    print(f"\n   Все сводные отчеты успешно сохранены в папку: {output_dir}")
