"""EMA Fan Bot — Streamlit monitor dashboard.

Run:
    poetry run streamlit run ema_fan_bot/monitor.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_bot_dir = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_bot_dir) not in sys.path:
    sys.path.insert(0, str(_bot_dir))
os.chdir(_bot_dir)

from shared.config_loader import set_active_bot, get_config
set_active_bot("ema_fan_bot")

import subprocess
import threading
import time
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from shared.data.storage import DataStorage
from ema_fan_bot.strategy.signals import EmaFanSignalGenerator
from ema_fan_bot.backtest.engine import EmaFanBacktestEngine

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EMA Fan Strategy",
    page_icon="📐",
    layout="wide",
)

st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
  .metric-positive { color: #00c896; }
  .metric-negative { color: #ff4b4b; }
  .stTabs [data-baseweb="tab"] { font-size: 0.9rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("📐 EMA Fan Strategy — Mean-Reversion on EMA50/100/150")
st.caption("Bullish fan → LONG below EMA150 · Bearish fan → SHORT above EMA150 · Exit: Chandelier ATR")

storage = DataStorage()
cfg = get_config()
generator = EmaFanSignalGenerator()

HIST_4H = Path(__file__).parent / "backtest" / "trades_history_4h.csv"
HIST_1D = Path(__file__).parent / "backtest" / "trades_history_1d.csv"
LOG_FILE = Path(__file__).parent / "logs" / "ema_fan.log"

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# Live log helpers
# ─────────────────────────────────────────────────────────────────────

def read_log_tail(n_lines: int = 60) -> str:
    """Read last N lines from the log file."""
    if not LOG_FILE.exists():
        return "Лог-файл не найден. Запустите бэктест."
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        tail = lines[-n_lines:] if len(lines) >= n_lines else lines
        return "".join(tail)
    except Exception as e:
        return f"Ошибка чтения лога: {e}"


def is_backtest_running() -> bool:
    """Heuristic: check if trades_history CSV was recently modified vs log."""
    # If log is newer than results file — probably still running
    if not LOG_FILE.exists():
        return False
    log_mtime = LOG_FILE.stat().st_mtime
    now = time.time()
    # If log was written within last 10 seconds, likely running
    return (now - log_mtime) < 10


def parse_progress_from_log() -> dict:
    """Extract key stats from log tail."""
    text = read_log_tail(200)
    info = {}
    # Extract simulation bar count
    m = re.search(r"Simulating (\d+) bars across (\d+) symbols", text)
    if m:
        info["total_bars"] = int(m.group(1))
        info["total_symbols"] = int(m.group(2))
    # Extract loaded symbols
    m = re.search(r"Loaded and processed (\d+) symbols", text)
    if m:
        info["loaded_symbols"] = int(m.group(1))
    # Extract closed trades count from summary
    m = re.search(r"Total trades:\s+(\d+)", text)
    if m:
        info["total_trades"] = int(m.group(1))
    m = re.search(r"Win rate:\s+([\d.]+)%", text)
    if m:
        info["win_rate"] = float(m.group(1))
    m = re.search(r"Total PnL:\s+\$([\-\d.]+)", text)
    if m:
        info["total_pnl"] = float(m.group(1))
    m = re.search(r"Profit factor:\s+([\d.]+)", text)
    if m:
        info["profit_factor"] = float(m.group(1))
    # Check if done
    info["done"] = "BACKTEST SUMMARY" in text
    return info


@st.cache_data(show_spinner="Загрузка сделок...", ttl=300)
def load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
    return df


def compute_equity_curve(df: pd.DataFrame, initial: float = 10_000.0) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    df_sorted = df.sort_values("exit_time").copy()
    df_sorted["cumulative_pnl"] = df_sorted["pnl_usdt"].cumsum()
    df_sorted["equity"] = initial + df_sorted["cumulative_pnl"]
    return df_sorted.set_index("exit_time")["equity"]


def metrics_row(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("Нет данных. Запустите бэктест.")
        return

    total = len(df)
    wins = (df["pnl_usdt"] > 0).sum()
    wr = wins / total * 100 if total else 0
    total_pnl = df["pnl_usdt"].sum()
    avg_win = df.loc[df["pnl_usdt"] > 0, "pnl_usdt"].mean() if wins else 0
    avg_loss = df.loc[df["pnl_usdt"] <= 0, "pnl_usdt"].mean() if (total - wins) else 0
    pf_num = df.loc[df["pnl_usdt"] > 0, "pnl_usdt"].sum()
    pf_den = abs(df.loc[df["pnl_usdt"] <= 0, "pnl_usdt"].sum())
    pf = pf_num / pf_den if pf_den > 0 else float("inf")

    # Per-direction
    longs = df[df["direction"] == "long"]
    shorts = df[df["direction"] == "short"]
    l_wr = (longs["pnl_usdt"] > 0).mean() * 100 if len(longs) else 0
    s_wr = (shorts["pnl_usdt"] > 0).mean() * 100 if len(shorts) else 0

    cols = st.columns(8)
    cols[0].metric("📊 Сделок", total)
    cols[1].metric("🏆 Winrate", f"{wr:.1f}%")
    cols[2].metric("📈 Лонги", f"{len(longs)} ({l_wr:.0f}%)")
    cols[3].metric("📉 Шорты", f"{len(shorts)} ({s_wr:.0f}%)")
    cols[4].metric("💰 Total PnL", f"${total_pnl:.2f}", delta=f"{total_pnl:+.2f}")
    cols[5].metric("✅ Avg Win", f"${avg_win:.2f}")
    cols[6].metric("❌ Avg Loss", f"${avg_loss:.2f}")
    cols[7].metric("⚖️ Profit Factor", f"{pf:.2f}" if not np.isinf(pf) else "∞")


def equity_chart(df: pd.DataFrame, title: str) -> None:
    if df.empty:
        return
    eq = compute_equity_curve(df)
    if eq.empty:
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values,
        mode="lines",
        line=dict(color="#6c63ff", width=2),
        fill="tozeroy",
        fillcolor="rgba(108,99,255,0.1)",
        name="Equity",
    ))
    fig.update_layout(
        title=title,
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)", tickprefix="$"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def pnl_distribution(df: pd.DataFrame) -> None:
    if df.empty:
        return
    fig = go.Figure()
    wins = df.loc[df["pnl_usdt"] > 0, "pnl_usdt"]
    losses = df.loc[df["pnl_usdt"] <= 0, "pnl_usdt"]
    fig.add_trace(go.Histogram(x=wins, name="Прибыль", marker_color="#00c896", opacity=0.75))
    fig.add_trace(go.Histogram(x=losses, name="Убыток", marker_color="#ff4b4b", opacity=0.75))
    fig.update_layout(
        barmode="overlay",
        title="Распределение PnL",
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickprefix="$"),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)


def winrate_by_symbol(df: pd.DataFrame) -> None:
    if df.empty or len(df) < 5:
        return
    grp = df.groupby("symbol").agg(
        trades=("pnl_usdt", "count"),
        winrate=("pnl_usdt", lambda x: (x > 0).mean() * 100),
        total_pnl=("pnl_usdt", "sum"),
    ).reset_index().sort_values("total_pnl", ascending=False)

    fig = go.Figure()
    colors = ["#00c896" if p > 0 else "#ff4b4b" for p in grp["total_pnl"]]
    fig.add_trace(go.Bar(
        x=grp["symbol"], y=grp["total_pnl"],
        marker_color=colors,
        text=grp["winrate"].map(lambda x: f"{x:.0f}%"),
        textposition="outside",
        name="Total PnL",
    ))
    fig.update_layout(
        title="PnL по монетам (процент над баром — winrate)",
        height=360,
        margin=dict(l=10, r=10, t=40, b=80),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=-45),
        yaxis=dict(tickprefix="$"),
    )
    st.plotly_chart(fig, use_container_width=True)


def trades_table(df: pd.DataFrame) -> None:
    if df.empty:
        return
    display = df.copy()
    display["pnl_usdt"] = display["pnl_usdt"].map(lambda x: f"${x:+.3f}")
    display["return_pct"] = display["return_pct"].map(lambda x: f"{x*100:+.2f}%")
    st.dataframe(
        display[["symbol", "direction", "entry_time", "exit_time",
                 "entry_price", "exit_price", "bars_held",
                 "pnl_usdt", "return_pct", "exit_reason"]],
        use_container_width=True,
        height=400,
    )


# ─────────────────────────────────────────────────────────────────────
# Signal chart for a single symbol
# ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Вычисление сигналов...", ttl=600)
def get_signal_data(symbol: str, timeframe: str) -> pd.DataFrame | None:
    df = storage.load_ohlcv(symbol, timeframe)
    if df is None or len(df) < 200:
        return None
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = generator.compute_signals(df)
    return df


def signal_chart(df: pd.DataFrame, symbol: str, trades_df: pd.DataFrame) -> None:
    if df is None or df.empty:
        st.warning("Недостаточно данных.")
        return

    # Limit to last 300 bars for readability
    df = df.tail(300).copy()

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f"{symbol} — Свечи + EMA Fan", "ATR"],
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df["timestamp"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="OHLCV", increasing_line_color="#00c896", decreasing_line_color="#ff4b4b",
        showlegend=False,
    ), row=1, col=1)

    ema_names = [
        (f"ema_{generator.ema_fast}", f"EMA{generator.ema_fast}", "#f7b731"),
        (f"ema_{generator.ema_mid}", f"EMA{generator.ema_mid}", "#fc5c65"),
        (f"ema_{generator.ema_slow}", f"EMA{generator.ema_slow}", "#45aaf2"),
    ]
    for col_name, label, color in ema_names:
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df[col_name],
                mode="lines", name=label,
                line=dict(color=color, width=1.5),
            ), row=1, col=1)

    # Entry signals
    long_entries = df[df["entry_long"] == True]
    short_entries = df[df["entry_short"] == True]

    if not long_entries.empty:
        fig.add_trace(go.Scatter(
            x=long_entries["timestamp"], y=long_entries["low"] * 0.997,
            mode="markers", name="LONG entry",
            marker=dict(symbol="triangle-up", size=12, color="#00c896"),
        ), row=1, col=1)

    if not short_entries.empty:
        fig.add_trace(go.Scatter(
            x=short_entries["timestamp"], y=short_entries["high"] * 1.003,
            mode="markers", name="SHORT entry",
            marker=dict(symbol="triangle-down", size=12, color="#ff4b4b"),
        ), row=1, col=1)

    # Highlight fan regions
    bull_mask = df["fan_bull"] == True
    bear_mask = df["fan_bear"] == True
    for mask, color, label in [
        (bull_mask, "rgba(0,200,150,0.07)", "Bullish Fan"),
        (bear_mask, "rgba(255,75,75,0.07)", "Bearish Fan"),
    ]:
        if mask.any():
            in_region = False
            start_idx = None
            for i, is_active in enumerate(mask):
                if is_active and not in_region:
                    in_region = True
                    start_idx = i
                elif not is_active and in_region:
                    in_region = False
                    fig.add_vrect(
                        x0=df["timestamp"].iloc[start_idx],
                        x1=df["timestamp"].iloc[i - 1],
                        fillcolor=color, layer="below", line_width=0,
                        annotation_text="" if i - start_idx < 10 else "",
                    )
            if in_region:
                fig.add_vrect(
                    x0=df["timestamp"].iloc[start_idx],
                    x1=df["timestamp"].iloc[-1],
                    fillcolor=color, layer="below", line_width=0,
                )

    # ATR subplot
    if "atr" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["atr"],
            mode="lines", name="ATR",
            line=dict(color="#a29bfe", width=1.5),
        ), row=2, col=1)

    fig.update_layout(
        height=700,
        margin=dict(l=10, r=10, t=60, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────

tab_progress, tab_4h, tab_1d, tab_signal, tab_config = st.tabs([
    "🟢 Прогресс бэктеста",
    "📊 Результаты 4H",
    "📊 Результаты 1D",
    "🔍 Анализ сигналов",
    "⚙️ Конфигурация",
])

# ── Tab: Progress ────────────────────────────────────────────────────
with tab_progress:
    st.subheader("🟢 Запуск и мониторинг бэктеста")

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("**Запустить бэктест:**")
        c1, c2, c3 = st.columns(3)
        run_4h = c1.button("▶ 4H", type="primary", key="run_4h_prog")
        run_1d = c2.button("▶ 1D", type="primary", key="run_1d_prog")
        run_both = c3.button("▶ Оба", key="run_both_prog")

    with col_r:
        auto_refresh = st.toggle("🔄 Авто-обновление (5 сек)", value=False)

    st.divider()

    # Launch in subprocess so Streamlit stays responsive
    if run_4h or run_1d or run_both:
        tf_arg = []
        if run_4h:
            tf_arg = ["--tf", "4h"]
        elif run_1d:
            tf_arg = ["--tf", "1d"]
        # run_both = no --tf flag
        cmd = [sys.executable, str(_bot_dir / "main.py"), "backtest"] + tf_arg
        subprocess.Popen(cmd, cwd=str(_root))
        st.success("✅ Бэктест запущен в фоне! Лог обновится ниже.")
        time.sleep(1)
        st.rerun()

    # Parse progress
    progress_info = parse_progress_from_log()
    is_done = progress_info.get("done", False)
    is_running = is_backtest_running() and not is_done

    # Status indicator
    if is_running:
        st.markdown("### ⏳ Бэктест выполняется...")
    elif is_done and progress_info:
        st.markdown("### ✅ Бэктест завершён")
    else:
        st.markdown("### 💤 Бэктест не запущен")

    # Key metrics from log
    if progress_info:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Символов загружено", progress_info.get("loaded_symbols", "—"))
        m2.metric("Баров в симуляции", f"{progress_info.get('total_bars', 0):,}" if progress_info.get('total_bars') else "—")
        m3.metric("Сделок найдено", progress_info.get("total_trades", "—"))
        wr = progress_info.get("win_rate")
        m4.metric("Win Rate", f"{wr:.1f}%" if wr is not None else "—")
        pnl = progress_info.get("total_pnl")
        m5.metric("Total PnL", f"${pnl:.2f}" if pnl is not None else "—",
                  delta=f"{pnl:+.2f}" if pnl is not None else None)

    st.divider()

    # Live log display
    st.markdown("**📋 Лог (последние 60 строк):**")
    log_text = read_log_tail(60)

    # Color-code log levels
    log_lines = log_text.split("\n")
    colored = []
    for line in log_lines:
        if "ERROR" in line or "CRITICAL" in line:
            colored.append(f"🔴 {line}")
        elif "WARNING" in line:
            colored.append(f"🟡 {line}")
        elif "SUMMARY" in line or "Done" in line or "Saved" in line:
            colored.append(f"🟢 {line}")
        else:
            colored.append(line)

    st.code("\n".join(colored), language="", line_numbers=False)

    col_ref, col_hint = st.columns([1, 4])
    with col_ref:
        if st.button("🔄 Обновить лог"):
            st.rerun()
    with col_hint:
        st.caption("Или включи авто-обновление выше.")

    # Auto-refresh
    if auto_refresh and is_running:
        time.sleep(5)
        st.rerun()

# ── Tab: 4H ──────────────────────────────────────────────────────────
with tab_4h:
    st.subheader("Таймфрейм 4H — EMA Fan Mean-Reversion")

    col_run, col_info = st.columns([1, 4])
    with col_run:
        if st.button("▶ Запустить бэктест 4H", type="primary"):
            with st.spinner("Запуск бэктеста 4H..."):
                engine = EmaFanBacktestEngine()
                engine.run("4h")
            st.cache_data.clear()
            st.success("Готово!")
            st.rerun()
    with col_info:
        st.caption("Нажми кнопку для запуска или используй: `poetry run python ema_fan_bot/main.py backtest --tf 4h`")

    df_4h = load_trades(HIST_4H)
    metrics_row(df_4h)

    if not df_4h.empty:
        c1, c2 = st.columns([2, 1])
        with c1:
            equity_chart(df_4h, "Equity Curve — 4H")
        with c2:
            pnl_distribution(df_4h)

        winrate_by_symbol(df_4h)

        # Direction breakdown
        if "direction" in df_4h.columns:
            c1, c2 = st.columns(2)
            with c1:
                longs = df_4h[df_4h["direction"] == "long"]
                st.markdown(f"**📈 Лонги** — {len(longs)} сделок")
                if not longs.empty:
                    winrate_by_symbol(longs)
            with c2:
                shorts = df_4h[df_4h["direction"] == "short"]
                st.markdown(f"**📉 Шорты** — {len(shorts)} сделок")
                if not shorts.empty:
                    winrate_by_symbol(shorts)

        with st.expander("📋 Таблица сделок (4H)"):
            trades_table(df_4h)


# ── Tab: 1D ──────────────────────────────────────────────────────────
with tab_1d:
    st.subheader("Таймфрейм 1D — EMA Fan Mean-Reversion")

    col_run, col_info = st.columns([1, 4])
    with col_run:
        if st.button("▶ Запустить бэктест 1D", type="primary"):
            with st.spinner("Запуск бэктеста 1D..."):
                engine = EmaFanBacktestEngine()
                engine.run("1d")
            st.cache_data.clear()
            st.success("Готово!")
            st.rerun()
    with col_info:
        st.caption("Нажми кнопку для запуска или используй: `poetry run python ema_fan_bot/main.py backtest --tf 1d`")

    df_1d = load_trades(HIST_1D)
    metrics_row(df_1d)

    if not df_1d.empty:
        c1, c2 = st.columns([2, 1])
        with c1:
            equity_chart(df_1d, "Equity Curve — 1D")
        with c2:
            pnl_distribution(df_1d)

        winrate_by_symbol(df_1d)

        if "direction" in df_1d.columns:
            c1, c2 = st.columns(2)
            with c1:
                longs = df_1d[df_1d["direction"] == "long"]
                st.markdown(f"**📈 Лонги** — {len(longs)} сделок")
                if not longs.empty:
                    winrate_by_symbol(longs)
            with c2:
                shorts = df_1d[df_1d["direction"] == "short"]
                st.markdown(f"**📉 Шорты** — {len(shorts)} сделок")
                if not shorts.empty:
                    winrate_by_symbol(shorts)

        with st.expander("📋 Таблица сделок (1D)"):
            trades_table(df_1d)

    # Side-by-side comparison
    if not df_4h.empty and not df_1d.empty:
        st.divider()
        st.subheader("📊 Сравнение таймфреймов")
        compare_data = []
        for label, df in [("4H", df_4h), ("1D", df_1d)]:
            if df.empty:
                continue
            wr = (df["pnl_usdt"] > 0).mean() * 100
            pnl = df["pnl_usdt"].sum()
            wins = df.loc[df["pnl_usdt"] > 0, "pnl_usdt"]
            losses = df.loc[df["pnl_usdt"] <= 0, "pnl_usdt"]
            pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else float("inf")
            compare_data.append({
                "Таймфрейм": label,
                "Сделок": len(df),
                "Winrate %": f"{wr:.1f}%",
                "Total PnL $": f"${pnl:.2f}",
                "Profit Factor": f"{pf:.2f}" if not np.isinf(pf) else "∞",
                "Avg bars held": f"{df['bars_held'].mean():.1f}",
            })
        if compare_data:
            st.dataframe(pd.DataFrame(compare_data), use_container_width=True)


# ── Tab: Signal Viewer ────────────────────────────────────────────────
with tab_signal:
    st.subheader("🔍 Визуальный аудит сигналов")

    col_sym, col_tf = st.columns([3, 1])
    all_symbols_4h = storage.list_symbols("4h")
    all_symbols_1d = storage.list_symbols("1d")
    all_symbols = sorted(set(all_symbols_4h + all_symbols_1d))
    all_symbols = [s for s in all_symbols if s != "BTC/USDT:USDT"]

    with col_sym:
        selected_sym = st.selectbox("Монета", all_symbols, key="signal_sym")
    with col_tf:
        selected_tf = st.selectbox("Таймфрейм", ["4h", "1d"], key="signal_tf")

    if selected_sym:
        df_sig = get_signal_data(selected_sym, selected_tf)

        if df_sig is not None:
            n_long = df_sig.get("entry_long", pd.Series(dtype=bool)).sum()
            n_short = df_sig.get("entry_short", pd.Series(dtype=bool)).sum()
            n_bull_fan = df_sig.get("fan_bull", pd.Series(dtype=bool)).sum()
            n_bear_fan = df_sig.get("fan_bear", pd.Series(dtype=bool)).sum()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📈 LONG сигналов", int(n_long))
            m2.metric("📉 SHORT сигналов", int(n_short))
            m3.metric("🐂 Bullish Fan баров", int(n_bull_fan))
            m4.metric("🐻 Bearish Fan баров", int(n_bear_fan))

            # Get trades for this symbol
            symbol_trades = pd.DataFrame()
            for df_t, tf_label in [(df_4h, "4h"), (df_1d, "1d")]:
                if not df_t.empty and selected_tf == tf_label:
                    sym_mask = df_t["symbol"] == selected_sym
                    if sym_mask.any():
                        symbol_trades = df_t[sym_mask]

            signal_chart(df_sig, selected_sym, symbol_trades)

            if not symbol_trades.empty:
                st.markdown(f"**Сделки по {selected_sym}:**")
                trades_table(symbol_trades)
        else:
            st.warning(f"Недостаточно данных для {selected_sym} [{selected_tf}] (нужно >200 свечей)")


# ── Tab: Config ───────────────────────────────────────────────────────
with tab_config:
    st.subheader("⚙️ Параметры стратегии")

    strat = cfg.get("strategy", {})
    risk = cfg.get("risk", {})
    bt = cfg.get("backtest", {})

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**📐 EMA & Сигналы**")
        st.markdown(f"- EMA быстрая: **{strat.get('ema_fast', 50)}**")
        st.markdown(f"- EMA средняя: **{strat.get('ema_mid', 100)}**")
        st.markdown(f"- EMA медленная: **{strat.get('ema_slow', 150)}**")
        st.markdown(f"- ATR период: **{strat.get('atr_period', 14)}**")
        st.markdown(f"- Fan gap min %: **{strat.get('fan_gap_min_pct', 0.0)}**")

    with c2:
        st.markdown("**💰 Риск**")
        st.markdown(f"- Размер позиции: **${risk.get('trade_size_usdt', 10)} USDT**")
        st.markdown(f"- Stop-loss ATR ×: **{risk.get('stop_loss_atr_mult', 2.0)}**")
        st.markdown(f"- Trailing ATR ×: **{risk.get('trailing_atr_mult', 2.5)}**")

    with c3:
        st.markdown("**🧪 Бэктест**")
        st.markdown(f"- Комиссия: **{bt.get('taker_fee', 0.0005)*100:.2f}%**")
        st.markdown(f"- Проскальзывание: **{bt.get('slippage', 0.0003)*100:.2f}%**")
        st.markdown(f"- Макс. позиций: **{bt.get('max_concurrent_positions', 20)}**")

    st.divider()
    st.markdown("**📋 Логика стратегии**")
    st.info("""
**Бычий веер** (EMA50 > EMA100 > EMA150):
→ Сигнал LONG: цена закрылась ниже EMA150 (пробой поддержки в тренде вверх)
→ Идея: временная слабость → возврат к тренду

**Медвежий веер** (EMA50 < EMA100 < EMA150):
→ Сигнал SHORT: цена закрылась выше EMA150 (пробой сопротивления в тренде вниз)
→ Идея: временный рост → возврат к нисходящему тренду

**Выход**: Chandelier Exit — trailing stop на базе ATR от максимума/минимума позиции.
**SL**: 2.0 × ATR от цены входа. **Вход**: на открытии следующей свечи.
    """)

    with st.expander("Сырой YAML конфиг"):
        import yaml
        st.code(yaml.dump(cfg, allow_unicode=True, default_flow_style=False), language="yaml")
