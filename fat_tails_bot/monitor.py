import os
import sys
from pathlib import Path
import time
import re

_root = Path(__file__).resolve().parent.parent
_bot_dir = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_bot_dir) not in sys.path:
    sys.path.insert(0, str(_bot_dir))
os.chdir(_bot_dir)

from shared.config_loader import set_active_bot, get_config
set_active_bot("fat_tails_bot")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from shared.data.storage import DataStorage
from fat_tails_bot.screener.screener import FatTailsScreener, load_daily_ohlcv
from fat_tails_bot.strategy.signals import SignalGenerator
from fat_tails_bot.backtest.engine import BacktestEngine

st.set_page_config(
    page_title="Fat Tails Bot Farm (D1)",
    page_icon="🌪️",
    layout="wide"
)

st.title("🌪️ Fat Tails Bot Farm (D1)")

storage = DataStorage()
cfg = get_config()
project_root = str(_root)
HISTORY_FILE = "backtest/trades_history.csv"

@st.cache_data(show_spinner="Выполняется математический расчет скринера...", ttl=3600)
def run_screener_cached() -> pd.DataFrame:
    screener = FatTailsScreener()
    return screener.screen_universe()

tab_step1, tab_step2, tab_step3, tab_theory = st.tabs([
    "Step 1: Отбор монет (Скринер)",
    "Step 2: Бэктест (Ферма)",
    "Step 3: Результаты и Сделки",
    "Теория и Правила"
])

# =====================================================================
# STEP 1: SCREENER & DATA SYNC
# =====================================================================
with tab_step1:
    st.subheader("Шаг 1: Синхронизация данных и Отбор (Скринер)")
    st.write("Обнови локальную базу данных свечей с Bybit, затем отфильтруй монеты по свойствам тяжелых хвостов.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🚀 Запустить синхронизацию данных (Фоновый режим)", type="primary", use_container_width=True):
            import subprocess
            subprocess.Popen(["poetry", "run", "python", "fat_tails_bot/main.py", "collect"])
            st.toast("Процесс запущен! Смотрите логи ниже.")
    with col2:
        auto_refresh = st.checkbox("🔄 Автообновление таблицы и логов (каждые 3 сек)", value=False)

    st.markdown("### 🖥️ Консоль (Live Logs)")
    log_file = os.path.join(project_root, "fat_tails_bot", "logs", "fat_tails.log")
    log_text = "Лог файл пока пуст или не создан."
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()[-15:]
            log_text = "".join(lines)
    st.code(log_text, language="log")

    st.markdown("### 📊 Отбор Монет (Скринер)")
    st.cache_data.clear()
    
    total_target = "?"
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            all_logs = f.read()
            matches = re.findall(r"\[\d+/(\d+)\] Collecting", all_logs)
            if matches:
                total_target = matches[-1]

    df_screener = run_screener_cached()
    if df_screener.empty:
        st.warning("Скринер еще не нашел подходящих монет.")
    else:
        passed_count = int(df_screener["selected"].sum())
        total_count = len(storage.list_symbols("4h"))
        
        c1, c2 = st.columns(2)
        c1.metric("Прогресс скачивания", f"{total_count} из {total_target}")
        c2.metric("Прошли фильтр (Hurst & Kurtosis)", f"{passed_count} монет")
        
        st.dataframe(
            df_screener,
            use_container_width=True,
            height=400
        )

# =====================================================================
# STEP 2: BACKTEST
# =====================================================================
def render_coin_badges(all_symbols, processed_set):
    html = "<div style='display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 15px;'>"
    for sym in all_symbols:
        name = sym.split('/')[0]
        if sym in processed_set:
            style = "border: 1px solid #10b981; color: #10b981; background-color: rgba(16, 185, 129, 0.1); padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; transition: all 0.3s ease;"
        else:
            style = "border: 1px solid #4b5563; color: #9ca3af; background-color: transparent; padding: 4px 12px; border-radius: 20px; font-size: 13px; transition: all 0.3s ease;"
        html += f"<div style='{style}'>{name}</div>"
    html += "</div>"
    return html

with tab_step2:
    st.subheader("Шаг 2: Запуск Мульти-активного Бэктеста")
    st.write("Прогон исторической симуляции (ферма ботов) с фиксой $10 USDT на сделку.")
    
    if st.button("Запустить Бэктест Фермы", type="primary"):
        with st.status("Выполняется историческая симуляция...", expanded=True) as status:
            st.write("1. Загрузка данных скринера...")
            df_screen = run_screener_cached()
            selected_symbols = df_screen[df_screen["selected"]]["symbol"].tolist()
            
            st.write(f"2. Подготовка признаков (Signals) для {len(selected_symbols)} монет...")
            prepared_data = {}
            progress_bar = st.progress(0)
            badges_placeholder = st.empty()
            
            generator = SignalGenerator()
            processed_symbols = set()
            
            badges_placeholder.markdown(render_coin_badges(selected_symbols, processed_symbols), unsafe_allow_html=True)
            
            for i, sym in enumerate(selected_symbols):
                df = load_daily_ohlcv(storage, sym)
                if df is not None:
                    prepared_data[sym] = generator.compute_signals(df)
                
                processed_symbols.add(sym)
                badges_placeholder.markdown(render_coin_badges(selected_symbols, processed_symbols), unsafe_allow_html=True)
                progress_bar.progress((i + 1) / len(selected_symbols))
            
            st.write("3. Запуск мульти-активного движка (Портфель $10,000)...")
            engine = BacktestEngine()
            result = engine.multi_asset_backtest(
                prepared_data,
                initial_capital=10000,
                max_concurrent_positions=cfg.get("strategy", {}).get("max_concurrent_positions", 5),
                execute_next_open=True
            )
            
            st.write("4. Сохранение сделок в trades_history.csv...")
            os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
            trade_dicts = []
            for t in result.trades:
                trade_dicts.append({
                    "symbol": t.symbol,
                    "entry_time": t.entry_date,
                    "exit_time": t.exit_date,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "initial_sl": t.stop_price,
                    "exit_reason": t.exit_reason,
                    "pnl_usdt": t.pnl,
                    "return_pct": t.pnl / (t.entry_price * t.size) if t.entry_price and t.size else 0
                })
            df_trades = pd.DataFrame(trade_dicts)
            df_trades.to_csv(HISTORY_FILE, index=False)
            
            status.update(label="Бэктест успешно завершен!", state="complete", expanded=False)
            st.success(f"Сохранено сделок: {len(df_trades)}. Перейдите на вкладку Результаты!")

# =====================================================================
# STEP 3: RESULTS
# =====================================================================
with tab_step3:
    st.subheader("Шаг 3: Результаты Фермы и Сделки")
    
    if not os.path.exists(HISTORY_FILE):
        st.warning("Файл trades_history.csv не найден. Запустите бэктест на Шаге 2.")
    else:
        df_trades = pd.read_csv(HISTORY_FILE)
        if df_trades.empty:
            st.warning("История сделок пуста.")
        else:
            total_trades = len(df_trades)
            wins = df_trades[df_trades["pnl_usdt"] > 0]
            losses = df_trades[df_trades["pnl_usdt"] <= 0]
            win_rate = len(wins) / total_trades * 100
            total_pnl = df_trades["pnl_usdt"].sum()
            
            st.markdown("### Общая статистика фермы")
            m1, m2, m3 = st.columns(3)
            m1.metric("Всего сделок", str(total_trades))
            m2.metric("Win Rate", f"{win_rate:.1f}%")
            m3.metric("Total PnL", f"${total_pnl:.2f}")
            
            st.markdown("### Рентген Сделок (Trade X-Ray)")
            df_trades['exit_reason_ru'] = df_trades['exit_reason'].replace({
                'stop_loss': 'Stop Loss (Chandelier)',
                'regime_exit': 'Regime Exit (EMA5 / CLV)'
            })
            
            options = [
                f"{str(row['entry_time'])[:10]} | {row['symbol']} | PnL: ${row['pnl_usdt']:+.2f} ({row['exit_reason_ru']})"
                for idx, row in df_trades.iterrows()
            ]
            
            selected_trade_idx = st.selectbox("Выберите сделку для анализа:", range(len(options)), format_func=lambda i: options[i])
            trade = df_trades.iloc[selected_trade_idx]
            t_symbol = trade["symbol"]
            t_entry_dt = pd.to_datetime(trade["entry_time"]).tz_localize(None)
            t_exit_dt = pd.to_datetime(trade["exit_time"]).tz_localize(None)
            
            df_ohlc = load_daily_ohlcv(storage, t_symbol)
            if df_ohlc is not None and len(df_ohlc) > 10:
                df_ohlc["dt"] = pd.to_datetime(df_ohlc["timestamp"]).dt.tz_localize(None)
                start_window = t_entry_dt - pd.Timedelta(days=20)
                end_window = t_exit_dt + pd.Timedelta(days=15)
                df_ohlc["ema5"] = df_ohlc["close"].ewm(span=5, adjust=False).mean()
                df_slice = df_ohlc[(df_ohlc["dt"] >= start_window) & (df_ohlc["dt"] <= end_window)]
                
                fig_cand = go.Figure()
                fig_cand.add_trace(go.Candlestick(
                    x=df_slice["dt"], open=df_slice["open"], high=df_slice["high"],
                    low=df_slice["low"], close=df_slice["close"], name=t_symbol
                ))
                fig_cand.add_trace(go.Scatter(
                    x=df_slice["dt"], y=df_slice["ema5"], mode="lines",
                    line=dict(color="#60a5fa", width=2), name="EMA5"
                ))
                
                fig_cand.add_trace(go.Scatter(
                    x=[t_entry_dt], y=[trade["entry_price"]], mode="markers+text",
                    marker=dict(symbol="triangle-down", size=14, color="#34d399"),
                    text=[f"ВХОД SHORT (${trade['entry_price']:.4f})"], textposition="top center", name="Вход"
                ))
                fig_cand.add_trace(go.Scatter(
                    x=[t_exit_dt], y=[trade["exit_price"]], mode="markers+text",
                    marker=dict(symbol="x", size=13, color="#ef4444" if trade["pnl_usdt"] < 0 else "#34d399"),
                    text=[f"ВЫХОД ({trade['exit_reason_ru']})"], textposition="bottom center", name="Выход"
                ))
                
                fig_cand.update_layout(
                    template="plotly_dark",
                    title=f"Детальный анализ сделки: {t_symbol} (PnL: ${trade['pnl_usdt']:+.2f})",
                    yaxis_title="Цена ($)", xaxis_title="Дата", height=450, margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig_cand, use_container_width=True)

# =====================================================================
# STEP 4: THEORY
# =====================================================================
with tab_theory:
    st.subheader("Теория и Правила (AGENTS.md)")
    st.write("Смотрите оригинальные правила стратегии Fat Tails. BTC исключается, 1% риска от капитала (макс $10 USDT на сделку).")

if auto_refresh:
    time.sleep(3)
    st.rerun()
