"""Streamlit Dashboard for Fat Tails D1 Bot — quantitative screening and trade analysis with full transparency."""

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
set_active_bot("fat_tails_bot")

import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy.stats import norm, kurtosis

from shared.data.storage import DataStorage
from fat_tails_bot.screener.screener import FatTailsScreener, load_daily_ohlcv
from fat_tails_bot.strategy.signals import SignalGenerator
from fat_tails_bot.backtest.engine import BacktestEngine


# ─────────────────────────────────────────────────────────────
# 1. СТИЛИ И НАСТРОЙКА СТРАНИЦЫ (UI/UX PRO MAX)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fat Tails D1 Quant Engine",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px;
        border-radius: 12px;
        border-left: 6px solid #3b82f6;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
        margin-bottom: 24px;
        color: white;
    }
    
    .info-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    
    .metric-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    
    .highlight-blue {
        color: #60a5fa;
        font-weight: 600;
    }
    
    .highlight-green {
        color: #34d399;
        font-weight: 600;
    }
    
    .highlight-red {
        color: #f87171;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 28px; font-weight: 700;">🏆 Fat Tails D1 Quant Engine & Dashboard</h1>
    <p style="margin-top: 8px; color: #94a3b8; font-size: 15px;">
        Количественная система на дневных свечах (D1). Поиск аномальных эксцессов распределения и ловля фазовых переходов волатильности.<br>
        <b>Мани-менеджмент:</b> Строго фиксированная позиция <span style="color:#60a5fa;">$10 USDT</span> | Плечо <span style="color:#34d399;">x1</span> | Выход без статического Take Profit (Chandelier Trailing).
    </p>
</div>
""", unsafe_allow_html=True)

storage = DataStorage()
cfg = get_config()


# ─────────────────────────────────────────────────────────────
# 2. КЭШИРОВАННЫЕ РАСЧЕТЫ И ОБРАБОТЧИКИ
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Выполняется математический расчет скринера...", ttl=3600)
def run_screener_cached() -> pd.DataFrame:
    screener = FatTailsScreener()
    return screener.screen_universe()


@st.cache_data(show_spinner="Загружается история сделок...", ttl=60)
def get_trades_history() -> pd.DataFrame:
    trades_file = Path("backtest/trades_history.csv")
    if not trades_file.exists():
        engine = BacktestEngine()
        return engine.run_backtest()
    return pd.read_csv(trades_file)


# ─────────────────────────────────────────────────────────────
# 3. НАВИГАЦИЯ ПО ВКАДКАМ
# ─────────────────────────────────────────────────────────────
tab_theory, tab_screener, tab_trades = st.tabs([
    "📖 Как это работает (Теория и правила)",
    "🔍 Отбор монет (Скринер и доказательства)",
    "📈 Бэктест и журнал сделок (Рентген)",
])

# =============================================================
# ВКЛАДКА 1: ТЕОРИЯ И ПРАВИЛА ВХОДА/ВЫХОДА
# ==============================================================
with tab_theory:
    st.subheader("💡 Математическое обоснование и правила без скрытых допущений")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(r"""
        ### 1. Почему «Тяжёлые хвосты» (Excess Kurtosis)?
        Классическая финансовая математика ошибочно предполагает, что доходности активов следуют нормальному распределению Гаусса. На реальном крипто-рынке распределения имеют аномально **тяжёлые хвосты** (Excess Kurtosis $K \gg 5.0$, в реальности от 10 до 45).
        
        **Что это значит для трейдера?**
        - Экстремально резкие ценовые импульсы и обвалы (чёрные лебеди, пробои флэтов) происходят на порядок чаще, чем предсказывает теория Гаусса.
        - Идеальная стратегия должна отказываться от торговли в боковике и поджидать именно эти редкие, но мощнейшие аномальные выбросы.
        
        ### 2. Зачем нужен показатель Хёрста ($H$)?
        Показатель Хёрста оценивает устойчивость (память) временного ряда:
        - $H \approx 0.5$ — случайное блуждание.
        - $H > 0.5$ — персистентный ряд (тренд имеет свойство продолжаться).
        - На дневных графиках крипты отбираются монеты с $H \ge 0.35$ и высоким эксцессом $K \ge 5.0$, что идентифицирует взрывоопасные активы с потенциалом затяжного обвала.
        """)
        
    with col2:
        st.markdown(r"""
        ### 3. Полная прозрачность: Как и почему мы входим?
        Вход в шорт осуществляется исключительно при совпадении двух фазовых условий:
        1. **Взрыв волатильности (Z-Score True Range $> 2.5$)** на свече $D-1$: истинный диапазон дня превышает среднее на 2.5+ стандартного отклонения (начало импульса).
        2. **Кульминация продаж ($CLV \le 0.25$)**: цена закрытия свечи $D-1$ находится в самых нижних 25% диапазона (медведи полностью захватили контроль).
        3. **Подтверждение (День $D$)**: свеча продолжения падения ($close_D < close_{D-1}$). Вход совершается по закрытию дня $D$.
        
        ### 4. Почему НЕТ фиксированного Take Profit?
        В стратегии на тяжёлых хвостах статический Take Profit является фатальной ошибкой: он обрезает асимметричный доход.
        - **Начальный стоп (SL):** устанавливается на $1.5 \times ATR_{14}$ выше точки входа.
        - **Динамический трейлинг (Chandelier Exit):** каждый последующий день стоп смещается вниз к отметке $low + 2.0 \times ATR$. Он **никогда не двигается вверх**.
        - Когда импульс иссякает и цена пересекает трейлинг-стоп снизу вверх, мы закрываем сделку, собирая движения до +50% и более!
        """)
        
    st.info("📌 **Никаких скрытых параметров:** Все пороги заданы в `fat_tails_bot/config/settings.yaml` и проверяются прозрачно в каждой сделке на соседних вкладках.")

# ==============================================================
# ВКЛАДКА 2: СКРИНЕР И ДОКАЗАТЕЛЬСТВО ХВОСТОВ
# ==============================================================
with tab_screener:
    st.subheader("🔍 Квантитативный фильтр монет по Эксцессу ($K$) и Хёрсту ($H$)")
    st.write("На этом этапе мы сканируем единую базу исторических свечей (D1), рассчитываем реальный эксцесс распределения доходностей и отфильтровываем шлак.")
    
    col_btn1, col_btn2 = st.columns([2, 5])
    with col_btn1:
        if st.button("⚡ Обновить / Запустить математический отбор", type="primary"):
            st.cache_data.clear()
            st.rerun()

    df_screener = run_screener_cached()
    
    if df_screener.empty:
        st.error("Нет данных в базе. Запустите сбор данных через `poetry run python ml_swing_bot/main.py collect`")
    else:
        passed_count = int(df_screener["selected"].sum())
        total_count = len(df_screener)
        max_k = df_screener["excess_kurtosis"].max()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Всего монет проанализировано", str(total_count))
        c2.metric(r"Отобрано ($H \ge 0.35, K \ge 5.0$)", f"{passed_count} монет")
        c3.metric("Максимальный Эксцесс (K)", f"{max_k:.1f}")
        c4.metric("Рабочий таймфрейм", "1D (Дневка)")

        st.markdown("### 📊 Табличный отчет по всем монетам (без купюр)")
        
        # Format df for presentation
        df_display = df_screener.copy()
        df_display["Статус отбора"] = df_display["selected"].apply(lambda x: "✅ Выбрана" if x else "❌ Отклонена")
        df_display.rename(columns={
            "symbol": "Торговая пара",
            "hurst_exponent": "Показатель Хёрста (H)",
            "excess_kurtosis": "Эксцесс доходностей (K)",
            "bars_1d": "Количество баров D1"
        }, inplace=True)
        
        st.dataframe(
            df_display[["Торговая пара", "Показатель Хёрста (H)", "Эксцесс доходностей (K)", "Количество баров D1", "Статус отбора"]],
            use_container_width=True,
            height=350
        )

        st.markdown("---")
        st.subheader("🔬 Визуальное доказательство «Тяжёлого хвоста» и взрывов волатильности")
        st.write("Выберите любую торговую пару из списка, чтобы своими глазами увидеть реальное распределение доходностей по сравнению с теоретической кривой Гаусса:")
        
        selected_sym = st.selectbox(
            "Выберите монету для детального математического осмотра:",
            df_screener["symbol"].tolist(),
            index=0
        )

        df_coin = load_daily_ohlcv(storage, selected_sym)
        if df_coin is not None and len(df_coin) > 30:
            returns = np.log(df_coin["close"] / df_coin["close"].shift(1)).dropna() * 100.0  # as percentage
            
            row_k = df_screener[df_screener["symbol"] == selected_sym]["excess_kurtosis"].values[0]
            row_h = df_screener[df_screener["symbol"] == selected_sym]["hurst_exponent"].values[0]

            col_chart1, col_chart2 = st.columns([1, 1])
            
            with col_chart1:
                st.markdown(f"**Распределение доходностей {selected_sym} (в % за день)**")
                st.caption(f"Фактический Эксцесс: **K = {row_k}** (Кривая Гаусса имеет K = 0). Выступающие «крылья» по бокам — это те самые тяжелые хвосты-катастрофы.")
                
                fig_dist = px.histogram(
                    x=returns, 
                    nbins=70, 
                    histnorm="probability density",
                    color_discrete_sequence=["#3b82f6"],
                    labels={"x": "Дневная доходность (%)", "y": "Плотность вероятности"}
                )
                
                # Overlay Gauss normal distribution
                mu, std = norm.fit(returns)
                x_vals = np.linspace(returns.min(), returns.max(), 100)
                y_gauss = norm.pdf(x_vals, mu, std)
                fig_dist.add_trace(go.Scatter(x=x_vals, y=y_gauss, mode="lines", name="Теоретическая нормальная (Гаусс)", line=dict(color="#f87171", width=2)))
                
                fig_dist.update_layout(
                    template="plotly_dark",
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=380,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                
            with col_chart2:
                st.markdown(f"**Фазовые переходы волатильности (Z-Score True Range) для {selected_sym}**")
                st.caption("Красная горизонтальная линия (Z=2.5) — порог взрыва волатильности, сигнализирующий о готовящемся сильном импульсе.")
                
                generator = SignalGenerator()
                df_coin_ind = generator.compute_signals(df_coin)
                
                fig_z = go.Figure()
                fig_z.add_trace(go.Bar(
                    x=df_coin_ind["timestamp"],
                    y=df_coin_ind["tr_zscore"],
                    name="Z-Score Истинного диапазона",
                    marker_color=np.where(df_coin_ind["tr_zscore"] >= 2.5, "#f59e0b", "#475569")
                ))
                fig_z.add_hline(y=2.5, line_dash="dash", line_color="#ef4444", annotation_text="Порог входа (Z = 2.5)")
                fig_z.update_layout(
                    template="plotly_dark",
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=380,
                    yaxis_title="Z-Score",
                    xaxis_title="Дата"
                )
                st.plotly_chart(fig_z, use_container_width=True)

# ==============================================================
# ВКЛАДКА 3: БЭКТЕСТ И РЕНТГЕН СДЕЛОК
# ==============================================================
with tab_trades:
    st.subheader("📈 Результаты симуляции на отобранных монетах и детальный рентген каждой сделки")
    
    col_bt1, col_bt2 = st.columns([2, 5])
    with col_bt1:
        if st.button("🚀 Пересчитать симуляцию (Бэктест)", type="primary"):
            engine = BacktestEngine()
            engine.run_backtest()
            st.cache_data.clear()
            st.rerun()

    df_trades = get_trades_history()
    
    if df_trades.empty:
        st.warning("В текущем прогоне не сгенерировано ни одной сделки или файл пуст.")
    else:
        # Calculate summary stats
        total_pnl = df_trades["pnl_usdt"].sum()
        win_rate = (df_trades["pnl_usdt"] > 0).mean() * 100.0
        total_trades = len(df_trades)
        best_trade = df_trades["pnl_usdt"].max()
        best_trade_row = df_trades.loc[df_trades["pnl_usdt"].idxmax()]
        avg_holding = df_trades["bars_held"].mean()
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Итоговый PnL (USDT)", f"${total_pnl:+.2f}", delta=f"от $10 фиксированного риска")
        m2.metric("Винрейт (Процент побед)", f"{win_rate:.1f}%", help="В стратегиях без фиксированного TP винрейт 15-25% является нормой: одна прибыль покрывает серию коротких стопов.")
        m3.metric("Самый крупный выигрыш", f"${best_trade:+.2f}", delta=f"{best_trade_row['return_pct']*100:.1f}% ({best_trade_row['symbol']})")
        m4.metric("Среднее время в сделке", f"{avg_holding:.1f} дней")

        st.markdown("### 📈 Кривая накопления капитала (USDT)")
        
        # Build cumulative equity curve
        df_eq = df_trades.copy()
        df_eq["cum_pnl"] = df_eq["pnl_usdt"].cumsum()
        
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=pd.to_datetime(df_eq["exit_time"]), 
            y=df_eq["cum_pnl"], 
            mode="lines+markers",
            name="Cumulative PnL ($)",
            line=dict(color="#10b981", width=3),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.1)"
        ))
        fig_eq.update_layout(
            template="plotly_dark",
            yaxis_title="Прибыль / Убыток ($ USDT)",
            xaxis_title="Дата закрытия сделки",
            height=320,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Полный журнал всех сделок (Никаких умолчаний)")
        st.write("Ниже представлен детальный аутдит каждой сделки: смотрите, при каких именно значениях индикаторов ($Z_{TR}$ и $CLV$) бот открыл позицию, и сколько дней держал трейлинг.")
        
        # Prepare table for display
        df_show = df_trades.copy()
        df_show["entry_time"] = pd.to_datetime(df_show["entry_time"]).dt.strftime("%Y-%m-%d")
        df_show["exit_time"] = pd.to_datetime(df_show["exit_time"]).dt.strftime("%Y-%m-%d")
        df_show["return_pct_formatted"] = (df_show["return_pct"] * 100).round(2).astype(str) + "%"
        df_show["pnl_formatted"] = df_show["pnl_usdt"].apply(lambda x: f"${x:+.2f}")
        
        df_show.rename(columns={
            "symbol": "Монета",
            "direction": "Направление",
            "entry_time": "Дата входа",
            "entry_price": "Цена входа",
            "initial_sl": "Начальный SL",
            "entry_tr_zscore": "Z-Score TR входа",
            "entry_clv": "CLV входа",
            "exit_time": "Дата выхода",
            "exit_price": "Цена выхода",
            "bars_held": "Удержание (дн)",
            "pnl_formatted": "PnL (USDT)",
            "return_pct_formatted": "Доходность %",
            "exit_reason": "Причина выхода"
        }, inplace=True)

        st.dataframe(
            df_show[[
                "Монета", "Дата входа", "Цена входа", "Z-Score TR входа", "CLV входа", 
                "Начальный SL", "Дата выхода", "Цена выхода", "Удержание (дн)", 
                "PnL (USDT)", "Доходность %", "Причина выхода"
            ]],
            use_container_width=True,
            height=380
        )

        st.markdown("---")
        st.subheader("🔍 Интерактивный рентген отдельной сделки")
        st.write("Выберите конкретно интересующую вас сделку, чтобы увидеть её график входа и выхода в контексте дневных свечей:")
        
        options = [
            f"Сделка #{idx+1} | {row['entry_time'][:10]} | {row['symbol']} | PnL: ${row['pnl_usdt']:+.2f} ({row['exit_reason']})"
            for idx, row in df_trades.iterrows()
        ]
        
        selected_trade_idx = st.selectbox(
            "Выберите сделку для графической инспекции на графике:",
            range(len(options)),
            format_func=lambda i: options[i]
        )

        trade = df_trades.iloc[selected_trade_idx]
        t_symbol = trade["symbol"]
        t_entry_dt = pd.to_datetime(trade["entry_time"]).tz_localize(None)
        t_exit_dt = pd.to_datetime(trade["exit_time"]).tz_localize(None)
        
        # Load OHLCV for chart around trade dates
        df_ohlc = load_daily_ohlcv(storage, t_symbol)
        if df_ohlc is not None and len(df_ohlc) > 10:
            df_ohlc["dt"] = pd.to_datetime(df_ohlc["timestamp"]).dt.tz_localize(None)
            
            # Filter +/- 25 days around trade
            start_window = t_entry_dt - pd.Timedelta(days=20)
            end_window = t_exit_dt + pd.Timedelta(days=15)
            df_slice = df_ohlc[(df_ohlc["dt"] >= start_window) & (df_ohlc["dt"] <= end_window)]
            
            fig_cand = go.Figure()
            fig_cand.add_trace(go.Candlestick(
                x=df_slice["dt"],
                open=df_slice["open"],
                high=df_slice["high"],
                low=df_slice["low"],
                close=df_slice["close"],
                name=t_symbol
            ))
            
            # Mark Entry
            fig_cand.add_trace(go.Scatter(
                x=[t_entry_dt],
                y=[trade["entry_price"]],
                mode="markers+text",
                marker=dict(symbol="triangle-down", size=14, color="#34d399"),
                text=[f"ВХОД SHORT (${trade['entry_price']:.4f})"],
                textposition="top center",
                name="Вход (Short)"
            ))

            # Mark Initial SL
            fig_cand.add_trace(go.Scatter(
                x=[t_entry_dt, t_entry_dt + pd.Timedelta(days=3)],
                y=[trade["initial_sl"], trade["initial_sl"]],
                mode="lines",
                line=dict(color="#ef4444", width=2, dash="dot"),
                name=f"Начальный SL (${trade['initial_sl']:.4f})"
            ))

            # Mark Exit
            fig_cand.add_trace(go.Scatter(
                x=[t_exit_dt],
                y=[trade["exit_price"]],
                mode="markers+text",
                marker=dict(symbol="x", size=13, color="#ef4444" if trade["pnl_usdt"] < 0 else "#34d399"),
                text=[f"ВЫХОД ({trade['exit_reason']})"],
                textposition="bottom center",
                name="Выход"
            ))
            
            fig_cand.update_layout(
                template="plotly_dark",
                title=f"Графика сделки по {t_symbol} (PnL: ${trade['pnl_usdt']:+.2f} / {trade['return_pct']*100:.2f}%)",
                yaxis_title="Цена ($)",
                xaxis_title="Дата",
                height=450,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig_cand, use_container_width=True)
            
            # Explain mathematically why this trade entered
            st.markdown(f"""
            <div style="background-color: #1e293b; padding: 16px; border-radius: 8px; border-left: 4px solid {'#34d399' if trade['pnl_usdt']>0 else '#ef4444'};">
                <h4 style="margin:0 0 8px 0; color:#fff;">💡 Математический аутдит входа и выхода:</h4>
                <ul style="margin:0; color:#cbd5e1; font-size:14px;">
                    <li><b>Почему был сигнал входа:</b> В момент входа Z-Score True Range составил <b>{trade.get('entry_tr_zscore', 0):.2f}</b> (что выше порогового условия > 2.5, фиксируя резкий скачок волатильности). Показатель положения закрытия CLV был равен <b>{trade.get('entry_clv', 0):.3f}</b> (свидетельствуя о закрытии у самого дна свечи).</li>
                    <li><b>Управление риском:</b> Начальный стоп-лосс был размещён на отметке <b>${trade.get('initial_sl', 0):.4f}</b> ($1.5 \times ATR$).</li>
                    <li><b>Почему произошел выход:</b> Сделка удерживалась <b>{trade.get('bars_held', 0)} дней</b> и закрылась по причине <b>{trade['exit_reason']}</b> на цене <b>${trade['exit_price']:.4f}</b>.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
