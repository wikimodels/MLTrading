import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="ML Trading Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for better aesthetics (Light Theme)
st.markdown("""
<style>
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #f0f0f0;
    }
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    try:
        df = pd.read_csv("backtest/trades_history.csv")
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        df['trade_pnl'] = df['trade_pnl'].astype(float) if 'trade_pnl' in df.columns else df['net_pnl_usdt'].astype(float)
        df['net_pnl_usdt'] = df['net_pnl_usdt'].astype(float)
        df['is_win'] = df['trade_pnl'] > 0
        df = df.sort_values("exit_time").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

st.title("🚀 ML Trading Dashboard (Strict $10 Sizing)")

if df.empty:
    st.warning("No trades history found. Run the backtest first.")
    st.stop()

tab_general, tab_patterns = st.tabs(["📊 General Performance", "🧠 Pattern Analytics"])

with tab_general:
    # --- OVERALL STATS ---
    total_trades = len(df)
    total_wins = len(df[df['trade_pnl'] > 0])
    win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    total_pnl = df['net_pnl_usdt'].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", f"{total_trades:,}")
    col2.metric("Total Wins", f"{total_wins:,}")
    col3.metric("Win Rate", f"{win_rate:.2f}%")
    col4.metric("Net PnL (USDT)", f"${total_pnl:,.2f}")

    # --- OVERALL EQUITY CURVE (FULL WIDTH) ---
    df['Cumulative_PnL'] = df['net_pnl_usdt'].cumsum()
    fig_equity = px.line(df, x='exit_time', y='Cumulative_PnL', title='Overall Equity Curve',
                         labels={'exit_time': 'Date', 'Cumulative_PnL': 'Cumulative Profit (USDT)'})
    fig_equity.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=500, margin=dict(l=0, r=0, t=40, b=0))
    fig_equity.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
    fig_equity.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
    st.plotly_chart(fig_equity, use_container_width=True)

    # --- CONCURRENCY ANALYSIS ---
    st.markdown("---")
    st.header("📊 Trade Concurrency (Simultaneous Entries)")

    concurrency_df = df.groupby('entry_time').agg(
        total_trades=('symbol', 'count'),
        wins=('is_win', 'sum')
    ).reset_index()
    concurrency_df['losses'] = concurrency_df['total_trades'] - concurrency_df['wins']

    melted_df = pd.melt(concurrency_df, id_vars=['entry_time'], value_vars=['wins', 'losses'], 
                        var_name='Outcome', value_name='Count')

    fig_concurrency = px.bar(melted_df, x='entry_time', y='Count', color='Outcome',
                             title='Trades Opened per Time (Candle)',
                             color_discrete_map={'wins': '#2ca02c', 'losses': '#d62728'},
                             labels={'entry_time': 'Entry Time', 'Count': 'Number of Trades'})
    fig_concurrency.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400, margin=dict(l=0, r=0, t=40, b=0), barmode='stack')
    fig_concurrency.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
    fig_concurrency.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
    st.plotly_chart(fig_concurrency, use_container_width=True)

    st.markdown("---")
    st.header("🪙 Coin Specific Analysis")

    symbols = sorted(df['symbol'].unique())

    def render_coin_card(symbol, container):
        coin_df = df[df['symbol'] == symbol].copy()
        c_trades = len(coin_df)
        c_wins = len(coin_df[coin_df['trade_pnl'] > 0])
        c_wr = (c_wins / c_trades) * 100 if c_trades > 0 else 0
        c_pnl = coin_df['net_pnl_usdt'].sum()
        
        with container:
            st.subheader(f"{symbol}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Trades", c_trades)
            m2.metric("Win Rate", f"{c_wr:.1f}%")
            m3.metric("PnL", f"${c_pnl:.2f}")
            
            coin_df['Cumulative_PnL'] = coin_df['net_pnl_usdt'].cumsum()
            fig_coin = px.line(coin_df, x='exit_time', y='Cumulative_PnL', 
                               labels={'exit_time': '', 'Cumulative_PnL': 'PnL'},
                               color_discrete_sequence=['#ff7f0e' if c_pnl > 0 else '#d62728'])
            fig_coin.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(l=0, r=0, t=10, b=0))
            fig_coin.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
            fig_coin.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
            st.plotly_chart(fig_coin, use_container_width=True, key=f"chart_{symbol}")

    for i in range(0, len(symbols), 2):
        cols = st.columns(2)
        with cols[0]:
            with st.container():
                render_coin_card(symbols[i], cols[0])
        if i + 1 < len(symbols):
            with cols[1]:
                with st.container():
                    render_coin_card(symbols[i+1], cols[1])


with tab_patterns:
    st.header("🔍 Deep Pattern Analytics")
    
    # 1. Hour of Day
    st.subheader("1. Time of Day Pattern (UTC)")
    df['entry_hour'] = df['entry_time'].dt.hour
    hourly_stats = df.groupby('entry_hour').agg(
        trades=('symbol', 'count'),
        win_rate=('is_win', 'mean')
    ).reset_index()
    hourly_stats['win_rate'] = hourly_stats['win_rate'] * 100
    
    fig_hour = go.Figure()
    fig_hour.add_trace(go.Bar(x=hourly_stats['entry_hour'], y=hourly_stats['trades'], name='Trades Volume', opacity=0.3, yaxis='y1', marker_color='blue'))
    fig_hour.add_trace(go.Scatter(x=hourly_stats['entry_hour'], y=hourly_stats['win_rate'], name='Win Rate %', mode='lines+markers', yaxis='y2', line=dict(color='green', width=3)))
    fig_hour.update_layout(
        xaxis=dict(title='Hour of Day (UTC)', tickmode='linear', tick0=0, dtick=1),
        yaxis=dict(title='Volume', side='left', showgrid=False),
        yaxis2=dict(title='Win Rate %', side='right', overlaying='y', range=[0, 100], showgrid=True, gridcolor='#e5e7eb'),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400, margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_hour, use_container_width=True)

    # 2. Day of Week
    st.markdown("---")
    st.subheader("2. Day of Week Pattern")
    df['day_of_week'] = df['entry_time'].dt.day_name()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_stats = df.groupby('day_of_week').agg(
        trades=('symbol', 'count'),
        win_rate=('is_win', 'mean')
    ).reindex()
    # Fix reindexing for days of week
    dow_stats = df.groupby('day_of_week').agg(trades=('symbol', 'count'), win_rate=('is_win', 'mean')).reset_index()
    dow_stats['day_of_week'] = pd.Categorical(dow_stats['day_of_week'], categories=days_order, ordered=True)
    dow_stats = dow_stats.sort_values('day_of_week')
    dow_stats['win_rate'] = dow_stats['win_rate'] * 100
    
    fig_dow = go.Figure()
    fig_dow.add_trace(go.Bar(x=dow_stats['day_of_week'], y=dow_stats['trades'], name='Trades Volume', opacity=0.3, yaxis='y1', marker_color='purple'))
    fig_dow.add_trace(go.Scatter(x=dow_stats['day_of_week'], y=dow_stats['win_rate'], name='Win Rate %', mode='lines+markers', yaxis='y2', line=dict(color='green', width=3)))
    fig_dow.update_layout(
        yaxis=dict(title='Volume', side='left', showgrid=False),
        yaxis2=dict(title='Win Rate %', side='right', overlaying='y', range=[0, 100], showgrid=True, gridcolor='#e5e7eb'),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400, margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_dow, use_container_width=True)

    # 3. Trade Duration
    st.markdown("---")
    st.subheader("3. Trade Duration Pattern")
    df['duration_hours'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600
    bins = [0, 4, 12, 24, 72, 1000]
    labels = ['< 4h', '4h-12h', '12h-24h', '1d-3d', '> 3d']
    # Add a fallback for categorical grouping to avoid issues
    df['duration_bin'] = pd.cut(df['duration_hours'], bins=bins, labels=labels, right=False)
    # Using observed=False explicitly to suppress future warnings, but using basic groupby
    dur_stats = df.groupby('duration_bin', observed=False).agg(
        trades=('symbol', 'count'),
        win_rate=('is_win', 'mean')
    ).reset_index()
    dur_stats['win_rate'] = dur_stats['win_rate'] * 100
    
    fig_dur = go.Figure()
    fig_dur.add_trace(go.Bar(x=dur_stats['duration_bin'], y=dur_stats['trades'], name='Trades Volume', opacity=0.3, yaxis='y1', marker_color='orange'))
    fig_dur.add_trace(go.Scatter(x=dur_stats['duration_bin'], y=dur_stats['win_rate'], name='Win Rate %', mode='lines+markers', yaxis='y2', line=dict(color='green', width=3)))
    fig_dur.update_layout(
        yaxis=dict(title='Volume', side='left', showgrid=False),
        yaxis2=dict(title='Win Rate %', side='right', overlaying='y', range=[0, 100], showgrid=True, gridcolor='#e5e7eb'),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400, margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_dur, use_container_width=True)
    
    # 4. Concurrency vs Win Rate
    st.markdown("---")
    st.subheader("4. Concurrency (Simultaneous Signals) vs Win Rate")
    # concurrency_df was calculated in general tab, but let's aggregate it by total_trades
    if 'concurrency_df' in locals() or 'concurrency_df' in globals():
        concurrency_stats = concurrency_df.groupby('total_trades').agg(
            occurrences=('entry_time', 'count'),
            total_wins=('wins', 'sum'),
            total_losses=('losses', 'sum')
        ).reset_index()
        concurrency_stats['win_rate'] = (concurrency_stats['total_wins'] / (concurrency_stats['total_wins'] + concurrency_stats['total_losses'])) * 100
        
        fig_conc = go.Figure()
        fig_conc.add_trace(go.Bar(x=concurrency_stats['total_trades'], y=concurrency_stats['occurrences'], name='Frequency', opacity=0.3, yaxis='y1', marker_color='red'))
        fig_conc.add_trace(go.Scatter(x=concurrency_stats['total_trades'], y=concurrency_stats['win_rate'], name='Win Rate %', mode='lines+markers', yaxis='y2', line=dict(color='green', width=3)))
        fig_conc.update_layout(
            xaxis=dict(title='Number of Simultaneous Trades (Concurrency)'),
            yaxis=dict(title='Frequency', side='left', showgrid=False),
            yaxis2=dict(title='Win Rate %', side='right', overlaying='y', range=[0, 100], showgrid=True, gridcolor='#e5e7eb'),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400, margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_conc, use_container_width=True)
