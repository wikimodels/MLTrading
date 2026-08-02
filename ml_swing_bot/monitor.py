"""
ML Trading — Live Training & Backtest Monitor
Запуск: poetry run streamlit run monitor.py
"""

from __future__ import annotations

import re
import time
import os
import sys
from datetime import datetime
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_bot_dir = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_bot_dir) not in sys.path:
    sys.path.insert(0, str(_bot_dir))
os.chdir(_bot_dir)
from shared.config_loader import set_active_bot
set_active_bot("ml_swing_bot")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yaml

st.set_page_config(
    page_title="ML Trading Monitor",
    page_icon="favicon.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session State for UI Navigation ──
if "show_results" not in st.session_state:
    st.session_state.show_results = False

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Roboto+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Roboto', sans-serif !important; }
.stApp { background-color: #F4F5F7 !important; color: #1D1B20 !important; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }
#MainMenu, footer, header { display: none !important; }

/* Material Typography */
h1, h2, h3, h4, h5, h6, p, div { font-family: 'Roboto', sans-serif; }
.main-title { font-size: 28px !important; font-weight: 400 !important; color: #1D1B20 !important; margin-bottom: 4px !important; letter-spacing: 0px !important; }
.sub-title { font-size: 14px !important; color: #49454F !important; margin-bottom: 24px !important; letter-spacing: 0.25px !important; }
.section-title { font-size: 16px !important; font-weight: 500 !important; color: #6750A4 !important; margin-bottom: 16px !important; padding-bottom: 8px !important; border-bottom: 2px solid #EADDFF !important; letter-spacing: 0.15px !important; text-transform: none !important; }

/* Material 3 Buttons */
button[kind="primary"] {
    background-color: #6750A4 !important;
    color: #FFFFFF !important;
    border-radius: 100px !important;
    border: none !important;
    padding: 10px 24px !important;
    font-weight: 500 !important;
    letter-spacing: 0.1px !important;
    text-transform: none !important;
    box-shadow: 0px 1px 3px 1px rgba(0,0,0,0.15), 0px 1px 2px 0px rgba(0,0,0,0.3) !important;
    transition: all 0.2s ease-in-out !important;
}
button[kind="primary"]:hover {
    background-color: #55428F !important;
    box-shadow: 0px 2px 6px 2px rgba(0,0,0,0.15), 0px 1px 2px 0px rgba(0,0,0,0.3) !important;
}

button[kind="secondary"] {
    background-color: transparent !important;
    color: #6750A4 !important;
    border: 1px solid #79747E !important;
    border-radius: 100px !important;
    padding: 10px 24px !important;
    font-weight: 500 !important;
    letter-spacing: 0.1px !important;
    transition: all 0.2s ease-in-out !important;
}
button[kind="secondary"]:hover {
    background-color: rgba(103, 80, 164, 0.08) !important;
    border-color: #6750A4 !important;
}

/* Elevated Cards (Metrics & Log) */
[data-testid="stTabs"] button {
    font-family: 'Roboto', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #49454F !important;
    border-bottom: 2px solid transparent !important;
    padding-bottom: 8px !important;
    border-radius: 4px 4px 0 0 !important;
    background-color: transparent !important;
    transition: all 0.2s ease-in-out !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #6750A4 !important;
    border-bottom: 2px solid #6750A4 !important;
}
[data-testid="stTabs"] button:hover {
    color: #1D1B20 !important;
    background-color: rgba(103, 80, 164, 0.08) !important;
}

[data-testid="stMetric"] {
    background-color: #FEF7FF !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 16px !important;
    box-shadow: 0px 1px 2px 0px rgba(0,0,0,0.3), 0px 1px 3px 1px rgba(0,0,0,0.15) !important;
    transition: box-shadow 0.2s ease-in-out !important;
}
[data-testid="stMetricLabel"] { color: #49454F !important; font-size: 14px !important; font-weight: 500 !important; letter-spacing: 0.1px !important; text-transform: none !important; }
[data-testid="stMetricValue"] { color: #1D1B20 !important; font-size: 24px !important; font-weight: 400 !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

.log-container {
    background-color: #FEF7FF !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px !important;
    font-family: 'Roboto Mono', monospace !important;
    font-size: 12px !important;
    line-height: 1.6 !important;
    max-height: 350px !important;
    overflow-y: auto !important;
    white-space: pre-wrap !important;
    word-break: break-all !important;
    box-shadow: 0px 1px 2px 0px rgba(0,0,0,0.3), 0px 1px 3px 1px rgba(0,0,0,0.15) !important;
}

/* Material Progress Bars */
.stProgress > div > div > div { background-color: #6750A4 !important; border-radius: 100px !important; }
.stProgress > div > div { background-color: #EADDFF !important; border-radius: 100px !important; height: 8px !important; }

/* Badges and Chips (M3 style) */
.badge { display: inline-flex !important; align-items: center !important; padding: 4px 12px !important; border-radius: 8px !important; font-size: 12px !important; font-weight: 500 !important; letter-spacing: 0.1px !important; border: none !important; }
.badge-running  { background-color: #E8DEF8 !important; color: #1D192B !important; }
.badge-done     { background-color: #C3E88D !important; color: #1D1B20 !important; }
.badge-idle     { background-color: #E0E0E0 !important; color: #1D1B20 !important; }
.badge-error    { background-color: #F9DEDC !important; color: #410E0B !important; }

.chip { display: inline-flex !important; align-items: center !important; background-color: #E8DEF8 !important; border: none !important; border-radius: 8px !important; padding: 4px 12px !important; font-size: 13px !important; font-weight: 500 !important; color: #1D192B !important; margin: 4px 4px 4px 0 !important; }
.chip-key { color: #49454F !important; font-weight: 400 !important; margin-right: 6px !important; }

.divider { border: none !important; border-top: 1px solid #CAC4D0 !important; margin: 24px 0 !important; }

.wf-item { display: inline-flex; align-items: center; gap: 4px; padding: 6px 12px; border-radius: 8px; margin: 0 6px 6px 0; font-size: 12px; font-weight: 500; font-family: 'Roboto Mono', monospace; }
.wf-done  { background-color: #E8DEF8 !important; color: #1D192B !important; }
.wf-curr  { background-color: #6750A4 !important; color: #FFFFFF !important; }
.wf-wait  { background-color: #E0E0E0 !important; color: #49454F !important; }

.err-item { background-color: #F9DEDC !important; border-left: 4px solid #B3261E !important; border-radius: 0 8px 8px 0 !important; padding: 8px 12px !important; margin-bottom: 4px !important; font-size: 12px !important; font-family: 'Roboto Mono', monospace !important; color: #410E0B !important; word-break: break-all !important; }
.warn-item { background-color: #FFEFD4 !important; border-left: 4px solid #D97706 !important; border-radius: 0 8px 8px 0 !important; padding: 8px 12px !important; margin-bottom: 4px !important; font-size: 12px !important; font-family: 'Roboto Mono', monospace !important; color: #78350F !important; word-break: break-all !important; }

/* Scroll to top FAB */
.scroll-to-top {
    position: fixed;
    bottom: 24px;
    right: 24px;
    background-color: #6750A4;
    color: white !important;
    width: 48px;
    height: 48px;
    border-radius: 50% !important;
    display: flex;
    justify-content: center;
    align-items: center;
    text-decoration: none !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    z-index: 999999;
    transition: background-color 0.2s, box-shadow 0.2s, transform 0.1s ease-out;
    outline: none !important;
    -webkit-tap-highlight-color: transparent !important;
}
.scroll-to-top:hover {
    background-color: #55428F;
    box-shadow: 0 6px 12px rgba(0,0,0,0.3) !important;
}
.scroll-to-top:active {
    transform: scale(0.92);
}
.scroll-to-top svg {
    width: 20px;
    height: 20px;
    stroke: currentColor;
    stroke-width: 2.5;
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
}
</style>
<a class="scroll-to-top" title="Наверх" style="cursor: pointer;">
    <svg viewBox="0 0 24 24"><line x1="12" y1="19" x2="12" y2="5"></line><polyline points="5 12 12 5 19 12"></polyline></svg>
</a>
""", unsafe_allow_html=True)

# ── Custom JS Scroll Animation (EaseOutExpo) — DEBUG MODE ──
components.html("""
<script>
    const parentWin = window.parent;
    const parentDoc = parentWin.document;

    console.log("[FAB Debug] ✅ Скрипт загружен. iframe origin:", window.location.href);
    console.log("[FAB Debug] parentWin доступен:", !!parentWin);
    console.log("[FAB Debug] parentDoc доступен:", !!parentDoc);

    // --- DEBUG: проверяем видимость кнопки в parentDoc ---
    setTimeout(function() {
        const btn = parentDoc.querySelector('.scroll-to-top');
        console.log("[FAB Debug] Кнопка .scroll-to-top найдена в parentDoc:", !!btn);
        if (btn) {
            const rect = btn.getBoundingClientRect();
            console.log("[FAB Debug] Кнопка rect:", JSON.stringify({top: rect.top, left: rect.left, width: rect.width, height: rect.height}));
            const style = parentWin.getComputedStyle(btn);
            console.log("[FAB Debug] Кнопка display:", style.display, "| visibility:", style.visibility, "| opacity:", style.opacity, "| z-index:", style.zIndex, "| pointer-events:", style.pointerEvents);
        } else {
            console.warn("[FAB Debug] ❌ Кнопка .scroll-to-top НЕ НАЙДЕНА в parentDoc!");
            const allLinks = parentDoc.querySelectorAll('a');
            console.log("[FAB Debug] Все <a> теги в parentDoc:", allLinks.length);
            allLinks.forEach(function(a, i) {
                console.log("  [FAB Debug] <a> #" + i + ":", a.className, "| href:", a.href, "| text:", a.textContent.trim().substring(0,30));
            });
        }
    }, 800);

    function easeOutExpo(t, b, c, d) {
        return t === d ? b + c : c * (-Math.pow(2, -10 * t / d) + 1) + b;
    }

    function doScroll() {
        console.log("[FAB Debug] doScroll() вызван!");

        // Перебираем все возможные скроллируемые контейнеры
        const selectors = [
            '[data-testid="stAppViewContainer"]',
            '[data-testid="block-container"]',
            '.main',
            'section.main',
            '[data-testid="stMain"]',
        ];
        let container = null;
        for (const sel of selectors) {
            const el = parentDoc.querySelector(sel);
            if (el) {
                console.log("[FAB Debug] Найден контейнер:", sel, "scrollTop:", el.scrollTop, "scrollHeight:", el.scrollHeight);
                if (el.scrollTop > 0) { container = el; break; }
            } else {
                console.log("[FAB Debug] НЕ найден:", sel);
            }
        }

        const startPosition = container ? container.scrollTop : parentWin.scrollY;
        console.log("[FAB Debug] Скроллируемый контейнер:", container ? container.getAttribute('data-testid') || container.className : "window");
        console.log("[FAB Debug] startPosition:", startPosition, "| window.scrollY:", parentWin.scrollY);

        if (startPosition === 0 && parentWin.scrollY === 0) {
            console.warn("[FAB Debug] ⚠️ Уже наверху (scrollTop=0, scrollY=0). Скролл не нужен.");
            return;
        }

        const distance = -startPosition;
        let startTime = null;
        const duration = 900;

        function animation(currentTime) {
            if (startTime === null) startTime = currentTime;
            const timeElapsed = currentTime - startTime;
            const pos = easeOutExpo(timeElapsed, startPosition, distance, duration);
            if (container) container.scrollTop = pos;
            parentWin.scrollTo(0, pos);
            if (timeElapsed < duration) {
                requestAnimationFrame(animation);
            } else {
                console.log("[FAB Debug] ✅ Анимация завершена.");
                if (container) container.scrollTop = 0;
                parentWin.scrollTo(0, 0);
            }
        }
        requestAnimationFrame(animation);
    }

    // Способ 1: глобальный клик в parentDoc (основной)
    if (!parentWin._fabScrollBound) {
        parentWin._fabScrollBound = true;
        console.log("[FAB Debug] Привязываем обработчик click к parentDoc (способ 1)...");

        parentDoc.addEventListener('click', function(e) {
            let node = e.target;
            console.log("[FAB Debug] Клик в parentDoc! target:", node.tagName, "| class:", node.className);
            while (node && node !== parentDoc) {
                if (node.classList && node.classList.contains('scroll-to-top')) {
                    console.log("[FAB Debug] 🎯 КНОПКА НАЖАТА (способ 1 — parentDoc listener)!");
                    e.preventDefault();
                    doScroll();
                    return;
                }
                node = node.parentNode;
            }
        });
        console.log("[FAB Debug] ✅ Обработчик способа 1 привязан.");
    } else {
        console.log("[FAB Debug] ⚠️ Обработчик способа 1 уже был привязан ранее.");
    }

    // Способ 2: прямая привязка onclick к кнопке (запасной)
    setTimeout(function() {
        const btn = parentDoc.querySelector('.scroll-to-top');
        if (btn) {
            if (!btn._directBound) {
                btn._directBound = true;
                btn.addEventListener('click', function(e) {
                    console.log("[FAB Debug] 🎯 КНОПКА НАЖАТА (способ 2 — прямой onclick на элементе)!");
                    e.preventDefault();
                    doScroll();
                });
                console.log("[FAB Debug] ✅ Прямой onclick привязан (способ 2).");
            }
        } else {
            console.warn("[FAB Debug] ❌ Способ 2: кнопка не найдена для прямой привязки.");
        }
    }, 1000);
</script>
""", height=0)

LOG_PATH      = Path("logs/mltrading.log")
SETTINGS_PATH = Path("config/settings.yaml")

RE_WINDOW    = re.compile(r"Walk-forward window(?: \[(\d+)/(\d+)\])?.*?cutoff=(\S+)")
RE_TRAINING  = re.compile(r"Training window:\s+(\S+)\s+->\s+(\S+)\s+\|\s+rows=(\d+)\s+\|\s+pos=(\d+)\s+\|\s+neg=(\d+)")
RE_SYMBOL    = re.compile(r"\[(\d+)/(\d+)\].*?(?:Engineering|Processing|feature).*?for\s+(\S+)", re.IGNORECASE)
RE_ERROR     = re.compile(r"\|\s+ERROR\s+\|(.+)")
RE_WARNING   = re.compile(r"\|\s+WARNING\s+\|(.+)")
RE_COLLECT   = re.compile(r"\[(\d+)/(\d+)\]\s+Collecting\s+(\S+)")
RE_COMMAND   = re.compile(r"Command:\s+(\w+)")
RE_SIM_START = re.compile(r"Starting simulation")
RE_DATASET   = re.compile(r"Dataset ready:\s+(\d+)\s+rows")
RE_MODEL_SAVED = re.compile(r"Model saved to\s+(\S+)")
RE_DONE      = re.compile(r"Equity curve saved|trades.*backtest", re.IGNORECASE)
RE_GROUP     = re.compile(r"SYMBOL GROUP:\s+(\w+).*?(\d+)\s+монет")

@st.cache_data(ttl=2)
def load_settings():
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def read_log_tail(path, n):
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.readlines()[-n:]
    except Exception:
        return []

def parse_log(lines):
    start_idx = 0
    for i in range(len(lines) - 1, -1, -1):
        if RE_COMMAND.search(lines[i]):
            start_idx = i
            break
    lines = lines[start_idx:]

    r = {
        "command": None, "windows": [], "current_window": None,
        "training_info": None, "symbols_done": 0, "symbols_total": 0,
        "current_symbol": None, "errors": [], "warnings": [],
        "dataset_rows": None, "model_saved": None, "is_done": False,
        "phase": "idle", "collect_done": 0, "collect_total": 0,
        "current_collect": None, "symbol_group": None, "symbol_count": None,
        "wf_done": 0, "wf_total": 0,
    }
    for line in lines:
        m = RE_COMMAND.search(line)
        if m: r["command"] = m.group(1)

        m = RE_WINDOW.search(line)
        if m:
            if m.group(1):
                r["wf_done"] = int(m.group(1))
                r["wf_total"] = int(m.group(2))
                cutoff = m.group(3)
            else:
                cutoff = m.group(1) if m.groups() == 1 else m.groups()[-1]
                
            r["windows"].append(cutoff)
            r["current_window"] = cutoff
            r["phase"] = "training"

        m = RE_TRAINING.search(line)
        if m:
            r["training_info"] = {"from": m.group(1), "to": m.group(2),
                                  "rows": int(m.group(3)), "pos": int(m.group(4)), "neg": int(m.group(5))}

        m = RE_SYMBOL.search(line)
        if m:
            r["symbols_done"] = int(m.group(1))
            r["symbols_total"] = int(m.group(2))
            r["current_symbol"] = m.group(3)
            if r["phase"] == "idle": r["phase"] = "engineering"

        m = RE_COLLECT.search(line)
        if m:
            r["collect_done"] = int(m.group(1))
            r["collect_total"] = int(m.group(2))
            r["current_collect"] = m.group(3)
            r["phase"] = "collecting"

        m = RE_GROUP.search(line)
        if m: r["symbol_group"] = m.group(1); r["symbol_count"] = int(m.group(2))

        if RE_SIM_START.search(line): r["phase"] = "simulating"

        m = RE_DATASET.search(line)
        if m: r["dataset_rows"] = int(m.group(1))

        m = RE_MODEL_SAVED.search(line)
        if m: r["model_saved"] = m.group(1)

        if RE_DONE.search(line): r["is_done"] = True; r["phase"] = "done"

        m = RE_ERROR.search(line)
        if m:
            txt = m.group(1).strip()
            if txt and txt not in r["errors"]: r["errors"].append(txt)

        m = RE_WARNING.search(line)
        if m:
            txt = m.group(1).strip()
            skip = ["No further splits", "LGBMDeprecation", "eval_set"]
            if txt and not any(p in txt for p in skip) and txt not in r["warnings"]:
                r["warnings"].append(txt)

    return r

PHASE_LABELS = {"idle": "Ожидание", "collecting": "Сбор данных", "engineering": "Инжиниринг фич",
                "training": "Обучение", "simulating": "Симуляция", "done": "Завершено"}
PHASE_CSS    = {"idle": "badge-idle", "collecting": "badge-running", "engineering": "badge-running",
                "training": "badge-running", "simulating": "badge-running", "done": "badge-done"}

def phase_badge(phase):
    return f'<span class="badge {PHASE_CSS.get(phase,"badge-idle")}">{PHASE_LABELS.get(phase,phase)}</span>'

def colorize(line):
    s = line.rstrip().replace("<","&lt;").replace(">","&gt;")
    if "| ERROR"    in s: return f'<span style="color:#dc2626; font-weight:600;">{s}</span>'
    if "| WARNING"  in s: return f'<span style="color:#d97706;">{s}</span>'
    if "| DEBUG"    in s: return f'<span style="color:#94a3b8;">{s}</span>'
    if "Walk-forward" in s or "Training window" in s: return f'<span style="color:#7c3aed;">{s}</span>'
    if "saved" in s.lower() or "complete" in s.lower() or "done" in s.lower():
        return f'<span style="color:#16a34a; font-weight:600;">{s}</span>'
    if "| INFO" in s: return f'<span style="color:#475569;">{s}</span>'
    return f'<span style="color:#334155;">{s}</span>'

# ══════════════════════════════════════════════════════════════════
# ВСТРОЕННЫЙ ДАШБОРД — функции отрисовки результатов бэктеста
# ══════════════════════════════════════════════════════════════════

def _load_trades_data() -> pd.DataFrame:
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
        st.error(f"Ошибка загрузки trades_history.csv: {e}")
        return pd.DataFrame()


def _render_results(df: pd.DataFrame):
    if df.empty:
        st.warning("Нет данных о сделках. Запустите бэктест.")
        return

    total_trades = len(df)
    total_wins   = len(df[df['trade_pnl'] > 0])
    win_rate     = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    total_pnl    = df['net_pnl_usdt'].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades",   f"{total_trades:,}")
    c2.metric("Total Wins",     f"{total_wins:,}")
    c3.metric("Win Rate",       f"{win_rate:.2f}%")
    c4.metric("Net PnL (USDT)", f"${total_pnl:,.2f}")

    df['Cumulative_PnL'] = df['net_pnl_usdt'].cumsum()
    fig_eq = px.line(df, x='exit_time', y='Cumulative_PnL', title='Overall Equity Curve',
                     labels={'exit_time': 'Date', 'Cumulative_PnL': 'Cumulative Profit (USDT)'})
    fig_eq.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                         height=500, margin=dict(l=0, r=0, t=40, b=0))
    fig_eq.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
    fig_eq.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
    st.plotly_chart(fig_eq, use_container_width=True)

    st.markdown("---")
    st.header("📊 Trade Concurrency (Simultaneous Entries)")
    conc_df = df.groupby('entry_time').agg(
        total_trades=('symbol', 'count'), wins=('is_win', 'sum')
    ).reset_index()
    conc_df['losses'] = conc_df['total_trades'] - conc_df['wins']
    melted = pd.melt(conc_df, id_vars=['entry_time'], value_vars=['wins', 'losses'],
                     var_name='Outcome', value_name='Count')
    fig_conc = px.bar(melted, x='entry_time', y='Count', color='Outcome',
                      title='Trades Opened per Time (Candle)',
                      color_discrete_map={'wins': '#2ca02c', 'losses': '#d62728'},
                      labels={'entry_time': 'Entry Time', 'Count': 'Number of Trades'})
    fig_conc.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           height=400, margin=dict(l=0, r=0, t=40, b=0), barmode='stack')
    fig_conc.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
    fig_conc.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
    st.plotly_chart(fig_conc, use_container_width=True)

    st.markdown("---")
    st.header("🪙 Coin Specific Analysis")
    symbols = sorted(df['symbol'].unique())

    def _coin_card(symbol, container):
        coin_df = df[df['symbol'] == symbol].copy()
        c_trades = len(coin_df)
        c_wins   = len(coin_df[coin_df['trade_pnl'] > 0])
        c_wr     = (c_wins / c_trades) * 100 if c_trades > 0 else 0
        c_pnl    = coin_df['net_pnl_usdt'].sum()
        with container:
            st.subheader(f"{symbol}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Trades", c_trades)
            m2.metric("Win Rate", f"{c_wr:.1f}%")
            m3.metric("PnL", f"${c_pnl:.2f}")
            coin_df['Cumulative_PnL'] = coin_df['net_pnl_usdt'].cumsum()
            fig_c = px.line(coin_df, x='exit_time', y='Cumulative_PnL',
                            labels={'exit_time': '', 'Cumulative_PnL': 'PnL'},
                            color_discrete_sequence=['#ff7f0e' if c_pnl > 0 else '#d62728'])
            fig_c.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                height=300, margin=dict(l=0, r=0, t=10, b=0))
            fig_c.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
            fig_c.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
            st.plotly_chart(fig_c, use_container_width=True, key=f"chart_{symbol}")

    for i in range(0, len(symbols), 2):
        cols = st.columns(2)
        _coin_card(symbols[i], cols[0])
        if i + 1 < len(symbols):
            _coin_card(symbols[i + 1], cols[1])


def _render_patterns(df: pd.DataFrame):
    if df.empty:
        st.warning("Нет данных о сделках. Запустите бэктест.")
        return

    st.header("🔍 Deep Pattern Analytics")

    st.subheader("1. Time of Day Pattern (UTC)")
    df['entry_hour'] = df['entry_time'].dt.hour
    hourly = df.groupby('entry_hour').agg(trades=('symbol', 'count'), win_rate=('is_win', 'mean')).reset_index()
    hourly['win_rate'] *= 100
    fig_h = go.Figure()
    fig_h.add_trace(go.Bar(x=hourly['entry_hour'], y=hourly['trades'], name='Volume', opacity=0.3, yaxis='y1', marker_color='blue'))
    fig_h.add_trace(go.Scatter(x=hourly['entry_hour'], y=hourly['win_rate'], name='Win Rate %', mode='lines+markers', yaxis='y2', line=dict(color='green', width=3)))
    fig_h.update_layout(xaxis=dict(title='Hour (UTC)', tickmode='linear', tick0=0, dtick=1),
                        yaxis=dict(title='Volume', side='left', showgrid=False),
                        yaxis2=dict(title='Win Rate %', side='right', overlaying='y', range=[0, 100], showgrid=True, gridcolor='#e5e7eb'),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("---")
    st.subheader("2. Day of Week Pattern")
    df['day_of_week'] = df['entry_time'].dt.day_name()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow = df.groupby('day_of_week').agg(trades=('symbol', 'count'), win_rate=('is_win', 'mean')).reset_index()
    dow['day_of_week'] = pd.Categorical(dow['day_of_week'], categories=days_order, ordered=True)
    dow = dow.sort_values('day_of_week')
    dow['win_rate'] *= 100
    fig_d = go.Figure()
    fig_d.add_trace(go.Bar(x=dow['day_of_week'], y=dow['trades'], name='Volume', opacity=0.3, yaxis='y1', marker_color='purple'))
    fig_d.add_trace(go.Scatter(x=dow['day_of_week'], y=dow['win_rate'], name='Win Rate %', mode='lines+markers', yaxis='y2', line=dict(color='green', width=3)))
    fig_d.update_layout(yaxis=dict(title='Volume', side='left', showgrid=False),
                        yaxis2=dict(title='Win Rate %', side='right', overlaying='y', range=[0, 100], showgrid=True, gridcolor='#e5e7eb'),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_d, use_container_width=True)

    st.markdown("---")
    st.subheader("3. Trade Duration Pattern")
    df['duration_hours'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600
    bins   = [0, 4, 12, 24, 72, 1000]
    labels = ['< 4h', '4h-12h', '12h-24h', '1d-3d', '> 3d']
    df['duration_bin'] = pd.cut(df['duration_hours'], bins=bins, labels=labels, right=False)
    dur = df.groupby('duration_bin', observed=False).agg(trades=('symbol', 'count'), win_rate=('is_win', 'mean')).reset_index()
    dur['win_rate'] *= 100
    fig_dur = go.Figure()
    fig_dur.add_trace(go.Bar(x=dur['duration_bin'], y=dur['trades'], name='Volume', opacity=0.3, yaxis='y1', marker_color='orange'))
    fig_dur.add_trace(go.Scatter(x=dur['duration_bin'], y=dur['win_rate'], name='Win Rate %', mode='lines+markers', yaxis='y2', line=dict(color='green', width=3)))
    fig_dur.update_layout(yaxis=dict(title='Volume', side='left', showgrid=False),
                          yaxis2=dict(title='Win Rate %', side='right', overlaying='y', range=[0, 100], showgrid=True, gridcolor='#e5e7eb'),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_dur, use_container_width=True)

    st.markdown("---")
    st.subheader("4. Concurrency vs Win Rate")
    conc_g = df.groupby('entry_time').agg(total_trades=('symbol', 'count'), wins=('is_win', 'sum')).reset_index()
    conc_g['losses'] = conc_g['total_trades'] - conc_g['wins']
    cs = conc_g.groupby('total_trades').agg(
        occurrences=('entry_time', 'count'), total_wins=('wins', 'sum'), total_losses=('losses', 'sum')
    ).reset_index()
    cs['win_rate'] = cs['total_wins'] / (cs['total_wins'] + cs['total_losses']) * 100
    fig_cs = go.Figure()
    fig_cs.add_trace(go.Bar(x=cs['total_trades'], y=cs['occurrences'], name='Frequency', opacity=0.3, yaxis='y1', marker_color='red'))
    fig_cs.add_trace(go.Scatter(x=cs['total_trades'], y=cs['win_rate'], name='Win Rate %', mode='lines+markers', yaxis='y2', line=dict(color='green', width=3)))
    fig_cs.update_layout(xaxis=dict(title='Simultaneous Trades'),
                         yaxis=dict(title='Frequency', side='left', showgrid=False),
                         yaxis2=dict(title='Win Rate %', side='right', overlaying='y', range=[0, 100], showgrid=True, gridcolor='#e5e7eb'),
                         plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_cs, use_container_width=True)


# ── FULL SCREEN DASHBOARD VIEW ──
if st.session_state.show_results:
    st.markdown('<div class="main-title"><svg xmlns="http://www.w3.org/2000/svg" height="32" viewBox="0 -960 960 960" width="32" fill="#6750A4" style="vertical-align: sub; margin-right: 8px;"><path d="M117.85-231.38 342.62-456.15 514-284.77 842.15-612.92l-51.38-51.39-276.77 276.77-171.38-171.38L66.46-282.77l51.39 51.39Z"/></svg>ML Trading Dashboard</div>', unsafe_allow_html=True)
    if st.button("⬅ Вернуться к живому монитору", type="secondary"):
        st.session_state.show_results = False
        st.rerun()
        
    st.markdown("---")
    df_res = _load_trades_data()
    if not df_res.empty:
        tab_res, tab_pat = st.tabs(["📊 General Performance", "🧠 Pattern Analytics"])
        with tab_res:
            _render_results(df_res)
        with tab_pat:
            _render_patterns(df_res)
    else:
        st.warning("Нет данных о сделках. Запустите бэктест.")
        
    st.stop() # Stop rendering the rest of the monitor

# ── Sidebar ──
with st.sidebar:
    st.markdown("### Настройки монитора")
    auto_refresh = st.toggle("Авто-обновление", value=True)
    refresh_sec  = st.slider("Интервал (сек)", 2, 30, 5)
    n_log_lines  = st.slider("Строк лога", 50, 500, 150, step=50)
    show_warns   = st.toggle("Показывать предупреждения", value=True)
    st.markdown("---")
    if st.button("Обновить сейчас", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ── Load ──
cfg       = load_settings()
all_lines = read_log_tail(LOG_PATH, 2000)
parsed    = parse_log(all_lines)
tail      = all_lines[-n_log_lines:]

# ── Header ──
col_title, col_btn = st.columns([3, 1])
with col_title:
    st.markdown('<div class="main-title"><svg xmlns="http://www.w3.org/2000/svg" height="32" viewBox="0 -960 960 960" width="32" fill="#6750A4" style="vertical-align: sub; margin-right: 8px;"><path d="M414.77-80 392-414.77H170.62L545.23-880l22.77 334.77H789.38L414.77-80Z"/></svg>ML Trading Monitor</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">Обновлено: {datetime.now().strftime("%H:%M:%S")} · {LOG_PATH}</div>', unsafe_allow_html=True)

with col_btn:
    if parsed["is_done"]:
        if st.button("📊 Посмотреть итоги", type="primary", use_container_width=True):
            st.session_state.show_results = True
            st.rerun()

# ── Settings ──
st.markdown('<div class="section-title">Активные настройки</div>', unsafe_allow_html=True)
universe_cfg  = cfg.get("symbol_universe", {})
active_group  = universe_cfg.get("active_group", "?")
exit_mode     = cfg.get("exit_mode", "?")
use_weekly    = cfg.get("features", {}).get("use_weekly_tf", False)
global_excl   = cfg.get("global_exclude_symbols", [])
exp_syms      = universe_cfg.get("groups", {}).get("experimental", {}).get("symbols", [])
effective     = [s for s in exp_syms if s not in set(global_excl)]
min_conf      = cfg.get("model", {}).get("min_confidence", "?")
capital       = cfg.get("backtest", {}).get("initial_capital", "?")

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Группа", active_group)
c2.metric("Exit mode", exit_mode)
c3.metric("Weekly TF", "ВКЛ" if use_weekly else "ВЫКЛ")
c4.metric("Монет", f"{len(effective)}")
c5.metric("Min conf", f"{min_conf:.2f}" if isinstance(min_conf,float) else str(min_conf))

exit_p = cfg.get("exit_params", {}).get(exit_mode, {})
chips  = [f'<span class="chip"><span class="chip-key">{k}:</span>{v}</span>' for k,v in exit_p.items()]
chips += [
    f'<span class="chip"><span class="chip-key">capital:</span>${capital}</span>',
    f'<span class="chip"><span class="chip-key">train-window:</span>{cfg.get("model",{}).get("train_window_days","?")}d</span>',
    f'<span class="chip"><span class="chip-key">retrain:</span>{cfg.get("model",{}).get("retrain_every_days","?")}d</span>',
    f'<span class="chip"><span class="chip-key">blacklist:</span>{len(global_excl)} монет</span>',
]
st.markdown(" ".join(chips), unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Process status ──
st.markdown('<div class="section-title">Текущий процесс</div>', unsafe_allow_html=True)
pA, pB = st.columns([1, 2])

with pA:
    st.markdown(f"**Команда:** `{parsed['command'] or '—'}` &nbsp;|&nbsp; **Фаза:** {phase_badge(parsed['phase'])}", unsafe_allow_html=True)
    if parsed["dataset_rows"]:
        st.markdown(f"**Общий датасет:** `{parsed['dataset_rows']:,}` строк")
    if parsed["training_info"] and not parsed["is_done"]:
        ti = parsed["training_info"]
        total = ti["pos"] + ti["neg"]
        st.markdown(f'<span style="color:#64748b;font-size:11px;">ℹ Последнее окно: {ti["from"]} → {ti["to"]} ({total:,} строк)</span>', unsafe_allow_html=True)

with pB:
    if parsed["symbols_total"] > 0:
        st.markdown(f"**Инжиниринг фич:** `{parsed['symbols_done']}/{parsed['symbols_total']}` · текущий: `{parsed['current_symbol'] or ''}`")
        st.progress(parsed["symbols_done"] / parsed["symbols_total"])

    if parsed["collect_total"] > 0:
        st.markdown(f"**Сбор данных:** `{parsed['collect_done']}/{parsed['collect_total']}` · текущий: `{parsed['current_collect'] or ''}`")
        st.progress(parsed["collect_done"] / parsed["collect_total"])

    if parsed["wf_total"] > 0:
        st.markdown(f"**Обучение (Walk-Forward):** `{parsed['wf_done']}` / `{parsed['wf_total']}` окон")
        st.progress(parsed["wf_done"] / parsed["wf_total"])
    elif parsed["windows"] and not parsed["is_done"]:
        st.markdown(f"**Обучение (Walk-Forward):** пройдено окон `{len(parsed['windows'])}`")
        
    if parsed["windows"]:
        recent = parsed["windows"][-12:] # inline flex allows more blocks
        html = ""
        for i, w in enumerate(recent):
            last = i == len(recent)-1 and not parsed["is_done"]
            css  = "wf-curr" if last else "wf-done"
            icon = "▶" if last else "✓"
            html += f'<div class="wf-item {css}">{icon} {w}</div>'
        st.markdown(html, unsafe_allow_html=True)

    if parsed["is_done"]:
        st.success("✓ Процесс завершён. Нажмите кнопку «Посмотреть итоги» вверху экрана.")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Errors & Warnings ──
eC, wC = st.columns(2)
with eC:
    ec = len(parsed["errors"])
    badge = f'<span class="badge {"badge-error" if ec else "badge-idle"}">{ec} ошибок</span>'
    st.markdown(f'<div class="section-title">Ошибки {badge}</div>', unsafe_allow_html=True)
    if parsed["errors"]:
        st.markdown("".join(f'<div class="err-item">{e}</div>' for e in reversed(parsed["errors"][-15:])), unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#64748b;font-size:11px">Ошибок нет</span>', unsafe_allow_html=True)

with wC:
    wc = len(parsed["warnings"])
    badge = f'<span class="badge {"badge-running" if wc else "badge-idle"}">{wc} предупреждений</span>'
    st.markdown(f'<div class="section-title">Предупреждения {badge}</div>', unsafe_allow_html=True)
    if show_warns and parsed["warnings"]:
        st.markdown("".join(f'<div class="warn-item">{w}</div>' for w in reversed(parsed["warnings"][-10:])), unsafe_allow_html=True)
    elif not show_warns:
        st.markdown('<span style="color:#64748b;font-size:11px">Скрыто (сайдбар)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#64748b;font-size:11px">Предупреждений нет</span>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Live log ──
st.markdown('<div class="section-title">Живой лог</div>', unsafe_allow_html=True)
log_html = "<br>".join(colorize(l) for l in tail)
st.markdown(f'<div class="log-container">{log_html}</div>', unsafe_allow_html=True)



# ── Auto-refresh ──
if auto_refresh and not parsed["is_done"]:
    time.sleep(refresh_sec)
    st.rerun()
