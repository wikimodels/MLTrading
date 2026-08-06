"""Build a standalone interactive HTML chart of hourly market breadth over the last 1y.

Breadth = fraction of universe (44 alts + BTC) whose Chandelier Exit direction
just flipped to short (-1) / long (1) at each hour (entry signal events, not state).
ECharts 5 is embedded inline (works offline)."""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
os.chdir(_root)

from loguru import logger
from shared.config_loader import set_active_bot
set_active_bot("breadth_bot")

from shared.data.storage import DataStorage
from breadth_bot.screener.screener import load_hourly_ohlcv
from breadth_bot.strategy.signals import BreadthSignalGenerator

HOURS_1Y = 24 * 365
ECHARTS_JS = Path(r"C:\Users\Vitali\AppData\Local\Temp\opencode\echarts.min.js").read_text(encoding="utf-8")


def main() -> None:
    storage = DataStorage()
    gen = BreadthSignalGenerator()
    symbols = storage.list_symbols("1h")

    data = {}
    for sym in symbols:
        df = load_hourly_ohlcv(storage, sym)
        if df is None:
            continue
        df = df.tail(HOURS_1Y)
        if len(df) < HOURS_1Y:
            logger.warning(f"{sym}: only {len(df)}h, skipped")
            continue
        sig = gen.compute_signals(df)
        data[sym] = sig[["ce_dir", "short_sig", "long_sig", "close"]]

    dates = sorted(set().union(*[set(df.index) for df in data.values()]))
    short_count = {}
    long_count = {}
    for t in dates:
        sc = 0; lc = 0
        for sym, df in data.items():
            if t in df.index:
                if df.at[t, "short_sig"]:
                    sc += 1
                if df.at[t, "long_sig"]:
                    lc += 1
        short_count[t] = sc / len(data)
        long_count[t] = lc / len(data)

    times = [t.strftime("%Y-%m-%d %H:%M") for t in dates]
    short_b = [round(short_count[t], 4) for t in dates]
    long_b = [round(long_count[t], 4) for t in dates]
    btc_close = []
    if "BTC/USDT:USDT" in data:
        btc_close = [round(float(data["BTC/USDT:USDT"].at[t, "close"]), 2) if t in data["BTC/USDT:USDT"].index else None for t in dates]

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Market Breadth (1y, 1h) - Chandelier Exit</title>
<script>
__ECHARTS__
</script>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; background:#0f1115; color:#e8e8e8; }
  h1 { font-size: 18px; }
  .note { color:#9aa0a6; font-size: 13px; margin-bottom: 8px; }
  .chart-wrap { position: relative; height: 460px; margin-bottom: 8px; }
  .chart-wrap.small { height: 260px; }
  .chart-wrap > div { position: absolute; inset: 0; }
</style>
</head>
<body>
<h1>Market Breadth &mdash; 45 symbols, 1h, last 1 year</h1>
<div class="note">Breadth = share of the universe whose Chandelier Exit(21, 2.5) direction JUST flipped to SHORT (red) or LONG (green) at each hour (entry events, not held state). Data range: {RANGE}. Mouse wheel = zoom, drag on the bottom dataZoom bar = pan, double-click = reset zoom.</div>
<div class="chart-wrap"><div id="breadth"></div></div>
<div class="note">BTC/USDT close for context (locked to breadth zoom).</div>
<div class="chart-wrap small"><div id="btc"></div></div>
<script>
const times = __TIMES__;
const shortB = __SHORT__;
const longB = __LONG__;
const btc = __BTC__;

const axisColor = '#9aa0a6';
const gridLine = { show: true, lineStyle: { color: 'rgba(255,255,255,0.07)' } };

const base = {
  backgroundColor: 'transparent',
  textStyle: { color: axisColor },
  grid: { left: 54, right: 16, top: 28, bottom: 58 },
  tooltip: { trigger: 'axis', backgroundColor: '#1a1d23', borderColor: '#333', textStyle: { color: '#e8e8e8', fontSize: 12 }, axisPointer: { type: 'cross', label: { backgroundColor: '#333' } } },
  legend: { top: 0, textStyle: { color: axisColor }, itemWidth: 12, itemHeight: 8 },
  xAxis: {
    type: 'category', data: times, boundaryGap: false,
    axisLine: { lineStyle: { color: 'rgba(255,255,255,0.15)' } },
    axisLabel: { color: axisColor },
    axisPointer: { show: true }
  },
  dataZoom: [
    { type: 'inside', xAxisIndex: 0, filterMode: 'none', zoomOnMouseWheel: true, moveOnMouseMove: true, moveOnMouseWheel: false },
    { type: 'slider', xAxisIndex: 0, height: 18, bottom: 8, filterMode: 'none', borderColor: '#333', backgroundColor: '#15171c', fillerColor: 'rgba(120,140,180,0.25)', handleStyle: { color: '#7a8aa8' }, textStyle: { color: axisColor, fontSize: 10 } }
  ]
};

const breadthChart = echarts.init(document.getElementById('breadth'));
breadthChart.setOption({
  ...base,
  grid: { ...base.grid, top: 44, bottom: 58 },
  legend: { ...base.legend, top: 12 },
  yAxis: {
    type: 'value', min: 0, max: 1,
    axisLabel: { color: axisColor, formatter: (v) => (v * 100) + '%' },
    splitLine: gridLine,
    name: 'share of market', nameTextStyle: { color: axisColor }
  },
  series: [
    { name: 'SHORT entry', type: 'line', data: shortB, symbol: 'none', lineStyle: { color: '#ef5350', width: 1 }, areaStyle: { color: 'rgba(239,83,80,0.18)' }, z: 2 },
    { name: 'LONG entry', type: 'line', data: longB, symbol: 'none', lineStyle: { color: '#66bb6a', width: 1 }, areaStyle: { color: 'rgba(102,187,106,0.18)' }, z: 2 },
    { name: '50% threshold', type: 'line', data: times.map(() => 0.5), symbol: 'none', lineStyle: { color: '#888', width: 1, type: 'dashed' }, z: 1 }
  ]
});

const btcChart = echarts.init(document.getElementById('btc'));
btcChart.setOption({
  ...base,
  grid: { ...base.grid, top: 28, bottom: 44 },
  legend: { ...base.legend, top: 4 },
  yAxis: {
    type: 'value', scale: true,
    axisLabel: { color: axisColor },
    splitLine: gridLine,
    name: 'price', nameTextStyle: { color: axisColor }
  },
  series: [
    { name: 'BTC/USDT', type: 'line', data: btc, symbol: 'none', lineStyle: { color: '#ffb300', width: 1 }, z: 2 }
  ]
});

breadthChart.on('datazoom', (params) => {
  const start = params.batch ? params.batch[0].start : params.start;
  const end = params.batch ? params.batch[0].end : params.end;
  btcChart.dispatchAction({ type: 'dataZoom', dataZoomIndex: 0, start, end });
});

const resetBtn = document.createElement('button');
resetBtn.textContent = 'Reset zoom';
resetBtn.style.cssText = 'position:fixed;bottom:14px;right:14px;background:#1a1d23;border:1px solid #333;color:#c9cdd3;padding:6px 12px;border-radius:6px;font-size:12px;cursor:pointer;z-index:10;';
resetBtn.onclick = () => {
  breadthChart.dispatchAction({ type: 'dataZoom', dataZoomIndex: 0, start: 0, end: 100 });
  btcChart.dispatchAction({ type: 'dataZoom', dataZoomIndex: 0, start: 0, end: 100 });
};
document.body.appendChild(resetBtn);

window.addEventListener('resize', () => { breadthChart.resize(); btcChart.resize(); });
</script>
</body>
</html>
"""

    html = (html
            .replace("{RANGE}", f"{dates[0]:%Y-%m-%d} -> {dates[-1]:%Y-%m-%d}")
            .replace("__ECHARTS__", ECHARTS_JS)
            .replace("__TIMES__", json.dumps(times))
            .replace("__SHORT__", json.dumps(short_b))
            .replace("__LONG__", json.dumps(long_b))
            .replace("__BTC__", json.dumps(btc_close)))

    out = _root / "breadth_bot" / "backtest" / "breadth_1y.html"
    out.write_text(html, encoding="utf-8")
    print(f"Saved: {out} ({out.stat().st_size/1024:.0f} KB)")
    print(f"bars: {len(dates)}, universe: {len(data)}, avg short breadth: {sum(short_b)/len(short_b):.3f}, avg long breadth: {sum(long_b)/len(long_b):.3f}")


if __name__ == "__main__":
    main()
