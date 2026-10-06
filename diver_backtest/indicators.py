import numpy as np
import pandas as pd

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

# ---------- RSI ----------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/n, adjust=False).mean()
    al = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

# ---------- VZO (Volume Zone Oscillator) ----------
def vzo(df: pd.DataFrame, n: int = 14) -> pd.Series:
    c = df["close"]
    v = df["volume"]
    sign = np.sign(c.diff()).fillna(0)
    vp = sign * v
    ev = ema(v, n).replace(0, np.nan)
    return 100 * ema(vp, n) / ev

# ---------- KVO (Klinger Volume Oscillator) ----------
def kvo(df: pd.DataFrame, fast: int = 34, slow: int = 55) -> pd.Series:
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
    dm = h - l
    trend = np.sign(c.diff()).fillna(0)
    cm = np.where(trend > 0, (h - l.shift()).abs(), (l - h.shift()).abs())
    cm = pd.Series(cm, index=df.index).replace(0, np.nan)
    vf = v * (2 * (dm / cm - 1)).abs() * trend * 100
    return ema(vf.fillna(0), fast) - ema(vf.fillna(0), slow)

# ---------- CMF (Chaikin Money Flow) ----------
def cmf(df: pd.DataFrame, n: int = 21) -> pd.Series:
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
    hl = (h - l).replace(0, np.nan)
    mfm = ((c - l) - (h - c)) / hl
    mfv = mfm * v
    vol_sum = v.rolling(n).sum().replace(0, np.nan)
    return mfv.rolling(n).sum() / vol_sum

# ---------- EFI (Elder Force Index) ----------
def efi(df: pd.DataFrame, n: int = 13) -> pd.Series:
    c, v = df["close"], df["volume"]
    return ema((c - c.shift()) * v, n)

# ---------- Fisher Transform (Оптимизирован на numpy) ----------
def fisher(df: pd.DataFrame, n: int = 10) -> pd.Series:
    h, l = df["high"], df["low"]
    hl = (h.rolling(n).max() - l.rolling(n).min()).replace(0, np.nan)
    x = 2 * ((df["close"] - l.rolling(n).min()) / hl - 0.5)
    x = x.clip(-0.999, 0.999).fillna(0).to_numpy()
    
    n_len = len(x)
    out = np.zeros(n_len, dtype=np.float64)
    prev = 0.0
    for i in range(n_len):
        xi = 0.33 * 2.0 * x[i] + 0.67 * prev
        xi = max(min(xi, 0.999), -0.999)
        f = 0.5 * np.log((1.0 + xi) / (1.0 - xi)) + 0.5 * prev
        out[i] = f
        prev = f
    return pd.Series(out, index=df.index)

# ---------- STC (Schaff Trend Cycle) ----------
def stc(df: pd.DataFrame, fast: int = 23, slow: int = 50, k: int = 10, d: int = 3) -> pd.Series:
    c = df["close"]
    macd = ema(c, fast) - ema(c, slow)
    def stoch(s: pd.Series, period: int) -> pd.Series:
        denom = (s.rolling(period).max() - s.rolling(period).min()).replace(0, np.nan)
        return 100 * (s - s.rolling(period).min()) / denom
    s1 = stoch(macd, k).fillna(0)
    s2 = ema(s1, d)
    s3 = stoch(s2, k).fillna(0)
    return ema(s3, d)

# ---------- MACD Histogram ----------
def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, sig)
    return m - s

# ---------- AO (Awesome Oscillator) ----------
def ao(df: pd.DataFrame, fast: int = 5, slow: int = 34) -> pd.Series:
    med = (df["high"] + df["low"]) / 2
    return sma(med, fast) - sma(med, slow)

# ---------- Z-score 200 EMA ----------
def zscore_ema(close: pd.Series, ma_n: int = 200, std_n: int = 20) -> pd.Series:
    e = ema(close, ma_n)
    d = close - e
    std_val = d.rolling(std_n).std().replace(0, np.nan)
    return d / std_val

# (Removed dead code for bbw, kc_width, anchored_vwap)

INDICATOR_MAP = {
    "RSI_10": lambda df: rsi(df["close"], n=10),
    "RSI_14": lambda df: rsi(df["close"], n=14),
    "VZO_10": lambda df: vzo(df, n=10),
    "VZO_14": lambda df: vzo(df, n=14)
}

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "volume" in df.columns:
        df["vol_ma"] = sma(df["volume"], 20).fillna(df["volume"])
    df["ATR"] = atr(df)
    for name, fn in INDICATOR_MAP.items():
        try:
            df[name] = fn(df)
        except Exception:
            df[name] = np.nan
    return df
