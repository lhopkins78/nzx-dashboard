"""
NZX Mean Reversion Dashboard — Jim Simons quant terminal style
Streamlit version for cloud hosting

Data sources:
  - NZXplorer API (X-API-Key required, free tier, ~18mo daily history)
  - Yahoo Finance (no key, 3mo daily — fallback if no NZX key)
"""

import streamlit as st
import urllib.request
import json
import math
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

NZX_UNIVERSE = [
    # === Large Cap ===
    ("WBC", "Westpac"),
    ("ANZ", "ANZ Group"),
    ("FPH", "Fisher & Paykel Healthcare"),
    ("MEL", "Meridian Energy"),
    ("IFT", "Infratil"),
    ("AIA", "Auckland Airport"),
    ("CEN", "Contact Energy"),
    ("FCG", "Fonterra"),
    ("MCY", "Mercury NZ"),
    ("VNT", "Ventia Services"),
    ("POT", "Port of Tauranga"),
    ("MFT", "Mainfreight"),
    ("VCT", "Vector"),
    ("ATM", "a2 Milk"),
    ("CNU", "Chorus"),
    ("EBO", "EBOS Group"),
    ("SPK", "Spark NZ"),
    ("GNE", "Genesis Energy"),
    ("FBU", "Fletcher Building"),
    ("GNZ", "Goodman Property Trust"),
    # === Mid Cap ===
    ("FRW", "Freightways"),
    ("RYM", "Ryman Healthcare"),
    ("PCT", "Precinct Properties"),
    ("SUM", "Summerset"),
    ("KPG", "Kiwi Property"),
    ("VHP", "Vital Healthcare"),
    ("CHI", "Channel Infrastructure"),
    ("AIR", "Air New Zealand"),
    ("PFI", "Property for Industry"),
    ("SKL", "Skellerup"),
    ("BGP", "Briscoe Group"),
    ("HGH", "Heartland Group"),
    ("VSL", "Vulcan Steel"),
    ("ARG", "Argosy Property"),
    ("SCL", "Scales Corp"),
    ("TRA", "Turners Auto"),
    ("NPH", "Napier Port"),
    ("TGG", "T&G Global"),
    ("RAK", "Rakon"),
    ("PEB", "Pacific Edge"),
    ("SML", "Synlait Milk"),
    ("CMO", "Colonial Motor Co"),
    ("SPN", "South Port NZ"),
    ("IKE", "ikeGPS"),
    ("SEK", "Seeka"),
    ("WHS", "The Warehouse"),
    ("GXH", "Green Cross Health"),
    ("NZM", "NZME"),
    ("CDI", "CDL Investments"),
    ("SCT", "Scott Technology"),
    ("SKO", "Serko"),
    ("RGI", "Rua Gold"),
    ("LIC", "Livestock Improvement"),
    ("PGW", "PGG Wrightson"),
    ("MHJ", "Michael Hill"),
    ("RAD", "Radius Residential"),
    ("KMD", "KMD Brands"),
    ("NZK", "NZ King Salmon"),
    ("CVT", "Comvita"),
    # === Smaller Cap ===
    ("OCA", "Oceania Healthcare"),
    ("SAN", "Sanford"),
    ("SKC", "SkyCity Entertainment"),
    ("SKT", "Sky Network Television"),
    ("VGL", "Vista Group"),
]

# NZXplorer config
NZX_KEY   = st.secrets.get("NZX_KEY", "")
NZX_BASE  = "https://nzxplorer.co.nz/api/v1/prices"

# Yahoo fallback
YAHOO_BASE = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}.NZ?interval=1d&range=3mo"

C = {
    "bg":      "#0d1117",
    "surface": "#161b22",
    "border":  "#30363d",
    "text":    "#e6edf3",
    "muted":   "#8b949e",
    "accent":  "#58a6ff",
    "buy":     "#3fb950",
    "sell":    "#f85149",
    "warn":    "#d29922",
    "sage":    "#8b9d8a",
    "neutral": "#6e7681",
}

METRIC_COLS = [
    "Ticker", "Name", "Price",
    "20dma", "50dma", "200dma",
    "Dev20", "Dev50",
    "ZScore", "RSI", "BBP",
    "52wkHigh", "52wkLow", "HighDist", "LowDist",
    "VolRatio", "BuyScore",
    "RSIDiv", "VolConfirm", "MAConv", "52wkDrop",
]


# ─────────────────────────────────────────────
# DATA FUNCTIONS
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_nzx(ticker):
    """Fetch daily OHLCV from NZXplorer API (18mo of daily data)."""
    url = f"{NZX_BASE}/{ticker}"
    headers = {"X-API-Key": NZX_KEY, "User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = json.loads(resp.read())
        prices = raw["data"]["prices"]
        if not prices:
            return None
        df = pd.DataFrame(prices)
        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={
            "adjusted_close": "adjclose",
            "date": "timestamp",
        })
        df = df.set_index("timestamp").sort_index()
        df["close"]  = df["close"].astype(float)
        df["high"]   = df["high"].astype(float)
        df["low"]    = df["low"].astype(float)
        df["open"]   = df["open"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df
    except Exception as e:
        return None


@st.cache_data(ttl=3600)
def fetch_yahoo(ticker):
    """Fetch from Yahoo Finance (3mo daily fallback)."""
    url = YAHOO_BASE.format(ticker=ticker)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = json.loads(resp.read())
        result = raw["chart"]["result"]
        if not result:
            return None
        r  = result[0]
        ts = r.get("timestamp", [])
        q  = r["indicators"]["quote"][0]
        df = pd.DataFrame({"timestamp": ts})
        df["close"]  = q.get("close")
        df["high"]   = q.get("high")
        df["low"]    = q.get("low")
        df["open"]   = q.get("open")
        df["volume"] = q.get("volume")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Pacific/Auckland")
        df = df.dropna(subset=["close"]).set_index("timestamp").sort_index()
        return df
    except Exception:
        return None


def calc_ma(s, w):   return s.rolling(w).mean()
def calc_rsi(s, w=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(w).mean()
    l = (-d.clip(upper=0)).rolling(w).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))
def calc_bbp(s, w=20):
    m = s.rolling(w).mean()
    std = s.rolling(w).std()
    return (s - m) / (2 * std)
def calc_zscore(s, w=63):
    m = s.rolling(w).mean()
    std = s.rolling(w).std().replace(0, np.nan)
    return (s - m) / std
def calc_vol_surge(v, w=20):
    avg = v.rolling(w).mean()
    return v / avg.replace(0, np.nan)


def calc_rsi_divergence(close, rsi, w=20):
    if len(close) < w + 2 or len(rsi) < w + 2:
        return None
    price_slice = close.iloc[-w:]
    rsi_slice   = rsi.iloc[-w:]
    price_dipped = price_slice.iloc[0] > price_slice.iloc[-1]
    rsi_lifted   = rsi_slice.iloc[0] < rsi_slice.iloc[-1]
    if price_dipped and rsi_lifted:
        return "Bull"
    price_raised = price_slice.iloc[0] < price_slice.iloc[-1]
    rsi_dropped  = rsi_slice.iloc[0] > rsi_slice.iloc[-1]
    if price_raised and rsi_dropped:
        return "Bear"
    return None


def calc_health_check(df, m):
    close  = df["close"]
    volume = df["volume"]
    rsi    = calc_rsi(close)

    div = calc_rsi_divergence(close, rsi)

    vol_now = float(calc_vol_surge(volume).iloc[-1])
    vol_confirm = "LowVol" if vol_now < 0.7 else ("HighVol" if vol_now > 1.5 else "Normal")

    price = float(close.iloc[-1])
    ma20_v  = m.get("20dma")
    ma50_v  = m.get("50dma")
    ma200_v = m.get("200dma")
    mas_below = sum([
        1 if ma20_v  and price < ma20_v  else 0,
        1 if ma50_v  and price < ma50_v  else 0,
        1 if ma200_v and price < ma200_v else 0,
    ])

    drop = m.get("HighDist", 0)

    z_part   = min(abs(m.get("ZScore", 0)) / 3 * 35, 35)
    rsi_part = max((50 - m.get("RSI", 50)) / 50 * 30, 0)
    dev20    = max(-m.get("Dev20", 0), 0) / 10 * 15
    dev50    = max(-m.get("Dev50", 0), 0) / 10 * 10
    div_bull = 10 if div == "Bull" else 0
    vol_low  = 5  if vol_confirm == "LowVol" else 0
    ma_conv  = mas_below / 3 * 10
    buy_score = round(z_part + rsi_part + dev20 + dev50 + div_bull + vol_low + ma_conv, 1)

    return {
        "BuyScore":   buy_score,
        "RSIDiv":    div or "—",
        "VolConfirm": vol_confirm,
        "MAConv":    f"{mas_below}/3",
        "52wkDrop":  round(drop, 1) if drop else 0,
    }


def calc_metrics(df):
    if df is None or len(df) < 65:
        return None
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    price  = float(close.iloc[-1])
    ma20   = float(calc_ma(close, 20).iloc[-1])
    ma50   = float(calc_ma(close, 50).iloc[-1])
    ma200  = float(calc_ma(close, 200).iloc[-1]) if len(close) >= 200 else None

    # Use 63-day z-score if we have enough data, else fall back to shorter window
    z_window = 63 if len(close) >= 63 else 20
    zscore   = float(calc_zscore(close, z_window).iloc[-1])

    rsi    = float(calc_rsi(close).iloc[-1])
    bbp    = float(calc_bbp(close).iloc[-1])
    h52    = float(high.rolling(252).max().iloc[-1]) if len(close) >= 252 else None
    l52    = float(low.rolling(252).min().iloc[-1])   if len(close) >= 252 else None
    vol20  = float(calc_vol_surge(volume).iloc[-1])

    base = {
        "Price":     round(price, 3),
        "20dma":     round(ma20, 3),
        "50dma":     round(ma50, 3),
        "200dma":    round(ma200, 3) if ma200 else None,
        "Dev20":     round((price - ma20) / ma20 * 100, 2),
        "Dev50":     round((price - ma50) / ma50 * 100, 2),
        "ZScore":    round(zscore, 2),
        "ZWindow":   z_window,
        "RSI":       round(rsi, 1),
        "BBP":       round(bbp, 3),
        "52wkHigh":  round(h52, 3) if h52 else None,
        "52wkLow":   round(l52, 3) if l52 else None,
        "HighDist":  round((price - h52) / h52 * 100, 2) if h52 else None,
        "LowDist":   round((price - l52) / l52 * 100, 2) if l52 else None,
        "VolRatio":  round(vol20, 2),
        "DataPts":   len(close),
        "DataSrc":   "NZX" if len(close) > 100 else "Yahoo",
    }
    base.update(calc_health_check(df, base))
    return base


@st.cache_data(ttl=3600)
def build_universe(use_nzx=True):
    rows  = []
    meta  = {}
    for ticker, name in NZX_UNIVERSE:
        df = fetch_nzx(ticker) if use_nzx else None
        if df is None:
            df = fetch_yahoo(ticker)
        m = calc_metrics(df)
        if m:
            row = {"Ticker": ticker, "Name": name}
            row.update(m)
            rows.append(row)
            meta[ticker] = {"status": "ok", "pts": m["DataPts"], "src": m["DataSrc"]}
        else:
            meta[ticker] = {"status": "fail", "pts": 0, "src": "-"}
    return pd.DataFrame(rows, columns=METRIC_COLS), meta


# ─────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────

def signal_color(v, col):
    try:
        v = float(v)
    except:
        return "normal"
    if col == "ZScore":
        if v < -1.5:  return "buy"
        if v < -0.5:  return "mild_buy"
        if v >  1.5:  return "sell"
        if v >  0.5:  return "mild_sell"
        return "neutral"
    if col == "RSI":
        if v < 30:    return "buy"
        if v < 45:    return "mild_buy"
        if v > 70:    return "sell"
        if v > 55:    return "mild_sell"
        return "neutral"
    if col in ("Dev20", "Dev50", "HighDist", "LowDist"):
        if v < -10:   return "buy"
        if v < -3:    return "mild_buy"
        if v >  10:   return "sell"
        if v >  3:    return "mild_sell"
        return "neutral"
    return "neutral"


# ─────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────

def make_chart(ticker, name, metrics, df_raw):
    close = df_raw["close"]
    high  = df_raw["high"]
    low   = df_raw["low"]
    ma20  = calc_ma(close, 20)
    ma50  = calc_ma(close, 50)
    ma200 = calc_ma(close, 200) if len(close) >= 200 else None
    bb_up = ma20 + 2 * close.rolling(20).std()
    bb_dn = ma20 - 2 * close.rolling(20).std()
    rsi   = calc_rsi(close)
    dates = close.index

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=close, name="Close",
                              line=dict(color=C["accent"], width=1.8)))
    fig.add_trace(go.Scatter(x=dates, y=ma20, name="MA20",
                              line=dict(color=C["warn"], width=1.2, dash="dot")))
    fig.add_trace(go.Scatter(x=dates, y=ma50, name="MA50",
                              line=dict(color=C["sage"], width=1.2, dash="dash")))
    if ma200 is not None:
        fig.add_trace(go.Scatter(x=dates, y=ma200, name="MA200",
                                  line=dict(color=C["muted"], width=1, dash="longdash")))
    fig.add_trace(go.Scatter(x=dates, y=bb_up, name="BB+",
                              line=dict(color=C["muted"], width=1, dash="dot"),
                              fill=None, mode="lines", visible="legendonly"))
    fig.add_trace(go.Scatter(x=dates, y=bb_dn, name="BB-",
                              line=dict(color=C["muted"], width=1, dash="dot"),
                              fill="tonexty", fillcolor="rgba(139,157,138,0.07)",
                              mode="lines", visible="legendonly"))
    fig.add_trace(go.Scatter(x=dates, y=rsi, name="RSI(14)",
                              line=dict(color=C["neutral"], width=1.2),
                              yaxis="y2"))

    if len(close) >= 252:
        high52 = high.rolling(252).max()
        low52  = low.rolling(252).min()
        fig.add_trace(go.Scatter(x=dates, y=high52, name="52wk High",
                                  line=dict(color=C["sell"], width=1, dash="dot"),
                                  visible="legendonly"))
        fig.add_trace(go.Scatter(x=dates, y=low52, name="52wk Low",
                                  line=dict(color=C["buy"], width=1, dash="dot"),
                                  visible="legendonly"))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg"],
        font=dict(color=C["text"], family="monospace"),
        title=dict(
            text=(f"<b>{ticker}</b> — {name}  |  "
                  f"${metrics.get('Price','—')}  |  "
                  f"Z={metrics.get('ZScore','—')}  RSI={metrics.get('RSI','—')}  "
                  f"({metrics.get('ZWindow',20)}d window)"),
            font=dict(color=C["text"], size=14),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(gridcolor=C["border"], color=C["muted"]),
        yaxis=dict(title="NZD", gridcolor=C["border"], color=C["muted"],
                   tickformat="$.2f"),
        yaxis2=dict(title="RSI", overlaying="y", side="right", range=[0, 100],
                    gridcolor=C["border"], color=C["neutral"], tickvals=[30, 50, 70]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=C["muted"])),
        margin=dict(l=60, r=60, t=80, b=60),
        height=420,
    )
    return fig


# ─────────────────────────────────────────────
# STREAMLIT PAGE
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="NZX Mean Reversion",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; }
    .stMetric { color: #e6edf3; }
    .stButton > button { background-color: #161b22; color: #e6edf3; border: 1px solid #30363d; }
    div[data-baseweb="table"] { background-color: #161b22; }
    table { font-family: monospace; font-size: 13px; }
    thead { background-color: #161b22; color: #8b949e; }
    tbody tr:nth-child(odd) { background-color: #0d1117; }
    tbody tr:nth-child(even) { background-color: #161b22; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ──
st.markdown("""
## 🌙 NZX Mean Reversion Dashboard
*Jim Simons–style quant terminal · Data: NZXplorer API (18mo daily)*
""")

# ── DATA SOURCE SELECTOR ──
col_hdr, col_src = st.columns([4, 1])
with col_src:
    use_nzx = st.toggle("Use NZX data", value=True, help="NZXplorer API (~18mo daily) vs Yahoo Finance (3mo daily)")
    if use_nzx:
        st.caption("📡 NZXplorer (18mo)")
    else:
        st.caption("📡 Yahoo (3mo)")

# ── LOAD DATA ──
df, meta = build_universe(use_nzx=use_nzx)

col1, col2, col3, col4 = st.columns(4)
oversold   = df[df["ZScore"] < -1]
neutral    = df[(df["ZScore"] >= -1) & (df["ZScore"] <= 1)]
overbought = df[df["ZScore"] > 1]

with col1:
    st.metric("Oversold  Z < -1", len(oversold))
with col2:
    st.metric("Neutral", len(neutral))
with col3:
    st.metric("Overbought  Z > 1", len(overbought))
with col4:
    ok_count = sum(1 for v in meta.values() if v["status"] == "ok")
    st.metric("Universe", f"{len(df)}/{len(NZX_UNIVERSE)}")

# ── WATCHLISTS ──
wc1, wc2, wc3 = st.columns(3)

with wc1:
    st.markdown("🔥 **Oversold Ranked by BuyScore**")
    oversold_bs = oversold.sort_values("BuyScore", ascending=False)
    for _, r in oversold_bs.head(6).iterrows():
        src = "🟢" if r["BuyScore"] >= 70 else ("🟡" if r["BuyScore"] >= 50 else "⚪")
        st.markdown(f"{src} `{r['Ticker']}` · Buy={r['BuyScore']} · Z={r['ZScore']} · RSI={r['RSI']}")

with wc2:
    st.markdown("📊 **Signal Summary**")
    st.markdown(f"Oversold (Z<-1): **{len(oversold)}**")
    st.markdown(f"Neutral: **{len(neutral)}**")
    st.markdown(f"Overbought (Z>1): **{len(overbought)}**")
    st.markdown("---")
    st.markdown(f"RSI &lt; 30: **{len(df[df['RSI'] < 30])}**")
    st.markdown(f"RSI &gt; 70: **{len(df[df['RSI'] > 70])}**")
    st.markdown(f"BBP &lt; 0: **{len(df[df['BBP'] < 0])}**")
    st.markdown(f"BBP &gt; 1: **{len(df[df['BBP'] > 1])}**")
    st.markdown(f"200dma avail: **{len(df[df['200dma'].notna()])}**")
    st.markdown("---")
    st.markdown(f"**BuyScore ≥70 (Strong):** {len(df[df['BuyScore'] >= 70])}")
    st.markdown(f"BuyScore 50-70 (Good): {len(df[(df['BuyScore'] >= 50) & (df['BuyScore'] < 70)])}")

with wc3:
    st.markdown("💤 **Overbought (Z > 1)**")
    for _, r in overbought.nlargest(6, "ZScore").iterrows():
        src = "🔴" if r["ZScore"] > 1.5 else "🟡"
        st.markdown(f"{src} `{r['Ticker']}` · Z={r['ZScore']} · RSI={r['RSI']}")

st.markdown("---")

# ── MAIN TABLE ──
st.markdown("### 📈 Full Universe — click ticker for chart")

def fmt_signal(v, col):
    sig = signal_color(v, col)
    icons = {"buy": "🟢", "mild_buy": "🟢", "sell": "🔴", "mild_sell": "🟡", "neutral": "  "}
    return f"{icons.get(sig, '')}{v}" if sig != "normal" else f"{v}"

display_df = df.copy()
for col in METRIC_COLS:
    if col in ("Ticker", "Name"):
        continue
    display_df[col] = display_df[col].apply(
        lambda x: fmt_signal(x, col) if not pd.isna(x) else "—"
    )

st.dataframe(display_df[METRIC_COLS], use_container_width=True, hide_index=True)

# ── STOCK DETAIL CHART ──
st.markdown("### Selected Stock Chart")
selected_ticker = st.selectbox(
    "Choose a ticker:",
    options=df["Ticker"].tolist(),
    index=0,
)

row  = df[df["Ticker"] == selected_ticker].iloc[0]
name = row["Name"]
m    = row.to_dict()

df_raw = fetch_nzx(selected_ticker) if use_nzx else fetch_yahoo(selected_ticker)
if df_raw is not None:
    fig = make_chart(selected_ticker, name, m, df_raw)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning(f"Could not load data for {selected_ticker}")

# ── HEALTH CHECK PANEL ──
st.markdown("### 🏥 Jim Simons Health Check")
bs   = m.get("BuyScore", 0)
div  = m.get("RSIDiv", "—")
vol  = m.get("VolConfirm", "—")
ma_n = int(str(m.get("MAConv", "0/3")).split("/")[0])
drop = m.get("52wkDrop", 0)

bs_label = "STRONG" if bs >= 70 else "GOOD" if bs >= 50 else "FAIR" if bs >= 30 else "WEAK"
bs_col   = "green" if bs >= 70 else ("olive" if bs >= 50 else ("grey" if bs >= 30 else "red"))

div_icon  = "✅ Bull" if div == "Bull" else ("⚠️ Bear" if div == "Bear" else "—")
vol_icon  = "🔇 LowVol (reliable)" if vol == "LowVol" else ("📈 HighVol (caution)" if vol == "HighVol" else "📊 Normal")
ma_col    = "green" if ma_n >= 2 else ("olive" if ma_n == 1 else "grey")

st.markdown(f"""
| Metric | Value |
|--------|-------|
| **BuyScore** | **{bs}** — {bs_label} |
| **RSI Divergence** | {div_icon} |
| **Volume** | {vol_icon} |
| **MAs Below** | {m.get('MAConv','—')} / 3 |
| **52wk Drop** | {abs(drop):.1f}% |
| **ZScore** | {m.get('ZScore','—')} |

*BuyScore = ZScore(35%) + RSI reversal room(30%) + Dev20%(15%) + Dev50%(10%) + RSI Divergence(10%) + Low Volume(5%) + MA Convergence(10%) · ≥70 = Strong buy setup · 50-70 = Good · 30-50 = Fair · <30 = Weak*
""")

# ── LEGEND ──
st.markdown("""
| Colour | Signal |
|--------|--------|
| 🟢 green | Oversold / mean reversion buy |
| 🟡 amber | Mild premium / slightly rich |
| 🔴 red | Overbought / expensive |

**Z-Window:** NZX data uses 63-day rolling window (quarterly). Yahoo uses 20-day (3-mo limit).
**200dma:** Available for stocks with 200+ trading days of history.
**52wk High/Low:** Available for stocks with 252+ trading days (~1 year).
""")

st.caption(
    f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
    f"Data: {'NZXplorer API' if use_nzx else 'Yahoo Finance'} · "
    f"Z-window: {'63d' if use_nzx else '20d'}"
)