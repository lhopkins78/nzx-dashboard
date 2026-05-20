"""
NZX Mean Reversion Dashboard — Jim Simons quant terminal style
Data: Yahoo Finance (no API key needed)
"""

import urllib.request
import json
import math
from datetime import datetime
from typing import Optional

import dash
from dash import dash_table, html, dcc, callback, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

NZX_UNIVERSE = [
    ("FPH", "Fisher & Paykel Healthcare"),
    ("CEN", "Contact Energy"),
    ("GEN", "Genesis Energy"),
    ("GNE", "Greencoat NZ Wind"),
    ("IPL", "Infratil"),
    ("MCY", "Meridian Energy"),
    ("MEL", "Mercury NZ"),
    ("OCA", "Oceania Healthcare"),
    ("PFI", "Property for Industry"),
    ("SAN", "Sanford"),
    ("SKC", "SkyCity Entertainment"),
    ("SKT", "Sky Network Television"),
    ("SPK", "Spark NZ"),
    ("VGL", "Vista Group"),
    ("WBC", "Westpac"),
    ("ANZ", "ANZ"),
    ("FBU", "Fletcher Building"),
    ("MFT", "Mainfreight"),
]

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

def fetch_ohlc(ticker: str) -> Optional[pd.DataFrame]:
    url = YAHOO_BASE.format(ticker=ticker)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = json.loads(resp.read())
        result = raw["chart"]["result"]
        if not result:
            return None
        r = result[0]
        ts  = r.get("timestamp", [])
        q   = r["indicators"]["quote"][0]
        adj = r["indicators"].get("adjclose", [q])[0]
        df  = pd.DataFrame({"timestamp": ts})
        df["close"]    = q.get("close")
        df["high"]     = q.get("high")
        df["low"]      = q.get("low")
        df["open"]     = q.get("open")
        df["volume"]   = q.get("volume")
        df["adjclose"] = adj.get("adjclose", df["close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Pacific/Auckland")
        df = df.dropna(subset=["close"]).set_index("timestamp").sort_index()
        return df
    except Exception as e:
        print(f"[WARN] {ticker}: {e}")
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
def calc_zscore(s, w=None):
    w = min(w or 20, len(s))
    m = s.rolling(w).mean()
    std = s.rolling(w).std().replace(0, np.nan)
    return (s - m) / std
def calc_vol_surge(v, w=20):
    avg = v.rolling(w).mean()
    return v / avg.replace(0, np.nan)


def calc_rsi_divergence(close, rsi, w=20):
    """Return 'Bull' if price makes lower low but RSI makes higher low (hidden bullish div).
    Return 'Bear' if price makes higher high but RSI makes lower high (hidden bearish div).
    Otherwise None."""
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


def calc_health_check(df: pd.DataFrame, m: dict) -> dict:
    """Jim Simons-style health check for oversold/overbought signals."""
    close  = df["close"]
    volume = df["volume"]
    rsi    = calc_rsi(close)

    div = calc_rsi_divergence(close, rsi)

    vol_now = float(calc_vol_surge(volume).iloc[-1])
    if vol_now < 0.7:
        vol_confirm = "LowVol"
    elif vol_now > 1.5:
        vol_confirm = "HighVol"
    else:
        vol_confirm = "Normal"

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
    vol_low  = 5  if vol_confirm == "LowVol"  else 0
    ma_conv  = mas_below / 3 * 10
    buy_score = round(z_part + rsi_part + dev20 + dev50 + div_bull + vol_low + ma_conv, 1)

    return {
        "BuyScore":   buy_score,
        "RSIDiv":    div or "—",
        "VolConfirm": vol_confirm,
        "MAConv":    f"{mas_below}/3",
        "52wkDrop":  round(drop, 1) if drop else 0,
    }


def calc_metrics(df: pd.DataFrame) -> Optional[dict]:
    if df is None or len(df) < 60:
        return None
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    price = float(close.iloc[-1])
    ma20  = float(calc_ma(close, 20).iloc[-1])
    ma50  = float(calc_ma(close, 50).iloc[-1])
    ma200 = float(calc_ma(close, 200).iloc[-1]) if len(close) >= 200 else None
    rsi   = float(calc_rsi(close).iloc[-1])
    bbp   = float(calc_bbp(close).iloc[-1])
    zscore = float(calc_zscore(close).iloc[-1])
    h52   = float(high.rolling(252).max().iloc[-1])
    l52   = float(low.rolling(252).min().iloc[-1])
    vol20 = float(calc_vol_surge(volume).iloc[-1])

    base = {
        "Price":     round(price, 3),
        "20dma":     round(ma20, 3),
        "50dma":     round(ma50, 3),
        "200dma":    round(ma200, 3) if ma200 else None,
        "Dev20":     round((price - ma20) / ma20 * 100, 2),
        "Dev50":     round((price - ma50) / ma50 * 100, 2),
        "ZScore":    round(zscore, 2),
        "RSI":       round(rsi, 1),
        "BBP":       round(bbp, 3),
        "52wkHigh":  round(h52, 3),
        "52wkLow":   round(l52, 3),
        "HighDist":  round((price - h52) / h52 * 100, 2),
        "LowDist":   round((price - l52) / l52 * 100, 2),
        "VolRatio":  round(vol20, 2),
    }
    base.update(calc_health_check(df, base))
    return base


def build_universe():
    rows = []
    for ticker, name in NZX_UNIVERSE:
        df = fetch_ohlc(ticker)
        m  = calc_metrics(df)
        if m:
            row = {"Ticker": ticker, "Name": name}
            row.update(m)
            rows.append(row)
            print(f"[OK] {ticker}: z={m['ZScore']:.2f} rsi={m['RSI']:.0f} buy={m['BuyScore']:.1f}")
        else:
            print(f"[FAIL] {ticker}")
    return pd.DataFrame(rows, columns=METRIC_COLS)


# ─────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────

def z_color(v):
    try:
        v = float(v)
        if v < -1.5: return C["buy"]
        if v < -0.5: return "#4caf50"
        if v >  1.5: return C["sell"]
        if v >  0.5: return C["warn"]
        return C["neutral"]
    except:
        return C["text"]

def rsi_color(v):
    try:
        v = float(v)
        if v < 30:   return C["buy"]
        if v < 45:   return "#4caf50"
        if v > 70:   return C["sell"]
        if v > 55:   return C["warn"]
        return C["neutral"]
    except:
        return C["text"]

def dev_color(v):
    try:
        v = float(v)
        if v < -10:  return C["buy"]
        if v < -3:   return "#4caf50"
        if v >  10:  return C["sell"]
        if v >  3:   return C["warn"]
        return C["neutral"]
    except:
        return C["text"]

def vol_color(v):
    try:
        v = float(v)
        if v > 2.0:  return C["warn"]
        return C["text"]
    except:
        return C["text"]

def buyscore_color(v):
    try:
        v = float(v)
        if v >= 70:  return C["buy"]
        if v >= 50:  return "#4caf50"
        if v >= 30:  return C["neutral"]
        return C["muted"]
    except:
        return C["text"]

def health_color(col, val):
    if col == "RSIDiv":
        return C["buy"] if val == "Bull" else (C["sell"] if val == "Bear" else C["neutral"])
    if col == "VolConfirm":
        return C["buy"] if val == "LowVol" else (C["warn"] if val == "HighVol" else C["neutral"])
    if col == "MAConv":
        try:
            n = int(str(val).split("/")[0])
            if n >= 2: return C["buy"]
            if n == 1: return "#4caf50"
            return C["muted"]
        except:
            return C["text"]
    return C["text"]

def cell_color(col, val):
    if col in ("ZScore",):              return z_color(val)
    if col in ("RSI",):                 return rsi_color(val)
    if col in ("Dev20", "Dev50", "HighDist", "LowDist"): return dev_color(val)
    if col in ("VolRatio",):            return vol_color(val)
    if col in ("BuyScore",):            return buyscore_color(val)
    if col in ("RSIDiv", "VolConfirm", "MAConv"): return health_color(col, val)
    return C["text"]


# ─────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────

def make_chart(ticker, df_raw, metrics):
    close = df_raw["close"]
    high  = df_raw["high"]
    low   = df_raw["low"]
    ma20  = calc_ma(close, 20)
    ma50  = calc_ma(close, 50)
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
            text=(f"<b>{ticker}</b> — {metrics.get('Name','')}  |  "
                  f"${metrics.get('Price','—')}  |  "
                  f"Z={metrics.get('ZScore','—')}  RSI={metrics.get('RSI','—')}"),
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
# HEALTH CHECK PANEL
# ─────────────────────────────────────────────

def health_check_panel(m):
    score   = m.get("BuyScore", 0)
    rsi_div = m.get("RSIDiv", "—")
    vol_c   = m.get("VolConfirm", "—")
    ma_conv = m.get("MAConv", "—")
    drop    = m.get("52wkDrop", 0)

    score_color = buyscore_color(score)
    score_label = "STRONG" if score >= 70 else "GOOD" if score >= 50 else "FAIR" if score >= 30 else "WEAK"

    div_color = C["buy"] if rsi_div == "Bull" else C["sell"] if rsi_div == "Bear" else C["neutral"]
    div_icon  = "✅" if rsi_div == "Bull" else "⚠️" if rsi_div == "Bear" else "—"

    vol_icon  = "🔇" if vol_c == "LowVol" else "📈" if vol_c == "HighVol" else "📊"
    vol_text  = "LowVol (reliable)" if vol_c == "LowVol" else "HighVol (caution)" if vol_c == "HighVol" else "Normal"

    ma_n = int(str(ma_conv).split("/")[0]) if ma_conv != "—" else 0
    ma_color = C["buy"] if ma_n >= 2 else "#4caf50" if ma_n == 1 else C["muted"]

    return html.Div(children=[
        html.H3("🏥 Jim Simons Health Check", style={"color": C["text"], "font-family": "monospace", "margin": "0 0 16px"}),
        html.Div(style={"display": "grid", "grid-template-columns": "1fr 1fr 1fr", "gap": "12px"}, children=[
            html.Div(style={"text-align": "center"}, children=[
                html.Div(f"{score_label}", style={"color": score_color, "font-size": "11px", "font-family": "monospace", "margin-bottom": "4px"}),
                html.Div(f"{score:.1f}", style={"color": score_color, "font-size": "28px", "font-weight": "bold", "font-family": "monospace"}),
                html.Div("Buy Score", style={"color": C["muted"], "font-size": "11px"}),
            ]),
            html.Div(style={"text-align": "center"}, children=[
                html.Div(div_icon, style={"font-size": "20px", "margin-bottom": "4px"}),
                html.Div(f"{rsi_div}", style={"color": div_color, "font-size": "16px", "font-family": "monospace", "font-weight": "bold"}),
                html.Div("RSI Div", style={"color": C["muted"], "font-size": "11px"}),
            ]),
            html.Div(style={"text-align": "center"}, children=[
                html.Div(vol_icon, style={"font-size": "20px", "margin-bottom": "4px"}),
                html.Div(vol_text, style={"color": C["text"], "font-size": "12px", "font-family": "monospace"}),
                html.Div("Volume", style={"color": C["muted"], "font-size": "11px"}),
            ]),
        ]),
        html.Div(style={"margin-top": "12px", "display": "grid", "grid-template-columns": "1fr 1fr 1fr"}, children=[
            html.Div(style={"text-align": "center"}, children=[
                html.Div(f"{ma_conv}", style={"color": ma_color, "font-size": "18px", "font-family": "monospace", "font-weight": "bold"}),
                html.Div("MAs Below (20/50/200)", style={"color": C["muted"], "font-size": "10px"}),
            ]),
            html.Div(style={"text-align": "center"}, children=[
                html.Div(f"{abs(drop):.1f}%", style={"color": C["warn"], "font-size": "18px", "font-family": "monospace", "font-weight": "bold"}),
                html.Div("52wk Drop", style={"color": C["muted"], "font-size": "10px"}),
            ]),
            html.Div(style={"text-align": "center"}, children=[
                html.Div(f"{m.get('ZScore','—')}", style={"color": z_color(m.get('ZScore')), "font-size": "18px", "font-family": "monospace", "font-weight": "bold"}),
                html.Div("ZScore", style={"color": C["muted"], "font-size": "10px"}),
            ]),
        ]),
        html.Div(style={"margin-top": "12px", "font-size": "11px", "color": C["muted"], "font-family": "monospace", "border-top": f"1px solid {C['border']}", "padding-top": "10px"}, children=[
            html.Span("BuyScore ", style={"color": C["accent"]}),
            html.Span("= Z(35%) + RSI(30%) + Dev20(15%) + Dev50(10%) + Div(10%) + Vol(5%) + MA(10%)"),
            html.Span("  |  ", style={"color": C["border"]}),
            html.Span("\u226570 = Strong  \u2022  50-70 = Good  \u2022  30-50 = Fair  \u2022  <30 = Weak", style={"color": C["muted"]}),
        ]),
    ])


# ─────────────────────────────────────────────
# LAYOUT BUILDER
# ─────────────────────────────────────────────

def make_table(df):
    table_data = df.to_dict("records")
    style_cond = []
    for col in METRIC_COLS:
        for row in table_data:
            val   = row.get(col)
            color = cell_color(col, val)
            if color != C["text"]:
                style_cond.append({
                    "if": {"filter_query": "{{Ticker}} = '{t}'".format(t=row["Ticker"]), "column_id": col},
                    "color": color,
                    "font-weight": "600",
                })
    style_cond.append({"if": {"row_index": "odd"}, "backgroundColor": C["surface"]})
    return dash_table.DataTable(
        id="heatmap-table",
        data=table_data,
        columns=[{"name": c, "id": c} for c in METRIC_COLS],
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto", "font-family": "monospace", "font-size": "13px"},
        style_header={"backgroundColor": C["surface"], "color": C["muted"], "fontWeight": "bold", "border": f"1px solid {C['border']}", "padding": "8px 12px", "textAlign": "center"},
        style_cell={"backgroundColor": C["bg"], "color": C["text"], "border": f"1px solid {C['border']}", "padding": "8px 12px", "textAlign": "right", "minWidth": "80px"},
        style_data_conditional=style_cond,
        style_as_list_view=True,
        row_selectable="single",
        selected_rows=[0],
    )


def watchlist_box(title, color, rows, score_col=None):
    if score_col and len(rows) > 0:
        rows = rows.sort_values(score_col, ascending=False)
    children = [html.H3(title, style={"color": color, "margin": "0 0 8px", "font-size": "13px"})]
    for _, r in rows.iterrows():
        children.append(html.Div(
            f"{r['Ticker']}  Buy={r.get('BuyScore','—')}  Z={r['ZScore']}  RSI={r['RSI']}",
            style={"color": C["text"], "font-family": "monospace", "font-size": "12px", "padding": "3px 0", "border-bottom": f"1px solid {C['border']}"}
        ))
    return html.Div(
        style={"backgroundColor": C["surface"], "border": f"1px solid {C['border']}", "border-radius": "6px", "padding": "16px"},
        children=children,
    )


def serve_layout(df, raw_data):
    oversold   = df[df["ZScore"] < -0.5].nsmallest(6, "ZScore")
    overbought = df[df["ZScore"] >  0.5].nlargest(5, "ZScore")

    return html.Div(
        style={"backgroundColor": C["bg"], "minHeight": "100vh", "padding": "20px"},
        children=[

            # Header
            html.Div(style={
                "border-bottom": f"2px solid {C['sage']}", "padding-bottom": "12px", "margin-bottom": "24px",
                "display": "flex", "justify-content": "space-between", "align-items": "flex-end",
            }, children=[
                html.Div(children=[
                    html.H1("NZX Mean Reversion", style={"font-family": "monospace", "color": C["text"], "font-size": "22px", "margin": "0"}),
                    html.P(f"Universe: {len(df)} stocks  |  Yahoo Finance (3-mo daily)", style={"color": C["muted"], "font-size": "12px", "margin": "4px 0 0"}),
                ]),
                html.Div(children=[
                    html.Span(id="last-updated", children=f"Updated: {datetime.now().strftime('%H:%M:%S')}", style={"color": C["muted"], "font-family": "monospace", "font-size": "12px"}),
                    html.Br(),
                    html.Button("Refresh", id="refresh-btn", n_clicks=0, style={"margin-top": "6px", "backgroundColor": C["surface"], "color": C["text"], "border": f"1px solid {C['border']}", "cursor": "pointer", "padding": "4px 16px", "font-family": "monospace"}),
                ]),
            ]),

            # Watchlists row
            html.Div(style={"display": "grid", "grid-template-columns": "1fr 1fr 1fr", "gap": "16px", "margin-bottom": "24px"}, children=[
                watchlist_box("Oversold Ranked \u2193 BuyScore", C["buy"], oversold, "BuyScore"),

                html.Div(style={"backgroundColor": C["surface"], "border": f"1px solid {C['border']}", "border-radius": "6px", "padding": "16px"}, children=[
                    html.H3("Signal Summary", style={"color": C["accent"], "margin": "0 0 8px", "font-size": "13px"}),
                    html.Div(f"Oversold (Z<-1) : {len(df[df['ZScore'] < -1])}", style={"color": C["buy"], "font-family": "monospace", "font-size": "12px", "padding": "3px 0"}),
                    html.Div(f"Neutral         : {len(df[(df['ZScore'] >= -1) & (df['ZScore'] <= 1)])}", style={"color": C["neutral"], "font-family": "monospace", "font-size": "12px", "padding": "3px 0"}),
                    html.Div(f"Overbought (Z>1): {len(df[df['ZScore'] > 1])}", style={"color": C["sell"], "font-family": "monospace", "font-size": "12px", "padding": "3px 0"}),
                    html.Hr(style={"border-color": C["border"]}),
                    html.Div(f"RSI < 30  : {len(df[df['RSI'] < 30])}", style={"color": C["buy"], "font-family": "monospace", "font-size": "12px", "padding": "3px 0"}),
                    html.Div(f"RSI > 70  : {len(df[df['RSI'] > 70])}", style={"color": C["sell"], "font-family": "monospace", "font-size": "12px", "padding": "3px 0"}),
                    html.Div(f"BBP < 0   : {len(df[df['BBP'] < 0])}", style={"color": C["buy"], "font-family": "monospace", "font-size": "12px", "padding": "3px 0"}),
                    html.Div(f"BBP > 1   : {len(df[df['BBP'] > 1])}", style={"color": C["sell"], "font-family": "monospace", "font-size": "12px", "padding": "3px 0"}),
                    html.Hr(style={"border-color": C["border"]}),
                    html.Div(f"BuyScore\u226570: {len(df[df['BuyScore'] >= 70])}", style={"color": C["buy"], "font-family": "monospace", "font-size": "12px", "padding": "3px 0"}),
                    html.Div(f"BuyScore 50-70: {len(df[(df['BuyScore'] >= 50) & (df['BuyScore'] < 70)])}", style={"color": "#4caf50", "font-family": "monospace", "font-size": "12px", "padding": "3px 0"}),
                ]),

                watchlist_box("Overbought  Z > 1", C["sell"], overbought),
            ]),

            # Legend
            html.Div(style={"backgroundColor": C["surface"], "border": f"1px solid {C['border']}", "border-radius": "6px", "padding": "12px 16px", "margin-bottom": "16px", "font-family": "monospace", "font-size": "11px", "color": C["muted"]}, children=[
                html.Span("Colour key:  ", style={"font-weight": "bold"}),
                html.Span("\u25a0 ", style={"color": C["buy"]}), html.Span("Oversold   ", style={"color": C["buy"]}),
                html.Span("\u25a0 ", style={"color": "#4caf50"}), html.Span("Mild disc.  ", style={"color": "#4caf50"}),
                html.Span("\u25a0 ", style={"color": C["neutral"]}), html.Span("Fair value  ", style={"color": C["neutral"]}),
                html.Span("\u25a0 ", style={"color": C["warn"]}), html.Span("Mild prem.  ", style={"color": C["warn"]}),
                html.Span("\u25a0 ", style={"color": C["sell"]}), html.Span("Overbought  ", style={"color": C["sell"]}),
                html.Span("  |  ZScore (63d)  RSI(14)  Bollinger %B  Dev% from MAs  52wk range  BuyScore composite", style={"margin-left": "16px"}),
            ]),

            # Table
            html.Div(style={"backgroundColor": C["surface"], "border": f"1px solid {C['border']}", "border-radius": "6px", "padding": "16px", "margin-bottom": "24px"}, children=[
                html.H3("Full Universe — click row for chart + health check", style={"color": C["text"], "font-family": "monospace", "margin": "0 0 12px"}),
                make_table(df),
            ]),

            # Chart
            html.Div(id="chart-wrap", style={"backgroundColor": C["surface"], "border": f"1px solid {C['border']}", "border-radius": "6px", "padding": "16px"}, children=[
                html.H3("Selected Stock", style={"color": C["text"], "font-family": "monospace", "margin": "0 0 12px"}),
                dcc.Graph(id="stock-chart", figure=go.Figure()),
            ]),

            # Health check panel
            html.Div(id="health-wrap", style={"backgroundColor": C["surface"], "border": f"1px solid {C['border']}", "border-radius": "6px", "padding": "16px", "margin-top": "16px"}, children=[
                health_check_panel(df.iloc[0].to_dict() if len(df) else {})
            ]),

            dcc.Store(id="df-store",  data=df.to_dict("records")),
            dcc.Store(id="raw-store", data={
                t: {k: float(v) for k, v in r.items() if v is not None and k != "Name"}
                for t, r in raw_data.items()
            }),
        ],
    )


# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────

def register_callbacks(app, df):

    @callback(
        Output("stock-chart", "figure"),
        Output("health-wrap", "children"),
        Input("heatmap-table", "derived_virtual_selected_rows"),
        Input("heatmap-table", "derived_virtual_data"),
    )
    def update_chart_and_health(rows, all_rows):
        if not rows or not all_rows:
            return go.Figure(), html.Div()
        row  = all_rows[rows[0]]
        tic  = row["Ticker"]
        name = row.get("Name", tic)
        df_r = fetch_ohlc(tic)
        if df_r is None:
            return go.Figure(), html.Div()
        m = calc_metrics(df_r) or {}
        m["Name"] = name
        return make_chart(tic, df_r, m), health_check_panel(m)

    @callback(
        Output("heatmap-table", "data"),
        Input("refresh-btn", "n_clicks"),
    )
    def refresh(n):
        if n == 0:
            raise dash.exceptions.PreventUpdate
        df2 = build_universe()
        return df2.to_dict("records")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Fetching NZX universe...")
    raw_data = {}
    for ticker, name in NZX_UNIVERSE:
        df = fetch_ohlc(ticker)
        m  = calc_metrics(df)
        if m:
            raw_data[ticker] = m
            print(f"  {ticker}: z={m['ZScore']:.2f} rsi={m['RSI']:.0f} buy={m['BuyScore']:.1f}")

    df = pd.DataFrame([
        {**{"Ticker": t, "Name": dict(NZX_UNIVERSE).get(t, t)}, **raw_data[t]}
        for t in raw_data
    ], columns=METRIC_COLS)

    print(f"\nLoaded {len(df)} stocks. Starting on http://0.0.0.0:8050")

    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.title = "NZX Mean Reversion"
    app.layout = lambda: serve_layout(df, raw_data)
    register_callbacks(app, df)
    app.run(host="0.0.0.0", port=8050, debug=False)
