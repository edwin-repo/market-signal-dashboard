#!/usr/bin/env python3
"""
Market Signal Dashboard — Alpha 🐺
Confluence: AAII Bull-Bear Spread + NAAIM + VIX (3-factor)
Macro Gate: Rates + DXY + Copper + HY Spread + Yield Curve + SPY > 200MA (6-factor)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path
import os

st.set_page_config(
    page_title="🐺 Market Signal Dashboard",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ────────────────────────────────────────────────────────────
AAII_SPREAD_THRESHOLD = -0.20
NAAIM_THRESHOLD       = 40
VIX_THRESHOLD         = 30
VIX_HIGH              = 40
VIX_EXTREME           = 50
FG_THRESHOLD          = 25
MACRO_GATE_MIN        = 4
ROC_DAYS              = 20
DATA_DIR = Path(__file__).parent.parent / 'backtest' / 'data'

# ── FRED Key ─────────────────────────────────────────────────────────────
try:
    FRED_KEY = st.secrets.get("FRED_API_KEY", "") or os.environ.get("FRED_API_KEY", "")
except Exception:
    FRED_KEY = os.environ.get("FRED_API_KEY", "")

# ── Signal Descriptions ───────────────────────────────────────────────────
SIGNAL_INFO = {
    "aaii": {
        "name": "AAII Bull-Bear Spread",
        "what": "Weekly survey of individual investors: % Bulls minus % Bears.",
        "low_is_good": True,
        "green_means": "Extreme pessimism → contrarian **buy signal**. When retail investors are most bearish, the market tends to rally.",
        "red_means": "Sentiment neutral or bullish → no edge. Signal requires bears to significantly outnumber bulls.",
        "threshold_explain": "Threshold: Spread < -20% (bears outnumber bulls by 20+ percentage points)",
        "backtest": "AAII Spread < -20%: **86% win rate at 12m, avg +18.9%** (377 signals, 1987–2026)",
        "update": "📅 Published every Thursday by AAII (aaii.com)",
        "levels": [
            ("< -30%", "🔴 Extreme fear — historically strongest buy zone"),
            ("-30% to -20%", "🟠 Deep pessimism — signal zone"),
            ("-20% to 0%", "🟡 Mildly bearish"),
            ("> 0%", "⚪ More bulls than bears — neutral/complacent"),
            ("> +30%", "🔵 Extreme euphoria — caution for longs"),
        ]
    },
    "naaim": {
        "name": "NAAIM Exposure Index",
        "what": "Average equity exposure of active fund managers (0 = fully out, 100 = fully invested, can go negative for short).",
        "low_is_good": True,
        "green_means": "Professional managers are mostly out of the market — they've capitulated. Historically this is when the best buying opportunities appear.",
        "red_means": "Managers are heavily invested → no contrarian signal. High readings (>80) indicate potential complacency.",
        "threshold_explain": "Threshold: NAAIM < 40 (managers significantly underweight equities)",
        "backtest": "NAAIM < 40: **98% win rate at 12m, avg +19.8%** (401 signals, 2006–2026)\nNAAIM < 25: **100% win rate at 12m, avg +23.3%** (112 signals)",
        "update": "📅 Published every Wednesday by NAAIM (naaim.org)",
        "levels": [
            ("< 0", "🔴 Managers are net short — extreme capitulation"),
            ("0 – 25", "🔴 Fully out/near-short — maximum fear"),
            ("25 – 40", "🟠 Heavily underweight — signal zone"),
            ("40 – 60", "🟡 Cautiously positioned"),
            ("60 – 80", "⚪ Normally invested"),
            ("> 80", "🔵 Heavily long — potential complacency"),
        ]
    },
    "vix": {
        "name": "VIX — CBOE Volatility Index",
        "what": "Market's expectation of 30-day S&P 500 volatility, derived from options prices. Known as the 'Fear Gauge'.",
        "low_is_good": False,
        "green_means": "Elevated fear/volatility → options prices spiked → historically a strong **mean-reversion buy signal**. Markets tend to recover sharply after VIX spikes.",
        "red_means": "Low volatility — no fear signal. VIX < 20 often indicates complacency.",
        "threshold_explain": "Thresholds: >30 (elevated), >40 (high conviction), >50 (extreme — rare but most powerful)",
        "backtest": "VIX > 30: **84% win at 3m, avg +8.7%** (254 signals)\nVIX > 40: **98% win at 3m, avg +16.5%** (54 signals)\nVIX > 50: **100% win rate every horizon** (19 signals)",
        "update": "📅 Live — updates every 15 seconds during market hours",
        "levels": [
            ("> 50", "🔴 Extreme fear — historically 100% win rate at all horizons"),
            ("40 – 50", "🔴 High fear — 98% win rate at 3m"),
            ("30 – 40", "🟠 Elevated fear — signal zone, 84% win at 3m"),
            ("20 – 30", "🟡 Moderate concern"),
            ("12 – 20", "⚪ Normal / complacent"),
            ("< 12", "🔵 Extreme complacency — caution"),
        ]
    },
    "fg": {
        "name": "CNN Fear & Greed Index",
        "what": "Composite of 7 market indicators: momentum, strength, breadth, put/call ratio, junk bond demand, market volatility, and safe-haven demand. Scale: 0 (Extreme Fear) → 100 (Extreme Greed).",
        "low_is_good": True,
        "green_means": "Market participants are extremely fearful → contrarian signal. When everyone is fearful, it's often time to buy.",
        "red_means": "No extreme fear signal. High readings (>75) indicate greed — potential caution for new longs.",
        "threshold_explain": "Threshold: F&G < 25 (Extreme Fear zone)",
        "backtest": "F&G < 25: **96% win rate at 3m, avg +8.5%** (56 signals — limited to ~1yr history)",
        "update": "📅 Live — updates daily (CNN Business)",
        "levels": [
            ("0 – 25", "🔴 Extreme Fear — buy signal zone"),
            ("25 – 45", "🟠 Fear"),
            ("45 – 55", "🟡 Neutral"),
            ("55 – 75", "⚪ Greed"),
            ("75 – 100", "🔵 Extreme Greed — caution"),
        ]
    },
}

MACRO_INFO = {
    "macro_rates": {
        "label": "10Y Yield ↓",
        "full": "10-Year Treasury Yield — Falling",
        "why": "Falling yields → lower discount rate → stocks more attractive (future earnings worth more today). Also signals Fed may be easing or growth slowing.",
        "bullish": "Yield is lower than 20 days ago",
        "bearish": "Yield is higher than 20 days ago — rising rates pressure valuations"
    },
    "macro_dxy": {
        "label": "DXY ↓",
        "full": "US Dollar Index — Falling",
        "why": "Weak dollar → better earnings for US multinationals (overseas revenue worth more in USD). Also signals risk-on appetite globally.",
        "bullish": "DXY lower than 20 days ago",
        "bearish": "DXY rising — strong dollar creates earnings headwinds"
    },
    "macro_copper": {
        "label": "Copper ↑",
        "full": "Copper Price — Rising",
        "why": "'Dr. Copper' is a leading economic indicator. Rising copper signals global industrial demand is healthy and the economy is expanding.",
        "bullish": "Copper higher than 20 days ago",
        "bearish": "Copper falling — growth concerns, industrial slowdown"
    },
    "macro_hy": {
        "label": "HY Spread ↓",
        "full": "High Yield Credit Spread — Tightening",
        "why": "HY spreads = gap between junk bond yields and Treasuries. Tightening spreads = credit markets are relaxed = companies can borrow cheaply = risk appetite healthy.",
        "bullish": "Spreads tightening (20d trend) — credit markets healthy",
        "bearish": "Spreads widening — credit stress, risk-off signal"
    },
    "macro_curve": {
        "label": "Curve >0",
        "full": "Yield Curve (2s10s) — Positive",
        "why": "Normal curve (10Y > 2Y) = banks profitable lending long, economy healthy. Inverted curve historically precedes recessions by 12–18 months.",
        "bullish": "2s10s > 0 — normal yield curve, no inversion",
        "bearish": "Inverted curve (<0) — recession warning signal"
    },
    "macro_breadth": {
        "label": "SPY >200MA",
        "full": "S&P 500 Above 200-Day Moving Average",
        "why": "SPY above its 200d MA = long-term uptrend intact. Buying dips (sentiment extremes) works best when the underlying trend is up. Buying into a structural downtrend is riskier.",
        "bullish": "SPY above 200d MA — trend is your friend",
        "bearish": "SPY below 200d MA — structural downtrend, higher risk for longs"
    },
}

# ── FRED API ─────────────────────────────────────────────────────────────
def fred_fetch(series_id, limit=60):
    if not FRED_KEY:
        return None
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={'series_id': series_id, 'api_key': FRED_KEY,
                    'file_type': 'json', 'limit': limit, 'sort_order': 'desc'},
            timeout=10
        )
        df = pd.DataFrame(r.json()['observations'])
        df['date']  = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df.dropna(subset=['value']).sort_values('date').reset_index(drop=True)
    except Exception:
        return None

# ── Data Fetchers ─────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_market():
    tickers = {'vix': '^VIX', 'spy': 'SPY', 'dxy': 'DX-Y.NYB',
               'copper': 'HG=F', 'yield10': '^TNX', 'hyg': 'HYG'}
    end   = datetime.today()
    start = end - timedelta(days=400)
    data  = {}
    for key, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if not df.empty:
                data[key] = df['Close'].dropna()
        except Exception:
            pass
    return data

@st.cache_data(ttl=3600)
def fetch_fred_data():
    result = {}
    for key, sid in {'curve_2s10s': 'T10Y2Y', 'hy_spread': 'BAMLH0A0HYM2',
                     'yield_2y': 'DGS2', 'yield_10y': 'DGS10'}.items():
        df = fred_fetch(sid)
        if df is not None and not df.empty:
            result[key] = df
    return result

@st.cache_data(ttl=86400)
def load_aaii():
    path = DATA_DIR / 'aaii.csv'
    if path.exists():
        df = pd.read_csv(path, parse_dates=['date'])
        df = df.dropna(subset=['bullish', 'bull_bear_spread']).sort_values('date')
        return df.iloc[-1]
    return None

@st.cache_data(ttl=86400)
def load_naaim():
    path = DATA_DIR / 'naaim.csv'
    if path.exists():
        df = pd.read_csv(path, parse_dates=['date'])
        df = df.dropna(subset=['naaim']).sort_values('date')
        return df.iloc[-1]
    return None

@st.cache_data(ttl=3600)
def fetch_fear_greed():
    try:
        r = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=10
        )
        data = r.json()
        return round(float(data['fear_and_greed']['score']), 1), data['fear_and_greed']['rating']
    except Exception:
        return None, None

# ── Signal Helpers ─────────────────────────────────────────────────────────
def to_bool(val):
    if val is None: return False
    try: return bool(val)
    except (ValueError, TypeError): return False

def roc(series, days=ROC_DAYS):
    if series is None or len(series) < days + 1:
        return None
    try:
        a = series.iloc[-1]
        b = series.iloc[-(days+1)]
        return float(a.item() if hasattr(a, 'item') else a) - float(b.item() if hasattr(b, 'item') else b)
    except Exception:
        return None

def latest(series):
    if series is None or len(series) == 0:
        return None
    try:
        v = series.iloc[-1]
        return float(v.item() if hasattr(v, 'item') else v)
    except Exception:
        return None

def spy_vs_200ma(spy_series):
    if spy_series is None or len(spy_series) < 200:
        return None, None, None
    try:
        ma  = spy_series.rolling(200).mean()
        p   = float(spy_series.iloc[-1].item() if hasattr(spy_series.iloc[-1], 'item') else spy_series.iloc[-1])
        m   = float(ma.iloc[-1].item() if hasattr(ma.iloc[-1], 'item') else ma.iloc[-1])
        return p, m, bool(p > m)
    except Exception:
        return None, None, None

def compute_signals(market, fred, aaii_row, naaim_row, aaii_override, naaim_override):
    sig = {}

    # AAII
    if aaii_override is not None:
        sig['aaii_spread'] = aaii_override / 100.0
        sig['aaii_date']   = 'Manual override'
        sig['aaii_bullish'] = None
    elif aaii_row is not None:
        sig['aaii_spread']  = float(aaii_row['bull_bear_spread'])
        sig['aaii_bullish'] = float(aaii_row['bullish']) * 100
        sig['aaii_bearish'] = float(aaii_row['bearish']) * 100
        sig['aaii_date']    = str(aaii_row['date'])[:10]
    else:
        sig['aaii_spread'] = None
        sig['aaii_date']   = 'No data'
    sig['aaii_fired'] = to_bool(sig['aaii_spread'] is not None and sig['aaii_spread'] < AAII_SPREAD_THRESHOLD)

    # NAAIM
    if naaim_override is not None:
        sig['naaim'] = naaim_override
        sig['naaim_date'] = 'Manual override'
    elif naaim_row is not None:
        sig['naaim']      = float(naaim_row['naaim'])
        sig['naaim_date'] = str(naaim_row['date'])[:10]
    else:
        sig['naaim'] = None
        sig['naaim_date'] = 'No data'
    sig['naaim_fired'] = to_bool(sig['naaim'] is not None and sig['naaim'] < NAAIM_THRESHOLD)

    # VIX
    sig['vix']          = latest(market.get('vix'))
    sig['vix_fired']    = to_bool(sig['vix'] is not None and sig['vix'] > VIX_THRESHOLD)
    sig['vix_high']     = to_bool(sig['vix'] is not None and sig['vix'] > VIX_HIGH)
    sig['vix_extreme']  = to_bool(sig['vix'] is not None and sig['vix'] > VIX_EXTREME)

    # Confluence
    sig['confluence'] = sum([sig['aaii_fired'], sig['naaim_fired'], sig['vix_fired']])

    # Macro: 10Y yield
    if 'yield_10y' in fred and fred['yield_10y'] is not None:
        ys = fred['yield_10y']['value']
        sig['yield_10y_now'] = latest(ys)
        sig['yield_roc']     = roc(ys) if len(ys) > ROC_DAYS else None
    else:
        y10 = market.get('yield10')
        sig['yield_10y_now'] = latest(y10)
        sig['yield_roc']     = roc(y10)
    sig['macro_rates'] = to_bool(sig.get('yield_roc') is not None and sig['yield_roc'] < 0)

    # Macro: DXY
    dxy = market.get('dxy')
    sig['dxy_now']   = latest(dxy)
    sig['dxy_roc']   = roc(dxy)
    sig['macro_dxy'] = to_bool(sig['dxy_roc'] is not None and sig['dxy_roc'] < 0)

    # Macro: Copper
    copper = market.get('copper')
    sig['copper_now']   = latest(copper)
    sig['copper_roc']   = roc(copper)
    sig['macro_copper'] = to_bool(sig['copper_roc'] is not None and sig['copper_roc'] > 0)

    # Macro: HY spread
    if 'hy_spread' in fred and fred['hy_spread'] is not None:
        hy = fred['hy_spread']['value']
        sig['hy_now']   = latest(hy)
        sig['hy_roc']   = roc(hy) if len(hy) > ROC_DAYS else None
        sig['macro_hy'] = to_bool(sig['hy_roc'] is not None and sig['hy_roc'] < 0)
        sig['hy_source'] = 'FRED'
    else:
        hyg = market.get('hyg')
        sig['hy_now']    = latest(hyg)
        sig['hy_roc']    = roc(hyg)
        sig['macro_hy']  = to_bool(sig['hy_roc'] is not None and sig['hy_roc'] > 0)
        sig['hy_source'] = 'HYG proxy'

    # Macro: Yield curve
    if 'curve_2s10s' in fred and fred['curve_2s10s'] is not None:
        cv = latest(fred['curve_2s10s']['value'])
        sig['curve_now']   = cv
        sig['macro_curve'] = to_bool(cv is not None and cv > 0)
    elif 'yield_2y' in fred and fred['yield_2y'] is not None and sig.get('yield_10y_now'):
        y2 = latest(fred['yield_2y']['value'])
        cv = (sig['yield_10y_now'] - y2) if y2 is not None else None
        sig['curve_now']   = cv
        sig['macro_curve'] = to_bool(cv is not None and cv > 0)
    else:
        sig['curve_now']   = None
        sig['macro_curve'] = False

    # Macro: SPY vs 200MA
    spy_price, spy_ma200, above_ma = spy_vs_200ma(market.get('spy'))
    sig['spy_price']     = spy_price
    sig['spy_ma200']     = spy_ma200
    sig['macro_breadth'] = to_bool(above_ma)

    macro_checks = ['macro_rates','macro_dxy','macro_copper','macro_hy','macro_curve','macro_breadth']
    sig['macro_score']   = sum(1 for k in macro_checks if sig.get(k))
    sig['macro_bullish'] = sig['macro_score'] >= MACRO_GATE_MIN

    return sig

# ── UI Components ─────────────────────────────────────────────────────────
def level_badge(value, levels, reverse=False):
    """Return color+label for a numeric value given level ranges."""
    for rng, label in levels:
        return label
    return "⚪ —"

def interpretation(val, info, fmt="{:.1f}"):
    """Return a human-readable interpretation of the current value."""
    if val is None:
        return "⚪ No data available"
    fired = (val < AAII_SPREAD_THRESHOLD * 100) if 'aaii' in str(info) else False
    return info['green_means'] if fired else info['red_means']

def big_signal_card(col, sig_key, title, current_str, threshold_str,
                    fired, info, note="", sub_stats=""):
    bg     = "#0f2d0f" if fired else "#1a0f0f"
    border = "#3fb950" if fired else "#f85149"
    icon   = "🟢" if fired else "🔴"
    status = "SIGNAL FIRING" if fired else "NOT YET"
    status_color = "#3fb950" if fired else "#f85149"

    with col:
        st.markdown(f"""
        <div style="background:{bg}; border:2px solid {border}; border-radius:10px; padding:18px; height:100%">
            <div style="display:flex; justify-content:space-between; align-items:flex-start">
                <div>
                    <div style="font-size:0.8em; color:#8b949e; text-transform:uppercase; letter-spacing:1px">{title}</div>
                    <div style="font-size:2em; font-weight:bold; margin:4px 0; color:{status_color}">{current_str}</div>
                    <div style="font-size:0.75em; color:#8b949e">{threshold_str}</div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:1.3em">{icon}</div>
                    <div style="font-size:0.7em; font-weight:bold; color:{status_color}; margin-top:2px">{status}</div>
                </div>
            </div>
            {f'<div style="margin-top:8px; font-size:0.72em; color:#8b949e">{note}</div>' if note else ''}
            {f'<div style="margin-top:6px; font-size:0.75em; color:#d29922">{sub_stats}</div>' if sub_stats else ''}
        </div>
        """, unsafe_allow_html=True)

        # Expandable info
        with st.expander("ℹ️ About this indicator"):
            st.markdown(f"**What it measures:** {info['what']}")
            st.markdown(f"**{threshold_str}**")
            st.markdown(f"**When signal fires (🟢):** {info['green_means']}")
            st.markdown(f"**When no signal (🔴):** {info['red_means']}")
            st.markdown(f"📊 **Backtest edge:** {info['backtest']}")
            st.markdown(f"{info['update']}")
            st.markdown("**Reading levels:**")
            for rng, label in info['levels']:
                st.markdown(f"- `{rng}` → {label}")

def macro_factor_row(key, fired, label, detail, info):
    bg     = "#0f2d0f" if fired else "#1a0f0f"
    border = "#3fb950" if fired else "#f85149"
    icon   = "✅" if fired else "❌"
    status = info['bullish'] if fired else info['bearish']

    st.markdown(f"""
    <div style="background:{bg}; border:1px solid {border}; border-radius:8px;
                padding:12px 16px; margin-bottom:8px; display:flex; align-items:flex-start; gap:12px">
        <div style="font-size:1.2em; min-width:24px">{icon}</div>
        <div style="flex:1">
            <div style="font-weight:bold; font-size:0.9em">{info['full']}</div>
            <div style="font-size:0.8em; color:#8b949e; margin-top:2px">{info['why']}</div>
            <div style="font-size:0.78em; color:{'#3fb950' if fired else '#f85149'}; margin-top:4px">
                → {status}
            </div>
            {f'<div style="font-size:0.75em; color:#8b949e; margin-top:2px">{detail}</div>' if detail else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        st.markdown("---")
        st.subheader("📅 Weekly Overrides")
        st.caption("AAII updates Thursdays · NAAIM updates Wednesdays")

        aaii_override  = None
        naaim_override = None

        if st.checkbox("Override AAII Spread"):
            aaii_override = st.number_input("AAII Bull-Bear Spread (%)",
                min_value=-100.0, max_value=100.0, value=-5.0, step=0.1,
                help="Bullish% minus Bearish%. Negative = more bears than bulls.")

        if st.checkbox("Override NAAIM"):
            naaim_override = st.number_input("NAAIM Exposure Index",
                min_value=-200.0, max_value=200.0, value=50.0, step=0.5,
                help="0=fully out, 100=fully invested")

        st.markdown("---")
        st.subheader("📡 Data Sources")
        st.caption("🟢 VIX / SPY / DXY / Copper — Yahoo Finance (live)")
        if FRED_KEY:
            st.caption("🟢 Yield curve / HY spread — FRED API (live)")
        else:
            st.caption("🟡 Yield curve / HY spread — proxy mode (add FRED key for exact data)")
        st.caption("🟡 AAII / NAAIM — last weekly CSV reading")

        st.markdown("---")
        if st.button("🔄 Refresh All Data"):
            st.cache_data.clear()
            st.rerun()

    # Fetch
    with st.spinner("Fetching live data..."):
        market    = fetch_market()
        fred      = fetch_fred_data()
        aaii_row  = load_aaii()
        naaim_row = load_naaim()
        fg_score, fg_rating = fetch_fear_greed()

    sig = compute_signals(market, fred, aaii_row, naaim_row, aaii_override, naaim_override)

    # ── Header ────────────────────────────────────────────────────────────
    st.title("🐺 Market Signal Dashboard")
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        st.caption(f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} · Backtest: 2010–2026 · 15yr dataset")
    with col_hdr2:
        st.caption("Strategy: Contrarian + Netto Macro Gate")

    st.markdown("---")

    # ── Strategy Explainer ────────────────────────────────────────────────
    with st.expander("📖 How this dashboard works", expanded=False):
        st.markdown("""
**Core idea:** Buy when fear is extreme + macro conditions support recovery.

**Step 1 — Confluence Score (0–3)**
Three sentiment indicators measure when investors are *maximally fearful*:
- **AAII Bull-Bear Spread < -20%** — retail investors more bearish than usual
- **NAAIM < 40** — professional fund managers have pulled out of equities
- **VIX > 30** — options market pricing in extreme fear/volatility

Score 2–3 = historically high-probability contrarian buy setup.

**Step 2 — Netto Macro Gate (0–6)**
Six macro factors check that conditions support a recovery (not just catching a falling knife):
- Yields falling, Dollar weakening, Copper rising, Credit spreads tightening, Yield curve positive, SPY in uptrend
- Need ≥ 4/6 = green light

**Step 3 — Standalone Confirmations**
Fear & Greed < 25 and Insider Cluster ≥ 3 add conviction but aren't required.

**Backtest results (2010–2026, 15yr):**
| Setup | N | 3m Win Rate | 12m Win Rate | 12m Avg Return |
|-------|---|-------------|--------------|----------------|
| Confluence ≥ 2 (raw) | 257 | 80% | 91% | +23.3% |
| **Confluence ≥ 2 + Macro Gate** | **43** | **91%** | **100%** | **+33.1%** |
| Confluence ≥ 3 | 25 | 96% | 96% | +18.5% |
| VIX > 40 | 54 | 98% | 100% | +44.2% |
        """)

    # ── Overall Banner ────────────────────────────────────────────────────
    conf     = sig['confluence']
    macro    = sig['macro_score']
    macro_ok = sig['macro_bullish']

    banners = {
        (3, True):  ("🚨", "#0f2d0f", "#3fb950", "HIGH CONVICTION SETUP — All 3 signals + Macro confirmed",
                     "Historically: 96%+ win rate at 3m/6m · Best risk/reward"),
        (3, False): ("⚡", "#1a2d0f", "#d29922", "STRONG SETUP — All 3 signals firing (macro not confirmed)",
                     "96% win rate at 3m raw · Consider waiting for macro gate"),
        (2, True):  ("✅", "#0f2a1a", "#3fb950", "SOLID SETUP — 2/3 signals + Macro Gate confirmed",
                     "91% win rate at 3m · 100% at 12m · Avg +33.1% over 12m"),
        (2, False): ("🟡", "#1a1a0f", "#d29922", "DEVELOPING — 2/3 signals (macro not confirmed)",
                     "80% win at 3m raw · Watch for macro confirmation"),
        (1, False): ("👀", "#0f0f1a", "#58a6ff", "WATCHING — 1/3 signals firing",
                     "Monitor weekly — NAAIM (Wed) + AAII (Thu)"),
    }
    icon, bg, border, title_text, sub_text = banners.get(
        (conf, macro_ok),
        banners.get((conf, False), ("😴", "#111", "#30363d", "NO SIGNAL — Market not at extreme", "0/3 signals firing"))
    )

    st.markdown(f"""
    <div style="background:{bg}; border:2px solid {border}; border-radius:12px; padding:20px 24px; margin-bottom:8px">
        <div style="font-size:1.5em; font-weight:bold">{icon} {title_text}</div>
        <div style="color:#8b949e; margin-top:6px; font-size:0.9em">{sub_text}</div>
        <div style="margin-top:14px; display:flex; gap:10px; flex-wrap:wrap">
            <span style="background:#21262d; border-radius:6px; padding:5px 12px; font-size:0.85em">
                Confluence: <b style="color:{'#3fb950' if conf >= 2 else '#f85149'}">{conf}/3</b>
            </span>
            <span style="background:#21262d; border-radius:6px; padding:5px 12px; font-size:0.85em">
                Macro Gate: <b style="color:{'#3fb950' if macro_ok else '#f85149'}">{macro}/6</b>
            </span>
            <span style="background:#21262d; border-radius:6px; padding:5px 12px; font-size:0.85em">
                VIX: <b style="color:{'#f85149' if sig['vix_fired'] else '#8b949e'}">{f"{sig['vix']:.1f}" if sig['vix'] else '—'}</b>
                {'🔥' if sig['vix_extreme'] else '⚡' if sig['vix_high'] else ''}
            </span>
            <span style="background:#21262d; border-radius:6px; padding:5px 12px; font-size:0.85em">
                NAAIM: <b style="color:{'#f85149' if sig['naaim_fired'] else '#8b949e'}">{f"{sig['naaim']:.0f}" if sig['naaim'] else '—'}</b>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 1: Confluence ─────────────────────────────────────────────
    st.subheader(f"📊 Confluence Score: {conf}/3")
    st.caption("All 3 have deep historical data (15+ years). Score ≥ 2 = actionable. Score = 3 = maximum conviction.")

    c1, c2, c3 = st.columns(3)

    aaii_val  = sig.get('aaii_spread')
    aaii_disp = f"{aaii_val*100:+.1f}%" if aaii_val is not None else "No data"
    aaii_sub  = f"Bulls: {sig.get('aaii_bullish', 0):.1f}%  Bears: {sig.get('aaii_bearish', 0):.1f}%" if sig.get('aaii_bullish') else ""
    big_signal_card(c1, "aaii", "AAII Bull-Bear Spread", aaii_disp,
                    "threshold: Spread < -20%", sig['aaii_fired'],
                    SIGNAL_INFO['aaii'],
                    note=f"as of {sig['aaii_date']} · {aaii_sub}",
                    sub_stats="📊 Spread < -20%: 86% win @12m, avg +18.9%")

    naaim_val  = sig.get('naaim')
    naaim_disp = f"{naaim_val:.1f}" if naaim_val is not None else "No data"
    naaim_sub  = ("🔴 Extreme — managers nearly flat/short" if naaim_val and naaim_val < 25
                  else "🟠 Heavily underweight" if naaim_val and naaim_val < 40
                  else "⚪ Normal allocation" if naaim_val else "")
    big_signal_card(c2, "naaim", "NAAIM Exposure Index", naaim_disp,
                    "threshold: < 40 | extreme: < 25",
                    sig['naaim_fired'], SIGNAL_INFO['naaim'],
                    note=f"as of {sig['naaim_date']} · {naaim_sub}",
                    sub_stats="📊 <40: 98% win @12m · <25: 100% win @12m")

    vix_val  = sig.get('vix')
    vix_disp = f"{vix_val:.1f}" if vix_val is not None else "No data"
    vix_sub  = ("🔴 EXTREME FEAR" if sig['vix_extreme']
                else "⚡ HIGH FEAR" if sig['vix_high']
                else "🟠 Elevated" if sig['vix_fired']
                else "⚪ Low/Normal")
    big_signal_card(c3, "vix", "VIX Fear Index", vix_disp,
                    "threshold: >30 | high: >40 | extreme: >50",
                    sig['vix_fired'], SIGNAL_INFO['vix'],
                    note=vix_sub,
                    sub_stats="📊 >40: 98% win @3m · >50: 100% win all horizons")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 2: Macro Gate ─────────────────────────────────────────────
    macro_color = "#3fb950" if macro_ok else "#f85149"
    st.subheader(f"🔭 Macro Gate: {macro}/6 — {'✅ GREEN LIGHT' if macro_ok else '❌ Not confirmed (need ≥4)'}")
    st.caption("Netto macro framework: confirms market conditions support a recovery. Improves confluence win rate from 80% → 91% at 3m.")

    with st.expander("ℹ️ Why the macro gate matters"):
        st.markdown("""
The macro gate answers: *"Even though sentiment is extreme, are conditions right for a recovery?"*

Without a macro filter, you can catch sentiment extremes during structural downtrends (like buying VIX spikes in 2022 while rates were still surging). The macro gate helps filter these out.

**Impact on backtest results:**
- Confluence ≥ 2 raw: 80% win rate at 3m (257 signals)
- Confluence ≥ 2 + macro gate: **91% win rate at 3m** (43 signals, also 100% win at 12m)

The gate reduces signal frequency by 6x but nearly eliminates losing trades.
        """)

    m1, m2 = st.columns(2)

    with m1:
        y10_now = sig.get('yield_10y_now')
        y10_roc = sig.get('yield_roc')
        detail  = f"Current: {y10_now:.2f}% · 20d change: {'↓' if y10_roc and y10_roc < 0 else '↑'}{abs(y10_roc):.3f}%" if y10_now else "No data"
        macro_factor_row("macro_rates",   sig['macro_rates'],  "macro_rates",  detail, MACRO_INFO['macro_rates'])

        dxy_now = sig.get('dxy_now')
        dxy_roc = sig.get('dxy_roc')
        detail  = f"Current: {dxy_now:.2f} · 20d change: {'↓' if dxy_roc and dxy_roc < 0 else '↑'}{abs(dxy_roc):.2f}" if dxy_now else "No data"
        macro_factor_row("macro_dxy",     sig['macro_dxy'],    "macro_dxy",    detail, MACRO_INFO['macro_dxy'])

        cop_now = sig.get('copper_now')
        cop_roc = sig.get('copper_roc')
        detail  = f"Current: ${cop_now:.3f}/lb · 20d change: {'↑' if cop_roc and cop_roc > 0 else '↓'}{abs(cop_roc):.3f}" if cop_now else "No data"
        macro_factor_row("macro_copper",  sig['macro_copper'], "macro_copper", detail, MACRO_INFO['macro_copper'])

    with m2:
        hy_now    = sig.get('hy_now')
        hy_roc    = sig.get('hy_roc')
        hy_src    = sig.get('hy_source', 'proxy')
        hy_unit   = '%' if hy_src == 'FRED' else ''
        detail    = f"Current: {hy_now:.2f}{hy_unit} ({hy_src}) · 20d change: {'↓' if hy_roc and hy_roc < 0 else '↑'}{abs(hy_roc):.3f}" if hy_now else "No data"
        macro_factor_row("macro_hy",      sig['macro_hy'],     "macro_hy",     detail, MACRO_INFO['macro_hy'])

        curve_now = sig.get('curve_now')
        detail    = f"2s10s spread: {curve_now:+.2f}% {'(normal curve)' if curve_now and curve_now > 0 else '(INVERTED — recession warning)' if curve_now is not None else ''}" if curve_now is not None else "No data (add FRED key)"
        macro_factor_row("macro_curve",   sig['macro_curve'],  "macro_curve",  detail, MACRO_INFO['macro_curve'])

        spy_p   = sig.get('spy_price')
        spy_ma  = sig.get('spy_ma200')
        pct_diff = ((spy_p - spy_ma) / spy_ma * 100) if spy_p and spy_ma else None
        detail   = f"SPY: ${spy_p:.0f} · 200MA: ${spy_ma:.0f} · {pct_diff:+.1f}% {'above' if pct_diff and pct_diff > 0 else 'below'}" if spy_p else "No data"
        macro_factor_row("macro_breadth", sig['macro_breadth'],"macro_breadth",detail, MACRO_INFO['macro_breadth'])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 3: Standalone ─────────────────────────────────────────────
    st.subheader("📌 Standalone Confirmation Signals")
    st.caption("Not part of the confluence score — use as bonus conviction when confluence ≥ 2")

    s1, s2 = st.columns(2)
    fg_fired = to_bool(fg_score is not None and fg_score < FG_THRESHOLD)
    fg_str   = f"{fg_score:.0f} — {fg_rating.title()}" if fg_score else "Unavailable"
    big_signal_card(s1, "fg", "CNN Fear & Greed Index", fg_str,
                    "threshold: < 25 (Extreme Fear)", fg_fired,
                    SIGNAL_INFO['fg'], note="Live daily · 0 = Extreme Fear, 100 = Extreme Greed",
                    sub_stats="📊 <25: 96% win @3m (limited history)")

    with s2:
        st.markdown("""
        <div style="background:#1a0f0f; border:2px solid #30363d; border-radius:10px; padding:18px; height:100%">
            <div style="font-size:0.8em; color:#8b949e; text-transform:uppercase; letter-spacing:1px">Insider Cluster</div>
            <div style="font-size:1.5em; font-weight:bold; margin:4px 0; color:#8b949e">Manual Check</div>
            <div style="font-size:0.75em; color:#8b949e">threshold: ≥ 3 companies buying in 10 days</div>
            <div style="margin-top:10px; font-size:0.8em; color:#d29922">
                📊 Cluster ≥ 3: 63–67% win rate (short-term; longer-term data limited)
            </div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("ℹ️ About Insider Cluster"):
            st.markdown("""
**What it measures:** When 3+ different company executives/directors buy their own stock within 10 days — "cluster buying" is a strong signal that insiders believe their stock is cheap.

**Why it matters:** Corporate insiders know their business best. When multiple insiders across different companies buy simultaneously, it suggests broad insider confidence at these prices.

**Where to check:** [openinsider.com](https://openinsider.com) → filter by "Cluster Buys" or look for multiple CEO/CFO purchases on the same day.

**What to look for:**
- Multiple different companies (not the same company)
- Officers (CEO, CFO, COO) > Directors (stronger signal)
- Larger $ amounts = stronger conviction
            """)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 4: Backtest Reference ─────────────────────────────────────
    with st.expander("📋 Full Backtest Reference Table (2010–2026)"):
        st.markdown("""
| Signal | N signals | 1m WR | 3m WR | 3m Avg | 6m WR | 12m WR | 12m Avg |
|--------|-----------|-------|-------|--------|-------|--------|---------|
| AAII Bulls < 20% | 94 | 52% | 61% | +1.7% | 67% | 77% | +11.9% |
| AAII Spread < -20% | 377 | 74% | 79% | +5.6% | 79% | 86% | **+18.9%** |
| NAAIM < 25 | 112 | 77% | 94% | +9.3% | 93% | **100%** | **+23.3%** |
| NAAIM < 40 | 401 | 73% | 81% | +6.1% | 86% | **98%** | +19.8% |
| VIX > 30 | 254 | 77% | 84% | +8.7% | 89% | 90% | +25.2% |
| VIX > 40 | 54 | 91% | **98%** | +16.5% | **100%** | **100%** | **+44.2%** |
| VIX > 50 | 19 | 95% | **100%** | +23.7% | **100%** | **100%** | **+57.6%** |
| Fear & Greed < 25 | 56 | 64% | 96% | +8.5% | 100% | n/a | n/a |
| **Confluence ≥ 2 (raw)** | **257** | 77% | 80% | +7.5% | 83% | 91% | +23.3% |
| **Confluence ≥ 2 + Macro** | **43** | 88% | **91%** | **+9.4%** | **93%** | **100%** | **+33.1%** |
| Confluence ≥ 3 (raw) | 25 | 92% | 96% | +7.5% | 96% | 96% | +18.5% |
        """)
        st.caption("WR = Win Rate (% of signals that resulted in positive forward returns). Source: 15yr backtest, S&P 500, 2010–2026.")

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption("🐺 Alpha · For research purposes only, not financial advice · AAII (1987–2026) · NAAIM (2006–2026) · Market data via Yahoo Finance & FRED")

if __name__ == "__main__":
    main()
