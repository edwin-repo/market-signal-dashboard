#!/usr/bin/env python3
"""
Market Signal Dashboard — Alpha 🐺
Core Sentiment: AAII Bull-Bear Spread + Put/Call + NAAIM + VIX (4-factor)
COT Positioning: S&P 500 + 10Y T-Note + Gold + Crude Oil + USD Index (5-factor)
Macro Gate: Rates + DXY + Copper + HY Spread + Yield Curve + SPY > 200MA (6-factor)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import os

st.set_page_config(
    page_title="🐺 Market Signal Dashboard",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="collapsed"
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
DATA_DIR = Path(__file__).parent / 'data'

# ── Backtest Stats (pre-computed from backtesting engine) ─────────────
# Each entry: best_horizon, win_rate, avg_return, n_signals, avg_loss (for EV)
INDICATOR_STATS = {
    'vix_extreme': {
        'label': 'VIX > 50', 'horizon': '3m',
        'win_rate': 1.00, 'avg_return': 0.237, 'avg_loss': 0.0, 'n': 19,
    },
    'vix_high': {
        'label': 'VIX > 40', 'horizon': '3m',
        'win_rate': 0.98, 'avg_return': 0.165, 'avg_loss': -0.05, 'n': 54,
    },
    'vix_fired': {
        'label': 'VIX > 30', 'horizon': '3m',
        'win_rate': 0.84, 'avg_return': 0.087, 'avg_loss': -0.08, 'n': 254,
    },
    'aaii_fired': {
        'label': 'AAII Spread < -20%', 'horizon': '12m',
        'win_rate': 0.86, 'avg_return': 0.189, 'avg_loss': -0.06, 'n': 377,
    },
    'naaim_fired': {
        'label': 'NAAIM < 40', 'horizon': '12m',
        'win_rate': 0.98, 'avg_return': 0.198, 'avg_loss': -0.03, 'n': 401,
    },
    'naaim_extreme': {
        'label': 'NAAIM < 25', 'horizon': '12m',
        'win_rate': 1.00, 'avg_return': 0.233, 'avg_loss': 0.0, 'n': 112,
    },
    'pc_10d_fired': {
        'label': 'Put/Call 10d > 0.70', 'horizon': '12m',
        'win_rate': 0.83, 'avg_return': 0.110, 'avg_loss': -0.05, 'n': 72,
    },
    'cot_tnote10_fired': {
        'label': '10Y T-Note COT → SPY', 'horizon': '3m', 'ticker': 'SPY',
        'win_rate': 0.89, 'avg_return': 0.061, 'avg_loss': -0.03, 'n': 27,
        'note': 'No edge on TLT — works as SPY macro signal only',
    },
    'cot_sp500_fired': {
        'label': 'S&P 500 COT → SPY', 'horizon': '12m', 'ticker': 'SPY',
        'win_rate': 0.833, 'avg_return': 0.148, 'avg_loss': -0.125, 'n': 30,
    },
    'cot_gold_fired': {
        'label': 'Gold COT → GLD', 'horizon': '6m', 'ticker': 'GLD',
        'win_rate': 0.722, 'avg_return': 0.045, 'avg_loss': -0.087, 'n': 36,
    },
    'cot_crude_fired': {
        'label': 'Crude Oil COT → USO', 'horizon': '12m', 'ticker': 'USO',
        'win_rate': 0.731, 'avg_return': 0.096, 'avg_loss': -0.168, 'n': 26,
    },
    'cot_usdx_fired': {
        'label': 'USD Index COT → EFA', 'horizon': '6m', 'ticker': 'EFA',
        'win_rate': 0.816, 'avg_return': 0.061, 'avg_loss': -0.086, 'n': 38,
    },
    'confluence_macro': {
        'label': 'Confluence ≥ 2 + Macro Confirm', 'horizon': '12m',
        'win_rate': 1.00, 'avg_return': 0.331, 'avg_loss': 0.0, 'n': 43,
    },
}

# ── Action Tier Config ─────────────────────────────────────────────────
ACTION_TIERS = {
    'MAX':    {'icon': '💎', 'color': '#7bf2da', 'bg': 'rgba(123,242,218,0.06)', 'border': 'rgba(123,242,218,0.25)', 'text': 'MAXIMUM CONVICTION', 'action': 'Historically never lost — strongest possible setup'},
    'HIGH':   {'icon': '🟢', 'color': '#3fb950', 'bg': 'rgba(63,185,80,0.06)', 'border': 'rgba(63,185,80,0.25)', 'text': 'HIGH CONVICTION BUY', 'action': 'Confluence ≥ 2 or VIX > 40 — size according to macro context'},
    'STRONG': {'icon': '🟡', 'color': '#e3b341', 'bg': 'rgba(227,179,65,0.06)', 'border': 'rgba(227,179,65,0.25)', 'text': 'STRONG SETUP', 'action': 'Consider building a position'},
    'ACTIVE': {'icon': '🔵', 'color': '#58a6ff', 'bg': 'rgba(88,166,255,0.06)', 'border': 'rgba(88,166,255,0.25)', 'text': 'SIGNAL ACTIVE', 'action': 'Monitor closely — individual signal firing'},
    'WATCH':  {'icon': '👀', 'color': '#8888a0', 'bg': 'rgba(136,136,160,0.04)', 'border': 'rgba(136,136,160,0.15)', 'text': 'WARMING UP', 'action': 'Indicators approaching signal zone — prepare watchlist'},
    'NONE':   {'icon': '😴', 'color': '#44445a', 'bg': 'rgba(68,68,90,0.04)', 'border': 'rgba(68,68,90,0.15)', 'text': 'NO SIGNAL', 'action': 'Market not at extremes — patience'},
}

def format_ev(indicator_key, progress=0.0):
    """Format EV string for display on card when indicator is warming/firing."""
    stats = INDICATOR_STATS.get(indicator_key)
    if not stats or progress < 0.66:
        return ""
    wr = stats['win_rate']
    avg = stats['avg_return']
    h = stats['horizon']
    n = stats['n']
    return f"💰 EV: {avg*100:+.1f}% over {h} ({wr*100:.0f}% win rate, {n} signals)"

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

MEISLER_INFO = {
    "putcall": {
        "name": "CBOE Equity Put/Call Ratio",
        "what": "Ratio of put volume to call volume on equity options. 10-day and 30-day moving averages smooth noise. Tracks what traders are *doing* (buying puts = fear).",
        "low_is_good": False,
        "green_means": "Elevated put/call ratio → traders are hedging/bearish → contrarian buy signal. Options data shows real behavior, not just opinions.",
        "red_means": "Low put/call ratio → traders complacent, buying calls → no fear signal.",
        "threshold_explain": "Thresholds: 10d MA > 0.70 (elevated fear), 30d MA > 0.65 (sustained fear)",
        "backtest": "PC 10d MA > 0.70: **83% win @12m, avg +11.0%** (72 signals, 2003–2026)\nPC 30d MA > 0.65: **82% win @12m, avg +10.9%** (99 signals)",
        "update": "📅 Updated daily by CBOE",
        "levels": [
            ("> 0.85 (10d MA)", "🔴 Extreme fear — heavy put buying"),
            ("0.75 – 0.85", "🟠 Elevated fear — signal zone"),
            ("0.65 – 0.75", "🟡 Mildly bearish"),
            ("0.50 – 0.65", "⚪ Neutral / balanced"),
            ("< 0.50", "🔵 Call-heavy — complacency / giddiness"),
        ]
    },
    "aaii_meisler": {
        "name": "AAII Sentiment Combo",
        "what": "Watches for specific extremes: Bulls dropping into the 20s% AND Bears rising above 50%. AAII is most valuable when both thresholds align.",
        "low_is_good": True,
        "green_means": "Bulls < 25% AND Bears > 50% → extreme pessimism combo flagging a potential bottom.",
        "red_means": "AAII not at extreme levels — bulls not low enough or bears not high enough.",
        "threshold_explain": "Threshold: Bulls < 25% + Bears > 50%",
        "backtest": "Bulls < 25%: **83% win @12m, avg +17.3%** (115 signals)\nBears > 50%: 65% win @12m (less reliable alone)\nCombo: 68% win @12m (rarer but powerful when confirmed by options data)",
        "update": "📅 Published every Thursday by AAII (aaii.com)",
        "levels": [
            ("Bulls < 20% + Bears > 55%", "🔴 Extreme — historically strongest"),
            ("Bulls < 25% + Bears > 50%", "🟠 Signal zone"),
            ("Bulls < 30% + Bears > 40%", "🟡 Getting pessimistic"),
            ("Bulls > 30% or Bears < 40%", "⚪ Not at extreme"),
        ]
    },
    "sector_igv": {
        "name": "IGV (Software) vs SPY",
        "what": "Relative strength of software stocks (IGV ETF) vs S&P 500. Compares sector performance to the index to find where investors are panicking.",
        "low_is_good": True,
        "green_means": "Software stocks getting hammered relative to SPY → sector capitulation → potential bounce. When a sector underperforms this much, panic selling may be overdone.",
        "red_means": "IGV performing normally vs SPY — no sector-specific panic.",
        "threshold_explain": "Signal: IGV/SPY ratio at bottom 10th percentile of its 60-day range",
        "backtest": "IGV/SPY at 10th pctile: **82% win @12m, avg +10.6%** (66 signals)",
        "update": "📅 Live — updates with market prices",
        "levels": [
            ("< 5th pctile (60d)", "🔴 Severe breakdown — sector panic"),
            ("5th – 10th pctile", "🟠 Significant underperformance"),
            ("10th – 25th pctile", "🟡 Underperforming"),
            ("> 25th pctile", "⚪ Normal range"),
        ]
    },
    "sector_kbe": {
        "name": "KBE (Banks) vs SPY",
        "what": "Relative strength of bank stocks (KBE ETF) vs S&P 500. Tracks the bank index for support levels to see if investors have reached a 'clean out' moment.",
        "low_is_good": True,
        "green_means": "Banks getting crushed vs SPY → financial sector panic → potential bottom. The 'death by a thousand cuts' may be ending.",
        "red_means": "Banks performing normally vs SPY — no financial sector distress signal.",
        "threshold_explain": "Signal: KBE/SPY ratio at bottom 10th percentile of its 60-day range",
        "backtest": "KBE/SPY at 10th pctile: **80% win @12m, avg +11.1%** (101 signals)",
        "update": "📅 Live — updates with market prices",
        "levels": [
            ("< 5th pctile (60d)", "🔴 Financial panic — sector washout"),
            ("5th – 10th pctile", "🟠 Significant stress"),
            ("10th – 25th pctile", "🟡 Underperforming"),
            ("> 25th pctile", "⚪ Normal range"),
        ]
    },
}

COT_INFO = {
    "cot_sp500": {
        "name": "S&P 500 E-mini — COT Positioning",
        "what": "Large speculator net positioning in S&P 500 E-mini futures (CFTC weekly). Shows how hedge funds and managed money are positioned.",
        "low_is_good": True,
        "green_means": "Speculators heavily net-short (capitulation) → contrarian buy signal. When futures traders are most bearish, reversals tend to follow.",
        "red_means": "Speculators neutral or net-long → no contrarian edge.",
        "threshold_explain": "Buy signal: Spec Net < 5th percentile of 3-year rolling window",
        "backtest": "Spec Net < 5th pctile: **77% win @1m, avg +2.2%** · 12m avg +14.8% (30 signals, 2006–2026)",
        "update": "📅 Weekly (Tuesday) from CFTC Commitments of Traders report",
        "levels": [
            ("< 5th pctile", "🔴 Extreme bearish positioning — contrarian buy zone"),
            ("5th – 20th pctile", "🟠 Bearish positioning — getting interesting"),
            ("20th – 80th pctile", "⚪ Normal range"),
            ("> 80th pctile", "🔵 Extremely long — potential complacency"),
        ]
    },
    "cot_vix": {
        "name": "VIX Futures — COT Positioning",
        "what": "Large speculator net positioning in VIX futures. Shows whether traders are betting on more or less volatility ahead.",
        "low_is_good": True,
        "green_means": "Speculators heavily short VIX (complacent about volatility) → vulnerable to spike. Can confirm VIX cash signals.",
        "red_means": "Positioning neutral → no extreme signal.",
        "threshold_explain": "Signal: Spec Net < 10th percentile of 3-year window",
        "backtest": "VIX COT signals included for completeness — strongest alpha comes from VIX cash level (>30)",
        "update": "📅 Weekly (Tuesday) from CFTC",
        "levels": [
            ("< 10th pctile", "🔴 Heavily short vol — vulnerable to squeeze"),
            ("10th – 30th pctile", "🟠 Moderately short"),
            ("30th – 70th pctile", "⚪ Normal range"),
            ("> 70th pctile", "🔵 Long vol — already hedged"),
        ]
    },
    "cot_tnote": {
        "name": "10Y T-Note — COT Positioning",
        "what": "Commercial hedger and speculator positioning in 10-Year Treasury Note futures. The strongest COT signal for equities.",
        "low_is_good": True,
        "green_means": "Commercials extremely net-short (or specs extremely net-long) → flight-to-safety trade is crowded → equities tend to rally. **Best COT signal found.**",
        "red_means": "Bond positioning neutral → no signal for equities.",
        "threshold_explain": "Buy signal: Comm Net < 10th pctile (89% WR) or Spec Net > 95th pctile (87% WR)",
        "backtest": "Comm Net < 10th pctile: **89% win @3m, avg +6.1%**, 12m +19.6% (27 signals)\nSpec Net > 95th pctile: **87% win @3m, avg +5.4%** (23 signals)",
        "update": "📅 Weekly (Tuesday) from CFTC",
        "levels": [
            ("Comm < 5th pctile", "🔴 Extreme — historically strongest equity buy signal"),
            ("Comm 5th–10th pctile", "🟠 Signal zone — bonds crowded"),
            ("Comm 10th–50th pctile", "🟡 Getting interesting"),
            ("Comm > 50th pctile", "⚪ Normal range"),
        ]
    },
    "cot_gold": {
        "name": "Gold — COT Positioning",
        "what": "Large speculator net positioning in gold futures. Gold specs at extremes can signal broader risk sentiment shifts.",
        "low_is_good": True,
        "green_means": "Speculators heavily short gold → capitulation in safe-haven trade → often coincides with equity bottoms. Slow-burn but powerful at 12m.",
        "red_means": "Gold positioning neutral → no signal.",
        "threshold_explain": "Buy signal: Spec Net < 10th percentile of 3-year window",
        "backtest": "Spec Net < 10th pctile: **97% win @12m**, avg +15.9% (36 signals)",
        "update": "📅 Weekly (Tuesday) from CFTC",
        "levels": [
            ("< 5th pctile", "🔴 Extreme bearish on gold — equity buy zone"),
            ("5th – 10th pctile", "🟠 Signal zone"),
            ("10th – 50th pctile", "🟡 Below average"),
            ("> 50th pctile", "⚪ Normal range"),
        ]
    },
    "cot_crude": {
        "name": "Crude Oil WTI — COT Positioning",
        "what": "Large speculator net positioning in WTI crude oil futures. Extreme bearish crude positioning often aligns with growth fear capitulation.",
        "low_is_good": True,
        "green_means": "Speculators heavily short crude → growth fears peaked → equities tend to recover. Strong 12m signal.",
        "red_means": "Crude positioning neutral → no signal.",
        "threshold_explain": "Buy signal: Spec Net < 5th percentile of 3-year window",
        "backtest": "Spec Net < 5th pctile: 12m avg **+17.1%** (31 signals, 2006–2026)",
        "update": "📅 Weekly (Tuesday) from CFTC",
        "levels": [
            ("< 5th pctile", "🔴 Extreme bearish — growth capitulation"),
            ("5th – 15th pctile", "🟠 Heavily bearish"),
            ("15th – 50th pctile", "🟡 Below average"),
            ("> 50th pctile", "⚪ Normal range"),
        ]
    },
    "cot_usd": {
        "name": "US Dollar Index — COT Positioning",
        "what": "Large speculator net positioning in USD Index futures. Dollar positioning extremes have strong equity implications.",
        "low_is_good": True,
        "green_means": "Speculators heavily short USD → weak dollar expected → bullish for equities (better multinational earnings, risk-on). Also a caution signal when specs are extremely long USD.",
        "red_means": "USD positioning neutral → no directional signal.",
        "threshold_explain": "Buy signal: Spec Net < 10th pctile (bullish equities) · Caution: Spec Net > 90th pctile (bearish equities)",
        "backtest": "Spec < 10th pctile: **85-88% win @3m, avg +4.3-5.0%** (41 signals)\n⚠️ Spec > 90th pctile: 3m avg **-1.2%** — one of few bearish signals",
        "update": "📅 Weekly (Tuesday) from CFTC",
        "levels": [
            ("< 10th pctile", "🔴 Specs bearish USD — bullish equities"),
            ("10th – 30th pctile", "🟠 Moderately bearish USD"),
            ("30th – 70th pctile", "⚪ Normal range"),
            ("70th – 90th pctile", "🟡 Specs bullish USD — headwind forming"),
            ("> 90th pctile", "🔴 Extreme long USD — ⚠️ equity caution signal"),
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
                # Flatten MultiIndex columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                # Use intraday High for VIX to capture spikes that close lower
                col = 'High' if key == 'vix' else 'Close'
                series = df[col].dropna()
                # Drop zero/placeholder rows (e.g. today before market open)
                series = series[series > 0]
                data[key] = series
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

@st.cache_data(ttl=86400)  # 24h cache — AAII publishes Thursdays
def load_aaii():
    """Load from bundled CSV (AAII blocks scrapers). Update CSV manually each Thursday."""
    path = DATA_DIR / 'aaii.csv'
    if path.exists():
        df = pd.read_csv(path, parse_dates=['date'])
        df = df.dropna(subset=['bullish', 'bull_bear_spread']).sort_values('date')
        row = df.iloc[-1].copy()
        row['source'] = 'csv'
        return row
    return None

@st.cache_data(ttl=43200)  # 12h cache — NAAIM publishes Wednesdays
def load_naaim():
    """Try live scrape from naaim.org first, fall back to bundled CSV."""
    try:
        r = requests.get(
            'https://www.naaim.org/resources/naaim-exposure-index/',
            headers={'User-Agent': 'Mozilla/5.0'}, timeout=10
        )
        if r.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.text, 'html.parser')
            tables = soup.find_all('table')
            if tables:
                rows = tables[0].find_all('tr')
                for row in rows[1:]:  # skip header
                    cells = [td.text.strip() for td in row.find_all(['td', 'th'])]
                    if len(cells) >= 2 and cells[0] and cells[1]:
                        try:
                            date = pd.to_datetime(cells[0], format='%m/%d/%Y')
                            naaim_val = float(cells[1])
                            return pd.Series({'date': date, 'naaim': naaim_val, 'source': 'live'})
                        except (ValueError, TypeError):
                            continue
    except Exception:
        pass
    # Fallback: bundled CSV
    path = DATA_DIR / 'naaim.csv'
    if path.exists():
        df = pd.read_csv(path, parse_dates=['date'])
        df = df.dropna(subset=['naaim']).sort_values('date')
        row = df.iloc[-1].copy()
        row['source'] = 'csv'
        return row
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

def refresh_naaim_csv():
    """Scrape latest NAAIM data and update CSV."""
    try:
        from bs4 import BeautifulSoup
        r = requests.get('https://www.naaim.org/resources/naaim-exposure-index/',
                          headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            tables = soup.find_all('table')
            if tables:
                new_rows = []
                for row in tables[0].find_all('tr')[1:]:
                    cells = [td.text.strip() for td in row.find_all(['td', 'th'])]
                    if len(cells) >= 2:
                        try:
                            date = pd.to_datetime(cells[0], format='%m/%d/%Y')
                            val = float(cells[1])
                            new_rows.append({'date': date, 'naaim': val})
                        except (ValueError, TypeError):
                            continue
                if new_rows:
                    new_df = pd.DataFrame(new_rows)
                    path = DATA_DIR / 'naaim.csv'
                    existing = pd.read_csv(path, parse_dates=['date']) if path.exists() else pd.DataFrame()
                    merged = pd.concat([existing, new_df]).drop_duplicates('date').sort_values('date')
                    merged.to_csv(path, index=False)
                    return True, f"NAAIM updated to {merged['date'].max().strftime('%Y-%m-%d')} ({len(new_rows)} new rows)"
    except Exception as e:
        return False, f"NAAIM refresh failed: {e}"
    return False, "NAAIM: no new data found"


def refresh_aaii_csv():
    """Try to scrape latest AAII sentiment from insights.aaii.com."""
    try:
        from bs4 import BeautifulSoup
        import re
        # Search for recent AAII articles on their Substack
        for slug in [
            'aaii-sentiment-survey-pessimism-pulls',
            'aaii-sentiment-survey-pessimism-spikes',
            'aaii-sentiment-survey-pessimism-rebounds',
            'aaii-sentiment-survey-bullish-sentiment',
            'aaii-sentiment-survey-optimism-retracts',
        ]:
            url = f'https://insights.aaii.com/p/{slug}'
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            if r.status_code == 200:
                text = BeautifulSoup(r.text, 'html.parser').get_text()[:5000]
                # Extract bullish/bearish/neutral percentages
                bull_m = re.search(r'[Bb]ullish.*?(?:to|at)\s+(\d+\.?\d*)%', text)
                bear_m = re.search(r'[Bb]earish.*?(?:to|at)\s+(\d+\.?\d*)%', text)
                neut_m = re.search(r'[Nn]eutral.*?(?:to|at)\s+(\d+\.?\d*)%', text)
                # Find date
                date_m = re.search(r'(?:March|February|January|April)\s+\d{1,2},?\s+2026', text)
                if bull_m and bear_m:
                    bull_val = float(bull_m.group(1)) / 100
                    bear_val = float(bear_m.group(1)) / 100
                    neut_val = float(neut_m.group(1)) / 100 if neut_m else 1 - bull_val - bear_val
                    spread = bull_val - bear_val
                    # Use latest Thursday as date estimate
                    if date_m:
                        survey_date = pd.to_datetime(date_m.group(0))
                    else:
                        survey_date = pd.Timestamp.today() - pd.DateOffset(days=pd.Timestamp.today().weekday() - 3)
                        if survey_date > pd.Timestamp.today():
                            survey_date -= pd.DateOffset(weeks=1)

                    new_row = pd.DataFrame([{
                        'date': survey_date,
                        'bullish': bull_val, 'neutral': neut_val,
                        'bearish': bear_val, 'bull_bear_spread': spread,
                    }])
                    path = DATA_DIR / 'aaii.csv'
                    existing = pd.read_csv(path, parse_dates=['date']) if path.exists() else pd.DataFrame()
                    merged = pd.concat([existing, new_row]).drop_duplicates('date').sort_values('date')
                    merged.to_csv(path, index=False)
                    return True, f"AAII updated: Bull {bull_val*100:.1f}% Bear {bear_val*100:.1f}% ({survey_date.strftime('%Y-%m-%d')})"
    except Exception as e:
        return False, f"AAII refresh failed: {e}"
    return False, "AAII: could not find latest data. Manual CSV update may be needed."


def refresh_putcall_csv():
    """Fetch CBOE equity put/call ratio from ycharts.com and append new rows to CSV."""
    try:
        from io import StringIO

        path = DATA_DIR / 'putcall.csv'
        existing = pd.read_csv(path, parse_dates=['date']) if path.exists() else pd.DataFrame()
        last_date = existing['date'].max() if not existing.empty else pd.Timestamp('2000-01-01')

        # Scrape ycharts — they show ~50 days of CBOE equity put/call ratio for free
        r = requests.get('https://ycharts.com/indicators/cboe_equity_put_call_ratio',
                          headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'},
                          timeout=15)
        if r.status_code != 200:
            return False, f"Put/Call: ycharts returned status {r.status_code}"

        tables = pd.read_html(StringIO(r.text))
        # Find tables with 'Date' and 'Value' columns (there are typically 2, each with ~25 rows)
        data_tables = [t for t in tables if 'Date' in t.columns and 'Value' in t.columns]
        if not data_tables:
            return False, "Put/Call: Could not find data table on ycharts"

        # Combine all data tables
        scraped = pd.concat(data_tables, ignore_index=True)
        scraped['date'] = pd.to_datetime(scraped['Date'])
        scraped['equity_pc_ratio'] = pd.to_numeric(scraped['Value'], errors='coerce')
        scraped = scraped[['date', 'equity_pc_ratio']].dropna()

        # Filter to only new dates
        new_rows = scraped[scraped['date'] > last_date].copy()
        if new_rows.empty:
            latest = scraped['date'].max().strftime('%Y-%m-%d')
            return True, f"Put/Call already up to date (latest: {latest})"

        # Add empty columns for total and index (ycharts only provides equity)
        new_rows['total_pc_ratio'] = ''
        new_rows['index_pc_ratio'] = ''
        new_rows = new_rows.sort_values('date')

        if not existing.empty:
            updated = pd.concat([existing, new_rows[['date', 'equity_pc_ratio', 'total_pc_ratio', 'index_pc_ratio']]], ignore_index=True)
        else:
            updated = new_rows[['date', 'equity_pc_ratio', 'total_pc_ratio', 'index_pc_ratio']]

        updated = updated.sort_values('date').drop_duplicates(subset='date', keep='last')
        updated.to_csv(path, index=False)
        n_new = len(new_rows)
        latest = new_rows['date'].max().strftime('%Y-%m-%d')
        return True, f"Put/Call updated: +{n_new} rows to {latest}"

    except Exception as e:
        return False, f"Put/Call refresh failed: {e}"


@st.cache_data(ttl=86400)
def load_putcall():
    """Load CBOE put/call ratio with moving averages."""
    path = DATA_DIR / 'putcall.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['pc_10d_ma'] = df['equity_pc_ratio'].rolling(10, min_periods=8).mean()
    df['pc_30d_ma'] = df['equity_pc_ratio'].rolling(30, min_periods=25).mean()
    return df

@st.cache_data(ttl=86400)
def load_cot_data():
    """Load all 37 COT contracts with rolling percentiles for all 3 trader types."""
    try:
        from cot_config import COT_CONTRACTS
        from backtest import load_cot
        data = {}
        for key in COT_CONTRACTS:
            df = load_cot(key)
            if not df.empty:
                data[key] = df
        return data
    except Exception:
        return {}

@st.cache_data(ttl=86400)  # 24h — monthly data, updates infrequently
def fetch_shiller_cape():
    """Fetch Shiller CAPE (P/E10) from multpl.com — 150+ years of history."""
    try:
        from io import StringIO
        r = requests.get('https://www.multpl.com/shiller-pe/table/by-month',
                          headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        df = pd.read_html(StringIO(r.text))[0]
        df['date'] = pd.to_datetime(df['Date'])
        df['cape'] = pd.to_numeric(df['Value'], errors='coerce')
        df = df.dropna(subset=['cape']).sort_values('date')
        return df[['date', 'cape']].reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=43200)  # 12h cache — forward PE updates daily
def fetch_forward_pe():
    """Fetch current S&P 500 forward P/E from WSJ."""
    try:
        from io import StringIO
        r = requests.get('https://www.wsj.com/market-data/stocks/peyields',
                          headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        if r.status_code == 200:
            tables = pd.read_html(StringIO(r.text))
            # Table 1 has S&P 500 with Estimate column (forward PE)
            for tbl in tables:
                flat_cols = ['_'.join(str(c) for c in col).strip() if isinstance(col, tuple) else str(col)
                             for col in tbl.columns]
                tbl.columns = flat_cols
                for idx, row in tbl.iterrows():
                    row_str = str(row.values)
                    if 'S&P 500' in row_str or 'S.P 500' in row_str:
                        # Find the Estimate column (forward PE)
                        for col in tbl.columns:
                            if 'Estimate' in col or 'estimate' in col:
                                val = pd.to_numeric(row[col], errors='coerce')
                                if pd.notna(val) and 5 < val < 50:
                                    return {'forward_pe': float(val), 'source': 'WSJ'}
                        # Fallback: trailing PE
                        for col in tbl.columns:
                            if 'P/E' in col or 'pe' in col.lower():
                                val = pd.to_numeric(row[col], errors='coerce')
                                if pd.notna(val) and 5 < val < 80:
                                    return {'trailing_pe': float(val), 'source': 'WSJ'}
    except Exception:
        pass
    return {}


@st.cache_data(ttl=86400)  # 24h cache — on-chain data updates daily
def fetch_mvrv():
    """Fetch BTC MVRV ratio from CoinMetrics free community API."""
    try:
        url = 'https://community-api.coinmetrics.io/v4/timeseries/asset-metrics'
        all_data = []
        start = '2011-01-01'
        while True:
            params = {
                'assets': 'btc',
                'metrics': 'CapMVRVCur',
                'start_time': start,
                'page_size': 10000,
                'frequency': '1d',
            }
            r = requests.get(url, params=params, timeout=15)
            data = r.json()
            rows = data.get('data', [])
            if not rows:
                break
            all_data.extend(rows)
            if data.get('next_page_url'):
                start = rows[-1]['time']
            else:
                break
        if all_data:
            df = pd.DataFrame(all_data)
            df['date'] = pd.to_datetime(df['time']).dt.tz_localize(None)
            df['mvrv'] = df['CapMVRVCur'].astype(float)
            df = df.drop_duplicates('date').sort_values('date')
            return df[['date', 'mvrv']]
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=86400)  # 24h cache — expensive calculation (downloads 500 tickers)
def fetch_sp500_breadth():
    """Calculate S&P 500 % stocks above 20-DMA from constituent data."""
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                              storage_options={'User-Agent': 'Mozilla/5.0'})[0]
        tickers = [t.replace('.', '-') for t in sp500['Symbol'].tolist()]

        data = yf.download(tickers, start='2006-01-01', progress=False, auto_adjust=True,
                           group_by='ticker', threads=True)

        close_dict = {}
        for t in tickers:
            try:
                s = data[t]['Close'].dropna()
                if len(s) > 20:
                    close_dict[t] = s
            except Exception:
                pass

        close_df = pd.DataFrame(close_dict)
        dma20 = close_df.rolling(20).mean()
        above = (close_df > dma20).sum(axis=1)
        total = close_df.notna().sum(axis=1)
        pct = (above / total * 100).dropna()
        # Drop incomplete rows (e.g. today mid-session with only a few tickers reporting)
        if len(total) >= 2 and total.iloc[-1] < total.iloc[-2] * 0.5:
            pct = pct.iloc[:-1]
        result = pd.DataFrame({'date': pct.index, 'pct_above_20dma': pct.values})
        result['date'] = pd.to_datetime(result['date']).dt.tz_localize(None)
        return result
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_sector_etfs():
    """Fetch IGV and KBE for sector relative strength (used by backtest explorer)."""
    end = datetime.today()
    start = end - timedelta(days=400)
    data = {}
    for ticker, key in [('IGV', 'igv'), ('KBE', 'kbe')]:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if not df.empty:
                close = df['Close']
                # Handle MultiIndex columns from yfinance
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                data[key] = close.dropna()
        except Exception:
            pass
    return data

# ── Leading Sector Tracker ─────────────────────────────────────────────────

SECTOR_ETFS = {
    # GICS Sectors
    'XLK':  ('Technology',            'gics'),
    'XLF':  ('Financials',            'gics'),
    'XLV':  ('Healthcare',            'gics'),
    'XLE':  ('Energy',                'gics'),
    'XLY':  ('Consumer Disc.',        'gics'),
    'XLP':  ('Consumer Staples',      'gics'),
    'XLI':  ('Industrials',           'gics'),
    'XLB':  ('Materials',             'gics'),
    'XLU':  ('Utilities',             'gics'),
    'XLRE': ('Real Estate',           'gics'),
    'XLC':  ('Communications',        'gics'),
    # Modern Thematic
    'SMH':  ('Semiconductors',        'theme'),
    'IGV':  ('Software',              'theme'),
    'CHAT': ('AI (Generative)',       'theme'),
    'BOTZ': ('Robotics & AI',         'theme'),
    'HACK': ('Cybersecurity',         'theme'),
    'UFO':  ('Space Industry',        'theme'),
    'ITA':  ('Defense & Aerospace',   'theme'),
    'URA':  ('Uranium/Nuclear',       'theme'),
    'QTUM': ('Quantum Computing',     'theme'),
    'IBIT': ('Bitcoin',               'theme'),
    'GDX':  ('Gold Miners',           'theme'),
    'XBI':  ('Biotech',               'theme'),
    'KRE':  ('Regional Banks',        'theme'),
    'TAN':  ('Solar',                 'theme'),
    'LIT':  ('Lithium/Battery',       'theme'),
    'XHB':  ('Homebuilders',          'theme'),
    # Reference Indices
    'SPY':  ('S&P 500',               'ref'),
    'QQQ':  ('Nasdaq 100',            'ref'),
    'IWM':  ('Russell 2000',          'ref'),
}


@st.cache_data(ttl=86400)  # 24h — holdings change infrequently
def fetch_etf_top_holdings(ticker: str, n: int = 10):
    """Get top N holdings of an ETF. Returns DataFrame with symbol, name, weight."""
    try:
        t = yf.Ticker(ticker)
        holdings = t.funds_data.top_holdings.head(n).copy()
        if holdings.empty:
            return pd.DataFrame()
        holdings = holdings.reset_index()
        holdings.columns = ['symbol', 'name', 'weight']
        return holdings
    except Exception:
        return pd.DataFrame()


def fetch_etf_holdings_performance(ticker: str, start_date, end_date):
    """Get top 10 holdings + compute their % return over the period."""
    holdings = fetch_etf_top_holdings(ticker, n=10)
    if holdings.empty:
        return pd.DataFrame()

    # Filter to symbols yfinance can reliably pull (skip foreign suffixes like .HK .TA .KS)
    us_mask = holdings['symbol'].apply(lambda s: '.' not in str(s))
    us_holdings = holdings[us_mask].copy()
    if us_holdings.empty:
        return holdings.assign(return_pct=float('nan'))

    # Compute returns for US-listed holdings
    symbols = us_holdings['symbol'].tolist()
    try:
        # Add a buffer day in case the start is non-trading
        fetch_start = pd.Timestamp(start_date) - pd.Timedelta(days=5)
        data = yf.download(symbols, start=fetch_start, end=pd.Timestamp(end_date) + pd.Timedelta(days=1),
                           progress=False, auto_adjust=True, group_by='ticker', threads=True)
        returns = {}
        for s in symbols:
            try:
                close = data[s]['Close'].dropna()
                # Slice to actual period
                close = close[close.index >= pd.Timestamp(start_date)]
                if len(close) >= 2:
                    ret = (close.iloc[-1] / close.iloc[0] - 1) * 100
                    returns[s] = round(ret, 2)
            except Exception:
                pass
        us_holdings['return_pct'] = us_holdings['symbol'].map(returns)
        # Drop any rows where we couldn't compute return
        us_holdings = us_holdings.dropna(subset=['return_pct'])
        return us_holdings
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_sector_universe():
    """Download 2 years of daily prices for all sector/thematic ETFs."""
    tickers = list(SECTOR_ETFS.keys())
    try:
        data = yf.download(tickers, period='2y', progress=False,
                           auto_adjust=True, group_by='ticker', threads=True)
        close_dict = {}
        for t in tickers:
            try:
                s = data[t]['Close'].dropna()
                if len(s) > 20:
                    close_dict[t] = s
            except Exception:
                pass
        df = pd.DataFrame(close_dict)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


def find_pivot_low(series: pd.Series, window: int = 5, min_age_days: int = 10) -> pd.Timestamp:
    """
    Find the most recent confirmed pivot low in a price series.
    A pivot low is a local minimum where price is lower than `window` days
    before AND after. Must be at least `min_age_days` old to be confirmed.
    Note: window and min_age_days are independent constraints (not additive).
    """
    min_right = max(window, min_age_days)  # Need enough days on right for BOTH checks
    if series is None or len(series) < window + min_right + 1:
        return None

    s = series.dropna()
    values = s.values
    idx = s.index

    # Scan from latest backwards. i must have:
    #   - `window` days on left (i >= window)
    #   - `window` days on right for pivot check
    #   - `min_age_days` days on right for age confirmation
    # → i < len - max(window, min_age_days)
    for i in range(len(values) - min_right - 1, window - 1, -1):
        left_min = values[i - window:i].min()
        right_min = values[i + 1:i + window + 1].min()
        if values[i] < left_min and values[i] < right_min:
            return idx[i]  # Most recent pivot low

    return None


def compute_sector_returns(start_date, end_date=None, universe: pd.DataFrame = None):
    """Compute % return for each sector between start and end dates."""
    if universe is None or universe.empty:
        return pd.DataFrame()

    df = universe.copy()
    if end_date is None:
        end_date = df.index.max()

    # Slice to [start, end]
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    sliced = df[(df.index >= start_date) & (df.index <= end_date)]
    if sliced.empty:
        return pd.DataFrame()

    # Use first non-NaN row as start price
    results = []
    spy_ret = None
    for t in sliced.columns:
        s = sliced[t].dropna()
        if len(s) < 2:
            continue
        start_px = s.iloc[0]
        end_px = s.iloc[-1]
        ret = (end_px / start_px - 1) * 100
        name, group = SECTOR_ETFS.get(t, (t, 'other'))
        results.append({
            'ticker': t,
            'name': name,
            'group': group,
            'return_pct': round(ret, 2),
            'start_price': round(start_px, 2),
            'end_price': round(end_px, 2),
            'start_date': s.index[0],
            'end_date': s.index[-1],
        })
        if t == 'SPY':
            spy_ret = ret

    rdf = pd.DataFrame(results)
    if not rdf.empty and spy_ret is not None:
        rdf['vs_spy'] = (rdf['return_pct'] - spy_ret).round(2)
    return rdf.sort_values('return_pct', ascending=False).reset_index(drop=True)


# ── Shapiro Divergence Score ───────────────────────────────────────────────
# Tracks news/market divergence: bad news + market up = bullish reversal,
# good news + market down = bearish reversal. Uses price action proxies
# (VIX regime, gaps, intraday strength) instead of news scraping for reliability.

@st.cache_data(ttl=3600)
def fetch_divergence_data(ticker: str = 'SPY', lookback_days: int = 1900):
    """Fetch OHLC for ticker + VIX + news sentiment history for divergence scoring.
    Default 1900 days (~5 years) to support longer display periods."""
    try:
        end = datetime.today()
        start = end - timedelta(days=lookback_days + 30)

        px = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        vix = yf.download('^VIX', start=start, end=end, progress=False, auto_adjust=True)

        if isinstance(px.columns, pd.MultiIndex):
            px.columns = px.columns.droplevel(1)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.droplevel(1)

        px.index = pd.to_datetime(px.index).tz_localize(None)
        vix.index = pd.to_datetime(vix.index).tz_localize(None)

        px['vix'] = vix['Close']
        # Drop today's row if incomplete (market still open, no close)
        px = px.dropna(subset=['Close', 'Open', 'High', 'Low'])
        # Drop rows where VIX is NaN or zero (pre-market days)
        px = px[px['vix'].notna() & (px['vix'] > 0)]

        # Merge news sentiment (if available)
        try:
            from news import load_sentiment_history
            news_df = load_sentiment_history()
            if not news_df.empty:
                news_df = news_df.set_index('date')
                px = px.join(news_df[['avg_sentiment', 'n_articles', 'n_bullish', 'n_bearish']],
                             how='left')
        except Exception:
            pass

        return px
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=7200)  # 2h cache — FRED data updates weekly/daily, not hourly
def fetch_liquidity_signals():
    """Fetch + compute liquidity regime signals."""
    try:
        from liquidity import compute_liquidity_signals
        return compute_liquidity_signals()
    except Exception as e:
        return {'error': str(e)}


@st.cache_data(ttl=1800)  # 30 min cache — refresh news every 30 min
def refresh_news_data(source_mode: str = 'rss+x'):
    """Refresh news sentiment from sources + Claude classification."""
    try:
        from news import refresh_news_sentiment
        return refresh_news_sentiment(source_mode=source_mode)
    except Exception as e:
        return {'error': str(e), 'n_headlines': 0}


def compute_divergence_score(df: pd.DataFrame, cum_window: int = 10,
                              use_news: bool = True,
                              min_articles: int = 0,
                              use_vix_filter: bool = False,
                              vix_bull_min: float = 18,
                              vix_bear_max: float = 15,
                              use_sentiment_weight: bool = False,
                              use_volume_weight: bool = False) -> pd.DataFrame:
    """
    News-based Shapiro divergence score with optional filters.

    Filters (toggleable):
    - min_articles: skip days with fewer articles (noise filter)
    - use_vix_filter: bull signals need VIX > vix_bull_min, bear signals need VIX < vix_bear_max
    - use_sentiment_weight: scale divergence points by sentiment magnitude (|sent|/0.15, capped at 3x)
    - use_volume_weight: multiply divergence by volume ratio (today vs 20d avg, clipped 0.5-2x)

    Bullish divergence: bearish news + market went up (absorption)
    Bearish divergence: bullish news + market went down (distribution)
    """
    if df.empty or len(df) < 2:
        return pd.DataFrame()

    d = df.copy()
    # Defensive: flatten MultiIndex columns and drop duplicates (fixes yfinance edge cases)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.loc[:, ~d.columns.duplicated()]
    # Ensure Close is a Series, not accidentally a DataFrame
    close = d['Close'].iloc[:, 0] if isinstance(d['Close'], pd.DataFrame) else d['Close']
    d['prev_close'] = close.shift(1)
    d['gap_pct'] = (d['Open'] - d['prev_close']) / d['prev_close'] * 100
    d['day_pct'] = (d['Close'] - d['prev_close']) / d['prev_close'] * 100
    d['intraday_pct'] = (d['Close'] - d['Open']) / d['Open'] * 100
    rng = (d['High'] - d['Low']).replace(0, 1e-9)
    d['close_pos'] = (d['Close'] - d['Low']) / rng  # 0 = close at low, 1 = close at high

    # Require news data
    has_news = 'avg_sentiment' in d.columns and d['avg_sentiment'].notna().any()
    d['bull_div'] = 0
    d['bear_div'] = 0

    if has_news:
        sentiment = d['avg_sentiment']

        # News regime flags (pure news, not VIX/price)
        d['news_bearish'] = (sentiment <= -0.15)
        d['news_bullish'] = (sentiment >= 0.15)

        # Bullish divergence — bearish news + market DIDN'T drop
        bearish_day = d['news_bearish']
        strong_bearish = (sentiment <= -0.3)
        very_strong_bearish = (sentiment <= -0.5)

        d.loc[bearish_day & (d['day_pct'] > 0), 'bull_div'] += 2                        # Green close despite bad news
        d.loc[bearish_day & (d['day_pct'] > 0.5), 'bull_div'] += 1                      # Strong up day
        d.loc[bearish_day & (d['gap_pct'] < -0.3) & (d['intraday_pct'] > 0), 'bull_div'] += 1  # Gap down reversal
        d.loc[bearish_day & (d['close_pos'] > 0.7) & (d['day_pct'] > 0), 'bull_div'] += 1  # Closed near high

        # Bonus for strength of the bad news (the worse the news, the stronger the signal)
        d.loc[strong_bearish & (d['day_pct'] > 0), 'bull_div'] += 1
        d.loc[very_strong_bearish & (d['day_pct'] > 0), 'bull_div'] += 1  # Double bonus for very bearish news

        # Bearish divergence — bullish news + market DIDN'T rally
        bullish_day = d['news_bullish']
        strong_bullish = (sentiment >= 0.3)
        very_strong_bullish = (sentiment >= 0.5)

        d.loc[bullish_day & (d['day_pct'] < 0), 'bear_div'] += 2                        # Red close despite good news
        d.loc[bullish_day & (d['day_pct'] < -0.5), 'bear_div'] += 1                     # Strong down day
        d.loc[bullish_day & (d['gap_pct'] > 0.3) & (d['intraday_pct'] < 0), 'bear_div'] += 1   # Gap up failure
        d.loc[bullish_day & (d['close_pos'] < 0.3) & (d['day_pct'] < 0), 'bear_div'] += 1  # Closed near low

        d.loc[strong_bullish & (d['day_pct'] < 0), 'bear_div'] += 1
        d.loc[very_strong_bullish & (d['day_pct'] < 0), 'bear_div'] += 1

        # Regime labels for UI (pure news-based)
        d['fear_regime'] = d['news_bearish'].fillna(False)
        d['euphoria_regime'] = d['news_bullish'].fillna(False)
        d['regime_source'] = 'news'
    else:
        # No news data → no signals
        d['fear_regime'] = False
        d['euphoria_regime'] = False
        d['regime_source'] = 'no_data'

    # ── Optional filters (applied to daily bull_div / bear_div before cumulative) ──

    # Filter #7: Scale by sentiment magnitude (stronger news = stronger signal)
    if use_sentiment_weight and has_news:
        # Multiplier: 1.0 at |sent|=0.15, linearly scaling to 3.0 at |sent|>=0.45
        abs_sent = d['avg_sentiment'].abs().fillna(0)
        mult = (abs_sent / 0.15).clip(lower=1.0, upper=3.0)
        d['bull_div'] = (d['bull_div'] * mult).round().astype(int)
        d['bear_div'] = (d['bear_div'] * mult).round().astype(int)

    # Filter #8: Volume confirmation (high volume days = stronger signal)
    if use_volume_weight and 'Volume' in d.columns:
        vol_ma20 = d['Volume'].rolling(20, min_periods=5).mean()
        vol_ratio = (d['Volume'] / vol_ma20).fillna(1.0).clip(lower=0.5, upper=2.0)
        d['vol_mult'] = vol_ratio
        d['bull_div'] = (d['bull_div'] * vol_ratio).round().astype(int)
        d['bear_div'] = (d['bear_div'] * vol_ratio).round().astype(int)

    # Filter #3: News density — zero out divergences on low-article days
    if min_articles > 0 and 'n_articles' in d.columns:
        low_density = d['n_articles'].fillna(0) < min_articles
        d.loc[low_density, 'bull_div'] = 0
        d.loc[low_density, 'bear_div'] = 0

    # Filter #4: VIX regime filter
    if use_vix_filter and 'vix' in d.columns:
        # Bull signals need VIX > threshold (panic exists for absorption to matter)
        vix_too_low_for_bull = d['vix'].fillna(0) <= vix_bull_min
        d.loc[vix_too_low_for_bull, 'bull_div'] = 0
        # Bear signals need VIX < threshold (complacency exists for distribution to matter)
        vix_too_high_for_bear = d['vix'].fillna(100) >= vix_bear_max
        d.loc[vix_too_high_for_bear, 'bear_div'] = 0

    # Rolling cumulative scores
    d['cum_bull'] = d['bull_div'].rolling(cum_window, min_periods=1).sum()
    d['cum_bear'] = d['bear_div'].rolling(cum_window, min_periods=1).sum()
    d['cum_net'] = d['cum_bull'] - d['cum_bear']

    return d


def get_divergence_signal(row) -> tuple[str, str, str]:
    """Return (signal_label, color, emoji) based on current cumulative scores."""
    bull = row.get('cum_bull', 0) or 0
    bear = row.get('cum_bear', 0) or 0
    net = bull - bear

    if bull >= 8 and bear < 3:
        return "STRONG Bullish Reversal", "#3fb950", "🟢"
    if bear >= 8 and bull < 3:
        return "STRONG Bearish Reversal", "#ff4444", "🔴"
    if bull >= 5 and net > 3:
        return "Moderate Bullish Reversal", "#7ee787", "🟡"
    if bear >= 5 and net < -3:
        return "Moderate Bearish Reversal", "#ff7b72", "🟡"
    if bull >= 3 and net > 0:
        return "Early Bullish Absorption", "#58a6ff", "⚪"
    if bear >= 3 and net < 0:
        return "Early Bearish Distribution", "#58a6ff", "⚪"
    return "Neutral — Trend Continuing", "#8888a0", "⚪"


@st.cache_data(ttl=86400)
def load_backtest_data():
    """Load backtesting results for all indicators."""
    try:
        from backtest import run_all_backtests, results_to_dataframe
        results = run_all_backtests()
        return results, results_to_dataframe(results)
    except Exception as e:
        return [], pd.DataFrame()

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
        sig['naaim']        = float(naaim_row['naaim'])
        sig['naaim_date']   = str(naaim_row['date'])[:10]
        sig['naaim_source'] = str(naaim_row.get('source', 'csv'))
    else:
        sig['naaim'] = None
        sig['naaim_date'] = 'No data'
    sig['naaim_fired'] = to_bool(sig['naaim'] is not None and sig['naaim'] < NAAIM_THRESHOLD)
    sig['naaim_extreme'] = to_bool(sig['naaim'] is not None and sig['naaim'] < 25)

    # VIX
    sig['vix']          = latest(market.get('vix'))
    sig['vix_fired']    = to_bool(sig['vix'] is not None and sig['vix'] > VIX_THRESHOLD)
    sig['vix_high']     = to_bool(sig['vix'] is not None and sig['vix'] > VIX_HIGH)
    sig['vix_extreme']  = to_bool(sig['vix'] is not None and sig['vix'] > VIX_EXTREME)

    # Confluence
    sig['confluence'] = sum([sig['aaii_fired'], sig['naaim_fired'], sig['vix_fired'], sig.get('pc_10d_fired', False)])

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

    # ── Put/Call Ratio ──
    putcall_df = load_putcall()
    if putcall_df is not None and not putcall_df.empty:
        last_pc = putcall_df.iloc[-1]
        sig['pc_10d_ma'] = float(last_pc['pc_10d_ma']) if pd.notna(last_pc.get('pc_10d_ma')) else None
        sig['pc_30d_ma'] = float(last_pc['pc_30d_ma']) if pd.notna(last_pc.get('pc_30d_ma')) else None
        sig['pc_date'] = str(last_pc['date'])[:10]
        sig['pc_10d_fired'] = to_bool(sig['pc_10d_ma'] is not None and sig['pc_10d_ma'] > 0.70)
        sig['pc_30d_fired'] = to_bool(sig['pc_30d_ma'] is not None and sig['pc_30d_ma'] > 0.65)
    else:
        sig['pc_10d_ma'] = sig['pc_30d_ma'] = None
        sig['pc_date'] = 'No data'
        sig['pc_10d_fired'] = sig['pc_30d_fired'] = False

    # ── AAII Combo (Bears > 50% + Bulls < 25%) ──
    aaii_bulls = sig.get('aaii_bullish')
    aaii_bears = sig.get('aaii_bearish')
    sig['meisler_aaii_combo'] = to_bool(
        aaii_bulls is not None and aaii_bears is not None
        and aaii_bulls < 25 and aaii_bears > 50
    )
    sig['meisler_bulls_low'] = to_bool(aaii_bulls is not None and aaii_bulls < 25)

    # ── Sector Internals ──
    sectors = fetch_sector_etfs()
    spy_series = market.get('spy')
    # Flatten spy_series if MultiIndex
    if spy_series is not None and isinstance(spy_series, pd.DataFrame):
        spy_series = spy_series.iloc[:, 0]
    for key, etf_series in [('igv', sectors.get('igv')), ('kbe', sectors.get('kbe'))]:
        if etf_series is not None and spy_series is not None and len(etf_series) > 60 and len(spy_series) > 60:
            try:
                # Ensure both are plain Series with matching index
                etf_s = etf_series.copy()
                spy_s = spy_series.reindex(etf_s.index, method='ffill')
                ratio = etf_s / spy_s
                ratio = ratio.dropna()
                if len(ratio) > 60:
                    rolling_min = ratio.rolling(60).min()
                    rolling_max = ratio.rolling(60).max()
                    pctile = (ratio - rolling_min) / (rolling_max - rolling_min + 1e-10)
                    val = pctile.iloc[-1]
                    current_pctile = float(val.item() if hasattr(val, 'item') else val)
                    sig[f'{key}_spy_pctile'] = current_pctile
                    sig[f'{key}_breakdown'] = to_bool(current_pctile < 0.10)
                else:
                    sig[f'{key}_spy_pctile'] = None
                    sig[f'{key}_breakdown'] = False
            except Exception:
                sig[f'{key}_spy_pctile'] = None
                sig[f'{key}_breakdown'] = False
        else:
            sig[f'{key}_spy_pctile'] = None
            sig[f'{key}_breakdown'] = False

    # ── COT Positioning (all 37 contracts) ──
    from cot_config import COT_CONTRACTS
    cot_data = load_cot_data()

    # Store all COT contract data for the COT tab
    sig['_cot_data'] = {}

    for contract, cfg in COT_CONTRACTS.items():
        df = cot_data.get(contract)
        is_eq = cfg.get('equity', False)

        if df is not None and not df.empty:
            last = df.iloc[-1]
            spec_p = float(last.get('spec_net_pctile', 0.5)) if pd.notna(last.get('spec_net_pctile')) else None
            comm_p = float(last.get('comm_net_pctile', 0.5)) if pd.notna(last.get('comm_net_pctile')) else None
            small_p = float(last.get('small_spec_net_pctile', 0.5)) if pd.notna(last.get('small_spec_net_pctile')) else None

            sig[f'cot_{contract}_spec_pctile'] = spec_p
            sig[f'cot_{contract}_comm_pctile'] = comm_p
            sig[f'cot_{contract}_small_pctile'] = small_p
            sig[f'cot_{contract}_date'] = str(last['date'])[:10]
            sig[f'cot_{contract}_spec_net'] = float(last['spec_net']) if pd.notna(last.get('spec_net')) else None
            sig[f'cot_{contract}_comm_net'] = float(last['comm_net']) if pd.notna(last.get('comm_net')) else None
            sig[f'cot_{contract}_small_net'] = float(last['small_spec_net']) if pd.notna(last.get('small_spec_net')) else None

            # Method A signal: single type extreme (backward compatible)
            if is_eq:
                # Equities: use commercials
                sig[f'cot_{contract}_fired'] = to_bool(comm_p is not None and (comm_p < 0.05 or comm_p > 0.95))
            else:
                sig[f'cot_{contract}_fired'] = to_bool(spec_p is not None and spec_p < 0.10)

            # Shapiro-style setup: all 3 types extreme
            if is_eq:
                # Equities: only commercials
                setup_long = comm_p is not None and comm_p > 0.95
                setup_short = comm_p is not None and comm_p < 0.05
            else:
                setup_long = (spec_p is not None and spec_p < 0.05 and
                              comm_p is not None and comm_p > 0.95 and
                              small_p is not None and small_p < 0.05)
                setup_short = (spec_p is not None and spec_p > 0.95 and
                               comm_p is not None and comm_p < 0.05 and
                               small_p is not None and small_p > 0.95)

            if setup_long:
                sig[f'cot_{contract}_setup'] = 'LONG'
            elif setup_short:
                sig[f'cot_{contract}_setup'] = 'SHORT'
            else:
                sig[f'cot_{contract}_setup'] = None

            # Warming: at least 1-2 types approaching extreme
            warming = False
            if spec_p is not None and (spec_p < 0.15 or spec_p > 0.85):
                warming = True
            if comm_p is not None and (comm_p < 0.15 or comm_p > 0.85):
                warming = True
            sig[f'cot_{contract}_warming'] = warming and sig[f'cot_{contract}_setup'] is None

            # Check for reversal confirmation on active setups
            reversal_confirmed = False
            if sig[f'cot_{contract}_setup'] is not None:
                try:
                    safe_name = cfg['yf'].replace('=', '_').replace('-', '_').replace('/', '_')
                    price_path = DATA_DIR / f'price_{safe_name}.csv'
                    if price_path.exists():
                        price_df = pd.read_csv(price_path, parse_dates=['date'])
                        cot_date = pd.to_datetime(sig[f'cot_{contract}_date'])
                        # Look at trading days AFTER the COT report date
                        recent = price_df[price_df['date'] > cot_date].head(5)
                        direction = sig[f'cot_{contract}_setup'].lower()
                        for _, prow in recent.iterrows():
                            if direction == 'long' and prow['close'] > prow['open']:
                                reversal_confirmed = True
                                break
                            elif direction == 'short' and prow['close'] < prow['open']:
                                reversal_confirmed = True
                                break
                except Exception:
                    pass

            # Store for COT tab
            sig['_cot_data'][contract] = {
                'label': cfg['label'], 'sector': cfg['sector'],
                'ticker': cfg['yf'], 'equity': is_eq,
                'spec_p': spec_p, 'comm_p': comm_p, 'small_p': small_p,
                'setup': sig[f'cot_{contract}_setup'],
                'warming': sig[f'cot_{contract}_warming'],
                'date': sig[f'cot_{contract}_date'],
                'reversal': reversal_confirmed,
            }
        else:
            sig[f'cot_{contract}_spec_pctile'] = None
            sig[f'cot_{contract}_comm_pctile'] = None
            sig[f'cot_{contract}_small_pctile'] = None
            sig[f'cot_{contract}_date'] = 'No data'
            sig[f'cot_{contract}_fired'] = False
            sig[f'cot_{contract}_setup'] = None

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

@st.cache_data(ttl=86400)
def get_last_fired_context(_version=2):
    """For each indicator, find the most recent signal date and its SPY forward return."""
    try:
        from backtest import load_spy, load_aaii as bt_aaii, load_naaim as bt_naaim, load_putcall as bt_pc, load_cot, backtest_threshold
        spy = load_spy()
        results = {}
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)

        configs = [
            ('aaii_fired', bt_aaii(), 'bull_bear_spread', -20, 'below', '12m'),
            ('naaim_fired', bt_naaim(), 'naaim', 40, 'below', '12m'),
            ('pc_10d_fired', bt_pc(), 'pc_10d_ma', 0.70, 'above', '12m'),
        ]
        # VIX
        vix_path = DATA_DIR / 'vix_daily.csv'
        if vix_path.exists():
            vix_df = pd.read_csv(vix_path, parse_dates=['date'])
            configs.append(('vix_fired', vix_df, 'vix', 30, 'above', '3m'))

        # COT
        for contract, col, thresh, direction, horizon in [
            ('sp500', 'spec_net_pctile', 0.05, 'below', '12m'),
            ('tnote10', 'comm_net_pctile', 0.10, 'below', '3m'),
            ('gold', 'spec_net_pctile', 0.10, 'below', '12m'),
            ('crude', 'spec_net_pctile', 0.05, 'below', '12m'),
            ('usdx', 'spec_net_pctile', 0.10, 'below', '3m'),
        ]:
            try:
                cot_df = load_cot(contract)
                if not cot_df.empty:
                    configs.append((f'cot_{contract}_fired', cot_df, col, thresh, direction, horizon))
            except Exception:
                pass

        for key, ind_df, col, thresh, direction, horizon in configs:
            if ind_df is None or (hasattr(ind_df, 'empty') and ind_df.empty):
                continue
            try:
                r = backtest_threshold(ind_df, spy, col, thresh, direction, name=key)
                if r.signals_df is not None and not r.signals_df.empty:
                    recent = r.signals_df[r.signals_df['date'] < cutoff].sort_values('date', ascending=False)
                    if not recent.empty:
                        row = recent.iloc[0]
                        fwd_col = f'fwd_{horizon}'
                        fwd = row.get(fwd_col)
                        results[key] = {
                            'date': row['date'].strftime('%b %Y'),
                            'return': f"{fwd*100:+.0f}%" if pd.notna(fwd) else "pending",
                            'horizon': horizon,
                        }
            except Exception:
                continue
        return results
    except Exception:
        return {}

@st.cache_data(ttl=86400)
def build_signal_timeline(_version=2):
    """Build signal history for timeline chart — last 5 years. _version param busts cache on data changes."""
    try:
        from backtest import load_spy, load_aaii as bt_aaii, load_naaim as bt_naaim, load_putcall as bt_pc, load_cot, backtest_threshold
        spy = load_spy()
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=5*365)
        timeline = {}

        configs = [
            ('AAII < -20%', bt_aaii(), 'bull_bear_spread', -20, 'below'),
            ('NAAIM < 40', bt_naaim(), 'naaim', 40, 'below'),
            ('Put/Call 10d > 0.70', bt_pc(), 'pc_10d_ma', 0.70, 'above'),
        ]
        vix_path = DATA_DIR / 'vix_daily.csv'
        if vix_path.exists():
            vix_df = pd.read_csv(vix_path, parse_dates=['date'])
            configs.append(('VIX > 30', vix_df, 'vix', 30, 'above'))

        for contract, label in [('sp500', 'S&P 500 COT'), ('tnote10', '10Y T-Note COT'),
                                ('gold', 'Gold COT'), ('crude', 'Crude Oil COT'), ('usdx', 'USD COT')]:
            try:
                cot_df = load_cot(contract)
                if not cot_df.empty:
                    col = 'comm_net_pctile' if contract == 'tnote10' else 'spec_net_pctile'
                    thresh = 0.10 if contract not in ('sp500', 'crude') else 0.05
                    configs.append((label, cot_df, col, thresh, 'below'))
            except Exception:
                pass

        for name, ind_df, col, thresh, direction in configs:
            if ind_df is None or (hasattr(ind_df, 'empty') and ind_df.empty):
                continue
            try:
                r = backtest_threshold(ind_df, spy, col, thresh, direction, name=name)
                if r.signals_df is not None and not r.signals_df.empty:
                    recent = r.signals_df[r.signals_df['date'] >= cutoff]
                    if not recent.empty:
                        timeline[name] = [(row['date'], row.get('fwd_3m')) for _, row in recent.iterrows()]
            except Exception:
                continue
        return timeline
    except Exception:
        return {}

def determine_action_tier(sig, progress_dict, sentiment_only=False):
    """Evaluate triggers and return (tier, active_signals, warming_signals).
    If sentiment_only=True, exclude COT signals from evaluation."""
    active = []
    # Check all firing signals (order = most to least conviction)
    signal_checks = [
        ('vix_extreme', sig.get('vix_extreme')),
        ('vix_high', sig.get('vix_high')),
        ('vix_fired', sig.get('vix_fired')),
        ('aaii_fired', sig.get('aaii_fired')),
        ('naaim_extreme', sig.get('naaim_extreme')),
        ('naaim_fired', sig.get('naaim_fired')),
        ('pc_10d_fired', sig.get('pc_10d_fired')),
    ]
    if not sentiment_only:
        signal_checks += [
            ('cot_tnote10_fired', sig.get('cot_tnote10_fired')),
            ('cot_sp500_fired', sig.get('cot_sp500_fired')),
            ('cot_gold_fired', sig.get('cot_gold_fired')),
            ('cot_crude_fired', sig.get('cot_crude_fired')),
            ('cot_usdx_fired', sig.get('cot_usdx_fired')),
        ]
    for key, fired in signal_checks:
        if fired:
            active.append(key)

    # Warming: progress > 0.66 but key not in active
    warming = []
    for key, prog in progress_dict.items():
        base_key = key
        if prog >= 0.66 and base_key not in active:
            warming.append(base_key)

    # Determine tier — macro gate is sizing context, NOT a filter
    if sig.get('vix_extreme'):
        tier = 'MAX'
    elif sig.get('vix_high') or sig['confluence'] >= 2:
        tier = 'HIGH'
    elif sig.get('naaim_extreme') or (not sentiment_only and sig.get('cot_tnote10_fired')) or sig['confluence'] >= 1:
        tier = 'STRONG'
    elif active:
        tier = 'ACTIVE'
    elif warming:
        tier = 'WATCH'
    else:
        tier = 'NONE'

    return tier, active, warming

def render_action_panel(tier, active, warming, sig):
    """Render the action panel at the top of the dashboard — Obsidian Glass style."""
    t = ACTION_TIERS[tier]
    icon, color, bg = t['icon'], t['color'], t['bg']
    border = t.get('border', f'{color}40')
    text, action = t['text'], t['action']

    # Build active signals list HTML
    active_html = ""
    if active:
        items = []
        for key in active:
            stats = INDICATOR_STATS.get(key, {})
            label = stats.get('label', key)
            wr = stats.get('win_rate', 0)
            h = stats.get('horizon', '12m')
            items.append(f'<span style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.08);border-radius:20px;padding:5px 14px;font-size:0.82em;margin:2px;display:inline-block;color:#c0c0d0">{label}: <b style="color:{color}">{wr*100:.0f}% win @{h}</b></span>')
        active_html = f'<div style="margin-top:12px;display:flex;flex-wrap:wrap;gap:4px">{"".join(items)}</div>'

    # Build warming signals
    warming_html = ""
    if warming:
        w_items = []
        for key in warming:
            stats = INDICATOR_STATS.get(key, {})
            label = stats.get('label', key)
            w_items.append(f'<span style="font-size:0.8em;color:#8888a0">{label}</span>')
        warming_html = f'<div style="margin-top:8px;font-size:0.8em;color:#8888a0">👀 Warming up: {" · ".join(w_items)}</div>'

    # Summary stats
    conf = sig['confluence']
    macro = sig['macro_score']
    macro_ok = sig['macro_bullish']
    conf_color = '#3fb950' if conf >= 2 else ('#e3b341' if conf >= 1 else '#ff6b6b')
    macro_color = '#3fb950' if macro_ok else ('#e3b341' if macro >= 3 else '#ff6b6b')

    # Macro as sizing context
    if macro_ok:
        sizing_text = 'Full size'
        sizing_color = '#3fb950'
    elif macro >= 3:
        sizing_text = 'Half size'
        sizing_color = '#e3b341'
    else:
        sizing_text = 'Quarter size'
        sizing_color = '#ff6b6b'

    badge_html = (
        f'<div style="margin-top:12px;display:flex;gap:10px;flex-wrap:wrap">'
        f'<span style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.08);border-radius:20px;padding:6px 14px;font-size:0.85em;color:#c0c0d0">'
        f'Confluence: <b style="color:{conf_color}">{conf}/4</b></span>'
        f'<span style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.08);border-radius:20px;padding:6px 14px;font-size:0.85em;color:#c0c0d0">'
        f'Macro: <b style="color:{macro_color}">{macro}/6</b></span>'
        f'<span style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.08);border-radius:20px;padding:6px 14px;font-size:0.85em;color:#c0c0d0">'
        f'Sizing: <b style="color:{sizing_color}">{sizing_text}</b></span>'
        f'<span style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.08);border-radius:20px;padding:6px 14px;font-size:0.85em;color:#c0c0d0">'
        f'Active Signals: <b style="color:{color}">{len(active)}</b></span>'
        f'</div>'
    )

    panel_html = (
        f'<div style="background:{bg};border:1px solid {border};border-radius:16px;padding:22px 28px;margin-bottom:8px;'
        f'box-shadow:0 8px 32px rgba(0,0,0,0.4),inset 0 1px 0 rgba(255,255,255,0.05)">'
        f'<div style="display:flex;align-items:center;gap:12px">'
        f'<span style="font-size:2em">{icon}</span>'
        f'<div>'
        f'<div style="font-size:1.3em;font-weight:700;color:{color}">{text}</div>'
        f'<div style="color:#8888a0;font-size:0.9em;margin-top:2px">{action}</div>'
        f'</div></div>'
        f'{badge_html}'
        f'{active_html}'
        f'{warming_html}'
        f'</div>'
    )
    st.markdown(panel_html, unsafe_allow_html=True)

def compute_signal_progress(value, neutral, threshold, direction="lower"):
    """Compute 0.0–1.2 progress toward signal threshold.
    direction='lower': value must drop below threshold (AAII, NAAIM, sector pctile)
    direction='higher': value must rise above threshold (VIX, Put/Call)
    """
    if value is None:
        return 0.0
    if direction == "lower":
        if neutral == threshold:
            return 1.0 if value <= threshold else 0.0
        raw = (neutral - value) / (neutral - threshold)
    else:
        if neutral == threshold:
            return 1.0 if value >= threshold else 0.0
        raw = (value - neutral) / (threshold - neutral)
    return max(0.0, min(1.2, raw))

def battery_color(progress):
    """Return (fill_color, bg_tint, border_color, status_text) for progress 0.0–1.2."""
    if progress >= 1.0:
        return "#3fb950", "rgba(63,185,80,0.05)", "rgba(63,185,80,0.3)", "Firing"
    elif progress >= 0.90:
        return "#3fb950", "rgba(63,185,80,0.04)", "rgba(63,185,80,0.25)", "Almost There"
    elif progress >= 0.66:
        return "#e3b341", "rgba(227,179,65,0.04)", "rgba(227,179,65,0.25)", "Getting Close"
    elif progress >= 0.33:
        return "#e3b341", "rgba(227,179,65,0.03)", "rgba(227,179,65,0.2)", "Warming Up"
    else:
        return "#ff6b6b", "rgba(255,107,107,0.03)", "rgba(255,107,107,0.2)", "Calm"

def battery_html(progress):
    """Return HTML for a slim glass-style progress bar."""
    fill_color, _, _, status_text = battery_color(progress)
    pct = min(progress, 1.0) * 100
    pct_display = int(progress * 100)

    # Gradient endpoint for glass effect
    if progress >= 1.0:
        grad_end = '#7bf2da'
    elif progress >= 0.66:
        grad_end = '#fde68a'
    elif progress >= 0.33:
        grad_end = '#fde68a'
    else:
        grad_end = '#fca5a5'

    bar_bg = "background:rgba(255,255,255,0.06);border-radius:10px;height:8px;position:relative;overflow:hidden"
    bar_fill = f"background:linear-gradient(90deg,{fill_color},{grad_end});width:{pct:.0f}%;height:100%;border-radius:10px;transition:width 0.5s ease"
    label_style = f"font-size:0.72em;color:{fill_color};font-weight:500;display:flex;justify-content:space-between;align-items:center"

    return (
        f'<div style="margin-top:12px">'
        f'<div style="{bar_bg}"><div style="{bar_fill}"></div></div>'
        f'<div style="{label_style};margin-top:6px"><span>{pct_display}%</span><span>{status_text}</span></div>'
        f'</div>'
    )

def big_signal_card(col, sig_key, title, current_str, threshold_str,
                    fired, info, note="", sub_stats="", progress=None):
    # Use progress-based colors if provided, otherwise fall back to binary
    if progress is not None:
        fill_color, bg, border, status = battery_color(progress)
        status_color = fill_color
    else:
        if fired:
            bg = "rgba(63,185,80,0.05)"
            border = "rgba(63,185,80,0.3)"
            status_color = "#3fb950"
        else:
            bg = "rgba(255,107,107,0.03)"
            border = "rgba(255,107,107,0.2)"
            status_color = "#ff6b6b"
        progress = 1.0 if fired else 0.0

    # Glass card with firing glow
    shadow = f"0 4px 30px rgba(63,185,80,0.1)" if fired else "0 4px 20px rgba(0,0,0,0.3)"

    gauge_html = battery_html(progress)
    note_html = f'<div style="margin-top:8px;font-size:0.72em;color:#8888a0">{note}</div>' if note else ''
    stats_html = f'<div style="margin-top:6px;font-size:0.75em;color:#e3b341">{sub_stats}</div>' if sub_stats else ''

    # Gradient text for firing values
    if progress >= 1.0:
        value_style = "font-size:2.2em;font-weight:700;margin:6px 0;background:linear-gradient(135deg,#3fb950,#7bf2da);-webkit-background-clip:text;-webkit-text-fill-color:transparent"
    else:
        value_style = f"font-size:2.2em;font-weight:700;margin:6px 0;color:{status_color}"

    card_html = (
        f'<div style="background:{bg};border:1px solid {border};border-radius:16px;padding:22px;height:100%;'
        f'box-shadow:{shadow};transition:all 0.3s">'
        f'<div style="font-size:0.78em;color:#7777a0;text-transform:uppercase;letter-spacing:0.5px;font-weight:500">{title}</div>'
        f'<div style="{value_style}">{current_str}</div>'
        f'<div style="font-size:0.75em;color:#7777a0">{threshold_str}</div>'
        f'{gauge_html}'
        f'{note_html}'
        f'{stats_html}'
        f'</div>'
    )

    with col:
        st.markdown(card_html, unsafe_allow_html=True)

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
    if fired:
        bg = "rgba(63,185,80,0.05)"
        border = "rgba(63,185,80,0.2)"
    else:
        bg = "rgba(255,107,107,0.03)"
        border = "rgba(255,107,107,0.15)"
    icon   = "✅" if fired else "❌"
    status = info['bullish'] if fired else info['bearish']

    status_color = '#3fb950' if fired else '#ff6b6b'
    detail_html  = f'<div style="font-size:0.75em; color:#7777a0; margin-top:2px">{detail}</div>' if detail else ''

    st.markdown(f"""
    <div style="background:{bg}; border:1px solid {border}; border-radius:12px;
                padding:14px 18px; margin-bottom:8px; display:flex; align-items:flex-start; gap:12px;
                box-shadow:0 2px 12px rgba(0,0,0,0.2)">
        <div style="font-size:1.2em; min-width:24px">{icon}</div>
        <div style="flex:1">
            <div style="font-weight:600; font-size:0.9em; color:#e0e0f0">{info['full']}</div>
            <div style="font-size:0.8em; color:#7777a0; margin-top:2px">{info['why']}</div>
            <div style="font-size:0.78em; color:{status_color}; margin-top:4px">
                &rarr; {status}
            </div>
            {detail_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────

def _render_sentiment_tab(sig, conf, macro, macro_ok, progress_dict, last_fired, signal_timeline):
    """Render the Sentiment Dashboard tab content."""

    # ── Strategy Explainer ────────────────────────────────────────────────
    with st.expander("📖 How this dashboard works", expanded=False):
        st.markdown("""
**Core idea:** Buy when fear is extreme + macro conditions support recovery.

**Section 1 — Core Sentiment (4 indicators)**
Measure when investors are *maximally fearful*:
- **AAII Bull-Bear Spread < -20%** — retail investors more bearish than usual (86% win @12m)
- **NAAIM < 40** — professional fund managers have pulled out of equities (98% win @12m)
- **VIX > 30** — options market pricing in extreme fear/volatility (84% win @3m)

Confluence Score 2–3 = historically high-probability contrarian buy setup.

**Section 2 — COT Futures Positioning (5 indicators)**
CFTC Commitments of Traders data shows what hedge funds and commercial hedgers are doing in futures:
- **10Y T-Note COT** ⭐ — best signal: 89% win @3m, +6.1% avg when commercials at extremes
- **S&P 500 E-mini, Gold, Crude Oil, USD Index** — all with 80%+ win rates at 12 months

**Section 3 — Macro Gate (6 factors)**
Confirms market conditions support a recovery (not just catching a falling knife):
- Yields falling, Dollar weakening, Copper rising, Credit spreads tightening, Yield curve positive, SPY in uptrend
- Need ≥ 4/6 = green light. Improves confluence win rate from 80% → 91% at 3m.

**Backtest results (2010–2026, 15yr):**
| Setup | N | 3m Win Rate | 12m Win Rate | 12m Avg Return |
|-------|---|-------------|--------------|----------------|
| Confluence ≥ 2 (raw) | 257 | 80% | 91% | +23.3% |
| **Confluence ≥ 2 + Macro Gate** | **43** | **91%** | **100%** | **+33.1%** |
| Confluence ≥ 3 | 25 | 96% | 96% | +18.5% |
| 10Y T-Note COT (comm <10th) | 27 | **89%** | — | +19.6% |
| VIX > 40 | 54 | 98% | 100% | +44.2% |
        """)

    # ── Signal History Timeline ──
    with st.expander("📅 Signal History Timeline — Last 5 Years", expanded=False):
        if signal_timeline:
            try:
                from charts import signal_timeline_chart
                fig_timeline = signal_timeline_chart(signal_timeline)
                st.plotly_chart(fig_timeline, use_container_width=True)
                st.caption("🟢 Green = positive 3m return · 🔴 Red = negative · ⚫ Gray = pending. Vertical clustering = multiple signals firing together (best setups).")
            except Exception as e:
                st.warning(f"Could not render timeline: {e}")
        else:
            st.info("Timeline data loading...")

    # ── Section 1: Core Sentiment ─────────────────────────────────────────
    st.subheader(f"📊 Core Sentiment — Confluence: {conf}/4")
    st.caption("Human emotion never changes. These contrarian indicators have 15+ years of consistent edge. Any 2 of 4 firing = actionable setup.")

    c1, c2, c3, c4 = st.columns(4)

    # Helper to build last-fired note prefix
    def lf_note(key):
        lf = last_fired.get(key)
        if lf:
            return f"⏱️ Last fired: {lf['date']} → SPY {lf['return']} over {lf['horizon']} · "
        return ""

    # Card 1: VIX (highest conviction single signal)
    vix_val  = sig.get('vix')
    vix_disp = f"{vix_val:.1f}" if vix_val is not None else "No data"
    vix_prog = progress_dict['vix_fired']
    vix_stats = "📊 >40: 98% win @3m · >50: 100% win all horizons"
    vix_ev = format_ev('vix_fired', vix_prog)
    if vix_ev:
        vix_stats += f"<br>{vix_ev}"
    big_signal_card(c1, "vix", "VIX Fear Index (High)", vix_disp,
                    "threshold: >30 | high: >40 | extreme: >50",
                    sig['vix_fired'], SIGNAL_INFO['vix'],
                    note=(lf_note('vix_fired') + "📈 Intraday high") if lf_note('vix_fired') else "📈 Intraday high",
                    sub_stats=vix_stats,
                    progress=vix_prog)

    # Card 2: Put/Call Ratio
    pc_10d = sig.get('pc_10d_ma')
    pc_30d = sig.get('pc_30d_ma')
    pc_disp = f"{pc_10d:.3f}" if pc_10d is not None else "No data"
    pc_prog = compute_signal_progress(pc_10d, 0.50, 0.70, "higher")
    pc_30d_note = f"30d MA: {pc_30d:.3f}" if pc_30d is not None else ""
    pc_stats = "📊 10d MA > 0.70: 83% win @12m, avg +11.0%"
    pc_ev = format_ev('pc_10d_fired', pc_prog)
    if pc_ev:
        pc_stats += f"<br>{pc_ev}"
    big_signal_card(c2, "putcall", "Put/Call Ratio", pc_disp,
                    "threshold: 10d MA > 0.70 (elevated fear)",
                    sig.get('pc_10d_fired', False), MEISLER_INFO['putcall'],
                    note=f"{lf_note('pc_10d_fired')}📁 CBOE daily · {sig.get('pc_date', 'N/A')} · {pc_30d_note}",
                    sub_stats=pc_stats,
                    progress=pc_prog)

    # Card 3: NAAIM
    naaim_val  = sig.get('naaim')
    naaim_disp = f"{naaim_val:.1f}" if naaim_val is not None else "No data"
    naaim_prog = progress_dict['naaim_fired']
    naaim_src_icon = "🟢 live" if sig.get('naaim_source') == 'live' else "📁 csv"
    naaim_stats = "📊 <40: 98% win @12m · <25: 100% win @12m"
    naaim_ev = format_ev('naaim_fired', naaim_prog)
    if naaim_ev:
        naaim_stats += f"<br>{naaim_ev}"
    big_signal_card(c3, "naaim", "NAAIM Exposure Index", naaim_disp,
                    "threshold: < 40 | extreme: < 25",
                    sig['naaim_fired'], SIGNAL_INFO['naaim'],
                    note=f"{lf_note('naaim_fired')}{naaim_src_icon} · {sig['naaim_date']}",
                    sub_stats=naaim_stats,
                    progress=naaim_prog)

    # Card 4: AAII
    aaii_val  = sig.get('aaii_spread')
    aaii_disp = f"{aaii_val*100:+.1f}%" if aaii_val is not None else "No data"
    aaii_sub  = f"Bulls: {sig.get('aaii_bullish', 0):.1f}%  Bears: {sig.get('aaii_bearish', 0):.1f}%" if sig.get('aaii_bullish') else ""
    aaii_prog = progress_dict['aaii_fired']
    aaii_ev = format_ev('aaii_fired', aaii_prog)
    aaii_stats = "📊 Spread < -20%: 86% win @12m, avg +18.9%"
    if aaii_ev:
        aaii_stats += f"<br>{aaii_ev}"
    big_signal_card(c4, "aaii", "AAII Bull-Bear Spread", aaii_disp,
                    "threshold: Spread < -20%", sig['aaii_fired'],
                    SIGNAL_INFO['aaii'],
                    note=f"{lf_note('aaii_fired')}📁 csv · {sig['aaii_date']} · {aaii_sub}",
                    sub_stats=aaii_stats,
                    progress=aaii_prog)

    # ── Confluence Indicator Charts ──
    with st.expander("📈 Indicator vs SPY Charts — AAII · NAAIM · VIX · Put/Call", expanded=False):
        try:
            from backtest import load_spy as bt_load_spy, load_aaii as bt_load_aaii, load_naaim as bt_load_naaim
            from charts import indicator_spy_chart
            _spy = bt_load_spy()

            chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs(["AAII Spread", "NAAIM", "VIX", "Put/Call Ratio"])

            with chart_tab1:
                _aaii = bt_load_aaii()
                if _aaii is not None and not _aaii.empty:
                    fig_aaii = indicator_spy_chart(
                        _aaii, _spy, 'date', 'bull_bear_spread',
                        title='AAII Bull-Bear Spread vs SPY',
                        y_label='Spread (%)',
                        threshold=-20, threshold_label='Signal: Spread < -20%',
                        threshold_direction='below',
                        invert_y=True)
                    st.plotly_chart(fig_aaii, use_container_width=True)
                    st.caption("When the spread drops below -20% (bears dominate), SPY tends to rally. Lower = more fear = better buy signal.")

            with chart_tab2:
                _naaim = bt_load_naaim()
                if _naaim is not None and not _naaim.empty:
                    fig_naaim = indicator_spy_chart(
                        _naaim, _spy, 'date', 'naaim',
                        title='NAAIM Exposure Index vs SPY',
                        y_label='NAAIM Exposure',
                        threshold=40, threshold_label='Signal: NAAIM < 40',
                        threshold_direction='below',
                        invert_y=True)
                    st.plotly_chart(fig_naaim, use_container_width=True)
                    st.caption("When managers pull out (NAAIM < 40), it's historically been an excellent time to buy. Lower = more capitulation.")

            with chart_tab3:
                vix_path = DATA_DIR / 'vix_daily.csv'
                if vix_path.exists():
                    _vix = pd.read_csv(vix_path, parse_dates=['date'])
                else:
                    import yfinance as yf
                    _vix_data = yf.download('^VIX', start='2000-01-01', progress=False, auto_adjust=True)
                    _vix = _vix_data[['High']].reset_index()
                    _vix.columns = ['date', 'vix']
                    _vix['date'] = pd.to_datetime(_vix['date']).dt.tz_localize(None)
                if _vix is not None and not _vix.empty:
                    fig_vix = indicator_spy_chart(
                        _vix, _spy, 'date', 'vix',
                        title='VIX Fear Index (Intraday High) vs SPY',
                        y_label='VIX',
                        threshold=30, threshold_label='Signal: VIX > 30',
                        color_1='#ff6b6b')
                    st.plotly_chart(fig_vix, use_container_width=True)
                    st.caption("VIX spikes (>30) coincide with market bottoms. The higher the VIX, the stronger the buy signal historically.")

            with chart_tab4:
                from backtest import load_putcall as bt_load_pc_sent
                _pc_sent = bt_load_pc_sent()
                if _pc_sent is not None and not _pc_sent.empty:
                    fig_pc = indicator_spy_chart(
                        _pc_sent, _spy, 'date', 'pc_10d_ma',
                        title='Put/Call Ratio (10d & 30d MA) vs SPY',
                        y_label='Put/Call Ratio',
                        threshold=0.70, threshold_label='Signal: 10d MA > 0.70',
                        threshold_direction='above',
                        value_col_2='pc_30d_ma',
                        label_1='10d MA', label_2='30d MA')
                    st.plotly_chart(fig_pc, use_container_width=True)
                    st.caption("Rising put/call = traders buying protection. Green zones show periods when 10d MA exceeded 0.70 — historically a buy signal.")
        except Exception as e:
            st.warning(f"Could not load chart data: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Macro Gate ────────────────────────────────────────────────────────
    macro_color = "#3fb950" if macro_ok else "#ff6b6b"
    if macro_ok:
        sizing_label = "✅ Full Size — macro confirms"
    elif macro >= 3:
        sizing_label = "🟡 Half Size — macro mixed"
    else:
        sizing_label = "🔴 Quarter Size — macro headwinds"
    st.subheader(f"🔭 Macro Context: {macro}/6 — {sizing_label}")
    st.caption("Use macro conditions to size positions, not to filter trades. More signals at 80% WR compounds better than fewer at 100%.")

    with st.expander("ℹ️ How macro context affects sizing"):
        st.markdown("""
    The macro score tells you: *"How favorable are conditions for a recovery?"*

    **Use it for position sizing, not as a filter:**
    - **≥4/6 → Full size** — macro tailwind, conditions support recovery
    - **3/6 → Half size** — mixed conditions, hedge your conviction
    - **≤2/6 → Quarter size** — macro headwinds, but sentiment extremes still have 80% WR

    **Why sizing beats filtering:**
    - Taking all confluence ≥ 2 trades (80% WR, ~6/yr) → **+20.8% CAGR**
    - Only taking macro-confirmed trades (100% WR, ~2/yr) → +18.5% CAGR
    - Frequency × edge > selectivity. The extra trades compound faster.
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
        cop_dir = '↑ rising' if cop_roc and cop_roc > 0 else '↓ falling'
        detail  = f"${cop_now:.3f}/lb · 20d ROC: {cop_dir} ({cop_roc:+.3f})" if cop_now else "No data"
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

    # ── Backtest Explorer ─────────────────────────────────────────────────
    with st.expander("📈 Backtest Explorer — Interactive Charts", expanded=False):
        st.markdown("**Visual backtesting** of indicators overlaid on SPY. Select an indicator to see when signals fired and outcomes.")

        try:
            from backtest import load_spy, load_putcall as bt_load_putcall, load_aaii as bt_load_aaii, backtest_threshold, load_sector_etfs, backtest_relative_strength
            from charts import spy_overlay_chart, putcall_chart, sector_strength_chart, backtest_summary_chart

            bt_spy = load_spy()
            bt_putcall = bt_load_putcall()
            bt_aaii = bt_load_aaii()

            tab_overlay, tab_putcall, tab_sectors, tab_summary = st.tabs(
                ["📊 Signal Overlay", "📉 Put/Call Chart", "🏗️ Sector Strength", "📋 Backtest Summary"])

            with tab_overlay:
                indicator_choice = st.selectbox("Select indicator to overlay on SPY:", [
                    "Put/Call 10d MA > 0.70",
                    "Put/Call 30d MA > 0.65",
                    "AAII Bulls < 25%",
                    "AAII Spread < -20%",
                    "VIX > 30",
                    "VIX > 40",
                ])
                horizon_choice = st.radio("Forward return horizon:", ['3m', '6m', '12m'], horizontal=True)

                indicator_map = {
                    "Put/Call 10d MA > 0.70": (bt_putcall, 'pc_10d_ma', 0.70, 'above'),
                    "Put/Call 30d MA > 0.65": (bt_putcall, 'pc_30d_ma', 0.65, 'above'),
                    "AAII Bulls < 25%": (bt_aaii, 'bullish', 25, 'below'),
                    "AAII Spread < -20%": (bt_aaii, 'bull_bear_spread', -20, 'below'),
                    "VIX > 30": (None, 'vix', 30, 'above'),
                    "VIX > 40": (None, 'vix', 40, 'above'),
                }

                ind_df, ind_col, threshold, direction = indicator_map[indicator_choice]

                # Load VIX separately if needed
                if ind_df is None:
                    vix_path = DATA_DIR / 'vix_daily.csv'
                    if vix_path.exists():
                        ind_df = pd.read_csv(vix_path, parse_dates=['date'])
                    else:
                        try:
                            vix_data = yf.download('^VIX', start='2000-01-01', progress=False, auto_adjust=True)
                            ind_df = vix_data[['High']].reset_index()
                            ind_df.columns = ['date', 'vix']
                            ind_df['date'] = pd.to_datetime(ind_df['date']).dt.tz_localize(None)
                        except Exception:
                            ind_df = pd.DataFrame()

                if ind_df is not None and not ind_df.empty:
                    result = backtest_threshold(ind_df, bt_spy, ind_col, threshold, direction,
                                               name=indicator_choice)
                    fig = spy_overlay_chart(bt_spy, result.signal_dates, result.signals_df,
                                           title=f"{indicator_choice} — Signals on SPY",
                                           horizon=horizon_choice)
                    st.plotly_chart(fig, use_container_width=True)

                    # Stats summary
                    if result.stats:
                        st.markdown("**Backtest Stats:**")
                        stats_cols = st.columns(4)
                        for i, (h, label) in enumerate([('1m', '1 Month'), ('3m', '3 Months'),
                                                          ('6m', '6 Months'), ('12m', '12 Months')]):
                            s = result.stats.get(h, {})
                            with stats_cols[i]:
                                wr = s.get('win_rate')
                                avg = s.get('avg_return')
                                n = s.get('n', 0)
                                wr_str = f"{wr*100:.0f}%" if wr is not None else "n/a"
                                avg_str = f"{avg*100:+.1f}%" if avg is not None else "n/a"
                                st.metric(label, wr_str, f"avg {avg_str} (n={n})")
                else:
                    st.warning("Indicator data not available")

            with tab_putcall:
                if bt_putcall is not None and not bt_putcall.empty:
                    fig_pc = putcall_chart(bt_putcall, bt_spy)
                    st.plotly_chart(fig_pc, use_container_width=True)
                    st.caption("CBOE equity put/call ratio with 10d and 30d moving averages. Dashed lines show signal thresholds.")
                else:
                    st.warning("Put/Call data not available")

            with tab_sectors:
                sectors_data = load_sector_etfs()
                if not sectors_data.empty:
                    sector_choice = st.selectbox("Select sector:", ['IGV (Software)', 'KBE (Banks)'])
                    ratio_col = 'igv_spy_ratio' if 'IGV' in sector_choice else 'kbe_spy_ratio'
                    etf = 'IGV' if 'IGV' in sector_choice else 'KBE'
                    fig_sect = sector_strength_chart(sectors_data, ratio_col, etf)
                    st.plotly_chart(fig_sect, use_container_width=True)
                    st.caption(f"Red dots show when {etf}/SPY ratio drops below 10th percentile of its 60-day range — sector capitulation signal.")
                else:
                    st.warning("Sector ETF data not available")

            with tab_summary:
                bt_results, bt_df = load_backtest_data()
                if not bt_df.empty:
                    fig_summary = backtest_summary_chart(bt_df)
                    st.plotly_chart(fig_summary, use_container_width=True)

                    st.markdown("**Full Results Table:**")
                    display_df = bt_df.copy()
                    for h in ['1m', '3m', '6m', '12m']:
                        if f'{h}_wr' in display_df.columns:
                            display_df[f'{h}_wr'] = display_df[f'{h}_wr'].apply(
                                lambda x: f"{x*100:.0f}%" if pd.notna(x) else "n/a")
                        if f'{h}_avg' in display_df.columns:
                            display_df[f'{h}_avg'] = display_df[f'{h}_avg'].apply(
                                lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "n/a")
                    cols_show = ['signal'] + [c for c in display_df.columns if c.endswith(('_n', '_wr', '_avg')) and not c.endswith('_med')]
                    st.dataframe(display_df[cols_show], use_container_width=True, hide_index=True)
                else:
                    st.warning("Backtest data not available — run `python backtest.py` first")

        except Exception as e:
            st.error(f"Backtest Explorer error: {e}")
            st.caption("Ensure backtest.py and charts.py are in the project root.")

    # ── Section 6: Backtest Reference ─────────────────────────────────────
    with st.expander("📋 Full Backtest Reference Table (2010–2026)"):
        st.markdown("""
    **Original Indicators:**

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

    **Additional Indicators (backtested 2003–2026 for Put/Call, 1987–2026 for AAII):**

    | Signal | N | 1m WR | 3m WR | 3m Avg | 6m WR | 12m WR | 12m Avg |
    |--------|---|-------|-------|--------|-------|--------|---------|
    | PC 10d MA > 0.70 | 72 | 71% | 75% | +3.3% | 76% | **83%** | +11.0% |
    | PC 30d MA > 0.65 | 99 | 68% | 76% | +3.0% | 78% | **82%** | +10.9% |
    | AAII Bulls < 25% | 115 | 68% | 77% | +5.0% | 75% | **83%** | **+17.3%** |
    | AAII Bears > 50% | 72 | 67% | 67% | +4.8% | 65% | 65% | +10.5% |
    | Bears>50 + Bulls<30 | 62 | 66% | 71% | +4.9% | 69% | 68% | +10.5% |
    | IGV/SPY 10th pctile | 66 | 70% | 75% | +2.7% | 76% | **82%** | +10.6% |
    | KBE/SPY 10th pctile | 101 | 71% | 74% | +3.5% | 75% | **80%** | +11.1% |
        """)
        st.caption("WR = Win Rate (% of signals that resulted in positive forward returns). Source: 15yr backtest, S&P 500, 2010–2026. Put/Call data: CBOE 2003–2026.")

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption("🐺 Alpha · For research purposes only, not financial advice · AAII (1987–2026) · NAAIM (2006–2026) · CBOE Put/Call (2003–2026) · COT (2006–2026) · Market data via Yahoo Finance, FRED & CFTC")


def _render_cot_tab(sig, progress_dict, last_fired):
    """Render the COT Market tab — heat-map table of all 37 contracts."""
    from cot_config import COT_CONTRACTS, SECTORS, SECTOR_COLORS

    st.subheader("🏦 COT Market — 37 Contracts")
    st.caption("CFTC Commitments of Traders positioning across all markets tracked by the Crowded Market Report. "
               "Inspired by Jason Shapiro's contrarian method: fade speculators when all 3 trader types are at extremes + price confirms.")

    # ── Load backtest summary ──
    bt_summary_path = DATA_DIR / 'cot_backtest_summary.csv'
    bt_summary = pd.read_csv(bt_summary_path) if bt_summary_path.exists() else pd.DataFrame()

    # ── Build heat-map data ──
    cot_info = sig.get('_cot_data', {})
    rows = []
    n_setup = 0
    n_warming = 0

    for key, cfg in COT_CONTRACTS.items():
        info = cot_info.get(key, {})
        spec_p = info.get('spec_p')
        comm_p = info.get('comm_p')
        small_p = info.get('small_p')
        setup = info.get('setup')
        warming = info.get('warming', False)
        reversal = info.get('reversal', False)

        if setup:
            n_setup += 1
            if reversal:
                status = f"🔴 {setup} ✅"
            else:
                status = f"🔴 {setup} ⏳"
        elif warming:
            n_warming += 1
            # Determine warming direction from commercials percentile
            if comm_p is not None and comm_p > 0.5:
                status = "🟡 Warming → Long"
            elif comm_p is not None and comm_p < 0.5:
                status = "🟡 Warming → Short"
            else:
                status = "🟡 Warming"
        else:
            status = "—"

        # Get backtest stats per method
        method_stats = {}
        if not bt_summary.empty:
            contract_bt = bt_summary[bt_summary['contract'] == key]
            for method in ['A', 'B', 'C']:
                m_rows = contract_bt[contract_bt['method'] == method]
                if not m_rows.empty:
                    # Pick the row with best win rate
                    best = m_rows.loc[m_rows['win_rate'].idxmax()]
                    wr = best['win_rate']
                    avg_ret = best['avg_return']
                    n = int(best['n_signals'])
                    method_stats[method] = {'wr': wr, 'avg': avg_ret, 'n': n}

        # Find best method
        best_method = None
        best_wr = 0
        for m, s in method_stats.items():
            if s['n'] >= 3 and s['wr'] > best_wr:
                best_wr = s['wr']
                best_method = m

        row = {
            'Contract': cfg['label'],
            'Sector': cfg['sector'],
            'Comm': round(comm_p * 100) if comm_p is not None else None,
            'L.Spec': round(spec_p * 100) if spec_p is not None else None,
            'S.Spec': round(small_p * 100) if small_p is not None else None,
            'Status': status,
            '_key': key,
            '_best_method': best_method,
        }

        # Method columns
        for m in ['A', 'B', 'C']:
            s = method_stats.get(m, {})
            if s:
                wr_str = f"{s['wr']*100:.0f}%"
                if m == best_method and s['n'] >= 3:
                    wr_str = f"★{wr_str}"
                row[f'M{m} WR'] = wr_str
                row[f'M{m} Avg'] = f"{s['avg']*100:+.1f}%"
                row[f'M{m} N'] = s['n']
            else:
                row[f'M{m} WR'] = '—'
                row[f'M{m} Avg'] = '—'
                row[f'M{m} N'] = 0

        # Numeric values for filtering (Method C best row)
        c_stats = method_stats.get('C', {})
        row['_mc_wr_num'] = c_stats.get('wr', 0) * 100 if c_stats else 0
        row['_mc_avg_num'] = c_stats.get('avg', 0) * 100 if c_stats else 0

        rows.append(row)

    df = pd.DataFrame(rows)

    # ── Summary badges ──
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Contracts Set Up", n_setup, help="All 3 trader types at extremes — ready for entry on price confirmation")
    with c2:
        st.metric("Warming Up", n_warming, help="1-2 trader types approaching extremes")
    with c3:
        if not bt_summary.empty:
            avg_c_wr = bt_summary[bt_summary['method'] == 'C']['win_rate'].mean()
            st.metric("Shapiro Method Avg WR", f"{avg_c_wr*100:.0f}%", help="Average win rate across Method C (strict) signals")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Method legend ──
    with st.expander("📖 Method Comparison Guide", expanded=False):
        st.markdown("""
| Method | Signal Trigger | Entry | Exit | Direction |
|--------|---------------|-------|------|-----------|
| **A — Current** | Single type < 10th pctile | Immediate | Hold 3-12m | Long only |
| **B — Hybrid** | All 3 types extreme (10th/90th) | Price reversal within 5 days | Any type → neutral (50th) | Long + Short |
| **C — Shapiro** | All 3 types extreme (5th/95th) | Price reversal within 5 days | Any type → neutral (50th) | Long + Short |

**⭐ = Best method** for that contract (highest win rate with ≥3 signals). Method C (Shapiro strict) typically has fewer signals but higher quality.
Equities use only commercials (Shapiro's rule). All other markets require all 3 types aligned.

**Status Icons:**
| Icon | Meaning |
|------|---------|
| 🔴 SHORT ✅ | COT extreme detected + **reversal candle confirmed** (price reversal within 5 days of COT report) |
| 🔴 LONG ⏳ | COT extreme detected, **awaiting reversal confirmation** (no reversal candle yet) |
| 🟡 Warming | 1-2 trader types approaching extremes — not yet a setup |

The ✅ reversal confirmation approximates Shapiro's "news failure" concept — it means price has started moving against the crowded direction, which is the entry trigger.
        """)

    # ── Filters ──
    filter_col1, filter_col2 = st.columns([3, 1])
    with filter_col1:
        selected_sectors = st.multiselect("Filter by sector", SECTORS, default=SECTORS,
                                          key="cot_sector_filter")
    with filter_col2:
        hide_low_ev = st.toggle("Hide low EV contracts", value=True, key="cot_hide_low_ev",
                               help="Hide contracts with C Avg < 5% and C WR < 70%")

    filtered_df = df[df['Sector'].isin(selected_sectors)].copy()

    # Default filter: hide low-quality contracts
    LOW_EV_FORCED = {
        'lumber', 'bond30', 'tnote10', 'gold', 'crude',
        'cocoa', 'cotton', 'oats', 'wheat', 'soybeans', 'soybean_oil',
        'chf',
    }
    if hide_low_ev:
        auto_mask = (filtered_df['_mc_avg_num'] >= 5) | (filtered_df['_mc_wr_num'] >= 70)
        forced_mask = ~filtered_df['_key'].isin(LOW_EV_FORCED)
        mask = auto_mask & forced_mask
        hidden_count = len(filtered_df) - mask.sum()
        filtered_df = filtered_df[mask]
        if hidden_count > 0:
            st.caption(f"Hiding {hidden_count} low-EV contracts (C Avg < 5% and C WR < 70%) · Toggle 'Show all' to reveal")

    # ── Compact CSS for table rows ──
    st.markdown("""<style>
    /* Shrink buttons in COT table rows */
    [data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) {
        gap: 0.3rem !important;
        margin-bottom: -0.8rem !important;
        border-bottom: 1px solid rgba(255,255,255,0.06) !important;
        padding-bottom: 0.3rem !important;
    }
    [data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) button {
        padding: 0.2rem 0.4rem !important;
        font-size: 0.9rem !important;
        min-height: 0 !important;
        line-height: 1.4 !important;
    }
    [data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) [data-testid="stMarkdownContainer"] {
        font-size: 0.9rem;
    }
    [data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) [data-testid="stMarkdownContainer"] p {
        margin-bottom: 0 !important;
        line-height: 1.4;
    }
    </style>""", unsafe_allow_html=True)

    # ── Render table with Streamlit columns — contract name is a clickable button ──
    def _pctile_html(val):
        """Return HTML span with heatmap background for a percentile value."""
        if val is None:
            return '<span style="display:block;text-align:center;padding:4px;font-weight:600">—</span>'
        v = round(val)
        if v <= 5:
            bg = 'rgba(255,107,107,0.35)'
        elif v <= 15:
            bg = 'rgba(255,107,107,0.15)'
        elif v >= 95:
            bg = 'rgba(63,185,80,0.35)'
        elif v >= 85:
            bg = 'rgba(63,185,80,0.15)'
        else:
            bg = 'transparent'
        return f'<span style="display:block;text-align:center;padding:4px;font-weight:600;white-space:nowrap;background:{bg};border-radius:4px">{v}</span>'

    # Column widths: Contract(button), Sector, Comm, L.Spec, S.Spec, Status, B WR, C WR, C Avg
    col_widths = [2.0, 1.0, 0.7, 0.7, 0.7, 1.5, 0.9, 0.9, 0.9]

    # Header row
    hdr = st.columns(col_widths)
    hdr_labels = ['Contract', 'Sector', 'Comm', 'L.Spec', 'S.Spec', 'Status', 'B WR', 'C WR', 'C Avg']
    for i, label in enumerate(hdr_labels):
        with hdr[i]:
            st.caption(label)

    st.markdown('<hr style="margin:0;border-color:rgba(255,255,255,0.1)">', unsafe_allow_html=True)

    # Data rows
    for _, row in filtered_df.iterrows():
        cols = st.columns(col_widths)
        key = row['_key']
        with cols[0]:
            if st.button(row['Contract'], key=f"tbl_{key}", use_container_width=True):
                st.session_state['_cot_chart_key'] = key
        with cols[1]:
            sector_color = SECTOR_COLORS.get(row['Sector'], '#8888a0')
            st.markdown(f'<span style="color:{sector_color};font-size:0.9em">{row["Sector"]}</span>', unsafe_allow_html=True)
        with cols[2]:
            st.markdown(_pctile_html(row['Comm']), unsafe_allow_html=True)
        with cols[3]:
            st.markdown(_pctile_html(row['L.Spec']), unsafe_allow_html=True)
        with cols[4]:
            st.markdown(_pctile_html(row['S.Spec']), unsafe_allow_html=True)
        with cols[5]:
            status = row['Status']
            if '🔴' in status:
                st.markdown(f'<span style="display:block;text-align:center;padding:4px;background:rgba(255,107,107,0.15);border-radius:4px;font-weight:600">{status}</span>', unsafe_allow_html=True)
            elif '🟡' in status:
                st.markdown(f'<span style="display:block;text-align:center;padding:4px;background:rgba(227,179,65,0.10);border-radius:4px;font-weight:600">{status}</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span style="display:block;text-align:center;padding:4px">{status}</span>', unsafe_allow_html=True)
        with cols[6]:
            st.markdown(f"<span style='white-space:nowrap'>{row['MB WR']}</span>", unsafe_allow_html=True)
        with cols[7]:
            st.markdown(f"<span style='white-space:nowrap'>{row['MC WR']}</span>", unsafe_allow_html=True)
        with cols[8]:
            st.markdown(f"<span style='white-space:nowrap;color:#7bf2da'>{row['MC Avg']}</span>", unsafe_allow_html=True)

    st.caption("Index scale: 0 = max short positioning, 100 = max long. Red ≤5 / Green ≥95 = extreme. "
               "WR = Win Rate. ★ = best performing method for that contract.")

    # ── Contract Detail View ──
    st.markdown("<br>", unsafe_allow_html=True)
    # Build options from visible (filtered) contracts so default is first visible row
    visible_keys = list(filtered_df['_key']) if not filtered_df.empty else []
    contract_options = [f"{COT_CONTRACTS[key]['label']} ({key})" for key in visible_keys
                        if key in COT_CONTRACTS and key in cot_info]
    # Fallback: include all contracts if nothing visible
    if not contract_options:
        contract_options = [f"{cfg['label']} ({key})" for key, cfg in COT_CONTRACTS.items()
                            if key in cot_info]
    if contract_options:

        st.markdown('<div id="cot-detail-anchor"></div>', unsafe_allow_html=True)

        # Use _cot_chart_key if set (from table click or quick button), otherwise selectbox
        preselected_key = st.session_state.get('_cot_chart_key', None)

        if preselected_key:
            # Find the matching option text for the preselected key
            preselected_option = None
            for opt in contract_options:
                if f'({preselected_key})' in opt:
                    preselected_option = opt
                    break
            # Force the selectbox widget to show the correct value
            if preselected_option:
                st.session_state['cot_detail_select_widget'] = preselected_option
            # Clear the override so selectbox works normally on next interaction
            del st.session_state['_cot_chart_key']

        selected = st.selectbox("📊 Select contract for detail view", contract_options,
                                key="cot_detail_select_widget")
        selected_key = selected.split('(')[-1].rstrip(')')

        with st.expander(f"📈 {COT_CONTRACTS[selected_key]['label']} — Positioning Chart", expanded=True):
            try:
                from backtest import load_cot
                from charts import cot_positioning_chart
                import yfinance as yf

                _cot = load_cot(selected_key)
                cfg = COT_CONTRACTS[selected_key]
                ticker = cfg['yf']

                if not _cot.empty:
                    _target = yf.download(ticker, start='2006-01-01', auto_adjust=True, progress=False)
                    if isinstance(_target.columns, pd.MultiIndex):
                        _target.columns = _target.columns.droplevel(1)
                    _target = _target.reset_index()
                    _target.columns = ['date' if c == 'Date' else c.lower() for c in _target.columns]
                    _target['date'] = pd.to_datetime(_target['date']).dt.tz_localize(None)

                    if not _target.empty:
                        # ── Filter toggles (above chart so they control markers) ──
                        st.markdown("**Backtest Comparison:**")
                        ft_col1, ft_col2 = st.columns(2)
                        with ft_col1:
                            dir_filter = st.radio("Direction", ["Both", "Long", "Short"],
                                                  horizontal=True, key=f"bt_dir_{selected_key}")
                        with ft_col2:
                            method_filter = st.radio("Method", ["Both", "Hybrid", "Shapiro"],
                                                     horizontal=True, key=f"bt_method_{selected_key}")

                        # ── Collect trades from selected methods ──
                        _all_trades = []
                        try:
                            from run_cot_backtests import backtest_method_b, backtest_method_c
                            is_eq = cfg.get('equity', False)

                            if method_filter in ("Both", "Hybrid"):
                                bt_b = backtest_method_b(_cot, _target, is_equity=is_eq)
                                _b_trades = bt_b.get('trades', [])
                                if isinstance(_b_trades, pd.DataFrame):
                                    _b_trades = _b_trades.to_dict('records')
                                for t in (_b_trades or []):
                                    t['_method'] = 'Hybrid'
                                _all_trades.extend(_b_trades or [])

                            if method_filter in ("Both", "Shapiro"):
                                bt_c = backtest_method_c(_cot, _target, is_equity=is_eq)
                                _c_trades = bt_c.get('trades', [])
                                if isinstance(_c_trades, pd.DataFrame):
                                    _c_trades = _c_trades.to_dict('records')
                                for t in (_c_trades or []):
                                    t['_method'] = 'Shapiro'
                                _all_trades.extend(_c_trades or [])
                        except Exception:
                            _all_trades = []

                        # Filter by direction
                        if dir_filter != "Both":
                            _all_trades = [t for t in _all_trades if t.get('direction') == dir_filter.lower()]

                        fig_cot = cot_positioning_chart(
                            _cot, _target,
                            contract_label=cfg['label'],
                            ticker=ticker,
                            sector_color=SECTOR_COLORS.get(cfg['sector'], '#e3b341'),
                            is_equity=cfg.get('equity', False),
                            trades=_all_trades if _all_trades else None,
                        )
                        st.plotly_chart(fig_cot, use_container_width=True)

                        # ── Backtest table (filtered same way) ──
                        if not bt_summary.empty:
                            contract_bt = bt_summary[bt_summary['contract'] == selected_key]
                            contract_bt = contract_bt[contract_bt['method_name'] != 'Current']
                            if not contract_bt.empty:
                                bt_filtered = contract_bt.copy()
                                if dir_filter != "Both":
                                    bt_filtered = bt_filtered[bt_filtered['direction'] == dir_filter.lower()]
                                if method_filter != "Both":
                                    bt_filtered = bt_filtered[bt_filtered['method_name'] == method_filter]

                                bt_display = bt_filtered[['method_name', 'direction', 'n_signals',
                                                           'win_rate', 'avg_return']].copy()
                                bt_display.columns = ['Method', 'Direction', 'Signals', 'Win Rate', 'Avg Return']
                                bt_display['Win Rate'] = bt_display['Win Rate'].apply(
                                    lambda x: f"{x*100:.0f}%" if pd.notna(x) else '—')
                                bt_display['Avg Return'] = bt_display['Avg Return'].apply(
                                    lambda x: f"{x*100:+.1f}%" if pd.notna(x) else '—')
                                st.dataframe(bt_display, use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Could not load chart data: {e}")


def _render_technical_lab():
    """Experimental technical charts for significance testing."""
    st.header("🧪 Technical Lab — Experimental Indicators")
    st.caption(
        "These charts are under evaluation. They are **not** part of the core signal system yet. "
        "Use them to cross-reference with sentiment + COT signals for additional confluence."
    )

    st.markdown("---")

    # ── Leading Sector Tracker ──
    with st.expander("🚀 Leading Sector Tracker — Sector Rotation & Leadership", expanded=True):
        st.caption(
            "Ranks sectors and thematic ETFs by % return over a selected period. "
            "Helps spot rotation — which themes are leading, which are lagging. "
            "Default: since the last confirmed pivot low on SPY (swing lows are actionable inflection points)."
        )

        with st.spinner("Loading sector universe (30 ETFs)..."):
            sector_universe = fetch_sector_universe()

        if sector_universe.empty:
            st.warning("Could not fetch sector data.")
        else:
            max_date = sector_universe.index.max()
            min_date = sector_universe.index.min()

            # Period selector
            mode = st.radio(
                "Period mode",
                ["Since Pivot Low", "Last N Days", "Last N Weeks", "Last N Months", "Custom Range"],
                horizontal=True,
                key="sector_mode",
            )

            start_date = None
            pivot_date = None

            if mode == "Since Pivot Low":
                spy_series = sector_universe.get('SPY') if 'SPY' in sector_universe.columns else None
                c1, c2 = st.columns([1, 1])
                with c1:
                    pivot_window = st.slider("Pivot window (days on each side)",
                                              min_value=3, max_value=15, value=5,
                                              help="Larger = stricter pivot definition",
                                              key="pivot_window")
                with c2:
                    min_age = st.slider("Min age to confirm (days)",
                                         min_value=5, max_value=30, value=10,
                                         help="How old the pivot must be to count as confirmed",
                                         key="pivot_min_age")
                pivot_date = find_pivot_low(spy_series, window=pivot_window, min_age_days=min_age)
                if pivot_date is not None:
                    start_date = pivot_date
                    spy_price = spy_series.loc[pivot_date]
                    spy_now = spy_series.iloc[-1]
                    st.info(
                        f"📍 Last confirmed SPY pivot low: **{pivot_date.strftime('%b %d, %Y')}** · "
                        f"SPY ${spy_price:.2f} → ${spy_now:.2f} "
                        f"(**{(spy_now/spy_price - 1)*100:+.1f}%**)"
                    )
                else:
                    st.warning("No pivot low found with these parameters. Try reducing the window or min age.")

            elif mode == "Last N Days":
                n = st.number_input("Days", min_value=1, max_value=730, value=30, step=1, key="sector_ndays")
                start_date = max_date - pd.Timedelta(days=int(n))

            elif mode == "Last N Weeks":
                n = st.number_input("Weeks", min_value=1, max_value=104, value=4, step=1, key="sector_nweeks")
                start_date = max_date - pd.Timedelta(weeks=int(n))

            elif mode == "Last N Months":
                n = st.number_input("Months", min_value=1, max_value=24, value=3, step=1, key="sector_nmonths")
                start_date = max_date - pd.DateOffset(months=int(n))

            elif mode == "Custom Range":
                c1, c2 = st.columns(2)
                with c1:
                    s_date = st.date_input("From", value=max_date - pd.DateOffset(months=3),
                                            min_value=min_date, max_value=max_date,
                                            key="sector_custom_start")
                with c2:
                    e_date = st.date_input("To", value=max_date,
                                            min_value=min_date, max_value=max_date,
                                            key="sector_custom_end")
                start_date = pd.Timestamp(s_date)
                end_date_override = pd.Timestamp(e_date)

            # Compute returns
            if start_date is not None:
                end_date = end_date_override if mode == "Custom Range" else max_date
                returns_df = compute_sector_returns(start_date, end_date, sector_universe)

                if returns_df.empty:
                    st.warning("No data in the selected range.")
                else:
                    spy_row = returns_df[returns_df['ticker'] == 'SPY']
                    spy_ret = float(spy_row['return_pct'].iloc[0]) if not spy_row.empty else 0

                    days_span = (end_date - start_date).days
                    st.markdown(
                        f"**Period:** {start_date.strftime('%b %d, %Y')} → {end_date.strftime('%b %d, %Y')} "
                        f"({days_span} days) · **SPY: {spy_ret:+.2f}%**"
                    )

                    # Summary: top 3 and bottom 3 (excluding refs)
                    no_ref = returns_df[returns_df['group'] != 'ref']
                    if not no_ref.empty:
                        top3 = no_ref.head(3)
                        bot3 = no_ref.tail(3)
                        lead_txt = " · ".join([f"**{r.ticker}** ({r['name']}) {r.return_pct:+.1f}%" for _, r in top3.iterrows()])
                        lag_txt = " · ".join([f"**{r.ticker}** ({r['name']}) {r.return_pct:+.1f}%" for _, r in bot3.iterrows()])
                        st.markdown(f"🟢 **Leading:** {lead_txt}")
                        st.markdown(f"🔴 **Lagging:** {lag_txt}")

                    st.markdown("")

                    # Bar chart
                    chart_df = returns_df.copy()
                    chart_df['label'] = chart_df.apply(lambda r: f"{r['ticker']} · {r['name']}", axis=1)

                    # Color: green if beats SPY, red if underperforms, blue for reference
                    def _color(row):
                        if row['group'] == 'ref':
                            return '#58a6ff'
                        if row['return_pct'] > spy_ret:
                            return '#3fb950'
                        return '#ff6b6b'
                    chart_df['color'] = chart_df.apply(_color, axis=1)

                    # Sort ascending for horizontal bar (best at top)
                    chart_df = chart_df.sort_values('return_pct', ascending=True)

                    fig_s = go.Figure()
                    fig_s.add_trace(go.Bar(
                        y=chart_df['label'],
                        x=chart_df['return_pct'],
                        orientation='h',
                        marker_color=chart_df['color'],
                        text=[f"{v:+.1f}%" for v in chart_df['return_pct']],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Return: %{x:+.2f}%<br>vs SPY: %{customdata:+.2f}%<extra></extra>',
                        customdata=chart_df['vs_spy'] if 'vs_spy' in chart_df.columns else chart_df['return_pct'],
                    ))
                    # SPY reference line
                    fig_s.add_vline(x=spy_ret, line_dash="dash", line_color="#58a6ff", opacity=0.5,
                                     annotation_text=f"SPY {spy_ret:+.1f}%", annotation_position="top")
                    fig_s.add_vline(x=0, line_color="#8888a0", opacity=0.3, line_width=1)
                    fig_s.update_layout(
                        template='plotly_dark', paper_bgcolor='#07070d', plot_bgcolor='#07070d',
                        height=max(500, len(chart_df) * 22), margin=dict(l=10, r=60, t=20, b=30),
                        xaxis=dict(title='% Return', zeroline=False),
                        yaxis=dict(title=''),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_s, use_container_width=True)

                    # Detail table with vs SPY column
                    with st.expander("📋 Detailed Table", expanded=False):
                        tbl = returns_df[['ticker', 'name', 'group', 'return_pct', 'vs_spy',
                                          'start_price', 'end_price']].copy()
                        tbl.columns = ['Ticker', 'Name', 'Group', 'Return %', 'vs SPY %',
                                       'Start $', 'End $']
                        tbl['Return %'] = tbl['Return %'].apply(lambda x: f"{x:+.2f}%")
                        tbl['vs SPY %'] = tbl['vs SPY %'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else '—')
                        tbl['Start $'] = tbl['Start $'].apply(lambda x: f"${x:,.2f}")
                        tbl['End $'] = tbl['End $'].apply(lambda x: f"${x:,.2f}")
                        st.dataframe(tbl, use_container_width=True, hide_index=True)

                    # ── Drill-Down: Show holdings of selected sector ──
                    st.markdown("---")
                    st.markdown("**🔍 Drill Down — See What's Inside Each Sector:**")

                    # Build selector with rankings
                    sector_options = [
                        f"{row['ticker']} · {row['name']}  ({row['return_pct']:+.1f}%)"
                        for _, row in returns_df.iterrows()
                    ]
                    sector_map = dict(zip(sector_options, returns_df['ticker'].tolist()))

                    selected_label = st.selectbox(
                        "Select ETF to see top 10 holdings + performance:",
                        sector_options,
                        key="sector_drill",
                    )
                    drill_ticker = sector_map[selected_label]

                    # Fetch top 10 holdings + compute their performance for the same period
                    holdings_df = fetch_etf_holdings_performance(
                        drill_ticker, start_date, end_date
                    )

                    if holdings_df is None or holdings_df.empty:
                        sector_row = returns_df[returns_df['ticker'] == drill_ticker].iloc[0]
                        if drill_ticker == 'IBIT':
                            st.info("💰 IBIT holds Bitcoin directly — no stock holdings to display.")
                        else:
                            st.warning(f"No holdings data available for {drill_ticker}")
                    else:
                        sector_row = returns_df[returns_df['ticker'] == drill_ticker].iloc[0]

                        # Header card
                        card_color = "#3fb950" if sector_row['return_pct'] > spy_ret else "#ff6b6b"
                        if sector_row['group'] == 'ref':
                            card_color = "#58a6ff"
                        st.markdown(f"""
                        <div style="border:1px solid rgba(255,255,255,0.08);background:rgba(255,255,255,0.02);border-radius:10px;padding:12px;margin:8px 0">
                            <div style="font-size:0.85em;color:#8888a0">
                                <b style="color:#e0e0f0;font-size:1.1em">{drill_ticker} — {sector_row['name']}</b> ·
                                Period Return: <span style="color:{card_color};font-weight:600">{sector_row['return_pct']:+.2f}%</span>
                                · vs SPY: <span style="color:{card_color}">{sector_row['vs_spy']:+.2f}%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Holdings chart — horizontal bars
                        hdf = holdings_df.sort_values('return_pct', ascending=True)
                        hdf['label'] = hdf.apply(lambda r: f"{r['symbol']} · {r['name'][:30]}", axis=1)

                        etf_ret = sector_row['return_pct']
                        def _holding_color(ret):
                            if ret < 0:
                                return '#ff4444'   # Red — actual loss
                            if ret > etf_ret:
                                return '#3fb950'   # Green — beat the ETF
                            return '#e3b341'       # Amber — positive but underperformed ETF
                        hdf['bar_color'] = hdf['return_pct'].apply(_holding_color)

                        fig_h = go.Figure()
                        fig_h.add_trace(go.Bar(
                            y=hdf['label'],
                            x=hdf['return_pct'],
                            orientation='h',
                            marker_color=hdf['bar_color'],
                            text=[f"{v:+.1f}% · {w:.1%}" for v, w in zip(hdf['return_pct'], hdf['weight'])],
                            textposition='outside',
                            hovertemplate='<b>%{y}</b><br>Return: %{x:+.2f}%<br>Weight in ETF: %{customdata:.2%}<extra></extra>',
                            customdata=hdf['weight'],
                        ))
                        # ETF itself as reference line
                        fig_h.add_vline(
                            x=sector_row['return_pct'], line_dash="dash", line_color="#e3b341", opacity=0.6,
                            annotation_text=f"{drill_ticker} {sector_row['return_pct']:+.1f}%",
                            annotation_position="top"
                        )
                        fig_h.add_vline(x=0, line_color="#8888a0", opacity=0.3, line_width=1)
                        fig_h.update_layout(
                            template='plotly_dark', paper_bgcolor='#07070d', plot_bgcolor='#07070d',
                            height=max(350, len(hdf) * 32), margin=dict(l=10, r=100, t=30, b=30),
                            xaxis=dict(title='% Return (same period)', zeroline=False),
                            yaxis=dict(title=''),
                            showlegend=False,
                        )
                        st.plotly_chart(fig_h, use_container_width=True)

                        st.caption(
                            f"Top 10 holdings by weight · Bar format: `Return% · Weight%` · "
                            f"🟢 Beat {drill_ticker}  ·  🟡 Positive but under {drill_ticker}  ·  🔴 Negative return"
                        )

    # ── Shapiro Divergence Score ──
    with st.expander("🎯 Shapiro Divergence — News vs Market Reaction", expanded=True):
        st.caption(
            "**100% news-based signal.** Compares each day's classified news sentiment against the market's actual reaction. "
            "Sources: Reuters, Bloomberg, WSJ, CNBC, FT, MarketWatch, BBC — classified by Claude. "
            "**🟢 Bullish divergence**: bearish news + market closed green (buyers absorbing). "
            "**🔴 Bearish divergence**: bullish news + market closed red (distribution/topping). "
            "Days with < 6 articles are filtered out (too few to be statistically meaningful)."
        )

        col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([1.2, 1.2, 1, 1.2])
        with col_ctrl1:
            div_ticker = st.selectbox(
                "Instrument",
                ['SPY', 'QQQ', 'IWM', 'DIA'],
                key="div_ticker",
            )
        with col_ctrl2:
            div_period = st.selectbox(
                "Display period",
                ['30 days', '60 days', '90 days', '6 months', '1 year', '2 years', '3 years', '5 years', 'Max'],
                index=1,
                key="div_period",
            )
        with col_ctrl3:
            div_cum_window = st.selectbox(
                "Cum window",
                [5, 10, 15, 20],
                index=1,
                key="div_cum_window",
            )
        with col_ctrl4:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            if st.button("🔄 Refresh News", key="refresh_news_btn",
                         help="Fetch latest headlines & classify via Claude (~$0.01)"):
                src_mode = st.session_state.get('news_source_mode', 'rss+x')
                with st.spinner(f"Fetching via {src_mode}..."):
                    news_result = refresh_news_data(source_mode=src_mode)
                    # Clear the fetch_divergence_data cache so it reloads with new news
                    fetch_divergence_data.clear()
                    if news_result.get('error'):
                        st.warning(f"News refresh: {news_result['error']}")
                    else:
                        breakdown = news_result.get('source_breakdown', {})
                        top_srcs = sorted(breakdown.items(), key=lambda x: -x[1])[:5]
                        src_str = ' · '.join(f"{s}({n})" for s, n in top_srcs)
                        st.success(
                            f"✓ Classified {news_result.get('n_headlines', 0)} headlines · "
                            f"Sources: {src_str}"
                        )
                        st.rerun()

        # Source mode selector — where news comes from
        src_col1, src_col2 = st.columns([1.5, 3])
        with src_col1:
            st.selectbox(
                "News sources",
                options=['rss+x', 'rss', 'x'],
                format_func=lambda x: {
                    'rss+x': '📰📱 RSS + X (both)',
                    'rss': '📰 RSS feeds only',
                    'x': '📱 X/Twitter only',
                }.get(x, x),
                key="news_source_mode",
                help="RSS = CNBC/WSJ/FT/BBC etc. · X = @WSJmarkets/@MarketWatch/@zerohedge",
            )
        with src_col2:
            current_mode = st.session_state.get('news_source_mode', 'rss+x')
            src_desc = {
                'rss+x': '📰 RSS feeds (8 outlets) **+** 📱 X handles (@WSJmarkets, @MarketWatch, @zerohedge)',
                'rss': '📰 RSS only: Reuters, Bloomberg, WSJ, CNBC, FT, MarketWatch, BBC',
                'x': '📱 X handles only: @WSJmarkets, @MarketWatch, @zerohedge (via Nitter)',
            }
            st.caption(src_desc.get(current_mode, ''))

        with st.spinner(f"Computing divergence scores for {div_ticker}..."):
            div_raw = fetch_divergence_data(div_ticker)
            # Always require ≥ 6 articles/day for statistical reliability
            div_df = compute_divergence_score(
                div_raw,
                cum_window=int(div_cum_window),
                min_articles=6,
            )

        if div_df.empty:
            st.warning("Could not compute divergence data.")
        else:
            # Slice to display period
            period_map = {
                '30 days': 30, '60 days': 60, '90 days': 90, '6 months': 126,
                '1 year': 252, '2 years': 504, '3 years': 756, '5 years': 1260,
                'Max': None,  # Use all data
            }
            n_days = period_map[div_period]
            display_df = div_df.copy() if n_days is None else div_df.tail(n_days).copy()

            # Current signal state
            latest_row = div_df.iloc[-1]
            signal_label, signal_color, signal_emoji = get_divergence_signal(latest_row)
            cum_bull = int(latest_row['cum_bull']) if pd.notna(latest_row['cum_bull']) else 0
            cum_bear = int(latest_row['cum_bear']) if pd.notna(latest_row['cum_bear']) else 0
            net_score = cum_bull - cum_bear

            # Signal card
            st.markdown(f"""
            <div style="border:1px solid {signal_color}55;background:rgba(255,255,255,0.02);border-radius:12px;padding:14px;max-width:720px">
                <div style="font-size:0.8em;color:#8888a0;font-weight:600;text-transform:uppercase;letter-spacing:0.5px">
                    {div_ticker} · Current Divergence Signal
                </div>
                <div style="display:flex;align-items:baseline;gap:14px;margin:6px 0">
                    <span style="font-size:1.4em;color:{signal_color};font-weight:700">{signal_emoji} {signal_label}</span>
                </div>
                <div style="display:flex;gap:24px;font-size:0.85em;color:#ccc;margin-top:6px">
                    <span>🟢 Bull score (last {div_cum_window}d): <b style="color:#3fb950">{cum_bull}</b></span>
                    <span>🔴 Bear score (last {div_cum_window}d): <b style="color:#ff4444">{cum_bear}</b></span>
                    <span>Net: <b style="color:{signal_color}">{net_score:+d}</b></span>
                </div>
                <div style="font-size:0.75em;color:#888;margin-top:8px">
                    Last market close: {div_df.index[-1].strftime('%b %d, %Y')}
                </div>
                {
                    f'<div style="font-size:0.8em;color:#ccc;margin-top:6px;padding:6px 10px;background:rgba(255,255,255,0.03);border-radius:6px">📰 Today\'s news: <b style="color:{"#ff4444" if latest_row.get("avg_sentiment", 0) < -0.1 else "#3fb950" if latest_row.get("avg_sentiment", 0) > 0.1 else "#8888a0"}">{latest_row.get("avg_sentiment", 0):+.2f}</b> sentiment · <b>{int(latest_row.get("n_articles", 0) or 0)}</b> articles (🟢 {int(latest_row.get("n_bullish", 0) or 0)} bullish · 🔴 {int(latest_row.get("n_bearish", 0) or 0)} bearish) · Classified by Claude</div>'
                    if pd.notna(latest_row.get('avg_sentiment')) else
                    '<div style="font-size:0.8em;color:#aaa;margin-top:6px;padding:6px 10px;background:rgba(255,179,71,0.08);border-radius:6px">⚠️ No news data logged yet. Click <b>🔄 Refresh News</b> to fetch headlines and start building history.</div>'
                }
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")

            # ── Chart: Candles + signal markers + cumulative score ──
            from plotly.subplots import make_subplots

            fig_d = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.04,
                row_heights=[0.65, 0.35],
                subplot_titles=(f'{div_ticker} Price + Divergence Signals', 'Cumulative Divergence Score'),
            )

            # Row 1: Candlesticks
            fig_d.add_trace(go.Candlestick(
                x=display_df.index,
                open=display_df['Open'], high=display_df['High'],
                low=display_df['Low'], close=display_df['Close'],
                name=div_ticker,
                increasing=dict(line=dict(color='#3fb950'), fillcolor='#3fb950'),
                decreasing=dict(line=dict(color='#ff4444'), fillcolor='#ff4444'),
                showlegend=False,
            ), row=1, col=1)

            # Load classified headlines for rich hover tooltips
            try:
                from news import load_headlines_log
                all_headlines = load_headlines_log(n_days=365)
                all_headlines['date'] = pd.to_datetime(all_headlines['timestamp']).dt.normalize()
            except Exception:
                all_headlines = pd.DataFrame()

            def _safe_int_hdr(v):
                try:
                    return int(v) if pd.notna(v) else 0
                except (ValueError, TypeError):
                    return 0

            def _safe_float_hdr(v, default=0.0):
                try:
                    return float(v) if pd.notna(v) else default
                except (ValueError, TypeError):
                    return default

            def _build_hover_text(row, is_bull: bool):
                """Build rich HTML hover with news context."""
                date = row.name
                sent = row.get('avg_sentiment', None)
                n_art = _safe_int_hdr(row.get('n_articles', 0))
                n_b = _safe_int_hdr(row.get('n_bullish', 0))
                n_r = _safe_int_hdr(row.get('n_bearish', 0))
                day_pct = _safe_float_hdr(row.get('day_pct', 0))
                gap_pct = _safe_float_hdr(row.get('gap_pct', 0))
                intraday = _safe_float_hdr(row.get('intraday_pct', 0))
                close_pos = _safe_float_hdr(row.get('close_pos', 0.5)) * 100
                score = _safe_int_hdr(row.get('bull_div' if is_bull else 'bear_div', 0))

                # Pull top headlines for this date
                top_headlines = []
                if not all_headlines.empty:
                    day_headlines = all_headlines[all_headlines['date'] == date]
                    if not day_headlines.empty:
                        if is_bull:
                            # For bull div, show top BEARISH headlines (what market ignored)
                            top = day_headlines.nsmallest(3, 'sentiment')
                        else:
                            # For bear div, show top BULLISH headlines (what market ignored)
                            top = day_headlines.nlargest(3, 'sentiment')
                        for _, h in top.iterrows():
                            s = float(h['sentiment'])
                            if abs(s) >= 0.1:
                                # Truncate title
                                title = str(h['title'])[:70] + ('...' if len(str(h['title'])) > 70 else '')
                                top_headlines.append(f"{s:+.2f} {title}")

                parts = [f"<b>{date.strftime('%b %d, %Y')}</b>"]
                parts.append(f"━━━━━━━━━━━━━━━━━━━━━━━━")
                if is_bull:
                    parts.append(f"🟢 <b>Bullish Divergence (score: {score})</b>")
                    parts.append(f"<i>Bearish news + market absorbed</i>")
                else:
                    parts.append(f"🔴 <b>Bearish Divergence (score: {score})</b>")
                    parts.append(f"<i>Bullish news + market failed to rally</i>")

                parts.append(f"")
                parts.append(f"<b>📰 News sentiment:</b> {sent:+.2f} ({n_art} articles)")
                parts.append(f"&nbsp;&nbsp;🟢 {n_b} bullish · 🔴 {n_r} bearish")
                parts.append(f"")
                parts.append(f"<b>📊 Market reaction:</b>")
                parts.append(f"&nbsp;&nbsp;Gap: {gap_pct:+.2f}% → Close: {day_pct:+.2f}%")
                parts.append(f"&nbsp;&nbsp;Intraday: {intraday:+.2f}% · Close in top {100-close_pos:.0f}% of range")

                if top_headlines:
                    parts.append(f"")
                    label = "Top bearish headlines:" if is_bull else "Top bullish headlines:"
                    parts.append(f"<b>{label}</b>")
                    for h in top_headlines[:3]:
                        parts.append(f"&nbsp;&nbsp;{h}")

                return "<br>".join(parts)

            # Bullish divergence markers (triangle up, below the low)
            bull_days = display_df[display_df['bull_div'] >= 2]
            if not bull_days.empty:
                bull_hover_text = [_build_hover_text(row, is_bull=True) for _, row in bull_days.iterrows()]
                fig_d.add_trace(go.Scatter(
                    x=bull_days.index,
                    y=bull_days['Low'] * 0.995,
                    mode='markers',
                    marker=dict(color='#3fb950', size=10, symbol='triangle-up',
                                 line=dict(color='white', width=1)),
                    name='Bull Divergence',
                    text=bull_hover_text,
                    hovertemplate='%{text}<extra></extra>',
                ), row=1, col=1)

            # Bearish divergence markers (triangle down, above the high)
            bear_days = display_df[display_df['bear_div'] >= 2]
            if not bear_days.empty:
                bear_hover_text = [_build_hover_text(row, is_bull=False) for _, row in bear_days.iterrows()]
                fig_d.add_trace(go.Scatter(
                    x=bear_days.index,
                    y=bear_days['High'] * 1.005,
                    mode='markers',
                    marker=dict(color='#ff4444', size=10, symbol='triangle-down',
                                 line=dict(color='white', width=1)),
                    name='Bear Divergence',
                    text=bear_hover_text,
                    hovertemplate='%{text}<extra></extra>',
                ), row=1, col=1)

            # Row 2: Cumulative bull & bear as overlaid bars
            def _safe_int(v):
                try:
                    return int(v) if pd.notna(v) else 0
                except (ValueError, TypeError):
                    return 0

            def _safe_float(v, default=0.0):
                try:
                    return float(v) if pd.notna(v) else default
                except (ValueError, TypeError):
                    return default

            def _cum_hover(row, is_bull: bool):
                date = row.name
                sent = row.get('avg_sentiment', None)
                day_pct = _safe_float(row.get('day_pct', 0))
                n_art = _safe_int(row.get('n_articles', 0))
                cum_b = _safe_int(row.get('cum_bull', 0))
                cum_r = _safe_int(row.get('cum_bear', 0))
                today_b = _safe_int(row.get('bull_div', 0))
                today_r = _safe_int(row.get('bear_div', 0))

                signal = 'Neutral'
                color_emoji = '⚪'
                if cum_b >= 8:
                    signal, color_emoji = 'STRONG Bull Reversal', '🟢'
                elif cum_r >= 8:
                    signal, color_emoji = 'STRONG Bear Reversal', '🔴'
                elif cum_b >= 5:
                    signal, color_emoji = 'Moderate Bull Absorption', '🟡'
                elif cum_r >= 5:
                    signal, color_emoji = 'Moderate Bear Distribution', '🟡'

                sent_str = f"{sent:+.2f}" if sent is not None and pd.notna(sent) else 'N/A'
                parts = [
                    f"<b>{date.strftime('%b %d, %Y')}</b>",
                    f"━━━━━━━━━━━━━━━━━━━━━",
                    f"{color_emoji} <b>{signal}</b>",
                    f"",
                    f"<b>Cumulative (10d):</b>",
                    f"&nbsp;&nbsp;🟢 Bull: {cum_b} · 🔴 Bear: {cum_r}",
                    f"",
                    f"<b>Today:</b>",
                    f"&nbsp;&nbsp;News sentiment: {sent_str} ({n_art} articles)",
                    f"&nbsp;&nbsp;SPY close: {day_pct:+.2f}%",
                    f"&nbsp;&nbsp;Bull divergence: +{today_b} · Bear: +{today_r}",
                ]
                return "<br>".join(parts)

            cum_hover_text = [_cum_hover(row, True) for _, row in display_df.iterrows()]

            fig_d.add_trace(go.Bar(
                x=display_df.index, y=display_df['cum_bull'],
                marker_color='#3fb950', opacity=0.7,
                name=f'Cum Bull ({div_cum_window}d)',
                text=cum_hover_text,
                hovertemplate='%{text}<extra></extra>',
            ), row=2, col=1)

            fig_d.add_trace(go.Bar(
                x=display_df.index, y=-display_df['cum_bear'],
                marker_color='#ff4444', opacity=0.7,
                name=f'Cum Bear ({div_cum_window}d)',
                text=cum_hover_text,
                hovertemplate='%{text}<extra></extra>',
            ), row=2, col=1)

            # Threshold lines on score chart
            fig_d.add_hline(y=8, row=2, line_dash="dash", line_color="#3fb950", opacity=0.5,
                             annotation_text="Strong Bull", annotation_position="top left",
                             annotation_font_color="#3fb950")
            fig_d.add_hline(y=-8, row=2, line_dash="dash", line_color="#ff4444", opacity=0.5,
                             annotation_text="Strong Bear", annotation_position="bottom left",
                             annotation_font_color="#ff4444")
            fig_d.add_hline(y=0, row=2, line_color="#8888a0", opacity=0.3, line_width=1)

            fig_d.update_layout(
                template='plotly_dark', paper_bgcolor='#07070d', plot_bgcolor='#07070d',
                height=700, margin=dict(l=50, r=20, t=50, b=30),
                xaxis_rangeslider_visible=False,
                xaxis2_rangeslider_visible=False,
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                barmode='relative',
            )
            fig_d.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig_d.update_yaxes(title_text="Score", row=2, col=1)

            st.plotly_chart(fig_d, use_container_width=True)

            # Thresholds legend
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:10px;font-size:0.8em;color:#aaa;margin:6px 0">
                <b style="color:#e0e0f0">How to read this:</b><br>
                🔺 <span style="color:#3fb950">Green triangles below bars</span> = bullish divergence (bearish news day but market closed green — absorption)<br>
                🔻 <span style="color:#ff4444">Red triangles above bars</span> = bearish divergence (bullish news day but market closed red — distribution)<br>
                <br>
                <b style="color:#e0e0f0">Cumulative score thresholds (last {div_cum_window} days):</b><br>
                <span style="color:#3fb950">Bull ≥ 8</span> = Strong bullish reversal · <span style="color:#3fb950">Bull ≥ 5</span> = Moderate absorption<br>
                <span style="color:#ff4444">Bear ≥ 8</span> = Strong bearish reversal · <span style="color:#ff4444">Bear ≥ 5</span> = Moderate distribution<br>
                <br>
                <span style="color:#888">Higher scores when the news is more extreme — very bearish news + green close scores higher than mildly bearish news + green close.</span>
            </div>
            """, unsafe_allow_html=True)

            # Recent divergence events table
            # ── Daily Breakdown Table: last 14 days with bull/bear article counts ──
            with st.expander("📰 Daily News Breakdown (last 14 days)", expanded=False):
                st.caption(
                    "Shows the news regime each day alongside market reaction. "
                    "**Ratio of bullish vs bearish articles matters more than absolute count** — "
                    "a day with 10 bearish / 2 bullish articles sets up a much stronger absorption signal than 5 neutral articles."
                )
                recent_14 = display_df.tail(14).iloc[::-1].copy()
                breakdown_rows = []
                for idx, r in recent_14.iterrows():
                    sent = r.get('avg_sentiment', None)
                    n_art = _safe_int(r.get('n_articles', 0))
                    n_b = _safe_int(r.get('n_bullish', 0))
                    n_r = _safe_int(r.get('n_bearish', 0))
                    n_neu = max(0, n_art - n_b - n_r)
                    day_pct = _safe_float(r.get('day_pct', 0))
                    bull_d = _safe_int(r.get('bull_div', 0))
                    bear_d = _safe_int(r.get('bear_div', 0))
                    cum_b = _safe_int(r.get('cum_bull', 0))
                    cum_r = _safe_int(r.get('cum_bear', 0))

                    # News regime based on ratio
                    if n_art == 0 or pd.isna(sent):
                        news_icon = '—'
                        news_txt = 'No data'
                    elif sent <= -0.15:
                        news_icon = '🔴'
                        news_txt = f"Bearish {sent:+.2f}"
                    elif sent >= 0.15:
                        news_icon = '🟢'
                        news_txt = f"Bullish {sent:+.2f}"
                    else:
                        news_icon = '⚪'
                        news_txt = f"Neutral {sent:+.2f}"

                    # Market reaction
                    if day_pct > 0.3:
                        mkt_icon = '🟢'
                    elif day_pct < -0.3:
                        mkt_icon = '🔴'
                    else:
                        mkt_icon = '⚪'

                    # Divergence status
                    if bull_d >= 2:
                        div_status = f"🟢 Bull +{bull_d}"
                    elif bear_d >= 2:
                        div_status = f"🔴 Bear +{bear_d}"
                    else:
                        div_status = '—'

                    breakdown_rows.append({
                        'Date': idx.strftime('%b %d'),
                        'News': f"{news_icon} {news_txt}",
                        'Articles': f"🟢{n_b} · ⚪{n_neu} · 🔴{n_r}",
                        'SPY': f"{mkt_icon} {day_pct:+.2f}%",
                        'Divergence': div_status,
                        f'Cum 10d': f"🟢{cum_b} · 🔴{cum_r}",
                    })

                bd_df = pd.DataFrame(breakdown_rows)
                st.dataframe(bd_df, use_container_width=True, hide_index=True,
                              column_config={
                                  'Articles': st.column_config.TextColumn(help="Bullish · Neutral · Bearish article counts"),
                                  'Divergence': st.column_config.TextColumn(help="Today's divergence score (bull_div or bear_div)"),
                                  'Cum 10d': st.column_config.TextColumn(help="Rolling 10-day cumulative bull vs bear score"),
                              })

                # Scoring rules reminder
                st.markdown("""
                <div style='font-size:0.78em;color:#aaa;margin-top:10px;padding:10px;background:rgba(255,255,255,0.03);border-radius:8px'>
                <b style='color:#e0e0f0'>Scoring rules recap:</b><br>
                🟢 <b>Bull divergence</b> fires when news is <b>bearish (≤-0.15)</b> AND SPY closes green. Max 6 points/day.<br>
                🔴 <b>Bear divergence</b> fires when news is <b>bullish (≥+0.15)</b> AND SPY closes red. Max 6 points/day.<br>
                <b>Cumulative</b> = rolling 10-day sum. <b>≥8</b> = Strong Reversal · <b>≥5</b> = Moderate.<br>
                <br>
                <b>The ratio of bullish vs bearish articles matters</b> — 3 bearish/0 bullish is a stronger setup than 3 bearish/3 bullish (which would average to neutral sentiment and not trigger).
                </div>
                """, unsafe_allow_html=True)

            with st.expander("📋 Recent Divergence Days (last 30 days)", expanded=False):
                recent = display_df.tail(30)
                div_events = recent[(recent['bull_div'] >= 2) | (recent['bear_div'] >= 2)].copy()
                if div_events.empty:
                    st.caption("No significant divergence days yet. Keep fetching news daily — signals appear when news sentiment diverges from market reaction.")
                else:
                    ev_rows = []
                    for idx, r in div_events.iterrows():
                        sent = r.get('avg_sentiment', None)
                        n_bull = int(r.get('n_bullish', 0) or 0)
                        n_bear = int(r.get('n_bearish', 0) or 0)
                        if r['bull_div'] >= 2:
                            kind = "🟢 Bullish"
                            score = int(r['bull_div'])
                            reason = [f"Bearish news ({sent:+.2f}) but closed {r['day_pct']:+.2f}%"]
                            if r['gap_pct'] < -0.3 and r['intraday_pct'] > 0:
                                reason.append("gap down → recovery")
                            if r['close_pos'] > 0.7:
                                reason.append("closed near high")
                        else:
                            kind = "🔴 Bearish"
                            score = int(r['bear_div'])
                            reason = [f"Bullish news ({sent:+.2f}) but closed {r['day_pct']:+.2f}%"]
                            if r['gap_pct'] > 0.3 and r['intraday_pct'] < 0:
                                reason.append("gap up → faded")
                            if r['close_pos'] < 0.3:
                                reason.append("closed near low")
                        ev_rows.append({
                            'Date': idx.strftime('%b %d'),
                            'Type': kind,
                            'Score': score,
                            'News': f"{sent:+.2f}" if pd.notna(sent) else '—',
                            'Articles': f"🟢{n_bull}/🔴{n_bear}",
                            'Day': f"{r['day_pct']:+.2f}%",
                            'Reason': ' · '.join(reason),
                        })
                    ev_df = pd.DataFrame(ev_rows)
                    st.dataframe(ev_df, use_container_width=True, hide_index=True)

            # Classified headlines log — show what's actually driving the scores
            with st.expander("📰 Recent Classified Headlines (last 7 days)", expanded=False):
                try:
                    from news import load_headlines_log
                    hl_df = load_headlines_log(n_days=7)
                    if hl_df.empty:
                        st.caption("No headlines logged yet. Click '🔄 Refresh News' to fetch.")
                    else:
                        # Format for display
                        hl_df = hl_df.copy()
                        hl_df['When'] = hl_df['timestamp'].dt.strftime('%b %d %H:%M')
                        hl_df['Sent'] = hl_df['sentiment'].apply(lambda x: f"{float(x):+.2f}")
                        hl_df['Label'] = hl_df['label'].str.replace('_', ' ').str.title()

                        # Emoji indicator
                        def _emoji(s):
                            try:
                                s = float(s)
                                if s >= 0.5: return "🟢🟢"
                                if s >= 0.1: return "🟢"
                                if s <= -0.5: return "🔴🔴"
                                if s <= -0.1: return "🔴"
                                return "⚪"
                            except: return "?"
                        hl_df['↕'] = hl_df['sentiment'].apply(_emoji)

                        display_cols = ['When', 'source', '↕', 'Sent', 'Label', 'title', 'reason']
                        hl_display = hl_df[display_cols].copy()
                        hl_display.columns = ['When', 'Source', '↕', 'Score', 'Label', 'Headline', 'Why']

                        st.dataframe(hl_display, use_container_width=True, hide_index=True,
                                      column_config={
                                          'Headline': st.column_config.TextColumn(width="large"),
                                          'Why': st.column_config.TextColumn(width="medium"),
                                      })

                        # Summary: breakdown by source
                        n_total = len(hl_df)
                        n_bull = len(hl_df[hl_df['sentiment'] >= 0.1])
                        n_bear = len(hl_df[hl_df['sentiment'] <= -0.1])
                        n_neu = n_total - n_bull - n_bear
                        st.caption(
                            f"Total: **{n_total}** headlines · "
                            f"🟢 Bullish: **{n_bull}** · 🔴 Bearish: **{n_bear}** · ⚪ Neutral: **{n_neu}** · "
                            f"Sources: {', '.join(hl_df['source'].value_counts().head(5).index.tolist())}"
                        )
                except Exception as e:
                    st.caption(f"Could not load headlines log: {e}")

            # ── Backtest: pull historical news for a date range ──
            with st.expander("📅 Backtest — Pull Historical News (GDELT)", expanded=False):
                st.caption(
                    "Pull historical news from **GDELT** (Reuters, Bloomberg, WSJ, CNBC, FT, BBC, etc.) "
                    "for a custom date range. Headlines are auto-classified by Claude and cached — "
                    "you won't pay to re-classify dates already fetched."
                )

                try:
                    from news import load_fetch_log, get_missing_dates, backfill_news_for_range

                    col_bt1, col_bt2 = st.columns(2)
                    default_start = pd.Timestamp.now().normalize() - pd.Timedelta(days=60)
                    default_end = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
                    with col_bt1:
                        bt_start = st.date_input(
                            "Start date",
                            value=default_start,
                            max_value=pd.Timestamp.now().normalize(),
                            key="bt_news_start",
                        )
                    with col_bt2:
                        bt_end = st.date_input(
                            "End date",
                            value=default_end,
                            max_value=pd.Timestamp.now().normalize(),
                            key="bt_news_end",
                        )

                    # Show cache status
                    bt_start_ts = pd.Timestamp(bt_start)
                    bt_end_ts = pd.Timestamp(bt_end)
                    total_days = (bt_end_ts - bt_start_ts).days + 1
                    missing = get_missing_dates(bt_start_ts, bt_end_ts)
                    cached = total_days - len(missing)

                    # Estimate cost: ~30 headlines/day × $0.000015 per headline (Haiku) = $0.00045/day
                    est_cost = len(missing) * 30 * 0.000015
                    est_time_sec = len(missing) * 7  # 6s rate limit + processing

                    status_color = "#3fb950" if not missing else "#e3b341"
                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:10px;margin:6px 0">
                        <div style="font-size:0.85em;color:#ccc">
                            📊 <b>Range:</b> {total_days} days
                            ({bt_start_ts.strftime('%b %d, %Y')} → {bt_end_ts.strftime('%b %d, %Y')})
                        </div>
                        <div style="font-size:0.82em;color:#aaa;margin-top:4px">
                            ✅ Already cached: <b style="color:#3fb950">{cached} days</b>
                            · ⏳ Missing: <b style="color:{status_color}">{len(missing)} days</b>
                        </div>
                        {f'<div style="font-size:0.78em;color:#888;margin-top:6px">Est. cost: ~${est_cost:.3f} · Est. time: ~{est_time_sec // 60}m {est_time_sec % 60}s (GDELT rate-limits 1 req/5s)</div>' if missing else '<div style="font-size:0.78em;color:#3fb950;margin-top:6px">All dates cached — no API calls needed.</div>'}
                    </div>
                    """, unsafe_allow_html=True)

                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        run_bt = st.button(
                            "🚀 Pull Missing Dates" if missing else "✅ All Cached",
                            disabled=(not missing),
                            key="bt_run_btn",
                            use_container_width=True,
                        )

                    if run_bt and missing:
                        prog_container = st.empty()
                        prog_bar = st.progress(0, text="Starting...")
                        with st.spinner(f"Fetching {len(missing)} days from GDELT..."):
                            def _progress(i, total, date):
                                prog_bar.progress((i + 1) / total,
                                                  text=f"Day {i+1}/{total}: {date} ({(i+1)*100//total}%)")

                            # Temporarily bypass the dict key lookup since our backfill
                            # uses get_missing_dates internally
                            result = backfill_news_for_range(bt_start_ts, bt_end_ts, progress=_progress)
                            prog_bar.empty()

                            if result['status'] == 'cached':
                                st.info(result['message'])
                            else:
                                st.success(
                                    f"✓ Fetched {result['fetched_count']} days · "
                                    f"Classified {result['n_headlines']} headlines"
                                )
                            # Clear cache so divergence recomputes with new data
                            fetch_divergence_data.clear()
                            st.rerun()

                    # Show cache log
                    fetch_log = load_fetch_log()
                    if not fetch_log.empty:
                        with st.expander(f"📜 Fetch log ({len(fetch_log)} days cached)", expanded=False):
                            display_log = fetch_log.copy()
                            display_log['Date'] = display_log['date'].dt.strftime('%Y-%m-%d')
                            display_log['Articles'] = display_log['n_articles']
                            display_log['Fetched'] = display_log['fetched_at'].dt.strftime('%m/%d %H:%M')
                            display_log['Source'] = display_log.get('source', 'gdelt')
                            st.dataframe(
                                display_log[['Date', 'Articles', 'Source', 'Fetched']]
                                    .sort_values('Date', ascending=False),
                                use_container_width=True, hide_index=True,
                            )
                except Exception as e:
                    st.error(f"Backtest panel error: {e}")

    # ── Global Liquidity Tracker ──
    with st.expander("💧 Global Liquidity Tracker — Backtested Alpha Signals", expanded=True):
        st.caption(
            "Tracks **only signals validated against 10 years of SPY data**. "
            "Each signal shows its backtested edge vs SPY baseline. "
            "This is a macro positioning tool (6-9 month horizon), not a daily trading signal. "
            "Data: FRED — free, no API key."
        )

        with st.spinner("Fetching liquidity data from FRED..."):
            liq = fetch_liquidity_signals()

        if liq.get('error'):
            st.error(f"Liquidity data error: {liq['error']}")
        else:
            nl = liq['net_liquidity']
            impulse = liq['impulse_13w']
            comp = liq['components']
            extr = liq['extremes']
            nfci = liq['nfci']
            active = liq.get('active_signals', [])
            catalog = liq.get('all_signals_catalog', {})

            # Main Net Liquidity card (informational — no regime label)
            impulse_color = "#3fb950" if impulse > 0 else "#ff4444"
            st.markdown(f"""
            <div style="border:1px solid rgba(255,255,255,0.1);background:rgba(255,255,255,0.02);border-radius:12px;padding:16px;max-width:760px">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                    <div>
                        <div style="font-size:0.82em;color:#8888a0;font-weight:600;text-transform:uppercase;letter-spacing:0.5px">Net Liquidity (Fed BS − TGA − RRP)</div>
                        <div style="font-size:2.6em;font-weight:700;color:#e0e0f0;line-height:1.1">${nl['current_trillions']:.2f}T</div>
                    </div>
                    <div style="text-align:right">
                        <div style="font-size:0.75em;color:#8888a0">Impulse (13w)</div>
                        <div style="font-size:1.6em;color:{impulse_color};font-weight:700">{impulse:+.2f}%</div>
                    </div>
                </div>
                <div style="display:flex;gap:20px;font-size:0.85em;color:#ccc;margin-top:10px;padding:8px 12px;background:rgba(255,255,255,0.03);border-radius:8px">
                    <span>1W: <b style="color:{'#3fb950' if nl['week_change_pct']>0 else '#ff6b6b'}">{nl['week_change_pct']:+.2f}%</b></span>
                    <span>1M: <b style="color:{'#3fb950' if nl['month_change_pct']>0 else '#ff6b6b'}">{nl['month_change_pct']:+.2f}%</b></span>
                    <span>3M: <b style="color:{'#3fb950' if nl['quarter_change_pct']>0 else '#ff6b6b'}">{nl['quarter_change_pct']:+.2f}%</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── ACTIVE BACKTESTED SIGNALS ──
            st.markdown("")
            st.markdown("### 🎯 Active Signals (Backtested)")

            if not active:
                st.info(
                    "⚪ **No alpha signals active.** "
                    "Waiting for: TGA > 85th pctile (currently "
                    f"{comp['tga_pctile_3y']:.0f}th), "
                    f"Impulse > +5% (currently {impulse:+.2f}%), "
                    "or RRP exhaustion (cash > $50B)."
                )
            else:
                for sig in active:
                    edge_text = (
                        f"<span style='color:#3fb950'>+{sig['edge_9m']:.2f}% edge vs baseline over 9m</span>"
                        if sig.get('edge_9m', 0) > 0 else
                        f"<span style='color:#ff6b6b'>{sig['edge_9m']:.2f}% underperformance vs baseline over 9m</span>"
                        if sig.get('edge_9m') is not None else
                        "<span style='color:#888'>edge not backtested</span>"
                    )
                    win_text = f"{sig.get('win_9m', '?')}% win rate"
                    n_text = f"n={sig.get('n_triggers_10y', '?')}"
                    st.markdown(f"""
                    <div style="border:1px solid {sig['color']}55;background:{sig['color']}10;border-radius:10px;padding:12px;margin:8px 0">
                        <div style="display:flex;justify-content:space-between;align-items:start">
                            <div>
                                <div style="font-size:1.05em;font-weight:700;color:{sig['color']}">
                                    {sig['emoji']} {sig['name']}
                                </div>
                                <div style="font-size:0.85em;color:#ccc;margin-top:4px">
                                    <b>Current:</b> {sig['current_reading']}
                                </div>
                                <div style="font-size:0.8em;color:#aaa;margin-top:4px">
                                    💡 {sig['thesis']}
                                </div>
                            </div>
                            <div style="text-align:right;font-size:0.78em;color:#888;min-width:200px">
                                <div><b style="color:#ccc">{sig['verdict']}</b></div>
                                <div style="margin-top:4px">{edge_text}</div>
                                <div>{win_text} · {n_text}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("")

            # Component cards
            col_c1, col_c2, col_c3, col_c4 = st.columns(4)

            with col_c1:
                walcl_t = comp['walcl'] / 1e6
                st.markdown(f"""
                <div style="border:1px solid rgba(255,255,255,0.08);background:rgba(255,255,255,0.02);border-radius:10px;padding:12px">
                    <div style="font-size:0.72em;color:#8888a0">FED BALANCE SHEET</div>
                    <div style="font-size:1.4em;font-weight:700;color:#e0e0f0">${walcl_t:.2f}T</div>
                    <div style="font-size:0.7em;color:#888;margin-top:4px">{comp['walcl_date']}</div>
                </div>
                """, unsafe_allow_html=True)

            with col_c2:
                tga_t = comp['tga'] / 1e6
                tga_sig = extr.get('tga', {})
                tga_color = '#3fb950' if tga_sig.get('signal') == 'drain_imminent' else '#ff4444' if tga_sig.get('signal') == 'refill_imminent' else '#e0e0f0'
                st.markdown(f"""
                <div style="border:1px solid rgba(255,255,255,0.08);background:rgba(255,255,255,0.02);border-radius:10px;padding:12px">
                    <div style="font-size:0.72em;color:#8888a0">TGA (Treasury Acct)</div>
                    <div style="font-size:1.4em;font-weight:700;color:{tga_color}">${tga_t:.2f}T</div>
                    <div style="font-size:0.7em;color:#888;margin-top:4px">{tga_sig.get('pctile_3y', 0):.0f}th pctile (3Y)</div>
                </div>
                """, unsafe_allow_html=True)

            with col_c3:
                rrp_b = comp['rrp_billions']
                rrp_sig = extr.get('rrp', {})
                rrp_color = '#ff4444' if rrp_sig.get('signal') == 'exhausted' else '#3fb950' if rrp_sig.get('signal') == 'flowing' else '#e0e0f0'
                st.markdown(f"""
                <div style="border:1px solid rgba(255,255,255,0.08);background:rgba(255,255,255,0.02);border-radius:10px;padding:12px">
                    <div style="font-size:0.72em;color:#8888a0">REVERSE REPO</div>
                    <div style="font-size:1.4em;font-weight:700;color:{rrp_color}">${rrp_b:.1f}B</div>
                    <div style="font-size:0.7em;color:#888;margin-top:4px">{rrp_sig.get('pct_of_peak', 0):.0f}% of 3Y peak</div>
                </div>
                """, unsafe_allow_html=True)

            with col_c4:
                nfci_val = nfci.get('value')
                nfci_color = '#ff4444' if nfci_val and nfci_val > 0 else '#3fb950' if nfci_val and nfci_val < -0.3 else '#e0e0f0'
                st.markdown(f"""
                <div style="border:1px solid rgba(255,255,255,0.08);background:rgba(255,255,255,0.02);border-radius:10px;padding:12px">
                    <div style="font-size:0.72em;color:#8888a0">NFCI (Financial Cond.)</div>
                    <div style="font-size:1.4em;font-weight:700;color:{nfci_color}">{nfci_val:+.2f}</div>
                    <div style="font-size:0.7em;color:#888;margin-top:4px">{'Tight' if nfci_val > 0 else 'Loose'}</div>
                </div>
                """, unsafe_allow_html=True)

            # Extreme readings alert card
            alerts = []
            for k, v in extr.items():
                if v.get('signal') not in (None, 'normal', 'neutral'):
                    alerts.append(v.get('interpretation', ''))

            if alerts:
                st.markdown(f"""
                <div style="background:rgba(227,179,65,0.08);border:1px solid rgba(227,179,65,0.25);border-radius:8px;padding:10px;margin:10px 0">
                    <div style="font-size:0.8em;color:#e3b341;font-weight:600;margin-bottom:6px">⚠️ EXTREME READINGS</div>
                    {'<br>'.join(f'<div style="font-size:0.82em;color:#ccc">{a}</div>' for a in alerts)}
                </div>
                """, unsafe_allow_html=True)

            # NFCI interpretation
            if nfci.get('interpretation'):
                st.caption(f"**Financial Conditions**: {nfci['interpretation']}")

            st.markdown("---")

            # ── Chart: Net Liquidity + SPY overlay ──
            st.markdown("**📊 Net Liquidity vs SPY — visual correlation check**")
            net_liq_df = liq['net_liquidity_series']
            spy_series = liq['spy']

            # Period selector
            liq_period = st.selectbox(
                "Period",
                ['1Y', '2Y', '3Y', '5Y'],
                index=1,
                key='liq_period',
            )
            yrs = {'1Y': 1, '2Y': 2, '3Y': 3, '5Y': 5}[liq_period]
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=yrs)

            nl_chart = net_liq_df[net_liq_df.index >= cutoff]
            spy_chart = spy_series[spy_series.index >= cutoff] if not spy_series.empty else pd.Series()

            from plotly.subplots import make_subplots
            fig_liq = make_subplots(specs=[[{"secondary_y": True}]])

            fig_liq.add_trace(go.Scatter(
                x=nl_chart.index, y=nl_chart['net_liquidity'] / 1e6,  # Trillions
                mode='lines', name='Net Liquidity ($T)',
                line=dict(color='#58a6ff', width=2),
            ), secondary_y=False)

            if not spy_chart.empty:
                fig_liq.add_trace(go.Scatter(
                    x=spy_chart.index, y=spy_chart.values,
                    mode='lines', name='SPY Price',
                    line=dict(color='#3fb950', width=1.5),
                ), secondary_y=True)

            fig_liq.update_layout(
                template='plotly_dark', paper_bgcolor='#07070d', plot_bgcolor='#07070d',
                height=400, margin=dict(l=50, r=50, t=30, b=30),
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            )
            fig_liq.update_yaxes(title_text="Net Liquidity ($T)", secondary_y=False, color='#58a6ff')
            fig_liq.update_yaxes(title_text="SPY ($)", secondary_y=True, color='#3fb950')
            st.plotly_chart(fig_liq, use_container_width=True)

            # ── Chart: Liquidity Impulse (13w) ──
            st.markdown("**⚡ Liquidity Impulse — 13-week rate of change**")

            # Show current impulse velocity stats
            accel_12w = liq.get('impulse_accel_12w', 0)
            accel_16w = liq.get('impulse_accel_16w', 0)
            accel_fired = liq.get('accel_trigger_date')
            velocity_color = '#3fb950' if accel_12w >= 10 else '#e3b341' if accel_12w >= 5 else '#8888a0'

            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:8px 12px;margin-bottom:8px;font-size:0.82em;color:#ccc">
                <b>Current level</b>: {impulse:+.2f}% (above +5% is the absolute alpha threshold)<br>
                <b>12w velocity</b>: <span style="color:{velocity_color}">{accel_12w:+.2f}pp</span> (above +10pp fires the acceleration signal — <b>+2.46% 9m edge, 83% win</b>)
                {f"<br><b>Acceleration signal last fired</b>: {accel_fired}" if accel_fired else ""}
            </div>
            """, unsafe_allow_html=True)
            st.caption(
                "**Two alpha thresholds**: (1) absolute >+5% · (2) velocity >+10pp in 12w. "
                "The velocity signal catches rapid policy pivots (Mar 2020 QE, Oct 2022 TGA drain, Nov 2022 pivot). "
                "Both have documented alpha. Readings below these thresholds = noise."
            )
            impulse_chart = liq['impulse_series']
            impulse_chart = impulse_chart[impulse_chart.index >= cutoff].dropna()

            fig_imp = go.Figure()
            fig_imp.add_trace(go.Scatter(
                x=impulse_chart.index, y=impulse_chart.values,
                mode='lines', name='Impulse (13w)',
                line=dict(color='#a78bfa', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(167,139,250,0.1)',
            ))
            # Alpha thresholds
            fig_imp.add_hline(y=5, line_dash="dash", line_color="#3fb950", opacity=0.5,
                              annotation_text="Level threshold (+5%)", annotation_position="top left",
                              annotation_font_color="#3fb950")
            fig_imp.add_hline(y=0, line_color="#444", opacity=0.3, line_width=1)

            # ── Mark velocity signal firings with triangle markers ──
            velocity_fires = liq.get('velocity_fires_history', [])
            massive_fires = liq.get('massive_bounce_fires_history', [])

            # Filter to display period + get impulse value at each fire date
            fires_in_period = []
            for fire_str in velocity_fires:
                fire_date = pd.Timestamp(fire_str)
                if fire_date >= cutoff and fire_date in impulse_chart.index:
                    fires_in_period.append((fire_date, impulse_chart.loc[fire_date]))
                elif fire_date >= cutoff:
                    closest = impulse_chart.index[impulse_chart.index.searchsorted(fire_date)] if len(impulse_chart) else None
                    if closest is not None:
                        fires_in_period.append((fire_date, impulse_chart.loc[closest]))

            if fires_in_period:
                fig_imp.add_trace(go.Scatter(
                    x=[d for d, _ in fires_in_period],
                    y=[v for _, v in fires_in_period],
                    mode='markers',
                    marker=dict(color='#3fb950', size=14, symbol='triangle-up',
                                line=dict(color='white', width=1.5)),
                    name='🟢 Velocity Fire (+10pp/12w)',
                    hovertemplate='<b>%{x|%b %d, %Y}</b><br>Velocity signal fired<br>Impulse: %{y:+.2f}%<extra></extra>',
                ))

            massive_in_period = []
            for fire_str in massive_fires:
                fire_date = pd.Timestamp(fire_str)
                if fire_date >= cutoff and fire_date in impulse_chart.index:
                    massive_in_period.append((fire_date, impulse_chart.loc[fire_date]))
                elif fire_date >= cutoff:
                    closest = impulse_chart.index[impulse_chart.index.searchsorted(fire_date)] if len(impulse_chart) else None
                    if closest is not None:
                        massive_in_period.append((fire_date, impulse_chart.loc[closest]))

            if massive_in_period:
                fig_imp.add_trace(go.Scatter(
                    x=[d for d, _ in massive_in_period],
                    y=[v for _, v in massive_in_period],
                    mode='markers',
                    marker=dict(color='#00ff88', size=20, symbol='star',
                                line=dict(color='white', width=2)),
                    name='🚀 Massive Bounce (+15pp/16w)',
                    hovertemplate='<b>%{x|%b %d, %Y}</b><br>Massive bounce fired<br>Impulse: %{y:+.2f}%<extra></extra>',
                ))

            # Summary
            n_fires_in_period = len(fires_in_period)
            n_massive_in_period = len(massive_in_period)

            # Summary of historical fires
            n_fires_in_period = len(fires_in_period)
            n_massive_in_period = len(massive_in_period)

            fig_imp.update_layout(
                template='plotly_dark', paper_bgcolor='#07070d', plot_bgcolor='#07070d',
                height=320, margin=dict(l=50, r=20, t=30, b=30),
                yaxis_title='13w % change',
                showlegend=bool(fires_in_period or massive_in_period),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            if n_fires_in_period or n_massive_in_period:
                st.caption(
                    f"📍 In this period: **{n_fires_in_period}** velocity signal fires · "
                    f"**{n_massive_in_period}** massive bounces"
                )

            # ── Global M2 → BTC tracker ──
            m2_btc = liq['m2_btc']
            btc_series = liq['btc']
            if not m2_btc.empty and not btc_series.empty:
                with st.expander("₿ Global M2 → BTC Impulse (12-week leading indicator)", expanded=False):
                    st.caption(
                        "**Documented alpha source**: Global M2 YoY leads BTC by ~12 weeks with strong correlation. "
                        "When M2 rises, BTC follows about 3 months later. Used by macro traders like Raoul Pal."
                    )

                    # Chart
                    fig_m2 = make_subplots(specs=[[{"secondary_y": True}]])

                    # Resample BTC to monthly for cleaner chart
                    btc_m = btc_series.resample('ME').last()
                    btc_recent = btc_m[btc_m.index >= cutoff]
                    m2_recent = liq['global_m2_series']
                    m2_recent = m2_recent[m2_recent.index >= cutoff]

                    fig_m2.add_trace(go.Scatter(
                        x=m2_recent.index, y=m2_recent.values / 1e6,  # Trillions
                        mode='lines', name='Global M2 ($T)',
                        line=dict(color='#58a6ff', width=2),
                    ), secondary_y=False)

                    fig_m2.add_trace(go.Scatter(
                        x=btc_recent.index, y=btc_recent.values,
                        mode='lines', name='BTC Price',
                        line=dict(color='#f7931a', width=1.5),
                    ), secondary_y=True)

                    fig_m2.update_layout(
                        template='plotly_dark', paper_bgcolor='#07070d', plot_bgcolor='#07070d',
                        height=350, margin=dict(l=50, r=50, t=30, b=30),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    )
                    fig_m2.update_yaxes(title_text="Global M2 ($T)", secondary_y=False, color='#58a6ff')
                    fig_m2.update_yaxes(title_text="BTC ($)", secondary_y=True, color='#f7931a')
                    st.plotly_chart(fig_m2, use_container_width=True)

                    # Current M2 status
                    if not m2_btc.empty:
                        latest_m2_yoy = m2_btc['m2_yoy'].iloc[-1] if pd.notna(m2_btc['m2_yoy'].iloc[-1]) else 0
                        signal_text = (
                            f"🟢 **M2 expanding at {latest_m2_yoy:+.2f}% YoY** — bullish for BTC over next 3 months"
                            if latest_m2_yoy > 2 else
                            f"🔴 **M2 contracting at {latest_m2_yoy:+.2f}% YoY** — bearish for BTC over next 3 months"
                            if latest_m2_yoy < 0 else
                            f"⚪ **M2 sideways at {latest_m2_yoy:+.2f}% YoY** — neutral for BTC"
                        )
                        st.markdown(signal_text)

            # ── SPY / Liquidity Divergence ──
            div = liq['divergence']
            div_series = liq['divergence_series']
            if div != 'none':
                div_color = '#ff4444' if div == 'bearish' else '#3fb950'
                div_msg = (
                    "🔴 **BEARISH DIVERGENCE**: SPY is rising while Net Liquidity is contracting. "
                    "Distribution signal — rally may not be sustainable."
                    if div == 'bearish' else
                    "🟢 **BULLISH DIVERGENCE**: SPY is falling while Net Liquidity is expanding. "
                    "Accumulation signal — buying opportunity."
                )
                st.markdown(f"""
                <div style="background:{div_color}15;border:1px solid {div_color}50;border-radius:8px;padding:12px;margin:10px 0">
                    <div style="color:{div_color};font-size:0.9em">{div_msg}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Full Signal Catalog with Backtest Stats ──
            with st.expander("📊 Signal Catalog — Backtest Results (2015-2026, 10yr SPY)", expanded=False):
                st.caption(
                    "All 4 signals tracked, with honest backtest results against SPY. "
                    "Baseline (random day) 9m return = **+10.66%** / 84% win rate. "
                    "Only signals beating this have real alpha."
                )
                cat_rows = []
                for key, sig in catalog.items():
                    edge = sig.get('edge_9m')
                    edge_str = f"{edge:+.2f}%" if edge is not None else '—'
                    cat_rows.append({
                        'Signal': sig['name'],
                        'Verdict': sig['verdict'],
                        '3m avg': f"{sig.get('ret_3m', 0):+.2f}%" if sig.get('ret_3m') else '—',
                        '6m avg': f"{sig.get('ret_6m', 0):+.2f}%" if sig.get('ret_6m') else '—',
                        '9m avg': f"{sig.get('ret_9m', 0):+.2f}%" if sig.get('ret_9m') else '—',
                        '9m win%': f"{sig.get('win_9m', 0):.0f}%" if sig.get('win_9m') else '—',
                        'Edge vs BL': edge_str,
                        'N (10y)': str(sig.get('n_triggers_10y', '—')),
                    })
                import pandas as _pd
                cat_df = _pd.DataFrame(cat_rows)
                st.dataframe(cat_df, use_container_width=True, hide_index=True)

                st.markdown("""
                <div style='font-size:0.8em;color:#aaa;margin-top:10px'>
                <b style='color:#e0e0f0'>Signals removed from dashboard (failed backtest):</b><br>
                • <s>Impulse zero-crossings (up or down)</s> — too noisy (62-64 triggers in 10y), underperforms baseline<br>
                • <s>Regime Easing/Tightening labels</s> — fast/slow MA crossover gives no alpha<br>
                • <s>Generic "Impulse positive = bullish"</s> — only matters above +5% threshold
                </div>
                """, unsafe_allow_html=True)

            # Source timestamps
            st.caption(
                f"**Data updates**: Fed BS (weekly Wed, latest {comp['walcl_date']}) · "
                f"TGA (weekly, {comp['tga_date']}) · "
                f"RRP (daily, {comp['rrp_date']}) · "
                f"Source: FRED (St. Louis Fed) via pandas-datareader · "
                f"Backtest period: 2015-2026"
            )

    # ── S&P 500 Market Breadth (slow — collapsed by default) ──
    with st.expander("📡 S&P 500 — Market Breadth (% Above 20-DMA)", expanded=False):
        breadth_df = fetch_sp500_breadth()
        if not breadth_df.empty:
            bval = breadth_df['pct_above_20dma'].iloc[-1]
            bdate = breadth_df['date'].iloc[-1].strftime('%Y-%m-%d')

            # Define zones
            if bval < 15:
                b_zone = "🟢 Opportunity Zone"
                b_color = "#3fb950"
                b_bg = "rgba(63,185,80,0.08)"
                b_border = "rgba(63,185,80,0.25)"
                b_action = "DCA In — historically 72% win @1m, +6% avg @3m"
            elif bval < 25:
                b_zone = "🟡 Warming Up"
                b_color = "#e3b341"
                b_bg = "rgba(227,179,65,0.05)"
                b_border = "rgba(227,179,65,0.2)"
                b_action = "Start DCA — breadth recovering from oversold"
            elif bval > 85:
                b_zone = "🔴 Fully Extended"
                b_color = "#ff6b6b"
                b_bg = "rgba(255,107,107,0.08)"
                b_border = "rgba(255,107,107,0.25)"
                b_action = "Stop adding — breadth at euphoric levels"
            elif bval > 75:
                b_zone = "🟠 Elevated"
                b_color = "#f0883e"
                b_bg = "rgba(240,136,62,0.05)"
                b_border = "rgba(240,136,62,0.2)"
                b_action = "Reduce new positions — limited upside edge"
            else:
                b_zone = "⚪ Neutral"
                b_color = "#8888a0"
                b_bg = "rgba(136,136,160,0.03)"
                b_border = "rgba(136,136,160,0.15)"
                b_action = "No edge — hold current positions"

            # Card
            st.markdown(f"""
            <div style="border:1px solid {b_border};background:{b_bg};border-radius:12px;padding:16px;max-width:600px">
                <div style="font-size:0.85em;color:#8888a0;font-weight:600;text-transform:uppercase;letter-spacing:0.5px">
                    S&P 500 Breadth — % Stocks Above 20-DMA
                </div>
                <div style="font-size:2.2em;font-weight:700;color:{b_color};margin:4px 0">{bval:.1f}%</div>
                <div style="font-size:0.8em;color:#8888a0;margin-bottom:8px">
                    {b_zone}
                </div>
                <div style="background:rgba(255,255,255,0.05);border-radius:6px;height:8px;overflow:hidden;margin:8px 0">
                    <div style="width:{min(bval, 100):.0f}%;height:100%;background:{b_color};border-radius:6px"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.75em;color:#666">
                    <span>0%</span><span style="color:{b_color}">{b_zone.split(' ', 1)[1] if ' ' in b_zone else ''}</span><span>100%</span>
                </div>
                <div style="font-size:0.78em;color:#aaa;margin-top:10px">
                    📊 {b_action}<br>
                    📅 As of {bdate} · 503 constituents
                </div>
                <details style="margin-top:8px;font-size:0.75em;color:#888">
                    <summary style="cursor:pointer">ℹ️ Backtest stats (2007–2026)</summary>
                    <div style="margin-top:6px;line-height:1.6">
                        <b>&lt;10%</b>: 72% win @1m (+2.9%), 73% win @3m (+7.3%)<br>
                        <b>&lt;15%</b>: 72% win @1m (+2.6%), 71% win @3m (+6.0%)<br>
                        <b>&lt;20%</b>: 72% win @1m (+2.5%), 73% win @3m (+5.8%)<br>
                        <b>&gt;85%</b>: Forward returns drop to +0.1% @1w — no edge for new DCA
                    </div>
                </details>
            </div>
            """, unsafe_allow_html=True)

            # Chart: breadth with period selector
            st.markdown("**Historical Chart:**")
            period_presets = {"1Y": 1, "2Y": 2, "5Y": 5, "10Y": 10, "Max": None, "Custom": -1}
            pcol1, pcol2, pcol3 = st.columns([2, 1.5, 1.5])
            with pcol1:
                period_choice = st.selectbox("Period", list(period_presets.keys()),
                                             index=1, key="breadth_period")
            max_date = breadth_df['date'].max()
            min_date = breadth_df['date'].min()
            if period_choice == "Custom":
                with pcol2:
                    custom_start = st.date_input("From", value=max_date - pd.DateOffset(years=2),
                                                  min_value=min_date, max_value=max_date,
                                                  key="breadth_start")
                with pcol3:
                    custom_end = st.date_input("To", value=max_date,
                                                min_value=min_date, max_value=max_date,
                                                key="breadth_end")
                chart_df = breadth_df[(breadth_df['date'] >= pd.Timestamp(custom_start)) &
                                      (breadth_df['date'] <= pd.Timestamp(custom_end))]
            else:
                years = period_presets[period_choice]
                if years is None:
                    chart_df = breadth_df
                else:
                    cutoff = max_date - pd.DateOffset(years=years)
                    chart_df = breadth_df[breadth_df['date'] >= cutoff]

            fig_b = go.Figure()
            fig_b.add_trace(go.Scatter(
                x=chart_df['date'], y=chart_df['pct_above_20dma'],
                mode='lines', name='% Above 20-DMA',
                line=dict(color='#58a6ff', width=1.5),
                fill='tozeroy', fillcolor='rgba(88,166,255,0.08)',
            ))
            fig_b.add_hline(y=15, line_dash="dash", line_color="#3fb950", opacity=0.5,
                            annotation_text="Opportunity (<15%)", annotation_position="bottom left",
                            annotation_font_color="#3fb950")
            fig_b.add_hline(y=85, line_dash="dash", line_color="#ff6b6b", opacity=0.5,
                            annotation_text="Extended (>85%)", annotation_position="top left",
                            annotation_font_color="#ff6b6b")
            fig_b.update_layout(
                template='plotly_dark', paper_bgcolor='#07070d', plot_bgcolor='#07070d',
                height=350, margin=dict(l=50, r=20, t=30, b=30),
                yaxis=dict(title='%', range=[0, 100]),
                showlegend=False,
            )
            st.plotly_chart(fig_b, use_container_width=True)
        else:
            st.info("Loading S&P 500 breadth data... (first load downloads 503 tickers, may take ~30s)")

    # ── Forward P/E ──
    with st.expander("📐 S&P 500 — Valuation (Forward P/E + Shiller CAPE)", expanded=True):
        fpe_data = fetch_forward_pe()
        fpe_val = fpe_data.get('forward_pe')

        if fpe_val:
            # Zones from the user's scatter plot chart
            if fpe_val < 17:
                f_zone, f_color = "🟢 Below 17x — Opportunity", "#3fb950"
                f_bg, f_border = "rgba(63,185,80,0.08)", "rgba(63,185,80,0.25)"
                f_action = "Historically 10-20%+ annualized 10yr returns. Best entry zone."
            elif fpe_val < 23:
                f_zone, f_color = "🟡 17-23x — Fair Value", "#e3b341"
                f_bg, f_border = "rgba(227,179,65,0.05)", "rgba(227,179,65,0.2)"
                f_action = "Expected 0-10% annualized 10yr returns. Neutral — hold positions."
            else:
                f_zone, f_color = "🔴 23x+ — Distribution / Danger Zone", "#ff6b6b"
                f_bg, f_border = "rgba(255,107,107,0.08)", "rgba(255,107,107,0.25)"
                f_action = "Historically marks distribution phases. Reduce risk, expect pullback."

            f_pct = min(max((fpe_val - 8) / 22 * 100, 0), 100)  # map 8-30 to 0-100%

            st.markdown(f"""
            <div style="border:1px solid {f_border};background:{f_bg};border-radius:12px;padding:16px;max-width:600px">
                <div style="font-size:0.85em;color:#8888a0;font-weight:600;text-transform:uppercase;letter-spacing:0.5px">
                    S&P 500 Forward P/E Ratio
                </div>
                <div style="font-size:2.2em;font-weight:700;color:{f_color};margin:4px 0">{fpe_val:.1f}x</div>
                <div style="font-size:0.8em;color:#8888a0;margin-bottom:8px">
                    {f_zone}
                </div>
                <div style="background:rgba(255,255,255,0.05);border-radius:6px;height:8px;overflow:hidden;margin:8px 0">
                    <div style="width:{f_pct:.0f}%;height:100%;background:{f_color};border-radius:6px"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.75em;color:#666">
                    <span>8x</span>
                    <span style="color:#3fb950">17x</span>
                    <span style="color:#e3b341">20x</span>
                    <span style="color:#ff6b6b">23x</span>
                    <span>30x</span>
                </div>
                <div style="font-size:0.78em;color:#aaa;margin-top:10px">
                    📊 {f_action}<br>
                    📅 Source: WSJ (S&P Dow Jones Indices) · Updated daily
                </div>
                <details style="margin-top:8px;font-size:0.75em;color:#888">
                    <summary style="cursor:pointer">ℹ️ Zone reference (from forward P/E vs 10yr return scatter)</summary>
                    <div style="margin-top:6px;line-height:1.6">
                        <b>&lt;17x</b>: 10-20%+ annualized 10yr returns — best entry<br>
                        <b>17–23x</b>: 0-10% annualized — fair value, muted returns<br>
                        <b>23x+</b>: Distribution / danger zone — historically precedes pullbacks<br>
                        <span style="color:#aaa">Based on S&P 500 forward P/E vs subsequent 10-year total returns</span>
                    </div>
                </details>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.caption("Forward P/E data unavailable — WSJ may be temporarily inaccessible.")

def main():
    # Obsidian Glass — global CSS injection
    st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #07070d !important; }
h1, h2, h3, .stSubheader { color: #e0e0f0 !important; font-weight: 700 !important; }
.stCaption { color: #7777a0 !important; }
.stExpander { border: 1px solid rgba(255,255,255,0.06) !important; border-radius: 12px !important; background: rgba(255,255,255,0.02) !important; }
div[data-testid="stExpander"] details { border: none !important; }
div[data-testid="stExpander"] summary { color: #8888a0 !important; }
section[data-testid="stSidebar"] { background: #0a0a14 !important; border-right: 1px solid rgba(255,255,255,0.06) !important; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px !important; color: #8888a0 !important; }
.stTabs [aria-selected="true"] { background: rgba(255,255,255,0.06) !important; color: #f0f0f5 !important; }
</style>""", unsafe_allow_html=True)

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
        st.caption("🟢 IGV / KBE — Yahoo Finance (live)")
        if FRED_KEY:
            st.caption("🟢 Yield curve / HY spread — FRED API (live)")
        else:
            st.caption("🟡 Yield curve / HY spread — proxy mode (add FRED key for exact data)")
        st.caption("🟡 AAII / NAAIM — last weekly CSV reading")
        st.caption("🟡 Put/Call Ratio — CBOE daily data (2003–present)")

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
    sig = compute_signals(market, fred, aaii_row, naaim_row, aaii_override, naaim_override)

    # ── Header ────────────────────────────────────────────────────────────
    st.title("🐺 Market Signal Dashboard")
    col_hdr1, col_hdr2, col_hdr3 = st.columns([3, 1, 0.8])
    with col_hdr1:
        st.caption(f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} · Backtest: 2010–2026 · 15yr dataset")
    with col_hdr2:
        st.caption("Strategy: Contrarian Sentiment + COT + Macro Gate")
    with col_hdr3:
        if st.button("🔄 Refresh Data", key="refresh_data_btn", help="Fetch latest NAAIM, AAII & Put/Call data and clear caches"):
            with st.spinner("Refreshing..."):
                results = []
                ok, msg = refresh_naaim_csv()
                results.append(("NAAIM", ok, msg))
                ok2, msg2 = refresh_aaii_csv()
                results.append(("AAII", ok2, msg2))
                ok3, msg3 = refresh_putcall_csv()
                results.append(("Put/Call", ok3, msg3))
                # Clear all caches to pick up new data
                st.cache_data.clear()
                for name, ok, msg in results:
                    if ok:
                        st.success(msg)
                    else:
                        st.warning(msg)
                st.rerun()

    st.markdown("---")

    # ── Pre-compute progress for all indicators ─────────────────────────
    conf     = sig['confluence']
    macro    = sig['macro_score']
    macro_ok = sig['macro_bullish']

    aaii_val  = sig.get('aaii_spread')
    naaim_val = sig.get('naaim')
    vix_val   = sig.get('vix')
    pc_10d    = sig.get('pc_10d_ma')

    progress_dict = {
        'aaii_fired': compute_signal_progress(aaii_val * 100 if aaii_val is not None else None, 0, -20, "lower"),
        'naaim_fired': compute_signal_progress(naaim_val, 80, 40, "lower"),
        'vix_fired': compute_signal_progress(vix_val, 15, 30, "higher"),
        'pc_10d_fired': compute_signal_progress(pc_10d, 0.50, 0.70, "higher"),
        'cot_tnote10_fired': compute_signal_progress(sig.get('cot_tnote10_comm_pctile'), 0.50, 0.10, "lower") if sig.get('cot_tnote10_comm_pctile') is not None else 0.0,
        'cot_sp500_fired': compute_signal_progress(sig.get('cot_sp500_spec_pctile'), 0.50, 0.05, "lower") if sig.get('cot_sp500_spec_pctile') is not None else 0.0,
        'cot_gold_fired': compute_signal_progress(sig.get('cot_gold_spec_pctile'), 0.50, 0.10, "lower") if sig.get('cot_gold_spec_pctile') is not None else 0.0,
        'cot_crude_fired': compute_signal_progress(sig.get('cot_crude_spec_pctile'), 0.50, 0.05, "lower") if sig.get('cot_crude_spec_pctile') is not None else 0.0,
        'cot_usdx_fired': compute_signal_progress(sig.get('cot_usdx_spec_pctile'), 0.50, 0.10, "lower") if sig.get('cot_usdx_spec_pctile') is not None else 0.0,
    }

    # ── Last Fired Context (cached) ──
    last_fired = get_last_fired_context()

    # ── Signal History Timeline (cached) ──
    timeline_data = build_signal_timeline()

    # ── Main Tabs (right at top) ──────────────────────────────────────────
    main_tab1, main_tab2, main_tab3 = st.tabs(["📊 Sentiment Dashboard", "🏦 COT Market", "🧪 Technical Lab"])

    with main_tab1:
        # Sentiment-only action panel (no COT signals)
        sentiment_progress = {k: v for k, v in progress_dict.items() if not k.startswith('cot_')}
        tier, active_signals, warming_signals = determine_action_tier(sig, sentiment_progress, sentiment_only=True)
        render_action_panel(tier, active_signals, warming_signals, sig)
        _render_sentiment_tab(sig, conf, macro, macro_ok, progress_dict, last_fired, timeline_data)
    with main_tab2:
        _render_cot_tab(sig, progress_dict, last_fired)
    with main_tab3:
        _render_technical_lab()

if __name__ == "__main__":
    main()
