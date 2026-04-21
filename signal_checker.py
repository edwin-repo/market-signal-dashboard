#!/usr/bin/env python3
"""
signal_checker.py — Daily signal check for Telegram alerts
Run via OpenClaw cron. Sends alert only when confluence ≥ 2 or extreme readings.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'backtest' / 'data'

AAII_SPREAD_THRESHOLD = -0.20
NAAIM_THRESHOLD       = 40
VIX_THRESHOLD         = 30
VIX_HIGH              = 40
VIX_EXTREME           = 50
FG_THRESHOLD          = 25
MACRO_GATE_MIN        = 4
ROC_DAYS              = 20
PC_10D_THRESHOLD      = 0.70
PC_30D_THRESHOLD      = 0.65

FRED_KEY       = os.environ.get("FRED_API_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT  = os.environ.get("TELEGRAM_CHAT_ID", "")


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


def fetch_all():
    end   = datetime.today()
    start = end - timedelta(days=400)  # need 400 calendar days for 200-day MA
    tickers = {'^VIX': 'vix', 'SPY': 'spy', 'DX-Y.NYB': 'dxy',
               'HG=F': 'copper', '^TNX': 'yield10', 'HYG': 'hyg'}
    market = {}
    for ticker, key in tickers.items():
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if not df.empty:
                market[key] = df['Close'].dropna()
        except Exception:
            pass
    return market


def load_latest(filename, col):
    path = DATA_DIR / filename
    if not path.exists():
        return None, None
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.dropna(subset=[col]).sort_values('date')
    if df.empty:
        return None, None
    row = df.iloc[-1]
    return float(row[col]), str(row['date'])[:10]


def roc(series, days=ROC_DAYS):
    if series is None or len(series) < days + 1:
        return None
    return float(series.iloc[-1]) - float(series.iloc[-(days+1)])


def fetch_fg():
    try:
        r = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=10
        )
        return round(float(r.json()['fear_and_greed']['score']), 1)
    except Exception:
        return None


def run_check():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Running signal check...")

    market = fetch_all()

    aaii_spread, aaii_date = load_latest('aaii.csv', 'bull_bear_spread')
    naaim_val,   naaim_date = load_latest('naaim.csv', 'naaim')

    vix    = float(market['vix'].iloc[-1].item()) if 'vix' in market else None
    dxy    = market.get('dxy')
    copper = market.get('copper')
    spy    = market.get('spy')
    y10    = market.get('yield10')
    hyg    = market.get('hyg')

    # ── Signal checks ────────────────────────────────────────────────────
    aaii_fired  = aaii_spread is not None  and aaii_spread  < AAII_SPREAD_THRESHOLD
    naaim_fired = naaim_val   is not None  and naaim_val    < NAAIM_THRESHOLD
    vix_fired   = vix         is not None  and vix          > VIX_THRESHOLD
    vix_high    = vix         is not None  and vix          > VIX_HIGH
    vix_extreme = vix         is not None  and vix          > VIX_EXTREME
    confluence  = sum([aaii_fired, naaim_fired, vix_fired])

    # ── Macro gate ───────────────────────────────────────────────────────
    fred_curve = fred_fetch('T10Y2Y')
    fred_hy    = fred_fetch('BAMLH0A0HYM2')

    macro_rates   = roc(y10) is not None    and roc(y10)    < 0
    macro_dxy     = roc(dxy) is not None    and roc(dxy)    < 0
    macro_copper  = roc(copper) is not None and roc(copper) > 0
    macro_hy      = (roc(fred_hy['value'] if fred_hy is not None else None) or
                     roc(hyg)) is not None and (
                     (fred_hy is not None and roc(fred_hy['value']) < 0) or
                     (fred_hy is None     and roc(hyg) > 0))
    curve_val     = float(fred_curve['value'].iloc[-1]) if fred_curve is not None else None
    macro_curve   = curve_val is not None and curve_val > 0

    spy_ma200    = float(spy.rolling(200).mean().iloc[-1]) if spy is not None and len(spy) >= 200 else None
    macro_breath = spy is not None and spy_ma200 is not None and float(spy.iloc[-1]) > spy_ma200

    macro_score   = sum([macro_rates, macro_dxy, macro_copper, macro_hy, macro_curve, macro_breath])
    macro_bullish = macro_score >= MACRO_GATE_MIN

    fg_score = fetch_fg()
    fg_fired = fg_score is not None and fg_score < FG_THRESHOLD

    # ── Meisler: Put/Call ratio ─────────────────────────────────────────
    pc_10d_ma = None
    pc_30d_ma = None
    pc_10d_fired = False
    pc_30d_fired = False
    try:
        pc_path = Path(__file__).parent / 'data' / 'putcall.csv'
        if pc_path.exists():
            pc_df = pd.read_csv(pc_path, parse_dates=['date']).sort_values('date')
            pc_df['pc_10d_ma'] = pc_df['equity_pc_ratio'].rolling(10, min_periods=8).mean()
            pc_df['pc_30d_ma'] = pc_df['equity_pc_ratio'].rolling(30, min_periods=25).mean()
            if not pc_df.empty:
                pc_10d_ma = float(pc_df['pc_10d_ma'].iloc[-1])
                pc_30d_ma = float(pc_df['pc_30d_ma'].iloc[-1])
                pc_10d_fired = pc_10d_ma > PC_10D_THRESHOLD
                pc_30d_fired = pc_30d_ma > PC_30D_THRESHOLD
    except Exception:
        pass

    # ── Meisler: AAII Bulls < 25% ─────────────────────────────────────
    aaii_bulls_low = False
    try:
        aaii_path = Path(__file__).parent / 'data' / 'aaii.csv'
        if aaii_path.exists():
            aaii_df = pd.read_csv(aaii_path, parse_dates=['date']).dropna(subset=['bullish']).sort_values('date')
            if not aaii_df.empty:
                last_bulls = float(aaii_df['bullish'].iloc[-1])
                if last_bulls <= 1.0:
                    last_bulls *= 100
                aaii_bulls_low = last_bulls < 25
    except Exception:
        pass

    # ── Determine if alert needed ─────────────────────────────────────────
    # Alert when: confluence ≥ 2, or VIX extreme, or any extreme individual reading
    should_alert = (
        confluence >= 2 or
        vix_extreme or
        (naaim_val is not None and naaim_val < 25) or
        (aaii_spread is not None and aaii_spread < -0.30) or
        fg_fired or
        (pc_10d_fired and pc_30d_fired) or
        aaii_bulls_low
    )

    # Build message
    def tick(b): return "✅" if b else "❌"
    def na(v, fmt="{:.1f}"): return fmt.format(v) if v is not None else "n/a"

    if vix_extreme:
        vix_label = f"{na(vix)} 🔥 EXTREME"
    elif vix_high:
        vix_label = f"{na(vix)} ⚡ HIGH"
    elif vix_fired:
        vix_label = f"{na(vix)} ⬆ elevated"
    else:
        vix_label = na(vix)

    if confluence == 3 and macro_bullish:
        header = "🚨 HIGH CONVICTION SETUP"
    elif confluence == 3:
        header = "⚡ STRONG SETUP (no macro gate)"
    elif confluence == 2 and macro_bullish:
        header = "✅ SOLID SETUP"
    elif confluence == 2:
        header = "🟡 DEVELOPING SETUP"
    elif vix_extreme:
        header = "🔥 VIX EXTREME — Standalone signal"
    else:
        header = "👀 Signal Watch"

    msg = f"""
{header}
────────────────────
📊 *CONFLUENCE: {confluence}/3*
{tick(aaii_fired)} AAII Spread: {na(aaii_spread*100 if aaii_spread else None)+"%" if aaii_spread else "n/a"} (as of {aaii_date}) · threshold < -20%
{tick(naaim_fired)} NAAIM: {na(naaim_val)} (as of {naaim_date}) · threshold < 40
{tick(vix_fired)} VIX: {vix_label} · threshold > 30

🔭 *MACRO GATE: {macro_score}/6* {"✅ CONFIRMED" if macro_bullish else "❌ not confirmed"}
{tick(macro_rates)} 10Y yield falling · {tick(macro_dxy)} DXY falling · {tick(macro_copper)} Copper rising
{tick(macro_hy)} HY spread tightening · {tick(macro_curve)} Curve +{f"{curve_val:.2f}%" if curve_val else "n/a"} · {tick(macro_breath)} SPY > 200MA

📌 *STANDALONE*
Fear & Greed: {na(fg_score)} {"🔴 EXTREME FEAR" if fg_fired else ""}

🎯 *MEISLER SENTIMENT*
{tick(pc_10d_fired)} Put/Call 10d MA: {na(pc_10d_ma, "{:.3f}")} · threshold > {PC_10D_THRESHOLD}
{tick(pc_30d_fired)} Put/Call 30d MA: {na(pc_30d_ma, "{:.3f}")} · threshold > {PC_30D_THRESHOLD}
{tick(aaii_bulls_low)} AAII Bulls < 25%: {"🟢 YES" if aaii_bulls_low else "🔴 NO"}

📅 {datetime.now().strftime("%Y-%m-%d %H:%M")} Bangkok time
    """.strip()

    print(msg)

    if should_alert:
        send_telegram(msg)
    else:
        print("No alert threshold met — silent.")

    return confluence, macro_score, should_alert


def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        print("No Telegram config — skipping send.")
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={'chat_id': TELEGRAM_CHAT, 'text': message, 'parse_mode': 'Markdown'},
            timeout=10
        )
        if r.ok:
            print("Telegram alert sent ✅")
        else:
            print(f"Telegram error: {r.text}")
    except Exception as e:
        print(f"Telegram send failed: {e}")


if __name__ == "__main__":
    run_check()
