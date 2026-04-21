"""
Global Liquidity Tracker — Regime gauge for V2 portfolio.

Tracks Net Liquidity (Fed BS - TGA - RRP) and global central bank aggregates.
Detects liquidity regime (Easing / Neutral / Tightening) and inflection points.
Includes Global M2 → BTC impulse tracker (documented alpha source).

Data source: FRED via pandas_datareader (no API key needed).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pandas_datareader.data as web


# ── FRED series IDs ────────────────────────────────────────────────────────
FRED_SERIES = {
    # US Liquidity Core
    'WALCL':     ('Fed Total Assets',       'millions_usd', 'weekly'),
    'WTREGEN':   ('Treasury General Acct',  'millions_usd', 'weekly'),
    'RRPONTSYD': ('Reverse Repo Facility',  'billions_usd', 'daily'),
    'RESPPANWW': ('Bank Reserves at Fed',   'millions_usd', 'weekly'),

    # Money Supply
    'M2SL':      ('US M2 Money Supply',     'billions_usd', 'monthly'),

    # Global Central Banks (in local currency — convert to USD via exchange rates)
    'ECBASSETSW':('ECB Total Assets',       'millions_eur', 'weekly'),
    'JPNASSETS': ('BoJ Total Assets',       'millions_jpy', 'monthly'),

    # Financial Conditions
    'NFCI':      ('Chicago Fed NFCI',       'index',        'weekly'),
    'ANFCI':     ('Adjusted NFCI',          'index',        'weekly'),

    # Rates & Dollar
    'SOFR':      ('SOFR Overnight',         'percent',      'daily'),
    'DTWEXBGS':  ('USD Broad Index',        'index',        'daily'),
    'DGS10':     ('10Y Treasury Yield',     'percent',      'daily'),
}


# ── Fetching ───────────────────────────────────────────────────────────────

def fetch_fred_series(series_id: str, years: int = 5) -> pd.DataFrame:
    """Fetch a single FRED series. Returns DataFrame with date index + value column."""
    try:
        start = datetime.now() - timedelta(days=365 * years)
        df = web.DataReader(series_id, 'fred', start)
        df.columns = ['value']
        df.index.name = 'date'
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.dropna()
    except Exception as e:
        return pd.DataFrame()


def fetch_all_liquidity(years: int = 5) -> dict:
    """Fetch all liquidity-relevant FRED series in parallel."""
    data = {}
    for sid in FRED_SERIES:
        df = fetch_fred_series(sid, years=years)
        if not df.empty:
            data[sid] = df
    return data


def fetch_spy_btc(years: int = 5) -> dict:
    """Fetch SPY and BTC for correlation/divergence analysis."""
    end = datetime.now()
    start = end - timedelta(days=365 * years)
    out = {}
    for ticker, key in [('SPY', 'spy'), ('BTC-USD', 'btc')]:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            out[key] = df['Close'].dropna()
        except Exception:
            pass
    return out


# ── Net Liquidity Computation ──────────────────────────────────────────────

def compute_net_liquidity(fred_data: dict) -> pd.DataFrame:
    """
    Net Liquidity = WALCL (Fed Assets) - WTREGEN (TGA) - RRP
    All in millions USD. Aligned to daily frequency (forward-fill weekly series).
    """
    if not all(k in fred_data for k in ['WALCL', 'WTREGEN', 'RRPONTSYD']):
        return pd.DataFrame()

    walcl = fred_data['WALCL']['value']       # Weekly, millions
    tga = fred_data['WTREGEN']['value']        # Weekly, millions
    rrp = fred_data['RRPONTSYD']['value'] * 1000  # Daily, billions → millions

    # Build daily index
    idx = pd.date_range(
        max(walcl.index[0], tga.index[0], rrp.index[0]),
        min(walcl.index[-1], max(tga.index[-1], rrp.index[-1])),
        freq='D'
    )

    # Forward-fill weekly to daily
    walcl_d = walcl.reindex(idx, method='ffill')
    tga_d = tga.reindex(idx, method='ffill')
    rrp_d = rrp.reindex(idx, method='ffill')

    net_liq = walcl_d - tga_d - rrp_d
    return pd.DataFrame({
        'net_liquidity': net_liq,
        'walcl': walcl_d,
        'tga': tga_d,
        'rrp': rrp_d,
    }).dropna()


# ── Impulse Computation (no regime, only backtested thresholds) ───────────

def compute_liquidity_impulse(net_liquidity: pd.Series, weeks: int = 13) -> pd.Series:
    """
    Liquidity Impulse = rate-of-change of net liquidity over N weeks.
    Only the +5% threshold has proven alpha (backtest: +2.14% 9m edge, 85% win).
    Zero-crossings are too noisy — do NOT use them.
    """
    days = weeks * 7
    return net_liquidity.pct_change(periods=days) * 100


# Backtest stats for each signal (computed from 2015-2026 10yr history)
# See /tmp/liq_bt.log for full methodology
SIGNAL_BACKTEST_STATS = {
    'tga_high': {
        'name': 'TGA > 85th percentile',
        'thesis': 'High TGA → drain coming = cash inflow to markets',
        'n_triggers_10y': 34,
        'ret_3m': 3.98,  'win_3m': 81,
        'ret_6m': 7.80,  'win_6m': 78,
        'ret_9m': 13.84, 'win_9m': 100,
        'edge_9m': 3.18,
        'verdict': 'STRONG ALPHA',
    },
    'impulse_high': {
        'name': 'Liquidity Impulse > +5%',
        'thesis': 'Rapid expansion = QE-like regime, SPY benefits',
        'n_triggers_10y': 26,
        'ret_3m': 4.15,  'win_3m': 81,
        'ret_6m': 7.70,  'win_6m': 81,
        'ret_9m': 12.80, 'win_9m': 85,
        'edge_9m': 2.14,
        'verdict': 'MODERATE ALPHA',
    },
    'rrp_exhausted': {
        'name': 'RRP Exhausted (< $50B)',
        'thesis': 'No more easy cash buffer — next drain event unmuted',
        'n_triggers_10y': 34,
        'ret_3m': 2.47, 'win_3m': 71,
        'ret_6m': 5.40, 'win_6m': 76,
        'ret_9m': 8.26, 'win_9m': 71,
        'edge_9m': -2.40,  # Negative = underperforms baseline
        'verdict': 'CAUTION SIGNAL',
    },
    'tga_low': {
        'name': 'TGA < 15th percentile',
        'thesis': 'TGA near zero → refill coming = liquidity drain',
        'n_triggers_10y': None,
        'ret_9m': None, 'win_9m': None,
        'edge_9m': None,
        'verdict': 'THEORETICAL — limited backtest samples',
    },
    'impulse_accel': {
        'name': 'Liquidity Acceleration (+10pp in 12w)',
        'thesis': 'Rapid impulse reversal = policy pivot, market front-runs Fed',
        'n_triggers_10y': 19,
        'ret_3m': 4.81, 'win_3m': 83,
        'ret_6m': 8.09, 'win_6m': 83,
        'ret_9m': 13.12, 'win_9m': 83,
        'edge_9m': 2.46,
        'verdict': 'STRONG ALPHA (velocity signal)',
    },
    'impulse_bounce': {
        'name': 'Massive Bounce (+15pp in 16w)',
        'thesis': 'Extreme trough reversal = major regime shift',
        'n_triggers_10y': 10,
        'ret_3m': 6.15, 'win_3m': 70,
        'ret_6m': 10.20, 'win_6m': 90,
        'ret_9m': 15.34, 'win_9m': 80,
        'edge_9m': 4.68,
        'verdict': 'BEST ALPHA (rare, high conviction)',
    },
}


# ── Global M2 → BTC Impulse ────────────────────────────────────────────────

def compute_global_m2(fred_data: dict, usd_values: bool = True) -> pd.Series:
    """
    Aggregate Global M2 = US M2 + ECB Balance Sheet + BoJ Balance Sheet
    All converted to USD using current exchange rates.
    Returns a monthly Series in USD.
    """
    us_m2 = fred_data.get('M2SL', pd.DataFrame()).get('value', pd.Series())
    ecb = fred_data.get('ECBASSETSW', pd.DataFrame()).get('value', pd.Series())
    boj = fred_data.get('JPNASSETS', pd.DataFrame()).get('value', pd.Series())

    # Rough conversion rates (static; could be dynamic from yfinance EUR=X, JPY=X)
    EUR_USD = 1.08
    JPY_USD = 1 / 150  # 1 JPY ≈ 0.0067 USD

    # Resample all to monthly
    us_m2_m = us_m2.resample('ME').last() * 1000  # billions → millions
    ecb_m = ecb.resample('ME').last() * EUR_USD
    boj_m = boj.resample('ME').last() * JPY_USD

    # Align on common dates
    combined = pd.concat([us_m2_m, ecb_m, boj_m], axis=1, keys=['US', 'EU', 'JP'])
    combined = combined.dropna()
    combined['global_m2'] = combined.sum(axis=1)
    return combined['global_m2']


def compute_m2_btc_impulse(global_m2: pd.Series, btc: pd.Series, lag_weeks: int = 12) -> pd.DataFrame:
    """
    Compare Global M2 YoY growth with BTC (lagged by N weeks).
    BTC correlates with M2 ~12 weeks later — when M2 rises, BTC follows.
    """
    m2_yoy = global_m2.pct_change(periods=12) * 100  # 12-month YoY
    btc_yoy = btc.resample('ME').last().pct_change(periods=12) * 100
    btc_shifted = btc_yoy.shift(-lag_weeks // 4)  # Weeks → months

    df = pd.DataFrame({
        'm2_yoy': m2_yoy,
        'btc_yoy_leading': btc_shifted,
        'btc_yoy': btc_yoy,
    }).dropna()
    return df


# ── SPY/Liquidity Divergence ───────────────────────────────────────────────

def detect_spy_liquidity_divergence(net_liq: pd.Series, spy: pd.Series,
                                     window_days: int = 60) -> pd.DataFrame:
    """
    Detect divergence between SPY and Net Liquidity.
    SPY up + liquidity draining = distribution (bearish divergence)
    SPY down + liquidity expanding = accumulation (bullish divergence)
    """
    # Align on daily
    net_liq_d = net_liq.resample('D').ffill()
    spy_d = spy.resample('D').ffill()
    aligned = pd.concat([net_liq_d, spy_d], axis=1, keys=['liq', 'spy']).dropna()

    # N-day rate of change
    aligned['liq_roc'] = aligned['liq'].pct_change(periods=window_days) * 100
    aligned['spy_roc'] = aligned['spy'].pct_change(periods=window_days) * 100

    # Divergence: signs are opposite
    aligned['div'] = 'none'
    both_signs = (aligned['liq_roc'].notna()) & (aligned['spy_roc'].notna())
    aligned.loc[both_signs & (aligned['spy_roc'] > 2) & (aligned['liq_roc'] < -1), 'div'] = 'bearish'
    aligned.loc[both_signs & (aligned['spy_roc'] < -2) & (aligned['liq_roc'] > 1), 'div'] = 'bullish'

    return aligned.dropna(subset=['liq_roc', 'spy_roc'])


# ── Extreme Readings Detection ─────────────────────────────────────────────

def detect_extremes(fred_data: dict, net_liq: pd.Series) -> dict:
    """
    Flag extreme readings in TGA, RRP, and Net Liquidity that historically
    precede inflection points.
    """
    results = {}

    # TGA percentile (where is it vs its 3-year range)
    tga = fred_data.get('WTREGEN', pd.DataFrame()).get('value', pd.Series())
    if not tga.empty:
        recent_tga = tga.last('3Y') if hasattr(tga, 'last') else tga.iloc[-156:]
        current_tga = tga.iloc[-1]
        pctile = (recent_tga < current_tga).sum() / len(recent_tga) * 100
        results['tga'] = {
            'current': float(current_tga),
            'pctile_3y': round(pctile, 0),
            'signal': 'drain_imminent' if pctile > 85 else 'refill_imminent' if pctile < 15 else 'neutral',
            'interpretation': (
                '🟢 TGA at extreme high — historical drawdown → liquidity inflow likely (bullish)'
                if pctile > 85 else
                '🔴 TGA at extreme low — refill coming → liquidity drain (bearish)'
                if pctile < 15 else
                '⚪ TGA in normal range'
            )
        }

    # RRP extreme
    rrp = fred_data.get('RRPONTSYD', pd.DataFrame()).get('value', pd.Series())
    if not rrp.empty:
        current_rrp = rrp.iloc[-1]
        recent_rrp = rrp.iloc[-252*2:]
        max_3y = recent_rrp.max()
        pct_of_peak = (current_rrp / max_3y * 100) if max_3y > 0 else 0
        results['rrp'] = {
            'current': float(current_rrp),
            'peak_3y': float(max_3y),
            'pct_of_peak': round(pct_of_peak, 0),
            'signal': 'exhausted' if current_rrp < 50 else 'flowing' if current_rrp < max_3y * 0.3 else 'high',
            'interpretation': (
                '🔴 RRP exhausted — cash source depleted, liquidity can only drain from here'
                if current_rrp < 50 else
                '🟢 RRP draining — cash flowing to markets (bullish)'
                if current_rrp < max_3y * 0.3 else
                '⚪ RRP still elevated — more liquidity available to flow out'
            )
        }

    # Net Liquidity Z-score
    if not net_liq.empty and len(net_liq) > 365:
        recent = net_liq.iloc[-365:]
        z_score = (net_liq.iloc[-1] - recent.mean()) / recent.std()
        results['netliq_zscore'] = {
            'z_score': round(float(z_score), 2),
            'signal': 'extreme_high' if z_score > 2 else 'extreme_low' if z_score < -2 else 'normal',
            'interpretation': (
                '🟢 Net Liq extreme high — unusual expansion'
                if z_score > 2 else
                '🔴 Net Liq extreme low — unusual contraction (potential inflection)'
                if z_score < -2 else
                '⚪ Net Liq in normal range'
            )
        }

    return results


# ── Main Pipeline ──────────────────────────────────────────────────────────

def compute_liquidity_signals() -> dict:
    """
    Full pipeline: fetch → compute → detect backtested signals only.
    Returns summary dict with alpha signals and their historical track records.
    """
    fred_data = fetch_all_liquidity(years=5)
    if not fred_data:
        return {'error': 'Failed to fetch FRED data'}

    net_liq_df = compute_net_liquidity(fred_data)
    if net_liq_df.empty:
        return {'error': 'Failed to compute net liquidity'}

    net_liq = net_liq_df['net_liquidity']

    # Impulse (level + velocity signals)
    impulse = compute_liquidity_impulse(net_liq, weeks=13)
    current_impulse = float(impulse.iloc[-1]) if not impulse.empty and pd.notna(impulse.iloc[-1]) else 0

    # Velocity: how much has impulse moved in last 12 weeks (acceleration signal)
    impulse_12w_ago = float(impulse.iloc[-84]) if len(impulse) >= 84 and pd.notna(impulse.iloc[-84]) else current_impulse
    impulse_16w_ago = float(impulse.iloc[-112]) if len(impulse) >= 112 and pd.notna(impulse.iloc[-112]) else current_impulse
    impulse_accel_12w = round(current_impulse - impulse_12w_ago, 2)  # pp change over 12 weeks
    impulse_accel_16w = round(current_impulse - impulse_16w_ago, 2)  # pp change over 16 weeks

    # Most recent trigger date for acceleration signal (in last 6 months)
    accel_trigger_date = None
    lookback_days = 180
    if len(impulse) > 84 + lookback_days:
        for i in range(len(impulse) - 1, len(impulse) - lookback_days, -1):
            if i < 84:
                break
            if pd.notna(impulse.iloc[i]) and pd.notna(impulse.iloc[i-84]):
                accel_here = impulse.iloc[i] - impulse.iloc[i-84]
                prev_accel = (impulse.iloc[i-1] - impulse.iloc[i-85]) if i > 84 else 0
                if accel_here >= 10 and prev_accel < 10:
                    accel_trigger_date = impulse.index[i]
                    break

    # Recent changes (informational)
    now = net_liq.iloc[-1]
    wk_ago = net_liq.iloc[-7] if len(net_liq) >= 7 else now
    mo_ago = net_liq.iloc[-30] if len(net_liq) >= 30 else now
    qtr_ago = net_liq.iloc[-91] if len(net_liq) >= 91 else now

    # Global M2 & BTC (M2→BTC alpha)
    global_m2 = compute_global_m2(fred_data)
    market_data = fetch_spy_btc(years=5)
    spy = market_data.get('spy', pd.Series())
    btc = market_data.get('btc', pd.Series())

    m2_btc = compute_m2_btc_impulse(global_m2, btc) if not btc.empty and not global_m2.empty else pd.DataFrame()

    # SPY vs Liquidity divergence
    divergence = detect_spy_liquidity_divergence(net_liq, spy) if not spy.empty else pd.DataFrame()
    current_div = divergence['div'].iloc[-1] if not divergence.empty else 'none'

    # Extremes
    extremes = detect_extremes(fred_data, net_liq)

    # ── Active backtested signals ──
    active_signals = []

    # Signal 1: TGA > 85th percentile
    tga_ext = extremes.get('tga', {})
    tga_pctile = tga_ext.get('pctile_3y', 50)
    if tga_pctile > 85:
        sig = {**SIGNAL_BACKTEST_STATS['tga_high'], 'status': 'ACTIVE',
               'color': '#3fb950', 'emoji': '🟢'}
        sig['current_reading'] = f"TGA at {tga_pctile:.0f}th percentile"
        active_signals.append(sig)
    elif tga_pctile < 15:
        sig = {**SIGNAL_BACKTEST_STATS['tga_low'], 'status': 'ACTIVE',
               'color': '#ff6b6b', 'emoji': '🔴'}
        sig['current_reading'] = f"TGA at {tga_pctile:.0f}th percentile"
        active_signals.append(sig)

    # Signal 2: Impulse > +5%
    if current_impulse > 5:
        sig = {**SIGNAL_BACKTEST_STATS['impulse_high'], 'status': 'ACTIVE',
               'color': '#3fb950', 'emoji': '🟢'}
        sig['current_reading'] = f"Impulse {current_impulse:+.2f}% (13w)"
        active_signals.append(sig)

    # Signal 2b: Liquidity Acceleration — FIRES if triggered in last 6 months (alpha plays out 3-9m)
    # Current velocity OR a recent firing within backtest horizon
    accel_fired_recently = accel_trigger_date is not None
    if impulse_accel_12w >= 10 or accel_fired_recently:
        sig = {**SIGNAL_BACKTEST_STATS['impulse_accel'], 'status': 'ACTIVE',
               'color': '#3fb950', 'emoji': '🟢'}
        if impulse_accel_12w >= 10:
            sig['current_reading'] = f"Impulse rose {impulse_accel_12w:+.1f}pp in 12 weeks (firing NOW)"
        else:
            days_since = (datetime.now() - accel_trigger_date).days if accel_trigger_date else 0
            sig['current_reading'] = (
                f"Fired {accel_trigger_date.strftime('%b %d, %Y')} ({days_since}d ago) · "
                f"alpha window active through {(accel_trigger_date + timedelta(days=270)).strftime('%b %d, %Y')}"
            )
        active_signals.append(sig)

    # Signal 2c: Massive Bounce (+15pp in 16w) — rare, high-conviction
    if impulse_accel_16w >= 15:
        sig = {**SIGNAL_BACKTEST_STATS['impulse_bounce'], 'status': 'ACTIVE',
               'color': '#3fb950', 'emoji': '🚀'}
        sig['current_reading'] = f"Impulse rose {impulse_accel_16w:+.1f}pp in 16 weeks (rare!)"
        active_signals.append(sig)

    # Signal 3: RRP Exhausted (caution)
    rrp_ext = extremes.get('rrp', {})
    if rrp_ext.get('signal') == 'exhausted':
        sig = {**SIGNAL_BACKTEST_STATS['rrp_exhausted'], 'status': 'ACTIVE',
               'color': '#e3b341', 'emoji': '🟡'}
        sig['current_reading'] = f"RRP at ${rrp_ext['current']:.1f}B (0% of peak)"
        active_signals.append(sig)

    return {
        'net_liquidity': {
            'current': float(now),
            'current_trillions': round(now / 1e6, 2),
            'week_change_pct': round(float((now - wk_ago) / wk_ago * 100), 2),
            'month_change_pct': round(float((now - mo_ago) / mo_ago * 100), 2),
            'quarter_change_pct': round(float((now - qtr_ago) / qtr_ago * 100), 2),
        },
        'impulse_13w': round(current_impulse, 2),
        'impulse_accel_12w': impulse_accel_12w,
        'impulse_accel_16w': impulse_accel_16w,
        'accel_trigger_date': str(accel_trigger_date.date()) if accel_trigger_date else None,
        'velocity_fires_history': _find_velocity_fires(impulse, threshold=10, window_days=84),
        'massive_bounce_fires_history': _find_velocity_fires(impulse, threshold=15, window_days=112),
        'components': {
            'walcl': float(fred_data['WALCL']['value'].iloc[-1]),
            'walcl_date': str(fred_data['WALCL']['value'].index[-1].date()),
            'tga': float(fred_data['WTREGEN']['value'].iloc[-1]),
            'tga_date': str(fred_data['WTREGEN']['value'].index[-1].date()),
            'tga_pctile_3y': tga_pctile,
            'rrp_billions': float(fred_data['RRPONTSYD']['value'].iloc[-1]),
            'rrp_date': str(fred_data['RRPONTSYD']['value'].index[-1].date()),
        },
        'extremes': extremes,
        'active_signals': active_signals,
        'all_signals_catalog': SIGNAL_BACKTEST_STATS,  # For backtest display
        'nfci': {
            'value': float(fred_data['NFCI']['value'].iloc[-1]) if 'NFCI' in fred_data else None,
            'interpretation': _interpret_nfci(fred_data.get('NFCI', pd.DataFrame())),
        },
        'divergence': current_div,
        'net_liquidity_series': net_liq_df,
        'impulse_series': impulse,
        'global_m2_series': global_m2,
        'm2_btc': m2_btc,
        'spy': spy,
        'btc': btc,
        'divergence_series': divergence,
        'timestamp': datetime.now().isoformat(),
    }


def _find_velocity_fires(impulse: pd.Series, threshold: float = 10,
                          window_days: int = 84, dedupe_days: int = 90) -> list:
    """
    Find all dates where impulse velocity crossed above threshold (pp change in window).
    Returns list of dates as ISO strings.
    """
    if impulse.empty or len(impulse) <= window_days:
        return []
    fires = []
    prev_change = None
    for i in range(window_days, len(impulse)):
        if pd.notna(impulse.iloc[i]) and pd.notna(impulse.iloc[i - window_days]):
            change = impulse.iloc[i] - impulse.iloc[i - window_days]
            if prev_change is not None and prev_change < threshold and change >= threshold:
                d = impulse.index[i]
                # Dedupe within window
                if not fires or (d - pd.Timestamp(fires[-1])).days > dedupe_days:
                    fires.append(str(d.date()))
            prev_change = change
    return fires


def _interpret_nfci(nfci_df: pd.DataFrame) -> str:
    if nfci_df.empty:
        return ''
    val = nfci_df['value'].iloc[-1]
    if val > 0.5:
        return '🔴 Tight financial conditions (risk-off)'
    elif val > 0:
        return '🟡 Slightly tight'
    elif val > -0.5:
        return '🟢 Loose conditions (risk-on)'
    else:
        return '🟢🟢 Very loose (strong risk-on)'


if __name__ == '__main__':
    import json
    result = compute_liquidity_signals()
    # Pretty print summary
    for k, v in result.items():
        if isinstance(v, (pd.DataFrame, pd.Series)):
            print(f"{k}: [{type(v).__name__} shape={v.shape}]")
        elif isinstance(v, dict):
            print(f"{k}:")
            for kk, vv in v.items():
                print(f"  {kk}: {vv}")
        elif isinstance(v, list):
            print(f"{k}: [{len(v)} items]")
        else:
            print(f"{k}: {v}")
