#!/usr/bin/env python3
"""
COT (Commitments of Traders) Backtest against SPY forward returns.

Downloads CFTC legacy futures-only reports, extracts key contracts,
computes net speculative and commercial positioning z-scores/percentiles,
and measures forward SPY returns at extremes.

Usage:
    python cot_backtest.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import urllib.request
import warnings
import sys
import os

warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent / 'data'
HORIZONS = {'1m': 21, '3m': 63, '6m': 126, '12m': 252}

# ── Contract definitions ────────────────────────────────────────────────
# CFTC Market_and_Exchange_Names (partial match) and CFTC_Contract_Market_Code
CONTRACTS = {
    'SP500': {
        'search_names': ['E-MINI S&P 500', 'S&P 500 STOCK INDEX'],
        'codes': ['13874A', '138741', '13874+'],
        'label': 'S&P 500 E-mini',
    },
    'VIX': {
        'search_names': ['VIX', 'CBOE VOLATILITY INDEX'],
        'codes': ['1170E1'],
        'label': 'VIX Futures',
    },
    'TNOTE10': {
        'search_names': ['10-YEAR', '10 YEAR U.S. TREASURY'],
        'codes': ['043602'],
        'label': '10Y T-Note',
    },
    'GOLD': {
        'search_names': ['GOLD'],
        'codes': ['088691'],
        'label': 'Gold',
    },
    'CRUDE': {
        'search_names': ['CRUDE OIL, LIGHT SWEET', 'WTI-PHYSICAL'],
        'codes': ['067651'],
        'label': 'Crude Oil WTI',
    },
    'USDX': {
        'search_names': ['U.S. DOLLAR INDEX', 'US DOLLAR INDEX'],
        'codes': ['098662'],
        'label': 'US Dollar Index',
    },
}


# ── Data Download ───────────────────────────────────────────────────────

def download_cot_year(year: int) -> pd.DataFrame:
    """Download a single year of CFTC legacy futures-only COT data."""
    # Current year uses the 'deacom' path for combined; legacy uses 'deafut'
    # Try legacy futures-only first
    url = f'https://www.cftc.gov/files/dea/history/deacot{year}.zip'
    cache_path = DATA_DIR / f'cot_raw_{year}.csv'

    if cache_path.exists():
        return pd.read_csv(cache_path, low_memory=False)

    print(f"  Downloading COT data for {year}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        zf = ZipFile(BytesIO(data))
        csv_name = zf.namelist()[0]
        df = pd.read_csv(zf.open(csv_name), low_memory=False)
        df.to_csv(cache_path, index=False)
        return df
    except Exception as e:
        print(f"    Failed legacy URL for {year}: {e}")
        # Try alternate URL pattern
        alt_url = f'https://www.cftc.gov/files/dea/history/deacot{year}.zip'
        try:
            req = urllib.request.Request(alt_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            zf = ZipFile(BytesIO(data))
            csv_name = zf.namelist()[0]
            df = pd.read_csv(zf.open(csv_name), low_memory=False)
            df.to_csv(cache_path, index=False)
            return df
        except Exception as e2:
            print(f"    Also failed alternate for {year}: {e2}")
            return pd.DataFrame()


def download_all_cot(start_year: int = 2006, end_year: int = 2026) -> pd.DataFrame:
    """Download and concatenate all COT yearly files."""
    combined_path = DATA_DIR / 'cot_legacy_combined.csv'
    if combined_path.exists():
        print("  Loading cached combined COT data...")
        return pd.read_csv(combined_path, low_memory=False)

    frames = []
    for year in range(start_year, end_year + 1):
        df = download_cot_year(year)
        if not df.empty:
            frames.append(df)

    if not frames:
        print("ERROR: No COT data could be downloaded.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(combined_path, index=False)
    print(f"  Combined COT data: {len(combined)} rows")
    return combined


def identify_columns(df: pd.DataFrame) -> dict:
    """Identify the correct column names in the COT data (they vary by year)."""
    cols = df.columns.tolist()
    col_map = {}

    # Date column
    for c in cols:
        if 'report' in c.lower() and 'date' in c.lower() and 'as_of' in c.lower().replace(' ', '_'):
            col_map['date'] = c
            break
        if c.strip().lower() in ['as_of_date_in_form_yyyymmdd', 'report_date_as_yyyy-mm-dd',
                                   'as_of_date_in_form_yyyy-mm-dd']:
            col_map['date'] = c
            break
    if 'date' not in col_map:
        for c in cols:
            if 'date' in c.lower() and 'yyyy' in c.lower():
                col_map['date'] = c
                break
    if 'date' not in col_map:
        for c in cols:
            if 'date' in c.lower():
                col_map['date'] = c
                break

    # Market name
    for c in cols:
        if 'market_and_exchange' in c.lower().replace(' ', '_'):
            col_map['market_name'] = c
            break
        if c.strip().lower() in ['market_and_exchange_names']:
            col_map['market_name'] = c
            break
    if 'market_name' not in col_map:
        for c in cols:
            if 'market' in c.lower() and 'name' in c.lower():
                col_map['market_name'] = c
                break

    # Contract code
    for c in cols:
        if 'cftc_contract_market_code' in c.lower().replace(' ', '_'):
            col_map['code'] = c
            break
        if 'contract_market' in c.lower() and 'code' in c.lower():
            col_map['code'] = c
            break

    # Noncommercial (Large Speculators) long and short
    for c in cols:
        cl = c.lower().replace(' ', '_').strip()
        if 'noncommercial' in cl and 'long' in cl and 'all' not in cl and 'spread' not in cl and 'change' not in cl and 'pct' not in cl:
            if 'positions' in cl or cl.endswith('_long'):
                col_map['spec_long'] = c
                break
    if 'spec_long' not in col_map:
        for c in cols:
            cl = c.strip().lower()
            if cl in ['noncommercial_positions-long_(all)',
                       'noncommercial_positions_long_all']:
                col_map['spec_long'] = c
                break

    for c in cols:
        cl = c.lower().replace(' ', '_').strip()
        if 'noncommercial' in cl and 'short' in cl and 'all' not in cl and 'spread' not in cl and 'change' not in cl and 'pct' not in cl:
            if 'positions' in cl or cl.endswith('_short'):
                col_map['spec_short'] = c
                break
    if 'spec_short' not in col_map:
        for c in cols:
            cl = c.strip().lower()
            if cl in ['noncommercial_positions-short_(all)',
                       'noncommercial_positions_short_all']:
                col_map['spec_short'] = c
                break

    # Commercial long and short
    for c in cols:
        cl = c.lower().replace(' ', '_').strip()
        if 'commercial' in cl and 'noncommercial' not in cl and 'long' in cl and 'all' not in cl and 'change' not in cl and 'pct' not in cl:
            if 'positions' in cl or cl.endswith('_long'):
                col_map['comm_long'] = c
                break

    for c in cols:
        cl = c.lower().replace(' ', '_').strip()
        if 'commercial' in cl and 'noncommercial' not in cl and 'short' in cl and 'all' not in cl and 'change' not in cl and 'pct' not in cl:
            if 'positions' in cl or cl.endswith('_short'):
                col_map['comm_short'] = c
                break

    # Nonreportable (Small Speculators) long and short
    for c in cols:
        cl = c.strip().lower()
        if 'nonreportable' in cl and 'long' in cl and '(all)' in cl and 'change' not in cl and '%' not in cl:
            col_map['nonrept_long'] = c
            break
    if 'nonrept_long' not in col_map:
        for c in cols:
            cl = c.strip().lower()
            if 'nonreportable' in cl and 'long' in cl and 'old' not in cl and 'other' not in cl and 'change' not in cl and '%' not in cl:
                col_map['nonrept_long'] = c
                break

    for c in cols:
        cl = c.strip().lower()
        if 'nonreportable' in cl and 'short' in cl and '(all)' in cl and 'change' not in cl and '%' not in cl:
            col_map['nonrept_short'] = c
            break
    if 'nonrept_short' not in col_map:
        for c in cols:
            cl = c.strip().lower()
            if 'nonreportable' in cl and 'short' in cl and 'old' not in cl and 'other' not in cl and 'change' not in cl and '%' not in cl:
                col_map['nonrept_short'] = c
                break

    # Open interest
    for c in cols:
        cl = c.lower().replace(' ', '_').strip()
        if 'open_interest' in cl and 'all' not in cl and 'change' not in cl and 'old' not in cl and 'other' not in cl:
            col_map['oi'] = c
            break

    return col_map


def extract_contract(cot_df: pd.DataFrame, contract_key: str, col_map: dict,
                     config: dict = None) -> pd.DataFrame:
    """Extract rows for a specific futures contract from combined COT data.

    Args:
        cot_df: Combined CFTC COT DataFrame
        contract_key: Key into CONTRACTS dict (legacy) or ignored if config provided
        col_map: Column mapping from identify_columns()
        config: Optional dict with 'search' list (from cot_config.py). If provided,
                uses these search strings instead of legacy CONTRACTS lookup.
    """
    if config is not None:
        search_names = config.get('search', [])
        codes = config.get('codes', [])
    else:
        contract = CONTRACTS[contract_key]
        search_names = contract.get('search_names', [])
        codes = contract.get('codes', [])

    # Try matching by code first
    mask = pd.Series(False, index=cot_df.index)
    if codes and 'code' in col_map:
        code_col = col_map['code']
        cot_df[code_col] = cot_df[code_col].astype(str).str.strip()
        for code in codes:
            mask = mask | (cot_df[code_col] == code)

    # If no matches by code, try by name (use regex=False for literal matching)
    if mask.sum() == 0 and 'market_name' in col_map:
        name_col = col_map['market_name']
        for search_name in search_names:
            mask = mask | cot_df[name_col].astype(str).str.upper().str.contains(
                search_name.upper(), na=False, regex=False)

    subset = cot_df[mask].copy()
    if subset.empty:
        return pd.DataFrame()

    # Parse date
    date_col = col_map['date']
    subset['date'] = pd.to_datetime(subset[date_col], format='mixed', errors='coerce')
    subset = subset.dropna(subset=['date'])

    # Extract numeric columns
    result = pd.DataFrame()
    result['date'] = subset['date']

    for key in ['spec_long', 'spec_short', 'comm_long', 'comm_short',
                'nonrept_long', 'nonrept_short', 'oi']:
        if key in col_map and col_map[key] in subset.columns:
            result[key] = pd.to_numeric(subset[col_map[key]], errors='coerce')

    # Compute net positioning for all 3 trader types
    if 'spec_long' in result.columns and 'spec_short' in result.columns:
        result['spec_net'] = result['spec_long'] - result['spec_short']
    if 'comm_long' in result.columns and 'comm_short' in result.columns:
        result['comm_net'] = result['comm_long'] - result['comm_short']
    if 'nonrept_long' in result.columns and 'nonrept_short' in result.columns:
        result['small_spec_net'] = result['nonrept_long'] - result['nonrept_short']

    result = result.sort_values('date').reset_index(drop=True)

    # Deduplicate by date (keep last if duplicates)
    result = result.drop_duplicates(subset='date', keep='last').reset_index(drop=True)

    return result


def extract_all_contracts(cot_df: pd.DataFrame = None, force: bool = False) -> dict:
    """Extract all 37 contracts from combined COT data using cot_config.py definitions.

    Returns dict of {contract_key: DataFrame} with spec_net, comm_net, small_spec_net.
    Caches individual CSVs to data/cot_{key}.csv.
    """
    from cot_config import COT_CONTRACTS

    # Load combined data if not provided
    if cot_df is None:
        cot_df = download_all_cot()

    col_map = identify_columns(cot_df)
    results = {}

    for key, cfg in COT_CONTRACTS.items():
        cache_path = DATA_DIR / f'cot_{key}.csv'

        # Use cache if available and not forcing re-extraction
        if cache_path.exists() and not force:
            df = pd.read_csv(cache_path, parse_dates=['date'])
            # Check if small_spec_net exists (old extractions won't have it)
            if 'small_spec_net' in df.columns:
                results[key] = df
                continue

        # Extract fresh
        df = extract_contract(cot_df, key, col_map, config=cfg)
        if df.empty:
            print(f"  WARNING: No data for {key} ({cfg['label']})")
            continue

        # Save to cache
        df.to_csv(cache_path, index=False)
        results[key] = df

    return results


def compute_rolling_percentile(series: pd.Series, window: int = 156) -> pd.Series:
    """
    Compute rolling percentile rank within trailing window.
    Default window = 156 weeks ~ 3 years of weekly data.
    """
    def pctile_rank(x):
        if len(x) < 20:  # need minimum data
            return np.nan
        current = x.iloc[-1]
        return (x.iloc[:-1] < current).sum() / (len(x) - 1)

    return series.rolling(window, min_periods=52).apply(pctile_rank, raw=False)


def compute_zscore(series: pd.Series, window: int = 156) -> pd.Series:
    """Compute rolling z-score within trailing window."""
    roll_mean = series.rolling(window, min_periods=52).mean()
    roll_std = series.rolling(window, min_periods=52).std()
    return (series - roll_mean) / (roll_std + 1e-10)


# ── SPY loader ──────────────────────────────────────────────────────────

def load_spy() -> pd.DataFrame:
    """Load SPY daily close with forward returns pre-computed."""
    path = DATA_DIR / 'spy_daily.csv'
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    for label, days in HORIZONS.items():
        df[f'fwd_{label}'] = df['close'].shift(-days) / df['close'] - 1
    return df


# ── Backtest logic ──────────────────────────────────────────────────────

def deduplicate_signals(dates: list, window_days: int = 30) -> list:
    """Cluster signals within a window, keep first."""
    if not dates:
        return []
    sorted_dates = sorted(dates)
    deduped = [sorted_dates[0]]
    for d in sorted_dates[1:]:
        if (d - deduped[-1]).days >= window_days:
            deduped.append(d)
    return deduped


def backtest_cot_signal(cot_df: pd.DataFrame, spy_df: pd.DataFrame,
                        signal_col: str, threshold_low: float, threshold_high: float,
                        signal_type: str, dedup_days: int = 30,
                        name: str = '') -> dict:
    """
    Backtest a COT percentile signal.
    signal_type='buy': fires when signal_col < threshold_low (contrarian buy)
    signal_type='sell': fires when signal_col > threshold_high (contrarian sell)

    Returns dict with stats per horizon.
    """
    # COT is weekly (Tuesday report date). Merge with nearest SPY trading day.
    cot = cot_df[['date', signal_col]].dropna().copy()
    spy = spy_df.copy()

    # For each COT date, find the nearest SPY trading day (same day or next)
    merged = pd.merge_asof(
        cot.sort_values('date'),
        spy.sort_values('date'),
        on='date',
        direction='forward',
        tolerance=pd.Timedelta(days=5)
    )
    merged = merged.dropna(subset=['close'])

    if signal_type == 'buy':
        mask = merged[signal_col] < threshold_low
    else:
        mask = merged[signal_col] > threshold_high

    signal_dates = merged.loc[mask, 'date'].tolist()
    signal_dates = deduplicate_signals(signal_dates, dedup_days)
    signals = merged[merged['date'].isin(signal_dates)].copy()

    stats = {}
    for label in HORIZONS:
        col = f'fwd_{label}'
        if col not in signals.columns:
            continue
        valid = signals[col].dropna()
        if len(valid) >= 3:
            stats[label] = {
                'n': len(valid),
                'win_rate': (valid > 0).mean(),
                'avg_return': valid.mean(),
                'median_return': valid.median(),
                'min_return': valid.min(),
                'max_return': valid.max(),
            }

    return {
        'name': name,
        'n_signals': len(signal_dates),
        'stats': stats,
        'signal_dates': signal_dates,
    }


# ── Unconditional baseline ─────────────────────────────────────────────

def compute_baseline(spy_df: pd.DataFrame) -> dict:
    """Compute unconditional (all dates) forward return stats for SPY."""
    stats = {}
    for label in HORIZONS:
        col = f'fwd_{label}'
        valid = spy_df[col].dropna()
        if len(valid) > 0:
            stats[label] = {
                'n': len(valid),
                'win_rate': (valid > 0).mean(),
                'avg_return': valid.mean(),
                'median_return': valid.median(),
            }
    return stats


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 100)
    print("COT (Commitments of Traders) Backtest vs SPY Forward Returns")
    print("=" * 100)

    # 1. Load SPY
    print("\n[1] Loading SPY data...")
    spy = load_spy()
    print(f"    SPY: {spy['date'].min().date()} to {spy['date'].max().date()}, {len(spy)} rows")

    baseline = compute_baseline(spy)

    # 2. Download COT data
    print("\n[2] Downloading CFTC COT data (legacy futures-only)...")
    cot_raw = download_all_cot(start_year=2006, end_year=2026)

    if cot_raw.empty:
        print("ERROR: No COT data available.")
        return

    # Identify columns
    col_map = identify_columns(cot_raw)
    print(f"    Column mapping: {col_map}")

    # 3. Extract each contract and compute signals
    print("\n[3] Extracting contracts and computing positioning signals...")
    all_results = []

    for contract_key, contract_info in CONTRACTS.items():
        label = contract_info['label']
        print(f"\n  --- {label} ---")

        contract_df = extract_contract(cot_raw, contract_key, col_map)
        if contract_df.empty or 'spec_net' not in contract_df.columns:
            print(f"    WARNING: No data found for {label}")
            continue

        print(f"    Data: {contract_df['date'].min().date()} to {contract_df['date'].max().date()}, "
              f"{len(contract_df)} weekly obs")

        # Save extracted contract data
        save_path = DATA_DIR / f'cot_{contract_key.lower()}.csv'
        contract_df.to_csv(save_path, index=False)
        print(f"    Saved to {save_path}")

        # Compute rolling percentiles (3-year trailing = 156 weeks)
        contract_df['spec_net_pctile'] = compute_rolling_percentile(contract_df['spec_net'], window=156)
        contract_df['spec_net_zscore'] = compute_zscore(contract_df['spec_net'], window=156)

        if 'comm_net' in contract_df.columns:
            contract_df['comm_net_pctile'] = compute_rolling_percentile(contract_df['comm_net'], window=156)
            contract_df['comm_net_zscore'] = compute_zscore(contract_df['comm_net'], window=156)

        # ── Backtest: Large Speculator extremes ──

        # Spec net at bottom 10th pctile -> contrarian BUY
        r = backtest_cot_signal(contract_df, spy, 'spec_net_pctile', 0.10, 0.90,
                                'buy', dedup_days=30,
                                name=f"{label}: Spec Net < 10th pctile (BUY)")
        all_results.append(r)

        # Spec net at bottom 5th pctile -> contrarian BUY (stricter)
        r = backtest_cot_signal(contract_df, spy, 'spec_net_pctile', 0.05, 0.95,
                                'buy', dedup_days=30,
                                name=f"{label}: Spec Net < 5th pctile (BUY)")
        all_results.append(r)

        # Spec net at top 90th pctile -> contrarian SELL
        r = backtest_cot_signal(contract_df, spy, 'spec_net_pctile', 0.10, 0.90,
                                'sell', dedup_days=30,
                                name=f"{label}: Spec Net > 90th pctile (SELL)")
        all_results.append(r)

        # Spec net at top 95th pctile -> contrarian SELL (stricter)
        r = backtest_cot_signal(contract_df, spy, 'spec_net_pctile', 0.05, 0.95,
                                'sell', dedup_days=30,
                                name=f"{label}: Spec Net > 95th pctile (SELL)")
        all_results.append(r)

        # ── Backtest: Commercial extremes ──
        if 'comm_net_pctile' in contract_df.columns:
            # Commercial net at top 90th pctile -> BUY (commercials are "smart money")
            r = backtest_cot_signal(contract_df, spy, 'comm_net_pctile', 0.10, 0.90,
                                    'sell', dedup_days=30,
                                    name=f"{label}: Comm Net > 90th pctile (BUY)")
            # For commercials, high net long = bullish, so we want 'buy' when pctile > 90
            # but our function uses signal_type='sell' for > threshold_high
            # We'll relabel in results: the signal fires, and we measure SPY returns
            all_results.append(r)

            # Commercial net at bottom 10th pctile -> SELL (commercials reducing longs)
            r = backtest_cot_signal(contract_df, spy, 'comm_net_pctile', 0.10, 0.90,
                                    'buy', dedup_days=30,
                                    name=f"{label}: Comm Net < 10th pctile (SELL)")
            all_results.append(r)

    # 4. Print results
    print("\n\n" + "=" * 130)
    print("BACKTEST RESULTS: COT Positioning Extremes vs SPY Forward Returns")
    print("=" * 130)

    # Print baseline first
    print(f"\n{'BASELINE (all days)':<50} ", end='')
    for label in HORIZONS:
        b = baseline.get(label, {})
        wr = b.get('win_rate', 0)
        avg = b.get('avg_return', 0)
        print(f"  {label}: WR={wr*100:.0f}% Avg={avg*100:+.1f}%", end='')
    print()

    print("\n" + "-" * 130)
    print(f"{'Signal':<50} {'N':>4}  {'1m WR':>6} {'1m Avg':>7}  {'3m WR':>6} {'3m Avg':>7}  "
          f"{'6m WR':>6} {'6m Avg':>7}  {'12m WR':>6} {'12m Avg':>8}")
    print("-" * 130)

    for r in all_results:
        name = r['name']
        stats = r['stats']

        if not stats:
            print(f"{name:<50} {'--':>4}  No signals (insufficient data)")
            continue

        n = next(iter(stats.values()), {}).get('n', 0)

        def fmt_wr(label):
            s = stats.get(label, {})
            wr = s.get('win_rate')
            return f"{wr*100:.0f}%" if wr is not None else "n/a"

        def fmt_avg(label):
            s = stats.get(label, {})
            avg = s.get('avg_return')
            return f"{avg*100:+.1f}%" if avg is not None else "n/a"

        print(f"{name:<50} {n:>4}  {fmt_wr('1m'):>6} {fmt_avg('1m'):>7}  "
              f"{fmt_wr('3m'):>6} {fmt_avg('3m'):>7}  "
              f"{fmt_wr('6m'):>6} {fmt_avg('6m'):>7}  "
              f"{fmt_wr('12m'):>6} {fmt_avg('12m'):>8}")

    print("=" * 130)

    # 5. Detailed view: best signals
    print("\n\n" + "=" * 100)
    print("TOP SIGNALS BY 3-MONTH AVERAGE RETURN (min 5 signals)")
    print("=" * 100)

    scored = []
    for r in all_results:
        s3 = r['stats'].get('3m', {})
        if s3.get('n', 0) >= 5:
            scored.append((r['name'], s3['avg_return'], s3['win_rate'], s3['n'],
                          r['stats'].get('6m', {}).get('avg_return', np.nan),
                          r['stats'].get('12m', {}).get('avg_return', np.nan)))

    scored.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Signal':<50} {'N':>4}  {'3m Avg':>8} {'3m WR':>7}  {'6m Avg':>8} {'12m Avg':>9}")
    print("-" * 100)
    for name, avg3, wr3, n, avg6, avg12 in scored[:15]:
        a6 = f"{avg6*100:+.1f}%" if not np.isnan(avg6) else "n/a"
        a12 = f"{avg12*100:+.1f}%" if not np.isnan(avg12) else "n/a"
        print(f"{name:<50} {n:>4}  {avg3*100:+.1f}%   {wr3*100:.0f}%   {a6:>8} {a12:>9}")

    print("\n" + "=" * 100)
    print("BOTTOM SIGNALS (worst 3m returns = potential short/avoid signals)")
    print("=" * 100)
    print(f"{'Signal':<50} {'N':>4}  {'3m Avg':>8} {'3m WR':>7}  {'6m Avg':>8} {'12m Avg':>9}")
    print("-" * 100)
    for name, avg3, wr3, n, avg6, avg12 in scored[-10:]:
        a6 = f"{avg6*100:+.1f}%" if not np.isnan(avg6) else "n/a"
        a12 = f"{avg12*100:+.1f}%" if not np.isnan(avg12) else "n/a"
        print(f"{name:<50} {n:>4}  {avg3*100:+.1f}%   {wr3*100:.0f}%   {a6:>8} {a12:>9}")

    # 6. Print most recent signals
    print("\n\n" + "=" * 100)
    print("MOST RECENT SIGNAL DATES (last 2 years)")
    print("=" * 100)
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=730)
    for r in all_results:
        recent = [d for d in r['signal_dates'] if d > cutoff]
        if recent:
            recent_str = ', '.join([d.strftime('%Y-%m-%d') for d in sorted(recent)[-5:]])
            print(f"  {r['name']:<50} Last signals: {recent_str}")


if __name__ == '__main__':
    main()
