#!/usr/bin/env python3
"""
Backtesting engine for market sentiment indicators.
Tests contrarian sentiment indicators against SPY forward returns.

Usage:
    python backtest.py              # Run all backtests, print results
    python backtest.py --verbose    # Include per-signal detail
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

DATA_DIR = Path(__file__).parent / 'data'

# Forward return periods (trading days)
HORIZONS = {'1m': 21, '3m': 63, '6m': 126, '12m': 252}

# ── Data Loaders ─────────────────────────────────────────────────────────

def load_spy() -> pd.DataFrame:
    """Load SPY daily close with forward returns pre-computed."""
    path = DATA_DIR / 'spy_daily.csv'
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    for label, days in HORIZONS.items():
        df[f'fwd_{label}'] = df['close'].shift(-days) / df['close'] - 1
    return df


def load_putcall() -> pd.DataFrame:
    """Load CBOE put/call ratio with moving averages."""
    path = DATA_DIR / 'putcall.csv'
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['pc_10d_ma'] = df['equity_pc_ratio'].rolling(10, min_periods=8).mean()
    df['pc_30d_ma'] = df['equity_pc_ratio'].rolling(30, min_periods=25).mean()
    return df


def load_aaii() -> pd.DataFrame:
    """Load AAII sentiment survey."""
    path = DATA_DIR / 'aaii.csv'
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.dropna(subset=['bullish', 'bearish']).sort_values('date').reset_index(drop=True)
    # Convert to percentages if in decimal form
    if df['bullish'].max() <= 1.0:
        df['bullish'] = df['bullish'] * 100
        df['bearish'] = df['bearish'] * 100
    df['bull_bear_spread'] = df['bullish'] - df['bearish']
    return df


def load_naaim() -> pd.DataFrame:
    """Load NAAIM exposure index."""
    path = DATA_DIR / 'naaim.csv'
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.dropna(subset=['naaim']).sort_values('date').reset_index(drop=True)
    return df


def load_sector_etfs() -> pd.DataFrame:
    """Load sector ETF data from yfinance (cached as CSV if available)."""
    path = DATA_DIR / 'sectors.csv'
    if path.exists():
        return pd.read_csv(path, parse_dates=['date'])
    # Generate on first use
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        end = datetime.today()
        start = end - timedelta(days=365 * 20)
        frames = []
        for ticker, col in [('IGV', 'igv'), ('KBE', 'kbe'), ('SPY', 'spy_close')]:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if not df.empty:
                s = df['Close'].reset_index()
                s.columns = ['date', col]
                s['date'] = pd.to_datetime(s['date']).dt.tz_localize(None)
                frames.append(s)
        if len(frames) >= 3:
            merged = frames[0]
            for f in frames[1:]:
                merged = merged.merge(f, on='date', how='inner')
            merged['igv_spy_ratio'] = merged['igv'] / merged['spy_close']
            merged['kbe_spy_ratio'] = merged['kbe'] / merged['spy_close']
            merged.to_csv(path, index=False)
            return merged
    except Exception as e:
        print(f"Warning: Could not load sector ETFs: {e}")
    return pd.DataFrame()


# ── COT Data Loaders ─────────────────────────────────────────────────────

def _compute_rolling_percentile(series: pd.Series, window: int = 156) -> pd.Series:
    """Rolling percentile rank over trailing window (156 weeks ≈ 3 years)."""
    def pctile_rank(x):
        if len(x) < 10:
            return np.nan
        return (x.values[:-1] < x.values[-1]).sum() / (len(x) - 1)
    return series.rolling(window, min_periods=52).apply(pctile_rank, raw=False)


def load_cot(contract: str) -> pd.DataFrame:
    """Load COT data for a contract, compute rolling percentiles for all trader types.
    Works with any of the 37 contracts from cot_config.py (data in data/cot_{key}.csv).
    """
    path = DATA_DIR / f'cot_{contract}.csv'
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    # Compute percentiles for all available trader types
    if 'spec_net' in df.columns:
        df['spec_net_pctile'] = _compute_rolling_percentile(df['spec_net'])
    if 'comm_net' in df.columns:
        df['comm_net_pctile'] = _compute_rolling_percentile(df['comm_net'])
    if 'small_spec_net' in df.columns:
        df['small_spec_net_pctile'] = _compute_rolling_percentile(df['small_spec_net'])
    return df.dropna(subset=['spec_net_pctile']) if 'spec_net_pctile' in df.columns else df


def load_cot_sp500():
    return load_cot('sp500')

def load_cot_vix():
    return load_cot('vix')

def load_cot_tnote10():
    return load_cot('tnote10')

def load_cot_gold():
    return load_cot('gold')

def load_cot_crude():
    return load_cot('crude')

def load_cot_usdx():
    return load_cot('usdx')


# ── Backtesting Core ─────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    name: str
    description: str
    signal_dates: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    signals_df: Optional[pd.DataFrame] = None


def deduplicate_signals(dates: list, window_days: int = 30) -> list:
    """Cluster signals within a window, keeping only the first occurrence."""
    if not dates:
        return []
    sorted_dates = sorted(dates)
    deduped = [sorted_dates[0]]
    for d in sorted_dates[1:]:
        if (d - deduped[-1]).days >= window_days:
            deduped.append(d)
    return deduped


def backtest_threshold(indicator_df: pd.DataFrame, spy_df: pd.DataFrame,
                       indicator_col: str, threshold: float,
                       direction: str = 'above', dedup_days: int = 30,
                       name: str = '', description: str = '') -> BacktestResult:
    """
    Backtest a threshold-based signal.
    direction='above': signal fires when indicator > threshold (e.g., VIX > 30)
    direction='below': signal fires when indicator < threshold (e.g., AAII spread < -20)
    """
    # Merge indicator with SPY
    merged = spy_df.merge(indicator_df[['date', indicator_col]].dropna(),
                          on='date', how='inner')

    # Find signal dates
    if direction == 'above':
        mask = merged[indicator_col] > threshold
    else:
        mask = merged[indicator_col] < threshold

    signal_dates = merged.loc[mask, 'date'].tolist()
    signal_dates = deduplicate_signals(signal_dates, dedup_days)

    # Get forward returns for signal dates
    signals = merged[merged['date'].isin(signal_dates)].copy()

    # Compute stats
    stats = {}
    for label in HORIZONS:
        col = f'fwd_{label}'
        valid = signals[col].dropna()
        if len(valid) > 0:
            stats[label] = {
                'n': len(valid),
                'win_rate': (valid > 0).mean(),
                'avg_return': valid.mean(),
                'median_return': valid.median(),
                'min_return': valid.min(),
                'max_return': valid.max(),
            }

    return BacktestResult(
        name=name or f"{indicator_col} {'>' if direction == 'above' else '<'} {threshold}",
        description=description,
        signal_dates=signal_dates,
        stats=stats,
        signals_df=signals,
    )


def backtest_combo(indicator_df: pd.DataFrame, spy_df: pd.DataFrame,
                   conditions: list, dedup_days: int = 30,
                   name: str = '', description: str = '') -> BacktestResult:
    """
    Backtest a combo signal where multiple conditions must be true simultaneously.
    conditions: list of (col, threshold, direction) tuples
    """
    merged = spy_df.copy()
    for col_name in indicator_df.columns:
        if col_name != 'date' and col_name not in merged.columns:
            temp = indicator_df[['date', col_name]].dropna()
            merged = merged.merge(temp, on='date', how='inner')

    mask = pd.Series(True, index=merged.index)
    for col, threshold, direction in conditions:
        if col not in merged.columns:
            return BacktestResult(name=name, description=description)
        if direction == 'above':
            mask = mask & (merged[col] > threshold)
        else:
            mask = mask & (merged[col] < threshold)

    signal_dates = merged.loc[mask, 'date'].tolist()
    signal_dates = deduplicate_signals(signal_dates, dedup_days)
    signals = merged[merged['date'].isin(signal_dates)].copy()

    stats = {}
    for label in HORIZONS:
        col = f'fwd_{label}'
        valid = signals[col].dropna()
        if len(valid) > 0:
            stats[label] = {
                'n': len(valid),
                'win_rate': (valid > 0).mean(),
                'avg_return': valid.mean(),
                'median_return': valid.median(),
                'min_return': valid.min(),
                'max_return': valid.max(),
            }

    return BacktestResult(name=name, description=description,
                          signal_dates=signal_dates, stats=stats, signals_df=signals)


def backtest_relative_strength(sectors_df: pd.DataFrame, spy_df: pd.DataFrame,
                               ratio_col: str, lookback: int = 20,
                               percentile: float = 0.10, dedup_days: int = 30,
                               name: str = '', description: str = '') -> BacktestResult:
    """
    Signal fires when a sector/SPY ratio drops to its N-day low (bottom percentile).
    This captures Meisler's "where are investors puking" concept.
    """
    df = sectors_df.copy()
    df[f'{ratio_col}_rolling_min'] = df[ratio_col].rolling(lookback).min()
    df[f'{ratio_col}_rolling_max'] = df[ratio_col].rolling(lookback).max()
    df['ratio_pctile'] = (df[ratio_col] - df[f'{ratio_col}_rolling_min']) / \
                         (df[f'{ratio_col}_rolling_max'] - df[f'{ratio_col}_rolling_min'] + 1e-10)

    merged = spy_df.merge(df[['date', ratio_col, 'ratio_pctile']].dropna(),
                          on='date', how='inner')

    mask = merged['ratio_pctile'] < percentile
    signal_dates = merged.loc[mask, 'date'].tolist()
    signal_dates = deduplicate_signals(signal_dates, dedup_days)
    signals = merged[merged['date'].isin(signal_dates)].copy()

    stats = {}
    for label in HORIZONS:
        col = f'fwd_{label}'
        valid = signals[col].dropna()
        if len(valid) > 0:
            stats[label] = {
                'n': len(valid),
                'win_rate': (valid > 0).mean(),
                'avg_return': valid.mean(),
                'median_return': valid.median(),
                'min_return': valid.min(),
                'max_return': valid.max(),
            }

    return BacktestResult(name=name, description=description,
                          signal_dates=signal_dates, stats=stats, signals_df=signals)


# ── Historical Analog Lookup ─────────────────────────────────────────────

def find_analogs(indicator_df: pd.DataFrame, spy_df: pd.DataFrame,
                 indicator_col: str, current_value: float,
                 tolerance_pct: float = 0.05, top_n: int = 10) -> pd.DataFrame:
    """
    Find historical dates where the indicator was near the current value.
    Returns those dates with SPY forward returns.
    """
    merged = spy_df.merge(indicator_df[['date', indicator_col]].dropna(),
                          on='date', how='inner')

    lower = current_value * (1 - tolerance_pct)
    upper = current_value * (1 + tolerance_pct)
    mask = (merged[indicator_col] >= lower) & (merged[indicator_col] <= upper)

    analogs = merged.loc[mask].copy()
    analogs = analogs.sort_values('date', ascending=False).head(top_n)

    return analogs[['date', indicator_col] + [f'fwd_{h}' for h in HORIZONS]]


# ── Run All Backtests ────────────────────────────────────────────────────

def run_all_backtests(verbose: bool = False) -> list:
    """Run the full Meisler indicator backtest suite."""
    print("Loading data...")
    spy = load_spy()
    putcall = load_putcall()
    aaii = load_aaii()
    naaim = load_naaim()

    results = []

    # ── Put/Call Ratio (Meisler's primary tool) ──
    print("\n=== CBOE Equity Put/Call Ratio ===")
    for threshold in [0.70, 0.75, 0.80, 0.85]:
        r = backtest_threshold(putcall, spy, 'pc_10d_ma', threshold, 'above',
                               name=f"PC 10d MA > {threshold}",
                               description=f"Put/Call 10-day MA above {threshold} (elevated fear)")
        results.append(r)

    for threshold in [0.65, 0.70, 0.75]:
        r = backtest_threshold(putcall, spy, 'pc_30d_ma', threshold, 'above',
                               name=f"PC 30d MA > {threshold}",
                               description=f"Put/Call 30-day MA above {threshold} (sustained fear)")
        results.append(r)

    # ── AAII Meisler-specific thresholds ──
    print("\n=== AAII (Meisler Thresholds) ===")
    r = backtest_threshold(aaii, spy, 'bearish', 50, 'above', dedup_days=7,
                           name="AAII Bears > 50%",
                           description="Extreme bearish reading — Meisler watches for this")
    results.append(r)

    r = backtest_threshold(aaii, spy, 'bullish', 25, 'below', dedup_days=7,
                           name="AAII Bulls < 25%",
                           description="Very few bulls — Meisler's 'bulls in the 20s'")
    results.append(r)

    r = backtest_combo(aaii, spy,
                       [('bearish', 50, 'above'), ('bullish', 30, 'below')],
                       dedup_days=7,
                       name="AAII Bears>50 + Bulls<30",
                       description="Meisler's combo: extreme bears + very few bulls = potential bottom")
    results.append(r)

    # ── AAII Spread (existing thresholds for comparison) ──
    r = backtest_threshold(aaii, spy, 'bull_bear_spread', -20, 'below', dedup_days=7,
                           name="AAII Spread < -20%",
                           description="Existing dashboard threshold")
    results.append(r)

    r = backtest_threshold(aaii, spy, 'bull_bear_spread', -30, 'below', dedup_days=7,
                           name="AAII Spread < -30%",
                           description="Extreme pessimism")
    results.append(r)

    # ── NAAIM (existing) ──
    print("\n=== NAAIM ===")
    for threshold in [40, 25]:
        r = backtest_threshold(naaim, spy, 'naaim', threshold, 'below', dedup_days=7,
                               name=f"NAAIM < {threshold}",
                               description=f"Fund managers underweight at {threshold}")
        results.append(r)

    # ── VIX (via SPY data — load from yfinance if needed) ──
    print("\n=== VIX ===")
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        vix_path = DATA_DIR / 'vix_daily.csv'
        if not vix_path.exists():
            vix_data = yf.download('^VIX', start='2000-01-01',
                                   end=datetime.today().strftime('%Y-%m-%d'),
                                   progress=False, auto_adjust=True)
            vix_df = vix_data[['High']].reset_index()
            vix_df.columns = ['date', 'vix']
            vix_df['date'] = pd.to_datetime(vix_df['date']).dt.tz_localize(None)
            vix_df.to_csv(vix_path, index=False)
        else:
            vix_df = pd.read_csv(vix_path, parse_dates=['date'])

        for threshold in [30, 40, 50]:
            r = backtest_threshold(vix_df, spy, 'vix', threshold, 'above',
                                   name=f"VIX > {threshold}",
                                   description=f"Fear gauge above {threshold}")
            results.append(r)
    except Exception as e:
        print(f"  VIX backtest skipped: {e}")

    # ── Sector Relative Strength ──
    print("\n=== Sector Internals ===")
    sectors = load_sector_etfs()
    if not sectors.empty:
        for ratio_col, etf_name in [('igv_spy_ratio', 'IGV (Software)'),
                                     ('kbe_spy_ratio', 'KBE (Banks)')]:
            for pctile in [0.05, 0.10]:
                r = backtest_relative_strength(
                    sectors, spy, ratio_col, lookback=60, percentile=pctile,
                    name=f"{etf_name}/SPY at {int(pctile*100)}th pctile (60d)",
                    description=f"{etf_name} relative to SPY at {int(pctile*100)}th percentile of 60-day range")
                results.append(r)
    else:
        print("  Sector data not available — skipping")

    return results


def print_results(results: list):
    """Print backtest results as a formatted table."""
    print("\n" + "=" * 120)
    print(f"{'Signal':<40} {'N':>5} {'1m WR':>7} {'3m WR':>7} {'3m Avg':>8} {'6m WR':>7} {'12m WR':>7} {'12m Avg':>9}")
    print("-" * 120)

    for r in results:
        if not r.stats:
            print(f"{r.name:<40} {'—':>5}  No signals found")
            continue

        n = r.stats.get('1m', r.stats.get('3m', {})).get('n', 0)
        wr_1m = r.stats.get('1m', {}).get('win_rate', None)
        wr_3m = r.stats.get('3m', {}).get('win_rate', None)
        avg_3m = r.stats.get('3m', {}).get('avg_return', None)
        wr_6m = r.stats.get('6m', {}).get('win_rate', None)
        wr_12m = r.stats.get('12m', {}).get('win_rate', None)
        avg_12m = r.stats.get('12m', {}).get('avg_return', None)

        def fmt_wr(v): return f"{v*100:.0f}%" if v is not None else "n/a"
        def fmt_ret(v): return f"{v*100:+.1f}%" if v is not None else "n/a"

        print(f"{r.name:<40} {n:>5} {fmt_wr(wr_1m):>7} {fmt_wr(wr_3m):>7} "
              f"{fmt_ret(avg_3m):>8} {fmt_wr(wr_6m):>7} {fmt_wr(wr_12m):>7} {fmt_ret(avg_12m):>9}")

    print("=" * 120)


def results_to_dataframe(results: list) -> pd.DataFrame:
    """Convert backtest results to a DataFrame for dashboard display."""
    rows = []
    for r in results:
        if not r.stats:
            continue
        row = {'signal': r.name, 'description': r.description}
        for label in HORIZONS:
            s = r.stats.get(label, {})
            row[f'{label}_n'] = s.get('n', 0)
            row[f'{label}_wr'] = s.get('win_rate', None)
            row[f'{label}_avg'] = s.get('avg_return', None)
            row[f'{label}_med'] = s.get('median_return', None)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import sys
    verbose = '--verbose' in sys.argv
    results = run_all_backtests(verbose=verbose)
    print_results(results)
