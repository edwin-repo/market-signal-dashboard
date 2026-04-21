#!/usr/bin/env python3
"""
COT Backtest Engine — 3 Methods × 37 Contracts

Method A: Current dashboard (single type percentile extreme, buy & hold)
Method B: Hybrid (all 3 types aligned, price reversal entry, dynamic exit at neutral)
Method C: Shapiro strict (all 3 at ≤5/≥95, reversal entry, exit at neutral, equities exception)

Usage:
    python run_cot_backtests.py          # Run all backtests, save to CSV
    python run_cot_backtests.py --quick  # Only run contracts without cached results
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import timedelta
import warnings
import sys

warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent / 'data'


# ── Price Data ────────────────────────────────────────────────────────────

def load_futures_price(yf_ticker: str, start: str = '2006-01-01') -> pd.DataFrame:
    """Download and cache daily price data for a futures contract or ETF."""
    safe_name = yf_ticker.replace('=', '_').replace('-', '_').replace('/', '_')
    cache_path = DATA_DIR / f'price_{safe_name}.csv'

    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=['date'])
        # Refresh if stale (>7 days old)
        if (pd.Timestamp.now() - df['date'].max()).days < 7:
            return df

    try:
        raw = yf.download(yf_ticker, start=start, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        if raw.empty:
            return pd.DataFrame()
        df = raw[['Open', 'Close']].reset_index()
        df.columns = ['date', 'open', 'close']
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df = df.sort_values('date').reset_index(drop=True)
        df.to_csv(cache_path, index=False)
        return df
    except Exception as e:
        print(f"    Price download failed for {yf_ticker}: {e}")
        return pd.DataFrame()


# ── Rolling Percentile ────────────────────────────────────────────────────

def compute_rolling_percentile(series: pd.Series, window: int = 156) -> pd.Series:
    """Rolling percentile rank over trailing window (156 weeks ≈ 3 years)."""
    def pctile_rank(x):
        if len(x) < 20:
            return np.nan
        return (x.values[:-1] < x.values[-1]).sum() / (len(x) - 1)
    return series.rolling(window, min_periods=52).apply(pctile_rank, raw=False)


def add_percentiles(df: pd.DataFrame, window: int = 156) -> pd.DataFrame:
    """Add rolling percentile columns for all 3 trader types."""
    df = df.copy()
    for col in ['spec_net', 'comm_net', 'small_spec_net']:
        if col in df.columns:
            df[f'{col}_pctile'] = compute_rolling_percentile(df[col], window)
    return df


# ── Helpers ───────────────────────────────────────────────────────────────

def find_reversal_entry(price_df: pd.DataFrame, signal_date: pd.Timestamp,
                        direction: str, window: int = 5):
    """
    After COT signal date, scan next `window` trading days for a reversal candle.
    Approximates Shapiro's "news failure" concept.

    direction='long': look for up-close day (close > open) = bullish reversal
    direction='short': look for down-close day (close < open) = bearish reversal

    Returns (entry_date, entry_price) or (None, None)
    """
    mask = (price_df['date'] > signal_date) & (price_df['date'] <= signal_date + timedelta(days=window * 2))
    candidates = price_df[mask].head(window)

    for _, row in candidates.iterrows():
        if direction == 'long' and row['close'] > row['open']:
            return row['date'], row['close']
        elif direction == 'short' and row['close'] < row['open']:
            return row['date'], row['close']

    return None, None


def find_dynamic_exit(cot_df: pd.DataFrame, entry_date: pd.Timestamp,
                      neutral: float = 0.50, max_weeks: int = 52):
    """
    Scan forward in weekly COT data until any trader type's percentile crosses neutral.
    Returns (exit_date, weeks_held) or (None, None) if no exit within max_weeks.
    """
    future = cot_df[cot_df['date'] > entry_date].head(max_weeks)

    for _, row in future.iterrows():
        for col in ['spec_net_pctile', 'comm_net_pctile', 'small_spec_net_pctile']:
            if col in row.index and not pd.isna(row[col]):
                val = row[col]
                # Neutral = between 0.40 and 0.60 (centered around 0.50)
                if 0.40 <= val <= 0.60:
                    weeks = (row['date'] - entry_date).days / 7
                    return row['date'], weeks

    return None, None


def get_price_at_date(price_df: pd.DataFrame, target_date: pd.Timestamp,
                      tolerance_days: int = 5):
    """Get the closing price nearest to target_date."""
    mask = (price_df['date'] >= target_date) & (price_df['date'] <= target_date + timedelta(days=tolerance_days))
    candidates = price_df[mask]
    if candidates.empty:
        # Try backward
        mask2 = (price_df['date'] >= target_date - timedelta(days=tolerance_days)) & (price_df['date'] <= target_date)
        candidates = price_df[mask2]
    if candidates.empty:
        return None
    return candidates.iloc[0]['close']


# ── Method A: Current Dashboard ──────────────────────────────────────────

def backtest_method_a(cot_df: pd.DataFrame, price_df: pd.DataFrame,
                      trader_type: str = 'spec', threshold: float = 0.10,
                      dedup_days: int = 30) -> dict:
    """
    Method A — Current approach: single trader type percentile extreme, buy & hold.

    For each contract, tests:
      - Long: spec/comm percentile < threshold → buy, hold fixed horizons
      - Also tests short: spec/comm percentile > (1-threshold) → sell

    Returns dict with 'long' and 'short' sub-dicts, each with horizon stats.
    """
    pctile_col = f'{trader_type}_net_pctile'
    if pctile_col not in cot_df.columns:
        return {'long': {}, 'short': {}, 'n_long': 0, 'n_short': 0}

    # Merge COT with price for forward returns
    cot = cot_df[['date', pctile_col]].dropna().copy()
    price = price_df[['date', 'close']].copy()

    merged = pd.merge_asof(cot.sort_values('date'), price.sort_values('date'),
                           on='date', direction='forward', tolerance=pd.Timedelta(days=5))
    merged = merged.dropna(subset=['close'])

    # Compute forward returns
    horizons = {'3m': 63, '6m': 126, '12m': 252}
    for label, days in horizons.items():
        merged[f'fwd_{label}'] = price.set_index('date').reindex(
            merged['date'] + timedelta(days=days), method='nearest', tolerance=timedelta(days=10)
        )['close'].values / merged['close'].values - 1
        # Fix: use merge_asof for forward prices
    # Better approach: look up each future price
    for label, days in horizons.items():
        fwd_returns = []
        for _, row in merged.iterrows():
            future_date = row['date'] + timedelta(days=days)
            fwd_price = get_price_at_date(price, future_date, tolerance_days=10)
            if fwd_price is not None:
                fwd_returns.append(fwd_price / row['close'] - 1)
            else:
                fwd_returns.append(np.nan)
        merged[f'fwd_{label}'] = fwd_returns

    results = {}
    for direction, op, thresh in [('long', 'lt', threshold), ('short', 'gt', 1 - threshold)]:
        if op == 'lt':
            mask = merged[pctile_col] < thresh
        else:
            mask = merged[pctile_col] > thresh

        signal_dates = merged.loc[mask, 'date'].tolist()

        # Deduplicate
        if signal_dates:
            sorted_dates = sorted(signal_dates)
            deduped = [sorted_dates[0]]
            for d in sorted_dates[1:]:
                if (d - deduped[-1]).days >= dedup_days:
                    deduped.append(d)
            signal_dates = deduped

        signals = merged[merged['date'].isin(signal_dates)].copy()

        horizon_stats = {}
        for label in horizons:
            col = f'fwd_{label}'
            valid = signals[col].dropna()
            if len(valid) >= 3:
                # For short trades, negate returns
                returns = -valid if direction == 'short' else valid
                horizon_stats[label] = {
                    'n': len(returns),
                    'win_rate': (returns > 0).mean(),
                    'avg_return': returns.mean(),
                    'avg_loss': returns[returns < 0].mean() if (returns < 0).any() else 0.0,
                }

        results[direction] = horizon_stats
        results[f'n_{direction}'] = len(signal_dates)

    return results


# ── Method B: Hybrid ─────────────────────────────────────────────────────

def backtest_method_b(cot_df: pd.DataFrame, price_df: pd.DataFrame,
                      threshold: float = 0.10, reversal_window: int = 5,
                      neutral_level: float = 0.50, is_equity: bool = False) -> dict:
    """
    Method B — Hybrid: all 3 types extreme + price reversal entry + dynamic exit.

    Long setup: spec < threshold AND small_spec < threshold AND comm > (1-threshold)
    Short setup: spec > (1-threshold) AND small_spec > (1-threshold) AND comm < threshold
    Equity exception: only comm needs to be extreme.
    """
    required_cols = ['spec_net_pctile', 'comm_net_pctile']
    if not is_equity:
        required_cols.append('small_spec_net_pctile')

    for col in required_cols:
        if col not in cot_df.columns:
            return {'long': {}, 'short': {}, 'n_long': 0, 'n_short': 0}

    trades = []

    for _, row in cot_df.iterrows():
        spec_p = row.get('spec_net_pctile', np.nan)
        comm_p = row.get('comm_net_pctile', np.nan)
        small_p = row.get('small_spec_net_pctile', np.nan)

        if pd.isna(spec_p) or pd.isna(comm_p):
            continue
        if not is_equity and pd.isna(small_p):
            continue

        signal_date = row['date']
        direction = None

        if is_equity:
            # Equity: only commercials needed (Shapiro's rule)
            if comm_p > (1 - threshold):
                direction = 'long'
            elif comm_p < threshold:
                direction = 'short'
        else:
            # All 3 types must be aligned
            if (spec_p < threshold and comm_p > (1 - threshold) and
                    small_p < threshold):
                direction = 'long'
            elif (spec_p > (1 - threshold) and comm_p < threshold and
                  small_p > (1 - threshold)):
                direction = 'short'

        if direction is None:
            continue

        # Check for reversal entry
        entry_date, entry_price = find_reversal_entry(price_df, signal_date,
                                                       direction, reversal_window)
        if entry_date is None:
            continue

        # Find dynamic exit
        exit_date, weeks_held = find_dynamic_exit(cot_df, entry_date, neutral_level)

        if exit_date is not None:
            exit_price = get_price_at_date(price_df, exit_date)
            if exit_price is not None:
                raw_return = (exit_price / entry_price - 1)
                trade_return = raw_return if direction == 'long' else -raw_return
                trades.append({
                    'date': signal_date,
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'weeks_held': weeks_held,
                })

    if not trades:
        return {'long': {}, 'short': {}, 'n_long': 0, 'n_short': 0, 'trades': []}

    trades_df = pd.DataFrame(trades)

    # Deduplicate overlapping trades (minimum 30 days apart)
    trades_df = trades_df.sort_values('date').reset_index(drop=True)
    keep = [0]
    for i in range(1, len(trades_df)):
        if (trades_df.loc[i, 'date'] - trades_df.loc[keep[-1], 'date']).days >= 30:
            keep.append(i)
    trades_df = trades_df.loc[keep].reset_index(drop=True)

    results = {'trades': trades_df.to_dict('records')}

    for direction in ['long', 'short']:
        subset = trades_df[trades_df['direction'] == direction]
        results[f'n_{direction}'] = len(subset)
        if len(subset) >= 2:
            results[direction] = {
                'n': len(subset),
                'win_rate': (subset['return'] > 0).mean(),
                'avg_return': subset['return'].mean(),
                'avg_loss': subset.loc[subset['return'] < 0, 'return'].mean() if (subset['return'] < 0).any() else 0.0,
                'avg_weeks': subset['weeks_held'].mean(),
                'reward_risk': abs(subset.loc[subset['return'] > 0, 'return'].mean() /
                                  subset.loc[subset['return'] < 0, 'return'].mean())
                    if (subset['return'] < 0).any() and (subset['return'] > 0).any() else np.nan,
            }
        else:
            results[direction] = {}

    return results


# ── Method C: Shapiro Strict ─────────────────────────────────────────────

def backtest_method_c(cot_df: pd.DataFrame, price_df: pd.DataFrame,
                      threshold: float = 0.05, reversal_window: int = 5,
                      neutral_level: float = 0.50, is_equity: bool = False) -> dict:
    """
    Method C — Shapiro strict: all 3 types at ≤5th or ≥95th percentile.
    Equities only need commercials.
    Same entry/exit as Method B but with stricter thresholds.
    """
    return backtest_method_b(cot_df, price_df,
                             threshold=threshold,
                             reversal_window=reversal_window,
                             neutral_level=neutral_level,
                             is_equity=is_equity)


# ── Main Runner ──────────────────────────────────────────────────────────

def run_all_backtests():
    """Run all 3 methods for all 37 contracts and save summary CSV."""
    from cot_config import COT_CONTRACTS
    from cot_backtest import extract_all_contracts

    print("=" * 100)
    print("COT Backtest Engine — 3 Methods × 37 Contracts")
    print("=" * 100)

    # Step 1: Extract all contracts
    print("\n[1] Extracting all 37 contracts...")
    all_cot = extract_all_contracts()
    print(f"    Loaded {len(all_cot)} contracts")

    # Step 2: Add percentiles
    print("\n[2] Computing rolling percentiles...")
    for key in all_cot:
        all_cot[key] = add_percentiles(all_cot[key])

    # Step 3: Download price data
    print("\n[3] Downloading price data...")
    prices = {}
    for key, cfg in COT_CONTRACTS.items():
        ticker = cfg['yf']
        if ticker not in prices:
            print(f"    {ticker}...", end=' ', flush=True)
            df = load_futures_price(ticker)
            if not df.empty:
                prices[ticker] = df
                print(f"OK ({len(df)} rows)")
            else:
                print("FAILED")

    # Step 4: Run backtests
    print("\n[4] Running backtests...")
    rows = []

    for key, cfg in COT_CONTRACTS.items():
        label = cfg['label']
        sector = cfg['sector']
        ticker = cfg['yf']
        is_eq = cfg.get('equity', False)

        cot = all_cot.get(key)
        price = prices.get(ticker)

        if cot is None or cot.empty:
            print(f"  ⚠ {label}: No COT data")
            continue
        if price is None or price.empty:
            print(f"  ⚠ {label}: No price data ({ticker})")
            continue

        print(f"  {label:25s}", end=' ', flush=True)

        # Method A: single type, threshold=0.10
        trader = 'comm' if is_eq else 'spec'
        res_a = backtest_method_a(cot, price, trader_type=trader, threshold=0.10)

        # Method B: hybrid, threshold=0.10
        res_b = backtest_method_b(cot, price, threshold=0.10, is_equity=is_eq)

        # Method C: strict, threshold=0.05
        res_c = backtest_method_c(cot, price, threshold=0.05, is_equity=is_eq)

        # Extract best horizon for Method A
        best_a_wr = 0
        best_a_horizon = '12m'
        for direction in ['long', 'short']:
            for h, stats in res_a.get(direction, {}).items():
                if isinstance(stats, dict) and stats.get('win_rate', 0) > best_a_wr:
                    best_a_wr = stats['win_rate']
                    best_a_horizon = h

        # Build summary row
        for method, res, method_name in [('A', res_a, 'Current'),
                                          ('B', res_b, 'Hybrid'),
                                          ('C', res_c, 'Shapiro')]:
            for direction in ['long', 'short']:
                stats = res.get(direction, {})
                if not stats:
                    continue

                row = {
                    'contract': key,
                    'label': label,
                    'sector': sector,
                    'method': method,
                    'method_name': method_name,
                    'direction': direction,
                    'n_signals': stats.get('n', 0),
                    'win_rate': stats.get('win_rate', np.nan),
                    'avg_return': stats.get('avg_return', np.nan),
                    'avg_loss': stats.get('avg_loss', np.nan),
                    'avg_weeks': stats.get('avg_weeks', np.nan),
                    'reward_risk': stats.get('reward_risk', np.nan),
                }

                # For Method A, pick best horizon
                if method == 'A':
                    for h in ['3m', '6m', '12m']:
                        h_stats = res.get(direction, {}).get(h, {})
                        if h_stats:
                            row.update({
                                'n_signals': h_stats.get('n', 0),
                                'win_rate': h_stats.get('win_rate', np.nan),
                                'avg_return': h_stats.get('avg_return', np.nan),
                                'avg_loss': h_stats.get('avg_loss', np.nan),
                                'horizon': h,
                            })
                            break  # Take first available horizon

                rows.append(row)

        n_a = res_a.get('n_long', 0) + res_a.get('n_short', 0)
        n_b = res_b.get('n_long', 0) + res_b.get('n_short', 0)
        n_c = res_c.get('n_long', 0) + res_c.get('n_short', 0)
        print(f"A:{n_a:3d} B:{n_b:3d} C:{n_c:3d} signals")

    # Step 5: Save summary
    summary_df = pd.DataFrame(rows)
    output_path = DATA_DIR / 'cot_backtest_summary.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"\n{'=' * 100}")
    print(f"Saved {len(summary_df)} result rows to {output_path}")

    # Print overview
    if not summary_df.empty:
        print(f"\n{'=' * 100}")
        print("SUMMARY BY METHOD")
        print(f"{'=' * 100}")
        for method in ['A', 'B', 'C']:
            sub = summary_df[summary_df['method'] == method]
            if sub.empty:
                continue
            name = sub['method_name'].iloc[0]
            n_contracts = sub['contract'].nunique()
            avg_wr = sub['win_rate'].mean()
            avg_ret = sub['avg_return'].mean()
            total_signals = sub['n_signals'].sum()
            print(f"\n  Method {method} ({name}):")
            print(f"    Contracts with signals: {n_contracts}")
            print(f"    Total signals: {total_signals}")
            print(f"    Average win rate: {avg_wr*100:.1f}%")
            print(f"    Average return: {avg_ret*100:+.2f}%")

            # Top 5 by win rate
            top = sub.nlargest(5, 'win_rate')
            print(f"    Top 5:")
            for _, r in top.iterrows():
                print(f"      {r['label']:20s} {r['direction']:5s} WR={r['win_rate']*100:.0f}% "
                      f"Avg={r['avg_return']*100:+.1f}% (n={r['n_signals']})")

    return summary_df


if __name__ == '__main__':
    run_all_backtests()
