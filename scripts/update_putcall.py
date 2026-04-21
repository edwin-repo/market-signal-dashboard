#!/usr/bin/env python3
"""
Download CBOE put/call ratio data and update data/putcall.csv.

Sources:
  1. Legacy CSV archives (2003–2019): cdn.cboe.com bulk CSVs
  2. Daily JSON API (2019-10-07–present): cdn.cboe.com/data/us/options/market_statistics/daily/

The script is incremental — it reads existing putcall.csv and only fetches
dates not already present.
"""

import requests
import pandas as pd
import numpy as np
import io
import time
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
HEADERS = {'User-Agent': 'Mozilla/5.0'}

JSON_API_START = pd.Timestamp('2019-10-07')
JSON_API_URL = 'https://cdn.cboe.com/data/us/options/market_statistics/daily/{}_daily_options'


def download_legacy_csv():
    """Download legacy CBOE CSV archives (2003–2019)."""
    frames = []

    # Archive: 2003-2012
    try:
        r = requests.get('https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypcarchive.csv',
                         headers=HEADERS, timeout=15)
        if r.status_code == 200:
            lines = r.text.strip().split('\n')
            df = pd.read_csv(io.StringIO('\n'.join(lines[2:])))
            df.columns = [c.strip() for c in df.columns]
            df['date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
            df['equity_pc_ratio'] = pd.to_numeric(df['Equity P/C Ratio'], errors='coerce')
            frames.append(df[['date', 'equity_pc_ratio']].dropna())
            print(f"  Archive CSV: {len(frames[-1])} rows")
    except Exception as e:
        print(f"  Archive CSV failed: {e}")

    # Main: 2006-2019
    try:
        r = requests.get('https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv',
                         headers=HEADERS, timeout=15)
        if r.status_code == 200:
            lines = r.text.strip().split('\n')
            df = pd.read_csv(io.StringIO('\n'.join(lines[2:])))
            df.columns = [c.strip() for c in df.columns]
            df['date'] = pd.to_datetime(df['DATE'], format='%m/%d/%Y', errors='coerce')
            df['equity_pc_ratio'] = pd.to_numeric(df['P/C Ratio'], errors='coerce')
            frames.append(df[['date', 'equity_pc_ratio']].dropna())
            print(f"  Main CSV:    {len(frames[-1])} rows")
    except Exception as e:
        print(f"  Main CSV failed: {e}")

    # Total and index ratios from legacy CSVs
    extra = {}
    for name, url, col in [
        ('total', 'https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv', 'total_pc_ratio'),
        ('index', 'https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpc.csv', 'index_pc_ratio'),
    ]:
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                lines = r.text.strip().split('\n')
                df = pd.read_csv(io.StringIO('\n'.join(lines[2:])))
                df.columns = [c.strip() for c in df.columns]
                df['date'] = pd.to_datetime(df['DATE'], format='%m/%d/%Y', errors='coerce')
                df[col] = pd.to_numeric(df['P/C Ratio'], errors='coerce')
                extra[col] = df[['date', col]].dropna()
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames).drop_duplicates(subset='date', keep='last').sort_values('date').reset_index(drop=True)
    for col, df in extra.items():
        combined = combined.merge(df, on='date', how='left')

    return combined


def fetch_daily_json(date_str):
    """Fetch a single day from the CBOE daily JSON API."""
    url = JSON_API_URL.format(date_str)
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        ratios = {item['name']: item['value'] for item in data.get('ratios', [])}
        equity = float(ratios.get('EQUITY PUT/CALL RATIO', np.nan))
        total = float(ratios.get('TOTAL PUT/CALL RATIO', np.nan))
        index = float(ratios.get('INDEX PUT/CALL RATIO', np.nan))
        return {'date': date_str, 'equity_pc_ratio': equity,
                'total_pc_ratio': total, 'index_pc_ratio': index}
    except Exception:
        return None


def trading_days(start, end):
    """Generate weekdays (Mon-Fri) between start and end."""
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon=0, Fri=4
            yield d
        d += timedelta(days=1)


def download_json_api(existing_dates):
    """Download recent data from CBOE daily JSON API, skipping dates we already have."""
    today = pd.Timestamp.now().normalize()
    rows = []
    dates_to_fetch = [d for d in trading_days(JSON_API_START, today)
                      if d not in existing_dates]

    if not dates_to_fetch:
        print("  JSON API: already up to date")
        return pd.DataFrame()

    print(f"  JSON API: fetching {len(dates_to_fetch)} days ({dates_to_fetch[0].strftime('%Y-%m-%d')} to {dates_to_fetch[-1].strftime('%Y-%m-%d')})...")

    fetched = 0
    failed = 0
    for i, d in enumerate(dates_to_fetch):
        date_str = d.strftime('%Y-%m-%d')
        result = fetch_daily_json(date_str)
        if result:
            rows.append(result)
            fetched += 1
        else:
            failed += 1  # likely a holiday

        # Progress every 200 days
        if (i + 1) % 200 == 0:
            print(f"    ... {i+1}/{len(dates_to_fetch)} ({fetched} fetched, {failed} holidays/errors)")

        # Small delay to be respectful
        if (i + 1) % 50 == 0:
            time.sleep(0.5)

    print(f"  JSON API: {fetched} rows fetched, {failed} holidays/gaps skipped")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    return df


def update():
    """Main update: load existing data, fetch missing dates, save."""
    out_path = DATA_DIR / 'putcall.csv'

    # Load existing
    if out_path.exists():
        existing = pd.read_csv(out_path, parse_dates=['date'])
        print(f"Existing data: {len(existing)} rows ({existing.date.min().strftime('%Y-%m-%d')} to {existing.date.max().strftime('%Y-%m-%d')})")
    else:
        existing = pd.DataFrame()
        print("No existing data — full download")

    existing_dates = set(existing['date'].dt.normalize()) if not existing.empty else set()

    # Fetch legacy CSV if we have no data before 2019
    if existing.empty or existing.date.min() > pd.Timestamp('2010-01-01'):
        print("\nDownloading legacy CSV archives...")
        legacy = download_legacy_csv()
        if not legacy.empty:
            existing = pd.concat([existing, legacy]).drop_duplicates(subset='date', keep='last')
            existing_dates = set(existing['date'].dt.normalize())

    # Fetch recent JSON API data
    print("\nDownloading recent data from CBOE JSON API...")
    recent = download_json_api(existing_dates)

    if not recent.empty:
        combined = pd.concat([existing, recent]).drop_duplicates(subset='date', keep='last')
    else:
        combined = existing

    combined = combined.sort_values('date').reset_index(drop=True)

    # Save
    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined)} rows to {out_path}")
    print(f"Date range: {combined.date.min().strftime('%Y-%m-%d')} to {combined.date.max().strftime('%Y-%m-%d')}")


if __name__ == '__main__':
    update()
