"""
Hormuz Strait Transit Monitor — Scoring Engine & Data Fetchers

Scrapes hormuztracker.com for real vessel transit data, combines with
Yahoo Finance market data, and computes a composite normalization score (0-100).

Signal states:
  GREEN (70-100): Normalization confirmed — physical flow resuming
  AMBER (30-70):  Headlines vs reality gap — fade rallies
  RED   (0-30):   Strait still closed — stay hedged
"""

import requests
import re
import yfinance as yf
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

# ── Constants ──

BASELINE_DAILY_TRANSITS = 138   # pre-crisis avg from hormuztracker
BASELINE_BRENT = 72.0           # Jan 2026 pre-crisis avg
THRESHOLD_GREEN = 70
THRESHOLD_AMBER = 30

MAINSTREAM_CARRIERS = {
    "Maersk", "MSC", "CMA CGM", "Hapag-Lloyd",
    "COSCO", "ONE", "HMM", "Evergreen", "PIL"
}

WEIGHTS = {
    'transit':   0.35,
    'flag':      0.20,
    'insurance': 0.15,
    'oil':       0.15,
    'vix':       0.15,
}


# ── Scoring Functions ──

def score_transit_count(daily: int, baseline: int = BASELINE_DAILY_TRANSITS) -> float:
    """0-100 based on % of pre-crisis baseline."""
    if baseline <= 0:
        return 0.0
    return min(100.0, (daily / baseline) * 100)


def score_carrier_status(carriers: list[dict], mainstream_active_count: int = 0) -> float:
    """
    All 9 suspended = 0-10.
    Some resuming = proportional.
    All active = 100.
    """
    total = len(carriers) if carriers else 9
    suspended = sum(1 for c in carriers if c.get('status', '').lower() == 'suspended')
    active = total - suspended
    if active == 0:
        return 5.0  # minimal — some ships still transit under flags
    return min(100.0, (active / total) * 100)


def score_oil_normalization(brent: float, baseline: float = BASELINE_BRENT) -> float:
    """How far Brent has returned to pre-crisis baseline."""
    if brent <= baseline:
        return 100.0
    premium_pct = ((brent - baseline) / baseline) * 100
    return max(0.0, 100 - premium_pct * 2.5)


def score_vix(vix: float) -> float:
    """VIX 12-15 = 100, VIX 25 = ~67, VIX 45+ = 0."""
    if vix <= 15:
        return 100.0
    if vix >= 45:
        return 0.0
    return max(0.0, 100 - (vix - 15) * 3.33)


def score_insurance_proxy(insurance_multiplier: float) -> float:
    """
    Insurance premium multiplier vs pre-crisis.
    1x (normal) = 100. 8x+ = 0.
    """
    if insurance_multiplier <= 1:
        return 100.0
    if insurance_multiplier >= 10:
        return 0.0
    return max(0.0, 100 - (insurance_multiplier - 1) * 11.1)


def compute_composite(scores: dict) -> float:
    """Weighted composite score 0-100."""
    return round(sum(scores.get(k, 0) * WEIGHTS[k] for k in WEIGHTS), 1)


def get_signal_state(score: float) -> tuple[str, str, str]:
    """Returns (state, emoji, color) based on composite score."""
    if score >= THRESHOLD_GREEN:
        return "GREEN", "🟢", "#3fb950"
    elif score >= THRESHOLD_AMBER:
        return "AMBER", "🟡", "#e3b341"
    else:
        return "RED", "🔴", "#ff4444"


def get_trade_action(state: str) -> str:
    """Trading signal based on signal state."""
    actions = {
        "GREEN": "Normalization confirmed — short VIX, long SPY, trim energy/gold hedges",
        "AMBER": "Headline deal, no physical flow — fade rally, stay hedged, sell-the-news setup",
        "RED": "Strait still closed — long vol, long gold, long XLE, defensive positioning",
    }
    return actions.get(state, "")


# ── Data Fetchers ──

def fetch_hormuz_data() -> dict:
    """Scrape hormuztracker.com for transit data."""
    data = {
        'transit_count': 0,
        'baseline': BASELINE_DAILY_TRANSITS,
        'days_disruption': 0,
        'carriers_suspended': 0,
        'carriers_total': 9,
        'vessels_trapped': 0,
        'insurance_multiplier': 1.0,
        'bypass_pct': 0,
        'carriers': [],
        'source': 'hormuztracker.com',
        'timestamp': datetime.utcnow().isoformat(),
        'error': None,
    }

    try:
        r = requests.get(
            'https://www.hormuztracker.com',
            headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'},
            timeout=15
        )
        if r.status_code != 200:
            data['error'] = f"HTTP {r.status_code}"
            return data

        soup = BeautifulSoup(r.text, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        lines = text.split('\n')

        # Transit count: pattern is "~" → number → "-" → pct → "%" → "vs X avg"
        for i, line in enumerate(lines):
            if line.strip() == '~' and i + 1 < len(lines):
                try:
                    data['transit_count'] = int(lines[i + 1].strip())
                except ValueError:
                    pass
                break

        # Baseline from "vs X avg"
        for line in lines:
            m = re.search(r'vs\s+(\d+)\s+avg', line)
            if m:
                data['baseline'] = int(m.group(1))
                break

        # Days of disruption
        for i, line in enumerate(lines):
            if 'Days of disruption' in line:
                for j in range(i - 1, max(0, i - 5), -1):
                    if re.match(r'^\d+$', lines[j].strip()):
                        data['days_disruption'] = int(lines[j].strip())
                        break
                break

        # Carriers suspended count
        for i, line in enumerate(lines):
            if 'Carriers suspended' in line:
                for j in range(i - 1, max(0, i - 5), -1):
                    if re.match(r'^\d+$', lines[j].strip()):
                        data['carriers_suspended'] = int(lines[j].strip())
                        break
                break

        # Vessels trapped
        m = re.search(r'(\d+)\+?\s*vessels?\s+trapped', text)
        if m:
            data['vessels_trapped'] = int(m.group(1))

        # Insurance multiplier
        m = re.search(r'(\d+)x\s+increase', text, re.IGNORECASE)
        if m:
            data['insurance_multiplier'] = int(m.group(1))

        # Bypass percentage
        for i, line in enumerate(lines):
            if 'Pipeline bypass' in line or 'bypass' in line.lower():
                for j in range(max(0, i - 3), min(len(lines), i + 3)):
                    m2 = re.search(r'(\d+)%', lines[j])
                    if m2:
                        data['bypass_pct'] = int(m2.group(1))
                        break
                break

        # Individual carrier details
        carrier_names = list(MAINSTREAM_CARRIERS)
        carriers = []
        for i, line in enumerate(lines):
            if line.strip() in carrier_names:
                status = lines[i + 1].strip() if i + 1 < len(lines) else 'Unknown'
                vessels_trapped = 0
                m = re.search(r'(\d+)\s+vessels?\s+trapped', ' '.join(lines[i:i+5]))
                if m:
                    vessels_trapped = int(m.group(1))
                carriers.append({
                    'name': line.strip(),
                    'status': status,
                    'vessels_trapped': vessels_trapped,
                })
        data['carriers'] = carriers
        data['carriers_total'] = len(carriers) if carriers else 9

    except Exception as e:
        data['error'] = str(e)

    return data


def fetch_market_data() -> dict:
    """Fetch VIX, Brent, WTI, SPY, Gold from Yahoo Finance."""
    tickers = {
        'vix': '^VIX',
        'brent': 'BZ=F',
        'wti': 'CL=F',
        'spy': 'SPY',
        'gold': 'GC=F',
    }
    market = {}
    for key, ticker in tickers.items():
        try:
            df = yf.download(ticker, period='5d', auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            col = 'High' if key == 'vix' else 'Close'
            series = df[col].dropna()
            series = series[series > 0]
            if not series.empty:
                market[key] = round(float(series.iloc[-1]), 2)
        except Exception:
            pass
    return market


def log_daily_transit(transit_count: int, data_dir: str = 'data'):
    """Append today's transit count to hormuz_daily.csv if not already logged."""
    from pathlib import Path
    path = Path(data_dir) / 'hormuz_daily.csv'
    today = datetime.now().strftime('%Y-%m-%d')

    if path.exists():
        df = pd.read_csv(path, parse_dates=['date'])
        if today in df['date'].astype(str).values:
            # Update today's row if count changed
            df.loc[df['date'].astype(str) == today, 'transit_count'] = transit_count
            df.loc[df['date'].astype(str) == today, 'source'] = 'hormuztracker'
            df.to_csv(path, index=False)
            return
    else:
        df = pd.DataFrame(columns=['date', 'transit_count', 'source'])

    new_row = pd.DataFrame([{
        'date': today,
        'transit_count': transit_count,
        'source': 'hormuztracker',
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(path, index=False)


def load_hormuz_history(data_dir: str = 'data') -> pd.DataFrame:
    """Load historical daily transit counts."""
    from pathlib import Path
    path = Path(data_dir) / 'hormuz_daily.csv'
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=['date'])
    return df.sort_values('date').reset_index(drop=True)


def compute_hormuz_signal() -> dict:
    """Full pipeline: fetch data → score → signal."""
    hormuz = fetch_hormuz_data()
    market = fetch_market_data()

    # Auto-log today's transit count
    if hormuz['transit_count'] > 0:
        log_daily_transit(hormuz['transit_count'])

    # Compute sub-scores
    scores = {
        'transit': score_transit_count(hormuz['transit_count'], hormuz['baseline']),
        'flag': score_carrier_status(hormuz['carriers']),
        'insurance': score_insurance_proxy(hormuz['insurance_multiplier']),
        'oil': score_oil_normalization(market.get('brent', 106.0)),
        'vix': score_vix(market.get('vix', 30.0)),
    }

    composite = compute_composite(scores)
    state, emoji, color = get_signal_state(composite)
    action = get_trade_action(state)

    return {
        'composite_score': composite,
        'signal_state': state,
        'signal_emoji': emoji,
        'signal_color': color,
        'trade_action': action,
        'scores': scores,
        'weights': WEIGHTS,
        'hormuz': hormuz,
        'market': market,
        'timestamp': datetime.now().isoformat(),
    }


if __name__ == '__main__':
    import json
    result = compute_hormuz_signal()
    print(json.dumps(result, indent=2, default=str))
