"""
News Sentiment Module — Shapiro Divergence Detector

Scrapes authoritative financial news from 5 major outlets (Reuters, Bloomberg,
WSJ, CNBC, FT), classifies each headline's market sentiment via Claude, and
aggregates to daily scores. Maintains a sentiment cache to avoid re-classifying
seen headlines. Appends daily scores to CSV for historical tracking.
"""

from __future__ import annotations
import os
import json
import hashlib
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import feedparser

# Load API key from .env (override any pre-existing env vars)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env', override=True)
except ImportError:
    pass

import anthropic


# ── Configuration ──────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / 'data'
SENTIMENT_CACHE = DATA_DIR / 'sentiment_cache.json'
SENTIMENT_CSV = DATA_DIR / 'news_sentiment.csv'
HEADLINES_LOG = DATA_DIR / 'headlines_log.csv'
FETCH_LOG = DATA_DIR / 'news_fetch_log.csv'  # tracks fetched date ranges for cache

# Authoritative financial news domains for GDELT queries
# GDELT has query length limits, so we query multiple domains separately
GDELT_TOP_DOMAINS = [
    'cnbc.com', 'reuters.com', 'wsj.com', 'marketwatch.com',
]

# Authoritative RSS feeds — widely read, reliable, free
NEWS_FEEDS = {
    'Reuters':      'https://news.google.com/rss/search?q=when:2d+site:reuters.com+market+OR+stocks+OR+fed+OR+economy',
    'Bloomberg':    'https://news.google.com/rss/search?q=when:2d+site:bloomberg.com+market+OR+stocks+OR+fed',
    'WSJ Markets':  'https://feeds.a.dj.com/rss/RSSMarketsMain.xml',
    'CNBC Markets': 'https://www.cnbc.com/id/15839069/device/rss/rss.html',
    'CNBC Top':     'https://www.cnbc.com/id/100003114/device/rss/rss.html',
    'FT Markets':   'https://www.ft.com/markets?format=rss',
    'MarketWatch':  'https://feeds.content.dowjones.io/public/rss/mw_topstories',
    'BBC Business': 'https://feeds.bbci.co.uk/news/business/rss.xml',
}

CLAUDE_MODEL = "claude-haiku-4-5"  # Fast & cheap; upgrade to sonnet if needed

# Only classify headlines with these keywords to save API calls (market-relevant)
MARKET_KEYWORDS = re.compile(
    r'\b(stock|stocks|market|markets|fed|powell|rate|rates|inflation|cpi|jobs|nfp|'
    r'earnings|gdp|recession|tariff|trump|biden|iran|china|russia|oil|crude|'
    r's&p|nasdaq|dow|treasury|yield|bond|dollar|trade|war|crisis|crash|rally|'
    r'bull|bear|hawkish|dovish|fomc|pce|unemployment|payroll|economy|economic|'
    r'tech|ai|nvidia|apple|google|tesla|amazon|microsoft|meta)\b',
    re.IGNORECASE
)


# ── RSS Fetching ───────────────────────────────────────────────────────────

def fetch_all_headlines(max_per_source: int = 50) -> list[dict]:
    """Fetch recent headlines from all configured feeds."""
    all_headlines = []
    seen_titles = set()

    for source_name, url in NEWS_FEEDS.items():
        try:
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
            if r.status_code != 200:
                continue
            feed = feedparser.parse(r.text)
            count = 0
            for entry in feed.entries[:max_per_source]:
                title = entry.get('title', '').strip()
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)

                # Parse publish date
                pub_str = entry.get('published', entry.get('updated', ''))
                try:
                    pub_date = pd.to_datetime(pub_str).tz_localize(None) if pub_str else pd.Timestamp.now()
                except Exception:
                    pub_date = pd.Timestamp.now()

                # Only market-relevant headlines
                if not MARKET_KEYWORDS.search(title):
                    continue

                all_headlines.append({
                    'title': title,
                    'source': source_name,
                    'url': entry.get('link', ''),
                    'published': pub_date,
                    'summary': entry.get('summary', '')[:200],
                })
                count += 1
            # print(f"  {source_name}: {count} market-relevant headlines")
        except Exception as e:
            continue

    return all_headlines


# ── X/Twitter Handle Fetcher (via Nitter RSS) ──────────────────────────────

# Curated list of authoritative financial X accounts
X_HANDLES = {
    'WSJmarkets':   ('WSJ Markets',          'news'),
    'MarketWatch':  ('MarketWatch',          'news'),
    'zerohedge':    ('ZeroHedge',            'analysis'),  # Bearish-leaning, high-signal
    'FirstSquawk':  ('FirstSquawk',          'wire'),       # Breaking wire-service news
    'Reuters':      ('Reuters',              'news'),
    'BloombergTV':  ('Bloomberg TV',         'news'),
    'DeItaone':     ('*Walter Bloomberg',    'wire'),       # Fast breaking headlines
    'unusual_whales':('Unusual Whales',      'analysis'),   # Options flow, market color
    'Jkylebass':    ('Kyle Bass',            'analysis'),
    'biancoresearch':('Jim Bianco',          'analysis'),
}

# Nitter instances to try in order (fallback if one goes down)
NITTER_INSTANCES = [
    'nitter.net',
    'nitter.privacydev.net',
    'xcancel.com',
]


def fetch_x_handle_tweets(handle: str, max_tweets: int = 20) -> list[dict]:
    """
    Fetch recent tweets from an X handle via Nitter RSS.
    Returns list of dicts compatible with the headline pipeline.
    """
    import time
    for instance in NITTER_INSTANCES:
        try:
            url = f'https://{instance}/{handle}/rss'
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            if r.status_code != 200 or len(r.text) < 500:
                continue

            feed = feedparser.parse(r.text)
            if not feed.entries:
                continue

            display_name, category = X_HANDLES.get(handle, (handle, 'other'))
            results = []
            for entry in feed.entries[:max_tweets]:
                title = entry.get('title', '').strip()
                if not title:
                    continue

                # Skip pure retweets if we want primary content only
                # (Comment out if retweets are desired)
                # if title.startswith('RT by @') or title.startswith('RT @'):
                #     continue

                # Strip Nitter's title prefix like "R to @user: "
                title_clean = re.sub(r'^(RT by @\w+:|R to @\w+:)\s*', '', title).strip()

                # Parse publish date
                try:
                    pub = pd.to_datetime(entry.get('published', '')).tz_localize(None)
                except Exception:
                    pub = pd.Timestamp.now()

                # Only keep market-relevant tweets
                if not MARKET_KEYWORDS.search(title_clean):
                    continue

                results.append({
                    'title': title_clean[:280],  # X char limit
                    'source': f'X/@{handle}',
                    'url': entry.get('link', ''),
                    'published': pub,
                    'summary': '',
                })
            return results
        except Exception:
            continue
        finally:
            time.sleep(1)  # Polite delay

    return []  # All instances failed


def fetch_all_x_handles(handles: list[str] = None, max_per_handle: int = 20) -> list[dict]:
    """Fetch tweets from multiple X handles and combine."""
    handles = handles or ['WSJmarkets', 'MarketWatch', 'zerohedge']
    all_tweets = []
    for h in handles:
        all_tweets.extend(fetch_x_handle_tweets(h, max_tweets=max_per_handle))
    return all_tweets


# ── Historical GDELT Fetcher ───────────────────────────────────────────────

def _gdelt_request(params, max_retries: int = 3):
    """GDELT API with rate-limit respect (1 req / 5-7s) and retry logic."""
    import time
    for attempt in range(max_retries):
        try:
            r = requests.get(
                'https://api.gdeltproject.org/api/v2/doc/doc',
                params=params,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; MarketSignalResearch/1.0)'},
                timeout=20,
            )
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                # Rate limited — wait longer and retry
                time.sleep(10 * (attempt + 1))
                continue
            return None
        except Exception:
            time.sleep(5)
    return None


def fetch_gdelt_headlines_for_date(date, max_records: int = 20) -> list[dict]:
    """
    Fetch headlines from major outlets for a single date via GDELT.
    Queries each top domain separately (GDELT has query length limits).
    """
    import time
    date_ts = pd.Timestamp(date).normalize()
    date_str = date_ts.strftime('%Y%m%d')

    source_map = {
        'reuters.com': 'Reuters', 'bloomberg.com': 'Bloomberg',
        'wsj.com': 'WSJ', 'cnbc.com': 'CNBC', 'ft.com': 'FT',
        'marketwatch.com': 'MarketWatch', 'bbc.com': 'BBC', 'bbc.co.uk': 'BBC',
        'nytimes.com': 'NYT', 'washingtonpost.com': 'WaPo',
        'forbes.com': 'Forbes', 'businessinsider.com': 'BusinessInsider',
    }

    headlines = []
    seen_titles = set()

    # Query per domain — simpler queries work better on GDELT
    for i, domain in enumerate(GDELT_TOP_DOMAINS):
        # Short query: keyword + single domain
        query = f'market domain:{domain}'

        data = _gdelt_request({
            'query': query,
            'mode': 'artlist',
            'format': 'json',
            'maxrecords': max_records,
            'startdatetime': f'{date_str}000000',
            'enddatetime': f'{date_str}235959',
            'sort': 'hybridrel',
        })

        if i < len(GDELT_TOP_DOMAINS) - 1:
            time.sleep(6)  # Rate limit between domain queries

        if not data:
            continue

        source = source_map.get(domain, domain.split('.')[0].title())

        for a in data.get('articles', []):
            title = a.get('title', '').strip()
            if not title or title in seen_titles:
                continue

            # Skip non-English content (often shows up as jp.reuters.com etc)
            # Filter by checking if title has ASCII letters
            if sum(1 for c in title if c.isascii() and c.isalpha()) < 10:
                continue

            seen_titles.add(title)

            try:
                pub = pd.to_datetime(a.get('seendate', '')).tz_localize(None)
            except Exception:
                pub = date_ts

            if not MARKET_KEYWORDS.search(title):
                continue

            headlines.append({
                'title': title,
                'source': source,
                'url': a.get('url', ''),
                'published': pub,
                'summary': '',
            })

    return headlines


def fetch_gdelt_range(start_date, end_date, progress=None) -> list[dict]:
    """
    Fetch all headlines from GDELT for a date range.
    Day-by-day with rate-limit respect. Returns combined list.
    """
    import time
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    all_headlines = []

    dates = pd.date_range(start, end, freq='D')
    for i, d in enumerate(dates):
        if progress:
            progress(i, len(dates), d)
        hl = fetch_gdelt_headlines_for_date(d)
        all_headlines.extend(hl)
        # Rate limit: 1 req per 5-7 seconds
        if i < len(dates) - 1:
            time.sleep(6)

    return all_headlines


# ── Fetch Log (cache tracking) ─────────────────────────────────────────────

def load_fetch_log() -> pd.DataFrame:
    """Load log of dates we've already fetched."""
    if not FETCH_LOG.exists():
        return pd.DataFrame(columns=['date', 'n_articles', 'source', 'fetched_at'])
    return pd.read_csv(FETCH_LOG, parse_dates=['date', 'fetched_at'])


def mark_dates_fetched(dates_counts: list[tuple], source: str = 'gdelt'):
    """Record (date, n_articles) pairs as fetched."""
    FETCH_LOG.parent.mkdir(exist_ok=True)
    existing = load_fetch_log()
    new_rows = pd.DataFrame([
        {'date': pd.Timestamp(d), 'n_articles': int(n),
         'source': source, 'fetched_at': pd.Timestamp.now()}
        for d, n in dates_counts
    ])
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined = combined.sort_values(['date', 'fetched_at']) \
                        .drop_duplicates(subset='date', keep='last')
    combined.to_csv(FETCH_LOG, index=False)


def get_missing_dates(start_date, end_date) -> list:
    """Return list of dates in the range that haven't been fetched yet."""
    log = load_fetch_log()
    fetched = set()
    if not log.empty:
        fetched = set(log['date'].dt.normalize().dt.date.tolist())

    all_dates = [d.date() for d in pd.date_range(start_date, end_date, freq='D')]
    return [d for d in all_dates if d not in fetched]


def backfill_news_for_range(start_date, end_date, progress=None) -> dict:
    """
    Backfill news + classification for a date range.
    Checks cache, only fetches missing dates.
    Returns summary dict.
    """
    missing = get_missing_dates(start_date, end_date)

    if not missing:
        return {
            'status': 'cached',
            'missing_count': 0,
            'fetched_count': 0,
            'n_headlines': 0,
            'message': f'All {(pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1} days already cached.'
        }

    # Group consecutive missing dates into ranges for efficient fetching
    # For simplicity, just fetch each missing date individually
    all_headlines = []
    dates_counts = []

    for i, d in enumerate(missing):
        if progress:
            progress(i, len(missing), d)
        headlines = fetch_gdelt_headlines_for_date(d)
        all_headlines.extend(headlines)
        dates_counts.append((d, len(headlines)))
        # Respect rate limit
        if i < len(missing) - 1:
            import time
            time.sleep(6)

    # Classify all new headlines via Claude (uses cache for seen headlines)
    if all_headlines:
        classified = classify_with_cache(all_headlines)
        daily = aggregate_daily_sentiment(classified)
        log_daily_sentiment(daily)
        log_headlines(classified)
    else:
        classified = []

    # Mark dates as fetched (even if 0 articles — don't re-try)
    mark_dates_fetched(dates_counts, source='gdelt')

    return {
        'status': 'fetched',
        'missing_count': len(missing),
        'fetched_count': len(missing),
        'n_headlines': len(classified),
        'message': f'Fetched {len(missing)} days, {len(classified)} market-relevant headlines classified.'
    }


# ── Sentiment Cache ────────────────────────────────────────────────────────

def _headline_hash(title: str) -> str:
    """Stable key for cache lookup."""
    return hashlib.md5(title.strip().lower().encode()).hexdigest()


def load_sentiment_cache() -> dict:
    if not SENTIMENT_CACHE.exists():
        return {}
    try:
        return json.loads(SENTIMENT_CACHE.read_text())
    except Exception:
        return {}


def save_sentiment_cache(cache: dict):
    SENTIMENT_CACHE.parent.mkdir(exist_ok=True)
    SENTIMENT_CACHE.write_text(json.dumps(cache, indent=2))


# ── Claude Classification ──────────────────────────────────────────────────

def classify_headlines_batch(headlines: list[str], api_key: str = None) -> list[dict]:
    """
    Classify a batch of headlines for market sentiment.
    Returns list of {sentiment: float [-1, 1], label: str, reason: str}.
    Batches multiple headlines into one API call for efficiency.
    """
    if not headlines:
        return []

    api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set. Check .env file.")

    client = anthropic.Anthropic(api_key=api_key)

    # Build numbered list for the prompt
    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))

    prompt = f"""You are a financial news sentiment analyst. For each headline below, classify its likely IMPACT ON US STOCK MARKET (S&P 500) over the next 1-5 trading days.

Output format: return ONLY a JSON array. CRITICAL: do not use apostrophes, double-quotes, or any special characters in the "reason" field. Use plain ASCII words only.

Example output:
[{{"n": 1, "sentiment": -0.8, "label": "bearish", "reason": "war raises oil prices"}}, {{"n": 2, "sentiment": 0.6, "label": "bullish", "reason": "rate cut expected"}}]

Rules:
- sentiment: float from -1.0 (very bearish for stocks) to +1.0 (very bullish for stocks), 0 = neutral/mixed
- label: one of: very_bearish, bearish, neutral, bullish, very_bullish
- reason: 3-6 words, ASCII only, NO apostrophes or quotes

Market knowledge:
- Fed cuts rates = BULLISH (lower rates = higher stocks)
- Inflation drops = BULLISH (supports rate cuts)
- Hawkish Fed / rate hike = BEARISH
- Weak jobs data = BULLISH (in current easing cycle)
- Tariff announced = BEARISH (trade disruption)
- Tariff deal/relief = BULLISH
- Iran/war/oil shock = BEARISH
- Profit warning = BEARISH
- Pure company-specific news = near neutral unless mega-cap

Headlines:
{numbered}

Return ONLY the JSON array, nothing else."""

    try:
        msg = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        # Strip code fences if present
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.MULTILINE).strip()

        # Try strict JSON parse first
        try:
            results = json.loads(text)
        except json.JSONDecodeError:
            # Robust fallback: extract each object via regex
            # Matches: {"n": 1, "sentiment": -0.8, "label": "bearish", "reason": "..."}
            results = []
            # Try to find each JSON object, handling unescaped apostrophes
            pattern = re.compile(
                r'\{\s*"n"\s*:\s*(\d+)\s*,\s*'
                r'"sentiment"\s*:\s*(-?\d+\.?\d*)\s*,\s*'
                r'"label"\s*:\s*"([^"]+)"\s*,\s*'
                r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
                re.DOTALL
            )
            for m in pattern.finditer(text):
                results.append({
                    'n': int(m.group(1)),
                    'sentiment': float(m.group(2)),
                    'label': m.group(3),
                    'reason': m.group(4),
                })
            if not results:
                # Ultimate fallback: regex extract per field
                n_matches = re.findall(r'"n"\s*:\s*(\d+)', text)
                s_matches = re.findall(r'"sentiment"\s*:\s*(-?\d+\.?\d*)', text)
                l_matches = re.findall(r'"label"\s*:\s*"([^"]+)"', text)
                for i in range(min(len(n_matches), len(s_matches), len(l_matches))):
                    results.append({
                        'n': int(n_matches[i]),
                        'sentiment': float(s_matches[i]),
                        'label': l_matches[i],
                        'reason': '',
                    })

        # Sort by 'n' to match input order
        results.sort(key=lambda x: x.get('n', 0))
        return results
    except Exception as e:
        # Return neutral on error so scoring continues
        return [{'n': i+1, 'sentiment': 0.0, 'label': 'neutral',
                 'reason': f'classify error: {str(e)[:50]}'}
                for i, _ in enumerate(headlines)]


def classify_with_cache(headlines: list[dict]) -> list[dict]:
    """
    Classify headlines, using cache for already-seen ones.
    Returns headlines with added 'sentiment', 'label', 'reason' fields.
    """
    cache = load_sentiment_cache()

    to_classify = []
    to_classify_idx = []
    for i, h in enumerate(headlines):
        key = _headline_hash(h['title'])
        if key in cache:
            h.update(cache[key])
        else:
            to_classify.append(h['title'])
            to_classify_idx.append(i)

    # Classify uncached headlines (batch of up to 30 at a time for reliability)
    BATCH = 30
    for i in range(0, len(to_classify), BATCH):
        batch_titles = to_classify[i:i+BATCH]
        batch_idx = to_classify_idx[i:i+BATCH]
        results = classify_headlines_batch(batch_titles)

        # Match results back to headlines
        for j, (title, idx) in enumerate(zip(batch_titles, batch_idx)):
            if j < len(results):
                r = results[j]
                classification = {
                    'sentiment': float(r.get('sentiment', 0)),
                    'label': r.get('label', 'neutral'),
                    'reason': r.get('reason', ''),
                }
            else:
                classification = {'sentiment': 0.0, 'label': 'neutral', 'reason': 'no result'}

            headlines[idx].update(classification)
            # Cache it
            cache[_headline_hash(title)] = classification

    save_sentiment_cache(cache)
    return headlines


# ── Daily Aggregation ──────────────────────────────────────────────────────

def aggregate_daily_sentiment(headlines: list[dict]) -> pd.DataFrame:
    """Aggregate headlines into daily sentiment scores."""
    if not headlines:
        return pd.DataFrame(columns=['date', 'avg_sentiment', 'n_articles',
                                      'n_bullish', 'n_bearish', 'n_neutral'])

    df = pd.DataFrame(headlines)
    df['date'] = pd.to_datetime(df['published']).dt.normalize()

    def _aggregate(group):
        sentiments = group['sentiment'].astype(float)
        labels = group['label']
        return pd.Series({
            'avg_sentiment': round(sentiments.mean(), 3),
            'n_articles': len(group),
            'n_bullish': int((labels.isin(['bullish', 'very_bullish'])).sum()),
            'n_bearish': int((labels.isin(['bearish', 'very_bearish'])).sum()),
            'n_neutral': int((labels == 'neutral').sum()),
            'max_bearish': round(sentiments.min(), 3),
            'max_bullish': round(sentiments.max(), 3),
        })

    daily = df.groupby('date').apply(_aggregate, include_groups=False).reset_index()
    return daily


def log_daily_sentiment(daily_df: pd.DataFrame):
    """Append/update today's sentiment to CSV."""
    if daily_df.empty:
        return

    SENTIMENT_CSV.parent.mkdir(exist_ok=True)
    if SENTIMENT_CSV.exists():
        existing = pd.read_csv(SENTIMENT_CSV, parse_dates=['date'])
        combined = pd.concat([existing, daily_df], ignore_index=True)
        combined = combined.sort_values('date').drop_duplicates(subset='date', keep='last')
    else:
        combined = daily_df.sort_values('date')

    combined.to_csv(SENTIMENT_CSV, index=False)


def log_headlines(headlines: list[dict]):
    """Append classified headlines to log CSV (for audit/debug)."""
    if not headlines:
        return

    df = pd.DataFrame(headlines)[['published', 'source', 'title', 'sentiment', 'label', 'reason']]
    df.columns = ['timestamp', 'source', 'title', 'sentiment', 'label', 'reason']

    HEADLINES_LOG.parent.mkdir(exist_ok=True)
    if HEADLINES_LOG.exists():
        existing = pd.read_csv(HEADLINES_LOG, parse_dates=['timestamp'])
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset='title', keep='last')
    else:
        combined = df

    combined.to_csv(HEADLINES_LOG, index=False)


def load_sentiment_history() -> pd.DataFrame:
    """Load accumulated daily sentiment history."""
    if not SENTIMENT_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(SENTIMENT_CSV, parse_dates=['date']).sort_values('date').reset_index(drop=True)


def load_headlines_log(n_days: int = 7) -> pd.DataFrame:
    """Load recent classified headlines for display."""
    if not HEADLINES_LOG.exists():
        return pd.DataFrame()
    df = pd.read_csv(HEADLINES_LOG, parse_dates=['timestamp'])
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=n_days)
    return df[df['timestamp'] >= cutoff].sort_values('timestamp', ascending=False).reset_index(drop=True)


# ── Main Pipeline ──────────────────────────────────────────────────────────

def refresh_news_sentiment(source_mode: str = 'rss+x',
                            x_handles: list = None) -> dict:
    """
    Full pipeline: fetch → classify → aggregate → log.
    source_mode:
      'rss'    — RSS feeds only (CNBC, WSJ, FT, BBC, etc.)
      'x'      — X/Twitter handles only
      'rss+x'  — both combined (default)
    x_handles: list of X handles (default: WSJmarkets, MarketWatch, zerohedge)
    """
    headlines = []

    if source_mode in ('rss', 'rss+x'):
        headlines.extend(fetch_all_headlines())

    if source_mode in ('x', 'rss+x'):
        handles = x_handles or ['WSJmarkets', 'MarketWatch', 'zerohedge']
        headlines.extend(fetch_all_x_handles(handles))

    if not headlines:
        return {'error': 'No headlines fetched', 'n_headlines': 0}

    # Dedupe by title (X and RSS may overlap)
    seen = set()
    unique = []
    for h in headlines:
        if h['title'] not in seen:
            seen.add(h['title'])
            unique.append(h)
    headlines = unique

    # Classify (uses cache)
    classified = classify_with_cache(headlines)

    # Aggregate to daily
    daily = aggregate_daily_sentiment(classified)

    # Log
    log_daily_sentiment(daily)
    log_headlines(classified)

    # Build summary
    today = pd.Timestamp.now().normalize()
    today_row = daily[daily['date'] == today]
    sources = set(h['source'] for h in classified) if classified else set()
    summary = {
        'n_headlines': len(classified),
        'n_sources': len(sources),
        'source_breakdown': {s: sum(1 for h in classified if h['source'] == s) for s in sources},
        'daily': daily.to_dict('records'),
        'today_sentiment': float(today_row['avg_sentiment'].iloc[0]) if not today_row.empty else None,
        'today_articles': int(today_row['n_articles'].iloc[0]) if not today_row.empty else 0,
        'timestamp': datetime.now().isoformat(),
        'source_mode': source_mode,
    }
    return summary


if __name__ == '__main__':
    print("Fetching news headlines...")
    headlines = fetch_all_headlines()
    print(f"  Got {len(headlines)} market-relevant headlines from {len(set(h['source'] for h in headlines))} sources")

    if headlines:
        print(f"\nClassifying via Claude ({CLAUDE_MODEL})...")
        classified = classify_with_cache(headlines)

        print(f"\nSample classifications:")
        for h in classified[:10]:
            print(f"  [{h['label']:14s}] {h['sentiment']:+.2f} — {h['title'][:80]}")
            print(f"     ({h['source']}) Reason: {h.get('reason', '')}")

        print(f"\nAggregating...")
        daily = aggregate_daily_sentiment(classified)
        print(daily.to_string(index=False))

        log_daily_sentiment(daily)
        log_headlines(classified)
        print(f"\nLogged to {SENTIMENT_CSV.name}")
