"""
TwitterAPI.io integration for historical X/Twitter backfill.

Uses the advanced_search endpoint with from:user + since:/until: filters
to fetch historical tweets for specific handles with full date range support.

Pricing: $0.15 per 1000 tweets (~$7 for 48K tweet backfill)
"""

from __future__ import annotations
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env', override=True)
except ImportError:
    pass


API_KEY = os.environ.get('TWITTERAPI_IO_KEY', '')
BASE_URL = 'https://api.twitterapi.io/twitter/tweet/advanced_search'
MAX_PAGES_PER_QUERY = 50  # Safety cap (~1000 tweets per date range query)

# Default handles to scrape (avoid duplicates with GDELT where possible)
DEFAULT_HANDLES = ['WSJmarkets', 'MarketWatch', 'zerohedge']

# Import from news module to reuse classification pipeline
from news import (
    MARKET_KEYWORDS, classify_with_cache,
    aggregate_daily_sentiment, log_daily_sentiment, log_headlines,
)


def fetch_tweets_single_query(handle: str, start_date, end_date,
                                max_pages: int = MAX_PAGES_PER_QUERY,
                                progress=None) -> list[dict]:
    """
    Fetch tweets from @handle within [start_date, end_date] via single query.
    Limited to max_pages × 20 tweets (~1000). Use fetch_tweets_for_range for
    larger ranges with chunking.
    """
    if not API_KEY:
        raise RuntimeError("TWITTERAPI_IO_KEY not set in .env")

    start = pd.Timestamp(start_date).strftime('%Y-%m-%d')
    end = pd.Timestamp(end_date).strftime('%Y-%m-%d')
    query = f'from:{handle} since:{start} until:{end}'

    all_tweets = []
    cursor = None
    page = 0

    while page < max_pages:
        params = {'query': query, 'queryType': 'Latest'}
        if cursor:
            params['cursor'] = cursor

        try:
            r = requests.get(BASE_URL,
                              headers={'X-API-Key': API_KEY},
                              params=params,
                              timeout=30)
            if r.status_code == 429:
                time.sleep(5)
                continue
            if r.status_code == 402:
                raise RuntimeError(
                    "TwitterAPI.io: Out of credits. Recharge at https://twitterapi.io/dashboard"
                )
            if r.status_code == 401:
                raise RuntimeError("TwitterAPI.io: Unauthorized — check API key in .env")
            if r.status_code != 200:
                if progress:
                    progress(handle, page, len(all_tweets),
                             error=f"HTTP {r.status_code}: {r.text[:100]}")
                break

            data = r.json()
            tweets = data.get('tweets', [])
            if not tweets:
                break

            all_tweets.extend(tweets)
            page += 1

            if progress:
                progress(handle, page, len(all_tweets))

            if not data.get('has_next_page') or not data.get('next_cursor'):
                break
            cursor = data['next_cursor']
            time.sleep(0.5)

        except RuntimeError:
            raise  # Propagate credit/auth errors
        except Exception as e:
            if progress:
                progress(handle, page, len(all_tweets), error=str(e))
            break

    return all_tweets


def fetch_tweets_for_range(handle: str, start_date, end_date,
                            chunk_days: int = 3,
                            max_pages_per_chunk: int = MAX_PAGES_PER_QUERY,
                            progress=None) -> list[dict]:
    """
    Fetch tweets for a date range, chunking by N days to avoid 1000-tweet cap.
    Zerohedge posts ~333/day → 3-day chunks = ~1000 tweets (fits in one query).
    WSJ/MW post ~50-80/day → same 3-day chunks work comfortably.
    """
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()

    all_tweets = []
    chunk_start = start_ts
    chunk_num = 0
    total_chunks = ((end_ts - start_ts).days // chunk_days) + 1

    while chunk_start < end_ts:
        chunk_end = min(chunk_start + pd.Timedelta(days=chunk_days), end_ts)
        chunk_num += 1

        if progress:
            progress(handle, chunk_num, len(all_tweets),
                     status=f'chunk {chunk_num}/{total_chunks}: {chunk_start.date()} → {chunk_end.date()}')

        try:
            tweets = fetch_tweets_single_query(
                handle, chunk_start, chunk_end,
                max_pages=max_pages_per_chunk,
            )
            all_tweets.extend(tweets)
        except RuntimeError:
            raise  # Stop on credit/auth errors

        chunk_start = chunk_end
        # Small delay between chunks to be polite
        time.sleep(0.3)

    return all_tweets


def tweet_to_headline(tweet: dict, handle: str) -> dict | None:
    """Convert a TwitterAPI.io tweet to the headline format used by the pipeline."""
    text = tweet.get('text', '').strip()
    if not text:
        return None

    # Skip pure media/retweets without text content
    if len(text) < 20:
        return None

    # Strip t.co URLs for cleaner classification
    text_clean = re.sub(r'https?://\S+', '', text).strip()
    if len(text_clean) < 15:
        return None

    # Only keep market-relevant tweets
    if not MARKET_KEYWORDS.search(text_clean):
        return None

    try:
        # Format: "Tue Apr 21 12:25:33 +0000 2026"
        pub = pd.to_datetime(tweet.get('createdAt', '')).tz_localize(None)
    except Exception:
        return None

    return {
        'title': text_clean[:280],
        'source': f'X/@{handle}',
        'url': tweet.get('url', ''),
        'published': pub,
        'summary': '',
        'engagement': {
            'retweets': tweet.get('retweetCount', 0),
            'likes': tweet.get('likeCount', 0),
            'replies': tweet.get('replyCount', 0),
            'views': tweet.get('viewCount', 0),
        },
    }


def backfill_twitter_handles(start_date, end_date,
                              handles: list = None,
                              progress=None) -> dict:
    """
    Full backfill: fetch tweets → classify via Claude → aggregate → log.
    Reuses existing news.py classification pipeline.
    """
    handles = handles or DEFAULT_HANDLES
    all_tweets_raw = []
    all_headlines = []

    per_handle = {}

    for handle in handles:
        if progress:
            progress(handle, 0, 0, status='starting')

        raw_tweets = fetch_tweets_for_range(handle, start_date, end_date, progress=progress)
        per_handle[handle] = {'raw': len(raw_tweets), 'market_relevant': 0}

        for t in raw_tweets:
            hl = tweet_to_headline(t, handle)
            if hl is not None:
                all_headlines.append(hl)
                per_handle[handle]['market_relevant'] += 1

        all_tweets_raw.extend(raw_tweets)

    if not all_headlines:
        return {
            'status': 'no_data',
            'n_raw_tweets': len(all_tweets_raw),
            'n_market_relevant': 0,
            'per_handle': per_handle,
        }

    # Deduplicate by title (handles can retweet each other)
    seen = set()
    unique = []
    for h in all_headlines:
        if h['title'] not in seen:
            seen.add(h['title'])
            unique.append(h)
    all_headlines = unique

    # Classify via Claude (uses cache — won't re-classify seen headlines)
    if progress:
        progress('CLASSIFY', 0, len(all_headlines), status='classifying')
    classified = classify_with_cache(all_headlines)

    # Aggregate daily
    daily = aggregate_daily_sentiment(classified)

    # Log
    log_daily_sentiment(daily)
    log_headlines(classified)

    return {
        'status': 'success',
        'n_raw_tweets': len(all_tweets_raw),
        'n_market_relevant': len(classified),
        'per_handle': per_handle,
        'daily_summary': daily.to_dict('records'),
        'date_range': f"{start_date} to {end_date}",
        'estimated_cost': round(len(all_tweets_raw) / 1000 * 0.15, 2),
    }


if __name__ == '__main__':
    import sys
    # Small test: 3 days of @zerohedge
    print("Testing TwitterAPI.io backfill for @zerohedge (Nov 1-3, 2024)...")

    def _prog(handle, page, count, status=None, error=None):
        if status:
            print(f"[{handle}] {status}")
        elif error:
            print(f"[{handle}] ERROR: {error}")
        else:
            print(f"[{handle}] page {page}, total tweets: {count}")

    result = backfill_twitter_handles(
        start_date='2024-11-01',
        end_date='2024-11-04',
        handles=['zerohedge'],
        progress=_prog,
    )

    print("\n=== RESULT ===")
    for k, v in result.items():
        if k != 'daily_summary':
            print(f"  {k}: {v}")
    if 'daily_summary' in result:
        print(f"\n  Daily summary:")
        for d in result['daily_summary']:
            print(f"    {d}")
