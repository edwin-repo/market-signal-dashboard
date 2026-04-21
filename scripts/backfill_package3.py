"""
Package 3 backfill — all 5 critical periods × 3 X handles.
Estimated cost: ~$17 (tweets + classification).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from news_twitterapi import backfill_twitter_handles
from datetime import datetime

# 5 critical periods
PERIODS = [
    ('A', '2025-01-15', '2025-02-05', 'Post-inauguration jitters'),
    ('B', '2025-02-15', '2025-05-01', 'Tariff crisis crash + recovery'),
    ('C', '2025-07-01', '2025-08-15', 'Summer pullback / yen carry'),
    ('D', '2025-11-01', '2025-12-20', 'Iran war buildup'),
    ('E', '2026-02-15', '2026-04-20', 'Iran war crisis + recovery'),
]

HANDLES = ['WSJmarkets', 'MarketWatch', 'zerohedge']


def main():
    print(f"=== PACKAGE 3 BACKFILL ===")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Periods: {len(PERIODS)}")
    print(f"Handles: {', '.join(HANDLES)}")
    print()

    total_raw = 0
    total_classified = 0
    total_cost_tweets = 0.0

    for label, start, end, desc in PERIODS:
        print(f"\n{'='*60}")
        print(f"Period {label}: {start} → {end} — {desc}")
        print(f"{'='*60}")

        def _prog(handle, chunk_or_page, count, status=None, error=None):
            if status:
                print(f"  [{handle}] {status}", flush=True)
            elif error:
                print(f"  [{handle}] ERROR: {error}", flush=True)
            else:
                print(f"  [{handle}] chunk {chunk_or_page}, total: {count}", flush=True)

        try:
            result = backfill_twitter_handles(
                start_date=start,
                end_date=end,
                handles=HANDLES,
                progress=_prog,
            )
            print(f"\n  Status: {result.get('status')}")
            print(f"  Raw tweets: {result.get('n_raw_tweets')}")
            print(f"  Market-relevant: {result.get('n_market_relevant')}")
            print(f"  Est cost: ${result.get('estimated_cost', 0):.2f}")
            for h, stats in result.get('per_handle', {}).items():
                print(f"    @{h}: raw={stats['raw']}, filtered={stats['market_relevant']}")

            total_raw += result.get('n_raw_tweets', 0)
            total_classified += result.get('n_market_relevant', 0)
            total_cost_tweets += result.get('estimated_cost', 0)
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            # Continue to next period rather than abort
            continue

    print(f"\n{'='*60}")
    print(f"=== FINAL SUMMARY ===")
    print(f"{'='*60}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total raw tweets: {total_raw:,}")
    print(f"Market-relevant classified: {total_classified:,}")
    print(f"Total TwitterAPI.io cost: ${total_cost_tweets:.2f}")


if __name__ == '__main__':
    main()
