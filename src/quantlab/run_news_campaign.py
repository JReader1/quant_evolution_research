"""Resume news collection and align it to the saved historical trading calendar.

Default: 45-day 2022 exploratory pilot, starting with the 14 days already cached.
Use --days 90 for a larger pilot later. Research only; no champion promotion.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys


def locate_root(source: Path) -> Path:
    here = source.resolve().parent
    for candidate in (here, *here.parents):
        if ((candidate / 'run_evolution.py').is_file()
                and (candidate / 'src' / 'quantlab' / 'news' / 'gdelt_daily.py').is_file()):
            return candidate
    raise FileNotFoundError('Cannot find Quant Research project root and news module')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2022-02-22', help='YYYY-MM-DD')
    parser.add_argument('--days', type=int, default=45, help='9–120 calendar days (default 45)')
    parser.add_argument('--offline', action='store_true', help='Only use cached historical archives')
    args = parser.parse_args()
    try:
        root = locate_root(Path(__file__))
        sys.path.insert(0, str(root / 'src'))
        from quantlab.news.campaign import collect_campaign
        folder, result = collect_campaign(root, date.fromisoformat(args.start),
                                          args.days, offline=args.offline)
        print('\nHISTORICAL NEWS CAMPAIGN RESULTS')
        print('Root:', root)
        print('Archive dates collected:', result['days_collected'], '/', result['days_requested'])
        print('Complete market-date windows:', result['complete_trading_windows'])
        print('Incomplete market-date windows:', result['incomplete_trading_windows'])
        print('Daily summary:', folder / 'daily_event_features.csv')
        print('Market-date alignment:', folder / 'market_date_news_alignment.csv')
        print('Missing archive audit:', folder / 'daily_coverage.csv')
        print('Research only: No news-based champion promotion or live trades.')
        if not result['finished_successfully']:
            print('INCOMPLETE: Re-run the same command; cached successful days will be reused.')
            print('First failure:', result['first_error'])
            return 1
        print('PASS: Historical news coverage expanded and aligned.')
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f'NEWS CAMPAIGN ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
