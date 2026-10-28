"""Resumable, audit-friendly GDELT 1.0 historical news collection.

Exploratory historical news proxies only. Historical date-based lags do NOT
certify true point-in-time data availability. No orders or champion promotion.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import json
import re

import pandas as pd

from quantlab.news.gdelt_daily import (
    fetch_archive, summarize_archive, align_to_market_dates,
    NEWS_LAG_CALENDAR_DAYS,
)

SUMMARY_SCHEMA = 1
REQUIRED_FIELDS = {
    'archive_date', 'events', 'conflict_events',
    'material_conflict_events', 'mean_tone',
    'conflict_share', 'material_conflict_share',
}


def _atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    try:
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding='utf-8')
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def daily_summary(root: Path, day: date, *, offline=False) -> tuple[dict, str]:
    """Read/download archive and reuse cached aggregate if source hash matches."""
    root = Path(root)
    raw, source = fetch_archive(root, day, offline=offline)
    summary_path = (
        root / 'data' / 'news' / 'gdelt_events_1_0' / 'daily_summaries'
        / f'{day:%Y%m%d}.json'
    )
    if summary_path.is_file():
        try:
            cached = json.loads(summary_path.read_text(encoding='utf-8'))
            if (cached.get('cache_schema') == SUMMARY_SCHEMA
                    and cached.get('source_sha256') == source['sha256']
                    and cached.get('archive_date') == day.isoformat()
                    and REQUIRED_FIELDS.issubset(cached)):
                return cached, 'CACHED_SUMMARY'
        except (OSError, ValueError, TypeError):
            pass  # Rebuild from verified raw archive, never trust bad cache.
    record = summarize_archive(raw, day)
    record.update({
        'cache_schema': SUMMARY_SCHEMA,
        'source_sha256': source['sha256'],
        'source_url': source['source_url'],
        'retrieved_utc': source.get('retrieved_utc'),
        'aggregated_utc': datetime.now(timezone.utc).isoformat(),
    })
    _atomic_json(summary_path, record)
    return record, 'SUMMARIZED_ARCHIVE'


def price_calendar(root: Path) -> pd.DatetimeIndex:
    """Historical exchange days from locally saved AAPL prices, not weekdays."""
    path = Path(root) / 'data' / 'raw_2018_2025' / 'AAPL.csv'
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing {path}; unable to verify actual trading sessions.'
        )
    prices = pd.read_csv(path, usecols=['Date'])
    if prices.empty:
        raise ValueError('AAPL price calendar is empty')
    dates = pd.DatetimeIndex(pd.to_datetime(prices['Date'], errors='raise'))
    if dates.has_duplicates:
        raise ValueError('Duplicate days in AAPL price calendar')
    return dates.sort_values().normalize()


def collect_campaign(root: Path, start: date, days: int, *, offline=False,
                     summary_fn=None, calendar_fn=None) -> tuple[Path, dict]:
    """Collect at most 120 calendar days per run. Supports safe retry/resume.

    On a failed download, writes a partial audit and stops downloading to avoid
    retry storms. Re-running reuses all successful raw files and summaries.
    """
    root = Path(root)
    if not 9 <= days <= 120:
        raise ValueError('Choose 9–120 consecutive calendar days per campaign')
    if start < date(2018, 1, 1):
        raise ValueError('Use 2018 or later for this equity project')
    if start + timedelta(days=days) > datetime.now(timezone.utc).date():
        raise ValueError('Campaign requests a future or incomplete calendar day')
    summary_fn = daily_summary if summary_fn is None else summary_fn
    calendar_fn = price_calendar if calendar_fn is None else calendar_fn

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = root / 'reports' / 'news_campaigns' / run_id
    folder.mkdir(parents=True, exist_ok=False)
    success_rows: list[dict] = []
    status_rows: list[dict] = []
    first_error = None

    print(f'GDELT NEWS CAMPAIGN: {start} for {days} days')
    for i in range(days):
        day = start + timedelta(days=i)
        if first_error is not None:
            status_rows.append({'archive_date': day.isoformat(),
                                'status': 'NOT_ATTEMPTED', 'reason': 'Stopped after first failed archive'})
            continue
        try:
            record, source = summary_fn(root, day, offline=offline)
            if record.get('archive_date') != day.isoformat():
                raise ValueError('Summary reports the wrong archive date')
            if not REQUIRED_FIELDS.issubset(record):
                raise ValueError('Invalid news summary schema')
            if int(record['events']) <= 0:
                raise ValueError('News summary has no events')
            success_rows.append(record)
            status_rows.append({'archive_date': day.isoformat(),
                                'status': 'OK', 'reason': '', 'source': source,
                                'events': int(record['events'])})
            print(f'  {day}: {int(record["events"]):,} events ({source})')
        except (Exception) as exc:
            first_error = f'{day}: {type(exc).__name__}: {exc}'
            status_rows.append({'archive_date': day.isoformat(),
                                'status': 'ERROR', 'reason': first_error})
            print(f'  STOPPED at {day}: {type(exc).__name__}: {exc}')

    daily = pd.DataFrame(success_rows)
    status = pd.DataFrame(status_rows)
    status.to_csv(folder / 'daily_coverage.csv', index=False)
    if daily.empty:
        daily = pd.DataFrame(columns=['archive_date', 'events', 'conflict_events',
                                      'material_conflict_events', 'mean_tone'])
    daily.to_csv(folder / 'daily_event_features.csv', index=False)

    trading_dates = calendar_fn(root)
    start_signal = pd.Timestamp(start + timedelta(days=8))
    end_signal = pd.Timestamp(start + timedelta(days=days + 3))
    trading_dates = trading_dates[(trading_dates >= start_signal) & (trading_dates <= end_signal)]
    if len(trading_dates) == 0:
        raise ValueError('No market trading dates in local price cache for requested period')
    aligned = align_to_market_dates(
        daily, trading_dates, window=7, lag_days=NEWS_LAG_CALENDAR_DAYS
    )
    aligned.to_csv(folder / 'market_date_news_alignment.csv', index=False)
    complete = int((aligned['news_coverage'] == 'COMPLETE').sum())
    incomplete = int((aligned['news_coverage'] == 'INCOMPLETE').sum())
    ok = int((status['status'] == 'OK').sum())
    result = {
        'run_id': run_id, 'start': start.isoformat(), 'days_requested': days,
        'days_collected': ok, 'days_failed_or_not_attempted': days-ok,
        'complete_trading_windows': complete, 'incomplete_trading_windows': incomplete,
        'first_error': first_error, 'finished_successfully': first_error is None,
        'source': 'GDELT 1.0 dated daily event archives',
        'time_delay_assumption_days': NEWS_LAG_CALENDAR_DAYS,
        'news_features_are_global': True,
        'point_in_time_verified': False,
        'champion_promotion_allowed': False,
        'orders_placed': False,
        'note': ('Global coverage proxy, not individual stock news. '
                 'Historical archive file dates are not proof of exact intraday news availability.'),
    }
    _atomic_json(folder / 'manifest.json', result)
    return folder, result
