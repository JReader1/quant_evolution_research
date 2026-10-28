"""One-command, resumable 2022 GDELT backfill + news-risk overlay comparison.

Research only. Preserves run_evolution.py, all champions and price files.
No model promotion, source-code rewriting, paper fills, or real trades.
"""
from __future__ import annotations

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import pandas as pd


def locate_root(path: Path) -> Path:
    for folder in (path.resolve().parent, *path.resolve().parents):
        if ((folder / 'run_evolution.py').exists() and
                (folder / 'src/quantlab/evolution/arena.py').exists() and
                (folder / 'src/quantlab/news/campaign.py').exists()):
            return folder
    raise FileNotFoundError('Cannot locate your Quant Research project root')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2022-02-22', help='Archive start date, YYYY-MM-DD')
    parser.add_argument('--days', type=int, default=120, help='9-120 consecutive calendar days')
    parser.add_argument('--offline', action='store_true',
                        help='Use cached archives only; refuse missing dates')
    args = parser.parse_args()
    root = locate_root(Path(__file__))
    sys.path.insert(0, str(root / 'src'))

    from quantlab.evolution.arena import BASELINE, Genome, HistoricalArena
    from quantlab.news.campaign import collect_campaign
    from quantlab.news.overlay import (
        OverlayRule, prepare_news_schedule, determine_experiment_bounds,
        evaluate_news_overlay,
    )

    start = date.fromisoformat(args.start)
    if args.days < 9 or args.days > 120:
        raise ValueError('Use 9-120 calendar days. Backfills resume from cache.')

    print('\nNEWS + PRICE EXPLORATORY EXPERIMENT (NO PROMOTION)')
    print(f'Backfill range: {start} through {start + timedelta(days=args.days - 1)}')
    print('Completed campaign reports are reused; otherwise archived days may be downloaded.')

    # Reuse archived campaigns when the source manifest still matches.
    campaign_dir = None
    campaign = None
    folder_root = root / 'reports' / 'news_campaigns'
    for manifest_file in sorted(folder_root.glob('*/manifest.json'), reverse=True):
        try:
            previous = json.loads(manifest_file.read_text(encoding='utf-8'))
            if (previous.get('start') == start.isoformat()
                    and previous.get('days_requested') == args.days
                    and previous.get('finished_successfully') is True
                    and (manifest_file.parent / 'market_date_news_alignment.csv').is_file()):
                campaign_dir, campaign = manifest_file.parent, previous
                print('Using existing completed news campaign:', campaign_dir)
                break
        except (OSError, ValueError, TypeError):
            continue
    if campaign_dir is None:
        campaign_dir, campaign = collect_campaign(root, start, args.days,
                                                 offline=args.offline)
    if not campaign['finished_successfully']:
        print('\nPARTIAL DOWNLOAD: results safely cached. Rerun the same command.')
        print('First error:', campaign['first_error'])
        print('Coverage audit:', campaign_dir / 'daily_coverage.csv')
        return 1

    aligned_path = campaign_dir / 'market_date_news_alignment.csv'
    aligned = pd.read_csv(aligned_path)
    rule = OverlayRule(warmup_sessions=20, threshold_z=1.0,
                       stressed_risk_multiplier=0.75)
    schedule = prepare_news_schedule(aligned, rule)
    arena = HistoricalArena(root, cost_bps=5.0)
    start_date, end_date, n_periods, signal_dates = determine_experiment_bounds(
        arena, schedule, (start + timedelta(days=args.days)).isoformat()
    )
    risk_map = dict(zip(
        schedule.loc[schedule['rule_ready'], 'signal_date'].dt.strftime('%Y-%m-%d'),
        schedule.loc[schedule['rule_ready'], 'risk_multiplier'],
    ))
    recent_championships = sorted(
        p for p in (root / 'reports/evolution_championships').glob('*/summary.json')
        if p.is_file()
    )
    if not recent_championships:
        raise FileNotFoundError('Need the earlier run_evolution.py championship summary')
    championship_path = recent_championships[-1]
    old_summary = json.loads(championship_path.read_text(encoding='utf-8'))
    old_champion = Genome(**old_summary['champion']['genome'])
    old_champion.validate()

    now = datetime.now(timezone.utc)
    output = root / 'reports' / 'news_experiments' / 'latest'
    output.mkdir(parents=True, exist_ok=True)


    rows = []
    trade_rows = []

    for label, genome in [
        ('five_day_baseline', BASELINE),
        ('previous_search_champion', old_champion)
    ]:

        price_only = arena.evaluate(
            genome,
            start_date,
            end_date,
            record_trades=True
        )

        # A constant-risk control distinguishes timing from reduced exposure.
        # The overlay accepts an exposure multiplier relative to the genome.
        if genome.risk_fraction < 0.75:
            raise ValueError(
                "Constant 75% control requires "
                "a genome with at least 75% risk capacity"
            )

        fixed_75_map = {
            day: 0.75 / genome.risk_fraction
            for day in risk_map
        }

        constant_75 = evaluate_news_overlay(
            arena,
            genome,
            start_date,
            end_date,
            fixed_75_map,
            record_trades=True
        )

        news_overlay = evaluate_news_overlay(
            arena,
            genome,
            start_date,
            end_date,
            risk_map,
            record_trades=True
        )

        for setting, result in [
            ('price_only', price_only),
            ('constant_75pct', constant_75),
            ('news_risk_overlay', news_overlay)
        ]:

            trades = result.pop('trades')

            rows.append({
                'strategy': label,
                'setting': setting,
                'genome': genome.name,
                **result
            })

            for trade in trades:
                trade_rows.append({
                    'strategy': label,
                    'setting': setting,
                    **trade
                })


    table = pd.DataFrame(rows)
    trades = pd.DataFrame(trade_rows)
    schedule.to_csv(output / 'news_rule_by_market_date.csv', index=False)
    table.to_csv(output / 'comparison.csv', index=False)
    trades.to_csv(output / 'portfolio_trades.csv', index=False)

    source_manifest = {
        'created_utc': now.isoformat(),
        'news_collection_report': str(campaign_dir),
        'aligned_source_sha256': sha256(aligned_path.read_bytes()).hexdigest(),
        'earlier_championship_summary': str(championship_path),
        'earlier_championship_sha256': sha256(championship_path.read_bytes()).hexdigest(),
        'stock_price_sha256': arena.input_hashes,
        'experiment_start': start_date, 'experiment_end_exclusive': end_date,
        'news_window_days': 7, 'news_embargo_calendar_days': 2,
        'overlay_rule': asdict(rule), 'rebalance_signals': signal_dates,
        'num_completed_rebalances': n_periods,
        'retrospective_news_is_point_in_time_verified': False,
        'training_period_contamination': '2022 overlaps previous evolutionary search folds',
        'statistical_conclusion': 'NONE: short exploratory window, historical selection and news availability limitations',
        'champion_promotion': 'FORBIDDEN', 'real_orders': False,
    }
    (output / 'manifest.json').write_text(json.dumps(source_manifest, indent=2), encoding='utf-8')
    lines = [
        '# Exploratory news-risk overlay comparison — NOT A CHAMPIONSHIP',
        '',
        f'Comparative window: {start_date} through before {end_date}.',
        f'Completed 21-session intervals: **{n_periods}**.',
        f'News rule: conflict share > prior {rule.warmup_sessions} COMPLETE market-day mean '
        f'+ {rule.threshold_z:.1f} std: multiply equity exposure by {rule.stressed_risk_multiplier:.0%}.',
        '21-session stock selection plus DAILY news-risk decisions; all executions next open.',
        'Historical adjusted OHLC and 5 bps cost for EVERY traded dollar, including mid-hold exposure changes.',
        '', '| Strategy | Version | Net return | Drawdown |', '|---|---|---:|---:|',
    ]
    for _, r in table.iterrows():
        lines.append(f"| {r['strategy']} | {r['setting']} | {r['return']:+.2%} | {r['max_drawdown']:.2%} |")
    lines += [
        '', '## Observed exposure changes',
        f'Daily risk-off signals: {int(table.loc[table["setting"] == "news_risk_overlay", "risk_off_signal_days"].iloc[0])}.',
        f'Extra mid-hold risk trades per strategy: '
        f'{table.loc[table["setting"] == "news_risk_overlay", "extra_risk_rebalances"].astype(int).tolist()}.',
        '', '## Why this cannot promote a champion',
        '- Only a handful of 21-session intervals. Outcomes have high sampling error.',
        '- News features come from retrospectively downloaded GDELT data, not verified exact historical release snapshots.',
        '- Date-lagging is conservative but cannot alone certify point-in-time availability.',
        '- Old equity universe is 33 retrospectively chosen survivors; survivor bias persists.',
        '- Historical year 2022 was already exposed to the evolutionary selection process.',
        '- Global event counts cannot choose sector winners; this overlay changes only market exposure.',
        '- A threshold of 1.0 std is predefined for engineering tests, not an optimized news strategy.',
        '- An apparent gain, loss or tie here is not evidence that news is predictive.',
        '', 'Saved: `comparison.csv`, `news_rule_by_market_date.csv`, `portfolio_trades.csv`, `manifest.json`.',
    ]
    (output / 'research_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print('\nPRICE VS NEWS-RISK EXPERIMENT')
    print('Completed 21-session periods:', n_periods)
    overlay_runs = table.loc[table['setting'] == 'news_risk_overlay']
    print('Daily news-signal days evaluated:', int(overlay_runs['daily_news_signal_days'].iloc[0]))
    print('Risk-off daily signals:', int(overlay_runs['risk_off_signal_days'].iloc[0]))
    for _, row in overlay_runs.iterrows():
        print(f"Extra mid-hold trades ({row['strategy']}): {int(row['extra_risk_rebalances'])}")
    print('News-driven exposure state transitions:',
          int(overlay_runs['risk_state_transitions'].iloc[0]))
    print(table[['strategy', 'setting', 'return', 'max_drawdown', 'periods']].to_string(
        index=False, formatters={'return': '{:+.2%}'.format,
                                 'max_drawdown': '{:.2%}'.format}))
    print('Saved research report:', output / 'research_report.md')
    print('PASS: Research experiment executed. No champion promotion or real orders.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(f'NEWS EXPERIMENT ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
