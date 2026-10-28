"""Reproducible evolutionary search with a locked, price-only evaluation arena."""
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import asdict
import csv
import json
import random
import statistics

from quantlab.evolution.arena import BASELINE, Genome, HistoricalArena

# Both search folds are reused by evolution, so these are NOT out-of-sample
# after the tournament has chosen its winner. 2023-24 is a once-per-run
# secondary historical audit, not untouched real-world forward evidence.
SEARCH_FOLDS = [('2019-01-01', '2021-01-01'), ('2021-01-01', '2023-01-01')]
AUDIT_FOLD = ('2023-01-01', '2025-01-01')
ALL_LOOKBACKS = (5, 10, 20, 40, 63, 126)
ALL_SKIPS = (0, 5, 21)
ALL_VOL = (0.0, 0.5, 1.0)
ALL_RISK = (0.75, 1.0)


def random_genome(rng: random.Random) -> Genome:
    lookback = rng.choice(ALL_LOOKBACKS)
    return Genome(lookback=lookback,
                  skip=rng.choice([v for v in ALL_SKIPS if v < lookback]),
                  volatility_penalty=rng.choice(ALL_VOL),
                  risk_fraction=rng.choice(ALL_RISK))


def mutate(parent: Genome, rng: random.Random) -> Genome:
    change = rng.choice(('lookback', 'skip', 'volatility_penalty', 'risk_fraction'))
    vals = asdict(parent)
    if change == 'lookback':
        vals['lookback'] = rng.choice(ALL_LOOKBACKS)
        if vals['skip'] >= vals['lookback']:
            vals['skip'] = 0
    elif change == 'skip':
        vals['skip'] = rng.choice([v for v in ALL_SKIPS if v < vals['lookback']])
    elif change == 'volatility_penalty':
        vals['volatility_penalty'] = rng.choice(ALL_VOL)
    else:
        vals['risk_fraction'] = rng.choice(ALL_RISK)
    return Genome(**vals)


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    new = Genome(
        lookback=rng.choice([a.lookback, b.lookback]),
        skip=rng.choice([a.skip, b.skip]),
        volatility_penalty=rng.choice([a.volatility_penalty, b.volatility_penalty]),
        risk_fraction=rng.choice([a.risk_fraction, b.risk_fraction]),
    )
    if new.skip >= new.lookback:
        new = Genome(new.lookback, 0, new.volatility_penalty, new.risk_fraction)
    new.validate()
    return new



def fitness(arena: HistoricalArena, genome: Genome,
            baseline_returns: list,
            bench_returns: list) -> tuple[float, list]:

    # Compare candidates against the original momentum
    # strategy at the same level of equity exposure.
    matched = Genome(
        lookback=BASELINE.lookback,
        skip=BASELINE.skip,
        volatility_penalty=BASELINE.volatility_penalty,
        risk_fraction=genome.risk_fraction
    )

    records = []

    for (start, end), baseline, benchmark in zip(
        SEARCH_FOLDS, baseline_returns, bench_returns
    ):

        metric = arena.evaluate(genome, start, end)

        matched_result = arena.evaluate(
            matched, start, end
        )

        # All returns are net of the arena's trading costs.
        vs_market = 100 * (
            metric["return"] - benchmark["return"]
        )

        vs_matched = 100 * (
            metric["return"] - matched_result["return"]
        )

        # Judge a strategy by its weaker comparison.
        edge = min(vs_market, vs_matched)

        # Additional drawdown compared with the
        # exposure-matched baseline.
        extra_drawdown_pp = 100 * max(
            0.0,
            matched_result["max_drawdown"]
            - metric["max_drawdown"]
        )

        records.append({
            "fold": f"{start[:4]}-{int(end[:4])-1}",
            **metric,
            "excess_vs_33stock_pp": vs_market,
            "excess_vs_5day_pp": 100 * (
                metric["return"] - baseline["return"]
            ),
            "excess_vs_risk_matched_5day_pp": vs_matched,
            "worst_comparison_pp": edge,
            "extra_drawdown_vs_matched_pp": extra_drawdown_pp
        })

    edges = [
        r["worst_comparison_pp"]
        for r in records
    ]

    # Penalize unstable performance across historical folds.
    instability = (
        0.5 * statistics.pstdev(edges)
    )

    # A strong period should not completely conceal
    # an unsuccessful period.
    losing_fold = (
        0.5 * max(0.0, -min(edges))
    )

    drawdown_penalty = (
        0.5 * statistics.mean(
            r["extra_drawdown_vs_matched_pp"]
            for r in records
        )
    )

    # Prefer simpler models when evidence is similar.
    complexity = (
        0.25 * bool(genome.volatility_penalty)
        + 0.25 * bool(genome.skip)
    )

    score = (
        statistics.mean(edges)
        - instability
        - losing_fold
        - drawdown_penalty
        - complexity
    )

    return score, records



def run_championship(root: Path, *, population: int = 16, generations: int = 5,
                     seed: int = 41, cost_bps: float = 5.0) -> tuple[Path, dict]:
    if not 4 <= population <= 100 or not 1 <= generations <= 100:
        raise ValueError('Population 4-100; generations 1-100')
    root = Path(root).resolve()
    rng = random.Random(seed)
    arena = HistoricalArena(root, cost_bps=cost_bps)
    baselines = [arena.evaluate(BASELINE, a, b) for a, b in SEARCH_FOLDS]
    benchmarks = [arena.evaluate(None, a, b) for a, b in SEARCH_FOLDS]
    baseline_score, _ = fitness(arena, BASELINE, baselines, benchmarks)
    cache: dict[Genome, tuple[float, list]] = {}
    population_now = [BASELINE]
    while len(population_now) < population:
        g = random_genome(rng)
        if g not in population_now:
            population_now.append(g)
    history = []
    fold_history = []
    for generation in range(1, generations + 1):
        ranked = []
        for g in population_now:
            if g not in cache:
                cache[g] = fitness(arena, g, baselines, benchmarks)
            score, folds = cache[g]
            ranked.append((score, g))
            history.append({'generation': generation, 'candidate': g.name,
                            'fitness': score, **asdict(g)})
            for f in folds:
                fold_history.append({'generation': generation, 'candidate': g.name, **f})
        ranked.sort(key=lambda x: (-x[0], x[1].name))
        top = ranked[0]
        print(f'Generation {generation:02d}: {top[1].name} | fitness {top[0]:+.3f}')
        if generation == generations:
            break
        elite_count = max(2, population // 4)
        elites = [g for _, g in ranked[:elite_count]]
        following = elites.copy()
        attempts = 0
        while len(following) < population:
            attempts += 1
            if attempts > 20000:
                raise RuntimeError('Cannot form distinct next generation')
            if rng.random() < 0.3:
                child = random_genome(rng)
            elif rng.random() < 0.6:
                child = mutate(rng.choice(elites), rng)
            else:
                child = crossover(rng.choice(elites), rng.choice(elites), rng)
                child = mutate(child, rng)
            if child not in following:
                following.append(child)
        population_now = following
    best = sorted(cache, key=lambda g: (-cache[g][0], g.name))[0]
    best_score, best_folds = cache[best]
    audit_champion = arena.evaluate(best, *AUDIT_FOLD, record_trades=True)
    audit_baseline = arena.evaluate(BASELINE, *AUDIT_FOLD)
    audit_benchmark = arena.evaluate(None, *AUDIT_FOLD)

    # HISTORICAL QUALIFICATION CHECK

    search_hurdles_passed = all(
        fold["excess_vs_33stock_pp"] > 0
        and fold["excess_vs_risk_matched_5day_pp"] > 0
        for fold in best_folds
    )

    # Evaluate the secondary period against a baseline
    # that uses the same equity exposure as the winner.
    audit_matched = arena.evaluate(
        Genome(
            BASELINE.lookback,
            BASELINE.skip,
            BASELINE.volatility_penalty,
            best.risk_fraction
        ),
        *AUDIT_FOLD
    )

    audit_hurdles_passed = (
        audit_champion["return"] > audit_matched["return"]
        and audit_champion["return"] > audit_benchmark["return"]
    )

    historical_hurdles_passed = (
        search_hurdles_passed and audit_hurdles_passed
    )

    print(
        "Historical hurdle check:",
        "PASS" if historical_hurdles_passed else "FAIL",
        "(research only; no automatic promotion)"
    )

    now = datetime.now(timezone.utc)
    run_id = now.strftime('%Y%m%dT%H%M%S%fZ')
    folder = root / 'reports' / 'evolution_championships' / run_id
    folder.mkdir(parents=True, exist_ok=False)
    def save_csv(filename, rows):
        with (folder / filename).open('x', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    save_csv('generations.csv', history)
    save_csv('fitness_folds.csv', fold_history)
    save_csv('champion_audit_trades.csv', audit_champion.pop('trades'))
    leaderboard = [{'candidate': g.name, 'fitness': s, **asdict(g)}
                   for g, (s, _) in sorted(cache.items(), key=lambda it: (-it[1][0], it[0].name))]
    save_csv('all_candidates.csv', leaderboard)
    summary = {
        'created_utc': now.isoformat(),
        'seed': seed, 'population': population, 'generations': generations,
        'unique_candidates_evaluated': len(cache), 'cost_bps_per_traded_dollar': cost_bps,
        'data_files_sha256': arena.input_hashes,
        'historical_universe_warning': '33 retrospectively selected surviving stocks; survivorship/selection bias remains',
        'news_mode': 'DISABLED: no verified historical point-in-time news features provided',
        'fitness_search_windows': SEARCH_FOLDS,
        'secondary_audit_window': AUDIT_FOLD,
        'audit_warning': '2023-24 and 2025 have been inspected in prior research; audit is descriptive and not fresh blind holdout',
        'baseline': {'genome': asdict(BASELINE), 'fitness': baseline_score, 'search_folds': baselines},
        'champion': {'genome': asdict(best), 'candidate': best.name,
                     'fitness': best_score, 'search_folds': best_folds},
        'secondary_audit': {'champion': audit_champion,
                            'five_day_baseline': audit_baseline,
                            'equal_weight_33': audit_benchmark,
                            'champion_minus_33stock_pp': 100 * (audit_champion['return'] - audit_benchmark['return']),
                            'champion_minus_five_day_pp': 100 * (audit_champion['return'] - audit_baseline['return'])},
        "historical_hurdles_passed": historical_hurdles_passed,
        'promotion_decision': 'NOT PROMOTED: research winner only; prospective validation required',
        'real_trading': False,
    }
    with (folder / 'summary.json').open('x', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    md = [
        '# Evolutionary research championship',
        f'Run: {run_id}; random seed {seed}; {population} candidates/generation; {generations} generations.',
        f'Unique genomes evaluated: {len(cache)}.',
        '',
        '## Search winner (NOT promoted for live trading)',
        f'**{best.name}**, fitness **{best_score:+.3f}** (baseline {baseline_score:+.3f}).',
        f'Configuration: `{asdict(best)}`.',
        '',
        '## Secondary historical audit (2023-2024; NOT independent blind test)',
        f"Champion net return: **{audit_champion['return']:+.2%}**, drawdown **{audit_champion['max_drawdown']:.2%}**.",
        f"Five-day momentum net return: **{audit_baseline['return']:+.2%}**.",
        f"All-33 equal-weight net return: **{audit_benchmark['return']:+.2%}**.",
        f"Champion minus equal-weight: **{audit_champion['return'] - audit_benchmark['return']:+.2%}**.",
        '**Promotion gate: NOT PASSED / no promotion.** Historical search optimization does not establish a dependable advantage.',
        '',
        '## Critical limits',
        '- All histories cover a retrospectively chosen 33-stock survivor universe; backtests may be optimistic.',
        '- Search-fold scores were repeatedly optimized; winner score is not unbiased out-of-sample evidence.',
        '- The later audit is descriptive, not an untouched holdout; 2025 already viewed elsewhere.',
        '- Adjusted OHLC, hypothetical fractional shares, 5 bps fees; spreads, slippage, taxes and dividends/execution realism remain limited.',
        '- Historical news **not included**: source publication time alone does not establish point-in-time availability.',
        '- No code rewriting, broker access or live-money trading enabled.',
        '',
        'See `summary.json`, `all_candidates.csv`, `fitness_folds.csv`, `champion_audit_trades.csv`.',
    ]
    (folder / 'research_report.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    return folder, summary
