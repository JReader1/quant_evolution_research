# Historical momentum-strategy comparison (2023–2024)

Source: [GitHub Actions run 43549194278](https://github.com/JReader1/quant_evolution_research/actions/runs/43549194278), successful on 2026-11-13 US Eastern / 2026-11-14 UTC. Report artifact: `historical-research-results` (includes the complete `summary.json`, parameters, input hashes and trades). The workflow successfully ran **9 tests** before simulating the strategy.

**Important:** This is a *descriptive, previously inspected historical audit*, not an independent out-of-sample test. All three portfolios use the same retrospectively chosen survivor universe of 33 stocks, and therefore suffer from selection/survivorship bias. Simulated execution uses adjusted open prices, fractional positions and 5-basis-point trading fees, not a complete real-world execution model.

| Metric | Selected evolutionary strategy | Fixed five-day momentum | Equal-weight 33 stocks |
| --- | ---: | ---: | ---: |
| Total net simulated return | +9.87% | +11.84% | +32.57% |
| Maximum drawdown | -16.26% | -17.79% | -9.35% |
| Annualized volatility | 12.36% | 13.27% | 11.25% |
| Annualized Sharpe ratio (zero risk-free) | 0.4583 | 0.5053 | 1.3615 |

The algorithm selected `m10_skip0_vol1_risk1`: 10-session momentum lookback, no skip, a full volatility penalty, and 100% target stock exposure. Its search used 16 candidates per generation, 5 generations and seed 41. The two training periods were 2019–2020 and 2021–2022.

**Interpretation:** The evolved strategy experienced 1.53 percentage points less maximum drawdown and 0.91 percentage points lower annualized volatility than fixed five-day momentum, but also 1.97 percentage points less total return. The equal-weight strategy had higher return, lower drawdown and a higher Sharpe ratio than either momentum strategy in this audit. The historical promotion hurdle failed.

Do not interpret these numbers as proof of future results or a statistically significant edge. To claim an independent result, freeze the strategy and its selection rules, use a point-in-time universe, and evaluate a genuinely untouched subsequent period. The workflow retains the exact input SHA-256 hashes in its archived summary. Yahoo Finance adjusted-price revisions can produce slightly different figures on rerun.
