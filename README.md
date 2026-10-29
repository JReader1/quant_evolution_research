# Evolutionary Equity Research

**Python · Quantitative research · Cross-sectional momentum · Evolutionary optimization · Backtesting**

**Research question:** Does an evolutionary search find sector-balanced equity momentum rules that outperform simpler, comparably implemented strategies—and does any apparent improvement survive careful evaluation?

I built a Python research framework to investigate that question. It combines equity data ingestion, momentum-based stock selection, an evolutionary parameter search, transaction-cost-aware portfolio simulation, and explicit checks against benchmark strategies. I also document two unsuccessful extensions: historical market analogs and geopolitical news signals.

**Project status:** Research prototype. Historical simulations only; no live trading, broker integration, or demonstrated predictive edge.

## Research design at a glance

| Component | Implementation |
| --- | --- |
| Equity universe | 33 retrospectively selected U.S. stocks, three in each of 11 sectors |
| Portfolio construction | Select the highest-scoring stock in each sector; 11 equally weighted positions |
| Signal | Momentum, optionally adjusted for recent returns and realized volatility |
| Search | Seeded evolutionary search using elitism, mutation, and crossover |
| Target exposure | 75% or 100% equities; remaining capital held as unremunerated cash |
| Rebalancing | Every 21 trading sessions |
| Execution assumption | Observe signals at a session's close; trade at the next adjusted open |
| Trading costs | Default 5 basis points per dollar traded |
| Controls | Risk-matched five-day momentum (same sector-selection rules), plus equal-weight ownership of all 33 stocks |
| Research windows | Search on 2019–2020 and 2021–2022; descriptive secondary audit on 2023–2024 |

The 33-stock sample is a deliberately manageable research universe, **not** a historical reconstruction of a point-in-time investable index. Its retrospective selection introduces substantial survivorship and selection bias.

## Signal and portfolio construction

For stock `i` and signal date `t`, the model computes:

```text
momentum(i,t) = Close(i,t-skip) / Close(i,t-lookback) - 1
score(i,t)    = momentum(i,t) /
                max(volatility20(i,t) * sqrt(21), 0.00001) ^ volatility_penalty
```

`volatility20` is the standard deviation of the previous 20 daily close-to-close returns. A volatility penalty of zero leaves the ranking as unadjusted momentum. The strategy selects **one stock per sector** using the highest score, then allocates equally across the 11 names. Target exposure is either 75% or 100%; the remainder stays in cash without modeled interest.

At each rebalance, the simulator uses information available through the signal close, enters at the following adjusted open, and holds through 21 trading sessions. It simulates portfolio value, buys and sells, and a proportional fee on the dollar amount traded. Trades assume fractional shares, with no explicit spread or market-impact model.

### Evolutionary search space

Each candidate (a `Genome` in the code) has four parameters:

| Parameter | Allowed values |
| --- | --- |
| Momentum lookback | 5, 10, 20, 40, 63, or 126 sessions |
| Skip period | 0, 5, or 21 sessions, strictly less than the lookback |
| Volatility penalty exponent | 0, 0.5, or 1 |
| Equity exposure | 0.75 or 1.00 |

The search begins with the standard five-day momentum candidate and randomly sampled valid candidates. In each generation, candidates are ranked by the fitness function below; the strongest candidates survive, while fresh random candidates and offspring produced by mutation or crossover populate the next generation. A fixed seed makes the *search procedure* repeatable for an unchanged price cache and software environment.

The documented example uses **16 candidates per generation, five generations, and seed 41**. This is a modest parameter search, not evidence that a globally optimal strategy has been found.

## Fitness and comparisons

The implementation evaluates each candidate in two separate training windows: **2019–2020** and **2021–2022**. For every window, it compares after-cost portfolio returns with:

1. An equal-weight 33-stock benchmark.
2. A five-day momentum strategy that uses the **same one-stock-per-sector portfolio construction and the candidate's target exposure**.

The second comparison is important: it isolates the effect of the candidate's ranking parameters more directly than comparing an 11-stock selection portfolio only against all 33 stocks.

The code computes the following quantities in **percentage points** for each training fold `f`:

```text
edge_f = min(
    100 * (candidate_return_f - equal_weight_33_return_f),
    100 * (candidate_return_f - risk_matched_5day_return_f)
)

extra_drawdown_f = 100 * max(
    0, risk_matched_5day_max_drawdown_f - candidate_max_drawdown_f
)

fitness = mean(edge)
          - 0.5 * population_std(edge)
          - 0.5 * max(0, -min(edge))
          - 0.5 * mean(extra_drawdown)
          - 0.25 * indicator(volatility_penalty > 0)
          - 0.25 * indicator(skip > 0)
```

Max drawdown is stored as a negative return, so the drawdown term penalizes a candidate that falls further than its risk-matched momentum comparator. The score rewards returns relative to **both** controls while penalizing inconsistency, a weak fold, excess drawdown, and added signal complexity. The weights are research design choices, **not statistically estimated measures of significance**.

Selection and repeated optimization on these windows mean the resulting champion's fitness is **in-sample**. The distinct 2023–2024 audit is descriptive, not a clean blind holdout: that period was examined during development.

## Results: what is and is not established

The repository's reproducible deliverable is the **research procedure**, rather than a verified profitable trading strategy.

Running the championship on a complete local price cache generates candidate rankings, fold-level scores, a selected champion, a 2023–2024 secondary audit, and simulated trades. The performance record for each evaluation includes net return, maximum drawdown, annualized volatility, number of holding periods, and total fees.

**I am intentionally not presenting a headline backtest return here.** An earlier local exploratory comparison reported attractive 2026 performance, but its original price inputs and output report are not committed, and the exact figures have not been reproduced from the published artifacts. Repeating them as independently verifiable evidence would overstate what the repository demonstrates.

A separate [fixed-strategy 2026 experiment](src/quantlab/evolution/frozen_2026.py) exists, but 2026 was also investigated during development. It should not be described as a previously untouched out-of-sample test.

**Conclusion from the evidence so far:** This framework demonstrates a quantitative research process and produces testable hypotheses. It does **not** establish a persistent excess return, statistical significance, or readiness for live trading.

## Experiments that did not improve the strategy

### Historical market analogs

I explored finding past market periods with similar recent returns, volatility, and drawdowns, including exploratory inflation and Treasury-yield inputs. The analogy-based forecasts did not reliably beat a simple historical-mean forecast. Using analogs to reduce equity exposure in difficult conditions also failed to improve the later comparison enough to justify inclusion.

### Geopolitical news risk signals

I experimented with daily risk features derived from GDELT event data and reduced portfolio exposure when those signals suggested elevated risk. In a limited 2022 comparison, this approach did not outperform a simpler constant lower-exposure control.

I could not fully establish whether all historical news information would have been available to an investor at the simulated decision time. **News features are therefore excluded from the main evolutionary search.** These negative results and timing concerns are part of the research, not omitted experiments.

## Validation timeline

| Data period | Role | Validation status |
| --- | --- | --- |
| 2018 | Historical context and indicator lookback | Not a scored search fold |
| 2019–2020 | Evolutionary fitness evaluation | In-sample / optimized repeatedly |
| 2021–2022 | Evolutionary fitness evaluation | In-sample / optimized repeatedly |
| 2023–2024 | Secondary historical audit | Development-inspected; **not blind** |
| 2025 | Included in the 2018–2025 price cache | Not a default championship scoring window; previously inspected in development |
| 2026 | Separate fixed-strategy explorations | Development-inspected; **not an untouched holdout** |

A later date is not automatically an out-of-sample test. I would need to pre-register the strategy, evaluation criteria, and an untouched observation window before making a prospective performance claim.

## Reproduce the research

Requires **Python 3.11+**. Run the following from the repository root.

**1. Install the package and run tests**

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

The test suite uses synthetic price data and does not require stock downloads. The repository includes GitHub Actions test configuration.

**2. Download and cache historical prices**

```bash
python -c "from quantlab.data import download_stock_data; download_stock_data('2026-01-01')"
```

This downloads auto-adjusted daily prices via `yfinance` from January 2018 through the end of 2025, storing one CSV per ticker under `data/raw_2018_2025/`. Files already present in the cache are reused rather than silently replaced.

**3. Run the evolutionary championship**

```bash
python run_evolution.py --population 16 --generations 5 --seed 41 --cost-bps 5
```

The run creates a timestamped folder under `reports/evolution_championships/` containing:

| Output | Contents |
| --- | --- |
| `summary.json` | Configuration, winning candidate, search-fold metrics, secondary audit, and SHA-256 hashes of input price files |
| `all_candidates.csv` | Unique candidate leaderboard and fitness |
| `generations.csv` | Population evaluated in each generation |
| `fitness_folds.csv` | Fold-by-fold returns and fitness components |
| `champion_audit_trades.csv` | Dates, selected stocks, and entry fees for the secondary audit |
| `research_report.md` | Compact summary of the selected candidate and limitations |

**Reproducibility boundary:** Code and synthetic tests are versioned, but historical raw market data and generated reports are not. A fixed random seed alone cannot guarantee identical numbers if Yahoo Finance revises adjusted prices or if dependencies change. The file hashes document which input CSVs were used in an individual local run.

## Repository map

| Path | Purpose |
| --- | --- |
| [`run_evolution.py`](run_evolution.py) | Championship command-line entry point |
| [`evolution/arena.py`](src/quantlab/evolution/arena.py) | Momentum scores, sector selection, portfolio accounting, fees, and metrics |
| [`evolution/championship.py`](src/quantlab/evolution/championship.py) | Search space, selection, mutation, crossover, fitness, and reports |
| [`data.py`](src/quantlab/data.py) | Download and local price-cache management |
| [`universe.py`](src/quantlab/universe.py) | Fixed stock universe and sector mapping |
| [`evolution/frozen_2026.py`](src/quantlab/evolution/frozen_2026.py) | Later fixed-strategy exploration |
| [`evolution/market_analogs.py`](src/quantlab/evolution/market_analogs.py) and [`macro_regimes.py`](src/quantlab/evolution/macro_regimes.py) | Historical market analogue research |
| [`news/`](src/quantlab/news) | GDELT-related exploratory news research |
| [`tests/`](tests) | Synthetic-data tests for timing, costs, risk, and end-to-end behavior |

## Limitations and next steps

The most consequential limitation is **survivorship and selection bias**: the 33 stocks were selected with hindsight. A point-in-time universe that includes historically eligible stocks and delistings is needed before taking strategy performance seriously.

Other important limitations are repeated in-sample search, development inspection of later windows, adjusted-price execution assumptions, fractional fills, simplistic transaction costs, and uncertain point-in-time availability of external data. The existing reports include volatility and drawdown but **do not yet supply a complete turnover analysis, Sharpe ratio, or statistical uncertainty estimates**. None should be inferred from a return comparison alone.

The next research priorities are:

1. Reconstruct a point-in-time eligible universe and preserve a versioned data snapshot.
2. Freeze the selection procedure and evaluation metrics before observing a new prospective period.
3. Add matched comparisons with Sharpe ratio, turnover, exposure, and uncertainty estimates, with more realistic execution assumptions.

The central lesson is that **finding an appealing historical strategy is much easier than establishing an investable edge**. This project is an exercise in building—and critically evaluating—the research process.
