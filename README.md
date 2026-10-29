# Evolutionary Equity Research

Python | Quantitative research | Cross-sectional momentum | Evolutionary optimization | Backtesting

Can evolutionary optimization improve sector-balanced equity momentum strategies over simpler momentum rules? And if it appears to, does the improvement hold up across different market periods?

The framework ranks stocks using momentum, searches for better parameter combinations using an evolutionary algorithm, and compares the results with simpler benchmarks after trading costs. Two unsuccessful experiments—historical market analogs and geopolitical news signals—are also documented.

This is a research prototype based entirely on historical simulations. No live trading or broker integration is involved.

## Overview

| Component | Description |
|----------|-------------|
| Universe of stocks | 33 stocks, with 3 from each of the 11 sectors (selected retrospectively). |
| Strategy | Select the top-ranked stock in each sector, giving 11 equally weighted stocks. |
| Signal | Momentum based signal. May be adjusted for recent returns and volatility. |
| Search | Evolutionary search using elitism, mutation and crossover. |
| Exposure | Either 75% or 100% in stocks. The rest stays in cash (no interest). |
| Rebalance Frequency | Every 21 trading sessions. |
| Trading Assumptions | Signal observed at the end of the session, trade executed at next adjusted open. |
| Costs | 5 bps by default. |
| Benchmark Strategies | A risk-matched 5-day momentum strategy with the same sector selection, and an equally weighted portfolio of all 33 stocks. |
| Search Period | 2019–2020 and 2021–2022. Secondary audit: 2023–2024 (not an untouched test). | |

The 33-stock universe keeps the experiment manageable. However, its constituents were selected retrospectively, introducing survivorship and selection bias.

## Signal Creation and Strategy

For stock i and date t:
```text
momentum(i,t) = Close(i,t-skip) / Close(i,t-lookback) - 1
score(i,t) = momentum(i,t) / (max(volatility20(i,t) * sqrt(21), 0.00001) ^ volatility_penalty)
```

Here, volatility20 is the standard deviation of the last 20 daily close-to-close returns. If the volatility penalty is zero, stocks are ranked by momentum alone. The highest-scoring stock in each sector is selected, giving 11 equally weighted holdings. Either 75% or 100% of the portfolio is invested in stocks, with the remainder in cash.

At each rebalance, the signal uses closing prices and trades take place at the next adjusted open. The portfolio holds for 21 trading sessions, with a proportional trading fee. The simulation allows fractional shares and doesn't model bid-ask spreads or market impact.

## Search Parameters

Each candidate (or Genome) has 4 parameters:

| Parameter | Options |
|----------|---------|
| Lookback | 5, 10, 20, 40, 63, 126 |
| Skip | 0, 5, 21 (must be less than lookback). |
| Volatility Penalty Exponent | 0, 0.5, 1 |
| Exposure | 0.75, 1.00 |

The search starts with a standard five-day momentum candidate and adds randomly generated candidates. At each generation, candidates are ranked by fitness. The strongest survive, while random candidates and candidates produced by mutation and crossover fill the next generation. A fixed seed allows the search to be repeated with the same input data and software.

The example run uses 16 candidates per generation, 5 generations and seed 41. It's a fairly small search.

## Fitness

Each candidate is evaluated on two training periods (2019–2020 and 2021–2022). After trading costs, returns are compared with:

1. An equally weighted portfolio of all 33 stocks.

2. A 5-day momentum strategy with the same portfolio creation (one stock per sector) and target exposure.

The second comparison gives a more direct comparison of the ranking parameters.

The code calculates the following for each training fold f (in percentage points):
```text
edge_f = min(100 * (candidate_return_f - equal_weight_33_return_f),
             100 * (candidate_return_f - risk_matched_5day_return_f))
extra_drawdown_f = 100 * max(0, risk_matched_5day_max_drawdown_f - candidate_max_drawdown_f)
fitness = mean(edge) - 0.5 * population_std(edge) - 0.5 * max(0, -min(edge))
          - 0.5 * mean(extra_drawdown) - 0.25 * indicator(volatility_penalty > 0)
          - 0.25 * indicator(skip > 0)
```

Max drawdown is stored as a negative number, so a candidate is penalized if its drawdown is worse than the risk-matched momentum strategy's. The fitness score rewards performance against both benchmarks, but also penalizes inconsistent results and some additional complexity. The penalty weights are research design choices rather than statistically estimated quantities.

The selected candidate's fitness is in-sample. The 2023–2024 secondary audit is not an independent test either, since that period was examined during development.

## Results

The evolutionary search produces a selected candidate and historical comparisons, but these results do not establish a reliable trading advantage.

With a complete local price cache, the championship ranks candidates, evaluates the selected candidate on 2023–2024, and saves simulated trades. Performance statistics include net return, maximum drawdown, annualized volatility, number of holding periods and total fees.

Earlier 2026 simulations showed promising returns, but their original input price files and report were not preserved in the repository. Those exact numbers have not been reproduced from the published artifacts and are therefore not included as verified results.

There is a separate [fixed strategy experiment for 2026](src/quantlab/evolution/frozen_2026.py). 2026 was also explored during development.

| Finding | Interpretation |
| --- | --- |
| Candidate selected by evolutionary fitness | Best candidate within the searched historical periods, not necessarily a reliable market edge |
| 2023–2024 secondary audit | Descriptive only; this period was inspected during development |
| Earlier 2026 results | Not independently reproducible from the published inputs and reports |
| Demonstrated live or out-of-sample edge | Not established |

The evidence does not establish a persistent edge, statistical significance, or readiness for live trading.

## Unsuccessful Experiments

### Market Analogy

Historical market periods were matched using recent returns, volatility and drawdowns, with further experiments using inflation and Treasury yields. The forecasts did not consistently beat a historical-mean forecast. Using analogs to reduce equity exposure did not improve the later comparison.

### Geopolitical Risk Signals

GDELT event data was used to create daily risk signals. When a signal indicated higher risk, the strategy reduced equity exposure. In a limited 2022 comparison, this did not outperform a constant lower-exposure strategy.

Not all historical news observations could be confirmed as available at the simulated decision time. News features were therefore excluded from the main evolutionary search.

## Validation Periods

| Period | Purpose | Independent test? |
|-------|--------|----------|
| 2018 | Historical and indicator lookback | No (not scored) |
| 2019–2020 | Evolutionary fitness | No (used for optimization) |
| 2021–2022 | Evolutionary fitness | No (used for optimization) |
| 2023–2024 | Secondary audit | No (looked at during development) |
| 2025 | Part of the 2018–2025 price cache | No (not a search fold; also explored during development) |
| 2026 | Fixed-strategy experiments | No (also explored during development) |

A later period is not automatically out of sample. A stronger performance claim would require fixing the strategy and evaluation rules before testing on genuinely unseen data.

## Replicating the Research

Requires Python 3.11 or later.

From the repository root:

Install package and run tests.
```bash
python -m pip install -e ".[dev]"

python -m pytest -q
```

The tests use synthetic price data and won’t require downloading any stocks. The tests are configured in GitHub Actions.

Download price data:
```bash
python -c "from quantlab.data import download_stock_data; download_stock_data('2026-01-01')"
```

This will download the auto-adjusted daily prices from 2018 using `yfinance`. The prices will be downloaded from Jan 2018 until the end of 2025. One CSV will be created per ticker in the `data/raw_2018_2025/` directory. If a file already exists it will be reused.

Run the evolutionary championship:
```bash
python run_evolution.py --population 16 --generations 5 --seed 41 --cost-bps 5
```

Each run creates a timestamped folder in `reports/evolution_championships/` containing:

| File | Description |
|------|-------------|
| summary.json | Summary of the configuration, champion candidate, metrics for each search fold, secondary audit and SHA-256 hash of the price files used. |
| all_candidates.csv | Unique list of candidates with their rankings and fitness. |
| generations.csv | List of populations evaluated for each generation. |
| fitness_folds.csv | Returns and fitness components for each fold. |
| champion_audit_trades.csv | Dates, stocks selected and entry fees for the secondary audit. |
| research_report.md | Summary of the chosen candidate and limitations. |

Note: The code is version controlled and the tests use synthetic data. The raw market data and reports aren’t version controlled. Even with a fixed random seed, the numbers may not be identical due to changes in the adjusted prices from Yahoo Finance and changes in dependencies. The hash will show which CSV files were used for this run.

## File Structure

| File | Description |
|-----|-------------|
| [`run_evolution.py`](run_evolution.py) | Command line to run the championship. |
| [`arena.py`](src/quantlab/evolution/arena.py) | Momentum ranking, sector selection, portfolio, fees and metrics. |
| [`championship.py`](src/quantlab/evolution/championship.py) | Search parameters, selection, mutation, crossover, fitness and reports. |
| [`data.py`](src/quantlab/data.py) | Download price data and manage local cache. |
| [`universe.py`](src/quantlab/universe.py) | Static universe of stocks and sectors. |
| [`frozen_2026.py`](src/quantlab/evolution/frozen_2026.py) | Further fixed strategy exploration. |
| [`market_analogs.py`](src/quantlab/evolution/market_analogs.py), [`macro_regimes.py`](src/quantlab/evolution/macro_regimes.py) | Market analogy research. |
| [`news/`](src/quantlab/news) | News research using GDELT. |
| [`tests/`](tests) | Tests using synthetic data to validate timing, costs, risk and end-to-end behavior. |

## Limitations and Next Steps

The biggest limitation is that the 33-stock universe was selected retrospectively, introducing survivorship and selection bias. Stronger evidence would require reconstructing the stocks eligible on each historical date, including delisted stocks.

Other key limitations are that the search was in-sample, the later periods were explored during development, trades assumed adjusted prices, fractional fills, simple costs and there is uncertainty around whether external data would have been available at the time. The reports do include volatility and drawdown but there isn’t a full turnover analysis, Sharpe ratio or estimates of statistical uncertainty. Nothing should be inferred from just comparing the returns.

Next steps for the research are:

1. Reconstruct a point-in-time universe, including delisted stocks, and keep a versioned copy of the data.

2. Fix the selection and evaluation process before looking at a new testing period.

3. Add Sharpe ratio, turnover, exposure and uncertainty estimates, with more realistic trading assumptions.

Finding a strategy that performs well on historical data is not the same as establishing a reliable edge. Further research must test whether the results survive realistic assumptions and independent evaluation.
