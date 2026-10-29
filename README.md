# Evolutionary Equity Research

Python | Quantitative research | Cross-sectional momentum | Evolutionary optimization | Backtesting

Is there an advantage to using an evolutionary search for sector-balanced equity momentum strategies? And if there is, how do we know the improvement is real?

I built a framework that loads stock data, ranks stocks using momentum, searches for better parameters, and compares the results to benchmark strategies after trading costs. I also kept two ideas I tested that didn't improve the results.

This is still a research prototype. Everything here is based on historical simulations; I haven't connected it to a broker or traded it live.

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

| Search Period | 2019–2020 and 2021–2022. Secondary audit: 2023–2024 (not an untouched test). |

I used 33 stocks to keep the project manageable. I selected them retrospectively, so the results have survivorship and selection bias.

## Signal Creation and Strategy

For stock i and date t:

momentum(i,t) = Close(i, t - skip) / Close(i, t - lookback) - 1

score(i,t) = momentum(i,t) / (max(volatility20(i,t) * sqrt(21), 0.00001) ^ volatility_penalty)

Here, volatility20 is the standard deviation of the last 20 daily close-to-close returns. If the volatility penalty is zero, stocks are ranked by momentum alone. I take the highest-scoring stock in each sector and weight the 11 holdings equally. The portfolio holds either 75% or 100% in stocks, with the remainder in cash.

At each rebalance, the signal uses closing prices and trades take place at the next adjusted open. The portfolio holds for 21 trading sessions, with a proportional trading fee. The simulation allows fractional shares and doesn't model bid-ask spreads or market impact.

## Search Parameters

Each candidate (or Genome) has 4 parameters:

| Parameter | Options |

|----------|---------|

| Lookback | 5, 10, 20, 40, 63, 126 |

| Skip | 0, 5, 21 (must be less than lookback). |

| Volatility Penalty Exponent | 0, 0.5, 1 |

| Exposure | 0.75, 1.00 |

The search starts with standard 5-day momentum and adds randomly generated candidates. At each generation, candidates are ranked by fitness. The best are kept, and the next generation is filled with random candidates and candidates produced through mutation and crossover. I use a fixed seed so the search can be repeated with the same price files and software.

The example run uses 16 candidates per generation, 5 generations and seed 41. It's a fairly small search.

## Fitness

Each candidate is evaluated on two training periods (2019–2020 and 2021–2022). After trading costs, I compare its returns to:

1. An equally weighted portfolio of all 33 stocks.

2. A 5-day momentum strategy with the same portfolio creation (one stock per sector) and target exposure.

The second comparison gives a more direct comparison of the ranking parameters.

The code calculates the following for each training fold f (in percentage points):

edge_f = min(100 * (candidate_return_f - equal_weight_33_return_f),
             100 * (candidate_return_f - risk_matched_5day_return_f))

extra_drawdown_f = 100 * max(0, risk_matched_5day_max_drawdown_f - candidate_max_drawdown_f)

fitness = mean(edge) - 0.5 * population_std(edge) - 0.5 * max(0, -min(edge))
          - 0.5 * mean(extra_drawdown) - 0.25 * indicator(volatility_penalty > 0)
          - 0.25 * indicator(skip > 0)

Max drawdown is stored as a negative number, so a candidate is penalized if its drawdown is worse than the risk-matched momentum strategy's. The fitness score rewards performance against both benchmarks, but also penalizes inconsistent results and some additional complexity. The penalty weights are choices I made for this experiment.

Because the champion is selected using these periods, its fitness is in-sample. I also check 2023–2024, but I had already looked at that period during development.

## What do we know?

The main result of this project is the research process, not a proven trading strategy.

When running the championship on a full local price cache, the candidates are ranked and scores calculated for each fold. A champion is selected and the secondary audit (2023-2024) is completed. The trades are simulated. Performance statistics are recorded for each evaluation including net return, max drawdown, volatility (annualised), number of periods held and total fees.

I'm not showing the earlier 2026 backtest returns here. They looked promising, but I don't have the original price files and report preserved in the repository. I can't reproduce those exact numbers from what I've published, so I'd rather leave them out.

There is a separate [fixed strategy experiment for 2026](src/quantlab/evolution/frozen_2026.py). 2026 was also explored during development.

The project shows how I approached the research and what I tested. It doesn't establish that the strategy has a reliable edge, that any improvement is statistically significant, or that it's ready for live trading.

## Unsuccessful attempts to improve the strategy

### Market Analogy

I tried matching current market conditions to earlier periods using returns, volatility and drawdowns. I also experimented with inflation and Treasury yields. These forecasts weren't consistently better than a simple historical-mean forecast. Using the analogs to reduce exposure didn't help enough to justify adding them to the main strategy.

### Geopolitical Risk Signals

I also tried daily risk signals based on GDELT event data. When a signal suggested higher risk, the portfolio held fewer stocks and more cash. In a limited 2022 test, this didn't outperform keeping exposure lower all the time.

I also couldn't verify that all of the historical news information would have been available at the time of each simulated trade. For that reason, I left news signals out of the main search.

## Validation Periods

| Period | Purpose | Independent test? |

|-------|--------|----------|

| 2018 | Historical and indicator lookback | No (not scored) |

| 2019–2020 | Evolutionary fitness | No (used for optimization) |

| 2021–2022 | Evolutionary fitness | No (used for optimization) |

| 2023–2024 | Secondary audit | No (looked at during development) |

| 2025 | Part of the 2018–2025 price cache | No (not a search fold; also explored during development) |

| 2026 | Fixed-strategy experiments | No (also explored during development) |

A later period isn't automatically out of sample. Before making a stronger performance claim, I'd need to fix the strategy and evaluation rules in advance, then test them on data I haven't examined.

## Replicating the Research

You’ll need Python 3.11+.

Within the root of the repository:

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

This will create a folder with a timestamp in the `reports/evolution_championships/` directory. Inside the folder you’ll find:

| File | Description |

|------|-------------|

| summary.json | Summary of the configuration, champion candidate, metrics for each search fold, secondary audit and SHA-256 hash of the price files used. |

| all_candidates.csv | Unique list of candidates with their rankings and fitness. |

| generations.csv | List of populations evaluated for each generation. |

| fitness_folds.csv | Returns and fitness components for each fold. |

| champion_audit_trades.csv | Dates, stocks selected and entry fees for the secondary audit. |

| research_report.md | Summary of the chosen candidate and limitations.

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

| [`tests/`](tests) | Tests using synthetic data to validate timing, costs, risk and end-to-end.

## Limitations and Next Steps

The biggest limitation is that I selected these 33 stocks retrospectively. That introduces survivorship and selection bias. I'd need to reconstruct which stocks were actually eligible at each date, including delisted stocks, before taking the performance results seriously.

Other key limitations are that the search was in-sample, the later periods were explored during development, trades assumed adjusted prices, fractional fills, simple costs and there is uncertainty around whether external data would have been available at the time. The reports do include volatility and drawdown but there isn’t a full turnover analysis, Sharpe ratio or estimates of statistical uncertainty. Nothing should be inferred from just comparing the returns.

Next steps for the research are:

1. Reconstruct a point-in-time universe, including delisted stocks, and keep a versioned copy of the data.

2. Fix the selection and evaluation process before looking at a new testing period.

3. Add Sharpe ratio, turnover, exposure and uncertainty estimates, with more realistic trading assumptions.

The biggest thing I've learned is that finding a strategy that looks good on historical data is much easier than figuring out whether it actually works.
