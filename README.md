# Evolutionary Equity Research



Python | Quantitative research | Cross-sectional momentum | Evolutionary optimization | Backtesting



Is there an advantage to using an evolutionary search to create sector balanced equity momentum strategies? Can we determine if there is truly an advantage?



I have developed a research framework. It loads equity data, creates a momentum-based stock screening signal, performs an evolutionary search, accounts for transaction costs and compares to benchmark strategies. I have also included 2 unsuccessful attempts at improving the research.

The research has been developed as a prototype. The research is performed using historical simulation only, and hasn’t been deployed with a broker.

## Overview



| Component | Description |

|----------|-------------|

| Universe of stocks | 33 stocks with 3 from each of the 11 sectors (retrospectively selected). |

| Strategy | Select the top stock for each sector to create a portfolio with 11 stocks equally weighted. |

| Signal | Momentum based signal. May be adjusted for recent returns and volatility. |

| Search | Evolutionary search using elitism, mutation and crossover. |

| Exposure | Either 75% or 100% equity exposure. Remaining investment in cash (no interest). |

| Rebalance Frequency | Every 21 trading sessions. |

| Trading Assumptions | Signal observed at the end of the session, trade executed at next adjusted open. |

| Costs | 5 bps by default. |

| Benchmark Strategies | Risk matched 5 day momentum strategy with same sector selection. Equal weighting of all 33 stocks. |

| Search Period | 2019-2020, 2021-2022. Descriptive secondary audit period: 2023-2024.



There are 33 stocks in the example. This is a deliberately small set of stocks to keep the research simple. This means the universe has significant survivorship and selection bias.



## Signal Creation and Strategy



For stock i and date t:



momentum(i,t) = (Close(i, t - skip) / Close(i, t - lookback)) - 1

score(i,t) = momentum(i,t) / (max(volatility20(i,t) sqrt(21), 0.00001) ^ volatility_penalty)



Where volatility20 is the standard deviation of the last 20 daily close-to-close returns. If volatility_penalty is zero, the ranking will be raw momentum. The strategy selects one stock from each sector with the highest score creating a portfolio of 11 equally weighted stocks. Either 75% or 100% equity exposure is created. The rest is invested in cash with no interest.



At rebalancing, the signal is known and the portfolio is invested at the next adjusted open. The portfolio value is simulated with 21 trading sessions. The trades are simulated with a proportional fee. Fractional fills and no spread or market impact modelling is assumed.



## Search Parameters



Each candidate (or Genome) has 4 parameters:



| Parameter | Options |

|----------|---------|

| Lookback | 5, 10, 20, 40, 63, 126 |

| Skip | 0, 5, 21 (must be less than lookback). |

| Volatility Penalty Exponent | 0, 0.5, 1 |

| Exposure | 0.75, 1.00 |



The search starts with a base candidate with standard 5-day momentum. Other valid candidates are randomly sampled. Each generation the candidates are sorted by the fitness function. The best candidates are retained and new randomly sampled candidates along with mutated and crossed-over candidates make up the new generation. A fixed seed is used to ensure the search can be replicated with no changes to the price cache and software.



The example has 16 candidates per generation with 5 generations and seed 41. This is a small parameter search.



## Fitness



Each candidate is evaluated on 2 training periods (2019-2020 and 2021-2022). The returns from the portfolio with costs applied are compared to:



1. An equally weighted portfolio of all 33 stocks.

2. A 5-day momentum strategy with the same portfolio creation (one stock per sector) and target exposure.



The second comparison gives a more direct comparison of the ranking parameters.



The code calculates the following for each training fold f (in percentage points):



edge_f = min(100 (candidate_return_f - equal_weight_33_return_f), 100 (candidate_return_f - risk_matched_5day_return_f))



extra_drawdown_f = 100 max(0, risk_matched_5day_max_drawdown_f - candidate_max_drawdown_f)



fitness = mean(edge) - 0.5 population_std(edge) - 0.5 max(0, -min(edge)) - 0.5 mean(extra_drawdown) - 0.25 indicator(volatility_penalty > 0) - 0.25 indicator(skip > 0)



Max drawdown is stored as a negative return so the drawdown penalty ensures that if the candidate goes further down than the risk matched momentum comparison it will be penalised. The score is designed to give a higher score for higher returns vs both controls. It also penalises inconsistent returns. The weights have been chosen as part of the research design.



As the champion has been selected and optimised on these periods the fitness will be in-sample. There is a separate descriptive audit period (2023-2024).

## What do we know?



The deliverable from this repository is the research process.



When running the championship on a full local price cache, the candidates are ranked and scores calculated for each fold. A champion is selected and the secondary audit (2023-2024) is completed. The trades are simulated. Performance statistics are recorded for each evaluation including net return, max drawdown, volatility (annualised), number of periods held and total fees.



I’m deliberately not showing the backtest return. An exploratory comparison was completed locally previously showing good performance in 2026. However, the original inputs aren’t fixed and neither is the output report. The exact figures haven’t been replicated from the published artifacts and wouldn’t be a fair representation of what is shown in this repository.



There is a separate [fixed strategy experiment for 2026](src/quantlab/evolution/frozen_2026.py). 2026 was also explored during development.



From the evidence we can say this shows a quantitative research process and hypotheses that can be tested. We can’t say there is a sustainable edge, statistically significant or ready for deployment.



## Unsuccessful attempts to improve the strategy



### Market Analogy



I have explored finding similar market periods in the past with similar recent returns, volatility and drawdowns. Some exploratory research was done using inflation and treasury yields. The analogy forecasts weren’t consistently better than a simple mean forecast. Attempting to reduce equity exposure using the analogy didn’t improve performance sufficiently for the later comparison.



### Geopolitical Risk Signals



I have experimented using daily risk signals based on GDELT events. When the signal indicated high risk, less exposure was taken. This didn’t outperform a simple lower exposure control in a limited 2022 comparison.



It couldn’t be fully proven that all the news information from history would be available to the investor at the time of decision making. So news features aren’t included in the main evolutionary search.



## Validation Periods



| Period | Purpose | Validated |

|-------|--------|----------|

| 2018 | Historical and indicator lookback | No (not scored) |

| 2019-2020 | Evolutionary fitness | Yes (in-sample, optimised) |

| 2021-2022 | Evolutionary fitness | Yes (in-sample, optimised) |

| 2023-2024 | Descriptive secondary audit | Yes (explored during development) |

| 2025 | Part of 2018-2025 price cache | No (not default scoring period, explored during development) |

| 2026 | Fixed strategy explorations | Yes (explored during development).

Just because it is later doesn’t mean it is out of sample. To claim performance I would need to pre-register the strategy, criteria and an unseen observation period.



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

The most impactful limitation is that the 33 stocks have been retrospectively selected meaning there is survivorship and selection bias. A time-point universe with stocks that would have been eligible and delistings would need to be created before we take the strategy performance seriously.

Other key limitations are that the search was in-sample, the later periods were explored during development, trades assumed adjusted prices, fractional fills, simple costs and there is uncertainty around whether external data would have been available at the time. The reports do include volatility and drawdown but there isn’t a full turnover analysis, Sharpe ratio or estimates of statistical uncertainty. Nothing should be inferred from just comparing the returns.

Next steps for the research are:



1. Reconstruct a time-point universe with eligible stocks. Ensure a versioned snapshot of the data.

2. Fix the selection and evaluation process before observing a new prospective period.

3. Include comparable metrics with Sharpe ratio, turnover, exposure and uncertainty estimates. Assume more realistic trading.



The key takeaway is that it is much easier to find an attractive strategy in history. This has been a learning exercise to create and critically analyse the research process.
