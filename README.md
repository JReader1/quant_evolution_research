# Evolutionary Equity Research

**Python · Quantitative research · Momentum strategies · Backtesting**

Can a systematic search improve a simple equity momentum strategy, and does that improvement persist outside the periods used to choose it?

I built this project to investigate that question. It combines a sector-balanced stock-selection strategy, an evolutionary parameter search, and a transaction-cost-aware historical simulator. The research also includes two unsuccessful extensions, historical market analogs and geopolitical news signals, that I kept because their failures are useful evidence.

**Status:** Research prototype. Historical simulations only; no live trading or broker integration.

## Research design

| Component | Implementation |
| --- | --- |
| Universe | 33 U.S. equities, three each from 11 sectors |
| Portfolio | Highest-ranked stock within each sector; 11 equal-weight positions |
| Signal | Trailing momentum, optional recent-period skip, optional volatility adjustment |
| Search | Seeded evolutionary search with selection, mutation, and crossover |
| Exposure | Candidate target equity exposure of 75% or 100% |
| Rebalancing | Every 21 trading sessions |
| Simulated execution | Signal at the close; trade at the following session's adjusted open |
| Trading costs | Default 5 basis points per dollar traded |
| Comparisons | Five-day momentum and equal-weight ownership of all 33 stocks |

Each candidate specifies a momentum lookback, skip period, volatility penalty, and equity exposure. The search evaluates candidates on **2019–2020** and **2021–2022** historical windows. Fitness considers benchmark-relative returns, performance instability between windows, excess drawdown, and a small complexity penalty.

The **2023–2024** period is reported separately as a secondary audit. It is *not* an untouched out-of-sample test: I had inspected historical performance repeatedly during development.

## Findings

A candidate explored in this project, `m10_skip0_vol1_risk1`, uses 10-session momentum, no skip, volatility adjustment, and 100% target equity exposure.

An earlier local simulation covering **nine completed holding periods in 2026** (final exit October 6) reported:

| Strategy | Simulated net return |
| --- | ---: |
| Evolutionary candidate | +18.90% |
| Five-day momentum baseline | +16.75% |
| Equal-weight 33-stock baseline | +15.27% |

**Interpretation:** These figures come from a prior local research run. The input price files and that run's report are not committed to this repository, so these exact numbers have **not been independently reproduced from the published code and data**. They are exploratory comparisons, not live returns or evidence of a reliable trading edge. Earlier historical audits did not consistently satisfy the strategy's benchmark hurdles, and the 2026 comparison is not a pristine blind holdout.

Two additional hypotheses were tested and **not adopted**:

- **Historical analogs:** Similarity matching using prior returns, volatility, and drawdowns, with exploratory inflation and Treasury-yield features. The analog forecasts did not consistently improve on a historical-mean forecast; a risk-reduction overlay also failed to improve the later equity-strategy comparison.
- **GDELT news overlay:** Daily event-based geopolitical risk signals. In a short 2022 experiment, reducing exposure on elevated-risk dates did not outperform a constant lower-exposure control.

The main evolutionary championship deliberately excludes historical news features because their point-in-time availability has not been established.

## Reproducing the research

Requires **Python 3.11+**. From the repository root:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

The tests use generated synthetic prices and do not need market downloads. GitHub Actions runs them on Linux (Python 3.11) and Windows (Python 3.13).

To build the historical price cache and run the evolutionary search:

```bash
python -c "from quantlab.data import download_stock_data; download_stock_data('2026-01-01')"
python run_evolution.py --population 16 --generations 5 --seed 41 --cost-bps 5
```

The download step requests adjusted daily Yahoo Finance prices through `yfinance` for 2018–2025 and caches the 33 input files in `data/raw_2018_2025/`. Existing cached CSVs are reused. Raw datasets and generated reports are excluded from Git; data-provider revisions may change rerun results.

Each championship run writes `summary.json`, fold-level fitness data, candidate rankings, simulated trades, and a Markdown report under `reports/evolution_championships/`. The summary includes SHA-256 hashes of the input files for auditability.

## Code map

| File | Purpose |
| --- | --- |
| [`run_evolution.py`](run_evolution.py) | Command-line entry point |
| [`evolution/arena.py`](src/quantlab/evolution/arena.py) | Signals, portfolio accounting, fees, and performance |
| [`evolution/championship.py`](src/quantlab/evolution/championship.py) | Candidate generation, fitness, and reports |
| [`data.py`](src/quantlab/data.py) | Historical data download and cache |
| [`evolution/frozen_2026.py`](src/quantlab/evolution/frozen_2026.py) | Later-period evaluation experiment |
| [`evolution/market_analogs.py`](src/quantlab/evolution/market_analogs.py) and [`macro_regimes.py`](src/quantlab/evolution/macro_regimes.py) | Historical analog research |
| [`news/`](src/quantlab/news) | GDELT collection and exploratory risk overlay |
| [`tests/`](tests) | Synthetic-data timing, fee, risk, and end-to-end tests |

## Limitations

The central limitation is **selection and survivorship bias**: these 33 stocks were chosen retrospectively rather than reconstructed from historically eligible constituents. The evolutionary search repeatedly examines its training windows, and the later periods have also been inspected; none of the results should be interpreted as independent proof of generalization.

The simulator assumes adjusted opening-price execution, fractional shares, and a simple transaction-cost model. Bid–ask spreads, market impact, constraints on execution, and point-in-time revisions to some external data are not fully modeled. The reported results are not a live or forward-tested track record.

My next research step is a point-in-time investable universe followed by a genuinely untouched evaluation period. Until those checks are complete, I regard this as a study in **research design and backtesting**, not a validated trading strategy.
