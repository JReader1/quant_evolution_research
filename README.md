# Evolutionary Equity Research

A Python project for testing momentum-based stock selection, evolutionary parameter search, and market-risk signals. I built it to find out whether a strategy that looks good in historical data still holds up against simple alternatives.

The main experiment is long-only: choose one stock from each of 11 sectors, hold the basket for 21 trading sessions, then rebalance. There is no broker connection or live trading.

## How it works

- **Universe:** 33 large-cap U.S. equities, three per sector. This is a retrospectively selected research universe, **not** a point-in-time S&P 500.
- **Signals:** momentum over different lookbacks, optional skipping of recent sessions, and volatility adjustment.
- **Search:** an evolutionary algorithm generates, crosses, and mutates strategy configurations (called `Genome` in the code).
- **Execution:** signals use information through the close; trades are simulated at the next session's adjusted open. The backtest charges 5 basis points per traded dollar.
- **Evaluation:** candidates are compared against five-day momentum and equal-weight controls. The search uses historical folds; the secondary audit and later 2026 evaluation are descriptive, not pristine blind tests.

The core model lives in [`arena.py`](src/quantlab/evolution/arena.py) (portfolio accounting) and [`championship.py`](src/quantlab/evolution/championship.py) (search and scoring). The optional research modules are [`market_analogs.py`](src/quantlab/evolution/market_analogs.py), [`macro_regimes.py`](src/quantlab/evolution/macro_regimes.py), and [`frozen_2026.py`](src/quantlab/evolution/frozen_2026.py).

## What I found

The evolutionary search selected `m10_skip0_vol1_risk1`: 10-session momentum, no skipped sessions, volatility adjustment, and full target equity exposure.

In a **later-period 2026 simulation** using nine completed holding periods (last exit: October 6), the results were:

| Portfolio | Simulated net return |
| --- | ---: |
| Evolutionary candidate | +18.90% |
| Original five-day momentum | +16.75% |
| Equal-weight 33 stocks | +15.27% |

These are outputs from local research runs, not verified live returns. The selected candidate was **not approved for live trading**. Earlier secondary-audit results did not consistently clear the benchmark hurdle, and the 2026 data is not a fully untouched holdout.

I also tested two additions that **did not earn a place in the main strategy**:

- **Market analogs:** matching recent price, volatility, and drawdown patterns to older market periods, then adding CPI inflation and Treasury yields. Similarity forecasts did not beat a simple historical-average return forecast. A defensive overlay looked interesting in one older broad-market simulation, but failed to improve the stock-selection strategy in the 2026 comparison.
- **Geopolitical news:** a GDELT-based daily risk overlay. It reduced losses relative to full exposure in a short 2022 sample, but did not beat constant 75% exposure.

Those experiments remain in the repository because unsuccessful tests are part of the research.

## Run the evolutionary experiment

Python 3.11 or newer is required.

```bash
python -m pip install -e .
python -c "from quantlab.data import download_stock_data; download_stock_data('2026-01-01')"
python run_evolution.py --population 16 --generations 5 --seed 41
```

The second command builds the local price cache in `data/raw_2018_2025/` using Yahoo Finance via `yfinance`. The repository intentionally excludes downloaded market and news data; Yahoo's adjusted historical data can be revised, so reproducing an older run byte-for-byte is not guaranteed. The championship writes reports under `reports/evolution_championships/`, which is also excluded from version control.

The news and analog experiments require additional cached datasets and are not part of the core championship command.

To run the synthetic-price tests without downloading market data:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

## What still needs work

The greatest weakness is **selection/survivorship bias**: the 33 stocks were chosen with hindsight, and failed or delisted companies are not represented. Other limitations include repeated inspection of historical periods, incomplete point-in-time provenance for news and macroeconomic features, a simplified execution/cost model, and no prospective trading track record.

Next steps are to reconstruct historically eligible stock universes, harden the evaluation protocol, and test new data without changing the rules after seeing results. Until then, I treat the reported returns as exploratory.
