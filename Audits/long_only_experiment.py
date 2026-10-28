
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd

from quantlab.data import download_stock_data
from quantlab.universe import SECTOR_BY_TICKER


# STEP 1: Load data

df = download_stock_data()

df = df.sort_values(
    ["Ticker", "Date"]
).copy()

df["Sector"] = df["Ticker"].map(
    SECTOR_BY_TICKER
)

close = df.groupby("Ticker")["Close"]

# Three preselected candidate signals
df["momentum_5d"] = close.pct_change(
    periods=5, fill_method=None
)

df["momentum_63d"] = close.pct_change(
    periods=63, fill_method=None
)

df["momentum_126_skip_21"] = (
    close.shift(21) / close.shift(126) - 1
)


# STEP 2: Prepare trading dates and prices

opens = df.pivot(
    index="Date",
    columns="Ticker",
    values="Open"
).sort_index()

if opens.isna().any().any():
    raise ValueError("Missing stock opening prices")

calendar = opens.index

first_i = calendar.searchsorted(
    pd.Timestamp("2022-01-01")
)

HOLD_DAYS = 21

# Ensure each selected holding period has an exit
rebalance_indices = list(
    range(first_i, len(calendar) - HOLD_DAYS - 1,
          HOLD_DAYS)
)

if not rebalance_indices:
    raise ValueError("No valid trading periods")


# STEP 3: Portfolio simulator

def simulate_long_only(signal, cost_bps=5.0, return_history=False):

    cash = 1.0
    shares = {}

    cost_rate = cost_bps / 10000
    equity_history = [1.0]

    for signal_i in rebalance_indices:

        signal_date = calendar[signal_i]
        entry_prices = opens.iloc[signal_i + 1]

        day = df[df["Date"] == signal_date]

        if day["Ticker"].nunique() != 33:
            raise ValueError("Incomplete stock universe")

        if signal is None:
            # Benchmark: hold all 33 stocks equally
            selected = list(opens.columns)

        else:
            if day[signal].isna().any():
                raise ValueError("Missing signal values")

            # Buy the highest-ranked stock in each sector
            winners = day.groupby("Sector")[signal].idxmax()

            selected = day.loc[
                winners, "Ticker"
            ].tolist()

            if len(selected) != 11:
                raise ValueError("Expected 11 sector winners")

        # Equity before trading
        equity = cash + sum(
            quantity * entry_prices[ticker]
            for ticker, quantity in shares.items()
        )

        if equity <= 0:
            raise ValueError("Portfolio equity exhausted")

        # Equal-weight selected stocks
        target_shares = {
            ticker: (
                equity / len(selected)
            ) / entry_prices[ticker]
            for ticker in selected
        }

        all_tickers = set(shares) | set(target_shares)

        trades = {
            ticker: (
                target_shares.get(ticker, 0)
                - shares.get(ticker, 0)
            ) * entry_prices[ticker]
            for ticker in all_tickers
        }

        trading_notional = sum(
            abs(value) for value in trades.values()
        )

        transaction_cost = (
            trading_notional * cost_rate
        )

        cash -= sum(trades.values()) + transaction_cost

        shares = target_shares


        # Record portfolio value after each subsequent
        # rebalance, including transaction costs.
        if signal_i != rebalance_indices[0]:

            current_equity = cash + sum(
                quantity * entry_prices[ticker]
                for ticker, quantity in shares.items()
            )

            equity_history.append(current_equity)


    # STEP 4: Liquidate the last portfolio
    final_i = rebalance_indices[-1] + HOLD_DAYS + 1

    final_prices = opens.iloc[final_i]

    liquidation_value = sum(
        quantity * final_prices[ticker]
        for ticker, quantity in shares.items()
    )

    liquidation_cost = (
        liquidation_value * cost_rate
    )

    cash += liquidation_value - liquidation_cost

    if return_history:
        equity_history.append(cash)

        return (
            cash,
            pd.Series(equity_history, dtype=float)
        )

    return cash



# STEP 5: Evaluate candidates against benchmark

strategies = {
    "Equal-weight 33-stock benchmark": None,
    "5-day momentum selection": "momentum_5d",
    "63-day momentum selection": "momentum_63d",
    "126-day skip-21 selection": "momentum_126_skip_21"
}

results = {}

print("\nLONG-ONLY STRATEGY EXPERIMENT")
print(f"Rebalancing intervals: {len(rebalance_indices)}")
print(f"Cost assumption: 5 bps per traded dollar")

for name, signal in strategies.items():

    gross_equity = simulate_long_only(
        signal, cost_bps=0
    )

    net_equity = simulate_long_only(
        signal, cost_bps=5
    )

    results[name] = net_equity - 1

    print(f"\nStrategy: {name}")
    print(f"Gross return: {gross_equity - 1:+.2%}")
    print(f"Net return: {net_equity - 1:+.2%}")


benchmark_return = results[
    "Equal-weight 33-stock benchmark"
]

print("\nPERFORMANCE RELATIVE TO BENCHMARK")

for name, result in results.items():

    if name == "Equal-weight 33-stock benchmark":
        continue

    difference = result - benchmark_return

    print(
        f"{name}: {difference:+.2%} "
        f"cumulative return difference"
    )


# STEP 6: Compare portfolio risk

import numpy as np

print("\nLONG-ONLY RISK COMPARISON")

return_series = {}

for name, signal in strategies.items():

    ending_equity, equity = simulate_long_only(
        signal,
        cost_bps=5.0,
        return_history=True
    )

    assert len(equity) == len(rebalance_indices) + 1

    assert abs(
        equity.iloc[-1] - ending_equity
    ) < 1e-10

    returns = equity.pct_change().dropna()
    returns = returns.reset_index(drop=True)

    return_series[name] = returns

    annual_volatility = (
        returns.std(ddof=1) * np.sqrt(252 / HOLD_DAYS)
    )

    max_drawdown = (
        equity / equity.cummax() - 1
    ).min()

    print(f"\nStrategy: {name}")

    print(f"Observed holding periods: {len(returns)}")

    print(
        f"Annualized return volatility: "
        f"{annual_volatility:.2%}"
    )

    print(
        f"Rebalance-level maximum drawdown: "
        f"{max_drawdown:.2%}"
    )


# STEP 7: Compare momentum directly with benchmark

print("\nMOMENTUM VS BENCHMARK")

benchmark = return_series[
    "Equal-weight 33-stock benchmark"
]

momentum = return_series[
    "5-day momentum selection"
]

active_returns = momentum - benchmark

print(
    f"Average excess return per period: "
    f"{active_returns.mean():+.4%}"
)

print(
    f"Periods outperforming benchmark: "
    f"{(active_returns > 0).mean():.2%}"
)

tracking_error = (
    active_returns.std(ddof=1)
    * np.sqrt(252 / HOLD_DAYS)
)

print(
    f"Annualized tracking error: "
    f"{tracking_error:.2%}"
)


# STEP 8: Paired block bootstrap

rng = np.random.default_rng(2026)

values = active_returns.to_numpy()
n = len(values)

block_length = 3
simulations = 3000

bootstrap_means = []

for _ in range(simulations):

    blocks_needed = (
        n + block_length - 1
    ) // block_length

    starts = rng.integers(
        0,
        n - block_length + 1,
        size=blocks_needed
    )

    sample = np.concatenate([
        values[start:start + block_length]
        for start in starts
    ])[:n]

    bootstrap_means.append(sample.mean())

lower, upper = np.percentile(
    bootstrap_means,
    [2.5, 97.5]
)

print(
    f"Exploratory 95% excess-return interval: "
    f"[{lower:+.4%}, {upper:+.4%}]"
)

if lower <= 0 <= upper:
    print("No clear advantage under this bootstrap")
else:
    print("Interval excludes zero")
