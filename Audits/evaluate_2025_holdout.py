
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd

from quantlab.data import download_stock_data
from quantlab.universe import SECTOR_BY_TICKER


# FROZEN STRATEGY SPECIFICATION

HOLD_DAYS = 21
COST_BPS = 5.0

df = download_stock_data(
    end_date="2026-01-01"
)

df = df.sort_values(
    ["Ticker", "Date"]
).copy()

df["Sector"] = df["Ticker"].map(
    SECTOR_BY_TICKER
)

df["momentum_5d"] = (
    df.groupby("Ticker")["Close"]
    .pct_change(5, fill_method=None)
)

opens = df.pivot(
    index="Date",
    columns="Ticker",
    values="Open"
).sort_index()

if opens.isna().any().any():
    raise ValueError("Missing opening prices")

if df["Sector"].isna().any():
    raise ValueError("Missing sector assignments")

calendar = opens.index


# CORRECTED, SELF-FINANCING REBALANCING

def rebalance(cash, shares, prices, selected, cost_rate):

    old_values = {
        ticker: quantity * float(prices[ticker])
        for ticker, quantity in shares.items()
    }

    equity_before = cash + sum(old_values.values())

    if equity_before <= 0:
        raise ValueError("Portfolio equity exhausted")

    selected = list(selected)
    selected_set = set(selected)
    all_tickers = set(old_values) | selected_set

    def traded_notional(invested_equity):

        target_value = invested_equity / len(selected)

        return sum(
            abs(
                (
                    target_value
                    if ticker in selected_set
                    else 0.0
                ) - old_values.get(ticker, 0.0)
            )
            for ticker in all_tickers
        )

    # Solve:
    # invested_equity + trading_cost = equity_before
    #
    # This prevents borrowing cash to cover fees.

    low = 0.0
    high = equity_before

    for _ in range(70):

        mid = (low + high) / 2

        required_cash = (
            mid + cost_rate * traded_notional(mid)
        )

        if required_cash > equity_before:
            high = mid
        else:
            low = mid

    invested_equity = (low + high) / 2

    transaction_cost = (
        cost_rate * traded_notional(invested_equity)
    )

    new_cash = (
        equity_before
        - invested_equity
        - transaction_cost
    )

    if new_cash < -1e-10:
        raise AssertionError("Negative cash after trade")

    if abs(new_cash) < 1e-10:
        new_cash = 0.0

    target_value = invested_equity / len(selected)

    new_shares = {
        ticker: target_value / float(prices[ticker])
        for ticker in selected
    }

    return new_cash, new_shares, transaction_cost


# RUN ONE STRATEGY ON ONE DATE WINDOW

def simulate(
    start_date,
    end_date,
    signal,
    cost_bps,
    return_daily=False,
    return_attribution=False
):

    start_i = calendar.searchsorted(
        pd.Timestamp(start_date)
    )

    last_i = (
        calendar.searchsorted(
            pd.Timestamp(end_date)
        ) - 1
    )

    # Full 21-session holding periods only.
    signal_indices = list(
        range(
            start_i,
            last_i - HOLD_DAYS,
            HOLD_DAYS
        )
    )

    if not signal_indices:
        raise ValueError("No complete holding periods")

    cash = 1.0
    shares = {}
    rate = cost_bps / 10000
    total_fees = 0.0

    equity_history = [1.0]
    daily_records = []
    attribution_records = []

    for signal_i in signal_indices:

        signal_date = calendar[signal_i]

        day = df.loc[
            df["Date"] == signal_date
        ]

        if day["Ticker"].nunique() != 33:
            raise ValueError("Incomplete universe")

        if signal is None:
            selected = list(opens.columns)

        else:
            if day[signal].isna().any():
                raise ValueError("Missing momentum signal")

            winners = (
                day.groupby("Sector")[signal].idxmax()
            )

            selected = day.loc[
                winners, "Ticker"
            ].tolist()

            if len(selected) != 11:
                raise ValueError("Expected 11 winners")

        entry_i = signal_i + 1
        exit_i = signal_i + HOLD_DAYS + 1

        assert calendar[exit_i] < pd.Timestamp(end_date)

        entry_prices = opens.iloc[entry_i]

        cash, shares, fee = rebalance(
            cash,
            shares,
            entry_prices,
            selected,
            rate
        )

        total_fees += fee

        # Value immediately after paying rebalance costs
        entry_equity = cash + sum(
            quantity * float(entry_prices[ticker])
            for ticker, quantity in shares.items()
        )

        equity_history.append(entry_equity)


        if return_daily:
            for day_i in range(entry_i, exit_i):
                day_prices = opens.iloc[day_i]

                day_equity = cash + sum(
                    quantity * float(day_prices[ticker])
                    for ticker, quantity in shares.items()
                )

                daily_records.append({
                    "Date": calendar[day_i],
                    "Equity": day_equity
                })


        # Value at the next scheduled exit opening
        exit_prices = opens.iloc[exit_i]

        exit_equity = cash + sum(
            quantity * float(exit_prices[ticker])
            for ticker, quantity in shares.items()
        )

        equity_history.append(exit_equity)

        # Attribute holding-period profits to each stock.
        if return_attribution:

            for ticker in selected:

                stock_profit = (
                    shares[ticker]
                    * (
                        float(exit_prices[ticker])
                        - float(entry_prices[ticker])
                    )
                )

                attribution_records.append({
                    "Signal_Date": signal_date,
                    "Ticker": ticker,
                    "Sector": SECTOR_BY_TICKER[ticker],
                    "Gross_PnL": stock_profit
                })


    # Liquidate the final holdings
    final_prices = opens.iloc[
        signal_indices[-1] + HOLD_DAYS + 1
    ]

    liquidation_value = sum(
        quantity * float(final_prices[ticker])
        for ticker, quantity in shares.items()
    )

    liquidation_fee = liquidation_value * rate

    cash += liquidation_value - liquidation_fee
    total_fees += liquidation_fee

    equity_history.append(cash)
    if return_daily:
        daily_records.append({
            "Date": calendar[signal_indices[-1] + HOLD_DAYS + 1],
            "Equity": cash
        })

    equity = pd.Series(equity_history, dtype=float)


    max_drawdown = (
        equity / equity.cummax() - 1
    ).min()

    result = {
        "return": cash - 1.0,
        "max_drawdown": max_drawdown,
        "periods": len(signal_indices),
        "first_trade": calendar[signal_indices[0] + 1],
        "last_trade": calendar[
            signal_indices[-1] + HOLD_DAYS + 1
        ],
        "total_fees": total_fees
    }

    # Option 1: Return portfolio statistics and attribution
    if return_attribution:
        return (
            result,
            pd.DataFrame(attribution_records)
        )

    # Option 2: Return portfolio statistics and daily equity
    if return_daily:
        daily = pd.DataFrame(daily_records)

        daily = daily.sort_values(
            "Date"
        ).reset_index(drop=True)

        assert len(daily) == (
            len(signal_indices) * HOLD_DAYS + 1
        )

        assert daily["Date"].is_unique
        assert daily["Date"].is_monotonic_increasing

        assert abs(
            daily["Equity"].iloc[-1]
            - (1 + result["return"])
        ) < 1e-10

        return result, daily

    # Option 3: Return only the portfolio statistics
    return result
    if return_attribution:
        return result, pd.DataFrame(attribution_records)



# EVALUATE THE FROZEN STRATEGY

def evaluate_window(start_date, end_date, label):

    print(f"\n{label}")

    results = {}

    for name, signal in [
        ("Equal-weight benchmark", None),
        ("5-day momentum", "momentum_5d")
    ]:

        gross = simulate(
            start_date,
            end_date,
            signal,
            cost_bps=0
        )

        net = simulate(
            start_date,
            end_date,
            signal,
            cost_bps=COST_BPS
        )

        results[name] = net["return"]

        print(f"\nStrategy: {name}")
        print(f"Trading periods: {net['periods']}")
        print(f"First entry: {net['first_trade']:%Y-%m-%d}")
        print(f"Final exit: {net['last_trade']:%Y-%m-%d}")
        print(f"Gross return: {gross['return']:+.2%}")
        print(f"Net return: {net['return']:+.2%}")
        print(
            f"Event-level max drawdown: "
            f"{net['max_drawdown']:.2%}"
        )

    excess = (
        results["5-day momentum"]
        - results["Equal-weight benchmark"]
    )

    print(
        f"\nMomentum minus benchmark: "
        f"{excess:+.2%} cumulative"
    )


evaluate_window(
    "2022-01-01",
    "2025-01-01",
    "DEVELOPMENT SAMPLE: 2022-2024"
)

evaluate_window(
    "2025-01-01",
    "2026-01-01",
    "INDEPENDENT EVALUATION: 2025"
)



# STEP 36: Compare 2025 momentum to SPY

import yfinance as yf

print("\n2025 BROAD-MARKET BENCHMARK")

momentum = simulate(
    "2025-01-01",
    "2026-01-01",
    "momentum_5d",
    cost_bps=COST_BPS
)

start = momentum["first_trade"]
end = momentum["last_trade"]

# Download adjusted SPY prices covering
# the exact dates used in our portfolio test.
spy = yf.download(
    "SPY",
    start=start.strftime("%Y-%m-%d"),
    end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    interval="1d",
    auto_adjust=True,
    multi_level_index=False,
    progress=False
)

if spy.empty:
    raise ValueError("SPY download failed")

spy.index = pd.to_datetime(spy.index)
spy.index = spy.index.tz_localize(None).normalize()

if start not in spy.index or end not in spy.index:
    raise ValueError("SPY is missing a required trading date")

start_price = float(spy.loc[start, "Open"])
end_price = float(spy.loc[end, "Open"])

# One purchase and one sale with modeled fees.
rate = COST_BPS / 10000

spy_gross = end_price / start_price - 1

spy_net = (
    (end_price / start_price)
    * (1 - rate)
    / (1 + rate)
    - 1
)

print(f"Evaluation start: {start:%Y-%m-%d}")
print(f"Evaluation end: {end:%Y-%m-%d}")

print(f"\nSPY gross return: {spy_gross:+.2%}")
print(f"SPY net return: {spy_net:+.2%}")

print(
    f"Momentum net return: "
    f"{momentum['return']:+.2%}"
)

print(
    f"Momentum minus SPY: "
    f"{momentum['return'] - spy_net:+.2%}"
)


# STEP 37: 2025 daily risk comparison

import numpy as np

print("\n2025 DAILY RISK COMPARISON")

momentum_stats, momentum_daily = simulate(
    "2025-01-01",
    "2026-01-01",
    "momentum_5d",
    cost_bps=COST_BPS,
    return_daily=True
)

benchmark_stats, benchmark_daily = simulate(
    "2025-01-01",
    "2026-01-01",
    None,
    cost_bps=COST_BPS,
    return_daily=True
)

assert momentum_daily["Date"].equals(
    benchmark_daily["Date"]
)

dates = pd.DatetimeIndex(
    momentum_daily["Date"]
)

# Use the SPY data downloaded in Step 36.
spy_prices = spy["Open"].reindex(dates)

if spy_prices.isna().any():
    raise ValueError("Missing aligned SPY prices")

rate = COST_BPS / 10000

# Model the initial purchase fee and final sale fee.
spy_equity = (
    spy_prices.to_numpy(dtype=float)
    / (float(spy.loc[dates[0], "Open"]) * (1 + rate))
)

spy_equity[-1] *= (1 - rate)

assert abs(
    spy_equity[-1] - (1 + spy_net)
) < 1e-10

risk_data = pd.DataFrame({
    "Momentum": momentum_daily["Equity"].to_numpy(),
    "Equal-weight": benchmark_daily["Equity"].to_numpy(),
    "SPY": spy_equity
}, index=dates)

daily_returns = risk_data.pct_change().dropna()

for name in risk_data.columns:

    annual_vol = (
        daily_returns[name].std(ddof=1)
        * np.sqrt(252)
    )

    # Include the initial $1 before the first trade.
    equity_values = pd.Series(
        np.r_[1.0, risk_data[name].to_numpy()]
    )

    daily_mdd = (
        equity_values / equity_values.cummax() - 1
    ).min()

    print(f"\n{name}")
    print(f"Annualized volatility: {annual_vol:.2%}")
    print(f"Daily-open max drawdown: {daily_mdd:.2%}")


# Market beta relative to SPY
spy_variance = daily_returns["SPY"].var(ddof=1)

momentum_beta = (
    daily_returns["Momentum"].cov(daily_returns["SPY"])
    / spy_variance
)

# Active performance relative to SPY
active = (
    daily_returns["Momentum"]
    - daily_returns["SPY"]
)

tracking_error = (
    active.std(ddof=1) * np.sqrt(252)
)

information_ratio = (
    active.mean() * 252 / tracking_error
    if tracking_error > 0
    else float("nan")
)

print("\nMOMENTUM RELATIVE TO SPY")
print(f"Daily observations: {len(risk_data)}")
print(f"Market beta: {momentum_beta:.3f}")
print(f"Annualized tracking error: {tracking_error:.2%}")
print(f"Information ratio: {information_ratio:+.3f}")

print("\nPASS: Daily valuations reconciled")


# STEP 38: Profit attribution by sector

print("\n2025 PROFIT ATTRIBUTION")

momentum_stats, momentum_holdings = simulate(
    "2025-01-01",
    "2026-01-01",
    "momentum_5d",
    cost_bps=COST_BPS,
    return_attribution=True
)

benchmark_stats, benchmark_holdings = simulate(
    "2025-01-01",
    "2026-01-01",
    None,
    cost_bps=COST_BPS,
    return_attribution=True
)

# Verify stock profits minus transaction costs
# explain the portfolio's total net profit.
for stats, holdings in [
    (momentum_stats, momentum_holdings),
    (benchmark_stats, benchmark_holdings)
]:
    reconstructed_profit = (
        holdings["Gross_PnL"].sum()
        - stats["total_fees"]
    )

    assert abs(
        reconstructed_profit - stats["return"]
    ) < 1e-9


# Aggregate gross profit contributions by sector.
momentum_sectors = (
    momentum_holdings.groupby("Sector")["Gross_PnL"]
    .sum() * 100
)

benchmark_sectors = (
    benchmark_holdings.groupby("Sector")["Gross_PnL"]
    .sum() * 100
)

sector_comparison = pd.DataFrame({
    "Momentum_pp": momentum_sectors,
    "Benchmark_pp": benchmark_sectors
}).fillna(0)

sector_comparison["Difference_pp"] = (
    sector_comparison["Momentum_pp"]
    - sector_comparison["Benchmark_pp"]
)

print("\nGROSS SECTOR CONTRIBUTIONS")
print(
    sector_comparison.sort_values(
        "Difference_pp",
        ascending=False
    ).round(2).to_string()
)


print("\nTOP FIVE MOMENTUM STOCK CONTRIBUTORS")

stock_contributions = (
    momentum_holdings.groupby("Ticker")["Gross_PnL"]
    .sum()
    .sort_values(ascending=False)
    * 100
)

print(stock_contributions.head(5).round(2))


print("\nATTRIBUTION RECONCILIATION")

gross_advantage = (
    sector_comparison["Difference_pp"].sum()
)

extra_costs = (
    momentum_stats["total_fees"]
    - benchmark_stats["total_fees"]
) * 100

net_advantage = (
    momentum_stats["return"]
    - benchmark_stats["return"]
) * 100

assert abs(
    gross_advantage - extra_costs - net_advantage
) < 1e-8

print(f"Gross sector advantage: {gross_advantage:+.2f} pp")
print(f"Additional trading costs: {extra_costs:+.2f} pp")
print(f"Net benchmark advantage: {net_advantage:+.2f} pp")

print("\nPASS: Profit attribution reconciles")


# STEP 39: Trading cost stress test

print("\n2025 TRADING COST STRESS TEST")

cost_scenarios = [0, 5, 10, 15, 20, 30]

for cost in cost_scenarios:

    momentum_result = simulate(
        "2025-01-01",
        "2026-01-01",
        "momentum_5d",
        cost_bps=cost
    )

    benchmark_result = simulate(
        "2025-01-01",
        "2026-01-01",
        None,
        cost_bps=cost
    )

    momentum_return = momentum_result["return"]
    benchmark_return = benchmark_result["return"]

    advantage = (
        momentum_return - benchmark_return
    )

    print(f"\nCost: {cost} bps per traded dollar")

    print(
        f"Momentum net return: "
        f"{momentum_return:+.2%}"
    )

    print(
        f"Benchmark net return: "
        f"{benchmark_return:+.2%}"
    )

    print(
        f"Momentum advantage: "
        f"{advantage:+.2%}"
    )

    print(
        f"Total momentum fees / initial capital: "
        f"{momentum_result['total_fees']:.2%}"
    )
