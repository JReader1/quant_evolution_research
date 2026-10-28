import os

# Limit numerical-library thread usage
# Must be set before importing NumPy or pandas
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import pandas as pd

from quantlab.data import download_stock_data

from quantlab.features import (
    calculate_daily_returns,
    calculate_five_day_features,
    calculate_additional_features,
    calculate_execution_targets
)

from quantlab.universe import SECTOR_BY_TICKER

from quantlab.model import (
    train_first_model,
    walk_forward_model,
    FEATURES
)


from quantlab.backtest import (
    prepare_trade_data,
    evaluate_portfolios,
    measure_position_persistence,
    backtest_with_shares
)


# Load data and construct predictors
df = download_stock_data()
df = calculate_daily_returns(df)
df = calculate_five_day_features(df)
df = calculate_additional_features(df)
df = calculate_execution_targets(df)

df["Sector"] = df["Ticker"].map(SECTOR_BY_TICKER)

if df["Sector"].isna().any():
    raise ValueError("Missing sector classifications")


# Generate model predictions
static_model, _, _ = train_first_model(df)

walk = walk_forward_model(df)

walk["static_prediction"] = static_model.predict(
    walk[FEATURES]
)


# Prepare next-open execution returns
panel = prepare_trade_data(df, walk)


# STEP 24: Verify prediction-target alignment

print("\nEXECUTION TARGET VALIDATION")

audit = panel.dropna(
    subset=[
        "future_open_return_5d",
        "trade_return"
    ]
).copy()

difference = (
    audit["future_open_return_5d"]
    - audit["trade_return"]
).abs()

max_error = difference.max()

print(f"Compared observations: {len(audit)}")
print(f"Maximum difference: {max_error:.12f}")

assert len(audit) > 0
assert max_error < 1e-10

print("PASS: Prediction target matches trade-return calculation")


signals = [
    "model_prediction",
    "static_prediction",
    "momentum_5d"
]

trades = evaluate_portfolios(
    panel,
    signal_columns=signals,
    cost_bps=5.0
)


# Compare results
print("\nFIRST LONG-SHORT BACKTEST")

for signal, group in trades.groupby("Signal"):

    group = group.sort_values("Date")

    gross_total = (
        (1 + group["Gross_Return"]).prod() - 1
    )

    net_total = (
        (1 + group["Net_Return"]).prod() - 1
    )

    print(f"\nStrategy: {signal}")

    print(f"Trading periods: {len(group)}")

    print(
        f"Average gross 5-day return: "
        f"{group['Gross_Return'].mean():.4%}"
    )

    print(
        f"Average net 5-day return: "
        f"{group['Net_Return'].mean():.4%}"
    )

    print(
        f"Compounded gross return: {gross_total:.2%}"
    )

    print(
        f"Compounded net return: {net_total:.2%}"
    )


# STEP 18: Transaction-cost sensitivity

print("\nTRANSACTION COST SENSITIVITY")

cost_scenarios = [0, 1, 2, 3, 5, 10]

for signal in signals:

    strategy = trades[
        trades["Signal"] == signal
    ].sort_values("Date")

    gross_returns = strategy["Gross_Return"]

    print(f"\nStrategy: {signal}")

    for cost_bps in cost_scenarios:

        # Cost on entry and exit
        round_trip_cost = 2 * cost_bps / 10000

        adjusted_returns = (
            gross_returns - round_trip_cost
        )

        compounded_return = (
            (1 + adjusted_returns).prod() - 1
        )

        print(
            f"{cost_bps:2d} bps per side: "
            f"{compounded_return:+.2%}"
        )


# STEP 19: Portfolio drawdown analysis

print("\nPORTFOLIO DRAWDOWN ANALYSIS")

for signal in signals:

    strategy = trades[
        trades["Signal"] == signal
    ].sort_values("Date")

    net_returns = strategy["Net_Return"]

    # Growth of $1 of starting portfolio equity
    equity_curve = (1 + net_returns).cumprod()

    # Highest portfolio value reached so far
    running_peak = equity_curve.cummax().clip(
        lower=1.0
    )

    # Decline from that previous peak
    drawdown = equity_curve / running_peak - 1

    maximum_drawdown = drawdown.min()

    print(f"\nStrategy: {signal}")

    print(
        f"Maximum drawdown: "
        f"{maximum_drawdown:.2%}"
    )

    print(
        f"Profitable periods: "
        f"{(net_returns > 0).mean():.2%}"
    )


# STEP 20: Examine portfolio position persistence

print("\nPORTFOLIO POSITION PERSISTENCE")

# Compare only dates used in the original backtest
valid_dates = trades["Date"].unique()

valid_panel = panel[
    panel["Date"].isin(valid_dates)
].copy()

for signal in signals:

    persistence = measure_position_persistence(
        valid_panel,
        signal
    )

    print(f"\nStrategy: {signal}")

    print(
        f"Compared rebalances: {len(persistence)}"
    )

    print(
        f"Average positions retained: "
        f"{persistence['Retained_Positions'].mean():.2f} / 22"
    )

    print(
        f"Average retention rate: "
        f"{persistence['Retention_Rate'].mean():.2%}"
    )

    print(
        f"Average target turnover: "
        f"{persistence['Target_Turnover'].mean():.2%}"
    )


# STEP 21: Audit the holding-period schedule

print("\nHOLDING-PERIOD CONTINUITY")

trading_dates = pd.DatetimeIndex(
    sorted(pd.to_datetime(df["Date"].unique()))
)

date_position = {
    date: i
    for i, date in enumerate(trading_dates)
}

used_dates = pd.DatetimeIndex(
    sorted(pd.to_datetime(trades["Date"].unique()))
)

intervals = [
    date_position[current] - date_position[previous]
    for previous, current in zip(
        used_dates[:-1],
        used_dates[1:]
    )
]

intervals = pd.Series(intervals)

print("Trading sessions between signals:")
print(intervals.value_counts().sort_index())

print(
    "Continuous rebalances:",
    (intervals == 5).sum()
)

print(
    "Gapped rebalances:",
    (intervals > 5).sum()
)


# STEP 22: Preliminary turnover-aware cost estimate

print("\nTURNOVER-AWARE COST ESTIMATES")

cost_bps = 5.0

for signal in signals:

    strategy = trades[
        trades["Signal"] == signal
    ].sort_values("Date").copy()

    persistence = measure_position_persistence(
        valid_panel,
        signal
    )

    turnover_lookup = persistence.set_index(
        "Date"
    )["Target_Turnover"]

    dates = list(pd.to_datetime(strategy["Date"]))

    traded_notional = []

    for i, date in enumerate(dates):

        if i == 0:
            # Opening the first portfolio
            notional = 1.0

        elif (
            date_position[date]
            - date_position[dates[i - 1]] == 5
        ):
            # Continuous rebalance:
            # trade only the change in target weights
            notional = turnover_lookup.loc[date]

        else:
            # A gap requires closing old positions
            # and opening a new portfolio
            notional = 2.0

        traded_notional.append(notional)

    # Close the last portfolio
    traded_notional[-1] += 1.0

    strategy["traded_notional"] = traded_notional

    strategy["estimated_cost"] = (
        strategy["traded_notional"]
        * cost_bps / 10000
    )

    strategy["proxy_net_return"] = (
        strategy["Gross_Return"]
        - strategy["estimated_cost"]
    )

    compounded_return = (
        (1 + strategy["proxy_net_return"]).prod() - 1
    )

    print(f"\nStrategy: {signal}")

    print(
        f"Average traded notional: "
        f"{strategy['traded_notional'].mean():.2%}"
    )

    print(
        f"Average estimated cost per period: "
        f"{strategy['estimated_cost'].mean():.4%}"
    )

    print(
        f"Compounded proxy net return: "
        f"{compounded_return:+.2%}"
    )




# STEP 23: Share-based portfolio accounting

print("\nSHARE-BASED BACKTEST RESULTS")

for signal in signals:

    gross = backtest_with_shares(
        df,
        valid_panel,
        signal,
        cost_bps=0
    )

    net = backtest_with_shares(
        df,
        valid_panel,
        signal,
        cost_bps=5
    )

    # Include starting equity in drawdown calculation
    equity = pd.concat([
        pd.Series([1.0]),
        net["Ending_Equity"]
    ], ignore_index=True)

    drawdown = equity / equity.cummax() - 1

    # Trading costs expressed in units of starting capital
    total_cost = net["Trading_Cost"].sum()

    print(f"\nStrategy: {signal}")

    print(f"Portfolio periods: {len(net)}")

    print(
        f"Gross compounded return: "
        f"{gross['Ending_Equity'].iloc[-1] - 1:+.2%}"
    )

    print(
        f"Net compounded return: "
        f"{net['Ending_Equity'].iloc[-1] - 1:+.2%}"
    )

    print(
        f"Maximum net drawdown: "
        f"{drawdown.min():.2%}"
    )

    print(
        f"Total transaction costs / initial equity: "
        f"{total_cost:.2%}"
    )

    print(
        f"Average traded notional / starting equity: "
        f"{(net['Traded_Notional'] / net['Starting_Equity']).mean():.2%}"
    )


# STEP 25: Performance by calendar year

print("\nYEAR-BY-YEAR PORTFOLIO PERFORMANCE")

for signal in signals:

    gross = backtest_with_shares(
        df, valid_panel, signal, cost_bps=0
    )

    net = backtest_with_shares(
        df, valid_panel, signal, cost_bps=5
    )

    gross["Year"] = pd.to_datetime(
        gross["Entry_Date"]
    ).dt.year

    net["Year"] = pd.to_datetime(
        net["Entry_Date"]
    ).dt.year

    print(f"\nStrategy: {signal}")

    for year in sorted(net["Year"].unique()):

        year_gross = gross[
            gross["Year"] == year
        ]["Period_Return"]

        year_net = net[
            net["Year"] == year
        ]["Period_Return"]

        gross_return = (
            (1 + year_gross).prod() - 1
        )

        net_return = (
            (1 + year_net).prod() - 1
        )

        print(
            f"{year}: "
            f"{len(year_net)} periods | "
            f"Gross: {gross_return:+.2%} | "
            f"Net: {net_return:+.2%}"
        )

    # Confirm individual periods compound
    # to the same final equity as the backtester

    compounded = (
        (1 + net["Period_Return"]).prod() - 1
    )

    ending_return = (
        net["Ending_Equity"].iloc[-1] - 1
    )

    assert abs(
        compounded - ending_return
    ) < 1e-9

    print("PASS: Portfolio compounding reconciles")


# STEP 26: Risk-adjusted portfolio performance

from math import sqrt

print("\nRISK-ADJUSTED PERFORMANCE")

PERIODS_PER_YEAR = 252 / 5

for signal in signals:

    gross = backtest_with_shares(
        df,
        valid_panel,
        signal,
        cost_bps=0
    )

    net = backtest_with_shares(
        df,
        valid_panel,
        signal,
        cost_bps=5
    )

    print(f"\nStrategy: {signal}")

    for label, results in [
        ("Gross", gross),
        ("Net", net)
    ]:

        returns = results["Period_Return"]

        average_return = returns.mean()

        volatility = returns.std(ddof=1)

        annual_volatility = (
            volatility * sqrt(PERIODS_PER_YEAR)
        )

        if volatility > 0:
            sharpe = (
                average_return
                / volatility
                * sqrt(PERIODS_PER_YEAR)
            )
        else:
            sharpe = float("nan")

        annualized_return = (
            (1 + returns).prod()
            ** (PERIODS_PER_YEAR / len(returns))
            - 1
        )

        print(f"\n{label} performance:")

        print(
            f"Average 5-day return: "
            f"{average_return:+.4%}"
        )

        print(
            f"5-day volatility: "
            f"{volatility:.4%}"
        )

        print(
            f"Annualized volatility: "
            f"{annual_volatility:.2%}"
        )

        print(
            f"Annualized compounded return: "
            f"{annualized_return:+.2%}"
        )

        print(
            f"Annualized Sharpe ratio: "
            f"{sharpe:+.3f}"
        )


# STEP 27: Bootstrap uncertainty analysis

import numpy as np

print("\nBOOTSTRAP RETURN ANALYSIS")

rng = np.random.default_rng(2026)

BOOTSTRAPS = 2000
BLOCK_LENGTH = 5


def bootstrap_mean_interval(returns):

    values = np.asarray(returns, dtype=float)

    n = len(values)

    if n < BLOCK_LENGTH:
        raise ValueError("Not enough observations")

    averages = []

    blocks_needed = (
        n + BLOCK_LENGTH - 1
    ) // BLOCK_LENGTH

    for _ in range(BOOTSTRAPS):

        starts = rng.integers(
            0,
            n - BLOCK_LENGTH + 1,
            size=blocks_needed
        )

        sample = np.concatenate([
            values[start:start + BLOCK_LENGTH]
            for start in starts
        ])[:n]

        averages.append(sample.mean())

    return np.percentile(
        averages, [2.5, 97.5]
    )


for signal in signals:

    gross = backtest_with_shares(
        df, valid_panel, signal, cost_bps=0
    )

    net = backtest_with_shares(
        df, valid_panel, signal, cost_bps=5
    )

    print(f"\nStrategy: {signal}")

    for label, results in [
        ("Gross", gross),
        ("Net", net)
    ]:

        returns = results["Period_Return"]

        lower, upper = bootstrap_mean_interval(
            returns
        )

        # 1 basis point = 0.0001
        mean_bps = returns.mean() * 10000

        lower_bps = lower * 10000
        upper_bps = upper * 10000

        print(f"\n{label} returns:")

        print(
            f"Average 5-day return: "
            f"{mean_bps:+.2f} bps"
        )

        print(
            f"Exploratory 95% interval: "
            f"[{lower_bps:+.2f}, "
            f"{upper_bps:+.2f}] bps"
        )

        if lower <= 0 <= upper:
            print("Interval includes zero")
        else:
            print("Interval excludes zero")


# STEP 28: Paired strategy comparison

print("\nPAIRED STRATEGY COMPARISON")

strategy_returns = {}

for signal in signals:

    result = backtest_with_shares(
        df,
        valid_panel,
        signal,
        cost_bps=5
    )

    # Align each strategy by the same signal date
    strategy_returns[signal] = (
        result.set_index("Date")["Period_Return"]
        .sort_index()
    )

paired = pd.concat(
    strategy_returns,
    axis=1
).dropna()

print(f"Matched trading periods: {len(paired)}")

assert len(paired) == 150


def paired_bootstrap_interval(
    differences,
    block_length=5,
    simulations=3000
):

    values = np.asarray(differences, dtype=float)

    n = len(values)

    rng = np.random.default_rng(2026)

    blocks_needed = (
        n + block_length - 1
    ) // block_length

    averages = []

    for _ in range(simulations):

        starts = rng.integers(
            0,
            n - block_length + 1,
            size=blocks_needed
        )

        sample = np.concatenate([
            values[start:start + block_length]
            for start in starts
        ])[:n]

        averages.append(sample.mean())

    return np.percentile(
        averages,
        [2.5, 97.5]
    )


comparisons = [
    ("momentum_5d", "model_prediction"),
    ("momentum_5d", "static_prediction"),
    ("model_prediction", "static_prediction")
]

for strategy_a, strategy_b in comparisons:

    differences = (
        paired[strategy_a] - paired[strategy_b]
    )

    lower, upper = paired_bootstrap_interval(
        differences
    )

    print(f"\n{strategy_a} vs {strategy_b}")

    print(
        f"Average advantage: "
        f"{differences.mean() * 10000:+.2f} bps"
    )

    print(
        f"95% bootstrap interval: "
        f"[{lower * 10000:+.2f}, "
        f"{upper * 10000:+.2f}] bps"
    )

    if lower > 0:
        print("Evidence favors the first strategy")

    elif upper < 0:
        print("Evidence favors the second strategy")

    else:
        print("No clear advantage under this test")


# STEP 29: Daily marked-to-market risk analysis

print("\nDAILY PORTFOLIO RISK")

for signal in signals:

    periods, daily = backtest_with_shares(
        df,
        valid_panel,
        signal,
        cost_bps=5,
        return_daily=True
    )

    daily = daily.sort_values("Date").reset_index(
        drop=True
    )

    # Verify the daily records form one continuous series
    assert daily["Date"].is_monotonic_increasing
    assert not daily["Date"].duplicated().any()

    # Final daily equity must match final backtest equity
    assert abs(
        daily["Equity"].iloc[-1]
        - periods["Ending_Equity"].iloc[-1]
    ) < 1e-10

    # Include starting portfolio equity of $1
    equity = pd.concat([
        pd.Series([1.0]),
        daily["Equity"]
    ], ignore_index=True)

    running_peak = equity.cummax()

    drawdown = equity / running_peak - 1

    daily_max_drawdown = drawdown.min()

    worst_position = drawdown.iloc[1:].idxmin() - 1
    worst_date = daily.iloc[worst_position]["Date"]

    print(f"\nStrategy: {signal}")

    print(
        f"Recorded trading days: {len(daily)}"
    )

    print(
        f"Daily-open maximum drawdown: "
        f"{daily_max_drawdown:.2%}"
    )

    print(
        f"Worst drawdown date: "
        f"{worst_date:%Y-%m-%d}"
    )

    print(
        f"Final portfolio equity: "
        f"${daily['Equity'].iloc[-1]:.4f}"
    )

    print("PASS: Daily equity reconciles")


# STEP 30: Rebalance-event drawdown audit

print("\nREBALANCE-EVENT DRAWDOWN AUDIT")

for signal in signals:

    periods, daily = backtest_with_shares(
        df,
        valid_panel,
        signal,
        cost_bps=5,
        return_daily=True
    )

    # Portfolio value at the end of each holding period,
    # before a possible new rebalance.
    exit_events = periods[
        ["Exit_Date", "Ending_Equity"]
    ].rename(columns={
        "Exit_Date": "Date",
        "Ending_Equity": "Equity"
    }).copy()

    exit_events["Date"] = pd.to_datetime(
        exit_events["Date"]
    )
    exit_events["Event_Order"] = 0

    # Daily opening-price valuations, after trades.
    daily_events = daily[
        ["Date", "Equity"]
    ].copy()

    daily_events["Date"] = pd.to_datetime(
        daily_events["Date"]
    )
    daily_events["Event_Order"] = 1

    # On shared dates, the holding period ends
    # before the new portfolio is rebalanced.
    events = pd.concat(
        [exit_events, daily_events],
        ignore_index=True
    ).sort_values(
        ["Date", "Event_Order"],
        kind="stable"
    ).reset_index(drop=True)

    def calculate_drawdown(equity_values):

        values = pd.concat([
            pd.Series([1.0]),
            equity_values.reset_index(drop=True)
        ], ignore_index=True)

        return values / values.cummax() - 1

    endpoint_mdd = calculate_drawdown(
        periods["Ending_Equity"]
    ).min()

    daily_mdd = calculate_drawdown(
        daily["Equity"]
    ).min()

    event_drawdowns = calculate_drawdown(
        events["Equity"]
    )

    combined_mdd = event_drawdowns.min()

    # Including all observations cannot make
    # the historical maximum drawdown smaller.
    assert combined_mdd <= endpoint_mdd + 1e-10
    assert combined_mdd <= daily_mdd + 1e-10

    # Ensure the final values still reconcile.
    assert abs(
        events["Equity"].iloc[-1]
        - periods["Ending_Equity"].iloc[-1]
    ) < 1e-10

    worst_idx = event_drawdowns.iloc[1:].idxmin() - 1
    worst_date = events.iloc[worst_idx]["Date"]

    print(f"\nStrategy: {signal}")
    print(f"Endpoint drawdown: {endpoint_mdd:.2%}")
    print(f"Daily-open drawdown: {daily_mdd:.2%}")
    print(f"Combined event drawdown: {combined_mdd:.2%}")
    print(f"Worst date: {worst_date:%Y-%m-%d}")
    print("PASS: All drawdown observations reconciled")



# STEP 31: Share-based transaction cost sensitivity

print("\nSHARE-BASED COST SENSITIVITY")

cost_scenarios = [0, 1, 2, 3, 4, 5, 7.5, 10]

for cost in cost_scenarios:

    results = backtest_with_shares(
        df,
        valid_panel,
        signal="momentum_5d",
        cost_bps=cost
    )

    final_equity = results[
        "Ending_Equity"
    ].iloc[-1]

    total_return = final_equity - 1

    print(
        f"{cost:4.1f} bps per traded dollar: "
        f"{total_return:+.2%}"
    )
