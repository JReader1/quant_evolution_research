
import os

# Reduce numerical-library thread usage
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt

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
    backtest_with_shares
)


# 1. Set up the output folder
ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


# 2. Prepare the data
df = download_stock_data()
df = calculate_daily_returns(df)
df = calculate_five_day_features(df)
df = calculate_additional_features(df)
df = calculate_execution_targets(df)

df["Sector"] = df["Ticker"].map(SECTOR_BY_TICKER)

if df["Sector"].isna().any():
    raise ValueError("Missing sector classification")


# 3. Generate predictions
static_model, _, _ = train_first_model(df)

walk = walk_forward_model(df)

walk["static_prediction"] = static_model.predict(
    walk[FEATURES]
)


# 4. Prepare a common evaluation sample
panel = prepare_trade_data(df, walk)

signals = [
    "momentum_5d",
    "model_prediction",
    "static_prediction"
]

trades = evaluate_portfolios(
    panel,
    signal_columns=signals,
    cost_bps=5
)

valid_panel = panel[
    panel["Date"].isin(trades["Date"].unique())
].copy()


# 5. Evaluate each strategy once per cost scenario
summary_rows = []
equity_curves = {}

for signal in signals:

    gross = backtest_with_shares(
        df, valid_panel, signal, cost_bps=0
    )

    net, daily = backtest_with_shares(
        df,
        valid_panel,
        signal,
        cost_bps=5,
        return_daily=True
    )

    assert len(gross) == len(net)

    assert abs(
        daily["Equity"].iloc[-1]
        - net["Ending_Equity"].iloc[-1]
    ) < 1e-10

    gross_return = gross["Ending_Equity"].iloc[-1] - 1
    net_return = net["Ending_Equity"].iloc[-1] - 1

    period_returns = net["Period_Return"]
    period_volatility = period_returns.std(ddof=1)

    sharpe = (
        period_returns.mean()
        / period_volatility
        * sqrt(252 / 5)
        if period_volatility > 0
        else float("nan")
    )

    # Include both daily and rebalance-end valuations
    exit_events = net[
        ["Exit_Date", "Ending_Equity"]
    ].rename(columns={
        "Exit_Date": "Date",
        "Ending_Equity": "Equity"
    }).copy()

    exit_events["Event_Order"] = 0

    daily_events = daily[
        ["Date", "Equity"]
    ].copy()

    daily_events["Event_Order"] = 1

    events = pd.concat(
        [exit_events, daily_events],
        ignore_index=True
    ).sort_values(
        ["Date", "Event_Order"],
        kind="stable"
    )

    equity_values = pd.concat([
        pd.Series([1.0]),
        events["Equity"].reset_index(drop=True)
    ], ignore_index=True)

    max_drawdown = (
        equity_values / equity_values.cummax() - 1
    ).min()

    summary_rows.append({
        "Strategy": signal,
        "Periods": len(net),
        "Gross_Return_Pct": gross_return * 100,
        "Net_Return_Pct": net_return * 100,
        "Net_Sharpe": sharpe,
        "Max_Drawdown_Pct": max_drawdown * 100
    })

    equity_curves[signal] = (
        daily.set_index("Date")["Equity"]
    )


# 6. Save the comparison table
summary = pd.DataFrame(summary_rows)

summary.to_csv(
    REPORTS / "strategy_summary.csv",
    index=False
)

print("\nRESEARCH SUMMARY")
print(summary.round(3).to_string(index=False))


# 7. Save the daily equity curves
equity_table = pd.concat(
    equity_curves,
    axis=1
).sort_index()

equity_table.index.name = "Date"

equity_table.to_csv(
    REPORTS / "equity_curves.csv"
)


# 8. Create a portfolio comparison chart
fig, ax = plt.subplots(figsize=(11, 6))

for signal in signals:
    ax.plot(
        equity_table.index,
        equity_table[signal],
        label=signal
    )

ax.axhline(1.0, linestyle="--", linewidth=1)

ax.set_title(
    "Long-Short Strategy Performance (2022-2024)"
)

ax.set_ylabel("Portfolio equity (starting at $1)")
ax.set_xlabel("Date")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()

fig.savefig(
    REPORTS / "equity_curves.png",
    dpi=180
)

plt.close(fig)

print("\nResearch files saved to reports/")
