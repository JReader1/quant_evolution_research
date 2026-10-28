
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from math import sqrt

from quantlab.data import download_stock_data

from quantlab.features import (
    calculate_daily_returns,
    calculate_five_day_features,
    calculate_additional_features,
    calculate_execution_targets
)

from quantlab.universe import SECTOR_BY_TICKER
from quantlab.model import walk_forward_model

from quantlab.backtest import (
    prepare_trade_data,
    evaluate_portfolios,
    backtest_with_shares
)


# STEP 1: Prepare historical stock data

df = download_stock_data()

df = calculate_daily_returns(df)
df = calculate_five_day_features(df)
df = calculate_additional_features(df)
df = calculate_execution_targets(df)

df["Sector"] = df["Ticker"].map(SECTOR_BY_TICKER)


# STEP 2: Create longer-term momentum signals

df = df.sort_values(
    ["Ticker", "Date"]
).copy()

close = df.groupby("Ticker")["Close"]

# Return over approximately three months
df["momentum_63d"] = close.pct_change(
    periods=63,
    fill_method=None
)

# Return from 126 days ago through 21 days ago
# Skips approximately the most recent month
df["momentum_126_skip_21"] = (
    close.shift(21) / close.shift(126) - 1
)


# STEP 3: Use our existing evaluation calendar

walk = walk_forward_model(df)

panel = prepare_trade_data(df, walk)

signals = [
    "momentum_5d",
    "momentum_63d",
    "momentum_126_skip_21"
]

trades = evaluate_portfolios(
    panel,
    signal_columns=signals,
    cost_bps=5
)

valid_panel = panel[
    panel["Date"].isin(trades["Date"].unique())
].copy()


# STEP 4: Compare gross and net performance

print("\nMOMENTUM HORIZON EXPERIMENT")

for signal in signals:

    gross = backtest_with_shares(
        df, valid_panel, signal, cost_bps=0
    )

    net = backtest_with_shares(
        df, valid_panel, signal, cost_bps=5
    )

    gross_return = (
        gross["Ending_Equity"].iloc[-1] - 1
    )

    net_return = (
        net["Ending_Equity"].iloc[-1] - 1
    )

    returns = net["Period_Return"]

    volatility = returns.std(ddof=1)

    net_sharpe = (
        returns.mean()
        / volatility
        * sqrt(252 / 5)
        if volatility > 0
        else float("nan")
    )

    turnover = (
        net["Traded_Notional"]
        / net["Starting_Equity"]
    ).mean()

    print(f"\nStrategy: {signal}")
    print(f"Trading periods: {len(net)}")
    print(f"Gross return: {gross_return:+.2%}")
    print(f"Net return: {net_return:+.2%}")
    print(f"Net Sharpe: {net_sharpe:+.3f}")
    print(f"Average traded notional: {turnover:.2%}")
