
import pandas as pd


def prepare_trade_data(prices, predictions):

    prices = prices.sort_values(
        ["Ticker", "Date"]
    ).copy()

    grouped = prices.groupby("Ticker")

    # Signal is known after today's close.
    # Enter at the next trading day's open.
    prices["entry_open"] = grouped["Open"].shift(-1)

    # Exit five trading sessions after entry.
    prices["exit_open"] = grouped["Open"].shift(-6)

    prices["trade_return"] = (
        prices["exit_open"] / prices["entry_open"] - 1
    )

    trade_prices = prices[
        ["Date", "Ticker", "trade_return"]
    ]

    panel = predictions.merge(
        trade_prices,
        on=["Date", "Ticker"],
        how="left",
        validate="one_to_one"
    )

    # Rebalance once every five trading days.
    calendar = (
        prices.loc[
            (prices["Date"] >= "2022-01-01")
            & (prices["Date"] < "2025-01-01"),
            "Date"
        ]
        .drop_duplicates()
        .sort_values()
    )

    rebalance_dates = calendar.iloc[::5]

    return panel[
        panel["Date"].isin(rebalance_dates)
    ].copy()


def evaluate_portfolios(
    panel,
    signal_columns,
    cost_bps=5.0
):

    rows = []

    for date, day in panel.groupby("Date"):

        # Require our full 33-stock universe.
        if day["Ticker"].nunique() != 33:
            continue

        if day["Sector"].nunique() != 11:
            continue

        if not (
            day.groupby("Sector")["Ticker"]
            .nunique().eq(3).all()
        ):
            continue

        required = signal_columns + ["trade_return"]

        if day[required].isna().any().any():
            continue

        for signal in signal_columns:

            long_returns = []
            short_returns = []

            for _, sector in day.groupby("Sector"):

                # Buy highest-ranked stock in sector.
                long_stock = sector.nlargest(
                    1, signal
                )

                # Short lowest-ranked stock in sector.
                short_stock = sector.nsmallest(
                    1, signal
                )

                long_returns.append(
                    long_stock["trade_return"].iloc[0]
                )

                short_returns.append(
                    short_stock["trade_return"].iloc[0]
                )

            # 50% long, 50% short.
            gross_return = 0.5 * (
                sum(long_returns) / 11
                - sum(short_returns) / 11
            )

            # Approximate opening and closing costs
            # on 100% gross portfolio notional.
            trading_cost = 2 * cost_bps / 10000

            net_return = gross_return - trading_cost

            rows.append({
                "Date": date,
                "Signal": signal,
                "Gross_Return": gross_return,
                "Net_Return": net_return
            })

    return pd.DataFrame(rows)

def measure_position_persistence(panel, signal):

    previous_weights = None
    records = []

    for date, day in panel.groupby("Date", sort=True):

        if day["Ticker"].nunique() != 33:
            continue

        if day["Sector"].nunique() != 11:
            continue

        if not day.groupby("Sector")["Ticker"].nunique().eq(3).all():
            continue

        if day[[signal, "trade_return"]].isna().any().any():
            continue

        weights = {}

        for _, sector in day.groupby("Sector"):

            long_ticker = sector.loc[
                sector[signal].idxmax(), "Ticker"
            ]

            short_ticker = sector.loc[
                sector[signal].idxmin(), "Ticker"
            ]

            weights[long_ticker] = 1 / 22
            weights[short_ticker] = -1 / 22

        if previous_weights is not None:

            # Count positions with unchanged tickers and sides
            retained = sum(
                previous_weights.get(ticker) == weight
                for ticker, weight in weights.items()
            )

            # Measure changes in desired portfolio weights
            all_tickers = (
                set(weights) | set(previous_weights)
            )

            turnover = sum(
                abs(
                    weights.get(ticker, 0)
                    - previous_weights.get(ticker, 0)
                )
                for ticker in all_tickers
            )

            records.append({
                "Date": date,
                "Retained_Positions": retained,
                "Retention_Rate": retained / 22,
                "Target_Turnover": turnover
            })

        previous_weights = weights

    return pd.DataFrame(records)

def backtest_with_shares(
    prices: pd.DataFrame,
    panel: pd.DataFrame,
    signal: str,
    cost_bps: float = 5.0,
    return_daily: bool = False
):

    # Opening prices for every stock and trading date
    opens = prices.pivot(
        index="Date",
        columns="Ticker",
        values="Open"
    ).sort_index()

    if opens.isna().any().any():
        raise ValueError("Missing opening prices")

    calendar = opens.index

    date_index = {
        date: i
        for i, date in enumerate(calendar)
    }

    # Build portfolio instructions before trading
    plans = []

    for date, day in panel.groupby("Date", sort=True):

        counts = day.groupby("Sector")["Ticker"].nunique()

        if (
            day["Ticker"].nunique() != 33
            or len(counts) != 11
            or not counts.eq(3).all()
            or day[signal].isna().any()
        ):
            continue

        i = date_index[date]

        if i + 6 >= len(calendar):
            continue

        weights = {}

        for _, sector in day.groupby("Sector"):

            long_ticker = sector.loc[
                sector[signal].idxmax(), "Ticker"
            ]

            short_ticker = sector.loc[
                sector[signal].idxmin(), "Ticker"
            ]

            if long_ticker == short_ticker:
                raise ValueError("Tied long and short selection")

            weights[long_ticker] = 1 / 22
            weights[short_ticker] = -1 / 22

        plans.append({
            "signal_date": date,
            "entry_i": i + 1,
            "exit_i": i + 6,
            "weights": weights
        })

    if not plans:
        raise ValueError("No valid portfolio periods")

    # Start with $1 of simulated portfolio equity
    cash = 1.0
    shares = {}
    cost_rate = cost_bps / 10000

    # Store completed holding periods
    records = []

    # Store daily portfolio values
    daily_records = []

    for k, plan in enumerate(plans):

        entry_i = plan["entry_i"]
        exit_i = plan["exit_i"]

        if k > 0 and entry_i < plans[k - 1]["exit_i"]:
            raise ValueError("Overlapping portfolio periods")

        entry_prices = opens.iloc[entry_i]

        # Cash plus current marked-to-market holdings
        starting_equity = cash + sum(
            quantity * entry_prices[ticker]
            for ticker, quantity in shares.items()
        )

        if starting_equity <= 0:
            raise ValueError("Portfolio equity exhausted")

        # Target share quantities at today's opening prices
        target_shares = {
            ticker: weight * starting_equity / entry_prices[ticker]
            for ticker, weight in plan["weights"].items()
        }

        all_tickers = set(shares) | set(target_shares)

        # Shares bought or sold to reach new targets
        trade_values = {
            ticker: (
                target_shares.get(ticker, 0)
                - shares.get(ticker, 0)
            ) * entry_prices[ticker]
            for ticker in all_tickers
        }

        entry_notional = sum(
            abs(value) for value in trade_values.values()
        )

        entry_cost = entry_notional * cost_rate

        cash -= sum(trade_values.values()) + entry_cost
        shares = target_shares


        # Record portfolio equity at each opening price
        # up to, but not including, the exit date.
        for day_i in range(entry_i, exit_i):

            day_prices = opens.iloc[day_i]

            day_equity = cash + sum(
                quantity * day_prices[ticker]
                for ticker, quantity in shares.items()
            )

            daily_records.append({
                "Date": calendar[day_i],
                "Equity": day_equity
            })

        # Mark portfolio at the planned exit open
        exit_prices = opens.iloc[exit_i]

        exit_equity = cash + sum(
            quantity * exit_prices[ticker]
            for ticker, quantity in shares.items()
        )

        # If next entry is later, close positions now.
        # Also liquidate at the end of the entire test.
        close_positions = (
            k == len(plans) - 1
            or plans[k + 1]["entry_i"] > exit_i
        )

        exit_notional = 0.0
        exit_cost = 0.0

        if close_positions:

            liquidation_value = sum(
                quantity * exit_prices[ticker]
                for ticker, quantity in shares.items()
            )

            exit_notional = sum(
                abs(quantity * exit_prices[ticker])
                for ticker, quantity in shares.items()
            )

            exit_cost = exit_notional * cost_rate

            cash += liquidation_value - exit_cost
            shares = {}

            exit_equity = cash


            # Record the portfolio after liquidation
            daily_records.append({
                "Date": calendar[exit_i],
                "Equity": cash
            })

            # If there's a gap before the next trade,
            # the portfolio stays in cash.
            if k + 1 < len(plans):

                next_entry_i = plans[k + 1]["entry_i"]

                for idle_i in range(
                    exit_i + 1,
                    next_entry_i
                ):
                    daily_records.append({
                        "Date": calendar[idle_i],
                        "Equity": cash
                    })

        period_return = (
            exit_equity / starting_equity - 1
        )

        records.append({
            "Date": plan["signal_date"],
            "Entry_Date": calendar[entry_i],
            "Exit_Date": calendar[exit_i],
            "Starting_Equity": starting_equity,
            "Ending_Equity": exit_equity,
            "Period_Return": period_return,
            "Traded_Notional": (
                entry_notional + exit_notional
            ),
            "Trading_Cost": entry_cost + exit_cost
        })


    # Return both datasets for daily risk analysis
    if return_daily:
        return (
            pd.DataFrame(records),
            pd.DataFrame(daily_records)
        )

    # Default behavior for existing experiments
    return pd.DataFrame(records)

