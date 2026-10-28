
import pandas as pd


def prepare_trade_data(prices, predictions):

    prices = prices.sort_values(
        ["Ticker", "Date"]
    ).copy()

    grouped = prices.groupby("Ticker")

    # Form signals at the close; orders execute at the following open.
    prices["entry_open"] = grouped["Open"].shift(-1)

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

        # Reject incomplete panels instead of silently changing the universe.
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

                long_stock = sector.nlargest(
                    1, signal
                )

                short_stock = sector.nsmallest(
                    1, signal
                )

                long_returns.append(
                    long_stock["trade_return"].iloc[0]
                )

                short_returns.append(
                    short_stock["trade_return"].iloc[0]
                )

            # Equal long and short gross exposures.
            gross_return = 0.5 * (
                sum(long_returns) / 11
                - sum(short_returns) / 11
            )

            # Approximate round-trip costs on gross notional.
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

            retained = sum(
                previous_weights.get(ticker) == weight
                for ticker, weight in weights.items()
            )

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

    cash = 1.0
    shares = {}
    cost_rate = cost_bps / 10000

    records = []

    daily_records = []

    for k, plan in enumerate(plans):

        entry_i = plan["entry_i"]
        exit_i = plan["exit_i"]

        if k > 0 and entry_i < plans[k - 1]["exit_i"]:
            raise ValueError("Overlapping portfolio periods")

        entry_prices = opens.iloc[entry_i]

        starting_equity = cash + sum(
            quantity * entry_prices[ticker]
            for ticker, quantity in shares.items()
        )

        if starting_equity <= 0:
            raise ValueError("Portfolio equity exhausted")

        target_shares = {
            ticker: weight * starting_equity / entry_prices[ticker]
            for ticker, weight in plan["weights"].items()
        }

        all_tickers = set(shares) | set(target_shares)

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


        # Mark equity at the open until the scheduled exit.
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

        exit_prices = opens.iloc[exit_i]

        exit_equity = cash + sum(
            quantity * exit_prices[ticker]
            for ticker, quantity in shares.items()
        )

        # Liquidate across gaps between holding periods.
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


            daily_records.append({
                "Date": calendar[exit_i],
                "Equity": cash
            })

            # Portfolio remains in cash while no positions are held.
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


    if return_daily:
        return (
            pd.DataFrame(records),
            pd.DataFrame(daily_records)
        )

    return pd.DataFrame(records)

