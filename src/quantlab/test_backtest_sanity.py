
import pandas as pd

from quantlab.backtest import backtest_with_shares
from quantlab.universe import SECTOR_BY_TICKER


def make_synthetic_data(
    long_multiplier=1.0,
    short_multiplier=1.0,
    n_days=7
):
    dates = pd.date_range(
        "2022-01-03",
        periods=n_days,
        freq="B"
    )

    sectors = {}

    for ticker, sector in SECTOR_BY_TICKER.items():
        sectors.setdefault(sector, []).append(ticker)

    price_rows = []
    signal_rows = []

    for sector, tickers in sectors.items():

        for rank, ticker in enumerate(sorted(tickers)):

            signal_rows.append({
                "Date": dates[0],
                "Ticker": ticker,
                "Sector": sector,
                "test_signal": rank
            })

            for i, date in enumerate(dates):

                price = 100.0

                if i == n_days - 1:

                    if rank == 2:
                        price *= long_multiplier

                    elif rank == 0:
                        price *= short_multiplier

                price_rows.append({
                    "Date": date,
                    "Ticker": ticker,
                    "Open": price
                })

    return (
        pd.DataFrame(price_rows),
        pd.DataFrame(signal_rows),
        dates
    )


def check_case(name, long_move, short_move, cost, expected):

    prices, panel, _ = make_synthetic_data(
        long_multiplier=long_move,
        short_multiplier=short_move
    )

    results = backtest_with_shares(
        prices,
        panel,
        signal="test_signal",
        cost_bps=cost
    )

    actual = results["Ending_Equity"].iloc[-1]

    assert abs(actual - expected) < 1e-10, (
        f"{name}: expected {expected}, got {actual}"
    )

    print(f"PASS: {name} — final equity ${actual:.4f}")


check_case("Flat prices, no costs", 1.0, 1.0, 0, 1.0)

check_case("Flat prices, trading costs", 1.0, 1.0, 5, 0.999)

check_case("Long stocks rise 10%", 1.1, 1.0, 0, 1.05)

check_case("Short stocks rise 10%", 1.0, 1.1, 0, 0.95)


# Unchanged positions should not incur unnecessary turnover.
prices, panel, dates = make_synthetic_data(n_days=12)

second_cycle = panel.copy()
second_cycle["Date"] = dates[5]

panel = pd.concat(
    [panel, second_cycle],
    ignore_index=True
)

results = backtest_with_shares(
    prices, panel, "test_signal", cost_bps=0
)

assert len(results) == 2
assert abs(results["Ending_Equity"].iloc[-1] - 1.0) < 1e-10

assert abs(results["Traded_Notional"].sum() - 2.0) < 1e-10

print("PASS: Unchanged positions are retained")
print("\nALL BACKTEST SANITY CHECKS PASSED")
