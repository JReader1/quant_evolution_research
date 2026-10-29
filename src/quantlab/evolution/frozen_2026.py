"""Evaluation of fixed strategies on later-period prices."""

from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.evolution.arena import BASELINE, HOLD_SESSIONS, Genome, HistoricalArena


def evaluate_frozen_2026(project_root):
    """
    Evaluate frozen strategies using 2026 prices.
    Does not retrain, mutate, or promote candidates.
    """

    root = Path(project_root).resolve()

    cache = (
            root / "data" / "reference"
            / "frozen_2026_panel.csv"
    )

    anchor = pd.Timestamp("2025-12-31")
    cutoff = pd.Timestamp("2026-10-24")

    arena = HistoricalArena(root, cost_bps=5.0)

    if arena.calendar[-1] != anchor:
        raise ValueError(
            "Historical cache must end on 2025-12-31"
        )

    if not cache.exists():

        import yfinance as yf

        print("Downloading 2026 evaluation data...")

        parts = []

        for ticker in arena.tickers:

            print("Fetching", ticker)

            prices = yf.download(
                ticker,
                start="2025-12-15",
                end="2026-10-24",
                auto_adjust=True,
                multi_level_index=False,
                progress=False,
                threads=False
            )

            if (
                    prices.empty
                    or not {"Open", "Close"}.issubset(prices.columns)
            ):
                raise ValueError(
                    f"Missing price data: {ticker}"
                )

            prices = prices.reset_index()[
                ["Date", "Open", "Close"]
            ]

            prices["Ticker"] = ticker

            parts.append(prices)

        panel = pd.concat(parts, ignore_index=True)

        cache.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        panel.to_csv(cache, index=False)

        print("Cached:", cache)

    else:

        print("Using preserved 2026 price data")

    panel = pd.read_csv(
        cache,
        parse_dates=["Date"]
    )

    if set(panel.columns) != {
        "Date", "Ticker", "Open", "Close"
    }:
        raise ValueError("Unexpected cache columns")

    if set(panel["Ticker"]) != set(arena.tickers):
        raise ValueError("Incomplete stock universe")

    if (
            panel["Date"].isna().any()
            or panel.duplicated(["Date", "Ticker"]).any()
    ):
        raise ValueError("Invalid trading dates")

    if (
            not np.isfinite(
                panel[["Open", "Close"]].to_numpy(
                    dtype=float
                )
            ).all()
            or (panel[["Open", "Close"]] <= 0).any().any()
    ):
        raise ValueError("Invalid stock prices")


    # Extend only the in-memory price panel; preserve the original files.

    arena.opens = arena.opens.copy()
    arena.closes = arena.closes.copy()

    new_dates = None
    new_opens = []
    new_closes = []


    for j, ticker in enumerate(arena.tickers):

        stock = (
            panel.loc[panel["Ticker"] == ticker]
            .sort_values("Date")
            .set_index("Date")
        )

        if anchor not in stock.index:
            raise ValueError(
                f"Missing overlap date: {ticker}"
            )

        later = stock.loc[
            (stock.index > anchor)
            & (stock.index < cutoff)
            ]

        if len(later) < 150:
            raise ValueError(
                f"Insufficient 2026 data: {ticker}"
            )

        if new_dates is None:
            new_dates = later.index

        elif not new_dates.equals(later.index):
            raise ValueError(
                f"Trading calendar mismatch: {ticker}"
            )

        # Align newer adjusted prices to the old cache on the overlap date.
        scale = (
                stock.loc[anchor, "Close"]
                / arena.closes[-1, j]
        )

        arena.opens[:, j] *= scale
        arena.closes[:, j] *= scale

        new_opens.append(
            later["Open"].to_numpy(dtype=float)
        )

        new_closes.append(
            later["Close"].to_numpy(dtype=float)
        )

    if new_dates[-1] != pd.Timestamp("2026-10-23"):
        raise ValueError(
            "Expected prices through 2026-10-23"
        )

    arena.calendar = arena.calendar.append(new_dates)

    arena.opens = np.vstack([
        arena.opens,
        np.column_stack(new_opens)
    ])

    arena.closes = np.vstack([
        arena.closes,
        np.column_stack(new_closes)
    ])

    returns = pd.DataFrame(
        arena.closes
    ).pct_change(fill_method=None)

    arena.vol20 = (
        returns.rolling(20, min_periods=20)
        .std()
        .to_numpy(dtype=float)
    )

    arena._score_cache.clear()
    arena._result_cache.clear()

    # Fixed candidates; the 2026 sample is an evaluation, not a search.
    champion = Genome(
        lookback=10,
        skip=0,
        volatility_penalty=1.0,
        risk_fraction=1.0
    )

    lower_risk = Genome(
        lookback=5,
        skip=0,
        volatility_penalty=0.0,
        risk_fraction=0.75
    )

    strategies = {
        "Evolution champion": champion,
        "Original momentum": BASELINE,
        "75% momentum": lower_risk,
        "Equal-weight 33": None
    }

    print("\nFROZEN 2026 STRATEGY EVALUATION")

    results = {}

    for name, genome in strategies.items():
        result = arena.evaluate(
            genome,
            "2026-01-02",
            "2026-10-24"
        )

        results[name] = result

        print(
            f"\n{name}"
            f"\nNet return: {result['return']:+.2%}"
            f"\nMax drawdown: {result['max_drawdown']:.2%}"
            f"\nCompleted periods: {result['periods']}"
            f"\nLast exit: {result['last_exit']}"
        )

    # Evaluate the analog risk overlay without changing the selected strategy.
    french_path = (
        root / "data" / "reference" / "french_market_daily.csv"
    )

    french = pd.read_csv(
        french_path, parse_dates=["Date"]
    )

    french = (
        french.sort_values("Date")
        .set_index("Date")
        .loc[:"2026-08-31"]
    )

    if (
        french.empty
        or french.index.max() < pd.Timestamp("2026-08-31")
        or french.index.has_duplicates
    ):
        raise ValueError("Invalid market-factor coverage")

    daily = french["Market_Return"].astype(float)
    wealth = (1.0 + daily).cumprod()

    features = pd.DataFrame(index=daily.index)
    features["r20"] = wealth / wealth.shift(20) - 1
    features["r60"] = wealth / wealth.shift(60) - 1
    features["vol20"] = (
        daily.rolling(20).std() * np.sqrt(252)
    )
    features["dd60"] = (
        wealth / wealth.rolling(60).max() - 1
    )

    features = features.dropna()
    future = wealth.shift(-21) / wealth - 1

    scales = pd.Series({
        "r20": 0.10,
        "r60": 0.18,
        "vol20": 0.20,
        "dd60": 0.15
    })

    start, end = "2026-01-02", "2026-09-01"

    first = int(
        arena.calendar.searchsorted(pd.Timestamp(start))
    )
    stop = int(
        arena.calendar.searchsorted(pd.Timestamp(end))
    )

    signal_positions = list(
        range(first, stop - HOLD_SESSIONS - 1, HOLD_SESSIONS)
    )

    if not signal_positions:
        raise ValueError("No completed evaluation periods")

    schedule = {}

    for si in signal_positions:

        signal_date = arena.calendar[si]

        # Use the previous factor date; same-day factor data is not assumed available.
        pos = (
            features.index.searchsorted(
                signal_date, side="left"
            ) - 1
        )

        if pos < 0:
            raise ValueError("Missing historical market signal")

        as_of = features.index[pos]

        if (signal_date - as_of).days > 5:
            raise ValueError("Stale market observation")

        current = features.loc[as_of]

        candidates = features.loc[
            :as_of - pd.Timedelta(days=365)
        ]

        scaled = (
            (candidates - current) / scales
        ).astype(float)

        distances = np.sqrt(
            (scaled ** 2).sum(axis=1)
        )

        matches = []

        for candidate_date in distances.sort_values().index:

            if all(
                abs((candidate_date - earlier).days) >= 90
                for earlier in matches
            ):
                matches.append(candidate_date)

            if len(matches) == 5:
                break

        if len(matches) != 5:
            raise ValueError("Insufficient historical analogs")

        known_outcomes = future.reindex(matches)

        if known_outcomes.isna().any():
            raise ValueError("Unavailable historical outcome")

        forecast = float(known_outcomes.mean())

        schedule[signal_date] = (
            0.75 if forecast < 0 else 1.0
        )

    matched = float(np.mean(list(schedule.values())))

    print("\n2026 CHAMPION VS ANALOG OVERLAY")
    print("Analog warnings:",
          sum(value < 1 for value in schedule.values()),
          "of", len(schedule))
    print("Average exposure:", f"{matched:.2%}")

    competitors = {
        "Frozen champion": {
            day: 1.0 for day in schedule
        },
        "Constant matched exposure": {
            day: matched for day in schedule
        },
        "Champion + price analogs": schedule
    }

    for label, risk in competitors.items():

        outcome = arena.evaluate(
            champion,
            start,
            end,
            risk_schedule=risk
        )

        print(f"\n{label}")
        print(f"Net return: {outcome['return']:+.2%}")
        print(
            f"Max drawdown: "
            f"{outcome['max_drawdown']:.2%}"
        )
        print("Completed periods:", outcome["periods"])
        print("Last exit:", outcome["last_exit"])

    print("\nSAME-WINDOW STOCK-SELECTION BENCHMARKS")

    for name, genome in [
        ("Original momentum", BASELINE),
        ("Equal-weight 33", None)
    ]:

        outcome = arena.evaluate(
            genome,
            start,
            end
        )

        print(f"\n{name}")
        print(f"Net return: {outcome['return']:+.2%}")
        print(
            "Max drawdown:",
            f"{outcome['max_drawdown']:.2%}"
        )
        print("Completed periods:", outcome["periods"])

    print("\nEXHIBITION ONLY - NO CHAMPION PROMOTION")

    print("\nPASS: Frozen evaluation completed")
    print("No strategy was retrained or promoted.")

    return results
