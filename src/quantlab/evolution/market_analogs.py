"""Historical-market similarity experiments."""

def preview_market_analogs(project_root, as_of="2025-12-31", n=5, backtest=False):
    """
    Find historically similar U.S. market environments.
    Exploratory only. Does not change trading decisions.
    """
    from io import BytesIO, StringIO
    from pathlib import Path
    from urllib.request import Request, urlopen
    import csv
    import zipfile
    import re

    import numpy as np
    import pandas as pd

    cache = (
        Path(project_root)
        / "data"
        / "reference"
        / "french_market_daily.csv"
    )

    if not cache.exists():

        url = (
            "https://mba.tuck.dartmouth.edu/pages/"
            "faculty/ken.french/ftp/"
            "F-F_Research_Data_Factors_daily_CSV.zip"
        )

        print("Downloading long-run U.S. market returns...")

        request = Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urlopen(request, timeout=40) as response:
            archive = response.read()

        with zipfile.ZipFile(BytesIO(archive)) as z:
            source_text = z.read(
                z.namelist()[0]
            ).decode("utf-8-sig", errors="replace")

        observations = []

        for row in csv.reader(StringIO(source_text)):

            if len(row) < 5:
                continue

            if not re.fullmatch(r"\d{8}", row[0].strip()):
                continue

            try:
                market_excess = float(row[1])
                risk_free = float(row[4])

                if not (
                    np.isfinite(market_excess)
                    and np.isfinite(risk_free)
                ):
                    continue

                if (
                    abs(market_excess) >= 90
                    or abs(risk_free) >= 90
                ):
                    continue

                # French values are reported in percent.
                market_return = (
                    market_excess + risk_free
                ) / 100.0

                observations.append(
                    (row[0].strip(), market_return)
                )

            except ValueError:
                continue

        if len(observations) < 10000:
            raise ValueError(
                "Unexpected historical market data"
            )

        cache.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        pd.DataFrame(
            observations,
            columns=["Date", "Market_Return"]
        ).to_csv(cache, index=False)

    market = pd.read_csv(
        cache,
        parse_dates=["Date"]
    )

    if (
        market["Date"].isna().any()
        or market["Date"].duplicated().any()
        or market["Market_Return"].isna().any()
    ):
        raise ValueError("Invalid cached market data")

    market = (
        market.sort_values("Date")
        .set_index("Date")
        .loc[:pd.Timestamp(as_of)]
    )

    if len(market) < 1000:
        raise ValueError("Not enough historical data")

    r = market["Market_Return"].astype(float)

    if ((r <= -1) | (~np.isfinite(r))).any():
        raise ValueError("Invalid historical returns")

    wealth = (1 + r).cumprod()

    features = pd.DataFrame(index=r.index)

    features["r20"] = wealth / wealth.shift(20) - 1

    features["r60"] = wealth / wealth.shift(60) - 1

    features["vol20"] = (
        r.rolling(20).std() * np.sqrt(252)
    )

    features["dd60"] = (
        wealth / wealth.rolling(60).max() - 1
    )

    features = features.dropna()

    current = features.iloc[-1]

    # Limit neighbors to dates at least a year before the decision date.
    cutoff = (
        features.index[-1]
        - pd.Timedelta(days=365)
    )

    candidates = features.loc[
        features.index <= cutoff
    ]

    if candidates.empty:
        raise ValueError("No eligible historical analogs")

    # Scales normalize features with different magnitudes.
    scales = pd.Series({
        "r20": 0.10,
        "r60": 0.18,
        "vol20": 0.20,
        "dd60": 0.15
    })

    distance = np.sqrt(
        (
            ((candidates - current) / scales) ** 2
        ).sum(axis=1)
    )

    # Separate neighbors to avoid counting the same episode repeatedly.
    chosen = []

    for date in distance.sort_values().index:

        if all(
            abs((date - previous).days) >= 90
            for previous in chosen
        ):
            chosen.append(date)

        if len(chosen) == n:
            break

    result = candidates.loc[chosen].copy()

    result.insert(
        0,
        "distance",
        distance.loc[chosen]
    )



    # Outcomes are measured over the next 21 trading sessions.
    future_21d = (
        wealth.shift(-21) / wealth - 1
    )

    result["next_21d_return"] = (
        future_21d.reindex(result.index)
    )

    # Require the full subsequent return for each selected neighbor.
    if result["next_21d_return"].isna().any():
        raise ValueError(
            "Historical analog has an unknown future outcome"
        )

    print("\n21-SESSION RETURNS AFTER HISTORICAL ANALOGS")

    print(
        result["next_21d_return"].map(
            lambda value: f"{value:+.2%}"
        ).to_string()
    )

    print(
        "\nAverage analog return:",
        f"{result['next_21d_return'].mean():+.2%}"
    )

    print(
        "Positive outcomes:",
        int((result["next_21d_return"] > 0).sum()),
        "of",
        len(result)
    )


    print(
        "\nMarket as of:",
        features.index[-1].date()
    )

    print(
        "Historical dates searched:",
        len(candidates)
    )

    print(
        "Lower distance means greater similarity."
    )

    print(
        "This is NOT a future-return prediction."
    )


    if backtest:

        print("\nHISTORICAL ANALOG WALK-FORWARD TEST")

        future = wealth.shift(-21) / wealth - 1

        # Sample one test date per approximate holding period.
        test_dates = features.loc[
            "2018-01-01":"2024-12-31"
        ].index[::21]

        records = []

        for test_date in test_dates:

            actual = future.loc[test_date]

            if pd.isna(actual):
                continue

            # Keep training neighbors and their subsequent outcomes in the past.
            eligible = features.loc[
                :test_date - pd.Timedelta(days=365)
            ]

            if len(eligible) < 500:
                continue

            distances = np.sqrt(
                (
                    ((eligible - features.loc[test_date])
                     / scales) ** 2
                ).sum(axis=1)
            )

            matches = []

            for historical_date in distances.sort_values().index:

                if all(
                    abs((historical_date - chosen).days) >= 90
                    for chosen in matches
                ):
                    matches.append(historical_date)

                if len(matches) == n:
                    break

            if len(matches) < n:
                continue

            analog_outcomes = future.reindex(matches)

            if analog_outcomes.isna().any():
                continue

            # Compare against the unconditional historical mean.
            historical_outcomes = (
                future.reindex(eligible.index).dropna()
            )

            records.append({
                "Date": test_date,
                "Analog_Forecast": analog_outcomes.mean(),
                "Simple_Forecast": historical_outcomes.mean(),
                "Actual_Return": actual
            })

        comparison = pd.DataFrame(
            records
        ).set_index("Date")

        if len(comparison) < 20:
            raise ValueError(
                "Insufficient historical testing dates"
            )

        print("Historical forecasts:", len(comparison))


        positive_rate = (
            comparison["Actual_Return"] > 0
        ).mean()

        print(
            "Always-positive directional accuracy:",
            f"{positive_rate:.1%}"
        )

        print(
            "Analog positive predictions:",
            int((comparison["Analog_Forecast"] > 0).sum()),
            "of",
            len(comparison)
        )


        for label in ["Analog_Forecast", "Simple_Forecast"]:

            errors = (
                comparison[label]
                - comparison["Actual_Return"]
            )

            mae = errors.abs().mean()

            rmse = np.sqrt(
                (errors ** 2).mean()
            )

            directional_accuracy = (
                (comparison[label] > 0)
                ==
                (comparison["Actual_Return"] > 0)
            ).mean()

            print(f"\n{label}")
            print(f"Mean absolute error: {mae:.2%}")
            print(f"Root mean squared error: {rmse:.2%}")
            print(
                "Directional accuracy:",
                f"{directional_accuracy:.1%}"
            )

    return result.round(4)
