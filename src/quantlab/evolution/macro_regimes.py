"""Historical CPI and Treasury-rate signals for market analog comparisons."""

from pathlib import Path

import numpy as np
import pandas as pd


def preview_macro_regimes(project_root):
    """
    Explore historical inflation and interest rates.
    Research only; does not affect trading decisions.
    """

    from io import StringIO
    from urllib.request import Request, urlopen

    root = Path(project_root).resolve()
    cache = root / "data" / "reference" / "macro_regimes.csv"

    if not cache.exists():

        print("Downloading economic history from FRED...")

        tables = {}

        for series in ["CPIAUCNS", "GS10"]:

            url = (
                "https://fred.stlouisfed.org/graph/"
                f"fredgraph.csv?id={series}"
            )

            request = Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"}
            )

            with urlopen(request, timeout=40) as response:
                content = response.read().decode("utf-8-sig")

            data = pd.read_csv(StringIO(content))

            if len(data.columns) != 2:
                raise ValueError(f"Unexpected FRED data: {series}")

            data.columns = ["Date", series]

            data["Date"] = pd.to_datetime(data["Date"])

            data[series] = pd.to_numeric(
                data[series], errors="coerce"
            )

            tables[series] = data.dropna()

        cpi = tables["CPIAUCNS"].sort_values("Date")

        cpi["inflation_yoy"] = (
            cpi["CPIAUCNS"].pct_change(12) * 100
        )

        macro = cpi[["Date", "inflation_yoy"]].merge(
            tables["GS10"],
            on="Date",
            how="inner"
        )

        macro = macro.dropna().sort_values("Date")

        # Approximate reporting lag; revised FRED data are not point-in-time vintages.
        macro["usable_from"] = (
            macro["Date"]
            + pd.offsets.MonthEnd(0)
            + pd.Timedelta(days=45)
        )

        cache.parent.mkdir(parents=True, exist_ok=True)

        macro.to_csv(cache, index=False)

        print("Economic history cached.")

    else:

        print("Using cached economic history.")

    macro = pd.read_csv(
        cache,
        parse_dates=["Date", "usable_from"]
    )

    required = {
        "Date", "usable_from", "inflation_yoy", "GS10"
    }

    if not required.issubset(macro.columns):
        raise ValueError("Incomplete macroeconomic dataset")

    if macro["Date"].duplicated().any():
        raise ValueError("Duplicate economic dates")

    if macro[list(required)].isna().any().any():
        raise ValueError("Missing economic values")

    print("\nHISTORICAL ECONOMIC REGIMES")

    for date_string in ["1976-09-13", "2025-12-31"]:

        as_of = pd.Timestamp(date_string)

        available = macro.loc[
            macro["usable_from"] <= as_of
        ].sort_values("Date")

        if available.empty:
            raise ValueError(
                f"No available economic history for {date_string}"
            )

        latest = available.iloc[-1]

        print(f"\nMarket date: {date_string}")
        print(f"Economic observation: {latest['Date'].date()}")

        print(
            "Annual CPI inflation:",
            f"{latest['inflation_yoy']:+.2f}%"
        )

        print(
            "10-year Treasury yield:",
            f"{latest['GS10']:.2f}%"
        )

    print("\nPASS: Economic-regime preview completed")

    return macro

def compare_macro_analogs(project_root):
    """
    Compare price-only and macro-aware analog forecasts.
    Exploratory research; never changes trading strategies.
    """

    root = Path(project_root).resolve()
    reference = root / "data" / "reference"

    market = pd.read_csv(
        reference / "french_market_daily.csv",
        parse_dates=["Date"]
    ).sort_values("Date").set_index("Date")

    market = market.loc[:"2025-12-31"]

    macro = pd.read_csv(
        reference / "macro_regimes.csv",
        parse_dates=["usable_from"]
    ).sort_values("usable_from")

    # Reconstruct the original market features.
    r = market["Market_Return"].astype(float)
    wealth = (1 + r).cumprod()

    features = pd.DataFrame(index=r.index)
    features.index.name = "Date"

    features["r20"] = wealth / wealth.shift(20) - 1
    features["r60"] = wealth / wealth.shift(60) - 1
    features["vol20"] = (
        r.rolling(20).std() * np.sqrt(252)
    )
    features["dd60"] = (
        wealth / wealth.rolling(60).max() - 1
    )

    features = features.dropna()

    # Attach only economic observations that would
    # have passed our assumed 45-day availability lag.
    combined = pd.merge_asof(
        features.reset_index().sort_values("Date"),
        macro[
            ["usable_from", "inflation_yoy", "GS10"]
        ].sort_values("usable_from"),
        left_on="Date",
        right_on="usable_from",
        direction="backward"
    )

    combined = (
        combined.set_index("Date")
        .dropna()
        .sort_index()
    )

    # Feature scales were fixed before comparing these variants.
    scales = pd.Series({
        "r20": 0.10,
        "r60": 0.18,
        "vol20": 0.20,
        "dd60": 0.15,
        "inflation_yoy": 4.0,
        "GS10": 3.0
    })

    price_columns = [
        "r20", "r60", "vol20", "dd60"
    ]

    macro_columns = price_columns + [
        "inflation_yoy", "GS10"
    ]

    future = wealth.shift(-21) / wealth - 1

    def analog_forecast(eligible, current, columns):


        scaled = (
            eligible[columns].astype(float)
            - current[columns].astype(float)
        ) / scales[columns]

        distance = np.sqrt(
            (scaled ** 2).sum(axis=1)
        )


        matches = []

        for date in distance.sort_values().index:

            if all(
                abs((date - earlier).days) >= 90
                for earlier in matches
            ):
                matches.append(date)

            if len(matches) == 5:
                break

        if len(matches) != 5:
            return np.nan

        outcomes = future.reindex(matches)

        if outcomes.isna().any():
            return np.nan

        return float(outcomes.mean())

    test_dates = combined.loc[
        "2018-01-01":"2024-12-31"
    ].index[::21]

    records = []

    for test_date in test_dates:

        # Only compare to neighbors with known subsequent outcomes.
        cutoff = test_date - pd.Timedelta(days=365)

        eligible = combined.loc[:cutoff]

        if len(eligible) < 500:
            continue

        actual = future.loc[test_date]

        if pd.isna(actual):
            continue

        current = combined.loc[test_date]

        price_prediction = analog_forecast(
            eligible, current, price_columns
        )

        macro_prediction = analog_forecast(
            eligible, current, macro_columns
        )

        benchmark = future.reindex(
            eligible.index
        ).mean()

        if not np.isfinite([
            price_prediction,
            macro_prediction,
            benchmark
        ]).all():
            continue

        records.append({
            "Date": test_date,
            "Actual": actual,
            "Price_Only": price_prediction,
            "Macro_Aware": macro_prediction,
            "Historical_Average": benchmark
        })

    if len(records) < 20:
        raise ValueError(
            "Insufficient paired evaluation dates"
        )

    comparison = pd.DataFrame(
        records
    ).set_index("Date")

    actual = comparison["Actual"]

    print("\nPRICE VS ECONOMIC-REGIME ANALOG TEST")
    print("Matched forecasts:", len(comparison))
    print(
        "Always-positive accuracy:",
        f"{(actual > 0).mean():.1%}"
    )

    for label in [
        "Historical_Average",
        "Price_Only",
        "Macro_Aware"
    ]:

        forecast = comparison[label]
        error = forecast - actual

        mae = error.abs().mean()
        rmse = np.sqrt((error ** 2).mean())

        direction = (
            (forecast > 0) == (actual > 0)
        ).mean()

        declining = actual < 0

        bearish_recall = (
            ((forecast < 0) & declining).sum()
            / declining.sum()
            if declining.any()
            else np.nan
        )

        print(f"\n{label}")
        print(f"Mean absolute error: {mae:.2%}")
        print(f"RMSE: {rmse:.2%}")
        print(f"Directional accuracy: {direction:.1%}")
        print(
            "Decline detection:",
            f"{bearish_recall:.1%}"
        )

    print("\nNo strategies retrained or promoted.")


    print("\nDOWNSIDE SIGNAL AUDIT")

    actual_down = comparison["Actual"] < 0

    for label in [
        "Historical_Average",
        "Price_Only",
        "Macro_Aware"
    ]:

        predicted_down = comparison[label] < 0

        tp = int((predicted_down & actual_down).sum())
        fp = int((predicted_down & ~actual_down).sum())
        fn = int((~predicted_down & actual_down).sum())
        tn = int((~predicted_down & ~actual_down).sum())

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        specificity = tn / (tn + fp) if tn + fp else 0

        balanced_accuracy = (recall + specificity) / 2

        print(f"\n{label}")
        print("Correct decline warnings:", tp)
        print("False decline warnings:", fp)
        print("Missed declines:", fn)
        print("Correct rising periods:", tn)
        print(f"Warning precision: {precision:.1%}")
        print(f"Balanced accuracy: {balanced_accuracy:.1%}")



    # These exposure rules are exploratory; they do not select a champion.

    print("\nHISTORICAL ANALOG EXHIBITION MATCH")

    actual = comparison["Actual"].to_numpy(dtype=float)

    # Cut market exposure to 75% when the analog mean is negative.
    analog_exposure = np.where(
        comparison["Price_Only"].to_numpy() < 0,
        0.75,
        1.0
    )


    # Match the average invested fraction to isolate timing effects.
    matched_exposure = float(analog_exposure.mean())

    strategies = {
        "Full market": np.ones(len(actual)),
        "Constant 75%": np.full(len(actual), 0.75),
        "Constant matched exposure": np.full(
            len(actual), matched_exposure
        ),
        "Analog challenger": analog_exposure
    }

    print(
        "Analog average exposure:",
        f"{matched_exposure:.2%}"
    )

    for name, exposure in strategies.items():

        period_returns = exposure * actual

        equity = np.r_[
            1.0,
            np.cumprod(1 + period_returns)
        ]

        total_return = equity[-1] - 1

        drawdown = (
            equity / np.maximum.accumulate(equity) - 1
        ).min()

        print(f"\n{name}")
        print(f"Compounded return: {total_return:+.2%}")
        print(f"Worst period: {period_returns.min():+.2%}")
        print(f"Period-end drawdown: {drawdown:.2%}")

    print(
        "\nAnalog risk-off periods:",
        int((analog_exposure < 1).sum()),
        "of",
        len(actual)
    )

    print("\nEXHIBITION ONLY - NO CHAMPION PROMOTION")


    print("\nANALOG PERFORMANCE BY YEAR")

    years = comparison.index.year

    for year in sorted(set(years)):

        mask = years == year

        year_returns = actual[mask]
        year_exposure = analog_exposure[mask]

        analog_return = (
            np.prod(1 + year_exposure * year_returns) - 1
        )

        matched_return = (
            np.prod(1 + matched_exposure * year_returns) - 1
        )

        advantage = analog_return - matched_return

        warnings = int((year_exposure < 1).sum())

        print(
            f"{year}: "
            f"Analog {analog_return:+.2%} | "
            f"Matched {matched_return:+.2%} | "
            f"Difference {advantage:+.2%} | "
            f"Warnings {warnings}"
        )


    return comparison
