import pandas as pd


def calculate_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Ticker", "Date"]).copy()

    df["daily_return"] = (
        df.groupby("Ticker")["Close"].pct_change()
    )

    return df


def calculate_five_day_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Ticker", "Date"]).copy()

    close = df.groupby("Ticker")["Close"]

    df["momentum_5d"] = close.pct_change(periods=5)

    # Future outcomes are labels, never features at signal time.
    df["future_return_5d"] = (
        close.shift(-5) / df["Close"] - 1
    )

    return df


def calculate_additional_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.sort_values(["Ticker", "Date"]).copy()

    grouped = df.groupby("Ticker")

    df["momentum_20d"] = (
        grouped["Close"].pct_change(
            periods=20,
            fill_method=None
        )
    )

    df["volatility_20d"] = (
        grouped["daily_return"].transform(
            lambda x: x.rolling(
                window=20,
                min_periods=20
            ).std()
        )
    )

    average_volume = (
        grouped["Volume"].transform(
            lambda x: x.rolling(
                window=20,
                min_periods=20
            ).mean()
        )
    )

    df["volume_ratio_20d"] = (
        df["Volume"] / average_volume
    )

    return df


def calculate_execution_targets(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.sort_values(
        ["Ticker", "Date"]
    ).copy()

    grouped = df.groupby("Ticker")

    # Targets assume next-open entry, five-session holding, and exit-open pricing.
    entry_open = grouped["Open"].shift(-1)

    exit_open = grouped["Open"].shift(-6)

    df["future_open_return_5d"] = (
        exit_open / entry_open - 1
    )

    # Label availability controls which rows may enter training.
    df["execution_outcome_date"] = (
        grouped["Date"].shift(-6)
    )

    return df
