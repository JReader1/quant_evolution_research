
import pandas as pd


def calculate_rank_ic(
    df: pd.DataFrame,
    feature: str = "momentum_5d",
    sector_neutral: bool = False
) -> pd.DataFrame:

    required = [feature, "future_return_5d"]

    if sector_neutral:
        required.append("Sector")

    clean = df.dropna(subset=required).copy()

    if sector_neutral:

        groups = clean.groupby(["Date", "Sector"])

        # Remove sector-average feature values
        clean["signal"] = (
            clean[feature]
            - groups[feature].transform("mean")
        )

        # Remove sector-average future returns
        clean["outcome"] = (
            clean["future_return_5d"]
            - groups["future_return_5d"].transform("mean")
        )

    else:

        clean["signal"] = clean[feature]
        clean["outcome"] = clean["future_return_5d"]

    results = []

    for date, group in clean.groupby("Date"):

        if group["Ticker"].nunique() < 20:
            continue

        signal_rank = group["signal"].rank()
        outcome_rank = group["outcome"].rank()

        ic = signal_rank.corr(outcome_rank)

        if pd.notna(ic):
            results.append({
                "Date": date,
                "rank_ic": ic
            })

    return pd.DataFrame(
        results,
        columns=["Date", "rank_ic"]
    )
