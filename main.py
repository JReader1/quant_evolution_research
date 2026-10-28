
import pandas as pd

from quantlab.data import download_stock_data

from quantlab.features import (
    calculate_daily_returns,
    calculate_five_day_features,
    calculate_additional_features
)

from quantlab.analysis import calculate_rank_ic
from quantlab.universe import SECTOR_BY_TICKER

from quantlab.model import (
    train_first_model,
    walk_forward_model,
    FEATURES
)

# STEP 1: Load historical stock data
df = download_stock_data()

# STEP 2: Calculate daily returns
df = calculate_daily_returns(df)

# STEP 3: Calculate five-day momentum and future returns
df = calculate_five_day_features(df)

# STEP 4: Calculate additional predictive features
df = calculate_additional_features(df)

# STEP 5: Assign sectors
df["Sector"] = df["Ticker"].map(SECTOR_BY_TICKER)

# STEP 6: Verify that every stock has a sector
if df["Sector"].isna().any():
    raise ValueError("Some stocks have no sector assigned.")

print("\nSECTOR COUNTS:")
print(df.groupby("Sector")["Ticker"].nunique())

# STEP 7: Check missing values
print("\nMISSING DAILY RETURNS:")
print(df["daily_return"].isna().sum())

print("\nMISSING FIVE-DAY VALUES:")
print(
    df[["momentum_5d", "future_return_5d"]].isna().sum()
)

# STEP 8: Calculate average daily returns
print("\nAVERAGE DAILY RETURN BY STOCK:")
print(
    df.groupby("Ticker")["daily_return"].mean()
)

# STEP 9: Investigate momentum and future returns
print("\nFIVE-DAY MOMENTUM CORRELATIONS:")

for ticker, stock in df.groupby("Ticker"):

    clean = stock.dropna(
        subset=["momentum_5d", "future_return_5d"]
    )

    correlation = clean["momentum_5d"].corr(
        clean["future_return_5d"]
    )

    print(
        f"{ticker}: {correlation:.4f} "
        f"({len(clean)} observations)"
    )


# STEP 10: Test whether correlations change over time

periods = {
    "2018–2021": ("2018-01-01", "2021-12-31"),
    "2022–2024": ("2022-01-01", "2024-12-31")
}

for period_name, (start, end) in periods.items():

    print(f"\nRESULTS FOR {period_name}")

    segment = df[
        df["Date"].between(start, end)
    ].sort_values(["Ticker", "Date"]).copy()

    # Recalculate within each period to avoid
    # using future outcomes beyond its boundary.
    close = segment.groupby("Ticker")["Close"]

    segment["past_5d"] = close.pct_change(periods=5)

    segment["future_5d"] = (
        close.shift(-5) / segment["Close"] - 1
    )

    for ticker, stock in segment.groupby("Ticker"):

        clean = stock.dropna(
            subset=["past_5d", "future_5d"]
        )

        correlation = clean["past_5d"].corr(
            clean["future_5d"]
        )

        print(f"{ticker}: {correlation:.4f}")


# STEP 11: Cross-sectional rank correlation

daily_ic = calculate_rank_ic(df)

print("\nCROSS-SECTIONAL RANK IC:")

print("\nFIRST FIVE DATES:")
print(daily_ic.head())

print("\nAVERAGE RANK IC:")
print(daily_ic["rank_ic"].mean())

print("\nPERCENTAGE OF POSITIVE IC DAYS:")
print((daily_ic["rank_ic"] > 0).mean() * 100)

print("\nNUMBER OF EVALUATED DATES:")
print(len(daily_ic))


# STEP 12: Evaluate rank IC by year

daily_ic["year"] = pd.to_datetime(
    daily_ic["Date"]
).dt.year

yearly_results = daily_ic.groupby("year").agg(
    average_ic=("rank_ic", "mean"),
    median_ic=("rank_ic", "median"),
    positive_days=("rank_ic", lambda x: (x > 0).mean() * 100),
    observations=("rank_ic", "count")
)

print("\nYEARLY RANK IC RESULTS:")
print(yearly_results.round(4))

# STEP 13: Compare candidate predictive signals

features = [
    "momentum_5d",
    "momentum_20d",
    "volatility_20d",
    "volume_ratio_20d"
]

print("\nCOMPARING PREDICTIVE FEATURES:")

for feature in features:

    results = calculate_rank_ic(df, feature)

    mean_ic = results["rank_ic"].mean()

    positive_pct = (
        (results["rank_ic"] > 0).mean() * 100
    )

    print(f"\nFeature: {feature}")
    print(f"Average Rank IC: {mean_ic:.4f}")
    print(f"Positive IC dates: {positive_pct:.2f}%")
    print(f"Evaluated dates: {len(results)}")


# STEP 14: Compare raw and sector-relative signals

features = [
    "momentum_5d",
    "momentum_20d",
    "volatility_20d",
    "volume_ratio_20d"
]

print("\nRAW VS SECTOR-RELATIVE RANK IC")

for feature in features:

    raw = calculate_rank_ic(
        df,
        feature=feature
    )

    sector_relative = calculate_rank_ic(
        df,
        feature=feature,
        sector_neutral=True
    )

    raw_ic = raw["rank_ic"].mean()
    sector_ic = sector_relative["rank_ic"].mean()

    print(f"\n{feature}")
    print(f"Raw Rank IC: {raw_ic:.4f}")
    print(f"Sector-relative Rank IC: {sector_ic:.4f}")


# STEP 15: Diagnose our first predictive model

model, train, validation = train_first_model(df)

# Predict on training data for diagnostic purposes
train = train.copy()

train["model_prediction"] = model.predict(
    train[FEATURES]
)

# Calculate model performance in each period
train_ic = calculate_rank_ic(
    train,
    feature="model_prediction"
)

validation_ic = calculate_rank_ic(
    validation,
    feature="model_prediction"
)

# Same-period momentum benchmark
baseline_ic = calculate_rank_ic(
    validation,
    feature="momentum_5d"
)

print("\nMODEL DIAGNOSTICS")

print(f"Training observations: {len(train)}")
print(f"Validation observations: {len(validation)}")

print("\nTRAINING VS VALIDATION RANK IC")

print(
    f"Training Ridge IC: "
    f"{train_ic['rank_ic'].mean():.4f}"
)

print(
    f"Validation Ridge IC: "
    f"{validation_ic['rank_ic'].mean():.4f}"
)

print(
    f"Validation Momentum IC: "
    f"{baseline_ic['rank_ic'].mean():.4f}"
)

# Examine learned coefficients
ridge = model.named_steps["ridge"]

coefficients = pd.Series(
    ridge.coef_,
    index=FEATURES
)

print("\nRIDGE COEFFICIENTS")
print(coefficients.round(6))

# Compare model and baseline year by year
model_yearly = (
    validation_ic.assign(
        year=validation_ic["Date"].dt.year
    )
    .groupby("year")["rank_ic"]
    .mean()
)

baseline_yearly = (
    baseline_ic.assign(
        year=baseline_ic["Date"].dt.year
    )
    .groupby("year")["rank_ic"]
    .mean()
)

comparison = pd.DataFrame({
    "ridge_ic": model_yearly,
    "momentum_ic": baseline_yearly
})

print("\nYEARLY MODEL VS BENCHMARK")
print(comparison.round(4))


# STEP 16: Compare feature relationships across periods

print("\nFEATURE STABILITY ANALYSIS")

print(
    f"{'Feature':<22}"
    f"{'Training IC':>14}"
    f"{'Validation IC':>16}"
)

for feature in FEATURES:

    # Measure the relationship during training
    training_results = calculate_rank_ic(
        train,
        feature=feature
    )

    # Measure the relationship during validation
    validation_results = calculate_rank_ic(
        validation,
        feature=feature
    )

    training_mean = (
        training_results["rank_ic"].mean()
    )

    validation_mean = (
        validation_results["rank_ic"].mean()
    )

    print(
        f"{feature:<22}"
        f"{training_mean:>+14.4f}"
        f"{validation_mean:>+16.4f}"
    )

# STEP 17: Walk-forward model evaluation

print("\nWALK-FORWARD TRAINING")
walk = walk_forward_model(df)

# Use the original static Ridge model as a comparison
walk["static_prediction"] = model.predict(
    walk[FEATURES]
)

comparison_rows = []

for year, group in walk.groupby("evaluation_year"):

    walk_ic = calculate_rank_ic(
        group, feature="model_prediction"
    )["rank_ic"].mean()

    static_ic = calculate_rank_ic(
        group, feature="static_prediction"
    )["rank_ic"].mean()

    momentum_ic = calculate_rank_ic(
        group, feature="momentum_5d"
    )["rank_ic"].mean()

    comparison_rows.append({
        "Year": year,
        "Walk_Forward_IC": walk_ic,
        "Static_Ridge_IC": static_ic,
        "Momentum_IC": momentum_ic
    })

comparison = pd.DataFrame(
    comparison_rows
).set_index("Year")

print("\nWALK-FORWARD RESULTS:")
print(comparison.round(4))

print("\nAVERAGE IC BY APPROACH:")
print(comparison.mean().round(4))
