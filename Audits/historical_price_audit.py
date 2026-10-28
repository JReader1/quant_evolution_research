
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from pathlib import Path
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

START = pd.Timestamp("2022-01-04")
END = pd.Timestamp("2025-01-02")


# STEP 1: Load preserved historical membership

members_2022 = pd.read_csv(
    REPORTS / "sp500_membership_2022-01-03.csv",
    dtype={"symbol": str}
)

members_2025 = pd.read_csv(
    REPORTS / "sp500_membership_2025-01-02.csv",
    dtype={"symbol": str}
)

symbols_2025 = set(members_2025["symbol"])

exited = members_2022[
    ~members_2022["symbol"].isin(symbols_2025)
].copy()

retained = members_2022[
    members_2022["symbol"].isin(symbols_2025)
].copy()

print("\nHISTORICAL MEMBERSHIP CHANGES")
print("2022 members:", len(members_2022))
print("2025 members:", len(members_2025))
print("No longer members:", len(exited))
print("Retained members:", len(retained))


# STEP 2: Select reproducible audit samples

sample_exited = exited.sample(
    n=min(6, len(exited)),
    random_state=41
).copy()

sample_exited["Group"] = "Exited index"

sample_retained = retained.sample(
    n=min(6, len(retained)),
    random_state=41
).copy()

sample_retained["Group"] = "Retained"

sample = pd.concat(
    [sample_exited, sample_retained],
    ignore_index=True
)


# STEP 3: Establish a reference trading calendar

spy = yf.download(
    "SPY",
    start="2022-01-04",
    end="2025-01-02",
    auto_adjust=True,
    progress=False
)

if spy.empty:
    raise ValueError("Unable to obtain SPY calendar")

spy_dates = pd.DatetimeIndex(
    spy.index
).tz_localize(None).normalize()


# STEP 4: Check historical price coverage

records = []

for _, row in sample.iterrows():

    original_symbol = row["symbol"]

    # Yahoo often uses hyphens for share classes.
    yahoo_symbol = original_symbol.replace(".", "-")

    removed_date = pd.to_datetime(
        row.get("date_removed"),
        errors="coerce"
    )

    eligible_end = END

    if pd.notna(removed_date):
        eligible_end = min(END, removed_date)

    expected_dates = spy_dates[
        (spy_dates >= START)
        & (spy_dates < eligible_end)
    ]

    print(f"\nChecking {original_symbol}...")

    try:
        prices = yf.download(
            yahoo_symbol,
            start="2022-01-04",
            end="2025-01-02",
            auto_adjust=True,
            multi_level_index=False,
            progress=False
        )

        observed_dates = pd.DatetimeIndex(
            prices.index
        ).tz_localize(None).normalize()

        available = expected_dates.intersection(
            observed_dates
        )

        missing = expected_dates.difference(
            observed_dates
        )

        if prices.empty:
            status = "NO DATA"
        elif len(missing) > 0:
            status = "PARTIAL"
        else:
            status = "COMPLETE"

    except Exception as exc:
        status = "DOWNLOAD ERROR"
        available = pd.DatetimeIndex([])
        missing = expected_dates

        print(f"Error: {type(exc).__name__}: {exc}")

    records.append({
        "Ticker": original_symbol,
        "Group": row["Group"],
        "Expected_Sessions": len(expected_dates),
        "Available_Sessions": len(available),
        "Missing_Sessions": len(missing),
        "Status": status
    })


# STEP 5: Save the audit results

audit = pd.DataFrame(records)

print("\nHISTORICAL PRICE COVERAGE")

print(
    audit.sort_values(
        ["Group", "Ticker"]
    ).to_string(index=False)
)

print("\nDATA QUALITY SUMMARY")

print(audit["Status"].value_counts())

audit.to_csv(
    REPORTS / "historical_price_coverage.csv",
    index=False
)

print("\nAudit completed.")
print("No missing stocks were silently removed.")
