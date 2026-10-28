
from pathlib import Path

import pandas as pd
import yfinance as yf



# The script lives inside Quant Research/Audits/
# Go up one directory to reach the project root.

ROOT = Path(__file__).resolve().parent.parent

REPORTS = ROOT / "reports"


START = pd.Timestamp("2022-01-04")
END = pd.Timestamp("2025-01-02")

# Verified historical ticker changes
RENAMES = {
    "BLL": ("BALL", "2022-05-10"),
    "RE": ("EG", "2023-07-10"),
    "CDAY": ("DAY", "2024-02-01")
}


# STEP 1: Load historical membership

members = pd.read_csv(
    REPORTS / "sp500_membership_2022-01-03.csv",
    dtype={"symbol": str}
)

members["date_removed"] = pd.to_datetime(
    members["date_removed"],
    errors="coerce"
)


# STEP 2: Establish expected trading sessions

spy = yf.download(
    "SPY",
    start=START.strftime("%Y-%m-%d"),
    end=END.strftime("%Y-%m-%d"),
    auto_adjust=True,
    multi_level_index=False,
    progress=False
)

if spy.empty:
    raise ValueError("SPY calendar unavailable")

calendar = pd.DatetimeIndex(spy.index)

if calendar.tz is not None:
    calendar = calendar.tz_localize(None)

calendar = calendar.normalize()


# STEP 3: Investigate each ticker change

records = []

print("\nHISTORICAL TICKER RENAME AUDIT")

for old, (new, change_date) in RENAMES.items():

    matching = members[
        members["symbol"] == old
    ]

    if len(matching) != 1:
        raise ValueError(
            f"Expected one historical record for {old}"
        )

    removed = matching.iloc[0]["date_removed"]

    expected_end = END

    if pd.notna(removed):
        expected_end = min(END, removed)

    expected = calendar[
        (calendar >= START)
        & (calendar < expected_end)
    ]

    print(f"\nHistorical ticker: {old}")
    print(f"Replacement ticker: {new}")
    print(f"Symbol change: {change_date}")
    print(f"Membership end: {expected_end.date()}")

    try:
        prices = yf.download(
            new,
            start=START.strftime("%Y-%m-%d"),
            end=END.strftime("%Y-%m-%d"),
            auto_adjust=True,
            multi_level_index=False,
            progress=False
        )

        observed = pd.DatetimeIndex(
            prices.index
        )

        if observed.tz is not None:
            observed = observed.tz_localize(None)

        observed = observed.normalize()

    except Exception as exc:
        print(f"Download error: {exc}")
        observed = pd.DatetimeIndex([])

    available = expected.intersection(observed)
    missing = expected.difference(observed)

    status = (
        "COVERED"
        if len(missing) == 0 and len(expected) > 0
        else "MISSING DATA"
    )

    print(f"Expected sessions: {len(expected)}")
    print(f"Available sessions: {len(available)}")
    print(f"Missing sessions: {len(missing)}")
    print(f"Status: {status}")

    records.append({
        "Historical_Ticker": old,
        "Download_Ticker": new,
        "Change_Date": change_date,
        "Expected_Sessions": len(expected),
        "Available_Sessions": len(available),
        "Missing_Sessions": len(missing),
        "Status": status
    })


# STEP 4: Save the findings

audit = pd.DataFrame(records)

audit.to_csv(
    REPORTS / "ticker_rename_audit.csv",
    index=False
)

print("\nTICKER RENAME SUMMARY")
print(audit.to_string(index=False))

print("\nAudit completed.")
