
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import Request, urlopen

import pandas as pd

from quantlab.universe import SECTOR_BY_TICKER


# STEP 1: Set up directories

ROOT = Path(__file__).resolve().parent

REFERENCE = ROOT / "data" / "reference"
REPORTS = ROOT / "reports"

REFERENCE.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


# STEP 2: Download and preserve source data

SOURCE = (
    "https://raw.githubusercontent.com/"
    "lawcal/sp500-components-history/"
    "main/data/components_history.csv"
)

snapshot = REFERENCE / "sp500_components_history.csv"

if not snapshot.exists():

    print("Downloading historical S&P 500 membership...")

    request = Request(
        SOURCE,
        headers={"User-Agent": "QuantResearchEducational/1.0"}
    )

    with urlopen(request, timeout=30) as response:
        content = response.read()

    if len(content) < 10000:
        raise ValueError("Unexpectedly small source file")

    snapshot.write_bytes(content)

    metadata = REFERENCE / "membership_source.txt"

    metadata.write_text(
        f"Source: {SOURCE}\n"
        f"Downloaded UTC: "
        f"{datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8"
    )

else:
    print("Using preserved membership snapshot")


# STEP 3: Read membership history

history = pd.read_csv(
    snapshot,
    dtype="string"
)

required = {
    "symbol",
    "sector",
    "date_added",
    "date_removed",
    "created_at"
}

if not required.issubset(history.columns):
    raise ValueError("Unexpected membership file format")


# Remove date footnote markers.
for column in [
    "date_added",
    "date_removed",
    "created_at"
]:
    history[column] = pd.to_datetime(
        history[column].str.replace(
            "*", "", regex=False
        ),
        errors="coerce"
    )

print("\nHISTORICAL UNIVERSE AUDIT")
print("Historical membership rows:", len(history))


# STEP 4: Find stocks active on a past date


def membership_on(date_string):

    date = pd.Timestamp(date_string)

    # Stock was added before or on this date.
    # Missing addition dates are treated as unknown.
    added = (
        history["date_added"].isna()
        | (history["date_added"] <= date)
    )

    # Stock had not yet been removed.
    removed = (
        history["date_removed"].isna()
        | (history["date_removed"] > date)
    )

    # The symbol must have been recorded by this date.
    created = (
        history["created_at"].isna()
        | (history["created_at"] <= date)
    )

    active = history.loc[
        added & removed & created
    ].copy()

    # Check for inconsistent duplicate symbol records.
    if active["symbol"].duplicated().any():
        duplicates = active.loc[
            active["symbol"].duplicated(keep=False),
            "symbol"
        ].tolist()

        raise ValueError(
            f"Duplicate historical symbols: {duplicates}"
        )

    # Ensure no included record has a future creation date.
    known_creation_dates = active["created_at"].dropna()

    assert (
        known_creation_dates <= date
    ).all()

    return active



# STEP 5: Compare three dated universes

for date_string in [
    "2022-01-03",
    "2025-01-02",
    "2026-10-23"
]:

    active = membership_on(date_string)

    print(f"\nDATE: {date_string}")
    print("Active membership rows:", len(active))
    print("Unique tickers:", active["symbol"].nunique())
    print("Sectors:", active["sector"].nunique())

    duplicate_count = active["symbol"].duplicated().sum()
    print("Duplicate symbols:", duplicate_count)

    original = set(SECTOR_BY_TICKER)
    historical = set(active["symbol"].dropna())

    print(
        "Original 33 stocks active:",
        len(original & historical)
    )

    print(
        "Original stocks not listed:",
        sorted(original - historical)
    )

    filename = (
        f"sp500_membership_{date_string}.csv"
    )

    active.to_csv(
        REPORTS / filename,
        index=False
    )

print("\nHistorical membership snapshots saved.")
print("PASS: Membership audit completed")


# STEP 42: Audit historical date completeness

print("\nMEMBERSHIP DATE QUALITY")

for column in [
    "date_added",
    "date_removed",
    "created_at"
]:
    missing = history[column].isna().sum()

    print(f"{column}: {missing} missing values")

print("\nPASS: Historical date audit completed")
