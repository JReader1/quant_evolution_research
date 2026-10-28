
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json

import pandas as pd


ROOT = Path(__file__).resolve().parent

MARKET_DIR = ROOT / "data" / "market_snapshots"
SIGNAL_DIR = ROOT / "reports" / "signal_snapshots"


# STEP 1: Find the latest market snapshot

snapshots = sorted(
    p for p in MARKET_DIR.iterdir()
    if p.is_dir()
)

if not snapshots:
    raise ValueError("No market snapshots available")

source_folder = snapshots[-1]

manifest_path = source_folder / "download_manifest.csv"

audit = pd.read_csv(manifest_path)

print("\nAUTOMATED MOMENTUM SIGNAL ENGINE")
print("Market snapshot:", source_folder.name)


# STEP 2: Verify source data

if len(audit) != 33:
    raise ValueError("Expected 33 stocks in pilot sample")

if audit["Ticker"].duplicated().any():
    raise ValueError("Duplicate tickers")

if audit["Sector"].nunique() != 11:
    raise ValueError("Expected 11 sectors")

if not (audit["Status"] == "OK").all():
    raise ValueError("Unresolved market data failures")

if audit["Last_Date"].nunique() != 1:
    raise ValueError("Stocks have different latest dates")


# STEP 3: Calculate five-day momentum

records = []
reference_dates = None

for _, row in audit.iterrows():

    ticker = row["Ticker"]
    sector = row["Sector"]

    filename = ticker.replace(".", "_") + ".csv"

    prices = pd.read_csv(
        source_folder / filename,
        parse_dates=["Date"]
    )

    prices = prices.sort_values("Date")

    if len(prices) < 6:
        raise ValueError(f"Insufficient data: {ticker}")

    if prices["Date"].duplicated().any():
        raise ValueError(f"Duplicate dates: {ticker}")

    if not prices["Ticker"].eq(ticker).all():
        raise ValueError(f"Ticker mismatch: {ticker}")

    recent = prices.tail(6)

    if recent["Close"].isna().any():
        raise ValueError(f"Missing prices: {ticker}")

    if (recent["Close"] <= 0).any():
        raise ValueError(f"Invalid prices: {ticker}")

    dates = recent["Date"].tolist()

    if reference_dates is None:
        reference_dates = dates
    elif dates != reference_dates:
        raise ValueError(
            f"Trading calendar mismatch: {ticker}"
        )

    current_close = float(recent["Close"].iloc[-1])
    previous_close = float(recent["Close"].iloc[0])

    momentum = current_close / previous_close - 1

    records.append({
        "Ticker": ticker,
        "Sector": sector,
        "Date": recent["Date"].iloc[-1],
        "Momentum_5d": momentum,
        "Latest_Close": current_close
    })


# STEP 4: Rank stocks within sectors

ranked = pd.DataFrame(records)

ranked = ranked.sort_values(
    ["Sector", "Momentum_5d", "Ticker"],
    ascending=[True, False, True]
).reset_index(drop=True)

ranked["Sector_Rank"] = (
    ranked.groupby("Sector").cumcount() + 1
)

assert ranked.groupby("Sector").size().eq(3).all()

leaders = ranked[
    ranked["Sector_Rank"] == 1
].copy()

assert len(leaders) == 11


# STEP 5: Save immutable prediction records

generated = datetime.now(timezone.utc)

run_id = generated.strftime(
    "%Y%m%dT%H%M%S%fZ"
)

output_folder = SIGNAL_DIR / run_id

output_folder.mkdir(
    parents=True,
    exist_ok=False
)

ranked.to_csv(
    output_folder / "all_rankings.csv",
    index=False
)

leaders.to_csv(
    output_folder / "sector_leaders.csv",
    index=False
)

metadata = {
    "generated_utc": generated.isoformat(),
    "data_date": str(ranked["Date"].iloc[0].date()),
    "market_snapshot": str(source_folder),
    "input_manifest_sha256": sha256(
        manifest_path.read_bytes()
    ).hexdigest(),
    "signal": "five_day_momentum",
    "lookback_sessions": 5,
    "candidate_stocks": len(ranked),
    "selected_stocks": len(leaders),
    "paper_only": True
}

with (
    output_folder / "metadata.json"
).open("x", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)


# STEP 6: Print results

print("\nSIGNAL GENERATION RESULTS")
print("Data date:", metadata["data_date"])
print("Stocks ranked:", len(ranked))
print("Sector leaders:", len(leaders))

print("\nMOMENTUM SECTOR LEADERS")

print(
    leaders[[
        "Sector",
        "Ticker",
        "Momentum_5d"
    ]].to_string(
        index=False,
        formatters={
            "Momentum_5d": "{:+.2%}".format
        }
    )
)

print("\nSIGNAL SNAPSHOT SAVED")
print(output_folder)

print("\nPASS: Momentum signals generated")
print("No real or hypothetical trades executed.")
