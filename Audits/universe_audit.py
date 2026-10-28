
from pathlib import Path
import pandas as pd

from quantlab.data import download_stock_data
from quantlab.universe import SECTOR_BY_TICKER


# STEP 1: Load the full dataset

df = download_stock_data(
    end_date="2026-01-01"
)

df = df.sort_values(
    ["Ticker", "Date"]
)

# STEP 2: Build the universe inventory

records = []

for ticker, stock in df.groupby("Ticker"):

    before_2022 = stock[
        stock["Date"] < "2022-01-01"
    ]

    records.append({
        "Ticker": ticker,
        "Sector": SECTOR_BY_TICKER[ticker],
        "First_Date": stock["Date"].min(),
        "Last_Date": stock["Date"].max(),
        "Total_Sessions": stock["Date"].nunique(),
        "Pre_2022_Sessions": before_2022["Date"].nunique()
    })

inventory = pd.DataFrame(records)

inventory = inventory.sort_values(
    ["Sector", "Ticker"]
).reset_index(drop=True)


# STEP 3: Inspect sector composition

print("\nUNIVERSE AUDIT")

print("\nSTOCKS PER SECTOR")

sector_counts = (
    inventory.groupby("Sector")["Ticker"]
    .nunique()
)

print(sector_counts.to_string())

print("\nHISTORICAL COVERAGE")
print(inventory.to_string(index=False))

print("\nSUMMARY")

print(
    "Total stocks:",
    inventory["Ticker"].nunique()
)

print(
    "Total sectors:",
    inventory["Sector"].nunique()
)

print(
    "All stocks have pre-2022 data:",
    (inventory["Pre_2022_Sessions"] > 0).all()
)


# STEP 4: Save the inventory

report_dir = Path("../reports")
report_dir.mkdir(exist_ok=True)

inventory.to_csv(
    report_dir / "universe_inventory.csv",
    index=False
)

print("\nUniverse inventory saved.")
print(
    "NOTE: Historical price coverage does not "
    "prove that a universe was selected without "
    "future information."
)
