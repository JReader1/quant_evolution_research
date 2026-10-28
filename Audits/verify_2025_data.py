
from quantlab.data import download_stock_data
import pandas as pd

print("\nVERIFYING 2025 DATASET")

# Explicitly request the expanded dataset
df = download_stock_data(
    end_date="2026-01-01"
)

# Isolate the new evaluation year
data_2025 = df[
    (df["Date"] >= "2025-01-01") &
    (df["Date"] < "2026-01-01")
].copy()

print("\n2025 DATA COVERAGE")
print("Tickers:", data_2025["Ticker"].nunique())
print("First date:", data_2025["Date"].min())
print("Last date:", data_2025["Date"].max())
print("Total rows:", len(data_2025))

# Verify all 33 stocks are represented
assert data_2025["Ticker"].nunique() == 33

# Verify dates extend into late December
assert data_2025["Date"].max() >= pd.Timestamp(
    "2025-12-15"
)

print("\nPASS: 2025 dataset verified")
