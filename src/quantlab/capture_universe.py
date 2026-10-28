
from quantlab.universe_manager import capture_universe
from quantlab.universe import SECTOR_BY_TICKER


print("\nCAPTURING STOCK UNIVERSE")

metadata, universe = capture_universe()

print("\nUNIVERSE SNAPSHOT")

print("Captured UTC:", metadata["captured_utc"])
print("Membership date:", metadata["membership_date"])
print("Active stocks:", metadata["active_symbols"])
print("Sectors:", metadata["sectors"])

print("\nSTOCKS BY SECTOR")

print(
    universe.groupby("sector")["symbol"]
    .nunique()
    .to_string()
)

original = set(SECTOR_BY_TICKER)

active = set(universe["symbol"])

print("\nOriginal 33 still members:",
      len(original & active))

print("Original stocks no longer members:",
      sorted(original - active))

print("\nSnapshot saved:")
print(metadata["universe_file"])

print("\nPASS: Universe snapshot captured")
