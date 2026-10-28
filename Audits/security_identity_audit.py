
from pathlib import Path
import pandas as pd

# The script is inside Quant Research/Audits/
ROOT = Path(__file__).resolve().parent.parent

REPORTS = ROOT / "reports"
REFERENCE = ROOT / "data" / "reference"

# Load the results from Lesson 44
audit = pd.read_csv(
    REPORTS / "ticker_rename_audit.csv"
)

# Load the preserved historical index membership
history = pd.read_csv(
    REFERENCE / "sp500_components_history.csv",
    dtype="string"
)

# These are internal labels, not official security IDs.
identities = {
    "BLL": "BALL_CORPORATION",
    "RE": "EVEREST_GROUP",
    "CDAY": "DAYFORCE"
}

records = []

print("\nSECURITY IDENTITY AUDIT")

for _, row in audit.iterrows():

    old = row["Historical_Ticker"]
    new = row["Download_Ticker"]

    old_records = history[
        history["symbol"] == old
    ]

    new_records = history[
        history["symbol"] == new
    ]

    if old_records.empty:
        raise ValueError(
            f"Historical symbol missing: {old}"
        )

    # Complete calendar coverage is necessary,
    # but does not prove price accuracy.
    covered = row["Status"] == "COVERED"

    records.append({
        "Local_Identity_Key": identities[old],
        "Old_Ticker": old,
        "New_Ticker": new,
        "Change_Date": row["Change_Date"],
        "Old_Membership_Records": len(old_records),
        "New_Membership_Records": len(new_records),
        "Expected_Sessions": row["Expected_Sessions"],
        "Available_Sessions": row["Available_Sessions"],
        "Missing_Sessions": row["Missing_Sessions"],
        "Price_Coverage_Verified": covered,
        "Requires_Investigation": not covered
    })

    print(f"\nCompany: {identities[old]}")
    print(f"Old symbol: {old}")
    print(f"New symbol: {new}")
    print(f"Historical coverage: {row['Status']}")
    print(f"New membership records: {len(new_records)}")


manifest = pd.DataFrame(records)

manifest.to_csv(
    REPORTS / "security_identity_manifest.csv",
    index=False
)

print("\nRESOLUTION SUMMARY")

print(
    manifest[[
        "Old_Ticker",
        "New_Ticker",
        "Price_Coverage_Verified",
        "Requires_Investigation"
    ]].to_string(index=False)
)

unresolved = manifest[
    manifest["Requires_Investigation"]
]

print("\nUnresolved historical identities:",
      len(unresolved))

if not unresolved.empty:
    print(
        "WARNING: Do not treat this sample as "
        "a fully resolved historical price universe."
    )

print("\nSecurity identity report saved.")
