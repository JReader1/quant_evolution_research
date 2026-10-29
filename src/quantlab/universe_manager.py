
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from io import BytesIO
from hashlib import sha256
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SOURCE_URL = (
    "https://raw.githubusercontent.com/"
    "lawcal/sp500-components-history/"
    "main/data/components_history.csv"
)

ARCHIVE_DIR = ROOT / "data" / "reference" / "live_snapshots"
OUTPUT_DIR = ROOT / "reports" / "universe_snapshots"


def capture_universe():

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Snapshot the source as retrieved, not as later revised.
    request = Request(
        SOURCE_URL,
        headers={"User-Agent": "QuantResearchEducational/1.0"}
    )

    with urlopen(request, timeout=30) as response:
        raw = response.read()

    history = pd.read_csv(
        BytesIO(raw),
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
        raise ValueError("Membership schema changed")

    for column in [
        "date_added",
        "date_removed",
        "created_at"
    ]:

        cleaned = (
            history[column]
            .str.replace("*", "", regex=False)
            .str.strip()
        )

        parsed = pd.to_datetime(
            cleaned,
            errors="coerce"
        )

        invalid = cleaned.notna() & parsed.isna()

        if invalid.any():
            raise ValueError(
                f"Invalid dates in {column}"
            )

        history[column] = parsed

    captured = datetime.now(timezone.utc)
    capture_id = captured.strftime("%Y%m%dT%H%M%S%fZ")

    as_of = pd.Timestamp(captured.date())

    active = history.loc[
        (history["date_added"] <= as_of)
        &
        (
            history["date_removed"].isna()
            | (history["date_removed"] > as_of)
        )
        &
        (history["created_at"] <= as_of)
    ].copy()

    if active["symbol"].isna().any():
        raise ValueError("Missing active symbols")

    if active["symbol"].duplicated().any():
        raise ValueError("Duplicate active symbols")

    if not 450 <= len(active) <= 600:
        raise ValueError(
            "Unexpected active membership count"
        )

    if active["sector"].nunique() != 11:
        raise ValueError(
            "Unexpected number of sectors"
        )

    active = active.sort_values(
        "symbol"
    ).reset_index(drop=True)

    # Retain raw source bytes for later provenance checks.
    archive_path = (
        ARCHIVE_DIR / f"{capture_id}_source.csv"
    )

    with archive_path.open("xb") as file:
        file.write(raw)

    universe_path = (
        OUTPUT_DIR / f"{capture_id}_universe.csv"
    )

    with universe_path.open("x", encoding="utf-8") as file:
        active.to_csv(file, index=False)

    metadata = {
        "captured_utc": captured.isoformat(),
        "membership_date": str(as_of.date()),
        "source_url": SOURCE_URL,
        "source_sha256": sha256(raw).hexdigest(),
        "active_symbols": int(active["symbol"].nunique()),
        "sectors": int(active["sector"].nunique()),
        "archive_file": str(archive_path),
        "universe_file": str(universe_path)
    }

    metadata_path = (
        ARCHIVE_DIR / f"{capture_id}_metadata.json"
    )

    with metadata_path.open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return metadata, active
