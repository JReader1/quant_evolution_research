
from pathlib import Path
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[2]

UNIVERSE_DIR = (
    ROOT / "reports" / "universe_snapshots"
)

MARKET_DIR = ROOT / "data" / "market_snapshots"


def update_sample_market_data(per_sector=1):

    snapshots = sorted(
        UNIVERSE_DIR.glob("*_universe.csv")
    )

    if not snapshots:
        raise FileNotFoundError(
            "No captured universe snapshot found"
        )

    universe_file = snapshots[-1]

    universe = pd.read_csv(universe_file)

    required = {"symbol", "sector"}

    if not required.issubset(universe.columns):
        raise ValueError("Invalid universe snapshot")

    if universe["symbol"].duplicated().any():
        raise ValueError("Duplicate stock symbols")

    sample = (
        universe.sort_values(["sector", "symbol"])
        .groupby("sector", sort=True)
        .head(per_sector)
    )

    captured = datetime.now(timezone.utc)

    run_id = captured.strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    run_dir = MARKET_DIR / run_id

    run_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    # Reject unfinished daily candles.
    local_today = datetime.now(
        ZoneInfo("America/New_York")
    ).date()

    cutoff = pd.Timestamp(local_today)

    records = []

    for _, row in sample.iterrows():

        symbol = str(row["symbol"])
        sector = str(row["sector"])

        provider_symbol = symbol.replace(".", "-")

        print(f"Fetching {symbol}...")

        result = {
            "Ticker": symbol,
            "Provider_Ticker": provider_symbol,
            "Sector": sector,
            "Status": "ERROR",
            "Sessions": 0,
            "Last_Date": "",
            "Reason": ""
        }

        try:
            prices = yf.download(
                provider_symbol,
                period="6mo",
                interval="1d",
                auto_adjust=True,
                multi_level_index=False,
                progress=False,
                threads=False
            )

            if prices.empty:
                raise ValueError("No prices returned")

            prices.index = pd.to_datetime(
                prices.index
            )

            if prices.index.tz is not None:
                prices.index = (
                    prices.index
                    .tz_convert("America/New_York")
                    .tz_localize(None)
                )

            prices.index = prices.index.normalize()

            prices = prices.loc[
                prices.index < cutoff
            ].sort_index()

            columns = [
                "Open", "High", "Low",
                "Close", "Volume"
            ]

            if not set(columns).issubset(prices.columns):
                raise ValueError("Missing OHLCV columns")

            if len(prices) < 30:
                raise ValueError(
                    "Fewer than 30 completed sessions"
                )

            if prices.index.duplicated().any():
                raise ValueError("Duplicate price dates")

            if prices[columns].isna().any().any():
                raise ValueError("Missing price values")

            if (prices[
                ["Open", "High", "Low", "Close"]
            ] <= 0).any().any():
                raise ValueError("Nonpositive prices")

            last_date = prices.index[-1].date()

            is_stale = (
                local_today - last_date
            ) > timedelta(days=7)

            prices.index.name = "Date"

            prices = prices.reset_index()

            prices["Ticker"] = symbol

            filename = (
                symbol.replace(".", "_") + ".csv"
            )

            prices.to_csv(
                run_dir / filename,
                index=False
            )

            result["Status"] = (
                "STALE" if is_stale else "OK"
            )

            result["Sessions"] = len(prices)
            result["Last_Date"] = str(last_date)

            if is_stale:
                result["Reason"] = (
                    "Latest observation is too old"
                )

        except Exception as exc:
            result["Reason"] = (
                f"{type(exc).__name__}: {exc}"
            )

        records.append(result)

    audit = pd.DataFrame(records)

    audit.to_csv(
        run_dir / "download_manifest.csv",
        index=False
    )

    return run_dir, universe_file, audit
