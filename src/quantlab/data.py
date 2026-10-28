
from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TICKERS = [
    # Technology
    "AAPL", "MSFT", "CSCO",

    # Financials
    "JPM", "BAC", "GS",

    # Healthcare
    "JNJ", "PFE", "MRK",

    # Energy
    "XOM", "CVX", "COP",

    # Consumer Staples
    "WMT", "PG", "KO",

    # Consumer Discretionary
    "AMZN", "HD", "MCD",

    # Industrials
    "CAT", "DE", "HON",

    # Communication Services
    "GOOGL", "DIS", "VZ",

    # Utilities
    "NEE", "DUK", "SO",

    # Materials
    "NUE", "NEM", "APD",

    # Real Estate
    "PLD", "AMT", "SPG"
]



def download_stock_data(end_date="2025-01-01"):

    # Keep the original research data separate
    # from the new 2025 evaluation dataset.
    if end_date == "2025-01-01":
        folder_name = "raw"

    elif end_date == "2026-01-01":
        folder_name = "raw_2018_2025"

    else:
        raise ValueError(
            f"Unsupported end date: {end_date}"
        )

    all_stocks = []

    raw_folder = (
        PROJECT_ROOT / "data" / folder_name
    )

    raw_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    for ticker in TICKERS:

        file_path = raw_folder / f"{ticker}.csv"

        if file_path.exists():

            print(f"Loading saved {ticker} data...")

            stock = pd.read_csv(
                file_path,
                parse_dates=["Date"]
            )

        else:

            print(f"Downloading {ticker}...")

            stock = yf.download(
                ticker,
                start="2018-01-01",
                end=end_date,
                interval="1d",
                auto_adjust=True,
                multi_level_index=False,
                progress=False
            )

            if stock.empty:
                raise ValueError(
                    f"No data downloaded for {ticker}"
                )

            stock = stock.reset_index()

            stock.to_csv(
                file_path,
                index=False
            )

        stock["Date"] = pd.to_datetime(
            stock["Date"]
        )

        stock = stock.sort_values("Date")

        if stock["Date"].duplicated().any():
            raise ValueError(
                f"Duplicate dates for {ticker}"
            )

        if end_date == "2026-01-01":

            latest_date = stock["Date"].max()

            if latest_date < pd.Timestamp("2025-12-15"):
                raise ValueError(
                    f"Incomplete 2025 data for "
                    f"{ticker}: {latest_date}"
                )

        stock["Ticker"] = ticker

        all_stocks.append(stock)

    df = pd.concat(
        all_stocks,
        ignore_index=True
    )

    df = df.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    return df

