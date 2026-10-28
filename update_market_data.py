"""Refresh the current 33-stock market-data sample."""

from quantlab.market_data_manager import update_sample_market_data


def main() -> None:
    folder, universe_source, audit = update_sample_market_data(per_sector=3)
    successful = audit["Status"].eq("OK")

    if len(audit) != 33 or not audit["Ticker"].is_unique:
        raise ValueError("Expected 33 unique stocks")
    if audit["Sector"].nunique() != 11:
        raise ValueError("Expected three names in each of 11 sectors")

    print(f"Universe source: {universe_source}")
    print(f"Market data: {folder}")
    print(f"Successful downloads: {successful.sum()} / {len(audit)}")

    if not successful.all():
        print(audit.loc[~successful, ["Ticker", "Status", "Reason"]].to_string(index=False))
        raise RuntimeError("Some price histories could not be updated")


if __name__ == "__main__":
    main()
