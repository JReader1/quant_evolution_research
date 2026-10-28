from quantlab.market_data_manager import (
    update_sample_market_data
)


print("\nAUTOMATED MARKET DATA UPDATE")


# Collect three stocks from each of our 11 sectors.
# This is a downloader test, not a trading selection.

folder, universe_source, audit = (
    update_sample_market_data(
        per_sector=3
    )
)


print("\nSOURCE UNIVERSE")
print(universe_source)

print("\nDOWNLOAD RESULTS")

print(
    audit[[
        "Ticker",
        "Sector",
        "Status",
        "Sessions",
        "Last_Date"
    ]].to_string(index=False)
)

print("\nSTATUS SUMMARY")
print(audit["Status"].value_counts().to_string())

print("\nDATA STORAGE")
print(folder)

if (audit["Status"] == "OK").all():
    print("\nPASS: All sampled prices updated")
else:
    print("\nWARNING: Some stocks need investigation")

    # STEP 48: Validate expanded sample coverage

    print("\nEXPANDED MARKET COVERAGE AUDIT")

    expected_stocks = 33

    actual_stocks = len(audit)

    successful = (
            audit["Status"] == "OK"
    ).sum()

    failed = actual_stocks - successful

    print(f"Expected stocks: {expected_stocks}")
    print(f"Attempted stocks: {actual_stocks}")
    print(f"Successful downloads: {successful}")
    print(f"Unsuccessful or stale: {failed}")

    assert actual_stocks == expected_stocks
    assert audit["Ticker"].is_unique
    assert audit["Sector"].nunique() == 11

    if successful == expected_stocks:
        print("\nPASS: Expanded price coverage verified")
    else:
        print("\nWARNING: Investigate these tickers:")
        print(
            audit.loc[
                audit["Status"] != "OK",
                ["Ticker", "Status", "Reason"]
            ].to_string(index=False)
        )
