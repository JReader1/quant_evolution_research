"""Save a snapshot of the current constituent universe."""

from quantlab.universe import SECTOR_BY_TICKER
from quantlab.universe_manager import capture_universe


def main() -> None:
    metadata, universe = capture_universe()

    original = set(SECTOR_BY_TICKER)
    current = set(universe["symbol"])

    print(f"Snapshot: {metadata['universe_file']}")
    print(f"Captured: {metadata['captured_utc']}")
    print(f"Membership date: {metadata['membership_date']}")
    print(f"Active securities: {metadata['active_symbols']}")
    print(f"Sectors: {metadata['sectors']}")
    print(f"Original 33 still listed: {len(original & current)}")
    print(f"Original names no longer listed: {sorted(original - current)}")


if __name__ == "__main__":
    main()
