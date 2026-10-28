"""Collect FREE historical GDELT events; no news-based trading signals or real orders."""
import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from quantlab.news.gdelt_daily import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2022-02-22", help="YYYY-MM-DD, default is a small 2022 pilot")
    parser.add_argument("--days", default=3, type=int, help="Calendar days to collect, maximum 31")
    parser.add_argument("--offline", action="store_true", help="Use previously downloaded ZIPs only")
    args = parser.parse_args()
    try:
        start = date.fromisoformat(args.start)
        path, features = collect(ROOT, start, args.days, offline=args.offline)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"\nNEWS AUDIT FAILED: {exc}", file=sys.stderr)
        print("No feature report was committed; missing days are never filled with zeros.", file=sys.stderr)
        return 1

    print("\nHISTORICAL NEWS AUDIT: PASS")
    print("Archived days:", len(features))
    print("Events examined:", f"{features['events'].sum():,}")
    print("Material conflict events:", f"{features['material_conflict_events'].sum():,}")
    print("Report:", path / "daily_event_features.csv")
    print("IMPORTANT: Retrospectively downloaded coverage proxy, NOT verified point-in-time news.")
    print("The evolutionary champion and fitness scores are UNCHANGED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
