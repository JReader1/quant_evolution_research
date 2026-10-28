"""One-command historical GDELT collection and safe market-date alignment.

Research only: retrospectively downloaded news features are not verified point-in-time.
Run from anywhere in Quant Research using the project virtual environment.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import zipfile

import pandas as pd


def find_root(script_path: Path) -> Path:
    path = script_path.resolve().parent
    for candidate in (path, *path.parents):
        if ((candidate / "src" / "quantlab" / "news" / "gdelt_daily.py").is_file()
                and (candidate / "run_evolution.py").is_file()):
            return candidate
    raise FileNotFoundError(
        "Cannot locate Quant Research project root. "
        "Install the add-on into your current Quant Research folder."
    )


def move_legacy_cache(root: Path) -> int:
    """Copy, never delete, valid archives previously saved under src/quantlab."""
    source = root / "src" / "quantlab" / "data" / "news" / "gdelt_events_1_0" / "raw"
    destination = root / "data" / "news" / "gdelt_events_1_0" / "raw"
    if not source.is_dir():
        return 0
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(source.glob("*.export.CSV.zip")):
        if not zipfile.is_zipfile(src):
            raise ValueError(f"Legacy archive is not a valid ZIP: {src}")
        old_bytes_hash = sha256(src.read_bytes()).hexdigest()
        src_meta = src.with_name(src.name + ".metadata.json")
        if src_meta.is_file():
            obj = json.loads(src_meta.read_text(encoding="utf-8"))
            if obj.get("sha256") != old_bytes_hash:
                raise ValueError(f"Legacy metadata SHA-256 mismatch: {src}")
        dst = destination / src.name
        if dst.is_file():
            if sha256(dst.read_bytes()).hexdigest() != old_bytes_hash:
                raise ValueError(f"Conflicting archive content at destination: {dst}")
        else:
            shutil.copy2(src, dst)
            copied += 1
        if src_meta.is_file():
            dst_meta = destination / src_meta.name
            if dst_meta.is_file():
                obj = json.loads(dst_meta.read_text(encoding="utf-8"))
                if obj.get("sha256") != old_bytes_hash:
                    raise ValueError(f"Conflicting cache metadata at destination: {dst_meta}")
            else:
                shutil.copy2(src_meta, dst_meta)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2022-02-22", help="First archive date, YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=14, help="Consecutive archive days, max 31")
    parser.add_argument("--offline", action="store_true", help="Only use locally cached ZIPs")
    args = parser.parse_args()
    try:
        root = find_root(Path(__file__))
        sys.path.insert(0, str(root / "src"))
        from quantlab.news.gdelt_daily import collect, align_to_market_dates
        start = date.fromisoformat(args.start)
        print("\nHISTORICAL NEWS ALIGNMENT — RESEARCH ONLY")
        print("Project:", root)
        migrated = move_legacy_cache(root)
        print("Reused existing legacy news archives:", migrated)
        folder, features = collect(root, start, args.days, offline=args.offline)
        # Use complete event windows ending at least two days before a trading decision.
        first_candidate = start + timedelta(days=8)
        last_candidate = start + timedelta(days=args.days + 3)
        trading_dates = pd.bdate_range(first_candidate, last_candidate)
        aligned = align_to_market_dates(features, trading_dates, window=7, lag_days=2)
        if aligned.empty:
            raise ValueError("No aligned market dates")
        n_complete = int((aligned["news_coverage"] == "COMPLETE").sum())
        n_incomplete = int((aligned["news_coverage"] == "INCOMPLETE").sum())
        if n_complete < 1:
            raise ValueError("No full seven-calendar-day news windows; collect at least 7 days")
        # Refuse changes to the previously established alignment settings.
        assert args.days >= 7, "At least seven consecutive news days required"
        report = root / "reports" / "news_alignment" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        report.mkdir(parents=True, exist_ok=False)
        aligned.to_csv(report / "market_date_news_alignment.csv", index=False)
        metadata = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "news_archive_report": str(folder),
            "start": start.isoformat(), "days": args.days,
            "window_calendar_days": 7, "assumed_lag_calendar_days": 2,
            "complete_market_dates": n_complete, "incomplete_market_dates": n_incomplete,
            "verified_point_in_time": False,
            "news_fitness_integration": "DISABLED",
            "champion_promotion": "DISABLED",
            "note": "These are global GDELT event proxies collected retrospectively; timing embargo is assumed, not verified."
        }
        with (report / "metadata.json").open("x", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print("\nMARKET-DATE NEWS ALIGNMENT")
        print(aligned.to_string(index=False, na_rep="MISSING", formatters={
            "news_conflict_share": lambda x: "MISSING" if pd.isna(x) else f"{x:.2%}",
            "news_material_conflict_share": lambda x: "MISSING" if pd.isna(x) else f"{x:.2%}",
        }))
        print("\nComplete trading-date windows:", n_complete)
        print("Incomplete trading-date windows:", n_incomplete)
        print("Report:", report / "market_date_news_alignment.csv")
        print("PASS: News-market alignment constructed; missing windows never filled with zero.")
        print("Research only: fitness and champion UNCHANGED. Historical news availability NOT verified.")
        return 0
    except (ValueError, OSError, RuntimeError, AssertionError, ImportError) as exc:
        print(f"\nNEWS ALIGNMENT FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
