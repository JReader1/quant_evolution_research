"""GDELT 1.0 daily *event* archive collector.

The archives measure machine-coded coverage, not stock-specific news sentiment.
Retrospective downloaded archives are NOT certified point-in-time histories.
No trade signal or champion promotion is permitted by this module.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import json
import time
import zipfile

import pandas as pd

BASE_URL = "https://data.gdeltproject.org/events"
# 1.0 daily event row: 58 tab-delimited columns, zero-based indices.
# https://www.gdeltproject.org/data.html -> GDELT 1.0 data format.
FIELDS = {
    1: "SQLDATE", 28: "EventRootCode", 29: "QuadClass",
    30: "GoldsteinScale", 33: "NumArticles", 34: "AvgTone",
    56: "DATEADDED",
}
USECOLS = sorted(FIELDS)
NEWS_LAG_CALENDAR_DAYS = 2  # Conservative *assumption*; NOT archival release proof
MAX_ARCHIVE_BYTES = 35 * 1024 * 1024


def archive_name(day: date) -> str:
    return f"{day:%Y%m%d}.export.CSV.zip"


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def fetch_archive(root: Path, day: date, *, offline: bool = False,
                  attempts: int = 3) -> tuple[Path, dict]:
    """Cache exact ZIP bytes and retrieval metadata; never fabricate missing days."""
    folder = root / "data" / "news" / "gdelt_events_1_0" / "raw"
    folder.mkdir(parents=True, exist_ok=True)
    filename = archive_name(day)
    path = folder / filename
    metadata_path = folder / (filename + ".metadata.json")
    url = f"{BASE_URL}/{filename}"

    if path.exists():
        if not zipfile.is_zipfile(path):
            raise ValueError(f"Corrupt cached ZIP: {path}")
        # Someone might have copied the file manually. Timestamp accordingly.
        if not metadata_path.exists():
            _write_json(metadata_path, {
                "source_url": url, "retrieved_utc": None,
                "provenance_note": "Preexisting file; retrieval time unknown",
                "sha256": sha256(path.read_bytes()).hexdigest(),
            })
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if sha256(path.read_bytes()).hexdigest() != metadata.get("sha256"):
            raise ValueError(f"Cached archive hash mismatch: {path}")
        return path, metadata

    if offline:
        raise FileNotFoundError(f"Offline cache miss: {filename}. Run without --offline.")

    error = None
    for i in range(attempts):
        temp_path = None
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 (personal research project)"})
            with urlopen(req, timeout=35) as response:
                header_size = response.headers.get("Content-Length")
                if header_size and int(header_size) > MAX_ARCHIVE_BYTES:
                    raise ValueError(f"Archive too large: {header_size} bytes")
                with NamedTemporaryFile(mode="wb", dir=folder, prefix="gdelt_",
                                        suffix=".part", delete=False) as output:
                    temp_path = Path(output.name)
                    size = 0
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > MAX_ARCHIVE_BYTES:
                            raise ValueError("Archive exceeds safe download limit")
                        output.write(chunk)
            if not zipfile.is_zipfile(temp_path):
                raise ValueError("Server did not return a ZIP archive")
            # Ensure every downloaded ZIP passes integrity checks before caching it.
            with zipfile.ZipFile(temp_path) as zf:
                if zf.testzip() is not None:
                    raise ValueError("Damaged ZIP archive")
            checksum = sha256(temp_path.read_bytes()).hexdigest()
            temp_path.replace(path)
            metadata = {"source_url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                        "sha256": checksum, "archive_date": day.isoformat()}
            _write_json(metadata_path, metadata)
            return path, metadata
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            error = exc
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            if i + 1 < attempts:
                time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"Failed to download {url} after {attempts} attempts: {error}")


def summarize_archive(path: Path, day: date) -> dict:
    """Stream tab-separated GDELT events, validate schema and aggregate per archive day."""
    with zipfile.ZipFile(path) as zf:
        files = [m for m in zf.infolist() if not m.is_dir()]
        if len(files) != 1:
            raise ValueError(f"Expected one GDELT data file in {path.name}")
        member = files[0]
        if member.file_size > 250 * 1024 * 1024:
            raise ValueError("Uncompressed file exceeds safe size limit")
        with zf.open(member) as stream:
            first = stream.readline()
            if first.count(b"\t") != 57:
                raise ValueError(f"Expected 58 tab-delimited fields; got {first.count(bytes([9])) + 1}")

        total = conflict = material = bad_quad = 0
        tone_total = goldstein_total = 0.0
        tone_n = goldstein_n = 0
        article_sum = 0
        mismatched_added = 0
        with zf.open(member) as stream:
            chunks = pd.read_csv(stream, sep="\t", header=None,
                                 usecols=USECOLS, dtype="string", chunksize=40_000,
                                 encoding="utf-8", encoding_errors="replace",
                                 on_bad_lines="error", low_memory=False)
            for chunk in chunks:
                chunk = chunk.rename(columns=FIELDS)
                n = len(chunk)
                total += n
                added = chunk["DATEADDED"].str.strip()
                # Daily export files should be date-indexed. Nonmatching rows
                # are visible in metadata, rather than ignored or backdated.
                mismatched_added += int((added != day.strftime("%Y%m%d")).fillna(True).sum())
                quad = pd.to_numeric(chunk["QuadClass"], errors="coerce")
                conflict += int(quad.isin([3, 4]).sum())
                material += int((quad == 4).sum())
                bad_quad += int((~quad.isin([1, 2, 3, 4])).sum())
                tone = pd.to_numeric(chunk["AvgTone"], errors="coerce")
                tone_total += float(tone.sum())
                tone_n += int(tone.count())
                gold = pd.to_numeric(chunk["GoldsteinScale"], errors="coerce")
                goldstein_total += float(gold.sum())
                goldstein_n += int(gold.count())
                articles = pd.to_numeric(chunk["NumArticles"], errors="coerce")
                article_sum += int(articles.clip(lower=0).fillna(0).sum())

    if total == 0:
        raise ValueError(f"Empty archive: {path}")
    if mismatched_added:
        raise ValueError(f"DATEADDED mismatch on {mismatched_added} rows of {path.name}; investigate")
    if bad_quad > total // 100:
        raise ValueError(f"More than 1% invalid QuadClass values: {path.name}")
    return {
        "archive_date": day.isoformat(),
        "assumed_usable_from_utc": (day + timedelta(days=NEWS_LAG_CALENDAR_DAYS)).isoformat() + "T12:00:00Z",
        "events": total,
        "conflict_events": conflict,
        "material_conflict_events": material,
        "conflict_share": conflict / total,
        "material_conflict_share": material / total,
        "mean_tone": tone_total / tone_n if tone_n else None,
        "mean_goldstein": goldstein_total / goldstein_n if goldstein_n else None,
        "sum_num_articles": article_sum,
        "invalid_quad_rows": bad_quad,
        "dateadded_mismatches": mismatched_added,
    }


def align_to_market_dates(features: pd.DataFrame, market_dates, *, window=7, lag_days=2) -> pd.DataFrame:
    """Only join a full consecutive calendar window; never equate missing days to zero."""
    if window < 1 or lag_days < NEWS_LAG_CALENDAR_DAYS:
        raise ValueError("Window must be positive and lag must be >= conservative two-day embargo")
    data = features.copy()
    data["archive_date"] = pd.to_datetime(data["archive_date"], errors="raise")
    if data["archive_date"].duplicated().any():
        raise ValueError("Duplicate GDELT archive dates")
    data = data.set_index("archive_date").sort_index()
    rows = []
    for market_date in pd.to_datetime(market_dates):
        d = pd.Timestamp(market_date).normalize()
        right = d - pd.Timedelta(days=lag_days)
        expected = pd.date_range(end=right, periods=window, freq="D")
        selected = data.reindex(expected)
        if selected["events"].isna().any():
            rows.append({"signal_date": d.date().isoformat(), "news_coverage": "INCOMPLETE",
                         "news_event_count": None, "news_conflict_share": None,
                         "news_material_conflict_share": None, "news_mean_tone": None})
            continue
        total = float(selected["events"].sum())
        rows.append({
            "signal_date": d.date().isoformat(), "news_coverage": "COMPLETE",
            "news_event_count": int(total),
            "news_conflict_share": float(selected["conflict_events"].sum() / total),
            "news_material_conflict_share": float(selected["material_conflict_events"].sum() / total),
            "news_mean_tone": float((selected["mean_tone"] * selected["events"]).sum() / total),
        })
    return pd.DataFrame(rows)


def collect(root: Path, start: date, days: int, *, offline: bool = False) -> tuple[Path, pd.DataFrame]:
    """Run one small auditable collection. Returns only when ALL requested dates succeeded."""
    if not 1 <= days <= 31:
        raise ValueError("1–31 calendar days per batch; use cache and additional runs for more history")
    if start < date(2013, 4, 1):
        raise ValueError("GDELT 1.0 daily export starts April 1, 2013")
    if start + timedelta(days=days) > datetime.now(timezone.utc).date():
        raise ValueError("Only completed historical days are allowed")
    root = Path(root)
    records = []
    print(f"GDELT historical event archives: {start} for {days} days")
    for i in range(days):
        day = start + timedelta(days=i)
        path, metadata = fetch_archive(root, day, offline=offline)
        record = summarize_archive(path, day)
        record["source_sha256"] = metadata["sha256"]
        record["retrieved_utc"] = metadata.get("retrieved_utc")
        record["source_url"] = metadata["source_url"]
        records.append(record)
        print(f"  {day}: {record['events']:,} events, {record['material_conflict_events']:,} material-conflict")

    features = pd.DataFrame(records)
    report_dir = root / "reports" / "news_history" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_dir.mkdir(parents=True, exist_ok=False)
    features.to_csv(report_dir / "daily_event_features.csv", index=False)
    _write_json(report_dir / "manifest.json", {
        "source": "GDELT 1.0 daily event archives", "source_url": BASE_URL,
        "archive_dates": [r["archive_date"] for r in records],
        "archives": [{"day": r["archive_date"], "sha256": r["source_sha256"],
                      "retrieved_utc": r["retrieved_utc"]} for r in records],
        "assumed_lookahead_embargo_calendar_days": NEWS_LAG_CALENDAR_DAYS,
        "point_in_time_verified": False,
        "historical_promotion_allowed": False,
        "coverage": "ALL_REQUESTED_DAYS_PRESENT",
        "warning": "Event-level news-coverage proxy; source retroactively downloaded. Not verified PIT; never use as an independent trading-signal test.",
    })
    return report_dir, features
