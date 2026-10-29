"""End-to-end championship test using generated prices, not market downloads."""

import json

import numpy as np
import pandas as pd

from quantlab.evolution.championship import run_championship
from quantlab.universe import SECTOR_BY_TICKER


def test_championship_writes_a_consistent_research_report(tmp_path):
    folder = tmp_path / "data" / "raw_2018_2025"
    folder.mkdir(parents=True)
    dates = pd.bdate_range("2018-01-02", "2024-12-31")
    t = np.arange(len(dates), dtype=float)

    for index, ticker in enumerate(sorted(SECTOR_BY_TICKER)):
        trend = 0.00014 + index * 0.000004
        cycle = 0.025 * np.sin(t / (19 + index % 5) + index)
        close = 90 * np.exp(trend * t + cycle)
        opened = close * (1 + 0.001 * np.cos(t / 7 + index))
        pd.DataFrame({
            "Date": dates,
            "Open": opened,
            "Close": close,
        }).to_csv(folder / f"{ticker}.csv", index=False)

    output, summary = run_championship(
        tmp_path, population=4, generations=1, seed=41, cost_bps=5.0
    )

    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["champion"]["candidate"] == summary["champion"]["candidate"]
    assert saved["seed"] == 41
    assert saved["unique_candidates_evaluated"] == 4
    assert len(saved["data_files_sha256"]) == 33
    assert saved["real_trading"] is False
    assert "NOT PROMOTED" in saved["promotion_decision"]
    assert saved["secondary_audit"]["champion"]["periods"] > 0

    for filename in (
        "generations.csv",
        "fitness_folds.csv",
        "champion_audit_trades.csv",
        "all_candidates.csv",
        "research_report.md",
    ):
        assert (output / filename).stat().st_size > 0
