"""Regression checks for the historical portfolio simulator.

All prices are synthetic; tests do not require downloaded market data.
"""

import numpy as np
import pandas as pd
import pytest

from quantlab.evolution import frozen_2026, macro_regimes, market_analogs
from quantlab.evolution.arena import BASELINE, HOLD_SESSIONS, Genome, HistoricalArena
from quantlab.universe import SECTOR_BY_TICKER


@pytest.fixture
def arena(tmp_path):
    folder = tmp_path / "data" / "raw_2018_2025"
    folder.mkdir(parents=True)
    dates = pd.bdate_range("2019-01-02", periods=180)
    prices = 100 + 0.15 * np.arange(len(dates))
    for ticker in SECTOR_BY_TICKER:
        pd.DataFrame({"Date": dates, "Open": prices, "Close": prices}).to_csv(
            folder / f"{ticker}.csv", index=False
        )
    return HistoricalArena(tmp_path, cost_bps=0.0)


def test_identical_prices_match_equal_weight_control(arena):
    selected = arena.evaluate(BASELINE, "2019-04-01", "2019-07-31")
    benchmark = arena.evaluate(None, "2019-04-01", "2019-07-31")
    assert selected["periods"] > 0
    assert selected["return"] == pytest.approx(benchmark["return"], abs=1e-10)


def test_trades_follow_the_signal_at_next_open(arena):
    start, end = "2019-04-01", "2019-07-31"
    result = arena.evaluate(BASELINE, start, end, record_trades=True)
    trade = result["trades"][0]
    signal_index = int(arena.calendar.searchsorted(pd.Timestamp(start)))
    assert trade["signal_date"] == str(arena.calendar[signal_index].date())
    assert trade["entry_date"] == str(arena.calendar[signal_index + 1].date())
    assert trade["exit_date"] == str(arena.calendar[signal_index + 22].date())
    assert len(trade["selected"].split(";")) == 11


def test_future_closes_do_not_change_past_signal(arena):
    i = int(arena.calendar.searchsorted(pd.Timestamp("2019-04-01")))
    before = arena.scores(BASELINE)[i].copy()
    arena.closes[i + 1 :] *= 3
    arena._score_cache.clear()
    np.testing.assert_allclose(arena.scores(BASELINE)[i], before)


def test_risk_schedule_requires_every_decision_date(arena):
    with pytest.raises(ValueError, match="Missing risk target"):
        arena.evaluate(BASELINE, "2019-04-01", "2019-07-31", risk_schedule={})


def test_invalid_genome_rejected():
    with pytest.raises(ValueError, match="Invalid momentum skip"):
        Genome(lookback=5, skip=5).validate()


def test_experiment_modules_are_importable():
    assert callable(market_analogs.preview_market_analogs)
    assert callable(macro_regimes.compare_macro_analogs)
    assert callable(frozen_2026.evaluate_frozen_2026)


def test_transaction_costs_reduce_net_returns(arena):
    start, end = "2019-04-01", "2019-07-31"
    no_cost = arena.evaluate(BASELINE, start, end)
    with_cost = HistoricalArena(arena.root, cost_bps=5.0).evaluate(BASELINE, start, end)
    assert with_cost["total_fees"] > 0
    assert with_cost["return"] < no_cost["return"]


def test_full_risk_schedule_matches_original_evaluation(arena):
    start, end = "2019-04-01", "2019-07-31"
    first = int(arena.calendar.searchsorted(pd.Timestamp(start)))
    last = int(arena.calendar.searchsorted(pd.Timestamp(end)))
    schedule = {
        arena.calendar[i]: 1.0
        for i in range(first, last - HOLD_SESSIONS - 1, HOLD_SESSIONS)
    }
    original = arena.evaluate(BASELINE, start, end)
    scheduled = arena.evaluate(BASELINE, start, end, risk_schedule=schedule)
    assert scheduled["return"] == pytest.approx(original["return"], abs=1e-12)
    assert scheduled["max_drawdown"] == pytest.approx(original["max_drawdown"], abs=1e-12)
    assert scheduled["total_fees"] == pytest.approx(original["total_fees"], abs=1e-12)
