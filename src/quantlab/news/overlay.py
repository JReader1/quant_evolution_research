"""Exploratory market-wide news risk overlay for the existing equity research arena.

Existing championship code is NEVER modified. Use the same date-based GDELT
7-day features, a 2-CALENDAR-day embargo, and next-open equity execution.
Historical GDELT archives are retrospectively obtained, not authenticated
point-in-time signal data. Never use this for promotion or real trades.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np
import pandas as pd

from quantlab.evolution.arena import HOLD_SESSIONS, Genome, HistoricalArena


@dataclass(frozen=True)
class OverlayRule:
    # Precommitted research rule; not fitted to the currently available 2022 data.
    warmup_sessions: int = 20
    threshold_z: float = 1.0
    stressed_risk_multiplier: float = 0.75

    def validate(self):
        if not 10 <= self.warmup_sessions <= 252:
            raise ValueError('Warmup needs 10–252 prior complete market dates')
        if not np.isfinite(self.threshold_z) or self.threshold_z < 0:
            raise ValueError('Threshold must be a nonnegative finite number')
        if not 0 <= self.stressed_risk_multiplier <= 1:
            raise ValueError('Risk multiplier must be in [0, 1]')


def prepare_news_schedule(aligned: pd.DataFrame, rule: OverlayRule) -> pd.DataFrame:
    """Use only past COMPLETE lagged news windows to calculate risk signals.

    Signal t is compared with a rolling history ending at t-1. Missing dates
    are never treated as zero. Noncomplete observations cannot enter a signal.
    """
    rule.validate()
    required = {'signal_date', 'news_coverage', 'news_material_conflict_share'}
    if not required.issubset(aligned.columns):
        raise ValueError(f'Missing news-alignment columns: {sorted(required - set(aligned.columns))}')
    data = aligned.copy()
    data['signal_date'] = pd.to_datetime(data['signal_date'], errors='raise').dt.normalize()
    if data['signal_date'].duplicated().any():
        raise ValueError('Duplicate news-market signal dates')
    data = data.sort_values('signal_date').reset_index(drop=True)
    if not data['news_coverage'].isin(['COMPLETE', 'INCOMPLETE']).all():
        raise ValueError('Invalid news coverage labels')
    data['news_material_conflict_share'] = pd.to_numeric(
        data['news_material_conflict_share'], errors='coerce'
    )
    full = data['news_coverage'].eq('COMPLETE')
    partial = ~full
    if data.loc[full, 'news_material_conflict_share'].isna().any():
        raise ValueError('COMPLETE news date has no numeric rate')
    if data.loc[partial, 'news_material_conflict_share'].notna().any():
        raise ValueError('INCOMPLETE news date must have missing features')
    observed = data.loc[full, 'news_material_conflict_share']
    if not observed.between(0, 1).all():
        raise ValueError('Invalid conflict share outside [0,1]')
    # Preserve missing windows in the chronological rolling history.
    rates = data['news_material_conflict_share']
    previous = rates.shift(1)
    mu = previous.rolling(rule.warmup_sessions, min_periods=rule.warmup_sessions).mean()
    sigma = previous.rolling(rule.warmup_sessions, min_periods=rule.warmup_sessions).std(ddof=0)
    z = (rates - mu) / sigma.replace(0, np.nan)
    # With a constant prior window the Z-score is undefined; retain this fact
    # in the report. A strictly higher value is still a deterministic shock.
    # Incomplete dates leave rolling stats missing rather than implying zero.
    ready = full & mu.notna() & sigma.notna()
    shock = (rates > mu) & (rates > (mu + rule.threshold_z * sigma))
    data['prior_mean'] = mu
    data['zscore_vs_prior'] = z.where(ready)
    data['risk_multiplier'] = np.nan
    data.loc[ready, 'risk_multiplier'] = np.where(
        shock.loc[ready], rule.stressed_risk_multiplier, 1.0
    )
    data['rule_ready'] = ready
    return data


def determine_experiment_bounds(arena: HistoricalArena, schedule: pd.DataFrame,
                                coverage_end_exclusive: str) -> tuple[str, str, int, list[str]]:
    """Choose fixed complete 21-session periods; require news on EVERY decision day."""
    ready = schedule.loc[schedule['rule_ready'], 'signal_date']
    if ready.empty:
        raise ValueError('Insufficient complete news history for the warmup')
    first = ready.iloc[0]
    end = pd.Timestamp(coverage_end_exclusive)
    calendar = arena.calendar
    i_start = int(calendar.searchsorted(first))
    i_end = int(calendar.searchsorted(end))
    signals = list(range(i_start, i_end - HOLD_SESSIONS - 1, HOLD_SESSIONS))
    if not signals:
        raise ValueError('No full 21-session holding interval after news warmup')
    last_exit = signals[-1] + HOLD_SESSIONS + 1
    valid = set(schedule.loc[schedule['rule_ready'], 'signal_date'].dt.strftime('%Y-%m-%d'))
    # Signal at close t can change risk only at the open t+1.
    # The last open is final liquidation, not a new risk decision.
    required = [calendar[i].strftime('%Y-%m-%d') for i in range(i_start, last_exit - 1)]
    missing = [d for d in required if d not in valid]
    if missing:
        raise ValueError(f'Missing complete daily news signals: {missing[:6]}'
                            + (' ...' if len(missing) > 6 else ''))
    signal_dates = [calendar[i].strftime('%Y-%m-%d') for i in signals]
    return first.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), len(signals), signal_dates


def evaluate_news_overlay(arena: HistoricalArena, genome: Genome, start: str, end: str,
                          risk_multipliers: Mapping[str, float],
                          *, record_trades: bool = False) -> dict:
    """Monthly stock selection, daily close news decisions, execution next open.

    A risk change scales CURRENT positions, preserving their weights. Only
    21-session rebalances choose new names/equal weights. Both operations pay
    fees on the dollar notional traded, and every order is self-financing.
    Missing news for any required decision date is an ERROR, never neutral.
    A 100%-risk schedule reproduces HistoricalArena.evaluate() exactly.
    """
    genome.validate()
    calendar = arena.calendar
    i_start = int(calendar.searchsorted(pd.Timestamp(start)))
    i_end = int(calendar.searchsorted(pd.Timestamp(end)))
    signals = list(range(i_start, i_end - HOLD_SESSIONS - 1, HOLD_SESSIONS))
    if not signals:
        raise ValueError('Not enough completed 21-session holding periods')
    last_exit = signals[-1] + HOLD_SESSIONS + 1
    required_indices = range(i_start, last_exit - 1)
    multipliers = []
    for i in required_indices:
        day = calendar[i].strftime('%Y-%m-%d')
        if day not in risk_multipliers:
            raise ValueError(f'No complete news observation for {day}')
        multiplier = float(risk_multipliers[day])
        if not np.isfinite(multiplier) or not 0 <= multiplier <= 1:
            raise ValueError(f'Invalid news risk multiplier on {day}')
        multipliers.append(multiplier)

    scores = arena.scores(genome)
    shares = np.zeros(len(arena.tickers), dtype=float)
    cash, fees = 1.0, 0.0
    daily_equity = [1.0]
    transactions = []
    prior_multiplier = None
    held = None
    extra_risk_rebalances = 0
    risk_state_transitions = 0

    # Iterate all opens where risk could be executed, not just monthly opens.
    for open_i in range(i_start + 1, last_exit):
        signal_i = open_i - 1
        signal_day = calendar[signal_i].strftime('%Y-%m-%d')
        multiplier = multipliers[signal_i - i_start]
        monthly = (signal_i - i_start) % HOLD_SESSIONS == 0
        changed = prior_multiplier is not None and multiplier != prior_multiplier
        if changed:
            risk_state_transitions += 1
        target_risk = genome.risk_fraction * multiplier
        opening_prices = arena.opens[open_i]
        equity_before = cash + float(np.dot(shares, opening_prices))
        if equity_before <= 0:
            raise AssertionError('Portfolio equity depleted')

        if monthly:
            row = scores[signal_i]
            if not np.isfinite(row).all():
                raise ValueError(f'Missing stock scores on {signal_day}')
            held = np.array([group[int(np.argmax(row[group]))]
                             for group in arena.sector_indices], dtype=int)
            target_weights = np.zeros_like(shares)
            target_weights[held] = 1.0 / len(held)
        elif changed:
            # A news shock changes gross exposure, NOT the 11 chosen names.
            # Keep drifted stock proportions; do not rebalance selection daily.
            current_dollars = shares * opening_prices
            invested_now = float(current_dollars.sum())
            target_weights = np.zeros_like(shares)
            if invested_now > 0:
                target_weights = current_dollars / invested_now
            else:
                # Possible after a prior signal reduced exposure to 0%.
                target_weights[held] = 1.0 / len(held)
            extra_risk_rebalances += 1
        else:
            target_weights = None

        if target_weights is not None:
            current_dollars = shares * opening_prices

            def turnover(invested: float) -> float:
                return float(np.abs(target_weights * invested - current_dollars).sum())

            # Invest no more than target exposure and leave enough for fees.
            low, high = 0.0, equity_before * target_risk
            for _ in range(55):
                mid = (low + high) / 2
                if mid + arena.cost_rate * turnover(mid) > equity_before:
                    high = mid
                else:
                    low = mid
            invested = (low + high) / 2
            fee = arena.cost_rate * turnover(invested)
            fees += fee
            cash = equity_before - invested - fee
            if cash < -1e-10:
                raise AssertionError('Negative cash after fees')
            if abs(cash) < 1e-10:
                cash = 0.0
            shares = (target_weights * invested) / opening_prices
            if record_trades:
                trade = {
                    'event': 'stock_rebalance' if monthly else 'news_risk_change',
                    'signal_date': signal_day,
                    'entry_date': str(calendar[open_i].date()),
                    'selected': ';'.join(arena.tickers[int(j)] for j in held),
                    'news_risk_multiplier': multiplier,
                    'effective_target_risk': target_risk,
                    'entry_fee_fraction': fee,
                }
                if monthly:
                    trade['exit_date'] = str(calendar[signal_i + HOLD_SESSIONS + 1].date())
                transactions.append(trade)

        daily_equity.append(cash + float(np.dot(shares, opening_prices)))
        prior_multiplier = multiplier

    # Mandatory final liquidation at scheduled exit open, with costs.
    liquidation = float(np.dot(shares, arena.opens[last_exit]))
    exit_fee = liquidation * arena.cost_rate
    cash += liquidation - exit_fee
    fees += exit_fee
    daily_equity.append(cash)
    equity = np.asarray(daily_equity, dtype=float)
    drawdown = float(np.min(equity / np.maximum.accumulate(equity) - 1))
    daily_returns = equity[1:] / equity[:-1] - 1
    result = {
        'return': float(cash - 1),
        'max_drawdown': drawdown,
        'annualized_volatility': float(np.std(daily_returns, ddof=1) * np.sqrt(252)),
        'periods': len(signals),
        'total_fees': float(fees),
        'first_entry': str(calendar[i_start + 1].date()),
        'last_exit': str(calendar[last_exit].date()),
    }
    if record_trades:
        result['trades'] = transactions
        result['daily_news_signal_days'] = len(multipliers)
        result['risk_off_signal_days'] = sum(m < 1 for m in multipliers)
        result['risk_state_transitions'] = risk_state_transitions
        result['extra_risk_rebalances'] = extra_risk_rebalances
    return result
