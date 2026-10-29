"""Historical portfolio simulator for a fixed 33-stock equity basket.

Signals use adjusted closing prices. Portfolios trade at the following open,
rebalance every 21 sessions, and pay fees on the amount traded.

The stocks were selected retrospectively; these results are not a point-in-time
S&P 500 backtest.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import hashlib

import numpy as np
import pandas as pd

from quantlab.universe import SECTOR_BY_TICKER

HOLD_SESSIONS = 21
SECTORS = sorted(set(SECTOR_BY_TICKER.values()))
EXPECTED_TICKERS = sorted(SECTOR_BY_TICKER)


@dataclass(frozen=True, order=True)
class Genome:
    """Momentum parameters and target equity exposure."""
    lookback: int = 5
    skip: int = 0
    volatility_penalty: float = 0.0
    risk_fraction: float = 1.0
    # News factor deliberately excluded until timestamp-safe data exists.

    def validate(self) -> None:
        if self.lookback not in (5, 10, 20, 40, 63, 126):
            raise ValueError(f"Unsupported lookback: {self.lookback}")
        if self.skip not in (0, 5, 21) or self.skip >= self.lookback:
            raise ValueError("Invalid momentum skip")
        if self.volatility_penalty not in (0.0, 0.5, 1.0):
            raise ValueError("Unsupported volatility penalty")
        if self.risk_fraction not in (0.75, 1.0):
            raise ValueError("Unsupported equity exposure")

    @property
    def name(self) -> str:
        return f"m{self.lookback}_skip{self.skip}_vol{self.volatility_penalty:g}_risk{self.risk_fraction:g}"


BASELINE = Genome(lookback=5, skip=0, volatility_penalty=0.0, risk_fraction=1.0)


class HistoricalArena:
    def __init__(self, project_root: Path, cost_bps: float = 5.0):
        self.root = Path(project_root).resolve()
        self.cost_rate = cost_bps / 10000.0
        if not 0 <= cost_bps <= 100:
            raise ValueError("Cost must be 0-100 basis points")
        folder = self.root / 'data' / 'raw_2018_2025'
        panels = []
        self.input_hashes = {}
        for ticker in EXPECTED_TICKERS:
            path = folder / (ticker + '.csv')
            if not path.exists():
                raise FileNotFoundError(f"Missing offline input: {path}")
            self.input_hashes[ticker] = hashlib.sha256(path.read_bytes()).hexdigest()
            stock = pd.read_csv(path, parse_dates=['Date'])
            for col in ('Date', 'Open', 'Close'):
                if col not in stock:
                    raise ValueError(f"{ticker}: missing {col}")
            if stock['Date'].isna().any() or stock['Date'].duplicated().any():
                raise ValueError(f"{ticker}: bad trading dates")
            if stock[['Open', 'Close']].isna().any().any():
                raise ValueError(f"{ticker}: missing price")
            if not np.isfinite(stock[['Open', 'Close']].to_numpy(dtype=float)).all():
                raise ValueError(f"{ticker}: nonfinite prices")
            if (stock[['Open', 'Close']] <= 0).any().any():
                raise ValueError(f"{ticker}: nonpositive prices")
            stock['Ticker'] = ticker
            panels.append(stock[['Date', 'Ticker', 'Open', 'Close']])
        df = pd.concat(panels, ignore_index=True)
        opens = df.pivot(index='Date', columns='Ticker', values='Open').sort_index()
        closes = df.pivot(index='Date', columns='Ticker', values='Close').sort_index()
        if opens.isna().any().any() or closes.isna().any().any():
            raise ValueError("Missing/uneven trading calendars: refuse to drop a stock")
        if not opens.index.equals(closes.index) or len(opens.columns) != 33:
            raise ValueError("Unexpected panel shape")
        self.calendar = opens.index
        self.tickers = list(opens.columns)
        self.opens = opens.to_numpy(dtype=float, copy=True)
        self.closes = closes.to_numpy(dtype=float, copy=True)
        self.sector_indices = [
            np.array([self.tickers.index(t) for t in self.tickers
                      if SECTOR_BY_TICKER[t] == sector], dtype=int)
            for sector in SECTORS
        ]
        if len(self.sector_indices) != 11 or any(len(v) != 3 for v in self.sector_indices):
            raise ValueError("The provided historical cache is not 3 names per 11 sectors")
        returns = pd.DataFrame(self.closes, index=self.calendar).pct_change(fill_method=None)
        self.vol20 = returns.rolling(20, min_periods=20).std().to_numpy(dtype=float)
        self._score_cache = {}
        self._result_cache = {}

    def scores(self, genome: Genome) -> np.ndarray:
        genome.validate()
        if genome not in self._score_cache:
            c = pd.DataFrame(self.closes, index=self.calendar)
            momentum = c.shift(genome.skip) / c.shift(genome.lookback) - 1.0
            scores = momentum.to_numpy(dtype=float)
            if genome.volatility_penalty:
                scale = np.maximum(self.vol20 * np.sqrt(21.0), 1e-5)
                scores = scores / np.power(scale, genome.volatility_penalty)
            self._score_cache[genome] = scores
        return self._score_cache[genome]


    def evaluate(self, genome: Optional[Genome], start: str, end: str,
                 *, record_trades: bool = False,
                 risk_schedule: Optional[dict] = None) -> dict:

        """Run one independent window. None means all-33 equal-weight benchmark.

        The last completed exit-open must be before `end`. Each window restarts
        with $1 of cash, so fold metrics must not be compounded across folds.
        """
        if genome is not None:
            genome.validate()
        key = (
            (genome, start, end)
            if risk_schedule is None
            else None
        )
        if not record_trades and key in self._result_cache:
            return self._result_cache[key].copy()
        i_start = int(self.calendar.searchsorted(pd.Timestamp(start)))
        i_end = int(self.calendar.searchsorted(pd.Timestamp(end)))
        # Signal at i -> enter i+1 -> exit i+HOLD+1, with exit date < end
        signals = range(i_start, i_end - HOLD_SESSIONS - 1, HOLD_SESSIONS)
        if not signals:
            raise ValueError(f"Not enough full periods in {start}:{end}")
        score_panel = self.scores(genome) if genome is not None else None
        shares = np.zeros(len(self.tickers), dtype=float)
        cash = 1.0
        fees = 0.0
        daily_equity = [1.0]
        transactions = []
        prev_end_i = None
        for si in signals:
            entry_i = si + 1
            exit_i = si + HOLD_SESSIONS + 1
            if prev_end_i is not None and entry_i != prev_end_i:
                raise AssertionError("Non-contiguous rebalance calendar")
            if genome is None:
                chosen = np.arange(len(self.tickers))
                equity_target = 1.0
            else:
                row = score_panel[si]
                if not np.isfinite(row).all():
                    raise ValueError(f"Missing {genome.name} signals on {self.calendar[si].date()}")
                chosen = np.array([
                    group[int(np.argmax(row[group]))]
                    for group in self.sector_indices
                ], dtype=int)
                equity_target = genome.risk_fraction

                if risk_schedule is not None:
                    signal_date = self.calendar[si]

                    if signal_date not in risk_schedule:
                        raise ValueError(
                            f"Missing risk target: {signal_date.date()}"
                        )

                    equity_target = float(
                        risk_schedule[signal_date]
                    )

                    if not (
                        np.isfinite(equity_target)
                        and 0.75 <= equity_target <= 1.0
                    ):
                        raise ValueError(
                            "Scheduled exposure must be between 75% and 100%"
                        )

            entry_prices = self.opens[entry_i]
            value_before = cash + float(np.dot(shares, entry_prices))
            if value_before <= 0:
                raise AssertionError("Portfolio equity depleted")
            # Fee-aware root solve for invested equity; avoids negative cash.
            def notional(invested):
                desired = np.zeros_like(shares)
                desired[chosen] = invested / len(chosen)
                return float(np.abs(desired - shares * entry_prices).sum())
            lo, hi = 0.0, value_before * equity_target
            for _ in range(55):
                mid = (lo + hi) / 2
                required = mid + self.cost_rate * notional(mid)
                if required > value_before:
                    hi = mid
                else:
                    lo = mid
            invested = (lo + hi) / 2
            fee = self.cost_rate * notional(invested)
            fees += fee
            cash = value_before - invested - fee
            if cash < -1e-10:
                raise AssertionError("Negative cash")
            if abs(cash) < 1e-10:
                cash = 0.0
            shares = np.zeros_like(shares)
            shares[chosen] = (invested / len(chosen)) / entry_prices[chosen]
            if record_trades:
                transactions.append({
                    'signal_date': str(self.calendar[si].date()),
                    'entry_date': str(self.calendar[entry_i].date()),
                    'exit_date': str(self.calendar[exit_i].date()),
                    'selected': ';'.join(self.tickers[int(j)] for j in chosen),
                    'entry_fee_fraction': fee,
                })
            for di in range(entry_i, exit_i):
                daily_equity.append(cash + float(np.dot(shares, self.opens[di])))
            prev_end_i = exit_i
        last_exit = prev_end_i
        final_open = self.opens[last_exit]
        liquidation = float(np.dot(shares, final_open))
        fee = liquidation * self.cost_rate
        cash += liquidation - fee
        fees += fee
        daily_equity.append(cash)
        equity = np.asarray(daily_equity, dtype=float)
        running_peak = np.maximum.accumulate(equity)
        drawdown = float(np.min(equity / running_peak - 1))
        daily_returns = equity[1:] / equity[:-1] - 1
        result = {
            'return': float(cash - 1.0),
            'max_drawdown': drawdown,
            'annualized_volatility': float(np.std(daily_returns, ddof=1) * np.sqrt(252)),
            'periods': len(signals),
            'total_fees': float(fees),
            'first_entry': str(self.calendar[i_start + 1].date()),
            'last_exit': str(self.calendar[last_exit].date()),
        }

        if record_trades:
            result["trades"] = transactions

        if not record_trades and key is not None:
            self._result_cache[key] = result.copy()

        return result

def preview_market_analogs(project_root, as_of="2025-12-31", n=5, backtest=False):
    from quantlab.evolution.market_analogs import preview_market_analogs as run
    return run(project_root, as_of=as_of, n=n, backtest=backtest)


def evaluate_frozen_2026(project_root):
    from quantlab.evolution.frozen_2026 import evaluate_frozen_2026 as run
    return run(project_root)


def preview_macro_regimes(project_root):
    from quantlab.evolution.macro_regimes import preview_macro_regimes as run
    return run(project_root)


def compare_macro_analogs(project_root):
    from quantlab.evolution.macro_regimes import compare_macro_analogs as run
    return run(project_root)
