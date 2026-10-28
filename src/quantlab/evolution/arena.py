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
        self.opens = opens.to_numpy(dtype=float)
        self.closes = closes.to_numpy(dtype=float)
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
    """
    Find historically similar U.S. market environments.
    Exploratory only. Does not change trading decisions.
    """
    from io import BytesIO, StringIO
    from pathlib import Path
    from urllib.request import Request, urlopen
    import csv
    import zipfile
    import re

    import numpy as np
    import pandas as pd

    cache = (
        Path(project_root)
        / "data"
        / "reference"
        / "french_market_daily.csv"
    )

    if not cache.exists():

        url = (
            "https://mba.tuck.dartmouth.edu/pages/"
            "faculty/ken.french/ftp/"
            "F-F_Research_Data_Factors_daily_CSV.zip"
        )

        print("Downloading long-run U.S. market returns...")

        request = Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urlopen(request, timeout=40) as response:
            archive = response.read()

        with zipfile.ZipFile(BytesIO(archive)) as z:
            source_text = z.read(
                z.namelist()[0]
            ).decode("utf-8-sig", errors="replace")

        observations = []

        for row in csv.reader(StringIO(source_text)):

            if len(row) < 5:
                continue

            if not re.fullmatch(r"\d{8}", row[0].strip()):
                continue

            try:
                market_excess = float(row[1])
                risk_free = float(row[4])

                if not (
                    np.isfinite(market_excess)
                    and np.isfinite(risk_free)
                ):
                    continue

                if (
                    abs(market_excess) >= 90
                    or abs(risk_free) >= 90
                ):
                    continue

                # French values are reported in percent.
                market_return = (
                    market_excess + risk_free
                ) / 100.0

                observations.append(
                    (row[0].strip(), market_return)
                )

            except ValueError:
                continue

        if len(observations) < 10000:
            raise ValueError(
                "Unexpected historical market data"
            )

        cache.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        pd.DataFrame(
            observations,
            columns=["Date", "Market_Return"]
        ).to_csv(cache, index=False)

    market = pd.read_csv(
        cache,
        parse_dates=["Date"]
    )

    if (
        market["Date"].isna().any()
        or market["Date"].duplicated().any()
        or market["Market_Return"].isna().any()
    ):
        raise ValueError("Invalid cached market data")

    market = (
        market.sort_values("Date")
        .set_index("Date")
        .loc[:pd.Timestamp(as_of)]
    )

    if len(market) < 1000:
        raise ValueError("Not enough historical data")

    r = market["Market_Return"].astype(float)

    if ((r <= -1) | (~np.isfinite(r))).any():
        raise ValueError("Invalid historical returns")

    wealth = (1 + r).cumprod()

    features = pd.DataFrame(index=r.index)

    features["r20"] = wealth / wealth.shift(20) - 1

    features["r60"] = wealth / wealth.shift(60) - 1

    features["vol20"] = (
        r.rolling(20).std() * np.sqrt(252)
    )

    features["dd60"] = (
        wealth / wealth.rolling(60).max() - 1
    )

    features = features.dropna()

    current = features.iloc[-1]

    # Limit neighbors to dates at least a year before the decision date.
    cutoff = (
        features.index[-1]
        - pd.Timedelta(days=365)
    )

    candidates = features.loc[
        features.index <= cutoff
    ]

    if candidates.empty:
        raise ValueError("No eligible historical analogs")

    # Scales normalize features with different magnitudes.
    scales = pd.Series({
        "r20": 0.10,
        "r60": 0.18,
        "vol20": 0.20,
        "dd60": 0.15
    })

    distance = np.sqrt(
        (
            ((candidates - current) / scales) ** 2
        ).sum(axis=1)
    )

    # Separate neighbors to avoid counting the same episode repeatedly.
    chosen = []

    for date in distance.sort_values().index:

        if all(
            abs((date - previous).days) >= 90
            for previous in chosen
        ):
            chosen.append(date)

        if len(chosen) == n:
            break

    result = candidates.loc[chosen].copy()

    result.insert(
        0,
        "distance",
        distance.loc[chosen]
    )



    # Outcomes are measured over the next 21 trading sessions.
    future_21d = (
        wealth.shift(-21) / wealth - 1
    )

    result["next_21d_return"] = (
        future_21d.reindex(result.index)
    )

    # Require the full subsequent return for each selected neighbor.
    if result["next_21d_return"].isna().any():
        raise ValueError(
            "Historical analog has an unknown future outcome"
        )

    print("\n21-SESSION RETURNS AFTER HISTORICAL ANALOGS")

    print(
        result["next_21d_return"].map(
            lambda value: f"{value:+.2%}"
        ).to_string()
    )

    print(
        "\nAverage analog return:",
        f"{result['next_21d_return'].mean():+.2%}"
    )

    print(
        "Positive outcomes:",
        int((result["next_21d_return"] > 0).sum()),
        "of",
        len(result)
    )


    print(
        "\nMarket as of:",
        features.index[-1].date()
    )

    print(
        "Historical dates searched:",
        len(candidates)
    )

    print(
        "Lower distance means greater similarity."
    )

    print(
        "This is NOT a future-return prediction."
    )


    if backtest:

        print("\nHISTORICAL ANALOG WALK-FORWARD TEST")

        future = wealth.shift(-21) / wealth - 1

        # Sample one test date per approximate holding period.
        test_dates = features.loc[
            "2018-01-01":"2024-12-31"
        ].index[::21]

        records = []

        for test_date in test_dates:

            actual = future.loc[test_date]

            if pd.isna(actual):
                continue

            # Keep training neighbors and their subsequent outcomes in the past.
            eligible = features.loc[
                :test_date - pd.Timedelta(days=365)
            ]

            if len(eligible) < 500:
                continue

            distances = np.sqrt(
                (
                    ((eligible - features.loc[test_date])
                     / scales) ** 2
                ).sum(axis=1)
            )

            matches = []

            for historical_date in distances.sort_values().index:

                if all(
                    abs((historical_date - chosen).days) >= 90
                    for chosen in matches
                ):
                    matches.append(historical_date)

                if len(matches) == n:
                    break

            if len(matches) < n:
                continue

            analog_outcomes = future.reindex(matches)

            if analog_outcomes.isna().any():
                continue

            # Compare against the unconditional historical mean.
            historical_outcomes = (
                future.reindex(eligible.index).dropna()
            )

            records.append({
                "Date": test_date,
                "Analog_Forecast": analog_outcomes.mean(),
                "Simple_Forecast": historical_outcomes.mean(),
                "Actual_Return": actual
            })

        comparison = pd.DataFrame(
            records
        ).set_index("Date")

        if len(comparison) < 20:
            raise ValueError(
                "Insufficient historical testing dates"
            )

        print("Historical forecasts:", len(comparison))


        positive_rate = (
            comparison["Actual_Return"] > 0
        ).mean()

        print(
            "Always-positive directional accuracy:",
            f"{positive_rate:.1%}"
        )

        print(
            "Analog positive predictions:",
            int((comparison["Analog_Forecast"] > 0).sum()),
            "of",
            len(comparison)
        )


        for label in ["Analog_Forecast", "Simple_Forecast"]:

            errors = (
                comparison[label]
                - comparison["Actual_Return"]
            )

            mae = errors.abs().mean()

            rmse = np.sqrt(
                (errors ** 2).mean()
            )

            directional_accuracy = (
                (comparison[label] > 0)
                ==
                (comparison["Actual_Return"] > 0)
            ).mean()

            print(f"\n{label}")
            print(f"Mean absolute error: {mae:.2%}")
            print(f"Root mean squared error: {rmse:.2%}")
            print(
                "Directional accuracy:",
                f"{directional_accuracy:.1%}"
            )

    return result.round(4)

def evaluate_frozen_2026(project_root):
    """
    Evaluate frozen strategies using 2026 prices.
    Does not retrain, mutate, or promote candidates.
    """

    root = Path(project_root).resolve()

    cache = (
            root / "data" / "reference"
            / "frozen_2026_panel.csv"
    )

    anchor = pd.Timestamp("2025-12-31")
    cutoff = pd.Timestamp("2026-10-24")

    arena = HistoricalArena(root, cost_bps=5.0)

    if arena.calendar[-1] != anchor:
        raise ValueError(
            "Historical cache must end on 2025-12-31"
        )

    if not cache.exists():

        import yfinance as yf

        print("Downloading 2026 evaluation data...")

        parts = []

        for ticker in arena.tickers:

            print("Fetching", ticker)

            prices = yf.download(
                ticker,
                start="2025-12-15",
                end="2026-10-24",
                auto_adjust=True,
                multi_level_index=False,
                progress=False,
                threads=False
            )

            if (
                    prices.empty
                    or not {"Open", "Close"}.issubset(prices.columns)
            ):
                raise ValueError(
                    f"Missing price data: {ticker}"
                )

            prices = prices.reset_index()[
                ["Date", "Open", "Close"]
            ]

            prices["Ticker"] = ticker

            parts.append(prices)

        panel = pd.concat(parts, ignore_index=True)

        cache.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        panel.to_csv(cache, index=False)

        print("Cached:", cache)

    else:

        print("Using preserved 2026 price data")

    panel = pd.read_csv(
        cache,
        parse_dates=["Date"]
    )

    if set(panel.columns) != {
        "Date", "Ticker", "Open", "Close"
    }:
        raise ValueError("Unexpected cache columns")

    if set(panel["Ticker"]) != set(arena.tickers):
        raise ValueError("Incomplete stock universe")

    if (
            panel["Date"].isna().any()
            or panel.duplicated(["Date", "Ticker"]).any()
    ):
        raise ValueError("Invalid trading dates")

    if (
            not np.isfinite(
                panel[["Open", "Close"]].to_numpy(
                    dtype=float
                )
            ).all()
            or (panel[["Open", "Close"]] <= 0).any().any()
    ):
        raise ValueError("Invalid stock prices")


    # Extend only the in-memory price panel; preserve the original files.

    arena.opens = arena.opens.copy()
    arena.closes = arena.closes.copy()

    new_dates = None
    new_opens = []
    new_closes = []


    for j, ticker in enumerate(arena.tickers):

        stock = (
            panel.loc[panel["Ticker"] == ticker]
            .sort_values("Date")
            .set_index("Date")
        )

        if anchor not in stock.index:
            raise ValueError(
                f"Missing overlap date: {ticker}"
            )

        later = stock.loc[
            (stock.index > anchor)
            & (stock.index < cutoff)
            ]

        if len(later) < 150:
            raise ValueError(
                f"Insufficient 2026 data: {ticker}"
            )

        if new_dates is None:
            new_dates = later.index

        elif not new_dates.equals(later.index):
            raise ValueError(
                f"Trading calendar mismatch: {ticker}"
            )

        # Align newer adjusted prices to the old cache on the overlap date.
        scale = (
                stock.loc[anchor, "Close"]
                / arena.closes[-1, j]
        )

        arena.opens[:, j] *= scale
        arena.closes[:, j] *= scale

        new_opens.append(
            later["Open"].to_numpy(dtype=float)
        )

        new_closes.append(
            later["Close"].to_numpy(dtype=float)
        )

    if new_dates[-1] != pd.Timestamp("2026-10-23"):
        raise ValueError(
            "Expected prices through 2026-10-23"
        )

    arena.calendar = arena.calendar.append(new_dates)

    arena.opens = np.vstack([
        arena.opens,
        np.column_stack(new_opens)
    ])

    arena.closes = np.vstack([
        arena.closes,
        np.column_stack(new_closes)
    ])

    returns = pd.DataFrame(
        arena.closes
    ).pct_change(fill_method=None)

    arena.vol20 = (
        returns.rolling(20, min_periods=20)
        .std()
        .to_numpy(dtype=float)
    )

    arena._score_cache.clear()
    arena._result_cache.clear()

    # Fixed candidates; the 2026 sample is an evaluation, not a search.
    champion = Genome(
        lookback=10,
        skip=0,
        volatility_penalty=1.0,
        risk_fraction=1.0
    )

    lower_risk = Genome(
        lookback=5,
        skip=0,
        volatility_penalty=0.0,
        risk_fraction=0.75
    )

    strategies = {
        "Evolution champion": champion,
        "Original momentum": BASELINE,
        "75% momentum": lower_risk,
        "Equal-weight 33": None
    }

    print("\nFROZEN 2026 STRATEGY EVALUATION")

    results = {}

    for name, genome in strategies.items():
        result = arena.evaluate(
            genome,
            "2026-01-02",
            "2026-10-24"
        )

        results[name] = result

        print(
            f"\n{name}"
            f"\nNet return: {result['return']:+.2%}"
            f"\nMax drawdown: {result['max_drawdown']:.2%}"
            f"\nCompleted periods: {result['periods']}"
            f"\nLast exit: {result['last_exit']}"
        )

    # Evaluate the analog risk overlay without changing the selected strategy.
    french_path = (
        root / "data" / "reference" / "french_market_daily.csv"
    )

    french = pd.read_csv(
        french_path, parse_dates=["Date"]
    )

    french = (
        french.sort_values("Date")
        .set_index("Date")
        .loc[:"2026-08-31"]
    )

    if (
        french.empty
        or french.index.max() < pd.Timestamp("2026-08-31")
        or french.index.has_duplicates
    ):
        raise ValueError("Invalid market-factor coverage")

    daily = french["Market_Return"].astype(float)
    wealth = (1.0 + daily).cumprod()

    features = pd.DataFrame(index=daily.index)
    features["r20"] = wealth / wealth.shift(20) - 1
    features["r60"] = wealth / wealth.shift(60) - 1
    features["vol20"] = (
        daily.rolling(20).std() * np.sqrt(252)
    )
    features["dd60"] = (
        wealth / wealth.rolling(60).max() - 1
    )

    features = features.dropna()
    future = wealth.shift(-21) / wealth - 1

    scales = pd.Series({
        "r20": 0.10,
        "r60": 0.18,
        "vol20": 0.20,
        "dd60": 0.15
    })

    start, end = "2026-01-02", "2026-09-01"

    first = int(
        arena.calendar.searchsorted(pd.Timestamp(start))
    )
    stop = int(
        arena.calendar.searchsorted(pd.Timestamp(end))
    )

    signal_positions = list(
        range(first, stop - HOLD_SESSIONS - 1, HOLD_SESSIONS)
    )

    if not signal_positions:
        raise ValueError("No completed evaluation periods")

    schedule = {}

    for si in signal_positions:

        signal_date = arena.calendar[si]

        # Use the previous factor date; same-day factor data is not assumed available.
        pos = (
            features.index.searchsorted(
                signal_date, side="left"
            ) - 1
        )

        if pos < 0:
            raise ValueError("Missing historical market signal")

        as_of = features.index[pos]

        if (signal_date - as_of).days > 5:
            raise ValueError("Stale market observation")

        current = features.loc[as_of]

        candidates = features.loc[
            :as_of - pd.Timedelta(days=365)
        ]

        scaled = (
            (candidates - current) / scales
        ).astype(float)

        distances = np.sqrt(
            (scaled ** 2).sum(axis=1)
        )

        matches = []

        for candidate_date in distances.sort_values().index:

            if all(
                abs((candidate_date - earlier).days) >= 90
                for earlier in matches
            ):
                matches.append(candidate_date)

            if len(matches) == 5:
                break

        if len(matches) != 5:
            raise ValueError("Insufficient historical analogs")

        known_outcomes = future.reindex(matches)

        if known_outcomes.isna().any():
            raise ValueError("Unavailable historical outcome")

        forecast = float(known_outcomes.mean())

        schedule[signal_date] = (
            0.75 if forecast < 0 else 1.0
        )

    matched = float(np.mean(list(schedule.values())))

    print("\n2026 CHAMPION VS ANALOG OVERLAY")
    print("Analog warnings:",
          sum(value < 1 for value in schedule.values()),
          "of", len(schedule))
    print("Average exposure:", f"{matched:.2%}")

    competitors = {
        "Frozen champion": {
            day: 1.0 for day in schedule
        },
        "Constant matched exposure": {
            day: matched for day in schedule
        },
        "Champion + price analogs": schedule
    }

    for label, risk in competitors.items():

        outcome = arena.evaluate(
            champion,
            start,
            end,
            risk_schedule=risk
        )

        print(f"\n{label}")
        print(f"Net return: {outcome['return']:+.2%}")
        print(
            f"Max drawdown: "
            f"{outcome['max_drawdown']:.2%}"
        )
        print("Completed periods:", outcome["periods"])
        print("Last exit:", outcome["last_exit"])

    print("\nSAME-WINDOW STOCK-SELECTION BENCHMARKS")

    for name, genome in [
        ("Original momentum", BASELINE),
        ("Equal-weight 33", None)
    ]:

        outcome = arena.evaluate(
            genome,
            start,
            end
        )

        print(f"\n{name}")
        print(f"Net return: {outcome['return']:+.2%}")
        print(
            "Max drawdown:",
            f"{outcome['max_drawdown']:.2%}"
        )
        print("Completed periods:", outcome["periods"])

    print("\nEXHIBITION ONLY - NO CHAMPION PROMOTION")

    print("\nPASS: Frozen evaluation completed")
    print("No strategy was retrained or promoted.")

    return results


def preview_macro_regimes(project_root):
    """
    Explore historical inflation and interest rates.
    Research only; does not affect trading decisions.
    """

    from io import StringIO
    from urllib.request import Request, urlopen

    root = Path(project_root).resolve()
    cache = root / "data" / "reference" / "macro_regimes.csv"

    if not cache.exists():

        print("Downloading economic history from FRED...")

        tables = {}

        for series in ["CPIAUCNS", "GS10"]:

            url = (
                "https://fred.stlouisfed.org/graph/"
                f"fredgraph.csv?id={series}"
            )

            request = Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"}
            )

            with urlopen(request, timeout=40) as response:
                content = response.read().decode("utf-8-sig")

            data = pd.read_csv(StringIO(content))

            if len(data.columns) != 2:
                raise ValueError(f"Unexpected FRED data: {series}")

            data.columns = ["Date", series]

            data["Date"] = pd.to_datetime(data["Date"])

            data[series] = pd.to_numeric(
                data[series], errors="coerce"
            )

            tables[series] = data.dropna()

        cpi = tables["CPIAUCNS"].sort_values("Date")

        cpi["inflation_yoy"] = (
            cpi["CPIAUCNS"].pct_change(12) * 100
        )

        macro = cpi[["Date", "inflation_yoy"]].merge(
            tables["GS10"],
            on="Date",
            how="inner"
        )

        macro = macro.dropna().sort_values("Date")

        # Approximate reporting lag; revised FRED data are not point-in-time vintages.
        macro["usable_from"] = (
            macro["Date"]
            + pd.offsets.MonthEnd(0)
            + pd.Timedelta(days=45)
        )

        cache.parent.mkdir(parents=True, exist_ok=True)

        macro.to_csv(cache, index=False)

        print("Economic history cached.")

    else:

        print("Using cached economic history.")

    macro = pd.read_csv(
        cache,
        parse_dates=["Date", "usable_from"]
    )

    required = {
        "Date", "usable_from", "inflation_yoy", "GS10"
    }

    if not required.issubset(macro.columns):
        raise ValueError("Incomplete macroeconomic dataset")

    if macro["Date"].duplicated().any():
        raise ValueError("Duplicate economic dates")

    if macro[list(required)].isna().any().any():
        raise ValueError("Missing economic values")

    print("\nHISTORICAL ECONOMIC REGIMES")

    for date_string in ["1976-09-13", "2025-12-31"]:

        as_of = pd.Timestamp(date_string)

        available = macro.loc[
            macro["usable_from"] <= as_of
        ].sort_values("Date")

        if available.empty:
            raise ValueError(
                f"No available economic history for {date_string}"
            )

        latest = available.iloc[-1]

        print(f"\nMarket date: {date_string}")
        print(f"Economic observation: {latest['Date'].date()}")

        print(
            "Annual CPI inflation:",
            f"{latest['inflation_yoy']:+.2f}%"
        )

        print(
            "10-year Treasury yield:",
            f"{latest['GS10']:.2f}%"
        )

    print("\nPASS: Economic-regime preview completed")

    return macro


def compare_macro_analogs(project_root):
    """
    Compare price-only and macro-aware analog forecasts.
    Exploratory research; never changes trading strategies.
    """

    root = Path(project_root).resolve()
    reference = root / "data" / "reference"

    market = pd.read_csv(
        reference / "french_market_daily.csv",
        parse_dates=["Date"]
    ).sort_values("Date").set_index("Date")

    market = market.loc[:"2025-12-31"]

    macro = pd.read_csv(
        reference / "macro_regimes.csv",
        parse_dates=["usable_from"]
    ).sort_values("usable_from")

    # Reconstruct the original market features.
    r = market["Market_Return"].astype(float)
    wealth = (1 + r).cumprod()

    features = pd.DataFrame(index=r.index)
    features.index.name = "Date"

    features["r20"] = wealth / wealth.shift(20) - 1
    features["r60"] = wealth / wealth.shift(60) - 1
    features["vol20"] = (
        r.rolling(20).std() * np.sqrt(252)
    )
    features["dd60"] = (
        wealth / wealth.rolling(60).max() - 1
    )

    features = features.dropna()

    # Attach only economic observations that would
    # have passed our assumed 45-day availability lag.
    combined = pd.merge_asof(
        features.reset_index().sort_values("Date"),
        macro[
            ["usable_from", "inflation_yoy", "GS10"]
        ].sort_values("usable_from"),
        left_on="Date",
        right_on="usable_from",
        direction="backward"
    )

    combined = (
        combined.set_index("Date")
        .dropna()
        .sort_index()
    )

    # Feature scales were fixed before comparing these variants.
    scales = pd.Series({
        "r20": 0.10,
        "r60": 0.18,
        "vol20": 0.20,
        "dd60": 0.15,
        "inflation_yoy": 4.0,
        "GS10": 3.0
    })

    price_columns = [
        "r20", "r60", "vol20", "dd60"
    ]

    macro_columns = price_columns + [
        "inflation_yoy", "GS10"
    ]

    future = wealth.shift(-21) / wealth - 1

    def analog_forecast(eligible, current, columns):


        scaled = (
            eligible[columns].astype(float)
            - current[columns].astype(float)
        ) / scales[columns]

        distance = np.sqrt(
            (scaled ** 2).sum(axis=1)
        )


        matches = []

        for date in distance.sort_values().index:

            if all(
                abs((date - earlier).days) >= 90
                for earlier in matches
            ):
                matches.append(date)

            if len(matches) == 5:
                break

        if len(matches) != 5:
            return np.nan

        outcomes = future.reindex(matches)

        if outcomes.isna().any():
            return np.nan

        return float(outcomes.mean())

    test_dates = combined.loc[
        "2018-01-01":"2024-12-31"
    ].index[::21]

    records = []

    for test_date in test_dates:

        # Only compare to neighbors with known subsequent outcomes.
        cutoff = test_date - pd.Timedelta(days=365)

        eligible = combined.loc[:cutoff]

        if len(eligible) < 500:
            continue

        actual = future.loc[test_date]

        if pd.isna(actual):
            continue

        current = combined.loc[test_date]

        price_prediction = analog_forecast(
            eligible, current, price_columns
        )

        macro_prediction = analog_forecast(
            eligible, current, macro_columns
        )

        benchmark = future.reindex(
            eligible.index
        ).mean()

        if not np.isfinite([
            price_prediction,
            macro_prediction,
            benchmark
        ]).all():
            continue

        records.append({
            "Date": test_date,
            "Actual": actual,
            "Price_Only": price_prediction,
            "Macro_Aware": macro_prediction,
            "Historical_Average": benchmark
        })

    if len(records) < 20:
        raise ValueError(
            "Insufficient paired evaluation dates"
        )

    comparison = pd.DataFrame(
        records
    ).set_index("Date")

    actual = comparison["Actual"]

    print("\nPRICE VS ECONOMIC-REGIME ANALOG TEST")
    print("Matched forecasts:", len(comparison))
    print(
        "Always-positive accuracy:",
        f"{(actual > 0).mean():.1%}"
    )

    for label in [
        "Historical_Average",
        "Price_Only",
        "Macro_Aware"
    ]:

        forecast = comparison[label]
        error = forecast - actual

        mae = error.abs().mean()
        rmse = np.sqrt((error ** 2).mean())

        direction = (
            (forecast > 0) == (actual > 0)
        ).mean()

        declining = actual < 0

        bearish_recall = (
            ((forecast < 0) & declining).sum()
            / declining.sum()
            if declining.any()
            else np.nan
        )

        print(f"\n{label}")
        print(f"Mean absolute error: {mae:.2%}")
        print(f"RMSE: {rmse:.2%}")
        print(f"Directional accuracy: {direction:.1%}")
        print(
            "Decline detection:",
            f"{bearish_recall:.1%}"
        )

    print("\nNo strategies retrained or promoted.")


    print("\nDOWNSIDE SIGNAL AUDIT")

    actual_down = comparison["Actual"] < 0

    for label in [
        "Historical_Average",
        "Price_Only",
        "Macro_Aware"
    ]:

        predicted_down = comparison[label] < 0

        tp = int((predicted_down & actual_down).sum())
        fp = int((predicted_down & ~actual_down).sum())
        fn = int((~predicted_down & actual_down).sum())
        tn = int((~predicted_down & ~actual_down).sum())

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        specificity = tn / (tn + fp) if tn + fp else 0

        balanced_accuracy = (recall + specificity) / 2

        print(f"\n{label}")
        print("Correct decline warnings:", tp)
        print("False decline warnings:", fp)
        print("Missed declines:", fn)
        print("Correct rising periods:", tn)
        print(f"Warning precision: {precision:.1%}")
        print(f"Balanced accuracy: {balanced_accuracy:.1%}")



    # These exposure rules are exploratory; they do not select a champion.

    print("\nHISTORICAL ANALOG EXHIBITION MATCH")

    actual = comparison["Actual"].to_numpy(dtype=float)

    # Cut market exposure to 75% when the analog mean is negative.
    analog_exposure = np.where(
        comparison["Price_Only"].to_numpy() < 0,
        0.75,
        1.0
    )


    # Match the average invested fraction to isolate timing effects.
    matched_exposure = float(analog_exposure.mean())

    strategies = {
        "Full market": np.ones(len(actual)),
        "Constant 75%": np.full(len(actual), 0.75),
        "Constant matched exposure": np.full(
            len(actual), matched_exposure
        ),
        "Analog challenger": analog_exposure
    }

    print(
        "Analog average exposure:",
        f"{matched_exposure:.2%}"
    )

    for name, exposure in strategies.items():

        period_returns = exposure * actual

        equity = np.r_[
            1.0,
            np.cumprod(1 + period_returns)
        ]

        total_return = equity[-1] - 1

        drawdown = (
            equity / np.maximum.accumulate(equity) - 1
        ).min()

        print(f"\n{name}")
        print(f"Compounded return: {total_return:+.2%}")
        print(f"Worst period: {period_returns.min():+.2%}")
        print(f"Period-end drawdown: {drawdown:.2%}")

    print(
        "\nAnalog risk-off periods:",
        int((analog_exposure < 1).sum()),
        "of",
        len(actual)
    )

    print("\nEXHIBITION ONLY - NO CHAMPION PROMOTION")


    print("\nANALOG PERFORMANCE BY YEAR")

    years = comparison.index.year

    for year in sorted(set(years)):

        mask = years == year

        year_returns = actual[mask]
        year_exposure = analog_exposure[mask]

        analog_return = (
            np.prod(1 + year_exposure * year_returns) - 1
        )

        matched_return = (
            np.prod(1 + matched_exposure * year_returns) - 1
        )

        advantage = analog_return - matched_return

        warnings = int((year_exposure < 1).sum())

        print(
            f"{year}: "
            f"Analog {analog_return:+.2%} | "
            f"Matched {matched_return:+.2%} | "
            f"Difference {advantage:+.2%} | "
            f"Warnings {warnings}"
        )


    return comparison
