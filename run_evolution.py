"""Run the evolutionary momentum search against cached historical prices."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantlab.evolution.championship import run_championship


def main() -> None:
    parser = argparse.ArgumentParser(description="Search historical equity momentum strategies.")
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    args = parser.parse_args()

    folder, result = run_championship(
        PROJECT_ROOT,
        population=args.population,
        generations=args.generations,
        seed=args.seed,
        cost_bps=args.cost_bps,
    )

    champion = result["champion"]
    audit = result["secondary_audit"]

    print(f"Champion: {champion['candidate']}")
    print(f"Fitness: {champion['fitness']:+.3f} "
          f"(baseline {result['baseline']['fitness']:+.3f})")
    print(f"Secondary audit return: {audit['champion']['return']:+.2%}")
    print(f"Excess vs equal weight: {audit['champion_minus_33stock_pp']:+.2f} pp")
    print(f"Unique candidates evaluated: {result['unique_candidates_evaluated']}")
    print(f"Report: {folder / 'research_report.md'}")
    print("Historical simulation only; no live trades or strategy promotion.")


if __name__ == "__main__":
    main()
