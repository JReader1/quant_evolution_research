import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

from quantlab.evolution.championship import run_championship

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--population', type=int, default=16)
    parser.add_argument('--generations', type=int, default=5)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--cost-bps', type=float, default=5.0)
    args = parser.parse_args()
    print('\nEVOLUTION CHAMPIONSHIP - RESEARCH ONLY')
    print('Offline historical prices; news disabled (no point-in-time history).')
    output, result = run_championship(
        ROOT, population=args.population, generations=args.generations,
        seed=args.seed, cost_bps=args.cost_bps)
    print('\nCHAMPION:', result['champion']['candidate'])
    print('Fitness:', f"{result['champion']['fitness']:+.3f}")
    print('Baseline fitness:', f"{result['baseline']['fitness']:+.3f}")
    print('Secondary audit (2023-24) return:',
          f"{result['secondary_audit']['champion']['return']:+.2%}")
    print('Champion minus equal-weight on secondary audit:',
          f"{result['secondary_audit']['champion_minus_33stock_pp']:+.2f} percentage points")
    print('Unique candidates:', result['unique_candidates_evaluated'])
    print('Saved report:', output / 'research_report.md')
    print('No live orders. No automatic promotion.')
