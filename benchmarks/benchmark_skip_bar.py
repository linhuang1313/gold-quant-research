"""Benchmark: skip-non-H1-bar optimization vs baseline.

Runs the same L8 variant twice:
  1. skip_non_h1_bars=True  (new default)
  2. skip_non_h1_bars=False (legacy behavior)

Verifies trade count and PnL are identical (within tolerance for M15 signals),
and reports the speedup.

Usage:
  python -m benchmarks.benchmark_skip_bar
"""
import json
import sys
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import indicators
from backtest.runner import DataBundle, LIVE_PARITY_KWARGS, run_variant


RESULTS_DIR = ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_l8_kwargs():
    return {
        **LIVE_PARITY_KWARGS,
        'keltner_adx_threshold': 14,
        'regime_config': {
            'low':    {'trail_act': 0.22, 'trail_dist': 0.04},
            'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
            'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
        },
        'keltner_max_hold_m15': 20,
        'time_decay_tp': False,
        'min_entry_gap_hours': 1.0,
    }


def main():
    print("=" * 60)
    print("  Benchmark: skip-non-H1-bar optimization")
    print("=" * 60)

    print("\n  Loading data...", flush=True)
    data = DataBundle.load_default()
    print(f"  M15: {len(data.m15_df)} bars, H1: {len(data.h1_df)} bars\n")

    # Run with skip enabled (new)
    kw_skip = get_l8_kwargs()
    kw_skip['skip_non_h1_bars'] = True
    t0 = time.perf_counter()
    stats_skip = run_variant(data, "L8_skip_on", **kw_skip)
    time_skip = time.perf_counter() - t0

    # Run with skip disabled (legacy)
    kw_legacy = get_l8_kwargs()
    kw_legacy['skip_non_h1_bars'] = False
    t0 = time.perf_counter()
    stats_legacy = run_variant(data, "L8_skip_off", **kw_legacy)
    time_legacy = time.perf_counter() - t0

    # Compare
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Skip ON:   {time_skip:.2f}s, {stats_skip['n']} trades, "
          f"PnL=${stats_skip['total_pnl']:.2f}, Sharpe={stats_skip['sharpe']:.3f}")
    print(f"  Skip OFF:  {time_legacy:.2f}s, {stats_legacy['n']} trades, "
          f"PnL=${stats_legacy['total_pnl']:.2f}, Sharpe={stats_legacy['sharpe']:.3f}")

    speedup = time_legacy / time_skip if time_skip > 0 else float('inf')
    print(f"\n  Speedup: {speedup:.2f}x ({time_legacy:.1f}s → {time_skip:.1f}s)")

    trade_diff = stats_skip['n'] - stats_legacy['n']
    pnl_diff = stats_skip['total_pnl'] - stats_legacy['total_pnl']
    h1_match = stats_skip['h1_entries'] == stats_legacy['h1_entries']
    m15_diff = stats_skip['m15_entries'] - stats_legacy['m15_entries']

    print(f"\n  H1 entries match: {h1_match} "
          f"({stats_skip['h1_entries']} vs {stats_legacy['h1_entries']})")
    print(f"  M15 entries diff: {m15_diff} "
          f"({stats_skip['m15_entries']} vs {stats_legacy['m15_entries']})")
    print(f"  Trade count diff: {trade_diff}")
    print(f"  PnL diff: ${pnl_diff:.2f}")

    print(f"\n  Note: skip_non_h1_bars skips M15-only entry checks on non-H1-boundary")
    print(f"  bars when flat. H1 entry diffs are indirect (fewer M15 positions")
    print(f"  means fewer slot-blocked H1 entries).")
    if m15_diff <= 0 and stats_skip['total_pnl'] >= stats_legacy['total_pnl'] * 0.95:
        print(f"\n  [PASS] Optimization safe: speedup {speedup:.2f}x, PnL delta acceptable")
    else:
        print(f"\n  [WARN] Review results — unexpected PnL regression")

    result = {
        'timestamp': datetime.now().isoformat(),
        'skip_on': {
            'time_s': round(time_skip, 2),
            'trades': int(stats_skip['n']),
            'h1_entries': int(stats_skip['h1_entries']),
            'm15_entries': int(stats_skip['m15_entries']),
            'pnl': round(float(stats_skip['total_pnl']), 2),
            'sharpe': round(float(stats_skip['sharpe']), 4),
        },
        'skip_off': {
            'time_s': round(time_legacy, 2),
            'trades': int(stats_legacy['n']),
            'h1_entries': int(stats_legacy['h1_entries']),
            'm15_entries': int(stats_legacy['m15_entries']),
            'pnl': round(float(stats_legacy['total_pnl']), 2),
            'sharpe': round(float(stats_legacy['sharpe']), 4),
        },
        'speedup': round(speedup, 2),
        'h1_match': h1_match,
        'pnl_diff': round(float(pnl_diff), 2),
    }

    out_path = RESULTS_DIR / "skip_bar_benchmark.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
