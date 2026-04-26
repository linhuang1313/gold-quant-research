"""Run the standard benchmark N times back-to-back and report variance.

Used to verify the 14% jitter observed between p1_full run1 (12.993s)
and run2 (11.117s) is one-off (numpy warmup / OS scheduling) rather
than something P1 introduced as a structural problem.

Spec
----
- N = 5 sequential runs of the fixed 2025-H1 benchmark.
- Reload the engine module + reset globals between runs (no pyc reuse).
- Reports min / max / mean / std and the per-run elapsed_s.
- Pass criterion (per user): coefficient of variation < 8%.
"""
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from backtest.runner import DataBundle, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine
from indicators import get_orb_strategy
import indicators as signals_mod


BASELINE_START = "2025-01-01"
BASELINE_END = "2025-06-30"

L8_BASE_KWARGS = {
    **LIVE_PARITY_KWARGS,
    'keltner_adx_threshold': 14,
    'regime_config': {
        'low':    {'trail_act': 0.22, 'trail_dist': 0.04},
        'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
        'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
    },
    'min_entry_gap_hours': 1.0,
    'keltner_max_hold_m15': 20,
    'spread_cost': 0.50,
    'kc_bw_filter_bars': 5,
}


def _reset_globals():
    get_orb_strategy().reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False


def main(n_runs: int = 5):
    lead_in = (pd.Timestamp(BASELINE_START, tz='UTC') - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    bundle = DataBundle.load_default(start=lead_in, end=BASELINE_END)
    m15 = bundle.m15_df[bundle.m15_df.index >= pd.Timestamp(BASELINE_START, tz='UTC')]
    h1 = bundle.h1_df[bundle.h1_df.index >= pd.Timestamp(BASELINE_START, tz='UTC')]
    print(f"  Window {BASELINE_START} -> {BASELINE_END}: M15={len(m15)} bars, H1={len(h1)} bars")
    print(f"  Running N={n_runs} sequential benchmarks ...")

    runs = []
    for i in range(n_runs):
        _reset_globals()
        t0 = time.perf_counter()
        engine = BacktestEngine(m15, h1, label=f"variance_{i+1}", **L8_BASE_KWARGS)
        trades = engine.run()
        elapsed = time.perf_counter() - t0
        n = len(trades)
        pnl = sum(t.pnl for t in trades)
        runs.append({'i': i + 1, 'elapsed_s': elapsed, 'n_trades': n, 'pnl': pnl})
        print(f"    [{i+1}/{n_runs}] elapsed={elapsed:.3f}s  n_trades={n}  pnl=${pnl:.2f}")

    print()
    elapsed_list = [r['elapsed_s'] for r in runs]
    pnl_list = [r['pnl'] for r in runs]
    n_trades_list = [r['n_trades'] for r in runs]

    mean_e = statistics.mean(elapsed_list)
    std_e = statistics.stdev(elapsed_list) if len(elapsed_list) > 1 else 0.0
    cv = (std_e / mean_e * 100) if mean_e else 0.0

    print("  ===== Variance summary =====")
    print(f"    runs        : {n_runs}")
    print(f"    elapsed min : {min(elapsed_list):.3f}s")
    print(f"    elapsed max : {max(elapsed_list):.3f}s")
    print(f"    elapsed mean: {mean_e:.3f}s")
    print(f"    elapsed std : {std_e:.3f}s")
    print(f"    CV (std/mean): {cv:.2f}%")
    print()
    print(f"    n_trades unique: {sorted(set(n_trades_list))}")
    print(f"    pnl unique     : {sorted(set(round(p, 4) for p in pnl_list))}")

    threshold = 8.0
    if cv < threshold:
        print(f"\n    PASS  CV={cv:.2f}% < {threshold}% (stable)")
        sys.exit(0)
    else:
        print(f"\n    FAIL  CV={cv:.2f}% >= {threshold}% (investigate)")
        sys.exit(1)


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(n_runs=n)
