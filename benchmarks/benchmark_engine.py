"""Engine performance & correctness benchmark.

Spec
----
- Fixed data window:  2025-01-01 ~ 2025-06-30 (6 months, M15)
    Rationale (per user spec):
      * Large enough for stable timing measurement (~25-60s per run currently)
      * Includes Q1 USD strength + spring gold rally + Q2 chop -> regime diversity
      * Avoids R38 / R42 already-cached windows (later 2025)
      * Stable fixed dates -> historical benchmarks comparable forever
- Fixed strategy:     L8_BASE (LIVE_PARITY_KWARGS + L8 overrides)
- Metrics tracked:    elapsed_s, n_trades, total_pnl, sharpe, max_dd, win_rate
- Tolerance:
    * n_trades:  exact match (delta == 0)
    * total_pnl: |delta| <= 0.01 USD
    * sharpe:    |delta| <= 0.001
- Output JSON: benchmarks/results/<tag>.json

Usage
-----
    python -m benchmarks.benchmark_engine save baseline      # record current
    python -m benchmarks.benchmark_engine compare baseline   # compare current vs saved
"""
import json
import sys
import time
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from backtest.runner import DataBundle, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine
from indicators import get_orb_strategy
import indicators as signals_mod
from backtest.stats import calc_stats


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

RESULTS_DIR = ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _reset_globals():
    get_orb_strategy().reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False


def load_fixed_window() -> Dict:
    """Load fixed 2025-H1 window with indicators applied (uses DataBundle to ensure parity).

    We pull a ~6 month lead-in so atr_percentile (rolling 500) and EMAs are warm.
    """
    lead_in = (pd.Timestamp(BASELINE_START, tz='UTC') - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    bundle = DataBundle.load_default(start=lead_in, end=BASELINE_END)
    m15 = bundle.m15_df[bundle.m15_df.index >= pd.Timestamp(BASELINE_START, tz='UTC')]
    h1 = bundle.h1_df[bundle.h1_df.index >= pd.Timestamp(BASELINE_START, tz='UTC')]
    print(f"  Window {BASELINE_START} -> {BASELINE_END}: M15={len(m15)} bars, H1={len(h1)} bars")
    return {'m15': m15, 'h1': h1}


def run_benchmark(data: Dict, label: str) -> Dict:
    _reset_globals()
    print(f"\n  Running benchmark [{label}] ...", flush=True)
    t0 = time.perf_counter()
    engine = BacktestEngine(data['m15'], data['h1'], label=label, **L8_BASE_KWARGS)
    trades = engine.run()
    elapsed = time.perf_counter() - t0

    stats = calc_stats(trades, engine.equity_curve)
    pnls = [t.pnl for t in trades]

    out = {
        'label': label,
        'elapsed_s': round(elapsed, 3),
        'n_trades': int(stats['n']),
        'total_pnl': round(float(stats['total_pnl']), 4),
        'sharpe': round(float(stats['sharpe']), 6),
        'max_dd': round(float(stats['max_dd']), 4),
        'win_rate': round(float(stats['win_rate']), 4),
        'avg_win': round(float(stats['avg_win']), 6),
        'avg_loss': round(float(stats['avg_loss']), 6),
        'pnl_checksum': round(sum(pnls), 6),
        'first_5_pnls': [round(p, 6) for p in pnls[:5]],
        'last_5_pnls': [round(p, 6) for p in pnls[-5:]],
    }
    return out


def print_result(r: Dict):
    print(f"  {'='*55}")
    print(f"  {r['label']}")
    print(f"  {'='*55}")
    print(f"    elapsed_s  : {r['elapsed_s']:.3f}s   ({r['elapsed_s']/60:.2f} min)")
    print(f"    n_trades   : {r['n_trades']}")
    print(f"    total_pnl  : ${r['total_pnl']:.4f}")
    print(f"    sharpe     : {r['sharpe']:.4f}")
    print(f"    max_dd     : ${r['max_dd']:.2f}")
    print(f"    win_rate   : {r['win_rate']:.2f}%")
    print(f"    pnl_checksum: {r['pnl_checksum']:.6f}")
    print(f"    first_5_pnls: {r['first_5_pnls']}")
    print(f"    last_5_pnls : {r['last_5_pnls']}")
    print(f"  {'='*55}")


def compare(baseline: Dict, current: Dict) -> bool:
    """Return True if all tolerances pass."""
    print(f"\n  COMPARE  {baseline['label']}  vs  {current['label']}")
    print(f"  {'-'*70}")

    tol_n = 0
    tol_pnl = 0.01
    tol_sharpe = 0.001

    dn = current['n_trades'] - baseline['n_trades']
    dp = current['total_pnl'] - baseline['total_pnl']
    ds = current['sharpe'] - baseline['sharpe']
    dd = current['max_dd'] - baseline['max_dd']

    speedup = baseline['elapsed_s'] / current['elapsed_s'] if current['elapsed_s'] > 0 else float('inf')

    print(f"    elapsed   : {baseline['elapsed_s']:.2f}s -> {current['elapsed_s']:.2f}s   "
          f"(speedup={speedup:.2f}x)")
    print(f"    n_trades  : {baseline['n_trades']:>6}  -> {current['n_trades']:>6}      "
          f"delta={dn:+d}            tol=0    {'PASS' if abs(dn) <= tol_n else 'FAIL'}")
    print(f"    total_pnl : {baseline['total_pnl']:>10.4f} -> {current['total_pnl']:>10.4f} "
          f"delta={dp:+.4f}    tol={tol_pnl}  {'PASS' if abs(dp) <= tol_pnl else 'FAIL'}")
    print(f"    sharpe    : {baseline['sharpe']:>10.4f} -> {current['sharpe']:>10.4f} "
          f"delta={ds:+.4f}    tol={tol_sharpe} {'PASS' if abs(ds) <= tol_sharpe else 'FAIL'}")
    print(f"    max_dd    : {baseline['max_dd']:>10.2f} -> {current['max_dd']:>10.2f} "
          f"delta={dd:+.4f}")
    print(f"    pnl_chksum: {baseline['pnl_checksum']:.6f} -> {current['pnl_checksum']:.6f}")

    if baseline['first_5_pnls'] != current['first_5_pnls']:
        print(f"    [WARN] first_5_pnls differ:")
        print(f"           base   : {baseline['first_5_pnls']}")
        print(f"           current: {current['first_5_pnls']}")
    if baseline['last_5_pnls'] != current['last_5_pnls']:
        print(f"    [WARN] last_5_pnls differ:")
        print(f"           base   : {baseline['last_5_pnls']}")
        print(f"           current: {current['last_5_pnls']}")

    pass_n = abs(dn) <= tol_n
    pass_p = abs(dp) <= tol_pnl
    pass_s = abs(ds) <= tol_sharpe
    all_pass = pass_n and pass_p and pass_s

    print(f"  {'-'*70}")
    print(f"  RESULT: {'PASS  (all tolerances met)' if all_pass else 'FAIL  (see above)'}"
          f"   speedup={speedup:.2f}x")
    print(f"  {'-'*70}")
    return all_pass


def main():
    args = sys.argv[1:]
    mode = args[0] if args else 'save'
    tag = args[1] if len(args) >= 2 else 'baseline'

    data = load_fixed_window()
    current = run_benchmark(data, tag)
    print_result(current)

    save_path = RESULTS_DIR / f"{tag}.json"

    if mode == 'save':
        with open(save_path, 'w') as f:
            json.dump(current, f, indent=2)
        print(f"\n  Saved to {save_path}")

    elif mode == 'compare':
        if not save_path.exists():
            print(f"\n  ERROR: baseline file not found: {save_path}")
            print(f"         Run with `save {tag}` first.")
            sys.exit(2)
        with open(save_path) as f:
            baseline = json.load(f)
        ok = compare(baseline, current)
        sys.exit(0 if ok else 1)

    elif mode == 'compare-to':
        # python -m benchmarks.benchmark_engine compare-to phase_a baseline
        cmp_tag = args[1] if len(args) >= 2 else 'phase_a'
        base_tag = args[2] if len(args) >= 3 else 'baseline'
        base_path = RESULTS_DIR / f"{base_tag}.json"
        cmp_path = RESULTS_DIR / f"{cmp_tag}.json"
        if not base_path.exists() or not cmp_path.exists():
            print(f"\n  ERROR: missing {base_path} or {cmp_path}")
            sys.exit(2)
        with open(base_path) as f:
            baseline = json.load(f)
        with open(cmp_path) as f:
            cur = json.load(f)
        ok = compare(baseline, cur)
        sys.exit(0 if ok else 1)

    else:
        print(f"Unknown mode: {mode}. Use save | compare | compare-to")
        sys.exit(2)


if __name__ == '__main__':
    main()
