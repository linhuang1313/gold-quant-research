"""
R28: L7(MH=8) Full-Data 6-Fold K-Fold Validation
=================================================
Runs the complete L7(MH=8) config through the real engine's run_kfold
on full data (2015-01-01 to 2026-04-10).
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from datetime import datetime

from backtest.runner import DataBundle, run_variant, run_kfold, LIVE_PARITY_KWARGS


OUT_DIR = Path("results/round28_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            try: f.write(data)
            except: pass
            f.flush()
    def flush(self):
        for f in self.files: f.flush()


def get_l6():
    kw = {**LIVE_PARITY_KWARGS}
    kw['regime_config'] = {
        'low':    {'trail_act': 0.30, 'trail_dist': 0.06},
        'normal': {'trail_act': 0.20, 'trail_dist': 0.04},
        'high':   {'trail_act': 0.08, 'trail_dist': 0.01},
    }
    return kw


def get_l7_mh8():
    kw = get_l6()
    kw['time_adaptive_trail'] = True
    kw['time_adaptive_trail_start'] = 2
    kw['time_adaptive_trail_decay'] = 0.75
    kw['time_adaptive_trail_floor'] = 0.003
    kw['min_entry_gap_hours'] = 1.0
    kw['keltner_max_hold_m15'] = 8
    return kw


def main():
    t0 = time.time()
    out_path = OUT_DIR / "R28_L7MH8_kfold.txt"
    out = open(out_path, 'w', encoding='utf-8', buffering=1)
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, out)

    print(f"# R28: L7(MH=8) Full-Data K-Fold Validation")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    data = DataBundle.load_default()

    # ── 1. Full-sample baseline ──
    print("=" * 70)
    print("1. Full-Sample Comparison: L6 vs L7(MH=20) vs L7(MH=8)")
    print("=" * 70)

    configs = [
        ("L6_baseline", get_l6()),
        ("L7_MH20", {**get_l6(),
                     'time_adaptive_trail': True,
                     'time_adaptive_trail_start': 2,
                     'time_adaptive_trail_decay': 0.75,
                     'time_adaptive_trail_floor': 0.003,
                     'min_entry_gap_hours': 1.0,
                     'keltner_max_hold_m15': 20}),
        ("L7_MH8", get_l7_mh8()),
    ]

    print(f"\n  {'Label':<15} {'N':>6} {'Sharpe':>8} {'PnL':>10} {'WR':>6} {'MaxDD':>8}")
    print(f"  {'-'*55}")
    for label, kw in configs:
        s = run_variant(data, label, verbose=False, **kw)
        print(f"  {label:<15} {s['n']:>6} {s['sharpe']:>8.2f} ${s['total_pnl']:>9.0f} "
              f"{s['win_rate']:>5.1f}% ${s['max_dd']:>7.0f}")
    out.flush()

    # ── 2. L7(MH=8) 6-Fold K-Fold ──
    print(f"\n{'='*70}")
    print("2. L7(MH=8) 6-Fold K-Fold (Real Engine)")
    print("=" * 70)

    kw = get_l7_mh8()
    kf_results = run_kfold(data, kw, n_folds=6, label_prefix="L7MH8_")

    print(f"\n  {'Fold':<8} {'Period':<25} {'N':>5} {'Sharpe':>8} {'PnL':>9} {'WR':>6} {'MaxDD':>8}")
    print(f"  {'-'*70}")
    sharpes = []
    for r in kf_results:
        fold = r.get('fold', '?')
        period = f"{r.get('test_start','?')} - {r.get('test_end','?')}"
        print(f"  {fold:<8} {period:<25} {r['n']:>5} {r['sharpe']:>8.2f} ${r['total_pnl']:>8.0f} "
              f"{r['win_rate']:>5.1f}% ${r['max_dd']:>7.0f}")
        sharpes.append(r['sharpe'])
    out.flush()

    pos = sum(1 for s in sharpes if s > 0)
    print(f"\n  K-Fold Result: {pos}/{len(sharpes)} positive")
    print(f"  Mean Sharpe: {np.mean(sharpes):.2f}")
    print(f"  Min Sharpe:  {np.min(sharpes):.2f}")
    print(f"  Max Sharpe:  {np.max(sharpes):.2f}")
    print(f"  Std Sharpe:  {np.std(sharpes):.2f}")

    if pos == len(sharpes):
        print(f"\n  >>> PASS: L7(MH=8) K-Fold {pos}/{len(sharpes)} — safe for deployment")
    else:
        print(f"\n  >>> WARN: {len(sharpes)-pos} folds negative — review before deployment")
    out.flush()

    # ── 3. L6 K-Fold for comparison ──
    print(f"\n{'='*70}")
    print("3. L6 Baseline 6-Fold K-Fold (for comparison)")
    print("=" * 70)

    kf_l6 = run_kfold(data, get_l6(), n_folds=6, label_prefix="L6_")

    print(f"\n  {'Fold':<8} {'N':>5} {'Sharpe':>8} {'PnL':>9}")
    print(f"  {'-'*35}")
    l6_sharpes = []
    for r in kf_l6:
        print(f"  {r.get('fold','?'):<8} {r['n']:>5} {r['sharpe']:>8.2f} ${r['total_pnl']:>8.0f}")
        l6_sharpes.append(r['sharpe'])
    out.flush()

    print(f"\n  L6 Mean Sharpe: {np.mean(l6_sharpes):.2f}")
    print(f"  L7(MH=8) Mean Sharpe: {np.mean(sharpes):.2f}")
    print(f"  Improvement: {np.mean(sharpes) - np.mean(l6_sharpes):+.2f}")

    elapsed = time.time() - t0
    print(f"\n# Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Elapsed: {elapsed/60:.1f} minutes")

    sys.stdout = old_stdout
    out.close()
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
