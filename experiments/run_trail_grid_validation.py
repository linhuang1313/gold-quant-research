#!/usr/bin/env python3
"""
Trail Parameter Grid — Fixed Engine + LIVE_PARITY + MaxHold=20 + Spread Cost
=============================================================================
Base: LIVE_PARITY_KWARGS with MaxHold=20 (validated K-Fold 6/6)
Grid: Trail Activate × Trail Distance (normal regime), regime offsets auto-computed
Test: $0.30 + $0.50 spread, K-Fold top candidates

The regime table is derived from normal params:
  low:    trail_act + 0.2,  trail_dist + 0.10
  normal: trail_act,        trail_dist
  high:   trail_act * 0.5,  trail_dist * 0.33  (T7 OnlyHigh style tightening)
"""
import sys, os, time, gc
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from backtest import DataBundle, run_variant
from backtest.runner import LIVE_PARITY_KWARGS

OUTPUT_FILE = "trail_grid_validation_output.txt"

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-01"),
]


class TeeOutput:
    def __init__(self, fp):
        self.file = open(fp, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, d):
        self.stdout.write(d)
        self.file.write(d)
        self.file.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()


tee = TeeOutput(OUTPUT_FILE)
sys.stdout = tee


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def make_regime(ta, td):
    """Derive regime table from normal params, keeping T7 OnlyHigh style for high."""
    return {
        'low':    {'trail_act': round(ta + 0.2, 2),  'trail_dist': round(td + 0.10, 2)},
        'normal': {'trail_act': ta,                    'trail_dist': td},
        'high':   {'trail_act': round(ta * 0.5, 2),   'trail_dist': round(td * 0.33, 2)},
    }


print("=" * 80)
print("TRAIL PARAMETER GRID — LIVE_PARITY + MaxHold=20 + Spread Cost")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()

print("\nLoading data...")
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

BASE_KWARGS = {**LIVE_PARITY_KWARGS, "keltner_max_hold_m15": 20}

TRAIL_ACT_RANGE  = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
TRAIL_DIST_RANGE = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]

total = len(TRAIL_ACT_RANGE) * len(TRAIL_DIST_RANGE)
print(f"\nGrid: {len(TRAIL_ACT_RANGE)} TA x {len(TRAIL_DIST_RANGE)} TD = {total} combos")
print(f"Base: LIVE_PARITY + MaxHold=20, Regime auto-derived from normal params")
print(f"Current live: TA=0.50, TD=0.15 (normal regime)")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Full grid at $0.30
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PHASE 1: FULL GRID @ $0.30 SPREAD")
print("=" * 80)

header = f"{'TA':>5s} {'TD':>5s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'WR%':>5s}  {'$/t':>7s}  {'MaxDD':>11s}"
print(f"\n{header}")
print("-" * 75)

grid_results = {}
idx = 0
for ta in TRAIL_ACT_RANGE:
    for td in TRAIL_DIST_RANGE:
        idx += 1
        regime = make_regime(ta, td)
        kwargs = {
            **BASE_KWARGS,
            "trailing_activate_atr": ta,
            "trailing_distance_atr": td,
            "regime_config": regime,
            "spread_cost": 0.30,
        }
        label = f"TA{ta}_TD{td}_sp030"
        s = run_variant(data, label, verbose=False, **kwargs)
        n = s['n']
        avg = s['total_pnl'] / n if n > 0 else 0
        marker = " <-- current" if (ta == 0.50 and td == 0.15) else ""
        print(f" {ta:>4.2f} {td:>5.2f}  {n:>5d}  {s['sharpe']:>6.2f}  "
              f"{fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  "
              f"${avg:>6.2f}  {fmt(s['max_dd'])}{marker}")
        grid_results[(ta, td)] = s
        gc.collect()

# Rank by Sharpe
ranked = sorted(grid_results.items(), key=lambda x: -x[1]['sharpe'])

print(f"\n--- Top 10 by Sharpe ($0.30) ---")
for i, ((ta, td), s) in enumerate(ranked[:10]):
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    cur = " <-- current" if (ta == 0.50 and td == 0.15) else ""
    print(f"  #{i+1}: TA={ta:.2f} TD={td:.2f}  Sharpe={s['sharpe']:.2f}  "
          f"PnL={fmt(s['total_pnl'])}  MaxDD={fmt(s['max_dd'])}{cur}")

current_s = grid_results.get((0.50, 0.15))
current_rank = [i+1 for i, ((ta,td),_) in enumerate(ranked) if ta==0.50 and td==0.15]
if current_s:
    print(f"\n  Current (TA=0.50, TD=0.15): rank #{current_rank[0]}/{total}, Sharpe={current_s['sharpe']:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Top-3 (excl current) at $0.50 stress test
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PHASE 2: TOP-3 CANDIDATES @ $0.50 STRESS TEST")
print("=" * 80)

top3_candidates = [(ta, td) for (ta, td), _ in ranked if not (ta == 0.50 and td == 0.15)][:3]
print(f"  Candidates: {top3_candidates}")
print(f"  + Current (0.50, 0.15) for comparison\n")

test_params = top3_candidates + [(0.50, 0.15)]

print(f"{'TA':>5s} {'TD':>5s}  {'Spread':>6s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'$/t':>7s}  {'MaxDD':>11s}")
print("-" * 75)

for ta, td in test_params:
    regime = make_regime(ta, td)
    for spread in [0.30, 0.50]:
        kwargs = {
            **BASE_KWARGS,
            "trailing_activate_atr": ta,
            "trailing_distance_atr": td,
            "regime_config": regime,
            "spread_cost": spread,
        }
        s = run_variant(data, f"TA{ta}_TD{td}_sp{int(spread*100):03d}", verbose=False, **kwargs)
        n = s['n']
        avg = s['total_pnl'] / n if n > 0 else 0
        cur = " <-- current" if (ta == 0.50 and td == 0.15) else ""
        print(f" {ta:>4.2f} {td:>5.2f}  ${spread:.2f}  {n:>5d}  {s['sharpe']:>6.2f}  "
              f"{fmt(s['total_pnl'])}  ${avg:>6.2f}  {fmt(s['max_dd'])}{cur}")
    gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: K-Fold validation for candidates that beat current at both spreads
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PHASE 3: K-FOLD VALIDATION @ $0.30")
print("  Pass criteria: >= 5/6 folds")
print("=" * 80)

for ta, td in top3_candidates:
    regime = make_regime(ta, td)
    print(f"\n  TA={ta:.2f}, TD={td:.2f} (regime: low={regime['low']}, high={regime['high']}):")

    wins = 0
    for fold_name, start, end in FOLDS:
        fold_data = data.slice(start, end)
        if len(fold_data.m15_df) < 1000 or len(fold_data.h1_df) < 200:
            continue

        # Baseline = current params (TA=0.50, TD=0.15) + MaxHold=20
        sb = run_variant(fold_data, f"Base_{fold_name}", verbose=False,
                         **BASE_KWARGS, spread_cost=0.30)

        # Test variant
        st = run_variant(fold_data, f"TA{ta}_TD{td}_{fold_name}", verbose=False,
                         **{**BASE_KWARGS,
                            "trailing_activate_atr": ta,
                            "trailing_distance_atr": td,
                            "regime_config": regime},
                         spread_cost=0.30)

        delta = st['sharpe'] - sb['sharpe']
        won = delta > 0
        if won:
            wins += 1
        print(f"    {fold_name}: Base={sb['sharpe']:>6.2f}  Test={st['sharpe']:>6.2f}  "
              f"delta={delta:>+.2f} {'V' if won else 'X'}")

    result = "PASS" if wins >= 5 else "FAIL"
    print(f"    Result: {wins}/6 {result}")


# ══════════════════════════════════════════════════════════════════════════════
# MARGINAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("MARGINAL ANALYSIS ($0.30 spread)")
print("=" * 80)

import numpy as np

print("\n  Trail Activate (avg Sharpe across all TD):")
for ta in TRAIL_ACT_RANGE:
    sharpes = [grid_results[(ta, td)]['sharpe'] for td in TRAIL_DIST_RANGE]
    print(f"    TA={ta:.2f}: avg={np.mean(sharpes):.2f}, min={min(sharpes):.2f}, max={max(sharpes):.2f}")

print("\n  Trail Distance (avg Sharpe across all TA):")
for td in TRAIL_DIST_RANGE:
    sharpes = [grid_results[(ta, td)]['sharpe'] for ta in TRAIL_ACT_RANGE]
    print(f"    TD={td:.2f}: avg={np.mean(sharpes):.2f}, min={min(sharpes):.2f}, max={max(sharpes):.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

elapsed = time.time() - t_total
print(f"\n  Total runtime: {elapsed/60:.1f} minutes")
print(f"  Completed: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nResults saved to {OUTPUT_FILE}")
