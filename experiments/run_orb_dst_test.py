#!/usr/bin/env python3
"""
ORB Strategy DST & Diagnosis Test
===================================
Tests ORB with current LIVE_PARITY parameters:
  1. Baseline: ORB ON, UTC 14 (current config)
  2. No ORB: Keltner only
  3. DST fix: ORB ON, UTC 13 (summer time NY open)
  4. ORB ON, UTC 13+14 both (hybrid: scan both hours)

Also tests ORB max_hold variants: 4/6/8/12/16/24 bars.

Usage: python experiments/run_orb_dst_test.py
"""
import sys, os, time, gc
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import research_config as config
from backtest import DataBundle, run_variant
from backtest.runner import LIVE_PARITY_KWARGS

OUTPUT_FILE = "orb_dst_test_output.txt"


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

print("=" * 80)
print("ORB STRATEGY DST & DIAGNOSIS TEST")
print(f"Started: {datetime.now()}")
print(f"Base params: LIVE_PARITY_KWARGS (L8_BASE)")
print("=" * 80)

t_total = time.time()

BASE_KW = {
    **LIVE_PARITY_KWARGS,
    "spread_cost": 0.30,
    "min_entry_gap_hours": 1.0,
    "maxloss_cap": 37.0,
}


def run_with_orb_config(label, orb_enabled, orb_hour=14, extra_kwargs=None):
    """Run variant with ORB config monkey-patched."""
    old_enabled = config.ORB_ENABLED
    old_hour = config.ORB_NY_OPEN_HOUR_UTC
    config.ORB_ENABLED = orb_enabled
    config.ORB_NY_OPEN_HOUR_UTC = orb_hour
    kwargs = {**BASE_KW}
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    stats = run_variant(data, label, **kwargs)
    config.ORB_ENABLED = old_enabled
    config.ORB_NY_OPEN_HOUR_UTC = old_hour
    return stats


# Load data
data = DataBundle.load_default()
gc.collect()

results = []

# ══════════════════════════════════════════════════
# Part 1: UTC Hour comparison
# ══════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 1: ORB HOUR COMPARISON (UTC 14 vs UTC 13 vs OFF)")
print("=" * 80)

s = run_with_orb_config("A: ORB OFF (Keltner only)", False)
results.append(s)

s = run_with_orb_config("B: ORB UTC14 (current)", True, 14)
results.append(s)

s = run_with_orb_config("C: ORB UTC13 (DST fix)", True, 13)
results.append(s)

gc.collect()

# ══════════════════════════════════════════════════
# Part 2: ORB max_hold scan (on UTC 14 baseline)
# ══════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 2: ORB MAX HOLD SCAN (UTC 14)")
print("=" * 80)

hold_results = []
for hold in [4, 6, 8, 12, 16, 24]:
    label = f"Hold={hold} (~{hold*15/60:.1f}h)"
    s = run_with_orb_config(label, True, 14, {"orb_max_hold_m15": hold})
    hold_results.append(s)
    gc.collect()

# ══════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════

def strat_breakdown(trades):
    strats = {}
    for t in trades:
        s = t.strategy
        if s not in strats:
            strats[s] = {'n': 0, 'pnl': 0.0, 'wins': 0}
        strats[s]['n'] += 1
        strats[s]['pnl'] += t.pnl
        if t.pnl > 0:
            strats[s]['wins'] += 1
    return strats


print("\n\n" + "=" * 80)
print("PART 1 SUMMARY: UTC HOUR COMPARISON")
print("=" * 80)
hdr = f"{'Variant':<32} {'N':>5} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'MaxDD':>8}"
print(hdr)
print("-" * len(hdr))
for r in results:
    print(f"{r['label']:<32} {r['n']:>5} {r['sharpe']:>7.2f} ${r['total_pnl']:>9,.0f} "
          f"{r['win_rate']:>5.1f}% ${r['max_dd']:>7,.0f}")

print("\n\nPER-STRATEGY BREAKDOWN")
print("-" * 80)
for r in results:
    sb = strat_breakdown(r['_trades'])
    print(f"\n  {r['label']}:")
    print(f"  {'Strategy':<15} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
    print(f"  {'-'*48}")
    for s in sorted(sb.keys()):
        d = sb[s]
        wr = 100.0 * d['wins'] / d['n'] if d['n'] > 0 else 0
        ppt = d['pnl'] / d['n'] if d['n'] > 0 else 0
        print(f"  {s:<15} {d['n']:>6} ${d['pnl']:>9,.0f} ${ppt:>6.2f} {wr:>5.1f}%")


print("\n\n" + "=" * 80)
print("PART 2 SUMMARY: MAX HOLD SCAN (UTC 14)")
print("=" * 80)
print(f"\n  {'Hold':<20} {'Total':>6} {'Sharpe':>8} {'ORB_N':>6} "
      f"{'ORB_PnL':>9} {'ORB_WR':>7} {'K_PnL':>9}")
print("  " + "-" * 70)
for s in hold_results:
    sb = strat_breakdown(s['_trades'])
    orb = sb.get('orb', {'n': 0, 'pnl': 0, 'wins': 0})
    k = sb.get('keltner', {'n': 0, 'pnl': 0, 'wins': 0})
    orb_wr = 100.0 * orb['wins'] / orb['n'] if orb['n'] > 0 else 0
    print(f"  {s['label']:<20} {s['n']:>6} {s['sharpe']:>8.2f} {orb['n']:>6} "
          f"${orb['pnl']:>8.0f} {orb_wr:>6.1f}% ${k['pnl']:>8.0f}")


print("\n\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
if len(results) >= 3:
    no_orb = results[0]
    utc14 = results[1]
    utc13 = results[2]
    print(f"  No ORB:    Sharpe={no_orb['sharpe']:.2f}, PnL=${no_orb['total_pnl']:,.0f}")
    print(f"  ORB UTC14: Sharpe={utc14['sharpe']:.2f}, PnL=${utc14['total_pnl']:,.0f} (current)")
    print(f"  ORB UTC13: Sharpe={utc13['sharpe']:.2f}, PnL=${utc13['total_pnl']:,.0f} (DST fix)")
    
    sh_diff_14 = utc14['sharpe'] - no_orb['sharpe']
    sh_diff_13 = utc13['sharpe'] - no_orb['sharpe']
    
    print(f"\n  ORB UTC14 vs No ORB: Sharpe delta = {sh_diff_14:+.2f}")
    print(f"  ORB UTC13 vs No ORB: Sharpe delta = {sh_diff_13:+.2f}")
    print(f"  ORB UTC13 vs UTC14:  Sharpe delta = {utc13['sharpe'] - utc14['sharpe']:+.2f}")
    
    if utc13['sharpe'] > utc14['sharpe'] + 0.05:
        print("\n  >>> UTC13 (DST fix) is better. Consider switching.")
    elif utc14['sharpe'] > utc13['sharpe'] + 0.05:
        print("\n  >>> UTC14 (current) is better. Keep as is.")
    else:
        print("\n  >>> Minimal difference. DST fix is cosmetic, not impactful.")
    
    if no_orb['sharpe'] > max(utc14['sharpe'], utc13['sharpe']) + 0.05:
        print("  >>> ORB is dragging the portfolio. Consider disabling.")
    elif max(utc14['sharpe'], utc13['sharpe']) > no_orb['sharpe'] + 0.05:
        print("  >>> ORB adds value. Keep it.")
    else:
        print("  >>> ORB has minimal impact on overall portfolio.")

total_elapsed = time.time() - t_total
print(f"\n  Total runtime: {total_elapsed/60:.1f} minutes")
print(f"  Output saved to: {OUTPUT_FILE}")
print(f"  Finished: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nDone! Results in {OUTPUT_FILE}")
