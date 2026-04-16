#!/usr/bin/env python3
"""
EXP33: Mega Trail + Hold Time 联合测试
========================================
EXP30: Hold=20 Sharpe+0.58
EXP32: Mega T0.5/D0.15 Sharpe+3.25
两者独立发现，测试组合效果是否叠加。

无点差。
"""
import sys, os, time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS, run_kfold

print("=" * 70)
print("EXP33: MEGA TRAIL + HOLD TIME COMBINED")
print(f"Started: {datetime.now()}")
print("=" * 70)

t0 = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
print(f"  Load time: {time.time()-t0:.1f}s")

CURRENT = {**C12_KWARGS, "intraday_adaptive": True}

MEGA_REGIME = {
    'low': {'trail_act': 0.7, 'trail_dist': 0.25},
    'normal': {'trail_act': 0.5, 'trail_dist': 0.15},
    'high': {'trail_act': 0.4, 'trail_dist': 0.10},
}

MEGA_BASE = {
    **C12_KWARGS,
    "intraday_adaptive": True,
    "trailing_activate_atr": 0.5,
    "trailing_distance_atr": 0.15,
    "regime_config": MEGA_REGIME,
}

# ═══════════════════════════════════════════════════════════════
# Part 1: Grid — Mega x Hold Time
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 1: MEGA x HOLD TIME GRID (no spread)")
print("=" * 70)

hold_values = [12, 16, 20, 24, 28, 32, 40, 60]

results = []

# A: Current baseline (T0.8/D0.25, Hold=60)
stats = run_variant(data, "A: Current (T0.8 H60)", **CURRENT)
stats['trail'] = 'current'
stats['hold'] = 60
results.append(stats)

# B: Mega only (T0.5/D0.15, Hold=60)
stats = run_variant(data, "B: Mega (T0.5 H60)", **MEGA_BASE)
stats['trail'] = 'mega'
stats['hold'] = 60
results.append(stats)

# C: Current + shorter hold times
for hold in [20, 24, 32]:
    label = f"C: Current T0.8 H{hold}"
    stats = run_variant(data, label, **CURRENT, keltner_max_hold_m15=hold)
    stats['trail'] = 'current'
    stats['hold'] = hold
    results.append(stats)

# D: Mega + all hold times
for hold in hold_values:
    label = f"D: Mega T0.5 H{hold}"
    stats = run_variant(data, label, **MEGA_BASE, keltner_max_hold_m15=hold)
    stats['trail'] = 'mega'
    stats['hold'] = hold
    results.append(stats)

baseline = results[0]

print("\n  RESULTS RANKED BY SHARPE:")
print(f"  {'Variant':<30} {'N':>6} {'Sharpe':>8} {'Delta':>7} {'PnL':>10} {'MaxDD':>10} {'WR%':>6} {'$/t':>7}")
print(f"  {'-'*85}")
for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
    d = r['sharpe'] - baseline['sharpe']
    ppt = r['total_pnl'] / r['n'] if r['n'] > 0 else 0
    print(f"  {r['label']:<30} {r['n']:>6} {r['sharpe']:>8.2f} {d:>+7.2f} "
          f"${r['total_pnl']:>9,.0f} ${r['max_dd']:>9,.0f} {r['win_rate']:>5.1f}% ${ppt:>6.2f}")


# ═══════════════════════════════════════════════════════════════
# Part 2: Exit reason — Current vs Mega H60 vs Best combo
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: EXIT REASON COMPARISON")
print("=" * 70)

best = max(results, key=lambda x: x['sharpe'])
compare = [results[0], results[1], best] if best != results[1] else [results[0], results[1]]

for r in compare:
    trades = r.get('_trades', [])
    if not trades:
        continue
    reasons = {}
    for t in trades:
        key = t.exit_reason.split(':')[0] if ':' in t.exit_reason else t.exit_reason
        if key not in reasons:
            reasons[key] = {'n': 0, 'pnl': 0}
        reasons[key]['n'] += 1
        reasons[key]['pnl'] += t.pnl

    print(f"\n  {r['label']}:")
    print(f"  {'Reason':<20} {'N':>6} {'PnL':>10} {'$/t':>7} {'%trades':>8}")
    for reason, d in sorted(reasons.items(), key=lambda x: x[1]['pnl']):
        ppt = d['pnl'] / d['n'] if d['n'] > 0 else 0
        pct = d['n'] / len(trades) * 100
        print(f"  {reason:<20} {d['n']:>6} ${d['pnl']:>9,.0f} ${ppt:>6.2f} {pct:>7.1f}%")


# ═══════════════════════════════════════════════════════════════
# Part 3: Regime stress test — top 3 configs
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: REGIME STRESS TEST")
print("=" * 70)

scenarios = [
    ("S1 War Escalation", "2020-01-01", "2020-03-01"),
    ("S3 Liquidity Crisis", "2020-03-09", "2020-03-23"),
    ("S4 Slow Decline", "2022-09-01", "2022-12-01"),
    ("S6 Tariff Whipsaw", "2025-04-01", "2025-04-10"),
    ("Low Vol 2018", "2018-01-01", "2018-12-31"),
]

top3 = sorted(results, key=lambda x: x['sharpe'], reverse=True)[:3]
top3_configs = []
for r in top3:
    if r['trail'] == 'mega':
        kw = {**MEGA_BASE}
        if r['hold'] != 60:
            kw['keltner_max_hold_m15'] = r['hold']
    else:
        kw = {**CURRENT}
        if r['hold'] != 60:
            kw['keltner_max_hold_m15'] = r['hold']
    top3_configs.append((r['label'], kw))

# Always include current baseline
if all('Current (T0.8 H60)' not in c[0] for c in top3_configs):
    top3_configs.append(("A: Current (T0.8 H60)", CURRENT))

print(f"\n  {'Scenario':<25}", end="")
for label, _ in top3_configs:
    short = label.split(': ')[1] if ': ' in label else label
    print(f" {short:>18}", end="")
print()
print(f"  {'-' * (25 + 19 * len(top3_configs))}")

for s_name, s_start, s_end in scenarios:
    s_data = data.slice(s_start, s_end)
    if len(s_data.m15_df) < 100:
        continue
    print(f"  {s_name:<25}", end="")
    for label, kw in top3_configs:
        s = run_variant(s_data, f"{s_name}_{label[:10]}", verbose=False, **kw)
        print(f"  Sh={s['sharpe']:>5.2f} ${s['total_pnl']:>6,.0f}", end="")
    print()


# ═══════════════════════════════════════════════════════════════
# Part 4: K-Fold validation
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4: K-FOLD VALIDATION (6 FOLDS)")
print("=" * 70)

kfold_configs = [
    ("Current H60", CURRENT),
    ("Mega H60", MEGA_BASE),
]
best_r = max(results, key=lambda x: x['sharpe'])
if best_r['trail'] == 'mega' and best_r['hold'] != 60:
    kfold_configs.append((
        f"Mega H{best_r['hold']}",
        {**MEGA_BASE, "keltner_max_hold_m15": best_r['hold']},
    ))
# Also test Mega H20 explicitly if not already included
if all('H20' not in c[0] for c in kfold_configs):
    kfold_configs.append((
        "Mega H20",
        {**MEGA_BASE, "keltner_max_hold_m15": 20},
    ))

kfold_results = {}
for name, kwargs in kfold_configs:
    folds = run_kfold(data, kwargs, n_folds=6, label_prefix=f"{name[:6]}_")
    fold_sharpes = [f['sharpe'] for f in folds]
    avg = np.mean(fold_sharpes)
    std = np.std(fold_sharpes)
    kfold_results[name] = {'folds': fold_sharpes, 'avg': avg, 'std': std}
    print(f"\n  {name}: Avg={avg:.2f} Std={std:.2f}")
    for f in folds:
        print(f"    {f['fold']}: Sh={f['sharpe']:.2f}  N={f['n']:,}  PnL=${f['total_pnl']:,.0f}")

base_folds = kfold_results["Current H60"]['folds']
print(f"\n  K-Fold Summary:")
print(f"  {'Config':<20} {'Avg':>6} {'Std':>6} {'Wins vs Current':>16}")
for name, res in kfold_results.items():
    wins = sum(1 for a, b in zip(res['folds'], base_folds) if a > b) if name != "Current H60" else "-"
    w_str = f"{wins}/6" if isinstance(wins, int) else "   -"
    print(f"  {name:<20} {res['avg']:>6.2f} {res['std']:>6.2f} {w_str:>10}")


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

elapsed = time.time() - t0
print("\n" + "=" * 70)
print("EXP33 SUMMARY")
print("=" * 70)
c = results[0]
b = max(results, key=lambda x: x['sharpe'])
d = b['sharpe'] - c['sharpe']
print(f"  Current (T0.8/H60): Sharpe={c['sharpe']:.2f}  PnL=${c['total_pnl']:,.0f}  N={c['n']:,}")
print(f"  Best ({b['label']}): Sharpe={b['sharpe']:.2f} ({d:+.2f})  PnL=${b['total_pnl']:,.0f}  N={b['n']:,}")
mega_h60 = results[1]
d2 = b['sharpe'] - mega_h60['sharpe']
print(f"  vs Mega H60: {d2:+.2f} Sharpe — {'COMBO ADDS VALUE' if d2 > 0.1 else 'NO SIGNIFICANT ADDITIONAL VALUE'}")
print(f"\n  Runtime: {elapsed/60:.1f} minutes")
print(f"  Finished: {datetime.now()}")
