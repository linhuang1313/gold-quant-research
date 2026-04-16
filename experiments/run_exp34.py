#!/usr/bin/env python3
"""
EXP34: 周内择时 Day-of-Week 过滤
==================================
IC 扫描显示 day_of_week 是最稳定因子之一 (IC=+0.033, WF=100%)。
测试: 哪些天的 Keltner 信号最赚钱? 某些天是否应降低仓位或跳过?

无点差。
"""
import sys, os, time
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS, run_kfold

print("=" * 70)
print("EXP34: DAY-OF-WEEK TIMING ANALYSIS")
print(f"Started: {datetime.now()}")
print("=" * 70)

t0 = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
print(f"  Load time: {time.time()-t0:.1f}s")

CURRENT = {**C12_KWARGS, "intraday_adaptive": True}
DOW_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

# ═══════════════════════════════════════════════════════════════
# Part 1: Per-day-of-week trade analysis (from baseline trades)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 1: BASELINE TRADE BREAKDOWN BY DAY-OF-WEEK")
print("=" * 70)

baseline = run_variant(data, "Baseline", **CURRENT)
trades = baseline.get('_trades', [])

by_dow = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0, 'losses': 0,
                                'sl_n': 0, 'sl_pnl': 0, 'trail_n': 0, 'trail_pnl': 0,
                                'tp_n': 0, 'tp_pnl': 0, 'timeout_n': 0, 'timeout_pnl': 0})

for t in trades:
    dow = t.entry_time.weekday()
    d = by_dow[dow]
    d['n'] += 1
    d['pnl'] += t.pnl
    if t.pnl > 0:
        d['wins'] += 1
    else:
        d['losses'] += 1

    reason = t.exit_reason.split(':')[0] if ':' in t.exit_reason else t.exit_reason
    if reason == 'sl':
        d['sl_n'] += 1; d['sl_pnl'] += t.pnl
    elif 'trailing' in reason:
        d['trail_n'] += 1; d['trail_pnl'] += t.pnl
    elif reason == 'tp':
        d['tp_n'] += 1; d['tp_pnl'] += t.pnl
    elif reason in ('timeout', 'time_stop'):
        d['timeout_n'] += 1; d['timeout_pnl'] += t.pnl

print(f"\n  {'Day':<5} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6} {'SL_N':>5} {'SL_PnL':>9} {'Trail_N':>8} {'Trail_PnL':>10} {'TO_N':>5} {'TO_PnL':>9}")
print(f"  {'-'*90}")
for dow in range(5):
    d = by_dow[dow]
    if d['n'] == 0:
        continue
    wr = 100 * d['wins'] / d['n']
    ppt = d['pnl'] / d['n']
    print(f"  {DOW_NAMES[dow]:<5} {d['n']:>6} ${d['pnl']:>9,.0f} ${ppt:>6.2f} {wr:>5.1f}% "
          f"{d['sl_n']:>5} ${d['sl_pnl']:>8,.0f} {d['trail_n']:>8} ${d['trail_pnl']:>9,.0f} "
          f"{d['timeout_n']:>5} ${d['timeout_pnl']:>8,.0f}")

# Find worst and best days
day_ppt = {dow: by_dow[dow]['pnl'] / by_dow[dow]['n'] for dow in range(5) if by_dow[dow]['n'] > 0}
worst_dow = min(day_ppt, key=day_ppt.get)
best_dow = max(day_ppt, key=day_ppt.get)
print(f"\n  Best day: {DOW_NAMES[best_dow]} (${day_ppt[best_dow]:+.2f}/trade)")
print(f"  Worst day: {DOW_NAMES[worst_dow]} (${day_ppt[worst_dow]:+.2f}/trade)")


# ═══════════════════════════════════════════════════════════════
# Part 2: Per-day direction analysis (BUY vs SELL per day)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: DAY x DIRECTION BREAKDOWN")
print("=" * 70)

by_dow_dir = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0})
for t in trades:
    dow = t.entry_time.weekday()
    key = (dow, t.direction)
    d = by_dow_dir[key]
    d['n'] += 1
    d['pnl'] += t.pnl
    if t.pnl > 0:
        d['wins'] += 1

print(f"\n  {'Day':<5} {'Dir':<5} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
print(f"  {'-'*45}")
for dow in range(5):
    for direction in ['BUY', 'SELL']:
        d = by_dow_dir[(dow, direction)]
        if d['n'] == 0:
            continue
        wr = 100 * d['wins'] / d['n']
        ppt = d['pnl'] / d['n']
        print(f"  {DOW_NAMES[dow]:<5} {direction:<5} {d['n']:>6} ${d['pnl']:>9,.0f} ${ppt:>6.2f} {wr:>5.1f}%")


# ═══════════════════════════════════════════════════════════════
# Part 3: Per-day strategy breakdown (keltner vs orb vs rsi etc)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: DAY x STRATEGY BREAKDOWN")
print("=" * 70)

by_dow_strat = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0})
strat_names = set()
for t in trades:
    dow = t.entry_time.weekday()
    strat = t.strategy
    strat_names.add(strat)
    d = by_dow_strat[(dow, strat)]
    d['n'] += 1
    d['pnl'] += t.pnl
    if t.pnl > 0:
        d['wins'] += 1

for strat in sorted(strat_names):
    print(f"\n  Strategy: {strat}")
    print(f"  {'Day':<5} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
    print(f"  {'-'*38}")
    for dow in range(5):
        d = by_dow_strat[(dow, strat)]
        if d['n'] == 0:
            continue
        wr = 100 * d['wins'] / d['n']
        ppt = d['pnl'] / d['n']
        print(f"  {DOW_NAMES[dow]:<5} {d['n']:>6} ${d['pnl']:>9,.0f} ${ppt:>6.2f} {wr:>5.1f}%")


# ═══════════════════════════════════════════════════════════════
# Part 4: Skip worst day — backtest comparison
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4: SKIP-DAY BACKTEST COMPARISON")
print("=" * 70)

# To test "skip day X", we filter trades post-hoc
# (engine doesn't have skip-day param, so we compute from existing trades)

print(f"\n  Baseline: N={baseline['n']:,} Sharpe={baseline['sharpe']:.2f} PnL=${baseline['total_pnl']:,.0f}")

skip_results = []
for skip_dow in range(5):
    kept = [t for t in trades if t.entry_time.weekday() != skip_dow]
    from backtest.stats import calc_stats
    if len(kept) < 10:
        continue
    equity = [0.0]
    for t in kept:
        equity.append(equity[-1] + t.pnl)
    stats = calc_stats(kept, equity)
    stats['skipped'] = DOW_NAMES[skip_dow]
    skip_results.append(stats)
    d = stats['sharpe'] - baseline['sharpe']
    print(f"  Skip {DOW_NAMES[skip_dow]}: N={stats['n']:,} Sharpe={stats['sharpe']:.2f} ({d:+.2f}) "
          f"PnL=${stats['total_pnl']:,.0f} MaxDD=${stats['max_dd']:,.0f}")

# Also test skip worst 2 days
if len(day_ppt) >= 3:
    sorted_days = sorted(day_ppt, key=day_ppt.get)
    worst2 = sorted_days[:2]
    kept = [t for t in trades if t.entry_time.weekday() not in worst2]
    if len(kept) >= 10:
        equity = [0.0]
        for t in kept:
            equity.append(equity[-1] + t.pnl)
        stats = calc_stats(kept, equity)
        d = stats['sharpe'] - baseline['sharpe']
        names = '+'.join(DOW_NAMES[w] for w in worst2)
        print(f"  Skip {names}: N={stats['n']:,} Sharpe={stats['sharpe']:.2f} ({d:+.2f}) "
              f"PnL=${stats['total_pnl']:,.0f} MaxDD=${stats['max_dd']:,.0f}")


# ═══════════════════════════════════════════════════════════════
# Part 5: Yearly stability of day-of-week patterns
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 5: YEARLY DOW PATTERN STABILITY")
print("=" * 70)

years = range(2015, 2027)
print(f"\n  {'Year':<6}", end="")
for d in DOW_NAMES:
    print(f" {d:>10}", end="")
print()
print(f"  {'-'*60}")

year_dow_data = {}
for year in years:
    start = f"{year}-01-01"
    end = f"{year+1}-01-01" if year < 2026 else "2026-04-01"
    yr_trades = [t for t in trades
                 if t.entry_time.year == year or (year == 2026 and t.entry_time.year == 2026)]
    # More precise filter
    yr_trades = [t for t in trades
                 if start <= t.entry_time.strftime('%Y-%m-%d') < end]
    if not yr_trades:
        continue

    print(f"  {year:<6}", end="")
    for dow in range(5):
        dt = [t for t in yr_trades if t.entry_time.weekday() == dow]
        if not dt:
            print(f" {'--':>10}", end="")
        else:
            ppt = sum(t.pnl for t in dt) / len(dt)
            print(f" ${ppt:>8.2f}", end="")
    print()

# Count how many years each day is the worst
worst_counts = defaultdict(int)
best_counts = defaultdict(int)
for year in years:
    start = f"{year}-01-01"
    end = f"{year+1}-01-01" if year < 2026 else "2026-04-01"
    yr_trades = [t for t in trades if start <= t.entry_time.strftime('%Y-%m-%d') < end]
    if not yr_trades:
        continue
    yr_dow_ppt = {}
    for dow in range(5):
        dt = [t for t in yr_trades if t.entry_time.weekday() == dow]
        if dt:
            yr_dow_ppt[dow] = sum(t.pnl for t in dt) / len(dt)
    if yr_dow_ppt:
        worst_counts[min(yr_dow_ppt, key=yr_dow_ppt.get)] += 1
        best_counts[max(yr_dow_ppt, key=yr_dow_ppt.get)] += 1

print(f"\n  Worst day frequency (across years):")
for dow in range(5):
    print(f"    {DOW_NAMES[dow]}: worst in {worst_counts.get(dow, 0)} years, best in {best_counts.get(dow, 0)} years")


# ═══════════════════════════════════════════════════════════════
# Part 6: Mega Trail + DOW filter combined
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 6: MEGA TRAIL x DOW ANALYSIS")
print("=" * 70)

MEGA = {
    **C12_KWARGS,
    "intraday_adaptive": True,
    "trailing_activate_atr": 0.5,
    "trailing_distance_atr": 0.15,
    "regime_config": {
        'low': {'trail_act': 0.7, 'trail_dist': 0.25},
        'normal': {'trail_act': 0.5, 'trail_dist': 0.15},
        'high': {'trail_act': 0.4, 'trail_dist': 0.10},
    },
}

mega_stats = run_variant(data, "Mega Baseline", **MEGA)
mega_trades = mega_stats.get('_trades', [])

print(f"\n  Mega DOW breakdown:")
print(f"  {'Day':<5} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
print(f"  {'-'*38}")
mega_dow_ppt = {}
for dow in range(5):
    dt = [t for t in mega_trades if t.entry_time.weekday() == dow]
    if not dt:
        continue
    pnl = sum(t.pnl for t in dt)
    wins = sum(1 for t in dt if t.pnl > 0)
    wr = 100 * wins / len(dt)
    ppt = pnl / len(dt)
    mega_dow_ppt[dow] = ppt
    print(f"  {DOW_NAMES[dow]:<5} {len(dt):>6} ${pnl:>9,.0f} ${ppt:>6.2f} {wr:>5.1f}%")

# Skip worst day for Mega
if mega_dow_ppt:
    worst_mega = min(mega_dow_ppt, key=mega_dow_ppt.get)
    kept = [t for t in mega_trades if t.entry_time.weekday() != worst_mega]
    if len(kept) >= 10:
        equity = [0.0]
        for t in kept:
            equity.append(equity[-1] + t.pnl)
        from backtest.stats import calc_stats
        stats = calc_stats(kept, equity)
        d = stats['sharpe'] - mega_stats['sharpe']
        print(f"\n  Mega skip {DOW_NAMES[worst_mega]}: Sharpe={stats['sharpe']:.2f} ({d:+.2f}) "
              f"PnL=${stats['total_pnl']:,.0f}")


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

elapsed = time.time() - t0
print("\n" + "=" * 70)
print("EXP34 SUMMARY")
print("=" * 70)
print(f"  Baseline: Sharpe={baseline['sharpe']:.2f} PnL=${baseline['total_pnl']:,.0f}")
print(f"  Best DOW: {DOW_NAMES[best_dow]} (${day_ppt[best_dow]:+.2f}/t)")
print(f"  Worst DOW: {DOW_NAMES[worst_dow]} (${day_ppt[worst_dow]:+.2f}/t)")
if skip_results:
    best_skip = max(skip_results, key=lambda x: x['sharpe'])
    d = best_skip['sharpe'] - baseline['sharpe']
    print(f"  Best skip-day: Skip {best_skip['skipped']} → Sharpe={best_skip['sharpe']:.2f} ({d:+.2f})")
print(f"\n  Runtime: {elapsed/60:.1f} minutes")
print(f"  Finished: {datetime.now()}")
