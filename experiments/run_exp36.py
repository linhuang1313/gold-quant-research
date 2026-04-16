#!/usr/bin/env python3
"""
EXP36: 交易时段过滤 Hour-of-Day
=================================
黄金在不同时段波动率差异大:
- 亚盘 (UTC 0-7): 低波动, 均值回归为主
- 伦敦 (UTC 7-13): 中高波动, 趋势启动
- 纽约 (UTC 13-17): 最高波动, 趋势延续
- 尾盘 (UTC 17-21): 波动收缩

Keltner 是趋势策略, 理论上只在高波动时段开仓可能提升 Sharpe.
无点差.
"""
import sys, os, time
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS, run_kfold
from backtest.stats import calc_stats

print("=" * 70)
print("EXP36: HOUR-OF-DAY SESSION ANALYSIS")
print(f"Started: {datetime.now()}")
print("=" * 70)

t0 = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
print(f"  Load time: {time.time()-t0:.1f}s")

CURRENT = {**C12_KWARGS, "intraday_adaptive": True}
MEGA = {
    **C12_KWARGS, "intraday_adaptive": True,
    "trailing_activate_atr": 0.5, "trailing_distance_atr": 0.15,
    "regime_config": {
        'low': {'trail_act': 0.7, 'trail_dist': 0.25},
        'normal': {'trail_act': 0.5, 'trail_dist': 0.15},
        'high': {'trail_act': 0.4, 'trail_dist': 0.10},
    },
}

SESSIONS = {
    'Asia (0-7)': range(0, 7),
    'London (7-13)': range(7, 13),
    'NY (13-17)': range(13, 17),
    'LDN-NY Overlap (13-16)': range(13, 16),
    'Late (17-21)': range(17, 21),
    'Night (21-24)': range(21, 24),
}


def analyze_by_hour(trades, label):
    print(f"\n  {label}:")
    by_hour = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0,
                                    'sl': 0, 'trail': 0, 'timeout': 0})
    for t in trades:
        h = t.entry_time.hour
        d = by_hour[h]
        d['n'] += 1
        d['pnl'] += t.pnl
        if t.pnl > 0:
            d['wins'] += 1
        reason = t.exit_reason.split(':')[0] if ':' in t.exit_reason else t.exit_reason
        if reason == 'SL':
            d['sl'] += 1
        elif 'railing' in reason:
            d['trail'] += 1
        elif reason in ('Timeout', 'time_stop'):
            d['timeout'] += 1

    print(f"  {'Hour':>4} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6} {'SL':>4} {'Trail':>6} {'TO':>4}")
    print(f"  {'-'*55}")
    for h in range(24):
        d = by_hour[h]
        if d['n'] == 0:
            continue
        wr = 100 * d['wins'] / d['n']
        ppt = d['pnl'] / d['n']
        print(f"  {h:>4} {d['n']:>6} ${d['pnl']:>9,.0f} ${ppt:>6.2f} {wr:>5.1f}% {d['sl']:>4} {d['trail']:>6} {d['timeout']:>4}")

    return by_hour


def analyze_sessions(trades, by_hour, label):
    print(f"\n  Session summary ({label}):")
    print(f"  {'Session':<25} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
    print(f"  {'-'*58}")
    session_ppt = {}
    for sname, hours in SESSIONS.items():
        n = sum(by_hour[h]['n'] for h in hours)
        pnl = sum(by_hour[h]['pnl'] for h in hours)
        wins = sum(by_hour[h]['wins'] for h in hours)
        if n == 0:
            continue
        wr = 100 * wins / n
        ppt = pnl / n
        session_ppt[sname] = ppt
        print(f"  {sname:<25} {n:>6} ${pnl:>9,.0f} ${ppt:>6.2f} {wr:>5.1f}%")
    return session_ppt


# ═══════════════════════════════════════════════════════════════
# Part 1: Current strategy hourly breakdown
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 1: CURRENT STRATEGY — HOURLY BREAKDOWN")
print("=" * 70)

baseline = run_variant(data, "Current", **CURRENT)
trades_cur = baseline.get('_trades', [])
by_hour_cur = analyze_by_hour(trades_cur, "Current (T0.8/H60)")
session_ppt_cur = analyze_sessions(trades_cur, by_hour_cur, "Current")


# ═══════════════════════════════════════════════════════════════
# Part 2: Mega Trail hourly breakdown
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: MEGA TRAIL — HOURLY BREAKDOWN")
print("=" * 70)

mega_stats = run_variant(data, "Mega", **MEGA)
trades_mega = mega_stats.get('_trades', [])
by_hour_mega = analyze_by_hour(trades_mega, "Mega (T0.5/D0.15)")
session_ppt_mega = analyze_sessions(trades_mega, by_hour_mega, "Mega")


# ═══════════════════════════════════════════════════════════════
# Part 3: Strategy-specific hourly breakdown (keltner vs orb)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: BY STRATEGY x HOUR")
print("=" * 70)

strat_names = set(t.strategy for t in trades_cur)
for strat in sorted(strat_names):
    strat_trades = [t for t in trades_cur if t.strategy == strat]
    if len(strat_trades) < 10:
        continue
    analyze_by_hour(strat_trades, f"Strategy: {strat}")


# ═══════════════════════════════════════════════════════════════
# Part 4: Skip-hour backtest (post-hoc)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4: SKIP-HOUR ANALYSIS (CURRENT)")
print("=" * 70)

# Find worst hours
hour_ppt = {h: by_hour_cur[h]['pnl'] / by_hour_cur[h]['n']
            for h in range(24) if by_hour_cur[h]['n'] > 20}
sorted_hours = sorted(hour_ppt, key=hour_ppt.get)

print(f"\n  Worst 5 hours: {[(h, f'${hour_ppt[h]:.2f}') for h in sorted_hours[:5]]}")
print(f"  Best 5 hours:  {[(h, f'${hour_ppt[h]:.2f}') for h in sorted_hours[-5:]]}")

# Test various session filters
print(f"\n  Baseline: N={baseline['n']:,} Sharpe={baseline['sharpe']:.2f} PnL=${baseline['total_pnl']:,.0f}")

filters = [
    ("Skip Asia (0-7)", lambda t: t.entry_time.hour not in range(0, 7)),
    ("Skip Late (17-21)", lambda t: t.entry_time.hour not in range(17, 21)),
    ("Skip Night (21-24)", lambda t: t.entry_time.hour not in range(21, 24)),
    ("Only London+NY (7-17)", lambda t: 7 <= t.entry_time.hour < 17),
    ("Only LDN-NY Overlap (13-16)", lambda t: 13 <= t.entry_time.hour < 16),
    ("Skip worst 3 hours", lambda t: t.entry_time.hour not in sorted_hours[:3]),
    ("Skip worst 5 hours", lambda t: t.entry_time.hour not in sorted_hours[:5]),
]

print(f"\n  {'Filter':<35} {'N':>6} {'Sharpe':>8} {'Delta':>7} {'PnL':>10} {'$/t':>7}")
print(f"  {'-'*75}")

for fname, ffunc in filters:
    kept = [t for t in trades_cur if ffunc(t)]
    if len(kept) < 50:
        continue
    equity = [0.0]
    for t in kept:
        equity.append(equity[-1] + t.pnl)
    stats = calc_stats(kept, equity)
    d = stats['sharpe'] - baseline['sharpe']
    ppt = stats['total_pnl'] / stats['n'] if stats['n'] > 0 else 0
    print(f"  {fname:<35} {stats['n']:>6} {stats['sharpe']:>8.2f} {d:>+7.2f} ${stats['total_pnl']:>9,.0f} ${ppt:>6.2f}")


# ═══════════════════════════════════════════════════════════════
# Part 5: Skip-hour for Mega Trail
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 5: SKIP-HOUR ANALYSIS (MEGA)")
print("=" * 70)

mega_hour_ppt = {h: by_hour_mega[h]['pnl'] / by_hour_mega[h]['n']
                 for h in range(24) if by_hour_mega[h]['n'] > 20}
mega_sorted = sorted(mega_hour_ppt, key=mega_hour_ppt.get)

print(f"\n  Mega Baseline: N={mega_stats['n']:,} Sharpe={mega_stats['sharpe']:.2f} PnL=${mega_stats['total_pnl']:,.0f}")

for fname, ffunc in filters:
    kept = [t for t in trades_mega if ffunc(t)]
    if len(kept) < 50:
        continue
    equity = [0.0]
    for t in kept:
        equity.append(equity[-1] + t.pnl)
    stats = calc_stats(kept, equity)
    d = stats['sharpe'] - mega_stats['sharpe']
    ppt = stats['total_pnl'] / stats['n'] if stats['n'] > 0 else 0
    print(f"  {fname:<35} {stats['n']:>6} {stats['sharpe']:>8.2f} {d:>+7.2f} ${stats['total_pnl']:>9,.0f} ${ppt:>6.2f}")


# ═══════════════════════════════════════════════════════════════
# Part 6: Yearly stability of hourly pattern
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 6: YEARLY SESSION STABILITY")
print("=" * 70)

years = range(2015, 2027)
sessions_simple = {
    'Asia': range(0, 7), 'London': range(7, 13),
    'NY': range(13, 17), 'Late': range(17, 22),
}

print(f"\n  {'Year':<6}", end="")
for s in sessions_simple:
    print(f" {s:>12}", end="")
print()
print(f"  {'-'*60}")

for year in years:
    start = f"{year}-01-01"
    end = f"{year+1}-01-01" if year < 2026 else "2026-04-01"
    yr_trades = [t for t in trades_cur if start <= t.entry_time.strftime('%Y-%m-%d') < end]
    if not yr_trades:
        continue
    print(f"  {year:<6}", end="")
    for sname, hours in sessions_simple.items():
        st = [t for t in yr_trades if t.entry_time.hour in hours]
        if not st:
            print(f" {'--':>12}", end="")
        else:
            ppt = sum(t.pnl for t in st) / len(st)
            print(f"  ${ppt:>9.2f}", end="")
    print()


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

elapsed = time.time() - t0
print("\n" + "=" * 70)
print("EXP36 SUMMARY")
print("=" * 70)
print(f"  Current: Sharpe={baseline['sharpe']:.2f} PnL=${baseline['total_pnl']:,.0f}")
print(f"  Mega: Sharpe={mega_stats['sharpe']:.2f} PnL=${mega_stats['total_pnl']:,.0f}")
print(f"\n  Runtime: {elapsed/60:.1f} minutes")
print(f"  Finished: {datetime.now()}")
