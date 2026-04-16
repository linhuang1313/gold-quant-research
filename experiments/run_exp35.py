#!/usr/bin/env python3
"""
EXP35: 连续亏损自适应减仓分析
================================
待办: 连续 5+ SELL 后减仓 (均值 -$0.07/笔, 7+ SELL 后 -$0.34/笔)
扩展: 连续 N 笔亏损后, 下一笔的期望收益是什么? 应该减仓/暂停/无影响?

无点差。
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
print("EXP35: CONSECUTIVE LOSS ADAPTIVE SIZING")
print(f"Started: {datetime.now()}")
print("=" * 70)

t0 = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
print(f"  Load time: {time.time()-t0:.1f}s")

CURRENT = {**C12_KWARGS, "intraday_adaptive": True}

baseline = run_variant(data, "Baseline", **CURRENT)
trades = baseline.get('_trades', [])

print(f"\n  Baseline: N={baseline['n']:,} Sharpe={baseline['sharpe']:.2f} PnL=${baseline['total_pnl']:,.0f}")


# ═══════════════════════════════════════════════════════════════
# Part 1: Consecutive loss streak analysis
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 1: AFTER N CONSECUTIVE LOSSES — NEXT TRADE EXPECTATION")
print("=" * 70)

streaks = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0, 'next_pnl_list': []})
loss_streak = 0

for i, t in enumerate(trades):
    if t.pnl <= 0:
        loss_streak += 1
    else:
        loss_streak = 0

    if i + 1 < len(trades):
        next_t = trades[i + 1]
        streaks[loss_streak]['n'] += 1
        streaks[loss_streak]['pnl'] += next_t.pnl
        streaks[loss_streak]['next_pnl_list'].append(next_t.pnl)
        if next_t.pnl > 0:
            streaks[loss_streak]['wins'] += 1

print(f"\n  {'Streak':<10} {'Next_N':>7} {'Next_AvgPnL':>12} {'Next_WR%':>9} {'Next_StdPnL':>12} {'ShouldTrade':>12}")
print(f"  {'-'*65}")
for streak in sorted(streaks.keys()):
    d = streaks[streak]
    if d['n'] < 5:
        continue
    avg_pnl = d['pnl'] / d['n']
    wr = 100 * d['wins'] / d['n']
    std_pnl = np.std(d['next_pnl_list']) if len(d['next_pnl_list']) > 1 else 0
    should = "YES" if avg_pnl > 0 else "REDUCE" if avg_pnl > -2 else "SKIP"
    print(f"  {streak:<10} {d['n']:>7} ${avg_pnl:>10.2f} {wr:>8.1f}% ${std_pnl:>10.2f} {should:>12}")


# ═══════════════════════════════════════════════════════════════
# Part 2: Consecutive same-direction analysis
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: AFTER N CONSECUTIVE SAME-DIRECTION TRADES")
print("=" * 70)

for direction in ['BUY', 'SELL']:
    dir_streaks = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0})
    same_dir_count = 0

    for i, t in enumerate(trades):
        if t.direction == direction:
            same_dir_count += 1
        else:
            same_dir_count = 0

        if i + 1 < len(trades):
            next_t = trades[i + 1]
            if next_t.direction == direction and same_dir_count >= 1:
                dir_streaks[same_dir_count]['n'] += 1
                dir_streaks[same_dir_count]['pnl'] += next_t.pnl
                if next_t.pnl > 0:
                    dir_streaks[same_dir_count]['wins'] += 1

    print(f"\n  Direction: {direction}")
    print(f"  {'Consec':>7} {'Next_N':>7} {'Next_AvgPnL':>12} {'Next_WR%':>9}")
    print(f"  {'-'*40}")
    for streak in sorted(dir_streaks.keys()):
        d = dir_streaks[streak]
        if d['n'] < 5:
            continue
        avg_pnl = d['pnl'] / d['n']
        wr = 100 * d['wins'] / d['n']
        print(f"  {streak:>7} {d['n']:>7} ${avg_pnl:>10.2f} {wr:>8.1f}%")


# ═══════════════════════════════════════════════════════════════
# Part 3: Daily cumulative loss — should we stop after N daily losses?
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: DAILY LOSS COUNT — NEXT TRADE EXPECTATION")
print("=" * 70)

daily_trades = defaultdict(list)
for t in trades:
    day = t.entry_time.strftime('%Y-%m-%d')
    daily_trades[day].append(t)

daily_loss_next = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0})

for day, day_ts in daily_trades.items():
    loss_so_far = 0
    for i, t in enumerate(day_ts):
        if t.pnl <= 0:
            loss_so_far += 1
        if i + 1 < len(day_ts):
            next_t = day_ts[i + 1]
            daily_loss_next[loss_so_far]['n'] += 1
            daily_loss_next[loss_so_far]['pnl'] += next_t.pnl
            if next_t.pnl > 0:
                daily_loss_next[loss_so_far]['wins'] += 1

print(f"\n  {'Daily_Losses':>12} {'Next_N':>7} {'Next_AvgPnL':>12} {'Next_WR%':>9} {'Action':>10}")
print(f"  {'-'*55}")
for losses in sorted(daily_loss_next.keys()):
    d = daily_loss_next[losses]
    if d['n'] < 5:
        continue
    avg_pnl = d['pnl'] / d['n']
    wr = 100 * d['wins'] / d['n']
    action = "NORMAL" if avg_pnl > 1 else "CAUTION" if avg_pnl > 0 else "REDUCE"
    print(f"  {losses:>12} {d['n']:>7} ${avg_pnl:>10.2f} {wr:>8.1f}% {action:>10}")


# ═══════════════════════════════════════════════════════════════
# Part 4: Simulate reduced-size after N losses
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4: SIMULATED REDUCED SIZING AFTER CONSECUTIVE LOSSES")
print("=" * 70)

# Strategies: after N consecutive losses, scale PnL by multiplier
strategies_test = [
    ("Baseline (no change)", {}),
    ("After 2 losses → 0.5x", {2: 0.5}),
    ("After 3 losses → 0.5x", {3: 0.5}),
    ("After 2 losses → 0.5x, 4+ → 0.25x", {2: 0.5, 4: 0.25}),
    ("After 3 losses → skip", {3: 0.0}),
    ("After 4 losses → skip", {4: 0.0}),
    ("After 2 losses → 0.75x", {2: 0.75}),
    ("After 3 losses → 0.75x, 5+ → 0.5x", {3: 0.75, 5: 0.5}),
]

print(f"\n  {'Strategy':<45} {'N_active':>9} {'PnL':>10} {'Sharpe':>8} {'Delta':>7}")
print(f"  {'-'*82}")

for strat_name, rules in strategies_test:
    loss_streak = 0
    adjusted_trades = []
    n_reduced = 0

    for t in trades:
        scale = 1.0
        if rules:
            for threshold in sorted(rules.keys(), reverse=True):
                if loss_streak >= threshold:
                    scale = rules[threshold]
                    break

        if scale > 0:
            adj_pnl = t.pnl * scale
            adjusted_trades.append(adj_pnl)
            if scale < 1.0:
                n_reduced += 1
        else:
            n_reduced += 1

        if t.pnl <= 0:
            loss_streak += 1
        else:
            loss_streak = 0

    total_pnl = sum(adjusted_trades)
    n_active = len(adjusted_trades)

    # Compute Sharpe from daily returns
    daily_pnl = defaultdict(float)
    idx = 0
    loss_streak = 0
    for t in trades:
        scale = 1.0
        if rules:
            for threshold in sorted(rules.keys(), reverse=True):
                if loss_streak >= threshold:
                    scale = rules[threshold]
                    break
        if scale > 0:
            day = t.entry_time.strftime('%Y-%m-%d')
            daily_pnl[day] += t.pnl * scale
        if t.pnl <= 0:
            loss_streak += 1
        else:
            loss_streak = 0

    daily_returns = list(daily_pnl.values())
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    else:
        sharpe = 0
    d = sharpe - baseline['sharpe']
    print(f"  {strat_name:<45} {n_active:>9} ${total_pnl:>9,.0f} {sharpe:>8.2f} {d:>+7.2f}")


# ═══════════════════════════════════════════════════════════════
# Part 5: Weekly drawdown auto-reduce
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 5: WEEKLY CUMULATIVE DRAWDOWN TRIGGER")
print("=" * 70)

# Compute weekly PnL and test: if week DD > threshold, reduce next week's size
weekly_pnl = defaultdict(float)
for t in trades:
    week = t.entry_time.strftime('%Y-W%W')
    weekly_pnl[week] += t.pnl

weeks = sorted(weekly_pnl.keys())
weekly_vals = [weekly_pnl[w] for w in weeks]

print(f"\n  Weekly PnL stats: mean=${np.mean(weekly_vals):.1f} std=${np.std(weekly_vals):.1f} "
      f"min=${min(weekly_vals):.1f} max=${max(weekly_vals):.1f}")
print(f"  Weeks with loss: {sum(1 for v in weekly_vals if v < 0)}/{len(weekly_vals)}")

# Simulate: after a losing week, scale next week's trades
thresholds = [0, -25, -50, -75, -100, -150]
print(f"\n  {'Week_DD_Threshold':>18} {'Reduced_Weeks':>14} {'Total_PnL':>10} {'Sharpe':>8}")
print(f"  {'-'*55}")

for threshold in thresholds:
    prev_week_loss = False
    adjusted_pnl = defaultdict(float)
    reduced_weeks = 0

    for i, week in enumerate(weeks):
        scale = 0.5 if prev_week_loss else 1.0
        if prev_week_loss:
            reduced_weeks += 1
        adjusted_pnl[week] = weekly_pnl[week] * scale
        prev_week_loss = (weekly_pnl[week] < threshold)

    total = sum(adjusted_pnl.values())
    vals = list(adjusted_pnl.values())
    if len(vals) > 1 and np.std(vals) > 0:
        sharpe = np.mean(vals) / np.std(vals) * np.sqrt(52)
    else:
        sharpe = 0
    label = f"<${threshold}" if threshold < 0 else "no filter"
    print(f"  {label:>18} {reduced_weeks:>14} ${total:>9,.0f} {sharpe:>8.2f}")


# ═══════════════════════════════════════════════════════════════
# Part 6: Mega Trail + loss streak
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 6: MEGA TRAIL + CONSECUTIVE LOSS ANALYSIS")
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

mega_stats = run_variant(data, "Mega", **MEGA)
mega_trades = mega_stats.get('_trades', [])

mega_streaks = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0})
loss_streak = 0
for i, t in enumerate(mega_trades):
    if t.pnl <= 0:
        loss_streak += 1
    else:
        loss_streak = 0
    if i + 1 < len(mega_trades):
        next_t = mega_trades[i + 1]
        mega_streaks[loss_streak]['n'] += 1
        mega_streaks[loss_streak]['pnl'] += next_t.pnl
        if next_t.pnl > 0:
            mega_streaks[loss_streak]['wins'] += 1

print(f"\n  Mega: After N consecutive losses — next trade:")
print(f"  {'Streak':<10} {'Next_N':>7} {'Next_AvgPnL':>12} {'Next_WR%':>9}")
print(f"  {'-'*42}")
for streak in sorted(mega_streaks.keys()):
    d = mega_streaks[streak]
    if d['n'] < 5:
        continue
    avg_pnl = d['pnl'] / d['n']
    wr = 100 * d['wins'] / d['n']
    print(f"  {streak:<10} {d['n']:>7} ${avg_pnl:>10.2f} {wr:>8.1f}%")

# Max streak length
max_loss_streak = 0
curr = 0
for t in mega_trades:
    if t.pnl <= 0:
        curr += 1
        max_loss_streak = max(max_loss_streak, curr)
    else:
        curr = 0
print(f"\n  Mega max consecutive losses: {max_loss_streak}")

curr = 0
for t in trades:
    if t.pnl <= 0:
        curr += 1
        max_loss_streak = max(max_loss_streak, curr)
    else:
        curr = 0
print(f"  Current max consecutive losses: {max_loss_streak}")


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

elapsed = time.time() - t0
print("\n" + "=" * 70)
print("EXP35 SUMMARY")
print("=" * 70)
print(f"  Baseline: Sharpe={baseline['sharpe']:.2f} PnL=${baseline['total_pnl']:,.0f}")
print(f"  Key findings:")
print(f"    - Consecutive loss streaks and their impact on next trade expectation")
print(f"    - Daily loss count impact")
print(f"    - Simulated reduced sizing strategies")
print(f"    - Weekly drawdown triggers")
print(f"\n  Runtime: {elapsed/60:.1f} minutes")
print(f"  Finished: {datetime.now()}")
