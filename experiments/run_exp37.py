#!/usr/bin/env python3
"""
EXP37: 分批止盈 Partial Profit Taking
=======================================
当前策略 all-in/all-out. 测试分批止盈:
- 到达 N×ATR 时平掉一半, 剩余继续 trailing
- 对比全量 trailing vs 分批锁利

通过 post-hoc 交易回放模拟, 无需改引擎. 无点差.
"""
import sys, os, time
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS
from backtest.stats import calc_stats

print("=" * 70)
print("EXP37: PARTIAL PROFIT TAKING ANALYSIS")
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


def simulate_partial(trades, partial_at_pct=0.5, partial_ratio=0.5):
    """
    Post-hoc: for each trade, simulate taking partial profit.
    partial_at_pct: take partial when PnL reaches this fraction of final MFE
    partial_ratio: fraction of position to close at partial level

    For trailing exits with positive MFE: split into two portions.
    For SL/Timeout exits: partial doesn't help if MFE was never reached.
    """
    adjusted_pnls = []
    n_partial_taken = 0

    for t in trades:
        mfe = getattr(t, 'mfe', 0) or 0
        pnl = t.pnl
        reason = t.exit_reason.split(':')[0] if ':' in t.exit_reason else t.exit_reason

        # MFE in dollar terms (already per lot in our system)
        # partial_at_pct means "take partial when profit = partial_at_pct of sl_distance"
        partial_level = t.sl_distance * partial_at_pct

        if mfe > partial_level and partial_level > 0:
            # Partial was taken: partial_ratio of position locked at partial_level
            locked_pnl = partial_level * partial_ratio
            # Remaining runs to actual exit
            remaining_pnl = pnl * (1 - partial_ratio)
            total = locked_pnl + remaining_pnl
            adjusted_pnls.append(total)
            n_partial_taken += 1
        else:
            adjusted_pnls.append(pnl)

    return adjusted_pnls, n_partial_taken


def compute_sharpe_from_trades(trades, pnls):
    """Compute daily Sharpe from trade-level PnLs."""
    daily = defaultdict(float)
    for t, pnl in zip(trades, pnls):
        day = t.entry_time.strftime('%Y-%m-%d')
        daily[day] += pnl
    vals = list(daily.values())
    if len(vals) > 1 and np.std(vals) > 0:
        return np.mean(vals) / np.std(vals) * np.sqrt(252)
    return 0


# ═══════════════════════════════════════════════════════════════
# Part 1: Current strategy — partial profit scan
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 1: CURRENT STRATEGY — PARTIAL PROFIT SCAN")
print("=" * 70)

baseline = run_variant(data, "Current", **CURRENT)
trades_cur = baseline.get('_trades', [])

base_pnls = [t.pnl for t in trades_cur]
base_sharpe = compute_sharpe_from_trades(trades_cur, base_pnls)
base_total = sum(base_pnls)

print(f"\n  Baseline: N={len(trades_cur):,} Sharpe={base_sharpe:.2f} PnL=${base_total:,.0f}")

configs = [
    ("No partial (baseline)", 0, 0),
    ("Partial @0.3xSL, close 30%", 0.3, 0.3),
    ("Partial @0.3xSL, close 50%", 0.3, 0.5),
    ("Partial @0.5xSL, close 30%", 0.5, 0.3),
    ("Partial @0.5xSL, close 50%", 0.5, 0.5),
    ("Partial @0.5xSL, close 70%", 0.5, 0.7),
    ("Partial @1.0xSL, close 30%", 1.0, 0.3),
    ("Partial @1.0xSL, close 50%", 1.0, 0.5),
    ("Partial @1.5xSL, close 50%", 1.5, 0.5),
    ("Partial @2.0xSL, close 50%", 2.0, 0.5),
]

print(f"\n  {'Config':<35} {'N_partial':>10} {'PnL':>10} {'Sharpe':>8} {'Delta':>7} {'$/t':>7}")
print(f"  {'-'*80}")

for cname, at_pct, ratio in configs:
    if at_pct == 0:
        pnls = base_pnls
        n_p = 0
    else:
        pnls, n_p = simulate_partial(trades_cur, at_pct, ratio)
    total = sum(pnls)
    sharpe = compute_sharpe_from_trades(trades_cur, pnls)
    d = sharpe - base_sharpe
    ppt = total / len(pnls) if pnls else 0
    print(f"  {cname:<35} {n_p:>10} ${total:>9,.0f} {sharpe:>8.2f} {d:>+7.2f} ${ppt:>6.2f}")


# ═══════════════════════════════════════════════════════════════
# Part 2: Mega Trail — partial profit scan
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: MEGA TRAIL — PARTIAL PROFIT SCAN")
print("=" * 70)

mega_stats = run_variant(data, "Mega", **MEGA)
trades_mega = mega_stats.get('_trades', [])

mega_pnls = [t.pnl for t in trades_mega]
mega_sharpe = compute_sharpe_from_trades(trades_mega, mega_pnls)
mega_total = sum(mega_pnls)

print(f"\n  Mega Baseline: N={len(trades_mega):,} Sharpe={mega_sharpe:.2f} PnL=${mega_total:,.0f}")

print(f"\n  {'Config':<35} {'N_partial':>10} {'PnL':>10} {'Sharpe':>8} {'Delta':>7} {'$/t':>7}")
print(f"  {'-'*80}")

for cname, at_pct, ratio in configs:
    if at_pct == 0:
        pnls = mega_pnls
        n_p = 0
    else:
        pnls, n_p = simulate_partial(trades_mega, at_pct, ratio)
    total = sum(pnls)
    sharpe = compute_sharpe_from_trades(trades_mega, pnls)
    d = sharpe - mega_sharpe
    ppt = total / len(pnls) if pnls else 0
    print(f"  {cname:<35} {n_p:>10} ${total:>9,.0f} {sharpe:>8.2f} {d:>+7.2f} ${ppt:>6.2f}")


# ═══════════════════════════════════════════════════════════════
# Part 3: MFE distribution analysis
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: MFE DISTRIBUTION (when does max profit occur?)")
print("=" * 70)

for label, trades in [("Current", trades_cur), ("Mega", trades_mega)]:
    mfes = [getattr(t, 'mfe', 0) or 0 for t in trades]
    sls = [t.sl_distance for t in trades if t.sl_distance > 0]
    mfe_sl_ratios = [m / t.sl_distance for t, m in zip(trades, mfes)
                     if t.sl_distance > 0 and m > 0]

    if not mfe_sl_ratios:
        continue

    print(f"\n  {label}:")
    print(f"    MFE/SL ratio: mean={np.mean(mfe_sl_ratios):.2f} "
          f"median={np.median(mfe_sl_ratios):.2f} "
          f"p25={np.percentile(mfe_sl_ratios, 25):.2f} "
          f"p75={np.percentile(mfe_sl_ratios, 75):.2f}")

    # How many trades reach various MFE levels
    for threshold in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
        reached = sum(1 for r in mfe_sl_ratios if r >= threshold)
        pct = 100 * reached / len(mfe_sl_ratios)
        print(f"    MFE >= {threshold:.1f}xSL: {reached:,} ({pct:.1f}%)")

    # MFE of losing trades specifically
    losing = [(t, getattr(t, 'mfe', 0) or 0) for t in trades if t.pnl <= 0]
    if losing:
        losing_mfe_ratios = [m / t.sl_distance for t, m in losing
                             if t.sl_distance > 0]
        if losing_mfe_ratios:
            print(f"    Losing trades MFE/SL: mean={np.mean(losing_mfe_ratios):.2f} "
                  f"median={np.median(losing_mfe_ratios):.2f}")
            reached_05 = sum(1 for r in losing_mfe_ratios if r >= 0.5)
            print(f"    Losing trades with MFE >= 0.5xSL: {reached_05} "
                  f"({100*reached_05/len(losing_mfe_ratios):.1f}%) — salvageable with partial")


# ═══════════════════════════════════════════════════════════════
# Part 4: Exit reason split — where does partial help most?
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4: PARTIAL PROFIT IMPACT BY EXIT REASON")
print("=" * 70)

for label, trades in [("Current", trades_cur), ("Mega", trades_mega)]:
    best_at, best_ratio = 0.5, 0.5
    pnls, _ = simulate_partial(trades, best_at, best_ratio)

    by_reason = defaultdict(lambda: {'orig': 0, 'partial': 0, 'n': 0})
    for t, adj_pnl in zip(trades, pnls):
        reason = t.exit_reason.split(':')[0] if ':' in t.exit_reason else t.exit_reason
        by_reason[reason]['orig'] += t.pnl
        by_reason[reason]['partial'] += adj_pnl
        by_reason[reason]['n'] += 1

    print(f"\n  {label} (Partial @{best_at}xSL, close {best_ratio*100:.0f}%):")
    print(f"  {'Reason':<20} {'N':>6} {'Orig_PnL':>10} {'Partial_PnL':>12} {'Diff':>10}")
    print(f"  {'-'*60}")
    for reason in sorted(by_reason, key=lambda r: by_reason[r]['orig']):
        d = by_reason[reason]
        diff = d['partial'] - d['orig']
        print(f"  {reason:<20} {d['n']:>6} ${d['orig']:>9,.0f} ${d['partial']:>11,.0f} ${diff:>9,.0f}")


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

elapsed = time.time() - t0
print("\n" + "=" * 70)
print("EXP37 SUMMARY")
print("=" * 70)
print(f"  Current: Sharpe={base_sharpe:.2f} PnL=${base_total:,.0f}")
print(f"  Mega: Sharpe={mega_sharpe:.2f} PnL=${mega_total:,.0f}")
print(f"\n  Runtime: {elapsed/60:.1f} minutes")
print(f"  Finished: {datetime.now()}")
