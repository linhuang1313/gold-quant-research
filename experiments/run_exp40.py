#!/usr/bin/env python3
"""
EXP40: K线质量过滤 — 突破K线实体/影线比
==========================================
研究表明: 突破K线的实体占比越大, 突破越真实.
- 影线长实体短 = 假突破(wick rejection)
- 实体大影线短 = 真突破(conviction candle)

测试: 在 Keltner 信号触发时, 检查突破K线的实体占比:
- body_ratio = |close-open| / (high-low)
- body_ratio > 0.5 → 强突破 → 正常入场
- body_ratio < 0.3 → 弱突破(wick拒绝) → 跳过

通过 post-hoc 交易分析. 无点差.
"""
import sys, os, time
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS
from backtest.stats import calc_stats

print("=" * 70)
print("EXP40: CANDLE BODY RATIO FILTER")
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

h1_df = data.h1_df.copy()


def get_body_ratio(entry_time):
    """Get body/range ratio of the H1 candle at entry time."""
    ts = pd.Timestamp(entry_time, tz='UTC')
    # Find the H1 bar closest to entry time
    idx = h1_df.index.get_indexer([ts], method='ffill')[0]
    if idx < 0 or idx >= len(h1_df):
        return None
    bar = h1_df.iloc[idx]
    range_ = float(bar['High'] - bar['Low'])
    if range_ <= 0:
        return None
    body = abs(float(bar['Close'] - bar['Open']))
    return body / range_


def get_upper_wick_ratio(entry_time, direction):
    """Get rejection wick ratio — opposing wick / range."""
    ts = pd.Timestamp(entry_time, tz='UTC')
    idx = h1_df.index.get_indexer([ts], method='ffill')[0]
    if idx < 0 or idx >= len(h1_df):
        return None
    bar = h1_df.iloc[idx]
    range_ = float(bar['High'] - bar['Low'])
    if range_ <= 0:
        return None
    if direction == 'BUY':
        # Upper wick = rejection if long
        wick = float(bar['High'] - max(bar['Close'], bar['Open']))
    else:
        wick = float(min(bar['Close'], bar['Open']) - bar['Low'])
    return wick / range_


# ═══════════════════════════════════════════════════════════════
# Part 1: Body ratio distribution by trade outcome
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 1: BODY RATIO vs TRADE OUTCOME (CURRENT)")
print("=" * 70)

baseline = run_variant(data, "Current", **CURRENT)
trades_cur = baseline.get('_trades', [])
keltner_cur = [t for t in trades_cur if t.strategy == 'keltner']

ratios_win = []
ratios_lose = []
for t in keltner_cur:
    br = get_body_ratio(t.entry_time)
    if br is None:
        continue
    if t.pnl > 0:
        ratios_win.append(br)
    else:
        ratios_lose.append(br)

print(f"\n  Winning trades body ratio: mean={np.mean(ratios_win):.3f} median={np.median(ratios_win):.3f}")
print(f"  Losing trades body ratio:  mean={np.mean(ratios_lose):.3f} median={np.median(ratios_lose):.3f}")
print(f"  Difference: {np.mean(ratios_win) - np.mean(ratios_lose):+.4f}")

# Distribution buckets
buckets = [(0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]
print(f"\n  {'Body Ratio':<15} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
print(f"  {'-'*48}")

for lo, hi in buckets:
    bucket_trades = []
    for t in keltner_cur:
        br = get_body_ratio(t.entry_time)
        if br is not None and lo <= br < hi:
            bucket_trades.append(t)
    if not bucket_trades:
        continue
    pnl = sum(t.pnl for t in bucket_trades)
    wins = sum(1 for t in bucket_trades if t.pnl > 0)
    wr = 100 * wins / len(bucket_trades)
    ppt = pnl / len(bucket_trades)
    print(f"  [{lo:.1f}-{hi:.1f}){' ':>6} {len(bucket_trades):>6} ${pnl:>9,.0f} ${ppt:>6.2f} {wr:>5.1f}%")


# ═══════════════════════════════════════════════════════════════
# Part 2: Rejection wick analysis
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: REJECTION WICK RATIO vs OUTCOME")
print("=" * 70)

wick_buckets = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.6), (0.6, 1.01)]
print(f"\n  {'Wick Ratio':<15} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
print(f"  {'-'*48}")

for lo, hi in wick_buckets:
    bucket_trades = []
    for t in keltner_cur:
        wr_val = get_upper_wick_ratio(t.entry_time, t.direction)
        if wr_val is not None and lo <= wr_val < hi:
            bucket_trades.append(t)
    if not bucket_trades:
        continue
    pnl = sum(t.pnl for t in bucket_trades)
    wins = sum(1 for t in bucket_trades if t.pnl > 0)
    wr = 100 * wins / len(bucket_trades)
    ppt = pnl / len(bucket_trades)
    print(f"  [{lo:.1f}-{hi:.1f}){' ':>6} {len(bucket_trades):>6} ${pnl:>9,.0f} ${ppt:>6.2f} {wr:>5.1f}%")


# ═══════════════════════════════════════════════════════════════
# Part 3: Body ratio filter — Sharpe comparison
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: BODY RATIO FILTER — SHARPE COMPARISON")
print("=" * 70)

print(f"\n  Baseline: N={baseline['n']:,} Sharpe={baseline['sharpe']:.2f} PnL=${baseline['total_pnl']:,.0f}")

filters = [
    ("No filter (baseline)", lambda t: True),
    ("Body > 0.2", lambda t: t.strategy != 'keltner' or (get_body_ratio(t.entry_time) or 0.5) > 0.2),
    ("Body > 0.3", lambda t: t.strategy != 'keltner' or (get_body_ratio(t.entry_time) or 0.5) > 0.3),
    ("Body > 0.4", lambda t: t.strategy != 'keltner' or (get_body_ratio(t.entry_time) or 0.5) > 0.4),
    ("Body > 0.5", lambda t: t.strategy != 'keltner' or (get_body_ratio(t.entry_time) or 0.5) > 0.5),
    ("Wick < 0.3", lambda t: t.strategy != 'keltner' or (get_upper_wick_ratio(t.entry_time, t.direction) or 0) < 0.3),
    ("Wick < 0.4", lambda t: t.strategy != 'keltner' or (get_upper_wick_ratio(t.entry_time, t.direction) or 0) < 0.4),
    ("Body>0.3 + Wick<0.4", lambda t: t.strategy != 'keltner' or (
        (get_body_ratio(t.entry_time) or 0.5) > 0.3 and
        (get_upper_wick_ratio(t.entry_time, t.direction) or 0) < 0.4)),
]

print(f"\n  {'Filter':<30} {'N':>6} {'Sharpe':>8} {'Delta':>7} {'PnL':>10} {'$/t':>7}")
print(f"  {'-'*70}")

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
    print(f"  {fname:<30} {stats['n']:>6} {stats['sharpe']:>8.2f} {d:>+7.2f} ${stats['total_pnl']:>9,.0f} ${ppt:>6.2f}")


# ═══════════════════════════════════════════════════════════════
# Part 4: Mega Trail — body ratio analysis
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4: MEGA TRAIL — BODY RATIO ANALYSIS")
print("=" * 70)

mega_stats = run_variant(data, "Mega", **MEGA)
trades_mega = mega_stats.get('_trades', [])

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
    print(f"  {fname:<30} {stats['n']:>6} {stats['sharpe']:>8.2f} {d:>+7.2f} ${stats['total_pnl']:>9,.0f} ${ppt:>6.2f}")


# ═══════════════════════════════════════════════════════════════
# Part 5: Yearly stability
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 5: YEARLY BODY RATIO PATTERN (CURRENT)")
print("=" * 70)

years = range(2015, 2027)
print(f"\n  {'Year':<6} {'LowBody(<0.3)':>14} {'MidBody':>10} {'HighBody(>0.5)':>15}")
print(f"  {'-'*48}")

for year in years:
    start = f"{year}-01-01"
    end = f"{year+1}-01-01" if year < 2026 else "2026-04-01"
    yr_trades = [t for t in keltner_cur if start <= t.entry_time.strftime('%Y-%m-%d') < end]
    if not yr_trades:
        continue

    lo_ppt = mid_ppt = hi_ppt = '--'
    lo = [t for t in yr_trades if (get_body_ratio(t.entry_time) or 0.5) < 0.3]
    mid = [t for t in yr_trades if 0.3 <= (get_body_ratio(t.entry_time) or 0.5) < 0.5]
    hi = [t for t in yr_trades if (get_body_ratio(t.entry_time) or 0.5) >= 0.5]

    lo_ppt = f"${sum(t.pnl for t in lo)/len(lo):.2f}" if lo else "--"
    mid_ppt = f"${sum(t.pnl for t in mid)/len(mid):.2f}" if mid else "--"
    hi_ppt = f"${sum(t.pnl for t in hi)/len(hi):.2f}" if hi else "--"

    print(f"  {year:<6} {lo_ppt:>14} {mid_ppt:>10} {hi_ppt:>15}")


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

elapsed = time.time() - t0
print("\n" + "=" * 70)
print("EXP40 SUMMARY")
print("=" * 70)
print(f"  Win body ratio mean: {np.mean(ratios_win):.3f}")
print(f"  Lose body ratio mean: {np.mean(ratios_lose):.3f}")
print(f"\n  Runtime: {elapsed/60:.1f} minutes")
print(f"  Finished: {datetime.now()}")
