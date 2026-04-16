#!/usr/bin/env python3
"""
EXP41: ATR Regime 动态仓位 — 反波动率加权
============================================
研究表明: 低波动时加大仓位, 高波动时减小仓位 (inverse volatility weighting)
可以显著提升 Sharpe ratio.

当前系统固定 $50 风险/笔. 测试:
- 低 ATR 百分位 (<30%): 风险 $75 (1.5x)
- 正常 ATR: 风险 $50 (1x)
- 高 ATR 百分位 (>70%): 风险 $30 (0.6x)

通过 post-hoc PnL 缩放模拟. 无点差.
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
print("EXP41: ATR REGIME DYNAMIC POSITION SIZING")
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

# Pre-compute ATR percentile for each timestamp
atr_pct_cache = {}
atr_series = h1_df['ATR'].dropna()
for i in range(50, len(atr_series)):
    ts = atr_series.index[i]
    window = atr_series.iloc[i-50:i]
    current_atr = atr_series.iloc[i]
    pct = float((window < current_atr).mean())
    atr_pct_cache[ts] = pct


def get_atr_percentile(entry_time):
    ts = pd.Timestamp(entry_time, tz='UTC')
    idx = h1_df.index.get_indexer([ts], method='ffill')[0]
    if idx < 0:
        return 0.5
    bar_ts = h1_df.index[idx]
    return atr_pct_cache.get(bar_ts, 0.5)


def compute_sharpe(trades, pnls):
    daily = defaultdict(float)
    for t, pnl in zip(trades, pnls):
        day = t.entry_time.strftime('%Y-%m-%d')
        daily[day] += pnl
    vals = list(daily.values())
    if len(vals) > 1 and np.std(vals) > 0:
        return np.mean(vals) / np.std(vals) * np.sqrt(252)
    return 0


# ═══════════════════════════════════════════════════════════════
# Part 1: Trade performance by ATR percentile
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 1: TRADE PERFORMANCE BY ATR PERCENTILE")
print("=" * 70)

baseline = run_variant(data, "Current", **CURRENT)
trades_cur = baseline.get('_trades', [])

buckets = [(0, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 1.01)]
print(f"\n  {'ATR Pct':<12} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6} {'Avg_Bars':>9}")
print(f"  {'-'*55}")

for lo, hi in buckets:
    bt = [(t, get_atr_percentile(t.entry_time)) for t in trades_cur]
    bucket = [t for t, p in bt if lo <= p < hi]
    if not bucket:
        continue
    pnl = sum(t.pnl for t in bucket)
    wins = sum(1 for t in bucket if t.pnl > 0)
    wr = 100 * wins / len(bucket)
    ppt = pnl / len(bucket)
    avg_bars = np.mean([t.bars_held for t in bucket])
    print(f"  [{lo:.1f}-{hi:.1f}){' ':>4} {len(bucket):>6} ${pnl:>9,.0f} ${ppt:>6.2f} {wr:>5.1f}% {avg_bars:>8.1f}")


# ═══════════════════════════════════════════════════════════════
# Part 2: Inverse volatility sizing simulation
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: INVERSE VOLATILITY SIZING — CURRENT")
print("=" * 70)

base_pnls = [t.pnl for t in trades_cur]
base_sharpe = compute_sharpe(trades_cur, base_pnls)
base_total = sum(base_pnls)

print(f"\n  Baseline: N={len(trades_cur):,} Sharpe={base_sharpe:.2f} PnL=${base_total:,.0f}")

sizing_schemes = [
    ("Flat 1.0x (baseline)", {(0, 1.01): 1.0}),
    ("InvVol: Low=1.5x Normal=1.0x High=0.6x", {(0, 0.30): 1.5, (0.30, 0.70): 1.0, (0.70, 1.01): 0.6}),
    ("InvVol: Low=1.3x Normal=1.0x High=0.7x", {(0, 0.30): 1.3, (0.30, 0.70): 1.0, (0.70, 1.01): 0.7}),
    ("InvVol: Low=2.0x Normal=1.0x High=0.5x", {(0, 0.30): 2.0, (0.30, 0.70): 1.0, (0.70, 1.01): 0.5}),
    ("ProVol: Low=0.7x Normal=1.0x High=1.3x", {(0, 0.30): 0.7, (0.30, 0.70): 1.0, (0.70, 1.01): 1.3}),
    ("Skip HighVol (>0.8)", {(0, 0.80): 1.0, (0.80, 1.01): 0.0}),
    ("Skip LowVol (<0.2)", {(0, 0.20): 0.0, (0.20, 1.01): 1.0}),
    ("Smooth: 1.5-atr_pct", {}),  # Special case
]

print(f"\n  {'Scheme':<45} {'N_active':>9} {'PnL':>10} {'Sharpe':>8} {'Delta':>7}")
print(f"  {'-'*82}")

for sname, scheme in sizing_schemes:
    adjusted = []
    n_active = 0
    for t in trades_cur:
        pct = get_atr_percentile(t.entry_time)

        if sname.startswith("Smooth"):
            scale = max(0.5, min(2.0, 1.5 - pct))
        else:
            scale = 1.0
            for (lo, hi), s in scheme.items():
                if lo <= pct < hi:
                    scale = s
                    break

        if scale > 0:
            adjusted.append((t, t.pnl * scale))
            n_active += 1
        # skip if scale == 0

    pnls = [p for _, p in adjusted]
    trades_active = [t for t, _ in adjusted]
    total = sum(pnls)
    sharpe = compute_sharpe(trades_active, pnls)
    d = sharpe - base_sharpe
    print(f"  {sname:<45} {n_active:>9} ${total:>9,.0f} {sharpe:>8.2f} {d:>+7.2f}")


# ═══════════════════════════════════════════════════════════════
# Part 3: Mega Trail — inverse volatility sizing
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: INVERSE VOLATILITY SIZING — MEGA")
print("=" * 70)

mega_stats = run_variant(data, "Mega", **MEGA)
trades_mega = mega_stats.get('_trades', [])

mega_pnls = [t.pnl for t in trades_mega]
mega_sharpe = compute_sharpe(trades_mega, mega_pnls)
mega_total = sum(mega_pnls)

print(f"\n  Mega Baseline: N={len(trades_mega):,} Sharpe={mega_sharpe:.2f} PnL=${mega_total:,.0f}")

print(f"\n  {'Scheme':<45} {'N_active':>9} {'PnL':>10} {'Sharpe':>8} {'Delta':>7}")
print(f"  {'-'*82}")

for sname, scheme in sizing_schemes:
    adjusted = []
    n_active = 0
    for t in trades_mega:
        pct = get_atr_percentile(t.entry_time)

        if sname.startswith("Smooth"):
            scale = max(0.5, min(2.0, 1.5 - pct))
        else:
            scale = 1.0
            for (lo, hi), s in scheme.items():
                if lo <= pct < hi:
                    scale = s
                    break

        if scale > 0:
            adjusted.append((t, t.pnl * scale))
            n_active += 1

    pnls = [p for _, p in adjusted]
    trades_active = [t for t, _ in adjusted]
    total = sum(pnls)
    sharpe = compute_sharpe(trades_active, pnls)
    d = sharpe - mega_sharpe
    print(f"  {sname:<45} {n_active:>9} ${total:>9,.0f} {sharpe:>8.2f} {d:>+7.2f}")


# ═══════════════════════════════════════════════════════════════
# Part 4: Yearly stability of ATR regime sizing
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4: YEARLY — INVERSE VOL (1.5/1.0/0.6) vs BASELINE")
print("=" * 70)

inv_scheme = {(0, 0.30): 1.5, (0.30, 0.70): 1.0, (0.70, 1.01): 0.6}

years = range(2015, 2027)
print(f"\n  {'Year':<6} {'Base_Sh':>8} {'InvVol_Sh':>10} {'Delta':>7} {'Base_PnL':>10} {'InvVol_PnL':>11}")
print(f"  {'-'*55}")

for year in years:
    start = f"{year}-01-01"
    end = f"{year+1}-01-01" if year < 2026 else "2026-04-01"
    yr_trades = [t for t in trades_cur if start <= t.entry_time.strftime('%Y-%m-%d') < end]
    if len(yr_trades) < 20:
        continue

    base_p = [t.pnl for t in yr_trades]
    base_sh = compute_sharpe(yr_trades, base_p)
    base_pnl = sum(base_p)

    inv_p = []
    for t in yr_trades:
        pct = get_atr_percentile(t.entry_time)
        scale = 1.0
        for (lo, hi), s in inv_scheme.items():
            if lo <= pct < hi:
                scale = s
                break
        inv_p.append(t.pnl * scale)
    inv_sh = compute_sharpe(yr_trades, inv_p)
    inv_pnl = sum(inv_p)
    d = inv_sh - base_sh

    print(f"  {year:<6} {base_sh:>8.2f} {inv_sh:>10.2f} {d:>+7.2f} ${base_pnl:>9,.0f} ${inv_pnl:>10,.0f}")


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

elapsed = time.time() - t0
print("\n" + "=" * 70)
print("EXP41 SUMMARY")
print("=" * 70)
print(f"  Current: Sharpe={base_sharpe:.2f} PnL=${base_total:,.0f}")
print(f"  Mega: Sharpe={mega_sharpe:.2f} PnL=${mega_total:,.0f}")
print(f"\n  Runtime: {elapsed/60:.1f} minutes")
print(f"  Finished: {datetime.now()}")
