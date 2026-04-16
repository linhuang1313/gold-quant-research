#!/usr/bin/env python3
"""
EXP39: D1 日线方向过滤
=======================
当前用 H1 EMA100 做趋势过滤. 测试加一层 D1 级别:
- D1 EMA50 向上 → 只允许 BUY
- D1 EMA50 向下 → 只允许 SELL
- D1 无方向/横盘 → 双向都可

通过 post-hoc 交易过滤: 构建 D1 趋势, 然后过滤不符合方向的交易.
无点差.
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
print("EXP39: D1 DAILY TREND DIRECTION FILTER")
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

# ═══════════════════════════════════════════════════════════════
# Build D1 trend signals from H1 data
# ═══════════════════════════════════════════════════════════════

print("\n  Building D1 trend indicators from H1 data...")

h1_df = data.h1_df.copy()

# Resample H1 to D1
d1_df = h1_df.resample('1D').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
    'Volume': 'sum'
}).dropna()

# Compute D1 EMAs
for span in [20, 50, 100, 200]:
    d1_df[f'D1_EMA{span}'] = d1_df['Close'].ewm(span=span, adjust=False).mean()

# D1 ADX
high = d1_df['High']
low = d1_df['Low']
close = d1_df['Close']
tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
d1_df['D1_ATR'] = tr.rolling(14).mean()

plus_dm = (high - high.shift()).clip(lower=0)
minus_dm = (low.shift() - low).clip(lower=0)
plus_dm[plus_dm < minus_dm] = 0
minus_dm[minus_dm < plus_dm] = 0
smooth_plus = plus_dm.rolling(14).mean()
smooth_minus = minus_dm.rolling(14).mean()
plus_di = 100 * smooth_plus / d1_df['D1_ATR']
minus_di = 100 * smooth_minus / d1_df['D1_ATR']
dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
d1_df['D1_ADX'] = dx.rolling(14).mean()

# D1 trend direction
d1_df['D1_trend'] = 'NEUTRAL'
d1_df.loc[d1_df['Close'] > d1_df['D1_EMA50'], 'D1_trend'] = 'UP'
d1_df.loc[d1_df['Close'] < d1_df['D1_EMA50'], 'D1_trend'] = 'DOWN'

print(f"  D1 bars: {len(d1_df):,}")
trend_counts = d1_df['D1_trend'].value_counts()
for t, c in trend_counts.items():
    print(f"    {t}: {c} days ({100*c/len(d1_df):.1f}%)")


def get_d1_trend(entry_time, ema_span=50):
    """Get D1 trend at the time of entry (using previous day's close)."""
    ts = pd.Timestamp(entry_time, tz='UTC')
    prev_days = d1_df.loc[:ts]
    if len(prev_days) < 2:
        return 'NEUTRAL'
    row = prev_days.iloc[-2]  # Use previous completed day
    close = row['Close']
    ema = row.get(f'D1_EMA{ema_span}', close)
    if pd.isna(ema):
        return 'NEUTRAL'
    if close > ema * 1.001:
        return 'UP'
    elif close < ema * 0.999:
        return 'DOWN'
    return 'NEUTRAL'


def get_d1_trend_strict(entry_time, ema_span=50, adx_min=20):
    """D1 trend with ADX confirmation."""
    ts = pd.Timestamp(entry_time, tz='UTC')
    prev_days = d1_df.loc[:ts]
    if len(prev_days) < 2:
        return 'NEUTRAL'
    row = prev_days.iloc[-2]
    close = row['Close']
    ema = row.get(f'D1_EMA{ema_span}', close)
    adx = row.get('D1_ADX', 0)
    if pd.isna(ema) or pd.isna(adx):
        return 'NEUTRAL'
    if adx < adx_min:
        return 'NEUTRAL'
    if close > ema:
        return 'UP'
    elif close < ema:
        return 'DOWN'
    return 'NEUTRAL'


# ═══════════════════════════════════════════════════════════════
# Part 1: Trade performance by D1 trend
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 1: CURRENT TRADES BREAKDOWN BY D1 TREND")
print("=" * 70)

baseline = run_variant(data, "Current", **CURRENT)
trades_cur = baseline.get('_trades', [])

for ema_span in [20, 50, 100]:
    by_combo = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0})

    for t in trades_cur:
        trend = get_d1_trend(t.entry_time, ema_span)
        aligned = (trend == 'UP' and t.direction == 'BUY') or \
                  (trend == 'DOWN' and t.direction == 'SELL')
        against = (trend == 'UP' and t.direction == 'SELL') or \
                  (trend == 'DOWN' and t.direction == 'BUY')
        if aligned:
            key = 'Aligned'
        elif against:
            key = 'Against'
        else:
            key = 'Neutral'

        by_combo[key]['n'] += 1
        by_combo[key]['pnl'] += t.pnl
        if t.pnl > 0:
            by_combo[key]['wins'] += 1

    print(f"\n  D1 EMA{ema_span}:")
    print(f"  {'Alignment':<12} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
    print(f"  {'-'*45}")
    for key in ['Aligned', 'Against', 'Neutral']:
        d = by_combo[key]
        if d['n'] == 0:
            continue
        wr = 100 * d['wins'] / d['n']
        ppt = d['pnl'] / d['n']
        print(f"  {key:<12} {d['n']:>6} ${d['pnl']:>9,.0f} ${ppt:>6.2f} {wr:>5.1f}%")


# ═══════════════════════════════════════════════════════════════
# Part 2: D1 filter — post-hoc Sharpe comparison
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: D1 FILTER SHARPE COMPARISON (CURRENT)")
print("=" * 70)

print(f"\n  Baseline: N={baseline['n']:,} Sharpe={baseline['sharpe']:.2f} PnL=${baseline['total_pnl']:,.0f}")

filters = [
    ("No filter (baseline)", lambda t: True),
    ("D1 EMA50 aligned only", lambda t: (get_d1_trend(t.entry_time, 50) == 'UP' and t.direction == 'BUY') or
                                         (get_d1_trend(t.entry_time, 50) == 'DOWN' and t.direction == 'SELL') or
                                         get_d1_trend(t.entry_time, 50) == 'NEUTRAL'),
    ("D1 EMA50 strict (block against)", lambda t: not ((get_d1_trend(t.entry_time, 50) == 'UP' and t.direction == 'SELL') or
                                                        (get_d1_trend(t.entry_time, 50) == 'DOWN' and t.direction == 'BUY'))),
    ("D1 EMA20 strict", lambda t: not ((get_d1_trend(t.entry_time, 20) == 'UP' and t.direction == 'SELL') or
                                        (get_d1_trend(t.entry_time, 20) == 'DOWN' and t.direction == 'BUY'))),
    ("D1 EMA100 strict", lambda t: not ((get_d1_trend(t.entry_time, 100) == 'UP' and t.direction == 'SELL') or
                                         (get_d1_trend(t.entry_time, 100) == 'DOWN' and t.direction == 'BUY'))),
    ("D1 EMA50+ADX>20 strict", lambda t: not ((get_d1_trend_strict(t.entry_time, 50, 20) == 'UP' and t.direction == 'SELL') or
                                               (get_d1_trend_strict(t.entry_time, 50, 20) == 'DOWN' and t.direction == 'BUY'))),
    ("D1 EMA50 aligned ONLY (drop neutral)", lambda t: (get_d1_trend(t.entry_time, 50) == 'UP' and t.direction == 'BUY') or
                                                         (get_d1_trend(t.entry_time, 50) == 'DOWN' and t.direction == 'SELL')),
]

print(f"\n  {'Filter':<45} {'N':>6} {'Sharpe':>8} {'Delta':>7} {'PnL':>10} {'$/t':>7}")
print(f"  {'-'*85}")

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
    print(f"  {fname:<45} {stats['n']:>6} {stats['sharpe']:>8.2f} {d:>+7.2f} ${stats['total_pnl']:>9,.0f} ${ppt:>6.2f}")


# ═══════════════════════════════════════════════════════════════
# Part 3: Mega Trail — D1 filter analysis
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: MEGA TRAIL — D1 FILTER ANALYSIS")
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
    print(f"  {fname:<45} {stats['n']:>6} {stats['sharpe']:>8.2f} {d:>+7.2f} ${stats['total_pnl']:>9,.0f} ${ppt:>6.2f}")


# ═══════════════════════════════════════════════════════════════
# Part 4: Yearly stability of D1 filter
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4: YEARLY STABILITY — ALIGNED vs AGAINST (D1 EMA50)")
print("=" * 70)

years = range(2015, 2027)
print(f"\n  {'Year':<6} {'Aligned_N':>10} {'Aligned_$/t':>12} {'Against_N':>10} {'Against_$/t':>12} {'Filter_helps':>13}")
print(f"  {'-'*66}")

for year in years:
    start = f"{year}-01-01"
    end = f"{year+1}-01-01" if year < 2026 else "2026-04-01"
    yr_trades = [t for t in trades_cur if start <= t.entry_time.strftime('%Y-%m-%d') < end]
    if not yr_trades:
        continue

    aligned = [t for t in yr_trades
               if (get_d1_trend(t.entry_time, 50) == 'UP' and t.direction == 'BUY') or
                  (get_d1_trend(t.entry_time, 50) == 'DOWN' and t.direction == 'SELL')]
    against = [t for t in yr_trades
               if (get_d1_trend(t.entry_time, 50) == 'UP' and t.direction == 'SELL') or
                  (get_d1_trend(t.entry_time, 50) == 'DOWN' and t.direction == 'BUY')]

    a_ppt = sum(t.pnl for t in aligned) / len(aligned) if aligned else 0
    ag_ppt = sum(t.pnl for t in against) / len(against) if against else 0
    helps = "YES" if a_ppt > ag_ppt else "NO"

    print(f"  {year:<6} {len(aligned):>10} ${a_ppt:>10.2f} {len(against):>10} ${ag_ppt:>10.2f} {helps:>13}")


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

elapsed = time.time() - t0
print("\n" + "=" * 70)
print("EXP39 SUMMARY")
print("=" * 70)
print(f"  Current: Sharpe={baseline['sharpe']:.2f} PnL=${baseline['total_pnl']:,.0f}")
print(f"  Mega: Sharpe={mega_stats['sharpe']:.2f} PnL=${mega_stats['total_pnl']:,.0f}")
print(f"\n  Runtime: {elapsed/60:.1f} minutes")
print(f"  Finished: {datetime.now()}")
