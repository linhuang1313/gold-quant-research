#!/usr/bin/env python3
"""
EXP38: RSI 背离提前出场
========================
不改变入场, 在持仓过程中检测 RSI 背离作为提前出场信号:
- BUY 持仓: 价格创新高但 RSI 未创新高 → 看跌背离 → 提前平仓
- 用 RSI(9) 替代 RSI(14) (研究建议黄金用更短周期)

通过 post-hoc 分析: 检查最终亏损的交易, 过程中是否出现过 RSI 背离可以挽救.
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
print("EXP38: RSI DIVERGENCE EARLY EXIT ANALYSIS")
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

# Compute RSI(9) if not present
if 'RSI9' not in h1_df.columns:
    delta = h1_df['Close'].diff()
    gain = delta.clip(lower=0).rolling(9).mean()
    loss = (-delta.clip(upper=0)).rolling(9).mean()
    rs = gain / loss
    h1_df['RSI9'] = 100 - (100 / (1 + rs))


def detect_divergence_in_window(h1_window, direction, rsi_col='RSI9', lookback=5):
    """
    Check if there's a bearish (for BUY) or bullish (for SELL) RSI divergence
    in the last `lookback` bars of h1_window.
    Returns: (divergence_detected, divergence_bar_idx, rsi_at_div)
    """
    if h1_window is None or len(h1_window) < lookback + 2:
        return False, -1, 0

    window = h1_window.iloc[-(lookback+2):]
    closes = window['Close'].values
    rsis = window[rsi_col].values

    if any(np.isnan(rsis)):
        return False, -1, 0

    if direction == 'BUY':
        # Bearish divergence: price makes higher high, RSI makes lower high
        for i in range(2, len(closes)):
            if closes[i] > closes[i-2] and rsis[i] < rsis[i-2]:
                return True, i, rsis[i]
    else:
        # Bullish divergence: price makes lower low, RSI makes higher low
        for i in range(2, len(closes)):
            if closes[i] < closes[i-2] and rsis[i] > rsis[i-2]:
                return True, i, rsis[i]

    return False, -1, 0


def analyze_divergence_impact(trades, h1_df, label, rsi_col='RSI9'):
    """For each trade, check if RSI divergence occurred during holding period."""
    results = {'div_found': 0, 'div_winning': 0, 'div_losing': 0,
               'no_div': 0, 'no_div_winning': 0, 'no_div_losing': 0,
               'div_pnl': 0, 'no_div_pnl': 0,
               'salvageable': 0, 'salvage_pnl_saved': 0}

    for t in trades:
        if t.strategy != 'keltner':
            continue

        entry_time = t.entry_time
        exit_time = getattr(t, 'exit_time', None)
        if exit_time is None:
            continue

        # Get H1 bars during holding period
        mask = (h1_df.index >= pd.Timestamp(entry_time, tz='UTC')) & \
               (h1_df.index <= pd.Timestamp(exit_time, tz='UTC'))
        holding_bars = h1_df.loc[mask]

        if len(holding_bars) < 3:
            continue

        # Check for divergence at any point during holding
        div_found = False
        for i in range(3, len(holding_bars)):
            window = holding_bars.iloc[:i+1]
            found, _, _ = detect_divergence_in_window(window, t.direction, rsi_col)
            if found:
                div_found = True
                break

        if div_found:
            results['div_found'] += 1
            results['div_pnl'] += t.pnl
            if t.pnl > 0:
                results['div_winning'] += 1
            else:
                results['div_losing'] += 1
                # If we had exited at divergence point, we'd have saved some loss
                results['salvageable'] += 1
                results['salvage_pnl_saved'] += abs(t.pnl)
        else:
            results['no_div'] += 1
            results['no_div_pnl'] += t.pnl
            if t.pnl > 0:
                results['no_div_winning'] += 1
            else:
                results['no_div_losing'] += 1

    return results


# ═══════════════════════════════════════════════════════════════
# Part 1: RSI divergence frequency during trades
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 1: RSI DIVERGENCE FREQUENCY DURING KELTNER TRADES")
print("=" * 70)

baseline = run_variant(data, "Current", **CURRENT)
trades_cur = baseline.get('_trades', [])
keltner_cur = [t for t in trades_cur if t.strategy == 'keltner']

print(f"\n  Keltner trades: {len(keltner_cur):,}")

for rsi_col, rsi_name in [('RSI9', 'RSI(9)'), ('RSI14', 'RSI(14)')]:
    if rsi_col not in h1_df.columns:
        continue
    results = analyze_divergence_impact(keltner_cur, h1_df, "Current", rsi_col)
    total = results['div_found'] + results['no_div']
    if total == 0:
        continue

    div_pct = 100 * results['div_found'] / total
    div_wr = 100 * results['div_winning'] / results['div_found'] if results['div_found'] > 0 else 0
    no_div_wr = 100 * results['no_div_winning'] / results['no_div'] if results['no_div'] > 0 else 0

    print(f"\n  {rsi_name}:")
    print(f"    Divergence found: {results['div_found']:,} ({div_pct:.1f}%)")
    print(f"      Winning: {results['div_winning']:,}, Losing: {results['div_losing']:,}, WR: {div_wr:.1f}%")
    print(f"      Total PnL: ${results['div_pnl']:,.0f}")
    print(f"    No divergence: {results['no_div']:,}")
    print(f"      Winning: {results['no_div_winning']:,}, Losing: {results['no_div_losing']:,}, WR: {no_div_wr:.1f}%")
    print(f"      Total PnL: ${results['no_div_pnl']:,.0f}")
    print(f"    Salvageable losing trades: {results['salvageable']:,}")
    div_avg = results['div_pnl'] / results['div_found'] if results['div_found'] > 0 else 0
    no_div_avg = results['no_div_pnl'] / results['no_div'] if results['no_div'] > 0 else 0
    print(f"    $/trade: Div={div_avg:.2f} vs NoDIV={no_div_avg:.2f}")


# ═══════════════════════════════════════════════════════════════
# Part 2: By exit reason — where does divergence predict loss?
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: DIVERGENCE x EXIT REASON")
print("=" * 70)

by_reason_div = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0})
by_reason_nodiv = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0})

for t in keltner_cur:
    entry_time = t.entry_time
    exit_time = getattr(t, 'exit_time', None)
    if exit_time is None:
        continue

    mask = (h1_df.index >= pd.Timestamp(entry_time, tz='UTC')) & \
           (h1_df.index <= pd.Timestamp(exit_time, tz='UTC'))
    holding_bars = h1_df.loc[mask]

    if len(holding_bars) < 3:
        continue

    div_found = False
    for i in range(3, len(holding_bars)):
        found, _, _ = detect_divergence_in_window(holding_bars.iloc[:i+1], t.direction, 'RSI9')
        if found:
            div_found = True
            break

    reason = t.exit_reason.split(':')[0] if ':' in t.exit_reason else t.exit_reason
    target = by_reason_div if div_found else by_reason_nodiv
    target[reason]['n'] += 1
    target[reason]['pnl'] += t.pnl
    if t.pnl > 0:
        target[reason]['wins'] += 1

print(f"\n  {'Reason':<15} {'DIV_N':>6} {'DIV_$/t':>8} {'DIV_WR':>7} | {'NoDIV_N':>8} {'NoDIV_$/t':>9} {'NoDIV_WR':>8}")
print(f"  {'-'*75}")
all_reasons = set(list(by_reason_div.keys()) + list(by_reason_nodiv.keys()))
for reason in sorted(all_reasons):
    dd = by_reason_div[reason]
    nd = by_reason_nodiv[reason]
    d_ppt = dd['pnl'] / dd['n'] if dd['n'] > 0 else 0
    n_ppt = nd['pnl'] / nd['n'] if nd['n'] > 0 else 0
    d_wr = 100 * dd['wins'] / dd['n'] if dd['n'] > 0 else 0
    n_wr = 100 * nd['wins'] / nd['n'] if nd['n'] > 0 else 0
    print(f"  {reason:<15} {dd['n']:>6} ${d_ppt:>6.2f} {d_wr:>6.1f}% | {nd['n']:>8} ${n_ppt:>7.2f} {n_wr:>7.1f}%")


# ═══════════════════════════════════════════════════════════════
# Part 3: Simulate early exit on divergence
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: SIMULATED EARLY EXIT ON RSI DIVERGENCE")
print("=" * 70)

# Simple simulation: if divergence detected, use breakeven exit
# vs original exit. This is optimistic (assumes perfect detection).
print(f"\n  Strategy: On RSI(9) divergence, exit at current close instead of waiting")

for label, trades in [("Current", keltner_cur)]:
    original_pnls = []
    modified_pnls = []

    for t in trades:
        original_pnls.append(t.pnl)

        entry_time = t.entry_time
        exit_time = getattr(t, 'exit_time', None)
        if exit_time is None:
            modified_pnls.append(t.pnl)
            continue

        mask = (h1_df.index >= pd.Timestamp(entry_time, tz='UTC')) & \
               (h1_df.index <= pd.Timestamp(exit_time, tz='UTC'))
        holding_bars = h1_df.loc[mask]

        if len(holding_bars) < 3:
            modified_pnls.append(t.pnl)
            continue

        # Find first divergence
        div_bar_close = None
        for i in range(3, len(holding_bars)):
            found, _, _ = detect_divergence_in_window(
                holding_bars.iloc[:i+1], t.direction, 'RSI9')
            if found:
                div_bar_close = float(holding_bars.iloc[i]['Close'])
                break

        if div_bar_close is not None:
            if t.direction == 'BUY':
                alt_pnl = (div_bar_close - t.entry_price) * t.lots * 100
            else:
                alt_pnl = (t.entry_price - div_bar_close) * t.lots * 100
            # Only use early exit if it's better than original
            # (no look-ahead: we always exit on divergence)
            modified_pnls.append(alt_pnl)
        else:
            modified_pnls.append(t.pnl)

    orig_total = sum(original_pnls)
    mod_total = sum(modified_pnls)
    orig_sharpe = compute_sharpe(trades, original_pnls)
    mod_sharpe = compute_sharpe(trades, modified_pnls)
    n_better = sum(1 for o, m in zip(original_pnls, modified_pnls) if m > o)
    n_worse = sum(1 for o, m in zip(original_pnls, modified_pnls) if m < o)

    print(f"\n  {label}:")
    print(f"    Original:  PnL=${orig_total:,.0f} Sharpe={orig_sharpe:.2f}")
    print(f"    Div-exit:  PnL=${mod_total:,.0f} Sharpe={mod_sharpe:.2f}")
    print(f"    Diff: ${mod_total - orig_total:+,.0f} Sharpe {mod_sharpe - orig_sharpe:+.2f}")
    print(f"    Trades improved: {n_better:,}, worsened: {n_worse:,}")


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
# Part 4: Mega Trail divergence analysis
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4: MEGA TRAIL RSI DIVERGENCE ANALYSIS")
print("=" * 70)

mega_stats = run_variant(data, "Mega", **MEGA)
trades_mega = mega_stats.get('_trades', [])
keltner_mega = [t for t in trades_mega if t.strategy == 'keltner']

print(f"\n  Mega Keltner trades: {len(keltner_mega):,}")

results = analyze_divergence_impact(keltner_mega, h1_df, "Mega", 'RSI9')
total = results['div_found'] + results['no_div']
if total > 0:
    div_pct = 100 * results['div_found'] / total
    div_wr = 100 * results['div_winning'] / results['div_found'] if results['div_found'] > 0 else 0
    no_div_wr = 100 * results['no_div_winning'] / results['no_div'] if results['no_div'] > 0 else 0
    div_avg = results['div_pnl'] / results['div_found'] if results['div_found'] > 0 else 0
    no_div_avg = results['no_div_pnl'] / results['no_div'] if results['no_div'] > 0 else 0

    print(f"  RSI(9):")
    print(f"    Divergence: {results['div_found']:,} ({div_pct:.1f}%) | WR={div_wr:.1f}% | $/t=${div_avg:.2f}")
    print(f"    No divergence: {results['no_div']:,} | WR={no_div_wr:.1f}% | $/t=${no_div_avg:.2f}")
    print(f"    Key: Div $/t vs NoDIV $/t diff = ${div_avg - no_div_avg:+.2f}")


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

elapsed = time.time() - t0
print("\n" + "=" * 70)
print("EXP38 SUMMARY")
print("=" * 70)
print(f"\n  Runtime: {elapsed/60:.1f} minutes")
print(f"  Finished: {datetime.now()}")
