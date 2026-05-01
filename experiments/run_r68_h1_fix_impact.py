#!/usr/bin/env python3
"""
R68 — H1 Lookahead Fix Impact Test
====================================
Core question: How much does fixing the H1 exit lookahead bias affect L8_MAX?

The engine's _check_exits() uses the CURRENT H1 bar (which may be unclosed)
for ATR, atr_percentile, trailing decisions. At M15 14:15, the H1[14:00] bar
has not closed yet, but we already see its full OHLC.

Test: Run L8_MAX twice — original vs exit_h1_closed_only=True — and compare.
"""
import sys, os, io, time, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r68_h1_fix")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPREAD = 0.30
BASE_LOT = 0.03

def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def run_l8(m15_df, h1_df, label, exit_h1_closed_only=False):
    from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS
    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': 37, 'spread_cost': SPREAD,
          'initial_capital': 2000, 'min_lot_size': BASE_LOT, 'max_lot_size': BASE_LOT,
          'exit_h1_closed_only': exit_h1_closed_only}
    data = DataBundle(m15_df, h1_df)
    result = run_variant(data, label, verbose=True, **kw)
    raw = result.get('_trades', [])
    trades = []
    for t in raw:
        pnl = t.pnl if hasattr(t, 'pnl') else t.get('pnl', 0)
        ext = t.exit_time if hasattr(t, 'exit_time') else t.get('exit_time', '')
        trades.append({'pnl': pnl, 'exit_time': ext})
    return trades, result


def trades_stats(trades):
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    if not daily:
        return {'sharpe': 0, 'pnl': 0, 'max_dd': 0, 'n_trades': 0, 'n_days': 0, 'win_rate': 0}
    arr = np.array([daily[k] for k in sorted(daily.keys())])
    eq = np.cumsum(arr)
    dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 0 else 0
    std = arr.std()
    sharpe = float(arr.mean() / std * np.sqrt(252)) if std > 0 else 0
    wins = sum(1 for t in trades if t['pnl'] > 0)
    return {
        'sharpe': round(sharpe, 4),
        'pnl': round(float(arr.sum()), 2),
        'max_dd': round(dd, 2),
        'n_trades': len(trades),
        'n_days': len(arr),
        'win_rate': round(wins / len(trades) * 100, 1) if trades else 0,
    }


def main():
    t0 = time.time()
    print("=" * 80)
    print("  R68: H1 Lookahead Fix Impact Test")
    print("=" * 80)

    from backtest.runner import DataBundle
    data = DataBundle.load_default()
    h1_df = data.h1_df.copy(); m15_df = data.m15_df.copy()
    print(f"  H1: {len(h1_df)} bars | M15: {len(m15_df)} bars", flush=True)

    # Run 1: Original (with potential H1 lookahead)
    print("\n  [Run 1] Original (exit uses current H1 bar)...", flush=True)
    trades_orig, _ = run_l8(m15_df, h1_df, "L8_ORIG", exit_h1_closed_only=False)
    stats_orig = trades_stats(trades_orig)
    print(f"    Original: Sharpe={stats_orig['sharpe']:.2f} PnL={fmt(stats_orig['pnl'])} "
          f"Trades={stats_orig['n_trades']} WR={stats_orig['win_rate']:.1f}%", flush=True)

    # Run 2: Fixed (exit uses closed H1 bar only)
    print("\n  [Run 2] Fixed (exit uses previous closed H1 bar)...", flush=True)
    trades_fixed, _ = run_l8(m15_df, h1_df, "L8_FIXED", exit_h1_closed_only=True)
    stats_fixed = trades_stats(trades_fixed)
    print(f"    Fixed:    Sharpe={stats_fixed['sharpe']:.2f} PnL={fmt(stats_fixed['pnl'])} "
          f"Trades={stats_fixed['n_trades']} WR={stats_fixed['win_rate']:.1f}%", flush=True)

    # Compare
    sharpe_diff = stats_orig['sharpe'] - stats_fixed['sharpe']
    sharpe_pct = (sharpe_diff / stats_orig['sharpe'] * 100) if stats_orig['sharpe'] != 0 else 0
    pnl_diff = stats_orig['pnl'] - stats_fixed['pnl']

    elapsed = time.time() - t0

    lines = [
        "R68 H1 Lookahead Fix Impact — Summary",
        "=" * 70,
        f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)\n",
        f"{'Metric':<20} {'Original':>12} {'Fixed':>12} {'Diff':>12} {'Impact%':>10}",
        "-" * 66,
        f"{'Sharpe':<20} {stats_orig['sharpe']:>12.2f} {stats_fixed['sharpe']:>12.2f} "
        f"{sharpe_diff:>12.2f} {sharpe_pct:>9.1f}%",
        f"{'PnL':<20} {fmt(stats_orig['pnl']):>12} {fmt(stats_fixed['pnl']):>12} "
        f"{fmt(pnl_diff):>12}",
        f"{'MaxDD':<20} {fmt(stats_orig['max_dd']):>12} {fmt(stats_fixed['max_dd']):>12}",
        f"{'Trades':<20} {stats_orig['n_trades']:>12} {stats_fixed['n_trades']:>12}",
        f"{'Win Rate':<20} {stats_orig['win_rate']:>11.1f}% {stats_fixed['win_rate']:>11.1f}%",
    ]

    if abs(sharpe_pct) < 5:
        lines.append(f"\nVERDICT: NEGLIGIBLE — H1 lookahead impact < 5% ({sharpe_pct:.1f}%)")
        lines.append("  The bias exists but does not materially affect strategy performance.")
    elif abs(sharpe_pct) < 15:
        lines.append(f"\nVERDICT: MODERATE — H1 lookahead impact {sharpe_pct:.1f}%")
        lines.append("  Consider using exit_h1_closed_only=True for conservative estimates.")
    else:
        lines.append(f"\nVERDICT: SIGNIFICANT — H1 lookahead impact {sharpe_pct:.1f}%")
        lines.append("  The original results are inflated. Use fixed version for production.")

    summary = "\n".join(lines)
    print(f"\n{summary}", flush=True)
    with open(OUTPUT_DIR / "r68_summary.txt", 'w', encoding='utf-8') as f: f.write(summary)
    with open(OUTPUT_DIR / "r68_results.json", 'w', encoding='utf-8') as f:
        json.dump({'original': stats_orig, 'fixed': stats_fixed,
                   'sharpe_diff': round(sharpe_diff, 4),
                   'sharpe_impact_pct': round(sharpe_pct, 1),
                   'pnl_diff': round(pnl_diff, 2),
                   'elapsed_s': round(elapsed, 1)}, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  R68 Complete — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
