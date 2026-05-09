#!/usr/bin/env python3
"""
TSMOM Max Hold Time Sweep
==========================
Test how varying the max_hold parameter affects TSMOM profitability.
Current default: max_hold=20 (H1 bars).
Sweep: 5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 60, 80, 100, no_limit(999)
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH

OUTPUT_DIR = Path("results/tsmom_hold_sweep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01; LOT = 0.09

t0 = time.time()


def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


def _mk(pos, exit_p, exit_t, reason, exit_bar, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_t,
            'pnl': round(pnl, 4), 'reason': reason,
            'bars_held': exit_bar - pos['bar']}


def bt_tsmom(h1_df, spread, lot, max_hold=20, sl_atr=4.5, tp_atr=6.0,
             trail_act=0.14, trail_dist=0.025, maxloss_cap=0):
    fast = 480; slow = 720
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    max_lb = max(fast, slow)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        if c[i-fast] > 0: s += 0.5 * np.sign(c[i]/c[i-fast] - 1.0)
        if c[i-slow] > 0: s += 0.5 * np.sign(c[i]/c[i-slow] - 1.0)
        score[i] = s

    trades = []; pos = None; last_exit = -999
    for i in range(max_lb+1, n):
        if pos is not None:
            d = pos['dir']; entry = pos['entry']; bar0 = pos['bar']; a = pos['atr']
            sl_dist = sl_atr * a; tp_dist = tp_atr * a
            held = i - bar0

            if d == 'BUY':
                sl_price = entry - sl_dist; tp_price = entry + tp_dist
                if lo[i] <= sl_price:
                    pnl = (sl_price - entry - spread) * lot * PV
                    pnl = max(pnl, -maxloss_cap) if maxloss_cap > 0 else pnl
                    trades.append(_mk(pos, sl_price, times[i], 'SL', i, pnl)); pos = None; last_exit = i; continue
                if h[i] >= tp_price:
                    pnl = (tp_price - entry - spread) * lot * PV
                    trades.append(_mk(pos, tp_price, times[i], 'TP', i, pnl)); pos = None; last_exit = i; continue
                profit = h[i] - entry - spread
                if profit >= trail_act * a:
                    trail_stop = h[i] - trail_dist * a
                    if lo[i] <= trail_stop:
                        pnl = (trail_stop - entry - spread) * lot * PV
                        trades.append(_mk(pos, trail_stop, times[i], 'Trail', i, pnl)); pos = None; last_exit = i; continue
                if held >= max_hold:
                    pnl = (c[i] - entry - spread) * lot * PV
                    trades.append(_mk(pos, c[i], times[i], 'Timeout', i, pnl)); pos = None; last_exit = i; continue
                if score[i] < 0:
                    pnl = (c[i] - entry - spread) * lot * PV
                    trades.append(_mk(pos, c[i], times[i], 'Reversal', i, pnl)); pos = None; last_exit = i; continue
            else:
                sl_price = entry + sl_dist; tp_price = entry - tp_dist
                if h[i] >= sl_price:
                    pnl = (entry - sl_price - spread) * lot * PV
                    pnl = max(pnl, -maxloss_cap) if maxloss_cap > 0 else pnl
                    trades.append(_mk(pos, sl_price, times[i], 'SL', i, pnl)); pos = None; last_exit = i; continue
                if lo[i] <= tp_price:
                    pnl = (entry - tp_price - spread) * lot * PV
                    trades.append(_mk(pos, tp_price, times[i], 'TP', i, pnl)); pos = None; last_exit = i; continue
                profit = entry - lo[i] - spread
                if profit >= trail_act * a:
                    trail_stop = lo[i] + trail_dist * a
                    if h[i] >= trail_stop:
                        pnl = (entry - trail_stop - spread) * lot * PV
                        trades.append(_mk(pos, trail_stop, times[i], 'Trail', i, pnl)); pos = None; last_exit = i; continue
                if held >= max_hold:
                    pnl = (entry - c[i] - spread) * lot * PV
                    trades.append(_mk(pos, c[i], times[i], 'Timeout', i, pnl)); pos = None; last_exit = i; continue
                if score[i] > 0:
                    pnl = (entry - c[i] - spread) * lot * PV
                    trades.append(_mk(pos, c[i], times[i], 'Reversal', i, pnl)); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(score[i]) or np.isnan(score[i-1]): continue
        if score[i] > 0 and score[i-1] <= 0:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def trades_to_daily(trades):
    if not trades: return pd.Series(dtype=float)
    df = pd.DataFrame(trades)
    df['date'] = pd.to_datetime(df['exit_time']).dt.date
    return df.groupby('date')['pnl'].sum()


def sharpe(daily):
    if len(daily) < 2: return 0
    m = daily.mean(); s = daily.std()
    return round(m / s * np.sqrt(252), 3) if s > 0 else 0


def max_dd(daily):
    cum = daily.cumsum(); peak = cum.cummax()
    dd = cum - peak
    return round(abs(dd.min()), 2) if len(dd) > 0 else 0


def compute_stats(trades, label=""):
    if not trades:
        return {'label': label, 'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0,
                'max_dd': 0, 'avg_pnl': 0, 'worst_trade': 0}
    pnls = np.array([t['pnl'] for t in trades])
    daily = trades_to_daily(trades)
    bars_held = [t['bars_held'] for t in trades]
    exit_reasons = {}
    for t in trades:
        r = t['reason']
        exit_reasons[r] = exit_reasons.get(r, 0) + 1
    return {
        'label': label,
        'n': len(trades),
        'sharpe': sharpe(daily),
        'pnl': round(float(pnls.sum()), 2),
        'wr': round(100 * np.mean(pnls > 0), 1),
        'max_dd': max_dd(daily),
        'avg_pnl': round(float(pnls.mean()), 4),
        'worst_trade': round(float(pnls.min()), 2),
        'avg_bars_held': round(float(np.mean(bars_held)), 1),
        'median_bars_held': int(np.median(bars_held)),
        'exit_reasons': exit_reasons,
    }


FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-07"),
]


def main():
    print("=" * 80)
    print("  TSMOM Max Hold Time Sweep")
    print("  Current: max_hold=20, SL=4.5xATR, TP=6.0xATR, trail=0.14/0.025")
    print("=" * 80, flush=True)

    print("\n  Loading H1 data...", flush=True)
    m15 = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15.index[0])
    print(f"  H1 range: {h1_df.index[0]} -> {h1_df.index[-1]}", flush=True)

    # Phase 1: Full sample max_hold sweep
    print("\n" + "=" * 80)
    print("  Phase 1: Max Hold Sweep (full sample)")
    print("=" * 80, flush=True)

    hold_values = [5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 60, 80, 100, 999]
    results = {}
    phase1 = {}

    print(f"\n  {'MaxHold':>8} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'MaxDD':>8} {'WR':>6} "
          f"{'AvgPnL':>8} {'AvgBars':>8} {'Timeout%':>9} {'Trail%':>8} {'Reversal%':>10}")
    print("  " + "-" * 110)

    for mh in hold_values:
        label = f"MH={mh}" if mh < 999 else "NoLimit"
        trades = bt_tsmom(h1_df, SPREAD, LOT, max_hold=mh)
        stats = compute_stats(trades, label)
        phase1[str(mh)] = stats
        
        n_total = stats['n'] if stats['n'] > 0 else 1
        timeout_pct = round(100 * stats.get('exit_reasons', {}).get('Timeout', 0) / n_total, 1)
        trail_pct = round(100 * stats.get('exit_reasons', {}).get('Trail', 0) / n_total, 1)
        rev_pct = round(100 * stats.get('exit_reasons', {}).get('Reversal', 0) / n_total, 1)
        
        print(f"  {label:>8} {stats['n']:>6} {stats['sharpe']:>8.3f} {stats['pnl']:>12.2f} "
              f"{stats['max_dd']:>8.2f} {stats['wr']:>5.1f}% {stats['avg_pnl']:>8.4f} "
              f"{stats.get('avg_bars_held', 0):>8.1f} {timeout_pct:>8.1f}% {trail_pct:>7.1f}% "
              f"{rev_pct:>9.1f}%", flush=True)

    results['phase1_full_sample'] = phase1

    # Phase 2: Analyze holding time distribution for baseline (MH=20)
    print("\n" + "=" * 80)
    print("  Phase 2: Holding Time Distribution (MH=20)")
    print("=" * 80, flush=True)

    trades_20 = bt_tsmom(h1_df, SPREAD, LOT, max_hold=20)
    bars_held = [t['bars_held'] for t in trades_20]
    pnls_by_hold = {}
    for t in trades_20:
        bh = t['bars_held']
        if bh not in pnls_by_hold:
            pnls_by_hold[bh] = []
        pnls_by_hold[bh].append(t['pnl'])

    print(f"\n  {'Bars':>5} {'Count':>6} {'AvgPnL':>10} {'WR':>6} {'TotalPnL':>12}")
    print("  " + "-" * 50)
    for bh in sorted(pnls_by_hold.keys()):
        pnls = pnls_by_hold[bh]
        avg = np.mean(pnls)
        wr = 100 * np.mean(np.array(pnls) > 0)
        total = sum(pnls)
        print(f"  {bh:>5} {len(pnls):>6} {avg:>10.4f} {wr:>5.1f}% {total:>12.2f}", flush=True)

    # Grouped analysis
    print(f"\n  Grouped:")
    groups = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 50), (51, 100), (101, 999)]
    phase2_groups = {}
    for lo_b, hi_b in groups:
        g_trades = [t for t in trades_20 if lo_b <= t['bars_held'] <= hi_b]
        if not g_trades: continue
        g_pnls = np.array([t['pnl'] for t in g_trades])
        grp_label = f"{lo_b}-{hi_b}H1"
        phase2_groups[grp_label] = {
            'n': len(g_trades), 'avg_pnl': round(float(g_pnls.mean()), 4),
            'wr': round(100 * np.mean(g_pnls > 0), 1),
            'total_pnl': round(float(g_pnls.sum()), 2),
        }
        print(f"  {grp_label:>12}: N={len(g_trades):>4}, AvgPnL={g_pnls.mean():>8.4f}, "
              f"WR={100*np.mean(g_pnls>0):>5.1f}%, Total=${g_pnls.sum():>10.2f}", flush=True)

    results['phase2_distribution'] = phase2_groups

    # Phase 3: PnL by exit reason
    print("\n" + "=" * 80)
    print("  Phase 3: PnL by Exit Reason (MH=20)")
    print("=" * 80, flush=True)

    reason_groups = {}
    for t in trades_20:
        r = t['reason']
        if r not in reason_groups: reason_groups[r] = []
        reason_groups[r].append(t)

    phase3 = {}
    print(f"\n  {'Reason':>10} {'N':>6} {'AvgPnL':>10} {'WR':>6} {'TotalPnL':>12} {'AvgBars':>8}")
    print("  " + "-" * 60)
    for reason, trades_r in sorted(reason_groups.items()):
        pnls_r = np.array([t['pnl'] for t in trades_r])
        bars_r = [t['bars_held'] for t in trades_r]
        phase3[reason] = {
            'n': len(trades_r), 'avg_pnl': round(float(pnls_r.mean()), 4),
            'wr': round(100 * np.mean(pnls_r > 0), 1),
            'total_pnl': round(float(pnls_r.sum()), 2),
            'avg_bars': round(float(np.mean(bars_r)), 1),
        }
        print(f"  {reason:>10} {len(trades_r):>6} {pnls_r.mean():>10.4f} "
              f"{100*np.mean(pnls_r>0):>5.1f}% {pnls_r.sum():>12.2f} {np.mean(bars_r):>8.1f}",
              flush=True)

    results['phase3_by_reason'] = phase3

    # Phase 4: K-Fold on top 3 candidates
    print("\n" + "=" * 80)
    print("  Phase 4: K-Fold Validation")
    print("=" * 80, flush=True)

    # Find best 3 from phase1
    sorted_by_sharpe = sorted(phase1.items(), key=lambda x: x[1].get('sharpe', 0), reverse=True)
    top_candidates = [int(k) for k, v in sorted_by_sharpe[:5] if v.get('n', 0) > 0]
    if 20 not in top_candidates:
        top_candidates.append(20)  # always include baseline

    print(f"  Testing: {top_candidates}", flush=True)

    kfold_results = {}
    for mh in top_candidates:
        label = f"MH={mh}" if mh < 999 else "NoLimit"
        fold_sharpes = []
        for fname, start, end in FOLDS:
            h1_fold = h1_df[(h1_df.index >= pd.Timestamp(start, tz='UTC')) &
                            (h1_df.index < pd.Timestamp(end, tz='UTC'))]
            if len(h1_fold) < 800:
                fold_sharpes.append(0)
                continue
            trades_fold = bt_tsmom(h1_fold, SPREAD, LOT, max_hold=mh)
            pnls_f = np.array([t['pnl'] for t in trades_fold]) if trades_fold else np.array([0])
            daily_f = trades_to_daily(trades_fold)
            fold_sharpes.append(sharpe(daily_f))
        kfold_results[str(mh)] = {
            'fold_sharpes': fold_sharpes,
            'mean_sharpe': round(float(np.mean(fold_sharpes)), 3),
            'positive_folds': sum(1 for s in fold_sharpes if s > 0),
        }
        print(f"  {label:>8}: folds={[round(s,2) for s in fold_sharpes]}, "
              f"mean={np.mean(fold_sharpes):.3f}, positive={sum(1 for s in fold_sharpes if s > 0)}/6",
              flush=True)

    results['phase4_kfold'] = kfold_results

    # Phase 5: Also test SL/TP sensitivity with best max_hold
    print("\n" + "=" * 80)
    print("  Phase 5: SL/TP Sensitivity (with best max_hold)")
    print("=" * 80, flush=True)

    best_mh = int(sorted_by_sharpe[0][0])
    print(f"  Using max_hold={best_mh}", flush=True)

    sl_tp_grid = [
        (3.0, 4.0), (3.5, 5.0), (4.0, 5.5), (4.5, 6.0),
        (5.0, 6.5), (5.5, 7.0), (6.0, 8.0), (3.5, 8.0),
    ]
    phase5 = {}
    print(f"\n  {'SL':>5} {'TP':>5} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>6} {'AvgBars':>8}")
    print("  " + "-" * 60)
    for sl, tp in sl_tp_grid:
        trades_st = bt_tsmom(h1_df, SPREAD, LOT, max_hold=best_mh, sl_atr=sl, tp_atr=tp)
        stats_st = compute_stats(trades_st, f"SL{sl}_TP{tp}")
        phase5[f"sl{sl}_tp{tp}"] = stats_st
        print(f"  {sl:>5.1f} {tp:>5.1f} {stats_st['n']:>6} {stats_st['sharpe']:>8.3f} "
              f"{stats_st['pnl']:>12.2f} {stats_st['wr']:>5.1f}% "
              f"{stats_st.get('avg_bars_held', 0):>8.1f}", flush=True)

    results['phase5_sl_tp'] = phase5

    # Summary
    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)
    results['best_max_hold'] = best_mh
    results['baseline_max_hold'] = 20

    print(f"\n{'='*80}")
    print(f"  TSMOM Hold Sweep Complete - {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Best max_hold: {best_mh} (Sharpe {sorted_by_sharpe[0][1]['sharpe']:.3f})")
    print(f"  Baseline MH=20: Sharpe {phase1['20']['sharpe']:.3f}")
    print(f"{'='*80}", flush=True)

    with open(OUTPUT_DIR / "tsmom_hold_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/tsmom_hold_results.json", flush=True)


if __name__ == "__main__":
    main()
