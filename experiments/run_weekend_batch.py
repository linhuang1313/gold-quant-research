#!/usr/bin/env python3
"""
Weekend Batch Experiments — 6 experiments in parallel
=====================================================
All based on L3 (LIVE_PARITY_KWARGS).
Uses multiprocessing for parallelism on 128-core server.
"""
import sys, os, time, multiprocessing as mp
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_FILE = "exp_weekend_batch_output.txt"

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-01"),
]


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def _load_base():
    from backtest.runner import LIVE_PARITY_KWARGS
    return {**LIVE_PARITY_KWARGS}


def _run_one(args):
    """Generic: run one variant, return tuple."""
    label, extra_kwargs, spread, start, end = args
    from backtest import DataBundle, run_variant
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    if start and end:
        data = data.slice(start, end)
    BASE = _load_base()
    kwargs = {**BASE, **extra_kwargs}
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kwargs)
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    return (label, n, s['sharpe'], s['total_pnl'], s['win_rate'], avg, s['max_dd'])


# ═══════════════════════════════════════════════════════════════
# EXP 1: TDTP ON/OFF
# ═══════════════════════════════════════════════════════════════

def exp1_tdtp(p):
    p("\n" + "=" * 80)
    p("EXP-1: TDTP ON vs OFF (L3 context)")
    p("=" * 80)

    tasks = [
        ("TDTP_ON", {"time_decay_tp": True}, 0.30, None, None),
        ("TDTP_OFF", {"time_decay_tp": False}, 0.30, None, None),
    ]
    with mp.Pool(2) as pool:
        results = pool.map(_run_one, tasks)

    p(f"\n{'Variant':<15s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'$/t':>8s}  {'MaxDD':>11s}")
    p("-" * 72)
    res_map = {}
    for label, n, sharpe, pnl, wr, avg, maxdd in sorted(results, key=lambda x: x[0]):
        res_map[label] = sharpe
        p(f"  {label:<13s}  {n:>6d}  {sharpe:>7.2f}  {fmt(pnl)}  {wr:>5.1f}%  ${avg:>7.2f}  {fmt(maxdd)}")

    delta = res_map.get('TDTP_OFF', 0) - res_map.get('TDTP_ON', 0)
    p(f"\n  Delta (OFF - ON): {delta:>+.2f}")

    # K-Fold
    p(f"\n--- K-Fold: TDTP ON vs OFF ---")
    fold_tasks = []
    for fname, start, end in FOLDS:
        fold_tasks.append((f"KF_ON_{fname}", {"time_decay_tp": True}, 0.30, start, end))
        fold_tasks.append((f"KF_OFF_{fname}", {"time_decay_tp": False}, 0.30, start, end))

    with mp.Pool(12) as pool:
        fold_results = pool.map(_run_one, fold_tasks)

    fold_map = {r[0]: r for r in fold_results}
    wins = 0
    deltas = []
    for fname, _, _ in FOLDS:
        on_s = fold_map[f"KF_ON_{fname}"][2]
        off_s = fold_map[f"KF_OFF_{fname}"][2]
        d = off_s - on_s
        won = d > 0
        if won:
            wins += 1
        deltas.append(d)
        p(f"    {fname}: ON={on_s:>6.2f}  OFF={off_s:>6.2f}  delta={d:>+.2f} {'V' if won else 'X'}")
    avg_d = sum(deltas) / len(deltas) if deltas else 0
    p(f"    Result: {wins}/6 {'PASS' if wins >= 5 else 'FAIL'}  avg_delta={avg_d:>+.3f}")

    # Stress $0.50
    p(f"\n--- Stress test @ $0.50 ---")
    stress_tasks = [
        ("TDTP_ON_sp50", {"time_decay_tp": True}, 0.50, None, None),
        ("TDTP_OFF_sp50", {"time_decay_tp": False}, 0.50, None, None),
    ]
    with mp.Pool(2) as pool:
        stress = pool.map(_run_one, stress_tasks)
    for label, n, sharpe, pnl, wr, avg, maxdd in stress:
        p(f"  {label}: Sharpe={sharpe:.2f}  PnL={fmt(pnl)}  N={n}")


# ═══════════════════════════════════════════════════════════════
# EXP 2: Historical Spread
# ═══════════════════════════════════════════════════════════════

def _run_hist_spread(args):
    label, extra_kwargs, spread_cost, spread_model = args
    from backtest import DataBundle, run_variant
    from backtest.runner import load_spread_series
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    BASE = _load_base()
    kwargs = {**BASE, **extra_kwargs}
    if spread_model == "historical":
        ss = load_spread_series()
        s = run_variant(data, label, verbose=False, spread_model="historical",
                        spread_series=ss, **kwargs)
    else:
        s = run_variant(data, label, verbose=False, spread_cost=spread_cost, **kwargs)
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    return (label, n, s['sharpe'], s['total_pnl'], s['win_rate'], avg, s['max_dd'])


def exp2_historical_spread(p):
    p("\n" + "=" * 80)
    p("EXP-2: L3 with Historical Spread")
    p("=" * 80)

    tasks = [
        ("Fixed_$0.30", {}, 0.30, "fixed"),
        ("Fixed_$0.50", {}, 0.50, "fixed"),
        ("Historical", {}, 0, "historical"),
    ]
    with mp.Pool(3) as pool:
        results = pool.map(_run_hist_spread, tasks)

    p(f"\n{'Variant':<15s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'$/t':>8s}  {'MaxDD':>11s}")
    p("-" * 72)
    for label, n, sharpe, pnl, wr, avg, maxdd in results:
        p(f"  {label:<13s}  {n:>6d}  {sharpe:>7.2f}  {fmt(pnl)}  {wr:>5.1f}%  ${avg:>7.2f}  {fmt(maxdd)}")


# ═══════════════════════════════════════════════════════════════
# EXP 3: Breakout Strength Sizing (post-hoc)
# ═══════════════════════════════════════════════════════════════

def exp3_breakout_sizing(p):
    p("\n" + "=" * 80)
    p("EXP-3: Breakout Strength Sizing (post-hoc simulation, L3 context)")
    p("=" * 80)

    import numpy as np
    from backtest import DataBundle, run_variant
    from backtest.engine import BacktestEngine

    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    BASE = _load_base()
    s = run_variant(data, "sizing_base", verbose=False, spread_cost=0.30, **BASE)

    trades = s.get('trades', [])
    if not trades:
        p("  No trade records available for post-hoc analysis")
        return

    h1_df = data.h1_df

    def kc_strength(trade):
        from run_exp_v_breakout_sizing import keltner_breakout_strength
        return keltner_breakout_strength(trade, h1_df)

    strengths = [kc_strength(t) for t in trades]
    kc_trades = [(t, s) for t, s in zip(trades, strengths)
                 if t.strategy == 'keltner' and np.isfinite(s)]

    if not kc_trades:
        p("  No Keltner trades with valid strength data")
        return

    import research_config as cfg
    capital = float(cfg.CAPITAL)

    p(f"\n  Keltner trades with strength data: {len(kc_trades)}")
    p(f"\n  Factors tested: 0.3, 0.5, 0.7, 1.0")
    p(f"\n{'Factor':>8s}  {'PnL':>11s}  {'$/trade':>9s}  {'Sharpe_approx':>14s}")
    p("-" * 50)

    for factor in [0.0, 0.3, 0.5, 0.7, 1.0]:
        adjusted_pnls = []
        for t, st in kc_trades:
            mult = 1.0 + st * factor
            mult = max(0.5, min(2.0, mult))
            adjusted_pnls.append(t.pnl * mult)
        total = sum(adjusted_pnls)
        avg = total / len(adjusted_pnls)
        std = np.std(adjusted_pnls, ddof=1) if len(adjusted_pnls) > 1 else 1
        sharpe_approx = (np.mean(adjusted_pnls) / std * np.sqrt(252 * 6)) if std > 0 else 0
        marker = " <-- baseline" if factor == 0.0 else ""
        p(f"  {factor:>6.1f}  {fmt(total)}  ${avg:>8.2f}  {sharpe_approx:>13.2f}{marker}")


# ═══════════════════════════════════════════════════════════════
# EXP 4: SL sweep in L3 context
# ═══════════════════════════════════════════════════════════════

def exp4_sl_sweep(p):
    p("\n" + "=" * 80)
    p("EXP-4: SL ATR Multiplier Sweep (L3 context)")
    p("=" * 80)

    SL_VALUES = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    tasks = [(f"SL_{sl}", {"sl_atr_mult": sl}, 0.30, None, None) for sl in SL_VALUES]

    with mp.Pool(len(tasks)) as pool:
        results = pool.map(_run_one, tasks)

    p(f"\n{'SL_mult':>8s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'$/t':>8s}  {'MaxDD':>11s}")
    p("-" * 72)
    res_map = {}
    for label, n, sharpe, pnl, wr, avg, maxdd in sorted(results, key=lambda x: x[0]):
        sl = float(label.split('_')[1])
        marker = " <-- L3" if sl == 4.5 else ""
        res_map[sl] = sharpe
        p(f"  {sl:>6.1f}  {n:>6d}  {sharpe:>7.2f}  {fmt(pnl)}  {wr:>5.1f}%  ${avg:>7.2f}  {fmt(maxdd)}{marker}")

    # K-Fold for top-2 (excl current 4.5)
    ranked = sorted(res_map.items(), key=lambda x: -x[1])
    top2 = [sl for sl, _ in ranked[:2] if sl != 4.5][:2]
    if not top2:
        top2 = [ranked[0][0]]

    for sl in top2:
        p(f"\n--- K-Fold: SL={sl} vs SL=4.5 ---")
        fold_tasks = []
        for fname, start, end in FOLDS:
            fold_tasks.append((f"SL45_{fname}", {"sl_atr_mult": 4.5}, 0.30, start, end))
            fold_tasks.append((f"SL{sl}_{fname}", {"sl_atr_mult": sl}, 0.30, start, end))
        with mp.Pool(12) as pool:
            fold_res = pool.map(_run_one, fold_tasks)
        fmap = {r[0]: r for r in fold_res}
        wins = 0
        for fname, _, _ in FOLDS:
            base_s = fmap[f"SL45_{fname}"][2]
            test_s = fmap[f"SL{sl}_{fname}"][2]
            d = test_s - base_s
            won = d > 0
            if won:
                wins += 1
            p(f"    {fname}: SL4.5={base_s:>6.2f}  SL{sl}={test_s:>6.2f}  delta={d:>+.2f} {'V' if won else 'X'}")
        p(f"    Result: {wins}/6 {'PASS' if wins >= 5 else 'FAIL'}")


# ═══════════════════════════════════════════════════════════════
# EXP 5: ADX threshold sweep in L3 context
# ═══════════════════════════════════════════════════════════════

def exp5_adx_sweep(p):
    p("\n" + "=" * 80)
    p("EXP-5: ADX Threshold Sweep (L3 context)")
    p("=" * 80)

    ADX_VALUES = [14, 15, 16, 17, 18, 19, 20, 22, 25]
    tasks = [(f"ADX_{adx}", {"keltner_adx_threshold": adx}, 0.30, None, None) for adx in ADX_VALUES]

    with mp.Pool(len(tasks)) as pool:
        results = pool.map(_run_one, tasks)

    p(f"\n{'ADX':>5s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'$/t':>8s}  {'MaxDD':>11s}")
    p("-" * 72)
    res_map = {}
    for label, n, sharpe, pnl, wr, avg, maxdd in sorted(results, key=lambda x: int(x[0].split('_')[1])):
        adx = int(label.split('_')[1])
        marker = " <-- L3" if adx == 18 else ""
        res_map[adx] = {'sharpe': sharpe, 'n': n}
        p(f"  {adx:>3d}  {n:>6d}  {sharpe:>7.2f}  {fmt(pnl)}  {wr:>5.1f}%  ${avg:>7.2f}  {fmt(maxdd)}{marker}")

    # Trade count impact
    base_n = res_map[18]['n']
    p(f"\n  Trade count vs ADX=18 ({base_n}):")
    for adx in ADX_VALUES:
        if adx == 18:
            continue
        diff = res_map[adx]['n'] - base_n
        p(f"    ADX={adx}: {res_map[adx]['n']} ({diff:+d} trades, {diff/base_n*100:+.1f}%)")

    # K-Fold for top-2
    ranked = sorted(res_map.items(), key=lambda x: -x[1]['sharpe'])
    top2 = [adx for adx, _ in ranked[:2] if adx != 18][:2]
    if not top2:
        top2 = [ranked[0][0]]

    for adx in top2:
        p(f"\n--- K-Fold: ADX={adx} vs ADX=18 ---")
        fold_tasks = []
        for fname, start, end in FOLDS:
            fold_tasks.append((f"ADX18_{fname}", {"keltner_adx_threshold": 18}, 0.30, start, end))
            fold_tasks.append((f"ADX{adx}_{fname}", {"keltner_adx_threshold": adx}, 0.30, start, end))
        with mp.Pool(12) as pool:
            fold_res = pool.map(_run_one, fold_tasks)
        fmap = {r[0]: r for r in fold_res}
        wins = 0
        for fname, _, _ in FOLDS:
            base_s = fmap[f"ADX18_{fname}"][2]
            test_s = fmap[f"ADX{adx}_{fname}"][2]
            d = test_s - base_s
            won = d > 0
            if won:
                wins += 1
            p(f"    {fname}: ADX18={base_s:>6.2f}  ADX{adx}={test_s:>6.2f}  delta={d:>+.2f} {'V' if won else 'X'}")
        p(f"    Result: {wins}/6 {'PASS' if wins >= 5 else 'FAIL'}")


# ═══════════════════════════════════════════════════════════════
# EXP 6: Recent performance detail
# ═══════════════════════════════════════════════════════════════

def exp6_recent_detail(p):
    p("\n" + "=" * 80)
    p("EXP-6: L3 Recent Performance Detail (2025-2026)")
    p("=" * 80)

    from backtest import DataBundle, run_variant
    import pandas as pd
    import numpy as np

    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    recent = data.slice("2025-01-01", "2026-04-10")
    BASE = _load_base()
    s = run_variant(recent, "L3_recent", verbose=False, spread_cost=0.30, **BASE)

    p(f"\n  Overall: N={s['n']}, Sharpe={s['sharpe']:.2f}, PnL={fmt(s['total_pnl'])}, "
      f"WR={s['win_rate']:.1f}%, MaxDD={fmt(s['max_dd'])}")

    trades = s.get('trades', [])
    if not trades:
        p("  No trade records for detail analysis")
        return

    # Monthly breakdown
    p(f"\n--- Monthly PnL ---")
    p(f"{'Month':<10s}  {'N':>5s}  {'PnL':>10s}  {'WR%':>6s}  {'$/t':>8s}")
    p("-" * 45)

    monthly = {}
    for t in trades:
        month = pd.Timestamp(t.exit_time).strftime('%Y-%m')
        if month not in monthly:
            monthly[month] = {'n': 0, 'wins': 0, 'pnl': 0.0}
        monthly[month]['n'] += 1
        monthly[month]['pnl'] += t.pnl
        if t.pnl > 0:
            monthly[month]['wins'] += 1

    for month in sorted(monthly.keys()):
        m = monthly[month]
        wr = m['wins'] / m['n'] * 100 if m['n'] > 0 else 0
        avg = m['pnl'] / m['n'] if m['n'] > 0 else 0
        p(f"  {month:<8s}  {m['n']:>5d}  {fmt(m['pnl'])}  {wr:>5.1f}%  ${avg:>7.2f}")

    # Exit type distribution
    p(f"\n--- Exit Type Distribution ---")
    p(f"{'Exit_type':<20s}  {'N':>5s}  {'PnL':>10s}  {'WR%':>6s}  {'$/t':>8s}  {'Avg_bars':>9s}")
    p("-" * 65)

    by_exit = {}
    for t in trades:
        reason = t.exit_reason or 'unknown'
        if reason not in by_exit:
            by_exit[reason] = {'n': 0, 'wins': 0, 'pnl': 0.0, 'bars': []}
        by_exit[reason]['n'] += 1
        by_exit[reason]['pnl'] += t.pnl
        if t.pnl > 0:
            by_exit[reason]['wins'] += 1
        by_exit[reason]['bars'].append(t.bars_held)

    for reason in sorted(by_exit.keys(), key=lambda r: -by_exit[r]['n']):
        e = by_exit[reason]
        wr = e['wins'] / e['n'] * 100 if e['n'] > 0 else 0
        avg = e['pnl'] / e['n'] if e['n'] > 0 else 0
        avg_bars = np.mean(e['bars']) if e['bars'] else 0
        p(f"  {reason:<18s}  {e['n']:>5d}  {fmt(e['pnl'])}  {wr:>5.1f}%  ${avg:>7.2f}  {avg_bars:>8.1f}")

    # Strategy breakdown
    p(f"\n--- Strategy Breakdown ---")
    by_strat = {}
    for t in trades:
        strat = t.strategy or 'unknown'
        if strat not in by_strat:
            by_strat[strat] = {'n': 0, 'wins': 0, 'pnl': 0.0}
        by_strat[strat]['n'] += 1
        by_strat[strat]['pnl'] += t.pnl
        if t.pnl > 0:
            by_strat[strat]['wins'] += 1

    for strat, d in sorted(by_strat.items(), key=lambda x: -x[1]['n']):
        wr = d['wins'] / d['n'] * 100 if d['n'] > 0 else 0
        avg = d['pnl'] / d['n'] if d['n'] > 0 else 0
        p(f"  {strat}: N={d['n']}, PnL={fmt(d['pnl'])}, WR={wr:.1f}%, $/t=${avg:.2f}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        def p(msg=""):
            print(msg, flush=True)
            f.write(msg + "\n")
            f.flush()

        p("=" * 80)
        p("WEEKEND BATCH EXPERIMENTS — L3 Context")
        p(f"CPUs: {mp.cpu_count()}")
        p(f"Started: {datetime.now()}")
        p("=" * 80)

        t_total = time.time()

        exp1_tdtp(p)
        p(f"\n  [EXP-1 elapsed: {(time.time()-t_total)/60:.1f} min]")

        t2 = time.time()
        exp2_historical_spread(p)
        p(f"\n  [EXP-2 elapsed: {(time.time()-t2)/60:.1f} min]")

        t3 = time.time()
        exp3_breakout_sizing(p)
        p(f"\n  [EXP-3 elapsed: {(time.time()-t3)/60:.1f} min]")

        t4 = time.time()
        exp4_sl_sweep(p)
        p(f"\n  [EXP-4 elapsed: {(time.time()-t4)/60:.1f} min]")

        t5 = time.time()
        exp5_adx_sweep(p)
        p(f"\n  [EXP-5 elapsed: {(time.time()-t5)/60:.1f} min]")

        t6 = time.time()
        exp6_recent_detail(p)
        p(f"\n  [EXP-6 elapsed: {(time.time()-t6)/60:.1f} min]")

        elapsed = time.time() - t_total
        p(f"\n{'='*80}")
        p(f"ALL EXPERIMENTS COMPLETE")
        p(f"Total runtime: {elapsed/60:.1f} minutes")
        p(f"Completed: {datetime.now()}")
        p(f"{'='*80}")

    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
