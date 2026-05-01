#!/usr/bin/env python3
"""
R59 — Session-Based Lot Weighting (Post-Hoc Analysis)
=====================================================
Instead of filtering by session (already rejected), adjust lot size by
session quality via post-hoc PnL re-weighting.

Phase 1: Run ONE full backtest, tag trades by entry-hour session, report
         per-session stats (N, WR, Sharpe, AvgPnL).
Phase 2: Grid-search 48 session-multiplier combos, rank by Sharpe.
Phase 3: K-Fold 6-Fold validation for top 5 weightings.

Output → results/r59_session_weight/
"""
import sys, os, io, time, json, traceback
import multiprocessing as mp
import numpy as np
from datetime import datetime
from collections import defaultdict
from itertools import product

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "results/r59_session_weight"
MAX_WORKERS = 6
FIXED_LOT = 0.05
SPREAD = 0.30

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]

SESSION_DEFS = {
    "Asia":     (0, 7),
    "London":   (8, 11),
    "NY":       (12, 16),
    "OffHours": (17, 23),
}

ASIA_MULTS    = [0.3, 0.5, 0.7, 1.0]
LONDON_MULTS  = [0.7, 0.85, 1.0]
NY_MULTS      = [1.0]
OFFHOURS_MULTS = [0.3, 0.5, 0.7, 1.0]


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def get_base():
    from backtest.runner import LIVE_PARITY_KWARGS
    return {**LIVE_PARITY_KWARGS, 'min_lot_size': FIXED_LOT, 'max_lot_size': FIXED_LOT,
            'maxloss_cap': 37}


def _classify_session(entry_time_str):
    hour = int(entry_time_str[11:13])
    for name, (h_start, h_end) in SESSION_DEFS.items():
        if h_start <= hour <= h_end:
            return name
    return "OffHours"


def _run_one_trades(args):
    label, kw, spread, start, end = args
    from backtest.runner import DataBundle, run_variant
    data = DataBundle.load_custom()
    if start and end:
        data = data.slice(start, end)
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kw)
    trades = s.get('_trades', [])
    trade_list = []
    for t in trades:
        trade_list.append({
            'entry_time': str(t.entry_time)[:19],
            'direction': t.direction,
            'pnl': round(t.pnl, 2),
            'exit_reason': t.exit_reason or '',
            'bars_held': t.bars_held,
            'lots': t.lots,
        })
    return {'label': label, 'trades': trade_list, 'n': s['n'], 'sharpe': s['sharpe'],
            'total_pnl': s['total_pnl'], 'win_rate': s['win_rate'], 'max_dd': s['max_dd']}


def _run_one(args):
    label, kw, spread, start, end = args
    from backtest.runner import DataBundle, run_variant
    data = DataBundle.load_custom()
    if start and end:
        data = data.slice(start, end)
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kw)
    return {
        'label': label, 'n': s['n'], 'sharpe': s['sharpe'],
        'total_pnl': s['total_pnl'], 'win_rate': s['win_rate'],
        'max_dd': s['max_dd'],
    }


def run_pool(tasks, func=_run_one):
    n_workers = min(MAX_WORKERS, len(tasks))
    with mp.Pool(n_workers) as pool:
        return pool.map(func, tasks)


# ═══════════════════════════════════════════════════════════════
# Helpers: post-hoc PnL re-weighting
# ═══════════════════════════════════════════════════════════════

def _apply_session_weights(trades, weights):
    """Return list of adjusted PnLs given per-session multipliers."""
    adjusted = []
    for t in trades:
        sess = _classify_session(t['entry_time'])
        mult = weights.get(sess, 1.0)
        adjusted.append(t['pnl'] * mult)
    return adjusted


def _daily_pnl_from_trades(trades, pnl_list):
    """Aggregate trade PnLs by entry date → {date_str: sum_pnl}."""
    daily = defaultdict(float)
    for t, pnl in zip(trades, pnl_list):
        d = t['entry_time'][:10]
        daily[d] += pnl
    return daily


def _sharpe_from_daily(daily_dict):
    if not daily_dict:
        return 0.0
    vals = np.array(list(daily_dict.values()))
    if len(vals) < 2 or vals.std() == 0:
        return 0.0
    return float(vals.mean() / vals.std() * np.sqrt(252))


def _max_dd_from_pnls(pnl_list):
    if not pnl_list:
        return 0.0
    eq = np.cumsum(pnl_list) + 2000
    peak = np.maximum.accumulate(eq)
    return float((peak - eq).max())


# ═══════════════════════════════════════════════════════════════
# Phase 1: Full backtest + per-session analysis
# ═══════════════════════════════════════════════════════════════

def run_phase1(out):
    print("\n" + "=" * 70)
    print("Phase 1: Full Backtest + Session Analysis")
    print("=" * 70)

    base = get_base()
    task = ("R59_Baseline", base, SPREAD, None, None)
    results = run_pool([task], func=_run_one_trades)
    r = results[0]
    trades = r['trades']

    print(f"  Baseline: N={r['n']}  Sharpe={r['sharpe']:.2f}  "
          f"PnL={fmt(r['total_pnl'])}  WR={r['win_rate']:.1f}%")

    sess_stats = {}
    for name in SESSION_DEFS:
        st = [t for t in trades if _classify_session(t['entry_time']) == name]
        wins = [t for t in st if t['pnl'] > 0]
        pnls = [t['pnl'] for t in st]
        daily = _daily_pnl_from_trades(st, pnls)
        sharpe = _sharpe_from_daily(daily)
        sess_stats[name] = {
            'n': len(st), 'win_rate': (len(wins) / len(st) * 100) if st else 0,
            'total_pnl': sum(pnls), 'avg_pnl': np.mean(pnls) if pnls else 0,
            'sharpe': sharpe,
        }

    with open(f"{out}/session_analysis.txt", 'w', encoding='utf-8') as f:
        f.write("R59 Phase 1: Per-Session Trade Analysis\n")
        f.write(f"Baseline: N={r['n']}  Sharpe={r['sharpe']:.2f}  PnL={fmt(r['total_pnl'])}\n")
        f.write("=" * 80 + "\n\n")
        header = f"{'Session':<12} {'N':>6} {'WR':>7} {'Sharpe':>8} {'AvgPnL':>10} {'TotalPnL':>12}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for name in ["Asia", "London", "NY", "OffHours"]:
            s = sess_stats[name]
            f.write(f"{name:<12} {s['n']:>6} {s['win_rate']:>6.1f}% {s['sharpe']:>8.2f} "
                    f"{s['avg_pnl']:>10.2f} {fmt(s['total_pnl']):>12}\n")

    print("\n  Per-Session Stats:")
    for name in ["Asia", "London", "NY", "OffHours"]:
        s = sess_stats[name]
        print(f"    {name:<10} N={s['n']:>4}  WR={s['win_rate']:>5.1f}%  "
              f"Sharpe={s['sharpe']:>6.2f}  AvgPnL={s['avg_pnl']:>7.2f}")

    return trades, sess_stats


# ═══════════════════════════════════════════════════════════════
# Phase 2: Session multiplier grid search (post-hoc)
# ═══════════════════════════════════════════════════════════════

def run_phase2(out, trades):
    print("\n" + "=" * 70)
    print("Phase 2: Session-Weight Grid Search (48 combos, post-hoc)")
    print("=" * 70)

    combos = list(product(ASIA_MULTS, LONDON_MULTS, NY_MULTS, OFFHOURS_MULTS))
    print(f"  {len(combos)} combinations to evaluate")

    orig_pnls = [t['pnl'] for t in trades]
    baseline_daily = _daily_pnl_from_trades(trades, orig_pnls)
    baseline_sharpe = _sharpe_from_daily(baseline_daily)

    grid_results = []
    for asia_m, london_m, ny_m, off_m in combos:
        weights = {"Asia": asia_m, "London": london_m, "NY": ny_m, "OffHours": off_m}
        adj_pnls = _apply_session_weights(trades, weights)
        daily = _daily_pnl_from_trades(trades, adj_pnls)
        sharpe = _sharpe_from_daily(daily)
        total_pnl = sum(adj_pnls)
        max_dd = _max_dd_from_pnls(adj_pnls)
        wins = sum(1 for p in adj_pnls if p > 0)
        wr = wins / len(adj_pnls) * 100 if adj_pnls else 0

        label = f"A{asia_m}_L{london_m}_NY{ny_m}_O{off_m}"
        grid_results.append({
            'label': label, 'asia': asia_m, 'london': london_m,
            'ny': ny_m, 'offhours': off_m,
            'sharpe': round(sharpe, 4), 'total_pnl': round(total_pnl, 2),
            'max_dd': round(max_dd, 2), 'win_rate': round(wr, 1),
            'delta_sharpe': round(sharpe - baseline_sharpe, 4),
        })

    grid_results.sort(key=lambda x: x['sharpe'], reverse=True)

    with open(f"{out}/weight_grid.txt", 'w', encoding='utf-8') as f:
        f.write("R59 Phase 2: Session Weight Grid Search\n")
        f.write(f"Baseline Sharpe: {baseline_sharpe:.4f}\n")
        f.write("=" * 110 + "\n\n")
        header = (f"{'Rank':>4} {'Label':<22} {'Asia':>5} {'Lon':>5} {'NY':>5} {'Off':>5} "
                  f"{'Sharpe':>8} {'Delta':>8} {'PnL':>12} {'MaxDD':>10} {'WR':>7}")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for i, g in enumerate(grid_results, 1):
            f.write(f"{i:>4} {g['label']:<22} {g['asia']:>5.1f} {g['london']:>5.2f} "
                    f"{g['ny']:>5.1f} {g['offhours']:>5.1f} "
                    f"{g['sharpe']:>8.4f} {g['delta_sharpe']:>+8.4f} "
                    f"{fmt(g['total_pnl']):>12} {fmt(g['max_dd']):>10} "
                    f"{g['win_rate']:>6.1f}%\n")

    with open(f"{out}/weight_grid.json", 'w') as f:
        json.dump(grid_results, f, indent=2, default=str)

    print(f"\n  Baseline Sharpe: {baseline_sharpe:.4f}")
    print("  Top 5 weightings:")
    for g in grid_results[:5]:
        print(f"    {g['label']:<22} Sharpe={g['sharpe']:.4f} ({g['delta_sharpe']:>+.4f})  "
              f"PnL={fmt(g['total_pnl'])}")

    return grid_results, baseline_sharpe


# ═══════════════════════════════════════════════════════════════
# Phase 3: K-Fold validation for top 5 weightings
# ═══════════════════════════════════════════════════════════════

def run_phase3(out, top5_weights):
    print("\n" + "=" * 70)
    print("Phase 3: K-Fold 6-Fold Validation (Top 5 Weightings)")
    print("=" * 70)

    base = get_base()
    tasks = []
    for fold_name, s, e in FOLDS:
        tasks.append((f"Baseline_{fold_name}", base, SPREAD, s, e))

    fold_trades_results = run_pool(tasks, func=_run_one_trades)

    fold_trades = {}
    fold_baseline_sharpe = {}
    for r in fold_trades_results:
        fn = r['label'].replace("Baseline_", "")
        fold_trades[fn] = r['trades']
        fold_baseline_sharpe[fn] = r['sharpe']

    kfold_results = []
    for wt in top5_weights:
        weights = {"Asia": wt['asia'], "London": wt['london'],
                   "NY": wt['ny'], "OffHours": wt['offhours']}
        fold_sharpes = []
        fold_detail = {}
        for fold_name, _, _ in FOLDS:
            trades_f = fold_trades.get(fold_name, [])
            if not trades_f:
                fold_sharpes.append(0.0)
                fold_detail[fold_name] = 0.0
                continue
            adj = _apply_session_weights(trades_f, weights)
            daily = _daily_pnl_from_trades(trades_f, adj)
            sh = _sharpe_from_daily(daily)
            fold_sharpes.append(sh)
            fold_detail[fold_name] = round(sh, 2)

        pos_folds = sum(1 for s in fold_sharpes if s > 0)
        beats_baseline = sum(1 for fn in fold_detail
                             if fold_detail[fn] > fold_baseline_sharpe.get(fn, 0))
        kfold_results.append({
            'label': wt['label'],
            'weights': weights,
            'fold_sharpes': fold_detail,
            'mean_sharpe': round(np.mean(fold_sharpes), 4),
            'min_sharpe': round(min(fold_sharpes), 4),
            'positive_folds': pos_folds,
            'beats_baseline': beats_baseline,
            'passed': pos_folds >= 4,
        })

    with open(f"{out}/kfold_results.txt", 'w', encoding='utf-8') as f:
        f.write("R59 Phase 3: K-Fold 6-Fold Validation (Top 5 Weightings)\n")
        f.write("=" * 100 + "\n\n")

        f.write("Baseline per-fold Sharpe:\n")
        for fn in sorted(fold_baseline_sharpe.keys()):
            f.write(f"  {fn}: {fold_baseline_sharpe[fn]:.2f}\n")
        f.write("\n")

        for kf in kfold_results:
            status = "PASS" if kf['passed'] else "FAIL"
            f.write(f"{kf['label']} — [{status}] Pos={kf['positive_folds']}/6  "
                    f"Beats={kf['beats_baseline']}/6  Mean={kf['mean_sharpe']:.4f}  "
                    f"Min={kf['min_sharpe']:.4f}\n")
            f.write(f"  Weights: Asia={kf['weights']['Asia']}  London={kf['weights']['London']}  "
                    f"NY={kf['weights']['NY']}  OffHours={kf['weights']['OffHours']}\n")
            for fn in sorted(kf['fold_sharpes'].keys()):
                delta = kf['fold_sharpes'][fn] - fold_baseline_sharpe.get(fn, 0)
                marker = "+" if delta >= 0 else ""
                f.write(f"    {fn}: {kf['fold_sharpes'][fn]:>6.2f}  "
                        f"(vs baseline {marker}{delta:.2f})\n")
            f.write("\n")

    with open(f"{out}/kfold_results.json", 'w') as f:
        json.dump(kfold_results, f, indent=2, default=str)

    print("\n  K-Fold Results:")
    for kf in kfold_results:
        status = "PASS" if kf['passed'] else "FAIL"
        print(f"    {kf['label']:<22} {kf['positive_folds']}/6 pos  "
              f"Mean={kf['mean_sharpe']:.4f}  [{status}]")

    return kfold_results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()
    print(f"\n{'=' * 70}")
    print(f"R59: Session-Based Lot Weighting")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Lot={FIXED_LOT}  Spread=${SPREAD}  Cap=$37")
    print(f"{'=' * 70}")

    # Phase 1
    trades, sess_stats = run_phase1(OUTPUT_DIR)

    # Phase 2
    grid_results, baseline_sharpe = run_phase2(OUTPUT_DIR, trades)

    # Phase 3: top 5 by Sharpe
    top5 = grid_results[:5]
    kfold_results = run_phase3(OUTPUT_DIR, top5)

    total = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"R59 Complete — Total: {total:.0f}s ({total / 60:.1f}min)")
    print(f"{'=' * 70}")

    with open(f"{OUTPUT_DIR}/r59_summary.txt", 'w', encoding='utf-8') as f:
        f.write(f"R59 Session-Based Lot Weighting — Summary\n{'=' * 60}\n")
        f.write(f"Total time: {total:.0f}s ({total / 60:.1f}min)\n")
        f.write(f"Lot={FIXED_LOT}  Spread=${SPREAD}  Cap=$37\n\n")

        f.write("Session Quality:\n")
        for name in ["Asia", "London", "NY", "OffHours"]:
            s = sess_stats[name]
            f.write(f"  {name:<10} N={s['n']:>4}  WR={s['win_rate']:>5.1f}%  "
                    f"Sharpe={s['sharpe']:>6.2f}  AvgPnL={s['avg_pnl']:>7.2f}\n")

        f.write(f"\nBaseline Sharpe: {baseline_sharpe:.4f}\n")
        f.write(f"\nTop 5 Weightings (from {len(grid_results)} combos):\n")
        for g in grid_results[:5]:
            f.write(f"  {g['label']:<22} Sharpe={g['sharpe']:.4f} "
                    f"({g['delta_sharpe']:>+.4f})\n")

        f.write(f"\nK-Fold Validation:\n")
        for kf in kfold_results:
            status = "PASS" if kf['passed'] else "FAIL"
            f.write(f"  {kf['label']:<22} {kf['positive_folds']}/6 pos  "
                    f"Mean={kf['mean_sharpe']:.4f}  [{status}]\n")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
