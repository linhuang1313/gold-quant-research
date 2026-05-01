#!/usr/bin/env python3
"""
R53 — Lot × MaxLoss Cap 联合优化
==================================
Phase 1: 固定 0.04 手，Cap $30-$85 网格 + 无 Cap 基线 (13 变体)
Phase 2: Phase 1 最优 Cap × 手数网格 0.02-0.06 (15 变体)
Phase 3: Top 3 K-Fold 6 折验证
"""
import sys, os, io, time, json, traceback
import multiprocessing as mp
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "results/r53_lot_cap"
MAX_WORKERS = 6

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


def get_base():
    from backtest.runner import LIVE_PARITY_KWARGS
    return {**LIVE_PARITY_KWARGS}


def _run_one(args):
    label, kw, spread, start, end = args
    from backtest.runner import DataBundle, run_variant
    data = DataBundle.load_custom()
    if start and end:
        data = data.slice(start, end)
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kw)
    return {
        'label': label,
        'n': s['n'],
        'sharpe': s['sharpe'],
        'total_pnl': s['total_pnl'],
        'win_rate': s['win_rate'],
        'max_dd': s['max_dd'],
        'max_dd_pct': s.get('max_dd_pct', 0),
        'avg_pnl': s['total_pnl'] / s['n'] if s['n'] > 0 else 0,
        'maxloss_cap_count': s.get('maxloss_cap_count', 0),
        'year_pnl': s.get('year_pnl', {}),
        'elapsed_s': s.get('elapsed_s', 0),
    }


def _run_one_trades(args):
    label, kw, spread, start, end = args
    from backtest.runner import DataBundle, run_variant
    data = DataBundle.load_custom()
    if start and end:
        data = data.slice(start, end)
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kw)
    trades = s.get('_trades', [])
    exit_reasons = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0})
    for t in trades:
        r = t.exit_reason or 'Unknown'
        exit_reasons[r]['n'] += 1
        exit_reasons[r]['pnl'] += t.pnl
        if t.pnl > 0:
            exit_reasons[r]['wins'] += 1
    loss_pnls = [t.pnl for t in trades if t.pnl < 0]
    return {
        'label': label,
        'n': s['n'],
        'sharpe': s['sharpe'],
        'total_pnl': s['total_pnl'],
        'win_rate': s['win_rate'],
        'max_dd': s['max_dd'],
        'maxloss_cap_count': s.get('maxloss_cap_count', 0),
        'exit_reasons': dict(exit_reasons),
        'avg_loss': np.mean(loss_pnls) if loss_pnls else 0,
        'median_loss': np.median(loss_pnls) if loss_pnls else 0,
        'max_loss': min(loss_pnls) if loss_pnls else 0,
    }


def run_pool(tasks, func=_run_one):
    n_workers = min(MAX_WORKERS, len(tasks))
    with mp.Pool(n_workers) as pool:
        return pool.map(func, tasks)


# ═══════════════════════════════════════════════════════════════
# Phase 1: Cap 网格 (固定 0.04 手)
# ═══════════════════════════════════════════════════════════════

def run_phase1(out):
    print("\n" + "=" * 70)
    print("Phase 1: Cap Grid Search (fixed lot=0.04)")
    print("=" * 70)

    base = get_base()
    cap_values = [0, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
    tasks = []
    for cap in cap_values:
        kw = {**base, 'min_lot_size': 0.04, 'max_lot_size': 0.04}
        if cap > 0:
            kw['maxloss_cap'] = cap
        label = f"CAP_{cap}" if cap > 0 else "CAP_NONE"
        tasks.append((label, kw, 0.30, None, None))

    results = run_pool(tasks, func=_run_one_trades)
    results.sort(key=lambda x: x['sharpe'], reverse=True)

    with open(f"{out}/phase1_cap_grid.txt", 'w', encoding='utf-8') as f:
        f.write("Phase 1: Cap Grid Search (lot=0.04, spread=$0.30)\n")
        f.write("=" * 100 + "\n\n")
        header = (f"{'Label':<12} {'N':>5} {'Sharpe':>8} {'PnL':>12} {'WR':>7} "
                  f"{'MaxDD':>10} {'AvgLoss':>10} {'MaxLoss':>10} "
                  f"{'CapHits':>8} {'CapHit%':>8}")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for r in results:
            cap_pct = (r['maxloss_cap_count'] / r['n'] * 100) if r['n'] > 0 else 0
            f.write(f"{r['label']:<12} {r['n']:>5} {r['sharpe']:>8.2f} "
                    f"{fmt(r['total_pnl']):>12} {r['win_rate']:>6.1f}% "
                    f"{fmt(r['max_dd']):>10} {r['avg_loss']:>10.2f} "
                    f"{r['max_loss']:>10.2f} "
                    f"{r['maxloss_cap_count']:>8} {cap_pct:>7.1f}%\n")

        f.write("\n\n--- Exit Reason Breakdown ---\n")
        for r in results:
            f.write(f"\n{r['label']}:\n")
            for reason, stats in sorted(r['exit_reasons'].items()):
                wr = (stats['wins'] / stats['n'] * 100) if stats['n'] > 0 else 0
                avg = stats['pnl'] / stats['n'] if stats['n'] > 0 else 0
                f.write(f"  {reason:<15} N={stats['n']:>4}  PnL={fmt(stats['pnl']):>10}  "
                        f"WR={wr:>5.1f}%  Avg={avg:>8.2f}\n")

    with open(f"{out}/phase1_cap_grid.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n  Phase 1 Results (sorted by Sharpe):")
    for r in results[:5]:
        print(f"    {r['label']:<12} Sharpe={r['sharpe']:>6.2f}  PnL={fmt(r['total_pnl'])}  "
              f"CapHits={r['maxloss_cap_count']}")

    best_cap = results[0]
    print(f"\n  >>> Best Cap: {best_cap['label']} (Sharpe={best_cap['sharpe']:.2f})")
    return results


# ═══════════════════════════════════════════════════════════════
# Phase 2: 手数网格 (固定最优 Cap + 当前 Cap + 无 Cap)
# ═══════════════════════════════════════════════════════════════

def run_phase2(out, best_cap_value):
    print("\n" + "=" * 70)
    print(f"Phase 2: Lot Grid Search (best_cap=${best_cap_value}, current=$80, none)")
    print("=" * 70)

    base = get_base()
    lot_values = [0.02, 0.03, 0.04, 0.05, 0.06]
    cap_values = [best_cap_value, 80, 0]
    if best_cap_value == 80:
        cap_values = [80, 0]

    tasks = []
    for lot in lot_values:
        for cap in cap_values:
            kw = {**base, 'min_lot_size': lot, 'max_lot_size': lot}
            if cap > 0:
                kw['maxloss_cap'] = cap
            cap_label = f"Cap{cap}" if cap > 0 else "NoCap"
            label = f"Lot{lot}_{cap_label}"
            tasks.append((label, kw, 0.30, None, None))

    results = run_pool(tasks, func=_run_one_trades)
    results.sort(key=lambda x: x['sharpe'], reverse=True)

    with open(f"{out}/phase2_lot_grid.txt", 'w', encoding='utf-8') as f:
        f.write(f"Phase 2: Lot Grid (best_cap=${best_cap_value})\n")
        f.write("=" * 100 + "\n\n")
        header = (f"{'Label':<20} {'N':>5} {'Sharpe':>8} {'PnL':>12} {'WR':>7} "
                  f"{'MaxDD':>10} {'AvgLoss':>10} {'CapHits':>8}")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for r in results:
            f.write(f"{r['label']:<20} {r['n']:>5} {r['sharpe']:>8.2f} "
                    f"{fmt(r['total_pnl']):>12} {r['win_rate']:>6.1f}% "
                    f"{fmt(r['max_dd']):>10} {r['avg_loss']:>10.2f} "
                    f"{r['maxloss_cap_count']:>8}\n")

        f.write("\n\n--- Lot x Cap Matrix (Sharpe) ---\n")
        f.write(f"{'Lot':>6}")
        for cap in cap_values:
            cl = f"Cap{cap}" if cap > 0 else "NoCap"
            f.write(f" {cl:>10}")
        f.write("\n")
        for lot in lot_values:
            f.write(f"{lot:>6.2f}")
            for cap in cap_values:
                cl = f"Cap{cap}" if cap > 0 else "NoCap"
                label = f"Lot{lot}_{cl}"
                r = next((x for x in results if x['label'] == label), None)
                sh = r['sharpe'] if r else 0
                f.write(f" {sh:>10.2f}")
            f.write("\n")

    with open(f"{out}/phase2_lot_grid.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n  Phase 2 Results (top 5 by Sharpe):")
    for r in results[:5]:
        print(f"    {r['label']:<20} Sharpe={r['sharpe']:>6.2f}  PnL={fmt(r['total_pnl'])}")

    return results


# ═══════════════════════════════════════════════════════════════
# Phase 3: K-Fold 6 折验证 (Top 3)
# ═══════════════════════════════════════════════════════════════

def run_phase3(out, top_configs):
    """
    top_configs: list of (label, lot, cap) tuples
    """
    print("\n" + "=" * 70)
    print("Phase 3: K-Fold Validation (Top 3)")
    print("=" * 70)

    base = get_base()
    tasks = []

    baseline_lot, baseline_cap = 0.04, 80
    for fold_name, s, e in FOLDS:
        kw = {**base, 'min_lot_size': baseline_lot, 'max_lot_size': baseline_lot,
              'maxloss_cap': baseline_cap}
        tasks.append((f"Baseline_{fold_name}", kw, 0.30, s, e))

    for label, lot, cap in top_configs:
        for fold_name, s, e in FOLDS:
            kw = {**base, 'min_lot_size': lot, 'max_lot_size': lot}
            if cap > 0:
                kw['maxloss_cap'] = cap
            tasks.append((f"{label}_{fold_name}", kw, 0.30, s, e))

    results = run_pool(tasks)

    with open(f"{out}/phase3_kfold.txt", 'w', encoding='utf-8') as f:
        f.write("Phase 3: K-Fold 6-Fold Validation\n")
        f.write("=" * 100 + "\n\n")

        baseline_folds = {}
        for r in results:
            if r['label'].startswith("Baseline_"):
                fn = r['label'].replace("Baseline_", "")
                baseline_folds[fn] = r['sharpe']

        f.write("Baseline (Lot=0.04, Cap=$80):\n")
        for fn in sorted(baseline_folds.keys()):
            f.write(f"  {fn}: Sharpe={baseline_folds[fn]:>6.2f}\n")
        f.write("\n")

        kfold_summary = []
        for label, lot, cap in top_configs:
            cfg_folds = {}
            for r in results:
                if r['label'].startswith(f"{label}_"):
                    fn = r['label'].replace(f"{label}_", "")
                    cfg_folds[fn] = r['sharpe']

            wins = sum(1 for fn in cfg_folds
                       if cfg_folds.get(fn, 0) >= baseline_folds.get(fn, 0))
            passed = wins >= 4
            kfold_summary.append((label, wins, passed))

            f.write(f"{label} (Lot={lot}, Cap={'$'+str(cap) if cap>0 else 'None'}): "
                    f"K-Fold {wins}/6 {'PASS' if passed else 'FAIL'}\n")
            for fn in sorted(cfg_folds.keys()):
                delta = cfg_folds.get(fn, 0) - baseline_folds.get(fn, 0)
                marker = "+" if delta >= 0 else ""
                f.write(f"  {fn}: Sharpe={cfg_folds[fn]:>6.2f} "
                        f"(vs baseline {marker}{delta:.2f})\n")
            f.write("\n")

    with open(f"{out}/phase3_kfold.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n  K-Fold Results:")
    for label, wins, passed in kfold_summary:
        status = "PASS" if passed else "FAIL"
        print(f"    {label:<20} {wins}/6 folds  [{status}]")

    return kfold_summary


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()
    print(f"\n{'=' * 70}")
    print(f"R53: Lot x MaxLoss Cap Joint Optimization")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")

    # Phase 1
    p1_results = run_phase1(OUTPUT_DIR)

    # Extract best Cap value for Phase 2
    best = p1_results[0]
    best_label = best['label']
    if best_label == 'CAP_NONE':
        best_cap_value = 0
    else:
        best_cap_value = int(best_label.split('_')[1])
    print(f"\n  Phase 1 winner: {best_label} → Cap={best_cap_value}")

    # Phase 2
    p2_results = run_phase2(OUTPUT_DIR, best_cap_value)

    # Top 3 for K-Fold: best from P1 + P2 combined, deduplicated
    all_results = []
    for r in p1_results:
        label = r['label']
        if label == 'CAP_NONE':
            lot, cap = 0.04, 0
        else:
            lot, cap = 0.04, int(label.split('_')[1])
        all_results.append((label, lot, cap, r['sharpe']))
    for r in p2_results:
        label = r['label']
        parts = label.split('_')
        lot = float(parts[0].replace('Lot', ''))
        cap_str = parts[1]
        cap = int(cap_str.replace('Cap', '')) if cap_str != 'NoCap' else 0
        all_results.append((label, lot, cap, r['sharpe']))

    # Deduplicate by (lot, cap), keep best sharpe
    seen = {}
    for label, lot, cap, sharpe in all_results:
        key = (lot, cap)
        if key not in seen or sharpe > seen[key][3]:
            seen[key] = (label, lot, cap, sharpe)

    top3 = sorted(seen.values(), key=lambda x: x[3], reverse=True)[:3]
    top3_configs = [(t[0], t[1], t[2]) for t in top3]
    print(f"\n  Top 3 for K-Fold: {[t[0] for t in top3]}")

    # Phase 3
    kfold_results = run_phase3(OUTPUT_DIR, top3_configs)

    total = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"R53 Complete — Total: {total:.0f}s ({total / 60:.1f}min)")
    print(f"{'=' * 70}")

    with open(f"{OUTPUT_DIR}/r53_summary.txt", 'w', encoding='utf-8') as f:
        f.write(f"R53 Lot x Cap Summary\n{'=' * 60}\n")
        f.write(f"Total time: {total:.0f}s ({total / 60:.1f}min)\n\n")
        f.write(f"Phase 1 best: {best_label} (Sharpe={best['sharpe']:.2f})\n")
        f.write(f"Phase 2 best: {p2_results[0]['label']} (Sharpe={p2_results[0]['sharpe']:.2f})\n\n")
        f.write("K-Fold results:\n")
        for label, wins, passed in kfold_results:
            f.write(f"  {label}: {wins}/6 {'PASS' if passed else 'FAIL'}\n")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
