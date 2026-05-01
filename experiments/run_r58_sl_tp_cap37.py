#!/usr/bin/env python3
"""
R58 — SL/TP Fine Grid Under Cap$37
====================================
Phase 1: Full sample, 36 variants (6 SL x 6 TP)
Phase 2: Recent 2023-2026, top 10 from Phase 1
Phase 3: K-Fold 6-fold for top 5

Baseline: SL=3.5, TP=8.0, MaxLossCap=$37
"""
import sys, os, io, time
import multiprocessing as mp
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "results/r58_sl_tp_cap37"
MAX_WORKERS = 6
FIXED_LOT = 0.05
SPREAD = 0.30

SL_MULTS = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
TP_MULTS = [6.0, 7.0, 8.0, 9.0, 10.0, 12.0]

BASELINE_SL = 3.5
BASELINE_TP = 8.0

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def get_base():
    from backtest.runner import LIVE_PARITY_KWARGS
    return {**LIVE_PARITY_KWARGS, 'min_lot_size': FIXED_LOT, 'max_lot_size': FIXED_LOT, 'maxloss_cap': 37}


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
        'max_dd': s['max_dd'], 'maxloss_cap_count': s.get('maxloss_cap_count', 0),
    }


def run_pool(tasks):
    n = min(MAX_WORKERS, len(tasks))
    with mp.Pool(n) as pool:
        return pool.map(_run_one, tasks)


def _variant_label(sl, tp):
    return f"SL{sl}_TP{tp}"


def _is_baseline(sl, tp):
    return sl == BASELINE_SL and tp == BASELINE_TP


def _write_table(f, rows, mark_baseline=True):
    header = (f"{'Variant':<16} {'N':>5} {'Sharpe':>8} {'PnL':>12} "
              f"{'WR':>7} {'MaxDD':>10} {'CapHits':>8} {'CapHit%':>8}")
    f.write(header + "\n")
    f.write("-" * len(header) + "\n")
    for r in rows:
        cap_pct = (r['maxloss_cap_count'] / r['n'] * 100) if r['n'] > 0 else 0
        marker = " <<<" if mark_baseline and r.get('is_baseline') else ""
        f.write(f"{r['label']:<16} {r['n']:>5} {r['sharpe']:>8.2f} "
                f"{fmt(r['total_pnl']):>12} {r['win_rate']:>6.1f}% "
                f"{fmt(r['max_dd']):>10} {r['maxloss_cap_count']:>8} "
                f"{cap_pct:>7.1f}%{marker}\n")


def phase1_full():
    """Phase 1: Full sample, all 36 variants."""
    print(f"\n{'='*70}")
    print("Phase 1: Full Sample — 36 SL x TP Variants")
    print(f"{'='*70}")
    t0 = time.time()

    tasks = []
    for sl in SL_MULTS:
        for tp in TP_MULTS:
            kw = get_base()
            kw['sl_atr_mult'] = sl
            kw['tp_atr_mult'] = tp
            tasks.append((_variant_label(sl, tp), kw, SPREAD, None, None))

    results = run_pool(tasks)

    for r in results:
        for sl in SL_MULTS:
            for tp in TP_MULTS:
                if r['label'] == _variant_label(sl, tp):
                    r['is_baseline'] = _is_baseline(sl, tp)

    results.sort(key=lambda x: x['sharpe'], reverse=True)

    with open(f"{OUTPUT_DIR}/phase1_full.txt", 'w', encoding='utf-8') as f:
        f.write("R58 Phase 1: Full Sample — SL x TP Grid (36 variants, Cap$37)\n")
        f.write(f"Lot={FIXED_LOT}, Spread=${SPREAD}, MaxLossCap=$37\n")
        f.write(f"Baseline: SL={BASELINE_SL}, TP={BASELINE_TP}\n")
        f.write("=" * 100 + "\n\n")
        _write_table(f, results)

        f.write(f"\n\n--- SL x TP Sharpe Matrix ---\n")
        header = 'SL\\TP'
        f.write(f"{header:<8}")
        for tp in TP_MULTS:
            f.write(f" {'TP'+str(tp):>8}")
        f.write("\n")
        f.write("-" * (8 + 9 * len(TP_MULTS)) + "\n")
        for sl in SL_MULTS:
            f.write(f"SL{sl:<6}")
            for tp in TP_MULTS:
                r = next((x for x in results if x['label'] == _variant_label(sl, tp)), None)
                sh = r['sharpe'] if r else 0
                marker = "*" if _is_baseline(sl, tp) else " "
                f.write(f" {sh:>7.2f}{marker}")
            f.write("\n")
        f.write("  (* = baseline)\n")

    bl = next((r for r in results if r.get('is_baseline')), None)
    bl_rank = next((i + 1 for i, r in enumerate(results) if r.get('is_baseline')), None)
    best = results[0]
    elapsed = time.time() - t0

    print(f"  Phase 1 done in {elapsed:.0f}s")
    print(f"  Best: {best['label']}  Sharpe={best['sharpe']:.2f}  PnL={fmt(best['total_pnl'])}")
    if bl:
        print(f"  Baseline rank: #{bl_rank}/{len(results)}  Sharpe={bl['sharpe']:.2f}")

    return results, elapsed


def phase2_recent(phase1_results):
    """Phase 2: Recent 2023-2026, top 10 from Phase 1."""
    print(f"\n{'='*70}")
    print("Phase 2: Recent 2023-2026 — Top 10 from Phase 1")
    print(f"{'='*70}")
    t0 = time.time()

    top10_labels = [r['label'] for r in phase1_results[:10]]
    bl_label = _variant_label(BASELINE_SL, BASELINE_TP)
    if bl_label not in top10_labels:
        top10_labels.append(bl_label)

    tasks = []
    for lbl in top10_labels:
        sl = float(lbl.split('_TP')[0].replace('SL', ''))
        tp = float(lbl.split('_TP')[1])
        kw = get_base()
        kw['sl_atr_mult'] = sl
        kw['tp_atr_mult'] = tp
        tasks.append((lbl, kw, SPREAD, "2023-01-01", "2026-05-01"))

    results = run_pool(tasks)
    for r in results:
        r['is_baseline'] = (r['label'] == bl_label)
    results.sort(key=lambda x: x['sharpe'], reverse=True)

    with open(f"{OUTPUT_DIR}/phase2_recent.txt", 'w', encoding='utf-8') as f:
        f.write("R58 Phase 2: Recent 2023-2026 — Top 10 + Baseline\n")
        f.write(f"Lot={FIXED_LOT}, Spread=${SPREAD}, MaxLossCap=$37\n")
        f.write("=" * 100 + "\n\n")
        _write_table(f, results)

    elapsed = time.time() - t0
    print(f"  Phase 2 done in {elapsed:.0f}s")
    best = results[0]
    print(f"  Best recent: {best['label']}  Sharpe={best['sharpe']:.2f}")

    return results, elapsed


def phase3_kfold(phase1_results):
    """Phase 3: K-Fold 6-fold for top 5 variants + baseline."""
    print(f"\n{'='*70}")
    print("Phase 3: K-Fold 6-Fold — Top 5 + Baseline")
    print(f"{'='*70}")
    t0 = time.time()

    top5_labels = [r['label'] for r in phase1_results[:5]]
    bl_label = _variant_label(BASELINE_SL, BASELINE_TP)
    if bl_label not in top5_labels:
        top5_labels.append(bl_label)

    tasks = []
    for lbl in top5_labels:
        sl = float(lbl.split('_TP')[0].replace('SL', ''))
        tp = float(lbl.split('_TP')[1])
        for fold_name, start, end in FOLDS:
            kw = get_base()
            kw['sl_atr_mult'] = sl
            kw['tp_atr_mult'] = tp
            tasks.append((f"{lbl}_{fold_name}", kw, SPREAD, start, end))

    results = run_pool(tasks)

    bl_sharpes = {}
    for fold_name, _, _ in FOLDS:
        bl_r = next((r for r in results if r['label'] == f"{bl_label}_{fold_name}"), None)
        bl_sharpes[fold_name] = bl_r['sharpe'] if bl_r else 0

    with open(f"{OUTPUT_DIR}/phase3_kfold.txt", 'w', encoding='utf-8') as f:
        f.write("R58 Phase 3: K-Fold 6-Fold Validation\n")
        f.write(f"Lot={FIXED_LOT}, Spread=${SPREAD}, MaxLossCap=$37\n")
        f.write("=" * 100 + "\n\n")

        fold_names = [fn for fn, _, _ in FOLDS]
        header_parts = [f"{'Variant':<16}"]
        for fn in fold_names:
            header_parts.append(f"{fn:>8}")
        header_parts.extend([f"{'Mean':>8}", f"{'Std':>8}", f"{'BeatBL':>7}"])
        header = " ".join(header_parts)
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        summary_rows = []
        for lbl in top5_labels:
            fold_sharpes = []
            beats = 0
            parts = [f"{lbl:<16}"]
            for fold_name in fold_names:
                r = next((x for x in results if x['label'] == f"{lbl}_{fold_name}"), None)
                sh = r['sharpe'] if r else 0
                fold_sharpes.append(sh)
                if sh > bl_sharpes.get(fold_name, 0) or lbl == bl_label:
                    beats += 1
                parts.append(f"{sh:>8.2f}")
            import numpy as np
            mean_sh = np.mean(fold_sharpes)
            std_sh = np.std(fold_sharpes)
            is_bl = lbl == bl_label
            if is_bl:
                beats_str = "  base"
            else:
                beats_str = f"  {beats}/6"
            parts.extend([f"{mean_sh:>8.2f}", f"{std_sh:>8.2f}", beats_str])
            line = " ".join(parts)
            if is_bl:
                line += " <<<"
            f.write(line + "\n")
            summary_rows.append({
                'label': lbl, 'mean_sharpe': mean_sh, 'std_sharpe': std_sh,
                'beats': beats, 'is_baseline': is_bl, 'fold_sharpes': fold_sharpes,
            })

    elapsed = time.time() - t0
    print(f"  Phase 3 done in {elapsed:.0f}s")
    for row in summary_rows:
        print(f"    {row['label']:<16} mean={row['mean_sharpe']:.2f}  std={row['std_sharpe']:.2f}  "
              f"beats_bl={'base' if row['is_baseline'] else str(row['beats']) + '/6'}")

    return summary_rows, elapsed


def write_summary(p1_results, p2_results, p3_rows, times):
    """Write final summary."""
    with open(f"{OUTPUT_DIR}/r58_summary.txt", 'w', encoding='utf-8') as f:
        f.write("R58 Summary: SL/TP Fine Grid Under Cap$37\n")
        f.write(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Lot={FIXED_LOT}, Spread=${SPREAD}, MaxLossCap=$37\n")
        f.write(f"Grid: {len(SL_MULTS)} SL x {len(TP_MULTS)} TP = "
                f"{len(SL_MULTS)*len(TP_MULTS)} variants\n")
        f.write(f"Baseline: SL={BASELINE_SL}, TP={BASELINE_TP}\n")
        f.write("=" * 100 + "\n\n")

        f.write("--- Phase 1: Full Sample Top 5 ---\n")
        for i, r in enumerate(p1_results[:5], 1):
            marker = " <<< BASELINE" if r.get('is_baseline') else ""
            f.write(f"  #{i} {r['label']:<16} Sharpe={r['sharpe']:.2f}  "
                    f"PnL={fmt(r['total_pnl'])}  WR={r['win_rate']:.1f}%  "
                    f"MaxDD={fmt(r['max_dd'])}{marker}\n")

        bl = next((r for r in p1_results if r.get('is_baseline')), None)
        bl_rank = next((i + 1 for i, r in enumerate(p1_results) if r.get('is_baseline')), None)
        if bl:
            f.write(f"\n  Baseline rank: #{bl_rank}/{len(p1_results)}\n")

        f.write(f"\n--- Phase 2: Recent 2023-2026 Top 5 ---\n")
        for i, r in enumerate(p2_results[:5], 1):
            marker = " <<< BASELINE" if r.get('is_baseline') else ""
            f.write(f"  #{i} {r['label']:<16} Sharpe={r['sharpe']:.2f}  "
                    f"PnL={fmt(r['total_pnl'])}  WR={r['win_rate']:.1f}%{marker}\n")

        f.write(f"\n--- Phase 3: K-Fold Summary ---\n")
        p3_sorted = sorted(p3_rows, key=lambda x: x['mean_sharpe'], reverse=True)
        for row in p3_sorted:
            marker = " <<< BASELINE" if row['is_baseline'] else ""
            beats = 'base' if row['is_baseline'] else f"{row['beats']}/6"
            f.write(f"  {row['label']:<16} mean_Sharpe={row['mean_sharpe']:.2f}  "
                    f"std={row['std_sharpe']:.2f}  beats_BL={beats}{marker}\n")

        f.write(f"\n--- Timing ---\n")
        for phase, t in times.items():
            f.write(f"  {phase}: {t:.0f}s ({t/60:.1f}min)\n")
        total = sum(times.values())
        f.write(f"  Total: {total:.0f}s ({total/60:.1f}min)\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_total = time.time()

    print(f"\n{'='*70}")
    print(f"R58: SL/TP Fine Grid Under Cap$37")
    print(f"Grid: {len(SL_MULTS)} SL x {len(TP_MULTS)} TP = "
          f"{len(SL_MULTS)*len(TP_MULTS)} variants")
    print(f"Baseline: SL={BASELINE_SL}, TP={BASELINE_TP}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    times = {}

    p1_results, t1 = phase1_full()
    times['Phase 1 (Full)'] = t1

    p2_results, t2 = phase2_recent(p1_results)
    times['Phase 2 (Recent)'] = t2

    p3_rows, t3 = phase3_kfold(p1_results)
    times['Phase 3 (K-Fold)'] = t3

    write_summary(p1_results, p2_results, p3_rows, times)

    total = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"R58 Complete — Total: {total:.0f}s ({total/60:.1f}min)")
    print(f"Results: {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
