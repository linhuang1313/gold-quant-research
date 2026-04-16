#!/usr/bin/env python3
"""
EXP-CHOPPY-FINE: Choppy threshold fine sweep 0.30-0.50 (step 0.01)
===================================================================
Parallel version — one process per threshold.
Base = L3 config (MaxHold=20, Tight_all trail).
21 variants full-sample + K-Fold for top-3.
"""
import sys, os, time, multiprocessing as mp
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_FILE = "exp_choppy_fine_output.txt"

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-01"),
]

THRESHOLDS = [round(0.30 + i * 0.01, 2) for i in range(21)]


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def run_one_threshold(args):
    """Run a single threshold on full sample. Loads data per-process to avoid pickle."""
    th, = args
    from backtest import DataBundle, run_variant
    from backtest.runner import LIVE_PARITY_KWARGS
    BASE = {**LIVE_PARITY_KWARGS}

    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    kwargs = {**BASE, "choppy_threshold": th}
    s = run_variant(data, f"choppy_{th:.2f}", verbose=False, **kwargs, spread_cost=0.30)
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    return (th, n, s['sharpe'], s['total_pnl'], s['win_rate'], avg, s['max_dd'])


def run_kfold_one(args):
    """Run K-Fold for one threshold."""
    th, = args
    from backtest import DataBundle, run_variant
    from backtest.runner import LIVE_PARITY_KWARGS
    BASE = {**LIVE_PARITY_KWARGS}

    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    results = []
    for fold_name, start, end in FOLDS:
        fold_data = data.slice(start, end)
        if len(fold_data.m15_df) < 1000 or len(fold_data.h1_df) < 200:
            continue
        sb = run_variant(fold_data, f"CF_base_{fold_name}", verbose=False,
                         **{**BASE, "choppy_threshold": 0.35}, spread_cost=0.30)
        st = run_variant(fold_data, f"CF_{th}_{fold_name}", verbose=False,
                         **{**BASE, "choppy_threshold": th}, spread_cost=0.30)
        delta = st['sharpe'] - sb['sharpe']
        results.append((fold_name, sb['sharpe'], st['sharpe'], delta))
    return (th, results)


def main():
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        def p(msg=""):
            print(msg, flush=True)
            f.write(msg + "\n")
            f.flush()

        p("=" * 80)
        p("EXP-CHOPPY-FINE: Choppy Threshold 0.30-0.50 (step 0.01) [PARALLEL]")
        p(f"CPUs: {mp.cpu_count()}, using {min(21, mp.cpu_count())} workers")
        p(f"Started: {datetime.now()}")
        p("=" * 80)

        t_total = time.time()

        # Part 1: parallel full-sample sweep
        p(f"\n--- Part 1: Full-sample sweep (21 thresholds, parallel) ---")
        p(f"{'Choppy':>7s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'$/trade':>8s}  {'MaxDD':>11s}")
        p("-" * 72)

        n_workers = min(21, mp.cpu_count())
        with mp.Pool(n_workers) as pool:
            results_raw = pool.map(run_one_threshold, [(th,) for th in THRESHOLDS])

        results = {}
        for th, n, sharpe, pnl, wr, avg, maxdd in sorted(results_raw, key=lambda x: x[0]):
            marker = ""
            if th == 0.35:
                marker = " <-- old"
            elif th == 0.50:
                marker = " <-- L3"
            results[th] = {'sharpe': sharpe, 'n': n, 'total_pnl': pnl, 'win_rate': wr, 'max_dd': maxdd}
            p(f"  {th:>5.2f}  {n:>6d}  {sharpe:>7.2f}  "
              f"{fmt(pnl)}  {wr:>5.1f}%  "
              f"${avg:>7.2f}  {fmt(maxdd)}{marker}")

        elapsed1 = time.time() - t_total
        p(f"\n  Part 1 done in {elapsed1/60:.1f} minutes")

        # Part 2: K-Fold for top-5 thresholds (parallel)
        ranked = sorted(results.items(), key=lambda x: -x[1]['sharpe'])
        top5 = [th for th, _ in ranked[:5]]
        p(f"\n--- Part 2: K-Fold for top-5 thresholds: {top5} (parallel) ---")

        t2 = time.time()
        with mp.Pool(min(len(top5), mp.cpu_count())) as pool:
            kfold_raw = pool.map(run_kfold_one, [(th,) for th in top5])

        for th, fold_results in sorted(kfold_raw, key=lambda x: x[0]):
            p(f"\n  Choppy={th:.2f}:")
            wins = 0
            deltas = []
            for fold_name, base_s, test_s, delta in fold_results:
                won = delta > 0
                if won:
                    wins += 1
                deltas.append(delta)
                p(f"    {fold_name}: Base(0.35)={base_s:>6.2f}  "
                  f"Test({th:.2f})={test_s:>6.2f}  delta={delta:>+.2f} {'V' if won else 'X'}")
            avg_d = sum(deltas) / len(deltas) if deltas else 0
            p(f"    Result: {wins}/{len(fold_results)} {'PASS' if wins >= 5 else 'FAIL'}  avg_delta={avg_d:>+.3f}")

        elapsed2 = time.time() - t2
        p(f"\n  Part 2 done in {elapsed2/60:.1f} minutes")

        # Part 3: Stress test top-1 at $0.50 spread
        best_th = ranked[0][0]
        p(f"\n--- Part 3: Stress test Choppy={best_th:.2f} @ $0.50 spread ---")

        from backtest import DataBundle, run_variant
        from backtest.runner import LIVE_PARITY_KWARGS
        BASE = {**LIVE_PARITY_KWARGS}
        data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

        s_base = run_variant(data, "CF_stress_base", verbose=False,
                             **{**BASE, "choppy_threshold": 0.35}, spread_cost=0.50)
        s_best = run_variant(data, "CF_stress_best", verbose=False,
                             **{**BASE, "choppy_threshold": best_th}, spread_cost=0.50)

        p(f"  Base(0.35) @ $0.50: Sharpe={s_base['sharpe']:.2f}  PnL={fmt(s_base['total_pnl'])}  N={s_base['n']}")
        p(f"  Best({best_th:.2f}) @ $0.50: Sharpe={s_best['sharpe']:.2f}  PnL={fmt(s_best['total_pnl'])}  N={s_best['n']}")
        p(f"  Delta: {s_best['sharpe'] - s_base['sharpe']:>+.2f}")

        elapsed = time.time() - t_total
        p(f"\nTotal runtime: {elapsed/60:.1f} minutes")
        p(f"Completed: {datetime.now()}")

    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
