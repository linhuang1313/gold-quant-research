#!/usr/bin/env python3
"""Re-run R6-B4 only (standalone, no import from run_local_remaining)"""
import sys, os, io, time
import multiprocessing as mp
from datetime import datetime

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT = "results/round6_results/r6_b4_interaction.txt"
MAX_WORKERS = 6

def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"

def _run_one(args):
    label, kw, spread, start, end = args
    from backtest.runner import DataBundle, run_variant
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    if start and end:
        data = data.slice(start, end)
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kw)
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'])

def get_base():
    from backtest.runner import LIVE_PARITY_KWARGS
    return {**LIVE_PARITY_KWARGS}

if __name__ == "__main__":
    os.makedirs("results/round6_results", exist_ok=True)

    lines = []
    lines.append(f"# R6-B4: 参数交互效应")
    lines.append(f"# Started: {datetime.now()}")
    lines.append(f"# Local re-run (format fix)\n")

    def p(msg=""):
        lines.append(msg)
        print(msg, flush=True)

    t0 = time.time()

    p("=" * 80)
    p("R6-B4: 参数交互效应 — AllTight Trail x Choppy 二维网格")
    p("=" * 80)

    L5 = get_base()
    L5_regime = L5['regime_config']

    trail_mults = [0.8, 0.9, 1.0, 1.1, 1.2]
    choppy_vals = [0.40, 0.45, 0.50, 0.55, 0.60]

    tasks = []
    for tm in trail_mults:
        for ch in choppy_vals:
            regime = {}
            for rk in ['low', 'normal', 'high']:
                regime[rk] = {
                    'trail_act': round(L5_regime[rk]['trail_act'] * tm, 4),
                    'trail_dist': round(L5_regime[rk]['trail_dist'] * tm, 4),
                }
            kw = {**L5,
                  "regime_config": regime,
                  "trailing_activate_atr": regime['normal']['trail_act'],
                  "trailing_distance_atr": regime['normal']['trail_dist'],
                  "choppy_threshold": ch}
            tasks.append((f"T{tm:.1f}_C{ch:.2f}", kw, 0.30, None, None))

    print(f"Running {len(tasks)} combinations with {MAX_WORKERS} workers...", flush=True)

    with mp.Pool(min(MAX_WORKERS, len(tasks))) as pool:
        results = pool.map(_run_one, tasks)

    p(f"\n--- Sharpe 矩阵 ---")
    col_header = "Trail/Choppy"
    header = f"  {col_header:<14}"
    for ch in choppy_vals:
        ch_str = f"{ch:.2f}"
        header += f" {ch_str:>8}"
    p(header)

    result_map = {r[0]: r for r in results}
    best_sharpe = -999
    best_combo = ""
    for tm in trail_mults:
        marker = " <-L5" if abs(tm - 1.0) < 0.01 else ""
        row_label = f"x{tm:.1f}{marker}"
        row = f"  {row_label:<14}"
        for ch in choppy_vals:
            key = f"T{tm:.1f}_C{ch:.2f}"
            r = result_map.get(key)
            if r:
                if r[2] > best_sharpe:
                    best_sharpe = r[2]
                    best_combo = key
                row += f" {r[2]:>8.2f}"
            else:
                row += f" {'N/A':>8}"
        p(row)

    p(f"\n  最优组合: {best_combo} (Sharpe={best_sharpe:.2f})")
    p(f"  当前 L5: T1.0_C0.50")

    p(f"\n--- PnL 矩阵 ---")
    header = f"  {col_header:<14}"
    for ch in choppy_vals:
        ch_str = f"{ch:.2f}"
        header += f" {ch_str:>8}"
    p(header)
    for tm in trail_mults:
        row_label = f"x{tm:.1f}"
        row = f"  {row_label:<14}"
        for ch in choppy_vals:
            key = f"T{tm:.1f}_C{ch:.2f}"
            r = result_map.get(key)
            if r:
                row += f" {r[3]:>8,.0f}"
            else:
                row += f" {'N/A':>8}"
        p(row)

    elapsed = (time.time() - t0) / 60
    lines.append(f"\n# Completed: {datetime.now()}")
    lines.append(f"# Elapsed: {elapsed:.1f} minutes")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nDONE in {elapsed:.1f} min -> {OUTPUT}", flush=True)
