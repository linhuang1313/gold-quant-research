#!/usr/bin/env python3
"""
本地补跑: R6B-B4/B5/B6 + R7-6
服务器断连后未完成的实验
"""
import sys, os, io, time, traceback
import multiprocessing as mp
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "results/round6_results"
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

def _run_one_trades(args):
    label, kw, spread, start, end = args
    from backtest.runner import DataBundle, run_variant
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    if start and end:
        data = data.slice(start, end)
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kw)
    trades = s.get('_trades', [])
    td = [(round(t.pnl, 2), t.exit_reason or '', t.bars_held, t.strategy or '',
           str(t.entry_time)[:16], t.direction or '',
           round(getattr(t, 'max_favorable', 0) or 0, 2))
          for t in trades[:5000]]
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'], td)

def run_pool(tasks, func=None):
    if func is None:
        func = _run_one
    with mp.Pool(min(MAX_WORKERS, len(tasks))) as pool:
        return pool.map(func, tasks)

def get_base():
    from backtest.runner import LIVE_PARITY_KWARGS
    return {**LIVE_PARITY_KWARGS}

ULTRA2 = {
    'low': {'trail_act': 0.30, 'trail_dist': 0.06},
    'normal': {'trail_act': 0.20, 'trail_dist': 0.04},
    'high': {'trail_act': 0.08, 'trail_dist': 0.01},
}


# ═══════════════════════════════════════════
# R6-B4: 参数交互效应
# ═══════════════════════════════════════════
def r6_b4_interaction_grid(p):
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

    results = run_pool(tasks)

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


# ═══════════════════════════════════════════
# R6-B5: 2025-2026 近期数据放大镜
# ═══════════════════════════════════════════
def r6_b5_recent_zoom(p):
    p("=" * 80)
    p("R6-B5: 2025-2026 近期数据放大镜")
    p("=" * 80)
    p("\n  焦点: Trump 关税行情下策略是否退化\n")

    L5 = get_base()
    L6 = {**L5, "regime_config": ULTRA2,
          "trailing_activate_atr": 0.20, "trailing_distance_atr": 0.04,
          "max_positions": 1}

    configs = [
        ("L5", L5),
        ("L6", L6),
        ("L5_MP1", {**L5, "max_positions": 1}),
        ("L5_UT2", {**L5, "regime_config": ULTRA2,
                    "trailing_activate_atr": 0.20, "trailing_distance_atr": 0.04}),
    ]

    periods = [
        ("2025_full", "2025-01-01", "2026-04-10"),
        ("2025_Q1", "2025-01-01", "2025-04-01"),
        ("2025_Q2", "2025-04-01", "2025-07-01"),
        ("2025_Q3", "2025-07-01", "2025-10-01"),
        ("2025_Q4", "2025-10-01", "2026-01-01"),
        ("2026_Q1", "2026-01-01", "2026-04-10"),
        ("tariff_period", "2025-03-01", "2026-04-10"),
    ]

    tasks = []
    for period_name, start, end in periods:
        for config_name, kw in configs:
            tasks.append((f"{config_name}_{period_name}", kw, 0.30, start, end))

    results = run_pool(tasks)

    for period_name, start, end in periods:
        p(f"\n--- {period_name} ({start} -> {end}) ---")
        p(f"  {'Config':<12} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR%':>6} {'MaxDD':>10}")
        period_results = [r for r in results if r[0].endswith(period_name)]
        for r in period_results:
            config = r[0].replace(f"_{period_name}", "")
            p(f"  {config:<12} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1%} {fmt(r[6]):>10}")


# ═══════════════════════════════════════════
# R6-B6: 年度稳定性热力图
# ═══════════════════════════════════════════
def r6_b6_monthly_heatmap(p):
    p("=" * 80)
    p("R6-B6: 年度稳定性热力图 (L5 月度 PnL)")
    p("=" * 80)

    L5 = get_base()

    tasks = []
    for y in range(2015, 2027):
        for m in range(1, 13):
            if y == 2026 and m > 3:
                continue
            start = f"{y}-{m:02d}-01"
            if m < 12:
                end = f"{y}-{m+1:02d}-01"
            else:
                end = f"{y+1}-01-01"
            tasks.append((f"M_{y}_{m:02d}", L5, 0.30, start, end))

    results = run_pool(tasks)

    result_map = {}
    for r in results:
        parts = r[0].split('_')
        y, m = int(parts[1]), int(parts[2])
        result_map[(y, m)] = r

    months = 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split()

    p(f"\n--- 月度 PnL 矩阵 ---")
    header = f"  {'Year':<6}"
    for m in range(1, 13):
        header += f" {months[m-1]:>6}"
    header += f" {'Total':>8}"
    p(header)

    for y in range(2015, 2027):
        row = f"  {y:<6}"
        yr_total = 0
        for m in range(1, 13):
            r = result_map.get((y, m))
            if r and r[1] > 0:
                row += f" {r[3]:>6.0f}"
                yr_total += r[3]
            else:
                row += f" {'--':>6}"
        row += f" {yr_total:>8.0f}"
        p(row)

    p(f"\n--- 月度 Sharpe 矩阵 ---")
    header = f"  {'Year':<6}"
    for m in range(1, 13):
        header += f" {months[m-1]:>6}"
    p(header)

    for y in range(2015, 2027):
        row = f"  {y:<6}"
        for m in range(1, 13):
            r = result_map.get((y, m))
            if r and r[1] > 0:
                row += f" {r[2]:>6.1f}"
            else:
                row += f" {'--':>6}"
        p(row)

    neg_months = sum(1 for r in results if r[1] > 0 and r[3] < 0)
    total_months = sum(1 for r in results if r[1] > 0)
    p(f"\n  亏损月份: {neg_months}/{total_months} ({neg_months/total_months*100:.0f}%)")


# ═══════════════════════════════════════════
# R7-6: 近期放大镜 (修复版)
# ═══════════════════════════════════════════
def r7_6_recent_zoom(p):
    L51 = get_base()
    p("=" * 80)
    p("R7-6: 2025-2026 近期高金价环境验证")
    p("=" * 80)

    p("\n--- Part A: 季度分解 ---")
    quarters = [
        ("2025-Q1", "2025-01-01", "2025-04-01"),
        ("2025-Q2", "2025-04-01", "2025-07-01"),
        ("2025-Q3", "2025-07-01", "2025-10-01"),
        ("2025-Q4", "2025-10-01", "2026-01-01"),
        ("2026-Q1", "2026-01-01", "2026-04-01"),
    ]
    tasks = [(name, L51, 0.30, start, end) for name, start, end in quarters]
    results = run_pool(tasks)
    p(f"  {'Quarter':<12} {'N':>5} {'Sharpe':>8} {'PnL':>14} {'WR%':>8} {'MaxDD':>10}")
    for r in sorted(results, key=lambda x: x[0]):
        p(f"  {r[0]:<12} {r[1]:>5} {r[2]:>8.2f} {fmt(r[3]):>14} {r[4]:>8.1f}% {fmt(r[6]):>10}")

    p("\n--- Part B: L5 vs L5.1 对比 (2025-2026) ---")
    L5_old = {**L51, "sl_atr_mult": 4.5, "max_positions": 2}
    tasks = [
        ("L5_2025",   L5_old, 0.30, "2025-01-01", "2026-01-01"),
        ("L5.1_2025", L51,    0.30, "2025-01-01", "2026-01-01"),
        ("L5_2026Q1", L5_old, 0.30, "2026-01-01", "2026-04-01"),
        ("L5.1_2026Q1", L51,  0.30, "2026-01-01", "2026-04-01"),
    ]
    results = run_pool(tasks)
    p(f"\n  {'Config':<15} {'N':>5} {'Sharpe':>8} {'PnL':>14} {'WR%':>8} {'MaxDD':>10}")
    for r in sorted(results, key=lambda x: x[0]):
        p(f"  {r[0]:<15} {r[1]:>5} {r[2]:>8.2f} {fmt(r[3]):>14} {r[4]:>8.1f}% {fmt(r[6]):>10}")

    p("\n--- Part C: 出场分布 (2025-2026) ---")
    detail_tasks = [("L51_detail", L51, 0.30, "2025-01-01", "2026-04-01")]
    detail_results = run_pool(detail_tasks, func=_run_one_trades)
    if detail_results and len(detail_results[0]) > 7:
        trades = detail_results[0][7]
        exit_counter = Counter()
        exit_pnl = {}
        for pnl, reason, bars, strat, etime, direction, mfe in trades:
            base_reason = reason.split(':')[0] if ':' in reason else reason
            base_reason = base_reason[:25]
            exit_counter[base_reason] += 1
            if base_reason not in exit_pnl:
                exit_pnl[base_reason] = []
            exit_pnl[base_reason].append(pnl)
        p(f"\n  {'Exit Reason':<28} {'Count':>6} {'Total PnL':>12} {'Avg PnL':>10}")
        for reason, count in exit_counter.most_common(10):
            total = sum(exit_pnl[reason])
            avg = total / count
            p(f"  {reason:<28} {count:>6} {fmt(total):>12} {fmt(avg):>10}")


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("results/round7_results", exist_ok=True)

    phases = [
        (OUTPUT_DIR,       "r6_b4_interaction.txt",  "R6-B4: 参数交互效应",     r6_b4_interaction_grid),
        (OUTPUT_DIR,       "r6_b5_recent_zoom.txt",  "R6-B5: 近期放大镜",       r6_b5_recent_zoom),
        (OUTPUT_DIR,       "r6_b6_heatmap.txt",      "R6-B6: 月度热力图",       r6_b6_monthly_heatmap),
        ("round7_results", "r7_6_recent_zoom.txt",   "R7-6: 近期放大镜(修复)",  r7_6_recent_zoom),
    ]

    print(f"Local Remaining Experiments")
    print(f"Started: {datetime.now()}")
    print(f"MAX_WORKERS: {MAX_WORKERS}")
    print("=" * 60)

    for out_dir, fname, title, func in phases:
        fpath = os.path.join(out_dir, fname)
        t0 = time.time()
        print(f"\n>>> Starting {title} ...")
        try:
            lines = []
            lines.append(f"# {title}")
            lines.append(f"# Started: {datetime.now()}")
            lines.append(f"# Local run (6 workers)\n")

            def p(msg=""):
                lines.append(msg)
                print(msg)

            func(p)
            elapsed = (time.time() - t0) / 60
            lines.append(f"\n# Completed: {datetime.now()}")
            lines.append(f"# Elapsed: {elapsed:.1f} minutes")

            with open(fpath, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            print(f"\n  >>> {title}: DONE ({elapsed:.1f} min)")
        except Exception as e:
            elapsed = (time.time() - t0) / 60
            print(f"\n  >>> {title}: FAILED ({elapsed:.1f} min) - {e}")
            traceback.print_exc()
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n# FAILED: {e}\n{traceback.format_exc()}")

    print(f"\nAll done! {datetime.now()}")
