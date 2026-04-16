#!/usr/bin/env python3
"""
Round 7 — L5.1 验证 + 探索 (双服务器均可运行)
===============================================
R7-1: L5.1 全面基准 + K-Fold 确认
R7-2: Entry Gap 1h K-Fold 验证
R7-3: L6 候选在 L5.1 基础上的增量
R7-4: Monte Carlo 参数扰动 (L5.1)
R7-5: TP 倍数在新 SL=3.5 下的交互
R7-6: 2025-2026 近期数据放大镜
"""
import sys, os, io, time, traceback
import multiprocessing as mp
import numpy as np
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "results/round7_results"
MAX_WORKERS = 8

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

def run_pool(tasks, func=_run_one):
    with mp.Pool(MAX_WORKERS) as pool:
        return pool.map(func, tasks)

def get_base():
    from backtest.runner import LIVE_PARITY_KWARGS
    return {**LIVE_PARITY_KWARGS}

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-10"),
]

ULTRA2 = {
    'low': {'trail_act': 0.30, 'trail_dist': 0.06},
    'normal': {'trail_act': 0.20, 'trail_dist': 0.04},
    'high': {'trail_act': 0.08, 'trail_dist': 0.01},
}


def run_kfold(base_kw, var_kw, spread=0.30, prefix=""):
    tasks = []
    for fname, start, end in FOLDS:
        tasks.append((f"{prefix}Base_{fname}", base_kw, spread, start, end))
        tasks.append((f"{prefix}Var_{fname}", var_kw, spread, start, end))
    results = run_pool(tasks)
    base_r = [r for r in results if 'Base_' in r[0]]
    var_r = [r for r in results if 'Var_' in r[0]]
    return base_r, var_r

def print_kfold(p, base_r, var_r, bl="Baseline", vl="Variant"):
    p(f"\n  {'Fold':<8} {bl+' Sharpe':>18} {vl+' Sharpe':>18} {'Delta':>10} {'Pass?':>6}")
    p(f"  {'-'*65}")
    wins = 0
    for b, v in zip(base_r, var_r):
        delta = v[2] - b[2]
        passed = delta > -0.5
        wins += int(passed)
        mark = "YES" if passed else " no"
        p(f"  {b[0].split('_')[-1]:<8} {b[2]:>18.2f} {v[2]:>18.2f} {delta:>+10.2f} {mark:>6}")
    p(f"\n  K-Fold: {wins}/{len(base_r)} PASS")
    return wins


# ════════════════════════════════════════════════════════════════
# R7-1: L5.1 全面基准
# ════════════════════════════════════════════════════════════════
def r7_1_baseline(p):
    L51 = get_base()
    p("=" * 80)
    p("R7-1: L5.1 全面基准确认")
    p("=" * 80)

    p("\n--- Part A: 全样本 多点差 ---")
    tasks = []
    for sp in [0.00, 0.20, 0.30, 0.40, 0.50]:
        tasks.append((f"L51_sp{sp:.2f}", L51, sp, None, None))
    results = run_pool(tasks)
    p(f"  {'Config':<20} {'N':>5} {'Sharpe':>8} {'PnL':>14} {'WR%':>8} {'MaxDD':>10}")
    for r in sorted(results, key=lambda x: x[0]):
        p(f"  {r[0]:<20} {r[1]:>5} {r[2]:>8.2f} {fmt(r[3]):>14} {r[4]:>8.1f}% {fmt(r[6]):>10}")

    p("\n--- Part B: 逐年 $0.30 ---")
    years = list(range(2015, 2027))
    tasks = [(f"Y{y}", L51, 0.30, f"{y}-01-01", f"{y+1}-01-01" if y < 2026 else "2026-12-31")
             for y in years]
    results = run_pool(tasks)
    for r in sorted(results, key=lambda x: x[0]):
        p(f"  {r[0]:<8} Sharpe={r[2]:>6.2f}  PnL={fmt(r[3]):>12}  N={r[1]:>5}")
    pos_years = sum(1 for r in results if r[3] > 0)
    p(f"\n  盈利年份: {pos_years}/{len(years)}")

    p("\n--- Part C: K-Fold vs L5 (验证叠加效果) ---")
    L5_old = {**L51, "sl_atr_mult": 4.5, "max_positions": 2}
    base_r, var_r = run_kfold(L5_old, L51, 0.30, "L51vL5_")
    print_kfold(p, base_r, var_r, "L5(old)", "L5.1")

    p("\n--- Part D: K-Fold $0.50 ---")
    base_r, var_r = run_kfold(L5_old, L51, 0.50, "L51vL5_50_")
    print_kfold(p, base_r, var_r, "L5(old)$0.50", "L5.1$0.50")

    p("\n--- Part E: Anchored Walk-Forward ---")
    wf_tasks = []
    for y in range(2016, 2027):
        train_end = f"{y}-01-01"
        test_end = f"{y+1}-01-01" if y < 2026 else "2026-12-31"
        wf_tasks.append((f"WF_{y}", L51, 0.30, train_end, test_end))
    wf_results = run_pool(wf_tasks)
    p(f"  {'Train':<20} {'Test':>8} {'N':>5} {'Sharpe':>8} {'PnL':>14} {'WR%':>8}")
    for r in sorted(wf_results, key=lambda x: x[0]):
        year = r[0].split('_')[1]
        p(f"  Train 2015-{int(year)-1}       {year:>8} {r[1]:>5} {r[2]:>8.2f} {fmt(r[3]):>14} {r[4]:>8.1f}%")
    wf_pos = sum(1 for r in wf_results if r[3] > 0)
    p(f"\n  盈利年份: {wf_pos}/{len(wf_results)}")


# ════════════════════════════════════════════════════════════════
# R7-2: Entry Gap 1h K-Fold 验证
# ════════════════════════════════════════════════════════════════
def r7_2_entry_gap(p):
    L51 = get_base()
    p("=" * 80)
    p("R7-2: Entry Gap K-Fold 验证")
    p("=" * 80)

    p("\n--- Part A: Gap 全样本扫描 (L5.1 基线) ---")
    gaps = [0.0, 0.5, 1.0, 1.5, 2.0]
    tasks = [(f"Gap={g:.1f}h", {**L51, "min_entry_gap_hours": g}, 0.30, None, None)
             for g in gaps]
    results = run_pool(tasks)
    p(f"  {'Gap':<12} {'N':>5} {'Sharpe':>8} {'PnL':>14} {'WR%':>8}")
    for r in sorted(results, key=lambda x: float(x[0].split('=')[1].replace('h',''))):
        p(f"  {r[0]:<12} {r[1]:>5} {r[2]:>8.2f} {fmt(r[3]):>14} {r[4]:>8.1f}%")

    p("\n--- Part B: Gap=1.0h K-Fold vs L5.1 ---")
    gap1h = {**L51, "min_entry_gap_hours": 1.0}
    base_r, var_r = run_kfold(L51, gap1h, 0.30, "Gap1h_")
    wins = print_kfold(p, base_r, var_r, "L5.1(no gap)", "Gap=1h")

    if wins >= 5:
        p("\n--- Part C: Gap=1.0h K-Fold $0.50 ---")
        base_r, var_r = run_kfold(L51, gap1h, 0.50, "Gap1h50_")
        print_kfold(p, base_r, var_r, "L5.1$0.50", "Gap=1h$0.50")


# ════════════════════════════════════════════════════════════════
# R7-3: L6 候选在 L5.1 基础上的增量
# ════════════════════════════════════════════════════════════════
def r7_3_l6_on_l51(p):
    L51 = get_base()
    p("=" * 80)
    p("R7-3: L6 候选在 L5.1 基础上的增量")
    p("=" * 80)

    L6 = {**L51,
          "regime_config": ULTRA2,
          "trailing_activate_atr": ULTRA2['normal']['trail_act'],
          "trailing_distance_atr": ULTRA2['normal']['trail_dist']}

    p("\n--- Part A: 全样本 ---")
    tasks = [
        ("L5.1_sp030", L51, 0.30, None, None),
        ("L6_sp030",   L6,  0.30, None, None),
        ("L5.1_sp050", L51, 0.50, None, None),
        ("L6_sp050",   L6,  0.50, None, None),
    ]
    results = run_pool(tasks)
    p(f"  {'Config':<20} {'N':>5} {'Sharpe':>8} {'PnL':>14} {'WR%':>8} {'MaxDD':>10}")
    for r in sorted(results, key=lambda x: x[0]):
        p(f"  {r[0]:<20} {r[1]:>5} {r[2]:>8.2f} {fmt(r[3]):>14} {r[4]:>8.1f}% {fmt(r[6]):>10}")

    p("\n--- Part B: L6 vs L5.1 K-Fold $0.30 ---")
    base_r, var_r = run_kfold(L51, L6, 0.30, "L6vL51_")
    wins = print_kfold(p, base_r, var_r, "L5.1", "L6")

    p("\n--- Part C: L6 vs L5.1 K-Fold $0.50 ---")
    base_r, var_r = run_kfold(L51, L6, 0.50, "L6vL51_50_")
    print_kfold(p, base_r, var_r, "L5.1$0.50", "L6$0.50")

    p("\n--- Part D: Walk-Forward L6 ---")
    wf_tasks = []
    for y in range(2016, 2027):
        test_end = f"{y+1}-01-01" if y < 2026 else "2026-12-31"
        wf_tasks.append((f"L6_WF_{y}", L6, 0.30, f"{y}-01-01", test_end))
    wf_results = run_pool(wf_tasks)
    p(f"  {'Test Year':<12} {'N':>5} {'Sharpe':>8} {'PnL':>14}")
    for r in sorted(wf_results, key=lambda x: x[0]):
        year = r[0].split('_')[-1]
        p(f"  {year:<12} {r[1]:>5} {r[2]:>8.2f} {fmt(r[3]):>14}")
    wf_pos = sum(1 for r in wf_results if r[3] > 0)
    p(f"\n  盈利年份: {wf_pos}/{len(wf_results)}")


# ════════════════════════════════════════════════════════════════
# R7-4: Monte Carlo 参数扰动 (L5.1)
# ════════════════════════════════════════════════════════════════
def r7_4_monte_carlo(p):
    L51 = get_base()
    p("=" * 80)
    p("R7-4: Monte Carlo 参数扰动 (L5.1, +/-15%)")
    p("=" * 80)

    n_runs = 80
    batch_size = 20
    all_results = []

    for batch_i in range(0, n_runs, batch_size):
        batch_end = min(batch_i + batch_size, n_runs)
        p(f"\n  Batch {batch_i//batch_size+1}/{(n_runs+batch_size-1)//batch_size} ({batch_i}-{batch_end-1})...")
        tasks = []
        for i in range(batch_i, batch_end):
            np.random.seed(42 + i)
            kw = {**L51}
            kw['sl_atr_mult'] = round(L51['sl_atr_mult'] * (1 + np.random.uniform(-0.15, 0.15)), 2)
            kw['tp_atr_mult'] = round(L51['tp_atr_mult'] * (1 + np.random.uniform(-0.15, 0.15)), 2)
            kw['choppy_threshold'] = round(L51['choppy_threshold'] * (1 + np.random.uniform(-0.15, 0.15)), 2)
            kw['keltner_adx_threshold'] = max(10, int(L51['keltner_adx_threshold'] * (1 + np.random.uniform(-0.15, 0.15))))
            rc = L51['regime_config']
            new_rc = {}
            for regime_name, rc_vals in rc.items():
                noise_act = 1 + np.random.uniform(-0.15, 0.15)
                noise_dist = 1 + np.random.uniform(-0.15, 0.15)
                new_rc[regime_name] = {
                    'trail_act': round(rc_vals['trail_act'] * noise_act, 3),
                    'trail_dist': round(rc_vals['trail_dist'] * noise_dist, 3),
                }
            kw['regime_config'] = new_rc
            kw['trailing_activate_atr'] = new_rc['normal']['trail_act']
            kw['trailing_distance_atr'] = new_rc['normal']['trail_dist']
            tasks.append((f"MC_{i:03d}", kw, 0.30, None, None))
        batch_results = run_pool(tasks)
        all_results.extend(batch_results)
        p(f"    done ({len(all_results)}/{n_runs})")

    sharpes = [r[2] for r in all_results]
    pnls = [r[3] for r in all_results]
    p(f"\n--- {n_runs} 次参数扰动结果 ---")
    p(f"  Sharpe: mean={np.mean(sharpes):.2f}, std={np.std(sharpes):.2f}, min={np.min(sharpes):.2f}, max={np.max(sharpes):.2f}")
    p(f"  PnL:    mean={fmt(np.mean(pnls))}, std={fmt(np.std(pnls))}, min={fmt(np.min(pnls))}, max={fmt(np.max(pnls))}")
    p(f"  盈利组合: {sum(1 for x in pnls if x>0)}/{n_runs}")
    p(f"  Sharpe>2: {sum(1 for x in sharpes if x>2)}/{n_runs}")
    p(f"  Sharpe>3: {sum(1 for x in sharpes if x>3)}/{n_runs}")
    p(f"  Sharpe>4: {sum(1 for x in sharpes if x>4)}/{n_runs}")

    p(f"\n--- Sharpe 分布 ---")
    bins = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,100)]
    for lo, hi in bins:
        cnt = sum(1 for x in sharpes if lo <= x < hi)
        bar = '#' * cnt
        p(f"  [{lo:>2}-{hi:>3}): {cnt:>3} {bar}")


# ════════════════════════════════════════════════════════════════
# R7-5: TP 倍数在新 SL=3.5 下的交互
# ════════════════════════════════════════════════════════════════
def r7_5_tp_interaction(p):
    L51 = get_base()
    p("=" * 80)
    p("R7-5: TP 倍数在 SL=3.5 下的交互效应")
    p("=" * 80)

    p("\n--- Part A: SL x TP 二维扫描 ---")
    sl_vals = [3.0, 3.5, 4.0, 4.5]
    tp_vals = [5.0, 6.0, 7.0, 8.0, 10.0, 999.0]
    tasks = []
    for sl in sl_vals:
        for tp in tp_vals:
            tp_label = "OFF" if tp > 100 else f"{tp:.0f}"
            tasks.append((f"SL{sl:.1f}_TP{tp_label}", {**L51, "sl_atr_mult": sl, "tp_atr_mult": tp}, 0.30, None, None))
    results = run_pool(tasks)
    result_map = {r[0]: r for r in results}

    col_header = "SL/TP"
    header = f"  {col_header:<14}"
    for tp in tp_vals:
        tp_label = "OFF" if tp > 100 else f"{tp:.0f}"
        header += f" {tp_label:>8}"
    p(header)
    for sl in sl_vals:
        marker = " <-" if abs(sl - 3.5) < 0.01 else ""
        row_label = f"SL={sl:.1f}{marker}"
        row = f"  {row_label:<14}"
        for tp in tp_vals:
            tp_label = "OFF" if tp > 100 else f"{tp:.0f}"
            key = f"SL{sl:.1f}_TP{tp_label}"
            r = result_map.get(key)
            if r:
                row += f" {r[2]:>8.2f}"
            else:
                row += f" {'N/A':>8}"
        p(row)

    p("\n--- Part B: 最优 TP K-Fold 验证 ---")
    best = max(results, key=lambda x: x[2] if 'SL3.5' in x[0] else -999)
    p(f"  最优组合: {best[0]} (Sharpe={best[2]:.2f})")
    best_tp = 999.0
    for tp in tp_vals:
        tp_label = "OFF" if tp > 100 else f"{tp:.0f}"
        if tp_label in best[0].split('_')[1]:
            best_tp = tp
            break
    if abs(best_tp - 8.0) > 0.1:
        var_kw = {**L51, "tp_atr_mult": best_tp}
        base_r, var_r = run_kfold(L51, var_kw, 0.30, "TP_")
        tp_str = "OFF" if best_tp > 100 else f"{best_tp:.0f}"
        print_kfold(p, base_r, var_r, "TP=8(current)", f"TP={tp_str}")
    else:
        p("  当前 TP=8.0 已是最优或接近最优，无需 K-Fold")


# ════════════════════════════════════════════════════════════════
# R7-6: 2025-2026 近期数据放大镜
# ════════════════════════════════════════════════════════════════
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
        ("2026-Apr", "2026-04-01", "2026-04-13"),
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
        ("L5_2026",   L5_old, 0.30, "2026-01-01", "2026-12-31"),
        ("L5.1_2026", L51,    0.30, "2026-01-01", "2026-12-31"),
    ]
    results = run_pool(tasks)
    p(f"\n  {'Config':<15} {'N':>5} {'Sharpe':>8} {'PnL':>14} {'WR%':>8} {'MaxDD':>10}")
    for r in sorted(results, key=lambda x: x[0]):
        p(f"  {r[0]:<15} {r[1]:>5} {r[2]:>8.2f} {fmt(r[3]):>14} {r[4]:>8.1f}% {fmt(r[6]):>10}")

    p("\n--- Part C: 出场分布 (2025-2026 逐笔) ---")
    detail_tasks = [("L51_detail", L51, 0.30, "2025-01-01", "2026-12-31")]
    detail_results = run_pool(detail_tasks, func=_run_one_trades)
    if detail_results and len(detail_results[0]) > 7:
        trades = detail_results[0][7]
        from collections import Counter
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


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    master_log = os.path.join(OUTPUT_DIR, "00_master_log.txt")

    phases = [
        ("r7_1_baseline.txt",     "R7-1: L5.1 全面基准",        r7_1_baseline),
        ("r7_2_entry_gap.txt",    "R7-2: Entry Gap 验证",        r7_2_entry_gap),
        ("r7_3_l6_on_l51.txt",    "R7-3: L6 候选增量",           r7_3_l6_on_l51),
        ("r7_4_monte_carlo.txt",  "R7-4: Monte Carlo",          r7_4_monte_carlo),
        ("r7_5_tp_interact.txt",  "R7-5: TP交互效应",            r7_5_tp_interaction),
        ("r7_6_recent_zoom.txt",  "R7-6: 近期放大镜",            r7_6_recent_zoom),
    ]

    with open(master_log, "w", encoding="utf-8") as mf:
        mf.write(f"Round 7 (L5.1 Validation & Exploration)\n")
        mf.write(f"Started: {datetime.now()}\n")
        mf.write(f"MAX_WORKERS: {MAX_WORKERS}\n")
        mf.write("=" * 60 + "\n\n")

    for fname, phase_name, func in phases:
        fpath = os.path.join(OUTPUT_DIR, fname)
        t0 = time.time()
        print(f"\n>>> Starting {phase_name} ...")
        try:
            lines = []
            lines.append(f"# {phase_name}")
            lines.append(f"# Started: {datetime.now()}")

            def p(msg=""):
                lines.append(msg)
                print(msg)

            func(p)
            elapsed = (time.time() - t0) / 60
            lines.append(f"\n# Completed: {datetime.now()}")
            lines.append(f"# Elapsed: {elapsed:.1f} minutes")

            with open(fpath, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            with open(master_log, "a", encoding="utf-8") as mf:
                mf.write(f"  {phase_name}: DONE ({elapsed:.1f} min)\n")

            print(f"<<< {phase_name} done in {elapsed:.1f} min")

        except Exception as e:
            elapsed = (time.time() - t0) / 60
            tb = traceback.format_exc()
            print(f"!!! {phase_name} FAILED: {e}")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(f"# {phase_name} FAILED\n{tb}\n")
            with open(master_log, "a", encoding="utf-8") as mf:
                mf.write(f"  {phase_name}: FAILED ({elapsed:.1f} min) - {e}\n")

    with open(master_log, "a", encoding="utf-8") as mf:
        mf.write(f"\nRound 7 Finished: {datetime.now()}\n")
    print(f"\n{'='*60}")
    print(f"Round 7 ALL DONE!")
