#!/usr/bin/env python3
"""
Round 13 — "Alpha 淬炼" (Alpha Refinement)
=============================================
目标: 在入场信号已饱和、出场已最优的基础上，开辟全新维度寻找边际 alpha
预计总耗时: ~20小时 (服务器 25核)

=== Phase A: KC 参数空间精细扫描 (~3h) ===
R13-A1: KC EMA 精细扫描 (15/18/20/22/25/28/30/35) × $0.30/$0.50
R13-A2: KC Mult 精细扫描 (0.8/1.0/1.2/1.4/1.6/1.8/2.0) × $0.30/$0.50
R13-A3: 最优 EMA×Mult 热力图 (8×7=56 组合)
R13-A4: 最优组合 K-Fold 6折验证

=== Phase B: Breakeven Stop 深度研究 (~3h) ===
R13-B1: BE 阈值扫描 (0.3/0.5/0.7/1.0/1.5/2.0 ATR) + L5.1 vs L6
R13-B2: 最优 BE K-Fold 验证
R13-B3: BE 对出场画像的影响分析 (Trailing/Timeout/SL 占比变化)

=== Phase C: 多速度 KC 信号 (~4h) ===
R13-C1: 快KC(15/0.8)+慢KC(35/1.6) 三模式 (union/intersect/fast_confirmed)
R13-C2: 不同快慢 KC 参数组合扫描
R13-C3: 最优 Dual KC K-Fold 验证

=== Phase D: 自适应 MA 中轨 (~3h) ===
R13-D1: HMA 替换 EMA 扫描: HMA 15/20/25/30
R13-D2: KAMA 替换 EMA 扫描: KAMA 10/15/20/25/30
R13-D3: 最优自适应 MA K-Fold 验证

=== Phase E: 滚动窗口自适应参数 (~2h) ===
R13-E1: 2年滚动最优 trail vs 固定参数
R13-E2: 3年/5年窗口对比

=== Phase F: Purged Walk-Forward 验证 (~2h) ===
R13-F1: L5.1 Purged WF (embargo=20 bars, purge 跨 fold 持仓)
R13-F2: L6 Purged WF
R13-F3: L5.1 vs L6 Purged WF 差异

=== Phase G: 组合 + L7 候选 (~3h) ===
R13-G1: 通过验证的改进叠加
R13-G2: L7 候选 Monte Carlo 100 次
R13-G3: L7 vs L6 vs L5.1 全面对比 + Walk-Forward
"""
import sys, os, io, time, traceback, json
import multiprocessing as mp
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "results/round13_results"
MAX_WORKERS = 22

def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def _run_one(args):
    label, kw, spread, start, end = args
    from backtest.runner import DataBundle, run_variant
    kc_ema = kw.pop('_kc_ema', 25)
    kc_mult = kw.pop('_kc_mult', 1.2)
    kc_ma = kw.pop('_kc_ma_type', 'ema')
    data = DataBundle.load_custom(kc_ema=kc_ema, kc_mult=kc_mult, kc_ma_type=kc_ma)
    if start and end:
        data = data.slice(start, end)
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kw)
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'], s.get('max_dd_pct', 0),
            s.get('keltner_n', 0), s.get('keltner_pnl', 0),
            s.get('breakeven_triggered', 0),
            s.get('dual_kc_filtered', 0), s.get('dual_kc_added', 0),
            s.get('skipped_gsr', 0),
            s.get('year_pnl', {}))


def _run_one_trades(args):
    label, kw, spread, start, end = args
    from backtest.runner import DataBundle, run_variant
    kc_ema = kw.pop('_kc_ema', 25)
    kc_mult = kw.pop('_kc_mult', 1.2)
    kc_ma = kw.pop('_kc_ma_type', 'ema')
    data = DataBundle.load_custom(kc_ema=kc_ema, kc_mult=kc_mult, kc_ma_type=kc_ma)
    if start and end:
        data = data.slice(start, end)
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kw)
    trades = s.get('_trades', [])
    td = [(round(t.pnl, 2), t.exit_reason or '', t.bars_held, t.strategy or '',
           str(t.entry_time)[:16], t.direction or '')
          for t in trades[:30000]]
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'], td,
            s.get('breakeven_triggered', 0))


def run_pool(tasks, func=_run_one):
    with mp.Pool(min(MAX_WORKERS, len(tasks))) as pool:
        return pool.map(func, tasks)


def get_base():
    from backtest.runner import LIVE_PARITY_KWARGS
    return {**LIVE_PARITY_KWARGS}


def get_l6():
    base = get_base()
    base['regime_config'] = {
        'low':    {'trail_act': 0.30, 'trail_dist': 0.06},
        'normal': {'trail_act': 0.20, 'trail_dist': 0.04},
        'high':   {'trail_act': 0.08, 'trail_dist': 0.01},
    }
    return base


FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-01"),
]

YEARS = [(str(y), f"{y}-01-01", f"{y+1}-01-01") for y in range(2015, 2026)]


def write_table(f, results, extra_cols=None):
    cols = f"{'Label':<40} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} {'MaxDD':>10}"
    if extra_cols:
        cols += f" {extra_cols}"
    f.write(cols + "\n")
    f.write("-" * len(cols) + "\n")
    for r in results:
        line = f"{r[0]:<40} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1f}% {fmt(r[6]):>10}"
        f.write(line + "\n")


# ═══════════════════════════════════════════════════════════════
# Phase A: KC 参数空间精细扫描
# ═══════════════════════════════════════════════════════════════

def run_r13_a1(out):
    """R13-A1: KC EMA 精细扫描"""
    print("\n" + "="*70)
    print("R13-A1: KC EMA Grid Search")
    print("="*70)

    base = get_base()
    ema_values = [15, 18, 20, 22, 25, 28, 30, 35]
    tasks = []
    for ema in ema_values:
        for sp in [0.30, 0.50]:
            kw = {**base, '_kc_ema': ema, '_kc_mult': 1.2}
            tasks.append((f"EMA{ema}_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R13-A1_ema_scan.txt", 'w') as f:
        f.write("KC EMA Grid Search\n" + "="*80 + "\n\n")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread ${sp} ---\n")
            sub = [r for r in results if f"sp{sp}" in r[0]]
            sub.sort(key=lambda x: x[2], reverse=True)
            write_table(f, sub)
    for r in sorted(results, key=lambda x: x[2], reverse=True)[:5]:
        print(f"  {r[0]:<25} N={r[1]:>5} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


def run_r13_a2(out):
    """R13-A2: KC Mult 精细扫描"""
    print("\n" + "="*70)
    print("R13-A2: KC Mult Grid Search")
    print("="*70)

    base = get_base()
    mult_values = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    tasks = []
    for mult in mult_values:
        for sp in [0.30, 0.50]:
            kw = {**base, '_kc_ema': 25, '_kc_mult': mult}
            tasks.append((f"Mult{mult}_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R13-A2_mult_scan.txt", 'w') as f:
        f.write("KC Mult Grid Search\n" + "="*80 + "\n\n")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread ${sp} ---\n")
            sub = [r for r in results if f"sp{sp}" in r[0]]
            sub.sort(key=lambda x: x[2], reverse=True)
            write_table(f, sub)
    for r in sorted(results, key=lambda x: x[2], reverse=True)[:5]:
        print(f"  {r[0]:<25} N={r[1]:>5} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


def run_r13_a3(out):
    """R13-A3: EMA × Mult 热力图"""
    print("\n" + "="*70)
    print("R13-A3: EMA x Mult Heatmap")
    print("="*70)

    base = get_base()
    ema_values = [15, 18, 20, 22, 25, 28, 30, 35]
    mult_values = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    tasks = []
    for ema in ema_values:
        for mult in mult_values:
            kw = {**base, '_kc_ema': ema, '_kc_mult': mult}
            tasks.append((f"E{ema}_M{mult}", kw, 0.30, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R13-A3_heatmap.txt", 'w') as f:
        f.write("EMA x Mult Heatmap (Sharpe, spread=$0.30)\n" + "="*80 + "\n\n")
        rmap = {r[0]: r for r in results}
        header_label = "EMA" + chr(92) + "Mult"
        f.write(f"{header_label:<10}")
        for mult in mult_values:
            f.write(f" {mult:>8}")
        f.write("\n")
        for ema in ema_values:
            f.write(f"{ema:<10}")
            for mult in mult_values:
                key = f"E{ema}_M{mult}"
                r = rmap.get(key)
                sh = r[2] if r else 0
                f.write(f" {sh:>8.2f}")
            f.write("\n")

        f.write("\n\nTop 10 by Sharpe:\n")
        top10 = sorted(results, key=lambda x: x[2], reverse=True)[:10]
        write_table(f, top10)

        f.write("\n\nPnL heatmap:\n")
        f.write(f"{header_label:<10}")
        for mult in mult_values:
            f.write(f" {mult:>10}")
        f.write("\n")
        for ema in ema_values:
            f.write(f"{ema:<10}")
            for mult in mult_values:
                key = f"E{ema}_M{mult}"
                r = rmap.get(key)
                pnl = r[3] if r else 0
                f.write(f" {fmt(pnl):>10}")
            f.write("\n")

    top3 = sorted(results, key=lambda x: x[2], reverse=True)[:3]
    for r in top3:
        print(f"  {r[0]:<25} N={r[1]:>5} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")
    return top3


def run_r13_a4(out, top_configs=None):
    """R13-A4: K-Fold 验证最优 KC 参数"""
    print("\n" + "="*70)
    print("R13-A4: K-Fold Validation of Top KC Params")
    print("="*70)

    if not top_configs:
        top_configs = [("E25_M1.2", 25, 1.2), ("E22_M1.2", 22, 1.2), ("E25_M1.0", 25, 1.0)]

    base = get_base()
    all_results = {}
    for cfg_label, ema, mult in top_configs:
        tasks = []
        for sp in [0.30, 0.50]:
            for fold_name, s, e in FOLDS:
                kw = {**base, '_kc_ema': ema, '_kc_mult': mult}
                tasks.append((f"{cfg_label}_{fold_name}_sp{sp}", kw, sp, s, e))
            kw = {**base, '_kc_ema': ema, '_kc_mult': mult}
            tasks.append((f"{cfg_label}_Full_sp{sp}", kw, sp, None, None))

        results = run_pool(tasks)
        all_results[cfg_label] = results

    baseline_tasks = []
    for sp in [0.30, 0.50]:
        for fold_name, s, e in FOLDS:
            kw = {**base, '_kc_ema': 25, '_kc_mult': 1.2}
            baseline_tasks.append((f"Baseline_{fold_name}_sp{sp}", kw, sp, s, e))
    baseline_results = run_pool(baseline_tasks)

    with open(f"{out}/R13-A4_kfold.txt", 'w') as f:
        f.write("K-Fold Validation of Top KC Params\n" + "="*80 + "\n\n")

        for sp in [0.30, 0.50]:
            f.write(f"\n=== Spread ${sp} ===\n")
            base_folds = {r[0].split('_')[1]: r[2] for r in baseline_results if f"sp{sp}" in r[0]}

            for cfg_label, results in all_results.items():
                cfg_folds = {r[0].split('_')[1]: r[2]
                             for r in results if f"sp{sp}" in r[0] and "Full" not in r[0]}
                full = [r for r in results if f"Full_sp{sp}" in r[0]]

                wins = sum(1 for fn in cfg_folds if cfg_folds.get(fn, 0) > base_folds.get(fn, 0))
                f.write(f"\n{cfg_label}: K-Fold {wins}/6 {'PASS' if wins >= 5 else 'FAIL'}\n")
                for fn in sorted(cfg_folds.keys()):
                    delta = cfg_folds[fn] - base_folds.get(fn, 0)
                    f.write(f"  {fn}: Sharpe={cfg_folds[fn]:>6.2f} (delta={delta:>+.2f})\n")
                if full:
                    f.write(f"  Full: Sharpe={full[0][2]:>6.2f} PnL={fmt(full[0][3])}\n")

    for cfg_label, results in all_results.items():
        full_30 = [r for r in results if "Full_sp0.3" in r[0]]
        if full_30:
            print(f"  {cfg_label}: Sharpe={full_30[0][2]:.2f}")


# ═══════════════════════════════════════════════════════════════
# Phase B: Breakeven Stop 深度研究
# ═══════════════════════════════════════════════════════════════

def run_r13_b1(out):
    """R13-B1: BE 阈值扫描"""
    print("\n" + "="*70)
    print("R13-B1: Breakeven Stop Threshold Scan")
    print("="*70)

    be_values = [0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    tasks = []
    for be in be_values:
        for config_name, get_fn in [("L5.1", get_base), ("L6", get_l6)]:
            kw = get_fn()
            kw['breakeven_after_atr'] = be
            for sp in [0.30, 0.50]:
                label = f"{config_name}_BE{be}_sp{sp}"
                tasks.append((label, {**kw}, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R13-B1_breakeven.txt", 'w') as f:
        f.write("Breakeven Stop Threshold Scan\n" + "="*80 + "\n\n")
        for config_name in ["L5.1", "L6"]:
            for sp in [0.30, 0.50]:
                f.write(f"\n--- {config_name} Spread ${sp} ---\n")
                sub = [r for r in results if r[0].startswith(config_name) and f"sp{sp}" in r[0]]
                sub.sort(key=lambda x: float(x[0].split('_BE')[1].split('_')[0]))
                write_table(f, sub, "BE_triggered")
                for r in sub:
                    f.write(f"  BE triggered: {r[10]}\n")

    for r in sorted(results, key=lambda x: x[2], reverse=True)[:5]:
        print(f"  {r[0]:<35} Sharpe={r[2]:>6.2f} BE_trig={r[10]}")


def run_r13_b2(out, best_be=None):
    """R13-B2: BE K-Fold 验证"""
    print("\n" + "="*70)
    print("R13-B2: Breakeven K-Fold Validation")
    print("="*70)

    if best_be is None:
        best_be = [0.5, 1.0]

    tasks = []
    for be in best_be:
        for config_name, get_fn in [("L5.1", get_base), ("L6", get_l6)]:
            for sp in [0.30, 0.50]:
                for fold_name, s, e in FOLDS:
                    kw = get_fn()
                    kw['breakeven_after_atr'] = be
                    tasks.append((f"{config_name}_BE{be}_{fold_name}_sp{sp}", {**kw}, sp, s, e))
                kw_full = get_fn()
                kw_full['breakeven_after_atr'] = be
                tasks.append((f"{config_name}_BE{be}_Full_sp{sp}", {**kw_full}, sp, None, None))

    base_tasks = []
    for config_name, get_fn in [("L5.1", get_base), ("L6", get_l6)]:
        for sp in [0.30, 0.50]:
            for fold_name, s, e in FOLDS:
                tasks.append((f"{config_name}_BE0_{fold_name}_sp{sp}", {**get_fn()}, sp, s, e))

    results = run_pool(tasks + base_tasks)

    with open(f"{out}/R13-B2_be_kfold.txt", 'w') as f:
        f.write("Breakeven K-Fold Validation\n" + "="*80 + "\n\n")
        for config_name in ["L5.1", "L6"]:
            for sp in [0.30, 0.50]:
                base_folds = {}
                for r in results:
                    if r[0].startswith(f"{config_name}_BE0_") and f"sp{sp}" in r[0]:
                        fn = r[0].split('_')[2]
                        base_folds[fn] = r[2]

                for be in best_be:
                    cfg_folds = {}
                    for r in results:
                        if r[0].startswith(f"{config_name}_BE{be}_") and f"sp{sp}" in r[0] and "Full" not in r[0]:
                            fn = r[0].split('_')[2]
                            cfg_folds[fn] = r[2]
                    wins = sum(1 for fn in cfg_folds if cfg_folds.get(fn, 0) > base_folds.get(fn, 0))
                    f.write(f"\n{config_name} BE={be} sp${sp}: K-Fold {wins}/6 {'PASS' if wins >= 5 else 'FAIL'}\n")
                    for fn in sorted(cfg_folds.keys()):
                        delta = cfg_folds[fn] - base_folds.get(fn, 0)
                        f.write(f"  {fn}: Sharpe={cfg_folds[fn]:>6.2f} (delta={delta:>+.2f})\n")

    print("  K-Fold results written")


def run_r13_b3(out):
    """R13-B3: BE 对出场画像的影响"""
    print("\n" + "="*70)
    print("R13-B3: Breakeven Exit Profile Impact")
    print("="*70)

    tasks = []
    for be in [0.0, 0.5, 1.0]:
        kw = get_base()
        kw['breakeven_after_atr'] = be
        tasks.append((f"L5.1_BE{be}", {**kw}, 0.30, None, None))
        kw6 = get_l6()
        kw6['breakeven_after_atr'] = be
        tasks.append((f"L6_BE{be}", {**kw6}, 0.30, None, None))

    results = run_pool(tasks, func=_run_one_trades)

    with open(f"{out}/R13-B3_exit_profile.txt", 'w') as f:
        f.write("Breakeven Exit Profile Impact\n" + "="*80 + "\n\n")
        for r in results:
            trades = r[7]
            f.write(f"\n--- {r[0]} (N={r[1]}, Sharpe={r[2]:.2f}, BE_trig={r[8]}) ---\n")
            exit_groups = defaultdict(list)
            for pnl, reason, bars, strat, etime, dirn in trades:
                exit_groups[reason].append((pnl, bars))
            f.write(f"{'Exit':<20} {'N':>6} {'TotalPnL':>12} {'AvgPnL':>10} {'WR':>7} {'AvgBars':>8}\n")
            for reason in sorted(exit_groups.keys()):
                grp = exit_groups[reason]
                n = len(grp)
                total = sum(p for p, _ in grp)
                avg = total / n if n else 0
                wr = sum(1 for p, _ in grp if p > 0) / n * 100 if n else 0
                avg_bars = sum(b for _, b in grp) / n if n else 0
                f.write(f"{reason:<20} {n:>6} {fmt(total):>12} {avg:>10.2f} {wr:>6.1f}% {avg_bars:>8.1f}\n")

    print("  Exit profile written")


# ═══════════════════════════════════════════════════════════════
# Phase C: 多速度 KC 信号
# ═══════════════════════════════════════════════════════════════

def run_r13_c1(out):
    """R13-C1: Dual KC 三模式测试"""
    print("\n" + "="*70)
    print("R13-C1: Dual KC Mode Test")
    print("="*70)

    base = get_base()
    tasks = [
        ("Baseline", {**base}, 0.30, None, None),
    ]
    for mode in ["union", "intersect", "fast_confirmed"]:
        kw = {**base, 'dual_kc_mode': mode}
        tasks.append((f"DualKC_{mode}", kw, 0.30, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R13-C1_dual_kc.txt", 'w') as f:
        f.write("Dual KC Mode Test (fast=EMA15/M0.8, slow=EMA35/M1.6)\n" + "="*80 + "\n\n")
        write_table(f, results, "DualFiltered DualAdded")
        for r in results:
            f.write(f"  {r[0]}: dual_filtered={r[11]}, dual_added={r[12]}\n")

    for r in results:
        print(f"  {r[0]:<25} Sharpe={r[2]:>6.2f} filtered={r[11]} added={r[12]}")


def run_r13_c2(out):
    """R13-C2: Dual KC 参数扫描"""
    print("\n" + "="*70)
    print("R13-C2: Dual KC Parameter Scan")
    print("="*70)

    base = get_base()
    tasks = [("Baseline", {**base}, 0.30, None, None)]

    fast_configs = [(12, 0.6), (15, 0.8), (18, 1.0)]
    slow_configs = [(30, 1.4), (35, 1.6), (40, 1.8)]

    for mode in ["fast_confirmed", "intersect"]:
        for f_ema, f_mult in fast_configs:
            for s_ema, s_mult in slow_configs:
                kw = {**base,
                       'dual_kc_mode': mode,
                       'dual_kc_fast_ema': f_ema,
                       'dual_kc_fast_mult': f_mult,
                       'dual_kc_slow_ema': s_ema,
                       'dual_kc_slow_mult': s_mult}
                label = f"{mode}_F{f_ema}x{f_mult}_S{s_ema}x{s_mult}"
                tasks.append((label, kw, 0.30, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R13-C2_dual_params.txt", 'w') as f:
        f.write("Dual KC Parameter Scan\n" + "="*80 + "\n\n")
        results_sorted = sorted(results, key=lambda x: x[2], reverse=True)
        write_table(f, results_sorted)

    top3 = sorted(results, key=lambda x: x[2], reverse=True)[:3]
    for r in top3:
        print(f"  {r[0]:<50} Sharpe={r[2]:>6.2f}")


def run_r13_c3(out, best_dual=None):
    """R13-C3: Dual KC K-Fold"""
    print("\n" + "="*70)
    print("R13-C3: Dual KC K-Fold Validation")
    print("="*70)

    if best_dual is None:
        best_dual = [
            ("fast_confirmed", 15, 0.8, 35, 1.6),
        ]

    base = get_base()
    tasks = []
    for sp in [0.30, 0.50]:
        for fold_name, s, e in FOLDS:
            tasks.append((f"Base_{fold_name}_sp{sp}", {**base}, sp, s, e))

        for mode, f_ema, f_mult, s_ema, s_mult in best_dual:
            kw = {**base,
                   'dual_kc_mode': mode,
                   'dual_kc_fast_ema': f_ema,
                   'dual_kc_fast_mult': f_mult,
                   'dual_kc_slow_ema': s_ema,
                   'dual_kc_slow_mult': s_mult}
            label_prefix = f"{mode}_F{f_ema}_S{s_ema}"
            for fold_name, s, e in FOLDS:
                tasks.append((f"{label_prefix}_{fold_name}_sp{sp}", {**kw}, sp, s, e))

    results = run_pool(tasks)

    with open(f"{out}/R13-C3_dual_kfold.txt", 'w') as f:
        f.write("Dual KC K-Fold Validation\n" + "="*80 + "\n\n")
        for sp in [0.30, 0.50]:
            base_folds = {r[0].split('_')[1]: r[2] for r in results
                          if r[0].startswith("Base_") and f"sp{sp}" in r[0]}
            for mode, f_ema, f_mult, s_ema, s_mult in best_dual:
                prefix = f"{mode}_F{f_ema}_S{s_ema}"
                cfg_folds = {}
                for r in results:
                    if r[0].startswith(prefix) and f"sp{sp}" in r[0]:
                        fn = r[0].replace(f"{prefix}_", "").replace(f"_sp{sp}", "")
                        cfg_folds[fn] = r[2]
                wins = sum(1 for fn in cfg_folds if cfg_folds.get(fn, 0) > base_folds.get(fn, 0))
                f.write(f"\n{prefix} sp${sp}: K-Fold {wins}/6 {'PASS' if wins >= 5 else 'FAIL'}\n")
                for fn in sorted(cfg_folds.keys()):
                    delta = cfg_folds[fn] - base_folds.get(fn, 0)
                    f.write(f"  {fn}: {cfg_folds[fn]:>6.2f} (delta={delta:>+.2f})\n")

    print("  Dual KC K-Fold written")


# ═══════════════════════════════════════════════════════════════
# Phase D: 自适应 MA 中轨
# ═══════════════════════════════════════════════════════════════

def run_r13_d1(out):
    """R13-D1: HMA 替换 EMA"""
    print("\n" + "="*70)
    print("R13-D1: HMA as KC Midline")
    print("="*70)

    base = get_base()
    tasks = [("EMA25_Baseline", {**base}, 0.30, None, None)]
    for period in [15, 20, 25, 30]:
        kw = {**base, '_kc_ema': period, '_kc_ma_type': 'hma'}
        tasks.append((f"HMA{period}", kw, 0.30, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R13-D1_hma.txt", 'w') as f:
        f.write("HMA as KC Midline\n" + "="*80 + "\n\n")
        write_table(f, sorted(results, key=lambda x: x[2], reverse=True))

    for r in sorted(results, key=lambda x: x[2], reverse=True)[:3]:
        print(f"  {r[0]:<25} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


def run_r13_d2(out):
    """R13-D2: KAMA 替换 EMA"""
    print("\n" + "="*70)
    print("R13-D2: KAMA as KC Midline")
    print("="*70)

    base = get_base()
    tasks = [("EMA25_Baseline", {**base}, 0.30, None, None)]
    for period in [10, 15, 20, 25, 30]:
        kw = {**base, '_kc_ema': period, '_kc_ma_type': 'kama'}
        tasks.append((f"KAMA{period}", kw, 0.30, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R13-D2_kama.txt", 'w') as f:
        f.write("KAMA as KC Midline\n" + "="*80 + "\n\n")
        write_table(f, sorted(results, key=lambda x: x[2], reverse=True))

    for r in sorted(results, key=lambda x: x[2], reverse=True)[:3]:
        print(f"  {r[0]:<25} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


def run_r13_d3(out, best_ma=None):
    """R13-D3: 自适应 MA K-Fold"""
    print("\n" + "="*70)
    print("R13-D3: Adaptive MA K-Fold Validation")
    print("="*70)

    if best_ma is None:
        best_ma = [("hma", 25), ("kama", 20)]

    base = get_base()
    tasks = []
    for sp in [0.30, 0.50]:
        for fold_name, s, e in FOLDS:
            tasks.append((f"EMA25_{fold_name}_sp{sp}", {**base}, sp, s, e))
        for ma_type, period in best_ma:
            kw = {**base, '_kc_ema': period, '_kc_ma_type': ma_type}
            for fold_name, s, e in FOLDS:
                tasks.append((f"{ma_type.upper()}{period}_{fold_name}_sp{sp}", {**kw}, sp, s, e))

    results = run_pool(tasks)

    with open(f"{out}/R13-D3_ma_kfold.txt", 'w') as f:
        f.write("Adaptive MA K-Fold Validation\n" + "="*80 + "\n\n")
        for sp in [0.30, 0.50]:
            base_folds = {r[0].split('_')[1]: r[2] for r in results
                          if r[0].startswith("EMA25_") and f"sp{sp}" in r[0]}
            for ma_type, period in best_ma:
                prefix = f"{ma_type.upper()}{period}"
                cfg_folds = {}
                for r in results:
                    if r[0].startswith(f"{prefix}_") and f"sp{sp}" in r[0]:
                        fn = r[0].replace(f"{prefix}_", "").replace(f"_sp{sp}", "")
                        cfg_folds[fn] = r[2]
                wins = sum(1 for fn in cfg_folds if cfg_folds.get(fn, 0) > base_folds.get(fn, 0))
                f.write(f"\n{prefix} sp${sp}: K-Fold {wins}/6 {'PASS' if wins >= 5 else 'FAIL'}\n")
                for fn in sorted(cfg_folds.keys()):
                    delta = cfg_folds[fn] - base_folds.get(fn, 0)
                    f.write(f"  {fn}: {cfg_folds[fn]:>6.2f} (delta={delta:>+.2f})\n")

    print("  Adaptive MA K-Fold written")


# ═══════════════════════════════════════════════════════════════
# Phase E: 滚动窗口自适应参数
# ═══════════════════════════════════════════════════════════════

def run_r13_e1(out):
    """R13-E1: 滚动窗口最优 trail"""
    print("\n" + "="*70)
    print("R13-E1: Rolling Window Adaptive Trail")
    print("="*70)

    from backtest.runner import DataBundle, run_variant
    data = DataBundle.load_custom()

    trail_grid = []
    for act in [0.15, 0.20, 0.28, 0.35, 0.40, 0.50]:
        for dist in [0.03, 0.05, 0.06, 0.08, 0.10, 0.15]:
            if dist < act:
                trail_grid.append((act, dist))

    windows = [
        ("2Y", 2),
        ("3Y", 3),
        ("5Y", 5),
    ]

    results_all = {}
    for win_name, win_years in windows:
        print(f"\n  Testing {win_name} rolling window...")
        yearly_results = []

        for test_year in range(2017, 2026):
            train_start = f"{test_year - win_years}-01-01"
            train_end = f"{test_year}-01-01"
            test_start = f"{test_year}-01-01"
            test_end = f"{test_year + 1}-01-01"

            train_data = data.slice(train_start, train_end)

            best_sharpe = -999
            best_trail = (0.28, 0.06)
            for act, dist in trail_grid:
                base = get_base()
                base['trailing_activate_atr'] = act
                base['trailing_distance_atr'] = dist
                base['regime_config']['normal']['trail_act'] = act
                base['regime_config']['normal']['trail_dist'] = dist
                s = run_variant(train_data, f"train_{act}_{dist}", verbose=False, spread_cost=0.30, **base)
                if s['sharpe'] > best_sharpe:
                    best_sharpe = s['sharpe']
                    best_trail = (act, dist)

            test_data = data.slice(test_start, test_end)
            base_test = get_base()
            base_test['trailing_activate_atr'] = best_trail[0]
            base_test['trailing_distance_atr'] = best_trail[1]
            base_test['regime_config']['normal']['trail_act'] = best_trail[0]
            base_test['regime_config']['normal']['trail_dist'] = best_trail[1]
            s_adaptive = run_variant(test_data, f"adaptive_{test_year}", verbose=False,
                                     spread_cost=0.30, **base_test)

            s_fixed = run_variant(test_data, f"fixed_{test_year}", verbose=False,
                                   spread_cost=0.30, **get_base())

            yearly_results.append({
                'year': test_year,
                'best_trail': best_trail,
                'train_sharpe': best_sharpe,
                'adaptive_sharpe': s_adaptive['sharpe'],
                'adaptive_pnl': s_adaptive['total_pnl'],
                'fixed_sharpe': s_fixed['sharpe'],
                'fixed_pnl': s_fixed['total_pnl'],
            })

        results_all[win_name] = yearly_results

    with open(f"{out}/R13-E1_rolling_trail.txt", 'w') as f:
        f.write("Rolling Window Adaptive Trail\n" + "="*80 + "\n\n")
        for win_name, yearly in results_all.items():
            f.write(f"\n=== {win_name} Rolling Window ===\n")
            f.write(f"{'Year':<6} {'BestTrail':>15} {'Train_Sh':>10} {'Adapt_Sh':>10} {'Fixed_Sh':>10} {'dSh':>8} {'Adapt_PnL':>12} {'Fixed_PnL':>12}\n")
            f.write("-"*90 + "\n")
            total_a, total_f = 0, 0
            for yr in yearly:
                bt = f"{yr['best_trail'][0]}/{yr['best_trail'][1]}"
                dsh = yr['adaptive_sharpe'] - yr['fixed_sharpe']
                total_a += yr['adaptive_pnl']
                total_f += yr['fixed_pnl']
                f.write(f"{yr['year']:<6} {bt:>15} {yr['train_sharpe']:>10.2f} "
                        f"{yr['adaptive_sharpe']:>10.2f} {yr['fixed_sharpe']:>10.2f} {dsh:>+7.2f} "
                        f"{fmt(yr['adaptive_pnl']):>12} {fmt(yr['fixed_pnl']):>12}\n")
            f.write(f"\nTotal: Adaptive={fmt(total_a)} Fixed={fmt(total_f)} Delta={fmt(total_a - total_f)}\n")

    print("  Rolling window analysis written")


# ═══════════════════════════════════════════════════════════════
# Phase F: Purged Walk-Forward
# ═══════════════════════════════════════════════════════════════

def run_r13_f1(out):
    """R13-F1/F2/F3: Purged Walk-Forward for L5.1 and L6"""
    print("\n" + "="*70)
    print("R13-F: Purged Walk-Forward Validation")
    print("="*70)

    from backtest.runner import DataBundle, run_variant
    data = DataBundle.load_custom()

    EMBARGO_BARS = 20
    WF_FOLDS = [
        ("WF1", "2015-01-01", "2017-01-01", "2017-01-01", "2019-01-01"),
        ("WF2", "2015-01-01", "2019-01-01", "2019-01-01", "2021-01-01"),
        ("WF3", "2015-01-01", "2021-01-01", "2021-01-01", "2023-01-01"),
        ("WF4", "2015-01-01", "2023-01-01", "2023-01-01", "2025-01-01"),
        ("WF5", "2015-01-01", "2025-01-01", "2025-01-01", "2026-04-01"),
    ]

    results = {}
    for config_name, get_fn in [("L5.1", get_base), ("L6", get_l6)]:
        fold_results = []
        for fold_name, train_s, train_e, test_s, test_e in WF_FOLDS:
            test_data = data.slice(test_s, test_e)
            if len(test_data.m15_df) < 200:
                continue

            if EMBARGO_BARS > 0:
                embargo_idx = min(EMBARGO_BARS * 4, len(test_data.m15_df) // 10)
                m15_purged = test_data.m15_df.iloc[embargo_idx:]
                if len(m15_purged) < 200:
                    continue
                h1_start = m15_purged.index[0]
                h1_purged = test_data.h1_df[test_data.h1_df.index >= h1_start - np.timedelta64(200, 'h')]
                from backtest.runner import DataBundle as DB
                test_purged = DB(m15_purged, h1_purged)
            else:
                test_purged = test_data

            kw = get_fn()
            s = run_variant(test_purged, f"{config_name}_{fold_name}", verbose=False,
                           spread_cost=0.30, **kw)
            fold_results.append({
                'fold': fold_name,
                'test_period': f"{test_s} -> {test_e}",
                'n': s['n'],
                'sharpe': s['sharpe'],
                'pnl': s['total_pnl'],
                'max_dd': s['max_dd'],
                'win_rate': s['win_rate'],
            })
        results[config_name] = fold_results

    with open(f"{out}/R13-F_purged_wf.txt", 'w') as f:
        f.write(f"Purged Walk-Forward (embargo={EMBARGO_BARS} H1 bars)\n" + "="*80 + "\n\n")
        for config_name, folds in results.items():
            f.write(f"\n=== {config_name} ===\n")
            f.write(f"{'Fold':<6} {'Period':<30} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'MaxDD':>10} {'WR':>7}\n")
            f.write("-"*85 + "\n")
            for fr in folds:
                f.write(f"{fr['fold']:<6} {fr['test_period']:<30} {fr['n']:>6} {fr['sharpe']:>8.2f} "
                        f"{fmt(fr['pnl']):>12} {fmt(fr['max_dd']):>10} {fr['win_rate']:>6.1f}%\n")
            wins = sum(1 for fr in folds if fr['sharpe'] > 0)
            f.write(f"\nPositive Sharpe folds: {wins}/{len(folds)}\n")

        f.write("\n\n=== L5.1 vs L6 Comparison ===\n")
        l51 = results.get("L5.1", [])
        l6 = results.get("L6", [])
        for i in range(min(len(l51), len(l6))):
            delta = l6[i]['sharpe'] - l51[i]['sharpe']
            f.write(f"{l51[i]['fold']}: L5.1={l51[i]['sharpe']:.2f} L6={l6[i]['sharpe']:.2f} delta={delta:+.2f}\n")

    for config_name, folds in results.items():
        wins = sum(1 for fr in folds if fr['sharpe'] > 0)
        print(f"  {config_name}: {wins}/{len(folds)} folds positive Sharpe")


# ═══════════════════════════════════════════════════════════════
# Phase G: 组合 + L7 候选
# ═══════════════════════════════════════════════════════════════

def run_r13_g1(out):
    """R13-G1: 组合所有通过验证的改进"""
    print("\n" + "="*70)
    print("R13-G1: Combined Improvements")
    print("="*70)

    tasks = [
        ("L5.1_Baseline", {**get_base()}, 0.30, None, None),
        ("L6_Baseline", {**get_l6()}, 0.30, None, None),
    ]

    for be in [0.0, 0.5, 1.0]:
        kw = get_l6()
        kw['breakeven_after_atr'] = be
        tasks.append((f"L6_BE{be}", kw, 0.30, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R13-G1_combined.txt", 'w') as f:
        f.write("Combined Improvements\n" + "="*80 + "\n\n")
        write_table(f, sorted(results, key=lambda x: x[2], reverse=True))
        for r in results:
            f.write(f"  {r[0]}: year_pnl={r[14]}\n")

    for r in sorted(results, key=lambda x: x[2], reverse=True):
        print(f"  {r[0]:<30} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


def run_r13_g2(out, l7_kwargs=None):
    """R13-G2: Monte Carlo 鲁棒性验证"""
    print("\n" + "="*70)
    print("R13-G2: Monte Carlo Robustness")
    print("="*70)

    if l7_kwargs is None:
        l7_kwargs = get_l6()

    import random
    base = {**l7_kwargs}
    tasks = [("L7_Base", {**base}, 0.30, None, None)]

    rng = random.Random(42)
    param_keys = ['trailing_activate_atr', 'trailing_distance_atr', 'sl_atr_mult',
                  'tp_atr_mult', 'keltner_adx_threshold', 'choppy_threshold']

    for i in range(100):
        kw = {**base}
        for pk in param_keys:
            if pk in kw:
                orig = kw[pk]
                kw[pk] = orig * (1 + rng.uniform(-0.15, 0.15))
        if 'regime_config' in kw:
            rc = {}
            for regime, params in kw['regime_config'].items():
                rc[regime] = {k: v * (1 + rng.uniform(-0.15, 0.15)) for k, v in params.items()}
            kw['regime_config'] = rc
        tasks.append((f"MC_{i:03d}", kw, 0.30, None, None))

    results = run_pool(tasks)

    sharpes = [r[2] for r in results if r[0] != "L7_Base"]
    pnls = [r[3] for r in results if r[0] != "L7_Base"]
    base_r = [r for r in results if r[0] == "L7_Base"][0]

    with open(f"{out}/R13-G2_monte_carlo.txt", 'w') as f:
        f.write("Monte Carlo Robustness (100 runs, ±15% param perturbation)\n" + "="*80 + "\n\n")
        f.write(f"Base: Sharpe={base_r[2]:.2f} PnL={fmt(base_r[3])}\n\n")
        f.write(f"Sharpe: mean={np.mean(sharpes):.2f} std={np.std(sharpes):.2f} "
                f"min={np.min(sharpes):.2f} max={np.max(sharpes):.2f}\n")
        f.write(f"PnL:    mean={fmt(np.mean(pnls))} min={fmt(np.min(pnls))} max={fmt(np.max(pnls))}\n")
        profitable = sum(1 for p in pnls if p > 0)
        f.write(f"Profitable: {profitable}/100 ({profitable}%)\n")
        f.write(f"Sharpe > 4.0: {sum(1 for s in sharpes if s > 4.0)}/100\n")
        f.write(f"Sharpe > 2.0: {sum(1 for s in sharpes if s > 2.0)}/100\n\n")

        f.write("Worst 5 runs:\n")
        worst = sorted(results[1:], key=lambda x: x[2])[:5]
        for r in worst:
            f.write(f"  {r[0]}: Sharpe={r[2]:.2f} PnL={fmt(r[3])}\n")

    print(f"  MC: Sharpe mean={np.mean(sharpes):.2f} min={np.min(sharpes):.2f} "
          f"profitable={profitable}/100")


def run_r13_g3(out):
    """R13-G3: L5.1 vs L6 vs L7 Walk-Forward"""
    print("\n" + "="*70)
    print("R13-G3: Final Comparison Walk-Forward")
    print("="*70)

    tasks = []
    configs = [("L5.1", get_base()), ("L6", get_l6())]

    for config_name, kw in configs:
        for sp in [0.30, 0.50]:
            tasks.append((f"{config_name}_Full_sp{sp}", {**kw}, sp, None, None))
            for yr_name, s, e in YEARS:
                tasks.append((f"{config_name}_{yr_name}_sp{sp}", {**kw}, sp, s, e))

    results = run_pool(tasks)

    with open(f"{out}/R13-G3_comparison.txt", 'w') as f:
        f.write("L5.1 vs L6 Walk-Forward Comparison\n" + "="*80 + "\n\n")
        for sp in [0.30, 0.50]:
            f.write(f"\n=== Spread ${sp} ===\n")
            f.write(f"{'Year':<6}")
            for cn, _ in configs:
                f.write(f" {cn+' Sharpe':>12} {cn+' PnL':>12}")
            f.write(f" {'Delta_Sh':>10}\n")
            f.write("-"*70 + "\n")

            for yr_name, _, _ in YEARS + [("Full", None, None)]:
                f.write(f"{yr_name:<6}")
                sharpes = []
                for cn, _ in configs:
                    r = [x for x in results if x[0] == f"{cn}_{yr_name}_sp{sp}"]
                    if r:
                        f.write(f" {r[0][2]:>12.2f} {fmt(r[0][3]):>12}")
                        sharpes.append(r[0][2])
                    else:
                        f.write(f" {'N/A':>12} {'N/A':>12}")
                        sharpes.append(0)
                if len(sharpes) >= 2:
                    f.write(f" {sharpes[1]-sharpes[0]:>+9.2f}")
                f.write("\n")

    for r in [x for x in results if "Full" in x[0]]:
        print(f"  {r[0]:<30} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


# ═══════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_time = time.time()
    print(f"\n{'='*70}")
    print(f"Round 13 — Alpha Refinement")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    phases = [
        ("Phase A1", run_r13_a1),
        ("Phase A2", run_r13_a2),
        ("Phase A3", run_r13_a3),
        ("Phase A4", lambda o: run_r13_a4(o)),
        ("Phase B1", run_r13_b1),
        ("Phase B2", lambda o: run_r13_b2(o)),
        ("Phase B3", run_r13_b3),
        ("Phase C1", run_r13_c1),
        ("Phase C2", run_r13_c2),
        ("Phase C3", lambda o: run_r13_c3(o)),
        ("Phase D1", run_r13_d1),
        ("Phase D2", run_r13_d2),
        ("Phase D3", lambda o: run_r13_d3(o)),
        ("Phase E1", run_r13_e1),
        ("Phase F", run_r13_f1),
        ("Phase G1", run_r13_g1),
        ("Phase G2", lambda o: run_r13_g2(o)),
        ("Phase G3", run_r13_g3),
    ]

    completed = []
    for phase_name, phase_fn in phases:
        try:
            t0 = time.time()
            print(f"\n>>> Starting {phase_name}...")
            phase_fn(OUTPUT_DIR)
            elapsed = time.time() - t0
            completed.append((phase_name, elapsed, "OK"))
            print(f"<<< {phase_name} done in {elapsed:.0f}s")
        except Exception as e:
            traceback.print_exc()
            completed.append((phase_name, 0, f"FAIL: {e}"))
            print(f"<<< {phase_name} FAILED: {e}")

    total = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Round 13 Complete — Total: {total:.0f}s ({total/3600:.1f}h)")
    print(f"{'='*70}")

    with open(f"{OUTPUT_DIR}/R13_summary.txt", 'w') as f:
        f.write(f"Round 13 Summary\n{'='*60}\n")
        f.write(f"Total: {total:.0f}s ({total/3600:.1f}h)\n\n")
        for name, elapsed, status in completed:
            f.write(f"{name:<20} {elapsed:>8.0f}s  {status}\n")

    for name, elapsed, status in completed:
        print(f"  {name:<20} {elapsed:>8.0f}s  {status}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
