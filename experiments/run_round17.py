#!/usr/bin/env python3
"""
Round 17 — "Capital Curve Engineering" (资金曲线工程)
=====================================================
目标: 在策略信号不变的前提下，通过资金管理提升风险调整收益
预计总耗时: ~18 小时 (208核服务器)

背景:
- L7 策略信号层已接近优化上限 (Sharpe ~6.3+)
- 本轮从完全不同的维度 — 资金管理 — 探索收益增强
- 方向: 复利、Kelly、回撤保护、反马丁格尔、利润再投资、资金曲线过滤

=== Phase A: Compounding 基础验证 (~2h) ===
R17-A1: 不同初始本金下的复利 vs 固定仓位 (500/1000/2000/5000/10000)
R17-A2: 复利在不同 Spread 下的效果 ($0.20~$0.60)
R17-A3: 复利 K-Fold 验证 (是否所有时段都受益)

=== Phase B: Kelly Criterion (~3h) ===
R17-B1: Kelly 分数扫描 (0.05/0.10/0.15/0.20/0.25/0.30/0.50/1.0)
R17-B2: Kelly + 复利 组合
R17-B3: Kelly 在不同市场条件 (Walk-Forward) 下的表现
R17-B4: Kelly 最优分数 K-Fold 验证

=== Phase C: 回撤保护 (~3h) ===
R17-C1: 回撤暂停阈值扫描 (5%/8%/10%/15%/20%)
R17-C2: 回撤缩仓阈值 × 缩仓比例 网格
R17-C3: 回撤保护 + 复利 组合 (保护复利的缺点)
R17-C4: 回撤保护 K-Fold 验证

=== Phase D: Anti-Martingale (~3h) ===
R17-D1: Anti-Martingale 参数扫描 (win_mult × loss_mult × max_streak)
R17-D2: Anti-Martingale + 复利 + 回撤保护 三重叠加
R17-D3: Anti-Martingale Walk-Forward 验证
R17-D4: Anti-Martingale K-Fold 验证

=== Phase E: 利润再投资 + 资金曲线过滤 (~3h) ===
R17-E1: 利润再投资比例扫描 (20%/40%/60%/80%/100%)
R17-E2: 资金曲线过滤 — MA 周期扫描 (20/30/50/75/100)
R17-E3: 再投资 + 曲线过滤 组合
R17-E4: 组合 K-Fold 验证

=== Phase F: 最优组合 + 鲁棒性 (~2.5h) ===
R17-F1: 各特性单独 vs 叠加对比矩阵
R17-F2: Monte Carlo 200x ±15% 参数扰动
R17-F3: Spread 敏感性分析

=== Phase G: 最终确认 (~1.5h) ===
R17-G1: 最终组合 K-Fold 双 Spread
R17-G2: 资本增长曲线对比 (固定 vs 最优方案)
R17-G3: 破产概率模拟 (含资金管理)
"""
import sys, os, io, time, traceback, json
import multiprocessing as mp
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.join(_script_dir, '..')
_grandparent = os.path.join(_script_dir, '..', '..')
for _candidate in [_parent, _grandparent, os.getcwd()]:
    _candidate = os.path.abspath(_candidate)
    if os.path.isdir(os.path.join(_candidate, 'backtest')):
        sys.path.insert(0, _candidate)
        os.chdir(_candidate)
        break

OUTPUT_DIR = "results/round17_results"
MAX_WORKERS = 40


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def _run_one(args):
    label, kw, spread, start, end = args
    from backtest.runner import DataBundle, run_variant
    kc_ema = kw.pop('_kc_ema', 25)
    kc_mult = kw.pop('_kc_mult', 1.2)
    data = DataBundle.load_custom(kc_ema=kc_ema, kc_mult=kc_mult)
    if start and end:
        data = data.slice(start, end)
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kw)
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'], s.get('max_dd_pct', 0),
            s.get('year_pnl', {}),
            s.get('avg_win', 0), s.get('avg_loss', 0), s.get('rr', 0),
            s.get('final_capital', 0), s.get('equity_peak', 0),
            s.get('dd_pause_count', 0), s.get('dd_reduce_count', 0),
            s.get('equity_filter_skip', 0))


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


def get_l7():
    """L7 = L6 + TATrail(best from R15) + Gap1h"""
    kw = get_l6()
    kw['time_adaptive_trail'] = True
    kw['time_adaptive_trail_start'] = 2
    kw['time_adaptive_trail_decay'] = 0.75
    kw['time_adaptive_trail_floor'] = 0.003
    kw['min_entry_gap_hours'] = 1.0
    return kw


FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-01"),
]

YEARS = [(str(y), f"{y}-01-01", f"{y+1}-01-01") for y in range(2015, 2026)]


def write_header(f, title, subtitle=""):
    f.write(f"{title}\n")
    f.write("=" * 80 + "\n")
    if subtitle:
        f.write(f"\n{subtitle}\n")
    f.write("\n")


# ===============================================================
# Phase A: Compounding 基础验证
# ===============================================================

def phase_a1(out):
    """不同本金下的复利 vs 固定仓位"""
    print("\n" + "=" * 70)
    print("R17-A1: Compounding vs Fixed Sizing at Different Capitals")
    print("=" * 70)

    capitals = [500, 1000, 2000, 5000, 10000]
    tasks = []

    for cap in capitals:
        kw = get_l7()
        kw['initial_capital'] = cap
        tasks.append((f"Fixed_{cap}", kw, 0.30, None, None))

        kw = get_l7()
        kw['initial_capital'] = cap
        kw['compounding'] = True
        tasks.append((f"Compound_{cap}", kw, 0.30, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R17_A1_compounding_vs_fixed.txt", 'w') as f:
        write_header(f, "R17-A1: Compounding vs Fixed Sizing",
                     "L7 strategy, Spread=$0.30")
        f.write(f"{'Config':<20} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'Final':>12} {'MaxDD':>10} {'DD%':>6} {'Peak':>12}\n")
        f.write("-" * 95 + "\n")

        for cap in capitals:
            fx = next((r for r in results if r[0] == f"Fixed_{cap}"), None)
            cp = next((r for r in results if r[0] == f"Compound_{cap}"), None)
            if fx:
                f.write(f"{'Fixed_'+str(cap):<20} {fx[1]:>6} {fx[2]:>8.2f} {fmt(fx[3]):>12} {fmt(fx[12]):>12} {fmt(fx[6]):>10} {fx[7]:>5.1f}% {fmt(fx[13]):>12}\n")
            if cp:
                f.write(f"{'Compound_'+str(cap):<20} {cp[1]:>6} {cp[2]:>8.2f} {fmt(cp[3]):>12} {fmt(cp[12]):>12} {fmt(cp[6]):>10} {cp[7]:>5.1f}% {fmt(cp[13]):>12}\n")
            f.write("\n")

    print(f"  A1 done")


def phase_a2(out):
    """复利在不同 Spread 下的效果"""
    print("\n" + "=" * 70)
    print("R17-A2: Compounding Across Spreads")
    print("=" * 70)

    spreads = [0.20, 0.30, 0.40, 0.50, 0.60]
    tasks = []
    for sp in spreads:
        kw = get_l7()
        kw['initial_capital'] = 2000
        tasks.append((f"Fixed_sp{sp}", kw, sp, None, None))

        kw = get_l7()
        kw['initial_capital'] = 2000
        kw['compounding'] = True
        tasks.append((f"Compound_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R17_A2_compounding_spreads.txt", 'w') as f:
        write_header(f, "R17-A2: Compounding Across Spreads",
                     "Capital=$2000, L7 strategy")
        f.write(f"{'Spread':>8} {'Fixed_Sh':>10} {'Fixed_PnL':>12} {'Comp_Sh':>10} {'Comp_PnL':>12} {'Comp_Final':>12} {'dSh':>6}\n")
        f.write("-" * 75 + "\n")
        for sp in spreads:
            fx = next((r for r in results if r[0] == f"Fixed_sp{sp}"), None)
            cp = next((r for r in results if r[0] == f"Compound_sp{sp}"), None)
            if fx and cp:
                dsh = cp[2] - fx[2]
                f.write(f"${sp:<7} {fx[2]:>10.2f} {fmt(fx[3]):>12} {cp[2]:>10.2f} {fmt(cp[3]):>12} {fmt(cp[12]):>12} {dsh:>+6.2f}\n")

    print(f"  A2 done")


def phase_a3(out):
    """复利 K-Fold"""
    print("\n" + "=" * 70)
    print("R17-A3: Compounding K-Fold Validation")
    print("=" * 70)

    tasks = []
    for fn, start, end in FOLDS:
        kw = get_l7()
        kw['initial_capital'] = 2000
        tasks.append((f"Fixed_{fn}", kw, 0.30, start, end))

        kw = get_l7()
        kw['initial_capital'] = 2000
        kw['compounding'] = True
        tasks.append((f"Compound_{fn}", kw, 0.30, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R17_A3_compounding_kfold.txt", 'w') as f:
        write_header(f, "R17-A3: Compounding K-Fold")
        f.write(f"{'Config':<15}")
        for fn, _, _ in FOLDS:
            f.write(f" {fn:>10}")
        f.write(f" {'Avg':>10} {'Win':>6}\n")
        f.write("-" * 90 + "\n")

        for prefix in ["Fixed", "Compound"]:
            f.write(f"{prefix:<15}")
            vals = []
            for fn, _, _ in FOLDS:
                r = next((x for x in results if x[0] == f"{prefix}_{fn}"), None)
                sh = r[2] if r else 0
                vals.append(sh)
                f.write(f" {sh:>10.2f}")
            avg = np.mean(vals)
            if prefix == "Fixed":
                f.write(f" {avg:>10.2f} {'ref':>6}\n")
            else:
                fx_vals = [next((x[2] for x in results if x[0] == f"Fixed_{fn}"), 0) for fn, _, _ in FOLDS]
                wins = sum(1 for a, b in zip(vals, fx_vals) if a > b)
                f.write(f" {avg:>10.2f} {wins}/6\n")

    print(f"  A3 done")


# ===============================================================
# Phase B: Kelly Criterion
# ===============================================================

def phase_b1(out):
    """Kelly 分数扫描"""
    print("\n" + "=" * 70)
    print("R17-B1: Kelly Fraction Scan")
    print("=" * 70)

    kelly_fracs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.0]
    tasks = []

    for sp in [0.30, 0.50]:
        kw = get_l7()
        kw['initial_capital'] = 2000
        tasks.append((f"NoKelly_sp{sp}", kw, sp, None, None))

        for kf in kelly_fracs:
            kw = get_l7()
            kw['initial_capital'] = 2000
            kw['kelly_fraction'] = kf
            tasks.append((f"Kelly_{int(kf*100)}_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R17_B1_kelly_scan.txt", 'w') as f:
        write_header(f, "R17-B1: Kelly Fraction Scan",
                     "Capital=$2000, L7 strategy")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            sub = [r for r in results if r[0].endswith(f"_sp{sp}")]
            base = next((r for r in sub if r[0].startswith("NoKelly")), sub[0])
            f.write(f"{'Config':<20} {'N':>6} {'Sharpe':>8} {'dSh':>6} {'PnL':>12} {'Final':>12} {'MaxDD':>10} {'DD%':>6}\n")
            f.write("-" * 90 + "\n")
            f.write(f"{'NoKelly':<20} {base[1]:>6} {base[2]:>8.2f} {'+0.00':>6} {fmt(base[3]):>12} {fmt(base[12]):>12} {fmt(base[6]):>10} {base[7]:>5.1f}%\n")
            for r in sorted(sub, key=lambda x: -x[2]):
                if r[0].startswith("NoKelly"):
                    continue
                dsh = r[2] - base[2]
                name = r[0].replace(f"_sp{sp}", "")
                f.write(f"{name:<20} {r[1]:>6} {r[2]:>8.2f} {dsh:>+6.2f} {fmt(r[3]):>12} {fmt(r[12]):>12} {fmt(r[6]):>10} {r[7]:>5.1f}%\n")

    print(f"  B1 done")
    return results


def phase_b2(out, b1_results):
    """Kelly + Compounding"""
    print("\n" + "=" * 70)
    print("R17-B2: Kelly + Compounding")
    print("=" * 70)

    b1_030 = [r for r in b1_results if r[0].endswith("_sp0.3") and r[0].startswith("Kelly_")]
    best_kelly = max(b1_030, key=lambda x: x[2]) if b1_030 else None
    best_kf = int(best_kelly[0].split("_")[1]) / 100 if best_kelly else 0.25
    print(f"  Best Kelly from B1: {best_kf}")

    tasks = []
    for sp in [0.30, 0.50]:
        kw = get_l7()
        kw['initial_capital'] = 2000
        tasks.append((f"Baseline_sp{sp}", kw, sp, None, None))

        kw = get_l7()
        kw['initial_capital'] = 2000
        kw['compounding'] = True
        tasks.append((f"Compound_sp{sp}", kw, sp, None, None))

        kw = get_l7()
        kw['initial_capital'] = 2000
        kw['kelly_fraction'] = best_kf
        tasks.append((f"Kelly_sp{sp}", kw, sp, None, None))

        kw = get_l7()
        kw['initial_capital'] = 2000
        kw['kelly_fraction'] = best_kf
        kw['compounding'] = True
        tasks.append((f"Kelly+Comp_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R17_B2_kelly_compounding.txt", 'w') as f:
        write_header(f, "R17-B2: Kelly + Compounding Combinations",
                     f"Best Kelly fraction={best_kf}")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            sub = sorted([r for r in results if r[0].endswith(f"_sp{sp}")], key=lambda x: -x[2])
            f.write(f"{'Config':<20} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'Final':>12} {'MaxDD':>10}\n")
            f.write("-" * 75 + "\n")
            for r in sub:
                name = r[0].replace(f"_sp{sp}", "")
                f.write(f"{name:<20} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {fmt(r[12]):>12} {fmt(r[6]):>10}\n")

    print(f"  B2 done")
    return best_kf


def phase_b3(out, best_kf):
    """Kelly Walk-Forward"""
    print("\n" + "=" * 70)
    print("R17-B3: Kelly Walk-Forward 11 Years")
    print("=" * 70)

    tasks = []
    for yr_name, start, end in YEARS:
        kw = get_l7()
        kw['initial_capital'] = 2000
        tasks.append((f"NoKelly_{yr_name}", kw, 0.30, start, end))

        kw = get_l7()
        kw['initial_capital'] = 2000
        kw['kelly_fraction'] = best_kf
        tasks.append((f"Kelly_{yr_name}", kw, 0.30, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R17_B3_kelly_walkforward.txt", 'w') as f:
        write_header(f, "R17-B3: Kelly Walk-Forward 11 Years")
        f.write(f"{'Year':<6} {'NoKelly_Sh':>10} {'NoKelly_PnL':>12} {'Kelly_Sh':>10} {'Kelly_PnL':>12} {'dSh':>6}\n")
        f.write("-" * 60 + "\n")
        for yr, _, _ in YEARS:
            nk = next((r for r in results if r[0] == f"NoKelly_{yr}"), None)
            k = next((r for r in results if r[0] == f"Kelly_{yr}"), None)
            if nk and k:
                f.write(f"{yr:<6} {nk[2]:>10.2f} {fmt(nk[3]):>12} {k[2]:>10.2f} {fmt(k[3]):>12} {k[2]-nk[2]:>+6.2f}\n")

    print(f"  B3 done")


def phase_b4(out, best_kf):
    """Kelly K-Fold"""
    print("\n" + "=" * 70)
    print("R17-B4: Kelly K-Fold Validation")
    print("=" * 70)

    tasks = []
    for fn, start, end in FOLDS:
        kw = get_l7()
        kw['initial_capital'] = 2000
        tasks.append((f"NoKelly_{fn}", kw, 0.30, start, end))

        kw = get_l7()
        kw['initial_capital'] = 2000
        kw['kelly_fraction'] = best_kf
        tasks.append((f"Kelly_{fn}", kw, 0.30, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R17_B4_kelly_kfold.txt", 'w') as f:
        write_header(f, f"R17-B4: Kelly (f={best_kf}) K-Fold")
        f.write(f"{'Config':<15}")
        for fn, _, _ in FOLDS:
            f.write(f" {fn:>10}")
        f.write(f" {'Avg':>10} {'Win':>6}\n")
        f.write("-" * 90 + "\n")

        for prefix in ["NoKelly", "Kelly"]:
            f.write(f"{prefix:<15}")
            vals = []
            for fn, _, _ in FOLDS:
                r = next((x for x in results if x[0] == f"{prefix}_{fn}"), None)
                sh = r[2] if r else 0
                vals.append(sh)
                f.write(f" {sh:>10.2f}")
            avg = np.mean(vals)
            if prefix == "NoKelly":
                f.write(f" {avg:>10.2f} {'ref':>6}\n")
            else:
                nk_vals = [next((x[2] for x in results if x[0] == f"NoKelly_{fn}"), 0) for fn, _, _ in FOLDS]
                wins = sum(1 for a, b in zip(vals, nk_vals) if a > b)
                f.write(f" {avg:>10.2f} {wins}/6\n")

    print(f"  B4 done")


# ===============================================================
# Phase C: 回撤保护
# ===============================================================

def phase_c1(out):
    """回撤暂停阈值扫描"""
    print("\n" + "=" * 70)
    print("R17-C1: Drawdown Protection Threshold Scan")
    print("=" * 70)

    dd_thresholds = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
    tasks = []

    kw = get_l7()
    kw['initial_capital'] = 2000
    kw['compounding'] = True
    tasks.append(("Baseline_Comp", kw, 0.30, None, None))

    for dd in dd_thresholds:
        kw = get_l7()
        kw['initial_capital'] = 2000
        kw['compounding'] = True
        kw['drawdown_protection'] = True
        kw['drawdown_max_pct'] = dd
        tasks.append((f"DD_{int(dd*100)}", kw, 0.30, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R17_C1_drawdown_threshold.txt", 'w') as f:
        write_header(f, "R17-C1: Drawdown Pause Threshold Scan",
                     "L7 + Compounding, Capital=$2000")
        base = next((r for r in results if r[0] == "Baseline_Comp"), results[0])
        f.write(f"{'Config':<20} {'N':>6} {'Sharpe':>8} {'dSh':>6} {'PnL':>12} {'Final':>12} {'MaxDD':>10} {'Pauses':>8}\n")
        f.write("-" * 95 + "\n")
        f.write(f"{'Baseline_Comp':<20} {base[1]:>6} {base[2]:>8.2f} {'+0.00':>6} {fmt(base[3]):>12} {fmt(base[12]):>12} {fmt(base[6]):>10} {'—':>8}\n")
        for r in sorted(results, key=lambda x: -x[2]):
            if r[0] == "Baseline_Comp":
                continue
            dsh = r[2] - base[2]
            f.write(f"{r[0]:<20} {r[1]:>6} {r[2]:>8.2f} {dsh:>+6.2f} {fmt(r[3]):>12} {fmt(r[12]):>12} {fmt(r[6]):>10} {r[14]:>8}\n")

    print(f"  C1 done")
    return results


def phase_c2(out):
    """回撤缩仓网格"""
    print("\n" + "=" * 70)
    print("R17-C2: Drawdown Reduction Grid")
    print("=" * 70)

    reduce_pcts = [0.03, 0.05, 0.08, 0.10]
    reduce_factors = [0.3, 0.5, 0.7]

    tasks = []
    kw = get_l7()
    kw['initial_capital'] = 2000
    kw['compounding'] = True
    tasks.append(("Baseline", kw, 0.30, None, None))

    for rp in reduce_pcts:
        for rf in reduce_factors:
            kw = get_l7()
            kw['initial_capital'] = 2000
            kw['compounding'] = True
            kw['drawdown_protection'] = True
            kw['drawdown_max_pct'] = 0.15
            kw['drawdown_reduce_pct'] = rp
            kw['drawdown_reduce_factor'] = rf
            tasks.append((f"Red_p{int(rp*100)}_f{int(rf*10)}", kw, 0.30, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R17_C2_reduction_grid.txt", 'w') as f:
        write_header(f, "R17-C2: Drawdown Reduction Grid",
                     "DD pause=15%, scanning reduce_pct × reduce_factor")
        base = next((r for r in results if r[0] == "Baseline"), results[0])
        f.write(f"{'Config':<25} {'N':>6} {'Sharpe':>8} {'dSh':>6} {'PnL':>12} {'MaxDD':>10} {'Reduces':>8}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Baseline':<25} {base[1]:>6} {base[2]:>8.2f} {'+0.00':>6} {fmt(base[3]):>12} {fmt(base[6]):>10} {'—':>8}\n")
        for r in sorted(results, key=lambda x: -x[2]):
            if r[0] == "Baseline":
                continue
            dsh = r[2] - base[2]
            f.write(f"{r[0]:<25} {r[1]:>6} {r[2]:>8.2f} {dsh:>+6.2f} {fmt(r[3]):>12} {fmt(r[6]):>10} {r[15]:>8}\n")

        # Heatmap
        f.write(f"\n  Sharpe Heatmap (reduce_pct × reduce_factor):\n")
        header = 'Pct\\Factor'
        f.write(f"  {header:>12}")
        for rf in reduce_factors:
            f.write(f" {rf:>7}")
        f.write("\n  " + "-" * (12 + 8 * len(reduce_factors)) + "\n")
        for rp in reduce_pcts:
            f.write(f"  {rp*100:>11.0f}%")
            for rf in reduce_factors:
                lbl = f"Red_p{int(rp*100)}_f{int(rf*10)}"
                r = next((x for x in results if x[0] == lbl), None)
                f.write(f" {r[2]:>7.2f}" if r else f" {'—':>7}")
            f.write("\n")

    print(f"  C2 done")
    return results


def phase_c3(out, c1_results, c2_results):
    """回撤保护 + 复利 K-Fold"""
    print("\n" + "=" * 70)
    print("R17-C3: Drawdown Protection K-Fold")
    print("=" * 70)

    # Find best DD params from C1/C2
    c1_best = max([r for r in c1_results if r[0] != "Baseline_Comp"], key=lambda x: x[2])
    best_dd_max = int(c1_best[0].split("_")[1]) / 100

    c2_best = max([r for r in c2_results if r[0] != "Baseline"], key=lambda x: x[2])
    parts = c2_best[0].split("_")
    best_reduce_pct = int(parts[1][1:]) / 100
    best_reduce_factor = int(parts[2][1:]) / 10

    tasks = []
    for fn, start, end in FOLDS:
        kw = get_l7()
        kw['initial_capital'] = 2000
        kw['compounding'] = True
        tasks.append((f"Comp_{fn}", kw, 0.30, start, end))

        kw = get_l7()
        kw['initial_capital'] = 2000
        kw['compounding'] = True
        kw['drawdown_protection'] = True
        kw['drawdown_max_pct'] = best_dd_max
        kw['drawdown_reduce_pct'] = best_reduce_pct
        kw['drawdown_reduce_factor'] = best_reduce_factor
        tasks.append((f"CompDD_{fn}", kw, 0.30, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R17_C3_dd_kfold.txt", 'w') as f:
        write_header(f, "R17-C3: Compounding + DD Protection K-Fold",
                     f"DD: max={best_dd_max}, reduce_pct={best_reduce_pct}, reduce_factor={best_reduce_factor}")
        f.write(f"{'Config':<15}")
        for fn, _, _ in FOLDS:
            f.write(f" {fn:>10}")
        f.write(f" {'Avg':>10} {'Win':>6}\n")
        f.write("-" * 90 + "\n")

        for prefix in ["Comp", "CompDD"]:
            f.write(f"{prefix:<15}")
            vals = []
            for fn, _, _ in FOLDS:
                r = next((x for x in results if x[0] == f"{prefix}_{fn}"), None)
                sh = r[2] if r else 0
                vals.append(sh)
                f.write(f" {sh:>10.2f}")
            avg = np.mean(vals)
            if prefix == "Comp":
                f.write(f" {avg:>10.2f} {'ref':>6}\n")
            else:
                ref_vals = [next((x[2] for x in results if x[0] == f"Comp_{fn}"), 0) for fn, _, _ in FOLDS]
                wins = sum(1 for a, b in zip(vals, ref_vals) if a > b)
                f.write(f" {avg:>10.2f} {wins}/6\n")

    print(f"  C3 done")
    return best_dd_max, best_reduce_pct, best_reduce_factor


# ===============================================================
# Phase D: Anti-Martingale
# ===============================================================

def phase_d1(out):
    """Anti-Martingale 参数扫描"""
    print("\n" + "=" * 70)
    print("R17-D1: Anti-Martingale Parameter Scan")
    print("=" * 70)

    win_mults = [1.1, 1.2, 1.3, 1.5]
    loss_mults = [0.6, 0.7, 0.8, 0.9]
    max_streaks = [2, 3, 5]

    tasks = []
    kw = get_l7()
    kw['initial_capital'] = 2000
    tasks.append(("Baseline", kw, 0.30, None, None))

    for wm in win_mults:
        for lm in loss_mults:
            for ms in max_streaks:
                kw = get_l7()
                kw['initial_capital'] = 2000
                kw['anti_martingale'] = True
                kw['anti_martingale_win_mult'] = wm
                kw['anti_martingale_loss_mult'] = lm
                kw['anti_martingale_max_streak'] = ms
                tasks.append((f"AM_w{int(wm*10)}_l{int(lm*10)}_s{ms}", kw, 0.30, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R17_D1_antimartingale_scan.txt", 'w') as f:
        write_header(f, "R17-D1: Anti-Martingale Parameter Scan")
        base = next((r for r in results if r[0] == "Baseline"), results[0])
        f.write(f"{'Config':<25} {'N':>6} {'Sharpe':>8} {'dSh':>6} {'PnL':>12} {'MaxDD':>10} {'WR':>6}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Baseline':<25} {base[1]:>6} {base[2]:>8.2f} {'+0.00':>6} {fmt(base[3]):>12} {fmt(base[6]):>10} {base[4]:>5.1f}%\n")
        for r in sorted(results, key=lambda x: -x[2])[:20]:
            if r[0] == "Baseline":
                continue
            dsh = r[2] - base[2]
            f.write(f"{r[0]:<25} {r[1]:>6} {r[2]:>8.2f} {dsh:>+6.2f} {fmt(r[3]):>12} {fmt(r[6]):>10} {r[4]:>5.1f}%\n")

    print(f"  D1 done")
    return results


def phase_d2(out, d1_results, best_dd_max, best_reduce_pct, best_reduce_factor):
    """Anti-Martingale + 复利 + DD 保护 三重叠加"""
    print("\n" + "=" * 70)
    print("R17-D2: Anti-Martingale + Compounding + DD Protection")
    print("=" * 70)

    d1_non_base = [r for r in d1_results if r[0] != "Baseline"]
    best_am = max(d1_non_base, key=lambda x: x[2]) if d1_non_base else None
    if best_am:
        parts = best_am[0].split("_")
        best_wm = int(parts[1][1:]) / 10
        best_lm = int(parts[2][1:]) / 10
        best_ms = int(parts[3][1:])
    else:
        best_wm, best_lm, best_ms = 1.2, 0.8, 3

    tasks = []
    combos = [
        ("Baseline", {}),
        ("Compound", {'compounding': True}),
        ("AM", {'anti_martingale': True, 'anti_martingale_win_mult': best_wm,
                'anti_martingale_loss_mult': best_lm, 'anti_martingale_max_streak': best_ms}),
        ("DD", {'compounding': True, 'drawdown_protection': True,
                'drawdown_max_pct': best_dd_max, 'drawdown_reduce_pct': best_reduce_pct,
                'drawdown_reduce_factor': best_reduce_factor}),
        ("Comp+AM", {'compounding': True, 'anti_martingale': True,
                     'anti_martingale_win_mult': best_wm, 'anti_martingale_loss_mult': best_lm,
                     'anti_martingale_max_streak': best_ms}),
        ("Comp+DD+AM", {'compounding': True, 'drawdown_protection': True,
                        'drawdown_max_pct': best_dd_max, 'drawdown_reduce_pct': best_reduce_pct,
                        'drawdown_reduce_factor': best_reduce_factor,
                        'anti_martingale': True, 'anti_martingale_win_mult': best_wm,
                        'anti_martingale_loss_mult': best_lm, 'anti_martingale_max_streak': best_ms}),
    ]

    for sp in [0.30, 0.50]:
        for name, extra in combos:
            kw = get_l7()
            kw['initial_capital'] = 2000
            kw.update(extra)
            tasks.append((f"{name}_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R17_D2_triple_stack.txt", 'w') as f:
        write_header(f, "R17-D2: Anti-Martingale + Compounding + DD Protection",
                     f"AM: w={best_wm}, l={best_lm}, s={best_ms}\n"
                     f"DD: max={best_dd_max}, reduce={best_reduce_pct}, factor={best_reduce_factor}")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            sub = sorted([r for r in results if r[0].endswith(f"_sp{sp}")], key=lambda x: -x[2])
            base = next((r for r in sub if r[0].startswith("Baseline")), sub[0])
            f.write(f"{'Config':<15} {'N':>6} {'Sharpe':>8} {'dSh':>6} {'PnL':>12} {'Final':>12} {'MaxDD':>10} {'DD%':>6}\n")
            f.write("-" * 85 + "\n")
            for r in sub:
                name = r[0].replace(f"_sp{sp}", "")
                dsh = r[2] - base[2]
                f.write(f"{name:<15} {r[1]:>6} {r[2]:>8.2f} {dsh:>+6.2f} {fmt(r[3]):>12} {fmt(r[12]):>12} {fmt(r[6]):>10} {r[7]:>5.1f}%\n")

    print(f"  D2 done")
    return best_wm, best_lm, best_ms


# ===============================================================
# Phase E: 利润再投资 + 资金曲线过滤
# ===============================================================

def phase_e1(out):
    """利润再投资比例扫描"""
    print("\n" + "=" * 70)
    print("R17-E1: Profit Reinvestment Ratio Scan")
    print("=" * 70)

    reinvest_pcts = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]
    tasks = []

    for rp in reinvest_pcts:
        kw = get_l7()
        kw['initial_capital'] = 2000
        if rp > 0:
            kw['profit_reinvest_pct'] = rp
        tasks.append((f"Reinvest_{int(rp*100)}", kw, 0.30, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R17_E1_reinvest_scan.txt", 'w') as f:
        write_header(f, "R17-E1: Profit Reinvestment Ratio Scan")
        base = results[0]
        f.write(f"{'Config':<20} {'N':>6} {'Sharpe':>8} {'dSh':>6} {'PnL':>12} {'Final':>12} {'MaxDD':>10}\n")
        f.write("-" * 80 + "\n")
        for r in results:
            dsh = r[2] - base[2]
            f.write(f"{r[0]:<20} {r[1]:>6} {r[2]:>8.2f} {dsh:>+6.2f} {fmt(r[3]):>12} {fmt(r[12]):>12} {fmt(r[6]):>10}\n")

    print(f"  E1 done")
    return results


def phase_e2(out):
    """资金曲线过滤 MA 周期扫描"""
    print("\n" + "=" * 70)
    print("R17-E2: Equity Curve Filter MA Period Scan")
    print("=" * 70)

    ma_periods = [20, 30, 50, 75, 100]
    tasks = []

    kw = get_l7()
    kw['initial_capital'] = 2000
    tasks.append(("NoFilter", kw, 0.30, None, None))

    for ma in ma_periods:
        kw = get_l7()
        kw['initial_capital'] = 2000
        kw['equity_curve_filter'] = True
        kw['equity_ma_period'] = ma
        tasks.append((f"EqFilter_MA{ma}", kw, 0.30, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R17_E2_equity_filter.txt", 'w') as f:
        write_header(f, "R17-E2: Equity Curve Filter MA Period Scan")
        base = results[0]
        f.write(f"{'Config':<20} {'N':>6} {'Sharpe':>8} {'dSh':>6} {'PnL':>12} {'MaxDD':>10} {'Skipped':>8}\n")
        f.write("-" * 80 + "\n")
        for r in results:
            dsh = r[2] - base[2]
            f.write(f"{r[0]:<20} {r[1]:>6} {r[2]:>8.2f} {dsh:>+6.2f} {fmt(r[3]):>12} {fmt(r[6]):>10} {r[16]:>8}\n")

    print(f"  E2 done")
    return results


# ===============================================================
# Phase F: 最优组合 + 鲁棒性
# ===============================================================

def phase_f1(out, best_kf, best_dd_max, best_reduce_pct, best_reduce_factor,
             best_wm, best_lm, best_ms):
    """各特性叠加矩阵"""
    print("\n" + "=" * 70)
    print("R17-F1: Feature Combination Matrix")
    print("=" * 70)

    combos = [
        ("Baseline", {}),
        ("Compound", {'compounding': True}),
        ("Kelly", {'kelly_fraction': best_kf}),
        ("DD", {'drawdown_protection': True, 'drawdown_max_pct': best_dd_max,
                'drawdown_reduce_pct': best_reduce_pct, 'drawdown_reduce_factor': best_reduce_factor}),
        ("AM", {'anti_martingale': True, 'anti_martingale_win_mult': best_wm,
                'anti_martingale_loss_mult': best_lm, 'anti_martingale_max_streak': best_ms}),
        ("Comp+Kelly", {'compounding': True, 'kelly_fraction': best_kf}),
        ("Comp+DD", {'compounding': True, 'drawdown_protection': True,
                     'drawdown_max_pct': best_dd_max, 'drawdown_reduce_pct': best_reduce_pct,
                     'drawdown_reduce_factor': best_reduce_factor}),
        ("Comp+AM", {'compounding': True, 'anti_martingale': True,
                     'anti_martingale_win_mult': best_wm, 'anti_martingale_loss_mult': best_lm,
                     'anti_martingale_max_streak': best_ms}),
        ("Full", {'compounding': True, 'kelly_fraction': best_kf,
                  'drawdown_protection': True, 'drawdown_max_pct': best_dd_max,
                  'drawdown_reduce_pct': best_reduce_pct, 'drawdown_reduce_factor': best_reduce_factor,
                  'anti_martingale': True, 'anti_martingale_win_mult': best_wm,
                  'anti_martingale_loss_mult': best_lm, 'anti_martingale_max_streak': best_ms}),
    ]

    tasks = []
    for sp in [0.30, 0.50]:
        for name, extra in combos:
            kw = get_l7()
            kw['initial_capital'] = 2000
            kw.update(extra)
            tasks.append((f"{name}_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R17_F1_combo_matrix.txt", 'w') as f:
        write_header(f, "R17-F1: Feature Combination Matrix")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            sub = sorted([r for r in results if r[0].endswith(f"_sp{sp}")], key=lambda x: -x[2])
            base = next((r for r in sub if r[0].startswith("Baseline")), sub[0])
            f.write(f"{'Config':<15} {'N':>6} {'Sharpe':>8} {'dSh':>6} {'PnL':>12} {'Final':>12} {'MaxDD':>10} {'DD%':>6}\n")
            f.write("-" * 85 + "\n")
            for r in sub:
                name = r[0].replace(f"_sp{sp}", "")
                dsh = r[2] - base[2]
                f.write(f"{name:<15} {r[1]:>6} {r[2]:>8.2f} {dsh:>+6.2f} {fmt(r[3]):>12} {fmt(r[12]):>12} {fmt(r[6]):>10} {r[7]:>5.1f}%\n")

    # Determine best combo
    best_030 = max([r for r in results if r[0].endswith("_sp0.3")], key=lambda x: x[2])
    best_combo_name = best_030[0].replace("_sp0.3", "")
    best_combo_extra = dict(next((e for n, e in combos if n == best_combo_name), {}))

    print(f"  F1 done: best={best_combo_name}")
    return best_combo_name, best_combo_extra


def phase_f2(out, best_combo_extra):
    """Monte Carlo 200x"""
    print("\n" + "=" * 70)
    print("R17-F2: Monte Carlo 200x Parameter Perturbation")
    print("=" * 70)

    rng = np.random.default_rng(42)
    n_mc = 200
    perturb_pct = 0.15

    float_params = {
        'kelly_fraction', 'drawdown_max_pct', 'drawdown_reduce_pct',
        'drawdown_reduce_factor', 'anti_martingale_win_mult', 'anti_martingale_loss_mult',
        'trailing_activate_atr', 'trailing_distance_atr', 'sl_atr_mult', 'tp_atr_mult',
        'time_adaptive_trail_decay', 'time_adaptive_trail_floor',
    }
    int_params = {
        'anti_martingale_max_streak', 'time_adaptive_trail_start',
    }

    tasks = []
    for i in range(n_mc):
        kw = get_l7()
        kw['initial_capital'] = 2000
        kw.update(best_combo_extra)
        for p in float_params:
            if p in kw and isinstance(kw[p], (int, float)) and kw[p] != 0:
                kw[p] = kw[p] * rng.uniform(1 - perturb_pct, 1 + perturb_pct)
        for p in int_params:
            if p in kw and isinstance(kw[p], int) and kw[p] != 0:
                kw[p] = max(1, int(kw[p] * rng.uniform(1 - perturb_pct, 1 + perturb_pct)))
        tasks.append((f"MC_{i:03d}", kw, 0.30, None, None))

    results = run_pool(tasks)
    sharpes = [r[2] for r in results]
    pnls = [r[3] for r in results]
    finals = [r[12] for r in results]

    with open(f"{out}/R17_F2_montecarlo.txt", 'w') as f:
        write_header(f, "R17-F2: Monte Carlo 200x ±15% Perturbation")
        f.write(f"Sharpe: mean={np.mean(sharpes):.2f}, std={np.std(sharpes):.2f}, "
                f"5th={np.percentile(sharpes, 5):.2f}, median={np.median(sharpes):.2f}, "
                f"95th={np.percentile(sharpes, 95):.2f}\n")
        f.write(f"PnL:    mean={fmt(np.mean(pnls))}, 5th={fmt(np.percentile(pnls, 5))}, "
                f"median={fmt(np.median(pnls))}, 95th={fmt(np.percentile(pnls, 95))}\n")
        f.write(f"Final:  mean={fmt(np.mean(finals))}, median={fmt(np.median(finals))}\n\n")

        f.write(f"Sharpe distribution:\n")
        for lo, hi in [(0, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 99)]:
            cnt = len([s for s in sharpes if lo <= s < hi])
            f.write(f"  [{lo}-{hi}): {cnt:>4} ({cnt/len(sharpes)*100:.1f}%)\n")

    print(f"  F2 done: median Sharpe={np.median(sharpes):.2f}")


def phase_f3(out, best_combo_extra):
    """Spread 敏感性"""
    print("\n" + "=" * 70)
    print("R17-F3: Spread Sensitivity")
    print("=" * 70)

    spreads = [0.20, 0.30, 0.40, 0.50, 0.60]
    tasks = []
    for sp in spreads:
        kw = get_l7()
        kw['initial_capital'] = 2000
        tasks.append((f"L7_sp{sp}", kw, sp, None, None))

        kw = get_l7()
        kw['initial_capital'] = 2000
        kw.update(best_combo_extra)
        tasks.append((f"L7MM_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R17_F3_spread_sensitivity.txt", 'w') as f:
        write_header(f, "R17-F3: Spread Sensitivity — L7 vs L7+MoneyMgmt")
        f.write(f"{'Spread':>8} {'L7_Sh':>8} {'L7_PnL':>12} {'L7MM_Sh':>8} {'L7MM_PnL':>12} {'L7MM_Final':>12} {'dSh':>6}\n")
        f.write("-" * 70 + "\n")
        for sp in spreads:
            l7 = next((r for r in results if r[0] == f"L7_sp{sp}"), None)
            mm = next((r for r in results if r[0] == f"L7MM_sp{sp}"), None)
            if l7 and mm:
                dsh = mm[2] - l7[2]
                f.write(f"${sp:<7} {l7[2]:>8.2f} {fmt(l7[3]):>12} {mm[2]:>8.2f} {fmt(mm[3]):>12} {fmt(mm[12]):>12} {dsh:>+6.2f}\n")

    print(f"  F3 done")


# ===============================================================
# Phase G: 最终确认
# ===============================================================

def phase_g1(out, best_combo_extra):
    """最终 K-Fold 双 Spread"""
    print("\n" + "=" * 70)
    print("R17-G1: Final K-Fold Dual Spread")
    print("=" * 70)

    tasks = []
    for sp in [0.30, 0.50]:
        for fn, start, end in FOLDS:
            kw = get_l7()
            kw['initial_capital'] = 2000
            tasks.append((f"L7_{fn}_sp{sp}", kw, sp, start, end))

            kw = get_l7()
            kw['initial_capital'] = 2000
            kw.update(best_combo_extra)
            tasks.append((f"L7MM_{fn}_sp{sp}", kw, sp, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R17_G1_final_kfold.txt", 'w') as f:
        write_header(f, "R17-G1: Final K-Fold — L7 vs L7+MoneyMgmt")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            f.write(f"{'Config':<15}")
            for fn, _, _ in FOLDS:
                f.write(f" {fn:>10}")
            f.write(f" {'Avg':>10} {'Win':>6}\n")
            f.write("-" * 90 + "\n")

            for prefix in ["L7", "L7MM"]:
                f.write(f"{prefix:<15}")
                vals = []
                for fn, _, _ in FOLDS:
                    r = next((x for x in results if x[0] == f"{prefix}_{fn}_sp{sp}"), None)
                    sh = r[2] if r else 0
                    vals.append(sh)
                    f.write(f" {sh:>10.2f}")
                avg = np.mean(vals)
                if prefix == "L7":
                    f.write(f" {avg:>10.2f} {'ref':>6}\n")
                else:
                    l7_vals = [next((x[2] for x in results if x[0] == f"L7_{fn}_sp{sp}"), 0) for fn, _, _ in FOLDS]
                    wins = sum(1 for a, b in zip(vals, l7_vals) if a > b)
                    f.write(f" {avg:>10.2f} {wins}/6\n")

    print(f"  G1 done")


def phase_g2(out, best_combo_extra):
    """破产概率"""
    print("\n" + "=" * 70)
    print("R17-G2: Ruin Probability with Money Management")
    print("=" * 70)

    from backtest.runner import DataBundle, run_variant

    configs = [
        ("L7_Fixed", {}),
        ("L7_MM", best_combo_extra),
    ]

    capitals = [500, 1000, 1500, 2000, 3000, 5000, 10000]
    n_boot = 1000
    rng = np.random.default_rng(42)

    with open(f"{out}/R17_G2_ruin_probability.txt", 'w') as f:
        write_header(f, "R17-G2: Ruin Probability — Fixed vs Money Management")

        for config_name, extra in configs:
            kw = get_l7()
            kw['initial_capital'] = 2000
            kw.update(extra)
            data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
            s = run_variant(data, config_name, verbose=False, spread_cost=0.30, **kw)
            trades = s.get('_trades', [])
            pnls = np.array([t.pnl for t in trades])

            f.write(f"\n--- {config_name} ({len(pnls)} trades) ---\n\n")
            f.write(f"{'Capital':>10} {'Ruin50%':>10} {'Ruin75%':>10} {'MaxDD_med':>10} {'MaxDD_95':>10} {'Final_med':>12}\n")
            f.write("-" * 65 + "\n")

            for cap in capitals:
                ruin50 = ruin75 = 0
                max_dds = []
                finals = []
                scale = cap / 2000.0

                for _ in range(n_boot):
                    sample = rng.choice(pnls, size=len(pnls), replace=True) * scale
                    equity = np.cumsum(sample)
                    dd = np.maximum.accumulate(equity) - equity
                    max_dd = float(np.max(dd))
                    max_dds.append(max_dd)
                    finals.append(cap + float(equity[-1]))
                    if max_dd >= cap * 0.50:
                        ruin50 += 1
                    if max_dd >= cap * 0.75:
                        ruin75 += 1

                f.write(f"${cap:<9} {ruin50/n_boot*100:>9.1f}% {ruin75/n_boot*100:>9.1f}% "
                        f"{fmt(np.median(max_dds)):>10} {fmt(np.percentile(max_dds, 95)):>10} "
                        f"{fmt(np.median(finals)):>12}\n")

    print(f"  G2 done")


# ===============================================================
# Main
# ===============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t0 = time.time()
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    phases = []

    def run_phase(name, func, *args):
        pt = time.time()
        try:
            result = func(OUTPUT_DIR, *args)
            elapsed = time.time() - pt
            phases.append((name, elapsed, "OK"))
            print(f"  {name} completed in {elapsed:.0f}s")
            return result
        except Exception as e:
            elapsed = time.time() - pt
            phases.append((name, elapsed, f"FAIL: {e}"))
            print(f"  {name} FAILED: {e}")
            traceback.print_exc()
            return None

    # Phase A: 复利基础
    run_phase("Phase A1: Compounding vs Fixed", phase_a1)
    run_phase("Phase A2: Compounding Spreads", phase_a2)
    run_phase("Phase A3: Compounding K-Fold", phase_a3)

    # Phase B: Kelly
    b1_results = run_phase("Phase B1: Kelly Scan", phase_b1)
    best_kf = run_phase("Phase B2: Kelly+Compound", phase_b2, b1_results) or 0.25
    run_phase("Phase B3: Kelly Walk-Forward", phase_b3, best_kf)
    run_phase("Phase B4: Kelly K-Fold", phase_b4, best_kf)

    # Phase C: 回撤保护
    c1_results = run_phase("Phase C1: DD Threshold", phase_c1)
    c2_results = run_phase("Phase C2: DD Reduction Grid", phase_c2)
    dd_result = run_phase("Phase C3: DD K-Fold", phase_c3, c1_results, c2_results)
    best_dd_max, best_reduce_pct, best_reduce_factor = dd_result or (0.10, 0.05, 0.5)

    # Phase D: Anti-Martingale
    d1_results = run_phase("Phase D1: AM Scan", phase_d1)
    am_result = run_phase("Phase D2: Triple Stack", phase_d2, d1_results,
                          best_dd_max, best_reduce_pct, best_reduce_factor)
    best_wm, best_lm, best_ms = am_result or (1.2, 0.8, 3)

    # Phase E: 再投资 + 曲线过滤
    run_phase("Phase E1: Reinvest Scan", phase_e1)
    run_phase("Phase E2: Equity Filter", phase_e2)

    # Phase F: 最优组合
    f1_result = run_phase("Phase F1: Combo Matrix", phase_f1,
                          best_kf, best_dd_max, best_reduce_pct, best_reduce_factor,
                          best_wm, best_lm, best_ms)
    best_combo_name, best_combo_extra = f1_result or ("Compound", {'compounding': True})
    run_phase("Phase F2: Monte Carlo 200x", phase_f2, best_combo_extra)
    run_phase("Phase F3: Spread Sensitivity", phase_f3, best_combo_extra)

    # Phase G: 最终确认
    run_phase("Phase G1: Final K-Fold", phase_g1, best_combo_extra)
    run_phase("Phase G2: Ruin Probability", phase_g2, best_combo_extra)

    total = time.time() - t0

    with open(f"{OUTPUT_DIR}/R17_summary.txt", 'w') as f:
        f.write(f"Round 17 — Capital Curve Engineering\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total: {total:.0f}s ({total/3600:.1f}h)\n")
        f.write(f"Started: {start_time}\n")
        f.write(f"Best combo: {best_combo_name}\n")
        f.write(f"Kelly={best_kf}, DD_max={best_dd_max}, reduce={best_reduce_pct}/{best_reduce_factor}\n")
        f.write(f"AM: w={best_wm}, l={best_lm}, s={best_ms}\n\n")
        for name, elapsed, status in phases:
            f.write(f"{name:<40} {elapsed:>6.0f}s  {status}\n")

    print(f"\n{'='*60}")
    print(f"Round 17 COMPLETE: {total:.0f}s ({total/3600:.1f}h)")
    print(f"Best combo: {best_combo_name}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
