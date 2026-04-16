#!/usr/bin/env python3
"""
Round 15 — "TATrail 深度验证"
==============================
目标: 围绕 R14 发现的 Time-Adaptive Trailing Stop 做全面深度验证
预计总耗时: ~18-20 小时 (服务器 208核)

背景:
- R14-B3 发现 TATrail_s4_d85 在 L5.1 上 Sharpe +0.10, K-Fold 6/6 (仅测了 s4_d95)
- 但未测试: L6叠加、更激进参数、不同衰减函数、组合优化、Monte Carlo

=== Phase A: TATrail 参数精细扫描 (~2h) ===
R15-A1: Start bar 精细扫描 (2/3/4/5/6/8) × Decay (0.80/0.85/0.90/0.95) × L5.1/L6
R15-A2: Floor 精细扫描 (0.001/0.003/0.005/0.008/0.01/0.02) × 最优 start/decay
R15-A3: 最优参数 × 热力图 (Start × Decay, Sharpe为值)

=== Phase B: 衰减函数对比 (~3h) ===
R15-B1: 指数衰减 (当前) vs 线性衰减 vs 阶梯衰减
R15-B2: 持仓时间分段策略 (bar 1-3 不动, 4-8 温和衰减, 9+ 激进衰减)
R15-B3: 最优衰减函数 K-Fold 验证

=== Phase C: L6 + TATrail 叠加验证 (~3h) ===
R15-C1: L6 + TATrail 全参数 (top 5 from A) K-Fold 6/6
R15-C2: L6 + TATrail + Entry Gap 1h 三重叠加
R15-C3: L7 候选组合 Walk-Forward 11年逐年
R15-C4: L7 候选 Bootstrap CI (1000x)

=== Phase D: 出场行为深度分析 (~3h) ===
R15-D1: TATrail 前后 Trailing/Timeout/SL 出场占比变化
R15-D2: Timeout→Trailing 转化率分析 (TATrail 拯救了多少 timeout 交易)
R15-D3: 持仓时间分布变化 (TATrail 如何改变 bars_held 分布)
R15-D4: MFE/MAE 画像对比 (有无 TATrail)

=== Phase E: 鲁棒性压力测试 (~4h) ===
R15-E1: Monte Carlo 200次 ±15% 参数扰动 (L6+TATrail)
R15-E2: 不同 Spread 下 TATrail 效果 ($0.20/$0.30/$0.40/$0.50/$0.60)
R15-E3: 逐年 TATrail delta 稳定性 (每年 Sharpe 改善是否一致)
R15-E4: 2024-2026 近期窗口放大镜 (TATrail 在近期数据上表现)

=== Phase F: L7 候选最终确认 (~3h) ===
R15-F1: L7 = L6 + TATrail(best) + Gap1h 完整 K-Fold (双 spread)
R15-F2: L7 vs L6 vs L5.1 全面对比表
R15-F3: L7 破产概率模拟 (7 个本金级别)
R15-F4: L7 Kelly Criterion 分析
"""
import sys, os, io, time, traceback, json
import multiprocessing as mp
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
# Support running from scripts/experiments/ or experiments/ — find project root
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.join(_script_dir, '..')
_grandparent = os.path.join(_script_dir, '..', '..')
for _candidate in [_parent, _grandparent, os.getcwd()]:
    _candidate = os.path.abspath(_candidate)
    if os.path.isdir(os.path.join(_candidate, 'backtest')):
        sys.path.insert(0, _candidate)
        os.chdir(_candidate)
        break

OUTPUT_DIR = "results/round15_results"
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
            s.get('avg_win', 0), s.get('avg_loss', 0), s.get('rr', 0))


def _run_one_trades(args):
    label, kw, spread, start, end = args
    from backtest.runner import DataBundle, run_variant
    kc_ema = kw.pop('_kc_ema', 25)
    kc_mult = kw.pop('_kc_mult', 1.2)
    data = DataBundle.load_custom(kc_ema=kc_ema, kc_mult=kc_mult)
    if start and end:
        data = data.slice(start, end)
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kw)
    trades = s.get('_trades', [])
    td = [(round(t.pnl, 2), t.exit_reason or '', t.bars_held, t.strategy or '',
           str(t.entry_time)[:16], t.direction or '', t.lots, t.entry_price,
           round(getattr(t, 'max_favorable', 0), 2),
           round(getattr(t, 'max_adverse', 0), 2))
          for t in trades[:50000]]
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'], s.get('max_dd_pct', 0), td)


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


def add_tatrail(kw, start=4, decay=0.85, floor=0.005):
    kw['time_adaptive_trail'] = True
    kw['time_adaptive_trail_start'] = start
    kw['time_adaptive_trail_decay'] = decay
    kw['time_adaptive_trail_floor'] = floor
    return kw


def write_header(f, title, subtitle=""):
    f.write(f"{title}\n")
    f.write("=" * 80 + "\n")
    if subtitle:
        f.write(f"\n{subtitle}\n")
    f.write("\n")


# ═══════════════════════════════════════════════════════════════
# Phase A: TATrail 参数精细扫描
# ═══════════════════════════════════════════════════════════════

def phase_a1(out):
    """Start × Decay × Config 精细扫描"""
    print("\n" + "=" * 70)
    print("R15-A1: TATrail Start × Decay Grid (L5.1 + L6)")
    print("=" * 70)

    starts = [2, 3, 4, 5, 6, 8]
    decays = [0.75, 0.80, 0.85, 0.90, 0.95]

    tasks = []
    for sp in [0.30, 0.50]:
        for config_name, get_kw in [("L51", get_base), ("L6", get_l6)]:
            # baseline
            kw = get_kw()
            tasks.append((f"Baseline_{config_name}_sp{sp}", kw, sp, None, None))
            for s in starts:
                for d in decays:
                    kw = get_kw()
                    add_tatrail(kw, start=s, decay=d, floor=0.005)
                    tasks.append((f"TA_s{s}_d{int(d*100)}_{config_name}_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R15_A1_start_decay_grid.txt", 'w') as f:
        write_header(f, "R15-A1: TATrail Start × Decay Grid",
                     "Grid: Start=[2,3,4,5,6,8] × Decay=[0.75,0.80,0.85,0.90,0.95], Floor=0.005")

        for sp in [0.30, 0.50]:
            for config_name in ["L51", "L6"]:
                f.write(f"\n--- {config_name} / Spread = ${sp} ---\n\n")
                sub = [r for r in results if r[0].endswith(f"_{config_name}_sp{sp}")]
                base = [r for r in sub if r[0].startswith("Baseline")][0]
                base_sh = base[2]

                f.write(f"{'Config':<30} {'Start':>5} {'Decay':>5} {'N':>6} {'Sharpe':>8} {'dSh':>6} {'PnL':>12} {'MaxDD':>10} {'WR':>6}\n")
                f.write("-" * 95 + "\n")
                f.write(f"{'Baseline':<30} {'—':>5} {'—':>5} {base[1]:>6} {base[2]:>8.2f} {'+0.00':>6} {fmt(base[3]):>12} {fmt(base[6]):>10} {base[4]:>5.1f}%\n")
                for r in sorted(sub, key=lambda x: -x[2]):
                    if r[0].startswith("Baseline"):
                        continue
                    dsh = r[2] - base_sh
                    f.write(f"{r[0].replace(f'_{config_name}_sp{sp}',''):<30} ")
                    parts = r[0].split("_")
                    s_val = parts[1][1:]
                    d_val = str(int(parts[2][1:]) / 100)
                    f.write(f"{s_val:>5} {d_val:>5} {r[1]:>6} {r[2]:>8.2f} {dsh:>+6.2f} {fmt(r[3]):>12} {fmt(r[6]):>10} {r[4]:>5.1f}%\n")

            # heatmap
            f.write(f"\n--- Sharpe Heatmap (Spread ${sp}) ---\n\n")
            for config_name in ["L51", "L6"]:
                f.write(f"\n{config_name}:\n")
                header = 'Start\\Decay'
                f.write(f"{header:>12}")
                for d in decays:
                    f.write(f" {d:>7}")
                f.write("\n" + "-" * (12 + 8 * len(decays)) + "\n")
                for s in starts:
                    f.write(f"{s:>12}")
                    for d in decays:
                        label = f"TA_s{s}_d{int(d*100)}_{config_name}_sp{sp}"
                        r = next((x for x in results if x[0] == label), None)
                        if r:
                            f.write(f" {r[2]:>7.2f}")
                        else:
                            f.write(f" {'—':>7}")
                    f.write("\n")

    print(f"  A1 done: {len(results)} configs")
    return results


def phase_a2(out, a1_results):
    """Floor 精细扫描 — 用 A1 最优 start/decay"""
    print("\n" + "=" * 70)
    print("R15-A2: TATrail Floor Scan")
    print("=" * 70)

    # find best start/decay from A1 for L6 $0.30
    l6_030 = [r for r in a1_results
              if r[0].endswith("_L6_sp0.3") and not r[0].startswith("Baseline")]
    best = max(l6_030, key=lambda x: x[2])
    parts = best[0].split("_")
    best_start = int(parts[1][1:])
    best_decay = int(parts[2][1:]) / 100
    print(f"  Best from A1: start={best_start}, decay={best_decay}, Sharpe={best[2]:.2f}")

    floors = [0.001, 0.002, 0.003, 0.005, 0.008, 0.010, 0.015, 0.020]
    tasks = []
    for sp in [0.30, 0.50]:
        for config_name, get_kw in [("L51", get_base), ("L6", get_l6)]:
            kw = get_kw()
            tasks.append((f"Baseline_{config_name}_sp{sp}", kw, sp, None, None))
            for fl in floors:
                kw = get_kw()
                add_tatrail(kw, start=best_start, decay=best_decay, floor=fl)
                tasks.append((f"Floor_{fl}_{config_name}_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R15_A2_floor_scan.txt", 'w') as f:
        write_header(f, "R15-A2: TATrail Floor Scan",
                     f"Best Start={best_start}, Decay={best_decay} from A1. Scanning Floor values.")
        for sp in [0.30, 0.50]:
            for config_name in ["L51", "L6"]:
                f.write(f"\n--- {config_name} / Spread = ${sp} ---\n\n")
                sub = [r for r in results if r[0].endswith(f"_{config_name}_sp{sp}")]
                base = [r for r in sub if r[0].startswith("Baseline")][0]
                f.write(f"{'Config':<30} {'Floor':>8} {'N':>6} {'Sharpe':>8} {'dSh':>6} {'PnL':>12} {'MaxDD':>10}\n")
                f.write("-" * 85 + "\n")
                f.write(f"{'Baseline':<30} {'—':>8} {base[1]:>6} {base[2]:>8.2f} {'+0.00':>6} {fmt(base[3]):>12} {fmt(base[6]):>10}\n")
                for r in sorted(sub, key=lambda x: -x[2]):
                    if r[0].startswith("Baseline"):
                        continue
                    dsh = r[2] - base[2]
                    fl_val = r[0].split("_")[1]
                    f.write(f"{r[0].replace(f'_{config_name}_sp{sp}',''):<30} {fl_val:>8} {r[1]:>6} {r[2]:>8.2f} {dsh:>+6.2f} {fmt(r[3]):>12} {fmt(r[6]):>10}\n")

    print(f"  A2 done: {len(results)} configs")
    return best_start, best_decay


# ═══════════════════════════════════════════════════════════════
# Phase B: 衰减函数对比
# ═══════════════════════════════════════════════════════════════

def phase_b1(out, best_start, best_decay):
    """指数 vs 线性 vs 阶梯衰减"""
    print("\n" + "=" * 70)
    print("R15-B1: Decay Function Comparison")
    print("=" * 70)

    tasks = []
    for sp in [0.30, 0.50]:
        for config_name, get_kw in [("L51", get_base), ("L6", get_l6)]:
            # baseline
            kw = get_kw()
            tasks.append((f"Baseline_{config_name}_sp{sp}", kw, sp, None, None))

            # exponential (current) — best from A1
            kw = get_kw()
            add_tatrail(kw, start=best_start, decay=best_decay)
            tasks.append((f"Exp_s{best_start}_d{int(best_decay*100)}_{config_name}_sp{sp}", kw, sp, None, None))

            # more aggressive exponential
            for d in [0.70, 0.75, 0.80]:
                kw = get_kw()
                add_tatrail(kw, start=best_start, decay=d)
                tasks.append((f"Exp_s{best_start}_d{int(d*100)}_{config_name}_sp{sp}", kw, sp, None, None))

            # stepped: no change bars 1-3, moderate 4-6, aggressive 7+
            for mild, aggressive in [(0.92, 0.80), (0.90, 0.75), (0.95, 0.85)]:
                kw = get_kw()
                add_tatrail(kw, start=best_start, decay=mild)
                tasks.append((f"Step_{int(mild*100)}_{int(aggressive*100)}_{config_name}_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R15_B1_decay_functions.txt", 'w') as f:
        write_header(f, "R15-B1: Decay Function Comparison",
                     "Comparing exponential decay rates. Stepped variants use mild decay.\n"
                     f"Best exponential: start={best_start}, decay={best_decay}")
        for sp in [0.30, 0.50]:
            for config_name in ["L51", "L6"]:
                f.write(f"\n--- {config_name} / Spread = ${sp} ---\n\n")
                sub = [r for r in results if r[0].endswith(f"_{config_name}_sp{sp}")]
                base = [r for r in sub if r[0].startswith("Baseline")][0]
                f.write(f"{'Config':<45} {'N':>6} {'Sharpe':>8} {'dSh':>6} {'PnL':>12} {'MaxDD':>10}\n")
                f.write("-" * 90 + "\n")
                f.write(f"{'Baseline':<45} {base[1]:>6} {base[2]:>8.2f} {'+0.00':>6} {fmt(base[3]):>12} {fmt(base[6]):>10}\n")
                for r in sorted(sub, key=lambda x: -x[2]):
                    if r[0].startswith("Baseline"):
                        continue
                    dsh = r[2] - base[2]
                    name = r[0].replace(f"_{config_name}_sp{sp}", "")
                    f.write(f"{name:<45} {r[1]:>6} {r[2]:>8.2f} {dsh:>+6.2f} {fmt(r[3]):>12} {fmt(r[6]):>10}\n")

    print(f"  B1 done: {len(results)} configs")


def phase_b2(out, best_start, best_decay):
    """分段衰减策略 K-Fold"""
    print("\n" + "=" * 70)
    print("R15-B2: Top Configs K-Fold Validation")
    print("=" * 70)

    configs = [
        ("Baseline", {}),
        (f"Exp_best", {'time_adaptive_trail': True, 'time_adaptive_trail_start': best_start,
                       'time_adaptive_trail_decay': best_decay, 'time_adaptive_trail_floor': 0.005}),
        ("Exp_d80", {'time_adaptive_trail': True, 'time_adaptive_trail_start': best_start,
                     'time_adaptive_trail_decay': 0.80, 'time_adaptive_trail_floor': 0.005}),
        ("Exp_d75", {'time_adaptive_trail': True, 'time_adaptive_trail_start': best_start,
                     'time_adaptive_trail_decay': 0.75, 'time_adaptive_trail_floor': 0.005}),
    ]

    tasks = []
    for sp in [0.30, 0.50]:
        for config_name, extra in configs:
            for fold_name, start, end in FOLDS:
                kw = get_base()
                kw.update(extra)
                tasks.append((f"{config_name}_{fold_name}_sp{sp}", kw, sp, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R15_B2_tatrail_kfold.txt", 'w') as f:
        write_header(f, "R15-B2: TATrail K-Fold Validation (L5.1)")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            f.write(f"{'Config':<20}")
            for fold_name, _, _ in FOLDS:
                f.write(f" {fold_name:>10}")
            f.write(f" {'Avg':>10} {'Pass':>6}\n")
            f.write("-" * 95 + "\n")

            for config_name, _ in configs:
                f.write(f"{config_name:<20}")
                base_vals = []
                cfg_vals = []
                for fold_name, _, _ in FOLDS:
                    label = f"{config_name}_{fold_name}_sp{sp}"
                    r = next((x for x in results if x[0] == label), None)
                    sh = r[2] if r else 0
                    f.write(f" {sh:>10.2f}")
                    cfg_vals.append(sh)

                    base_label = f"Baseline_{fold_name}_sp{sp}"
                    br = next((x for x in results if x[0] == base_label), None)
                    base_vals.append(br[2] if br else 0)

                avg = np.mean(cfg_vals) if cfg_vals else 0
                if config_name == "Baseline":
                    f.write(f" {avg:>10.2f} {'—':>6}\n")
                else:
                    wins = sum(1 for c, b in zip(cfg_vals, base_vals) if c > b)
                    f.write(f" {avg:>10.2f} {wins}/6\n")

    print(f"  B2 done")


# ═══════════════════════════════════════════════════════════════
# Phase C: L6 + TATrail 叠加验证
# ═══════════════════════════════════════════════════════════════

def phase_c1(out, best_start, best_decay):
    """L6 + TATrail K-Fold"""
    print("\n" + "=" * 70)
    print("R15-C1: L6 + TATrail K-Fold Validation")
    print("=" * 70)

    configs = [
        ("L6_Baseline", get_l6, {}),
        ("L6_TATrail_best", get_l6, {
            'time_adaptive_trail': True, 'time_adaptive_trail_start': best_start,
            'time_adaptive_trail_decay': best_decay, 'time_adaptive_trail_floor': 0.005}),
        ("L6_TATrail_d80", get_l6, {
            'time_adaptive_trail': True, 'time_adaptive_trail_start': best_start,
            'time_adaptive_trail_decay': 0.80, 'time_adaptive_trail_floor': 0.005}),
        ("L6_TATrail_d85", get_l6, {
            'time_adaptive_trail': True, 'time_adaptive_trail_start': best_start,
            'time_adaptive_trail_decay': 0.85, 'time_adaptive_trail_floor': 0.005}),
        ("L6_TATrail_d90", get_l6, {
            'time_adaptive_trail': True, 'time_adaptive_trail_start': best_start,
            'time_adaptive_trail_decay': 0.90, 'time_adaptive_trail_floor': 0.005}),
    ]

    tasks = []
    for sp in [0.30, 0.50]:
        for config_name, get_kw, extra in configs:
            for fold_name, start, end in FOLDS:
                kw = get_kw()
                kw.update(extra)
                tasks.append((f"{config_name}_{fold_name}_sp{sp}", kw, sp, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R15_C1_l6_tatrail_kfold.txt", 'w') as f:
        write_header(f, "R15-C1: L6 + TATrail K-Fold Validation")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            f.write(f"{'Config':<25}")
            for fold_name, _, _ in FOLDS:
                f.write(f" {fold_name:>10}")
            f.write(f" {'Avg':>10} {'Pass':>6}\n")
            f.write("-" * 100 + "\n")

            for config_name, _, _ in configs:
                f.write(f"{config_name:<25}")
                base_vals = []
                cfg_vals = []
                for fold_name, _, _ in FOLDS:
                    label = f"{config_name}_{fold_name}_sp{sp}"
                    r = next((x for x in results if x[0] == label), None)
                    sh = r[2] if r else 0
                    f.write(f" {sh:>10.2f}")
                    cfg_vals.append(sh)

                    base_label = f"L6_Baseline_{fold_name}_sp{sp}"
                    br = next((x for x in results if x[0] == base_label), None)
                    base_vals.append(br[2] if br else 0)

                avg = np.mean(cfg_vals)
                if config_name == "L6_Baseline":
                    f.write(f" {avg:>10.2f} {'—':>6}\n")
                else:
                    wins = sum(1 for c, b in zip(cfg_vals, base_vals) if c > b)
                    f.write(f" {avg:>10.2f} {wins}/6\n")

    print(f"  C1 done")


def phase_c2(out, best_start, best_decay):
    """L6 + TATrail + Gap1h 三重叠加"""
    print("\n" + "=" * 70)
    print("R15-C2: L6 + TATrail + Gap1h Triple Stack")
    print("=" * 70)

    configs = [
        ("L51_Baseline", get_base, {}),
        ("L6_Baseline", get_l6, {}),
        ("L6_TATrail", get_l6, {
            'time_adaptive_trail': True, 'time_adaptive_trail_start': best_start,
            'time_adaptive_trail_decay': best_decay, 'time_adaptive_trail_floor': 0.005}),
        ("L6_Gap1h", get_l6, {'min_entry_gap_hours': 1.0}),
        ("L7_L6_TATrail_Gap1h", get_l6, {
            'time_adaptive_trail': True, 'time_adaptive_trail_start': best_start,
            'time_adaptive_trail_decay': best_decay, 'time_adaptive_trail_floor': 0.005,
            'min_entry_gap_hours': 1.0}),
    ]

    tasks = []
    for sp in [0.30, 0.50]:
        for config_name, get_kw, extra in configs:
            kw = get_kw()
            kw.update(extra)
            tasks.append((f"{config_name}_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R15_C2_triple_stack.txt", 'w') as f:
        write_header(f, "R15-C2: L7 Candidate = L6 + TATrail + Gap1h")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            sub = [r for r in results if r[0].endswith(f"_sp{sp}")]
            f.write(f"{'Config':<30} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'MaxDD':>10} {'WR':>6} {'AvgW':>8} {'AvgL':>8} {'RR':>6}\n")
            f.write("-" * 105 + "\n")
            for r in sorted(sub, key=lambda x: -x[2]):
                name = r[0].replace(f"_sp{sp}", "")
                f.write(f"{name:<30} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {fmt(r[6]):>10} {r[4]:>5.1f}% {r[9]:>8.2f} {r[10]:>8.2f} {r[11]:>6.2f}\n")

    print(f"  C2 done")


def phase_c3(out, best_start, best_decay):
    """L7 Walk-Forward 11年"""
    print("\n" + "=" * 70)
    print("R15-C3: L7 Walk-Forward Year-by-Year")
    print("=" * 70)

    configs = [
        ("L51", get_base, {}),
        ("L6", get_l6, {}),
        ("L7", get_l6, {
            'time_adaptive_trail': True, 'time_adaptive_trail_start': best_start,
            'time_adaptive_trail_decay': best_decay, 'time_adaptive_trail_floor': 0.005,
            'min_entry_gap_hours': 1.0}),
    ]

    tasks = []
    for sp in [0.30, 0.50]:
        for config_name, get_kw, extra in configs:
            for yr_name, start, end in YEARS:
                kw = get_kw()
                kw.update(extra)
                tasks.append((f"{config_name}_{yr_name}_sp{sp}", kw, sp, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R15_C3_walkforward.txt", 'w') as f:
        write_header(f, "R15-C3: L7 vs L6 vs L5.1 Walk-Forward Year-by-Year")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            f.write(f"{'Year':>6}")
            for cn, _, _ in configs:
                f.write(f" {cn+'_Sh':>10} {cn+'_PnL':>12}")
            f.write(f" {'L7-L6':>8} {'L7-L51':>8}\n")
            f.write("-" * 80 + "\n")

            totals = {cn: 0 for cn, _, _ in configs}
            pos_years = {cn: 0 for cn, _, _ in configs}
            deltas_l7_l6 = []

            for yr_name, _, _ in YEARS:
                f.write(f"{yr_name:>6}")
                yr_sharpes = {}
                for cn, _, _ in configs:
                    label = f"{cn}_{yr_name}_sp{sp}"
                    r = next((x for x in results if x[0] == label), None)
                    sh = r[2] if r else 0
                    pnl = r[3] if r else 0
                    f.write(f" {sh:>10.2f} {fmt(pnl):>12}")
                    yr_sharpes[cn] = sh
                    totals[cn] += pnl
                    if pnl > 0:
                        pos_years[cn] += 1

                d_l7_l6 = yr_sharpes.get("L7", 0) - yr_sharpes.get("L6", 0)
                d_l7_l51 = yr_sharpes.get("L7", 0) - yr_sharpes.get("L51", 0)
                deltas_l7_l6.append(d_l7_l6)
                f.write(f" {d_l7_l6:>+8.2f} {d_l7_l51:>+8.2f}\n")

            f.write("\n")
            f.write(f"{'Total':>6}")
            for cn, _, _ in configs:
                f.write(f" {'':>10} {fmt(totals[cn]):>12}")
            f.write("\n")
            f.write(f"{'PosYrs':>6}")
            for cn, _, _ in configs:
                f.write(f" {pos_years[cn]:>7}/11 {'':>12}")
            f.write("\n")
            f.write(f"L7>L6: {sum(1 for d in deltas_l7_l6 if d > 0)}/11 years\n")

    print(f"  C3 done")


def phase_c4(out, best_start, best_decay):
    """L7 Bootstrap CI"""
    print("\n" + "=" * 70)
    print("R15-C4: L7 Bootstrap Confidence Intervals (1000x)")
    print("=" * 70)

    from backtest.runner import DataBundle, run_variant

    configs = [
        ("L51", get_base, {}),
        ("L6", get_l6, {}),
        ("L7", get_l6, {
            'time_adaptive_trail': True, 'time_adaptive_trail_start': best_start,
            'time_adaptive_trail_decay': best_decay, 'time_adaptive_trail_floor': 0.005,
            'min_entry_gap_hours': 1.0}),
    ]

    all_results = {}
    for config_name, get_kw, extra in configs:
        kw = get_kw()
        kw.update(extra)
        data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
        s = run_variant(data, config_name, verbose=False, spread_cost=0.30, **kw)
        trades = s.get('_trades', [])
        pnls = [t.pnl for t in trades]

        n_boot = 1000
        boot_sharpes = []
        boot_pnls = []
        rng = np.random.default_rng(42)
        for _ in range(n_boot):
            sample = rng.choice(pnls, size=len(pnls), replace=True)
            boot_pnls.append(float(np.sum(sample)))
            std = np.std(sample, ddof=1)
            sh = float(np.mean(sample) / std * np.sqrt(252)) if std > 0 else 0
            boot_sharpes.append(sh)

        all_results[config_name] = {
            'observed_sharpe': s['sharpe'],
            'observed_pnl': s['total_pnl'],
            'n': s['n'],
            'boot_sharpes': boot_sharpes,
            'boot_pnls': boot_pnls,
        }

    with open(f"{out}/R15_C4_bootstrap_ci.txt", 'w') as f:
        write_header(f, "R15-C4: L7 vs L6 vs L5.1 Bootstrap Confidence Intervals")

        for cn, data in all_results.items():
            f.write(f"\nConfig: {cn}\n")
            f.write(f"  Observed: Sharpe={data['observed_sharpe']:.2f}, PnL={fmt(data['observed_pnl'])}, N={data['n']}\n")
            bs = np.array(data['boot_sharpes'])
            bp = np.array(data['boot_pnls'])
            f.write(f"  Bootstrap (1000 samples):\n")
            f.write(f"    Sharpe: mean={np.mean(bs):.2f}, std={np.std(bs):.2f}\n")
            f.write(f"    95% CI: [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]\n")
            f.write(f"    99% CI: [{np.percentile(bs, 0.5):.2f}, {np.percentile(bs, 99.5):.2f}]\n")
            f.write(f"    PnL 95% CI: [{fmt(np.percentile(bp, 2.5))}, {fmt(np.percentile(bp, 97.5))}]\n")
            f.write(f"    P(Sharpe>0): {(bs > 0).mean() * 100:.1f}%\n")
            f.write(f"    P(Sharpe>2): {(bs > 2).mean() * 100:.1f}%\n\n")

        # CI overlap analysis
        f.write("\n--- L7 vs L6 CI Overlap ---\n")
        l7_bs = np.array(all_results['L7']['boot_sharpes'])
        l6_bs = np.array(all_results['L6']['boot_sharpes'])
        l7_lo, l7_hi = np.percentile(l7_bs, 2.5), np.percentile(l7_bs, 97.5)
        l6_lo, l6_hi = np.percentile(l6_bs, 2.5), np.percentile(l6_bs, 97.5)
        f.write(f"  L7 95% CI: [{l7_lo:.2f}, {l7_hi:.2f}]\n")
        f.write(f"  L6 95% CI: [{l6_lo:.2f}, {l6_hi:.2f}]\n")
        overlap = max(0, min(l7_hi, l6_hi) - max(l7_lo, l6_lo))
        f.write(f"  Overlap width: {overlap:.2f}\n")
        f.write(f"  P(L7 > L6): {(l7_bs > l6_bs).mean() * 100:.1f}%\n")

    print(f"  C4 done")


# ═══════════════════════════════════════════════════════════════
# Phase D: 出场行为分析
# ═══════════════════════════════════════════════════════════════

def phase_d(out, best_start, best_decay):
    """出场行为对比分析"""
    print("\n" + "=" * 70)
    print("R15-D: Exit Behavior Deep Analysis")
    print("=" * 70)

    configs = [
        ("L6_NoTA", get_l6, {}),
        ("L6_TATrail", get_l6, {
            'time_adaptive_trail': True, 'time_adaptive_trail_start': best_start,
            'time_adaptive_trail_decay': best_decay, 'time_adaptive_trail_floor': 0.005}),
        ("L7_Full", get_l6, {
            'time_adaptive_trail': True, 'time_adaptive_trail_start': best_start,
            'time_adaptive_trail_decay': best_decay, 'time_adaptive_trail_floor': 0.005,
            'min_entry_gap_hours': 1.0}),
    ]

    tasks = []
    for config_name, get_kw, extra in configs:
        kw = get_kw()
        kw.update(extra)
        tasks.append((f"{config_name}_sp0.3", kw, 0.30, None, None))

    results = run_pool(tasks, func=_run_one_trades)

    with open(f"{out}/R15_D_exit_behavior.txt", 'w') as f:
        write_header(f, "R15-D: TATrail Exit Behavior Deep Analysis",
                     "Comparing exit patterns with and without TATrail (Spread $0.30)")

        for r in results:
            config_name = r[0].replace("_sp0.3", "")
            trades = r[8]
            f.write(f"\n{'='*60}\n")
            f.write(f"Config: {config_name} — N={r[1]}, Sharpe={r[2]:.2f}, PnL={fmt(r[3])}\n")
            f.write(f"{'='*60}\n\n")

            # exit reason breakdown
            exit_counts = Counter()
            exit_pnl = defaultdict(float)
            exit_bars = defaultdict(list)
            exit_mfe = defaultdict(list)
            exit_mae = defaultdict(list)
            bars_dist = Counter()

            for pnl, exit_r, bars, strat, etime, direction, lots, eprice, mfe, mae in trades:
                exit_counts[exit_r] += 1
                exit_pnl[exit_r] += pnl
                exit_bars[exit_r].append(bars)
                exit_mfe[exit_r].append(mfe)
                exit_mae[exit_r].append(mae)
                if bars <= 2:
                    bars_dist['1-2'] += 1
                elif bars <= 5:
                    bars_dist['3-5'] += 1
                elif bars <= 10:
                    bars_dist['6-10'] += 1
                elif bars <= 15:
                    bars_dist['11-15'] += 1
                else:
                    bars_dist['16+'] += 1

            f.write("--- Exit Reason Summary ---\n")
            f.write(f"{'Reason':<20} {'N':>6} {'%':>6} {'TotalPnL':>12} {'AvgPnL':>10} {'AvgBars':>8} {'AvgMFE':>8} {'AvgMAE':>8}\n")
            f.write("-" * 85 + "\n")
            total_n = sum(exit_counts.values())
            for reason in sorted(exit_counts.keys(), key=lambda x: -exit_counts[x]):
                n = exit_counts[reason]
                pnl_total = exit_pnl[reason]
                avg_pnl = pnl_total / n if n else 0
                avg_bars = np.mean(exit_bars[reason]) if exit_bars[reason] else 0
                avg_mfe = np.mean(exit_mfe[reason]) if exit_mfe[reason] else 0
                avg_mae = np.mean(exit_mae[reason]) if exit_mae[reason] else 0
                pct = n / total_n * 100
                f.write(f"{reason:<20} {n:>6} {pct:>5.1f}% {fmt(pnl_total):>12} {avg_pnl:>+10.2f} {avg_bars:>8.1f} {avg_mfe:>8.2f} {avg_mae:>8.2f}\n")

            f.write("\n--- Bars Held Distribution ---\n")
            for bucket in ['1-2', '3-5', '6-10', '11-15', '16+']:
                n = bars_dist.get(bucket, 0)
                pct = n / total_n * 100
                f.write(f"  {bucket:>5} bars: {n:>6} ({pct:>5.1f}%)\n")

    print(f"  D done")


# ═══════════════════════════════════════════════════════════════
# Phase E: 鲁棒性压力测试
# ═══════════════════════════════════════════════════════════════

def phase_e1(out, best_start, best_decay):
    """Monte Carlo 200次"""
    print("\n" + "=" * 70)
    print("R15-E1: L7 Monte Carlo Parameter Perturbation (200x)")
    print("=" * 70)

    rng = np.random.default_rng(42)
    tasks = []
    for i in range(200):
        kw = get_l6()
        sl_mult = 3.5 * (1 + rng.uniform(-0.15, 0.15))
        kw['sl_atr_mult'] = sl_mult
        choppy = 0.50 * (1 + rng.uniform(-0.15, 0.15))
        kw['choppy_threshold'] = choppy

        decay_pert = best_decay * (1 + rng.uniform(-0.10, 0.10))
        decay_pert = max(0.5, min(0.99, decay_pert))
        start_pert = max(2, best_start + rng.integers(-1, 2))

        add_tatrail(kw, start=int(start_pert), decay=decay_pert)
        kw['min_entry_gap_hours'] = 1.0
        tasks.append((f"MC_{i:03d}", kw, 0.30, None, None))

    results = run_pool(tasks)
    sharpes = [r[2] for r in results]
    pnls = [r[3] for r in results]
    dds = [r[6] for r in results]

    with open(f"{out}/R15_E1_monte_carlo.txt", 'w') as f:
        write_header(f, "R15-E1: L7 Monte Carlo Parameter Perturbation (200x, +/-15%)",
                     f"N simulations: 200\nPerturbation: SL/Choppy +/-15%, TATrail decay +/-10%, start +/-1")

        f.write("\n--- Sharpe Distribution ---\n")
        f.write(f"  Mean:   {np.mean(sharpes):.2f}\n")
        f.write(f"  Std:    {np.std(sharpes):.2f}\n")
        f.write(f"  Min:    {np.min(sharpes):.2f}\n")
        f.write(f"  Max:    {np.max(sharpes):.2f}\n")
        f.write(f"  P5:     {np.percentile(sharpes, 5):.2f}\n")
        f.write(f"  P25:    {np.percentile(sharpes, 25):.2f}\n")
        f.write(f"  P50:    {np.percentile(sharpes, 50):.2f}\n")
        f.write(f"  P75:    {np.percentile(sharpes, 75):.2f}\n")
        f.write(f"  P95:    {np.percentile(sharpes, 95):.2f}\n")
        f.write(f"  100% profitable: {all(p > 0 for p in pnls)}\n")
        f.write(f"  % Sharpe > 0: {sum(1 for s in sharpes if s > 0) / len(sharpes) * 100:.1f}%\n")
        f.write(f"  % Sharpe > 4: {sum(1 for s in sharpes if s > 4) / len(sharpes) * 100:.1f}%\n")

        f.write(f"\n--- PnL Distribution ---\n")
        f.write(f"  Mean:   {fmt(np.mean(pnls))}\n")
        f.write(f"  Min:    {fmt(np.min(pnls))}\n")
        f.write(f"  Max:    {fmt(np.max(pnls))}\n")
        f.write(f"  100% positive: {all(p > 0 for p in pnls)}\n")

        f.write(f"\n--- MaxDD Distribution ---\n")
        f.write(f"  Mean:   {fmt(np.mean(dds))}\n")
        f.write(f"  Worst:  {fmt(np.max(dds))}\n")
        f.write(f"  P95:    {fmt(np.percentile(dds, 95))}\n")

        f.write(f"\n--- Worst 10 Runs ---\n")
        worst = sorted(results, key=lambda x: x[2])[:10]
        for r in worst:
            f.write(f"  {r[0]}: Sharpe={r[2]:.2f}, PnL={fmt(r[3])}, MaxDD={fmt(r[6])}\n")

    print(f"  E1 done: min Sharpe={min(sharpes):.2f}, mean={np.mean(sharpes):.2f}")


def phase_e2(out, best_start, best_decay):
    """不同 Spread 下 TATrail 效果"""
    print("\n" + "=" * 70)
    print("R15-E2: TATrail Effect Across Spread Levels")
    print("=" * 70)

    spreads = [0.20, 0.30, 0.40, 0.50, 0.60]
    tasks = []
    for sp in spreads:
        for config_name, get_kw, extra in [
            ("L6", get_l6, {}),
            ("L7", get_l6, {
                'time_adaptive_trail': True, 'time_adaptive_trail_start': best_start,
                'time_adaptive_trail_decay': best_decay, 'time_adaptive_trail_floor': 0.005,
                'min_entry_gap_hours': 1.0}),
        ]:
            kw = get_kw()
            kw.update(extra)
            tasks.append((f"{config_name}_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R15_E2_spread_sensitivity.txt", 'w') as f:
        write_header(f, "R15-E2: L7 vs L6 Across Spread Levels")
        f.write(f"{'Spread':>8} {'L6_Sharpe':>10} {'L7_Sharpe':>10} {'Delta':>8} {'L6_PnL':>12} {'L7_PnL':>12} {'L6_MaxDD':>10} {'L7_MaxDD':>10}\n")
        f.write("-" * 85 + "\n")
        for sp in spreads:
            l6 = next((r for r in results if r[0] == f"L6_sp{sp}"), None)
            l7 = next((r for r in results if r[0] == f"L7_sp{sp}"), None)
            if l6 and l7:
                delta = l7[2] - l6[2]
                f.write(f"${sp:<7} {l6[2]:>10.2f} {l7[2]:>10.2f} {delta:>+8.2f} {fmt(l6[3]):>12} {fmt(l7[3]):>12} {fmt(l6[6]):>10} {fmt(l7[6]):>10}\n")

    print(f"  E2 done")


def phase_e3(out, best_start, best_decay):
    """逐年 TATrail delta 稳定性"""
    print("\n" + "=" * 70)
    print("R15-E3: TATrail Delta Stability Per Year")
    print("=" * 70)

    tasks = []
    for yr_name, start, end in YEARS:
        for config_name, get_kw, extra in [
            ("L6", get_l6, {}),
            ("L6_TA", get_l6, {
                'time_adaptive_trail': True, 'time_adaptive_trail_start': best_start,
                'time_adaptive_trail_decay': best_decay, 'time_adaptive_trail_floor': 0.005}),
        ]:
            kw = get_kw()
            kw.update(extra)
            for sp in [0.30, 0.50]:
                tasks.append((f"{config_name}_{yr_name}_sp{sp}", kw, sp, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R15_E3_yearly_delta.txt", 'w') as f:
        write_header(f, "R15-E3: TATrail Delta Stability Per Year",
                     "Does TATrail consistently improve L6 across every year?")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            f.write(f"{'Year':>6} {'L6_Sharpe':>10} {'L6+TA_Sh':>10} {'Delta':>8} {'L6_PnL':>12} {'L6+TA_PnL':>12}\n")
            f.write("-" * 65 + "\n")
            wins = 0
            for yr_name, _, _ in YEARS:
                l6 = next((r for r in results if r[0] == f"L6_{yr_name}_sp{sp}"), None)
                l6ta = next((r for r in results if r[0] == f"L6_TA_{yr_name}_sp{sp}"), None)
                if l6 and l6ta:
                    delta = l6ta[2] - l6[2]
                    if delta > 0:
                        wins += 1
                    f.write(f"{yr_name:>6} {l6[2]:>10.2f} {l6ta[2]:>10.2f} {delta:>+8.2f} {fmt(l6[3]):>12} {fmt(l6ta[3]):>12}\n")
            f.write(f"\nTA wins: {wins}/11 years\n")

    print(f"  E3 done")


# ═══════════════════════════════════════════════════════════════
# Phase F: L7 最终确认
# ═══════════════════════════════════════════════════════════════

def phase_f1(out, best_start, best_decay):
    """L7 K-Fold 双 spread"""
    print("\n" + "=" * 70)
    print("R15-F1: L7 Full K-Fold Validation (Dual Spread)")
    print("=" * 70)

    configs = [
        ("L51_Baseline", get_base, {}),
        ("L6_Baseline", get_l6, {}),
        ("L7_Full", get_l6, {
            'time_adaptive_trail': True, 'time_adaptive_trail_start': best_start,
            'time_adaptive_trail_decay': best_decay, 'time_adaptive_trail_floor': 0.005,
            'min_entry_gap_hours': 1.0}),
    ]

    tasks = []
    for sp in [0.30, 0.50]:
        for config_name, get_kw, extra in configs:
            for fold_name, start, end in FOLDS:
                kw = get_kw()
                kw.update(extra)
                tasks.append((f"{config_name}_{fold_name}_sp{sp}", kw, sp, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R15_F1_l7_kfold_dual.txt", 'w') as f:
        write_header(f, "R15-F1: L7 vs L6 vs L5.1 K-Fold Validation (Dual Spread)")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            f.write(f"{'Config':<20}")
            for fold_name, _, _ in FOLDS:
                f.write(f" {fold_name:>10}")
            f.write(f" {'Avg':>10} {'Pass':>6}\n")
            f.write("-" * 95 + "\n")

            for config_name, _, _ in configs:
                f.write(f"{config_name:<20}")
                cfg_vals = []
                l6_vals = []
                for fold_name, _, _ in FOLDS:
                    label = f"{config_name}_{fold_name}_sp{sp}"
                    r = next((x for x in results if x[0] == label), None)
                    sh = r[2] if r else 0
                    f.write(f" {sh:>10.2f}")
                    cfg_vals.append(sh)

                    l6_label = f"L6_Baseline_{fold_name}_sp{sp}"
                    l6r = next((x for x in results if x[0] == l6_label), None)
                    l6_vals.append(l6r[2] if l6r else 0)

                avg = np.mean(cfg_vals)
                if config_name == "L6_Baseline":
                    f.write(f" {avg:>10.2f} {'—':>6}\n")
                elif config_name == "L51_Baseline":
                    f.write(f" {avg:>10.2f} {'ref':>6}\n")
                else:
                    wins = sum(1 for c, b in zip(cfg_vals, l6_vals) if c > b)
                    f.write(f" {avg:>10.2f} {wins}/6\n")

    print(f"  F1 done")


def phase_f2(out, best_start, best_decay):
    """L7 vs L6 vs L5.1 全面对比"""
    print("\n" + "=" * 70)
    print("R15-F2: L7 vs L6 vs L5.1 Complete Comparison")
    print("=" * 70)

    configs = [
        ("L51", get_base, {}),
        ("L6", get_l6, {}),
        ("L7", get_l6, {
            'time_adaptive_trail': True, 'time_adaptive_trail_start': best_start,
            'time_adaptive_trail_decay': best_decay, 'time_adaptive_trail_floor': 0.005,
            'min_entry_gap_hours': 1.0}),
    ]

    tasks = []
    for sp in [0.30, 0.50]:
        for config_name, get_kw, extra in configs:
            kw = get_kw()
            kw.update(extra)
            tasks.append((f"{config_name}_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R15_F2_full_comparison.txt", 'w') as f:
        write_header(f, "R15-F2: L7 vs L6 vs L5.1 Complete Comparison")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            sub = sorted([r for r in results if r[0].endswith(f"_sp{sp}")], key=lambda x: -x[2])
            f.write(f"{'Config':<15} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'MaxDD':>10} {'DD%':>6} {'WR':>6} {'AvgW':>8} {'AvgL':>8} {'RR':>6}\n")
            f.write("-" * 95 + "\n")
            for r in sub:
                name = r[0].replace(f"_sp{sp}", "")
                f.write(f"{name:<15} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {fmt(r[6]):>10} {r[7]:>5.1f}% {r[4]:>5.1f}% {r[9]:>8.2f} {r[10]:>8.2f} {r[11]:>6.2f}\n")

    print(f"  F2 done")


def phase_f3(out, best_start, best_decay):
    """L7 破产概率"""
    print("\n" + "=" * 70)
    print("R15-F3: L7 Ruin Probability Simulation")
    print("=" * 70)

    from backtest.runner import DataBundle, run_variant

    kw = get_l6()
    add_tatrail(kw, start=best_start, decay=best_decay)
    kw['min_entry_gap_hours'] = 1.0
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    s = run_variant(data, "L7", verbose=False, spread_cost=0.30, **kw)
    trades = s.get('_trades', [])
    pnls = np.array([t.pnl for t in trades])

    capitals = [500, 1000, 1500, 2000, 3000, 5000, 10000]
    n_boot = 1000
    rng = np.random.default_rng(42)

    with open(f"{out}/R15_F3_ruin_probability.txt", 'w') as f:
        write_header(f, "R15-F3: L7 Ruin Probability Simulation",
                     f"Method: Bootstrap {n_boot}x resampling of {len(pnls)} L7 trades\n"
                     "Ruin = losing X% of starting capital at any point")

        f.write(f"{'Capital':>10} {'Ruin50%':>10} {'Ruin75%':>10} {'MaxDD_med':>10} {'MaxDD_95':>10} {'Final_med':>12}\n")
        f.write("-" * 65 + "\n")

        for cap in capitals:
            ruin50 = 0
            ruin75 = 0
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

    print(f"  F3 done")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

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

    # Phase A
    a1_results = run_phase("Phase A1: Start×Decay Grid", phase_a1)
    best_start, best_decay = run_phase("Phase A2: Floor Scan", phase_a2, a1_results) or (4, 0.85)

    # Phase B
    run_phase("Phase B1: Decay Functions", phase_b1, best_start, best_decay)
    run_phase("Phase B2: TATrail K-Fold", phase_b2, best_start, best_decay)

    # Phase C
    run_phase("Phase C1: L6+TATrail K-Fold", phase_c1, best_start, best_decay)
    run_phase("Phase C2: Triple Stack", phase_c2, best_start, best_decay)
    run_phase("Phase C3: Walk-Forward", phase_c3, best_start, best_decay)
    run_phase("Phase C4: Bootstrap CI", phase_c4, best_start, best_decay)

    # Phase D
    run_phase("Phase D: Exit Behavior", phase_d, best_start, best_decay)

    # Phase E
    run_phase("Phase E1: Monte Carlo 200x", phase_e1, best_start, best_decay)
    run_phase("Phase E2: Spread Sensitivity", phase_e2, best_start, best_decay)
    run_phase("Phase E3: Yearly Delta", phase_e3, best_start, best_decay)

    # Phase F
    run_phase("Phase F1: L7 K-Fold Dual", phase_f1, best_start, best_decay)
    run_phase("Phase F2: Full Comparison", phase_f2, best_start, best_decay)
    run_phase("Phase F3: Ruin Probability", phase_f3, best_start, best_decay)

    total = time.time() - t0

    with open(f"{OUTPUT_DIR}/R15_summary.txt", 'w') as f:
        f.write(f"Round 15 — TATrail Deep Validation\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total: {total:.0f}s ({total/3600:.1f}h)\n")
        f.write(f"Started: {start_time}\n")
        f.write(f"Best TATrail: start={best_start}, decay={best_decay}\n\n")
        for name, elapsed, status in phases:
            f.write(f"{name:<40} {elapsed:>6.0f}s  {status}\n")

    print(f"\n{'='*60}")
    print(f"Round 15 COMPLETE: {total:.0f}s ({total/3600:.1f}h)")
    print(f"Best TATrail: start={best_start}, decay={best_decay}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
