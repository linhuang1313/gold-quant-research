#!/usr/bin/env python3
"""
Round 16 — "Timeout Sniper" (Timeout 专项优化)
================================================
目标: 系统性解决 Timeout 出场 — 当前最大亏损来源
预计总耗时: ~18 小时 (208核服务器)

背景:
- Timeout 占出场的 ~30-40%, 且 Timeout 交易平均亏损最大
- R14 Progressive SL 有一定效果但未深入
- 本轮全面探索: 提前锁利、不利信号退出、动量衰减、动态持仓时间

=== Phase A: Timeout 画像诊断 (~1h) ===
R16-A1: Baseline Timeout 交易全面分析 (PnL分布、方向、时段、持仓bar分布)
R16-A2: Timeout 交易的 MFE/MAE 画像 (最大浮盈/浮亏 vs 最终结果)
R16-A3: Timeout 前N bar 的价格行为模式 (逆转 vs 震荡 vs 缓慢衰减)

=== Phase B: 预超时锁利 (~3h) ===
R16-B1: Profit Lock ATR 阈值扫描 (0.05/0.10/0.15/0.20/0.30/0.50) × Lock Bar 扫描
R16-B2: 最优 Profit Lock 在 L6 + TATrail 上的叠加效果
R16-B3: Profit Lock K-Fold 验证

=== Phase C: 不利信号退出 (~3h) ===
R16-C1: KC Mid 穿越退出 — Adverse Bar 阈值扫描 (max_hold//4, //3, //2, 2//3)
R16-C2: EMA100 穿越退出 (Momentum Decay) — 阈值扫描
R16-C3: 组合: Adverse + Momentum — 哪个先触发就退出
R16-C4: 最优不利信号退出 K-Fold 验证

=== Phase D: 动态持仓时间 (~3h) ===
R16-D1: Dynamic Timeout — 扩展/缩短阈值扫描
R16-D2: extend_bars × cut_bars 网格
R16-D3: 动态 Timeout + 其他 R16 特性叠加
R16-D4: 动态 Timeout K-Fold 验证

=== Phase E: 最优组合 + Progressive SL 回顾 (~3h) ===
R16-E1: R16 各特性单独 vs 叠加对比矩阵
R16-E2: Progressive SL (R14) 精细调参 + R16 特性叠加
R16-E3: 最终 Timeout Sniper 组合 K-Fold 验证
R16-E4: 组合 Walk-Forward 11年逐年

=== Phase F: 鲁棒性 + L7/L8 候选 (~3h) ===
R16-F1: Monte Carlo 200x ±15% 参数扰动
R16-F2: Spread 敏感性 ($0.20/$0.30/$0.40/$0.50/$0.60)
R16-F3: L7+Timeout 最优 vs L7 baseline 全面对比
R16-F4: Timeout 出场占比变化分析 (优化前后)

=== Phase G: 最终确认 (~2h) ===
R16-G1: 最终组合 K-Fold 双 Spread
R16-G2: 破产概率模拟 (7 个本金级别)
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

OUTPUT_DIR = "results/round16_results"
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
            s.get('timeout_profit_lock', 0), s.get('timeout_adverse_exit', 0),
            s.get('timeout_momentum_exit', 0), s.get('timeout_dynamic_extend', 0),
            s.get('timeout_dynamic_cut', 0))


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
           str(t.entry_time)[:16], t.direction or '', t.lots, t.entry_price)
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
# Phase A: Timeout 画像诊断
# ===============================================================

def phase_a1(out):
    """Baseline Timeout 交易全面分析"""
    print("\n" + "=" * 70)
    print("R16-A1: Baseline Timeout Trade Analysis")
    print("=" * 70)

    tasks = []
    for config_name, get_kw in [("L51", get_base), ("L6", get_l6), ("L7", get_l7)]:
        kw = get_kw()
        tasks.append((f"{config_name}_sp0.3", kw, 0.30, None, None))

    results = run_pool(tasks, _run_one_trades)

    with open(f"{out}/R16_A1_timeout_profile.txt", 'w') as f:
        write_header(f, "R16-A1: Baseline Timeout Trade Profile",
                     "Analyzing Timeout exits across L5.1/L6/L7")

        for r in results:
            label, n, sharpe, pnl, wr, _, maxdd, _, trades_data = r
            f.write(f"\n--- {label} (N={n}, Sharpe={sharpe:.2f}, PnL={fmt(pnl)}) ---\n\n")

            timeout_trades = [t for t in trades_data if 'Timeout' in t[1]]
            non_timeout = [t for t in trades_data if 'Timeout' not in t[1]]
            sl_trades = [t for t in trades_data if t[1] == 'SL']
            trail_trades = [t for t in trades_data if t[1] == 'Trailing']
            tp_trades = [t for t in trades_data if t[1] == 'TP']

            f.write(f"Exit Breakdown:\n")
            f.write(f"  Timeout: {len(timeout_trades)} ({len(timeout_trades)/max(n,1)*100:.1f}%)\n")
            f.write(f"  SL:      {len(sl_trades)} ({len(sl_trades)/max(n,1)*100:.1f}%)\n")
            f.write(f"  Trail:   {len(trail_trades)} ({len(trail_trades)/max(n,1)*100:.1f}%)\n")
            f.write(f"  TP:      {len(tp_trades)} ({len(tp_trades)/max(n,1)*100:.1f}%)\n\n")

            if timeout_trades:
                to_pnls = [t[0] for t in timeout_trades]
                to_wins = [p for p in to_pnls if p > 0]
                to_losses = [p for p in to_pnls if p <= 0]
                f.write(f"Timeout Stats:\n")
                f.write(f"  Total PnL: {fmt(sum(to_pnls))}\n")
                f.write(f"  Win Rate:  {len(to_wins)/len(timeout_trades)*100:.1f}%\n")
                f.write(f"  Avg PnL:   ${np.mean(to_pnls):.2f}\n")
                f.write(f"  Median:    ${np.median(to_pnls):.2f}\n")
                f.write(f"  Avg Win:   ${np.mean(to_wins):.2f}\n" if to_wins else "")
                f.write(f"  Avg Loss:  ${np.mean(to_losses):.2f}\n" if to_losses else "")

                bars = [t[2] for t in timeout_trades]
                f.write(f"\n  Bars held: mean={np.mean(bars):.1f}, median={np.median(bars):.0f}\n")

                buy_to = [t for t in timeout_trades if t[5] == 'BUY']
                sell_to = [t for t in timeout_trades if t[5] == 'SELL']
                f.write(f"  BUY timeouts:  {len(buy_to)} (avg={np.mean([t[0] for t in buy_to]):.2f})\n" if buy_to else "")
                f.write(f"  SELL timeouts: {len(sell_to)} (avg={np.mean([t[0] for t in sell_to]):.2f})\n" if sell_to else "")

                # PnL distribution buckets
                f.write(f"\n  PnL Distribution:\n")
                for bucket, lo, hi in [("<-$5", -999, -5), ("-$5~-$2", -5, -2),
                                        ("-$2~$0", -2, 0), ("$0~$2", 0, 2),
                                        ("$2~$5", 2, 5), (">$5", 5, 999)]:
                    cnt = len([p for p in to_pnls if lo <= p < hi])
                    f.write(f"    {bucket:>10}: {cnt:>5} ({cnt/len(to_pnls)*100:>5.1f}%)\n")

    print(f"  A1 done")


def phase_a2(out):
    """Timeout 前价格行为分析"""
    print("\n" + "=" * 70)
    print("R16-A2: Timeout Pre-exit Behavior")
    print("=" * 70)

    kw = get_l7()
    tasks = [(f"L7_sp0.3", kw, 0.30, None, None)]
    results = run_pool(tasks, _run_one_trades)

    with open(f"{out}/R16_A2_timeout_behavior.txt", 'w') as f:
        write_header(f, "R16-A2: Timeout Pre-exit Price Behavior",
                     "Categorizing what happens before timeout")

        r = results[0]
        _, _, _, _, _, _, _, _, trades_data = r
        timeout_trades = [t for t in trades_data if 'Timeout' in t[1]]

        profitable = [t for t in timeout_trades if t[0] > 0]
        scratch = [t for t in timeout_trades if -1 <= t[0] <= 1]
        losing = [t for t in timeout_trades if t[0] < -1]

        f.write(f"Total Timeout trades: {len(timeout_trades)}\n")
        f.write(f"  Profitable (>$0):    {len(profitable)} ({len(profitable)/max(len(timeout_trades),1)*100:.1f}%)\n")
        f.write(f"  Scratch (-$1~$1):    {len(scratch)} ({len(scratch)/max(len(timeout_trades),1)*100:.1f}%)\n")
        f.write(f"  Losing (<-$1):       {len(losing)} ({len(losing)/max(len(timeout_trades),1)*100:.1f}%)\n\n")

        if profitable:
            f.write(f"Profitable timeouts (could have been locked):\n")
            f.write(f"  Avg PnL: ${np.mean([t[0] for t in profitable]):.2f}\n")
            f.write(f"  Total salvageable: {fmt(sum(t[0] for t in profitable))}\n\n")

    print(f"  A2 done")


# ===============================================================
# Phase B: 预超时锁利
# ===============================================================

def phase_b1(out):
    """Profit Lock 参数扫描"""
    print("\n" + "=" * 70)
    print("R16-B1: Timeout Profit Lock Parameter Scan")
    print("=" * 70)

    lock_atrs = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    lock_bars_offsets = [2, 3, 4, 6, 8]

    tasks = []
    for sp in [0.30, 0.50]:
        for config_name, get_kw in [("L7", get_l7)]:
            kw = get_kw()
            tasks.append((f"Baseline_{config_name}_sp{sp}", kw, sp, None, None))

            for lock_atr in lock_atrs:
                for bar_off in lock_bars_offsets:
                    kw = get_kw()
                    kw['timeout_profit_lock_atr'] = lock_atr
                    kw['timeout_profit_lock_bar'] = 20 - bar_off
                    tasks.append((f"PL_a{int(lock_atr*100)}_b{bar_off}_{config_name}_sp{sp}",
                                  kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R16_B1_profit_lock_scan.txt", 'w') as f:
        write_header(f, "R16-B1: Timeout Profit Lock Parameter Scan",
                     f"Lock ATR: {lock_atrs}\nLock Bar offsets from max_hold: {lock_bars_offsets}")

        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            sub = [r for r in results if r[0].endswith(f"_L7_sp{sp}")]
            base = [r for r in sub if r[0].startswith("Baseline")][0]

            f.write(f"{'Config':<25} {'N':>6} {'Sharpe':>8} {'dSh':>6} {'PnL':>12} {'MaxDD':>10} {'WR':>6} {'Locks':>6}\n")
            f.write("-" * 85 + "\n")
            f.write(f"{'Baseline':<25} {base[1]:>6} {base[2]:>8.2f} {'+0.00':>6} {fmt(base[3]):>12} {fmt(base[6]):>10} {base[4]:>5.1f}% {'—':>6}\n")

            for r in sorted(sub, key=lambda x: -x[2]):
                if r[0].startswith("Baseline"):
                    continue
                dsh = r[2] - base[2]
                name = r[0].replace(f"_L7_sp{sp}", "")
                f.write(f"{name:<25} {r[1]:>6} {r[2]:>8.2f} {dsh:>+6.2f} {fmt(r[3]):>12} {fmt(r[6]):>10} {r[4]:>5.1f}% {r[12]:>6}\n")

            # Heatmap
            f.write(f"\n  Sharpe Heatmap (ATR x BarOffset):\n")
            header = 'ATR\\BarOff'
            f.write(f"  {header:>12}")
            for bo in lock_bars_offsets:
                f.write(f" {bo:>7}")
            f.write("\n  " + "-" * (12 + 8 * len(lock_bars_offsets)) + "\n")
            for la in lock_atrs:
                f.write(f"  {la:>12.2f}")
                for bo in lock_bars_offsets:
                    lbl = f"PL_a{int(la*100)}_b{bo}_L7_sp{sp}"
                    r = next((x for x in results if x[0] == lbl), None)
                    f.write(f" {r[2]:>7.2f}" if r else f" {'—':>7}")
                f.write("\n")

    print(f"  B1 done: {len(results)} configs")
    return results


def phase_b2(out, b1_results):
    """最优 Profit Lock K-Fold"""
    print("\n" + "=" * 70)
    print("R16-B2: Best Profit Lock K-Fold Validation")
    print("=" * 70)

    l7_030 = [r for r in b1_results if r[0].endswith("_L7_sp0.3") and not r[0].startswith("Baseline")]
    if not l7_030:
        print("  No B1 results, skipping B2")
        return 0.15, 4
    best = max(l7_030, key=lambda x: x[2])
    parts = best[0].split("_")
    best_lock_atr = int(parts[1][1:]) / 100
    best_bar_off = int(parts[2][1:])
    print(f"  Best from B1: lock_atr={best_lock_atr}, bar_offset={best_bar_off}, Sharpe={best[2]:.2f}")

    tasks = []
    for sp in [0.30, 0.50]:
        for fold_name, start, end in FOLDS:
            # baseline L7
            kw = get_l7()
            tasks.append((f"L7_{fold_name}_sp{sp}", kw, sp, start, end))
            # L7 + ProfitLock
            kw = get_l7()
            kw['timeout_profit_lock_atr'] = best_lock_atr
            kw['timeout_profit_lock_bar'] = 20 - best_bar_off
            tasks.append((f"L7PL_{fold_name}_sp{sp}", kw, sp, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R16_B2_profit_lock_kfold.txt", 'w') as f:
        write_header(f, "R16-B2: Best Profit Lock K-Fold",
                     f"Best lock_atr={best_lock_atr}, bar_offset={best_bar_off}")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            f.write(f"{'Config':<15}")
            for fn, _, _ in FOLDS:
                f.write(f" {fn:>10}")
            f.write(f" {'Avg':>10} {'Win':>6}\n")
            f.write("-" * 90 + "\n")

            for prefix in ["L7", "L7PL"]:
                f.write(f"{prefix:<15}")
                vals = []
                for fn, _, _ in FOLDS:
                    lbl = f"{prefix}_{fn}_sp{sp}"
                    r = next((x for x in results if x[0] == lbl), None)
                    sh = r[2] if r else 0
                    vals.append(sh)
                    f.write(f" {sh:>10.2f}")
                avg = np.mean(vals)
                if prefix == "L7":
                    f.write(f" {avg:>10.2f} {'ref':>6}\n")
                else:
                    l7_vals = []
                    for fn, _, _ in FOLDS:
                        l7r = next((x for x in results if x[0] == f"L7_{fn}_sp{sp}"), None)
                        l7_vals.append(l7r[2] if l7r else 0)
                    wins = sum(1 for a, b in zip(vals, l7_vals) if a > b)
                    f.write(f" {avg:>10.2f} {wins}/6\n")

    print(f"  B2 done")
    return best_lock_atr, best_bar_off


# ===============================================================
# Phase C: 不利信号退出
# ===============================================================

def phase_c1(out):
    """Adverse Signal Exit — KC Mid 穿越扫描"""
    print("\n" + "=" * 70)
    print("R16-C1: Adverse Signal Exit (KC Mid Crossover)")
    print("=" * 70)

    bar_fractions = [0.25, 0.33, 0.50, 0.67, 0.75]

    tasks = []
    for sp in [0.30, 0.50]:
        kw = get_l7()
        tasks.append((f"Baseline_sp{sp}", kw, sp, None, None))

        for frac in bar_fractions:
            kw = get_l7()
            kw['timeout_adverse_exit'] = True
            kw['timeout_adverse_bar'] = max(1, int(20 * frac))
            tasks.append((f"Adverse_b{int(frac*100)}_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R16_C1_adverse_exit.txt", 'w') as f:
        write_header(f, "R16-C1: Adverse Signal Exit (KC Mid Crossover)",
                     "Exit when price crosses KC midline on adverse side")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            sub = [r for r in results if r[0].endswith(f"_sp{sp}")]
            base = [r for r in sub if r[0].startswith("Baseline")][0]
            f.write(f"{'Config':<25} {'N':>6} {'Sharpe':>8} {'dSh':>6} {'PnL':>12} {'MaxDD':>10} {'WR':>6} {'AdvExit':>8}\n")
            f.write("-" * 85 + "\n")
            f.write(f"{'Baseline':<25} {base[1]:>6} {base[2]:>8.2f} {'+0.00':>6} {fmt(base[3]):>12} {fmt(base[6]):>10} {base[4]:>5.1f}% {'—':>8}\n")
            for r in sorted(sub, key=lambda x: -x[2]):
                if r[0].startswith("Baseline"):
                    continue
                dsh = r[2] - base[2]
                name = r[0].replace(f"_sp{sp}", "")
                f.write(f"{name:<25} {r[1]:>6} {r[2]:>8.2f} {dsh:>+6.2f} {fmt(r[3]):>12} {fmt(r[6]):>10} {r[4]:>5.1f}% {r[13]:>8}\n")

    print(f"  C1 done")
    return results


def phase_c2(out):
    """Momentum Decay Exit — EMA100 穿越扫描"""
    print("\n" + "=" * 70)
    print("R16-C2: Momentum Decay Exit (EMA100 Crossover)")
    print("=" * 70)

    bar_fractions = [0.25, 0.33, 0.50, 0.67, 0.75]

    tasks = []
    for sp in [0.30, 0.50]:
        kw = get_l7()
        tasks.append((f"Baseline_sp{sp}", kw, sp, None, None))

        for frac in bar_fractions:
            kw = get_l7()
            kw['timeout_momentum_exit'] = True
            kw['timeout_momentum_bar'] = max(1, int(20 * frac))
            tasks.append((f"Momentum_b{int(frac*100)}_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R16_C2_momentum_exit.txt", 'w') as f:
        write_header(f, "R16-C2: Momentum Decay Exit (EMA100 Crossover)",
                     "Exit when price crosses EMA100 on adverse side")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            sub = [r for r in results if r[0].endswith(f"_sp{sp}")]
            base = [r for r in sub if r[0].startswith("Baseline")][0]
            f.write(f"{'Config':<25} {'N':>6} {'Sharpe':>8} {'dSh':>6} {'PnL':>12} {'MaxDD':>10} {'WR':>6} {'MomExit':>8}\n")
            f.write("-" * 85 + "\n")
            f.write(f"{'Baseline':<25} {base[1]:>6} {base[2]:>8.2f} {'+0.00':>6} {fmt(base[3]):>12} {fmt(base[6]):>10} {base[4]:>5.1f}% {'—':>8}\n")
            for r in sorted(sub, key=lambda x: -x[2]):
                if r[0].startswith("Baseline"):
                    continue
                dsh = r[2] - base[2]
                name = r[0].replace(f"_sp{sp}", "")
                f.write(f"{name:<25} {r[1]:>6} {r[2]:>8.2f} {dsh:>+6.2f} {fmt(r[3]):>12} {fmt(r[6]):>10} {r[4]:>5.1f}% {r[14]:>8}\n")

    print(f"  C2 done")
    return results


def phase_c3(out, c1_results, c2_results):
    """组合: Adverse + Momentum 退出"""
    print("\n" + "=" * 70)
    print("R16-C3: Combined Adverse + Momentum Exit")
    print("=" * 70)

    # Find best adverse bar from C1
    c1_030 = [r for r in c1_results if r[0].endswith("_sp0.3") and not r[0].startswith("Baseline")]
    best_adv = max(c1_030, key=lambda x: x[2]) if c1_030 else None
    adv_bar = int(best_adv[0].split("_")[1][1:]) if best_adv else 50
    adv_bar_actual = max(1, int(20 * adv_bar / 100))

    # Find best momentum bar from C2
    c2_030 = [r for r in c2_results if r[0].endswith("_sp0.3") and not r[0].startswith("Baseline")]
    best_mom = max(c2_030, key=lambda x: x[2]) if c2_030 else None
    mom_bar = int(best_mom[0].split("_")[1][1:]) if best_mom else 50
    mom_bar_actual = max(1, int(20 * mom_bar / 100))

    tasks = []
    for sp in [0.30, 0.50]:
        kw = get_l7()
        tasks.append((f"Baseline_sp{sp}", kw, sp, None, None))

        # Adverse only
        kw = get_l7()
        kw['timeout_adverse_exit'] = True
        kw['timeout_adverse_bar'] = adv_bar_actual
        tasks.append((f"Adverse_sp{sp}", kw, sp, None, None))

        # Momentum only
        kw = get_l7()
        kw['timeout_momentum_exit'] = True
        kw['timeout_momentum_bar'] = mom_bar_actual
        tasks.append((f"Momentum_sp{sp}", kw, sp, None, None))

        # Combined
        kw = get_l7()
        kw['timeout_adverse_exit'] = True
        kw['timeout_adverse_bar'] = adv_bar_actual
        kw['timeout_momentum_exit'] = True
        kw['timeout_momentum_bar'] = mom_bar_actual
        tasks.append((f"Combined_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R16_C3_combined_exit.txt", 'w') as f:
        write_header(f, "R16-C3: Combined Adverse + Momentum Exit",
                     f"Best adverse bar={adv_bar_actual}, momentum bar={mom_bar_actual}")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            sub = [r for r in results if r[0].endswith(f"_sp{sp}")]
            f.write(f"{'Config':<20} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'MaxDD':>10} {'WR':>6} {'Adv':>6} {'Mom':>6}\n")
            f.write("-" * 80 + "\n")
            for r in sorted(sub, key=lambda x: -x[2]):
                name = r[0].replace(f"_sp{sp}", "")
                f.write(f"{name:<20} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {fmt(r[6]):>10} {r[4]:>5.1f}% {r[13]:>6} {r[14]:>6}\n")

    print(f"  C3 done")
    return adv_bar_actual, mom_bar_actual


def phase_c4(out, adv_bar, mom_bar):
    """最优不利信号退出 K-Fold"""
    print("\n" + "=" * 70)
    print("R16-C4: Best Signal Exit K-Fold Validation")
    print("=" * 70)

    tasks = []
    for sp in [0.30, 0.50]:
        for fn, start, end in FOLDS:
            kw = get_l7()
            tasks.append((f"L7_{fn}_sp{sp}", kw, sp, start, end))

            kw = get_l7()
            kw['timeout_adverse_exit'] = True
            kw['timeout_adverse_bar'] = adv_bar
            kw['timeout_momentum_exit'] = True
            kw['timeout_momentum_bar'] = mom_bar
            tasks.append((f"L7Sig_{fn}_sp{sp}", kw, sp, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R16_C4_signal_exit_kfold.txt", 'w') as f:
        write_header(f, "R16-C4: Signal Exit K-Fold",
                     f"adverse_bar={adv_bar}, momentum_bar={mom_bar}")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            f.write(f"{'Config':<15}")
            for fn, _, _ in FOLDS:
                f.write(f" {fn:>10}")
            f.write(f" {'Avg':>10} {'Win':>6}\n")
            f.write("-" * 90 + "\n")
            for prefix in ["L7", "L7Sig"]:
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

    print(f"  C4 done")


# ===============================================================
# Phase D: 动态持仓时间
# ===============================================================

def phase_d1(out):
    """Dynamic Timeout 参数扫描"""
    print("\n" + "=" * 70)
    print("R16-D1: Dynamic Timeout Parameter Scan")
    print("=" * 70)

    extend_vals = [2, 3, 4, 6, 8]
    cut_vals = [2, 3, 4, 6, 8]
    extend_thresholds = [0.1, 0.2, 0.3, 0.5]
    cut_thresholds = [-0.3, -0.5, -0.8, -1.0]

    tasks = []
    for sp in [0.30]:
        kw = get_l7()
        tasks.append((f"Baseline_sp{sp}", kw, sp, None, None))

        for et in extend_thresholds:
            for ct in cut_thresholds:
                kw = get_l7()
                kw['timeout_dynamic'] = True
                kw['timeout_extend_bars'] = 4
                kw['timeout_cut_bars'] = 4
                kw['timeout_extend_threshold_atr'] = et
                kw['timeout_cut_threshold_atr'] = ct
                tasks.append((f"Dyn_et{int(et*10)}_ct{int(abs(ct)*10)}_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R16_D1_dynamic_timeout.txt", 'w') as f:
        write_header(f, "R16-D1: Dynamic Timeout Threshold Scan",
                     "extend/cut bars=4, scanning extend/cut thresholds")
        for sp in [0.30]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            sub = [r for r in results if r[0].endswith(f"_sp{sp}")]
            base = [r for r in sub if r[0].startswith("Baseline")][0]
            f.write(f"{'Config':<25} {'N':>6} {'Sharpe':>8} {'dSh':>6} {'PnL':>12} {'MaxDD':>10} {'Ext':>5} {'Cut':>5}\n")
            f.write("-" * 85 + "\n")
            f.write(f"{'Baseline':<25} {base[1]:>6} {base[2]:>8.2f} {'+0.00':>6} {fmt(base[3]):>12} {fmt(base[6]):>10} {'—':>5} {'—':>5}\n")
            for r in sorted(sub, key=lambda x: -x[2]):
                if r[0].startswith("Baseline"):
                    continue
                dsh = r[2] - base[2]
                name = r[0].replace(f"_sp{sp}", "")
                f.write(f"{name:<25} {r[1]:>6} {r[2]:>8.2f} {dsh:>+6.2f} {fmt(r[3]):>12} {fmt(r[6]):>10} {r[15]:>5} {r[16]:>5}\n")

    print(f"  D1 done")
    return results


def phase_d2(out, d1_results):
    """extend_bars × cut_bars 网格 with best thresholds"""
    print("\n" + "=" * 70)
    print("R16-D2: Dynamic Timeout Bars Grid")
    print("=" * 70)

    d1_030 = [r for r in d1_results if r[0].endswith("_sp0.3") and not r[0].startswith("Baseline")]
    best_dyn = max(d1_030, key=lambda x: x[2]) if d1_030 else None

    if best_dyn:
        parts = best_dyn[0].split("_")
        best_et = int(parts[1][2:]) / 10
        best_ct = -int(parts[2][2:]) / 10
    else:
        best_et, best_ct = 0.3, -0.5

    extend_bars = [2, 3, 4, 6, 8]
    cut_bars = [2, 3, 4, 6, 8]

    tasks = []
    for sp in [0.30]:
        kw = get_l7()
        tasks.append((f"Baseline_sp{sp}", kw, sp, None, None))

        for eb in extend_bars:
            for cb in cut_bars:
                kw = get_l7()
                kw['timeout_dynamic'] = True
                kw['timeout_extend_bars'] = eb
                kw['timeout_cut_bars'] = cb
                kw['timeout_extend_threshold_atr'] = best_et
                kw['timeout_cut_threshold_atr'] = best_ct
                tasks.append((f"Bars_e{eb}_c{cb}_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R16_D2_dynamic_bars_grid.txt", 'w') as f:
        write_header(f, "R16-D2: Dynamic Timeout Bars Grid",
                     f"Best thresholds: extend={best_et}, cut={best_ct}")

        f.write(f"\nSharpe Heatmap (extend × cut bars):\n")
        header = 'Ext\\Cut'
        f.write(f"  {header:>8}")
        for cb in cut_bars:
            f.write(f" {cb:>7}")
        f.write("\n  " + "-" * (8 + 8 * len(cut_bars)) + "\n")
        for eb in extend_bars:
            f.write(f"  {eb:>8}")
            for cb in cut_bars:
                lbl = f"Bars_e{eb}_c{cb}_sp0.3"
                r = next((x for x in results if x[0] == lbl), None)
                f.write(f" {r[2]:>7.2f}" if r else f" {'—':>7}")
            f.write("\n")

    print(f"  D2 done")
    return best_et, best_ct


# ===============================================================
# Phase E: 最优组合 + Progressive SL 回顾
# ===============================================================

def phase_e1(out, best_lock_atr, best_bar_off, adv_bar, mom_bar, best_et, best_ct):
    """R16 特性叠加对比矩阵"""
    print("\n" + "=" * 70)
    print("R16-E1: Feature Combination Matrix")
    print("=" * 70)

    combos = [
        ("Baseline", {}),
        ("ProfitLock", {'timeout_profit_lock_atr': best_lock_atr, 'timeout_profit_lock_bar': 20 - best_bar_off}),
        ("SignalExit", {'timeout_adverse_exit': True, 'timeout_adverse_bar': adv_bar,
                        'timeout_momentum_exit': True, 'timeout_momentum_bar': mom_bar}),
        ("DynTimeout", {'timeout_dynamic': True, 'timeout_extend_bars': 4, 'timeout_cut_bars': 4,
                        'timeout_extend_threshold_atr': best_et, 'timeout_cut_threshold_atr': best_ct}),
        ("PL+Signal", {'timeout_profit_lock_atr': best_lock_atr, 'timeout_profit_lock_bar': 20 - best_bar_off,
                       'timeout_adverse_exit': True, 'timeout_adverse_bar': adv_bar,
                       'timeout_momentum_exit': True, 'timeout_momentum_bar': mom_bar}),
        ("PL+Dyn", {'timeout_profit_lock_atr': best_lock_atr, 'timeout_profit_lock_bar': 20 - best_bar_off,
                    'timeout_dynamic': True, 'timeout_extend_bars': 4, 'timeout_cut_bars': 4,
                    'timeout_extend_threshold_atr': best_et, 'timeout_cut_threshold_atr': best_ct}),
        ("Signal+Dyn", {'timeout_adverse_exit': True, 'timeout_adverse_bar': adv_bar,
                        'timeout_momentum_exit': True, 'timeout_momentum_bar': mom_bar,
                        'timeout_dynamic': True, 'timeout_extend_bars': 4, 'timeout_cut_bars': 4,
                        'timeout_extend_threshold_atr': best_et, 'timeout_cut_threshold_atr': best_ct}),
        ("AllThree", {'timeout_profit_lock_atr': best_lock_atr, 'timeout_profit_lock_bar': 20 - best_bar_off,
                      'timeout_adverse_exit': True, 'timeout_adverse_bar': adv_bar,
                      'timeout_momentum_exit': True, 'timeout_momentum_bar': mom_bar,
                      'timeout_dynamic': True, 'timeout_extend_bars': 4, 'timeout_cut_bars': 4,
                      'timeout_extend_threshold_atr': best_et, 'timeout_cut_threshold_atr': best_ct}),
    ]

    tasks = []
    for sp in [0.30, 0.50]:
        for name, extra in combos:
            kw = get_l7()
            kw.update(extra)
            tasks.append((f"{name}_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R16_E1_combo_matrix.txt", 'w') as f:
        write_header(f, "R16-E1: Feature Combination Matrix")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            sub = sorted([r for r in results if r[0].endswith(f"_sp{sp}")], key=lambda x: -x[2])
            base = next((r for r in sub if r[0].startswith("Baseline")), sub[0])
            f.write(f"{'Config':<15} {'N':>6} {'Sharpe':>8} {'dSh':>6} {'PnL':>12} {'MaxDD':>10} {'WR':>6} {'PL':>4} {'Adv':>4} {'Mom':>4} {'Ext':>4} {'Cut':>4}\n")
            f.write("-" * 100 + "\n")
            for r in sub:
                name = r[0].replace(f"_sp{sp}", "")
                dsh = r[2] - base[2]
                f.write(f"{name:<15} {r[1]:>6} {r[2]:>8.2f} {dsh:>+6.2f} {fmt(r[3]):>12} {fmt(r[6]):>10} {r[4]:>5.1f}% {r[12]:>4} {r[13]:>4} {r[14]:>4} {r[15]:>4} {r[16]:>4}\n")

    print(f"  E1 done")
    # Find best combo
    best_030 = max([r for r in results if r[0].endswith("_sp0.3")], key=lambda x: x[2])
    best_name = best_030[0].replace("_sp0.3", "")
    best_extra = dict(next((e for n, e in combos if n == best_name), {}))
    return best_name, best_extra


def phase_e2(out, best_name, best_extra):
    """Progressive SL + R16 叠加"""
    print("\n" + "=" * 70)
    print("R16-E2: Progressive SL + R16 Features")
    print("=" * 70)

    prog_configs = [
        (0, 0, 0),
        (6, 2.0, 4),
        (8, 2.0, 4),
        (6, 1.5, 4),
        (8, 1.5, 6),
        (10, 2.5, 5),
    ]

    tasks = []
    for sp in [0.30]:
        for start, target, steps in prog_configs:
            kw = get_l7()
            kw.update(best_extra)
            if start > 0:
                kw['progressive_sl_start_bar'] = start
                kw['progressive_sl_target_mult'] = target
                kw['progressive_sl_steps'] = steps
            lbl_suffix = f"ProgSL_s{start}_t{int(target*10)}_n{steps}" if start > 0 else "NoProg"
            tasks.append((f"{lbl_suffix}_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R16_E2_progressive_sl.txt", 'w') as f:
        write_header(f, "R16-E2: Progressive SL + R16 Best Combo",
                     f"Base = L7 + {best_name}")
        f.write(f"{'Config':<30} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'MaxDD':>10} {'WR':>6}\n")
        f.write("-" * 75 + "\n")
        for r in sorted(results, key=lambda x: -x[2]):
            name = r[0].replace("_sp0.3", "")
            f.write(f"{name:<30} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {fmt(r[6]):>10} {r[4]:>5.1f}%\n")

    print(f"  E2 done")


def phase_e3(out, best_extra):
    """最终组合 Walk-Forward"""
    print("\n" + "=" * 70)
    print("R16-E3: Walk-Forward 11 Years")
    print("=" * 70)

    tasks = []
    for yr_name, start, end in YEARS:
        kw = get_l7()
        tasks.append((f"L7_{yr_name}", kw, 0.30, start, end))

        kw = get_l7()
        kw.update(best_extra)
        tasks.append((f"L7TS_{yr_name}", kw, 0.30, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R16_E3_walkforward.txt", 'w') as f:
        write_header(f, "R16-E3: Walk-Forward 11-Year Comparison",
                     "L7 vs L7+TimeoutSniper, Spread=$0.30")

        f.write(f"{'Year':<6}")
        f.write(f" {'L7_Sharpe':>10} {'L7_PnL':>12} {'L7TS_Sharpe':>12} {'L7TS_PnL':>12} {'dSh':>6}\n")
        f.write("-" * 65 + "\n")

        for yr_name, _, _ in YEARS:
            l7 = next((r for r in results if r[0] == f"L7_{yr_name}"), None)
            ts = next((r for r in results if r[0] == f"L7TS_{yr_name}"), None)
            if l7 and ts:
                dsh = ts[2] - l7[2]
                f.write(f"{yr_name:<6} {l7[2]:>10.2f} {fmt(l7[3]):>12} {ts[2]:>12.2f} {fmt(ts[3]):>12} {dsh:>+6.2f}\n")

    print(f"  E3 done")


# ===============================================================
# Phase F: 鲁棒性测试
# ===============================================================

def phase_f1(out, best_extra):
    """Monte Carlo 200x"""
    print("\n" + "=" * 70)
    print("R16-F1: Monte Carlo 200x Parameter Perturbation")
    print("=" * 70)

    rng = np.random.default_rng(42)
    n_mc = 200
    perturb_pct = 0.15

    float_params = {
        'timeout_profit_lock_atr', 'timeout_extend_threshold_atr',
        'timeout_cut_threshold_atr', 'trailing_activate_atr',
        'trailing_distance_atr', 'sl_atr_mult', 'tp_atr_mult',
        'time_adaptive_trail_decay', 'time_adaptive_trail_floor',
    }
    int_params = {
        'timeout_profit_lock_bar', 'timeout_adverse_bar', 'timeout_momentum_bar',
        'timeout_extend_bars', 'timeout_cut_bars', 'time_adaptive_trail_start',
    }

    tasks = []
    for i in range(n_mc):
        kw = get_l7()
        kw.update(best_extra)
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

    with open(f"{out}/R16_F1_montecarlo.txt", 'w') as f:
        write_header(f, "R16-F1: Monte Carlo 200x ±15% Perturbation")
        f.write(f"Sharpe: mean={np.mean(sharpes):.2f}, std={np.std(sharpes):.2f}, "
                f"5th={np.percentile(sharpes, 5):.2f}, median={np.median(sharpes):.2f}, "
                f"95th={np.percentile(sharpes, 95):.2f}\n")
        f.write(f"PnL:    mean={fmt(np.mean(pnls))}, std={fmt(np.std(pnls))}, "
                f"5th={fmt(np.percentile(pnls, 5))}, median={fmt(np.median(pnls))}, "
                f"95th={fmt(np.percentile(pnls, 95))}\n\n")

        f.write(f"Sharpe distribution:\n")
        for lo, hi in [(0, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 99)]:
            cnt = len([s for s in sharpes if lo <= s < hi])
            f.write(f"  [{lo}-{hi}): {cnt:>4} ({cnt/len(sharpes)*100:.1f}%)\n")

    print(f"  F1 done: median Sharpe={np.median(sharpes):.2f}")


def phase_f2(out, best_extra):
    """Spread 敏感性"""
    print("\n" + "=" * 70)
    print("R16-F2: Spread Sensitivity")
    print("=" * 70)

    spreads = [0.20, 0.30, 0.40, 0.50, 0.60]
    tasks = []
    for sp in spreads:
        kw = get_l7()
        tasks.append((f"L7_sp{sp}", kw, sp, None, None))

        kw = get_l7()
        kw.update(best_extra)
        tasks.append((f"L7TS_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R16_F2_spread_sensitivity.txt", 'w') as f:
        write_header(f, "R16-F2: Spread Sensitivity Analysis")
        f.write(f"{'Spread':>8} {'L7_Sharpe':>10} {'L7_PnL':>12} {'L7TS_Sharpe':>12} {'L7TS_PnL':>12} {'dSh':>6}\n")
        f.write("-" * 65 + "\n")
        for sp in spreads:
            l7 = next((r for r in results if r[0] == f"L7_sp{sp}"), None)
            ts = next((r for r in results if r[0] == f"L7TS_sp{sp}"), None)
            if l7 and ts:
                dsh = ts[2] - l7[2]
                f.write(f"${sp:<7} {l7[2]:>10.2f} {fmt(l7[3]):>12} {ts[2]:>12.2f} {fmt(ts[3]):>12} {dsh:>+6.2f}\n")

    print(f"  F2 done")


def phase_f3(out, best_extra):
    """Timeout 出场占比变化"""
    print("\n" + "=" * 70)
    print("R16-F3: Timeout Exit Ratio Before vs After")
    print("=" * 70)

    tasks = []
    kw = get_l7()
    tasks.append(("L7_before", kw, 0.30, None, None))

    kw = get_l7()
    kw.update(best_extra)
    tasks.append(("L7TS_after", kw, 0.30, None, None))

    results = run_pool(tasks, _run_one_trades)

    with open(f"{out}/R16_F3_exit_ratio_change.txt", 'w') as f:
        write_header(f, "R16-F3: Timeout Exit Ratio — Before vs After")
        for r in results:
            label, n, sharpe, pnl, wr, _, maxdd, _, trades_data = r
            f.write(f"\n--- {label} (N={n}, Sharpe={sharpe:.2f}) ---\n")

            reasons = Counter(t[1].split(":")[0] for t in trades_data)
            total = sum(reasons.values())
            f.write(f"  {'Reason':<20} {'Count':>6} {'Pct':>6} {'Avg PnL':>10}\n")
            f.write("  " + "-" * 45 + "\n")
            for reason, cnt in reasons.most_common():
                rpnls = [t[0] for t in trades_data if t[1].split(":")[0] == reason]
                f.write(f"  {reason:<20} {cnt:>6} {cnt/total*100:>5.1f}% ${np.mean(rpnls):>9.2f}\n")

    print(f"  F3 done")


# ===============================================================
# Phase G: 最终确认
# ===============================================================

def phase_g1(out, best_extra):
    """最终 K-Fold 双 Spread"""
    print("\n" + "=" * 70)
    print("R16-G1: Final K-Fold Dual Spread")
    print("=" * 70)

    tasks = []
    for sp in [0.30, 0.50]:
        for fn, start, end in FOLDS:
            kw = get_l7()
            tasks.append((f"L7_{fn}_sp{sp}", kw, sp, start, end))

            kw = get_l7()
            kw.update(best_extra)
            tasks.append((f"L7TS_{fn}_sp{sp}", kw, sp, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R16_G1_final_kfold.txt", 'w') as f:
        write_header(f, "R16-G1: Final L7+TimeoutSniper K-Fold (Dual Spread)")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            f.write(f"{'Config':<15}")
            for fn, _, _ in FOLDS:
                f.write(f" {fn:>10}")
            f.write(f" {'Avg':>10} {'Win':>6}\n")
            f.write("-" * 90 + "\n")

            for prefix in ["L7", "L7TS"]:
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


def phase_g2(out, best_extra):
    """破产概率"""
    print("\n" + "=" * 70)
    print("R16-G2: Ruin Probability Simulation")
    print("=" * 70)

    from backtest.runner import DataBundle, run_variant
    kw = get_l7()
    kw.update(best_extra)
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    s = run_variant(data, "L7TS", verbose=False, spread_cost=0.30, **kw)
    trades = s.get('_trades', [])
    pnls = np.array([t.pnl for t in trades])

    capitals = [500, 1000, 1500, 2000, 3000, 5000, 10000]
    n_boot = 1000
    rng = np.random.default_rng(42)

    with open(f"{out}/R16_G2_ruin_probability.txt", 'w') as f:
        write_header(f, "R16-G2: L7+TimeoutSniper Ruin Probability",
                     f"Bootstrap {n_boot}x resampling of {len(pnls)} trades")

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

    # Phase A: 诊断
    run_phase("Phase A1: Timeout Profile", phase_a1)
    run_phase("Phase A2: Timeout Behavior", phase_a2)

    # Phase B: 锁利
    b1_results = run_phase("Phase B1: Profit Lock Scan", phase_b1)
    best_lock_atr, best_bar_off = run_phase("Phase B2: Profit Lock K-Fold", phase_b2, b1_results) or (0.15, 4)

    # Phase C: 不利信号退出
    c1_results = run_phase("Phase C1: Adverse Exit", phase_c1)
    c2_results = run_phase("Phase C2: Momentum Exit", phase_c2)
    adv_bar, mom_bar = run_phase("Phase C3: Combined Exit", phase_c3, c1_results, c2_results) or (10, 10)
    run_phase("Phase C4: Signal Exit K-Fold", phase_c4, adv_bar, mom_bar)

    # Phase D: 动态持仓
    d1_results = run_phase("Phase D1: Dynamic Timeout", phase_d1)
    best_et, best_ct = run_phase("Phase D2: Dynamic Bars Grid", phase_d2, d1_results) or (0.3, -0.5)

    # Phase E: 组合优化
    best_name, best_extra = run_phase("Phase E1: Combo Matrix", phase_e1,
                                      best_lock_atr, best_bar_off, adv_bar, mom_bar,
                                      best_et, best_ct) or ("AllThree", {})
    run_phase("Phase E2: Progressive SL", phase_e2, best_name, best_extra)
    run_phase("Phase E3: Walk-Forward", phase_e3, best_extra)

    # Phase F: 鲁棒性
    run_phase("Phase F1: Monte Carlo 200x", phase_f1, best_extra)
    run_phase("Phase F2: Spread Sensitivity", phase_f2, best_extra)
    run_phase("Phase F3: Exit Ratio Change", phase_f3, best_extra)

    # Phase G: 最终确认
    run_phase("Phase G1: Final K-Fold", phase_g1, best_extra)
    run_phase("Phase G2: Ruin Probability", phase_g2, best_extra)

    total = time.time() - t0

    with open(f"{OUTPUT_DIR}/R16_summary.txt", 'w') as f:
        f.write(f"Round 16 — Timeout Sniper\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total: {total:.0f}s ({total/3600:.1f}h)\n")
        f.write(f"Started: {start_time}\n")
        f.write(f"Best combo: {best_name}\n")
        f.write(f"Best params: lock_atr={best_lock_atr}, bar_off={best_bar_off}\n")
        f.write(f"  adv_bar={adv_bar}, mom_bar={mom_bar}\n")
        f.write(f"  extend_th={best_et}, cut_th={best_ct}\n\n")
        for name, elapsed, status in phases:
            f.write(f"{name:<40} {elapsed:>6.0f}s  {status}\n")

    print(f"\n{'='*60}")
    print(f"Round 16 COMPLETE: {total:.0f}s ({total/3600:.1f}h)")
    print(f"Best combo: {best_name}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
