#!/usr/bin/env python3
"""
Round 2 Experiments — 围绕 L4* 深挖 + 新方向探索
=================================================
Phase 4 发现 High Regime Trail H0.15/D0.02 K-Fold 6/6，Normal N0.33/D0.08 也更优。
本轮实验:
  R1: L5 组合验证 — L4* + H0.15/D0.02 + N0.33/D0.08 全样本 + 12-Fold + spread 压力
  R2: KC 参数扫描 (修复 pickle 错误) — EMA/Mult 网格在 L4* 基础上
  R3: Low Regime Trail 精调 — 当前 L0.50/D0.15 是否还可以收紧
  R4: SL/TP 精调 — SL 3.5~5.5 步进 0.5, TP 6~10 步进 1
  R5: 滑动窗口 Walk-Forward — 2年训练/1年测试滚动验证
  R6: 最终 L5 vs L4* vs L3 全面对比（含 historical spread）
  R7: L5 出场效率审计 — 详细分析各出场类型在不同 regime 下的表现
"""
import sys, os, time, traceback
import multiprocessing as mp
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "results/round2_results"
MAX_WORKERS = 14

FOLDS_6 = [
    ("2015-01-01", "2016-12-31", "2017-01-01", "2018-12-31"),
    ("2017-01-01", "2018-12-31", "2019-01-01", "2020-12-31"),
    ("2019-01-01", "2020-12-31", "2021-01-01", "2022-12-31"),
    ("2021-01-01", "2022-12-31", "2023-01-01", "2024-12-31"),
    ("2015-01-01", "2018-12-31", "2019-01-01", "2022-12-31"),
    ("2019-01-01", "2022-12-31", "2023-01-01", "2026-04-10"),
]

FOLDS_12 = [
    ("2015-01-01", "2015-12-31", "2016-01-01", "2016-12-31"),
    ("2016-01-01", "2016-12-31", "2017-01-01", "2017-12-31"),
    ("2017-01-01", "2017-12-31", "2018-01-01", "2018-12-31"),
    ("2018-01-01", "2018-12-31", "2019-01-01", "2019-12-31"),
    ("2019-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
    ("2020-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2021-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2022-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2023-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2024-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("2015-01-01", "2020-06-30", "2020-07-01", "2026-04-10"),
    ("2018-01-01", "2022-06-30", "2022-07-01", "2026-04-10"),
]


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def _run_one(args):
    """Worker: (label, engine_kwargs, spread, start, end) -> stats tuple"""
    label, kw, spread, start, end = args
    from backtest import DataBundle, run_variant
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    if start and end:
        data = data.slice(start, end)
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kw)
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'])


def run_pool(tasks):
    with mp.Pool(min(MAX_WORKERS, len(tasks))) as pool:
        return pool.map(_run_one, tasks)


def _run_one_with_trades(args):
    """Worker that returns trades list too."""
    label, kw, spread, start, end = args
    from backtest import DataBundle, run_variant
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    if start and end:
        data = data.slice(start, end)
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kw)
    trades = s.get('_trades', [])
    trade_data = [(t.pnl, t.exit_reason, t.bars_held, t.strategy,
                   str(t.entry_time), str(t.exit_time)) for t in trades]
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'], trade_data)


# ─────────────────────────────────────────
# Base configs
# ─────────────────────────────────────────
def get_base_configs():
    from backtest.runner import LIVE_PARITY_KWARGS
    L3 = {**LIVE_PARITY_KWARGS}  # current live (TDTP ON)
    L4_STAR = {**LIVE_PARITY_KWARGS, "time_decay_tp": False}

    # L5 candidate: L4* + improved regime trails
    L5_regime = {
        'high': {'trail_act': 0.15, 'trail_dist': 0.02},
        'normal': {'trail_act': 0.33, 'trail_dist': 0.08},
        'low': {'trail_act': 0.50, 'trail_dist': 0.15},
    }
    L5 = {**L4_STAR, "regime_config": {**L4_STAR.get("regime_config", {}), **L5_regime}}
    # need to merge properly
    rc = dict(L4_STAR.get("regime_config", {}))
    rc.update(L5_regime)
    L5["regime_config"] = rc

    return L3, L4_STAR, L5


# ═══════════════════════════════════════════
# R1: L5 组合验证
# ═══════════════════════════════════════════
def r1_l5_combo(p):
    p("=" * 80)
    p("R1: L5 组合验证 — L4* + H0.15/D0.02 + N0.33/D0.08")
    p("=" * 80)

    L3, L4_STAR, L5 = get_base_configs()

    # Full sample comparison
    tasks = [
        ("L3", L3, 0.30, None, None),
        ("L4*", L4_STAR, 0.30, None, None),
        ("L5", L5, 0.30, None, None),
        ("L5_sp50", L5, 0.50, None, None),
    ]
    results = run_pool(tasks)
    p(f"\n{'Config':<12s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'MaxDD':>11s}")
    p("-" * 60)
    for r in results:
        p(f"  {r[0]:<10s}  {r[1]:>6d}  {r[2]:>7.2f}  {fmt(r[3])}  {r[4]:>5.1f}%  {fmt(r[6])}")

    # 12-Fold K-Fold: L5 vs L4*
    p(f"\n--- 12-Fold: L5 vs L4* (sp $0.30) ---")
    fold_tasks = []
    for i, (_, _, ts, te) in enumerate(FOLDS_12):
        fold_tasks.append((f"L4*_F{i+1}", L4_STAR, 0.30, ts, te))
        fold_tasks.append((f"L5_F{i+1}", L5, 0.30, ts, te))
    fold_results = run_pool(fold_tasks)

    l4_folds = {r[0]: r for r in fold_results if r[0].startswith("L4*")}
    l5_folds = {r[0]: r for r in fold_results if r[0].startswith("L5")}

    pass_count = 0
    deltas = []
    for i in range(len(FOLDS_12)):
        l4r = l4_folds[f"L4*_F{i+1}"]
        l5r = l5_folds[f"L5_F{i+1}"]
        delta = l5r[2] - l4r[2]
        deltas.append(delta)
        passed = delta >= 0
        if passed:
            pass_count += 1
        marker = "V" if passed else "X"
        p(f"    Fold{i+1:>2d}: L4*={l4r[2]:>5.2f}  L5={l5r[2]:>5.2f}  delta={delta:>+.2f} {marker}")
    p(f"    Result: {pass_count}/{len(FOLDS_12)} PASS  avg_delta={np.mean(deltas):>+.3f}")

    # Spread stress for L5
    p(f"\n--- L5 Spread 压力测试 ---")
    sp_tasks = [(f"L5_sp{s:.1f}", L5, s, None, None)
                for s in [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]]
    sp_results = run_pool(sp_tasks)
    sp_results.sort(key=lambda x: float(x[0].split('sp')[1]))
    p(f"  {'Spread':<10s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'MaxDD':>11s}")
    p("-" * 50)
    for r in sp_results:
        p(f"  {r[0]:<10s}  {r[1]:>6d}  {r[2]:>7.2f}  {fmt(r[3])}  {fmt(r[6])}")


# ═══════════════════════════════════════════
# R2: KC 参数扫描
# ═══════════════════════════════════════════
def _run_kc_variant(args):
    """Worker for KC parameter sweep — must be module-level for pickling."""
    ema, mult, base_kw, spread = args
    from backtest import DataBundle, run_variant
    data = DataBundle.load_custom(kc_ema=ema, kc_mult=mult)
    label = f"KC_E{ema}_M{mult:.1f}"
    s = run_variant(data, label, verbose=False, spread_cost=spread, **base_kw)
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'])


def r2_kc_params(p):
    p("=" * 80)
    p("R2: KC 参数扫描 (L4* context)")
    p("=" * 80)

    _, L4_STAR, _ = get_base_configs()

    ema_vals = [15, 20, 25, 30, 35]
    mult_vals = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

    kc_configs = [(ema, mult, L4_STAR, 0.30) for ema in ema_vals for mult in mult_vals]

    with mp.Pool(min(MAX_WORKERS, len(kc_configs))) as pool:
        results = pool.map(_run_kc_variant, kc_configs)

    results.sort(key=lambda x: -x[2])
    p(f"\n{'Config':<18s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'MaxDD':>11s}")
    p("-" * 65)
    for r in results:
        marker = " <-- current" if "E25_M1.2" in r[0] else ""
        p(f"  {r[0]:<16s}  {r[1]:>6d}  {r[2]:>7.2f}  {fmt(r[3])}  {r[4]:>5.1f}%  {fmt(r[6])}{marker}")

    # K-Fold for top-3 non-current configs
    top3 = [r for r in results if "E25_M1.2" not in r[0]][:3]
    current = [r for r in results if "E25_M1.2" in r[0]][0] if any("E25_M1.2" in r[0] for r in results) else None

    if current and top3:
        for r in top3:
            label = r[0]
            parts = label.split('_')
            ema = int(parts[1][1:])
            mult = float(parts[2][1:])
            p(f"\n--- K-Fold: {label} vs Current (E25/M1.2) ---")

            fold_tasks_a = []
            fold_tasks_b = []
            for i, (_, _, ts, te) in enumerate(FOLDS_6):
                fold_tasks_a.append((ema, mult, L4_STAR, 0.30))
                fold_tasks_b.append((25, 1.2, L4_STAR, 0.30))

            # Run folds sequentially (different data loads per fold)
            pass_cnt = 0
            for i, (_, _, ts, te) in enumerate(FOLDS_6):
                from backtest import DataBundle, run_variant
                data_a = DataBundle.load_custom(kc_ema=ema, kc_mult=mult).slice(ts, te)
                data_b = DataBundle.load_custom(kc_ema=25, kc_mult=1.2).slice(ts, te)
                sa = run_variant(data_a, f"{label}_F{i+1}", verbose=False, spread_cost=0.30, **L4_STAR)
                sb = run_variant(data_b, f"Current_F{i+1}", verbose=False, spread_cost=0.30, **L4_STAR)
                delta = sa['sharpe'] - sb['sharpe']
                passed = delta >= 0
                if passed:
                    pass_cnt += 1
                marker = "V" if passed else "X"
                p(f"    Fold{i+1}: Cur={sb['sharpe']:>5.2f}  {label}={sa['sharpe']:>5.2f}  delta={delta:>+.2f} {marker}")
            p(f"    Result: {pass_cnt}/6 PASS")


# ═══════════════════════════════════════════
# R3: Low Regime Trail 精调
# ═══════════════════════════════════════════
def r3_low_regime(p):
    p("=" * 80)
    p("R3: Low Regime Trail 精调")
    p("  当前 L0.50/D0.15, 测试 0.35~0.65 / 0.05~0.20")
    p("=" * 80)

    _, L4_STAR, _ = get_base_configs()

    configs = []
    for l_act in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        for l_dist in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
            if l_dist >= l_act:
                continue
            rc = dict(L4_STAR.get("regime_config", {}))
            rc['low'] = {'trail_act': l_act, 'trail_dist': l_dist}
            label = f"L{l_act:.2f}D{l_dist:.2f}"
            kw = {**L4_STAR, "regime_config": rc}
            configs.append((label, kw))

    p(f"  Testing {len(configs)} configs...")
    tasks = [(label, kw, 0.30, None, None) for label, kw in configs]
    results = run_pool(tasks)
    results.sort(key=lambda x: -x[2])

    p(f"\n{'Config':<18s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'MaxDD':>11s}")
    p("-" * 65)
    for r in results[:15]:
        marker = " <-- current" if "L0.50D0.15" in r[0] else ""
        p(f"  {r[0]:<16s}  {r[1]:>6d}  {r[2]:>7.2f}  {fmt(r[3])}  {r[4]:>5.1f}%  {fmt(r[6])}{marker}")
    p(f"  ... ({len(results)} total)")


# ═══════════════════════════════════════════
# R4: SL/TP 精调
# ═══════════════════════════════════════════
def r4_sl_tp(p):
    p("=" * 80)
    p("R4: SL/TP ATR Multiplier 精调")
    p("  SL: 3.0~6.0 步进 0.5, TP: 5.0~12.0 步进 1.0")
    p("=" * 80)

    _, L4_STAR, _ = get_base_configs()

    configs = []
    for sl in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
        for tp in [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]:
            label = f"SL{sl:.1f}_TP{tp:.1f}"
            kw = {**L4_STAR, "sl_atr_mult": sl, "tp_atr_mult": tp}
            configs.append((label, kw))

    p(f"  Testing {len(configs)} configs...")
    tasks = [(label, kw, 0.30, None, None) for label, kw in configs]
    results = run_pool(tasks)
    results.sort(key=lambda x: -x[2])

    p(f"\n{'Config':<18s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'MaxDD':>11s}")
    p("-" * 65)
    for r in results[:20]:
        marker = " <-- current" if "SL4.5_TP8.0" in r[0] else ""
        p(f"  {r[0]:<16s}  {r[1]:>6d}  {r[2]:>7.2f}  {fmt(r[3])}  {r[4]:>5.1f}%  {fmt(r[6])}{marker}")
    p(f"  ... ({len(results)} total)")

    # K-Fold for top-3
    top3 = [r for r in results if "SL4.5_TP8.0" not in r[0]][:3]
    for r in top3:
        label = r[0]
        parts = label.split('_')
        sl = float(parts[0][2:])
        tp = float(parts[1][2:])
        p(f"\n--- K-Fold: {label} vs Current (SL4.5/TP8.0) ---")
        test_kw = {**L4_STAR, "sl_atr_mult": sl, "tp_atr_mult": tp}
        pass_cnt = 0
        for i, (_, _, ts, te) in enumerate(FOLDS_6):
            fold_tasks = [
                (f"Cur_F{i+1}", L4_STAR, 0.30, ts, te),
                (f"Test_F{i+1}", test_kw, 0.30, ts, te),
            ]
            fr = run_pool(fold_tasks)
            cur_s = [x for x in fr if x[0].startswith("Cur")][0][2]
            tst_s = [x for x in fr if x[0].startswith("Test")][0][2]
            delta = tst_s - cur_s
            passed = delta >= 0
            if passed:
                pass_cnt += 1
            marker = "V" if passed else "X"
            p(f"    Fold{i+1}: Cur={cur_s:>5.2f}  {label}={tst_s:>5.2f}  delta={delta:>+.2f} {marker}")
        p(f"    Result: {pass_cnt}/6 PASS")


# ═══════════════════════════════════════════
# R5: Walk-Forward 验证
# ═══════════════════════════════════════════
def r5_walk_forward(p):
    p("=" * 80)
    p("R5: 滑动窗口 Walk-Forward (2年训练 / 1年测试)")
    p("=" * 80)

    _, L4_STAR, L5 = get_base_configs()

    windows = [
        ("2015-2016→2017", "2017-01-01", "2017-12-31"),
        ("2016-2017→2018", "2018-01-01", "2018-12-31"),
        ("2017-2018→2019", "2019-01-01", "2019-12-31"),
        ("2018-2019→2020", "2020-01-01", "2020-12-31"),
        ("2019-2020→2021", "2021-01-01", "2021-12-31"),
        ("2020-2021→2022", "2022-01-01", "2022-12-31"),
        ("2021-2022→2023", "2023-01-01", "2023-12-31"),
        ("2022-2023→2024", "2024-01-01", "2024-12-31"),
        ("2023-2024→2025", "2025-01-01", "2025-12-31"),
        ("2024-2025→2026", "2026-01-01", "2026-04-10"),
    ]

    tasks = []
    for desc, ts, te in windows:
        tasks.append((f"L4*_{desc}", L4_STAR, 0.30, ts, te))
        tasks.append((f"L5_{desc}", L5, 0.30, ts, te))

    results = run_pool(tasks)
    l4_results = {r[0]: r for r in results if r[0].startswith("L4*")}
    l5_results = {r[0]: r for r in results if r[0].startswith("L5")}

    p(f"\n{'Window':<22s}  {'L4* Sharpe':>10s}  {'L5 Sharpe':>10s}  {'Delta':>8s}  {'L4* PnL':>10s}  {'L5 PnL':>10s}")
    p("-" * 75)
    l5_wins = 0
    for desc, ts, te in windows:
        l4 = l4_results[f"L4*_{desc}"]
        l5 = l5_results[f"L5_{desc}"]
        delta = l5[2] - l4[2]
        if delta >= 0:
            l5_wins += 1
        marker = " V" if delta >= 0 else " X"
        p(f"  {desc:<20s}  {l4[2]:>10.2f}  {l5[2]:>10.2f}  {delta:>+7.2f}{marker}  {fmt(l4[3])}  {fmt(l5[3])}")
    p(f"\n  L5 wins: {l5_wins}/{len(windows)} OOS periods")


# ═══════════════════════════════════════════
# R6: 最终 L5 vs L4* vs L3 全面对比
# ═══════════════════════════════════════════
def r6_final_comparison(p):
    p("=" * 80)
    p("R6: L5 vs L4* vs L3 全面对比 (含 historical spread)")
    p("=" * 80)

    L3, L4_STAR, L5 = get_base_configs()

    # Load historical spread
    try:
        from backtest.runner import load_spread_series
        spread_series = load_spread_series()
        has_hist = True
    except Exception:
        has_hist = False

    configs = [
        ("L3_sp30", L3, 0.30),
        ("L4*_sp30", L4_STAR, 0.30),
        ("L5_sp30", L5, 0.30),
        ("L3_sp50", L3, 0.50),
        ("L4*_sp50", L4_STAR, 0.50),
        ("L5_sp50", L5, 0.50),
    ]

    tasks = [(label, kw, sp, None, None) for label, kw, sp in configs]

    if has_hist:
        for name, kw in [("L3", L3), ("L4*", L4_STAR), ("L5", L5)]:
            hist_kw = {**kw, "spread_model": "historical", "spread_series": spread_series}
            tasks.append((f"{name}_hist", hist_kw, 0.0, None, None))

    results = run_pool(tasks)
    results.sort(key=lambda x: -x[2])

    p(f"\n{'Config':<15s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'$/t':>8s}  {'MaxDD':>11s}")
    p("-" * 75)
    for r in results:
        avg_pnl = r[3] / r[1] if r[1] > 0 else 0
        p(f"  {r[0]:<13s}  {r[1]:>6d}  {r[2]:>7.2f}  {fmt(r[3])}  {r[4]:>5.1f}%  ${avg_pnl:>6.2f}  {fmt(r[6])}")


# ═══════════════════════════════════════════
# R7: 出场效率审计
# ═══════════════════════════════════════════
def r7_exit_audit(p):
    p("=" * 80)
    p("R7: L5 出场效率审计")
    p("=" * 80)

    _, _, L5 = get_base_configs()
    from backtest import DataBundle, run_variant
    import pandas as pd

    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    s = run_variant(data, "L5_audit", verbose=False, spread_cost=0.30, **L5)
    trades = s.get('_trades', [])
    p(f"\n  Total: N={s['n']}, Sharpe={s['sharpe']:.2f}, PnL={fmt(s['total_pnl'])}")
    p(f"  Trade records: {len(trades)}")

    if not trades:
        p("  No trade records!")
        return

    # Categorize exit reasons
    categories = {}
    for t in trades:
        reason = t.exit_reason or 'unknown'
        if 'Trailing' in reason:
            cat = 'Trailing'
        elif 'SL' in reason:
            cat = 'SL'
        elif 'TP' in reason:
            cat = 'TP'
        elif 'Timeout' in reason:
            cat = 'Timeout'
        elif 'RSI' in reason:
            cat = 'RSI_exit'
        else:
            cat = 'Other'

        if cat not in categories:
            categories[cat] = {'n': 0, 'pnl': 0.0, 'wins': 0, 'bars': []}
        categories[cat]['n'] += 1
        categories[cat]['pnl'] += t.pnl
        if t.pnl > 0:
            categories[cat]['wins'] += 1
        categories[cat]['bars'].append(t.bars_held)

    p(f"\n--- Exit Category Summary ---")
    p(f"{'Category':<12s}  {'N':>6s}  {'%':>6s}  {'PnL':>11s}  {'WR%':>6s}  {'$/t':>8s}  {'AvgBars':>8s}")
    p("-" * 65)
    for cat in sorted(categories.keys(), key=lambda c: -categories[c]['n']):
        d = categories[cat]
        pct = d['n'] / len(trades) * 100
        wr = d['wins'] / d['n'] * 100 if d['n'] > 0 else 0
        avg_pnl = d['pnl'] / d['n'] if d['n'] > 0 else 0
        avg_bars = np.mean(d['bars']) if d['bars'] else 0
        p(f"  {cat:<10s}  {d['n']:>6d}  {pct:>5.1f}%  {fmt(d['pnl'])}  {wr:>5.1f}%  ${avg_pnl:>6.2f}  {avg_bars:>7.1f}")

    # By strategy
    p(f"\n--- By Strategy ---")
    by_strat = {}
    for t in trades:
        strat = t.strategy or 'unknown'
        if strat not in by_strat:
            by_strat[strat] = {'n': 0, 'pnl': 0.0, 'wins': 0}
        by_strat[strat]['n'] += 1
        by_strat[strat]['pnl'] += t.pnl
        if t.pnl > 0:
            by_strat[strat]['wins'] += 1
    for strat, d in sorted(by_strat.items(), key=lambda x: -x[1]['n']):
        wr = d['wins'] / d['n'] * 100 if d['n'] > 0 else 0
        avg = d['pnl'] / d['n'] if d['n'] > 0 else 0
        p(f"  {strat}: N={d['n']}, PnL={fmt(d['pnl'])}, WR={wr:.1f}%, $/t=${avg:.2f}")

    # Bars-held distribution
    p(f"\n--- Holding Period (M15 bars) ---")
    all_bars = [t.bars_held for t in trades]
    for bucket, lo, hi in [("1-2", 1, 2), ("3-4", 3, 4), ("5-8", 5, 8),
                            ("9-12", 9, 12), ("13-16", 13, 16), ("17-20", 17, 20),
                            ("21+", 21, 999)]:
        count = sum(1 for b in all_bars if lo <= b <= hi)
        pnl = sum(t.pnl for t in trades if lo <= t.bars_held <= hi)
        if count > 0:
            p(f"  {bucket:>5s} bars: N={count:>5d} ({count/len(trades)*100:>5.1f}%)  PnL={fmt(pnl)}  $/t=${pnl/count:.2f}")

    # Floating profit erosion: trades that exit at Timeout with negative PnL
    p(f"\n--- Timeout 出场分析 ---")
    timeout_trades = [t for t in trades if 'Timeout' in (t.exit_reason or '')]
    if timeout_trades:
        to_win = sum(1 for t in timeout_trades if t.pnl > 0)
        to_loss = sum(1 for t in timeout_trades if t.pnl <= 0)
        to_pnl = sum(t.pnl for t in timeout_trades)
        p(f"  Timeout total: {len(timeout_trades)}, Win={to_win}, Loss={to_loss}")
        p(f"  Timeout PnL: {fmt(to_pnl)}, $/t=${to_pnl/len(timeout_trades):.2f}")
        p(f"  Loss Timeout avg PnL: ${np.mean([t.pnl for t in timeout_trades if t.pnl <= 0]):.2f}")


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    master_log = os.path.join(OUTPUT_DIR, "00_master_log.txt")

    with open(master_log, 'w', encoding='utf-8') as mf:
        def mlog(msg):
            ts = datetime.now().strftime('%H:%M:%S')
            line = f"[{ts}] {msg}"
            print(line, flush=True)
            mf.write(line + "\n")
            mf.flush()

        mlog("=" * 60)
        mlog("ROUND 2 EXPERIMENTS")
        mlog(f"CPUs: {mp.cpu_count()}, Workers: {MAX_WORKERS}")
        mlog(f"Started: {datetime.now()}")
        mlog("=" * 60)

        phases = [
            ("r1_l5_combo.txt", "R1: L5 组合验证", r1_l5_combo),
            ("r2_kc_params.txt", "R2: KC 参数扫描", r2_kc_params),
            ("r3_low_regime.txt", "R3: Low Regime Trail", r3_low_regime),
            ("r4_sl_tp.txt", "R4: SL/TP 精调", r4_sl_tp),
            ("r5_walk_forward.txt", "R5: Walk-Forward", r5_walk_forward),
            ("r6_final_compare.txt", "R6: 最终对比", r6_final_comparison),
            ("r7_exit_audit.txt", "R7: 出场审计", r7_exit_audit),
        ]

        total_start = time.time()
        for filename, desc, func in phases:
            filepath = os.path.join(OUTPUT_DIR, filename)
            mlog(f"--- STARTING: {desc} ---")
            t0 = time.time()

            with open(filepath, 'w', encoding='utf-8') as f:
                def p(msg="", _f=f):
                    print(msg, flush=True)
                    _f.write(msg + "\n")
                    _f.flush()

                p(f"# {desc}")
                p(f"# Started: {datetime.now()}")
                p("")
                try:
                    func(p)
                except Exception as e:
                    p(f"\n# FAILED: {e}")
                    p(traceback.format_exc())
                elapsed = time.time() - t0
                p(f"\n# Completed: {datetime.now()}")
                p(f"# Elapsed: {elapsed/60:.1f} minutes")

            mlog(f"--- COMPLETED: {desc} [{(time.time()-t0)/60:.1f} min] ---")

        total_elapsed = time.time() - total_start
        mlog("=" * 60)
        mlog(f"ALL PHASES COMPLETE: {total_elapsed/60:.1f} min ({total_elapsed/3600:.1f} hours)")
        mlog(f"Completed: {datetime.now()}")
        mlog("=" * 60)

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
