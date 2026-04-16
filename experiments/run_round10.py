#!/usr/bin/env python3
"""
Round 10 — L8 候选构建 + 追踪止盈终极优化 + 新方向探索 + 鲁棒性压力测试
========================================================================
25-core server, ~30h budget

=== Phase 1: L8 候选构建与深度验证 (~8h) ===
R10-1:  L8 构建 — UT3g_micro + L7 叠加
R10-2:  L8 K-Fold + Walk-Forward (四代对比)
R10-3:  L8 Monte Carlo 500 次参数扰动
R10-4:  L8 逐年 + 月度热力图
R10-5:  L8 vs L7 极端市场分期

=== Phase 2: UT3g 边界探索 (~4h) ===
R10-6:  UT3g 精细网格 — 3 regime 独立扫描
R10-7:  非线性 Trail — 持仓时间递减
R10-8:  Trail + SL 联动 — 保本止损

=== Phase 3: 新方向探索 (~10h) ===
R10-9:  Keltner 状态机 vs 简单信号
R10-10: 多时间框架 H4 趋势过滤
R10-11: KC 参数探索 (修复 R9-15 pickle bug)
R10-12: Historical Spread 真实成本验证
R10-13: 亏损交易画像 v2
R10-14: 盘中动量衰减分析

=== Phase 4: 鲁棒性压力测试 (~8h) ===
R10-15: 破产概率 v2 — 多版本对比
R10-16: 参数悬崖检测 — L8 边界探索
R10-17: Walk-Forward 稳定性 — 不同训练窗口
R10-18: Purged K-Fold — 严格交叉验证
R10-19: Regime 转换压力测试
R10-20: 持仓重叠分析
"""
import sys, os, io, time, traceback, random
import multiprocessing as mp
import numpy as np
from datetime import datetime
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "results/round10_results"
MAX_WORKERS = 22

def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"

# ── Picklable worker functions (module-level) ──────────────────

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
           round(t.entry_price, 2), round(t.exit_price, 2))
          for t in trades[:20000]]
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'], td)

def _run_one_equity(args):
    label, kw, spread, start, end = args
    from backtest.runner import DataBundle, run_variant
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    if start and end:
        data = data.slice(start, end)
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kw)
    eq = list(s.get('_equity_curve', []))
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'], eq)

def _run_kc_variant(args):
    """Module-level KC params worker (fixes R9-15 pickle bug)."""
    label, kc_ema, kc_mult, kw, spread = args
    from backtest.runner import DataBundle, run_variant
    data = DataBundle.load_custom(kc_ema=kc_ema, kc_mult=kc_mult)
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kw)
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'])

def _run_spread_hist(args):
    """Worker with historical spread support."""
    label, kw, spread_model, start, end = args
    from backtest.runner import DataBundle, run_variant, load_spread_series
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    if start and end:
        data = data.slice(start, end)
    extra = {}
    if spread_model == "historical":
        ss = load_spread_series()
        if ss is not None:
            extra['spread_model'] = 'historical'
            extra['spread_series'] = ss
            extra['spread_cost'] = 0.30
        else:
            extra['spread_cost'] = 0.30
    else:
        extra['spread_cost'] = float(spread_model)
    s = run_variant(data, label, verbose=False, **extra, **kw)
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'])

def run_pool(tasks, func=_run_one):
    with mp.Pool(MAX_WORKERS) as pool:
        return pool.map(func, tasks)


# ── Config presets ─────────────────────────────────────────────

def get_base():
    from backtest.runner import LIVE_PARITY_KWARGS
    return {**LIVE_PARITY_KWARGS}

ULTRA2 = {
    'low': {'trail_act': 0.30, 'trail_dist': 0.06},
    'normal': {'trail_act': 0.20, 'trail_dist': 0.04},
    'high': {'trail_act': 0.08, 'trail_dist': 0.01},
}

UT3G_MICRO = {
    'low': {'trail_act': 0.22, 'trail_dist': 0.04},
    'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
    'high': {'trail_act': 0.06, 'trail_dist': 0.008},
}

def get_l6():
    L51 = get_base()
    return {**L51,
            "regime_config": ULTRA2,
            "trailing_activate_atr": ULTRA2['normal']['trail_act'],
            "trailing_distance_atr": ULTRA2['normal']['trail_dist']}

def get_l7():
    L6 = get_l6()
    return {**L6, "min_entry_gap_hours": 1.0, "keltner_adx_threshold": 14}

def get_l8():
    L7 = get_l7()
    return {**L7,
            "regime_config": UT3G_MICRO,
            "trailing_activate_atr": UT3G_MICRO['normal']['trail_act'],
            "trailing_distance_atr": UT3G_MICRO['normal']['trail_dist']}

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-01"),
]

def run_kfold(base_kw, label_prefix, spread=0.30):
    tasks = [(f"{label_prefix}_{fn}", {**base_kw}, spread, s, e) for fn, s, e in FOLDS]
    return run_pool(tasks)

def print_kfold(p, base_results, test_results, base_label, test_label):
    p(f"\n  {'Fold':<10}{base_label:>20} Sharpe {test_label:>20} Sharpe {'Delta':>10}  Pass?")
    p(f"  {'-'*65}")
    for i, fn in enumerate([f[0] for f in FOLDS]):
        bs = base_results[i][2]; ts = test_results[i][2]
        d = ts - bs; ok = "YES" if d > -1.0 else "NO"
        p(f"  {fn:<10}{bs:>20.2f} {ts:>20.2f} {d:>+10.2f}    {ok}")
    passed = sum(1 for i in range(len(FOLDS)) if test_results[i][2] - base_results[i][2] > -1.0)
    p(f"\n  K-Fold: {passed}/{len(FOLDS)} PASS")
    return passed


# ══════════════════════════════════════════════════════════════════
# Phase 1: L8 候选构建
# ══════════════════════════════════════════════════════════════════

def r10_1_l8_construction(p):
    """L8 = L7 + UT3g_micro trailing + optional micro-improvements"""
    p("="*80)
    p("R10-1: L8 候选构建 — UT3g_micro + L7 叠加")
    p("="*80)

    L7 = get_l7()
    L8 = get_l8()

    variants = {
        "L7_base":           {**L7},
        "L8_basic":          {**L8},
        "L8_Ch55":           {**L8, "choppy_threshold": 0.55},
        "L8_CD0":            {**L8, "cooldown_hours": 0.001},
        "L8_RSI30":          {**L8, "rsi_adx_filter": 30},
        "L8_ORB12":          {**L8, "orb_max_hold_m15": 12},
        "L8_max":            {**L8, "cooldown_hours": 0.001, "rsi_adx_filter": 30, "orb_max_hold_m15": 12},
        "L8_Ch55_max":       {**L8, "choppy_threshold": 0.55, "cooldown_hours": 0.001, "rsi_adx_filter": 30},
    }

    for sp, sp_label in [(0.30, "$0.30"), (0.50, "$0.50")]:
        tasks = [(name, kw, sp, None, None) for name, kw in variants.items()]
        results = run_pool(tasks)

        p(f"\n--- 全样本 {sp_label} ---")
        p(f"  {'Variant':<24}{'N':>8}{'Sharpe':>10}{'PnL':>14}{'WR%':>8}{'MaxDD':>12}")
        for r in sorted(results, key=lambda x: x[2], reverse=True):
            p(f"  {r[0]:<24}{r[1]:>8}{r[2]:>10.2f} {fmt(r[3]):>13}{r[4]:>8.1f}% {fmt(r[6]):>11}")


def r10_2_l8_kfold(p):
    """L5.1 vs L6 vs L7 vs L8 K-Fold + Walk-Forward"""
    p("="*80)
    p("R10-2: L8 K-Fold + Walk-Forward (四代对比)")
    p("="*80)

    L51 = get_base()
    L6 = get_l6()
    L7 = get_l7()
    L8 = get_l8()

    for spread, sp_label in [(0.30, "$0.30"), (0.50, "$0.50")]:
        p(f"\n--- K-Fold {sp_label} ---")
        r_51 = run_kfold(L51, "L5.1", spread)
        r_l6 = run_kfold(L6, "L6", spread)
        r_l7 = run_kfold(L7, "L7", spread)
        r_l8 = run_kfold(L8, "L8", spread)

        p(f"\n  {'Fold':<10}{'L5.1':>8}{'L6':>8}{'L7':>8}{'L8':>8}  L7-L6  L8-L7  L8-L51")
        p(f"  {'-'*80}")
        for i, fn in enumerate([f[0] for f in FOLDS]):
            s51, sl6, sl7, sl8 = r_51[i][2], r_l6[i][2], r_l7[i][2], r_l8[i][2]
            p(f"  {fn:<10}{s51:>8.2f}{sl6:>8.2f}{sl7:>8.2f}{sl8:>8.2f}  {sl7-sl6:>+6.2f}  {sl8-sl7:>+5.2f}  {sl8-s51:>+6.2f}")

    # Walk-Forward (annual)
    p(f"\n--- Walk-Forward (年度) ---")
    years = [(str(y), f"{y}-01-01", f"{y+1}-01-01") for y in range(2015, 2026)]
    years.append(("2026", "2026-01-01", "2026-04-10"))
    tasks = []
    for yr, s, e in years:
        for lbl, kw in [("L5.1", L51), ("L6", L6), ("L7", L7), ("L8", L8)]:
            tasks.append((f"{lbl}_{yr}", {**kw}, 0.30, s, e))
    results = run_pool(tasks)
    p(f"  {'Year':<8}{'L5.1':>10}{'L6':>10}{'L7':>10}{'L8':>10}")
    p(f"  {'-'*50}")
    for i, (yr, _, _) in enumerate(years):
        r51, rl6, rl7, rl8 = results[i*4], results[i*4+1], results[i*4+2], results[i*4+3]
        p(f"  {yr:<8}{r51[2]:>10.2f}{rl6[2]:>10.2f}{rl7[2]:>10.2f}{rl8[2]:>10.2f}")


def r10_3_l8_monte_carlo(p):
    """L8 Monte Carlo 500 runs with +/-15% perturbation"""
    p("="*80)
    p("R10-3: L8 Monte Carlo 参数扰动 (500次, +/-15%)")
    p("="*80)

    L8 = get_l8()
    perturb_keys = ['trailing_activate_atr', 'trailing_distance_atr', 'sl_atr_mult',
                    'tp_atr_mult', 'choppy_threshold']

    random.seed(42)
    N = 500
    BATCH = 100
    all_results = []

    for b in range(0, N, BATCH):
        batch_end = min(b + BATCH, N)
        p(f"\n  Batch {b//BATCH+1}/{(N+BATCH-1)//BATCH} ({b}-{batch_end-1})...")
        tasks = []
        for i in range(b, batch_end):
            kw = {**L8}
            for key in perturb_keys:
                if key in kw and isinstance(kw[key], (int, float)):
                    kw[key] = round(kw[key] * random.uniform(0.85, 1.15), 4)
            if 'regime_config' in kw:
                rc = {}
                for regime, vals in kw['regime_config'].items():
                    rc[regime] = {k: round(v * random.uniform(0.85, 1.15), 4) for k, v in vals.items()}
                kw['regime_config'] = rc
            tasks.append((f"MC_{i}", kw, 0.30, None, None))
        results = run_pool(tasks)
        all_results.extend(results)
        p(f"    done ({len(all_results)}/{N})")

    sharpes = [r[2] for r in all_results]
    pnls = [r[3] for r in all_results]
    p(f"\n--- {N} 次 L8 参数扰动结果 ---")
    p(f"  Sharpe: mean={np.mean(sharpes):.2f}, std={np.std(sharpes):.2f}, min={np.min(sharpes):.2f}, max={np.max(sharpes):.2f}")
    p(f"  PnL:    mean={fmt(np.mean(pnls))}, std={fmt(np.std(pnls))}, min={fmt(np.min(pnls))}, max={fmt(np.max(pnls))}")
    p(f"  盈利组合: {sum(1 for s in sharpes if s > 0)}/{N}")
    for lo, hi in [(0,2),(2,4),(4,5),(5,6),(6,7),(7,8),(8,10),(10,100)]:
        cnt = sum(1 for s in sharpes if lo <= s < hi)
        bar = '#' * min(cnt, 200)
        p(f"  [{lo:>2}-{hi:>3}): {cnt:>4} {bar}")


def r10_4_l8_yearly_heatmap(p):
    """L8 yearly + monthly heatmap"""
    p("="*80)
    p("R10-4: L8 逐年 + 月度热力图")
    p("="*80)

    L51 = get_base(); L6 = get_l6(); L7 = get_l7(); L8 = get_l8()
    years = [(str(y), f"{y}-01-01", f"{y+1}-01-01") for y in range(2015, 2026)]
    years.append(("2026", "2026-01-01", "2026-04-10"))
    tasks = []
    for yr, s, e in years:
        for lbl, kw in [("L5.1", L51), ("L6", L6), ("L7", L7), ("L8", L8)]:
            tasks.append((f"{lbl}_{yr}", {**kw}, 0.30, s, e))
    results = run_pool(tasks)

    p(f"\n--- 逐年全对比 ---")
    p(f"  {'Year':<8}{'L5.1 Sharpe':>14}{'L5.1 PnL':>12}{'L7 Sharpe':>12}{'L7 PnL':>12}"
      f"{'L8 Sharpe':>12}{'L8 PnL':>12}{'L8-L7':>8}")
    for i, (yr, _, _) in enumerate(years):
        r51, rl6, rl7, rl8 = results[i*4], results[i*4+1], results[i*4+2], results[i*4+3]
        p(f"  {yr:<8}{r51[2]:>14.2f} {fmt(r51[3]):>11}{rl7[2]:>12.2f} {fmt(rl7[3]):>11}"
          f"{rl8[2]:>12.2f} {fmt(rl8[3]):>11} {fmt(rl8[3]-rl7[3]):>7}")

    # Monthly heatmap (L8, 2020-2026)
    p(f"\n--- 月度热力图 (L8, $0.30, 2020-2026) ---")
    month_tasks = []
    for y in range(2020, 2027):
        for m in range(1, 13):
            if y == 2026 and m > 4: break
            ms = f"{y}-{m:02d}-01"
            if m == 12:
                me = f"{y+1}-01-01"
            else:
                me = f"{y}-{m+1:02d}-01"
            month_tasks.append((f"L8_{y}_{m:02d}", {**L8}, 0.30, ms, me))
    month_results = run_pool(month_tasks)

    p(f"    {'Year':>4}" + "".join(f"{m:>7}" for m in range(1, 13)) + "     Total")
    idx = 0
    for y in range(2020, 2027):
        row = f"    {y:>4}"
        total = 0
        for m in range(1, 13):
            if y == 2026 and m > 4:
                row += "     --"
            elif idx < len(month_results):
                pnl = month_results[idx][3]
                total += pnl
                sign = "+" if pnl >= 0 else ""
                row += f" {sign}{pnl:>5.0f}"
                idx += 1
            else:
                row += "     --"
        p(row + f" {fmt(total):>9}")


def r10_5_l8_extreme_periods(p):
    """L8 vs L7 extreme market periods"""
    p("="*80)
    p("R10-5: L8 vs L7 极端市场分期")
    p("="*80)

    L51 = get_base(); L7 = get_l7(); L8 = get_l8()
    periods = [
        ("COVID_crash",   "2020-02-01", "2020-05-01"),
        ("COVID_rally",   "2020-06-01", "2020-12-31"),
        ("Rate_hike_22",  "2022-01-01", "2022-12-31"),
        ("SVB_crisis",    "2023-03-01", "2023-06-01"),
        ("Rally_2024H2",  "2024-07-01", "2024-12-31"),
        ("Tariff_2025",   "2025-03-01", "2025-06-01"),
        ("Tariff_Apr25",  "2025-04-01", "2025-05-01"),
        ("Rally_2026Q1",  "2026-01-01", "2026-04-01"),
        ("Tariff_2ndWave","2025-06-01", "2025-09-01"),
    ]
    tasks = []
    for pname, s, e in periods:
        for lbl, kw in [("L5.1", L51), ("L7", L7), ("L8", L8)]:
            tasks.append((f"{lbl}_{pname}", {**kw}, 0.30, s, e))
    results = run_pool(tasks)

    p(f"\n  {'Period':<22}{'L5.1 Sharpe':>14}{'L5.1 PnL':>12}{'L7 Sharpe':>12}{'L7 PnL':>12}{'L8 Sharpe':>12}{'L8 PnL':>12}")
    for i, (pname, _, _) in enumerate(periods):
        r51, rl7, rl8 = results[i*3], results[i*3+1], results[i*3+2]
        p(f"  {pname:<22}{r51[2]:>14.2f} {fmt(r51[3]):>11}{rl7[2]:>12.2f} {fmt(rl7[3]):>11}{rl8[2]:>12.2f} {fmt(rl8[3]):>11}")


# ══════════════════════════════════════════════════════════════════
# Phase 2: UT3g 边界探索
# ══════════════════════════════════════════════════════════════════

def r10_6_trail_grid(p):
    """Fine grid search for each regime's trail params"""
    p("="*80)
    p("R10-6: UT3g 精细网格 — 3 regime 独立扫描")
    p("="*80)

    L7 = get_l7()

    for regime_name, acts, dists in [
        ("high",   [0.04, 0.05, 0.06, 0.07, 0.08], [0.005, 0.008, 0.010, 0.012]),
        ("normal", [0.10, 0.12, 0.14, 0.16, 0.18], [0.020, 0.025, 0.030, 0.035]),
        ("low",    [0.18, 0.20, 0.22, 0.25, 0.28], [0.030, 0.040, 0.050, 0.060]),
    ]:
        p(f"\n--- {regime_name} regime grid ---")
        tasks = []
        for act in acts:
            for dist in dists:
                rc = {**UT3G_MICRO}
                rc[regime_name] = {'trail_act': act, 'trail_dist': dist}
                kw = {**L7, "regime_config": rc,
                      "trailing_activate_atr": rc['normal']['trail_act'],
                      "trailing_distance_atr": rc['normal']['trail_dist']}
                tasks.append((f"{regime_name}_A{act}_D{dist}", kw, 0.30, None, None))
        results = run_pool(tasks)
        rd = {r[0]: r for r in results}

        col_hdr = "Act\\Dist"
        header = f"  {col_hdr:<14}" + "".join(f"{d:>8}" for d in dists)
        p(header)
        best_sharpe, best_label = 0, ""
        for act in acts:
            row = f"  {act:<14}"
            for dist in dists:
                lbl = f"{regime_name}_A{act}_D{dist}"
                s = rd[lbl][2]
                row += f"{s:>8.2f}"
                if s > best_sharpe:
                    best_sharpe = s; best_label = lbl
            p(row)
        p(f"\n  Best: {best_label} Sharpe={best_sharpe:.2f}")

    # K-Fold best grid vs UT3g_micro
    p(f"\n--- Grid 最优 vs UT3g_micro K-Fold ---")
    r_ut3g = run_kfold(get_l8(), "UT3g_micro")
    p(f"  UT3g_micro K-Fold: " + ", ".join(f"{r[2]:.2f}" for r in r_ut3g))


def r10_7_time_adaptive_trail(p):
    """Time-adaptive trailing: tighten as bars_held increases"""
    p("="*80)
    p("R10-7: 非线性 Trail — 持仓时间递减")
    p("="*80)

    L8 = get_l8()
    configs = [
        ("NoAdapt",         {}),
        ("Start4_D0.95",    {"time_adaptive_trail": True, "time_adaptive_trail_start": 4, "time_adaptive_trail_decay": 0.95}),
        ("Start4_D0.90",    {"time_adaptive_trail": True, "time_adaptive_trail_start": 4, "time_adaptive_trail_decay": 0.90}),
        ("Start4_D0.85",    {"time_adaptive_trail": True, "time_adaptive_trail_start": 4, "time_adaptive_trail_decay": 0.85}),
        ("Start8_D0.90",    {"time_adaptive_trail": True, "time_adaptive_trail_start": 8, "time_adaptive_trail_decay": 0.90}),
        ("Start8_D0.80",    {"time_adaptive_trail": True, "time_adaptive_trail_start": 8, "time_adaptive_trail_decay": 0.80}),
        ("Start2_D0.95",    {"time_adaptive_trail": True, "time_adaptive_trail_start": 2, "time_adaptive_trail_decay": 0.95}),
        ("Start12_D0.85",   {"time_adaptive_trail": True, "time_adaptive_trail_start": 12, "time_adaptive_trail_decay": 0.85}),
    ]
    tasks = [(name, {**L8, **kw}, 0.30, None, None) for name, kw in configs]
    results = run_pool(tasks)

    p(f"\n--- Time-Adaptive Trail 对比 ---")
    p(f"  {'Config':<20}{'N':>8}{'Sharpe':>10}{'PnL':>14}{'WR%':>8}{'MaxDD':>12}")
    for r in sorted(results, key=lambda x: x[2], reverse=True):
        p(f"  {r[0]:<20}{r[1]:>8}{r[2]:>10.2f} {fmt(r[3]):>13}{r[4]:>8.1f}% {fmt(r[6]):>11}")

    best = sorted(results, key=lambda x: x[2], reverse=True)[0]
    p(f"\n  Best: {best[0]} Sharpe={best[2]:.2f}")
    if best[0] != "NoAdapt":
        best_cfg = {name: kw for name, kw in configs if name == best[0]}
        if best_cfg:
            best_kw_extra = list(best_cfg.values())[0]
            p(f"  Running K-Fold for {best[0]}...")
            r_base = run_kfold(L8, "NoAdapt")
            r_best = run_kfold({**L8, **best_kw_extra}, best[0])
            print_kfold(p, r_base, r_best, "NoAdapt", best[0])


def r10_8_breakeven_stop(p):
    """Breakeven stop: move SL to entry after profit exceeds threshold"""
    p("="*80)
    p("R10-8: Trail + SL 联动 — 保本止损")
    p("="*80)

    L8 = get_l8()
    configs = [
        ("NoBE",     {}),
        ("BE_0.3",   {"breakeven_after_atr": 0.3}),
        ("BE_0.5",   {"breakeven_after_atr": 0.5}),
        ("BE_0.8",   {"breakeven_after_atr": 0.8}),
        ("BE_1.0",   {"breakeven_after_atr": 1.0}),
        ("BE_1.5",   {"breakeven_after_atr": 1.5}),
        ("BE_2.0",   {"breakeven_after_atr": 2.0}),
    ]
    tasks = [(name, {**L8, **kw}, 0.30, None, None) for name, kw in configs]
    results = run_pool(tasks)

    p(f"\n--- Breakeven Stop 对比 ---")
    p(f"  {'Config':<16}{'N':>8}{'Sharpe':>10}{'PnL':>14}{'WR%':>8}{'MaxDD':>12}")
    for r in sorted(results, key=lambda x: x[2], reverse=True):
        p(f"  {r[0]:<16}{r[1]:>8}{r[2]:>10.2f} {fmt(r[3]):>13}{r[4]:>8.1f}% {fmt(r[6]):>11}")


# ══════════════════════════════════════════════════════════════════
# Phase 3: 新方向探索
# ══════════════════════════════════════════════════════════════════

def r10_9_state_machine(p):
    """Keltner state machine vs simple signal"""
    p("="*80)
    p("R10-9: Keltner 状态机 vs 简单信号")
    p("="*80)
    p("  NOTE: 状态机需要持续状态追踪，与 multiprocessing 不兼容。")
    p("  使用单进程顺序执行。")

    from backtest.runner import DataBundle, run_variant
    from indicators import KeltnerStateMachine, scan_all_signals
    import indicators as sm

    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    L8 = get_l8()

    # Test 1: Simple signal (baseline)
    s_simple = run_variant(data, "SimpleKC", verbose=False, spread_cost=0.30, **L8)
    p(f"\n  SimpleKC: N={s_simple['n']}, Sharpe={s_simple['sharpe']:.2f}, PnL={fmt(s_simple['total_pnl'])}")

    # Test 2: State machine — monkey-patch scan_all_signals for H1
    ksm = KeltnerStateMachine()
    original_scan = sm.scan_all_signals

    def _sm_scan(df, timeframe):
        if timeframe == 'H1':
            sig = ksm.update(df)
            return [sig] if sig else []
        return original_scan(df, timeframe)

    sm.scan_all_signals = _sm_scan
    try:
        s_sm = run_variant(data, "StateMachine", verbose=False, spread_cost=0.30, **L8)
        p(f"  StateMachine: N={s_sm['n']}, Sharpe={s_sm['sharpe']:.2f}, PnL={fmt(s_sm['total_pnl'])}")
    finally:
        sm.scan_all_signals = original_scan

    p(f"\n  Delta: Sharpe {s_sm['sharpe'] - s_simple['sharpe']:+.2f}")


def r10_10_h4_filter(p):
    """H4 trend filter using aggregated bars"""
    p("="*80)
    p("R10-10: 多时间框架 H4 趋势过滤")
    p("="*80)
    p("  NOTE: H4 过滤在引擎外实现——通过 h1_allowed_sessions 模拟。")
    p("  历史结论: 所有时间框架过滤都降低 Sharpe。验证 L8 环境下是否改变。")

    L8 = get_l8()
    configs = [
        ("No_filter",    {}),
        ("London_NY",    {"h1_allowed_sessions": list(range(7, 21))}),
        ("No_rollover",  {"h1_allowed_sessions": list(range(0, 21)) + [23]}),
        ("Extended",     {"h1_allowed_sessions": list(range(5, 22))}),
    ]
    tasks = [(name, {**L8, **kw}, 0.30, None, None) for name, kw in configs]
    results = run_pool(tasks)

    p(f"\n--- H4/Session Filter 对比 (L8) ---")
    p(f"  {'Config':<16}{'N':>8}{'Sharpe':>10}{'PnL':>14}{'WR%':>8}{'MaxDD':>12}")
    for r in sorted(results, key=lambda x: x[2], reverse=True):
        p(f"  {r[0]:<16}{r[1]:>8}{r[2]:>10.2f} {fmt(r[3]):>13}{r[4]:>8.1f}% {fmt(r[6]):>11}")


def r10_11_kc_params(p):
    """KC parameter exploration (fixes R9-15 pickle bug)"""
    p("="*80)
    p("R10-11: KC 参数探索 (修复 R9-15)")
    p("="*80)

    L8_no_kc = get_l8()
    emas = [20, 25, 30, 35, 40]
    mults = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

    tasks = []
    for ema in emas:
        for mult in mults:
            tasks.append((f"KC_E{ema}_M{mult}", ema, mult, L8_no_kc, 0.30))
    results = run_pool(tasks, func=_run_kc_variant)

    rd = {r[0]: r for r in results}
    p(f"\n--- Sharpe Grid ---")
    col_hdr = "EMA\\Mult"
    header = f"  {col_hdr:<12}" + "".join(f"{m:>8}" for m in mults)
    p(header)
    best_s, best_l = 0, ""
    for ema in emas:
        row = f"  {ema:<12}"
        for mult in mults:
            lbl = f"KC_E{ema}_M{mult}"
            s = rd[lbl][2]
            row += f"{s:>8.2f}"
            if s > best_s: best_s = s; best_l = lbl
        p(row)
    p(f"\n  Best: {best_l} Sharpe={best_s:.2f}")


def r10_12_historical_spread(p):
    """Historical spread verification"""
    p("="*80)
    p("R10-12: Historical Spread 真实成本验证")
    p("="*80)

    L51 = get_base(); L6 = get_l6(); L7 = get_l7(); L8 = get_l8()

    tasks = []
    for lbl, kw in [("L5.1", L51), ("L6", L6), ("L7", L7), ("L8", L8)]:
        tasks.append((f"{lbl}_Fixed030", kw, "0.30", None, None))
        tasks.append((f"{lbl}_Historical", kw, "historical", None, None))
    results = run_pool(tasks, func=_run_spread_hist)

    p(f"\n--- Fixed $0.30 vs Historical Spread ---")
    p(f"  {'Config':<22}{'N':>8}{'Sharpe':>10}{'PnL':>14}{'WR%':>8}{'MaxDD':>12}")
    for r in results:
        p(f"  {r[0]:<22}{r[1]:>8}{r[2]:>10.2f} {fmt(r[3]):>13}{r[4]:>8.1f}% {fmt(r[6]):>11}")


def r10_13_loss_profile(p):
    """Loss profile v2: extract features from losing trades"""
    p("="*80)
    p("R10-13: 亏损交易画像 v2")
    p("="*80)

    L8 = get_l8()
    result = run_pool([("L8_trades", {**L8}, 0.30, None, None)], func=_run_one_trades)[0]
    trades = result[7]

    winners = [t for t in trades if t[0] > 0]
    losers = [t for t in trades if t[0] <= 0]
    p(f"\n  Total: {len(trades)}, Winners: {len(winners)}, Losers: {len(losers)}")

    if losers:
        loss_pnls = [t[0] for t in losers]
        loss_bars = [t[2] for t in losers]
        win_bars = [t[2] for t in winners]
        p(f"\n  Losers avg PnL: ${np.mean(loss_pnls):.2f}, median: ${np.median(loss_pnls):.2f}")
        p(f"  Losers avg bars: {np.mean(loss_bars):.1f}, Winners avg bars: {np.mean(win_bars):.1f}")

        # Exit reason distribution for losers
        loss_reasons = Counter(t[1].split(':')[0] for t in losers)
        p(f"\n  --- 亏损出场原因分布 ---")
        for reason, cnt in loss_reasons.most_common():
            avg_pnl = np.mean([t[0] for t in losers if t[1].startswith(reason)])
            p(f"  {reason:<16}: {cnt:>5} ({cnt/len(losers)*100:>5.1f}%) avg=${avg_pnl:.2f}")

        # bars_held distribution for losers
        p(f"\n  --- 亏损持仓时间分布 ---")
        for lo, hi in [(1,3),(3,5),(5,8),(8,12),(12,16),(16,20),(20,999)]:
            cnt = sum(1 for b in loss_bars if lo <= b < hi)
            avg_pnl = np.mean([t[0] for t in losers if lo <= t[2] < hi]) if cnt > 0 else 0
            p(f"  {lo:>2}-{hi:>3} bars: {cnt:>5} ({cnt/len(losers)*100:>5.1f}%) avg=${avg_pnl:.2f}")

        # Strategy distribution for losers
        loss_strats = Counter(t[3] for t in losers)
        p(f"\n  --- 亏损策略分布 ---")
        for strat, cnt in loss_strats.most_common():
            p(f"  {strat:<16}: {cnt:>5} ({cnt/len(losers)*100:>5.1f}%)")

        # Direction distribution
        loss_dirs = Counter(t[5] for t in losers)
        win_dirs = Counter(t[5] for t in winners)
        p(f"\n  --- 方向分布 ---")
        for d in ['BUY', 'SELL']:
            lc = loss_dirs.get(d, 0); wc = win_dirs.get(d, 0)
            wr = wc / (wc + lc) * 100 if (wc + lc) > 0 else 0
            p(f"  {d}: Win={wc}, Loss={lc}, WR={wr:.1f}%")


def r10_14_momentum_decay(p):
    """Post-entry momentum decay analysis"""
    p("="*80)
    p("R10-14: 盘中动量衰减分析")
    p("="*80)

    L8 = get_l8()
    result = run_pool([("L8_trades", {**L8}, 0.30, None, None)], func=_run_one_trades)[0]
    trades = result[7]

    # Analyze bars_held vs PnL relationship
    p(f"\n  --- bars_held vs avg PnL ---")
    p(f"  {'Bars':<8}{'Count':>8}{'Avg PnL':>12}{'WR%':>8}{'Total PnL':>14}")
    for b in range(1, 21):
        subset = [t for t in trades if t[2] == b]
        if not subset: continue
        avg = np.mean([t[0] for t in subset])
        wr = sum(1 for t in subset if t[0] > 0) / len(subset) * 100
        total = sum(t[0] for t in subset)
        p(f"  {b:<8}{len(subset):>8}{avg:>12.2f}{wr:>8.1f}% {fmt(total):>13}")

    # Cumulative PnL by bars_held
    p(f"\n  --- 持仓时间累积 PnL ---")
    all_pnls = sorted(trades, key=lambda t: t[2])
    cum = 0
    for b in [1,2,3,4,5,8,10,12,15,20]:
        sub = [t[0] for t in trades if t[2] <= b]
        p(f"  <=  {b:>2} bars: {len(sub):>6} trades, total PnL={fmt(sum(sub))}")


# ══════════════════════════════════════════════════════════════════
# Phase 4: 鲁棒性压力测试
# ══════════════════════════════════════════════════════════════════

def r10_15_bankruptcy(p):
    """Bankruptcy probability v2 — multi-version comparison"""
    p("="*80)
    p("R10-15: 破产概率 v2 — 多版本对比")
    p("="*80)

    for lbl, kw in [("L5.1", get_base()), ("L6", get_l6()), ("L7", get_l7()), ("L8", get_l8())]:
        result = run_pool([(f"{lbl}_trades", {**kw}, 0.30, None, None)], func=_run_one_trades)[0]
        trade_pnls = [t[0] for t in result[7]]
        if not trade_pnls: continue

        capitals = [500, 1000, 2000, 3000, 5000]
        N_SIM = 10000
        random.seed(42)

        p(f"\n  --- {lbl} (N={len(trade_pnls)}, avg=${np.mean(trade_pnls):.2f}, WR={sum(1 for x in trade_pnls if x>0)/len(trade_pnls)*100:.1f}%) ---")
        p(f"  {'Capital':>10}{'Bankrupt%':>12}{'Median End':>14}{'Max DD':>12}")
        p(f"  {'-'*50}")
        for cap in capitals:
            bankrupt = 0
            ends = []
            max_dds = []
            for _ in range(N_SIM):
                equity = cap
                peak = cap
                max_dd = 0
                for _ in range(len(trade_pnls)):
                    equity += random.choice(trade_pnls)
                    if equity > peak: peak = equity
                    dd = peak - equity
                    if dd > max_dd: max_dd = dd
                    if equity <= 0:
                        bankrupt += 1
                        break
                ends.append(max(0, equity))
                max_dds.append(max_dd)
            p(f"  {fmt(cap):>10}{bankrupt/N_SIM*100:>11.1f}% {fmt(np.median(ends)):>13} {fmt(np.median(max_dds)):>11}")


def r10_16_param_cliff(p):
    """Parameter cliff detection — single-parameter sweeps"""
    p("="*80)
    p("R10-16: 参数悬崖检测 — L8 边界探索")
    p("="*80)

    L8 = get_l8()

    sweeps = [
        ("SL", "sl_atr_mult", [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0]),
        ("MaxHold", "keltner_max_hold_m15", [4, 8, 12, 16, 20, 24, 32, 48, 64]),
        ("Choppy", "choppy_threshold", [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]),
        ("ADX", "keltner_adx_threshold", [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]),
    ]

    for sweep_name, param, values in sweeps:
        tasks = [(f"{sweep_name}={v}", {**L8, param: v}, 0.30, None, None) for v in values]
        results = run_pool(tasks)

        p(f"\n--- {sweep_name} Sensitivity ---")
        p(f"  {'Value':<12}{'N':>8}{'Sharpe':>10}{'PnL':>14}{'WR%':>8}{'MaxDD':>12}")
        for r in results:
            p(f"  {r[0]:<12}{r[1]:>8}{r[2]:>10.2f} {fmt(r[3]):>13}{r[4]:>8.1f}% {fmt(r[6]):>11}")

        sharpes = [r[2] for r in results]
        p(f"  Range: {min(sharpes):.2f} - {max(sharpes):.2f}, Spread: {max(sharpes)-min(sharpes):.2f}")


def r10_17_wf_windows(p):
    """Walk-forward stability with different training windows"""
    p("="*80)
    p("R10-17: Walk-Forward 稳定性 — 不同训练窗口")
    p("="*80)

    L8 = get_l8()
    # Half-year rolling windows
    windows = []
    for y in range(2017, 2026):
        for half in [0, 1]:
            ts = f"{y}-{'01' if half==0 else '07'}-01"
            te = f"{y}-{'07' if half==0 else '12'}-31" if half==0 else f"{y+1}-01-01"
            windows.append((f"WF_{y}H{half+1}", ts, te))

    tasks = [(name, {**L8}, 0.30, s, e) for name, s, e in windows]
    results = run_pool(tasks)

    p(f"\n--- L8 半年滚动窗口 ---")
    p(f"  {'Window':<14}{'N':>8}{'Sharpe':>10}{'PnL':>14}{'WR%':>8}")
    profitable = 0
    for r in results:
        p(f"  {r[0]:<14}{r[1]:>8}{r[2]:>10.2f} {fmt(r[3]):>13}{r[4]:>8.1f}%")
        if r[3] > 0: profitable += 1
    p(f"\n  盈利窗口: {profitable}/{len(results)}")


def r10_18_purged_kfold(p):
    """Purged K-Fold with embargo gap"""
    p("="*80)
    p("R10-18: Purged K-Fold (1个月 embargo)")
    p("="*80)

    L7 = get_l7(); L8 = get_l8()

    # 8-fold with 1-month purge gaps
    purged_folds_8 = [
        ("PF1", "2015-01-01", "2016-06-01"),
        ("PF2", "2016-07-01", "2017-12-01"),
        ("PF3", "2018-01-01", "2019-06-01"),
        ("PF4", "2019-07-01", "2020-12-01"),
        ("PF5", "2021-01-01", "2022-06-01"),
        ("PF6", "2022-07-01", "2023-12-01"),
        ("PF7", "2024-01-01", "2025-03-01"),
        ("PF8", "2025-04-01", "2026-04-01"),
    ]

    for lbl, kw in [("L7", L7), ("L8", L8)]:
        p(f"\n--- {lbl} Purged 8-Fold ($0.30) ---")
        tasks = [(f"{lbl}_P8_{fn}", {**kw}, 0.30, s, e) for fn, s, e in purged_folds_8]
        results = run_pool(tasks)
        p(f"  {'Fold':<10}{'Sharpe':>10}{'PnL':>14}{'N':>8}")
        for i, r in enumerate(results):
            p(f"  {purged_folds_8[i][0]:<10}{r[2]:>10.2f} {fmt(r[3]):>13}{r[1]:>8}")
        sharpes = [r[2] for r in results]
        p(f"  Mean Sharpe: {np.mean(sharpes):.2f}, Min: {np.min(sharpes):.2f}")
        p(f"  All positive: {all(s > 0 for s in sharpes)}")


def r10_19_regime_transition(p):
    """Regime transition stress test"""
    p("="*80)
    p("R10-19: Regime 转换压力测试")
    p("="*80)

    L8 = get_l8()
    transitions = [
        ("PreCOVID→COVID",     "2019-10-01", "2020-06-01"),
        ("COVID→Recovery",     "2020-06-01", "2021-03-01"),
        ("LowVol→RateHike",   "2021-06-01", "2022-06-01"),
        ("RateHike→Pause",    "2022-06-01", "2023-06-01"),
        ("Pause→Rally24",     "2023-06-01", "2024-06-01"),
        ("Rally→Tariff",      "2024-09-01", "2025-06-01"),
        ("Tariff→2ndWave",    "2025-03-01", "2025-09-01"),
        ("2025H2→2026Q1",     "2025-07-01", "2026-04-01"),
    ]

    tasks = [(name, {**L8}, 0.30, s, e) for name, s, e in transitions]
    results = run_pool(tasks)

    p(f"\n--- Regime 转换窗口 ---")
    p(f"  {'Transition':<24}{'N':>8}{'Sharpe':>10}{'PnL':>14}{'WR%':>8}")
    for r in results:
        p(f"  {r[0]:<24}{r[1]:>8}{r[2]:>10.2f} {fmt(r[3]):>13}{r[4]:>8.1f}%")
    sharpes = [r[2] for r in results]
    p(f"\n  全部盈利: {all(s > 0 for s in sharpes)}")
    p(f"  最弱窗口: {min(results, key=lambda x: x[2])[0]} Sharpe={min(sharpes):.2f}")


def r10_20_overlap_analysis(p):
    """Strategy overlap analysis"""
    p("="*80)
    p("R10-20: 持仓重叠分析")
    p("="*80)

    L8 = get_l8()
    result = run_pool([("L8_detail", {**L8}, 0.30, None, None)], func=_run_one_trades)[0]
    trades = result[7]

    # Per-strategy stats
    strategies = Counter(t[3] for t in trades)
    p(f"\n  --- 策略贡献 ---")
    p(f"  {'Strategy':<16}{'N':>8}{'PnL':>14}{'WR%':>8}{'Avg PnL':>12}")
    for strat, cnt in strategies.most_common():
        strat_trades = [t for t in trades if t[3] == strat]
        pnl = sum(t[0] for t in strat_trades)
        wr = sum(1 for t in strat_trades if t[0] > 0) / len(strat_trades) * 100
        avg = np.mean([t[0] for t in strat_trades])
        p(f"  {strat:<16}{cnt:>8} {fmt(pnl):>13}{wr:>8.1f}% {avg:>12.2f}")

    # Time-of-day analysis (entry hour)
    p(f"\n  --- 入场时间分布 (UTC) ---")
    hour_trades = {}
    for t in trades:
        try:
            hour = int(t[4].split(' ')[1].split(':')[0])
            if hour not in hour_trades: hour_trades[hour] = []
            hour_trades[hour].append(t[0])
        except (IndexError, ValueError):
            pass
    p(f"  {'Hour':>6}{'N':>8}{'Avg PnL':>12}{'WR%':>8}{'Total PnL':>14}")
    for h in sorted(hour_trades.keys()):
        pnls = hour_trades[h]
        wr = sum(1 for x in pnls if x > 0) / len(pnls) * 100
        p(f"  {h:>6}{len(pnls):>8}{np.mean(pnls):>12.2f}{wr:>8.1f}% {fmt(sum(pnls)):>13}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

PHASES = [
    ("r10_1_l8_construction.txt",   "R10-1: L8 Construction",         r10_1_l8_construction),
    ("r10_2_l8_kfold.txt",          "R10-2: L8 K-Fold+WF",            r10_2_l8_kfold),
    ("r10_3_l8_monte_carlo.txt",    "R10-3: L8 Monte Carlo 500",      r10_3_l8_monte_carlo),
    ("r10_4_l8_yearly.txt",         "R10-4: L8 Yearly+Heatmap",       r10_4_l8_yearly_heatmap),
    ("r10_5_l8_extreme.txt",        "R10-5: L8 Extreme Periods",      r10_5_l8_extreme_periods),
    ("r10_6_trail_grid.txt",        "R10-6: Trail Grid 3-Regime",      r10_6_trail_grid),
    ("r10_7_time_trail.txt",        "R10-7: Time-Adaptive Trail",      r10_7_time_adaptive_trail),
    ("r10_8_breakeven.txt",         "R10-8: Breakeven Stop",           r10_8_breakeven_stop),
    ("r10_9_statemachine.txt",      "R10-9: State Machine",            r10_9_state_machine),
    ("r10_10_h4_filter.txt",        "R10-10: H4/Session Filter",       r10_10_h4_filter),
    ("r10_11_kc_params.txt",        "R10-11: KC Params",               r10_11_kc_params),
    ("r10_12_hist_spread.txt",      "R10-12: Historical Spread",       r10_12_historical_spread),
    ("r10_13_loss_profile.txt",     "R10-13: Loss Profile v2",         r10_13_loss_profile),
    ("r10_14_momentum.txt",         "R10-14: Momentum Decay",          r10_14_momentum_decay),
    ("r10_15_bankruptcy.txt",       "R10-15: Bankruptcy v2",           r10_15_bankruptcy),
    ("r10_16_param_cliff.txt",      "R10-16: Parameter Cliff",         r10_16_param_cliff),
    ("r10_17_wf_windows.txt",       "R10-17: WF Stability",            r10_17_wf_windows),
    ("r10_18_purged_kfold.txt",     "R10-18: Purged K-Fold",           r10_18_purged_kfold),
    ("r10_19_regime_trans.txt",     "R10-19: Regime Transition",        r10_19_regime_transition),
    ("r10_20_overlap.txt",          "R10-20: Overlap Analysis",         r10_20_overlap_analysis),
]

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    master_log = os.path.join(OUTPUT_DIR, "00_master_log.txt")

    with open(master_log, "a", encoding="utf-8") as mf:
        def mlog(msg):
            mf.write(msg + "\n"); mf.flush()

        mlog(f"Round 10 (L8 Construction + Trail Optimization + New Explorations + Robustness)")
        mlog(f"Started: {datetime.now()}")
        mlog(f"MAX_WORKERS: {MAX_WORKERS}")
        mlog(f"Server: {os.cpu_count()} cores")
        mlog("="*60)

        for fname, title, func in PHASES:
            fpath = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(fpath) and os.path.getsize(fpath) > 100:
                mlog(f"\n  {title}: SKIPPED (exists)")
                continue

            t0 = time.time()
            lines = []
            def p(msg=""):
                lines.append(msg)
                print(msg, flush=True)

            p(f"# {fname.replace('.txt','')}")
            p(f"# Started: {datetime.now()}")
            p(f"# Workers: {MAX_WORKERS}")

            try:
                func(p)
            except Exception as e:
                p(f"\n!!! ERROR !!!")
                p(traceback.format_exc())

            elapsed = (time.time() - t0) / 60
            p(f"\n# Completed: {datetime.now()}")
            p(f"# Elapsed: {elapsed:.1f} minutes")

            with open(fpath, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            mlog(f"\n  {title}: DONE ({elapsed:.1f} min)")

        mlog(f"\nRound 10 Finished: {datetime.now()}")

    print("\n" + "="*60)
    print("Round 10 ALL DONE!")
