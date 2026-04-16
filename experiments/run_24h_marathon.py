#!/usr/bin/env python3
"""
24-Hour Marathon Test Suite — 无人值守自动调度
==============================================
设计目标：
  1. 每个 Phase 完成后自动启动下一个
  2. 每个 Phase 输出到独立文件，便于分段下载
  3. 16 核 cgroup 配额优化（不超开进程）
  4. 总计约 20-24 小时覆盖所有高价值实验

Phase 清单：
  Phase 1: 等待当前实验完成，收集结果 (~0h, 只是检查)
  Phase 2: L3+TDTP_OFF 组合确认 + K-Fold (~1h)
  Phase 3: L3+TDTP_OFF 极端环境压力测试 (~1.5h)
  Phase 4: Regime Trail 参数精调 Sweep (~3h)
  Phase 5: MaxHold 精调 (15-25 M15 bars) (~2h)
  Phase 6: 全参数最优组合 (L4*) 确认 + 12-Fold (~3h)
  Phase 7: 近期行情 (2024-2026) 专项分析 (~1h)
  Phase 8: Monte Carlo 随机抽样稳定性 (~2h)
  Phase 9: 年度逐年分析 + Regime 画像 (~1.5h)
  Phase 10: 最终部署配置确认 vs L3 (~1h)
  Phase 11: 备选实验（如有剩余时间）(~2-3h)

预计总计: ~18-20h，留 4-6h 缓冲
"""
import sys, os, time, traceback, multiprocessing as mp
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "marathon_results"
MAX_WORKERS = 14  # cgroup 16核，留2核给系统/监控

FOLDS_6 = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-10"),
]

FOLDS_12 = [
    ("F01", "2015-01-01", "2016-01-01"),
    ("F02", "2016-01-01", "2017-01-01"),
    ("F03", "2017-01-01", "2018-01-01"),
    ("F04", "2018-01-01", "2019-01-01"),
    ("F05", "2019-01-01", "2020-01-01"),
    ("F06", "2020-01-01", "2021-01-01"),
    ("F07", "2021-01-01", "2022-01-01"),
    ("F08", "2022-01-01", "2023-01-01"),
    ("F09", "2023-01-01", "2024-01-01"),
    ("F10", "2024-01-01", "2025-01-01"),
    ("F11", "2025-01-01", "2026-01-01"),
    ("F12", "2026-01-01", "2026-04-10"),
]


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def _load_base():
    from backtest.runner import LIVE_PARITY_KWARGS
    return {**LIVE_PARITY_KWARGS}


def _run_one(args):
    """Generic worker: run one backtest variant."""
    label, extra_kwargs, spread, start, end = args
    from backtest import DataBundle, run_variant
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    if start and end:
        data = data.slice(start, end)
    BASE = _load_base()
    kwargs = {**BASE, **extra_kwargs}
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kwargs)
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    return (label, n, s['sharpe'], s['total_pnl'], s['win_rate'], avg, s['max_dd'],
            s.get('trades', []))


def _run_one_no_trades(args):
    """Worker without returning trades (lighter)."""
    r = _run_one(args)
    return r[:7]


def run_pool(tasks, workers=None):
    w = workers or min(len(tasks), MAX_WORKERS)
    with mp.Pool(w) as pool:
        return pool.map(_run_one_no_trades, tasks)


def print_table(p, results, headers=None):
    if not headers:
        headers = f"{'Variant':<25s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'$/t':>8s}  {'MaxDD':>11s}"
    p(headers)
    p("-" * len(headers))
    for r in results:
        label, n, sharpe, pnl, wr, avg, maxdd = r[:7]
        marker = r[7] if len(r) > 7 else ""
        p(f"  {label:<23s}  {n:>6d}  {sharpe:>7.2f}  {fmt(pnl)}  {wr:>5.1f}%  ${avg:>7.2f}  {fmt(maxdd)}  {marker}")


def run_kfold_comparison(p, label_base, kwargs_base, label_test, kwargs_test,
                         spread=0.30, folds=None):
    """Run K-Fold comparison between two configs."""
    if folds is None:
        folds = FOLDS_6
    tasks = []
    for fname, start, end in folds:
        tasks.append((f"KF_B_{fname}", kwargs_base, spread, start, end))
        tasks.append((f"KF_T_{fname}", kwargs_test, spread, start, end))
    results = run_pool(tasks)
    rmap = {r[0]: r for r in results}
    wins = 0
    deltas = []
    for fname, _, _ in folds:
        bs = rmap[f"KF_B_{fname}"][2]
        ts = rmap[f"KF_T_{fname}"][2]
        d = ts - bs
        won = d > 0
        if won:
            wins += 1
        deltas.append(d)
        p(f"    {fname}: {label_base}={bs:>6.2f}  {label_test}={ts:>6.2f}  "
          f"delta={d:>+.2f} {'V' if won else 'X'}")
    avg_d = sum(deltas) / len(deltas) if deltas else 0
    n_folds = len(folds)
    threshold = n_folds * 5 // 6  # ~83%
    passed = wins >= threshold
    p(f"    Result: {wins}/{n_folds} {'PASS' if passed else 'FAIL'}  avg_delta={avg_d:>+.3f}")
    return wins, n_folds, avg_d, passed


# ═══════════════════════════════════════════════════════════════
# Phase 2: L3 + TDTP_OFF 组合确认
# ═══════════════════════════════════════════════════════════════

def phase2_tdtp_off_combo(p):
    p("\n" + "=" * 80)
    p("PHASE 2: L3 + TDTP_OFF 组合确认")
    p("=" * 80)

    L3 = {}  # LIVE_PARITY_KWARGS already has L3 params
    L3_NOTDTP = {"time_decay_tp": False}

    # Full sample comparison at $0.30 and $0.50
    tasks = [
        ("L3_TDTP_ON", L3, 0.30, None, None),
        ("L3_TDTP_OFF", L3_NOTDTP, 0.30, None, None),
        ("L3_TDTP_ON_sp50", L3, 0.50, None, None),
        ("L3_TDTP_OFF_sp50", L3_NOTDTP, 0.50, None, None),
    ]
    results = run_pool(tasks)
    p("\n--- Full Sample ---")
    print_table(p, sorted(results, key=lambda x: x[0]))

    # K-Fold 6-Fold
    p("\n--- K-Fold 6-Fold @ $0.30 ---")
    run_kfold_comparison(p, "TDTP_ON", L3, "TDTP_OFF", L3_NOTDTP, spread=0.30)

    # K-Fold 6-Fold @ $0.50
    p("\n--- K-Fold 6-Fold @ $0.50 ---")
    run_kfold_comparison(p, "TDTP_ON", L3, "TDTP_OFF", L3_NOTDTP, spread=0.50)


# ═══════════════════════════════════════════════════════════════
# Phase 3: 极端环境压力测试
# ═══════════════════════════════════════════════════════════════

def phase3_stress_tests(p):
    p("\n" + "=" * 80)
    p("PHASE 3: 极端环境压力测试")
    p("  测试场景: 高 spread, 不同数据窗口, 参数扰动")
    p("=" * 80)

    L3_OFF = {"time_decay_tp": False}

    # 3a: Spread 敏感性 ($0.00 到 $1.00)
    p("\n--- 3a: Spread 敏感性 ---")
    spreads = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    tasks = [(f"SP_{sp:.2f}", L3_OFF, sp, None, None) for sp in spreads]
    results = run_pool(tasks)
    p(f"\n{'Spread':>8s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'$/t':>8s}  {'MaxDD':>11s}")
    p("-" * 72)
    for r in sorted(results, key=lambda x: float(x[0].split('_')[1])):
        sp = float(r[0].split('_')[1])
        marker = " <-- ref" if sp == 0.30 else (" <-- break-even?" if r[2] < 0.1 else "")
        p(f"  ${sp:>5.2f}  {r[1]:>6d}  {r[2]:>7.2f}  {fmt(r[3])}  {r[4]:>5.1f}%  ${r[5]:>7.2f}  {fmt(r[6])}{marker}")

    # 3b: 近期 vs 远期 表现
    p("\n--- 3b: 时段稳定性 ---")
    periods = [
        ("2015-2017", "2015-01-01", "2017-01-01"),
        ("2017-2019", "2017-01-01", "2019-01-01"),
        ("2019-2021", "2019-01-01", "2021-01-01"),
        ("2021-2023", "2021-01-01", "2023-01-01"),
        ("2023-2025", "2023-01-01", "2025-01-01"),
        ("2025-2026", "2025-01-01", "2026-04-10"),
        ("Trump2.0", "2025-01-20", "2026-04-10"),
        ("Tariff_crisis", "2025-04-01", "2026-04-10"),
    ]
    tasks = [(name, L3_OFF, 0.30, start, end) for name, start, end in periods]
    results = run_pool(tasks)
    print_table(p, sorted(results, key=lambda x: x[0]))

    # 3c: 参数扰动（参数 ±10-20% 是否稳定）
    p("\n--- 3c: 参数扰动测试 ---")
    perturbations = [
        ("Trail_tight", {"regime_config": {
            'low': {'trail_act': 0.45, 'trail_dist': 0.12},
            'normal': {'trail_act': 0.30, 'trail_dist': 0.08},
            'high': {'trail_act': 0.17, 'trail_dist': 0.025},
        }, "time_decay_tp": False}),
        ("Trail_loose", {"regime_config": {
            'low': {'trail_act': 0.55, 'trail_dist': 0.18},
            'normal': {'trail_act': 0.40, 'trail_dist': 0.12},
            'high': {'trail_act': 0.23, 'trail_dist': 0.035},
        }, "time_decay_tp": False}),
        ("SL_4.0", {"sl_atr_mult": 4.0, "time_decay_tp": False}),
        ("SL_5.0", {"sl_atr_mult": 5.0, "time_decay_tp": False}),
        ("Choppy_0.45", {"choppy_threshold": 0.45, "time_decay_tp": False}),
        ("Choppy_0.55", {"choppy_threshold": 0.55, "time_decay_tp": False}),
        ("MaxHold_16", {"keltner_max_hold_m15": 16, "time_decay_tp": False}),
        ("MaxHold_24", {"keltner_max_hold_m15": 24, "time_decay_tp": False}),
        ("ADX_16", {"keltner_adx_threshold": 16, "time_decay_tp": False}),
        ("ADX_20", {"keltner_adx_threshold": 20, "time_decay_tp": False}),
    ]
    tasks = [("L3_OFF_baseline", L3_OFF, 0.30, None, None)]
    tasks += [(name, kw, 0.30, None, None) for name, kw in perturbations]
    results = run_pool(tasks)
    results_sorted = sorted(results, key=lambda x: -x[2])
    print_table(p, results_sorted)

    # 3d: Historical spread
    p("\n--- 3d: Historical Spread 对比 ---")
    from backtest import DataBundle, run_variant
    from backtest.runner import load_spread_series
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    BASE = _load_base()
    kw = {**BASE, "time_decay_tp": False}
    try:
        ss = load_spread_series()
        s_hist = run_variant(data, "L3_OFF_hist_spread", verbose=False,
                             spread_model="historical", spread_series=ss, **kw)
        s_fixed = run_variant(data, "L3_OFF_fixed_030", verbose=False,
                              spread_cost=0.30, **kw)
        for label, s in [("Fixed_$0.30", s_fixed), ("Historical", s_hist)]:
            n = s['n']
            avg = s['total_pnl'] / n if n > 0 else 0
            p(f"  {label:<15s}  N={n}  Sharpe={s['sharpe']:.2f}  PnL={fmt(s['total_pnl'])}  MaxDD={fmt(s['max_dd'])}")
    except Exception as e:
        p(f"  Historical spread test skipped: {e}")


# ═══════════════════════════════════════════════════════════════
# Phase 4: Regime Trail 参数精调
# ═══════════════════════════════════════════════════════════════

def phase4_regime_trail_sweep(p):
    p("\n" + "=" * 80)
    p("PHASE 4: Regime Trail 参数精调")
    p("  在 L3+TDTP_OFF 基础上，微调各 regime 的 trail 参数")
    p("=" * 80)

    base_off = {"time_decay_tp": False}

    configs = []
    # High regime 最重要（64% 交易）: act 0.15-0.25, dist 0.02-0.05
    for h_act in [0.15, 0.18, 0.20, 0.22, 0.25]:
        for h_dist in [0.02, 0.03, 0.04, 0.05]:
            label = f"H{h_act:.2f}D{h_dist:.2f}"
            kw = {
                "regime_config": {
                    'low': {'trail_act': 0.50, 'trail_dist': 0.15},
                    'normal': {'trail_act': 0.35, 'trail_dist': 0.10},
                    'high': {'trail_act': h_act, 'trail_dist': h_dist},
                },
                "time_decay_tp": False,
            }
            configs.append((label, kw))

    p(f"\n--- Phase 4a: High Regime Sweep ({len(configs)} configs) ---")
    tasks = [(label, kw, 0.30, None, None) for label, kw in configs]
    results = run_pool(tasks)
    results_sorted = sorted(results, key=lambda x: -x[2])

    p(f"\n{'Config':<25s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'MaxDD':>11s}")
    p("-" * 65)
    for r in results_sorted[:10]:
        marker = " <-- L3" if "H0.20D0.03" in r[0] else ""
        p(f"  {r[0]:<23s}  {r[1]:>6d}  {r[2]:>7.2f}  {fmt(r[3])}  {fmt(r[6])}{marker}")
    p(f"  ... ({len(results_sorted)} total configs)")

    # K-Fold for top-3 non-L3 configs
    top3 = [r for r in results_sorted if "H0.20D0.03" not in r[0]][:3]
    L3_kw = {**base_off}  # uses LIVE_PARITY defaults (H0.20/D0.03)

    for r in top3:
        label = r[0]
        test_kw = dict([c for c in configs if c[0] == label][0][1])
        p(f"\n--- K-Fold: {label} vs L3 ---")
        run_kfold_comparison(p, "L3", L3_kw, label, test_kw, spread=0.30)

    # Normal regime sweep
    p(f"\n--- Phase 4b: Normal Regime Sweep ---")
    n_configs = []
    for n_act in [0.30, 0.33, 0.35, 0.38, 0.40]:
        for n_dist in [0.08, 0.10, 0.12]:
            label = f"N{n_act:.2f}D{n_dist:.2f}"
            kw = {
                "regime_config": {
                    'low': {'trail_act': 0.50, 'trail_dist': 0.15},
                    'normal': {'trail_act': n_act, 'trail_dist': n_dist},
                    'high': {'trail_act': 0.20, 'trail_dist': 0.03},
                },
                "time_decay_tp": False,
            }
            n_configs.append((label, kw))

    tasks = [(label, kw, 0.30, None, None) for label, kw in n_configs]
    results = run_pool(tasks)
    results_sorted = sorted(results, key=lambda x: -x[2])

    p(f"\n{'Config':<25s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'MaxDD':>11s}")
    p("-" * 65)
    for r in results_sorted[:10]:
        marker = " <-- L3" if "N0.35D0.10" in r[0] else ""
        p(f"  {r[0]:<23s}  {r[1]:>6d}  {r[2]:>7.2f}  {fmt(r[3])}  {fmt(r[6])}{marker}")


# ═══════════════════════════════════════════════════════════════
# Phase 5: MaxHold 精调
# ═══════════════════════════════════════════════════════════════

def phase5_maxhold_sweep(p):
    p("\n" + "=" * 80)
    p("PHASE 5: MaxHold 精调 (L3+TDTP_OFF)")
    p("=" * 80)

    values = list(range(12, 30))  # 12 to 29 M15 bars (3h to 7.25h)
    tasks = [(f"MH_{v}", {"keltner_max_hold_m15": v, "time_decay_tp": False},
              0.30, None, None) for v in values]
    results = run_pool(tasks)
    results_sorted = sorted(results, key=lambda x: int(x[0].split('_')[1]))

    p(f"\n{'MaxHold':>8s}  {'Hours':>6s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'$/t':>8s}  {'MaxDD':>11s}")
    p("-" * 72)
    best_sharpe = 0
    best_mh = 20
    for r in results_sorted:
        mh = int(r[0].split('_')[1])
        hours = mh * 0.25
        marker = " <-- L3" if mh == 20 else ""
        if r[2] > best_sharpe:
            best_sharpe = r[2]
            best_mh = mh
        p(f"  {mh:>6d}  {hours:>5.1f}h  {r[1]:>6d}  {r[2]:>7.2f}  {fmt(r[3])}  ${r[5]:>7.2f}  {fmt(r[6])}{marker}")

    if best_mh != 20:
        p(f"\n--- K-Fold: MH={best_mh} vs MH=20 ---")
        run_kfold_comparison(
            p, "MH20", {"keltner_max_hold_m15": 20, "time_decay_tp": False},
            f"MH{best_mh}", {"keltner_max_hold_m15": best_mh, "time_decay_tp": False},
            spread=0.30)
    else:
        p(f"\n  MaxHold=20 already optimal, skipping K-Fold")


# ═══════════════════════════════════════════════════════════════
# Phase 6: 最优组合 (L4*) 确认 + 12-Fold
# ═══════════════════════════════════════════════════════════════

def phase6_optimal_combo(p):
    p("\n" + "=" * 80)
    p("PHASE 6: 最优组合确认 + 12-Fold 严格验证")
    p("  基于 Phase 2-5 最优参数，组合为 L4* 候选")
    p("=" * 80)

    # L4* = L3 + TDTP_OFF (+ any Phase 4/5 improvements if found)
    # Start with confirmed improvement only
    L3 = {}
    L4_STAR = {"time_decay_tp": False}

    # Full sample at 3 spread levels
    p("\n--- Full Sample @ 3 spreads ---")
    tasks = [
        ("L3_sp030", L3, 0.30, None, None),
        ("L4star_sp030", L4_STAR, 0.30, None, None),
        ("L3_sp050", L3, 0.50, None, None),
        ("L4star_sp050", L4_STAR, 0.50, None, None),
        ("L3_sp000", L3, 0.00, None, None),
        ("L4star_sp000", L4_STAR, 0.00, None, None),
    ]
    results = run_pool(tasks)
    print_table(p, sorted(results, key=lambda x: x[0]))

    # 12-Fold validation (stricter than 6-Fold)
    p("\n--- 12-Fold Validation @ $0.30 ---")
    run_kfold_comparison(p, "L3", L3, "L4*", L4_STAR, spread=0.30, folds=FOLDS_12)

    # 12-Fold @ $0.50
    p("\n--- 12-Fold Validation @ $0.50 ---")
    run_kfold_comparison(p, "L3", L3, "L4*", L4_STAR, spread=0.50, folds=FOLDS_12)


# ═══════════════════════════════════════════════════════════════
# Phase 7: 近期行情专项分析 (2024-2026)
# ═══════════════════════════════════════════════════════════════

def phase7_recent_analysis(p):
    p("\n" + "=" * 80)
    p("PHASE 7: 近期行情专项分析 (2024-2026)")
    p("  下周实盘最相关的时段深度分析")
    p("=" * 80)

    import numpy as np

    L4_STAR = {"time_decay_tp": False}

    # Run with trade records
    from backtest import DataBundle, run_variant
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    BASE = _load_base()
    kw = {**BASE, **L4_STAR}

    # 2024-2026 detailed
    recent = data.slice("2024-01-01", "2026-04-10")
    s = run_variant(recent, "L4star_2024_2026", verbose=False, spread_cost=0.30, **kw)
    trades = s.get('trades', [])

    p(f"\n  Overall 2024-2026: N={s['n']}, Sharpe={s['sharpe']:.2f}, "
      f"PnL={fmt(s['total_pnl'])}, WR={s['win_rate']:.1f}%, MaxDD={fmt(s['max_dd'])}")

    if not trades:
        p("  No trade records for detail analysis")
        return

    import pandas as pd

    # Monthly PnL
    p(f"\n--- Monthly PnL ---")
    p(f"{'Month':<10s}  {'N':>5s}  {'PnL':>10s}  {'WR%':>6s}  {'$/t':>8s}")
    p("-" * 45)
    monthly = {}
    for t in trades:
        month = pd.Timestamp(t.exit_time).strftime('%Y-%m')
        if month not in monthly:
            monthly[month] = {'n': 0, 'wins': 0, 'pnl': 0.0}
        monthly[month]['n'] += 1
        monthly[month]['pnl'] += t.pnl
        if t.pnl > 0:
            monthly[month]['wins'] += 1
    for month in sorted(monthly.keys()):
        m = monthly[month]
        wr = m['wins'] / m['n'] * 100 if m['n'] > 0 else 0
        avg = m['pnl'] / m['n'] if m['n'] > 0 else 0
        p(f"  {month:<8s}  {m['n']:>5d}  {fmt(m['pnl'])}  {wr:>5.1f}%  ${avg:>7.2f}")

    # Weekly PnL for 2026
    p(f"\n--- Weekly PnL (2026) ---")
    p(f"{'Week':<12s}  {'N':>5s}  {'PnL':>10s}  {'WR%':>6s}")
    p("-" * 40)
    weekly = {}
    for t in trades:
        ts = pd.Timestamp(t.exit_time)
        if ts.year < 2026:
            continue
        week = ts.strftime('%Y-W%V')
        if week not in weekly:
            weekly[week] = {'n': 0, 'wins': 0, 'pnl': 0.0}
        weekly[week]['n'] += 1
        weekly[week]['pnl'] += t.pnl
        if t.pnl > 0:
            weekly[week]['wins'] += 1
    for week in sorted(weekly.keys()):
        w = weekly[week]
        wr = w['wins'] / w['n'] * 100 if w['n'] > 0 else 0
        p(f"  {week:<10s}  {w['n']:>5d}  {fmt(w['pnl'])}  {wr:>5.1f}%")

    # Exit type distribution
    p(f"\n--- Exit Type Distribution (2024-2026) ---")
    p(f"{'Exit':<20s}  {'N':>5s}  {'PnL':>10s}  {'WR%':>6s}  {'$/t':>8s}  {'Bars':>6s}")
    p("-" * 60)
    by_exit = {}
    for t in trades:
        reason = t.exit_reason or 'unknown'
        if reason not in by_exit:
            by_exit[reason] = {'n': 0, 'wins': 0, 'pnl': 0.0, 'bars': []}
        by_exit[reason]['n'] += 1
        by_exit[reason]['pnl'] += t.pnl
        if t.pnl > 0:
            by_exit[reason]['wins'] += 1
        by_exit[reason]['bars'].append(t.bars_held)
    for reason in sorted(by_exit.keys(), key=lambda r: -by_exit[r]['n']):
        e = by_exit[reason]
        wr = e['wins'] / e['n'] * 100 if e['n'] > 0 else 0
        avg = e['pnl'] / e['n'] if e['n'] > 0 else 0
        avg_bars = np.mean(e['bars']) if e['bars'] else 0
        p(f"  {reason:<18s}  {e['n']:>5d}  {fmt(e['pnl'])}  {wr:>5.1f}%  ${avg:>7.2f}  {avg_bars:>5.1f}")

    # Consecutive loss analysis
    p(f"\n--- 连续亏损分析 ---")
    streaks = []
    current_streak = 0
    current_loss = 0.0
    for t in trades:
        if t.pnl < 0:
            current_streak += 1
            current_loss += t.pnl
        else:
            if current_streak > 0:
                streaks.append((current_streak, current_loss))
            current_streak = 0
            current_loss = 0.0
    if current_streak > 0:
        streaks.append((current_streak, current_loss))
    if streaks:
        max_streak = max(streaks, key=lambda x: x[0])
        max_loss = min(streaks, key=lambda x: x[1])
        p(f"  最长连亏: {max_streak[0]} 笔, 总亏损 {fmt(max_streak[1])}")
        p(f"  最大连亏金额: {max_loss[0]} 笔, 总亏损 {fmt(max_loss[1])}")
        streak_counts = {}
        for s, _ in streaks:
            streak_counts[s] = streak_counts.get(s, 0) + 1
        p(f"  连亏分布:")
        for k in sorted(streak_counts.keys()):
            p(f"    {k}连亏: {streak_counts[k]}次")

    # Strategy breakdown
    p(f"\n--- Strategy Breakdown (2024-2026) ---")
    by_strat = {}
    for t in trades:
        strat = t.strategy or 'unknown'
        if strat not in by_strat:
            by_strat[strat] = {'n': 0, 'wins': 0, 'pnl': 0.0}
        by_strat[strat]['n'] += 1
        by_strat[strat]['pnl'] += t.pnl
        if t.pnl > 0:
            by_strat[strat]['wins'] += 1
    for strat, d in sorted(by_strat.items(), key=lambda x: -x[1]['n']):
        wr = d['wins'] / d['n'] * 100 if d['n'] > 0 else 0
        avg = d['pnl'] / d['n'] if d['n'] > 0 else 0
        p(f"  {strat}: N={d['n']}, PnL={fmt(d['pnl'])}, WR={wr:.1f}%, $/t=${avg:.2f}")

    # Day-of-week analysis
    p(f"\n--- Day-of-Week (2024-2026) ---")
    by_dow = {}
    dow_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
    for t in trades:
        dow = pd.Timestamp(t.entry_time).dayofweek
        name = dow_names.get(dow, str(dow))
        if name not in by_dow:
            by_dow[name] = {'n': 0, 'pnl': 0.0, 'wins': 0}
        by_dow[name]['n'] += 1
        by_dow[name]['pnl'] += t.pnl
        if t.pnl > 0:
            by_dow[name]['wins'] += 1
    for name in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']:
        if name in by_dow:
            d = by_dow[name]
            wr = d['wins'] / d['n'] * 100 if d['n'] > 0 else 0
            avg = d['pnl'] / d['n'] if d['n'] > 0 else 0
            p(f"  {name}: N={d['n']}, PnL={fmt(d['pnl'])}, WR={wr:.1f}%, $/t=${avg:.2f}")

    # Hour distribution
    p(f"\n--- Entry Hour (2024-2026) ---")
    by_hour = {}
    for t in trades:
        hour = pd.Timestamp(t.entry_time).hour
        if hour not in by_hour:
            by_hour[hour] = {'n': 0, 'pnl': 0.0, 'wins': 0}
        by_hour[hour]['n'] += 1
        by_hour[hour]['pnl'] += t.pnl
        if t.pnl > 0:
            by_hour[hour]['wins'] += 1
    for h in sorted(by_hour.keys()):
        d = by_hour[h]
        wr = d['wins'] / d['n'] * 100 if d['n'] > 0 else 0
        avg = d['pnl'] / d['n'] if d['n'] > 0 else 0
        p(f"  {h:>2d}:00  N={d['n']:>4d}  PnL={fmt(d['pnl'])}  WR={wr:>5.1f}%  $/t=${avg:.2f}")


# ═══════════════════════════════════════════════════════════════
# Phase 8: Monte Carlo 随机抽样稳定性
# ═══════════════════════════════════════════════════════════════

def phase8_monte_carlo(p):
    p("\n" + "=" * 80)
    p("PHASE 8: Monte Carlo Bootstrap 稳定性")
    p("  随机抽取 80% 交易样本，重复 200 次，评估 Sharpe 分布")
    p("=" * 80)

    import numpy as np

    L4_STAR = {"time_decay_tp": False}

    from backtest import DataBundle, run_variant
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    BASE = _load_base()
    kw = {**BASE, **L4_STAR}
    s = run_variant(data, "MC_base", verbose=False, spread_cost=0.30, **kw)
    trades = s.get('trades', [])

    if not trades:
        p("  No trade records for Monte Carlo")
        return

    pnls = np.array([t.pnl for t in trades])
    n_total = len(pnls)
    n_sample = int(n_total * 0.8)
    n_bootstrap = 200

    p(f"  Total trades: {n_total}")
    p(f"  Sample size: {n_sample} (80%)")
    p(f"  Bootstrap iterations: {n_bootstrap}")

    np.random.seed(42)
    sharpes = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_total, n_sample, replace=True)
        sample = pnls[idx]
        if sample.std(ddof=1) > 0:
            sharpe = sample.mean() / sample.std(ddof=1) * np.sqrt(252 * 6)
            sharpes.append(sharpe)

    sharpes = np.array(sharpes)
    p(f"\n--- Monte Carlo Sharpe Distribution ---")
    p(f"  Mean:   {sharpes.mean():.2f}")
    p(f"  Median: {np.median(sharpes):.2f}")
    p(f"  Std:    {sharpes.std():.2f}")
    p(f"  5th %%:  {np.percentile(sharpes, 5):.2f}")
    p(f"  25th %%: {np.percentile(sharpes, 25):.2f}")
    p(f"  75th %%: {np.percentile(sharpes, 75):.2f}")
    p(f"  95th %%: {np.percentile(sharpes, 95):.2f}")
    p(f"  Min:    {sharpes.min():.2f}")
    p(f"  Max:    {sharpes.max():.2f}")
    p(f"  P(Sharpe > 1.0): {(sharpes > 1.0).mean()*100:.1f}%")
    p(f"  P(Sharpe > 2.0): {(sharpes > 2.0).mean()*100:.1f}%")
    p(f"  P(Sharpe > 0):   {(sharpes > 0).mean()*100:.1f}%")

    # Histogram
    p(f"\n  Histogram (bin width 0.2):")
    bins = np.arange(sharpes.min() - 0.1, sharpes.max() + 0.3, 0.2)
    hist, edges = np.histogram(sharpes, bins=bins)
    for i in range(len(hist)):
        bar = '#' * hist[i]
        p(f"  [{edges[i]:>5.1f}, {edges[i+1]:>5.1f}): {hist[i]:>3d} {bar}")

    # Monte Carlo drawdown
    p(f"\n--- Monte Carlo MaxDD Distribution ---")
    maxdds = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_total, n_sample, replace=True)
        sample = pnls[idx]
        cum = np.cumsum(sample)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        maxdds.append(dd.max())

    maxdds = np.array(maxdds)
    p(f"  Mean MaxDD:   {fmt(maxdds.mean())}")
    p(f"  Median MaxDD: {fmt(np.median(maxdds))}")
    p(f"  5th %%:        {fmt(np.percentile(maxdds, 5))}")
    p(f"  95th %%:       {fmt(np.percentile(maxdds, 95))}")
    p(f"  Worst case:   {fmt(maxdds.max())}")


# ═══════════════════════════════════════════════════════════════
# Phase 9: 逐年分析 + Regime 画像
# ═══════════════════════════════════════════════════════════════

def phase9_annual_regime(p):
    p("\n" + "=" * 80)
    p("PHASE 9: 逐年分析 + Regime 表现画像")
    p("=" * 80)

    L4_STAR = {"time_decay_tp": False}

    # Yearly performance
    years = [(str(y), f"{y}-01-01", f"{y+1}-01-01") for y in range(2015, 2026)]
    years.append(("2026_Q1", "2026-01-01", "2026-04-10"))

    tasks = [(f"Year_{name}", L4_STAR, 0.30, start, end) for name, start, end in years]
    results = run_pool(tasks)

    p(f"\n--- 逐年 Sharpe 和 PnL ---")
    p(f"{'Year':<12s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'$/t':>8s}  {'MaxDD':>11s}")
    p("-" * 72)
    total_pnl = 0
    neg_years = 0
    for r in sorted(results, key=lambda x: x[0]):
        year = r[0].replace("Year_", "")
        total_pnl += r[3]
        if r[3] < 0:
            neg_years += 1
        marker = " *** NEGATIVE ***" if r[3] < 0 else ""
        p(f"  {year:<10s}  {r[1]:>6d}  {r[2]:>7.2f}  {fmt(r[3])}  {r[4]:>5.1f}%  ${r[5]:>7.2f}  {fmt(r[6])}{marker}")
    p(f"\n  Total PnL: {fmt(total_pnl)}")
    p(f"  Negative years: {neg_years}/12")

    # Quarterly for 2025-2026
    p(f"\n--- 2025-2026 季度详情 ---")
    quarters = [
        ("2025Q1", "2025-01-01", "2025-04-01"),
        ("2025Q2", "2025-04-01", "2025-07-01"),
        ("2025Q3", "2025-07-01", "2025-10-01"),
        ("2025Q4", "2025-10-01", "2026-01-01"),
        ("2026Q1", "2026-01-01", "2026-04-10"),
    ]
    tasks = [(f"Q_{name}", L4_STAR, 0.30, start, end) for name, start, end in quarters]
    results = run_pool(tasks)
    p(f"{'Quarter':<10s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'$/t':>8s}")
    p("-" * 55)
    for r in sorted(results, key=lambda x: x[0]):
        name = r[0].replace("Q_", "")
        p(f"  {name:<8s}  {r[1]:>6d}  {r[2]:>7.2f}  {fmt(r[3])}  {r[4]:>5.1f}%  ${r[5]:>7.2f}")


# ═══════════════════════════════════════════════════════════════
# Phase 10: 最终部署确认
# ═══════════════════════════════════════════════════════════════

def phase10_final_deployment(p):
    p("\n" + "=" * 80)
    p("PHASE 10: 最终部署配置确认")
    p("=" * 80)

    L3 = {}
    L4_STAR = {"time_decay_tp": False}

    p("\n--- 最终对比: L3 vs L4* (L3+TDTP_OFF) ---")
    tasks = [
        ("L3_final_030", L3, 0.30, None, None),
        ("L4star_final_030", L4_STAR, 0.30, None, None),
        ("L3_final_050", L3, 0.50, None, None),
        ("L4star_final_050", L4_STAR, 0.50, None, None),
    ]
    results = run_pool(tasks)
    print_table(p, sorted(results, key=lambda x: x[0]))

    # Win/Loss stats for deployment sizing
    p("\n--- L4* 交易统计（用于实盘仓位计算）---")
    from backtest import DataBundle, run_variant
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    BASE = _load_base()
    kw = {**BASE, **L4_STAR}
    s = run_variant(data, "L4star_stats", verbose=False, spread_cost=0.30, **kw)
    trades = s.get('trades', [])
    if trades:
        import numpy as np
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]
        p(f"  Total trades: {len(trades)}")
        p(f"  Win count: {len(wins)}  ({len(wins)/len(trades)*100:.1f}%)")
        p(f"  Loss count: {len(losses)}  ({len(losses)/len(trades)*100:.1f}%)")
        p(f"  Avg win: ${np.mean(wins):.2f}")
        p(f"  Avg loss: ${np.mean(losses):.2f}")
        p(f"  Median win: ${np.median(wins):.2f}")
        p(f"  Median loss: ${np.median(losses):.2f}")
        p(f"  Max single win: ${max(wins):.2f}")
        p(f"  Max single loss: ${min(losses):.2f}")
        p(f"  Profit factor: {sum(wins)/abs(sum(losses)):.2f}")
        p(f"  Expectancy: ${np.mean([t.pnl for t in trades]):.2f}/trade")

    p("\n" + "=" * 80)
    p("推荐部署配置 (L4*):")
    p("  time_decay_tp: False  (关闭 TDTP)")
    p("  其余参数: 与 L3 相同")
    p("=" * 80)


# ═══════════════════════════════════════════════════════════════
# Phase 11: 备选实验
# ═══════════════════════════════════════════════════════════════

def phase11_optional(p):
    p("\n" + "=" * 80)
    p("PHASE 11: 备选实验")
    p("=" * 80)

    L4_STAR = {"time_decay_tp": False}

    # 11a: Cooldown 精调
    p("\n--- 11a: Cooldown Minutes Sweep ---")
    cooldowns = [0, 15, 20, 25, 30, 45, 60, 90, 120]
    tasks = [(f"CD_{cd}", {"cooldown_hours": cd / 60.0, "time_decay_tp": False},
              0.30, None, None) for cd in cooldowns]
    results = run_pool(tasks)
    results_sorted = sorted(results, key=lambda x: int(x[0].split('_')[1]))
    p(f"{'CD_min':>7s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'$/t':>8s}  {'MaxDD':>11s}")
    p("-" * 72)
    for r in results_sorted:
        cd = int(r[0].split('_')[1])
        marker = " <-- current" if cd == 30 else ""
        p(f"  {cd:>5d}  {r[1]:>6d}  {r[2]:>7.2f}  {fmt(r[3])}  {r[4]:>5.1f}%  ${r[5]:>7.2f}  {fmt(r[6])}{marker}")

    # 11b: KC EMA 和 Multiplier 微调（在 L4* 基础上）
    p("\n--- 11b: KC Parameters in L4* context ---")
    kc_configs = []
    for ema in [20, 22, 25, 28, 30]:
        for mult in [1.0, 1.1, 1.2, 1.3, 1.4]:
            kc_configs.append((f"KC_e{ema}_m{mult:.1f}", ema, mult))

    def _run_kc(args):
        label, ema, mult = args
        from backtest import DataBundle, run_variant
        data = DataBundle.load_custom(kc_ema=ema, kc_mult=mult)
        BASE = _load_base()
        kw = {**BASE, "time_decay_tp": False}
        s = run_variant(data, label, verbose=False, spread_cost=0.30, **kw)
        n = s['n']
        avg = s['total_pnl'] / n if n > 0 else 0
        return (label, n, s['sharpe'], s['total_pnl'], s['win_rate'], avg, s['max_dd'])

    with mp.Pool(MAX_WORKERS) as pool:
        results = pool.map(_run_kc, kc_configs)

    results_sorted = sorted(results, key=lambda x: -x[2])
    p(f"\n{'Config':<20s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'MaxDD':>11s}")
    p("-" * 60)
    for r in results_sorted[:15]:
        marker = " <-- current" if "e25_m1.2" in r[0] else ""
        p(f"  {r[0]:<18s}  {r[1]:>6d}  {r[2]:>7.2f}  {fmt(r[3])}  {fmt(r[6])}{marker}")

    # 11c: ORB 参数微调（如时间有余）
    p("\n--- 11c: ORB Max Hold Sweep ---")
    orb_holds = [4, 6, 8, 10, 12, 16, 20]
    tasks = [(f"ORB_{oh}", {"orb_max_hold_m15": oh, "time_decay_tp": False},
              0.30, None, None) for oh in orb_holds]
    results = run_pool(tasks)
    results_sorted = sorted(results, key=lambda x: int(x[0].split('_')[1]))
    p(f"{'ORB_MH':>8s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'MaxDD':>11s}")
    p("-" * 55)
    for r in results_sorted:
        oh = int(r[0].split('_')[1])
        p(f"  {oh:>6d}  {r[1]:>6d}  {r[2]:>7.2f}  {fmt(r[3])}  {fmt(r[6])}")


# ═══════════════════════════════════════════════════════════════
# Master Scheduler
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    master_log = os.path.join(OUTPUT_DIR, "00_master_log.txt")
    with open(master_log, 'w', encoding='utf-8') as mf:
        def mlog(msg):
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            line = f"[{ts}] {msg}"
            print(line, flush=True)
            mf.write(line + "\n")
            mf.flush()

        mlog("=" * 80)
        mlog("24-HOUR MARATHON TEST SUITE")
        mlog(f"CPUs visible: {mp.cpu_count()}, using max {MAX_WORKERS} workers")
        mlog(f"Started: {datetime.now()}")
        mlog("=" * 80)

        phases = [
            ("phase02_tdtp_combo.txt", "Phase 2: L3+TDTP_OFF 组合确认", phase2_tdtp_off_combo),
            ("phase03_stress.txt", "Phase 3: 极端环境压力测试", phase3_stress_tests),
            ("phase04_regime_trail.txt", "Phase 4: Regime Trail 精调", phase4_regime_trail_sweep),
            ("phase05_maxhold.txt", "Phase 5: MaxHold 精调", phase5_maxhold_sweep),
            ("phase06_optimal_12fold.txt", "Phase 6: 最优组合 12-Fold", phase6_optimal_combo),
            ("phase07_recent.txt", "Phase 7: 近期行情分析", phase7_recent_analysis),
            ("phase08_montecarlo.txt", "Phase 8: Monte Carlo", phase8_monte_carlo),
            ("phase09_annual.txt", "Phase 9: 逐年 + Regime", phase9_annual_regime),
            ("phase10_final.txt", "Phase 10: 最终确认", phase10_final_deployment),
            ("phase11_optional.txt", "Phase 11: 备选实验", phase11_optional),
        ]

        total_start = time.time()

        for filename, desc, func in phases:
            filepath = os.path.join(OUTPUT_DIR, filename)
            mlog(f"--- STARTING: {desc} ---")
            t0 = time.time()

            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    def p(msg=""):
                        print(msg, flush=True)
                        f.write(msg + "\n")
                        f.flush()

                    p(f"# {desc}")
                    p(f"# Started: {datetime.now()}")
                    p("")
                    func(p)
                    elapsed = time.time() - t0
                    p(f"\n# Completed: {datetime.now()}")
                    p(f"# Elapsed: {elapsed/60:.1f} minutes")

                mlog(f"--- COMPLETED: {desc} [{(time.time()-t0)/60:.1f} min] ---")

            except Exception as e:
                elapsed = time.time() - t0
                mlog(f"--- FAILED: {desc} [{elapsed/60:.1f} min] ERROR: {e} ---")
                with open(filepath, 'a', encoding='utf-8') as f:
                    f.write(f"\n\n# FAILED: {e}\n")
                    f.write(traceback.format_exc())
                continue

        total_elapsed = time.time() - total_start
        mlog("=" * 80)
        mlog(f"ALL PHASES COMPLETE")
        mlog(f"Total runtime: {total_elapsed/3600:.1f} hours ({total_elapsed/60:.1f} min)")
        mlog(f"Completed: {datetime.now()}")
        mlog("=" * 80)

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
