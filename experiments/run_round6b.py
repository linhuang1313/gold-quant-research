#!/usr/bin/env python3
"""
Round 6B — Server B (32核/120GB): 探索突破
==========================================
R6-B1: L6 全面评估 (UltraTight2 + MaxPos=1)
R6-B2: 出场机制分析 (Timeout 亏损诊断)
R6-B3: 多策略组合优化
R6-B4: 参数交互效应 (Trail x Choppy 二维网格)
R6-B5: 2025-2026 近期数据放大镜
R6-B6: 年度稳定性热力图
"""
import sys, os, io, time, traceback
import multiprocessing as mp
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "results/round6_results"
MAX_WORKERS = 16


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
          for t in trades]
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
        ok = "YES" if delta > 0 else "no"
        if delta > 0: wins += 1
        p(f"  {b[0].split('_')[-1]:<8} {b[2]:>18.2f} {v[2]:>18.2f} {delta:>+10.2f} {ok:>6}")
    p(f"\n  K-Fold: {wins}/{len(base_r)} PASS")
    return wins


# ═══════════════════════════════════════════
# R6-B1: L6 全面评估
# ═══════════════════════════════════════════
def r6_b1_l6_evaluation(p):
    p("=" * 80)
    p("R6-B1: L6 全面评估 (UltraTight2 + MaxPos=1)")
    p("=" * 80)

    L5 = get_base()
    L6 = {**L5, "regime_config": ULTRA2,
          "trailing_activate_atr": 0.20, "trailing_distance_atr": 0.04,
          "max_positions": 1}

    # Part A: 逐年详细统计
    p("\n--- Part A: L5 vs L6 逐年对比 ---")
    years = list(range(2015, 2027))
    tasks = []
    for y in years:
        end = f"{y+1}-01-01" if y < 2026 else "2026-04-10"
        tasks.append((f"L5_{y}", L5, 0.30, f"{y}-01-01", end))
        tasks.append((f"L6_{y}", L6, 0.30, f"{y}-01-01", end))
    results = run_pool(tasks)

    l5_yr = [r for r in results if r[0].startswith('L5')]
    l6_yr = [r for r in results if r[0].startswith('L6')]

    p(f"\n  {'Year':<6} {'L5 N':>5} {'L5 Sharpe':>10} {'L5 PnL':>10} {'L6 N':>5} {'L6 Sharpe':>10} {'L6 PnL':>10} {'Delta PnL':>10}")
    l5_total = l6_total = 0
    for r5, r6 in zip(l5_yr, l6_yr):
        yr = r5[0].split('_')[1]
        delta = r6[3] - r5[3]
        l5_total += r5[3]
        l6_total += r6[3]
        p(f"  {yr:<6} {r5[1]:>5} {r5[2]:>10.2f} {fmt(r5[3]):>10} {r6[1]:>5} {r6[2]:>10.2f} {fmt(r6[3]):>10} {fmt(delta):>10}")
    p(f"  {'TOTAL':<6} {'':>5} {'':>10} {fmt(l5_total):>10} {'':>5} {'':>10} {fmt(l6_total):>10} {fmt(l6_total-l5_total):>10}")

    # Part B: 按季度统计
    p("\n--- Part B: L6 季度统计 ---")
    quarters = []
    for y in range(2015, 2027):
        for q in range(1, 5):
            start_m = (q - 1) * 3 + 1
            end_m = q * 3 + 1
            start = f"{y}-{start_m:02d}-01"
            if end_m > 12:
                end = f"{y+1}-01-01"
            else:
                end = f"{y}-{end_m:02d}-01"
            if y == 2026 and q > 1:
                if q == 2:
                    end = "2026-04-10"
                else:
                    continue
            quarters.append((f"L6_Q{y}Q{q}", L6, 0.30, start, end))
    results_q = run_pool(quarters)

    p(f"  {'Quarter':<12} {'N':>5} {'Sharpe':>8} {'PnL':>10} {'WR%':>6}")
    neg_q = 0
    for r in results_q:
        if r[1] > 0:
            p(f"  {r[0].replace('L6_Q',''):<12} {r[1]:>5} {r[2]:>8.2f} {fmt(r[3]):>10} {r[4]:>6.1%}")
            if r[3] < 0:
                neg_q += 1
    p(f"\n  亏损季度: {neg_q}/{len([r for r in results_q if r[1] > 0])}")

    # Part C: K-Fold $0.30 + $0.50 + $0.80
    for spread in [0.30, 0.50, 0.80]:
        p(f"\n--- Part C: K-Fold L6 vs L5 (${spread:.2f}) ---")
        base_r, var_r = run_kfold(L5, L6, spread=spread, prefix=f"L6s{int(spread*100)}_")
        print_kfold(p, base_r, var_r, f"L5 ${spread}", f"L6 ${spread}")

    # Part D: Monte Carlo Bootstrap 200 次
    p(f"\n--- Part D: Monte Carlo Bootstrap (L6, 200 runs) ---")
    mc_tasks = [(f"L6_MC_{i:03d}", L6, 0.30, None, None) for i in range(4)]
    mc_base = run_pool(mc_tasks, func=_run_one_trades)
    if mc_base and mc_base[0][7]:
        all_pnls = [t[0] for t in mc_base[0][7]]
        n_trades = len(all_pnls)
        bootstrap_sharpes = []
        np.random.seed(42)
        for _ in range(200):
            sample = np.random.choice(all_pnls, size=n_trades, replace=True)
            s_sharpe = np.mean(sample) / np.std(sample) * np.sqrt(252 * 4) if np.std(sample) > 0 else 0
            bootstrap_sharpes.append(s_sharpe)
        p(f"  Bootstrap Sharpe (200 runs, {n_trades} trades):")
        p(f"  Mean={np.mean(bootstrap_sharpes):.2f}, Std={np.std(bootstrap_sharpes):.2f}")
        p(f"  5th pct={np.percentile(bootstrap_sharpes, 5):.2f}, "
          f"25th={np.percentile(bootstrap_sharpes, 25):.2f}, "
          f"50th={np.percentile(bootstrap_sharpes, 50):.2f}, "
          f"95th={np.percentile(bootstrap_sharpes, 95):.2f}")
        p(f"  P(Sharpe>0): {sum(1 for s in bootstrap_sharpes if s > 0)/200*100:.0f}%")
        p(f"  P(Sharpe>1): {sum(1 for s in bootstrap_sharpes if s > 1)/200*100:.0f}%")

    # Part E: OOS Walk-Forward
    p(f"\n--- Part E: Anchored Walk-Forward (L6) ---")
    p(f"  {'Train':<25} {'Test':>6} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR%':>6}")
    wf_years = list(range(2016, 2027))
    wf_tasks = []
    for test_year in wf_years:
        end_str = f"{test_year+1}-01-01" if test_year < 2026 else "2026-04-10"
        wf_tasks.append((f"OOS_{test_year}", L6, 0.30, f"{test_year}-01-01", end_str))
    wf_results = run_pool(wf_tasks)
    profit_years = 0
    for r in wf_results:
        year = int(r[0].split('_')[1])
        p(f"  Train 2015-{year-1}     {year:>6} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1%}")
        if r[3] > 0: profit_years += 1
    p(f"\n  盈利年份: {profit_years}/{len(wf_results)}")


# ═══════════════════════════════════════════
# R6-B2: 出场机制分析 (Post-hoc)
# ═══════════════════════════════════════════
def r6_b2_exit_analysis(p):
    p("=" * 80)
    p("R6-B2: 出场机制分析 — Timeout 亏损诊断")
    p("=" * 80)
    p("\n  引擎不支持 timeout_pretrigger / break_even 参数")
    p("  改用 post-hoc 交易分析: 诊断 Timeout 亏损的特征\n")

    L5 = get_base()
    tasks = [("L5_full", L5, 0.30, None, None)]
    results = run_pool(tasks, func=_run_one_trades)
    trades = results[0][7]

    exit_groups = defaultdict(list)
    for t in trades:
        exit_groups[t[1]].append(t)

    p("--- Part A: 出场类型统计 ---")
    p(f"  {'Exit Reason':<20} {'N':>6} {'PnL':>12} {'Avg PnL':>10} {'WR%':>6} {'Avg Bars':>8}")
    for reason in sorted(exit_groups.keys(), key=lambda k: -len(exit_groups[k])):
        group = exit_groups[reason]
        n = len(group)
        total_pnl = sum(t[0] for t in group)
        avg_pnl = total_pnl / n
        wins = sum(1 for t in group if t[0] > 0)
        avg_bars = np.mean([t[2] for t in group])
        p(f"  {reason:<20} {n:>6} {fmt(total_pnl):>12} {fmt(avg_pnl):>10} {wins/n:>6.1%} {avg_bars:>8.1f}")

    # Timeout analysis
    timeout_trades = exit_groups.get('timeout', []) + exit_groups.get('max_hold', [])
    if timeout_trades:
        p(f"\n--- Part B: Timeout 交易深度分析 ({len(timeout_trades)} 笔) ---")
        to_pnls = [t[0] for t in timeout_trades]
        to_wins = [t for t in timeout_trades if t[0] > 0]
        to_losses = [t for t in timeout_trades if t[0] < 0]

        p(f"  盈利: {len(to_wins)} 笔, 总PnL={fmt(sum(t[0] for t in to_wins))}")
        p(f"  亏损: {len(to_losses)} 笔, 总PnL={fmt(sum(t[0] for t in to_losses))}")
        p(f"  净PnL: {fmt(sum(to_pnls))}")

        if to_losses:
            loss_pnls = [t[0] for t in to_losses]
            p(f"  亏损分布: mean={fmt(np.mean(loss_pnls))}, "
              f"median={fmt(np.median(loss_pnls))}, "
              f"worst={fmt(min(loss_pnls))}")

            mfe_data = [t[6] for t in to_losses if t[6] > 0]
            if mfe_data:
                p(f"\n  亏损单的 max_favorable (浮盈峰值):")
                p(f"  曾浮盈: {len(mfe_data)}/{len(to_losses)} ({len(mfe_data)/len(to_losses)*100:.0f}%)")
                p(f"  平均浮盈峰值: {fmt(np.mean(mfe_data))}")
                p(f"  中位浮盈峰值: {fmt(np.median(mfe_data))}")
                p(f"  -> 这些是'浮盈回吐变亏'的单子, 更紧的 trail 可能拯救它们")

        bars_held = [t[2] for t in timeout_trades]
        p(f"\n  持仓 bars 分布:")
        p(f"  mean={np.mean(bars_held):.1f}, median={np.median(bars_held):.1f}, "
          f"min={min(bars_held)}, max={max(bars_held)}")

        by_strategy = defaultdict(list)
        for t in timeout_trades:
            by_strategy[t[3]].append(t[0])
        p(f"\n  按策略分:")
        for strat, pnls_s in sorted(by_strategy.items(), key=lambda x: sum(x[1])):
            p(f"    {strat:<15} N={len(pnls_s):>4} PnL={fmt(sum(pnls_s))}")

        by_dir = defaultdict(list)
        for t in timeout_trades:
            by_dir[t[5]].append(t[0])
        p(f"\n  按方向:")
        for d, pnls_d in by_dir.items():
            p(f"    {d:<6} N={len(pnls_d):>4} PnL={fmt(sum(pnls_d))} Avg={fmt(np.mean(pnls_d))}")

    # Trail analysis
    trail_trades = exit_groups.get('trailing_stop', [])
    if trail_trades:
        p(f"\n--- Part C: Trailing Stop 效率 ({len(trail_trades)} 笔) ---")
        tr_pnls = [t[0] for t in trail_trades]
        p(f"  总PnL: {fmt(sum(tr_pnls))}")
        p(f"  WR: {sum(1 for p in tr_pnls if p > 0)/len(tr_pnls):.1%}")
        p(f"  Avg PnL: {fmt(np.mean(tr_pnls))}")
        p(f"  Avg Bars: {np.mean([t[2] for t in trail_trades]):.1f}")


# ═══════════════════════════════════════════
# R6-B3: 多策略组合优化
# ═══════════════════════════════════════════
def r6_b3_strategy_combo(p):
    p("=" * 80)
    p("R6-B3: 多策略组合优化")
    p("=" * 80)

    L5 = get_base()

    p("\n--- Part A: ORB max_hold 扫描 ---")
    orb_holds = [0, 8, 12, 16, 20, 24, 32]
    tasks = [(f"ORB_MH={h}", {**L5, "orb_max_hold_m15": h}, 0.30, None, None)
             for h in orb_holds]
    results = run_pool(tasks)
    p(f"  {'Config':<14} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR%':>6}")
    for r in results:
        p(f"  {r[0]:<14} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1%}")

    p("\n--- Part B: RSI 开关测试 ---")
    rsi_configs = [
        ("RSI_ON", L5),
        ("RSI_OFF_buy", {**L5, "rsi_buy_threshold": 999}),
        ("RSI_OFF_sell", {**L5, "rsi_sell_enabled": False}),
        ("RSI_OFF_both", {**L5, "rsi_buy_threshold": 999, "rsi_sell_enabled": False}),
    ]
    tasks = [(label, kw, 0.30, None, None) for label, kw in rsi_configs]
    results = run_pool(tasks)
    p(f"  {'Config':<14} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR%':>6}")
    for r in results:
        p(f"  {r[0]:<14} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1%}")

    p("\n--- Part C: RSI 参数调整 ---")
    rsi_thresholds = [
        ("RSI_buy5", {**L5, "rsi_buy_threshold": 5}),
        ("RSI_buy10", {**L5, "rsi_buy_threshold": 10}),
        ("RSI_buy15", {**L5, "rsi_buy_threshold": 15}),
        ("RSI_buy20", {**L5, "rsi_buy_threshold": 20}),
        ("RSI_buy25", {**L5, "rsi_buy_threshold": 25}),
    ]
    tasks = [(label, kw, 0.30, None, None) for label, kw in rsi_thresholds]
    results = run_pool(tasks)
    p(f"  {'Config':<14} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR%':>6}")
    for r in results:
        p(f"  {r[0]:<14} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1%}")

    p("\n--- Part D: Keltner ADX 阈值附近微调 ---")
    adx_values = [15, 16, 17, 18, 19, 20, 22, 25]
    tasks = [(f"ADX={a}", {**L5, "keltner_adx_threshold": a}, 0.30, None, None)
             for a in adx_values]
    results = run_pool(tasks)
    p(f"  {'Config':<10} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR%':>6}")
    for r in results:
        marker = " <-- current" if "=18" in r[0] else ""
        p(f"  {r[0]:<10} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1%}{marker}")


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
        header += f" {ch:.2f:>8}"
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
        header += f" {ch:.2f:>8}"
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

    # Recent periods
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    phases = [
        ("r6_b1_l6_eval.txt",      "R6-B1: L6 全面评估",       r6_b1_l6_evaluation),
        ("r6_b2_exit_analysis.txt", "R6-B2: 出场机制分析",      r6_b2_exit_analysis),
        ("r6_b3_strategy_combo.txt","R6-B3: 策略组合优化",       r6_b3_strategy_combo),
        ("r6_b4_interaction.txt",   "R6-B4: 参数交互效应",       r6_b4_interaction_grid),
        ("r6_b5_recent_zoom.txt",   "R6-B5: 近期数据放大镜",     r6_b5_recent_zoom),
        ("r6_b6_heatmap.txt",       "R6-B6: 月度稳定性热力图",   r6_b6_monthly_heatmap),
    ]

    master_log = os.path.join(OUTPUT_DIR, "00_master_log.txt")
    with open(master_log, 'w') as mf:
        mf.write(f"Round 6B (Server B: Exploration)\nStarted: {datetime.now()}\n{'='*60}\n\n")

        for fname, title, func in phases:
            fpath = os.path.join(OUTPUT_DIR, fname)
            print(f"\n{'='*60}")
            print(f"  Starting: {title}")
            print(f"{'='*60}\n")

            t0 = time.time()
            try:
                with open(fpath, 'w') as f:
                    header = f"# {title}\n# Started: {datetime.now()}\n# Server: B (32-core)\n\n"
                    f.write(header)
                    def printer(msg):
                        print(msg)
                        f.write(msg + "\n")
                        f.flush()
                    func(printer)
                    elapsed = time.time() - t0
                    f.write(f"\n# Completed: {datetime.now()}\n# Elapsed: {elapsed/60:.1f} minutes\n")
                status = f"DONE ({elapsed/60:.1f} min)"
            except Exception as e:
                elapsed = time.time() - t0
                status = f"FAILED ({elapsed/60:.1f} min): {e}"
                traceback.print_exc()
                with open(fpath, 'a') as f:
                    f.write(f"\n# FAILED: {e}\n{traceback.format_exc()}\n")

            mf.write(f"  {title}: {status}\n")
            mf.flush()

        mf.write(f"\nRound 6B Finished: {datetime.now()}\n")
        print(f"\n{'='*60}")
        print(f"  Round 6B COMPLETE")
        print(f"{'='*60}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
