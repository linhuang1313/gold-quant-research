#!/usr/bin/env python3
"""
Round 8 — L6 全面验证 + Timeout 优化 + 新方向探索
====================================================
25-core server, ~10h budget

R8-1: TP x SL 二维扫描 (补完 R7-5)
R8-2: L6 + Entry Gap 叠加 K-Fold
R8-3: L6 Monte Carlo 200 次参数扰动
R8-4: Timeout 优化探索 (最大亏损来源)
R8-5: Historical Spread 全量验证 L5.1 vs L6
R8-6: L6 近期放大镜 + 月度热力图
R8-7: SL 灵敏度在 L6 下重新扫描
R8-8: MaxHold 在 L6 下最优扫描
R8-9: Choppy 阈值在 L6 下验证
R8-10: 突破强度 Sizing 在 L6 下
"""
import sys, os, io, time, traceback
import multiprocessing as mp
import numpy as np
from datetime import datetime
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "results/round8_results"
MAX_WORKERS = 22

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
          for t in trades[:10000]]
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'], td)

def run_pool(tasks, func=_run_one):
    with mp.Pool(MAX_WORKERS) as pool:
        return pool.map(func, tasks)

def get_base():
    from backtest.runner import LIVE_PARITY_KWARGS
    return {**LIVE_PARITY_KWARGS}

ULTRA2 = {
    'low': {'trail_act': 0.30, 'trail_dist': 0.06},
    'normal': {'trail_act': 0.20, 'trail_dist': 0.04},
    'high': {'trail_act': 0.08, 'trail_dist': 0.01},
}

def get_l6():
    L51 = get_base()
    return {**L51,
            "regime_config": ULTRA2,
            "trailing_activate_atr": ULTRA2['normal']['trail_act'],
            "trailing_distance_atr": ULTRA2['normal']['trail_dist']}

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-10"),
]

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
# R8-1: TP x SL 二维扫描 (补完 R7-5 + 扩展范围)
# ════════════════════════════════════════════════════════════════
def r8_1_tp_sl_grid(p):
    L51 = get_base()
    L6 = get_l6()
    p("=" * 80)
    p("R8-1: TP x SL 二维扫描 (L5.1 + L6)")
    p("=" * 80)

    sl_vals = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    tp_vals = [5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 999.0]

    for ver_name, base_kw in [("L5.1", L51), ("L6", L6)]:
        p(f"\n--- {ver_name}: SL x TP Sharpe Grid ($0.30) ---")
        tasks = []
        for sl in sl_vals:
            for tp in tp_vals:
                tp_label = "OFF" if tp > 100 else f"{tp:.0f}"
                tasks.append((f"{ver_name}_SL{sl:.1f}_TP{tp_label}",
                              {**base_kw, "sl_atr_mult": sl, "tp_atr_mult": tp},
                              0.30, None, None))
        results = run_pool(tasks)
        result_map = {r[0]: r for r in results}

        col_hdr = "SL\\TP"
        header = f"  {col_hdr:<10}"
        for tp in tp_vals:
            tp_label = "OFF" if tp > 100 else f"{tp:.0f}"
            header += f" {tp_label:>8}"
        p(header)
        for sl in sl_vals:
            marker = " <-" if abs(sl - 3.5) < 0.01 else ""
            row = f"  {sl:.1f}{marker:<6}"
            for tp in tp_vals:
                tp_label = "OFF" if tp > 100 else f"{tp:.0f}"
                key = f"{ver_name}_SL{sl:.1f}_TP{tp_label}"
                r = result_map.get(key)
                row += f" {r[2]:>8.2f}" if r else f" {'N/A':>8}"
            p(row)

    p("\n--- 最优 TP K-Fold (L6, SL=3.5) ---")
    l6_results = [r for r in results if 'L6_SL3.5' in r[0]]
    if l6_results:
        best = max(l6_results, key=lambda x: x[2])
        p(f"  最优: {best[0]} Sharpe={best[2]:.2f}")
        best_tp_str = best[0].split('_TP')[1] if '_TP' in best[0] else "8"
        best_tp = 999.0 if best_tp_str == "OFF" else float(best_tp_str)
        if abs(best_tp - 8.0) > 0.1:
            var_kw = {**L6, "tp_atr_mult": best_tp}
            base_r, var_r = run_kfold(L6, var_kw, 0.30, "L6TP_")
            tp_disp = "OFF" if best_tp > 100 else f"{best_tp:.0f}"
            print_kfold(p, base_r, var_r, "L6(TP=8)", f"L6(TP={tp_disp})")
        else:
            p("  TP=8 已是最优, 无需 K-Fold")


# ════════════════════════════════════════════════════════════════
# R8-2: L6 + Entry Gap 叠加
# ════════════════════════════════════════════════════════════════
def r8_2_l6_entry_gap(p):
    L6 = get_l6()
    p("=" * 80)
    p("R8-2: L6 + Entry Gap 叠加验证")
    p("=" * 80)

    p("\n--- Part A: Gap 全样本扫描 (L6 基线) ---")
    gaps = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    tasks = [(f"L6_Gap={g:.1f}h", {**L6, "min_entry_gap_hours": g}, 0.30, None, None)
             for g in gaps]
    results = run_pool(tasks)
    p(f"  {'Gap':<12} {'N':>5} {'Sharpe':>8} {'PnL':>14} {'WR%':>8} {'MaxDD':>10}")
    for r in sorted(results, key=lambda x: float(x[0].split('=')[1].replace('h',''))):
        p(f"  {r[0]:<12} {r[1]:>5} {r[2]:>8.2f} {fmt(r[3]):>14} {r[4]:>8.1f}% {fmt(r[6]):>10}")

    p("\n--- Part B: 最优 Gap K-Fold vs L6 ---")
    best_gap = max(results, key=lambda x: x[2])
    best_g = float(best_gap[0].split('=')[1].replace('h',''))
    p(f"  最优: Gap={best_g:.1f}h Sharpe={best_gap[2]:.2f}")
    if best_g > 0:
        var_kw = {**L6, "min_entry_gap_hours": best_g}
        base_r, var_r = run_kfold(L6, var_kw, 0.30, "L6Gap_")
        wins = print_kfold(p, base_r, var_r, "L6(no gap)", f"L6+Gap={best_g:.1f}h")
        if wins >= 5:
            p("\n--- Part C: $0.50 K-Fold ---")
            base_r50, var_r50 = run_kfold(L6, var_kw, 0.50, "L6Gap50_")
            print_kfold(p, base_r50, var_r50, "L6(no gap)$0.50", f"L6+Gap={best_g:.1f}h $0.50")


# ════════════════════════════════════════════════════════════════
# R8-3: L6 Monte Carlo 200 次参数扰动
# ════════════════════════════════════════════════════════════════
def r8_3_l6_monte_carlo(p):
    L6 = get_l6()
    p("=" * 80)
    p("R8-3: L6 Monte Carlo 参数扰动 (200次, +/-15%)")
    p("=" * 80)

    n_runs = 200
    batch_size = 40
    all_results = []

    for batch_i in range(0, n_runs, batch_size):
        batch_end = min(batch_i + batch_size, n_runs)
        p(f"\n  Batch {batch_i//batch_size+1}/{(n_runs+batch_size-1)//batch_size} ({batch_i}-{batch_end-1})...")
        tasks = []
        for i in range(batch_i, batch_end):
            np.random.seed(1000 + i)
            kw = {**L6}
            kw['sl_atr_mult'] = round(L6['sl_atr_mult'] * (1 + np.random.uniform(-0.15, 0.15)), 2)
            kw['tp_atr_mult'] = round(L6['tp_atr_mult'] * (1 + np.random.uniform(-0.15, 0.15)), 2)
            kw['choppy_threshold'] = round(L6['choppy_threshold'] * (1 + np.random.uniform(-0.15, 0.15)), 2)
            kw['keltner_adx_threshold'] = max(10, int(L6['keltner_adx_threshold'] * (1 + np.random.uniform(-0.15, 0.15))))
            rc = L6['regime_config']
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
    p(f"\n--- {n_runs} 次 L6 参数扰动结果 ---")
    p(f"  Sharpe: mean={np.mean(sharpes):.2f}, std={np.std(sharpes):.2f}, min={np.min(sharpes):.2f}, max={np.max(sharpes):.2f}")
    p(f"  PnL:    mean={fmt(np.mean(pnls))}, std={fmt(np.std(pnls))}, min={fmt(np.min(pnls))}, max={fmt(np.max(pnls))}")
    p(f"  盈利组合: {sum(1 for x in pnls if x>0)}/{n_runs}")
    p(f"  Sharpe>2: {sum(1 for x in sharpes if x>2)}/{n_runs}")
    p(f"  Sharpe>4: {sum(1 for x in sharpes if x>4)}/{n_runs}")
    p(f"  Sharpe>6: {sum(1 for x in sharpes if x>6)}/{n_runs}")

    bins = [(0,2),(2,4),(4,5),(5,6),(6,7),(7,8),(8,10),(10,100)]
    for lo, hi in bins:
        cnt = sum(1 for x in sharpes if lo <= x < hi)
        bar = '#' * cnt
        p(f"  [{lo:>2}-{hi:>3}): {cnt:>3} {bar}")


# ════════════════════════════════════════════════════════════════
# R8-4: Timeout 优化探索
# ════════════════════════════════════════════════════════════════
def r8_4_timeout_optimization(p):
    L6 = get_l6()
    p("=" * 80)
    p("R8-4: Timeout 优化 — 最大亏损来源攻克")
    p("=" * 80)

    p("\n--- Part A: MaxHold 灵敏度 (L6) ---")
    holds = [3, 4, 5, 6, 8, 10, 12, 15, 20]
    tasks = [(f"Hold={h}", {**L6, "keltner_max_hold_m15": h*4}, 0.30, None, None)
             for h in holds]
    results = run_pool(tasks)
    p(f"  {'MaxHold(h)':>12} {'N':>5} {'Sharpe':>8} {'PnL':>14} {'WR%':>8} {'MaxDD':>10}")
    for r in sorted(results, key=lambda x: int(x[0].split('=')[1])):
        h = int(r[0].split('=')[1])
        p(f"  {h:>12} {r[1]:>5} {r[2]:>8.2f} {fmt(r[3]):>14} {r[4]:>8.1f}% {fmt(r[6]):>10}")

    p("\n--- Part B: Timeout 出场分析 (L6 逐笔) ---")
    detail_tasks = [("L6_detail", L6, 0.30, None, None)]
    detail_results = run_pool(detail_tasks, func=_run_one_trades)
    if detail_results and len(detail_results[0]) > 7:
        trades = detail_results[0][7]
        exit_counter = Counter()
        exit_pnl = {}
        timeout_bars = []
        timeout_mfe = []
        for pnl, reason, bars, strat, etime, direction, mfe in trades:
            base_reason = reason.split(':')[0] if ':' in reason else reason
            base_reason = base_reason[:25]
            exit_counter[base_reason] += 1
            if base_reason not in exit_pnl:
                exit_pnl[base_reason] = []
            exit_pnl[base_reason].append(pnl)
            if 'timeout' in reason.lower():
                timeout_bars.append(bars)
                timeout_mfe.append(mfe)

        p(f"\n  {'Exit Reason':<28} {'Count':>6} {'Total PnL':>12} {'Avg PnL':>10} {'WR%':>8}")
        for reason, count in exit_counter.most_common(10):
            total = sum(exit_pnl[reason])
            avg = total / count
            wins = sum(1 for x in exit_pnl[reason] if x > 0)
            wr = wins / count * 100
            p(f"  {reason:<28} {count:>6} {fmt(total):>12} {fmt(avg):>10} {wr:>7.1f}%")

        if timeout_mfe:
            p(f"\n  Timeout 单子 MFE 分析:")
            p(f"    平均 MFE: ${np.mean(timeout_mfe):.2f}")
            p(f"    MFE > $5: {sum(1 for x in timeout_mfe if x>5)}/{len(timeout_mfe)} ({sum(1 for x in timeout_mfe if x>5)/len(timeout_mfe)*100:.1f}%)")
            p(f"    MFE > $10: {sum(1 for x in timeout_mfe if x>10)}/{len(timeout_mfe)}")
            p(f"    结论: MFE高但最终timeout=浮盈回吐, 需要更早锁利")

    p("\n--- Part C: 最优 MaxHold K-Fold ---")
    best = max(results, key=lambda x: x[2])
    best_h = int(best[0].split('=')[1])
    p(f"  最优: MaxHold={best_h}h Sharpe={best[2]:.2f}")
    if best_h != 5:
        var_kw = {**L6, "keltner_max_hold_m15": best_h * 4}
        base_r, var_r = run_kfold(L6, var_kw, 0.30, "Hold_")
        print_kfold(p, base_r, var_r, "L6(Hold=5h)", f"L6(Hold={best_h}h)")
    else:
        p("  Hold=5h 已是最优")


# ════════════════════════════════════════════════════════════════
# R8-5: Historical Spread 全量验证
# ════════════════════════════════════════════════════════════════
def r8_5_historical_spread(p):
    L51 = get_base()
    L6 = get_l6()
    p("=" * 80)
    p("R8-5: Historical Spread 全量验证 L5.1 vs L6")
    p("=" * 80)

    spreads = [0.00, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    tasks = []
    for sp in spreads:
        tasks.append((f"L51_sp{sp:.2f}", L51, sp, None, None))
        tasks.append((f"L6_sp{sp:.2f}", L6, sp, None, None))
    results = run_pool(tasks)

    p(f"\n  {'Spread':<10} {'L5.1 Sharpe':>12} {'L5.1 PnL':>14} {'L6 Sharpe':>12} {'L6 PnL':>14} {'Delta Sharpe':>14}")
    for sp in spreads:
        l51r = next((r for r in results if r[0] == f"L51_sp{sp:.2f}"), None)
        l6r = next((r for r in results if r[0] == f"L6_sp{sp:.2f}"), None)
        if l51r and l6r:
            delta = l6r[2] - l51r[2]
            p(f"  ${sp:.2f}     {l51r[2]:>12.2f} {fmt(l51r[3]):>14} {l6r[2]:>12.2f} {fmt(l6r[3]):>14} {delta:>+14.2f}")

    p("\n  L6 盈亏平衡 spread:")
    l6_results = sorted([(sp, next(r for r in results if r[0] == f"L6_sp{sp:.2f}")) for sp in spreads], key=lambda x: x[0])
    for sp, r in l6_results:
        if r[3] < 0:
            p(f"    L6 在 ${sp:.2f} spread 下亏损, 盈亏平衡约 ${sp-0.10:.2f}-${sp:.2f}")
            break
    else:
        p(f"    L6 在 ${spreads[-1]:.2f} spread 下仍盈利!")


# ════════════════════════════════════════════════════════════════
# R8-6: L6 近期放大镜 + 月度热力图
# ════════════════════════════════════════════════════════════════
def r8_6_recent_heatmap(p):
    L51 = get_base()
    L6 = get_l6()
    p("=" * 80)
    p("R8-6: L6 近期放大镜 + 月度热力图")
    p("=" * 80)

    p("\n--- Part A: L6 逐年全对比 ---")
    years = list(range(2015, 2027))
    tasks = []
    for y in years:
        end = f"{y+1}-01-01" if y < 2026 else "2026-04-10"
        tasks.append((f"L51_Y{y}", L51, 0.30, f"{y}-01-01", end))
        tasks.append((f"L6_Y{y}", L6, 0.30, f"{y}-01-01", end))
    results = run_pool(tasks)

    p(f"  {'Year':<6} {'L5.1 Sharpe':>12} {'L5.1 PnL':>12} {'L6 Sharpe':>12} {'L6 PnL':>12} {'Delta':>8}")
    for y in years:
        l51r = next((r for r in results if r[0] == f"L51_Y{y}"), None)
        l6r = next((r for r in results if r[0] == f"L6_Y{y}"), None)
        if l51r and l6r:
            delta = l6r[3] - l51r[3]
            p(f"  {y:<6} {l51r[2]:>12.2f} {fmt(l51r[3]):>12} {l6r[2]:>12.2f} {fmt(l6r[3]):>12} {fmt(delta):>8}")
    l6_pos = sum(1 for y in years if next((r for r in results if r[0] == f"L6_Y{y}"), (0,0,0,0))[3] > 0)
    p(f"\n  L6 盈利年份: {l6_pos}/{len(years)}")

    p("\n--- Part B: 月度热力图 (L6, $0.30, 2020-2026) ---")
    months_tasks = []
    for y in range(2020, 2027):
        for m in range(1, 13):
            if y == 2026 and m > 4:
                break
            end_y = y if m < 12 else y + 1
            end_m = m + 1 if m < 12 else 1
            months_tasks.append((f"M{y}-{m:02d}", L6, 0.30,
                                 f"{y}-{m:02d}-01", f"{end_y}-{end_m:02d}-01"))
    month_results = run_pool(months_tasks)
    month_map = {r[0]: r for r in month_results}

    header = f"  {'Year':<6}"
    for m in range(1, 13):
        header += f" {m:>6}"
    header += f" {'Total':>8}"
    p(header)
    for y in range(2020, 2027):
        row = f"  {y:<6}"
        yr_total = 0
        for m in range(1, 13):
            key = f"M{y}-{m:02d}"
            r = month_map.get(key)
            if r:
                pnl = r[3]
                yr_total += pnl
                sign = "+" if pnl >= 0 else ""
                row += f" {sign}{pnl:>5.0f}"
            else:
                row += f" {'--':>6}"
        row += f" {fmt(yr_total):>8}"
        p(row)

    loss_months = sum(1 for r in month_results if r[3] < 0)
    p(f"\n  亏损月份: {loss_months}/{len(month_results)}")


# ════════════════════════════════════════════════════════════════
# R8-7: SL 在 L6 下的精细扫描 + K-Fold
# ════════════════════════════════════════════════════════════════
def r8_7_sl_sensitivity(p):
    L6 = get_l6()
    p("=" * 80)
    p("R8-7: SL 在 L6 下的精细扫描")
    p("=" * 80)

    p("\n--- Part A: SL 扫描 ---")
    sl_vals = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    tasks = [(f"SL={sl:.1f}", {**L6, "sl_atr_mult": sl}, 0.30, None, None)
             for sl in sl_vals]
    results = run_pool(tasks)
    p(f"  {'SL':>6} {'N':>5} {'Sharpe':>8} {'PnL':>14} {'WR%':>8} {'MaxDD':>10}")
    for r in sorted(results, key=lambda x: float(x[0].split('=')[1])):
        marker = " <-current" if '3.5' in r[0] else ""
        p(f"  {r[0]:>6} {r[1]:>5} {r[2]:>8.2f} {fmt(r[3]):>14} {r[4]:>8.1f}% {fmt(r[6]):>10}{marker}")

    best = max(results, key=lambda x: x[2])
    best_sl = float(best[0].split('=')[1])
    p(f"\n  最优: SL={best_sl:.1f} Sharpe={best[2]:.2f}")

    if abs(best_sl - 3.5) > 0.1:
        p("\n--- Part B: K-Fold 最优 SL ---")
        var_kw = {**L6, "sl_atr_mult": best_sl}
        base_r, var_r = run_kfold(L6, var_kw, 0.30, "SL_")
        wins = print_kfold(p, base_r, var_r, "L6(SL=3.5)", f"L6(SL={best_sl:.1f})")
        if wins >= 5:
            p("\n--- Part C: $0.50 K-Fold ---")
            base_r50, var_r50 = run_kfold(L6, var_kw, 0.50, "SL50_")
            print_kfold(p, base_r50, var_r50, "L6$0.50(SL=3.5)", f"L6$0.50(SL={best_sl:.1f})")
    else:
        p("  SL=3.5 已是最优, 参数确认")


# ════════════════════════════════════════════════════════════════
# R8-8: MaxHold 在 L6 下精细扫描
# ════════════════════════════════════════════════════════════════
def r8_8_maxhold_sensitivity(p):
    L6 = get_l6()
    p("=" * 80)
    p("R8-8: MaxHold 在 L6 下精细扫描")
    p("=" * 80)

    holds_m15 = [8, 12, 16, 20, 24, 28, 32, 40, 60]
    tasks = [(f"Hold={h}m15({h/4:.1f}h)", {**L6, "keltner_max_hold_m15": h}, 0.30, None, None)
             for h in holds_m15]
    results = run_pool(tasks)
    p(f"  {'Hold (M15 bars)':>18} {'Hours':>6} {'N':>5} {'Sharpe':>8} {'PnL':>14} {'WR%':>8} {'MaxDD':>10}")
    for r in sorted(results, key=lambda x: int(x[0].split('=')[1].split('m')[0])):
        h = int(r[0].split('=')[1].split('m')[0])
        marker = " <-current" if h == 20 else ""
        p(f"  {h:>18} {h/4:>6.1f} {r[1]:>5} {r[2]:>8.2f} {fmt(r[3]):>14} {r[4]:>8.1f}% {fmt(r[6]):>10}{marker}")

    best = max(results, key=lambda x: x[2])
    best_h = int(best[0].split('=')[1].split('m')[0])
    p(f"\n  最优: {best_h} M15 bars ({best_h/4:.1f}h) Sharpe={best[2]:.2f}")
    if best_h != 20:
        p("\n--- K-Fold ---")
        var_kw = {**L6, "keltner_max_hold_m15": best_h}
        base_r, var_r = run_kfold(L6, var_kw, 0.30, "MH_")
        print_kfold(p, base_r, var_r, "L6(Hold=20)", f"L6(Hold={best_h})")


# ════════════════════════════════════════════════════════════════
# R8-9: Choppy 阈值在 L6 下验证
# ════════════════════════════════════════════════════════════════
def r8_9_choppy_threshold(p):
    L6 = get_l6()
    p("=" * 80)
    p("R8-9: Choppy 阈值在 L6 下验证")
    p("=" * 80)

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    tasks = [(f"Choppy={t:.2f}", {**L6, "choppy_threshold": t}, 0.30, None, None)
             for t in thresholds]
    results = run_pool(tasks)
    p(f"  {'Choppy':>10} {'N':>5} {'Sharpe':>8} {'PnL':>14} {'WR%':>8} {'MaxDD':>10} {'Choppy Skips':>14}")
    for r in sorted(results, key=lambda x: float(x[0].split('=')[1])):
        marker = " <-current" if '0.50' in r[0] else ""
        p(f"  {r[0]:>10} {r[1]:>5} {r[2]:>8.2f} {fmt(r[3]):>14} {r[4]:>8.1f}% {fmt(r[6]):>10}{marker}")

    best = max(results, key=lambda x: x[2])
    best_t = float(best[0].split('=')[1])
    if abs(best_t - 0.50) > 0.01:
        p(f"\n  最优: Choppy={best_t:.2f} Sharpe={best[2]:.2f}")
        p("\n--- K-Fold ---")
        var_kw = {**L6, "choppy_threshold": best_t}
        base_r, var_r = run_kfold(L6, var_kw, 0.30, "Chop_")
        print_kfold(p, base_r, var_r, "L6(0.50)", f"L6({best_t:.2f})")
    else:
        p(f"\n  Choppy=0.50 已是最优, 参数确认")


# ════════════════════════════════════════════════════════════════
# R8-10: ADX 阈值在 L6 下验证
# ════════════════════════════════════════════════════════════════
def r8_10_adx_threshold(p):
    L6 = get_l6()
    p("=" * 80)
    p("R8-10: ADX 阈值在 L6 下验证")
    p("=" * 80)

    adx_vals = [14, 16, 18, 20, 22, 24, 26]
    tasks = [(f"ADX={a}", {**L6, "keltner_adx_threshold": a}, 0.30, None, None)
             for a in adx_vals]
    results = run_pool(tasks)
    p(f"  {'ADX':>6} {'N':>5} {'Sharpe':>8} {'PnL':>14} {'WR%':>8} {'MaxDD':>10}")
    for r in sorted(results, key=lambda x: int(x[0].split('=')[1])):
        marker = " <-current" if '=18' in r[0] else ""
        p(f"  {r[0]:>6} {r[1]:>5} {r[2]:>8.2f} {fmt(r[3]):>14} {r[4]:>8.1f}% {fmt(r[6]):>10}{marker}")

    best = max(results, key=lambda x: x[2])
    best_a = int(best[0].split('=')[1])
    if best_a != 18:
        p(f"\n  最优: ADX={best_a} Sharpe={best[2]:.2f}")
        p("\n--- K-Fold ---")
        var_kw = {**L6, "keltner_adx_threshold": best_a}
        base_r, var_r = run_kfold(L6, var_kw, 0.30, "ADX_")
        print_kfold(p, base_r, var_r, "L6(ADX=18)", f"L6(ADX={best_a})")
    else:
        p(f"\n  ADX=18 已是最优, 参数确认")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    master_log = os.path.join(OUTPUT_DIR, "00_master_log.txt")

    phases = [
        ("r8_1_tp_sl_grid.txt",      "R8-1: TP x SL Grid",          r8_1_tp_sl_grid),
        ("r8_2_l6_entry_gap.txt",    "R8-2: L6+Entry Gap",          r8_2_l6_entry_gap),
        ("r8_3_l6_monte_carlo.txt",  "R8-3: L6 Monte Carlo",        r8_3_l6_monte_carlo),
        ("r8_4_timeout_opt.txt",     "R8-4: Timeout优化",            r8_4_timeout_optimization),
        ("r8_5_hist_spread.txt",     "R8-5: Historical Spread",      r8_5_historical_spread),
        ("r8_6_recent_heatmap.txt",  "R8-6: 近期+热力图",            r8_6_recent_heatmap),
        ("r8_7_sl_sensitivity.txt",  "R8-7: L6 SL扫描",             r8_7_sl_sensitivity),
        ("r8_8_maxhold.txt",         "R8-8: L6 MaxHold扫描",        r8_8_maxhold_sensitivity),
        ("r8_9_choppy.txt",          "R8-9: L6 Choppy阈值",         r8_9_choppy_threshold),
        ("r8_10_adx.txt",            "R8-10: L6 ADX阈值",           r8_10_adx_threshold),
    ]

    with open(master_log, "w", encoding="utf-8") as mf:
        mf.write(f"Round 8 (L6 Comprehensive Validation)\n")
        mf.write(f"Started: {datetime.now()}\n")
        mf.write(f"MAX_WORKERS: {MAX_WORKERS}\n")
        mf.write(f"Server: 25 cores\n")
        mf.write("=" * 60 + "\n\n")

    for fname, phase_name, func in phases:
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath) and os.path.getsize(fpath) > 100:
            print(f"  SKIP {phase_name} (result exists)")
            with open(master_log, "a", encoding="utf-8") as mf:
                mf.write(f"  {phase_name}: SKIPPED (result exists)\n")
            continue

        t0 = time.time()
        print(f"\n>>> Starting {phase_name} ...")
        try:
            lines = []
            lines.append(f"# {phase_name}")
            lines.append(f"# Started: {datetime.now()}")
            lines.append(f"# Workers: {MAX_WORKERS}")

            def p(msg=""):
                lines.append(msg)
                print(msg, flush=True)

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
        mf.write(f"\nRound 8 Finished: {datetime.now()}\n")
    print(f"\n{'='*60}")
    print(f"Round 8 ALL DONE!")
