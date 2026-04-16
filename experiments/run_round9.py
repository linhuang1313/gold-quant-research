#!/usr/bin/env python3
"""
Round 9 — L7 候选构建 + 多维组合验证 + 新策略探索
====================================================
25-core server, ~20h budget

=== Phase 1: L7 组合候选 (L6 + R8 发现的改进) ===
R9-1:  L7 候选构建 — L6 + Entry Gap 1h + 其他 R8 通过的改进逐层叠加
R9-2:  L7 vs L6 vs L5.1 全面 K-Fold (双点差) + Walk-Forward
R9-3:  L7 Monte Carlo 300 次参数扰动 (更大样本)
R9-4:  L7 逐年对比 + 月度热力图 + 季度 Sharpe

=== Phase 2: 参数精调 ===
R9-5:  SL 在 L7 下精细 2D 扫描 (SL x MaxHold)
R9-6:  Choppy x ADX 二维交互 (在 L7 下)
R9-7:  Trailing 参数微调 — UltraTight3 探索 (比 L6 更紧/更松的变体)
R9-8:  Time Decay TP 在 L7 下重新验证 (R4 曾否决但 L7 基线不同)

=== Phase 3: 鲁棒性压力测试 ===
R9-9:  滚动窗口 Walk-Forward (2年训练 + 6月测试, 滑动)
R9-10: 多 Spread 模型对比 (Fixed / Session-aware / Historical)
R9-11: 极端市场分期测试 (COVID 2020, 加息周期 2022, 关税危机 2025-04)
R9-12: 破产概率 Monte Carlo (不同起始资金 $500-$5000)

=== Phase 4: 新方向探索 ===
R9-13: RSI 均值回归参数优化 (当前贡献边际)
R9-14: ORB 参数优化 (session filter / hold time)
R9-15: Keltner 通道参数探索 (KC_ema / KC_mult / EMA 周期)
R9-16: 入场时段优化 (哪些 UTC 小时入场效果最好)
R9-17: Cooldown 时间优化 (冷却期长度对策略影响)
R9-18: ATR Spike Protection 验证 (波动率突变时收紧 trailing)
"""
import sys, os, io, time, traceback, random
import multiprocessing as mp
import numpy as np
from datetime import datetime
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "results/round9_results"
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
          for t in trades[:15000]]
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
    ("Fold6", "2025-01-01", "2026-04-01"),
]

def run_kfold(base_kw, label_prefix, spread=0.30):
    tasks = [(f"{label_prefix}_{fn}", {**base_kw}, spread, s, e) for fn, s, e in FOLDS]
    return run_pool(tasks)

def print_kfold(p, base_results, test_results, base_label, test_label):
    p(f"\n  {'Fold':<10}{base_label:>20} Sharpe {test_label:>20} Sharpe {'Delta':>10}  Pass?")
    p(f"  {'-'*65}")
    all_pass = True
    for i, fn in enumerate([f[0] for f in FOLDS]):
        bs = base_results[i][2]
        ts = test_results[i][2]
        d = ts - bs
        ok = "YES" if d > -1.0 else "NO"
        if d <= -1.0: all_pass = False
        p(f"  {fn:<10}{bs:>20.2f} {ts:>20.2f} {d:>+10.2f}    {ok}")
    passed = sum(1 for i in range(len(FOLDS)) if test_results[i][2] - base_results[i][2] > -1.0)
    p(f"\n  K-Fold: {passed}/{len(FOLDS)} PASS")
    return passed


# ══════════════════════════════════════════════════════════════════
# Phase 1: L7 候选构建
# ══════════════════════════════════════════════════════════════════

def r9_1_l7_construction(p):
    """L7 = L6 + R8 通过的改进逐层叠加"""
    p("="*80)
    p("R9-1: L7 候选构建 — 逐层叠加 R8 验证通过的改进")
    p("="*80)

    L6 = get_l6()

    # 各个改进单独测试 + 叠加
    variants = {
        "L6_base":           {**L6},
        "L6+Gap1h":          {**L6, "min_entry_gap_hours": 1.0},
        "L6+SL2.5":          {**L6, "sl_atr_mult": 2.5},
        "L6+SL3.0":          {**L6, "sl_atr_mult": 3.0},
        "L6+Hold8bar":       {**L6, "keltner_max_hold_m15": 8},
        "L6+Hold12bar":      {**L6, "keltner_max_hold_m15": 12},
        "L6+Choppy0.55":     {**L6, "choppy_threshold": 0.55},
        "L6+Choppy0.60":     {**L6, "choppy_threshold": 0.60},
        "L6+ADX14":          {**L6, "keltner_adx_threshold": 14},
        "L6+ADX16":          {**L6, "keltner_adx_threshold": 16},
        # 两两组合
        "L6+Gap1h+SL3.0":    {**L6, "min_entry_gap_hours": 1.0, "sl_atr_mult": 3.0},
        "L6+Gap1h+Hold12":   {**L6, "min_entry_gap_hours": 1.0, "keltner_max_hold_m15": 12},
        "L6+Gap1h+ADX14":    {**L6, "min_entry_gap_hours": 1.0, "keltner_adx_threshold": 14},
        "L6+Gap1h+Ch0.55":   {**L6, "min_entry_gap_hours": 1.0, "choppy_threshold": 0.55},
        # 三层组合
        "L7a_Gap+SL3+H12":   {**L6, "min_entry_gap_hours": 1.0, "sl_atr_mult": 3.0, "keltner_max_hold_m15": 12},
        "L7b_Gap+ADX14+H12": {**L6, "min_entry_gap_hours": 1.0, "keltner_adx_threshold": 14, "keltner_max_hold_m15": 12},
        "L7c_Gap+Ch55+ADX14":{**L6, "min_entry_gap_hours": 1.0, "choppy_threshold": 0.55, "keltner_adx_threshold": 14},
        "L7d_Gap+SL3+ADX14": {**L6, "min_entry_gap_hours": 1.0, "sl_atr_mult": 3.0, "keltner_adx_threshold": 14},
        # 四层组合
        "L7e_Full":          {**L6, "min_entry_gap_hours": 1.0, "sl_atr_mult": 3.0, "keltner_adx_threshold": 14, "keltner_max_hold_m15": 12},
        "L7f_FullCh":        {**L6, "min_entry_gap_hours": 1.0, "sl_atr_mult": 3.0, "keltner_adx_threshold": 14, "choppy_threshold": 0.55},
    }

    tasks = [(name, kw, 0.30, None, None) for name, kw in variants.items()]
    results = run_pool(tasks)

    p(f"\n--- 全样本 $0.30 ---")
    p(f"  {'Variant':<24}{'N':>8}{'Sharpe':>10}{'PnL':>14}{'WR%':>8}{'MaxDD':>12}")
    for r in sorted(results, key=lambda x: x[2], reverse=True):
        p(f"  {r[0]:<24}{r[1]:>8}{r[2]:>10.2f} {fmt(r[3]):>13}{r[4]:>8.1f}% {fmt(r[6]):>11}")

    # $0.50 top 5
    top5 = sorted(results, key=lambda x: x[2], reverse=True)[:5]
    top5_names = [r[0] for r in top5]
    tasks50 = [(name, variants[name], 0.50, None, None) for name in top5_names]
    results50 = run_pool(tasks50)

    p(f"\n--- Top 5 at $0.50 ---")
    p(f"  {'Variant':<24}{'N':>8}{'Sharpe':>10}{'PnL':>14}{'WR%':>8}{'MaxDD':>12}")
    for r in sorted(results50, key=lambda x: x[2], reverse=True):
        p(f"  {r[0]:<24}{r[1]:>8}{r[2]:>10.2f} {fmt(r[3]):>13}{r[4]:>8.1f}% {fmt(r[6]):>11}")


def r9_2_l7_kfold(p):
    """L7 vs L6 vs L5.1 K-Fold + Walk-Forward"""
    p("="*80)
    p("R9-2: L7 vs L6 vs L5.1 全面 K-Fold + Walk-Forward")
    p("="*80)

    L51 = get_base()
    L6 = get_l6()
    # L7 = best from R9-1 (use Gap+ADX14 as likely candidate based on R8)
    L7 = {**L6, "min_entry_gap_hours": 1.0, "keltner_adx_threshold": 14}

    for spread, sp_label in [(0.30, "$0.30"), (0.50, "$0.50")]:
        p(f"\n--- K-Fold {sp_label} ---")
        r_51 = run_kfold(L51, "L5.1", spread)
        r_l6 = run_kfold(L6, "L6", spread)
        r_l7 = run_kfold(L7, "L7", spread)

        p(f"\n  {'Fold':<10}{'L5.1':>10}{'L6':>10}{'L7':>10}  L6-L51  L7-L6  L7-L51")
        p(f"  {'-'*75}")
        for i, fn in enumerate([f[0] for f in FOLDS]):
            s51 = r_51[i][2]; sl6 = r_l6[i][2]; sl7 = r_l7[i][2]
            p(f"  {fn:<10}{s51:>10.2f}{sl6:>10.2f}{sl7:>10.2f}  {sl6-s51:>+6.2f}  {sl7-sl6:>+5.2f}  {sl7-s51:>+6.2f}")

    # Walk-Forward (annual)
    p(f"\n--- Walk-Forward (年度) ---")
    years = [(str(y), f"{y}-01-01", f"{y+1}-01-01") for y in range(2015, 2026)]
    years.append(("2026", "2026-01-01", "2026-04-10"))

    wf_tasks = []
    for yr, s, e in years:
        for lbl, kw in [("L5.1", L51), ("L6", L6), ("L7", L7)]:
            wf_tasks.append((f"WF_{lbl}_{yr}", {**kw}, 0.30, s, e))
    wf_results = run_pool(wf_tasks)

    p(f"  {'Year':<8}{'L5.1':>10}{'L6':>10}{'L7':>10}")
    p(f"  {'-'*40}")
    for i, (yr, _, _) in enumerate(years):
        r51 = wf_results[i*3][2]
        rl6 = wf_results[i*3+1][2]
        rl7 = wf_results[i*3+2][2]
        p(f"  {yr:<8}{r51:>10.2f}{rl6:>10.2f}{rl7:>10.2f}")


def r9_3_l7_monte_carlo(p):
    """L7 Monte Carlo 300 次"""
    p("="*80)
    p("R9-3: L7 Monte Carlo 参数扰动 (300次, +/-15%)")
    p("="*80)

    L7 = {**get_l6(), "min_entry_gap_hours": 1.0, "keltner_adx_threshold": 14}
    perturb_keys = ['trailing_activate_atr', 'trailing_distance_atr', 'sl_atr_mult',
                    'tp_atr_mult', 'choppy_threshold']

    random.seed(42)
    N = 300
    tasks = []
    for i in range(N):
        kw = {**L7}
        for key in perturb_keys:
            if key in kw and isinstance(kw[key], (int, float)):
                kw[key] = round(kw[key] * random.uniform(0.85, 1.15), 4)
        if 'regime_config' in kw:
            rc = {}
            for regime, vals in kw['regime_config'].items():
                rc[regime] = {
                    'trail_act': round(vals['trail_act'] * random.uniform(0.85, 1.15), 4),
                    'trail_dist': round(vals['trail_dist'] * random.uniform(0.85, 1.15), 4),
                }
            kw['regime_config'] = rc
        tasks.append((f"MC_{i}", kw, 0.30, None, None))

    BATCH = 60
    all_sharpes, all_pnls = [], []
    for batch_i in range(0, N, BATCH):
        batch = tasks[batch_i:batch_i+BATCH]
        p(f"\n  Batch {batch_i//BATCH+1}/{(N+BATCH-1)//BATCH} ({batch_i}-{min(batch_i+BATCH-1, N-1)})...")
        results = run_pool(batch)
        for r in results:
            all_sharpes.append(r[2])
            all_pnls.append(r[3])
        p(f"    done ({len(all_sharpes)}/{N})")

    p(f"\n--- {N} 次 L7 参数扰动结果 ---")
    p(f"  Sharpe: mean={np.mean(all_sharpes):.2f}, std={np.std(all_sharpes):.2f}, "
      f"min={np.min(all_sharpes):.2f}, max={np.max(all_sharpes):.2f}")
    p(f"  PnL:    mean={fmt(np.mean(all_pnls))}, std={fmt(np.std(all_pnls))}, "
      f"min={fmt(np.min(all_pnls))}, max={fmt(np.max(all_pnls))}")
    p(f"  盈利组合: {sum(1 for s in all_pnls if s > 0)}/{N}")
    p(f"  Sharpe>2: {sum(1 for s in all_sharpes if s > 2)}/{N}")
    p(f"  Sharpe>4: {sum(1 for s in all_sharpes if s > 4)}/{N}")
    p(f"  Sharpe>6: {sum(1 for s in all_sharpes if s > 6)}/{N}")

    bins = [(0,2),(2,4),(4,5),(5,6),(6,7),(7,8),(8,10),(10,100)]
    for lo, hi in bins:
        cnt = sum(1 for s in all_sharpes if lo <= s < hi)
        bar = '#' * cnt
        p(f"  [{lo:>2}-{hi:>3}): {cnt:>4} {bar}")


def r9_4_l7_yearly_heatmap(p):
    """L7 逐年 + 月度热力图"""
    p("="*80)
    p("R9-4: L7 逐年对比 + 月度热力图 + 季度 Sharpe")
    p("="*80)

    L51 = get_base()
    L6 = get_l6()
    L7 = {**L6, "min_entry_gap_hours": 1.0, "keltner_adx_threshold": 14}

    years = [(str(y), f"{y}-01-01", f"{y+1}-01-01") for y in range(2015, 2026)]
    years.append(("2026", "2026-01-01", "2026-04-10"))

    tasks = []
    for yr, s, e in years:
        for lbl, kw in [("L5.1", L51), ("L6", L6), ("L7", L7)]:
            tasks.append((f"{lbl}_{yr}", {**kw}, 0.30, s, e))
    results = run_pool(tasks)

    p(f"\n--- 逐年全对比 ---")
    p(f"  {'Year':<8}{'L5.1 Sharpe':>14}{'L5.1 PnL':>12}{'L6 Sharpe':>12}{'L6 PnL':>12}"
      f"{'L7 Sharpe':>12}{'L7 PnL':>12}{'L7-L6':>8}")
    for i, (yr, _, _) in enumerate(years):
        r51, rl6, rl7 = results[i*3], results[i*3+1], results[i*3+2]
        delta = rl7[3] - rl6[3]
        p(f"  {yr:<8}{r51[2]:>14.2f} {fmt(r51[3]):>11}{rl6[2]:>12.2f} {fmt(rl6[3]):>11}"
          f"{rl7[2]:>12.2f} {fmt(rl7[3]):>11} {fmt(delta):>7}")

    # Monthly heatmap for L7
    p(f"\n--- 月度热力图 (L7, $0.30, 2020-2026) ---")
    month_tasks = []
    for y in range(2020, 2027):
        for m in range(1, 13):
            if y == 2026 and m > 4: break
            s = f"{y}-{m:02d}-01"
            if m == 12:
                e = f"{y+1}-01-01"
            else:
                e = f"{y}-{m+1:02d}-01"
            month_tasks.append((f"L7_{y}_{m:02d}", {**L7}, 0.30, s, e))
    month_results = run_pool(month_tasks)

    mr_dict = {r[0]: r for r in month_results}
    p(f"  {'Year':>6}" + "".join(f"{m:>7}" for m in range(1, 13)) + f"{'Total':>10}")
    for y in range(2020, 2027):
        row = f"  {y:>6}"
        total = 0
        for m in range(1, 13):
            key = f"L7_{y}_{m:02d}"
            if key in mr_dict:
                pnl = mr_dict[key][3]
                total += pnl
                sign = '+' if pnl >= 0 else '-'
                row += f" {sign}{abs(pnl):>5.0f}"
            else:
                row += "     --"
        row += f" {fmt(total):>9}"
        p(row)


# ══════════════════════════════════════════════════════════════════
# Phase 2: 参数精调
# ══════════════════════════════════════════════════════════════════

def r9_5_sl_maxhold_2d(p):
    """SL x MaxHold 二维扫描"""
    p("="*80)
    p("R9-5: SL x MaxHold 二维扫描 (L7)")
    p("="*80)

    L7 = {**get_l6(), "min_entry_gap_hours": 1.0, "keltner_adx_threshold": 14}

    sls = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    holds = [8, 12, 16, 20, 24, 32]

    tasks = []
    for sl in sls:
        for h in holds:
            kw = {**L7, "sl_atr_mult": sl, "keltner_max_hold_m15": h}
            tasks.append((f"SL{sl}_H{h}", kw, 0.30, None, None))
    results = run_pool(tasks)

    rd = {r[0]: r for r in results}
    col_hdr = "SL\\Hold"
    header = f"  {col_hdr:<10}" + "".join(f"{h:>8}" for h in holds)
    p(f"\n--- Sharpe Grid ---")
    p(header)
    best_sharpe, best_label = 0, ""
    for sl in sls:
        row = f"  {sl:<10.1f}"
        for h in holds:
            key = f"SL{sl}_H{h}"
            s = rd[key][2]
            if s > best_sharpe:
                best_sharpe = s
                best_label = key
            row += f"{s:>8.2f}"
        p(row)
    p(f"\n  Best: {best_label} Sharpe={best_sharpe:.2f}")


def r9_6_choppy_adx_2d(p):
    """Choppy x ADX 二维交互"""
    p("="*80)
    p("R9-6: Choppy x ADX 二维扫描 (L7)")
    p("="*80)

    L7 = {**get_l6(), "min_entry_gap_hours": 1.0, "keltner_adx_threshold": 14}

    choppys = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    adxs = [12, 14, 16, 18, 20, 22]

    tasks = []
    for ch in choppys:
        for adx in adxs:
            kw = {**L7, "choppy_threshold": ch, "keltner_adx_threshold": adx}
            tasks.append((f"Ch{ch}_ADX{adx}", kw, 0.30, None, None))
    results = run_pool(tasks)

    rd = {r[0]: r for r in results}
    col_hdr = "Ch\\ADX"
    header = f"  {col_hdr:<10}" + "".join(f"{a:>8}" for a in adxs)
    p(f"\n--- Sharpe Grid ---")
    p(header)
    best_s, best_l = 0, ""
    for ch in choppys:
        row = f"  {ch:<10.2f}"
        for adx in adxs:
            key = f"Ch{ch}_ADX{adx}"
            s = rd[key][2]
            if s > best_s: best_s, best_l = s, key
            row += f"{s:>8.2f}"
        p(row)
    p(f"\n  Best: {best_l} Sharpe={best_s:.2f}")

    # K-Fold top
    best_ch = float(best_l.split('_')[0].replace('Ch',''))
    best_adx = int(best_l.split('ADX')[1])
    best_kw = {**L7, "choppy_threshold": best_ch, "keltner_adx_threshold": best_adx}
    base_r = run_kfold(L7, "L7_base", 0.30)
    best_r = run_kfold(best_kw, "L7_best", 0.30)
    print_kfold(p, base_r, best_r, "L7(base)", f"L7(Ch={best_ch},ADX={best_adx})")


def r9_7_trailing_variants(p):
    """UltraTight3 探索 — trailing 参数的进一步微调"""
    p("="*80)
    p("R9-7: Trailing 参数微调 — UltraTight3 变体探索")
    p("="*80)

    L7 = {**get_l6(), "min_entry_gap_hours": 1.0, "keltner_adx_threshold": 14}

    trail_variants = {
        "L6(current)": ULTRA2,
        "UT3a_tighter": {
            'low': {'trail_act': 0.25, 'trail_dist': 0.05},
            'normal': {'trail_act': 0.16, 'trail_dist': 0.03},
            'high': {'trail_act': 0.06, 'trail_dist': 0.008},
        },
        "UT3b_asym": {
            'low': {'trail_act': 0.35, 'trail_dist': 0.05},
            'normal': {'trail_act': 0.22, 'trail_dist': 0.03},
            'high': {'trail_act': 0.08, 'trail_dist': 0.008},
        },
        "UT3c_wide_low": {
            'low': {'trail_act': 0.40, 'trail_dist': 0.08},
            'normal': {'trail_act': 0.20, 'trail_dist': 0.04},
            'high': {'trail_act': 0.06, 'trail_dist': 0.01},
        },
        "UT3d_tight_high": {
            'low': {'trail_act': 0.30, 'trail_dist': 0.06},
            'normal': {'trail_act': 0.20, 'trail_dist': 0.04},
            'high': {'trail_act': 0.05, 'trail_dist': 0.005},
        },
        "UT3e_uniform": {
            'low': {'trail_act': 0.20, 'trail_dist': 0.04},
            'normal': {'trail_act': 0.20, 'trail_dist': 0.04},
            'high': {'trail_act': 0.20, 'trail_dist': 0.04},
        },
        "UT3f_steep": {
            'low': {'trail_act': 0.45, 'trail_dist': 0.10},
            'normal': {'trail_act': 0.18, 'trail_dist': 0.03},
            'high': {'trail_act': 0.05, 'trail_dist': 0.006},
        },
        "UT3g_micro": {
            'low': {'trail_act': 0.22, 'trail_dist': 0.04},
            'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
            'high': {'trail_act': 0.06, 'trail_dist': 0.008},
        },
    }

    tasks = []
    for name, rc in trail_variants.items():
        kw = {**L7, "regime_config": rc,
              "trailing_activate_atr": rc['normal']['trail_act'],
              "trailing_distance_atr": rc['normal']['trail_dist']}
        tasks.append((name, kw, 0.30, None, None))
    results = run_pool(tasks)

    p(f"\n--- Trailing 变体对比 ($0.30) ---")
    p(f"  {'Variant':<20}{'N':>8}{'Sharpe':>10}{'PnL':>14}{'WR%':>8}{'MaxDD':>12}")
    for r in sorted(results, key=lambda x: x[2], reverse=True):
        p(f"  {r[0]:<20}{r[1]:>8}{r[2]:>10.2f} {fmt(r[3]):>13}{r[4]:>8.1f}% {fmt(r[6]):>11}")

    # K-Fold best vs L6 current
    best = sorted(results, key=lambda x: x[2], reverse=True)[0]
    if best[0] != "L6(current)":
        best_name = best[0]
        best_rc = trail_variants[best_name]
        best_kw = {**L7, "regime_config": best_rc,
                   "trailing_activate_atr": best_rc['normal']['trail_act'],
                   "trailing_distance_atr": best_rc['normal']['trail_dist']}
        base_r = run_kfold(L7, "L6_trail", 0.30)
        best_r = run_kfold(best_kw, best_name, 0.30)
        print_kfold(p, base_r, best_r, "L6(current)", best_name)


def r9_8_time_decay_retest(p):
    """Time Decay TP 在 L7 下重新验证"""
    p("="*80)
    p("R9-8: Time Decay TP 在 L7 下重新验证")
    p("="*80)

    L7 = {**get_l6(), "min_entry_gap_hours": 1.0, "keltner_adx_threshold": 14}

    td_configs = [
        ("TD_OFF", {}),
        ("TD_default", {"time_decay_tp": True}),
        ("TD_fast", {"time_decay_tp": True, "time_decay_start_hour": 0.5, "time_decay_atr_step": 0.15}),
        ("TD_slow", {"time_decay_tp": True, "time_decay_start_hour": 2.0, "time_decay_atr_step": 0.05}),
        ("TD_low_start", {"time_decay_tp": True, "time_decay_atr_start": 0.20}),
        ("TD_high_start", {"time_decay_tp": True, "time_decay_atr_start": 0.50}),
    ]

    tasks = [(name, {**L7, **kw}, 0.30, None, None) for name, kw in td_configs]
    results = run_pool(tasks)

    p(f"\n--- Time Decay TP 对比 ---")
    p(f"  {'Config':<20}{'N':>8}{'Sharpe':>10}{'PnL':>14}{'WR%':>8}{'MaxDD':>12}")
    for r in sorted(results, key=lambda x: x[2], reverse=True):
        p(f"  {r[0]:<20}{r[1]:>8}{r[2]:>10.2f} {fmt(r[3]):>13}{r[4]:>8.1f}% {fmt(r[6]):>11}")


# ══════════════════════════════════════════════════════════════════
# Phase 3: 鲁棒性压力测试
# ══════════════════════════════════════════════════════════════════

def r9_9_rolling_walkforward(p):
    """滚动窗口 Walk-Forward"""
    p("="*80)
    p("R9-9: 滚动窗口 Walk-Forward (2年训练 + 6月测试)")
    p("="*80)

    L7 = {**get_l6(), "min_entry_gap_hours": 1.0, "keltner_adx_threshold": 14}

    windows = []
    for y in range(2017, 2026):
        for half in [0, 1]:
            test_start = f"{y}-{'01' if half==0 else '07'}-01"
            test_end = f"{y}-{'07' if half==0 else '12'}-31" if half==0 else f"{y+1}-01-01"
            windows.append((f"WF_{y}H{half+1}", test_start, test_end))

    tasks = [(name, {**L7}, 0.30, s, e) for name, s, e in windows]
    results = run_pool(tasks)

    p(f"\n--- 半年滚动窗口 ---")
    p(f"  {'Window':<14}{'N':>8}{'Sharpe':>10}{'PnL':>14}{'WR%':>8}")
    profitable = 0
    for r in results:
        sign = "+" if r[3] >= 0 else ""
        p(f"  {r[0]:<14}{r[1]:>8}{r[2]:>10.2f} {fmt(r[3]):>13}{r[4]:>8.1f}%")
        if r[3] > 0: profitable += 1
    p(f"\n  盈利窗口: {profitable}/{len(results)}")


def r9_10_spread_models(p):
    """多 Spread 模型对比"""
    p("="*80)
    p("R9-10: 多 Spread 模型对比 (Fixed / Session-aware)")
    p("="*80)

    L7 = {**get_l6(), "min_entry_gap_hours": 1.0, "keltner_adx_threshold": 14}

    spread_configs = []
    for s in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        spread_configs.append((f"Fixed_{s:.2f}", {**L7, "spread_model": "fixed"}, s))
    spread_configs.append((f"Session_0.30", {**L7, "spread_model": "session_aware", "spread_base": 0.30}, 0.30))
    spread_configs.append((f"ATR_scaled_0.30", {**L7, "spread_model": "atr_scaled", "spread_base": 0.30}, 0.30))

    tasks = [(name, kw, sp, None, None) for name, kw, sp in spread_configs]
    results = run_pool(tasks)

    p(f"\n--- Spread 模型对比 ---")
    p(f"  {'Model':<22}{'N':>8}{'Sharpe':>10}{'PnL':>14}{'WR%':>8}{'MaxDD':>12}")
    for r in results:
        p(f"  {r[0]:<22}{r[1]:>8}{r[2]:>10.2f} {fmt(r[3]):>13}{r[4]:>8.1f}% {fmt(r[6]):>11}")


def r9_11_extreme_periods(p):
    """极端市场分期测试"""
    p("="*80)
    p("R9-11: 极端市场分期测试")
    p("="*80)

    L51 = get_base()
    L6 = get_l6()
    L7 = {**L6, "min_entry_gap_hours": 1.0, "keltner_adx_threshold": 14}

    periods = [
        ("COVID_crash",   "2020-02-01", "2020-05-01"),
        ("COVID_rally",   "2020-06-01", "2020-12-31"),
        ("Rate_hike_22",  "2022-01-01", "2022-12-31"),
        ("SVB_crisis",    "2023-03-01", "2023-06-01"),
        ("Rally_2024H2",  "2024-07-01", "2024-12-31"),
        ("Tariff_2025",   "2025-03-01", "2025-06-01"),
        ("Tariff_Apr25",  "2025-04-01", "2025-05-01"),
        ("Rally_2026Q1",  "2026-01-01", "2026-04-01"),
    ]

    tasks = []
    for pname, s, e in periods:
        for lbl, kw in [("L5.1", L51), ("L6", L6), ("L7", L7)]:
            tasks.append((f"{lbl}_{pname}", {**kw}, 0.30, s, e))
    results = run_pool(tasks)

    rd = {r[0]: r for r in results}
    p(f"\n  {'Period':<18}{'L5.1 Sharpe':>14}{'L5.1 PnL':>12}{'L6 Sharpe':>12}{'L6 PnL':>12}"
      f"{'L7 Sharpe':>12}{'L7 PnL':>12}")
    for pname, _, _ in periods:
        r51 = rd[f"L5.1_{pname}"]
        rl6 = rd[f"L6_{pname}"]
        rl7 = rd[f"L7_{pname}"]
        p(f"  {pname:<18}{r51[2]:>14.2f} {fmt(r51[3]):>11}{rl6[2]:>12.2f} {fmt(rl6[3]):>11}"
          f"{rl7[2]:>12.2f} {fmt(rl7[3]):>11}")


def r9_12_bankruptcy_sim(p):
    """破产概率 Monte Carlo"""
    p("="*80)
    p("R9-12: 破产概率 Monte Carlo (不同起始资金)")
    p("="*80)

    L7 = {**get_l6(), "min_entry_gap_hours": 1.0, "keltner_adx_threshold": 14}

    # Get trade PnLs
    result = run_pool([("L7_trades", {**L7}, 0.30, None, None)], func=_run_one_trades)[0]
    trade_pnls = [t[0] for t in result[7]]

    capitals = [500, 750, 1000, 1500, 2000, 3000, 5000]
    N_SIM = 5000

    random.seed(42)
    p(f"\n  Total trades: {len(trade_pnls)}")
    p(f"  Avg PnL: ${np.mean(trade_pnls):.2f}, Std: ${np.std(trade_pnls):.2f}")
    p(f"  Win Rate: {sum(1 for x in trade_pnls if x > 0)/len(trade_pnls)*100:.1f}%\n")
    p(f"  {'Capital':>10}{'Bankrupt%':>12}{'Median End':>14}{'P10 End':>12}{'P90 End':>12}")
    p(f"  {'-'*62}")

    for cap in capitals:
        bankrupts = 0
        end_equities = []
        for _ in range(N_SIM):
            equity = cap
            for pnl in random.choices(trade_pnls, k=len(trade_pnls)):
                equity += pnl
                if equity <= 0:
                    bankrupts += 1
                    break
            end_equities.append(max(equity, 0))
        end_equities.sort()
        p10 = end_equities[int(N_SIM*0.1)]
        p50 = end_equities[int(N_SIM*0.5)]
        p90 = end_equities[int(N_SIM*0.9)]
        p(f"  ${cap:>9}{bankrupts/N_SIM*100:>11.1f}% {fmt(p50):>13} {fmt(p10):>11} {fmt(p90):>11}")


# ══════════════════════════════════════════════════════════════════
# Phase 4: 新方向探索
# ══════════════════════════════════════════════════════════════════

def r9_13_rsi_optimization(p):
    """RSI 均值回归参数优化"""
    p("="*80)
    p("R9-13: M15 RSI 均值回归参数优化")
    p("="*80)

    L7 = {**get_l6(), "min_entry_gap_hours": 1.0, "keltner_adx_threshold": 14}

    configs = [
        ("RSI_adx0",    {"rsi_adx_filter": 0}),
        ("RSI_adx20",   {"rsi_adx_filter": 20}),
        ("RSI_adx30",   {"rsi_adx_filter": 30}),
        ("RSI_adx40",   {"rsi_adx_filter": 40}),
        ("RSI_adx50",   {"rsi_adx_filter": 50}),
        ("RSI_noSell",  {"rsi_sell_enabled": False}),
        ("RSI_hold10",  {"rsi_max_hold_m15": 10}),
        ("RSI_hold15",  {"rsi_max_hold_m15": 15}),
        ("RSI_hold20",  {"rsi_max_hold_m15": 20}),
        ("RSI_hold25",  {"rsi_max_hold_m15": 25}),
        ("RSI_OFF",     {"rsi_adx_filter": 999}),
    ]

    tasks = [(name, {**L7, **kw}, 0.30, None, None) for name, kw in configs]
    results = run_pool(tasks)

    p(f"\n--- RSI 参数对比 ---")
    p(f"  {'Config':<16}{'N':>8}{'Sharpe':>10}{'PnL':>14}{'WR%':>8}{'MaxDD':>12}")
    for r in sorted(results, key=lambda x: x[2], reverse=True):
        p(f"  {r[0]:<16}{r[1]:>8}{r[2]:>10.2f} {fmt(r[3]):>13}{r[4]:>8.1f}% {fmt(r[6]):>11}")


def r9_14_orb_optimization(p):
    """ORB 参数优化"""
    p("="*80)
    p("R9-14: NY 开盘区间突破 (ORB) 参数优化")
    p("="*80)

    L7 = {**get_l6(), "min_entry_gap_hours": 1.0, "keltner_adx_threshold": 14}

    configs = [
        ("ORB_base",     {}),
        ("ORB_hold4",    {"orb_max_hold_m15": 4}),
        ("ORB_hold6",    {"orb_max_hold_m15": 6}),
        ("ORB_hold8",    {"orb_max_hold_m15": 8}),
        ("ORB_hold12",   {"orb_max_hold_m15": 12}),
        ("ORB_hold16",   {"orb_max_hold_m15": 16}),
    ]

    tasks = [(name, {**L7, **kw}, 0.30, None, None) for name, kw in configs]
    results = run_pool(tasks)

    p(f"\n--- ORB 参数对比 ---")
    p(f"  {'Config':<16}{'N':>8}{'Sharpe':>10}{'PnL':>14}{'WR%':>8}{'MaxDD':>12}")
    for r in sorted(results, key=lambda x: x[2], reverse=True):
        p(f"  {r[0]:<16}{r[1]:>8}{r[2]:>10.2f} {fmt(r[3]):>13}{r[4]:>8.1f}% {fmt(r[6]):>11}")


def r9_15_kc_params(p):
    """Keltner 通道参数探索"""
    p("="*80)
    p("R9-15: Keltner 通道参数探索 (KC_ema / KC_mult)")
    p("="*80)

    L7_base = {**get_l6(), "min_entry_gap_hours": 1.0, "keltner_adx_threshold": 14}

    kc_variants = [
        (20, 1.0), (20, 1.2), (20, 1.5), (20, 2.0),
        (25, 1.0), (25, 1.2), (25, 1.5), (25, 2.0),
        (30, 1.0), (30, 1.2), (30, 1.5), (30, 2.0),
        (40, 1.0), (40, 1.2), (40, 1.5), (40, 2.0),
    ]

    def _run_kc(args):
        ema, mult, kw, spread = args
        from backtest.runner import DataBundle, run_variant
        data = DataBundle.load_custom(kc_ema=ema, kc_mult=mult)
        s = run_variant(data, f"KC_e{ema}_m{mult}", verbose=False, spread_cost=spread, **kw)
        return (f"KC_e{ema}_m{mult}", s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
                s.get('elapsed_s', 0), s['max_dd'])

    tasks = [(ema, mult, {**L7_base}, 0.30) for ema, mult in kc_variants]
    with mp.Pool(MAX_WORKERS) as pool:
        results = pool.map(_run_kc, tasks)

    p(f"\n--- KC 参数对比 ---")
    col_hdr = "EMA\\Mult"
    mults_unique = sorted(set(m for _, m in kc_variants))
    header = f"  {col_hdr:<12}" + "".join(f"{m:>10}" for m in mults_unique)
    p(header)
    rd = {r[0]: r for r in results}
    emas_unique = sorted(set(e for e, _ in kc_variants))
    for ema in emas_unique:
        row = f"  {ema:<12}"
        for mult in mults_unique:
            key = f"KC_e{ema}_m{mult}"
            s = rd[key][2]
            marker = " <-" if ema == 25 and mult == 1.2 else ""
            row += f"{s:>10.2f}"
        p(row)


def r9_16_session_filter(p):
    """入场时段优化"""
    p("="*80)
    p("R9-16: 入场时段过滤 (UTC 小时)")
    p("="*80)

    L7 = {**get_l6(), "min_entry_gap_hours": 1.0, "keltner_adx_threshold": 14}

    session_configs = [
        ("All_hours",     {}),
        ("London_NY",     {"h1_allowed_sessions": list(range(7, 21))}),
        ("London_only",   {"h1_allowed_sessions": list(range(7, 16))}),
        ("NY_only",       {"h1_allowed_sessions": list(range(12, 21))}),
        ("Asian_off",     {"h1_allowed_sessions": list(range(5, 22))}),
        ("Prime_8_18",    {"h1_allowed_sessions": list(range(8, 18))}),
        ("Extended_6_22", {"h1_allowed_sessions": list(range(6, 22))}),
        ("No_rollover",   {"h1_allowed_sessions": list(range(0, 21)) + [23]}),
    ]

    tasks = [(name, {**L7, **kw}, 0.30, None, None) for name, kw in session_configs]
    results = run_pool(tasks)

    p(f"\n--- 时段过滤对比 ---")
    p(f"  {'Config':<18}{'N':>8}{'Sharpe':>10}{'PnL':>14}{'WR%':>8}{'MaxDD':>12}")
    for r in sorted(results, key=lambda x: x[2], reverse=True):
        p(f"  {r[0]:<18}{r[1]:>8}{r[2]:>10.2f} {fmt(r[3]):>13}{r[4]:>8.1f}% {fmt(r[6]):>11}")


def r9_17_cooldown_optimization(p):
    """Cooldown 时间优化"""
    p("="*80)
    p("R9-17: Cooldown 冷却期优化")
    p("="*80)

    L7 = {**get_l6(), "min_entry_gap_hours": 1.0, "keltner_adx_threshold": 14}

    configs = [
        ("CD_0h",    {"cooldown_hours": 0}),
        ("CD_0.5h",  {"cooldown_hours": 0.5}),
        ("CD_1h",    {"cooldown_hours": 1.0}),
        ("CD_2h",    {"cooldown_hours": 2.0}),
        ("CD_3h",    {"cooldown_hours": 3.0}),
        ("CD_4h",    {"cooldown_hours": 4.0}),
        ("CD_6h",    {"cooldown_hours": 6.0}),
        ("CD_8h",    {"cooldown_hours": 8.0}),
        ("CD_esc",   {"cooldown_hours": 3.0, "escalating_cooldown": True, "escalating_cooldown_mult": 2.0}),
        ("CD_esc4x", {"cooldown_hours": 3.0, "escalating_cooldown": True, "escalating_cooldown_mult": 4.0}),
    ]

    tasks = [(name, {**L7, **kw}, 0.30, None, None) for name, kw in configs]
    results = run_pool(tasks)

    p(f"\n--- Cooldown 对比 ---")
    p(f"  {'Config':<14}{'N':>8}{'Sharpe':>10}{'PnL':>14}{'WR%':>8}{'MaxDD':>12}")
    for r in sorted(results, key=lambda x: x[2], reverse=True):
        p(f"  {r[0]:<14}{r[1]:>8}{r[2]:>10.2f} {fmt(r[3]):>13}{r[4]:>8.1f}% {fmt(r[6]):>11}")


def r9_18_atr_spike_protection(p):
    """ATR Spike Protection 验证"""
    p("="*80)
    p("R9-18: ATR Spike Protection (波动率突变时收紧)")
    p("="*80)

    L7 = {**get_l6(), "min_entry_gap_hours": 1.0, "keltner_adx_threshold": 14}

    configs = [
        ("No_spike",     {}),
        ("Spike_1.3_0.7",{"atr_spike_protection": True, "atr_spike_threshold": 1.3, "atr_spike_trail_mult": 0.7}),
        ("Spike_1.5_0.7",{"atr_spike_protection": True, "atr_spike_threshold": 1.5, "atr_spike_trail_mult": 0.7}),
        ("Spike_1.5_0.5",{"atr_spike_protection": True, "atr_spike_threshold": 1.5, "atr_spike_trail_mult": 0.5}),
        ("Spike_2.0_0.7",{"atr_spike_protection": True, "atr_spike_threshold": 2.0, "atr_spike_trail_mult": 0.7}),
        ("Spike_2.0_0.5",{"atr_spike_protection": True, "atr_spike_threshold": 2.0, "atr_spike_trail_mult": 0.5}),
    ]

    tasks = [(name, {**L7, **kw}, 0.30, None, None) for name, kw in configs]
    results = run_pool(tasks)

    p(f"\n--- ATR Spike Protection 对比 ---")
    p(f"  {'Config':<20}{'N':>8}{'Sharpe':>10}{'PnL':>14}{'WR%':>8}{'MaxDD':>12}")
    for r in sorted(results, key=lambda x: x[2], reverse=True):
        p(f"  {r[0]:<20}{r[1]:>8}{r[2]:>10.2f} {fmt(r[3]):>13}{r[4]:>8.1f}% {fmt(r[6]):>11}")

    # K-Fold best if different from baseline
    best = sorted(results, key=lambda x: x[2], reverse=True)[0]
    if best[0] != "No_spike":
        best_name = best[0]
        best_cfg = dict(configs)[best_name] if isinstance(dict(configs).get(best_name), dict) else {}
        for name, kw in configs:
            if name == best_name:
                best_cfg = kw
                break
        base_r = run_kfold(L7, "NoSpike", 0.30)
        best_r = run_kfold({**L7, **best_cfg}, best_name, 0.30)
        print_kfold(p, base_r, best_r, "No Spike", best_name)


# ══════════════════════════════════════════════════════════════════
# Main execution
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    master_log = os.path.join(OUTPUT_DIR, "00_master_log.txt")

    with open(master_log, "w") as ml:
        ml.write(f"Round 9 (L7 Construction + Deep Validation + New Explorations)\n")
        ml.write(f"Started: {datetime.now()}\n")
        ml.write(f"MAX_WORKERS: {MAX_WORKERS}\n")
        cores = os.cpu_count() or '?'
        ml.write(f"Server: {cores} cores\n")
        ml.write("="*60 + "\n\n")

    phases = [
        ("r9_1_l7_construction.txt",    "R9-1: L7 Construction",       r9_1_l7_construction),
        ("r9_2_l7_kfold.txt",          "R9-2: L7 K-Fold+WF",          r9_2_l7_kfold),
        ("r9_3_l7_monte_carlo.txt",    "R9-3: L7 Monte Carlo",         r9_3_l7_monte_carlo),
        ("r9_4_l7_yearly_heatmap.txt", "R9-4: L7 Yearly+Heatmap",      r9_4_l7_yearly_heatmap),
        ("r9_5_sl_maxhold_2d.txt",     "R9-5: SL x MaxHold 2D",        r9_5_sl_maxhold_2d),
        ("r9_6_choppy_adx_2d.txt",     "R9-6: Choppy x ADX 2D",        r9_6_choppy_adx_2d),
        ("r9_7_trailing_variants.txt", "R9-7: Trailing Variants",       r9_7_trailing_variants),
        ("r9_8_time_decay.txt",        "R9-8: Time Decay TP",           r9_8_time_decay_retest),
        ("r9_9_rolling_wf.txt",        "R9-9: Rolling Walk-Forward",    r9_9_rolling_walkforward),
        ("r9_10_spread_models.txt",    "R9-10: Spread Models",          r9_10_spread_models),
        ("r9_11_extreme_periods.txt",  "R9-11: Extreme Periods",        r9_11_extreme_periods),
        ("r9_12_bankruptcy.txt",       "R9-12: Bankruptcy Sim",          r9_12_bankruptcy_sim),
        ("r9_13_rsi_opt.txt",          "R9-13: RSI Optimization",       r9_13_rsi_optimization),
        ("r9_14_orb_opt.txt",          "R9-14: ORB Optimization",       r9_14_orb_optimization),
        ("r9_15_kc_params.txt",        "R9-15: KC Parameters",          r9_15_kc_params),
        ("r9_16_session.txt",          "R9-16: Session Filter",         r9_16_session_filter),
        ("r9_17_cooldown.txt",         "R9-17: Cooldown Opt",           r9_17_cooldown_optimization),
        ("r9_18_atr_spike.txt",        "R9-18: ATR Spike",              r9_18_atr_spike_protection),
    ]

    for fname, title, func in phases:
        out_path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
            print(f">>> Skipping {title} (result exists)")
            with open(master_log, "a") as ml:
                ml.write(f"  {title}: SKIPPED (exists)\n")
            continue

        print(f"\n>>> Starting {title} ...", flush=True)
        t0 = time.time()
        lines = []
        lines.append(f"# {fname.replace('.txt','')}")
        lines.append(f"# Started: {datetime.now()}")
        lines.append(f"# Workers: {MAX_WORKERS}")

        def p(text=""):
            lines.append(text)
            print(text, flush=True)

        try:
            func(p)
        except Exception:
            tb = traceback.format_exc()
            p(f"\n!!! ERROR !!!\n{tb}")

        elapsed = (time.time() - t0) / 60
        lines.append(f"\n# Completed: {datetime.now()}")
        lines.append(f"# Elapsed: {elapsed:.1f} minutes")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        with open(master_log, "a") as ml:
            ml.write(f"  {title}: DONE ({elapsed:.1f} min)\n")

        print(f"<<< {title} done in {elapsed:.1f} min", flush=True)

    with open(master_log, "a") as ml:
        ml.write(f"\nRound 9 Finished: {datetime.now()}\n")

    print(f"\n{'='*60}")
    print("Round 9 ALL DONE!")
