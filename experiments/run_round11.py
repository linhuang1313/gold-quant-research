#!/usr/bin/env python3
"""
Round 11 — Price Action 因子全面验证
======================================
灵感来源: 张峻齐 (张Mr stock) 的裸K价格行为交易方法论 + 课程笔记
核心假设:
  H1: K线形态 (Pinbar/顶底分型/孕线/2B吞没) 在关键位出现时可确认信号质量
  H2: 支撑阻力位距离可作为入场过滤器
  H3: 多形态共振 (同一根bar多个形态) 提高信号质量
  H4: 日内波幅已过大时应停止入场 ($15 rule)

=== Phase 1: Pinbar 因子 IC 分析 (~10min) ===
R11-1:  Pinbar 因子扫描 — bull/bear pinbar 与后续收益的 IC
R11-2:  Pinbar 频率统计 — 出现频率、与 Keltner 信号的时间重叠率

=== Phase 2: Pinbar 作为入场确认过滤器 (~30min) ===
R11-3:  Pinbar 确认 — 只在 Keltner 突破 + Pinbar 对齐时入场
R11-4:  Pinbar 确认 K-Fold — 如果 R11-3 有效则做 6-Fold 验证
R11-5:  Pinbar 宽松版 — 前 1-3 根 H1 bar 有 Pinbar 即可

=== Phase 3: 支撑阻力位过滤器 (~30min) ===
R11-6:  S/R 过滤 — 做多时离阻力太近则跳过 (1.0/1.5/2.0/3.0 ATR)
R11-7:  S/R 过滤 K-Fold — 如果最优阈值有效则做 6-Fold 验证
R11-8:  S/R 距离因子 IC — dist_to_resistance/support 与收益的线性关系

=== Phase 4: Pinbar + S/R 组合策略 (~30min) ===
R11-9:  PinbarSR 独立策略 — Pinbar 在 S/R zone 出场 (独立于 Keltner)
R11-10: PinbarSR 叠加 — 在现有 Keltner 基础上叠加 PinbarSR 信号
R11-11: 全组合 — Pinbar确认 + S/R过滤 + PinbarSR 独立

=== Phase 5: 鲁棒性验证 (~60min) ===
R11-12: 最优配置 K-Fold 6折 ($0.30 + $0.50)
R11-13: 最优配置 Walk-Forward 逐年
R11-14: Monte Carlo 100 次参数扰动

=== Phase 6: 新K线形态因子 (顶底分型/孕线/2B) (~30min) ===
R11-15: 新形态 IC 扫描 — fractal/inside_bar/engulfing 因子分析
R11-16: 新形态频率统计 — 各形态频率 + Keltner 重叠率
R11-17: 各形态作为 Keltner 入场确认过滤器 — 单独效果对比
R11-18: 最优 PA 过滤器 K-Fold 验证

=== Phase 7: 日幅过滤 + PA共振 (~20min) ===
R11-19: 日幅过滤器 — $10/$15/$20/$25/$30 日内波幅上限
R11-20: PA 共振过滤 — 要求 >=2 个形态同方向确认
R11-21: 日幅 + PA共振组合

=== Phase 8: 独立 PA+SR 策略 + 全局最优 (~40min) ===
R11-22: 各形态+SR zone 独立策略 — Fractal/InsideBar/Engulf at S/R
R11-23: 全局最优组合 — 最佳过滤 + 最佳独立策略
R11-24: 全局最优 K-Fold 验证
"""
import sys, os, io, time, traceback
import multiprocessing as mp
import numpy as np
from datetime import datetime
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "results/round11_results"
MAX_WORKERS = 22

def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"

# ── Picklable worker functions ────────────────────────────────

def _run_one(args):
    label, kw, spread, start, end = args
    from backtest.runner import DataBundle, run_variant
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    if start and end:
        data = data.slice(start, end)
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kw)
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'],
            s.get('skipped_pinbar', 0), s.get('skipped_sr', 0),
            s.get('pinbar_sr_entries', 0))

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

def run_pool(tasks, func=_run_one):
    with mp.Pool(MAX_WORKERS) as pool:
        return pool.map(func, tasks)


# ── Config presets ─────────────────────────────────────────────

def get_base():
    from backtest.runner import LIVE_PARITY_KWARGS
    return {**LIVE_PARITY_KWARGS}

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-01"),
]

YEARS = [(str(y), f"{y}-01-01", f"{y+1}-01-01") for y in range(2015, 2026)]


# ═══════════════════════════════════════════════════════════════
# Phase 1: Pinbar 因子 IC 分析
# ═══════════════════════════════════════════════════════════════

def run_r11_1(out):
    """R11-1: Pinbar 因子扫描 — IC 分析"""
    print("\n" + "="*70)
    print("R11-1: Pinbar Factor IC Scan")
    print("="*70)

    from backtest.runner import DataBundle
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    h1 = data.h1_df.copy()

    fwd_bars = [1, 2, 4, 8]
    factors = ['pinbar_bull', 'pinbar_bear']

    results = []
    for fwd in fwd_bars:
        h1[f'ret_{fwd}'] = h1['Close'].pct_change(fwd).shift(-fwd)

        for factor in factors:
            if factor not in h1.columns:
                continue
            mask = h1[factor].notna() & h1[f'ret_{fwd}'].notna()
            if mask.sum() < 100:
                continue
            from scipy.stats import spearmanr
            ic, pval = spearmanr(h1.loc[mask, factor], h1.loc[mask, f'ret_{fwd}'])
            results.append((factor, f'ret_{fwd}', mask.sum(), round(ic, 4), round(pval, 4)))

    # Also compute conditional stats: when pinbar fires, what's the avg return?
    for fwd in fwd_bars:
        for factor, direction in [('pinbar_bull', 'BUY'), ('pinbar_bear', 'SELL')]:
            if factor not in h1.columns:
                continue
            fired = h1[h1[factor] > 0]
            if len(fired) < 10:
                continue
            ret_col = f'ret_{fwd}'
            if direction == 'SELL':
                avg_ret = -fired[ret_col].mean()
            else:
                avg_ret = fired[ret_col].mean()
            results.append((f'{factor}_dir_{direction}', f'ret_{fwd}',
                           len(fired), round(avg_ret * 10000, 2), 0))

    with open(f"{out}/R11-1_pinbar_ic.txt", 'w') as f:
        f.write("Pinbar Factor IC Scan\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Factor':<25} {'Return':<10} {'N':>8} {'IC/AvgRet':>10} {'p-value':>10}\n")
        f.write("-"*65 + "\n")
        for row in results:
            f.write(f"{row[0]:<25} {row[1]:<10} {row[2]:>8} {row[3]:>10.4f} {row[4]:>10.4f}\n")

    print(f"  Results saved to {out}/R11-1_pinbar_ic.txt")
    for r in results:
        print(f"    {r[0]:<25} {r[1]:<10} N={r[2]:>6} IC={r[3]:>8.4f}")


def run_r11_2(out):
    """R11-2: Pinbar 频率统计 + Keltner 重叠率"""
    print("\n" + "="*70)
    print("R11-2: Pinbar Frequency & Keltner Overlap")
    print("="*70)

    from backtest.runner import DataBundle, run_variant
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

    # Run baseline to get Keltner entry times
    base = get_base()
    s = run_variant(data, "baseline_for_overlap", verbose=False, spread_cost=0.30, **base)
    trades = s.get('_trades', [])
    kc_times = set()
    for t in trades:
        if t.strategy == 'keltner':
            kc_times.add(t.entry_time.replace(minute=0, second=0, microsecond=0))

    h1 = data.h1_df
    total_bars = len(h1)
    bull_bars = int((h1.get('pinbar_bull', 0) > 0).sum())
    bear_bars = int((h1.get('pinbar_bear', 0) > 0).sum())
    any_pinbar = int(((h1.get('pinbar_bull', 0) > 0) | (h1.get('pinbar_bear', 0) > 0)).sum())

    # Check overlap with Keltner entries
    overlap_count = 0
    for idx in h1.index:
        if h1.loc[idx].get('pinbar_bull', 0) > 0 or h1.loc[idx].get('pinbar_bear', 0) > 0:
            rounded = idx.replace(minute=0, second=0, microsecond=0)
            if rounded in kc_times:
                overlap_count += 1

    report = (
        f"Pinbar Frequency & Keltner Overlap\n{'='*50}\n\n"
        f"Total H1 bars: {total_bars:,}\n"
        f"Bull Pinbar: {bull_bars} ({bull_bars/total_bars*100:.2f}%)\n"
        f"Bear Pinbar: {bear_bars} ({bear_bars/total_bars*100:.2f}%)\n"
        f"Any Pinbar:  {any_pinbar} ({any_pinbar/total_bars*100:.2f}%)\n\n"
        f"Keltner entries: {len(kc_times)}\n"
        f"Pinbar + Keltner overlap: {overlap_count} ({overlap_count/max(1,len(kc_times))*100:.1f}% of KC entries)\n"
        f"Pinbar at entry = {overlap_count/max(1,any_pinbar)*100:.1f}% of all Pinbars\n"
    )
    with open(f"{out}/R11-2_pinbar_freq.txt", 'w') as f:
        f.write(report)
    print(report)


# ═══════════════════════════════════════════════════════════════
# Phase 2: Pinbar 作为入场确认过滤器
# ═══════════════════════════════════════════════════════════════

def run_r11_3(out):
    """R11-3: Pinbar confirmation filter — full sample"""
    print("\n" + "="*70)
    print("R11-3: Pinbar Confirmation Filter (Full Sample)")
    print("="*70)

    base = get_base()
    tasks = [
        ("Baseline_$0.30", {**base}, 0.30, None, None),
        ("Baseline_$0.50", {**base}, 0.50, None, None),
        ("Pinbar_Confirm_$0.30", {**base, 'pinbar_confirmation': True}, 0.30, None, None),
        ("Pinbar_Confirm_$0.50", {**base, 'pinbar_confirmation': True}, 0.50, None, None),
    ]
    results = run_pool(tasks)

    with open(f"{out}/R11-3_pinbar_confirm.txt", 'w') as f:
        f.write("Pinbar Confirmation Filter\n" + "="*80 + "\n\n")
        f.write(f"{'Label':<30} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} {'MaxDD':>10} {'Skip_PB':>8} {'Time':>6}\n")
        f.write("-"*90 + "\n")
        for r in results:
            f.write(f"{r[0]:<30} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1f}% {fmt(r[6]):>10} {r[7]:>8} {r[5]:>5.0f}s\n")
    for r in results:
        print(f"  {r[0]:<30} N={r[1]:>5} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])} Skip_PB={r[7]}")


def run_r11_4(out):
    """R11-4: Pinbar confirmation K-Fold (if R11-3 shows improvement)"""
    print("\n" + "="*70)
    print("R11-4: Pinbar Confirmation K-Fold 6x ($0.30 + $0.50)")
    print("="*70)

    base = get_base()
    tasks = []
    for spread in [0.30, 0.50]:
        sp_tag = f"sp{spread:.2f}"
        for fname, start, end in FOLDS:
            tasks.append((f"Base_{sp_tag}_{fname}", {**base}, spread, start, end))
            tasks.append((f"PB_{sp_tag}_{fname}", {**base, 'pinbar_confirmation': True}, spread, start, end))

    results = run_pool(tasks)
    base_by_fold = {}
    pb_by_fold = {}
    for r in results:
        label = r[0]
        if label.startswith("Base_"):
            base_by_fold[label] = r
        else:
            pb_by_fold[label] = r

    with open(f"{out}/R11-4_pinbar_kfold.txt", 'w') as f:
        f.write("Pinbar Confirmation K-Fold\n" + "="*80 + "\n\n")
        for spread in [0.30, 0.50]:
            sp_tag = f"sp{spread:.2f}"
            f.write(f"\n--- Spread ${spread:.2f} ---\n")
            f.write(f"{'Fold':<10} {'Base_N':>6} {'Base_Sharpe':>12} {'PB_N':>6} {'PB_Sharpe':>12} {'Delta':>8}\n")
            f.write("-"*60 + "\n")
            wins = 0
            for fname, _, _ in FOLDS:
                b = base_by_fold.get(f"Base_{sp_tag}_{fname}")
                p = pb_by_fold.get(f"PB_{sp_tag}_{fname}")
                if b and p:
                    delta = p[2] - b[2]
                    if delta > 0:
                        wins += 1
                    f.write(f"{fname:<10} {b[1]:>6} {b[2]:>12.2f} {p[1]:>6} {p[2]:>12.2f} {delta:>+8.2f}\n")
            f.write(f"\nK-Fold PASS: {wins}/6 folds (need >= 5)\n")
            print(f"  ${spread:.2f}: {wins}/6 folds win")


def run_r11_5(out):
    """R11-5: Relaxed Pinbar — check lookback 1-3 bars"""
    print("\n" + "="*70)
    print("R11-5: Relaxed Pinbar (lookback 1-3 bars)")
    print("="*70)

    base = get_base()
    tasks = [
        ("Baseline_$0.30", {**base}, 0.30, None, None),
    ]
    # We test by using pinbar_confirmation but changing our indicator
    # For now just test with strict pinbar as-is; lookback variants would
    # need engine changes. Instead test SR combinations.
    # Just re-run baseline for comparison and note the pinbar stats
    results = run_pool(tasks)
    with open(f"{out}/R11-5_relaxed_pinbar.txt", 'w') as f:
        f.write("Relaxed Pinbar (placeholder — strict only implemented)\n")
        f.write("See R11-3 for strict Pinbar confirmation results.\n")
    print("  Relaxed pinbar variants deferred — strict version in R11-3")


# ═══════════════════════════════════════════════════════════════
# Phase 3: 支撑阻力位过滤器
# ═══════════════════════════════════════════════════════════════

def run_r11_6(out):
    """R11-6: S/R proximity filter — grid search"""
    print("\n" + "="*70)
    print("R11-6: Support/Resistance Proximity Filter Grid")
    print("="*70)

    base = get_base()
    thresholds = [0.5, 1.0, 1.5, 2.0, 3.0]
    tasks = [("Baseline_$0.30", {**base}, 0.30, None, None)]
    for th in thresholds:
        tasks.append((f"SR_{th:.1f}ATR_$0.30", {**base, 'sr_filter_atr': th}, 0.30, None, None))
        tasks.append((f"SR_{th:.1f}ATR_$0.50", {**base, 'sr_filter_atr': th}, 0.50, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R11-6_sr_filter.txt", 'w') as f:
        f.write("S/R Proximity Filter Grid\n" + "="*80 + "\n\n")
        f.write(f"{'Label':<30} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} {'MaxDD':>10} {'Skip_SR':>8}\n")
        f.write("-"*85 + "\n")
        for r in results:
            f.write(f"{r[0]:<30} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1f}% {fmt(r[6]):>10} {r[8]:>8}\n")
    for r in results:
        print(f"  {r[0]:<30} N={r[1]:>5} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])} Skip_SR={r[8]}")


def run_r11_7(out):
    """R11-7: Best S/R threshold K-Fold"""
    print("\n" + "="*70)
    print("R11-7: S/R Filter K-Fold (best threshold from R11-6)")
    print("="*70)

    base = get_base()
    best_sr = 1.5  # will be adjusted after R11-6 results
    tasks = []
    for spread in [0.30, 0.50]:
        sp_tag = f"sp{spread:.2f}"
        for fname, start, end in FOLDS:
            tasks.append((f"Base_{sp_tag}_{fname}", {**base}, spread, start, end))
            tasks.append((f"SR{best_sr}_{sp_tag}_{fname}",
                         {**base, 'sr_filter_atr': best_sr}, spread, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R11-7_sr_kfold.txt", 'w') as f:
        f.write(f"S/R Filter K-Fold (threshold={best_sr})\n" + "="*80 + "\n\n")
        for spread in [0.30, 0.50]:
            sp_tag = f"sp{spread:.2f}"
            f.write(f"\n--- Spread ${spread:.2f} ---\n")
            f.write(f"{'Fold':<10} {'Base_Sharpe':>12} {'SR_Sharpe':>12} {'Delta':>8}\n")
            f.write("-"*45 + "\n")
            wins = 0
            for fname, _, _ in FOLDS:
                b = next((r for r in results if r[0] == f"Base_{sp_tag}_{fname}"), None)
                s = next((r for r in results if r[0] == f"SR{best_sr}_{sp_tag}_{fname}"), None)
                if b and s:
                    delta = s[2] - b[2]
                    if delta > 0:
                        wins += 1
                    f.write(f"{fname:<10} {b[2]:>12.2f} {s[2]:>12.2f} {delta:>+8.2f}\n")
            f.write(f"\nK-Fold: {wins}/6 folds win\n")
            print(f"  ${spread:.2f}: {wins}/6")


def run_r11_8(out):
    """R11-8: S/R distance factor IC — linear predictive power"""
    print("\n" + "="*70)
    print("R11-8: S/R Distance Factor IC Scan")
    print("="*70)

    from backtest.runner import DataBundle
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    h1 = data.h1_df.copy()

    factors = ['dist_to_resistance', 'dist_to_support']
    fwd_bars = [1, 2, 4, 8]
    results = []
    for fwd in fwd_bars:
        h1[f'ret_{fwd}'] = h1['Close'].pct_change(fwd).shift(-fwd)
        for factor in factors:
            if factor not in h1.columns:
                continue
            mask = h1[factor].notna() & h1[f'ret_{fwd}'].notna() & (h1[factor].abs() < 50)
            if mask.sum() < 100:
                continue
            from scipy.stats import spearmanr
            ic, pval = spearmanr(h1.loc[mask, factor], h1.loc[mask, f'ret_{fwd}'])
            results.append((factor, f'ret_{fwd}', mask.sum(), round(ic, 4), round(pval, 4)))

    with open(f"{out}/R11-8_sr_ic.txt", 'w') as f:
        f.write("S/R Distance Factor IC Scan\n" + "="*60 + "\n\n")
        f.write(f"{'Factor':<25} {'Return':<10} {'N':>8} {'IC':>10} {'p-val':>10}\n")
        f.write("-"*65 + "\n")
        for r in results:
            f.write(f"{r[0]:<25} {r[1]:<10} {r[2]:>8} {r[3]:>10.4f} {r[4]:>10.4f}\n")
    for r in results:
        print(f"  {r[0]:<25} {r[1]:<10} N={r[2]:>6} IC={r[3]:>8.4f}")


# ═══════════════════════════════════════════════════════════════
# Phase 4: Pinbar + S/R 组合策略
# ═══════════════════════════════════════════════════════════════

def run_r11_9(out):
    """R11-9: PinbarSR standalone strategy"""
    print("\n" + "="*70)
    print("R11-9: PinbarSR Standalone Strategy")
    print("="*70)

    base = get_base()
    tasks = [
        ("Baseline_$0.30", {**base}, 0.30, None, None),
        ("Baseline_$0.50", {**base}, 0.50, None, None),
    ]
    for zone in [1.0, 1.5, 2.0, 3.0]:
        tasks.append((f"PinbarSR_z{zone}_$0.30",
                      {**base, 'pinbar_sr_strategy': True, 'pinbar_sr_atr_zone': zone},
                      0.30, None, None))
        tasks.append((f"PinbarSR_z{zone}_$0.50",
                      {**base, 'pinbar_sr_strategy': True, 'pinbar_sr_atr_zone': zone},
                      0.50, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R11-9_pinbar_sr.txt", 'w') as f:
        f.write("PinbarSR Standalone Strategy\n" + "="*80 + "\n\n")
        f.write(f"{'Label':<30} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} {'MaxDD':>10} {'PB_SR':>6}\n")
        f.write("-"*85 + "\n")
        for r in results:
            f.write(f"{r[0]:<30} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1f}% {fmt(r[6]):>10} {r[9]:>6}\n")
    for r in results:
        print(f"  {r[0]:<30} N={r[1]:>5} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])} PB_SR={r[9]}")


def run_r11_10(out):
    """R11-10: Pinbar confirm + S/R filter combo"""
    print("\n" + "="*70)
    print("R11-10: Combined Pinbar+SR Filter")
    print("="*70)

    base = get_base()
    tasks = [
        ("Baseline", {**base}, 0.30, None, None),
        ("PB_only", {**base, 'pinbar_confirmation': True}, 0.30, None, None),
        ("SR_1.5_only", {**base, 'sr_filter_atr': 1.5}, 0.30, None, None),
        ("PB+SR_1.5", {**base, 'pinbar_confirmation': True, 'sr_filter_atr': 1.5}, 0.30, None, None),
        ("PB+SR_2.0", {**base, 'pinbar_confirmation': True, 'sr_filter_atr': 2.0}, 0.30, None, None),
        ("PB+PinbarSR", {**base, 'pinbar_confirmation': True, 'pinbar_sr_strategy': True}, 0.30, None, None),
        ("SR_1.5+PinbarSR", {**base, 'sr_filter_atr': 1.5, 'pinbar_sr_strategy': True}, 0.30, None, None),
        ("All_combo", {**base, 'pinbar_confirmation': True, 'sr_filter_atr': 1.5,
                       'pinbar_sr_strategy': True}, 0.30, None, None),
    ]
    results = run_pool(tasks)

    with open(f"{out}/R11-10_combo.txt", 'w') as f:
        f.write("Combined Price Action Filters\n" + "="*80 + "\n\n")
        f.write(f"{'Label':<25} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} {'MaxDD':>10}\n")
        f.write("-"*70 + "\n")
        for r in results:
            f.write(f"{r[0]:<25} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1f}% {fmt(r[6]):>10}\n")
    for r in results:
        print(f"  {r[0]:<25} N={r[1]:>5} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


# ═══════════════════════════════════════════════════════════════
# Phase 5: 鲁棒性验证
# ═══════════════════════════════════════════════════════════════

def run_r11_12(out):
    """R11-12: Best config K-Fold (placeholder — config adjusted after Phase 2-4)"""
    print("\n" + "="*70)
    print("R11-12: Best Config K-Fold ($0.30 + $0.50)")
    print("="*70)
    print("  [Deferred until Phase 2-4 results determine best config]")


def run_r11_13(out):
    """R11-13: Walk-Forward yearly"""
    print("\n" + "="*70)
    print("R11-13: Walk-Forward Yearly")
    print("="*70)

    base = get_base()
    # Test with the most promising config from Phase 2-4
    best_kw = {**base, 'pinbar_sr_strategy': True, 'pinbar_sr_atr_zone': 2.0}

    tasks = []
    for yname, start, end in YEARS:
        tasks.append((f"Base_{yname}", {**base}, 0.30, start, end))
        tasks.append((f"PBSR_{yname}", best_kw, 0.30, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R11-13_walkforward.txt", 'w') as f:
        f.write("Walk-Forward Yearly Comparison\n" + "="*80 + "\n\n")
        f.write(f"{'Year':<8} {'Base_N':>6} {'Base_Sharpe':>12} {'PBSR_N':>6} {'PBSR_Sharpe':>12} {'Delta':>8}\n")
        f.write("-"*55 + "\n")
        wins = 0
        for yname, _, _ in YEARS:
            b = next((r for r in results if r[0] == f"Base_{yname}"), None)
            p = next((r for r in results if r[0] == f"PBSR_{yname}"), None)
            if b and p:
                delta = p[2] - b[2]
                if delta > 0:
                    wins += 1
                f.write(f"{yname:<8} {b[1]:>6} {b[2]:>12.2f} {p[1]:>6} {p[2]:>12.2f} {delta:>+8.2f}\n")
        f.write(f"\nWins: {wins}/{len(YEARS)} years\n")
        print(f"  Walk-Forward: {wins}/{len(YEARS)} years win")


def run_r11_14(out):
    """R11-14: Monte Carlo 100x parameter perturbation"""
    print("\n" + "="*70)
    print("R11-14: Monte Carlo 100x Perturbation")
    print("="*70)

    import random
    base = get_base()
    best_kw = {**base, 'pinbar_sr_strategy': True, 'pinbar_sr_atr_zone': 2.0}

    tasks = []
    for i in range(100):
        kw = {**best_kw}
        rc = kw.get('regime_config', {})
        new_rc = {}
        for regime, params in rc.items():
            new_rc[regime] = {
                'trail_act': params['trail_act'] * random.uniform(0.85, 1.15),
                'trail_dist': params['trail_dist'] * random.uniform(0.85, 1.15),
            }
        kw['regime_config'] = new_rc
        if 'sl_atr_mult' in kw:
            kw['sl_atr_mult'] *= random.uniform(0.85, 1.15)
        if 'pinbar_sr_atr_zone' in kw:
            kw['pinbar_sr_atr_zone'] *= random.uniform(0.85, 1.15)
        tasks.append((f"MC_{i:03d}", kw, 0.30, None, None))

    results = run_pool(tasks)
    sharpes = [r[2] for r in results]
    pnls = [r[3] for r in results]
    profitable = sum(1 for p in pnls if p > 0)

    report = (
        f"Monte Carlo 100x Perturbation (±15%)\n{'='*50}\n\n"
        f"Sharpe: mean={np.mean(sharpes):.2f}, std={np.std(sharpes):.2f}, "
        f"min={np.min(sharpes):.2f}, max={np.max(sharpes):.2f}\n"
        f"PnL: mean=${np.mean(pnls):,.0f}, min=${np.min(pnls):,.0f}\n"
        f"Profitable: {profitable}/100 ({profitable}%)\n"
    )
    with open(f"{out}/R11-14_monte_carlo.txt", 'w') as f:
        f.write(report)
    print(report)


# ═══════════════════════════════════════════════════════════════
# Phase 6: 新K线形态因子扫描 (顶底分型 / 孕线 / 2B吞没)
# ═══════════════════════════════════════════════════════════════

def run_r11_15(out):
    """R11-15: 新K线形态因子 IC 扫描 (top/bot_fractal, inside_bar, engulfing)"""
    print("\n" + "="*70)
    print("R11-15: New PA Pattern Factor IC Scan")
    print("="*70)

    from backtest.runner import DataBundle
    from scipy.stats import spearmanr
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    h1 = data.h1_df.copy()

    factors = ['top_fractal', 'bot_fractal',
               'inside_bar_bull', 'inside_bar_bear',
               'engulf_bull', 'engulf_bear',
               'pa_bull_count', 'pa_bear_count']
    fwd_bars = [1, 2, 4, 8]
    results = []

    for fwd in fwd_bars:
        h1[f'ret_{fwd}'] = h1['Close'].pct_change(fwd).shift(-fwd)
        for factor in factors:
            if factor not in h1.columns:
                continue
            mask = h1[factor].notna() & h1[f'ret_{fwd}'].notna()
            if mask.sum() < 100:
                continue
            ic, pval = spearmanr(h1.loc[mask, factor], h1.loc[mask, f'ret_{fwd}'])
            results.append((factor, f'ret_{fwd}', mask.sum(), round(ic, 4), round(pval, 4)))

    # Conditional returns when patterns fire
    pa_pairs = [
        ('bot_fractal', 'BUY'), ('top_fractal', 'SELL'),
        ('inside_bar_bull', 'BUY'), ('inside_bar_bear', 'SELL'),
        ('engulf_bull', 'BUY'), ('engulf_bear', 'SELL'),
    ]
    for fwd in fwd_bars:
        for factor, direction in pa_pairs:
            if factor not in h1.columns:
                continue
            fired = h1[h1[factor] > 0]
            if len(fired) < 10:
                continue
            ret_col = f'ret_{fwd}'
            avg_ret = -fired[ret_col].mean() if direction == 'SELL' else fired[ret_col].mean()
            results.append((f'{factor}_dir_{direction}', f'ret_{fwd}',
                           len(fired), round(avg_ret * 10000, 2), 0))

    with open(f"{out}/R11-15_pa_ic.txt", 'w') as f:
        f.write("New PA Pattern Factor IC Scan\n" + "="*70 + "\n\n")
        hdr = f"{'Factor':<30} {'Return':<10} {'N':>8} {'IC/AvgRet':>10} {'p-value':>10}\n"
        f.write(hdr + "-"*70 + "\n")
        for r in results:
            f.write(f"{r[0]:<30} {r[1]:<10} {r[2]:>8} {r[3]:>10.4f} {r[4]:>10.4f}\n")

    print(f"  Results saved to {out}/R11-15_pa_ic.txt")
    for r in results:
        print(f"    {r[0]:<30} {r[1]:<10} N={r[2]:>6} IC={r[3]:>8.4f}")


def run_r11_16(out):
    """R11-16: 各形态频率统计 + Keltner 重叠率"""
    print("\n" + "="*70)
    print("R11-16: PA Pattern Frequency & Keltner Overlap")
    print("="*70)

    from backtest.runner import DataBundle, run_variant
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

    base = get_base()
    s = run_variant(data, "baseline_freq", verbose=False, spread_cost=0.30, **base)
    trades = s.get('_trades', [])
    kc_times = set()
    for t in trades:
        if t.strategy == 'keltner':
            kc_times.add(t.entry_time.replace(minute=0, second=0, microsecond=0))

    h1 = data.h1_df
    total = len(h1)
    patterns = {
        'top_fractal': 0, 'bot_fractal': 0,
        'inside_bar_bull': 0, 'inside_bar_bear': 0,
        'engulf_bull': 0, 'engulf_bear': 0,
        'pinbar_bull': 0, 'pinbar_bear': 0,
    }
    overlaps = {k: 0 for k in patterns}

    for idx in h1.index:
        row = h1.loc[idx]
        rounded = idx.replace(minute=0, second=0, microsecond=0)
        for pat in patterns:
            if float(row.get(pat, 0)) > 0:
                patterns[pat] += 1
                if rounded in kc_times:
                    overlaps[pat] += 1

    lines = [f"PA Pattern Frequency & Keltner Overlap\n{'='*60}\n\n"]
    lines.append(f"Total H1 bars: {total:,}   Keltner entries: {len(kc_times)}\n\n")
    lines.append(f"{'Pattern':<22} {'Count':>8} {'Freq%':>8} {'KC_Overlap':>10} {'Overlap%':>10}\n")
    lines.append("-"*60 + "\n")
    for p, cnt in sorted(patterns.items(), key=lambda x: -x[1]):
        ov = overlaps[p]
        lines.append(f"{p:<22} {cnt:>8} {cnt/total*100:>7.2f}% {ov:>10} {ov/max(1,cnt)*100:>9.1f}%\n")

    report = ''.join(lines)
    with open(f"{out}/R11-16_pa_freq.txt", 'w') as f:
        f.write(report)
    print(report)


def run_r11_17(out):
    """R11-17: 各形态单独作为 Keltner 入场确认过滤器"""
    print("\n" + "="*70)
    print("R11-17: Individual PA Confirmation Filters (Full Sample)")
    print("="*70)

    base = get_base()
    tasks = [
        ("Baseline_$0.30", {**base}, 0.30, None, None),
        ("Baseline_$0.50", {**base}, 0.50, None, None),
        ("Pinbar_$0.30", {**base, 'pinbar_confirmation': True}, 0.30, None, None),
        ("Fractal_$0.30", {**base, 'fractal_confirmation': True}, 0.30, None, None),
        ("InsideBar_$0.30", {**base, 'inside_bar_confirmation': True}, 0.30, None, None),
        ("Engulf_$0.30", {**base, 'engulf_confirmation': True}, 0.30, None, None),
        ("AnyPA_$0.30", {**base, 'any_pa_confirmation': True}, 0.30, None, None),
        ("Pinbar_$0.50", {**base, 'pinbar_confirmation': True}, 0.50, None, None),
        ("Fractal_$0.50", {**base, 'fractal_confirmation': True}, 0.50, None, None),
        ("InsideBar_$0.50", {**base, 'inside_bar_confirmation': True}, 0.50, None, None),
        ("Engulf_$0.50", {**base, 'engulf_confirmation': True}, 0.50, None, None),
        ("AnyPA_$0.50", {**base, 'any_pa_confirmation': True}, 0.50, None, None),
    ]
    results = run_pool(tasks)

    with open(f"{out}/R11-17_pa_filters.txt", 'w') as f:
        f.write("Individual PA Confirmation Filters\n" + "="*80 + "\n\n")
        f.write(f"{'Label':<25} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} {'MaxDD':>10} {'Time':>6}\n")
        f.write("-"*80 + "\n")
        for r in results:
            f.write(f"{r[0]:<25} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1f}% {fmt(r[6]):>10} {r[5]:>5.0f}s\n")
    for r in results:
        print(f"  {r[0]:<25} N={r[1]:>5} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


def run_r11_18(out):
    """R11-18: 最优 PA 确认过滤器 K-Fold 验证"""
    print("\n" + "="*70)
    print("R11-18: Best PA Filter K-Fold 6x")
    print("="*70)

    base = get_base()
    # Test all four individual PA filters in K-Fold
    pa_configs = {
        'Pinbar': {'pinbar_confirmation': True},
        'Fractal': {'fractal_confirmation': True},
        'InsideBar': {'inside_bar_confirmation': True},
        'Engulf': {'engulf_confirmation': True},
        'AnyPA': {'any_pa_confirmation': True},
    }

    tasks = []
    for spread in [0.30, 0.50]:
        sp_tag = f"sp{spread:.2f}"
        for fname, start, end in FOLDS:
            tasks.append((f"Base_{sp_tag}_{fname}", {**base}, spread, start, end))
            for pa_name, pa_kw in pa_configs.items():
                tasks.append((f"{pa_name}_{sp_tag}_{fname}", {**base, **pa_kw}, spread, start, end))

    results = run_pool(tasks)
    result_map = {r[0]: r for r in results}

    with open(f"{out}/R11-18_pa_kfold.txt", 'w') as f:
        f.write("PA Filter K-Fold Comparison\n" + "="*80 + "\n\n")
        for spread in [0.30, 0.50]:
            sp_tag = f"sp{spread:.2f}"
            f.write(f"\n--- Spread ${spread:.2f} ---\n")
            for pa_name in pa_configs:
                wins = 0
                f.write(f"\n  {pa_name}:\n")
                f.write(f"  {'Fold':<10} {'Base':>10} {'PA':>10} {'Delta':>8}\n")
                for fname, _, _ in FOLDS:
                    b = result_map.get(f"Base_{sp_tag}_{fname}")
                    p = result_map.get(f"{pa_name}_{sp_tag}_{fname}")
                    if b and p:
                        delta = p[2] - b[2]
                        if delta > 0:
                            wins += 1
                        f.write(f"  {fname:<10} {b[2]:>10.2f} {p[2]:>10.2f} {delta:>+8.2f}\n")
                f.write(f"  Result: {wins}/6 folds win\n")
                print(f"  {pa_name} ${spread:.2f}: {wins}/6")


# ═══════════════════════════════════════════════════════════════
# Phase 7: 日幅过滤 + PA 共振
# ═══════════════════════════════════════════════════════════════

def run_r11_19(out):
    """R11-19: 日幅过滤器 ($15 / $20 / $25 / $30 rule)"""
    print("\n" + "="*70)
    print("R11-19: Daily Range Filter Grid")
    print("="*70)

    base = get_base()
    thresholds = [10.0, 15.0, 20.0, 25.0, 30.0]
    tasks = [("Baseline_$0.30", {**base}, 0.30, None, None)]
    for th in thresholds:
        tasks.append((f"DR_{th:.0f}_$0.30", {**base, 'daily_range_filter': th}, 0.30, None, None))
        tasks.append((f"DR_{th:.0f}_$0.50", {**base, 'daily_range_filter': th}, 0.50, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R11-19_daily_range.txt", 'w') as f:
        f.write("Daily Range Filter Grid\n" + "="*80 + "\n\n")
        f.write(f"{'Label':<25} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} {'MaxDD':>10} {'Skipped':>8}\n")
        f.write("-"*80 + "\n")
        for r in results:
            f.write(f"{r[0]:<25} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1f}% {fmt(r[6]):>10}\n")
    for r in results:
        print(f"  {r[0]:<25} N={r[1]:>5} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


def run_r11_20(out):
    """R11-20: PA 共振过滤 (要求 >=2 个形态同方向确认)"""
    print("\n" + "="*70)
    print("R11-20: PA Confluence Filter (min 2+ aligned patterns)")
    print("="*70)

    base = get_base()
    tasks = [
        ("Baseline_$0.30", {**base}, 0.30, None, None),
        ("AnyPA_$0.30", {**base, 'any_pa_confirmation': True}, 0.30, None, None),
        ("Confluence2_$0.30", {**base, 'pa_confluence_min': 2}, 0.30, None, None),
        ("Confluence3_$0.30", {**base, 'pa_confluence_min': 3}, 0.30, None, None),
        ("AnyPA_$0.50", {**base, 'any_pa_confirmation': True}, 0.50, None, None),
        ("Confluence2_$0.50", {**base, 'pa_confluence_min': 2}, 0.50, None, None),
        ("Confluence3_$0.50", {**base, 'pa_confluence_min': 3}, 0.50, None, None),
    ]
    results = run_pool(tasks)

    with open(f"{out}/R11-20_confluence.txt", 'w') as f:
        f.write("PA Confluence Filter\n" + "="*80 + "\n\n")
        f.write(f"{'Label':<25} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} {'MaxDD':>10}\n")
        f.write("-"*70 + "\n")
        for r in results:
            f.write(f"{r[0]:<25} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1f}% {fmt(r[6]):>10}\n")
    for r in results:
        print(f"  {r[0]:<25} N={r[1]:>5} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


def run_r11_21(out):
    """R11-21: 日幅过滤 + PA 共振组合"""
    print("\n" + "="*70)
    print("R11-21: Daily Range + PA Confluence Combo")
    print("="*70)

    base = get_base()
    tasks = [
        ("Baseline", {**base}, 0.30, None, None),
        ("DR15", {**base, 'daily_range_filter': 15.0}, 0.30, None, None),
        ("AnyPA", {**base, 'any_pa_confirmation': True}, 0.30, None, None),
        ("DR15+AnyPA", {**base, 'daily_range_filter': 15.0, 'any_pa_confirmation': True}, 0.30, None, None),
        ("DR20+AnyPA", {**base, 'daily_range_filter': 20.0, 'any_pa_confirmation': True}, 0.30, None, None),
        ("DR15+Confl2", {**base, 'daily_range_filter': 15.0, 'pa_confluence_min': 2}, 0.30, None, None),
        ("DR20+Confl2", {**base, 'daily_range_filter': 20.0, 'pa_confluence_min': 2}, 0.30, None, None),
    ]
    results = run_pool(tasks)

    with open(f"{out}/R11-21_dr_pa_combo.txt", 'w') as f:
        f.write("Daily Range + PA Confluence Combo\n" + "="*80 + "\n\n")
        f.write(f"{'Label':<20} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} {'MaxDD':>10}\n")
        f.write("-"*65 + "\n")
        for r in results:
            f.write(f"{r[0]:<20} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1f}% {fmt(r[6]):>10}\n")
    for r in results:
        print(f"  {r[0]:<20} N={r[1]:>5} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


# ═══════════════════════════════════════════════════════════════
# Phase 8: 独立 PA+SR 策略 + 全局最优
# ═══════════════════════════════════════════════════════════════

def run_r11_22(out):
    """R11-22: 各形态+SR zone 独立策略对比"""
    print("\n" + "="*70)
    print("R11-22: PA+SR Standalone Strategy Comparison")
    print("="*70)

    base = get_base()
    tasks = [
        ("Baseline_$0.30", {**base}, 0.30, None, None),
    ]
    strats = [
        ('PinbarSR', {'pinbar_sr_strategy': True}),
        ('FractalSR', {'fractal_sr_strategy': True}),
        ('InsideBarSR', {'inside_bar_sr_strategy': True}),
        ('EngulfSR', {'engulf_sr_strategy': True}),
    ]
    for sname, skw in strats:
        for zone in [1.5, 2.0, 3.0]:
            tasks.append((f"{sname}_z{zone}_$0.30",
                          {**base, **skw, 'pinbar_sr_atr_zone': zone}, 0.30, None, None))
            tasks.append((f"{sname}_z{zone}_$0.50",
                          {**base, **skw, 'pinbar_sr_atr_zone': zone}, 0.50, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R11-22_pa_sr_strats.txt", 'w') as f:
        f.write("PA+SR Standalone Strategy Comparison\n" + "="*80 + "\n\n")
        f.write(f"{'Label':<30} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} {'MaxDD':>10}\n")
        f.write("-"*75 + "\n")
        for r in results:
            f.write(f"{r[0]:<30} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1f}% {fmt(r[6]):>10}\n")
    for r in results:
        print(f"  {r[0]:<30} N={r[1]:>5} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


def run_r11_23(out):
    """R11-23: 全局最优组合 — 最好的 PA 过滤 + 最好的独立策略"""
    print("\n" + "="*70)
    print("R11-23: Global Best Combo")
    print("="*70)

    base = get_base()
    tasks = [
        ("Baseline", {**base}, 0.30, None, None),
        # Combo 1: AnyPA filter + SR filter
        ("AnyPA+SR1.5", {**base, 'any_pa_confirmation': True, 'sr_filter_atr': 1.5}, 0.30, None, None),
        # Combo 2: AnyPA filter + PinbarSR standalone
        ("AnyPA+PinbarSR", {**base, 'any_pa_confirmation': True,
                            'pinbar_sr_strategy': True}, 0.30, None, None),
        # Combo 3: AnyPA + all PA+SR standalone
        ("AnyPA+AllPASR", {**base, 'any_pa_confirmation': True,
                           'pinbar_sr_strategy': True, 'fractal_sr_strategy': True,
                           'inside_bar_sr_strategy': True, 'engulf_sr_strategy': True}, 0.30, None, None),
        # Combo 4: DR15 + AnyPA + PinbarSR
        ("DR15+AnyPA+PbSR", {**base, 'daily_range_filter': 15.0,
                              'any_pa_confirmation': True,
                              'pinbar_sr_strategy': True}, 0.30, None, None),
        # Combo 5: full kitchen sink
        ("KitchenSink", {**base, 'daily_range_filter': 20.0,
                         'any_pa_confirmation': True, 'sr_filter_atr': 1.5,
                         'pinbar_sr_strategy': True, 'fractal_sr_strategy': True,
                         'engulf_sr_strategy': True}, 0.30, None, None),
    ]
    # Duplicate with $0.50
    tasks_050 = [(f"{l}_$0.50", kw, 0.50, s, e) for l, kw, _, s, e in tasks[1:]]
    tasks += tasks_050

    results = run_pool(tasks)

    with open(f"{out}/R11-23_global_best.txt", 'w') as f:
        f.write("Global Best Combo\n" + "="*80 + "\n\n")
        f.write(f"{'Label':<30} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} {'MaxDD':>10}\n")
        f.write("-"*75 + "\n")
        for r in results:
            f.write(f"{r[0]:<30} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1f}% {fmt(r[6]):>10}\n")
    for r in results:
        print(f"  {r[0]:<30} N={r[1]:>5} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


def run_r11_24(out):
    """R11-24: 全局最优配置 K-Fold 验证"""
    print("\n" + "="*70)
    print("R11-24: Global Best K-Fold Validation")
    print("="*70)

    base = get_base()
    combos = {
        'AnyPA+SR1.5': {**base, 'any_pa_confirmation': True, 'sr_filter_atr': 1.5},
        'AnyPA+PbSR': {**base, 'any_pa_confirmation': True, 'pinbar_sr_strategy': True},
        'AnyPA+AllSR': {**base, 'any_pa_confirmation': True,
                        'pinbar_sr_strategy': True, 'fractal_sr_strategy': True,
                        'inside_bar_sr_strategy': True, 'engulf_sr_strategy': True},
    }

    tasks = []
    for spread in [0.30, 0.50]:
        sp = f"sp{spread:.2f}"
        for fname, start, end in FOLDS:
            tasks.append((f"Base_{sp}_{fname}", {**base}, spread, start, end))
            for cname, ckw in combos.items():
                tasks.append((f"{cname}_{sp}_{fname}", ckw, spread, start, end))

    results = run_pool(tasks)
    rmap = {r[0]: r for r in results}

    with open(f"{out}/R11-24_global_kfold.txt", 'w') as f:
        f.write("Global Best K-Fold Validation\n" + "="*80 + "\n\n")
        for spread in [0.30, 0.50]:
            sp = f"sp{spread:.2f}"
            f.write(f"\n--- Spread ${spread:.2f} ---\n")
            for cname in combos:
                wins = 0
                f.write(f"\n  {cname}:\n")
                f.write(f"  {'Fold':<10} {'Base':>10} {'Combo':>10} {'Delta':>8}\n")
                for fname, _, _ in FOLDS:
                    b = rmap.get(f"Base_{sp}_{fname}")
                    c = rmap.get(f"{cname}_{sp}_{fname}")
                    if b and c:
                        delta = c[2] - b[2]
                        if delta > 0:
                            wins += 1
                        f.write(f"  {fname:<10} {b[2]:>10.2f} {c[2]:>10.2f} {delta:>+8.2f}\n")
                f.write(f"  Result: {wins}/6 folds win\n")
                print(f"  {cname} ${spread:.2f}: {wins}/6")


# ═══════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    phases = {
        '1': [run_r11_1, run_r11_2],
        '2': [run_r11_3, run_r11_4, run_r11_5],
        '3': [run_r11_6, run_r11_7, run_r11_8],
        '4': [run_r11_9, run_r11_10],
        '5': [run_r11_12, run_r11_13, run_r11_14],
        '6': [run_r11_15, run_r11_16, run_r11_17, run_r11_18],
        '7': [run_r11_19, run_r11_20, run_r11_21],
        '8': [run_r11_22, run_r11_23, run_r11_24],
    }

    # Run specific phase if provided as argument
    if len(sys.argv) > 1:
        phase_id = sys.argv[1]
        if phase_id in phases:
            print(f"\n{'#'*70}")
            print(f"# Round 11 — Phase {phase_id}")
            print(f"{'#'*70}")
            for fn in phases[phase_id]:
                try:
                    fn(OUTPUT_DIR)
                except Exception as e:
                    print(f"  ERROR in {fn.__name__}: {e}")
                    traceback.print_exc()
        elif phase_id.startswith('R11-'):
            # Run individual experiment like R11-1, R11-3, etc
            num = phase_id.split('-')[1]
            func_name = f"run_r11_{num}"
            fn = globals().get(func_name)
            if fn:
                fn(OUTPUT_DIR)
            else:
                print(f"Unknown experiment: {phase_id}")
        else:
            print(f"Unknown phase: {phase_id}")
            print("Usage: python run_round11.py [1|2|3|4|5|R11-N]")
        print(f"\nTotal elapsed: {time.time()-t0:.0f}s")
        return

    # Run all phases
    for phase_id, fns in sorted(phases.items()):
        print(f"\n{'#'*70}")
        print(f"# Round 11 — Phase {phase_id}")
        print(f"{'#'*70}")
        for fn in fns:
            try:
                fn(OUTPUT_DIR)
            except Exception as e:
                print(f"  ERROR in {fn.__name__}: {e}")
                traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"Round 11 COMPLETE — Total elapsed: {time.time()-t0:.0f}s")
    print(f"Results in: {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
