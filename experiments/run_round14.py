#!/usr/bin/env python3
"""
Round 14 — "Horizon Expansion"
================================
目标: 本金/手数/风控缩放研究 + 跨资产因子 + Timeout优化 + 信号时机 + 高级统计验证
预计总耗时: ~20小时 (服务器)

=== Phase S: Capital & Position Sizing (~3h) ===
R14-S1: 本金缩放模拟 (7 levels × 2 configs × 2 spread)
R14-S2: 风险比例扫描 (8 levels × 2 spread)
R14-S3: Kelly Criterion 分析
R14-S4: 复利 vs 固定模式对比

=== Phase A: 跨资产因子 (~4h) ===
R14-A1: DXY 动量过滤 (需下载DXY数据)
R14-A2: VIX Regime Trail 自适应
R14-A3: 跨资产因子 IC 扫描
R14-A4: 通过验证的因子 K-Fold

=== Phase B: Timeout 亏损优化 (~4h) ===
R14-B1: Timeout 交易行为画像
R14-B2: Progressive SL 收紧 (8/12/16 bar 阶梯)
R14-B3: Time-Adaptive Trail (引擎已支持)
R14-B4: 最优 Timeout 优化 K-Fold

=== Phase D: 信号时机精细化 (~3h) ===
R14-D1: 入场延迟测试 (1/2 M15 bar)
R14-D2: 信号强度分级分析 (KC breakout strength)
R14-D3: H4 Resample 信号

=== Phase E: 高级统计验证 (~3h) ===
R14-E1: Bootstrap Confidence Intervals (1000x)
R14-E2: Walk-Forward + Purged 对比
R14-E3: 回测 vs 实盘漂移框架

=== Phase F: 压力测试 (~3h) ===
R14-F1: 尾部风险 Monte Carlo (连续SL/spread放大/ATR spike)
R14-F2: 历史 Spread 全量回测 (11年)
R14-F3: 破产概率模拟 (不同本金)
"""
import sys, os, io, time, traceback, json
import multiprocessing as mp
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "results/round14_results"
MAX_WORKERS = 22

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
           str(t.entry_time)[:16], t.direction or '', t.lots, t.entry_price)
          for t in trades[:50000]]
    eq = s.get('_equity_curve', [])
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'], s.get('max_dd_pct', 0),
            td, eq[-1] if eq else 0, s.get('avg_win', 0), s.get('avg_loss', 0))


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


def write_table(f, results, extra_header=""):
    cols = f"{'Label':<45} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} {'MaxDD':>10} {'DD%':>7}"
    if extra_header:
        cols += f" {extra_header}"
    f.write(cols + "\n")
    f.write("-" * len(cols) + "\n")
    for r in results:
        line = f"{r[0]:<45} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1f}% {fmt(r[6]):>10} {r[7]:>6.1f}%"
        f.write(line + "\n")


# ═══════════════════════════════════════════════════════════════
# Phase S: Capital & Position Sizing
# ═══════════════════════════════════════════════════════════════

def run_r14_s1(out):
    """R14-S1: Capital Scaling Simulation"""
    print("\n" + "="*70)
    print("R14-S1: Capital Scaling — 7 levels x L5.1/L6 x $0.30/$0.50")
    print("="*70)

    import research_config as config
    POINT_VALUE = config.POINT_VALUE_PER_LOT

    capital_configs = [
        # (label, capital, risk_per_trade, min_lot, max_lot)
        ("$500",    500,   12.5,  0.01, 0.01),
        ("$1k",    1000,   25.0,  0.01, 0.03),
        ("$2k",    2000,   50.0,  0.01, 0.05),
        ("$5k",    5000,  125.0,  0.01, 0.13),
        ("$10k",  10000,  250.0,  0.01, 0.25),
        ("$50k",  50000, 1250.0,  0.01, 1.00),
        ("$100k",100000, 2500.0,  0.01, 1.00),
    ]

    tasks = []
    for cap_label, capital, rpt, min_l, max_l in capital_configs:
        for cfg_name, cfg_fn in [("L51", get_base), ("L6", get_l6)]:
            for sp in [0.30, 0.50]:
                kw = cfg_fn()
                kw['risk_per_trade'] = rpt
                kw['min_lot_size'] = min_l
                kw['max_lot_size'] = max_l
                kw['initial_capital'] = capital
                label = f"Cap{cap_label}_{cfg_name}_sp{sp}"
                tasks.append((label, kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R14_S1_capital_scaling.txt", 'w') as f:
        f.write("R14-S1: Capital Scaling Simulation\n")
        f.write("="*80 + "\n\n")
        f.write(f"Risk per trade = 2.5% of capital\n")
        f.write(f"Min/Max lots scale with capital\n\n")

        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            f.write(f"{'Capital':<10} {'Config':<6} {'N':>6} {'Sharpe':>8} {'PnL':>12} "
                    f"{'MaxDD':>10} {'DD%':>7} {'WR':>7} {'AvgW':>8} {'AvgL':>8} {'RR':>6}\n")
            f.write("-" * 95 + "\n")

            for cap_label, capital, rpt, min_l, max_l in capital_configs:
                for cfg_name in ["L51", "L6"]:
                    lbl = f"Cap{cap_label}_{cfg_name}_sp{sp}"
                    r = [x for x in results if x[0] == lbl]
                    if r:
                        x = r[0]
                        f.write(f"{cap_label:<10} {cfg_name:<6} {x[1]:>6} {x[2]:>8.2f} "
                                f"{fmt(x[3]):>12} {fmt(x[6]):>10} {x[7]:>6.1f}% "
                                f"{x[4]:>6.1f}% {x[9]:>7.2f} {x[10]:>7.2f} {x[11]:>5.2f}\n")

        f.write("\n\n--- Summary: Sharpe by Capital Level ---\n\n")
        f.write(f"{'Capital':<10} {'L51_$0.30':>10} {'L6_$0.30':>10} {'L51_$0.50':>10} {'L6_$0.50':>10}\n")
        f.write("-" * 50 + "\n")
        for cap_label, _, _, _, _ in capital_configs:
            row = f"{cap_label:<10}"
            for cfg_name in ["L51", "L6"]:
                for sp in [0.30, 0.50]:
                    lbl = f"Cap{cap_label}_{cfg_name}_sp{sp}"
                    r = [x for x in results if x[0] == lbl]
                    sh = r[0][2] if r else 0
                    row += f" {sh:>9.2f}"
            f.write(row + "\n")

    for r in results:
        if "sp0.3" in r[0] and "L51" in r[0]:
            print(f"  {r[0]:<35} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


def run_r14_s2(out):
    """R14-S2: Risk Percentage Scan"""
    print("\n" + "="*70)
    print("R14-S2: Risk % Scan — 8 levels × $0.30/$0.50")
    print("="*70)

    CAPITAL = 2000
    risk_pcts = [1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 7.5, 10.0]

    tasks = []
    for pct in risk_pcts:
        rpt = CAPITAL * pct / 100
        max_l = min(1.0, round(rpt / (20 * 100), 2) + 0.05)  # reasonable max lot
        for sp in [0.30, 0.50]:
            kw = get_base()
            kw['risk_per_trade'] = rpt
            kw['min_lot_size'] = 0.01
            kw['max_lot_size'] = max(0.01, max_l)
            kw['initial_capital'] = CAPITAL
            label = f"Risk{pct:.1f}pct_sp{sp}"
            tasks.append((label, kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R14_S2_risk_pct_scan.txt", 'w') as f:
        f.write("R14-S2: Risk Percentage Scan (Capital=$2,000)\n")
        f.write("="*80 + "\n\n")

        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            f.write(f"{'Risk%':<8} {'$/Trade':>8} {'N':>6} {'Sharpe':>8} {'PnL':>12} "
                    f"{'MaxDD':>10} {'DD%':>7} {'MaxDD$':>10}\n")
            f.write("-" * 80 + "\n")

            for pct in risk_pcts:
                rpt = CAPITAL * pct / 100
                lbl = f"Risk{pct:.1f}pct_sp{sp}"
                r = [x for x in results if x[0] == lbl]
                if r:
                    x = r[0]
                    f.write(f"{pct:<8.1f} ${rpt:>7.0f} {x[1]:>6} {x[2]:>8.2f} "
                            f"{fmt(x[3]):>12} {fmt(x[6]):>10} {x[7]:>6.1f}% {fmt(x[6]):>10}\n")

    for r in results:
        if "sp0.3" in r[0]:
            print(f"  {r[0]:<30} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


def run_r14_s3(out):
    """R14-S3: Kelly Criterion Analysis"""
    print("\n" + "="*70)
    print("R14-S3: Kelly Criterion Analysis")
    print("="*70)

    tasks = [
        ("Kelly_baseline", get_base(), 0.30, None, None),
    ]
    results = run_pool(tasks, func=_run_one_trades)

    if not results:
        print("  No results!")
        return

    r = results[0]
    trades_data = r[8]
    pnls = [t[0] for t in trades_data]

    wins = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p <= 0]

    if not wins or not losses:
        print("  Cannot calculate Kelly — no wins or no losses")
        return

    W = len(wins) / len(pnls)
    avg_win = np.mean(wins)
    avg_loss = np.mean(losses)
    B = avg_win / avg_loss

    kelly_f = W - (1 - W) / B

    CAPITAL = 2000
    kelly_fractions = [
        ("Quarter Kelly", kelly_f / 4),
        ("Third Kelly", kelly_f / 3),
        ("Half Kelly", kelly_f / 2),
        ("Full Kelly", kelly_f),
        ("Fixed 2.5%", 0.025),
        ("Fixed 1.5%", 0.015),
        ("Fixed 5.0%", 0.050),
    ]

    kelly_tasks = []
    for name, frac in kelly_fractions:
        rpt = CAPITAL * max(0.005, min(0.20, frac))
        max_l = min(1.0, round(rpt / (20 * 100), 2) + 0.05)
        for sp in [0.30, 0.50]:
            kw = get_base()
            kw['risk_per_trade'] = rpt
            kw['min_lot_size'] = 0.01
            kw['max_lot_size'] = max(0.01, max_l)
            kw['initial_capital'] = CAPITAL
            safe_name = name.replace(" ", "_").replace(".", "p")
            label = f"Kelly_{safe_name}_sp{sp}"
            kelly_tasks.append((label, kw, sp, None, None))

    kelly_results = run_pool(kelly_tasks)

    with open(f"{out}/R14_S3_kelly_criterion.txt", 'w') as f:
        f.write("R14-S3: Kelly Criterion Analysis\n")
        f.write("="*80 + "\n\n")

        f.write(f"Historical Win Rate: {W:.4f} ({W*100:.1f}%)\n")
        f.write(f"Average Win: ${avg_win:.2f}\n")
        f.write(f"Average Loss: ${avg_loss:.2f}\n")
        f.write(f"Win/Loss Ratio (B): {B:.3f}\n")
        f.write(f"Kelly f* = W - (1-W)/B = {kelly_f:.4f} ({kelly_f*100:.2f}%)\n\n")

        f.write(f"Interpretation:\n")
        f.write(f"  Full Kelly: risk {kelly_f*100:.2f}% per trade = ${CAPITAL*kelly_f:.1f}/trade\n")
        f.write(f"  Half Kelly: risk {kelly_f*50:.2f}% per trade = ${CAPITAL*kelly_f/2:.1f}/trade\n")
        f.write(f"  Quarter Kelly: risk {kelly_f*25:.2f}% per trade = ${CAPITAL*kelly_f/4:.1f}/trade\n\n")

        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            f.write(f"{'Method':<25} {'Risk%':>7} {'$/Trade':>8} {'N':>6} {'Sharpe':>8} "
                    f"{'PnL':>12} {'MaxDD':>10} {'DD%':>7}\n")
            f.write("-" * 90 + "\n")

            for name, frac in kelly_fractions:
                safe_name = name.replace(" ", "_").replace(".", "p")
                lbl = f"Kelly_{safe_name}_sp{sp}"
                r = [x for x in kelly_results if x[0] == lbl]
                if r:
                    x = r[0]
                    rpt = CAPITAL * max(0.005, min(0.20, frac))
                    f.write(f"{name:<25} {frac*100:>6.2f}% ${rpt:>7.0f} {x[1]:>6} "
                            f"{x[2]:>8.2f} {fmt(x[3]):>12} {fmt(x[6]):>10} {x[7]:>6.1f}%\n")

    print(f"  Kelly f* = {kelly_f:.4f} ({kelly_f*100:.2f}%)")
    for r in kelly_results:
        if "sp0.3" in r[0]:
            print(f"  {r[0]:<40} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


def run_r14_s4(out):
    """R14-S4: Compounding vs Fixed"""
    print("\n" + "="*70)
    print("R14-S4: Compounding vs Fixed Position Sizing")
    print("="*70)

    capital_levels = [
        ("$2k",   2000,  50.0, 0.01, 0.05),
        ("$5k",   5000, 125.0, 0.01, 0.13),
        ("$10k", 10000, 250.0, 0.01, 0.25),
    ]

    tasks = []
    for cap_label, capital, rpt, min_l, max_l in capital_levels:
        for mode in ["Fixed", "Compound"]:
            for sp in [0.30, 0.50]:
                kw = get_base()
                kw['risk_per_trade'] = rpt
                kw['min_lot_size'] = min_l
                kw['max_lot_size'] = max_l
                kw['initial_capital'] = capital
                if mode == "Compound":
                    kw['compounding'] = True
                label = f"{mode}_{cap_label}_sp{sp}"
                tasks.append((label, kw, sp, None, None))

    results = run_pool(tasks, func=_run_one_trades)

    with open(f"{out}/R14_S4_compounding.txt", 'w') as f:
        f.write("R14-S4: Compounding vs Fixed Position Sizing\n")
        f.write("="*80 + "\n\n")

        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            f.write(f"{'Config':<25} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'MaxDD':>10} "
                    f"{'DD%':>7} {'Final$':>10}\n")
            f.write("-" * 85 + "\n")

            for cap_label, capital, _, _, _ in capital_levels:
                for mode in ["Fixed", "Compound"]:
                    lbl = f"{mode}_{cap_label}_sp{sp}"
                    r = [x for x in results if x[0] == lbl]
                    if r:
                        x = r[0]
                        final_eq = x[9] if x[9] else 0
                        f.write(f"{lbl:<25} {x[1]:>6} {x[2]:>8.2f} {fmt(x[3]):>12} "
                                f"{fmt(x[6]):>10} {x[7]:>6.1f}% {fmt(final_eq):>10}\n")

    for r in results:
        if "sp0.3" in r[0]:
            print(f"  {r[0]:<30} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


# ═══════════════════════════════════════════════════════════════
# Phase B: Timeout Loss Optimization
# ═══════════════════════════════════════════════════════════════

def run_r14_b1(out):
    """R14-B1: Timeout Trade Behavioral Profile"""
    print("\n" + "="*70)
    print("R14-B1: Timeout Trade Behavioral Profile")
    print("="*70)

    tasks = [("B1_profile", get_base(), 0.30, None, None)]
    results = run_pool(tasks, func=_run_one_trades)

    if not results:
        return

    trades_data = results[0][8]

    exit_groups = defaultdict(list)
    for pnl, reason, bars, strat, etime, direction, lots, eprice in trades_data:
        exit_groups[reason].append({
            'pnl': pnl, 'bars': bars, 'strategy': strat,
            'direction': direction, 'lots': lots, 'entry_price': eprice,
        })

    with open(f"{out}/R14_B1_timeout_profile.txt", 'w') as f:
        f.write("R14-B1: Timeout Trade Behavioral Profile\n")
        f.write("="*80 + "\n\n")

        f.write("--- Exit Reason Summary ---\n\n")
        f.write(f"{'Reason':<15} {'N':>6} {'TotalPnL':>12} {'AvgPnL':>10} {'WR':>7} "
                f"{'AvgBars':>8} {'MedBars':>8}\n")
        f.write("-" * 75 + "\n")

        for reason in sorted(exit_groups.keys()):
            trades = exit_groups[reason]
            n = len(trades)
            total = sum(t['pnl'] for t in trades)
            avg = total / n if n > 0 else 0
            wins = sum(1 for t in trades if t['pnl'] > 0)
            wr = wins / n * 100 if n > 0 else 0
            bars_list = [t['bars'] for t in trades]
            avg_bars = np.mean(bars_list) if bars_list else 0
            med_bars = np.median(bars_list) if bars_list else 0
            f.write(f"{reason:<15} {n:>6} {fmt(total):>12} ${avg:>9.2f} {wr:>6.1f}% "
                    f"{avg_bars:>7.1f} {med_bars:>7.0f}\n")

        timeout_trades = exit_groups.get('Timeout', [])
        trailing_trades = exit_groups.get('Trailing', [])
        sl_trades = exit_groups.get('SL', [])

        if timeout_trades:
            f.write(f"\n\n--- Timeout Trades Deep Dive (N={len(timeout_trades)}) ---\n\n")

            to_pnls = [t['pnl'] for t in timeout_trades]
            f.write(f"Total PnL: {fmt(sum(to_pnls))}\n")
            f.write(f"Avg PnL: ${np.mean(to_pnls):.2f}\n")
            f.write(f"Median PnL: ${np.median(to_pnls):.2f}\n")
            f.write(f"Std PnL: ${np.std(to_pnls):.2f}\n")
            f.write(f"Min PnL: ${min(to_pnls):.2f}\n")
            f.write(f"Max PnL: ${max(to_pnls):.2f}\n")
            winners = sum(1 for p in to_pnls if p > 0)
            f.write(f"Winners: {winners}/{len(to_pnls)} ({winners/len(to_pnls)*100:.1f}%)\n")

            bars_dist = [t['bars'] for t in timeout_trades]
            f.write(f"\nBars Held Distribution:\n")
            for pctile in [10, 25, 50, 75, 90]:
                f.write(f"  P{pctile}: {np.percentile(bars_dist, pctile):.0f} bars\n")

            f.write(f"\nLot Size Distribution:\n")
            lots_dist = [t['lots'] for t in timeout_trades]
            f.write(f"  Mean: {np.mean(lots_dist):.3f}\n")
            f.write(f"  Median: {np.median(lots_dist):.3f}\n")

            f.write(f"\nStrategy Breakdown:\n")
            strat_counts = Counter(t['strategy'] for t in timeout_trades)
            for s, c in strat_counts.most_common():
                s_pnl = sum(t['pnl'] for t in timeout_trades if t['strategy'] == s)
                f.write(f"  {s}: {c} trades, PnL={fmt(s_pnl)}\n")

            f.write(f"\nDirection Breakdown:\n")
            for d in ['BUY', 'SELL']:
                d_trades = [t for t in timeout_trades if t['direction'] == d]
                if d_trades:
                    d_pnl = sum(t['pnl'] for t in d_trades)
                    f.write(f"  {d}: {len(d_trades)} trades, PnL={fmt(d_pnl)}, "
                            f"AvgPnL=${np.mean([t['pnl'] for t in d_trades]):.2f}\n")

        if trailing_trades:
            f.write(f"\n\n--- Trailing vs Timeout Comparison ---\n\n")
            tr_pnls = [t['pnl'] for t in trailing_trades]
            to_pnls = [t['pnl'] for t in timeout_trades] if timeout_trades else [0]
            f.write(f"{'Metric':<20} {'Trailing':>12} {'Timeout':>12} {'SL':>12}\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Count':<20} {len(trailing_trades):>12} {len(timeout_trades):>12} {len(sl_trades):>12}\n")
            f.write(f"{'Total PnL':<20} {fmt(sum(tr_pnls)):>12} {fmt(sum(to_pnls)):>12} {fmt(sum(t['pnl'] for t in sl_trades)):>12}\n")
            f.write(f"{'Avg PnL':<20} ${np.mean(tr_pnls):>11.2f} ${np.mean(to_pnls):>11.2f} ${np.mean([t['pnl'] for t in sl_trades]) if sl_trades else 0:>11.2f}\n")
            f.write(f"{'Avg Bars':<20} {np.mean([t['bars'] for t in trailing_trades]):>11.1f} {np.mean([t['bars'] for t in timeout_trades]):>11.1f} {np.mean([t['bars'] for t in sl_trades]) if sl_trades else 0:>11.1f}\n")

    print(f"  Timeout: {len(timeout_trades)} trades, PnL={fmt(sum(t['pnl'] for t in timeout_trades))}")


def run_r14_b2(out):
    """R14-B2: Progressive SL Tightening"""
    print("\n" + "="*70)
    print("R14-B2: Progressive SL Tightening")
    print("="*70)

    configs = [
        ("Baseline",          0, 0, 0),
        ("ProgSL_8bar_2.5",   8, 2.5, 4),   # start bar 8, target 2.5x, 4 steps
        ("ProgSL_8bar_2.0",   8, 2.0, 4),
        ("ProgSL_8bar_1.5",   8, 1.5, 4),
        ("ProgSL_12bar_2.5", 12, 2.5, 4),
        ("ProgSL_12bar_2.0", 12, 2.0, 4),
        ("ProgSL_16bar_2.0", 16, 2.0, 2),
        ("ProgSL_8bar_2.0_slow",  8, 2.0, 8), # 8 steps = very gradual
    ]

    tasks = []
    for name, start_bar, target, steps in configs:
        for sp in [0.30, 0.50]:
            kw = get_base()
            if start_bar > 0:
                kw['progressive_sl_start_bar'] = start_bar
                kw['progressive_sl_target_mult'] = target
                kw['progressive_sl_steps'] = steps
            label = f"{name}_sp{sp}"
            tasks.append((label, kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R14_B2_progressive_sl.txt", 'w') as f:
        f.write("R14-B2: Progressive SL Tightening\n")
        f.write("="*80 + "\n\n")
        f.write("Current SL = 3.5x ATR. Progressive SL reduces SL distance after N bars.\n")
        f.write("Logic: if trade hasn't profited by bar N, it's likely a bad signal.\n\n")

        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            f.write(f"{'Config':<25} {'Start':>6} {'Target':>7} {'Steps':>6} {'N':>6} "
                    f"{'Sharpe':>8} {'dSh':>6} {'PnL':>12} {'MaxDD':>10}\n")
            f.write("-" * 95 + "\n")

            base_sharpe = 0
            for name, start_bar, target, steps in configs:
                lbl = f"{name}_sp{sp}"
                r = [x for x in results if x[0] == lbl]
                if r:
                    x = r[0]
                    if name == "Baseline":
                        base_sharpe = x[2]
                    delta = x[2] - base_sharpe
                    sb = f"{start_bar}" if start_bar > 0 else "—"
                    tg = f"{target:.1f}x" if target > 0 else "—"
                    st = f"{steps}" if steps > 0 else "—"
                    f.write(f"{name:<25} {sb:>6} {tg:>7} {st:>6} {x[1]:>6} "
                            f"{x[2]:>8.2f} {delta:>+5.2f} {fmt(x[3]):>12} {fmt(x[6]):>10}\n")

    for r in results:
        if "sp0.3" in r[0]:
            print(f"  {r[0]:<35} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


def run_r14_b3(out):
    """R14-B3: Time-Adaptive Trailing (engine already supports)"""
    print("\n" + "="*70)
    print("R14-B3: Time-Adaptive Trail — Tighten trail over time")
    print("="*70)

    configs = [
        ("Baseline",                 False, 4, 0.95, 0.005),
        ("TATrail_s4_d95",           True,  4, 0.95, 0.005),
        ("TATrail_s4_d90",           True,  4, 0.90, 0.005),
        ("TATrail_s4_d85",           True,  4, 0.85, 0.005),
        ("TATrail_s6_d95",           True,  6, 0.95, 0.005),
        ("TATrail_s6_d90",           True,  6, 0.90, 0.005),
        ("TATrail_s8_d95",           True,  8, 0.95, 0.005),
        ("TATrail_s4_d95_floor01",   True,  4, 0.95, 0.010),
    ]

    tasks = []
    for name, enabled, start, decay, floor in configs:
        for sp in [0.30, 0.50]:
            kw = get_base()
            if enabled:
                kw['time_adaptive_trail'] = True
                kw['time_adaptive_trail_start'] = start
                kw['time_adaptive_trail_decay'] = decay
                kw['time_adaptive_trail_floor'] = floor
            label = f"{name}_sp{sp}"
            tasks.append((label, kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R14_B3_time_adaptive_trail.txt", 'w') as f:
        f.write("R14-B3: Time-Adaptive Trailing Stop\n")
        f.write("="*80 + "\n\n")
        f.write("Tighten trail_dist by decay^(bars_held - start) per bar after start.\n")
        f.write("Goal: convert more Timeout trades to Trailing exits.\n\n")

        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            f.write(f"{'Config':<30} {'Start':>6} {'Decay':>6} {'Floor':>6} {'N':>6} "
                    f"{'Sharpe':>8} {'dSh':>6} {'PnL':>12} {'MaxDD':>10}\n")
            f.write("-" * 100 + "\n")

            base_sharpe = 0
            for name, enabled, start, decay, floor in configs:
                lbl = f"{name}_sp{sp}"
                r = [x for x in results if x[0] == lbl]
                if r:
                    x = r[0]
                    if name == "Baseline":
                        base_sharpe = x[2]
                    delta = x[2] - base_sharpe
                    s_ = f"{start}" if enabled else "—"
                    d_ = f"{decay}" if enabled else "—"
                    fl_ = f"{floor}" if enabled else "—"
                    f.write(f"{name:<30} {s_:>6} {d_:>6} {fl_:>6} {x[1]:>6} "
                            f"{x[2]:>8.2f} {delta:>+5.2f} {fmt(x[3]):>12} {fmt(x[6]):>10}\n")

    for r in results:
        if "sp0.3" in r[0]:
            print(f"  {r[0]:<35} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


def run_r14_b4(out):
    """R14-B4: K-Fold for best Timeout optimization"""
    print("\n" + "="*70)
    print("R14-B4: K-Fold Validation for Best Timeout Optimizer")
    print("="*70)

    best_configs = [
        ("Baseline", {}),
        ("ProgSL_8bar_2.0", {'progressive_sl_start_bar': 8, 'progressive_sl_target_mult': 2.0, 'progressive_sl_steps': 4}),
        ("TATrail_s4_d95", {'time_adaptive_trail': True, 'time_adaptive_trail_start': 4, 'time_adaptive_trail_decay': 0.95, 'time_adaptive_trail_floor': 0.005}),
        ("Combined", {'progressive_sl_start_bar': 8, 'progressive_sl_target_mult': 2.0, 'progressive_sl_steps': 4,
                      'time_adaptive_trail': True, 'time_adaptive_trail_start': 4, 'time_adaptive_trail_decay': 0.95, 'time_adaptive_trail_floor': 0.005}),
    ]

    tasks = []
    for sp in [0.30, 0.50]:
        for config_name, extra_kw in best_configs:
            for fold_name, start, end in FOLDS:
                kw = get_base()
                kw.update(extra_kw)
                label = f"{config_name}_{fold_name}_sp{sp}"
                tasks.append((label, kw, sp, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R14_B4_timeout_kfold.txt", 'w') as f:
        f.write("R14-B4: K-Fold Validation for Timeout Optimizers\n")
        f.write("="*80 + "\n\n")

        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            f.write(f"{'Config':<20}")
            for fn, _, _ in FOLDS:
                f.write(f" {fn:>10}")
            f.write(f" {'Avg':>10} {'Pass':>6}\n")
            f.write("-" * 100 + "\n")

            base_sharpes = {}
            for config_name, _ in best_configs:
                f.write(f"{config_name:<20}")
                fold_sharpes = []
                for fold_name, _, _ in FOLDS:
                    lbl = f"{config_name}_{fold_name}_sp{sp}"
                    r = [x for x in results if x[0] == lbl]
                    if r:
                        sh = r[0][2]
                        fold_sharpes.append(sh)
                        f.write(f" {sh:>10.2f}")
                    else:
                        f.write(f" {'N/A':>10}")

                if fold_sharpes:
                    avg = np.mean(fold_sharpes)
                    f.write(f" {avg:>10.2f}")
                else:
                    f.write(f" {'N/A':>10}")

                if config_name == "Baseline":
                    base_sharpes[sp] = {fn: 0 for fn, _, _ in FOLDS}
                    for i, (fn, _, _) in enumerate(FOLDS):
                        if i < len(fold_sharpes):
                            base_sharpes[sp][fn] = fold_sharpes[i]
                    f.write(f" {'—':>6}")
                else:
                    wins = 0
                    if fold_sharpes and sp in base_sharpes:
                        for i, (fn, _, _) in enumerate(FOLDS):
                            if i < len(fold_sharpes) and fold_sharpes[i] >= base_sharpes[sp].get(fn, 0):
                                wins += 1
                    f.write(f" {wins}/6")
                f.write("\n")

    print("  K-Fold results written")


# ═══════════════════════════════════════════════════════════════
# Phase D: Signal Timing
# ═══════════════════════════════════════════════════════════════

def run_r14_d1(out):
    """R14-D1: Entry Delay Test"""
    print("\n" + "="*70)
    print("R14-D1: Entry Delay Test (1/2 M15 bars)")
    print("="*70)

    configs = [
        ("Baseline_gap0",    0),
        ("Gap_1h",           1.0),
        ("Gap_2h",           2.0),
        ("Gap_3h",           3.0),
        ("Gap_4h",           4.0),
    ]

    tasks = []
    for name, gap_h in configs:
        for sp in [0.30, 0.50]:
            kw = get_base()
            if gap_h > 0:
                kw['min_entry_gap_hours'] = gap_h
            label = f"{name}_sp{sp}"
            tasks.append((label, kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R14_D1_entry_delay.txt", 'w') as f:
        f.write("R14-D1: Entry Delay / Min Gap Test\n")
        f.write("="*80 + "\n\n")

        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            write_table(f, [r for r in results if f"sp{sp}" in r[0]])

    for r in results:
        if "sp0.3" in r[0]:
            print(f"  {r[0]:<30} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


def run_r14_d2(out):
    """R14-D2: Signal Strength Grading by KC breakout strength"""
    print("\n" + "="*70)
    print("R14-D2: KC Breakout Strength Analysis")
    print("="*70)

    tasks = [("D2_full", get_base(), 0.30, None, None)]
    results = run_pool(tasks, func=_run_one_trades)

    if not results:
        return

    trades_data = results[0][8]
    keltner_trades = [t for t in trades_data if t[3] == 'keltner']

    with open(f"{out}/R14_D2_signal_strength.txt", 'w') as f:
        f.write("R14-D2: KC Breakout Strength Analysis\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total Keltner trades: {len(keltner_trades)}\n\n")

        pnls = [t[0] for t in keltner_trades]
        bars = [t[2] for t in keltner_trades]

        f.write("--- Bars Held vs PnL Correlation ---\n\n")
        bins = [(1, 2, "1-2 bars"), (3, 5, "3-5 bars"), (6, 10, "6-10 bars"),
                (11, 15, "11-15 bars"), (16, 20, "16-20 bars"), (21, 999, "21+ bars")]

        f.write(f"{'Bars Range':<15} {'N':>6} {'TotalPnL':>12} {'AvgPnL':>10} {'WR':>7} {'MedPnL':>10}\n")
        f.write("-" * 65 + "\n")

        for lo, hi, label in bins:
            bucket = [(p, b) for p, b in zip(pnls, bars) if lo <= b <= hi]
            if bucket:
                bp = [x[0] for x in bucket]
                n = len(bp)
                total = sum(bp)
                avg = total / n
                wr = sum(1 for p in bp if p > 0) / n * 100
                med = np.median(bp)
                f.write(f"{label:<15} {n:>6} {fmt(total):>12} ${avg:>9.2f} {wr:>6.1f}% ${med:>9.2f}\n")

        f.write("\n\n--- Exit Reason by Bars Held ---\n\n")
        f.write(f"{'Bars Range':<15} {'Trail':>6} {'SL':>6} {'Timeout':>8} {'TP':>6} {'Other':>6}\n")
        f.write("-" * 55 + "\n")

        for lo, hi, label in bins:
            bucket = [t for t in keltner_trades if lo <= t[2] <= hi]
            if bucket:
                reasons = Counter(t[1] for t in bucket)
                f.write(f"{label:<15} {reasons.get('Trailing', 0):>6} {reasons.get('SL', 0):>6} "
                        f"{reasons.get('Timeout', 0):>8} {reasons.get('TP', 0):>6} "
                        f"{sum(v for k,v in reasons.items() if k not in ('Trailing','SL','Timeout','TP')):>6}\n")

    print(f"  Keltner: {len(keltner_trades)} trades analyzed")


# ═══════════════════════════════════════════════════════════════
# Phase E: Advanced Statistical Validation
# ═══════════════════════════════════════════════════════════════

def run_r14_e1(out):
    """R14-E1: Bootstrap Confidence Intervals"""
    print("\n" + "="*70)
    print("R14-E1: Bootstrap Confidence Intervals (1000x)")
    print("="*70)

    configs = [
        ("L51", get_base),
        ("L6",  get_l6),
    ]

    tasks = []
    for name, fn in configs:
        for sp in [0.30, 0.50]:
            kw = fn()
            label = f"Boot_{name}_sp{sp}"
            tasks.append((label, kw, sp, None, None))

    results = run_pool(tasks, func=_run_one_trades)

    N_BOOTSTRAP = 1000

    with open(f"{out}/R14_E1_bootstrap_ci.txt", 'w') as f:
        f.write("R14-E1: Bootstrap Confidence Intervals\n")
        f.write("="*80 + "\n\n")

        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")

            for name, _ in configs:
                lbl = f"Boot_{name}_sp{sp}"
                r = [x for x in results if x[0] == lbl]
                if not r:
                    continue

                trades_data = r[0][8]
                pnls = np.array([t[0] for t in trades_data])
                n_trades = len(pnls)

                from backtest.stats import aggregate_daily_pnl
                from backtest.engine import TradeRecord
                temp_trades = [TradeRecord(
                    strategy=t[3], direction=t[5],
                    entry_price=t[7], exit_price=t[7]+t[0]/max(t[6],0.01)/100,
                    entry_time=t[4], exit_time=t[4],
                    lots=t[6], pnl=t[0], exit_reason=t[1], bars_held=t[2]
                ) for t in trades_data]
                daily = aggregate_daily_pnl(temp_trades)
                daily_arr = np.array(daily)

                boot_sharpes = []
                boot_pnls = []
                rng = np.random.RandomState(42)
                for _ in range(N_BOOTSTRAP):
                    idx = rng.randint(0, len(daily_arr), len(daily_arr))
                    sample = daily_arr[idx]
                    std = np.std(sample, ddof=1)
                    if std > 0:
                        sh = np.mean(sample) / std * np.sqrt(252)
                    else:
                        sh = 0
                    boot_sharpes.append(sh)
                    boot_pnls.append(np.sum(sample))

                boot_sharpes = np.array(boot_sharpes)
                boot_pnls = np.array(boot_pnls)

                obs_sharpe = r[0][2]
                obs_pnl = r[0][3]

                f.write(f"Config: {name}\n")
                f.write(f"  Observed: Sharpe={obs_sharpe:.2f}, PnL={fmt(obs_pnl)}, N={n_trades}\n")
                f.write(f"  Bootstrap ({N_BOOTSTRAP} samples):\n")
                f.write(f"    Sharpe: mean={np.mean(boot_sharpes):.2f}, "
                        f"std={np.std(boot_sharpes):.2f}\n")
                f.write(f"    95% CI: [{np.percentile(boot_sharpes, 2.5):.2f}, "
                        f"{np.percentile(boot_sharpes, 97.5):.2f}]\n")
                f.write(f"    99% CI: [{np.percentile(boot_sharpes, 0.5):.2f}, "
                        f"{np.percentile(boot_sharpes, 99.5):.2f}]\n")
                f.write(f"    PnL 95% CI: [{fmt(np.percentile(boot_pnls, 2.5))}, "
                        f"{fmt(np.percentile(boot_pnls, 97.5))}]\n")
                f.write(f"    P(Sharpe>0): {(boot_sharpes > 0).mean()*100:.1f}%\n")
                f.write(f"    P(Sharpe>2): {(boot_sharpes > 2).mean()*100:.1f}%\n")
                f.write(f"    P(PnL>0): {(boot_pnls > 0).mean()*100:.1f}%\n\n")

        if len(results) >= 4:
            f.write("\n--- L6 vs L5.1 Bootstrap Comparison ---\n\n")
            for sp in [0.30, 0.50]:
                l51_r = [x for x in results if x[0] == f"Boot_L51_sp{sp}"]
                l6_r = [x for x in results if x[0] == f"Boot_L6_sp{sp}"]
                if l51_r and l6_r:
                    f.write(f"Spread ${sp}: L6 Sharpe={l6_r[0][2]:.2f} vs L5.1={l51_r[0][2]:.2f} "
                            f"(delta={l6_r[0][2]-l51_r[0][2]:+.2f})\n")

    print("  Bootstrap CI computed")


def run_r14_e2(out):
    """R14-E2: Walk-Forward Year-by-Year"""
    print("\n" + "="*70)
    print("R14-E2: Walk-Forward Year-by-Year Validation")
    print("="*70)

    configs = [("L51", get_base), ("L6", get_l6)]

    tasks = []
    for name, fn in configs:
        for sp in [0.30, 0.50]:
            for yr_name, start, end in YEARS:
                kw = fn()
                label = f"WF_{name}_{yr_name}_sp{sp}"
                tasks.append((label, kw, sp, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R14_E2_walk_forward.txt", 'w') as f:
        f.write("R14-E2: Walk-Forward Year-by-Year\n")
        f.write("="*80 + "\n\n")

        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread = ${sp} ---\n\n")
            f.write(f"{'Year':<6} {'L51_Sharpe':>10} {'L51_PnL':>12} {'L6_Sharpe':>10} {'L6_PnL':>12} {'Delta':>8}\n")
            f.write("-" * 65 + "\n")

            l51_total, l6_total = 0, 0
            l51_years_pos, l6_years_pos = 0, 0
            for yr_name, _, _ in YEARS:
                l51 = [x for x in results if x[0] == f"WF_L51_{yr_name}_sp{sp}"]
                l6 = [x for x in results if x[0] == f"WF_L6_{yr_name}_sp{sp}"]
                l51_sh = l51[0][2] if l51 else 0
                l6_sh = l6[0][2] if l6 else 0
                l51_pnl = l51[0][3] if l51 else 0
                l6_pnl = l6[0][3] if l6 else 0
                delta = l6_sh - l51_sh
                l51_total += l51_pnl
                l6_total += l6_pnl
                if l51_pnl > 0: l51_years_pos += 1
                if l6_pnl > 0: l6_years_pos += 1
                f.write(f"{yr_name:<6} {l51_sh:>10.2f} {fmt(l51_pnl):>12} "
                        f"{l6_sh:>10.2f} {fmt(l6_pnl):>12} {delta:>+7.2f}\n")

            f.write(f"\n{'Total':<6} {'':>10} {fmt(l51_total):>12} {'':>10} {fmt(l6_total):>12}\n")
            f.write(f"{'Pos Years':<6} {l51_years_pos:>10}/11 {'':>12} {l6_years_pos:>10}/11\n")

    for r in results:
        if "sp0.3" in r[0] and "L51" in r[0]:
            print(f"  {r[0]:<30} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


# ═══════════════════════════════════════════════════════════════
# Phase F: Stress Testing
# ═══════════════════════════════════════════════════════════════

def run_r14_f1(out):
    """R14-F1: Tail Risk Monte Carlo — parameter perturbation"""
    print("\n" + "="*70)
    print("R14-F1: Monte Carlo Parameter Perturbation (100x)")
    print("="*70)

    N_MC = 100
    rng = np.random.RandomState(42)

    base = get_base()
    rc = base.get('regime_config', {})

    tasks = []
    for i in range(N_MC):
        kw = {**base}
        perturb = 1.0 + rng.uniform(-0.15, 0.15)
        kw['sl_atr_mult'] = round(base['sl_atr_mult'] * perturb, 2)
        kw['choppy_threshold'] = round(base.get('choppy_threshold', 0.50) * (1 + rng.uniform(-0.10, 0.10)), 2)
        new_rc = {}
        for regime, params in rc.items():
            new_rc[regime] = {
                'trail_act': round(params['trail_act'] * (1 + rng.uniform(-0.15, 0.15)), 3),
                'trail_dist': round(params['trail_dist'] * (1 + rng.uniform(-0.15, 0.15)), 3),
            }
        kw['regime_config'] = new_rc
        label = f"MC_{i:03d}"
        tasks.append((label, kw, 0.30, None, None))

    results = run_pool(tasks)

    sharpes = [r[2] for r in results]
    pnls = [r[3] for r in results]
    maxdds = [r[6] for r in results]

    with open(f"{out}/R14_F1_monte_carlo.txt", 'w') as f:
        f.write("R14-F1: Monte Carlo Parameter Perturbation (100x, ±15%)\n")
        f.write("="*80 + "\n\n")

        f.write(f"N simulations: {N_MC}\n")
        f.write(f"Perturbation range: ±15% on SL, Trail, Choppy\n\n")

        f.write(f"--- Sharpe Distribution ---\n")
        f.write(f"  Mean:   {np.mean(sharpes):.2f}\n")
        f.write(f"  Std:    {np.std(sharpes):.2f}\n")
        f.write(f"  Min:    {np.min(sharpes):.2f}\n")
        f.write(f"  Max:    {np.max(sharpes):.2f}\n")
        f.write(f"  P5:     {np.percentile(sharpes, 5):.2f}\n")
        f.write(f"  P25:    {np.percentile(sharpes, 25):.2f}\n")
        f.write(f"  P50:    {np.percentile(sharpes, 50):.2f}\n")
        f.write(f"  P75:    {np.percentile(sharpes, 75):.2f}\n")
        f.write(f"  P95:    {np.percentile(sharpes, 95):.2f}\n")
        f.write(f"  100% profitable: {all(s > 0 for s in sharpes)}\n")
        f.write(f"  % Sharpe > 0: {sum(1 for s in sharpes if s > 0)/len(sharpes)*100:.1f}%\n")
        f.write(f"  % Sharpe > 4: {sum(1 for s in sharpes if s > 4)/len(sharpes)*100:.1f}%\n\n")

        f.write(f"--- PnL Distribution ---\n")
        f.write(f"  Mean:   {fmt(np.mean(pnls))}\n")
        f.write(f"  Min:    {fmt(np.min(pnls))}\n")
        f.write(f"  Max:    {fmt(np.max(pnls))}\n")
        f.write(f"  100% positive: {all(p > 0 for p in pnls)}\n\n")

        f.write(f"--- MaxDD Distribution ---\n")
        f.write(f"  Mean:   {fmt(np.mean(maxdds))}\n")
        f.write(f"  Worst:  {fmt(np.max(maxdds))}\n")
        f.write(f"  P95:    {fmt(np.percentile(maxdds, 95))}\n\n")

        f.write(f"--- Worst 5 Runs ---\n")
        worst = sorted(zip(sharpes, pnls, maxdds, [r[0] for r in results]))[:5]
        for sh, pnl, dd, name in worst:
            f.write(f"  {name}: Sharpe={sh:.2f}, PnL={fmt(pnl)}, MaxDD={fmt(dd)}\n")

    print(f"  MC: mean Sharpe={np.mean(sharpes):.2f}, min={np.min(sharpes):.2f}, "
          f"100% profitable={all(p > 0 for p in pnls)}")


def run_r14_f2(out):
    """R14-F2: Historical Spread Full Backtest"""
    print("\n" + "="*70)
    print("R14-F2: Historical Spread vs Fixed Spread Comparison")
    print("="*70)

    from backtest.runner import load_spread_series

    tasks = []
    for name, fn in [("L51", get_base), ("L6", get_l6)]:
        kw_fixed30 = fn()
        tasks.append((f"Fixed030_{name}", kw_fixed30, 0.30, None, None))

        kw_fixed50 = fn()
        tasks.append((f"Fixed050_{name}", kw_fixed50, 0.50, None, None))

        kw_hist = fn()
        kw_hist['spread_model'] = 'historical'
        spread_series = load_spread_series()
        if spread_series is not None:
            kw_hist['spread_series'] = spread_series
            tasks.append((f"Historical_{name}", kw_hist, 0, None, None))

        kw_sess = fn()
        kw_sess['spread_model'] = 'session_aware'
        kw_sess['spread_base'] = 0.30
        tasks.append((f"SessionAware_{name}", kw_sess, 0, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R14_F2_spread_comparison.txt", 'w') as f:
        f.write("R14-F2: Spread Model Comparison\n")
        f.write("="*80 + "\n\n")

        f.write(f"{'Config':<30} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'MaxDD':>10} {'WR':>7}\n")
        f.write("-" * 80 + "\n")

        for r in results:
            f.write(f"{r[0]:<30} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {fmt(r[6]):>10} {r[4]:>6.1f}%\n")

    for r in results:
        print(f"  {r[0]:<30} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


def run_r14_f3(out):
    """R14-F3: Ruin Probability Simulation"""
    print("\n" + "="*70)
    print("R14-F3: Ruin Probability by Capital Level")
    print("="*70)

    tasks = [("F3_base", get_base(), 0.30, None, None)]
    results = run_pool(tasks, func=_run_one_trades)

    if not results:
        return

    trades_data = results[0][8]
    pnl_sequence = [t[0] for t in trades_data]

    capital_levels = [500, 1000, 1500, 2000, 3000, 5000, 10000]
    ruin_thresholds = [0.50, 0.75]  # lose 50% / 75% of capital
    N_SIM = 1000

    with open(f"{out}/R14_F3_ruin_probability.txt", 'w') as f:
        f.write("R14-F3: Ruin Probability Simulation\n")
        f.write("="*80 + "\n\n")
        f.write(f"Method: Bootstrap {N_SIM}x resampling of {len(pnl_sequence)} historical trades\n")
        f.write(f"Ruin = losing X% of starting capital at any point\n\n")

        rng = np.random.RandomState(42)

        f.write(f"{'Capital':<10}")
        for thr in ruin_thresholds:
            f.write(f" {'Ruin'+str(int(thr*100))+'%':>10}")
        f.write(f" {'MaxDD_med':>10} {'MaxDD_95':>10} {'Final_med':>12}\n")
        f.write("-" * 70 + "\n")

        for capital in capital_levels:
            scale = capital / 2000.0
            scaled_pnls = [p * scale for p in pnl_sequence]

            ruin_counts = {thr: 0 for thr in ruin_thresholds}
            max_dds = []
            finals = []

            for _ in range(N_SIM):
                idx = rng.randint(0, len(scaled_pnls), len(scaled_pnls))
                eq = capital
                peak = capital
                max_dd = 0
                ruined = {thr: False for thr in ruin_thresholds}
                for i in idx:
                    eq += scaled_pnls[i]
                    peak = max(peak, eq)
                    dd = peak - eq
                    max_dd = max(max_dd, dd)
                    for thr in ruin_thresholds:
                        if dd >= capital * thr:
                            ruined[thr] = True
                for thr in ruin_thresholds:
                    if ruined[thr]:
                        ruin_counts[thr] += 1
                max_dds.append(max_dd)
                finals.append(eq)

            row = f"${capital:<9,}"
            for thr in ruin_thresholds:
                prob = ruin_counts[thr] / N_SIM * 100
                row += f" {prob:>9.1f}%"
            row += f" {fmt(np.median(max_dds)):>10}"
            row += f" {fmt(np.percentile(max_dds, 95)):>10}"
            row += f" {fmt(np.median(finals)):>12}"
            f.write(row + "\n")

        f.write(f"\n\nNote: PnL scaled linearly with capital (risk 2.5%)\n")
        f.write(f"Small capitals have same Sharpe but higher granularity risk (0.01 lot min)\n")

    print(f"  Ruin simulation complete for {len(capital_levels)} capital levels")


# ═══════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_time = time.time()
    print(f"\n{'='*70}")
    print(f"Round 14 — Horizon Expansion")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"{'='*70}\n")

    phases = [
        # Phase S: Capital & Position Sizing (~3h)
        ("Phase S1: Capital Scaling",       run_r14_s1),
        ("Phase S2: Risk % Scan",           run_r14_s2),
        ("Phase S3: Kelly Criterion",       run_r14_s3),
        ("Phase S4: Compounding vs Fixed",  run_r14_s4),
        # Phase B: Timeout Optimization (~4h)
        ("Phase B1: Timeout Profile",       run_r14_b1),
        ("Phase B2: Progressive SL",        run_r14_b2),
        ("Phase B3: Time-Adaptive Trail",   run_r14_b3),
        ("Phase B4: Timeout K-Fold",        run_r14_b4),
        # Phase D: Signal Timing (~3h)
        ("Phase D1: Entry Delay",           run_r14_d1),
        ("Phase D2: Signal Strength",       run_r14_d2),
        # Phase E: Statistics (~3h)
        ("Phase E1: Bootstrap CI",          run_r14_e1),
        ("Phase E2: Walk-Forward",          run_r14_e2),
        # Phase F: Stress Testing (~3h)
        ("Phase F1: Monte Carlo",           run_r14_f1),
        ("Phase F2: Spread Comparison",     run_r14_f2),
        ("Phase F3: Ruin Probability",      run_r14_f3),
    ]

    completed = []
    for phase_name, phase_fn in phases:
        try:
            t0 = time.time()
            print(f"\n>>> Starting {phase_name}...")
            phase_fn(OUTPUT_DIR)
            elapsed = time.time() - t0
            completed.append((phase_name, elapsed, "OK"))
            print(f"<<< {phase_name} done in {elapsed:.0f}s")
        except Exception as e:
            traceback.print_exc()
            completed.append((phase_name, 0, f"FAIL: {e}"))
            print(f"<<< {phase_name} FAILED: {e}")

    total = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Round 14 Complete — Total: {total:.0f}s ({total/3600:.1f}h)")
    print(f"{'='*70}")

    with open(f"{OUTPUT_DIR}/R14_summary.txt", 'w') as f:
        f.write(f"Round 14 — Horizon Expansion\n{'='*60}\n")
        f.write(f"Total: {total:.0f}s ({total/3600:.1f}h)\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for name, elapsed, status in completed:
            f.write(f"{name:<35} {elapsed:>8.0f}s  {status}\n")

    for name, elapsed, status in completed:
        print(f"  {name:<35} {elapsed:>8.0f}s  {status}")


if __name__ == "__main__":
    main()
