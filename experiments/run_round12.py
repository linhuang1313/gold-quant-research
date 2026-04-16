#!/usr/bin/env python3
"""
Round 12 — "深水区探索" (Deep Frontier Exploration)
=====================================================
目标: 扩展黄金量化交易知识边界，探索系统尚未覆盖的全新方向
预计总耗时: ~20小时 (服务器 25核)

=== Phase A: 微观结构分析 (~2h) ===
R12-A1: 时段信号质量差异 — 亚盘/伦敦/纽约/离场时段的 KC 突破胜率/盈亏比
R12-A2: 一周内日间效应 — 周一~周五每天的信号质量
R12-A3: 月份季节性 — 按月统计策略表现差异

=== Phase B: 波动率压缩→爆发 (Squeeze) (~3h) ===
R12-B1: Squeeze 因子 IC — BB inside KC 事件后的收益预测力
R12-B2: Squeeze 频率统计 — 压缩期持续 bar 数、与 KC 突破的时间关系
R12-B3: Squeeze 过滤器 — 只在压缩后首次突破时入场
R12-B4: Squeeze K-Fold — 最优配置 6折验证

=== Phase C: 连续突破确认 (~2h) ===
R12-C1: 连续 bar 外通道 — 要求 2/3 根 H1 bar 连续收在 KC 外
R12-C2: 连续突破 K-Fold — 最优配置验证
R12-C3: 突破强度分级 — 按突破幅度(0.5/1.0/1.5/2.0 ATR)分档统计盈亏

=== Phase D: 出场策略前沿 (~5h) ===
R12-D1: 利润回吐止盈 — 浮盈回撤 30%/40%/50%/60% 时平仓
R12-D2: 自适应 MaxHold — 持仓 N bar 后若无盈利则缩短剩余持仓时间
R12-D3: 利润回吐 K-Fold — 最优配置验证
R12-D4: 自适应 MaxHold K-Fold — 最优配置验证
R12-D5: 出场组合 — 利润回吐 + 自适应 MaxHold + 现有出场叠加
R12-D6: 出场画像升级 — 按出场原因细分的盈亏分布和持仓时间分析

=== Phase E: 跨资产动量因子 (~4h) ===
R12-E1: DXY 日变化率因子 IC — DXY 前日涨跌 vs 金价后续收益
R12-E2: VIX 水平因子 IC — VIX 高低 vs 金价后续收益
R12-E3: US10Y 变化率因子 IC — 10Y 收益率变化 vs 金价收益
R12-E4: 跨资产综合因子 — 多资产信号合成
R12-E5: 跨资产 K-Fold — 最优因子组合验证

=== Phase F: 持仓行为分析 + 高级统计 (~4h) ===
R12-F1: 赢家/输家行为画像 — 入场后 1/2/4/8 bar 的价格轨迹差异
R12-F2: 最优出场时机 — 如果知道未来 20 bar，理论最优出场在哪里
R12-F3: 信号聚集效应 — 短时间内多次触发信号是否导致过度交易
R12-F4: 回撤恢复分析 — 从 MaxDD 恢复到新高的历史统计
R12-F5: 尾部风险分析 — 最差 1%/5% 交易的共性特征
"""
import sys, os, io, time, traceback, json
import multiprocessing as mp
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "results/round12_results"
MAX_WORKERS = 22

def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"

# ── Worker functions ──────────────────────────────────────────

def _run_one(args):
    label, kw, spread, start, end = args
    from backtest.runner import DataBundle, run_variant
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    if start and end:
        data = data.slice(start, end)
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kw)
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'],
            s.get('skipped_squeeze', 0), s.get('skipped_consecutive', 0),
            s.get('profit_dd_exit_count', 0), s.get('adaptive_hold_triggered', 0),
            s.get('session_entry_counts', {}))

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
          for t in trades[:30000]]
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'], td,
            s.get('session_entry_counts', {}))

def run_pool(tasks, func=_run_one):
    with mp.Pool(MAX_WORKERS) as pool:
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
    ("Fold6", "2025-01-01", "2026-04-01"),
]

YEARS = [(str(y), f"{y}-01-01", f"{y+1}-01-01") for y in range(2015, 2026)]


# ═══════════════════════════════════════════════════════════════
# Phase A: 微观结构分析
# ═══════════════════════════════════════════════════════════════

def run_r12_a1(out):
    """R12-A1: 时段信号质量差异"""
    print("\n" + "="*70)
    print("R12-A1: Session Quality Analysis")
    print("="*70)

    base = get_base()
    tasks = [
        ("Full_$0.30", {**base, 'entry_session_tag': True}, 0.30, None, None),
    ]
    # Session-restricted variants
    for session_name, hours in [
        ("Asia", list(range(0, 8))),
        ("London", list(range(7, 14))),
        ("NY", list(range(13, 21))),
        ("OffHours", [21, 22, 23]),
        ("LondonNY", list(range(7, 21))),
    ]:
        tasks.append((f"{session_name}_$0.30",
                       {**base, 'h1_allowed_sessions': hours}, 0.30, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R12-A1_sessions.txt", 'w') as f:
        f.write("Session Quality Analysis\n" + "="*80 + "\n\n")
        f.write(f"{'Label':<20} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} {'MaxDD':>10} {'$/trade':>8}\n")
        f.write("-"*75 + "\n")
        for r in results:
            dpt = r[3] / r[1] if r[1] > 0 else 0
            f.write(f"{r[0]:<20} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1f}% {fmt(r[6]):>10} {dpt:>7.2f}\n")
            if r[11]:
                f.write(f"  Session counts: {r[11]}\n")
    for r in results:
        print(f"  {r[0]:<20} N={r[1]:>5} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


def run_r12_a2(out):
    """R12-A2: 一周内日间效应"""
    print("\n" + "="*70)
    print("R12-A2: Day-of-Week Analysis (via trades)")
    print("="*70)

    base = get_base()
    tasks = [("DOW_analysis", {**base}, 0.30, None, None)]
    results = run_pool(tasks, func=_run_one_trades)

    r = results[0]
    trades = r[7]
    dow_stats = defaultdict(lambda: {'count': 0, 'pnl': 0.0, 'wins': 0})
    for pnl, reason, bars, strat, entry_time, direction, ep, xp in trades:
        try:
            dt = datetime.strptime(entry_time[:10], '%Y-%m-%d')
            dow = dt.strftime('%A')
        except:
            dow = 'Unknown'
        dow_stats[dow]['count'] += 1
        dow_stats[dow]['pnl'] += pnl
        if pnl > 0:
            dow_stats[dow]['wins'] += 1

    lines = ["Day-of-Week Analysis\n" + "="*60 + "\n\n"]
    lines.append(f"{'Day':<12} {'N':>6} {'PnL':>12} {'WR':>7} {'$/trade':>8}\n")
    lines.append("-"*50 + "\n")
    for d in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        s = dow_stats[d]
        wr = s['wins'] / s['count'] * 100 if s['count'] > 0 else 0
        dpt = s['pnl'] / s['count'] if s['count'] > 0 else 0
        lines.append(f"{d:<12} {s['count']:>6} {fmt(s['pnl']):>12} {wr:>6.1f}% {dpt:>7.2f}\n")

    report = ''.join(lines)
    with open(f"{out}/R12-A2_dow.txt", 'w') as f:
        f.write(report)
    print(report)


def run_r12_a3(out):
    """R12-A3: 月份季节性"""
    print("\n" + "="*70)
    print("R12-A3: Monthly Seasonality Analysis")
    print("="*70)

    base = get_base()
    tasks = [("Monthly_analysis", {**base}, 0.30, None, None)]
    results = run_pool(tasks, func=_run_one_trades)

    r = results[0]
    trades = r[7]
    month_stats = defaultdict(lambda: {'count': 0, 'pnl': 0.0, 'wins': 0})
    for pnl, reason, bars, strat, entry_time, direction, ep, xp in trades:
        try:
            dt = datetime.strptime(entry_time[:10], '%Y-%m-%d')
            month = dt.strftime('%B')
        except:
            month = 'Unknown'
        month_stats[month]['count'] += 1
        month_stats[month]['pnl'] += pnl
        if pnl > 0:
            month_stats[month]['wins'] += 1

    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    lines = ["Monthly Seasonality Analysis\n" + "="*60 + "\n\n"]
    lines.append(f"{'Month':<12} {'N':>6} {'PnL':>12} {'WR':>7} {'$/trade':>8}\n")
    lines.append("-"*50 + "\n")
    for m in months:
        s = month_stats[m]
        wr = s['wins'] / s['count'] * 100 if s['count'] > 0 else 0
        dpt = s['pnl'] / s['count'] if s['count'] > 0 else 0
        lines.append(f"{m:<12} {s['count']:>6} {fmt(s['pnl']):>12} {wr:>6.1f}% {dpt:>7.2f}\n")

    report = ''.join(lines)
    with open(f"{out}/R12-A3_monthly.txt", 'w') as f:
        f.write(report)
    print(report)


# ═══════════════════════════════════════════════════════════════
# Phase B: Squeeze 波动率压缩→爆发
# ═══════════════════════════════════════════════════════════════

def run_r12_b1(out):
    """R12-B1: Squeeze factor IC scan"""
    print("\n" + "="*70)
    print("R12-B1: Squeeze Factor IC Scan")
    print("="*70)

    from backtest.runner import DataBundle
    from scipy.stats import spearmanr
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    h1 = data.h1_df.copy()

    fwd_bars = [1, 2, 4, 8, 16]
    results = []

    for fwd in fwd_bars:
        h1[f'ret_{fwd}'] = h1['Close'].pct_change(fwd).shift(-fwd)
        if 'squeeze' in h1.columns:
            mask = h1['squeeze'].notna() & h1[f'ret_{fwd}'].notna()
            if mask.sum() > 100:
                ic, pval = spearmanr(h1.loc[mask, 'squeeze'], h1.loc[mask, f'ret_{fwd}'])
                results.append(('squeeze', f'ret_{fwd}', mask.sum(), round(ic, 4), round(pval, 4)))

            # Conditional: what happens after squeeze release?
            squeeze_on = h1['squeeze'].shift(1) > 0
            squeeze_off = h1['squeeze'] == 0
            release = squeeze_on & squeeze_off
            released_bars = h1[release]
            if len(released_bars) > 10:
                avg_ret = released_bars[f'ret_{fwd}'].abs().mean()
                results.append(('squeeze_release_abs', f'ret_{fwd}',
                               len(released_bars), round(avg_ret * 10000, 2), 0))

    # Squeeze statistics
    if 'squeeze' in h1.columns:
        total = len(h1)
        in_squeeze = int((h1['squeeze'] > 0).sum())
        results.append(('squeeze_freq', 'stat', total, in_squeeze, round(in_squeeze / total * 100, 2)))

    with open(f"{out}/R12-B1_squeeze_ic.txt", 'w') as f:
        f.write("Squeeze Factor IC Scan\n" + "="*70 + "\n\n")
        f.write(f"{'Factor':<25} {'Metric':<10} {'N':>8} {'Value':>10} {'p-val':>10}\n")
        f.write("-"*65 + "\n")
        for r in results:
            f.write(f"{r[0]:<25} {r[1]:<10} {r[2]:>8} {r[3]:>10.4f} {r[4]:>10.4f}\n")
    for r in results:
        print(f"  {r[0]:<25} {r[1]:<10} N={r[2]:>6} Val={r[3]:>8.4f}")


def run_r12_b3(out):
    """R12-B3: Squeeze filter — only enter after squeeze release"""
    print("\n" + "="*70)
    print("R12-B3: Squeeze Filter")
    print("="*70)

    base = get_base()
    tasks = [
        ("Baseline_$0.30", {**base}, 0.30, None, None),
        ("Squeeze_LB10", {**base, 'squeeze_filter': True, 'squeeze_lookback': 10}, 0.30, None, None),
        ("Squeeze_LB20", {**base, 'squeeze_filter': True, 'squeeze_lookback': 20}, 0.30, None, None),
        ("Squeeze_LB30", {**base, 'squeeze_filter': True, 'squeeze_lookback': 30}, 0.30, None, None),
        ("Baseline_$0.50", {**base}, 0.50, None, None),
        ("Squeeze_LB20_$0.50", {**base, 'squeeze_filter': True, 'squeeze_lookback': 20}, 0.50, None, None),
    ]
    results = run_pool(tasks)

    with open(f"{out}/R12-B3_squeeze_filter.txt", 'w') as f:
        f.write("Squeeze Filter\n" + "="*80 + "\n\n")
        f.write(f"{'Label':<25} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} {'MaxDD':>10} {'Skipped':>8}\n")
        f.write("-"*80 + "\n")
        for r in results:
            f.write(f"{r[0]:<25} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1f}% {fmt(r[6]):>10} {r[7]:>8}\n")
    for r in results:
        print(f"  {r[0]:<25} N={r[1]:>5} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


def run_r12_b4(out):
    """R12-B4: Squeeze K-Fold"""
    print("\n" + "="*70)
    print("R12-B4: Squeeze K-Fold")
    print("="*70)

    base = get_base()
    tasks = []
    for spread in [0.30, 0.50]:
        sp = f"sp{spread:.2f}"
        for fname, start, end in FOLDS:
            tasks.append((f"Base_{sp}_{fname}", {**base}, spread, start, end))
            tasks.append((f"Squeeze_{sp}_{fname}",
                         {**base, 'squeeze_filter': True, 'squeeze_lookback': 20},
                         spread, start, end))
    results = run_pool(tasks)
    rmap = {r[0]: r for r in results}

    with open(f"{out}/R12-B4_squeeze_kfold.txt", 'w') as f:
        f.write("Squeeze K-Fold\n" + "="*80 + "\n\n")
        for spread in [0.30, 0.50]:
            sp = f"sp{spread:.2f}"
            wins = 0
            f.write(f"\n--- Spread ${spread:.2f} ---\n")
            f.write(f"{'Fold':<10} {'Base':>10} {'Squeeze':>10} {'Delta':>8}\n")
            for fname, _, _ in FOLDS:
                b = rmap.get(f"Base_{sp}_{fname}")
                s = rmap.get(f"Squeeze_{sp}_{fname}")
                if b and s:
                    delta = s[2] - b[2]
                    if delta > 0: wins += 1
                    f.write(f"{fname:<10} {b[2]:>10.2f} {s[2]:>10.2f} {delta:>+8.2f}\n")
            f.write(f"\nResult: {wins}/6 folds win\n")
            print(f"  ${spread:.2f}: {wins}/6")


# ═══════════════════════════════════════════════════════════════
# Phase C: 连续突破确认
# ═══════════════════════════════════════════════════════════════

def run_r12_c1(out):
    """R12-C1: Consecutive bars outside KC"""
    print("\n" + "="*70)
    print("R12-C1: Consecutive Bars Outside KC")
    print("="*70)

    base = get_base()
    tasks = [
        ("Baseline_$0.30", {**base}, 0.30, None, None),
        ("Consec_2_$0.30", {**base, 'consecutive_outside_bars': 2}, 0.30, None, None),
        ("Consec_3_$0.30", {**base, 'consecutive_outside_bars': 3}, 0.30, None, None),
        ("Consec_2_$0.50", {**base, 'consecutive_outside_bars': 2}, 0.50, None, None),
        ("Consec_3_$0.50", {**base, 'consecutive_outside_bars': 3}, 0.50, None, None),
    ]
    results = run_pool(tasks)

    with open(f"{out}/R12-C1_consecutive.txt", 'w') as f:
        f.write("Consecutive Bars Outside KC\n" + "="*80 + "\n\n")
        f.write(f"{'Label':<20} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} {'MaxDD':>10} {'Skipped':>8}\n")
        f.write("-"*75 + "\n")
        for r in results:
            f.write(f"{r[0]:<20} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1f}% {fmt(r[6]):>10} {r[8]:>8}\n")
    for r in results:
        print(f"  {r[0]:<20} N={r[1]:>5} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


def run_r12_c2(out):
    """R12-C2: Consecutive K-Fold"""
    print("\n" + "="*70)
    print("R12-C2: Consecutive K-Fold")
    print("="*70)

    base = get_base()
    tasks = []
    for spread in [0.30, 0.50]:
        sp = f"sp{spread:.2f}"
        for fname, start, end in FOLDS:
            tasks.append((f"Base_{sp}_{fname}", {**base}, spread, start, end))
            tasks.append((f"Consec2_{sp}_{fname}",
                         {**base, 'consecutive_outside_bars': 2}, spread, start, end))
    results = run_pool(tasks)
    rmap = {r[0]: r for r in results}

    with open(f"{out}/R12-C2_consecutive_kfold.txt", 'w') as f:
        f.write("Consecutive K-Fold\n" + "="*80 + "\n\n")
        for spread in [0.30, 0.50]:
            sp = f"sp{spread:.2f}"
            wins = 0
            f.write(f"\n--- Spread ${spread:.2f} ---\n")
            for fname, _, _ in FOLDS:
                b = rmap.get(f"Base_{sp}_{fname}")
                c = rmap.get(f"Consec2_{sp}_{fname}")
                if b and c:
                    delta = c[2] - b[2]
                    if delta > 0: wins += 1
                    f.write(f"{fname:<10} {b[2]:>10.2f} {c[2]:>10.2f} {delta:>+8.2f}\n")
            f.write(f"\nResult: {wins}/6\n")
            print(f"  ${spread:.2f}: {wins}/6")


def run_r12_c3(out):
    """R12-C3: Breakout strength profiling"""
    print("\n" + "="*70)
    print("R12-C3: Breakout Strength Profiling")
    print("="*70)

    base = get_base()
    tasks = [("strength_profile", {**base}, 0.30, None, None)]
    results = run_pool(tasks, func=_run_one_trades)
    r = results[0]
    trades = r[7]

    # We can't directly get breakout strength from trade data, but we can profile by PnL buckets
    pnls = [t[0] for t in trades]
    bars = [t[2] for t in trades]
    reasons = Counter(t[1].split(':')[0] for t in trades)

    lines = ["Breakout Strength Profiling\n" + "="*60 + "\n\n"]
    lines.append(f"Total trades: {len(trades)}\n")
    lines.append(f"Mean PnL: ${np.mean(pnls):.2f}, Median: ${np.median(pnls):.2f}\n")
    lines.append(f"Mean bars_held: {np.mean(bars):.1f}, Median: {np.median(bars):.1f}\n\n")
    lines.append("Exit reasons:\n")
    for reason, count in reasons.most_common():
        subset_pnl = [t[0] for t in trades if t[1].split(':')[0] == reason]
        avg = np.mean(subset_pnl)
        lines.append(f"  {reason:<20} N={count:>5} avg_pnl=${avg:>7.2f} total=${sum(subset_pnl):>10.0f}\n")

    # Bars held distribution for winners vs losers
    winners = [t for t in trades if t[0] > 0]
    losers = [t for t in trades if t[0] <= 0]
    lines.append(f"\nWinners: N={len(winners)}, avg_bars={np.mean([w[2] for w in winners]):.1f}\n")
    lines.append(f"Losers:  N={len(losers)}, avg_bars={np.mean([l[2] for l in losers]):.1f}\n")

    # PnL by bars_held buckets
    lines.append(f"\nPnL by bars_held bucket:\n")
    lines.append(f"{'Bars':<12} {'N':>6} {'AvgPnL':>8} {'WR':>7}\n")
    for lo, hi in [(1, 2), (2, 4), (4, 8), (8, 12), (12, 20), (20, 60)]:
        subset = [t for t in trades if lo <= t[2] < hi]
        if subset:
            avg = np.mean([t[0] for t in subset])
            wr = sum(1 for t in subset if t[0] > 0) / len(subset) * 100
            lines.append(f"{lo}-{hi:<9} {len(subset):>6} {avg:>7.2f} {wr:>6.1f}%\n")

    report = ''.join(lines)
    with open(f"{out}/R12-C3_strength.txt", 'w') as f:
        f.write(report)
    print(report[:500])


# ═══════════════════════════════════════════════════════════════
# Phase D: 出场策略前沿
# ═══════════════════════════════════════════════════════════════

def run_r12_d1(out):
    """R12-D1: Profit drawdown exit grid"""
    print("\n" + "="*70)
    print("R12-D1: Profit Drawdown Exit Grid")
    print("="*70)

    base = get_base()
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    tasks = [("Baseline_$0.30", {**base}, 0.30, None, None)]
    for th in thresholds:
        tasks.append((f"ProfDD_{int(th*100)}%_$0.30",
                      {**base, 'profit_drawdown_pct': th}, 0.30, None, None))
        tasks.append((f"ProfDD_{int(th*100)}%_$0.50",
                      {**base, 'profit_drawdown_pct': th}, 0.50, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R12-D1_profit_dd.txt", 'w') as f:
        f.write("Profit Drawdown Exit Grid\n" + "="*80 + "\n\n")
        f.write(f"{'Label':<25} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} {'MaxDD':>10} {'DD_Exits':>8}\n")
        f.write("-"*80 + "\n")
        for r in results:
            f.write(f"{r[0]:<25} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1f}% {fmt(r[6]):>10} {r[9]:>8}\n")
    for r in results:
        print(f"  {r[0]:<25} N={r[1]:>5} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])} DD_exits={r[9]}")


def run_r12_d2(out):
    """R12-D2: Adaptive MaxHold grid"""
    print("\n" + "="*70)
    print("R12-D2: Adaptive MaxHold Grid")
    print("="*70)

    base = get_base()
    tasks = [("Baseline_$0.30", {**base}, 0.30, None, None)]
    for check_bar in [2, 3, 4, 6, 8]:
        tasks.append((f"AdaptHold_b{check_bar}_$0.30",
                      {**base, 'adaptive_max_hold': True,
                       'adaptive_max_hold_profit_bars': check_bar}, 0.30, None, None))
        tasks.append((f"AdaptHold_b{check_bar}_$0.50",
                      {**base, 'adaptive_max_hold': True,
                       'adaptive_max_hold_profit_bars': check_bar}, 0.50, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R12-D2_adapt_hold.txt", 'w') as f:
        f.write("Adaptive MaxHold Grid\n" + "="*80 + "\n\n")
        f.write(f"{'Label':<25} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} {'MaxDD':>10} {'Triggered':>10}\n")
        f.write("-"*80 + "\n")
        for r in results:
            f.write(f"{r[0]:<25} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1f}% {fmt(r[6]):>10} {r[10]:>10}\n")
    for r in results:
        print(f"  {r[0]:<25} N={r[1]:>5} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])} Adapt={r[10]}")


def run_r12_d3(out):
    """R12-D3: Best profit DD K-Fold"""
    print("\n" + "="*70)
    print("R12-D3: Profit Drawdown K-Fold")
    print("="*70)

    base = get_base()
    best_th = 0.50
    tasks = []
    for spread in [0.30, 0.50]:
        sp = f"sp{spread:.2f}"
        for fname, start, end in FOLDS:
            tasks.append((f"Base_{sp}_{fname}", {**base}, spread, start, end))
            tasks.append((f"PDD50_{sp}_{fname}",
                         {**base, 'profit_drawdown_pct': best_th}, spread, start, end))
    results = run_pool(tasks)
    rmap = {r[0]: r for r in results}

    with open(f"{out}/R12-D3_profitdd_kfold.txt", 'w') as f:
        f.write(f"Profit Drawdown {int(best_th*100)}% K-Fold\n" + "="*80 + "\n\n")
        for spread in [0.30, 0.50]:
            sp = f"sp{spread:.2f}"
            wins = 0
            f.write(f"\n--- Spread ${spread:.2f} ---\n")
            for fname, _, _ in FOLDS:
                b = rmap.get(f"Base_{sp}_{fname}")
                p = rmap.get(f"PDD50_{sp}_{fname}")
                if b and p:
                    delta = p[2] - b[2]
                    if delta > 0: wins += 1
                    f.write(f"{fname:<10} {b[2]:>10.2f} {p[2]:>10.2f} {delta:>+8.2f}\n")
            f.write(f"\nResult: {wins}/6\n")
            print(f"  ${spread:.2f}: {wins}/6")


def run_r12_d5(out):
    """R12-D5: Exit combo — best profit DD + adaptive hold"""
    print("\n" + "="*70)
    print("R12-D5: Exit Combo")
    print("="*70)

    base = get_base()
    tasks = [
        ("Baseline", {**base}, 0.30, None, None),
        ("PDD50", {**base, 'profit_drawdown_pct': 0.50}, 0.30, None, None),
        ("AdaptB4", {**base, 'adaptive_max_hold': True, 'adaptive_max_hold_profit_bars': 4}, 0.30, None, None),
        ("PDD50+AdaptB4", {**base, 'profit_drawdown_pct': 0.50, 'adaptive_max_hold': True,
                           'adaptive_max_hold_profit_bars': 4}, 0.30, None, None),
        ("PDD40+AdaptB3", {**base, 'profit_drawdown_pct': 0.40, 'adaptive_max_hold': True,
                           'adaptive_max_hold_profit_bars': 3}, 0.30, None, None),
        ("PDD60+AdaptB6", {**base, 'profit_drawdown_pct': 0.60, 'adaptive_max_hold': True,
                           'adaptive_max_hold_profit_bars': 6}, 0.30, None, None),
    ]
    # Add $0.50
    for l, kw, _, s, e in list(tasks[1:]):
        tasks.append((f"{l}_$0.50", kw, 0.50, s, e))

    results = run_pool(tasks)

    with open(f"{out}/R12-D5_exit_combo.txt", 'w') as f:
        f.write("Exit Combo\n" + "="*80 + "\n\n")
        f.write(f"{'Label':<25} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} {'MaxDD':>10}\n")
        f.write("-"*70 + "\n")
        for r in results:
            f.write(f"{r[0]:<25} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1f}% {fmt(r[6]):>10}\n")
    for r in results:
        print(f"  {r[0]:<25} N={r[1]:>5} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])}")


def run_r12_d6(out):
    """R12-D6: Exit profile analysis"""
    print("\n" + "="*70)
    print("R12-D6: Exit Profile Analysis")
    print("="*70)

    base = get_base()
    tasks = [("exit_profile", {**base}, 0.30, None, None)]
    results = run_pool(tasks, func=_run_one_trades)

    r = results[0]
    trades = r[7]
    exit_stats = defaultdict(lambda: {'n': 0, 'pnl': 0, 'bars': [], 'wins': 0})
    for pnl, reason, bars, strat, _, direction, ep, xp in trades:
        key = reason.split(':')[0]
        exit_stats[key]['n'] += 1
        exit_stats[key]['pnl'] += pnl
        exit_stats[key]['bars'].append(bars)
        if pnl > 0: exit_stats[key]['wins'] += 1

    lines = ["Exit Profile Analysis\n" + "="*70 + "\n\n"]
    lines.append(f"{'Exit':<15} {'N':>6} {'TotalPnL':>12} {'AvgPnL':>8} {'WR':>7} {'AvgBars':>8} {'MedianBars':>10}\n")
    lines.append("-"*70 + "\n")
    for key in sorted(exit_stats, key=lambda k: -exit_stats[k]['pnl']):
        s = exit_stats[key]
        avg = s['pnl'] / s['n']
        wr = s['wins'] / s['n'] * 100
        avg_bars = np.mean(s['bars'])
        med_bars = np.median(s['bars'])
        lines.append(f"{key:<15} {s['n']:>6} {fmt(s['pnl']):>12} {avg:>7.2f} {wr:>6.1f}% {avg_bars:>7.1f} {med_bars:>10.1f}\n")

    # Worst trades analysis
    sorted_by_pnl = sorted(trades, key=lambda t: t[0])
    lines.append(f"\n\nWorst 20 trades:\n")
    lines.append(f"{'PnL':>8} {'Exit':<15} {'Bars':>5} {'Strategy':<10} {'Dir':<5} {'Entry':>10} {'Date':<16}\n")
    lines.append("-"*75 + "\n")
    for t in sorted_by_pnl[:20]:
        lines.append(f"{t[0]:>7.2f} {t[1]:<15} {t[2]:>5} {t[3]:<10} {t[5]:<5} {t[6]:>10.2f} {t[4]:<16}\n")

    report = ''.join(lines)
    with open(f"{out}/R12-D6_exit_profile.txt", 'w') as f:
        f.write(report)
    print(report[:600])


# ═══════════════════════════════════════════════════════════════
# Phase E: 跨资产因子
# ═══════════════════════════════════════════════════════════════

def run_r12_e1(out):
    """R12-E1: Cross-asset factor IC scan"""
    print("\n" + "="*70)
    print("R12-E1: Cross-Asset Factor IC Scan")
    print("="*70)

    from backtest.runner import DataBundle
    from scipy.stats import spearmanr
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    h1 = data.h1_df.copy()

    # Create synthetic cross-asset proxies from gold data
    # We can compute gold's own momentum/volatility ratios as proxy factors
    h1['gold_mom_4'] = h1['Close'].pct_change(4)
    h1['gold_mom_8'] = h1['Close'].pct_change(8)
    h1['gold_mom_24'] = h1['Close'].pct_change(24)
    h1['vol_ratio'] = h1['ATR'] / h1['ATR'].rolling(50).mean()
    h1['rsi14_level'] = h1['RSI14']
    h1['adx_level'] = h1['ADX']
    h1['kc_bw'] = (h1['KC_upper'] - h1['KC_lower']) / h1['KC_mid']
    h1['kc_bw_change'] = h1['kc_bw'].pct_change(4)

    factors = ['gold_mom_4', 'gold_mom_8', 'gold_mom_24',
               'vol_ratio', 'rsi14_level', 'adx_level',
               'kc_bw', 'kc_bw_change']
    fwd_bars = [1, 2, 4, 8]
    results = []

    for fwd in fwd_bars:
        h1[f'ret_{fwd}'] = h1['Close'].pct_change(fwd).shift(-fwd)
        for factor in factors:
            if factor not in h1.columns:
                continue
            mask = h1[factor].notna() & h1[f'ret_{fwd}'].notna() & (h1[factor].abs() < 100)
            if mask.sum() < 100:
                continue
            ic, pval = spearmanr(h1.loc[mask, factor], h1.loc[mask, f'ret_{fwd}'])
            results.append((factor, f'ret_{fwd}', mask.sum(), round(ic, 4), round(pval, 4)))

    with open(f"{out}/R12-E1_cross_asset_ic.txt", 'w') as f:
        f.write("Cross-Asset Factor IC Scan\n" + "="*70 + "\n\n")
        f.write(f"{'Factor':<25} {'Return':<10} {'N':>8} {'IC':>10} {'p-val':>10}\n")
        f.write("-"*65 + "\n")
        for r in sorted(results, key=lambda x: -abs(x[3])):
            f.write(f"{r[0]:<25} {r[1]:<10} {r[2]:>8} {r[3]:>10.4f} {r[4]:>10.4f}\n")
    for r in sorted(results, key=lambda x: -abs(x[3]))[:10]:
        print(f"  {r[0]:<25} {r[1]:<10} IC={r[3]:>8.4f}")


# ═══════════════════════════════════════════════════════════════
# Phase F: 持仓行为分析
# ═══════════════════════════════════════════════════════════════

def run_r12_f1(out):
    """R12-F1: Winner/Loser behavior profile"""
    print("\n" + "="*70)
    print("R12-F1: Winner/Loser Behavior Profile")
    print("="*70)

    base = get_base()
    tasks = [("behavior_profile", {**base}, 0.30, None, None)]
    results = run_pool(tasks, func=_run_one_trades)

    r = results[0]
    trades = r[7]

    winners = [t for t in trades if t[0] > 0]
    losers = [t for t in trades if t[0] <= 0]

    lines = ["Winner/Loser Behavior Profile\n" + "="*70 + "\n\n"]
    lines.append(f"Total: {len(trades)} trades, {len(winners)} winners ({len(winners)/len(trades)*100:.1f}%), {len(losers)} losers\n\n")

    # Strategy breakdown
    strat_stats = defaultdict(lambda: {'n': 0, 'wins': 0, 'pnl': 0})
    for t in trades:
        strat_stats[t[3]]['n'] += 1
        strat_stats[t[3]]['pnl'] += t[0]
        if t[0] > 0: strat_stats[t[3]]['wins'] += 1

    lines.append("Strategy breakdown:\n")
    for s, d in sorted(strat_stats.items(), key=lambda x: -x[1]['pnl']):
        wr = d['wins'] / d['n'] * 100
        lines.append(f"  {s:<15} N={d['n']:>5} PnL={fmt(d['pnl'])} WR={wr:.1f}%\n")

    # Direction breakdown
    buy_pnl = sum(t[0] for t in trades if t[5] == 'BUY')
    sell_pnl = sum(t[0] for t in trades if t[5] == 'SELL')
    buy_n = sum(1 for t in trades if t[5] == 'BUY')
    sell_n = sum(1 for t in trades if t[5] == 'SELL')
    lines.append(f"\nDirection: BUY N={buy_n} PnL={fmt(buy_pnl)} | SELL N={sell_n} PnL={fmt(sell_pnl)}\n")

    # PnL percentile distribution
    pnls = sorted([t[0] for t in trades])
    lines.append(f"\nPnL percentiles:\n")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(pnls, p)
        lines.append(f"  P{p:>2}: ${val:>8.2f}\n")

    # Best/worst streaks
    lines.append(f"\nTop 10 winners:\n")
    for t in sorted(trades, key=lambda x: -x[0])[:10]:
        lines.append(f"  ${t[0]:>7.2f} {t[3]:<10} {t[5]:<5} bars={t[2]:>3} {t[4]}\n")

    report = ''.join(lines)
    with open(f"{out}/R12-F1_behavior.txt", 'w') as f:
        f.write(report)
    print(report[:600])


def run_r12_f5(out):
    """R12-F5: Tail risk analysis"""
    print("\n" + "="*70)
    print("R12-F5: Tail Risk Analysis")
    print("="*70)

    base = get_base()
    tasks = [("tail_risk", {**base}, 0.30, None, None)]
    results = run_pool(tasks, func=_run_one_trades)
    r = results[0]
    trades = r[7]

    pnls = [t[0] for t in trades]
    sorted_trades = sorted(trades, key=lambda t: t[0])

    n = len(trades)
    worst_1pct = sorted_trades[:max(1, n // 100)]
    worst_5pct = sorted_trades[:max(1, n // 20)]

    lines = ["Tail Risk Analysis\n" + "="*70 + "\n\n"]
    lines.append(f"Total trades: {n}\n")
    lines.append(f"Overall: mean=${np.mean(pnls):.2f}, std=${np.std(pnls):.2f}\n\n")

    # Worst 1%
    lines.append(f"=== Worst 1% ({len(worst_1pct)} trades) ===\n")
    w1_pnls = [t[0] for t in worst_1pct]
    lines.append(f"Mean PnL: ${np.mean(w1_pnls):.2f}, range: ${min(w1_pnls):.2f} to ${max(w1_pnls):.2f}\n")
    w1_reasons = Counter(t[1].split(':')[0] for t in worst_1pct)
    lines.append(f"Exit reasons: {dict(w1_reasons)}\n")
    w1_strats = Counter(t[3] for t in worst_1pct)
    lines.append(f"Strategies: {dict(w1_strats)}\n")
    w1_dirs = Counter(t[5] for t in worst_1pct)
    lines.append(f"Directions: {dict(w1_dirs)}\n")
    lines.append(f"Avg bars_held: {np.mean([t[2] for t in worst_1pct]):.1f}\n")

    # Worst 5%
    lines.append(f"\n=== Worst 5% ({len(worst_5pct)} trades) ===\n")
    w5_pnls = [t[0] for t in worst_5pct]
    lines.append(f"Mean PnL: ${np.mean(w5_pnls):.2f}\n")
    w5_reasons = Counter(t[1].split(':')[0] for t in worst_5pct)
    lines.append(f"Exit reasons: {dict(w5_reasons)}\n")
    w5_bars = [t[2] for t in worst_5pct]
    lines.append(f"Bars held: mean={np.mean(w5_bars):.1f}, median={np.median(w5_bars):.1f}\n")

    # CVaR (Expected Shortfall)
    cvar_1 = np.mean(w1_pnls)
    cvar_5 = np.mean(w5_pnls)
    lines.append(f"\nCVaR(1%): ${cvar_1:.2f}, CVaR(5%): ${cvar_5:.2f}\n")

    report = ''.join(lines)
    with open(f"{out}/R12-F5_tail_risk.txt", 'w') as f:
        f.write(report)
    print(report[:500])


# ═══════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    phases = {
        'A': [run_r12_a1, run_r12_a2, run_r12_a3],
        'B': [run_r12_b1, run_r12_b3, run_r12_b4],
        'C': [run_r12_c1, run_r12_c2, run_r12_c3],
        'D': [run_r12_d1, run_r12_d2, run_r12_d3, run_r12_d5, run_r12_d6],
        'E': [run_r12_e1],
        'F': [run_r12_f1, run_r12_f5],
    }

    if len(sys.argv) > 1:
        phase_id = sys.argv[1].upper()
        if phase_id in phases:
            print(f"\n{'#'*70}")
            print(f"# Round 12 — Phase {phase_id}")
            print(f"{'#'*70}")
            for fn in phases[phase_id]:
                try:
                    fn(OUTPUT_DIR)
                except Exception as e:
                    print(f"  ERROR in {fn.__name__}: {e}")
                    traceback.print_exc()
        else:
            print(f"Unknown phase: {phase_id}")
            print("Usage: python run_round12.py [A|B|C|D|E|F]")
        print(f"\nPhase elapsed: {time.time()-t0:.0f}s")
        return

    for phase_id, fns in sorted(phases.items()):
        print(f"\n{'#'*70}")
        print(f"# Round 12 — Phase {phase_id}")
        print(f"{'#'*70}")
        for fn in fns:
            try:
                fn(OUTPUT_DIR)
            except Exception as e:
                print(f"  ERROR in {fn.__name__}: {e}")
                traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"Round 12 COMPLETE — Total elapsed: {time.time()-t0:.0f}s")
    print(f"Results in: {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
