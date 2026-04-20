#!/usr/bin/env python3
"""
Round 18 — "Temporal Relevance" (时间相关性衰减研究)
=====================================================
核心假设: 越接近当前时间的数据, 对未来预测的相关性越高;
         远期数据的统计特性可能与当前市场不同.

本轮不修改策略逻辑, 而是通过不同时间窗口/权重的回测,
量化这一假设的真实程度, 并据此优化训练/验证方法.

=== Phase A: 时间段切片对比 (~2h) ===
A1: 逐年 Sharpe/PnL/WinRate 详细画像 (2015-2026)
A2: 滚动2年窗口 Sharpe 趋势 (2015-2017, 2016-2018, ..., 2024-2026)
A3: 滚动3年窗口 Sharpe 趋势
A4: "历史年份 vs 近期年份" 对比 (2015-2019 vs 2020-2026, 2015-2021 vs 2022-2026)

=== Phase B: 参数稳定性时间衰减 (~3h) ===
B1: 只用远期数据 (2015-2019) 优化参数 → 在近期 (2020-2026) 验证
B2: 只用近期数据 (2020-2026) 优化参数 → 在远期 (2015-2019) 验证
B3: Expanding Window 训练 (2015-T → T+1, T+2 测试)
B4: 比较 "远期训练→近期测试" vs "近期训练→近期测试" 的 Sharpe 差异

=== Phase C: 渐进时间窗口 (~3h) ===
C1: 从当前向过去逐渐扩展: 最近1年/2年/3年/.../全部, Sharpe变化趋势
C2: 从远期向当前逐渐扩展: 2015起, +1年/+2年/...直到全部
C3: 确定 "最优回测起始年" — 从哪年开始回测, Sharpe 最稳定?
C4: Anchored Walk-Forward (每次训练最近N年, 测试下一年)

=== Phase D: 不同市场环境相关性 (~2h) ===
D1: 按 ATR 分位数 (低/中/高波动) 划分交易, 各年代分布变化
D2: 按金价趋势 (牛/熊/震荡) 划分区间, 策略表现差异
D3: 近期 vs 远期的 "市场环境分布" 差异分析

=== Phase E: Recency-Weighted 优化 (~4h) ===
E1: 指数衰减权重: 近期交易权重高, 远期交易权重低, 调整Sharpe计算
E2: 线性衰减权重: weight = (year - 2015) / (2026 - 2015)
E3: 阶梯权重: 最近3年 weight=1.0, 3-6年 weight=0.5, 6+年 weight=0.25
E4: 用不同权重方案分别优化关键参数 (KC EMA, KC Mult, Trail),
    对比结果是否比均等权重更优

=== Phase F: 最终验证 (~2h) ===
F1: 基于最优时间窗口/权重的策略 vs 全量数据策略, 前瞻测试
F2: Monte Carlo: 随机打乱年份顺序, Sharpe 是否有显著时间依赖?
F3: 结论汇总: 时间相关性强度量化, 最佳回测窗口建议

预计总耗时: ~16h (208核服务器)
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

OUTPUT_DIR = "results/round18_results"
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
            s.get('avg_win', 0), s.get('avg_loss', 0), s.get('rr', 0))


def _run_one_trades(args):
    """Return per-trade data for detailed analysis."""
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
           str(t.entry_time)[:16], t.direction or '', t.lots, t.entry_price,
           str(t.exit_time)[:16])
          for t in trades[:80000]]
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'], s.get('max_dd_pct', 0), td,
            s.get('year_pnl', {}))


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


FULL_START = "2015-01-01"
FULL_END = "2026-04-10"
SPREAD = 0.3

YEARS = list(range(2015, 2027))
YEAR_RANGES = [(str(y), f"{y}-01-01", f"{y+1}-01-01" if y < 2026 else "2026-04-10")
               for y in range(2015, 2026)]


# ═════════════════════════════════════════════════════════════
# Phase A: 时间段切片对比
# ═════════════════════════════════════════════════════════════
def phase_a1():
    """逐年 Sharpe/PnL/WinRate 详细画像"""
    print("=" * 70)
    print("R18-A1: Per-Year Performance Profile (L7)")
    print("=" * 70)

    tasks = []
    kw = get_l7()
    for name, start, end in YEAR_RANGES:
        tasks.append((f"Y{name}", {**kw}, SPREAD, start, end))

    tasks.append(("FullPeriod", {**kw}, SPREAD, FULL_START, FULL_END))
    results = run_pool(tasks)

    lines = ["R18-A1: Per-Year L7 Performance Profile",
             "=" * 80, ""]
    lines.append(f"{'Year':>6}  {'N':>6}  {'Sharpe':>8}  {'PnL':>12}  {'MaxDD':>10}  "
                 f"{'WR':>6}  {'AvgW':>7}  {'AvgL':>7}  {'RR':>5}")
    lines.append("-" * 80)
    for r in results:
        label, n, sh, pnl, wr, _, mdd, _, _, avgw, avgl, rr = r
        lines.append(f"{label:>6}  {n:>6}  {sh:>8.2f}  {fmt(pnl):>12}  {fmt(mdd):>10}  "
                     f"{wr:>5.1f}%  {avgw:>7.2f}  {avgl:>7.2f}  {rr:>5.2f}")

    return "\n".join(lines)


def phase_a2():
    """滚动2年窗口 Sharpe 趋势"""
    print("=" * 70)
    print("R18-A2: Rolling 2-Year Window Sharpe Trend")
    print("=" * 70)

    tasks = []
    kw = get_l7()
    for y in range(2015, 2025):
        start = f"{y}-01-01"
        end = f"{y+2}-01-01" if y + 2 <= 2026 else "2026-04-10"
        tasks.append((f"{y}-{y+1}", {**kw}, SPREAD, start, end))

    results = run_pool(tasks)

    lines = ["R18-A2: Rolling 2-Year Window Sharpe Trend (L7)",
             "=" * 80, ""]
    lines.append(f"{'Window':>12}  {'N':>6}  {'Sharpe':>8}  {'PnL':>12}  {'MaxDD':>10}  {'WR':>6}")
    lines.append("-" * 70)
    for r in results:
        label, n, sh, pnl, wr, _, mdd, *_ = r
        lines.append(f"{label:>12}  {n:>6}  {sh:>8.2f}  {fmt(pnl):>12}  {fmt(mdd):>10}  {wr:>5.1f}%")

    sharpes = [r[2] for r in results]
    lines.append("")
    lines.append(f"Sharpe trend: min={min(sharpes):.2f}, max={max(sharpes):.2f}, "
                 f"std={np.std(sharpes):.2f}")
    recent_avg = np.mean(sharpes[-3:])
    old_avg = np.mean(sharpes[:3])
    lines.append(f"Recent 3 windows avg: {recent_avg:.2f}, Old 3 windows avg: {old_avg:.2f}, "
                 f"Delta: {recent_avg - old_avg:+.2f}")

    return "\n".join(lines)


def phase_a3():
    """滚动3年窗口 Sharpe 趋势"""
    print("=" * 70)
    print("R18-A3: Rolling 3-Year Window Sharpe Trend")
    print("=" * 70)

    tasks = []
    kw = get_l7()
    for y in range(2015, 2024):
        start = f"{y}-01-01"
        end = f"{y+3}-01-01" if y + 3 <= 2026 else "2026-04-10"
        tasks.append((f"{y}-{y+2}", {**kw}, SPREAD, start, end))

    results = run_pool(tasks)

    lines = ["R18-A3: Rolling 3-Year Window Sharpe Trend (L7)",
             "=" * 80, ""]
    lines.append(f"{'Window':>12}  {'N':>6}  {'Sharpe':>8}  {'PnL':>12}  {'MaxDD':>10}  {'WR':>6}")
    lines.append("-" * 70)
    for r in results:
        label, n, sh, pnl, wr, _, mdd, *_ = r
        lines.append(f"{label:>12}  {n:>6}  {sh:>8.2f}  {fmt(pnl):>12}  {fmt(mdd):>10}  {wr:>5.1f}%")

    sharpes = [r[2] for r in results]
    lines.append("")
    lines.append(f"Sharpe trend: min={min(sharpes):.2f}, max={max(sharpes):.2f}, "
                 f"std={np.std(sharpes):.2f}")

    return "\n".join(lines)


def phase_a4():
    """历史年份 vs 近期年份 对比"""
    print("=" * 70)
    print("R18-A4: Old vs Recent Era Comparison")
    print("=" * 70)

    kw = get_l7()
    splits = [
        ("2015-2019", "2020-2026", "2015-01-01", "2020-01-01", "2020-01-01", "2026-04-10"),
        ("2015-2020", "2021-2026", "2015-01-01", "2021-01-01", "2021-01-01", "2026-04-10"),
        ("2015-2021", "2022-2026", "2015-01-01", "2022-01-01", "2022-01-01", "2026-04-10"),
        ("2015-2022", "2023-2026", "2015-01-01", "2023-01-01", "2023-01-01", "2026-04-10"),
    ]

    tasks = []
    for old_lbl, new_lbl, os_, oe, ns, ne in splits:
        tasks.append((f"Old_{old_lbl}", {**kw}, SPREAD, os_, oe))
        tasks.append((f"New_{new_lbl}", {**kw}, SPREAD, ns, ne))

    results = run_pool(tasks)

    lines = ["R18-A4: Old vs Recent Era Comparison (L7)",
             "=" * 80, ""]
    lines.append(f"{'Period':>16}  {'N':>6}  {'Sharpe':>8}  {'PnL':>12}  {'MaxDD':>10}  "
                 f"{'WR':>6}  {'AvgW':>7}  {'AvgL':>7}")
    lines.append("-" * 80)
    for i in range(0, len(results), 2):
        r_old = results[i]
        r_new = results[i + 1]
        for r in [r_old, r_new]:
            label, n, sh, pnl, wr, _, mdd, _, _, avgw, avgl, rr = r
            lines.append(f"{label:>16}  {n:>6}  {sh:>8.2f}  {fmt(pnl):>12}  {fmt(mdd):>10}  "
                         f"{wr:>5.1f}%  {avgw:>7.2f}  {avgl:>7.2f}")
        delta_sh = r_new[2] - r_old[2]
        lines.append(f"{'':>16}  {'':>6}  {delta_sh:>+8.2f}  {'(new-old)':>12}")
        lines.append("")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# Phase B: 参数稳定性时间衰减
# ═════════════════════════════════════════════════════════════
def phase_b1():
    """远期训练 → 近期验证: KC 参数扫描"""
    print("=" * 70)
    print("R18-B1: Train on Old (2015-2019), Test on Recent (2020-2026)")
    print("=" * 70)

    ema_range = [20, 22, 25, 28, 30]
    mult_range = [1.0, 1.1, 1.2, 1.3, 1.5]

    tasks = []
    for ema in ema_range:
        for mult in mult_range:
            kw = get_l7()
            kw['_kc_ema'] = ema
            kw['_kc_mult'] = mult
            tasks.append((f"E{ema}_M{mult}_OLD", {**kw}, SPREAD, "2015-01-01", "2020-01-01"))
            tasks.append((f"E{ema}_M{mult}_NEW", {**kw}, SPREAD, "2020-01-01", "2026-04-10"))

    results = run_pool(tasks)

    lines = ["R18-B1: KC Params — Train Old (2015-2019), Test Recent (2020-2026)",
             "=" * 80, ""]
    lines.append(f"{'EMA':>5}  {'Mult':>5}  {'Sh_Old':>8}  {'Sh_New':>8}  {'Delta':>8}  "
                 f"{'PnL_Old':>12}  {'PnL_New':>12}  {'WR_Old':>7}  {'WR_New':>7}")
    lines.append("-" * 85)

    best_old_sh, best_old_params = -999, ""
    best_new_sh, best_new_params = -999, ""
    for i in range(0, len(results), 2):
        r_old, r_new = results[i], results[i + 1]
        ema = int(r_old[0].split('_')[0][1:])
        mult = float(r_old[0].split('_')[1][1:])
        sh_old, sh_new = r_old[2], r_new[2]
        delta = sh_new - sh_old
        lines.append(f"{ema:>5}  {mult:>5.1f}  {sh_old:>8.2f}  {sh_new:>8.2f}  {delta:>+8.2f}  "
                     f"{fmt(r_old[3]):>12}  {fmt(r_new[3]):>12}  {r_old[4]:>6.1f}%  {r_new[4]:>6.1f}%")
        if sh_old > best_old_sh:
            best_old_sh = sh_old
            best_old_params = f"EMA={ema}, Mult={mult}"
        if sh_new > best_new_sh:
            best_new_sh = sh_new
            best_new_params = f"EMA={ema}, Mult={mult}"

    lines.append("")
    lines.append(f"Best in Old period: {best_old_params} (Sharpe={best_old_sh:.2f})")
    lines.append(f"Best in New period: {best_new_params} (Sharpe={best_new_sh:.2f})")
    lines.append(f"Same optimal? {'YES' if best_old_params == best_new_params else 'NO — params shifted!'}")

    return "\n".join(lines)


def phase_b2():
    """近期训练 → 远期验证: 反向测试"""
    print("=" * 70)
    print("R18-B2: Train on Recent (2020-2026), Test on Old (2015-2019)")
    print("=" * 70)

    ema_range = [20, 22, 25, 28, 30]
    mult_range = [1.0, 1.1, 1.2, 1.3, 1.5]

    tasks = []
    for ema in ema_range:
        for mult in mult_range:
            kw = get_l7()
            kw['_kc_ema'] = ema
            kw['_kc_mult'] = mult
            tasks.append((f"E{ema}_M{mult}_FULL", {**kw}, SPREAD, FULL_START, FULL_END))

    results = run_pool(tasks)

    lines = ["R18-B2: KC Params — Full Period Scan (baseline for B1 comparison)",
             "=" * 80, ""]
    lines.append(f"{'EMA':>5}  {'Mult':>5}  {'Sharpe':>8}  {'PnL':>12}  {'N':>6}  {'WR':>6}")
    lines.append("-" * 55)

    best_sh, best_params = -999, ""
    for r in results:
        label, n, sh, pnl, wr, *_ = r
        parts = label.split('_')
        ema = int(parts[0][1:])
        mult = float(parts[1][1:])
        lines.append(f"{ema:>5}  {mult:>5.1f}  {sh:>8.2f}  {fmt(pnl):>12}  {n:>6}  {wr:>5.1f}%")
        if sh > best_sh:
            best_sh = sh
            best_params = f"EMA={ema}, Mult={mult}"

    lines.append("")
    lines.append(f"Best on full period: {best_params} (Sharpe={best_sh:.2f})")

    return "\n".join(lines)


def phase_b3():
    """Expanding Window: 从2015起逐年扩展训练 → 测试下一年"""
    print("=" * 70)
    print("R18-B3: Expanding Window Train → Test Next Year")
    print("=" * 70)

    tasks = []
    kw = get_l7()
    for train_end_year in range(2017, 2026):
        train_start = "2015-01-01"
        train_end = f"{train_end_year}-01-01"
        test_start = f"{train_end_year}-01-01"
        test_end = f"{train_end_year + 1}-01-01" if train_end_year < 2025 else "2026-04-10"

        tasks.append((f"Train15-{train_end_year - 1}", {**kw}, SPREAD, train_start, train_end))
        tasks.append((f"Test{train_end_year}", {**kw}, SPREAD, test_start, test_end))

    results = run_pool(tasks)

    lines = ["R18-B3: Expanding Window — Train (2015→T) → Test (T+1)",
             "=" * 80, ""]
    lines.append(f"{'Train':>16}  {'Sh_Train':>10}  {'Test':>10}  {'Sh_Test':>10}  {'Delta':>8}")
    lines.append("-" * 65)

    for i in range(0, len(results), 2):
        r_train, r_test = results[i], results[i + 1]
        delta = r_test[2] - r_train[2]
        lines.append(f"{r_train[0]:>16}  {r_train[2]:>10.2f}  {r_test[0]:>10}  "
                     f"{r_test[2]:>10.2f}  {delta:>+8.2f}")

    train_sharpes = [results[i][2] for i in range(0, len(results), 2)]
    test_sharpes = [results[i][2] for i in range(1, len(results), 2)]
    corr = np.corrcoef(train_sharpes, test_sharpes)[0, 1] if len(train_sharpes) > 2 else 0
    lines.append("")
    lines.append(f"Train-Test Sharpe Correlation: {corr:.3f}")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# Phase C: 渐进时间窗口
# ═════════════════════════════════════════════════════════════
def phase_c1():
    """从当前向过去逐渐扩展: 最近1年/2年/.../全部"""
    print("=" * 70)
    print("R18-C1: Backward Expanding — Most Recent N Years")
    print("=" * 70)

    tasks = []
    kw = get_l7()
    end_date = "2026-04-10"

    windows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    for n in windows:
        start_year = 2026 - n
        start = f"{max(start_year, 2015)}-01-01"
        tasks.append((f"Recent_{n}y", {**kw}, SPREAD, start, end_date))

    results = run_pool(tasks)

    lines = ["R18-C1: Backward Expanding — Most Recent N Years (L7)",
             "=" * 80, ""]
    lines.append(f"{'Window':>12}  {'Start':>10}  {'N':>6}  {'Sharpe':>8}  {'PnL':>12}  "
                 f"{'MaxDD':>10}  {'WR':>6}  {'PnL/Year':>10}")
    lines.append("-" * 80)
    for i, r in enumerate(results):
        label, n, sh, pnl, wr, _, mdd, *_ = r
        years = windows[i]
        pnl_per_year = pnl / years if years > 0 else 0
        start_year = 2026 - years
        lines.append(f"{label:>12}  {start_year:>10}  {n:>6}  {sh:>8.2f}  {fmt(pnl):>12}  "
                     f"{fmt(mdd):>10}  {wr:>5.1f}%  {fmt(pnl_per_year):>10}")

    return "\n".join(lines)


def phase_c2():
    """从远期向当前逐渐扩展: 2015起 +1年/+2年/..."""
    print("=" * 70)
    print("R18-C2: Forward Expanding — From 2015 + N Years")
    print("=" * 70)

    tasks = []
    kw = get_l7()
    start_date = "2015-01-01"

    for n in range(1, 12):
        end_year = 2015 + n
        end = f"{min(end_year, 2026)}-01-01" if end_year < 2026 else "2026-04-10"
        tasks.append((f"From2015_{n}y", {**kw}, SPREAD, start_date, end))

    results = run_pool(tasks)

    lines = ["R18-C2: Forward Expanding — From 2015 + N Years (L7)",
             "=" * 80, ""]
    lines.append(f"{'Window':>14}  {'End':>10}  {'N':>6}  {'Sharpe':>8}  {'PnL':>12}  "
                 f"{'MaxDD':>10}  {'WR':>6}")
    lines.append("-" * 75)
    for r in results:
        label, n, sh, pnl, wr, _, mdd, *_ = r
        lines.append(f"{label:>14}  {'':>10}  {n:>6}  {sh:>8.2f}  {fmt(pnl):>12}  "
                     f"{fmt(mdd):>10}  {wr:>5.1f}%")

    return "\n".join(lines)


def phase_c3():
    """确定最优回测起始年"""
    print("=" * 70)
    print("R18-C3: Optimal Backtest Start Year")
    print("=" * 70)

    tasks = []
    kw = get_l7()
    end_date = "2026-04-10"
    for start_year in range(2015, 2024):
        start = f"{start_year}-01-01"
        tasks.append((f"Start{start_year}", {**kw}, SPREAD, start, end_date))

    results = run_pool(tasks)

    lines = ["R18-C3: Optimal Backtest Start Year (L7, end=2026-04-10)",
             "=" * 80, ""]
    lines.append(f"{'Start':>8}  {'Years':>6}  {'N':>6}  {'Sharpe':>8}  {'PnL':>12}  "
                 f"{'MaxDD':>10}  {'WR':>6}  {'Sharpe/√Yr':>10}")
    lines.append("-" * 75)
    for r in results:
        label, n, sh, pnl, wr, _, mdd, *_ = r
        start_year = int(label.replace("Start", ""))
        years = 2026 - start_year
        sh_per_sqrt_yr = sh / np.sqrt(years) if years > 0 else 0
        lines.append(f"{start_year:>8}  {years:>6}  {n:>6}  {sh:>8.2f}  {fmt(pnl):>12}  "
                     f"{fmt(mdd):>10}  {wr:>5.1f}%  {sh_per_sqrt_yr:>10.2f}")

    return "\n".join(lines)


def phase_c4():
    """Anchored Walk-Forward: 训练最近N年 → 测试下一年"""
    print("=" * 70)
    print("R18-C4: Anchored Walk-Forward (Train Recent N Years → Test Next)")
    print("=" * 70)

    kw = get_l7()
    train_lengths = [2, 3, 4, 5]

    lines = ["R18-C4: Anchored Walk-Forward — Train Recent N Years → Test Next Year",
             "=" * 80, ""]

    for tl in train_lengths:
        lines.append(f"\n--- Train Length: {tl} years ---")
        lines.append(f"{'Train':>16}  {'Test':>8}  {'Sh_Train':>10}  {'Sh_Test':>10}  {'Delta':>8}")
        lines.append("-" * 60)

        tasks = []
        for test_year in range(2015 + tl, 2026):
            train_start = f"{test_year - tl}-01-01"
            train_end = f"{test_year}-01-01"
            test_end = f"{test_year + 1}-01-01" if test_year < 2025 else "2026-04-10"
            tasks.append((f"Tr{test_year - tl}-{test_year - 1}", {**kw}, SPREAD, train_start, train_end))
            tasks.append((f"Tst{test_year}", {**kw}, SPREAD, train_end, test_end))

        results = run_pool(tasks)
        test_sharpes = []

        for i in range(0, len(results), 2):
            r_train, r_test = results[i], results[i + 1]
            delta = r_test[2] - r_train[2]
            lines.append(f"{r_train[0]:>16}  {r_test[0]:>8}  {r_train[2]:>10.2f}  "
                         f"{r_test[2]:>10.2f}  {delta:>+8.2f}")
            test_sharpes.append(r_test[2])

        lines.append(f"  Avg test Sharpe: {np.mean(test_sharpes):.2f}, "
                     f"Std: {np.std(test_sharpes):.2f}")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# Phase D: 不同市场环境相关性
# ═════════════════════════════════════════════════════════════
def phase_d1():
    """按 ATR 分位数划分交易, 各年代表现"""
    print("=" * 70)
    print("R18-D1: ATR Regime Distribution Across Eras")
    print("=" * 70)

    kw = get_l7()
    tasks = [("FullTrades", {**kw}, SPREAD, FULL_START, FULL_END)]
    results = run_pool(tasks, func=_run_one_trades)

    r = results[0]
    label, n, sh, pnl, wr, _, mdd, _, trades_data, year_pnl = r

    lines = ["R18-D1: Trade Performance by Era (L7 full period)",
             "=" * 80, ""]

    era_trades = defaultdict(list)
    for td in trades_data:
        pnl_val, reason, bars, strat, entry_time, direction, lots, eprice, exit_time = td
        year = int(entry_time[:4])
        if year <= 2019:
            era = "Old(15-19)"
        elif year <= 2022:
            era = "Mid(20-22)"
        else:
            era = "New(23-26)"
        era_trades[era].append(pnl_val)

    lines.append(f"{'Era':>12}  {'N':>6}  {'TotalPnL':>12}  {'AvgPnL':>8}  {'WR':>6}  {'Sharpe_est':>10}")
    lines.append("-" * 65)
    for era in ["Old(15-19)", "Mid(20-22)", "New(23-26)"]:
        pnls = era_trades.get(era, [])
        if not pnls:
            continue
        n_t = len(pnls)
        total = sum(pnls)
        avg = np.mean(pnls)
        wr_val = 100 * sum(1 for p in pnls if p > 0) / n_t
        std = np.std(pnls)
        sh_est = (avg / std * np.sqrt(252)) if std > 0 else 0
        lines.append(f"{era:>12}  {n_t:>6}  {fmt(total):>12}  {avg:>8.2f}  {wr_val:>5.1f}%  {sh_est:>10.2f}")

    exit_reasons_by_era = defaultdict(lambda: defaultdict(int))
    for td in trades_data:
        pnl_val, reason, bars, strat, entry_time, *_ = td
        year = int(entry_time[:4])
        era = "Old(15-19)" if year <= 2019 else ("Mid(20-22)" if year <= 2022 else "New(23-26)")
        exit_reasons_by_era[era][reason] += 1

    lines.append("")
    lines.append("Exit reason distribution by era:")
    for era in ["Old(15-19)", "Mid(20-22)", "New(23-26)"]:
        reasons = exit_reasons_by_era.get(era, {})
        total_n = sum(reasons.values())
        lines.append(f"  {era}:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            pct = 100 * count / total_n if total_n > 0 else 0
            lines.append(f"    {reason:>25}: {count:>5} ({pct:>5.1f}%)")

    return "\n".join(lines)


def phase_d2():
    """按金价绝对水平划分区间, 策略表现差异"""
    print("=" * 70)
    print("R18-D2: Performance by Gold Price Level")
    print("=" * 70)

    kw = get_l7()
    tasks = [("FullTrades", {**kw}, SPREAD, FULL_START, FULL_END)]
    results = run_pool(tasks, func=_run_one_trades)

    r = results[0]
    trades_data = r[8]

    price_buckets = defaultdict(list)
    for td in trades_data:
        pnl_val, reason, bars, strat, entry_time, direction, lots, eprice, exit_time = td
        if eprice < 1400:
            bucket = "<1400"
        elif eprice < 1800:
            bucket = "1400-1800"
        elif eprice < 2200:
            bucket = "1800-2200"
        elif eprice < 2600:
            bucket = "2200-2600"
        else:
            bucket = "2600+"
        price_buckets[bucket].append(pnl_val)

    lines = ["R18-D2: L7 Performance by Gold Price Level",
             "=" * 80, ""]
    lines.append(f"{'PriceRange':>12}  {'N':>6}  {'TotalPnL':>12}  {'AvgPnL':>8}  {'WR':>6}")
    lines.append("-" * 50)
    for bucket in ["<1400", "1400-1800", "1800-2200", "2200-2600", "2600+"]:
        pnls = price_buckets.get(bucket, [])
        if not pnls:
            continue
        n_t = len(pnls)
        total = sum(pnls)
        avg = np.mean(pnls)
        wr_val = 100 * sum(1 for p in pnls if p > 0) / n_t
        lines.append(f"{bucket:>12}  {n_t:>6}  {fmt(total):>12}  {avg:>8.2f}  {wr_val:>5.1f}%")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# Phase E: Recency-Weighted 分析
# ═════════════════════════════════════════════════════════════
def phase_e1():
    """用不同权重方案重新评估全量交易的加权 Sharpe"""
    print("=" * 70)
    print("R18-E1: Recency-Weighted Sharpe Analysis")
    print("=" * 70)

    kw = get_l7()
    tasks = [("FullTrades", {**kw}, SPREAD, FULL_START, FULL_END)]
    results = run_pool(tasks, func=_run_one_trades)

    r = results[0]
    trades_data = r[8]

    year_pnls = defaultdict(list)
    for td in trades_data:
        pnl_val = td[0]
        entry_time = td[4]
        year = int(entry_time[:4])
        year_pnls[year].append(pnl_val)

    lines = ["R18-E1: Recency-Weighted Sharpe Analysis (L7)",
             "=" * 80, ""]

    def calc_weighted_sharpe(pnls_by_year, weight_func, name):
        weighted_pnls = []
        weights = []
        for year in sorted(pnls_by_year.keys()):
            w = weight_func(year)
            for p in pnls_by_year[year]:
                weighted_pnls.append(p * w)
                weights.append(w)
        if not weighted_pnls:
            return 0
        arr = np.array(weighted_pnls)
        return np.mean(arr) / np.std(arr) * np.sqrt(252) if np.std(arr) > 0 else 0

    schemes = {
        "Equal": lambda y: 1.0,
        "Linear": lambda y: (y - 2014) / (2026 - 2014),
        "Exp_0.85": lambda y: 0.85 ** (2026 - y),
        "Exp_0.90": lambda y: 0.90 ** (2026 - y),
        "Exp_0.95": lambda y: 0.95 ** (2026 - y),
        "Step_3/6": lambda y: 1.0 if y >= 2023 else (0.5 if y >= 2020 else 0.25),
        "Step_5": lambda y: 1.0 if y >= 2021 else 0.3,
        "Recent5Only": lambda y: 1.0 if y >= 2021 else 0.0,
        "Recent3Only": lambda y: 1.0 if y >= 2023 else 0.0,
    }

    lines.append(f"{'Scheme':>15}  {'W_Sharpe':>10}  Weight distribution:")
    lines.append("-" * 70)
    for name, wfunc in schemes.items():
        wsh = calc_weighted_sharpe(year_pnls, wfunc, name)
        weight_str = ", ".join(f"{y}:{wfunc(y):.2f}" for y in range(2015, 2027))
        lines.append(f"{name:>15}  {wsh:>10.2f}  {weight_str}")

    lines.append("")
    lines.append("Per-year Sharpe (unweighted):")
    lines.append(f"{'Year':>6}  {'N':>6}  {'Sharpe':>8}  {'PnL':>12}  {'WR':>6}")
    lines.append("-" * 45)
    for year in sorted(year_pnls.keys()):
        pnls = year_pnls[year]
        n_t = len(pnls)
        total = sum(pnls)
        avg = np.mean(pnls)
        std = np.std(pnls)
        sh = avg / std * np.sqrt(252) if std > 0 else 0
        wr_val = 100 * sum(1 for p in pnls if p > 0) / n_t
        lines.append(f"{year:>6}  {n_t:>6}  {sh:>8.2f}  {fmt(total):>12}  {wr_val:>5.1f}%")

    return "\n".join(lines)


def phase_e2():
    """KC 参数: 近期加权 vs 均等权重优化对比"""
    print("=" * 70)
    print("R18-E2: Recency-Weighted Parameter Optimization")
    print("=" * 70)

    ema_range = [20, 22, 25, 28, 30]
    mult_range = [1.0, 1.1, 1.2, 1.3, 1.5]

    tasks = []
    for ema in ema_range:
        for mult in mult_range:
            kw = get_l7()
            kw['_kc_ema'] = ema
            kw['_kc_mult'] = mult
            for name, start, end in YEAR_RANGES:
                tasks.append((f"E{ema}_M{mult}_Y{name}", {**kw}, SPREAD, start, end))

    results = run_pool(tasks)

    result_map = {}
    for r in results:
        result_map[r[0]] = r

    lines = ["R18-E2: KC Params — Equal vs Recency-Weighted Ranking",
             "=" * 80, ""]

    def weighted_sharpe_from_years(ema, mult, weight_func):
        year_sharpes = []
        year_weights = []
        for name, _, _ in YEAR_RANGES:
            key = f"E{ema}_M{mult}_Y{name}"
            r = result_map.get(key)
            if r and r[1] > 10:
                year_sharpes.append(r[2])
                year_weights.append(weight_func(int(name)))
        if not year_sharpes:
            return 0
        ws = np.array(year_sharpes)
        ww = np.array(year_weights)
        return np.average(ws, weights=ww) if ww.sum() > 0 else 0

    equal_ranking = []
    recency_ranking = []
    for ema in ema_range:
        for mult in mult_range:
            eq_sh = weighted_sharpe_from_years(ema, mult, lambda y: 1.0)
            rc_sh = weighted_sharpe_from_years(ema, mult, lambda y: 0.90 ** (2026 - y))
            equal_ranking.append((ema, mult, eq_sh))
            recency_ranking.append((ema, mult, rc_sh))

    equal_ranking.sort(key=lambda x: -x[2])
    recency_ranking.sort(key=lambda x: -x[2])

    lines.append("Equal-Weight Top 5:")
    lines.append(f"{'Rank':>5}  {'EMA':>5}  {'Mult':>5}  {'AvgSharpe':>10}")
    for i, (ema, mult, sh) in enumerate(equal_ranking[:5]):
        lines.append(f"{i + 1:>5}  {ema:>5}  {mult:>5.1f}  {sh:>10.2f}")

    lines.append("")
    lines.append("Recency-Weighted (decay=0.90) Top 5:")
    lines.append(f"{'Rank':>5}  {'EMA':>5}  {'Mult':>5}  {'WtdSharpe':>10}")
    for i, (ema, mult, sh) in enumerate(recency_ranking[:5]):
        lines.append(f"{i + 1:>5}  {ema:>5}  {mult:>5.1f}  {sh:>10.2f}")

    eq_best = equal_ranking[0]
    rc_best = recency_ranking[0]
    lines.append("")
    lines.append(f"Equal-weight best: EMA={eq_best[0]}, Mult={eq_best[1]} (Sharpe={eq_best[2]:.2f})")
    lines.append(f"Recency-weighted best: EMA={rc_best[0]}, Mult={rc_best[1]} (Sharpe={rc_best[2]:.2f})")
    lines.append(f"Same? {'YES' if (eq_best[0] == rc_best[0] and eq_best[1] == rc_best[1]) else 'NO'}")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# Phase F: 最终验证
# ═════════════════════════════════════════════════════════════
def phase_f1():
    """Monte Carlo: 随机打乱年份顺序, 测试时间依赖性"""
    print("=" * 70)
    print("R18-F1: Year-Shuffle Monte Carlo — Is Sharpe Time-Dependent?")
    print("=" * 70)

    import random as rng

    kw = get_l7()
    tasks = [("FullTrades", {**kw}, SPREAD, FULL_START, FULL_END)]
    results = run_pool(tasks, func=_run_one_trades)
    trades_data = results[0][8]

    year_pnls = defaultdict(list)
    for td in trades_data:
        year = int(td[4][:4])
        year_pnls[year].append(td[0])

    years_list = sorted(year_pnls.keys())
    original_sharpes = []
    for y in years_list:
        pnls = year_pnls[y]
        avg = np.mean(pnls)
        std = np.std(pnls)
        sh = avg / std * np.sqrt(252) if std > 0 else 0
        original_sharpes.append(sh)

    n_sims = 1000
    rng.seed(42)
    shuffled_trends = []
    for _ in range(n_sims):
        order = list(range(len(years_list)))
        rng.shuffle(order)
        shuffled_sharpes = [original_sharpes[i] for i in order]
        trend = np.corrcoef(range(len(shuffled_sharpes)), shuffled_sharpes)[0, 1]
        shuffled_trends.append(trend)

    real_trend = np.corrcoef(range(len(original_sharpes)), original_sharpes)[0, 1]

    pct_above = 100 * sum(1 for t in shuffled_trends if t >= real_trend) / n_sims

    lines = ["R18-F1: Year-Shuffle Monte Carlo — Time Dependency Test",
             "=" * 80, ""]
    lines.append(f"Original year order Sharpe trend (correlation with time): {real_trend:+.3f}")
    lines.append(f"Shuffled distribution: mean={np.mean(shuffled_trends):.3f}, "
                 f"std={np.std(shuffled_trends):.3f}")
    lines.append(f"P(shuffled >= real): {pct_above:.1f}%")
    lines.append(f"Conclusion: {'Significant time trend (p<5%)' if pct_above < 5 else 'No significant time trend (p>=5%)'}")
    lines.append("")
    lines.append("Original per-year Sharpes:")
    for y, sh in zip(years_list, original_sharpes):
        lines.append(f"  {y}: {sh:.2f}")

    return "\n".join(lines)


def phase_f2():
    """结论汇总"""
    print("=" * 70)
    print("R18-F2: Summary & Recommendations")
    print("=" * 70)

    kw = get_l7()
    periods = [
        ("Full(15-26)", FULL_START, FULL_END),
        ("Recent5(21-26)", "2021-01-01", FULL_END),
        ("Recent3(23-26)", "2023-01-01", FULL_END),
        ("Old(15-19)", "2015-01-01", "2020-01-01"),
    ]

    tasks = [(name, {**kw}, SPREAD, s, e) for name, s, e in periods]
    for sp in [0.3, 0.5]:
        for name, s, e in periods:
            if sp != 0.3:
                tasks.append((f"{name}_sp{sp}", {**kw}, sp, s, e))

    results = run_pool(tasks)

    lines = ["R18-F2: Summary — Temporal Relevance Analysis",
             "=" * 80, ""]
    lines.append(f"{'Period':>20}  {'Spread':>6}  {'N':>6}  {'Sharpe':>8}  "
                 f"{'PnL':>12}  {'MaxDD':>10}  {'WR':>6}")
    lines.append("-" * 75)
    for r in results:
        label, n, sh, pnl, wr, _, mdd, *_ = r
        sp = 0.5 if '_sp0.5' in label else 0.3
        name = label.replace('_sp0.5', '')
        lines.append(f"{name:>20}  {sp:>6.1f}  {n:>6}  {sh:>8.2f}  "
                     f"{fmt(pnl):>12}  {fmt(mdd):>10}  {wr:>5.1f}%")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    t0 = time.time()
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    phase_times = []

    phases = [
        ("A1", "Per-Year Profile", phase_a1),
        ("A2", "Rolling 2Y Window", phase_a2),
        ("A3", "Rolling 3Y Window", phase_a3),
        ("A4", "Old vs Recent Era", phase_a4),
        ("B1", "Old Train → New Test", phase_b1),
        ("B2", "Full Period KC Scan", phase_b2),
        ("B3", "Expanding Window", phase_b3),
        ("C1", "Backward Expanding", phase_c1),
        ("C2", "Forward Expanding", phase_c2),
        ("C3", "Optimal Start Year", phase_c3),
        ("C4", "Anchored Walk-Forward", phase_c4),
        ("D1", "Era Trade Analysis", phase_d1),
        ("D2", "Price Level Analysis", phase_d2),
        ("E1", "Recency-Weighted Sharpe", phase_e1),
        ("E2", "Recency-Weighted Params", phase_e2),
        ("F1", "Year-Shuffle MC", phase_f1),
        ("F2", "Summary", phase_f2),
    ]

    for pid, pname, pfunc in phases:
        pt0 = time.time()
        try:
            result_text = pfunc()
            elapsed_p = time.time() - pt0
            phase_times.append((pid, pname, elapsed_p, "OK"))

            fname = os.path.join(OUTPUT_DIR, f"R18_{pid}_{pname.replace(' ', '_').lower()}.txt")
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(result_text)

            print(f"    {pid} done")
            print(f"    Phase {pid}: {pname} completed in {elapsed_p:.0f}s")
        except Exception as e:
            elapsed_p = time.time() - pt0
            phase_times.append((pid, pname, elapsed_p, f"FAIL: {e}"))
            print(f"    {pid} FAILED: {e}")
            traceback.print_exc()

    total = time.time() - t0
    summary_lines = [
        f"Round 18 — Temporal Relevance",
        "=" * 60,
        f"Total: {total:.0f}s ({total / 3600:.1f}h)",
        f"Started: {started}",
        "",
    ]
    for pid, pname, pt, status in phase_times:
        summary_lines.append(f"Phase {pid}: {pname:30s} {pt:>8.0f}s  {status}")

    summary = "\n".join(summary_lines)
    with open(os.path.join(OUTPUT_DIR, "R18_summary.txt"), 'w', encoding='utf-8') as f:
        f.write(summary)

    print()
    print(summary)
    print("=" * 60)
    print(f"Round 18 COMPLETE: {total:.0f}s ({total / 3600:.1f}h)")
    print("=" * 60)


if __name__ == "__main__":
    main()
