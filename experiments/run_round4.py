#!/usr/bin/env python3
"""
Round 4 Experiments — 边界条件 + 实战模拟 + 微观结构
====================================================
前 3 轮覆盖了参数优化和 K-Fold 验证。Round 4 聚焦:
  R4-1: 危机期专项（2020 COVID, 2022 加息, 2026 关税战）每段单独统计
  R4-2: Drawdown Recovery 模拟 — 从任意历史起点开始 $2000 本金，最终存活率
  R4-3: 交易频率与盈利关系 — 高频日 vs 低频日表现差异
  R4-4: 盈利因子衰减测试 — 前半段 vs 后半段 IC 对比
  R4-5: 最终候选配置 Ensemble — top-3 配置投票系统回测
  R4-6: 滑点敏感性 — 入场价偏移 0.1~1.0 ATR 对策略的影响
  R4-7: Regime 切换时的策略表现 — 从趋势→震荡 / 震荡→趋势 的过渡期
  R4-8: 双策略对冲 — Keltner + ORB 同向/反向信号时的表现差异
预计 ~2 小时
"""
import sys, os, time, traceback
import multiprocessing as mp
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "results/round4_results"
MAX_WORKERS = 14


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def _run_one(args):
    label, kw, spread, start, end = args
    from backtest import DataBundle, run_variant
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    if start and end:
        data = data.slice(start, end)
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kw)
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'])


def _run_one_trades(args):
    label, kw, spread, start, end = args
    from backtest import DataBundle, run_variant
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    if start and end:
        data = data.slice(start, end)
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kw)
    trades = s.get('_trades', [])
    td = [(t.pnl, t.exit_reason or '', t.bars_held, t.strategy or '',
           str(t.entry_time), str(t.exit_time)) for t in trades]
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'], td)


def run_pool(tasks):
    with mp.Pool(min(MAX_WORKERS, len(tasks))) as pool:
        return pool.map(_run_one, tasks)


def get_base():
    from backtest.runner import LIVE_PARITY_KWARGS
    return {**LIVE_PARITY_KWARGS, "time_decay_tp": False}


# ═══════════════════════════════════════════
# R4-0: 样本外 OOS 验证（最科学的方法）
# ═══════════════════════════════════════════
def r4_0_out_of_sample(p):
    p("=" * 80)
    p("R4-0: 样本外 (Out-of-Sample) 验证")
    p("=" * 80)
    p("")
    p("  核心问题: 策略参数是用全部 11 年数据调出来的。")
    p("  如果只看全样本 Sharpe，可能存在过拟合风险。")
    p("  以下测试用多种方式检验策略在'没见过的数据'上是否依然有效。")

    L4 = get_base()

    # ─── Part A: Anchored Walk-Forward (锚定式前推) ───
    # 固定起点 2015，不断扩大训练集，每次测试下一年
    # 核心思想：模拟"策略上线后每年的真实表现"
    p(f"\n{'='*60}")
    p("Part A: 锚定式前推 (Anchored Walk-Forward)")
    p("  固定从 2015 年开始，测试每一年是否都赚钱")
    p("  类似你说的'前N年训练→测试后面'，但更精细")
    p(f"{'='*60}")

    anchor_tests = [
        # (训练期描述, 测试期, 测试开始, 测试结束)
        ("Train 2015",         "Test 2016", "2016-01-01", "2016-12-31"),
        ("Train 2015-2016",    "Test 2017", "2017-01-01", "2017-12-31"),
        ("Train 2015-2017",    "Test 2018", "2018-01-01", "2018-12-31"),
        ("Train 2015-2018",    "Test 2019", "2019-01-01", "2019-12-31"),
        ("Train 2015-2019",    "Test 2020", "2020-01-01", "2020-12-31"),
        ("Train 2015-2020",    "Test 2021", "2021-01-01", "2021-12-31"),
        ("Train 2015-2021",    "Test 2022", "2022-01-01", "2022-12-31"),
        ("Train 2015-2022",    "Test 2023", "2023-01-01", "2023-12-31"),
        ("Train 2015-2023",    "Test 2024", "2024-01-01", "2024-12-31"),
        ("Train 2015-2024",    "Test 2025", "2025-01-01", "2025-12-31"),
        ("Train 2015-2025",    "Test 2026", "2026-01-01", "2026-04-10"),
    ]

    tasks = [(f"{test_desc}", L4, 0.30, ts, te)
             for _, test_desc, ts, te in anchor_tests]
    results = run_pool(tasks)
    result_map = {r[0]: r for r in results}

    p(f"\n{'训练数据':<20s}  {'测试年':>10s}  {'N':>5s}  {'Sharpe':>7s}  {'PnL':>10s}  {'WR%':>6s}  {'$/t':>7s}  {'MaxDD':>10s}")
    p("-" * 90)
    positive_years = 0
    all_sharpes = []
    for train_desc, test_desc, _, _ in anchor_tests:
        r = result_map.get(test_desc)
        if r:
            avg = r[3] / r[1] if r[1] > 0 else 0
            marker = ""
            if r[3] < 0:
                marker = " *** LOSS"
            elif r[2] < 1.0:
                marker = " * weak"
            else:
                positive_years += 1
            all_sharpes.append(r[2])
            p(f"  {train_desc:<18s}  {test_desc:>10s}  {r[1]:>5d}  {r[2]:>7.2f}  {fmt(r[3])}  {r[4]:>5.1f}%  ${avg:>5.2f}  {fmt(r[6])}{marker}")

    p(f"\n  盈利年份: {positive_years}/{len(anchor_tests)}")
    p(f"  平均 Sharpe: {np.mean(all_sharpes):.2f}")
    p(f"  最差 Sharpe: {min(all_sharpes):.2f}")
    p(f"  标准差: {np.std(all_sharpes):.2f}")

    # ─── Part B: 大切分 (你的直觉) ───
    # 前半 vs 后半，前1/3 vs 后2/3 等
    p(f"\n{'='*60}")
    p("Part B: 大切分 — 前半 vs 后半 / 前1/3 vs 后2/3")
    p("  测试策略在不同历史段的表现差异")
    p(f"{'='*60}")

    splits = [
        ("前3年(2015-2017)",  "2015-01-01", "2017-12-31"),
        ("后8年(2018-2026)",  "2018-01-01", "2026-04-10"),
        ("前5年(2015-2019)",  "2015-01-01", "2019-12-31"),
        ("后6年(2020-2026)",  "2020-01-01", "2026-04-10"),
        ("前7年(2015-2021)",  "2015-01-01", "2021-12-31"),
        ("后4年(2022-2026)",  "2022-01-01", "2026-04-10"),
        ("前9年(2015-2023)",  "2015-01-01", "2023-12-31"),
        ("后2年(2024-2026)",  "2024-01-01", "2026-04-10"),
        ("奇数年(15/17/19/21/23/25)", None, None),  # handled separately
        ("偶数年(16/18/20/22/24/26)", None, None),
    ]

    # Regular splits
    reg_tasks = [(name, L4, 0.30, s, e)
                 for name, s, e in splits if s is not None]
    reg_results = run_pool(reg_tasks)

    p(f"\n{'切分':<25s}  {'N':>5s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'$/t':>7s}")
    p("-" * 65)
    for r in reg_results:
        avg = r[3] / r[1] if r[1] > 0 else 0
        p(f"  {r[0]:<23s}  {r[1]:>5d}  {r[2]:>7.2f}  {fmt(r[3])}  {r[4]:>5.1f}%  ${avg:>5.2f}")

    # Odd/even year splits
    odd_years = [(f"ODD_{y}", L4, 0.30, f"{y}-01-01", f"{y}-12-31")
                 for y in [2015, 2017, 2019, 2021, 2023, 2025]]
    even_years = [(f"EVEN_{y}", L4, 0.30, f"{y}-01-01", f"{y}-12-31" if y < 2026 else "2026-04-10")
                  for y in [2016, 2018, 2020, 2022, 2024, 2026]]

    odd_results = run_pool(odd_years)
    even_results = run_pool(even_years)

    odd_pnl = sum(r[3] for r in odd_results)
    even_pnl = sum(r[3] for r in even_results)
    odd_n = sum(r[1] for r in odd_results)
    even_n = sum(r[1] for r in even_results)
    odd_sharpe_avg = np.mean([r[2] for r in odd_results])
    even_sharpe_avg = np.mean([r[2] for r in even_results])

    p(f"\n  奇数年合计: N={odd_n}, PnL={fmt(odd_pnl)}, 平均Sharpe={odd_sharpe_avg:.2f}")
    p(f"  偶数年合计: N={even_n}, PnL={fmt(even_pnl)}, 平均Sharpe={even_sharpe_avg:.2f}")
    p(f"  差异: {abs(odd_sharpe_avg - even_sharpe_avg):.2f} (越小越好)")

    # ─── Part C: 纯样本外模拟 ───
    # 假装策略是在 2015-2019 调出来的，看 2020-2026 表现
    # 再假装是 2015-2017 调出来的，看 2018-2026
    p(f"\n{'='*60}")
    p("Part C: 纯样本外模拟")
    p("  假设策略只用了前 N 年的数据开发")
    p("  然后看剩余年份（'未来'）的表现")
    p("  关键指标: 样本外 Sharpe 是否显著低于样本内")
    p(f"{'='*60}")

    oos_scenarios = [
        ("Dev 3Y / OOS 8Y", "2015-01-01", "2017-12-31", "2018-01-01", "2026-04-10"),
        ("Dev 4Y / OOS 7Y", "2015-01-01", "2018-12-31", "2019-01-01", "2026-04-10"),
        ("Dev 5Y / OOS 6Y", "2015-01-01", "2019-12-31", "2020-01-01", "2026-04-10"),
        ("Dev 6Y / OOS 5Y", "2015-01-01", "2020-12-31", "2021-01-01", "2026-04-10"),
        ("Dev 7Y / OOS 4Y", "2015-01-01", "2021-12-31", "2022-01-01", "2026-04-10"),
        ("Dev 8Y / OOS 3Y", "2015-01-01", "2022-12-31", "2023-01-01", "2026-04-10"),
        ("Dev 9Y / OOS 2Y", "2015-01-01", "2023-12-31", "2024-01-01", "2026-04-10"),
    ]

    oos_tasks = []
    for name, ds, de, os_s, os_e in oos_scenarios:
        oos_tasks.append((f"IN_{name}", L4, 0.30, ds, de))
        oos_tasks.append((f"OOS_{name}", L4, 0.30, os_s, os_e))

    oos_results = run_pool(oos_tasks)
    oos_map = {r[0]: r for r in oos_results}

    p(f"\n{'场景':<22s}  {'样本内Sharpe':>12s}  {'样本外Sharpe':>12s}  {'衰减':>8s}  {'样本外PnL':>11s}  {'样本外WR%':>9s}")
    p("-" * 85)
    for name, _, _, _, _ in oos_scenarios:
        in_r = oos_map.get(f"IN_{name}")
        oos_r = oos_map.get(f"OOS_{name}")
        if in_r and oos_r:
            decay = oos_r[2] - in_r[2]
            decay_pct = decay / in_r[2] * 100 if in_r[2] != 0 else 0
            marker = ""
            if oos_r[2] < 0:
                marker = " ⛔"
            elif decay_pct < -50:
                marker = " ⚠️ 大衰减"
            elif decay_pct < -20:
                marker = " 📉"
            p(f"  {name:<20s}  {in_r[2]:>12.2f}  {oos_r[2]:>12.2f}  {decay:>+7.2f}{marker}  {fmt(oos_r[3])}  {oos_r[4]:>8.1f}%")

    p(f"\n  解读:")
    p(f"  - 如果样本外 Sharpe 和样本内接近 → 策略稳健，没有过拟合")
    p(f"  - 如果样本外远低于样本内 → 过拟合风险，参数只适合过去")
    p(f"  - 如果样本外反而更高 → 策略近年表现更好（趋势增强）")

    # Also do all of Part C at $0.50 spread
    p(f"\n--- 同样场景，$0.50 spread ---")
    oos_tasks_50 = []
    for name, _, _, os_s, os_e in oos_scenarios:
        oos_tasks_50.append((f"OOS50_{name}", L4, 0.50, os_s, os_e))
    oos_50 = run_pool(oos_tasks_50)
    neg = sum(1 for r in oos_50 if r[2] < 0)
    p(f"  $0.50 spread 下负 Sharpe 的 OOS 期数: {neg}/{len(oos_scenarios)}")
    for r in sorted(oos_50, key=lambda x: x[2]):
        p(f"    {r[0]:<30s}  Sharpe={r[2]:>6.2f}  PnL={fmt(r[3])}")


# ═══════════════════════════════════════════
# R4-1: 危机期专项
# ═══════════════════════════════════════════
def r4_1_crisis_periods(p):
    p("=" * 80)
    p("R4-1: 危机期专项分析")
    p("=" * 80)

    L4 = get_base()

    periods = [
        ("COVID_crash",    "2020-02-20", "2020-04-15"),
        ("COVID_recovery", "2020-04-15", "2020-08-31"),
        ("Fed_hike_2022",  "2022-01-01", "2022-12-31"),
        ("SVB_crisis",     "2023-03-01", "2023-04-30"),
        ("Gold_rally_2024","2024-02-01", "2024-05-31"),
        ("Tariff_war",     "2025-11-01", "2026-04-10"),
        ("Trump_tariff_Q1","2026-01-01", "2026-04-10"),
        ("Low_vol_2018",   "2018-05-01", "2018-10-31"),
        ("Flat_2021_H1",   "2021-01-01", "2021-06-30"),
        ("Iran_conflict",  "2026-02-01", "2026-03-31"),
    ]

    tasks = [(name, L4, 0.30, start, end) for name, start, end in periods]
    results = run_pool(tasks)

    p(f"\n{'Period':<20s}  {'N':>5s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'$/t':>8s}  {'MaxDD':>11s}")
    p("-" * 80)
    for r in sorted(results, key=lambda x: x[0]):
        avg = r[3] / r[1] if r[1] > 0 else 0
        marker = ""
        if r[2] < 0:
            marker = " *** NEGATIVE"
        elif r[2] < 1.0:
            marker = " * weak"
        p(f"  {r[0]:<18s}  {r[1]:>5d}  {r[2]:>7.2f}  {fmt(r[3])}  {r[4]:>5.1f}%  ${avg:>6.2f}  {fmt(r[6])}{marker}")

    # Also test with $0.50 spread
    p(f"\n--- Same periods with $0.50 spread ---")
    tasks50 = [(f"{name}_sp50", L4, 0.50, start, end) for name, start, end in periods]
    results50 = run_pool(tasks50)
    neg_count = sum(1 for r in results50 if r[2] < 0)
    p(f"  Negative Sharpe periods at $0.50: {neg_count}/{len(periods)}")
    for r in sorted(results50, key=lambda x: x[2]):
        if r[2] < 1.0:
            avg = r[3] / r[1] if r[1] > 0 else 0
            p(f"  {r[0]:<25s}  N={r[1]:>4d}  Sharpe={r[2]:>6.2f}  PnL={fmt(r[3])}")


# ═══════════════════════════════════════════
# R4-2: $2000 本金存活模拟
# ═══════════════════════════════════════════
def r4_2_survival_simulation(p):
    p("=" * 80)
    p("R4-2: $2,000 本金存活模拟")
    p("  从每个可能的历史起点开始交易，计算破产概率")
    p("=" * 80)

    from backtest import DataBundle, run_variant
    L4 = get_base()

    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    s = run_variant(data, "survival", verbose=False, spread_cost=0.30, **L4)
    trades = s.get('_trades', [])

    if not trades:
        p("  No trades!")
        return

    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    capital = 2000.0
    max_loss = 1500.0  # risk_manager max total loss

    p(f"\n  Total trades: {n}")
    p(f"  Starting capital: ${capital:.0f}")
    p(f"  Max allowed loss: ${max_loss:.0f} (ruin at ${capital - max_loss:.0f})")

    # From each starting trade, simulate forward
    ruin_count = 0
    min_equity_all = []
    final_equity_all = []
    ruin_at_trade = []

    step = max(1, n // 500)  # sample ~500 starting points
    start_points = list(range(0, n - 50, step))

    for start_idx in start_points:
        equity = capital
        min_eq = capital
        ruined = False
        for i in range(start_idx, n):
            equity += pnls[i]
            min_eq = min(min_eq, equity)
            if equity <= capital - max_loss:
                ruined = True
                ruin_at_trade.append(i - start_idx)
                break
        if ruined:
            ruin_count += 1
        min_equity_all.append(min_eq)
        final_equity_all.append(equity)

    ruin_pct = ruin_count / len(start_points) * 100
    min_equity_all = np.array(min_equity_all)
    final_equity_all = np.array(final_equity_all)

    p(f"\n  Simulated {len(start_points)} starting points (step={step})")
    p(f"  Ruin probability: {ruin_pct:.1f}% ({ruin_count}/{len(start_points)})")
    p(f"  Min equity: mean=${min_equity_all.mean():.0f}, "
      f"worst=${min_equity_all.min():.0f}, "
      f"5th pct=${np.percentile(min_equity_all, 5):.0f}")
    p(f"  Final equity: mean=${final_equity_all.mean():.0f}, "
      f"median=${np.median(final_equity_all):.0f}")
    p(f"  P(final > start): {(final_equity_all > capital).mean()*100:.1f}%")

    if ruin_at_trade:
        p(f"  Ruin speed: mean={np.mean(ruin_at_trade):.0f} trades, "
          f"fastest={min(ruin_at_trade)} trades")

    # Also test with $0.50 spread
    p(f"\n--- Same simulation with $0.50 spread ---")
    s50 = run_variant(data, "surv50", verbose=False, spread_cost=0.50, **L4)
    trades50 = s50.get('_trades', [])
    if trades50:
        pnls50 = np.array([t.pnl for t in trades50])
        ruin50 = 0
        for start_idx in start_points:
            if start_idx >= len(pnls50) - 50:
                continue
            equity = capital
            for i in range(start_idx, len(pnls50)):
                equity += pnls50[i]
                if equity <= capital - max_loss:
                    ruin50 += 1
                    break
        valid = sum(1 for s in start_points if s < len(pnls50) - 50)
        p(f"  $0.50 spread ruin: {ruin50/max(1,valid)*100:.1f}% ({ruin50}/{valid})")


# ═══════════════════════════════════════════
# R4-3: 交易频率与盈利关系
# ═══════════════════════════════════════════
def r4_3_frequency_analysis(p):
    p("=" * 80)
    p("R4-3: 交易频率与盈利关系")
    p("=" * 80)

    from backtest import DataBundle, run_variant
    import pandas as pd
    L4 = get_base()

    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    s = run_variant(data, "freq", verbose=False, spread_cost=0.30, **L4)
    trades = s.get('_trades', [])

    if not trades:
        p("  No trades!")
        return

    # Group by date
    daily = {}
    for t in trades:
        day = pd.Timestamp(t.entry_time).strftime('%Y-%m-%d')
        if day not in daily:
            daily[day] = []
        daily[day].append(t.pnl)

    # Categorize days by trade count
    freq_buckets = {}
    for day, pnls in daily.items():
        n = len(pnls)
        if n <= 2:
            bucket = "1-2"
        elif n <= 4:
            bucket = "3-4"
        elif n <= 6:
            bucket = "5-6"
        elif n <= 10:
            bucket = "7-10"
        else:
            bucket = "11+"

        if bucket not in freq_buckets:
            freq_buckets[bucket] = {'days': 0, 'trades': 0, 'pnl': 0, 'win_days': 0}
        freq_buckets[bucket]['days'] += 1
        freq_buckets[bucket]['trades'] += n
        freq_buckets[bucket]['pnl'] += sum(pnls)
        if sum(pnls) > 0:
            freq_buckets[bucket]['win_days'] += 1

    p(f"\n{'Trades/Day':<12s}  {'Days':>6s}  {'Trades':>7s}  {'PnL':>11s}  {'$/day':>8s}  {'WinDay%':>8s}")
    p("-" * 60)
    for bucket in ["1-2", "3-4", "5-6", "7-10", "11+"]:
        if bucket in freq_buckets:
            d = freq_buckets[bucket]
            avg_day = d['pnl'] / d['days'] if d['days'] > 0 else 0
            wd_pct = d['win_days'] / d['days'] * 100 if d['days'] > 0 else 0
            p(f"  {bucket:<10s}  {d['days']:>6d}  {d['trades']:>7d}  {fmt(d['pnl'])}  ${avg_day:>6.2f}  {wd_pct:>6.1f}%")

    # Same-day multi-trade: does the 2nd+ trade of the day perform differently?
    p(f"\n--- 当日第 N 笔交易表现 ---")
    nth_trade = {}
    for day, pnls in daily.items():
        for i, pnl in enumerate(pnls):
            n = min(i + 1, 5)  # cap at 5+
            key = f"#{n}" if n < 5 else "#5+"
            if key not in nth_trade:
                nth_trade[key] = {'n': 0, 'pnl': 0, 'wins': 0}
            nth_trade[key]['n'] += 1
            nth_trade[key]['pnl'] += pnl
            if pnl > 0:
                nth_trade[key]['wins'] += 1

    p(f"{'Nth':<6s}  {'N':>6s}  {'PnL':>11s}  {'WR%':>6s}  {'$/t':>8s}")
    p("-" * 45)
    for key in ["#1", "#2", "#3", "#4", "#5+"]:
        if key in nth_trade:
            d = nth_trade[key]
            wr = d['wins'] / d['n'] * 100 if d['n'] > 0 else 0
            avg = d['pnl'] / d['n'] if d['n'] > 0 else 0
            p(f"  {key:<4s}  {d['n']:>6d}  {fmt(d['pnl'])}  {wr:>5.1f}%  ${avg:>6.2f}")


# ═══════════════════════════════════════════
# R4-4: 因子半衰期 — 前半 vs 后半
# ═══════════════════════════════════════════
def r4_4_factor_decay(p):
    p("=" * 80)
    p("R4-4: 策略因子半衰期 — 前半 vs 后半 + 滚动窗口")
    p("=" * 80)

    L4 = get_base()

    # 2-year rolling windows
    windows = []
    for year in range(2015, 2025):
        start = f"{year}-01-01"
        end = f"{year+1}-12-31"
        windows.append((f"{year}-{year+1}", start, end))

    tasks = [(name, L4, 0.30, start, end) for name, start, end in windows]
    results = run_pool(tasks)

    p(f"\n--- 2-Year Rolling Sharpe ---")
    p(f"{'Window':<12s}  {'N':>5s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'$/t':>8s}")
    p("-" * 55)
    sharpes = []
    for r in sorted(results, key=lambda x: x[0]):
        avg = r[3] / r[1] if r[1] > 0 else 0
        sharpes.append(r[2])
        trend = ""
        if len(sharpes) >= 2:
            if sharpes[-1] > sharpes[-2]:
                trend = " ▲"
            elif sharpes[-1] < sharpes[-2]:
                trend = " ▼"
        p(f"  {r[0]:<10s}  {r[1]:>5d}  {r[2]:>7.2f}  {fmt(r[3])}  {r[4]:>5.1f}%  ${avg:>6.2f}{trend}")

    # First half vs second half
    p(f"\n--- First Half (2015-2020) vs Second Half (2020-2026) ---")
    half_tasks = [
        ("First_half", L4, 0.30, "2015-01-01", "2020-06-30"),
        ("Second_half", L4, 0.30, "2020-07-01", "2026-04-10"),
    ]
    half_results = run_pool(half_tasks)
    for r in half_results:
        avg = r[3] / r[1] if r[1] > 0 else 0
        p(f"  {r[0]:<14s}  N={r[1]:>5d}  Sharpe={r[2]:>6.2f}  PnL={fmt(r[3])}  $/t=${avg:.2f}")

    if len(sharpes) >= 4:
        from scipy import stats as scipy_stats
        try:
            slope, _, r_value, p_val, _ = scipy_stats.linregress(range(len(sharpes)), sharpes)
            p(f"\n  Sharpe trend: slope={slope:+.3f}/window, R²={r_value**2:.3f}, p={p_val:.4f}")
            if p_val < 0.05:
                if slope > 0:
                    p("  ✅ 策略 Sharpe 有统计显著的上升趋势（改善中）")
                else:
                    p("  ⚠️ 策略 Sharpe 有统计显著的下降趋势（衰减警告）")
            else:
                p("  策略 Sharpe 无显著时间趋势（稳定）")
        except Exception:
            pass


# ═══════════════════════════════════════════
# R4-5: 滑点敏感性
# ═══════════════════════════════════════════
def r4_5_slippage(p):
    p("=" * 80)
    p("R4-5: 滑点敏感性 — 入场/出场价格偏移")
    p("  回测用固定 spread_cost 模拟，这里测试不同水平")
    p("=" * 80)

    L4 = get_base()

    # Spread from 0.00 to 1.00 in fine steps
    spreads = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
               0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    tasks = [(f"SP_{sp:.2f}", L4, sp, None, None) for sp in spreads]
    results = run_pool(tasks)
    results.sort(key=lambda x: float(x[0].split('_')[1]))

    p(f"\n{'Spread':<10s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'$/t':>8s}  {'MaxDD':>11s}")
    p("-" * 75)
    breakeven_sp = None
    for r in results:
        avg = r[3] / r[1] if r[1] > 0 else 0
        sp = float(r[0].split('_')[1])
        marker = ""
        if sp == 0.30:
            marker = " <-- reference"
        if r[2] < 0 and breakeven_sp is None:
            breakeven_sp = sp
            marker = " <-- BREAKEVEN"
        p(f"  ${sp:<7.2f}  {r[1]:>6d}  {r[2]:>7.2f}  {fmt(r[3])}  {r[4]:>5.1f}%  ${avg:>6.2f}  {fmt(r[6])}{marker}")

    if breakeven_sp:
        p(f"\n  Break-even spread: ~${breakeven_sp:.2f}")
        p(f"  Safety margin from $0.30: ${breakeven_sp - 0.30:.2f} ({(breakeven_sp - 0.30)/0.30*100:.0f}%)")
    else:
        p(f"\n  Strategy profitable at all tested spreads!")

    # Sharpe degradation per $0.10 spread
    sp_vals = [float(r[0].split('_')[1]) for r in results]
    sh_vals = [r[2] for r in results]
    if len(sp_vals) >= 3:
        coeffs = np.polyfit(sp_vals, sh_vals, 1)
        p(f"  Sharpe per $0.10 spread increase: {coeffs[0]*0.10:+.2f}")


# ═══════════════════════════════════════════
# R4-6: Regime 切换期表现
# ═══════════════════════════════════════════
def r4_6_regime_transition(p):
    p("=" * 80)
    p("R4-6: ATR Regime 切换期表现分析")
    p("=" * 80)

    from backtest import DataBundle, run_variant
    import pandas as pd
    L4 = get_base()

    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    s = run_variant(data, "regime_tr", verbose=False, spread_cost=0.30, **L4)
    trades = s.get('_trades', [])

    if not trades:
        p("  No trades!")
        return

    # We can approximate ATR regime from the trade data
    # Since we don't have ATR percentile per trade, analyze by month volatility proxy
    monthly = {}
    for t in trades:
        month = pd.Timestamp(t.entry_time).strftime('%Y-%m')
        if month not in monthly:
            monthly[month] = {'pnls': [], 'n': 0}
        monthly[month]['pnls'].append(t.pnl)
        monthly[month]['n'] += 1

    # Calculate monthly stats
    month_stats = []
    for month in sorted(monthly.keys()):
        d = monthly[month]
        pnl = sum(d['pnls'])
        std = np.std(d['pnls']) if len(d['pnls']) > 1 else 0
        month_stats.append({
            'month': month, 'n': d['n'], 'pnl': pnl,
            'std': std, 'avg': pnl/d['n'] if d['n'] > 0 else 0
        })

    # Classify months by volatility quartile
    stds = [m['std'] for m in month_stats if m['std'] > 0]
    if stds:
        q25, q75 = np.percentile(stds, 25), np.percentile(stds, 75)
        low_vol = [m for m in month_stats if m['std'] <= q25]
        mid_vol = [m for m in month_stats if q25 < m['std'] <= q75]
        high_vol = [m for m in month_stats if m['std'] > q75]

        p(f"\n--- Monthly Volatility Regime ---")
        p(f"  PnL std cutoffs: Q25=${q25:.2f}, Q75=${q75:.2f}")

        for label, group in [("Low vol", low_vol), ("Mid vol", mid_vol), ("High vol", high_vol)]:
            if group:
                total_pnl = sum(m['pnl'] for m in group)
                total_n = sum(m['n'] for m in group)
                neg = sum(1 for m in group if m['pnl'] < 0)
                p(f"  {label}: {len(group)} months, N={total_n}, PnL={fmt(total_pnl)}, "
                  f"$/t=${total_pnl/total_n:.2f}, neg_months={neg}/{len(group)}")

    # Transition months: was previous month different vol regime?
    p(f"\n--- Regime Transition Analysis ---")
    transitions = {'L→H': [], 'H→L': [], 'L→M': [], 'M→H': [], 'H→M': [], 'M→L': [], 'Same': []}
    for i in range(1, len(month_stats)):
        prev = month_stats[i-1]
        curr = month_stats[i]
        if prev['std'] <= q25:
            prev_r = 'L'
        elif prev['std'] > q75:
            prev_r = 'H'
        else:
            prev_r = 'M'

        if curr['std'] <= q25:
            curr_r = 'L'
        elif curr['std'] > q75:
            curr_r = 'H'
        else:
            curr_r = 'M'

        key = f"{prev_r}→{curr_r}" if prev_r != curr_r else "Same"
        transitions[key].append(curr['pnl'])

    p(f"{'Transition':<10s}  {'Count':>6s}  {'Avg PnL':>10s}  {'Total PnL':>11s}")
    p("-" * 42)
    for key in sorted(transitions.keys()):
        vals = transitions[key]
        if vals:
            p(f"  {key:<8s}  {len(vals):>6d}  ${np.mean(vals):>8.2f}  {fmt(sum(vals))}")


# ═══════════════════════════════════════════
# R4-7: Win/Loss 序列分析 (Runs Test)
# ═══════════════════════════════════════════
def r4_7_sequence_analysis(p):
    p("=" * 80)
    p("R4-7: Win/Loss 序列分析")
    p("  检验交易结果是否独立（Runs Test + 自相关）")
    p("=" * 80)

    from backtest import DataBundle, run_variant
    L4 = get_base()

    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    s = run_variant(data, "seq", verbose=False, spread_cost=0.30, **L4)
    trades = s.get('_trades', [])

    if not trades:
        p("  No trades!")
        return

    pnls = np.array([t.pnl for t in trades])
    wins = (pnls > 0).astype(int)
    n = len(wins)
    n_wins = wins.sum()
    n_losses = n - n_wins

    # Count runs
    runs = 1
    for i in range(1, n):
        if wins[i] != wins[i-1]:
            runs += 1

    # Expected runs under independence
    e_runs = 1 + 2 * n_wins * n_losses / n
    var_runs = 2 * n_wins * n_losses * (2 * n_wins * n_losses - n) / (n**2 * (n - 1))
    z_runs = (runs - e_runs) / np.sqrt(var_runs) if var_runs > 0 else 0

    p(f"\n  Total trades: {n}")
    p(f"  Wins: {n_wins} ({n_wins/n*100:.1f}%), Losses: {n_losses} ({n_losses/n*100:.1f}%)")
    p(f"\n--- Runs Test ---")
    p(f"  Observed runs: {runs}")
    p(f"  Expected runs: {e_runs:.1f}")
    p(f"  Z-statistic: {z_runs:.3f}")
    if abs(z_runs) < 1.96:
        p("  ✅ 无显著序列依赖性（交易结果近似独立）")
    elif z_runs < -1.96:
        p("  ⚠️ Runs 偏少 → 存在序列相关（连胜/连亏倾向）")
    else:
        p("  ⚠️ Runs 偏多 → 存在均值回归倾向（胜亏交替）")

    # Autocorrelation of PnL
    p(f"\n--- PnL 自相关 ---")
    mean_pnl = pnls.mean()
    centered = pnls - mean_pnl
    var_pnl = np.var(centered)
    if var_pnl > 0:
        for lag in [1, 2, 3, 5, 10]:
            if lag < n:
                ac = np.sum(centered[:-lag] * centered[lag:]) / ((n - lag) * var_pnl)
                sig = 1.96 / np.sqrt(n)
                significant = "***" if abs(ac) > sig else ""
                p(f"  Lag-{lag}: {ac:>+.4f} {significant}")

    # Conditional expectation: E[PnL_n+1 | PnL_n > 0] vs E[PnL_n+1 | PnL_n < 0]
    p(f"\n--- 条件期望 ---")
    after_win = [pnls[i+1] for i in range(n-1) if pnls[i] > 0]
    after_loss = [pnls[i+1] for i in range(n-1) if pnls[i] < 0]
    if after_win and after_loss:
        p(f"  E[next | prev win]:  ${np.mean(after_win):>+.2f} (N={len(after_win)})")
        p(f"  E[next | prev loss]: ${np.mean(after_loss):>+.2f} (N={len(after_loss)})")
        diff = np.mean(after_win) - np.mean(after_loss)
        p(f"  Difference: ${diff:>+.2f}")
        if abs(diff) < 0.5:
            p("  ✅ 几乎无条件依赖，不支持 martingale/anti-martingale")
        else:
            p(f"  ⚠️ 存在条件依赖 (${diff:+.2f})，值得关注但需更大样本验证")


# ═══════════════════════════════════════════
# R4-8: 策略间相关性
# ═══════════════════════════════════════════
def r4_8_strategy_correlation(p):
    p("=" * 80)
    p("R4-8: 策略间相关性 + 独立贡献分析")
    p("=" * 80)

    from backtest import DataBundle, run_variant
    import pandas as pd
    L4 = get_base()

    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    s = run_variant(data, "corr", verbose=False, spread_cost=0.30, **L4)
    trades = s.get('_trades', [])

    if not trades:
        p("  No trades!")
        return

    # Group by strategy and day
    strat_daily = {}
    for t in trades:
        day = pd.Timestamp(t.exit_time).strftime('%Y-%m-%d')
        strat = t.strategy or 'unknown'
        key = (strat, day)
        if key not in strat_daily:
            strat_daily[key] = 0.0
        strat_daily[key] += t.pnl

    # Build daily PnL per strategy
    all_days = sorted(set(day for _, day in strat_daily.keys()))
    strategies = sorted(set(strat for strat, _ in strat_daily.keys()))

    p(f"\n  Strategies found: {strategies}")
    p(f"  Trading days: {len(all_days)}")

    if len(strategies) < 2:
        p("  Only one strategy, skipping correlation")
        return

    # Create daily PnL arrays
    daily_pnls = {}
    for strat in strategies:
        daily_pnls[strat] = np.array([strat_daily.get((strat, day), 0.0) for day in all_days])

    # Correlation matrix
    p(f"\n--- Daily PnL Correlation ---")
    header = f"{'':>12s}" + "".join(f"  {s:>10s}" for s in strategies)
    p(header)
    for s1 in strategies:
        row = f"  {s1:>10s}"
        for s2 in strategies:
            active_days = (daily_pnls[s1] != 0) | (daily_pnls[s2] != 0)
            if active_days.sum() > 10:
                corr = np.corrcoef(daily_pnls[s1][active_days], daily_pnls[s2][active_days])[0, 1]
            else:
                corr = 0
            row += f"  {corr:>10.3f}"
        p(row)

    # Marginal contribution: full portfolio vs removing each strategy
    p(f"\n--- Marginal Contribution ---")
    total_pnl = sum(strat_daily.values())
    for strat in strategies:
        strat_pnl = sum(v for (s, _), v in strat_daily.items() if s == strat)
        strat_n = sum(1 for t in trades if t.strategy == strat)
        strat_wr = sum(1 for t in trades if t.strategy == strat and t.pnl > 0) / max(1, strat_n) * 100
        pct = strat_pnl / total_pnl * 100 if total_pnl != 0 else 0
        avg = strat_pnl / strat_n if strat_n > 0 else 0
        p(f"  {strat}: N={strat_n}, PnL={fmt(strat_pnl)} ({pct:.1f}%), WR={strat_wr:.1f}%, $/t=${avg:.2f}")


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
        mlog("ROUND 4 EXPERIMENTS")
        mlog(f"CPUs: {mp.cpu_count()}, Workers: {MAX_WORKERS}")
        mlog(f"Started: {datetime.now()}")
        mlog("=" * 60)

        phases = [
            ("r4_0_oos.txt", "R4-0: 样本外 OOS 验证", r4_0_out_of_sample),
            ("r4_1_crisis.txt", "R4-1: 危机期专项", r4_1_crisis_periods),
            ("r4_2_survival.txt", "R4-2: 存活模拟", r4_2_survival_simulation),
            ("r4_3_frequency.txt", "R4-3: 频率分析", r4_3_frequency_analysis),
            ("r4_4_decay.txt", "R4-4: 因子衰减", r4_4_factor_decay),
            ("r4_5_slippage.txt", "R4-5: 滑点敏感性", r4_5_slippage),
            ("r4_6_regime.txt", "R4-6: Regime 切换", r4_6_regime_transition),
            ("r4_7_sequence.txt", "R4-7: 序列分析", r4_7_sequence_analysis),
            ("r4_8_correlation.txt", "R4-8: 策略相关性", r4_8_strategy_correlation),
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
        mlog(f"ALL COMPLETE: {total_elapsed/60:.1f} min ({total_elapsed/3600:.1f} hours)")
        mlog(f"Completed: {datetime.now()}")
        mlog("=" * 60)

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
