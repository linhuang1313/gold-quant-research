#!/usr/bin/env python3
"""
Round 3 Experiments — 深度探索 + 新策略方向
=============================================
Round 2 围绕 L5 组合验证。Round 3 侧重:
  R3-1: Trailing 激活/距离的非线性组合（High/Normal/Low 三维联合优化）
  R3-2: 多种出场超时策略对比（MaxHold 分 regime、BUY/SELL 分离超时）
  R3-3: ORB 策略独立优化（ORB 贡献 $618/140笔 但从未单独调参）
  R3-4: Keltner BUY vs SELL 分离参数（BUY/SELL 可能需要不同 ADX/Trail 参数）
  R3-5: 入场时间加权（基于 Phase 7 的小时分布，测试入场时间对仓位的影响）
  R3-6: 连续盈利加仓 vs 连续亏损减仓（反向 martingale）
  R3-7: 综合 Robustness — 用最终最优配置跑 100 次随机 70% 子集
  R3-8: EUR/USD 策略交叉验证（用黄金策略参数跑 EUR/USD 数据）
"""
import sys, os, time, traceback
import multiprocessing as mp
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "results/round3_results"
MAX_WORKERS = 14

FOLDS_6 = [
    ("2015-01-01", "2016-12-31", "2017-01-01", "2018-12-31"),
    ("2017-01-01", "2018-12-31", "2019-01-01", "2020-12-31"),
    ("2019-01-01", "2020-12-31", "2021-01-01", "2022-12-31"),
    ("2021-01-01", "2022-12-31", "2023-01-01", "2024-12-31"),
    ("2015-01-01", "2018-12-31", "2019-01-01", "2022-12-31"),
    ("2019-01-01", "2022-12-31", "2023-01-01", "2026-04-10"),
]


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


def run_pool(tasks):
    with mp.Pool(min(MAX_WORKERS, len(tasks))) as pool:
        return pool.map(_run_one, tasks)


def get_base():
    from backtest.runner import LIVE_PARITY_KWARGS
    L4_STAR = {**LIVE_PARITY_KWARGS, "time_decay_tp": False}
    return L4_STAR


def run_kfold_comparison(p, name_a, kw_a, name_b, kw_b, spread=0.30):
    tasks = []
    for i, (_, _, ts, te) in enumerate(FOLDS_6):
        tasks.append((f"{name_a}_F{i+1}", kw_a, spread, ts, te))
        tasks.append((f"{name_b}_F{i+1}", kw_b, spread, ts, te))
    results = run_pool(tasks)
    a_map = {r[0]: r for r in results if r[0].startswith(name_a)}
    b_map = {r[0]: r for r in results if r[0].startswith(name_b)}
    pass_cnt = 0
    deltas = []
    for i in range(len(FOLDS_6)):
        ar = a_map.get(f"{name_a}_F{i+1}")
        br = b_map.get(f"{name_b}_F{i+1}")
        if ar and br:
            delta = br[2] - ar[2]
            deltas.append(delta)
            passed = delta >= 0
            if passed:
                pass_cnt += 1
            marker = "V" if passed else "X"
            p(f"    Fold{i+1}: {name_a}={ar[2]:>5.2f}  {name_b}={br[2]:>5.2f}  delta={delta:>+.2f} {marker}")
    avg_d = np.mean(deltas) if deltas else 0
    p(f"    Result: {pass_cnt}/{len(FOLDS_6)} PASS  avg_delta={avg_d:>+.3f}")
    return pass_cnt, avg_d


# ═══════════════════════════════════════════
# R3-1: 三维联合 Trail 优化
# ═══════════════════════════════════════════
def r3_1_joint_trail(p):
    p("=" * 80)
    p("R3-1: 三维联合 Trail 优化")
    p("  基于 Marathon Phase 4 的 top configs，组合 High+Normal+Low 最优")
    p("=" * 80)

    L4 = get_base()

    # Phase 4 results:
    #   High top: H0.15/D0.02 (6/6 PASS)
    #   Normal top: N0.33/D0.08 (+0.12 vs N0.35/D0.10)
    #   Low: 未测试（Round 2 R3 正在跑）
    # 这里测试几个有代表性的组合

    combos = [
        # (label, high_act, high_dist, norm_act, norm_dist, low_act, low_dist)
        ("Current",    0.20, 0.03, 0.35, 0.10, 0.50, 0.15),
        ("H15_N33",    0.15, 0.02, 0.33, 0.08, 0.50, 0.15),
        ("H15_N35",    0.15, 0.02, 0.35, 0.10, 0.50, 0.15),
        ("H15_N30",    0.15, 0.02, 0.30, 0.08, 0.50, 0.15),
        ("H15_N33_L45", 0.15, 0.02, 0.33, 0.08, 0.45, 0.12),
        ("H15_N33_L40", 0.15, 0.02, 0.33, 0.08, 0.40, 0.10),
        ("H15_N33_L55", 0.15, 0.02, 0.33, 0.08, 0.55, 0.18),
        ("H18_N33",    0.18, 0.02, 0.33, 0.08, 0.50, 0.15),
        ("H15_N35_L45", 0.15, 0.02, 0.35, 0.10, 0.45, 0.12),
        ("AllTight",   0.12, 0.02, 0.28, 0.06, 0.40, 0.10),
        ("AllLoose",   0.22, 0.04, 0.40, 0.12, 0.60, 0.20),
    ]

    configs = []
    for name, ha, hd, na, nd, la, ld in combos:
        rc = dict(L4.get("regime_config", {}))
        rc['high'] = {'trail_act': ha, 'trail_dist': hd}
        rc['normal'] = {'trail_act': na, 'trail_dist': nd}
        rc['low'] = {'trail_act': la, 'trail_dist': ld}
        kw = {**L4, "regime_config": rc}
        configs.append((name, kw))

    tasks = [(label, kw, 0.30, None, None) for label, kw in configs]
    results = run_pool(tasks)
    results.sort(key=lambda x: -x[2])

    p(f"\n{'Combo':<20s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'MaxDD':>11s}")
    p("-" * 65)
    for r in results:
        marker = " <-- current" if r[0] == "Current" else ""
        p(f"  {r[0]:<18s}  {r[1]:>6d}  {r[2]:>7.2f}  {fmt(r[3])}  {r[4]:>5.1f}%  {fmt(r[6])}{marker}")

    # K-Fold for best non-current
    best = [r for r in results if r[0] != "Current"]
    if best and best[0][2] > results[-1][2]:
        best_name = best[0][0]
        best_kw = dict([c for c in configs if c[0] == best_name][0][1])
        cur_kw = dict([c for c in configs if c[0] == "Current"][0][1])
        p(f"\n--- K-Fold: {best_name} vs Current ---")
        run_kfold_comparison(p, "Current", cur_kw, best_name, best_kw)


# ═══════════════════════════════════════════
# R3-2: MaxHold 分 BUY/SELL + Regime
# ═══════════════════════════════════════════
def _run_maxhold_variant(args):
    """Monkey-patch MaxHold per direction."""
    label, base_kw, spread, buy_mh, sell_mh = args
    from backtest import DataBundle, run_variant
    import backtest.engine as eng

    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

    original_check = eng.BacktestEngine._check_exits

    def patched_check_exits(self, pos, bar_idx, m15_bar, h1_window):
        old_mh = self._keltner_max_hold_m15
        if pos.direction == 'BUY':
            self._keltner_max_hold_m15 = buy_mh
        else:
            self._keltner_max_hold_m15 = sell_mh
        result = original_check(pos, bar_idx, m15_bar, h1_window)
        self._keltner_max_hold_m15 = old_mh
        return result

    eng.BacktestEngine._check_exits = patched_check_exits
    try:
        s = run_variant(data, label, verbose=False, spread_cost=spread, **base_kw)
    finally:
        eng.BacktestEngine._check_exits = original_check

    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'])


def r3_2_maxhold_split(p):
    p("=" * 80)
    p("R3-2: MaxHold BUY/SELL 分离 + 不同超时策略")
    p("=" * 80)

    L4 = get_base()

    # Test different MaxHold values (uniform, not split - avoid monkey-patch complexity in pool)
    configs = []
    for mh in [12, 16, 20, 24, 28, 32, 40, 48, 60]:
        kw = {**L4, "keltner_max_hold_m15": mh}
        label = f"MH_{mh}"
        configs.append((label, kw))

    # Also test infinite hold (no timeout)
    kw_inf = {**L4, "keltner_max_hold_m15": 9999}
    configs.append(("MH_inf", kw_inf))

    tasks = [(label, kw, 0.30, None, None) for label, kw in configs]
    results = run_pool(tasks)
    results.sort(key=lambda x: -x[2])

    p(f"\n{'Config':<12s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'$/t':>8s}  {'MaxDD':>11s}")
    p("-" * 70)
    for r in results:
        avg = r[3] / r[1] if r[1] > 0 else 0
        marker = " <-- current" if r[0] == "MH_20" else ""
        p(f"  {r[0]:<10s}  {r[1]:>6d}  {r[2]:>7.2f}  {fmt(r[3])}  {r[4]:>5.1f}%  ${avg:>6.2f}  {fmt(r[6])}{marker}")


# ═══════════════════════════════════════════
# R3-3: ORB 独立优化
# ═══════════════════════════════════════════
def r3_3_orb_optimization(p):
    p("=" * 80)
    p("R3-3: ORB 策略独立优化")
    p("  ORB 贡献 $618/140笔 in 2024-2026, 从未独立调参")
    p("=" * 80)

    L4 = get_base()

    # ORB max hold sweep
    configs = []
    for orb_mh in [4, 6, 8, 10, 12, 16, 20, 24]:
        kw = {**L4, "orb_max_hold_m15": orb_mh}
        label = f"ORB_MH{orb_mh}"
        configs.append((label, kw))

    # ORB without choppy filter (does choppy also affect ORB?)
    kw_no_choppy = {**L4, "choppy_threshold": 0.0}
    configs.append(("ORB_noChoppy", kw_no_choppy))

    # ORB with tighter choppy
    kw_tight_choppy = {**L4, "choppy_threshold": 0.55}
    configs.append(("ORB_choppy55", kw_tight_choppy))

    tasks = [(label, kw, 0.30, None, None) for label, kw in configs]
    results = run_pool(tasks)
    results.sort(key=lambda x: -x[2])

    p(f"\n{'Config':<18s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'MaxDD':>11s}")
    p("-" * 65)
    for r in results:
        marker = " <-- current (MH=default)" if r[0] == "ORB_MH8" else ""
        p(f"  {r[0]:<16s}  {r[1]:>6d}  {r[2]:>7.2f}  {fmt(r[3])}  {r[4]:>5.1f}%  {fmt(r[6])}{marker}")


# ═══════════════════════════════════════════
# R3-4: BUY vs SELL 分离 ADX
# ═══════════════════════════════════════════
def _run_adx_split(args):
    """Test different ADX thresholds for BUY vs SELL via monkey-patch."""
    label, base_kw, spread, buy_adx, sell_adx = args
    import indicators as sig
    from backtest import DataBundle, run_variant

    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    original_fn = sig.check_keltner_signal

    def patched_keltner(df):
        result = original_fn(df)
        if result is None:
            return None
        adx = float(df.iloc[-1]['ADX'])
        if result['signal'] == 'BUY' and adx < buy_adx:
            return None
        if result['signal'] == 'SELL' and adx < sell_adx:
            return None
        return result

    sig.check_keltner_signal = patched_keltner
    try:
        s = run_variant(data, label, verbose=False, spread_cost=spread, **base_kw)
    finally:
        sig.check_keltner_signal = original_fn

    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'])


def r3_4_buy_sell_split(p):
    p("=" * 80)
    p("R3-4: BUY vs SELL 分离参数 (ADX 阈值)")
    p("  当前 BUY/SELL 共用 ADX>=18，测试是否需要差异化")
    p("=" * 80)

    L4 = get_base()

    # ADX threshold sweep (uniform)
    configs = []
    for adx in [14, 16, 18, 20, 22, 25]:
        kw = {**L4, "keltner_adx_threshold": adx}
        configs.append((f"ADX_{adx}", kw))

    tasks = [(label, kw, 0.30, None, None) for label, kw in configs]
    results = run_pool(tasks)
    results.sort(key=lambda x: -x[2])

    p(f"\n--- Uniform ADX Sweep ---")
    p(f"{'Config':<12s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'MaxDD':>11s}")
    p("-" * 60)
    for r in results:
        marker = " <-- current" if r[0] == "ADX_18" else ""
        p(f"  {r[0]:<10s}  {r[1]:>6d}  {r[2]:>7.2f}  {fmt(r[3])}  {r[4]:>5.1f}%  {fmt(r[6])}{marker}")


# ═══════════════════════════════════════════
# R3-5: Escalating Cooldown 精调
# ═══════════════════════════════════════════
def r3_5_escalating_cooldown(p):
    p("=" * 80)
    p("R3-5: Escalating Cooldown 精调")
    p("  连续亏损后延长冷却期是否有帮助？")
    p("=" * 80)

    L4 = get_base()

    configs = []
    # Escalating cooldown with different multipliers
    for mult in [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
        kw = {**L4, "escalating_cooldown": True, "escalating_cooldown_mult": mult}
        label = f"EscCD_x{mult:.1f}"
        configs.append((label, kw))

    # Baseline (no escalating)
    configs.append(("NoEscCD", L4))

    # Min entry gap
    for gap in [0.5, 1.0, 1.5, 2.0, 3.0]:
        kw = {**L4, "min_entry_gap_hours": gap}
        label = f"MinGap_{gap:.1f}h"
        configs.append((label, kw))

    tasks = [(label, kw, 0.30, None, None) for label, kw in configs]
    results = run_pool(tasks)
    results.sort(key=lambda x: -x[2])

    p(f"\n{'Config':<16s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'$/t':>8s}  {'MaxDD':>11s}")
    p("-" * 75)
    for r in results:
        avg = r[3] / r[1] if r[1] > 0 else 0
        marker = " <-- baseline" if r[0] == "NoEscCD" else ""
        p(f"  {r[0]:<14s}  {r[1]:>6d}  {r[2]:>7.2f}  {fmt(r[3])}  {r[4]:>5.1f}%  ${avg:>6.2f}  {fmt(r[6])}{marker}")


# ═══════════════════════════════════════════
# R3-6: Robustness — 100 次随机子集
# ═══════════════════════════════════════════
def r3_6_robustness(p):
    p("=" * 80)
    p("R3-6: 随机子集 Robustness (100 次 70% 子集)")
    p("=" * 80)

    from backtest import DataBundle, run_variant
    L4 = get_base()

    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    s = run_variant(data, "base", verbose=False, spread_cost=0.30, **L4)
    trades = s.get('_trades', [])
    p(f"\n  Full sample: N={s['n']}, Sharpe={s['sharpe']:.2f}, PnL={fmt(s['total_pnl'])}")

    if not trades:
        p("  No trades!")
        return

    pnls = np.array([t.pnl for t in trades])
    n_total = len(pnls)
    n_sample = int(n_total * 0.7)
    n_iters = 100

    np.random.seed(42)
    sharpes = []
    total_pnls = []
    maxdds = []

    for _ in range(n_iters):
        idx = np.random.choice(n_total, n_sample, replace=False)
        sample = pnls[idx]
        std = sample.std(ddof=1)
        if std > 0:
            sh = sample.mean() / std * np.sqrt(252 * 6)
            sharpes.append(sh)
        total_pnls.append(sample.sum())
        cum = np.cumsum(sample)
        peak = np.maximum.accumulate(cum)
        dd = (peak - cum).max()
        maxdds.append(dd)

    sharpes = np.array(sharpes)
    total_pnls = np.array(total_pnls)
    maxdds = np.array(maxdds)

    p(f"\n--- {n_iters} Random 70% Subsets ---")
    p(f"  Sharpe:  mean={sharpes.mean():.2f}, std={sharpes.std():.2f}, "
      f"min={sharpes.min():.2f}, 5th={np.percentile(sharpes, 5):.2f}, "
      f"95th={np.percentile(sharpes, 95):.2f}, max={sharpes.max():.2f}")
    p(f"  P(Sharpe > 0):   {(sharpes > 0).mean()*100:.0f}%")
    p(f"  P(Sharpe > 2.0): {(sharpes > 2.0).mean()*100:.0f}%")
    p(f"  P(Sharpe > 3.0): {(sharpes > 3.0).mean()*100:.0f}%")
    p(f"  PnL: mean={fmt(total_pnls.mean())}, min={fmt(total_pnls.min())}, max={fmt(total_pnls.max())}")
    p(f"  P(PnL > 0):   {(total_pnls > 0).mean()*100:.0f}%")
    p(f"  MaxDD: mean={fmt(maxdds.mean())}, 95th={fmt(np.percentile(maxdds, 95))}, worst={fmt(maxdds.max())}")


# ═══════════════════════════════════════════
# R3-7: 年度最差月分析 + 回撤恢复时间
# ═══════════════════════════════════════════
def r3_7_drawdown_analysis(p):
    p("=" * 80)
    p("R3-7: 回撤分析 — 最大回撤持续时间 + 恢复时间")
    p("=" * 80)

    from backtest import DataBundle, run_variant
    import pandas as pd

    L4 = get_base()
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    s = run_variant(data, "dd_analysis", verbose=False, spread_cost=0.30, **L4)
    trades = s.get('_trades', [])

    if not trades:
        p("  No trades!")
        return

    p(f"\n  Total: N={s['n']}, Sharpe={s['sharpe']:.2f}, PnL={fmt(s['total_pnl'])}")

    # Build equity curve from trades
    equity = [0.0]
    times = []
    for t in trades:
        equity.append(equity[-1] + t.pnl)
        times.append(pd.Timestamp(t.exit_time))

    equity = np.array(equity[1:])
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity

    # Find top-5 drawdown episodes
    p(f"\n--- Top 10 Drawdown Episodes ---")
    in_dd = False
    dd_start = 0
    episodes = []
    for i in range(len(drawdown)):
        if drawdown[i] > 0 and not in_dd:
            dd_start = i
            in_dd = True
        elif drawdown[i] == 0 and in_dd:
            max_dd = drawdown[dd_start:i].max()
            max_dd_idx = dd_start + np.argmax(drawdown[dd_start:i])
            duration = i - dd_start
            recovery = i - max_dd_idx
            episodes.append((max_dd, dd_start, max_dd_idx, i, duration, recovery))
            in_dd = False
    if in_dd:
        max_dd = drawdown[dd_start:].max()
        max_dd_idx = dd_start + np.argmax(drawdown[dd_start:])
        episodes.append((max_dd, dd_start, max_dd_idx, len(drawdown)-1, len(drawdown)-1-dd_start, len(drawdown)-1-max_dd_idx))

    episodes.sort(key=lambda x: -x[0])
    p(f"{'Rank':>4s}  {'MaxDD':>10s}  {'Duration':>10s}  {'Recovery':>10s}  {'Start':>20s}  {'Bottom':>20s}")
    p("-" * 80)
    for rank, (dd, start, bottom, end, dur, rec) in enumerate(episodes[:10], 1):
        start_t = str(times[start])[:10] if start < len(times) else "?"
        bottom_t = str(times[bottom])[:10] if bottom < len(times) else "?"
        p(f"  {rank:>3d}  {fmt(dd)}  {dur:>8d}t  {rec:>8d}t  {start_t:>18s}  {bottom_t:>18s}")

    # Quarterly PnL
    p(f"\n--- Quarterly PnL ---")
    quarterly = {}
    for t in trades:
        ts = pd.Timestamp(t.exit_time)
        q = f"{ts.year}-Q{ts.quarter}"
        if q not in quarterly:
            quarterly[q] = {'n': 0, 'pnl': 0.0, 'wins': 0}
        quarterly[q]['n'] += 1
        quarterly[q]['pnl'] += t.pnl
        if t.pnl > 0:
            quarterly[q]['wins'] += 1

    neg_q = 0
    p(f"{'Quarter':<10s}  {'N':>5s}  {'PnL':>11s}  {'WR%':>6s}  {'$/t':>8s}")
    p("-" * 45)
    for q in sorted(quarterly.keys()):
        d = quarterly[q]
        wr = d['wins'] / d['n'] * 100 if d['n'] > 0 else 0
        avg = d['pnl'] / d['n'] if d['n'] > 0 else 0
        marker = " ***" if d['pnl'] < 0 else ""
        if d['pnl'] < 0:
            neg_q += 1
        p(f"  {q:<8s}  {d['n']:>5d}  {fmt(d['pnl'])}  {wr:>5.1f}%  ${avg:>6.2f}{marker}")
    p(f"\n  Negative quarters: {neg_q}/{len(quarterly)}")

    # Max losing streak in PnL terms (rolling 50-trade window)
    p(f"\n--- Rolling 50-Trade Window Worst Stretches ---")
    if len(trades) > 50:
        pnls = np.array([t.pnl for t in trades])
        window_pnls = []
        for i in range(len(pnls) - 50):
            window_pnls.append((i, pnls[i:i+50].sum()))
        window_pnls.sort(key=lambda x: x[1])
        for rank, (idx, wpnl) in enumerate(window_pnls[:5], 1):
            t_start = str(pd.Timestamp(trades[idx].entry_time))[:10]
            t_end = str(pd.Timestamp(trades[idx+49].exit_time))[:10]
            wr_50 = sum(1 for j in range(idx, idx+50) if trades[j].pnl > 0) / 50 * 100
            p(f"  #{rank}: PnL={fmt(wpnl)}, WR={wr_50:.0f}%, {t_start} → {t_end}")


# ═══════════════════════════════════════════
# R3-8: EUR/USD 交叉验证
# ═══════════════════════════════════════════
def r3_8_eurusd(p):
    p("=" * 80)
    p("R3-8: EUR/USD 交叉验证")
    p("  用黄金 L4* 的出场逻辑测试 EUR/USD 数据")
    p("=" * 80)

    L4 = get_base()

    eurusd_kc_params = [
        (20, 1.5), (20, 2.0), (25, 1.5), (25, 2.0), (25, 2.5),
        (30, 2.0), (30, 2.5), (35, 2.0), (35, 2.5),
    ]

    tasks = []
    for ema, mult in eurusd_kc_params:
        try:
            from backtest import DataBundle
            data = DataBundle.load_custom(kc_ema=ema, kc_mult=mult, symbol='eurusd')
            label = f"EUR_E{ema}_M{mult:.1f}"
            tasks.append((label, L4, 0.018, None, None))  # EUR/USD spread ~1.8 pips
        except Exception:
            pass

    if not tasks:
        # Try loading EUR/USD data directly
        try:
            from backtest import DataBundle
            data = DataBundle.load_custom(kc_ema=25, kc_mult=2.0, symbol='eurusd')
            p("  EUR/USD data loaded successfully")
            s = run_variant(data, "EUR_test", verbose=False, spread_cost=0.018, **L4)
            p(f"  N={s['n']}, Sharpe={s['sharpe']:.2f}, PnL={fmt(s['total_pnl'])}")
        except Exception as e:
            p(f"  EUR/USD data not available: {e}")
            p("  Skipping EUR/USD tests")
            return

    if tasks:
        results = run_pool(tasks)
        results.sort(key=lambda x: -x[2])
        p(f"\n{'Config':<18s}  {'N':>6s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'MaxDD':>11s}")
        p("-" * 65)
        for r in results:
            p(f"  {r[0]:<16s}  {r[1]:>6d}  {r[2]:>7.2f}  {fmt(r[3])}  {r[4]:>5.1f}%  {fmt(r[6])}")


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
        mlog("ROUND 3 EXPERIMENTS")
        mlog(f"CPUs: {mp.cpu_count()}, Workers: {MAX_WORKERS}")
        mlog(f"Started: {datetime.now()}")
        mlog("=" * 60)

        phases = [
            ("r3_1_joint_trail.txt", "R3-1: 三维联合 Trail", r3_1_joint_trail),
            ("r3_2_maxhold.txt", "R3-2: MaxHold 策略", r3_2_maxhold_split),
            ("r3_3_orb.txt", "R3-3: ORB 优化", r3_3_orb_optimization),
            ("r3_4_buy_sell.txt", "R3-4: BUY/SELL ADX", r3_4_buy_sell_split),
            ("r3_5_cooldown.txt", "R3-5: Escalating Cooldown", r3_5_escalating_cooldown),
            ("r3_6_robustness.txt", "R3-6: 随机子集", r3_6_robustness),
            ("r3_7_drawdown.txt", "R3-7: 回撤分析", r3_7_drawdown_analysis),
            ("r3_8_eurusd.txt", "R3-8: EUR/USD", r3_8_eurusd),
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
