#!/usr/bin/env python3
"""
Round 19 — "Signal Quality & Regime Enhancement"
=================================================
目标: 从信号质量和市场状态识别的角度探索收益增强
前提: R13 证明 KC 参数/trailing 已接近极限; R14 确认 TATrail/Gap 有效
预计总耗时: ~16 小时

=== Phase A: 多时间框架方向过滤 (~3h) ===
R19-A1: H1 EMA100 斜率过滤 — 不同回看窗口 (3/5/8/12 H1 bars)
         只做与 EMA100 方向一致的交易 vs baseline
R19-A2: H4 趋势方向过滤 — 需 H4 EMA 同向
R19-A3: 最优过滤器 K-Fold 6折 + 双 Spread 验证
R19-A4: 逐年对比最优 vs baseline

=== Phase B: 入场时段质量分析 (~3h) ===
R19-B1: 按 UTC 小时统计所有交易的 avg_pnl, WR, Sharpe
R19-B2: 最差时段屏蔽 — 单小时/组合小时屏蔽效果
R19-B3: 最优 session 过滤 K-Fold 验证
R19-B4: 亚洲/欧洲/美洲 三个大区的性能对比

=== Phase C: 波动率 Regime 细分 (~3h) ===
R19-C1: 5 档 regime (0-20/20-40/40-60/60-80/80-100 atr pct) 各自的交易表现
R19-C2: 5 档独立优化 trailing 参数 vs 当前 3 档
R19-C3: 4 档 (0-25/25-50/50-75/75-100) 方案对比
R19-C4: 最优细分方案 K-Fold 验证

=== Phase D: 持仓期动态管理 (~3h) ===
R19-D1: ADX 骤降提前退出 — 持仓中 ADX < threshold 时立即平仓
R19-D2: 不同 max_hold 对比 (15/18/20/24/30 M15 bars)
R19-D3: 方向反转早退 — EMA100 斜率反转时提前退出
R19-D4: 最优退出规则 K-Fold 验证

=== Phase E: 组合验证 (~4h) ===
R19-E1: 通过验证的改进叠加 (L8 候选)
R19-E2: Monte Carlo 100x ±15% 参数扰动
R19-E3: Purged Walk-Forward 5-Fold
R19-E4: 逐年稳定性 + 双 Spread
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

OUTPUT_DIR = "results/round19_results"
MAX_WORKERS = 10


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
            s.get('year_pnl', {}))


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
    trade_data = []
    for t in trades:
        trade_data.append((
            t.pnl, t.exit_reason, t.bars_held, t.strategy,
            str(t.entry_time), t.direction, t.lots, t.entry_price,
        ))
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'], s.get('max_dd_pct', 0),
            trade_data,
            s.get('skipped_ema_slope', 0), s.get('skipped_session', 0))


def run_pool(tasks, func=_run_one):
    n = min(MAX_WORKERS, len(tasks))
    if n == 0:
        return []
    with mp.Pool(n) as pool:
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
    """L7 = L6 + TATrail + Gap1h"""
    kw = get_l6()
    kw['time_adaptive_trail'] = True
    kw['time_adaptive_trail_start'] = 2
    kw['time_adaptive_trail_decay'] = 0.75
    kw['time_adaptive_trail_floor'] = 0.003
    kw['min_entry_gap_hours'] = 1.0
    return kw


FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-10"),
]

YEARS = [(str(y), f"{y}-01-01", f"{y+1}-01-01") for y in range(2015, 2026)]


def write_table(f, results, extra_cols=""):
    cols = f"{'Label':<45} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} {'MaxDD':>10}"
    if extra_cols:
        cols += f" {extra_cols}"
    f.write(cols + "\n")
    f.write("-" * len(cols) + "\n")
    for r in results:
        line = f"{r[0]:<45} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1f}% {fmt(r[6]):>10}"
        f.write(line + "\n")


# ═══════════════════════════════════════════════════════════════
# PHASE A: Multi-Timeframe Direction Filter
# ═══════════════════════════════════════════════════════════════

def run_r19_a1(out):
    """R19-A1: EMA100 Slope Filter — different lookback windows."""
    print("\n" + "=" * 70)
    print("R19-A1: EMA100 Slope Direction Filter")
    print("=" * 70)

    slope_bars = [0, 3, 5, 8, 12, 16, 20]
    tasks = []
    for sb in slope_bars:
        for sp in [0.30, 0.50]:
            kw = get_l6()
            if sb > 0:
                kw['block_buy_ema_slope'] = sb
            label = f"Slope{sb}_sp{sp}"
            tasks.append((label, kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R19-A1_ema_slope.txt", 'w') as f:
        f.write("R19-A1: EMA100 Slope Direction Filter\n" + "=" * 80 + "\n")
        f.write("block_buy_ema_slope=N: block BUY when EMA100(now) < EMA100(N bars ago)\n\n")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread ${sp} ---\n")
            sub = sorted([r for r in results if f"sp{sp}" in r[0]],
                         key=lambda x: -x[2])
            write_table(f, sub)
            base_sharpe = next((r[2] for r in sub if "Slope0" in r[0]), 0)
            f.write(f"\nBaseline Sharpe: {base_sharpe:.2f}\n")
            for r in sub:
                if "Slope0" not in r[0]:
                    f.write(f"  {r[0]}: delta={r[2] - base_sharpe:+.2f}\n")

    print(f"  Saved R19-A1 ({len(results)} variants)")


def run_r19_a2(out):
    """R19-A2: Bidirectional EMA slope — block both BUY and SELL against trend."""
    print("\n" + "=" * 70)
    print("R19-A2: Bidirectional EMA Slope + Session Combo")
    print("=" * 70)

    configs = [
        ("Baseline", {}),
        ("Slope5_buy_only", {"block_buy_ema_slope": 5}),
        ("Slope8_buy_only", {"block_buy_ema_slope": 8}),
        ("Slope5_sessions_7_20", {"block_buy_ema_slope": 5,
                                   "h1_allowed_sessions": list(range(7, 21))}),
        ("Slope8_sessions_8_19", {"block_buy_ema_slope": 8,
                                   "h1_allowed_sessions": list(range(8, 20))}),
        ("Sessions_7_20", {"h1_allowed_sessions": list(range(7, 21))}),
        ("Sessions_8_19", {"h1_allowed_sessions": list(range(8, 20))}),
        ("Sessions_6_21", {"h1_allowed_sessions": list(range(6, 22))}),
    ]

    tasks = []
    for name, extra in configs:
        for sp in [0.30, 0.50]:
            kw = get_l6()
            kw.update(extra)
            tasks.append((f"{name}_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R19-A2_slope_session_combo.txt", 'w') as f:
        f.write("R19-A2: EMA Slope + Session Combinations\n" + "=" * 80 + "\n\n")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread ${sp} ---\n")
            sub = sorted([r for r in results if f"sp{sp}" in r[0]],
                         key=lambda x: -x[2])
            write_table(f, sub)

    print(f"  Saved R19-A2 ({len(results)} variants)")


def run_r19_a3(out):
    """R19-A3: K-Fold validation for best A1/A2 filters."""
    print("\n" + "=" * 70)
    print("R19-A3: K-Fold Validation for Best Filters")
    print("=" * 70)

    best_configs = [
        ("Baseline", {}),
        ("Slope5", {"block_buy_ema_slope": 5}),
        ("Slope8", {"block_buy_ema_slope": 8}),
        ("Sessions_7_20", {"h1_allowed_sessions": list(range(7, 21))}),
        ("Slope5_S7_20", {"block_buy_ema_slope": 5,
                          "h1_allowed_sessions": list(range(7, 21))}),
    ]

    tasks = []
    for sp in [0.30, 0.50]:
        for config_name, extra_kw in best_configs:
            for fold_name, start, end in FOLDS:
                kw = get_l6()
                kw.update(extra_kw)
                tasks.append((f"{config_name}_{fold_name}_sp{sp}", kw, sp, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R19-A3_kfold.txt", 'w') as f:
        f.write("R19-A3: K-Fold Validation\n" + "=" * 80 + "\n\n")
        for sp in [0.30, 0.50]:
            f.write(f"\n=== Spread ${sp} ===\n")
            for config_name, _ in best_configs:
                if config_name == "Baseline":
                    continue
                base_folds = {}
                test_folds = {}
                for r in results:
                    if f"sp{sp}" not in r[0]:
                        continue
                    for fold_name, _, _ in FOLDS:
                        if fold_name in r[0]:
                            if f"Baseline_{fold_name}" in r[0]:
                                base_folds[fold_name] = r[2]
                            elif f"{config_name}_{fold_name}" in r[0]:
                                test_folds[fold_name] = r[2]

                wins = 0
                f.write(f"\n{config_name} sp${sp}:\n")
                for fold_name, _, _ in FOLDS:
                    bs = base_folds.get(fold_name, 0)
                    ts = test_folds.get(fold_name, 0)
                    d = ts - bs
                    won = d > 0
                    if won:
                        wins += 1
                    f.write(f"  {fold_name}: Base={bs:.2f} Test={ts:.2f} "
                            f"delta={d:+.2f} {'V' if won else 'X'}\n")
                result = "PASS" if wins >= 5 else "FAIL"
                f.write(f"  Result: {wins}/6 {result}\n")

    print(f"  Saved R19-A3 ({len(results)} variants)")


def run_r19_a4(out):
    """R19-A4: Year-by-year stability for best filter."""
    print("\n" + "=" * 70)
    print("R19-A4: Year-by-Year Stability")
    print("=" * 70)

    configs = [
        ("L6_Baseline", get_l6()),
        ("L6_Slope5", {**get_l6(), "block_buy_ema_slope": 5}),
        ("L6_Slope8", {**get_l6(), "block_buy_ema_slope": 8}),
        ("L6_S7_20", {**get_l6(), "h1_allowed_sessions": list(range(7, 21))}),
    ]

    tasks = []
    for name, kw in configs:
        for sp in [0.30, 0.50]:
            for yr_name, start, end in YEARS:
                tasks.append((f"{name}_{yr_name}_sp{sp}", {**kw}, sp, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R19-A4_yearly.txt", 'w') as f:
        f.write("R19-A4: Year-by-Year Comparison\n" + "=" * 80 + "\n\n")
        for sp in [0.30, 0.50]:
            f.write(f"\n=== Spread ${sp} ===\n")
            f.write(f"{'Year':<6}")
            for name, _ in configs:
                f.write(f"  {name:>20s}")
            f.write("\n" + "-" * (6 + 22 * len(configs)) + "\n")

            for yr_name, _, _ in YEARS:
                f.write(f"{yr_name:<6}")
                for name, _ in configs:
                    match = [r for r in results
                             if r[0] == f"{name}_{yr_name}_sp{sp}"]
                    if match:
                        f.write(f"  {match[0][2]:>20.2f}")
                    else:
                        f.write(f"  {'N/A':>20s}")
                f.write("\n")

    print(f"  Saved R19-A4 ({len(results)} variants)")


# ═══════════════════════════════════════════════════════════════
# PHASE B: Entry Session Quality
# ═══════════════════════════════════════════════════════════════

def run_r19_b1(out):
    """R19-B1: Per-hour trade quality analysis."""
    print("\n" + "=" * 70)
    print("R19-B1: Per-Hour Trade Quality Analysis")
    print("=" * 70)

    import pandas as pd

    tasks = [("B1_full", get_l6(), 0.30, None, None)]
    results = run_pool(tasks, func=_run_one_trades)

    if not results:
        print("  No results!")
        return

    trades_data = results[0][8]
    by_hour = defaultdict(list)
    by_session = defaultdict(list)

    for pnl, reason, bars, strat, etime, direction, lots, eprice in trades_data:
        hour = pd.Timestamp(etime).hour
        by_hour[hour].append({
            'pnl': pnl, 'reason': reason, 'bars': bars,
            'strategy': strat, 'direction': direction,
        })
        if 0 <= hour < 7:
            session = "Asia"
        elif 7 <= hour < 14:
            session = "London"
        elif 14 <= hour < 21:
            session = "NY"
        else:
            session = "Late"
        by_session[session].append(pnl)

    with open(f"{out}/R19-B1_hourly_quality.txt", 'w') as f:
        f.write("R19-B1: Per-Hour Trade Quality\n" + "=" * 80 + "\n\n")
        f.write(f"Total trades: {len(trades_data)}\n\n")

        f.write(f"{'Hour':>6s}  {'N':>5s}  {'WR%':>6s}  {'Avg$':>8s}  {'Total$':>10s}  {'Sharpe':>7s}  {'Top Exit':>15s}\n")
        f.write("-" * 65 + "\n")

        for h in range(24):
            trades = by_hour.get(h, [])
            if not trades:
                continue
            pnls = [t['pnl'] for t in trades]
            n = len(pnls)
            wr = 100.0 * sum(1 for p in pnls if p > 0) / n
            avg = np.mean(pnls)
            total = sum(pnls)
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0
            exits = Counter(t['reason'] for t in trades)
            top_exit = exits.most_common(1)[0][0] if exits else "N/A"
            f.write(f"{h:>4d}h  {n:>5d}  {wr:>5.1f}%  ${avg:>7.2f}  {fmt(total)}  "
                    f"{sharpe:>7.2f}  {top_exit:>15s}\n")

        f.write(f"\n\n{'Session':>8s}  {'N':>6s}  {'WR%':>6s}  {'Avg$':>8s}  {'Total$':>10s}  {'Sharpe':>7s}\n")
        f.write("-" * 55 + "\n")
        for sess in ["Asia", "London", "NY", "Late"]:
            pnls = by_session.get(sess, [])
            if not pnls:
                continue
            n = len(pnls)
            wr = 100.0 * sum(1 for p in pnls if p > 0) / n
            avg = np.mean(pnls)
            total = sum(pnls)
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0
            f.write(f"{sess:>8s}  {n:>6d}  {wr:>5.1f}%  ${avg:>7.2f}  {fmt(total)}  {sharpe:>7.2f}\n")

    print(f"  Saved R19-B1")


def run_r19_b2(out):
    """R19-B2: Session filtering — block worst hours."""
    print("\n" + "=" * 70)
    print("R19-B2: Session Hour Filtering")
    print("=" * 70)

    session_configs = [
        ("All_hours", None),
        ("Block_0_6", list(range(7, 24))),
        ("Block_0_7", list(range(8, 24))),
        ("Block_21_6", list(range(7, 21))),
        ("Block_22_6", list(range(7, 22))),
        ("Block_21_7", list(range(8, 21))),
        ("London_NY_8_20", list(range(8, 21))),
        ("London_NY_7_21", list(range(7, 22))),
        ("London_only_7_14", list(range(7, 15))),
        ("NY_only_14_21", list(range(14, 22))),
    ]

    tasks = []
    for name, hours in session_configs:
        for sp in [0.30, 0.50]:
            kw = get_l6()
            if hours is not None:
                kw['h1_allowed_sessions'] = hours
            tasks.append((f"{name}_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R19-B2_session_filter.txt", 'w') as f:
        f.write("R19-B2: Session Hour Filtering\n" + "=" * 80 + "\n\n")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread ${sp} ---\n")
            sub = sorted([r for r in results if f"sp{sp}" in r[0]],
                         key=lambda x: -x[2])
            write_table(f, sub)

    print(f"  Saved R19-B2 ({len(results)} variants)")


def run_r19_b3(out):
    """R19-B3: K-Fold for best session filters."""
    print("\n" + "=" * 70)
    print("R19-B3: Session Filter K-Fold")
    print("=" * 70)

    best_configs = [
        ("Baseline", {}),
        ("Block_21_6", {"h1_allowed_sessions": list(range(7, 21))}),
        ("London_NY_8_20", {"h1_allowed_sessions": list(range(8, 21))}),
    ]

    tasks = []
    for sp in [0.30, 0.50]:
        for config_name, extra_kw in best_configs:
            for fold_name, start, end in FOLDS:
                kw = get_l6()
                kw.update(extra_kw)
                tasks.append((f"{config_name}_{fold_name}_sp{sp}", kw, sp, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R19-B3_session_kfold.txt", 'w') as f:
        f.write("R19-B3: Session Filter K-Fold\n" + "=" * 80 + "\n\n")
        for sp in [0.30, 0.50]:
            f.write(f"\n=== Spread ${sp} ===\n")
            for config_name, _ in best_configs:
                if config_name == "Baseline":
                    continue
                wins = 0
                f.write(f"\n{config_name} sp${sp}:\n")
                for fold_name, _, _ in FOLDS:
                    base = [r for r in results
                            if r[0] == f"Baseline_{fold_name}_sp{sp}"]
                    test = [r for r in results
                            if r[0] == f"{config_name}_{fold_name}_sp{sp}"]
                    if base and test:
                        d = test[0][2] - base[0][2]
                        won = d > 0
                        if won:
                            wins += 1
                        f.write(f"  {fold_name}: Base={base[0][2]:.2f} "
                                f"Test={test[0][2]:.2f} delta={d:+.2f} "
                                f"{'V' if won else 'X'}\n")
                result = "PASS" if wins >= 5 else "FAIL"
                f.write(f"  Result: {wins}/6 {result}\n")

    print(f"  Saved R19-B3 ({len(results)} variants)")


# ═══════════════════════════════════════════════════════════════
# PHASE C: Volatility Regime Refinement
# ═══════════════════════════════════════════════════════════════

def run_r19_c1(out):
    """R19-C1: 5-tier regime trade quality analysis."""
    print("\n" + "=" * 70)
    print("R19-C1: 5-Tier ATR Regime Trade Quality")
    print("=" * 70)

    import pandas as pd

    tasks = [("C1_full", get_l6(), 0.30, None, None)]
    results = run_pool(tasks, func=_run_one_trades)

    if not results:
        return

    trades_data = results[0][8]

    from backtest.runner import DataBundle
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    h1_df = data.h1_df

    with open(f"{out}/R19-C1_regime_quality.txt", 'w') as f:
        f.write("R19-C1: 5-Tier ATR Regime Quality\n" + "=" * 80 + "\n\n")
        f.write("Tiers: VeryLow(0-20%), Low(20-40%), Normal(40-60%), "
                "High(60-80%), VeryHigh(80-100%)\n\n")

        tier_trades = defaultdict(list)
        for pnl, reason, bars, strat, etime, direction, lots, eprice in trades_data:
            ts = pd.Timestamp(etime)
            h1_time = ts.floor('h')
            idx = h1_df.index.get_indexer([h1_time], method='ffill')
            if len(idx) > 0 and idx[0] >= 50:
                atr_series = h1_df['ATR'].iloc[max(0, idx[0]-50):idx[0]+1].dropna()
                if len(atr_series) >= 10:
                    current_atr = float(atr_series.iloc[-1])
                    atr_pct = float((atr_series < current_atr).mean())
                else:
                    atr_pct = 0.5
            else:
                atr_pct = 0.5

            if atr_pct < 0.20:
                tier = "VeryLow"
            elif atr_pct < 0.40:
                tier = "Low"
            elif atr_pct < 0.60:
                tier = "Normal"
            elif atr_pct < 0.80:
                tier = "High"
            else:
                tier = "VeryHigh"
            tier_trades[tier].append({
                'pnl': pnl, 'reason': reason, 'bars': bars,
                'atr_pct': atr_pct,
            })

        f.write(f"{'Tier':>10s}  {'N':>6s}  {'WR%':>6s}  {'Avg$':>8s}  "
                f"{'Total$':>10s}  {'Sharpe':>7s}  {'AvgBars':>8s}\n")
        f.write("-" * 65 + "\n")
        for tier in ["VeryLow", "Low", "Normal", "High", "VeryHigh"]:
            trades = tier_trades.get(tier, [])
            if not trades:
                continue
            pnls = [t['pnl'] for t in trades]
            n = len(pnls)
            wr = 100.0 * sum(1 for p in pnls if p > 0) / n
            avg = np.mean(pnls)
            total = sum(pnls)
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0
            avg_bars = np.mean([t['bars'] for t in trades])
            f.write(f"{tier:>10s}  {n:>6d}  {wr:>5.1f}%  ${avg:>7.2f}  "
                    f"{fmt(total)}  {sharpe:>7.2f}  {avg_bars:>8.1f}\n")

        f.write(f"\nExit reason breakdown by tier:\n")
        for tier in ["VeryLow", "Low", "Normal", "High", "VeryHigh"]:
            trades = tier_trades.get(tier, [])
            if not trades:
                continue
            exits = Counter(t['reason'] for t in trades)
            total_n = len(trades)
            f.write(f"\n  {tier} (N={total_n}):\n")
            for reason, cnt in exits.most_common():
                f.write(f"    {reason}: {cnt} ({100*cnt/total_n:.1f}%)\n")

    print(f"  Saved R19-C1")


def run_r19_c2(out):
    """R19-C2: 5-tier vs 3-tier regime trailing optimization."""
    print("\n" + "=" * 70)
    print("R19-C2: 5-Tier vs 3-Tier Regime Trailing")
    print("=" * 70)

    regime_3tier = {
        'low':    {'trail_act': 0.30, 'trail_dist': 0.06},
        'normal': {'trail_act': 0.20, 'trail_dist': 0.04},
        'high':   {'trail_act': 0.08, 'trail_dist': 0.01},
    }

    regime_5tier_configs = [
        ("5T_v1", {
            'very_low':  {'trail_act': 0.40, 'trail_dist': 0.10},
            'low':       {'trail_act': 0.30, 'trail_dist': 0.06},
            'normal':    {'trail_act': 0.20, 'trail_dist': 0.04},
            'high':      {'trail_act': 0.10, 'trail_dist': 0.02},
            'very_high': {'trail_act': 0.06, 'trail_dist': 0.008},
        }),
        ("5T_v2", {
            'very_low':  {'trail_act': 0.35, 'trail_dist': 0.08},
            'low':       {'trail_act': 0.28, 'trail_dist': 0.05},
            'normal':    {'trail_act': 0.20, 'trail_dist': 0.04},
            'high':      {'trail_act': 0.12, 'trail_dist': 0.02},
            'very_high': {'trail_act': 0.05, 'trail_dist': 0.006},
        }),
        ("4T_v1", {
            'low':    {'trail_act': 0.35, 'trail_dist': 0.08},
            'normal': {'trail_act': 0.22, 'trail_dist': 0.04},
            'high':   {'trail_act': 0.12, 'trail_dist': 0.02},
            'very_high': {'trail_act': 0.06, 'trail_dist': 0.008},
        }),
    ]

    tasks = []
    for sp in [0.30, 0.50]:
        kw = get_base()
        kw['regime_config'] = regime_3tier
        tasks.append((f"3Tier_sp{sp}", kw, sp, None, None))

        for name, rc in regime_5tier_configs:
            kw = get_base()
            kw['regime_config'] = rc
            tasks.append((f"{name}_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R19-C2_regime_tiers.txt", 'w') as f:
        f.write("R19-C2: 5-Tier vs 3-Tier Regime Trailing\n" + "=" * 80 + "\n\n")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread ${sp} ---\n")
            sub = sorted([r for r in results if f"sp{sp}" in r[0]],
                         key=lambda x: -x[2])
            write_table(f, sub)

    print(f"  Saved R19-C2 ({len(results)} variants)")


def run_r19_c3(out):
    """R19-C3: K-Fold for best regime config."""
    print("\n" + "=" * 70)
    print("R19-C3: Regime Config K-Fold Validation")
    print("=" * 70)

    regime_3tier = {
        'low':    {'trail_act': 0.30, 'trail_dist': 0.06},
        'normal': {'trail_act': 0.20, 'trail_dist': 0.04},
        'high':   {'trail_act': 0.08, 'trail_dist': 0.01},
    }

    best_5tier = {
        'very_low':  {'trail_act': 0.40, 'trail_dist': 0.10},
        'low':       {'trail_act': 0.30, 'trail_dist': 0.06},
        'normal':    {'trail_act': 0.20, 'trail_dist': 0.04},
        'high':      {'trail_act': 0.10, 'trail_dist': 0.02},
        'very_high': {'trail_act': 0.06, 'trail_dist': 0.008},
    }

    configs = [
        ("3Tier", regime_3tier),
        ("5Tier", best_5tier),
    ]

    tasks = []
    for sp in [0.30, 0.50]:
        for config_name, rc in configs:
            for fold_name, start, end in FOLDS:
                kw = get_base()
                kw['regime_config'] = rc
                tasks.append((f"{config_name}_{fold_name}_sp{sp}", kw, sp, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R19-C3_regime_kfold.txt", 'w') as f:
        f.write("R19-C3: Regime K-Fold (5Tier vs 3Tier)\n" + "=" * 80 + "\n\n")
        for sp in [0.30, 0.50]:
            f.write(f"\n=== Spread ${sp} ===\n")
            wins = 0
            for fold_name, _, _ in FOLDS:
                base = [r for r in results
                        if r[0] == f"3Tier_{fold_name}_sp{sp}"]
                test = [r for r in results
                        if r[0] == f"5Tier_{fold_name}_sp{sp}"]
                if base and test:
                    d = test[0][2] - base[0][2]
                    won = d > 0
                    if won:
                        wins += 1
                    f.write(f"  {fold_name}: 3Tier={base[0][2]:.2f} "
                            f"5Tier={test[0][2]:.2f} delta={d:+.2f} "
                            f"{'V' if won else 'X'}\n")
            result = "PASS" if wins >= 5 else "FAIL"
            f.write(f"  Result: {wins}/6 {result}\n")

    print(f"  Saved R19-C3 ({len(results)} variants)")


# ═══════════════════════════════════════════════════════════════
# PHASE D: Dynamic Position Management
# ═══════════════════════════════════════════════════════════════

def run_r19_d1(out):
    """R19-D1: Max hold bar comparison."""
    print("\n" + "=" * 70)
    print("R19-D1: Max Hold Bars Comparison")
    print("=" * 70)

    hold_bars = [12, 15, 18, 20, 24, 30]

    tasks = []
    for hb in hold_bars:
        for sp in [0.30, 0.50]:
            kw = get_l6()
            kw['keltner_max_hold_m15'] = hb
            tasks.append((f"Hold{hb}_sp{sp}", kw, sp, None, None))

    results = run_pool(tasks)

    with open(f"{out}/R19-D1_max_hold.txt", 'w') as f:
        f.write("R19-D1: Max Hold Bars Comparison\n" + "=" * 80 + "\n\n")
        for sp in [0.30, 0.50]:
            f.write(f"\n--- Spread ${sp} ---\n")
            sub = sorted([r for r in results if f"sp{sp}" in r[0]],
                         key=lambda x: -x[2])
            write_table(f, sub)

    print(f"  Saved R19-D1 ({len(results)} variants)")


def run_r19_d2(out):
    """R19-D2: K-Fold for best max_hold + combined with best filters."""
    print("\n" + "=" * 70)
    print("R19-D2: Max Hold + Best Filters K-Fold")
    print("=" * 70)

    configs = [
        ("Hold20_baseline", {"keltner_max_hold_m15": 20}),
        ("Hold15", {"keltner_max_hold_m15": 15}),
        ("Hold18", {"keltner_max_hold_m15": 18}),
        ("Hold24", {"keltner_max_hold_m15": 24}),
    ]

    tasks = []
    for sp in [0.30, 0.50]:
        for config_name, extra_kw in configs:
            for fold_name, start, end in FOLDS:
                kw = get_l6()
                kw.update(extra_kw)
                tasks.append((f"{config_name}_{fold_name}_sp{sp}", kw, sp, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R19-D2_hold_kfold.txt", 'w') as f:
        f.write("R19-D2: Max Hold K-Fold\n" + "=" * 80 + "\n\n")
        for sp in [0.30, 0.50]:
            f.write(f"\n=== Spread ${sp} ===\n")
            for config_name, _ in configs:
                if "baseline" in config_name:
                    continue
                wins = 0
                f.write(f"\n{config_name} vs Hold20 sp${sp}:\n")
                for fold_name, _, _ in FOLDS:
                    base = [r for r in results
                            if r[0] == f"Hold20_baseline_{fold_name}_sp{sp}"]
                    test = [r for r in results
                            if r[0] == f"{config_name}_{fold_name}_sp{sp}"]
                    if base and test:
                        d = test[0][2] - base[0][2]
                        won = d > 0
                        if won:
                            wins += 1
                        f.write(f"  {fold_name}: Base={base[0][2]:.2f} "
                                f"Test={test[0][2]:.2f} delta={d:+.2f} "
                                f"{'V' if won else 'X'}\n")
                result = "PASS" if wins >= 5 else "FAIL"
                f.write(f"  Result: {wins}/6 {result}\n")

    print(f"  Saved R19-D2 ({len(results)} variants)")


# ═══════════════════════════════════════════════════════════════
# PHASE E: Combined Validation
# ═══════════════════════════════════════════════════════════════

def run_r19_e1(out):
    """R19-E1: Stack validated improvements."""
    print("\n" + "=" * 70)
    print("R19-E1: Combined Improvements (L8 Candidate)")
    print("=" * 70)

    configs = [
        ("L6_Baseline", get_l6()),
        ("L7_TATrail_Gap", get_l7()),
    ]

    tasks = []
    for name, kw in configs:
        for sp in [0.30, 0.50]:
            tasks.append((f"{name}_sp{sp}", {**kw}, sp, None, None))
            for yr_name, start, end in YEARS:
                tasks.append((f"{name}_{yr_name}_sp{sp}", {**kw}, sp, start, end))

    results = run_pool(tasks)

    with open(f"{out}/R19-E1_combined.txt", 'w') as f:
        f.write("R19-E1: Combined Improvements\n" + "=" * 80 + "\n\n")
        for sp in [0.30, 0.50]:
            f.write(f"\n=== Full Period (Spread ${sp}) ===\n")
            full = [r for r in results
                    if f"sp{sp}" in r[0] and not any(yr in r[0] for yr, _, _ in YEARS)]
            write_table(f, sorted(full, key=lambda x: -x[2]))

            f.write(f"\n--- Year-by-Year ---\n")
            f.write(f"{'Year':<6}")
            for name, _ in configs:
                f.write(f"  {name:>20s}")
            f.write("\n" + "-" * (6 + 22 * len(configs)) + "\n")
            for yr_name, _, _ in YEARS:
                f.write(f"{yr_name:<6}")
                for name, _ in configs:
                    match = [r for r in results
                             if r[0] == f"{name}_{yr_name}_sp{sp}"]
                    if match:
                        f.write(f"  {match[0][2]:>20.2f}")
                    else:
                        f.write(f"  {'N/A':>20s}")
                f.write("\n")

    print(f"  Saved R19-E1 ({len(results)} variants)")


def run_r19_e2(out):
    """R19-E2: Monte Carlo 100x parameter perturbation."""
    print("\n" + "=" * 70)
    print("R19-E2: Monte Carlo Parameter Perturbation (100x)")
    print("=" * 70)

    N_MC = 100
    rng = np.random.RandomState(42)

    base = get_l6()
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
                'trail_dist': round(params['trail_dist'] * (1 + rng.uniform(-0.15, 0.15)), 4),
            }
        kw['regime_config'] = new_rc
        tasks.append((f"MC_{i:03d}", kw, 0.30, None, None))

    results = run_pool(tasks)

    sharpes = [r[2] for r in results]
    pnls = [r[3] for r in results]

    with open(f"{out}/R19-E2_monte_carlo.txt", 'w') as f:
        f.write("R19-E2: Monte Carlo Robustness (100 runs, +/-15% param perturbation)\n"
                + "=" * 80 + "\n\n")

        base_result = run_pool([("L6_base", get_l6(), 0.30, None, None)])[0]
        f.write(f"Base: Sharpe={base_result[2]:.2f} PnL={fmt(base_result[3])}\n\n")
        f.write(f"Sharpe: mean={np.mean(sharpes):.2f} std={np.std(sharpes):.2f} "
                f"min={min(sharpes):.2f} max={max(sharpes):.2f}\n")
        f.write(f"PnL:    mean={fmt(np.mean(pnls))} min={fmt(min(pnls))} "
                f"max={fmt(max(pnls))}\n")
        f.write(f"Profitable: {sum(1 for p in pnls if p > 0)}/100 "
                f"({100*sum(1 for p in pnls if p > 0)/100:.0f}%)\n")
        f.write(f"Sharpe > 4.0: {sum(1 for s in sharpes if s > 4)}/100\n")
        f.write(f"Sharpe > 2.0: {sum(1 for s in sharpes if s > 2)}/100\n\n")

        worst = sorted(results, key=lambda x: x[2])[:5]
        f.write("Worst 5 runs:\n")
        for r in worst:
            f.write(f"  {r[0]}: Sharpe={r[2]:.2f} PnL={fmt(r[3])}\n")

    print(f"  Saved R19-E2")


def run_r19_e3(out):
    """R19-E3: Purged Walk-Forward."""
    print("\n" + "=" * 70)
    print("R19-E3: Purged Walk-Forward Validation")
    print("=" * 70)

    WF_FOLDS = [
        ("WF1", "2015-01-01", "2017-01-01", "2017-01-01", "2019-01-01"),
        ("WF2", "2017-01-01", "2019-01-01", "2019-01-01", "2021-01-01"),
        ("WF3", "2019-01-01", "2021-01-01", "2021-01-01", "2023-01-01"),
        ("WF4", "2021-01-01", "2023-01-01", "2023-01-01", "2025-01-01"),
        ("WF5", "2023-01-01", "2025-01-01", "2025-01-01", "2026-04-10"),
    ]

    configs = [("L6", get_l6)]
    results_map = {}

    from backtest.runner import DataBundle
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

    for config_name, get_fn in configs:
        fold_results = []
        for fold_name, train_s, train_e, test_s, test_e in WF_FOLDS:
            test_data = data.slice(test_s, test_e)
            if len(test_data.m15_df) < 1000:
                continue
            kw = get_fn()
            from backtest.runner import run_variant
            s = run_variant(test_data, f"WF_{config_name}_{fold_name}",
                            verbose=False, spread_cost=0.30, **kw)
            fold_results.append({
                'fold': fold_name, 'n': s['n'], 'sharpe': s['sharpe'],
                'pnl': s['total_pnl'], 'max_dd': s['max_dd'],
                'wr': s['win_rate'],
            })
        results_map[config_name] = fold_results

    with open(f"{out}/R19-E3_walk_forward.txt", 'w') as f:
        f.write("R19-E3: Purged Walk-Forward\n" + "=" * 80 + "\n\n")
        for config_name, folds in results_map.items():
            f.write(f"\n=== {config_name} ===\n")
            f.write(f"{'Fold':<8s} {'Period':>30s}  {'N':>6s} {'Sharpe':>8s} "
                    f"{'PnL':>12s} {'MaxDD':>10s} {'WR':>7s}\n")
            f.write("-" * 85 + "\n")
            for i, fr in enumerate(folds):
                _, _, _, test_s, test_e = WF_FOLDS[i]
                f.write(f"{fr['fold']:<8s} {test_s} -> {test_e:>10s}  "
                        f"{fr['n']:>6d} {fr['sharpe']:>8.2f} "
                        f"{fmt(fr['pnl']):>12s} {fmt(fr['max_dd']):>10s} "
                        f"{fr['wr']:>6.1f}%\n")
            wins = sum(1 for fr in folds if fr['sharpe'] > 0)
            f.write(f"\nPositive Sharpe folds: {wins}/{len(folds)}\n")

    print(f"  Saved R19-E3")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    phases = [
        ("A1", run_r19_a1),
        ("A2", run_r19_a2),
        ("A3", run_r19_a3),
        ("A4", run_r19_a4),
        ("B1", run_r19_b1),
        ("B2", run_r19_b2),
        ("B3", run_r19_b3),
        ("C1", run_r19_c1),
        ("C2", run_r19_c2),
        ("C3", run_r19_c3),
        ("D1", run_r19_d1),
        ("D2", run_r19_d2),
        ("E1", run_r19_e1),
        ("E2", run_r19_e2),
        ("E3", run_r19_e3),
    ]

    print("\n" + "=" * 70)
    print(f"  Round 19 — Signal Quality & Regime Enhancement")
    print(f"  Started: {datetime.now()}")
    print(f"  Phases: {len(phases)}")
    print("=" * 70)

    t0 = time.time()
    phase_times = {}

    for phase_name, phase_func in phases:
        print(f"\n>>> Starting Phase {phase_name}...")
        pt0 = time.time()
        try:
            phase_func(OUTPUT_DIR)
            elapsed = time.time() - pt0
            phase_times[phase_name] = ("OK", elapsed)
            print(f"<<< Phase {phase_name} done in {elapsed:.0f}s")
        except Exception as e:
            elapsed = time.time() - pt0
            phase_times[phase_name] = ("FAIL", elapsed)
            print(f"<<< Phase {phase_name} FAILED in {elapsed:.0f}s: {e}")
            traceback.print_exc()

    total = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Round 19 Complete — Total: {total:.0f}s ({total/3600:.1f}h)")
    print(f"{'=' * 70}")
    for pn, (status, pt) in phase_times.items():
        print(f"  Phase {pn:>4s}  {pt:>10.0f}s  {status}")

    with open(f"{OUTPUT_DIR}/R19_summary.txt", 'w') as f:
        f.write(f"Round 19 Summary\n{'=' * 60}\n")
        f.write(f"Total: {total:.0f}s ({total/3600:.1f}h)\n\n")
        for pn, (status, pt) in phase_times.items():
            f.write(f"Phase {pn:>4s}  {pt:>10.0f}s  {status}\n")


if __name__ == "__main__":
    main()
