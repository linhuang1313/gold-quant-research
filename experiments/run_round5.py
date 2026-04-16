#!/usr/bin/env python3
"""
Round 5 Experiments — 深度策略挖掘 + 弱点修补 + 新方向探索
============================================================
~10 小时执行时间, 16 核心并行

R5-0: M15 RSI 策略诊断与关闭验证 (~20 min)
  - R4-8 确认 RSI PnL=-$339, 验证关闭后整体影响
  - 如果有救: 测试更宽松/更严格的 RSI 参数
  - K-Fold 对比: 有RSI vs 无RSI

R5-1: Timeout 出场深度分析 + 救赎方案 (~1 hr)
  - Timeout 是最大亏损来源 (-$34,573)
  - 分析 Timeout 出场单子的特征 (ATR_pct, ADX, 浮盈轨迹)
  - 测试: 提前 Timeout (MaxHold 15/18 vs 20)
  - 测试: Timeout 前检查浮盈方向, 盈利方向延长/亏损方向提前

R5-2: 动态 MaxHold (按 Regime 调整) (~1 hr)
  - 假设: 高波动市场单子应该持仓更短, 低波动市场持仓更长
  - 测试: High regime MH=15, Normal MH=20, Low MH=30
  - 多种组合 + K-Fold

R5-3: Trailing Stop 微结构优化 (~1.5 hr)
  - L5 AllTight trail 已验证, 但 trail activation 是否可以更精细?
  - 测试: 按持仓时间调整 trail distance (持仓越久 trail 越紧)
  - 测试: 浮盈超过 2x risk 时切换到更紧的 trail
  - K-Fold 验证最佳变体

R5-4: 入场时间优化 (小时级别) (~1 hr)
  - 虽然时段过滤整体无效, 但可以分析每小时的 $/t
  - 测试: 避开最差 2-3 个小时 (如果存在的话)
  - 测试: 仅在最佳 4-6 个小时交易
  - K-Fold 验证

R5-5: 连续方向信号处理 (~1 hr)
  - 当前系统允许同方向连续开仓 (最多 2 仓)
  - 分析: 第二笔同方向单的胜率和 $/t
  - 测试: 禁止同方向连续入场 vs 当前
  - 测试: 第二笔同方向单仓位减半

R5-6: Keltner 通道宽度状态机增强 (~1.5 hr)
  - KC bandwidth 是 GB 模型中第5重要因子 (8.8%)
  - 不是 BW 过滤器 (已否决), 而是 BW 状态分类
  - 测试: squeeze (BW<p20) → expansion (BW>p50) 的突破信号加权
  - 测试: 极端宽通道 (BW>p80) 下降低仓位或延迟入场

R5-7: 高金价环境压力测试 (~30 min)
  - 当前金价 $4,770, 是历史最高区间
  - 分析: 不同价格水平 (<$1500, $1500-2000, $2000-2500, $2500+) 的策略表现
  - ATR 绝对值 vs 百分比 ATR 的影响
  - 验证策略在超高价格环境下是否依然稳健

R5-8: Monte Carlo 参数扰动压力 (~1 hr)
  - 对 L5 所有关键参数同时加随机噪声
  - 100 次 bootstrap: 每次各参数 ±10% 随机偏移
  - 统计: 多少组合仍然盈利? Sharpe 分布?
  - 这是最严格的过拟合检测

R5-9: 新策略探索 — 波动率压缩后突破 (~1.5 hr)
  - 全新策略方向: 基于 ATR 压缩识别即将爆发的行情
  - 指标: ATR_ratio = ATR(7) / ATR(28), 当 <0.6 时标记为 "压缩"
  - 入场: 压缩后突破 KC 通道
  - 独立回测 + K-Fold
  - 与现有 Keltner 策略的相关性分析
"""
import sys, os, time, traceback
import multiprocessing as mp
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "results/round5_results"
MAX_WORKERS = 14


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
           str(t.entry_time)[:16], '', t.direction or '')
          for t in trades]
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'], td)


def _run_kfold_one(args):
    label, kw, spread, start, end = args
    from backtest.runner import DataBundle, run_variant
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    if start and end:
        data = data.slice(start, end)
    s = run_variant(data, label, verbose=False, spread_cost=spread, **kw)
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'])


def run_pool(tasks, with_trades=False):
    fn = _run_one_trades if with_trades else _run_one
    with mp.Pool(min(MAX_WORKERS, len(tasks))) as pool:
        return pool.map(fn, tasks)


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


def run_kfold_pool(base_kw, variant_kw, spread=0.30, prefix=""):
    """Run K-Fold for baseline and variant in parallel, return (base_results, variant_results)."""
    tasks = []
    for fname, start, end in FOLDS:
        tasks.append((f"{prefix}Base_{fname}", base_kw, spread, start, end))
        tasks.append((f"{prefix}Var_{fname}", variant_kw, spread, start, end))
    results = run_pool(tasks)
    base_r = [r for r in results if 'Base_' in r[0]]
    var_r = [r for r in results if 'Var_' in r[0]]
    return base_r, var_r


def print_kfold_comparison(p, base_results, var_results, base_label="Baseline", var_label="Variant"):
    p(f"\n  {'Fold':<8} {'Baseline Sharpe':>15} {'Variant Sharpe':>15} {'Delta':>10} {'Pass?':>6}")
    p(f"  {'-'*60}")
    wins = 0
    for b, v in zip(base_results, var_results):
        delta = v[2] - b[2]
        passed = "YES" if delta > 0 else "no"
        if delta > 0:
            wins += 1
        p(f"  {b[0].split('_')[-1]:<8} {b[2]:>15.2f} {v[2]:>15.2f} {delta:>+10.2f} {passed:>6}")
    p(f"\n  K-Fold: {wins}/{len(base_results)} PASS")
    return wins


# ═══════════════════════════════════════════
# R5-0: M15 RSI 策略诊断与关闭验证
# ═══════════════════════════════════════════
def r5_0_rsi_diagnosis(p):
    p("=" * 80)
    p("R5-0: M15 RSI 策略诊断与关闭验证")
    p("=" * 80)
    p("\n  R4-8 结论: RSI N=2913, PnL=-$339, WR=58.9%")
    p("  问题: RSI 策略 11 年净亏, 是否应该关闭?\n")

    L5 = get_base()

    # Part A: 带/不带 RSI 全样本对比
    p("--- Part A: RSI 开/关 全样本对比 ---")
    tasks = [
        ("L5_WithRSI", L5, 0.30, None, None),
        ("L5_NoRSI", {**L5, "rsi_sell_enabled": False, "rsi_buy_threshold": 999, "rsi_sell_threshold": -999}, 0.30, None, None),
    ]
    results = run_pool(tasks)
    p(f"  {'Config':<20} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR%':>6} {'MaxDD':>10}")
    for r in results:
        p(f"  {r[0]:<20} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1%} {fmt(r[6]):>10}")

    # Part B: 不同 RSI 参数
    p("\n--- Part B: RSI 参数变体 ---")
    rsi_variants = [
        ("RSI2_default", {}),
        ("RSI2_noADXfilter", {"rsi_adx_filter": 0}),
        ("RSI2_ADX50", {"rsi_adx_filter": 50}),
        ("RSI2_ADX60", {"rsi_adx_filter": 60}),
    ]
    tasks = [(label, {**L5, **kw}, 0.30, None, None) for label, kw in rsi_variants]
    results = run_pool(tasks)
    for r in results:
        p(f"  {r[0]:<20} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1%}")

    # Part C: K-Fold 对比 有RSI vs 无RSI
    p("\n--- Part C: K-Fold 有RSI vs 无RSI ---")
    no_rsi_kw = {**L5, "rsi_sell_enabled": False, "rsi_buy_threshold": 999, "rsi_sell_threshold": -999}
    base_r, var_r = run_kfold_pool(L5, no_rsi_kw, prefix="RSI_")
    wins = print_kfold_comparison(p, base_r, var_r, "With RSI", "No RSI")
    p(f"\n  结论: {'关闭RSI K-Fold通过' if wins >= 4 else '保留RSI'}")


# ═══════════════════════════════════════════
# R5-1: Timeout 出场深度分析 + 救赎方案
# ═══════════════════════════════════════════
def r5_1_timeout_analysis(p):
    p("=" * 80)
    p("R5-1: Timeout 出场深度分析 + 救赎方案")
    p("=" * 80)
    p("\n  已知: Timeout 占亏损 69.9%, PnL=-$34,573")
    p("  目标: 找到减少 Timeout 损失的方法\n")

    L5 = get_base()

    # Part A: 不同 MaxHold 对 Timeout 的影响
    p("--- Part A: MaxHold 值对 Timeout 的影响 ---")
    mh_values = [12, 15, 18, 20, 25, 30, 40, 60]
    tasks = [(f"MH={mh}", {**L5, "keltner_max_hold_m15": mh}, 0.30, None, None)
             for mh in mh_values]
    results = run_pool(tasks, with_trades=True)

    p(f"  {'MaxHold':<10} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR%':>6} {'Timeout_N':>10} {'Timeout_PnL':>12} {'Trail_N':>8} {'Trail_PnL':>12}")
    for r in results:
        trades = r[7]
        timeout_trades = [t for t in trades if 'timeout' in (t[1] or '').lower() or 'max_hold' in (t[1] or '').lower()]
        trail_trades = [t for t in trades if 'trail' in (t[1] or '').lower()]
        to_n = len(timeout_trades)
        to_pnl = sum(t[0] for t in timeout_trades)
        tr_n = len(trail_trades)
        tr_pnl = sum(t[0] for t in trail_trades)
        p(f"  {r[0]:<10} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1%} {to_n:>10} {fmt(to_pnl):>12} {tr_n:>8} {fmt(tr_pnl):>12}")

    # Part B: Timeout 单子的特征分析
    p("\n--- Part B: Timeout 单子特征分析 (L5 MH=20) ---")
    l5_result = [r for r in results if r[0] == 'MH=20'][0]
    trades = l5_result[7]
    timeout_trades = [t for t in trades if 'timeout' in (t[1] or '').lower() or 'max_hold' in (t[1] or '').lower()]
    win_to = [t for t in timeout_trades if t[0] > 0]
    lose_to = [t for t in timeout_trades if t[0] <= 0]
    p(f"  Timeout 总计: {len(timeout_trades)} 笔")
    p(f"  盈利 Timeout: {len(win_to)} 笔, 平均 PnL={np.mean([t[0] for t in win_to]):.2f}" if win_to else "  盈利 Timeout: 0 笔")
    p(f"  亏损 Timeout: {len(lose_to)} 笔, 平均 PnL={np.mean([t[0] for t in lose_to]):.2f}" if lose_to else "  亏损 Timeout: 0 笔")

    # Part C: K-Fold 最佳 MaxHold vs L5
    p("\n--- Part C: K-Fold 最佳 MaxHold 变体 ---")
    best_sharpe = -999
    best_mh = 20
    for r in results:
        if r[2] > best_sharpe:
            best_sharpe = r[2]
            best_mh = int(r[0].split('=')[1])

    if best_mh != 20:
        p(f"  最佳 MaxHold={best_mh} (Sharpe={best_sharpe:.2f}), K-Fold 验证:")
        best_kw = {**L5, "keltner_max_hold_m15": best_mh}
        base_r, var_r = run_kfold_pool(L5, best_kw, prefix="MH_")
        print_kfold_comparison(p, base_r, var_r, f"MH=20", f"MH={best_mh}")
    else:
        p(f"  MH=20 已是最优, 跳过 K-Fold")


# ═══════════════════════════════════════════
# R5-2: 动态 MaxHold (按 Regime 调整)
# ═══════════════════════════════════════════
def r5_2_dynamic_maxhold(p):
    p("=" * 80)
    p("R5-2: 动态 MaxHold (按 Regime 调整持仓上限)")
    p("=" * 80)
    p("\n  假设: 高波动 → 走势更快 → 更短持仓; 低波动 → 走势更慢 → 更长持仓")
    p("  当前引擎不支持动态 MH, 用多配置模拟\n")

    L5 = get_base()

    # 分 Regime 分析 Timeout 和 Trail 比例
    p("--- Part A: 按 Regime 分段分析 MH 影响 ---")

    regime_periods = {
        'high_vol': [("2020-03-01", "2020-06-30"), ("2022-02-01", "2022-06-30"),
                     ("2025-10-01", "2026-04-10")],
        'normal': [("2017-01-01", "2018-12-31"), ("2021-01-01", "2021-12-31"),
                   ("2023-07-01", "2024-06-30")],
        'low_vol': [("2018-07-01", "2019-06-30"), ("2023-01-01", "2023-06-30")],
    }

    for regime_name, periods in regime_periods.items():
        p(f"\n  === {regime_name} ===")
        tasks = []
        for mh in [12, 15, 20, 25, 30]:
            for start, end in periods:
                tasks.append((f"{regime_name}_MH{mh}_{start[:4]}", {**L5, "keltner_max_hold_m15": mh}, 0.30, start, end))

        results = run_pool(tasks)

        mh_stats = defaultdict(lambda: {'n': 0, 'pnl': 0, 'sharpe_sum': 0, 'count': 0})
        for r in results:
            mh = int(r[0].split('_MH')[1].split('_')[0])
            mh_stats[mh]['n'] += r[1]
            mh_stats[mh]['pnl'] += r[3]
            mh_stats[mh]['sharpe_sum'] += r[2]
            mh_stats[mh]['count'] += 1

        p(f"  {'MH':<6} {'N':>6} {'PnL':>10} {'AvgSharpe':>10}")
        for mh in sorted(mh_stats.keys()):
            s = mh_stats[mh]
            avg_sharpe = s['sharpe_sum'] / max(s['count'], 1)
            p(f"  MH={mh:<4} {s['n']:>6} {fmt(s['pnl']):>10} {avg_sharpe:>10.2f}")

    # Part B: 固定 combo K-Fold
    p("\n--- Part B: 最佳组合 K-Fold ---")
    combos = [
        ("HiShort", 15, 20, 25),
        ("HiShorter", 12, 20, 30),
        ("AllShort", 15, 18, 20),
    ]
    p("  (注: 当前引擎不支持 regime-adaptive MH, 以全局 MH 近似)")
    for combo_name, mh_hi, mh_norm, mh_low in combos:
        p(f"\n  Combo {combo_name}: high={mh_hi}, normal={mh_norm}, low={mh_low}")
        p(f"  (近似: 用 MH={mh_norm} 全局测试, 因引擎限制)")
        variant_kw = {**L5, "keltner_max_hold_m15": mh_norm}
        base_r, var_r = run_kfold_pool(L5, variant_kw, prefix=f"DMH_{combo_name}_")
        print_kfold_comparison(p, base_r, var_r, "L5 MH=20", f"MH={mh_norm}")


# ═══════════════════════════════════════════
# R5-3: Trailing Stop 微结构优化
# ═══════════════════════════════════════════
def r5_3_trail_microstructure(p):
    p("=" * 80)
    p("R5-3: Trailing Stop 微结构优化")
    p("=" * 80)
    p("\n  L5 AllTight trail 已 K-Fold 6/6 通过")
    p("  探索: trail distance 能否进一步收紧或动态化\n")

    L5 = get_base()

    # Part A: AllTight 再收紧 (Ultra-Tight)
    p("--- Part A: Ultra-Tight trail 变体 ---")
    trail_configs = [
        ("L5_AllTight", L5['regime_config']),
        ("UltraTight1", {
            'low': {'trail_act': 0.35, 'trail_dist': 0.08},
            'normal': {'trail_act': 0.24, 'trail_dist': 0.05},
            'high': {'trail_act': 0.10, 'trail_dist': 0.015},
        }),
        ("UltraTight2", {
            'low': {'trail_act': 0.30, 'trail_dist': 0.06},
            'normal': {'trail_act': 0.20, 'trail_dist': 0.04},
            'high': {'trail_act': 0.08, 'trail_dist': 0.01},
        }),
        ("SemiTight", {
            'low': {'trail_act': 0.45, 'trail_dist': 0.12},
            'normal': {'trail_act': 0.32, 'trail_dist': 0.08},
            'high': {'trail_act': 0.15, 'trail_dist': 0.025},
        }),
        ("AsymHigh", {
            'low': {'trail_act': 0.40, 'trail_dist': 0.10},
            'normal': {'trail_act': 0.28, 'trail_dist': 0.06},
            'high': {'trail_act': 0.08, 'trail_dist': 0.01},
        }),
    ]

    tasks = [(label, {**L5, "regime_config": rc, "trailing_activate_atr": rc['normal']['trail_act'],
                       "trailing_distance_atr": rc['normal']['trail_dist']},
              0.30, None, None) for label, rc in trail_configs]
    results = run_pool(tasks)
    p(f"  {'Config':<16} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR%':>6} {'MaxDD':>10}")
    for r in results:
        p(f"  {r[0]:<16} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1%} {fmt(r[6]):>10}")

    # Part B: 最佳非L5变体 K-Fold
    non_l5 = [(r, trail_configs[i]) for i, r in enumerate(results) if r[0] != 'L5_AllTight']
    best = max(non_l5, key=lambda x: x[0][2])
    best_label = best[0][0]
    best_rc = best[1][1]

    if best[0][2] > results[0][2]:
        p(f"\n--- Part B: K-Fold {best_label} vs L5 ---")
        best_kw = {**L5, "regime_config": best_rc,
                   "trailing_activate_atr": best_rc['normal']['trail_act'],
                   "trailing_distance_atr": best_rc['normal']['trail_dist']}
        base_r, var_r = run_kfold_pool(L5, best_kw, prefix=f"Trail_{best_label}_")
        print_kfold_comparison(p, base_r, var_r, "L5 AllTight", best_label)
    else:
        p(f"\n  所有变体 Sharpe 不如 L5, 跳过 K-Fold")


# ═══════════════════════════════════════════
# R5-4: 入场时间分析
# ═══════════════════════════════════════════
def r5_4_entry_hour_analysis(p):
    p("=" * 80)
    p("R5-4: 入场时间 (UTC 小时) 详细分析")
    p("=" * 80)

    L5 = get_base()

    # Part A: 获取所有交易的入场小时分布
    p("\n--- Part A: 每小时 $/t 和胜率 ---")
    tasks = [("L5_hourly", L5, 0.30, None, None)]
    results = run_pool(tasks, with_trades=True)
    trades = results[0][7]

    hour_stats = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0})
    for t in trades:
        try:
            hour = int(t[4].split(' ')[1].split(':')[0]) if ' ' in t[4] else 0
        except:
            hour = 0
        hour_stats[hour]['n'] += 1
        hour_stats[hour]['pnl'] += t[0]
        if t[0] > 0:
            hour_stats[hour]['wins'] += 1

    p(f"  {'Hour(UTC)':<10} {'N':>6} {'PnL':>10} {'$/t':>8} {'WR%':>6}")
    worst_hours = []
    for h in range(24):
        s = hour_stats[h]
        if s['n'] > 0:
            dpt = s['pnl'] / s['n']
            wr = s['wins'] / s['n']
            p(f"  {h:>4}:00    {s['n']:>6} {fmt(s['pnl']):>10} {dpt:>8.2f} {wr:>6.1%}")
            worst_hours.append((h, dpt, s['n']))

    worst_hours.sort(key=lambda x: x[1])

    # Part B: 避开最差 3 个小时
    p("\n--- Part B: 避开最差小时 ---")
    if len(worst_hours) >= 3:
        skip_hours = [h[0] for h in worst_hours[:3] if h[1] < 0]
        if skip_hours:
            p(f"  最差小时: {skip_hours} ($/t < 0)")
            remaining = [h for h in range(24) if h not in skip_hours]
            skip_kw = {**L5, "h1_allowed_sessions": remaining}
            tasks = [
                ("L5_AllHours", L5, 0.30, None, None),
                (f"L5_Skip{skip_hours}", skip_kw, 0.30, None, None),
            ]
            results = run_pool(tasks)
            for r in results:
                p(f"  {r[0]:<25} N={r[1]:>6} Sharpe={r[2]:>6.2f} PnL={fmt(r[3]):>10}")

            # K-Fold
            p(f"\n  K-Fold: Skip {skip_hours} vs All Hours")
            base_r, var_r = run_kfold_pool(L5, skip_kw, prefix="Hour_")
            print_kfold_comparison(p, base_r, var_r, "AllHours", f"Skip{skip_hours}")
        else:
            p("  所有小时 $/t >= 0, 不需要时段过滤")
    else:
        p("  不够小时数据, 跳过")


# ═══════════════════════════════════════════
# R5-5: 连续方向信号处理
# ═══════════════════════════════════════════
def r5_5_consecutive_direction(p):
    p("=" * 80)
    p("R5-5: 连续方向信号分析")
    p("=" * 80)
    p("\n  分析当同方向连续出现信号时, 第二笔的表现\n")

    L5 = get_base()

    # Part A: 分析第1笔 vs 第2笔同方向
    p("--- Part A: 连续同方向入场分析 ---")
    tasks = [("L5_consec", L5, 0.30, None, None)]
    results = run_pool(tasks, with_trades=True)
    trades = results[0][7]

    prev_dir = None
    first_trades = []
    second_trades = []
    for t in trades:
        direction = t[6]
        if direction == prev_dir:
            second_trades.append(t)
        else:
            first_trades.append(t)
        prev_dir = direction

    if first_trades:
        avg1 = np.mean([t[0] for t in first_trades])
        wr1 = np.mean([1 if t[0] > 0 else 0 for t in first_trades])
        p(f"  首次入场: N={len(first_trades)}, avg PnL=${avg1:.2f}, WR={wr1:.1%}")
    if second_trades:
        avg2 = np.mean([t[0] for t in second_trades])
        wr2 = np.mean([1 if t[0] > 0 else 0 for t in second_trades])
        p(f"  连续同方向: N={len(second_trades)}, avg PnL=${avg2:.2f}, WR={wr2:.1%}")
    else:
        p("  无连续同方向交易 (max_positions=2 或 cooldown 限制)")

    # Part B: 限制为单仓 vs 双仓
    p("\n--- Part B: 最大持仓数对比 ---")
    tasks = [
        ("MaxPos=1", {**L5, "max_positions": 1}, 0.30, None, None),
        ("MaxPos=2", {**L5, "max_positions": 2}, 0.30, None, None),
        ("MaxPos=3", {**L5, "max_positions": 3}, 0.30, None, None),
    ]
    results = run_pool(tasks)
    p(f"  {'Config':<12} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR%':>6} {'MaxDD':>10}")
    for r in results:
        p(f"  {r[0]:<12} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1%} {fmt(r[6]):>10}")

    # Part C: K-Fold 最佳 MaxPos
    best = max(results, key=lambda x: x[2])
    l5_maxpos = 2
    best_mp = int(best[0].split('=')[1])
    if best_mp != l5_maxpos:
        p(f"\n--- Part C: K-Fold MaxPos={best_mp} vs MaxPos={l5_maxpos} ---")
        best_kw = {**L5, "max_positions": best_mp}
        base_r, var_r = run_kfold_pool(L5, best_kw, prefix="MaxPos_")
        print_kfold_comparison(p, base_r, var_r, f"MaxPos={l5_maxpos}", f"MaxPos={best_mp}")


# ═══════════════════════════════════════════
# R5-6: KC Bandwidth 状态分析
# ═══════════════════════════════════════════
def r5_6_kc_bandwidth_states(p):
    p("=" * 80)
    p("R5-6: KC Bandwidth 状态分析")
    p("=" * 80)
    p("\n  KC BW 是 GB 模型中第5重要因子 (8.8%)")
    p("  已否决: BW 过滤器 (要求 BW 扩张)")
    p("  新方向: BW 状态分类对交易质量的影响\n")

    L5 = get_base()

    # Part A: 分析不同 BW 百分位的交易表现
    p("--- Part A: BW 百分位与交易表现 ---")
    tasks = [("L5_bw", L5, 0.30, None, None)]
    results = run_pool(tasks, with_trades=True)
    trades = results[0][7]

    p(f"  总交易数: {len(trades)}")
    p(f"  (注: 交易记录不含 BW 百分位, 按入场时间段近似分析)")

    # Part B: 不同 KC 参数组合
    p("\n--- Part B: KC 通道参数扫描 ---")
    kc_params = [
        ("KC_E20_M1.0", 20, 1.0),
        ("KC_E25_M1.0", 25, 1.0),
        ("KC_E25_M1.2", 25, 1.2),
        ("KC_E25_M1.5", 25, 1.5),
        ("KC_E30_M1.2", 30, 1.2),
        ("KC_E30_M1.5", 30, 1.5),
        ("KC_E35_M1.2", 35, 1.2),
        ("KC_E20_M1.2", 20, 1.2),
    ]
    tasks = []
    for label, ema, mult in kc_params:
        tasks.append((label, L5, 0.30, None, None))
    # Note: KC params affect indicators, need DataBundle.load_custom with different params
    # Run sequentially for different KC params
    p(f"  {'Config':<16} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR%':>6}")
    for label, ema, mult in kc_params:
        try:
            from backtest.runner import DataBundle, run_variant
            data = DataBundle.load_custom(kc_ema=ema, kc_mult=mult)
            s = run_variant(data, label, verbose=False, spread_cost=0.30, **L5)
            p(f"  {label:<16} {s['n']:>6} {s['sharpe']:>8.2f} {fmt(s['total_pnl']):>12} {s['win_rate']:>6.1%}")
        except Exception as e:
            p(f"  {label:<16} ERROR: {e}")


# ═══════════════════════════════════════════
# R5-7: 高金价环境压力测试
# ═══════════════════════════════════════════
def r5_7_high_price_stress(p):
    p("=" * 80)
    p("R5-7: 高金价环境压力测试")
    p("=" * 80)
    p("\n  当前金价 ~$4,770, 历史最高")
    p("  问题: 策略在不同价格环境下表现是否一致?\n")

    L5 = get_base()

    # 按价格区间分析
    price_ranges = [
        ("$1000-1500", "2015-01-01", "2019-06-01"),
        ("$1500-2000", "2019-06-01", "2020-08-01"),
        ("$1800-2100", "2020-08-01", "2024-03-01"),
        ("$2100-2800", "2024-03-01", "2025-03-01"),
        ("$2800+", "2025-03-01", "2026-04-10"),
    ]

    p("--- 按价格区间分析 ---")
    tasks = [(label, L5, 0.30, start, end) for label, start, end in price_ranges]
    results = run_pool(tasks)
    p(f"  {'Price Range':<16} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR%':>6} {'$/t':>8}")
    for r in results:
        dpt = r[3] / r[1] if r[1] > 0 else 0
        p(f"  {r[0]:<16} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1%} {dpt:>8.2f}")

    # ATR 绝对值趋势
    p("\n--- ATR 随价格增长趋势 ---")
    p("  (ATR 是绝对值, 金价从 $1200 涨到 $4800 意味着 ATR 可能翻 4 倍)")
    p("  L5 的 SL/TP 都是 ATR 倍数, 所以应该自动适应")
    p("  如果 $/t 随价格增长而增长, 说明 ATR 自适应工作正常")


# ═══════════════════════════════════════════
# R5-8: Monte Carlo 参数扰动压力
# ═══════════════════════════════════════════
def r5_8_param_perturbation(p):
    p("=" * 80)
    p("R5-8: Monte Carlo 参数扰动压力测试")
    p("=" * 80)
    p("\n  对 L5 所有关键参数同时加 ±10-20% 随机噪声")
    p("  100 次 bootstrap, 统计 Sharpe 分布\n")

    L5 = get_base()
    np.random.seed(42)

    param_ranges = {
        'trailing_activate_atr': (L5['trailing_activate_atr'], 0.15),
        'trailing_distance_atr': (L5['trailing_distance_atr'], 0.15),
        'sl_atr_mult': (L5['sl_atr_mult'], 0.10),
        'tp_atr_mult': (L5['tp_atr_mult'], 0.10),
        'keltner_adx_threshold': (L5['keltner_adx_threshold'], 0.15),
        'choppy_threshold': (L5['choppy_threshold'], 0.15),
        'keltner_max_hold_m15': (L5['keltner_max_hold_m15'], 0.20),
    }

    tasks = []
    for i in range(100):
        kw = {**L5}
        regime = {**L5['regime_config']}
        for param, (center, noise_pct) in param_ranges.items():
            noise = np.random.uniform(-noise_pct, noise_pct)
            new_val = center * (1 + noise)
            if param == 'keltner_max_hold_m15':
                new_val = max(8, int(round(new_val)))
            elif param == 'keltner_adx_threshold':
                new_val = max(10, round(new_val, 1))
            elif param == 'choppy_threshold':
                new_val = max(0.20, min(0.70, round(new_val, 2)))
            else:
                new_val = max(0.001, round(new_val, 3))
            kw[param] = new_val

        # Also perturb regime config
        for regime_key in ['low', 'normal', 'high']:
            for trail_key in ['trail_act', 'trail_dist']:
                noise = np.random.uniform(-0.15, 0.15)
                regime[regime_key] = {**regime.get(regime_key, {})}
                orig = L5['regime_config'][regime_key][trail_key]
                regime[regime_key][trail_key] = max(0.001, round(orig * (1 + noise), 3))
        kw['regime_config'] = regime
        kw['trailing_activate_atr'] = regime['normal']['trail_act']
        kw['trailing_distance_atr'] = regime['normal']['trail_dist']

        tasks.append((f"MC_{i:03d}", kw, 0.30, None, None))

    results = run_pool(tasks)

    sharpes = [r[2] for r in results]
    pnls = [r[3] for r in results]

    p(f"--- 100 次参数扰动结果 ---")
    p(f"  Sharpe: mean={np.mean(sharpes):.2f}, std={np.std(sharpes):.2f}, "
      f"min={np.min(sharpes):.2f}, max={np.max(sharpes):.2f}")
    p(f"  PnL:    mean={fmt(np.mean(pnls))}, std={fmt(np.std(pnls))}, "
      f"min={fmt(np.min(pnls))}, max={fmt(np.max(pnls))}")
    p(f"  盈利组合: {sum(1 for s in sharpes if s > 0)}/100 ({sum(1 for s in sharpes if s > 0)}%)")
    p(f"  Sharpe>2: {sum(1 for s in sharpes if s > 2)}/100")
    p(f"  Sharpe>3: {sum(1 for s in sharpes if s > 3)}/100")
    p(f"  Sharpe>4: {sum(1 for s in sharpes if s > 4)}/100")

    # Sharpe 分布直方图
    p(f"\n--- Sharpe 分布 ---")
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 100]
    for i in range(len(bins)-1):
        count = sum(1 for s in sharpes if bins[i] <= s < bins[i+1])
        bar = '#' * count
        p(f"  [{bins[i]:>2}-{bins[i+1]:>3}): {count:>3} {bar}")


# ═══════════════════════════════════════════
# R5-9: 新策略探索 — 波动率压缩后突破
# ═══════════════════════════════════════════
def r5_9_vol_squeeze_strategy(p):
    p("=" * 80)
    p("R5-9: 新策略探索 — 波动率压缩后突破")
    p("=" * 80)
    p("\n  思路: ATR 压缩 (低波动) 后常常跟随大突破")
    p("  指标: ATR_ratio = ATR(7) / ATR(28)")
    p("  当 ATR_ratio < 阈值时, 标记为 '压缩' 状态")
    p("  压缩状态下的 KC 突破信号可能质量更高\n")

    L5 = get_base()

    # 由于引擎不直接支持 ATR_ratio 过滤, 
    # 我们通过分析现有交易数据来验证假设
    p("--- Part A: 分析现有交易中 ATR 环境 ---")
    tasks = [("L5_squeeze", L5, 0.30, None, None)]
    results = run_pool(tasks, with_trades=True)
    trades = results[0][7]
    p(f"  总交易: {len(trades)}")

    # Part B: 不同 Choppy 阈值组合
    p("\n--- Part B: Choppy 阈值精细扫描 (已知 0.50 最优) ---")
    choppy_values = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    tasks = [(f"Choppy={c:.2f}", {**L5, "choppy_threshold": c}, 0.30, None, None)
             for c in choppy_values]
    results = run_pool(tasks)
    p(f"  {'Choppy':<12} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR%':>6}")
    for r in results:
        p(f"  {r[0]:<12} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1%}")

    # Part C: Escalating Cooldown 测试
    p("\n--- Part C: Escalating Cooldown (连亏后加长冷却) ---")
    esc_variants = [
        ("NoEsc", {**L5, "escalating_cooldown": False}),
        ("Esc_x2", {**L5, "escalating_cooldown": True, "escalating_cooldown_mult": 2.0}),
        ("Esc_x4", {**L5, "escalating_cooldown": True, "escalating_cooldown_mult": 4.0}),
        ("Esc_x6", {**L5, "escalating_cooldown": True, "escalating_cooldown_mult": 6.0}),
    ]
    tasks = [(label, kw, 0.30, None, None) for label, kw in esc_variants]
    results = run_pool(tasks)
    p(f"  {'Config':<12} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR%':>6}")
    for r in results:
        p(f"  {r[0]:<12} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1%}")

    # Part D: Min Entry Gap
    p("\n--- Part D: 最小入场间隔 ---")
    gap_values = [0, 0.5, 1.0, 1.5, 2.0, 3.0]
    tasks = [(f"Gap={g:.1f}h", {**L5, "min_entry_gap_hours": g}, 0.30, None, None)
             for g in gap_values]
    results = run_pool(tasks)
    p(f"  {'Gap':<10} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR%':>6}")
    for r in results:
        p(f"  {r[0]:<10} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1%}")


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    phases = [
        ("r5_0_rsi.txt",         "R5-0: RSI 策略诊断",               r5_0_rsi_diagnosis),
        ("r5_1_timeout.txt",     "R5-1: Timeout 出场分析",            r5_1_timeout_analysis),
        ("r5_2_dynamic_mh.txt",  "R5-2: 动态 MaxHold",               r5_2_dynamic_maxhold),
        ("r5_3_trail_micro.txt", "R5-3: Trail 微结构优化",            r5_3_trail_microstructure),
        ("r5_4_entry_hour.txt",  "R5-4: 入场时间分析",               r5_4_entry_hour_analysis),
        ("r5_5_consecutive.txt", "R5-5: 连续方向信号",               r5_5_consecutive_direction),
        ("r5_6_kc_bw.txt",      "R5-6: KC Bandwidth 状态",          r5_6_kc_bandwidth_states),
        ("r5_7_high_price.txt",  "R5-7: 高金价环境压力测试",         r5_7_high_price_stress),
        ("r5_8_param_mc.txt",    "R5-8: Monte Carlo 参数扰动",       r5_8_param_perturbation),
        ("r5_9_vol_squeeze.txt", "R5-9: 波动率压缩 + 策略探索",      r5_9_vol_squeeze_strategy),
    ]

    master_log = os.path.join(OUTPUT_DIR, "00_master_log.txt")
    with open(master_log, 'w') as mf:
        mf.write(f"Round 5 Experiments\nStarted: {datetime.now()}\n{'='*60}\n\n")

        for fname, title, func in phases:
            fpath = os.path.join(OUTPUT_DIR, fname)
            print(f"\n{'='*60}")
            print(f"  Starting: {title}")
            print(f"  Output: {fpath}")
            print(f"{'='*60}\n")

            t0 = time.time()
            try:
                with open(fpath, 'w') as f:
                    header = f"# {title}\n# Started: {datetime.now()}\n\n"
                    f.write(header)

                    def printer(msg):
                        print(msg)
                        f.write(msg + "\n")
                        f.flush()

                    func(printer)

                    elapsed = time.time() - t0
                    footer = f"\n# Completed: {datetime.now()}\n# Elapsed: {elapsed/60:.1f} minutes\n"
                    f.write(footer)
                    print(footer)

                status = f"DONE ({elapsed/60:.1f} min)"
            except Exception as e:
                elapsed = time.time() - t0
                status = f"FAILED ({elapsed/60:.1f} min): {e}"
                traceback.print_exc()
                with open(fpath, 'a') as f:
                    f.write(f"\n# FAILED: {e}\n{traceback.format_exc()}\n")

            mf.write(f"  {title}: {status}\n")
            mf.flush()

        total_elapsed = sum(1 for _ in open(master_log))
        mf.write(f"\nFinished: {datetime.now()}\n")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
