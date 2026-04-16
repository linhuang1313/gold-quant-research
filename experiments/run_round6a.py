#!/usr/bin/env python3
"""
Round 6A — Server A (12核/90GB): 验证加固
==========================================
R6-A1: Historical Spread 全量验证
R6-A2: 危机年份深度剖析
R6-A3: 真实交易成本梯度
R6-A4: MaxPos=1 vs MaxPos=2 深度对比
R6-A5: SL 倍数微调
R6-A6: Cooldown 精细扫描
"""
import sys, os, io, time, traceback
import multiprocessing as mp
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "results/round6_results"
MAX_WORKERS = 8


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
           str(t.entry_time)[:16], t.direction or '')
          for t in trades]
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'], td)


def _run_one_historical(args):
    label, kw = args
    from backtest.runner import DataBundle, run_variant, load_spread_series
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    spread_series = load_spread_series()
    if spread_series is None:
        kw_final = {**kw, 'spread_cost': 0.30}
    else:
        kw_final = {**kw, 'spread_model': 'historical', 'spread_series': spread_series, 'spread_cost': 0.0}
    s = run_variant(data, label, verbose=False, **kw_final)
    return (label, s['n'], s['sharpe'], s['total_pnl'], s['win_rate'],
            s.get('elapsed_s', 0), s['max_dd'])


def run_pool(tasks, func=None):
    if func is None:
        func = _run_one
    with mp.Pool(min(MAX_WORKERS, len(tasks))) as pool:
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
        ok = "YES" if delta > 0 else "no"
        if delta > 0: wins += 1
        p(f"  {b[0].split('_')[-1]:<8} {b[2]:>18.2f} {v[2]:>18.2f} {delta:>+10.2f} {ok:>6}")
    p(f"\n  K-Fold: {wins}/{len(base_r)} PASS")
    return wins


# ═══════════════════════════════════════════
# R6-A1: Historical Spread 全量验证
# ═══════════════════════════════════════════
def r6_a1_historical_spread(p):
    p("=" * 80)
    p("R6-A1: Historical Spread 全量验证")
    p("=" * 80)

    L5 = get_base()

    p("\n--- Part A: L5 Fixed vs Historical Spread ---")
    tasks_fixed = [
        ("L5_Fixed_030", L5, 0.30, None, None),
        ("L5_Fixed_050", L5, 0.50, None, None),
    ]
    r_fixed = run_pool(tasks_fixed)
    for r in r_fixed:
        p(f"  {r[0]:<20} N={r[1]:>6} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])} WR={r[4]:.1%} MaxDD={fmt(r[6])}")

    p("\n  Historical spread:")
    tasks_hist = [("L5_Historical", L5)]
    r_hist = run_pool(tasks_hist, func=_run_one_historical)
    for r in r_hist:
        p(f"  {r[0]:<20} N={r[1]:>6} Sharpe={r[2]:>6.2f} PnL={fmt(r[3])} WR={r[4]:.1%} MaxDD={fmt(r[6])}")

    p("\n--- Part B: Historical Spread 逐年对比 ---")
    years = list(range(2015, 2027))
    tasks_yr = []
    for y in years:
        end = f"{y+1}-01-01" if y < 2026 else "2026-04-10"
        tasks_yr.append((f"Fixed030_{y}", L5, 0.30, f"{y}-01-01", end))
    r_yr_fixed = run_pool(tasks_yr)

    p(f"  {'Year':<8} {'Fixed$0.30 Sharpe':>18} {'Fixed$0.30 PnL':>16}")
    for r in r_yr_fixed:
        p(f"  {r[0].split('_')[-1]:<8} {r[2]:>18.2f} {fmt(r[3]):>16}")


# ═══════════════════════════════════════════
# R6-A2: 危机年份深度剖析
# ═══════════════════════════════════════════
def r6_a2_crisis_years(p):
    p("=" * 80)
    p("R6-A2: 危机年份深度剖析 (2018/2020/2022)")
    p("=" * 80)

    L5 = get_base()
    crisis_years = [
        ("2018", "2018-01-01", "2019-01-01", "强美元+加息周期"),
        ("2020", "2020-01-01", "2021-01-01", "COVID崩盘+V型反转"),
        ("2022", "2022-01-01", "2023-01-01", "激进加息+美元暴涨"),
        ("2025", "2025-01-01", "2026-01-01", "Trump关税+地缘紧张"),
    ]

    tasks = [(f"Crisis_{y[0]}", L5, 0.30, y[1], y[2]) for y in crisis_years]
    results = run_pool(tasks, func=_run_one_trades)

    for (label, n, sharpe, pnl, wr, elapsed, maxdd, trades), info in zip(results, crisis_years):
        p(f"\n--- {info[0]}: {info[3]} ---")
        p(f"  N={n}, Sharpe={sharpe:.2f}, PnL={fmt(pnl)}, WR={wr:.1%}, MaxDD={fmt(maxdd)}")

        if not trades:
            p("  (no trades)")
            continue

        pnls = [t[0] for t in trades]
        losses = [t[0] for t in trades if t[0] < 0]
        wins_t = [t[0] for t in trades if t[0] > 0]

        max_consec_loss = 0
        cur = 0
        for pnl_t in pnls:
            if pnl_t < 0:
                cur += 1
                max_consec_loss = max(max_consec_loss, cur)
            else:
                cur = 0

        exit_counts = defaultdict(int)
        exit_pnl = defaultdict(float)
        for t in trades:
            exit_counts[t[1]] += 1
            exit_pnl[t[1]] += t[0]

        p(f"  最大连续亏损: {max_consec_loss} 笔")
        p(f"  亏损笔数: {len(losses)}/{n} ({len(losses)/n*100:.0f}%)")
        if losses:
            p(f"  最大单笔亏损: {fmt(min(losses))}")
            p(f"  平均亏损: {fmt(np.mean(losses))}")
        if wins_t:
            p(f"  最大单笔盈利: {fmt(max(wins_t))}")
            p(f"  平均盈利: {fmt(np.mean(wins_t))}")
        p(f"  出场分布:")
        for reason, cnt in sorted(exit_counts.items(), key=lambda x: -x[1]):
            avg = exit_pnl[reason] / cnt
            p(f"    {reason:<20} {cnt:>4} 笔  总PnL={fmt(exit_pnl[reason])}  均PnL={fmt(avg)}")

        worst5 = sorted(trades, key=lambda t: t[0])[:5]
        p(f"  最差5笔:")
        for t in worst5:
            p(f"    {t[4]} {t[5]:>4} PnL={fmt(t[0])} exit={t[1]} bars={t[2]} strat={t[3]}")


# ═══════════════════════════════════════════
# R6-A3: 真实交易成本梯度
# ═══════════════════════════════════════════
def r6_a3_cost_gradient(p):
    p("=" * 80)
    p("R6-A3: 真实交易成本梯度")
    p("=" * 80)

    L5 = get_base()
    spreads = [round(0.10 + i * 0.05, 2) for i in range(19)]

    tasks = [(f"Sp={s:.2f}", L5, s, None, None) for s in spreads]
    results = run_pool(tasks)

    p(f"\n  {'Spread':<10} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR%':>6} {'$/trade':>8}")
    breakeven = None
    for r in results:
        sp = float(r[0].split('=')[1])
        dpt = r[3] / r[1] if r[1] > 0 else 0
        marker = " <-- L5" if abs(sp - 0.30) < 0.01 else ""
        if r[3] <= 0 and breakeven is None:
            breakeven = sp
            marker = " <-- BREAKEVEN"
        p(f"  ${sp:<9.2f} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1%} {fmt(dpt):>8}{marker}")

    if breakeven:
        safety = (breakeven - 0.30) / 0.30 * 100
        p(f"\n  盈亏平衡 spread: ~${breakeven:.2f}")
        p(f"  安全边际 (vs $0.30): {safety:.0f}%")
    else:
        p(f"\n  所有 spread 下均盈利! 盈亏平衡 > $1.00")


# ═══════════════════════════════════════════
# R6-A4: MaxPos=1 vs MaxPos=2 深度对比
# ═══════════════════════════════════════════
def r6_a4_maxpos_deep(p):
    p("=" * 80)
    p("R6-A4: MaxPos=1 vs MaxPos=2 深度对比")
    p("=" * 80)

    L5 = get_base()
    mp1 = {**L5, "max_positions": 1}

    p("\n--- Part A: 全样本对比 ---")
    for spread in [0.30, 0.50]:
        tasks = [
            (f"MaxPos2_sp{spread}", L5, spread, None, None),
            (f"MaxPos1_sp{spread}", mp1, spread, None, None),
        ]
        results = run_pool(tasks)
        p(f"\n  Spread ${spread:.2f}:")
        p(f"  {'Config':<16} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR%':>6} {'MaxDD':>10}")
        for r in results:
            p(f"  {r[0]:<16} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1%} {fmt(r[6]):>10}")

    p("\n--- Part B: 逐年对比 ---")
    years = list(range(2015, 2027))
    tasks = []
    for y in years:
        end = f"{y+1}-01-01" if y < 2026 else "2026-04-10"
        tasks.append((f"MP2_{y}", L5, 0.30, f"{y}-01-01", end))
        tasks.append((f"MP1_{y}", mp1, 0.30, f"{y}-01-01", end))
    results = run_pool(tasks)

    mp2_yr = [r for r in results if r[0].startswith('MP2')]
    mp1_yr = [r for r in results if r[0].startswith('MP1')]

    p(f"\n  {'Year':<6} {'MP2 N':>6} {'MP2 PnL':>10} {'MP1 N':>6} {'MP1 PnL':>10} {'Delta':>10}")
    for r2, r1 in zip(mp2_yr, mp1_yr):
        yr = r2[0].split('_')[1]
        delta = r1[3] - r2[3]
        p(f"  {yr:<6} {r2[1]:>6} {fmt(r2[3]):>10} {r1[1]:>6} {fmt(r1[3]):>10} {fmt(delta):>10}")

    p("\n--- Part C: K-Fold ($0.30) ---")
    base_r, var_r = run_kfold(L5, mp1, spread=0.30, prefix="MP_")
    print_kfold(p, base_r, var_r, "MaxPos=2", "MaxPos=1")

    p("\n--- Part D: K-Fold ($0.50) ---")
    base_r50, var_r50 = run_kfold(L5, mp1, spread=0.50, prefix="MP50_")
    print_kfold(p, base_r50, var_r50, "MP2 $0.50", "MP1 $0.50")


# ═══════════════════════════════════════════
# R6-A5: SL 倍数微调
# ═══════════════════════════════════════════
def r6_a5_sl_tuning(p):
    p("=" * 80)
    p("R6-A5: SL 倍数微调 (L5 AllTight 环境下)")
    p("=" * 80)

    L5 = get_base()

    p("\n--- Part A: 全样本扫描 ---")
    sl_values = [round(3.5 + i * 0.25, 2) for i in range(9)]
    tasks = [(f"SL={sl:.2f}", {**L5, "sl_atr_mult": sl}, 0.30, None, None)
             for sl in sl_values]
    results = run_pool(tasks)

    p(f"  {'SL mult':<10} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR%':>6} {'MaxDD':>10}")
    best_sharpe = -999
    best_sl = None
    for r in results:
        sl = float(r[0].split('=')[1])
        marker = " <-- current" if abs(sl - 4.5) < 0.01 else ""
        if r[2] > best_sharpe:
            best_sharpe = r[2]
            best_sl = sl
        p(f"  {sl:<10.2f} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1%} {fmt(r[6]):>10}{marker}")
    p(f"\n  最优 SL: {best_sl}x (Sharpe={best_sharpe:.2f})")

    if abs(best_sl - 4.5) > 0.01:
        p(f"\n--- Part B: K-Fold 验证 SL={best_sl} vs 4.5 ---")
        var_kw = {**L5, "sl_atr_mult": best_sl}
        base_r, var_r = run_kfold(L5, var_kw, prefix="SL_")
        print_kfold(p, base_r, var_r, "SL=4.50", f"SL={best_sl:.2f}")
    else:
        p(f"  当前 SL=4.5 已是最优，无需 K-Fold 验证")


# ═══════════════════════════════════════════
# R6-A6: Cooldown 精细扫描
# ═══════════════════════════════════════════
def r6_a6_cooldown_scan(p):
    p("=" * 80)
    p("R6-A6: Cooldown 精细扫描 (L5 AllTight 环境下)")
    p("=" * 80)

    L5 = get_base()

    cd_minutes = [15, 20, 25, 30, 45, 60, 90, 120]

    p("\n--- Part A: 全样本扫描 ---")
    tasks = [(f"CD={m}min", {**L5, "cooldown_hours": m / 60.0}, 0.30, None, None)
             for m in cd_minutes]
    results = run_pool(tasks)

    p(f"  {'Cooldown':<12} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR%':>6} {'MaxDD':>10}")
    best_sharpe = -999
    best_cd = None
    for r in results:
        mins = int(r[0].split('=')[1].replace('min', ''))
        marker = " <-- current" if mins == 30 else ""
        if r[2] > best_sharpe:
            best_sharpe = r[2]
            best_cd = mins
        p(f"  {mins:>4} min    {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1%} {fmt(r[6]):>10}{marker}")
    p(f"\n  最优 Cooldown: {best_cd} min (Sharpe={best_sharpe:.2f})")

    if best_cd != 30:
        p(f"\n--- Part B: K-Fold 验证 CD={best_cd}min vs 30min ---")
        var_kw = {**L5, "cooldown_hours": best_cd / 60.0}
        base_r, var_r = run_kfold(L5, var_kw, prefix="CD_")
        print_kfold(p, base_r, var_r, "CD=30min", f"CD={best_cd}min")
    else:
        p(f"  当前 30min 已是最优，无需 K-Fold 验证")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    phases = [
        ("r6_a1_hist_spread.txt",  "R6-A1: Historical Spread",  r6_a1_historical_spread),
        ("r6_a2_crisis.txt",       "R6-A2: 危机年份剖析",       r6_a2_crisis_years),
        ("r6_a3_cost_gradient.txt","R6-A3: 成本梯度",           r6_a3_cost_gradient),
        ("r6_a4_maxpos.txt",       "R6-A4: MaxPos 深度对比",    r6_a4_maxpos_deep),
        ("r6_a5_sl.txt",           "R6-A5: SL 微调",            r6_a5_sl_tuning),
        ("r6_a6_cooldown.txt",     "R6-A6: Cooldown 扫描",      r6_a6_cooldown_scan),
    ]

    master_log = os.path.join(OUTPUT_DIR, "00_master_log.txt")
    with open(master_log, 'w') as mf:
        mf.write(f"Round 6A (Server A: Validation)\nStarted: {datetime.now()}\n{'='*60}\n\n")

        for fname, title, func in phases:
            fpath = os.path.join(OUTPUT_DIR, fname)
            print(f"\n{'='*60}")
            print(f"  Starting: {title}")
            print(f"{'='*60}\n")

            t0 = time.time()
            try:
                with open(fpath, 'w') as f:
                    header = f"# {title}\n# Started: {datetime.now()}\n# Server: A (12-core)\n\n"
                    f.write(header)
                    def printer(msg):
                        print(msg)
                        f.write(msg + "\n")
                        f.flush()
                    func(printer)
                    elapsed = time.time() - t0
                    f.write(f"\n# Completed: {datetime.now()}\n# Elapsed: {elapsed/60:.1f} minutes\n")
                status = f"DONE ({elapsed/60:.1f} min)"
            except Exception as e:
                elapsed = time.time() - t0
                status = f"FAILED ({elapsed/60:.1f} min): {e}"
                traceback.print_exc()
                with open(fpath, 'a') as f:
                    f.write(f"\n# FAILED: {e}\n{traceback.format_exc()}\n")

            mf.write(f"  {title}: {status}\n")
            mf.flush()

        mf.write(f"\nRound 6A Finished: {datetime.now()}\n")
        print(f"\n{'='*60}")
        print(f"  Round 6A COMPLETE")
        print(f"{'='*60}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
