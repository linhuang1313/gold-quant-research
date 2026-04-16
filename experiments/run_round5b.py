#!/usr/bin/env python3
"""
Round 5B — 补跑 R5-8/R5-9 (降低并行度) + R5-10 叠加验证
=========================================================
R5-8 之前因 100 个并行进程内存溢出被 kill
修复: MAX_WORKERS=6, 分批执行

R5-10: UltraTight2 + MaxPos=1 叠加验证 (Round 5 两大发现)
"""
import sys, os, time, traceback
import multiprocessing as mp
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = "results/round5_results"
MAX_WORKERS = 10


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


def run_pool(tasks):
    with mp.Pool(min(MAX_WORKERS, len(tasks))) as pool:
        return pool.map(_run_one, tasks)


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
    tasks = []
    for fname, start, end in FOLDS:
        tasks.append((f"{prefix}Base_{fname}", base_kw, spread, start, end))
        tasks.append((f"{prefix}Var_{fname}", variant_kw, spread, start, end))
    results = run_pool(tasks)
    base_r = [r for r in results if 'Base_' in r[0]]
    var_r = [r for r in results if 'Var_' in r[0]]
    return base_r, var_r


def print_kfold_comparison(p, base_results, var_results, base_label="Baseline", var_label="Variant"):
    p(f"\n  {'Fold':<8} {base_label+' Sharpe':>18} {var_label+' Sharpe':>18} {'Delta':>10} {'Pass?':>6}")
    p(f"  {'-'*65}")
    wins = 0
    for b, v in zip(base_results, var_results):
        delta = v[2] - b[2]
        passed = "YES" if delta > 0 else "no"
        if delta > 0:
            wins += 1
        p(f"  {b[0].split('_')[-1]:<8} {b[2]:>18.2f} {v[2]:>18.2f} {delta:>+10.2f} {passed:>6}")
    p(f"\n  K-Fold: {wins}/{len(base_results)} PASS")
    return wins


# ═══════════════════════════════════════════
# R5-8: Monte Carlo 参数扰动 (修复版: 分批)
# ═══════════════════════════════════════════
def r5_8_param_perturbation(p):
    p("=" * 80)
    p("R5-8: Monte Carlo 参数扰动压力测试 (修复版)")
    p("=" * 80)
    p(f"\n  MAX_WORKERS={MAX_WORKERS}, 分 5 批 x 20 个")
    p("  对 L5 所有关键参数同时加 +/-15% 随机噪声\n")

    L5 = get_base()
    np.random.seed(42)

    all_tasks = []
    for i in range(100):
        kw = {**L5}
        regime = {}
        for rk in ['low', 'normal', 'high']:
            regime[rk] = {}
            for tk in ['trail_act', 'trail_dist']:
                orig = L5['regime_config'][rk][tk]
                regime[rk][tk] = max(0.001, round(orig * (1 + np.random.uniform(-0.15, 0.15)), 4))

        kw['regime_config'] = regime
        kw['trailing_activate_atr'] = regime['normal']['trail_act']
        kw['trailing_distance_atr'] = regime['normal']['trail_dist']
        kw['sl_atr_mult'] = round(L5['sl_atr_mult'] * (1 + np.random.uniform(-0.10, 0.10)), 2)
        kw['tp_atr_mult'] = round(L5['tp_atr_mult'] * (1 + np.random.uniform(-0.10, 0.10)), 2)
        kw['keltner_adx_threshold'] = max(10, round(L5['keltner_adx_threshold'] * (1 + np.random.uniform(-0.15, 0.15)), 1))
        kw['choppy_threshold'] = max(0.20, min(0.70, round(L5['choppy_threshold'] * (1 + np.random.uniform(-0.15, 0.15)), 2)))
        mh = L5['keltner_max_hold_m15'] * (1 + np.random.uniform(-0.20, 0.20))
        kw['keltner_max_hold_m15'] = max(8, int(round(mh)))

        all_tasks.append((f"MC_{i:03d}", kw, 0.30, None, None))

    # Run in batches of 20
    all_results = []
    batch_size = 20
    for batch_idx in range(0, 100, batch_size):
        batch = all_tasks[batch_idx:batch_idx+batch_size]
        p(f"  Batch {batch_idx//batch_size+1}/5 ({batch_idx}-{batch_idx+len(batch)-1})...")
        results = run_pool(batch)
        all_results.extend(results)
        p(f"    done ({len(all_results)}/100)")

    sharpes = [r[2] for r in all_results]
    pnls = [r[3] for r in all_results]

    p(f"\n--- 100 次参数扰动结果 ---")
    p(f"  Sharpe: mean={np.mean(sharpes):.2f}, std={np.std(sharpes):.2f}, "
      f"min={np.min(sharpes):.2f}, max={np.max(sharpes):.2f}")
    p(f"  PnL:    mean={fmt(np.mean(pnls))}, std={fmt(np.std(pnls))}, "
      f"min={fmt(np.min(pnls))}, max={fmt(np.max(pnls))}")
    p(f"  盈利组合: {sum(1 for s in sharpes if s > 0)}/100")
    p(f"  Sharpe>2: {sum(1 for s in sharpes if s > 2)}/100")
    p(f"  Sharpe>3: {sum(1 for s in sharpes if s > 3)}/100")
    p(f"  Sharpe>4: {sum(1 for s in sharpes if s > 4)}/100")

    p(f"\n--- Sharpe 分布 ---")
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 100]
    for i in range(len(bins)-1):
        count = sum(1 for s in sharpes if bins[i] <= s < bins[i+1])
        bar = '#' * count
        p(f"  [{bins[i]:>2}-{bins[i+1]:>3}): {count:>3} {bar}")


# ═══════════════════════════════════════════
# R5-9: 策略探索 (修复版)
# ═══════════════════════════════════════════
def r5_9_strategy_explore(p):
    p("=" * 80)
    p("R5-9: 策略探索 — Choppy/Cooldown/Gap 精调")
    p("=" * 80)

    L5 = get_base()

    # Part A: Choppy
    p("\n--- Part A: Choppy 阈值精细扫描 ---")
    choppy_values = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    tasks = [(f"Choppy={c:.2f}", {**L5, "choppy_threshold": c}, 0.30, None, None)
             for c in choppy_values]
    results = run_pool(tasks)
    p(f"  {'Choppy':<12} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR%':>6}")
    for r in results:
        p(f"  {r[0]:<12} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1%}")

    # Part B: Escalating Cooldown
    p("\n--- Part B: Escalating Cooldown ---")
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

    # Part C: Min Entry Gap
    p("\n--- Part C: 最小入场间隔 ---")
    gap_values = [0, 0.5, 1.0, 1.5, 2.0, 3.0]
    tasks = [(f"Gap={g:.1f}h", {**L5, "min_entry_gap_hours": g}, 0.30, None, None)
             for g in gap_values]
    results = run_pool(tasks)
    p(f"  {'Gap':<10} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR%':>6}")
    for r in results:
        p(f"  {r[0]:<10} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1%}")


# ═══════════════════════════════════════════
# R5-10: UltraTight2 + MaxPos=1 叠加验证
# ═══════════════════════════════════════════
def r5_10_l6_candidate(p):
    p("=" * 80)
    p("R5-10: L6 候选 — UltraTight2 + MaxPos=1 叠加验证")
    p("=" * 80)
    p("\n  R5-3: UltraTight2 K-Fold 6/6 (Sharpe 4.07→6.34)")
    p("  R5-5: MaxPos=1 K-Fold 6/6 (Sharpe 4.07→4.50)")
    p("  问题: 两者叠加是否 1+1>2 ?\n")

    L5 = get_base()

    ULTRA2 = {
        'low': {'trail_act': 0.30, 'trail_dist': 0.06},
        'normal': {'trail_act': 0.20, 'trail_dist': 0.04},
        'high': {'trail_act': 0.08, 'trail_dist': 0.01},
    }

    configs = [
        ("L5_Baseline", L5),
        ("UltraTight2", {**L5, "regime_config": ULTRA2,
                         "trailing_activate_atr": 0.20, "trailing_distance_atr": 0.04}),
        ("MaxPos1", {**L5, "max_positions": 1}),
        ("L6_combo", {**L5, "regime_config": ULTRA2,
                      "trailing_activate_atr": 0.20, "trailing_distance_atr": 0.04,
                      "max_positions": 1}),
    ]

    # Part A: 全样本 + $0.30/$0.50 spread
    p("--- Part A: 全样本对比 ---")
    for spread in [0.30, 0.50]:
        p(f"\n  Spread = ${spread:.2f}")
        tasks = [(label, kw, spread, None, None) for label, kw in configs]
        results = run_pool(tasks)
        p(f"  {'Config':<16} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR%':>6} {'MaxDD':>10}")
        for r in results:
            p(f"  {r[0]:<16} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1%} {fmt(r[6]):>10}")

    # Part B: K-Fold L6 vs L5
    p("\n--- Part B: K-Fold L6 combo vs L5 ---")
    l6_kw = {**L5, "regime_config": ULTRA2,
             "trailing_activate_atr": 0.20, "trailing_distance_atr": 0.04,
             "max_positions": 1}
    base_r, var_r = run_kfold_pool(L5, l6_kw, prefix="L6_")
    wins = print_kfold_comparison(p, base_r, var_r, "L5", "L6_combo")

    # Part C: K-Fold with $0.50 spread
    p("\n--- Part C: K-Fold L6 vs L5 ($0.50 spread) ---")
    base_r50, var_r50 = run_kfold_pool(L5, l6_kw, spread=0.50, prefix="L6_50_")
    wins50 = print_kfold_comparison(p, base_r50, var_r50, "L5 $0.50", "L6 $0.50")

    # Part D: OOS Walk-Forward
    p("\n--- Part D: Anchored Walk-Forward (L6) ---")
    p(f"  {'Train':<25} {'Test':>6} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR%':>6}")
    years = list(range(2016, 2027))
    tasks = []
    for test_year in years:
        end_str = f"{test_year+1}-01-01" if test_year < 2026 else "2026-04-10"
        tasks.append((f"OOS_{test_year}", l6_kw, 0.30,
                      f"{test_year}-01-01", end_str))
    results = run_pool(tasks)
    profit_years = 0
    for r in results:
        year = int(r[0].split('_')[1])
        dpt = r[3] / r[1] if r[1] > 0 else 0
        p(f"  Train 2015-{year-1}     {year:>6} {r[1]:>6} {r[2]:>8.2f} {fmt(r[3]):>12} {r[4]:>6.1%}")
        if r[3] > 0:
            profit_years += 1
    p(f"\n  盈利年份: {profit_years}/{len(results)}")

    # Part E: 最终结论
    p(f"\n--- 结论 ---")
    if wins >= 5 and wins50 >= 4:
        p(f"  L6 combo K-Fold {wins}/6 ($0.30) + {wins50}/6 ($0.50) → 强烈推荐部署")
    elif wins >= 4:
        p(f"  L6 combo K-Fold {wins}/6 ($0.30) + {wins50}/6 ($0.50) → 值得考虑")
    else:
        p(f"  L6 combo K-Fold {wins}/6 → 不推荐叠加, 单独使用更好")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    phases = [
        ("r5_8_param_mc.txt",     "R5-8: Monte Carlo 参数扰动 (修复)", r5_8_param_perturbation),
        ("r5_9_vol_squeeze.txt",  "R5-9: 策略探索",                    r5_9_strategy_explore),
        ("r5_10_l6_candidate.txt","R5-10: L6 候选叠加验证",            r5_10_l6_candidate),
    ]

    master_log = os.path.join(OUTPUT_DIR, "00_master_log.txt")
    with open(master_log, 'a') as mf:
        mf.write(f"\n--- Round 5B (补跑) Started: {datetime.now()} ---\n")

        for fname, title, func in phases:
            fpath = os.path.join(OUTPUT_DIR, fname)
            print(f"\n{'='*60}")
            print(f"  Starting: {title}")
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

        mf.write(f"\nRound 5B Finished: {datetime.now()}\n")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
