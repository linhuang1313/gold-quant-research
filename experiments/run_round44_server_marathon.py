"""R44: 15-Hour Server Marathon — 6-Phase Deep Research
=====================================================
服务器 15 小时实验包。覆盖 6 个高价值、从未探索过的方向。

Phase A: L8_BASE 部署验证 (当前实盘配置 vs R43 V0)      ~2h
Phase B: Spread 敏感度精密测量                           ~2h
Phase C: 多策略组合 Walk-Forward (最终部署组合)           ~3h
Phase D: 尾部风险压力测试 (2020 COVID / 2022加息 / 极端) ~2h
Phase E: 参数衰变监测 (最近2年 vs 前9年)                 ~2h
Phase F: 最优组合 Monte Carlo 生存性测试                  ~3h

USAGE
-----
    cd /root/gold-quant-research
    nohup python3 -u experiments/run_round44_server_marathon.py > results/r44_marathon.log 2>&1 &
"""
import sys, os, time, json, traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import research_config as config
from backtest.engine import BacktestEngine, TradeRecord
from backtest.runner import (
    DataBundle, run_variant, run_kfold,
    LIVE_PARITY_KWARGS, sanitize_for_json
)
from backtest.stats import calc_stats, print_comparison

OUT_DIR = ROOT / "results" / "round44_marathon"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARATHON_START = time.time()

# ══════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            try: f.write(data)
            except: pass
            try: f.flush()
            except: pass
    def flush(self):
        for f in self.files:
            try: f.flush()
            except: pass

def save_json(data, filename):
    path = OUT_DIR / filename
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")

def elapsed():
    return f"[{(time.time()-MARATHON_START)/60:.1f} min]"

def phase_header(name, desc):
    print(f"\n{'='*70}")
    print(f"  {name}: {desc}")
    print(f"  {elapsed()}")
    print(f"{'='*70}\n", flush=True)


# ══════════════════════════════════════════════════════════════
# Strategy configs
# ══════════════════════════════════════════════════════════════

L8_BASE = {
    **LIVE_PARITY_KWARGS,
    'keltner_adx_threshold': 14,
    'regime_config': {
        'low':    {'trail_act': 0.22, 'trail_dist': 0.04},
        'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
        'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
    },
    'keltner_max_hold_m15': 20,  # L8: MH=20 M15 bars (= 5h)
    'time_decay_tp': False,      # L8: TATrail OFF
    'min_entry_gap_hours': 1.0,
}

L7_BEST = {
    **LIVE_PARITY_KWARGS,
    'regime_config': {
        'low':    {'trail_act': 0.30, 'trail_dist': 0.06},
        'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
        'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
    },
    'keltner_max_hold_m15': 8,
    'time_decay_tp': True,
    'time_adaptive_trail': True,
    'time_adaptive_trail_start': 2,
    'time_adaptive_trail_decay': 0.75,
    'time_adaptive_trail_floor': 0.003,
    'min_entry_gap_hours': 1.0,
}


# ══════════════════════════════════════════════════════════════
# Phase A: L8_BASE 部署验证 — 当前实盘 vs L7_BEST 的差距到底多大？
# ══════════════════════════════════════════════════════════════

def phase_a(data):
    phase_header("Phase A", "L8_BASE(当前实盘) vs L7_BEST(最强组合) — 全样本 + 6-Fold")

    variants = {
        'L8_BASE_current': L8_BASE,
        'L7_BEST_R35': L7_BEST,
        'L8_MH8': {**L8_BASE, 'keltner_max_hold_m15': 8},
        'L8_TATrail': {**L8_BASE, 'time_adaptive_trail': True,
                       'time_adaptive_trail_start': 2, 'time_adaptive_trail_decay': 0.75,
                       'time_adaptive_trail_floor': 0.003},
        'L8_MH8_TATrail': {**L8_BASE, 'keltner_max_hold_m15': 8,
                           'time_adaptive_trail': True, 'time_adaptive_trail_start': 2,
                           'time_adaptive_trail_decay': 0.75, 'time_adaptive_trail_floor': 0.003},
    }

    # Full sample
    print("--- Full Sample ---")
    full_results = []
    for name, kw in variants.items():
        stats = run_variant(data, name, verbose=True, **kw)
        full_results.append(stats)
        print(f"  {name}: Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}, "
              f"N={stats['n']}, WR={stats['win_rate']:.1f}%, MaxDD=${stats['max_dd']:.0f}")

    print_comparison(full_results)

    # K-Fold for top 3
    print("\n--- K-Fold 6/6 ---")
    kfold_results = {}
    for name in ['L8_BASE_current', 'L8_MH8_TATrail', 'L7_BEST_R35']:
        kw = variants[name]
        folds = run_kfold(data, kw, n_folds=6, label_prefix=f"{name}_")
        sharpes = [f['sharpe'] for f in folds]
        pass_count = sum(1 for s in sharpes if s > 0)
        kfold_results[name] = {
            'folds': sanitize_for_json(folds),
            'sharpes': sharpes,
            'mean_sharpe': np.mean(sharpes),
            'min_sharpe': min(sharpes),
            'pass': f"{pass_count}/6",
        }
        print(f"  {name}: {pass_count}/6 PASS, mean={np.mean(sharpes):.2f}, "
              f"min={min(sharpes):.2f}, max={max(sharpes):.2f}")

    save_json({
        'full_sample': sanitize_for_json(full_results),
        'kfold': kfold_results,
    }, 'phase_a_l8_vs_l7.json')
    print(f"\n  Phase A complete. {elapsed()}")


# ══════════════════════════════════════════════════════════════
# Phase B: Spread 敏感度精密测量
# 4 周后用真实 spread 校准时需要的基准线
# ══════════════════════════════════════════════════════════════

def phase_b(data):
    phase_header("Phase B", "Spread 敏感度 — L8_BASE 在 $0.20 ~ $2.00 的 Sharpe 衰减曲线")

    spreads = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
               1.00, 1.20, 1.50, 2.00]

    results = []
    for sp in spreads:
        kw = {**L8_BASE, 'spread_cost': sp}
        stats = run_variant(data, f"L8_sp{sp:.2f}", verbose=False, **kw)
        results.append({
            'spread': sp, 'sharpe': stats['sharpe'],
            'total_pnl': stats['total_pnl'], 'n': stats['n'],
            'win_rate': stats['win_rate'], 'max_dd': stats['max_dd'],
        })
        print(f"  spread=${sp:.2f}: Sharpe={stats['sharpe']:.2f}, "
              f"PnL=${stats['total_pnl']:.0f}, N={stats['n']}")

    # L7_BEST comparison
    print("\n--- L7_BEST spread sensitivity ---")
    l7_results = []
    for sp in spreads:
        kw = {**L7_BEST, 'spread_cost': sp}
        stats = run_variant(data, f"L7best_sp{sp:.2f}", verbose=False, **kw)
        l7_results.append({
            'spread': sp, 'sharpe': stats['sharpe'],
            'total_pnl': stats['total_pnl'], 'n': stats['n'],
        })
        print(f"  spread=${sp:.2f}: Sharpe={stats['sharpe']:.2f}, "
              f"PnL=${stats['total_pnl']:.0f}")

    # Break-even spread (Sharpe > 0)
    for r in results:
        if r['sharpe'] <= 0:
            print(f"\n  L8_BASE break-even spread: ~${r['spread']:.2f}")
            break
    else:
        print(f"\n  L8_BASE profitable at all tested spreads (up to $2.00)")

    save_json({'L8_BASE': results, 'L7_BEST': l7_results}, 'phase_b_spread_sensitivity.json')
    print(f"\n  Phase B complete. {elapsed()}")


# ══════════════════════════════════════════════════════════════
# Phase C: 多策略组合 Walk-Forward (22 个半年窗口)
# R36 只用 L7+H1filter 做了 WF, 从未用 L8_BASE 做过
# ══════════════════════════════════════════════════════════════

def phase_c(data):
    phase_header("Phase C", "Walk-Forward 22 半年窗口 — L8_BASE vs L8_full_upgrade vs L7_BEST")

    # 22 half-year windows: 2015H1 ~ 2025H2
    windows = []
    for year in range(2015, 2026):
        windows.append((f"{year}H1", f"{year}-01-01", f"{year}-07-01"))
        if year < 2026:
            windows.append((f"{year}H2", f"{year}-07-01", f"{year+1}-01-01"))

    strategies = {
        'L8_BASE': L8_BASE,
        'L8_MH8_TATrail': {**L8_BASE, 'keltner_max_hold_m15': 8,
                           'time_adaptive_trail': True, 'time_adaptive_trail_start': 2,
                           'time_adaptive_trail_decay': 0.75, 'time_adaptive_trail_floor': 0.003},
        'L7_BEST': L7_BEST,
    }

    all_results = {}
    for strat_name, kw in strategies.items():
        print(f"\n  --- {strat_name} ---")
        strat_results = []
        for wname, wstart, wend in windows:
            try:
                wdata = data.slice(wstart, wend)
                if len(wdata.m15_df) < 500:
                    print(f"    {wname}: SKIP (too few bars)")
                    continue
                stats = run_variant(wdata, f"{strat_name}_{wname}", verbose=False, **kw)
                strat_results.append({
                    'window': wname, 'start': wstart, 'end': wend,
                    'sharpe': stats['sharpe'], 'total_pnl': stats['total_pnl'],
                    'n': stats['n'], 'win_rate': stats['win_rate'],
                    'max_dd': stats['max_dd'],
                })
                print(f"    {wname}: Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}, N={stats['n']}")
            except Exception as e:
                print(f"    {wname}: ERROR {e}")

        positive = sum(1 for r in strat_results if r['sharpe'] > 0)
        total = len(strat_results)
        sharpes = [r['sharpe'] for r in strat_results]
        print(f"  {strat_name} summary: {positive}/{total} positive, "
              f"mean Sharpe={np.mean(sharpes):.2f}, min={min(sharpes):.2f}")
        all_results[strat_name] = strat_results

    save_json(all_results, 'phase_c_walk_forward.json')
    print(f"\n  Phase C complete. {elapsed()}")


# ══════════════════════════════════════════════════════════════
# Phase D: 尾部风险压力测试 — 黑天鹅时期表现
# ══════════════════════════════════════════════════════════════

def phase_d(data):
    phase_header("Phase D", "尾部风险压力测试 — 6 个危机/极端时期")

    crisis_windows = [
        ("2020_COVID_crash",    "2020-02-01", "2020-05-01"),  # COVID 黄金暴跌→暴涨
        ("2020_gold_ATH",       "2020-07-01", "2020-09-01"),  # 黄金突破 2000 后回落
        ("2022_rate_hike",      "2022-03-01", "2022-07-01"),  # Fed 快速加息, 黄金承压
        ("2022_DXY_peak",       "2022-09-01", "2022-11-30"),  # DXY 见顶, 黄金见底
        ("2023_banking_crisis", "2023-03-01", "2023-04-30"),  # SVB/CS, 避险推金上涨
        ("2024_geopolitics",    "2024-01-01", "2024-04-30"),  # 中东+大选预期, 黄金 ATH
    ]

    strategies = {
        'L8_BASE': L8_BASE,
        'L8_MH8_TATrail': {**L8_BASE, 'keltner_max_hold_m15': 8,
                           'time_adaptive_trail': True, 'time_adaptive_trail_start': 2,
                           'time_adaptive_trail_decay': 0.75, 'time_adaptive_trail_floor': 0.003},
        'L7_BEST': L7_BEST,
    }

    all_results = {}
    for strat_name, kw in strategies.items():
        print(f"\n  --- {strat_name} ---")
        crisis_results = []
        for cname, cstart, cend in crisis_windows:
            try:
                cdata = data.slice(cstart, cend)
                if len(cdata.m15_df) < 200:
                    print(f"    {cname}: SKIP")
                    continue
                stats = run_variant(cdata, f"{strat_name}_{cname}", verbose=False, **kw)
                crisis_results.append({
                    'crisis': cname, 'start': cstart, 'end': cend,
                    'sharpe': stats['sharpe'], 'total_pnl': stats['total_pnl'],
                    'n': stats['n'], 'win_rate': stats['win_rate'],
                    'max_dd': stats['max_dd'],
                })
                print(f"    {cname}: Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}, "
                      f"N={stats['n']}, WR={stats['win_rate']:.1f}%, MaxDD=${stats['max_dd']:.0f}")
            except Exception as e:
                print(f"    {cname}: ERROR {e}")

        all_results[strat_name] = crisis_results
        survived = sum(1 for r in crisis_results if r['total_pnl'] > 0)
        print(f"  {strat_name}: {survived}/{len(crisis_results)} crises profitable")

    save_json(all_results, 'phase_d_crisis_stress.json')
    print(f"\n  Phase D complete. {elapsed()}")


# ══════════════════════════════════════════════════════════════
# Phase E: 参数衰变监测 — 最近 2 年 vs 前 9 年
# 核心问题: alpha 是否在衰退？
# ══════════════════════════════════════════════════════════════

def phase_e(data):
    phase_header("Phase E", "Alpha 衰变检测 — 逐年 Sharpe + 最近2年 vs 前9年")

    strategies = {
        'L8_BASE': L8_BASE,
        'L7_BEST': L7_BEST,
    }

    yearly_windows = [(str(y), f"{y}-01-01", f"{y+1}-01-01") for y in range(2015, 2026)]

    all_results = {}
    for strat_name, kw in strategies.items():
        print(f"\n  --- {strat_name} ---")
        yearly = []
        for yname, ystart, yend in yearly_windows:
            try:
                ydata = data.slice(ystart, yend)
                if len(ydata.m15_df) < 500:
                    continue
                stats = run_variant(ydata, f"{strat_name}_{yname}", verbose=False, **kw)
                yearly.append({
                    'year': yname, 'sharpe': stats['sharpe'],
                    'total_pnl': stats['total_pnl'], 'n': stats['n'],
                    'win_rate': stats['win_rate'], 'max_dd': stats['max_dd'],
                })
                print(f"    {yname}: Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}, "
                      f"N={stats['n']}, WR={stats['win_rate']:.1f}%")
            except Exception as e:
                print(f"    {yname}: ERROR {e}")

        if len(yearly) >= 9:
            early = [r['sharpe'] for r in yearly[:9]]
            recent = [r['sharpe'] for r in yearly[9:]]
            early_pnl = sum(r['total_pnl'] for r in yearly[:9])
            recent_pnl = sum(r['total_pnl'] for r in yearly[9:])
            print(f"\n  {strat_name} 前9年 (2015-2023): mean Sharpe={np.mean(early):.2f}, "
                  f"total PnL=${early_pnl:.0f}")
            print(f"  {strat_name} 后2年 (2024-2025): mean Sharpe={np.mean(recent):.2f}, "
                  f"total PnL=${recent_pnl:.0f}")
            if np.mean(recent) < np.mean(early) * 0.7:
                print(f"  ⚠️ WARNING: Recent Sharpe < 70% of historical → possible alpha decay")
            else:
                print(f"  ✓ No alpha decay detected")

        all_results[strat_name] = yearly

    # Regime shift test: rolling 2-year Sharpe
    print("\n\n  --- Rolling 2-Year Sharpe (L8_BASE) ---")
    rolling = []
    for y in range(2015, 2025):
        rstart = f"{y}-01-01"
        rend = f"{y+2}-01-01"
        try:
            rdata = data.slice(rstart, rend)
            stats = run_variant(rdata, f"L8_rolling_{y}_{y+2}", verbose=False, **L8_BASE)
            rolling.append({
                'window': f"{y}-{y+2}", 'sharpe': stats['sharpe'],
                'total_pnl': stats['total_pnl'], 'n': stats['n'],
            })
            print(f"    {y}-{y+2}: Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}")
        except Exception as e:
            print(f"    {y}-{y+2}: ERROR {e}")

    all_results['rolling_2yr'] = rolling
    save_json(all_results, 'phase_e_alpha_decay.json')
    print(f"\n  Phase E complete. {elapsed()}")


# ══════════════════════════════════════════════════════════════
# Phase F: Monte Carlo 生存性 — 200 次参数扰动
# R36 只测了 spread MC, 从未测过参数 MC 对 L8_BASE
# ══════════════════════════════════════════════════════════════

def phase_f(data):
    phase_header("Phase F", "Monte Carlo 参数扰动 200 次 — L8_BASE 生存性")

    np.random.seed(42)
    N_MC = 200

    base_params = {
        'keltner_adx_threshold': 14,
        'trailing_activate_atr': 0.14,
        'trailing_distance_atr': 0.025,
        'sl_atr_mult': 3.5,
        'tp_atr_mult': 8.0,
        'keltner_max_hold_m15': 20,
    }

    results = []
    for i in range(N_MC):
        # ±15% uniform perturbation on each param
        perturbed = {}
        for k, v in base_params.items():
            if isinstance(v, int):
                delta = max(1, int(v * 0.15))
                perturbed[k] = v + np.random.randint(-delta, delta + 1)
            else:
                perturbed[k] = v * (1 + np.random.uniform(-0.15, 0.15))

        perturbed['keltner_adx_threshold'] = max(8, perturbed['keltner_adx_threshold'])
        perturbed['keltner_max_hold_m15'] = max(4, perturbed['keltner_max_hold_m15'])

        kw = {
            **LIVE_PARITY_KWARGS,
            **perturbed,
            'regime_config': {
                'low':    {'trail_act': 0.22 * (1 + np.random.uniform(-0.15, 0.15)),
                           'trail_dist': 0.04 * (1 + np.random.uniform(-0.15, 0.15))},
                'normal': {'trail_act': perturbed['trailing_activate_atr'],
                           'trail_dist': perturbed['trailing_distance_atr']},
                'high':   {'trail_act': 0.06 * (1 + np.random.uniform(-0.15, 0.15)),
                           'trail_dist': 0.008 * (1 + np.random.uniform(-0.15, 0.15))},
            },
            'time_decay_tp': False,
            'min_entry_gap_hours': 1.0,
        }

        stats = run_variant(data, f"MC_{i:03d}", verbose=False, **kw)
        results.append({
            'run': i, 'sharpe': stats['sharpe'],
            'total_pnl': stats['total_pnl'], 'n': stats['n'],
            'win_rate': stats['win_rate'], 'max_dd': stats['max_dd'],
            'params': {k: round(v, 4) if isinstance(v, float) else v for k, v in perturbed.items()},
        })

        if (i + 1) % 20 == 0:
            sharpes_so_far = [r['sharpe'] for r in results]
            print(f"  MC {i+1}/{N_MC}: mean Sharpe={np.mean(sharpes_so_far):.2f}, "
                  f"min={min(sharpes_so_far):.2f}, "
                  f"positive={sum(1 for s in sharpes_so_far if s > 0)}/{len(sharpes_so_far)} "
                  f"{elapsed()}")

    sharpes = [r['sharpe'] for r in results]
    pnls = [r['total_pnl'] for r in results]

    print(f"\n  {'='*50}")
    print(f"  Monte Carlo Summary ({N_MC} runs, ±15% perturbation)")
    print(f"  {'='*50}")
    print(f"  Sharpe: mean={np.mean(sharpes):.2f}, median={np.median(sharpes):.2f}, "
          f"std={np.std(sharpes):.2f}")
    print(f"  Sharpe: min={min(sharpes):.2f}, p5={np.percentile(sharpes,5):.2f}, "
          f"p25={np.percentile(sharpes,25):.2f}, p75={np.percentile(sharpes,75):.2f}, "
          f"max={max(sharpes):.2f}")
    print(f"  PnL: mean=${np.mean(pnls):.0f}, min=${min(pnls):.0f}, max=${max(pnls):.0f}")
    print(f"  Positive Sharpe: {sum(1 for s in sharpes if s > 0)}/{N_MC} "
          f"({sum(1 for s in sharpes if s > 0)/N_MC*100:.1f}%)")
    print(f"  Sharpe > 3: {sum(1 for s in sharpes if s > 3)}/{N_MC}")
    print(f"  Sharpe > 5: {sum(1 for s in sharpes if s > 5)}/{N_MC}")

    save_json(results, 'phase_f_monte_carlo_params.json')
    print(f"\n  Phase F complete. {elapsed()}")


# ══════════════════════════════════════════════════════════════
# MAIN — Run all phases sequentially
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log_file = open(OUT_DIR / "00_master_log.txt", 'w')
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print(f"R44 Server Marathon — Started at {datetime.now()}")
    print(f"Server: {os.uname().nodename if hasattr(os, 'uname') else 'unknown'}")
    print(f"Python: {sys.version}")
    print(f"Output: {OUT_DIR}")

    print("\n\nLoading data (full 11-year sample)...")
    t0 = time.time()
    data = DataBundle.load_default()
    print(f"Data loaded in {time.time()-t0:.1f}s")

    phases = [
        ("Phase A", phase_a),
        ("Phase B", phase_b),
        ("Phase C", phase_c),
        ("Phase D", phase_d),
        ("Phase E", phase_e),
        ("Phase F", phase_f),
    ]

    completed = []
    for pname, pfunc in phases:
        try:
            t_phase = time.time()
            pfunc(data)
            dt = time.time() - t_phase
            completed.append((pname, dt))
            print(f"\n  {pname} took {dt/60:.1f} min")
        except Exception as e:
            print(f"\n  ❌ {pname} FAILED: {e}")
            traceback.print_exc()
            completed.append((pname, -1))

    total_elapsed = time.time() - MARATHON_START
    print(f"\n\n{'='*70}")
    print(f"  R44 MARATHON COMPLETE")
    print(f"  Total: {total_elapsed/3600:.1f} hours ({total_elapsed/60:.0f} minutes)")
    print(f"{'='*70}")
    for pname, dt in completed:
        status = f"{dt/60:.1f} min" if dt > 0 else "FAILED"
        print(f"  {pname}: {status}")

    print(f"\n  Results saved to: {OUT_DIR}")
    print(f"  Log: {OUT_DIR / '00_master_log.txt'}")

    log_file.close()
