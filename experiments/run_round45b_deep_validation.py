"""R45B: 新信号源深度验证
=========================
Phase A: S4 Chandelier + L8 组合回测 (daily PnL合并 → 组合 Sharpe/MaxDD)
Phase B: S3 Dual Thrust 深度验证 (扩大SL/TP网格 + 更多k值)
Phase C: S5 Z-Score Fold1修复 (更严格ADX/调整周期)
Phase D: 所有候选策略 spread 敏感度 ($0.30 ~ $1.50)
Phase E: 最佳组合 K-Fold 验证

USAGE
-----
    python -m experiments.run_round45b_deep_validation
"""
import sys, os, time, json, traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS
from backtest.stats import calc_stats
from backtest.engine import TradeRecord
from indicators import (
    calc_donchian, calc_chandelier, calc_zscore,
    calc_dual_thrust_range, calc_range_contraction, prepare_indicators
)
from experiments.run_round45_new_signals import (
    backtest_signals, trades_to_stats, daily_pnl_correlation,
    SimpleTrade, Tee, save_json, donchian_signals, chandelier_signals,
    dual_thrust_signals, zscore_signals, bb_squeeze_signals, kfold_test,
)

OUT_DIR = ROOT / "results" / "round45b_deep_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARATHON_START = time.time()

def elapsed():
    return f"[{(time.time()-MARATHON_START)/60:.1f} min]"

def phase_header(name, desc):
    print(f"\n{'='*70}")
    print(f"  {name}: {desc}")
    print(f"  {elapsed()}")
    print(f"{'='*70}\n", flush=True)


def combine_daily_pnl(*daily_dicts) -> Dict[str, float]:
    """Merge multiple daily PnL dicts into one combined."""
    combined = {}
    for d in daily_dicts:
        for date, pnl in d.items():
            combined[date] = combined.get(date, 0) + pnl
    return combined


def stats_from_daily(daily_pnl: Dict[str, float], label: str = "") -> Dict:
    """Compute Sharpe/MaxDD from a daily PnL dict."""
    if not daily_pnl:
        return {'label': label, 'sharpe': 0, 'total_pnl': 0, 'max_dd': 0, 'n_days': 0}
    dates = sorted(daily_pnl.keys())
    pnls = [daily_pnl[d] for d in dates]
    total = sum(pnls)
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
    else:
        sharpe = 0
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = running_max - cumsum
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0
    return {
        'label': label,
        'sharpe': round(sharpe, 2),
        'total_pnl': round(total, 2),
        'max_dd': round(max_dd, 2),
        'n_days': len(dates),
    }


def get_l8_daily_and_trades(data):
    """Run L8_BASE and return (stats, daily_pnl, trades)."""
    L8_BASE = {
        **LIVE_PARITY_KWARGS,
        'keltner_adx_threshold': 14,
        'regime_config': {
            'low':    {'trail_act': 0.22, 'trail_dist': 0.04},
            'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
            'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
        },
        'keltner_max_hold_m15': 20,
        'time_decay_tp': False,
        'min_entry_gap_hours': 1.0,
    }
    stats = run_variant(data, 'L8_BASE', verbose=False, **L8_BASE)
    trades = stats.get('_trades', [])
    daily = {}
    for t in trades:
        d = str(t.exit_time.date()) if hasattr(t.exit_time, 'date') else str(t.exit_time)[:10]
        daily[d] = daily.get(d, 0) + t.pnl
    return stats, daily, trades


# ══════════════════════════════════════════════════════════════
# Phase A: S4 Chandelier + L8 组合
# ══════════════════════════════════════════════════════════════

def phase_a(h1_df, data, l8_stats, l8_daily):
    phase_header("Phase A", "S4 Chandelier + L8 组合回测")

    best_ch_params = {'period': 10, 'mult': 3.0, 'ema_filter': False}

    print("  Running S4 Chandelier best config (full period)...")
    signals, atr = chandelier_signals(h1_df, **best_ch_params)
    ch_trades = backtest_signals(h1_df, signals, atr, sl_mult=3.0, tp_mult=8.0,
                                  max_hold=20, trail_act=0.28, trail_dist=0.06)
    ch_stats = trades_to_stats(ch_trades, "CH_best_full")

    ch_daily = ch_stats['daily_pnl']
    combo_daily = combine_daily_pnl(l8_daily, ch_daily)
    combo_stats = stats_from_daily(combo_daily, "L8+CH_combo")

    print(f"  L8_BASE alone:    Sharpe={l8_stats['sharpe']:.2f}, PnL=${l8_stats['total_pnl']:.0f}, MaxDD=${l8_stats['max_dd']:.0f}")
    print(f"  S4 Chandelier:    Sharpe={ch_stats['sharpe']},   PnL=${ch_stats['total_pnl']:.0f}, MaxDD=${ch_stats['max_dd']:.0f}")
    print(f"  ★ L8+CH 组合:     Sharpe={combo_stats['sharpe']}, PnL=${combo_stats['total_pnl']:.0f}, MaxDD=${combo_stats['max_dd']:.0f}")
    print(f"  Daily PnL 相关性:  {daily_pnl_correlation(l8_daily, ch_daily)}")

    # Chandelier 参数鲁棒性
    print("\n  Chandelier 参数鲁棒性检查:")
    robust_results = []
    for period in [8, 10, 12, 15]:
        for mult in [2.5, 3.0, 3.5]:
            sig, a = chandelier_signals(h1_df, period=period, mult=mult, ema_filter=False)
            tr = backtest_signals(h1_df, sig, a, sl_mult=3.0, tp_mult=8.0,
                                  max_hold=20, trail_act=0.28, trail_dist=0.06)
            st = trades_to_stats(tr, f"CH_p{period}_m{mult}")
            cd = st['daily_pnl']
            combo_d = combine_daily_pnl(l8_daily, cd)
            combo_s = stats_from_daily(combo_d, f"L8+CH_p{period}_m{mult}")
            robust_results.append({
                'period': period, 'mult': mult,
                'ch_sharpe': st['sharpe'], 'ch_pnl': round(st['total_pnl'], 0),
                'ch_n': st['n'],
                'combo_sharpe': combo_s['sharpe'], 'combo_pnl': combo_s['total_pnl'],
                'combo_maxdd': combo_s['max_dd'],
                'l8_corr': daily_pnl_correlation(l8_daily, cd),
            })
    robust_results.sort(key=lambda x: x['combo_sharpe'], reverse=True)
    for r in robust_results[:8]:
        print(f"    CH(p={r['period']},m={r['mult']}): CH_Sharpe={r['ch_sharpe']}, "
              f"Combo_Sharpe={r['combo_sharpe']}, Combo_PnL=${r['combo_pnl']}, "
              f"Combo_MaxDD=${r['combo_maxdd']}, L8_corr={r['l8_corr']}")

    # K-Fold on combo
    print("\n  K-Fold 组合测试:")
    folds = [
        ("Fold1", "2015-01-01", "2017-01-01"),
        ("Fold2", "2017-01-01", "2019-01-01"),
        ("Fold3", "2019-01-01", "2021-01-01"),
        ("Fold4", "2021-01-01", "2023-01-01"),
        ("Fold5", "2023-01-01", "2025-01-01"),
        ("Fold6", "2025-01-01", "2026-05-01"),
    ]

    kfold_results = []
    for fname, start, end in folds:
        fold_h1 = h1_df[start:end]
        fold_data = data.slice(start, end)

        if len(fold_h1) < 200:
            continue

        l8_fold = run_variant(fold_data, f'L8_{fname}', verbose=False,
                              **{**LIVE_PARITY_KWARGS,
                                 'keltner_adx_threshold': 14,
                                 'regime_config': {
                                     'low':    {'trail_act': 0.22, 'trail_dist': 0.04},
                                     'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
                                     'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
                                 },
                                 'keltner_max_hold_m15': 20,
                                 'time_decay_tp': False,
                                 'min_entry_gap_hours': 1.0})
        l8_fold_trades = l8_fold.get('_trades', [])
        l8_fold_daily = {}
        for t in l8_fold_trades:
            d = str(t.exit_time.date())
            l8_fold_daily[d] = l8_fold_daily.get(d, 0) + t.pnl

        sig, a = chandelier_signals(fold_h1, **best_ch_params)
        ch_fold_trades = backtest_signals(fold_h1, sig, a, sl_mult=3.0, tp_mult=8.0,
                                           max_hold=20, trail_act=0.28, trail_dist=0.06)
        ch_fold_stats = trades_to_stats(ch_fold_trades, f"CH_{fname}")

        combo_fold = combine_daily_pnl(l8_fold_daily, ch_fold_stats['daily_pnl'])
        combo_fold_stats = stats_from_daily(combo_fold, f"Combo_{fname}")

        kfold_results.append({
            'fold': fname, 'period': f"{start}~{end}",
            'l8_sharpe': round(l8_fold['sharpe'], 2),
            'ch_sharpe': ch_fold_stats['sharpe'],
            'combo_sharpe': combo_fold_stats['sharpe'],
            'combo_maxdd': combo_fold_stats['max_dd'],
        })
        print(f"    {fname}: L8={l8_fold['sharpe']:.2f}, CH={ch_fold_stats['sharpe']}, "
              f"Combo={combo_fold_stats['sharpe']}, MaxDD=${combo_fold_stats['max_dd']}")

    combo_sharpes = [r['combo_sharpe'] for r in kfold_results]
    l8_sharpes = [r['l8_sharpe'] for r in kfold_results]
    print(f"\n  ★ K-Fold 总结:")
    print(f"    L8 alone:  mean={np.mean(l8_sharpes):.2f}, min={min(l8_sharpes):.2f}")
    print(f"    L8+CH:     mean={np.mean(combo_sharpes):.2f}, min={min(combo_sharpes):.2f}")
    improvement = np.mean(combo_sharpes) - np.mean(l8_sharpes)
    print(f"    Combo 提升: {improvement:+.2f} Sharpe")

    save_json({'solo': {k: v for k, v in ch_stats.items() if k != 'daily_pnl'},
               'combo_full': combo_stats,
               'robustness': robust_results,
               'kfold': kfold_results,
               'kfold_summary': {
                   'l8_mean': round(np.mean(l8_sharpes), 2),
                   'combo_mean': round(np.mean(combo_sharpes), 2),
                   'improvement': round(improvement, 2),
               }},
              'A_chandelier_combo.json')
    print(f"  Phase A complete. {elapsed()}")
    return combo_stats, kfold_results


# ══════════════════════════════════════════════════════════════
# Phase B: S3 Dual Thrust 深度验证
# ══════════════════════════════════════════════════════════════

def phase_b(h1_df, l8_daily):
    phase_header("Phase B", "S3 Dual Thrust 深度验证")

    param_grid = []
    for n in [3, 4, 5, 6, 8]:
        for k in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for sl in [2.0, 3.0, 4.0]:
                for tp in [4.0, 6.0, 8.0]:
                    param_grid.append({
                        'n_bars': n, 'k_up': k, 'k_down': k,
                        'bt_kwargs': {'sl_mult': sl, 'tp_mult': tp, 'max_hold': 10,
                                      'trail_act': 0.28, 'trail_dist': 0.06},
                    })

    print(f"  Scanning {len(param_grid)} param combos...")
    results = []
    for p in param_grid:
        signals, atr = dual_thrust_signals(h1_df, **p)
        bt_kw = p.get('bt_kwargs', {})
        trades = backtest_signals(h1_df, signals, atr, **bt_kw)
        stats = trades_to_stats(trades, f"DT_n{p['n_bars']}_k{p['k_up']}_sl{bt_kw['sl_mult']}_tp{bt_kw['tp_mult']}")
        stats['params'] = {k_: v for k_, v in p.items() if k_ != 'bt_kwargs'}
        stats['params'].update(bt_kw)
        results.append(stats)

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    print("  Top 10 Dual Thrust configs:")
    for r in results[:10]:
        corr = daily_pnl_correlation(r['daily_pnl'], l8_daily)
        print(f"    {r['label']}: Sharpe={r['sharpe']}, PnL=${r['total_pnl']:.0f}, "
              f"N={r['n']}, WR={r['win_rate']}%, L8_corr={corr}")
        r['l8_corr'] = corr

    # K-Fold on top 3
    print("\n  K-Fold on top 3:")
    kfold_results = []
    for r in results[:3]:
        p = r['params']
        bt_kw = {k_: v for k_, v in p.items() if k_ in ('sl_mult', 'tp_mult', 'max_hold', 'trail_act', 'trail_dist')}
        sig_kw = {k_: v for k_, v in p.items() if k_ not in bt_kw}
        sig_kw['bt_kwargs'] = bt_kw
        kf = kfold_test(h1_df, dual_thrust_signals, sig_kw, r['label'])
        kfold_results.append(kf)
        print(f"    {r['label']}: K-Fold {kf['pass']}, mean={kf['mean_sharpe']}, min={kf['min_sharpe']}")

    save_json({'grid': [{k_: v for k_, v in r.items() if k_ != 'daily_pnl'} for r in results[:30]],
               'kfold': kfold_results},
              'B_dual_thrust_deep.json')
    print(f"  Phase B complete. {elapsed()}")
    return results[0] if results else None


# ══════════════════════════════════════════════════════════════
# Phase C: S5 Z-Score Fold1 修复
# ══════════════════════════════════════════════════════════════

def phase_c(h1_df, l8_daily):
    phase_header("Phase C", "S5 Z-Score Fold1修复 + 更严格过滤")

    param_grid = []
    for period in [50, 100, 150]:
        for z_th in [2.0, 2.5, 3.0, 3.5]:
            for adx_cap in [15, 18, 20, 22]:
                for sl in [1.5, 2.0, 2.5]:
                    param_grid.append({
                        'sma_period': period, 'z_threshold': z_th, 'adx_cap': adx_cap,
                        'bt_kwargs': {'sl_mult': sl, 'tp_mult': 4.0, 'max_hold': 10,
                                      'trail_act': 0.5, 'trail_dist': 0.1},
                    })

    print(f"  Scanning {len(param_grid)} param combos...")
    results = []
    for p in param_grid:
        signals, atr = zscore_signals(h1_df, **p)
        bt_kw = p.get('bt_kwargs', {})
        trades = backtest_signals(h1_df, signals, atr, **bt_kw)
        stats = trades_to_stats(trades, f"ZS_p{p['sma_period']}_z{p['z_threshold']}_adx{p['adx_cap']}_sl{bt_kw['sl_mult']}")
        stats['params'] = {k_: v for k_, v in p.items() if k_ != 'bt_kwargs'}
        stats['params'].update(bt_kw)
        results.append(stats)

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    print("  Top 10 Z-Score configs:")
    for r in results[:10]:
        corr = daily_pnl_correlation(r['daily_pnl'], l8_daily)
        print(f"    {r['label']}: Sharpe={r['sharpe']}, PnL=${r['total_pnl']:.0f}, "
              f"N={r['n']}, WR={r['win_rate']}%, L8_corr={corr}")
        r['l8_corr'] = corr

    # K-Fold on top 3, checking Fold1 specifically
    print("\n  K-Fold on top 3 (focus on Fold1):")
    kfold_results = []
    for r in results[:3]:
        p = r['params']
        bt_kw = {k_: v for k_, v in p.items() if k_ in ('sl_mult', 'tp_mult', 'max_hold', 'trail_act', 'trail_dist')}
        sig_kw = {k_: v for k_, v in p.items() if k_ not in bt_kw}
        sig_kw['bt_kwargs'] = bt_kw
        kf = kfold_test(h1_df, zscore_signals, sig_kw, r['label'])
        kfold_results.append(kf)
        fold1_sharpe = kf['folds'][0]['sharpe'] if kf['folds'] else '?'
        print(f"    {r['label']}: K-Fold {kf['pass']}, mean={kf['mean_sharpe']}, "
              f"min={kf['min_sharpe']}, Fold1={fold1_sharpe}")

    save_json({'grid': [{k_: v for k_, v in r.items() if k_ != 'daily_pnl'} for r in results[:30]],
               'kfold': kfold_results},
              'C_zscore_fix.json')
    print(f"  Phase C complete. {elapsed()}")
    return results[0] if results else None


# ══════════════════════════════════════════════════════════════
# Phase D: Spread 敏感度
# ══════════════════════════════════════════════════════════════

def phase_d(h1_df, l8_daily):
    phase_header("Phase D", "候选策略 Spread 敏感度测试")

    spreads = [0.0, 0.30, 0.50, 0.80, 1.00, 1.50]

    strategies = {
        'S4_Chandelier': {'func': chandelier_signals,
                          'params': {'period': 10, 'mult': 3.0, 'ema_filter': False},
                          'bt': {'sl_mult': 3.0, 'tp_mult': 8.0, 'max_hold': 20,
                                 'trail_act': 0.28, 'trail_dist': 0.06}},
        'S3_DualThrust': {'func': dual_thrust_signals,
                          'params': {'n_bars': 4, 'k_up': 0.5, 'k_down': 0.5},
                          'bt': {'sl_mult': 3.0, 'tp_mult': 6.0, 'max_hold': 10,
                                 'trail_act': 0.28, 'trail_dist': 0.06}},
        'S1_Donchian':   {'func': donchian_signals,
                          'params': {'lookback': 10, 'ema_filter': False},
                          'bt': {'sl_mult': 4.0, 'tp_mult': 8.0, 'max_hold': 10,
                                 'trail_act': 0.28, 'trail_dist': 0.06}},
        'S5_ZScore':     {'func': zscore_signals,
                          'params': {'sma_period': 100, 'z_threshold': 3.0, 'adx_cap': 20},
                          'bt': {'sl_mult': 2.0, 'tp_mult': 4.0, 'max_hold': 10,
                                 'trail_act': 0.5, 'trail_dist': 0.1}},
    }

    all_results = {}
    for sname, sconf in strategies.items():
        print(f"  {sname}:")
        signals, atr = sconf['func'](h1_df, **sconf['params'])
        spread_results = []
        for sp in spreads:
            bt_kw = {**sconf['bt'], 'spread_cost': sp}
            trades = backtest_signals(h1_df, signals, atr, **bt_kw)
            stats = trades_to_stats(trades, f"{sname}_sp{sp}")
            spread_results.append({
                'spread': sp,
                'sharpe': stats['sharpe'],
                'total_pnl': round(stats['total_pnl'], 0),
                'n': stats['n'],
                'win_rate': stats['win_rate'],
            })
            print(f"    spread=${sp:.2f}: Sharpe={stats['sharpe']}, PnL=${stats['total_pnl']:.0f}, "
                  f"N={stats['n']}, WR={stats['win_rate']}%")

        # Sharpe at $0.50 vs $0 = degradation
        s0 = spread_results[0]['sharpe']
        s50 = next((r['sharpe'] for r in spread_results if r['spread'] == 0.50), 0)
        degradation = (s0 - s50) / s0 * 100 if s0 > 0 else 0
        breakeven_spread = None
        for r in spread_results:
            if r['sharpe'] <= 0:
                breakeven_spread = r['spread']
                break
        print(f"    → Degradation at $0.50: {degradation:.1f}%")
        if breakeven_spread:
            print(f"    → Breakeven spread: ≈${breakeven_spread:.2f}")
        else:
            print(f"    → Still profitable at $1.50")

        all_results[sname] = {
            'spread_curve': spread_results,
            'degradation_50': round(degradation, 1),
            'breakeven_spread': breakeven_spread,
        }

    save_json(all_results, 'D_spread_sensitivity.json')
    print(f"\n  Phase D complete. {elapsed()}")
    return all_results


# ══════════════════════════════════════════════════════════════
# Phase E: 最佳三策略组合 K-Fold
# ══════════════════════════════════════════════════════════════

def phase_e(h1_df, data, l8_daily):
    phase_header("Phase E", "L8 + Chandelier + DualThrust 三策略组合 K-Fold")

    folds = [
        ("Fold1", "2015-01-01", "2017-01-01"),
        ("Fold2", "2017-01-01", "2019-01-01"),
        ("Fold3", "2019-01-01", "2021-01-01"),
        ("Fold4", "2021-01-01", "2023-01-01"),
        ("Fold5", "2023-01-01", "2025-01-01"),
        ("Fold6", "2025-01-01", "2026-05-01"),
    ]

    kfold_results = []
    for fname, start, end in folds:
        fold_h1 = h1_df[start:end]
        fold_data = data.slice(start, end)
        if len(fold_h1) < 200:
            continue

        # L8
        l8_fold = run_variant(fold_data, f'L8_{fname}', verbose=False,
                              **{**LIVE_PARITY_KWARGS,
                                 'keltner_adx_threshold': 14,
                                 'regime_config': {
                                     'low':    {'trail_act': 0.22, 'trail_dist': 0.04},
                                     'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
                                     'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
                                 },
                                 'keltner_max_hold_m15': 20,
                                 'time_decay_tp': False,
                                 'min_entry_gap_hours': 1.0})
        l8_daily_fold = {}
        for t in l8_fold.get('_trades', []):
            d = str(t.exit_time.date())
            l8_daily_fold[d] = l8_daily_fold.get(d, 0) + t.pnl

        # Chandelier
        sig_ch, atr_ch = chandelier_signals(fold_h1, period=10, mult=3.0, ema_filter=False)
        ch_trades = backtest_signals(fold_h1, sig_ch, atr_ch, sl_mult=3.0, tp_mult=8.0,
                                      max_hold=20, trail_act=0.28, trail_dist=0.06)
        ch_stats = trades_to_stats(ch_trades, f"CH_{fname}")

        # Dual Thrust
        sig_dt, atr_dt = dual_thrust_signals(fold_h1, n_bars=4, k_up=0.5, k_down=0.5)
        dt_trades = backtest_signals(fold_h1, sig_dt, atr_dt, sl_mult=3.0, tp_mult=6.0,
                                      max_hold=10, trail_act=0.28, trail_dist=0.06)
        dt_stats = trades_to_stats(dt_trades, f"DT_{fname}")

        # 2-combo: L8 + CH
        combo2_daily = combine_daily_pnl(l8_daily_fold, ch_stats['daily_pnl'])
        combo2_stats = stats_from_daily(combo2_daily, f"L8+CH_{fname}")

        # 3-combo: L8 + CH + DT
        combo3_daily = combine_daily_pnl(l8_daily_fold, ch_stats['daily_pnl'], dt_stats['daily_pnl'])
        combo3_stats = stats_from_daily(combo3_daily, f"L8+CH+DT_{fname}")

        kfold_results.append({
            'fold': fname,
            'l8_sharpe': round(l8_fold['sharpe'], 2),
            'ch_sharpe': ch_stats['sharpe'],
            'dt_sharpe': dt_stats['sharpe'],
            'l8_ch_sharpe': combo2_stats['sharpe'],
            'l8_ch_dt_sharpe': combo3_stats['sharpe'],
            'l8_ch_maxdd': combo2_stats['max_dd'],
            'l8_ch_dt_maxdd': combo3_stats['max_dd'],
        })
        print(f"  {fname}: L8={l8_fold['sharpe']:.2f}, CH={ch_stats['sharpe']}, DT={dt_stats['sharpe']}, "
              f"L8+CH={combo2_stats['sharpe']}, L8+CH+DT={combo3_stats['sharpe']}")

    l8_s = [r['l8_sharpe'] for r in kfold_results]
    c2_s = [r['l8_ch_sharpe'] for r in kfold_results]
    c3_s = [r['l8_ch_dt_sharpe'] for r in kfold_results]

    print(f"\n  ★ K-Fold 总结:")
    print(f"    L8 alone:     mean={np.mean(l8_s):.2f}, min={min(l8_s):.2f}")
    print(f"    L8+CH:        mean={np.mean(c2_s):.2f}, min={min(c2_s):.2f}")
    print(f"    L8+CH+DT:     mean={np.mean(c3_s):.2f}, min={min(c3_s):.2f}")

    save_json({'kfold': kfold_results,
               'summary': {
                   'l8_mean': round(np.mean(l8_s), 2),
                   'l8_ch_mean': round(np.mean(c2_s), 2),
                   'l8_ch_dt_mean': round(np.mean(c3_s), 2),
               }},
              'E_triple_combo_kfold.json')
    print(f"\n  Phase E complete. {elapsed()}")
    return kfold_results


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log_file = open(OUT_DIR / "00_master_log.txt", 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print(f"R45B Deep Validation — Started at {datetime.now()}")
    print(f"Output: {OUT_DIR}\n")

    print("Loading data...")
    t0 = time.time()
    data = DataBundle.load_default()
    h1_df = data.h1_df.copy()
    print(f"Data loaded in {time.time()-t0:.1f}s")
    print(f"  H1: {len(h1_df)} bars, {h1_df.index[0]} ~ {h1_df.index[-1]}")

    print("\nRunning L8_BASE reference...")
    l8_stats, l8_daily, l8_trades = get_l8_daily_and_trades(data)
    print(f"  L8 ref: Sharpe={l8_stats['sharpe']:.2f}, PnL=${l8_stats['total_pnl']:.0f}, N={l8_stats['n']}")

    phases = [
        ("A", phase_a, (h1_df, data, l8_stats, l8_daily)),
        ("B", phase_b, (h1_df, l8_daily)),
        ("C", phase_c, (h1_df, l8_daily)),
        ("D", phase_d, (h1_df, l8_daily)),
        ("E", phase_e, (h1_df, data, l8_daily)),
    ]

    completed = []
    for pname, pfunc, pargs in phases:
        try:
            t_phase = time.time()
            result = pfunc(*pargs)
            dt = time.time() - t_phase
            completed.append((pname, dt, result))
            print(f"\n  Phase {pname} took {dt/60:.1f} min")
        except Exception as e:
            print(f"\n  Phase {pname} FAILED: {e}")
            traceback.print_exc()
            completed.append((pname, -1, None))

    total_elapsed = time.time() - MARATHON_START
    print(f"\n\n{'='*70}")
    print(f"  R45B COMPLETE — {total_elapsed/60:.0f} minutes")
    print(f"{'='*70}")
    for pname, dt, _ in completed:
        status = f"{dt/60:.1f} min" if dt > 0 else "FAILED"
        print(f"  Phase {pname}: {status}")
    print(f"\n  Results: {OUT_DIR}")
    log_file.close()
