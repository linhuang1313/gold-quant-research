#!/usr/bin/env python3
"""R207: Drawdown-Responsive Lot Sizing Deep Validation
========================================================
R200 C1 found dd_responsive_200 boosts Sharpe from 7.85 to 10.83 (+38%)
with only -2.4% PnL reduction. This is the largest single-factor improvement
in the entire R200 mega research.

R200 C1 was a quick scan: one lot_mult=0.5 cutoff, four thresholds, no
cross-validation. This experiment does the full validation:

Phase 1 — Extended Grid
  dd_threshold: [80, 100, 120, 150, 180, 200, 250, 300, 400]
  lot_mult_dd:  [0.25, 0.33, 0.50, 0.67, 0.75]
  recovery:     [immediate, gradual_3d, gradual_5d]
  ~135 combos on full Keltner backtest

Phase 2 — 3-Gate Validation on top candidates
  6-fold CV, 19-window walk-forward, 4-era stability

Phase 3 — 5/11-5/12 Scenario Replay
  Would dd_responsive have saved money during the actual drawdown?
  Replay real trade sequence with dd logic applied.

Phase 4 — Interaction with R202 trail params
  Does dd_responsive still help with 0.06/0.015 trail (not just old 0.02/0.005)?

Run: python experiments/run_r207_dd_responsive_validation.py
"""
from __future__ import annotations
import sys
import json
import time
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.runner import (
    DataBundle, run_variant, LIVE_PARITY_KWARGS,
)
from backtest.engine import BacktestEngine
from backtest.stats import calc_stats
import research_config as config
import indicators as signals_mod
from indicators import get_orb_strategy

OUTPUT_DIR = Path("results/r207_dd_responsive")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAPITAL = getattr(config, 'CAPITAL', 10000)

ERA_SEGMENTS = {
    "Pre-COVID (2015-2019)":      ("2015-01-01", "2020-01-01"),
    "COVID+Recovery (2020-2021)": ("2020-01-01", "2022-01-01"),
    "Tightening (2022-2023)":     ("2022-01-01", "2024-01-01"),
    "Recent (2024-2026)":         ("2024-01-01", "2026-06-01"),
}

WF_WINDOWS = []
for yr in range(2017, 2027):
    train_s = f"{yr-2}-01-01"
    train_e = f"{yr}-01-01"
    test_s  = f"{yr}-01-01"
    test_e  = f"{yr}-07-01" if yr < 2026 else "2026-05-06"
    WF_WINDOWS.append((train_s, train_e, test_s, test_e))
    if yr < 2026:
        WF_WINDOWS.append((f"{yr-2}-07-01", f"{yr}-07-01",
                           f"{yr}-07-01", f"{yr+1}-01-01"))


def save(name, data):
    p = OUTPUT_DIR / f'{name}.json'
    with open(p, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f'  -> saved {p}')


def run_keltner_backtest(data: DataBundle, **extra_kwargs):
    """Run Keltner backtest, return list of trades with timing."""
    get_orb_strategy().reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False

    kw = dict(LIVE_PARITY_KWARGS)
    kw.update(extra_kwargs)
    engine = BacktestEngine(data.m15_df, data.h1_df, **kw)
    trades = engine.run()
    return trades, engine.equity_curve


def apply_dd_responsive(trades, dd_thresh, lot_mult_dd=0.5,
                        recovery='immediate', capital=None):
    """Apply drawdown-responsive sizing to a trade list.

    Args:
        trades: list of TradeRecord from backtest engine
        dd_thresh: $ drawdown threshold to trigger lot reduction
        lot_mult_dd: lot multiplier when in drawdown (0-1)
        recovery: 'immediate' = restore full lot when equity > peak - dd_thresh
                  'gradual_Nd' = linearly restore over N days after recovery
    Returns:
        dict with daily PnL series, stats, and trade-level details
    """
    if capital is None:
        capital = CAPITAL
    equity = capital
    peak_eq = equity
    daily_pnl = defaultdict(float)
    trade_details = []

    # Parse recovery mode
    gradual_days = 0
    if recovery.startswith('gradual_'):
        gradual_days = int(recovery.split('_')[1].replace('d', ''))

    recovery_start = None
    recovery_mult = 1.0

    for t in trades:
        dd = peak_eq - equity

        # Determine current lot multiplier
        if dd > dd_thresh:
            cur_mult = lot_mult_dd
            recovery_start = None
        elif gradual_days > 0 and recovery_start is not None:
            exit_time = pd.Timestamp(t.exit_time)
            days_since = (exit_time - recovery_start).total_seconds() / 86400
            if days_since < gradual_days:
                progress = days_since / gradual_days
                cur_mult = lot_mult_dd + (1.0 - lot_mult_dd) * progress
            else:
                cur_mult = 1.0
                recovery_start = None
        else:
            cur_mult = 1.0

        scaled_pnl = t.pnl * cur_mult
        equity += scaled_pnl
        if equity > peak_eq:
            peak_eq = equity

        # Track recovery transition
        if dd > dd_thresh and (peak_eq - equity) <= dd_thresh and gradual_days > 0:
            if recovery_start is None:
                recovery_start = pd.Timestamp(t.exit_time)

        exit_date = str(pd.Timestamp(t.exit_time).date())
        daily_pnl[exit_date] += scaled_pnl

        trade_details.append({
            'exit_time': str(t.exit_time),
            'orig_pnl': t.pnl,
            'scaled_pnl': scaled_pnl,
            'mult': round(cur_mult, 3),
            'dd_at_entry': round(dd, 2),
            'equity_after': round(equity, 2),
        })

    dpnl = np.array(list(daily_pnl.values()))
    total_pnl = float(dpnl.sum())
    sharpe = float(dpnl.mean() / dpnl.std(ddof=1) * np.sqrt(252)) if len(dpnl) > 1 and dpnl.std(ddof=1) > 0 else 0

    eq_series = np.array([capital] + [td['equity_after'] for td in trade_details])
    running_max = np.maximum.accumulate(eq_series)
    max_dd = float((running_max - eq_series).max())

    n_reduced = sum(1 for td in trade_details if td['mult'] < 1.0)

    return {
        'total_pnl': round(total_pnl, 2),
        'sharpe': round(sharpe, 3),
        'max_dd': round(max_dd, 2),
        'n_trades': len(trade_details),
        'n_reduced': n_reduced,
        'pct_reduced': round(100 * n_reduced / max(len(trade_details), 1), 2),
        'trade_details': trade_details,
        'daily_pnl': dict(daily_pnl),
    }


def main():
    t_start = time.time()
    print('=' * 80)
    print('R207: Drawdown-Responsive Lot Sizing Deep Validation')
    print('=' * 80)

    print('\nLoading data...')
    data = DataBundle.load_default()

    # Run baseline backtest once to get full trade list
    print('\nRunning baseline Keltner backtest...')
    trades, eq_curve = run_keltner_backtest(data)
    print(f'  Baseline: {len(trades)} trades')

    baseline_stats = calc_stats(trades, eq_curve)
    baseline_daily = defaultdict(float)
    for t in trades:
        d = str(pd.Timestamp(t.exit_time).date())
        baseline_daily[d] += t.pnl
    bl_dpnl = np.array(list(baseline_daily.values()))
    bl_sharpe = float(bl_dpnl.mean() / bl_dpnl.std(ddof=1) * np.sqrt(252)) if len(bl_dpnl) > 1 else 0
    bl_total = float(bl_dpnl.sum())
    print(f'  Baseline daily Sharpe: {bl_sharpe:.3f}  Total PnL: ${bl_total:.0f}')

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Extended Grid Search
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 1: Extended Grid Search')
    print('=' * 80)

    DD_THRESHOLDS = [80, 100, 120, 150, 180, 200, 250, 300, 400]
    LOT_MULTS     = [0.25, 0.33, 0.50, 0.67, 0.75]
    RECOVERIES    = ['immediate', 'gradual_3d', 'gradual_5d']

    grid_results = []
    best_sharpe = -999
    best_config = None

    for dd_t, lm, rec in itertools.product(DD_THRESHOLDS, LOT_MULTS, RECOVERIES):
        label = f'dd{dd_t}_lm{lm}_r{rec}'
        result = apply_dd_responsive(trades, dd_t, lm, rec)
        row = {
            'label': label,
            'dd_thresh': dd_t,
            'lot_mult': lm,
            'recovery': rec,
            'sharpe': result['sharpe'],
            'pnl': result['total_pnl'],
            'max_dd': result['max_dd'],
            'pct_reduced': result['pct_reduced'],
        }
        grid_results.append(row)
        if result['sharpe'] > best_sharpe:
            best_sharpe = result['sharpe']
            best_config = row

    grid_results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f'  Grid: {len(grid_results)} combos evaluated')
    print(f'\n  Top 10:')
    print(f'  {"Label":<35} {"Sharpe":>8} {"PnL":>12} {"MaxDD":>8} {"Reduced%":>9}')
    for r in grid_results[:10]:
        print(f'  {r["label"]:<35} {r["sharpe"]:>8.3f} {r["pnl"]:>12.0f} {r["max_dd"]:>8.0f} {r["pct_reduced"]:>9.1f}%')

    print(f'\n  Baseline: Sharpe={bl_sharpe:.3f}  PnL=${bl_total:.0f}')
    print(f'  Best:     {best_config["label"]}  Sharpe={best_sharpe:.3f}  PnL=${best_config["pnl"]:.0f}')

    # Save without trade_details (too large)
    save('phase1_grid', grid_results[:50])

    # Pick top 3 for validation
    # Force different thresholds in top 3 for diversity
    seen_thresh = set()
    top_candidates = []
    for r in grid_results:
        if r['dd_thresh'] not in seen_thresh and len(top_candidates) < 3:
            top_candidates.append(r)
            seen_thresh.add(r['dd_thresh'])
    if len(top_candidates) < 3:
        top_candidates = grid_results[:3]

    print(f'\n  Top 3 candidates for 3-Gate:')
    for c in top_candidates:
        print(f'    {c["label"]}: Sharpe={c["sharpe"]:.3f}')

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: 3-Gate Validation
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 2: 3-Gate Validation')
    print('=' * 80)

    gate_results = {}
    for cand in top_candidates:
        label = cand['label']
        dd_t = cand['dd_thresh']
        lm = cand['lot_mult']
        rec = cand['recovery']
        print(f'\n  === {label} ===')

        # Gate 1: 6-Fold CV
        print('    Gate 1: 6-Fold CV...')
        kf_sharpes = []
        kf_bl_sharpes = []
        all_dates = sorted(set(str(pd.Timestamp(t.exit_time).date()) for t in trades))
        fold_size = len(all_dates) // 6

        for fold in range(6):
            fold_start = all_dates[fold * fold_size] if fold * fold_size < len(all_dates) else all_dates[-1]
            fold_end = all_dates[min((fold + 1) * fold_size, len(all_dates) - 1)]
            fold_trades = [t for t in trades
                           if fold_start <= str(pd.Timestamp(t.exit_time).date()) <= fold_end]

            if len(fold_trades) < 50:
                kf_sharpes.append(0.0)
                kf_bl_sharpes.append(0.0)
                continue

            dd_result = apply_dd_responsive(fold_trades, dd_t, lm, rec)
            kf_sharpes.append(dd_result['sharpe'])

            bl_daily = defaultdict(float)
            for t in fold_trades:
                d = str(pd.Timestamp(t.exit_time).date())
                bl_daily[d] += t.pnl
            bl_arr = np.array(list(bl_daily.values()))
            bl_sh = float(bl_arr.mean() / bl_arr.std(ddof=1) * np.sqrt(252)) if len(bl_arr) > 1 and bl_arr.std(ddof=1) > 0 else 0
            kf_bl_sharpes.append(round(bl_sh, 3))

            print(f'      Fold {fold+1}: Sharpe={dd_result["sharpe"]:.3f} (baseline={bl_sh:.3f})')

        kf_wins = sum(1 for s, b in zip(kf_sharpes, kf_bl_sharpes) if s >= b)
        kf_pass = kf_wins >= 4
        print(f'    KF: {kf_wins}/6  PASS={kf_pass}')

        # Gate 2: Walk-Forward
        print('    Gate 2: Walk-Forward...')
        wf_wins = 0
        wf_details = []
        for train_s, train_e, test_s, test_e in WF_WINDOWS:
            ts_start = pd.Timestamp(test_s, tz='UTC')
            ts_end = pd.Timestamp(test_e, tz='UTC')
            wf_trades = [t for t in trades
                         if ts_start <= pd.Timestamp(t.exit_time) <= ts_end]
            if len(wf_trades) < 30:
                wf_details.append({'window': test_s, 'sharpe': 0, 'bl_sharpe': 0, 'win': False})
                continue

            dd_result = apply_dd_responsive(wf_trades, dd_t, lm, rec)
            bl_daily = defaultdict(float)
            for t in wf_trades:
                d = str(pd.Timestamp(t.exit_time).date())
                bl_daily[d] += t.pnl
            bl_arr = np.array(list(bl_daily.values()))
            bl_sh = float(bl_arr.mean() / bl_arr.std(ddof=1) * np.sqrt(252)) if len(bl_arr) > 1 and bl_arr.std(ddof=1) > 0 else 0

            win = dd_result['sharpe'] >= bl_sh
            if win:
                wf_wins += 1
            wf_details.append({'window': test_s, 'sharpe': round(dd_result['sharpe'], 3),
                               'bl_sharpe': round(bl_sh, 3), 'win': win})

        wf_pass = wf_wins >= len(WF_WINDOWS) * 0.6
        print(f'    WF: {wf_wins}/{len(WF_WINDOWS)}  PASS={wf_pass}')

        # Gate 3: Era Stability
        print('    Gate 3: Era Stability...')
        era_results = []
        for era_name, (es, ee) in ERA_SEGMENTS.items():
            ts_s = pd.Timestamp(es, tz='UTC')
            ts_e = pd.Timestamp(ee, tz='UTC')
            era_trades = [t for t in trades
                          if ts_s <= pd.Timestamp(t.exit_time) <= ts_e]
            if len(era_trades) < 30:
                era_results.append({'era': era_name, 'sharpe': 0, 'bl_sharpe': 0})
                continue

            dd_result = apply_dd_responsive(era_trades, dd_t, lm, rec)
            bl_daily = defaultdict(float)
            for t in era_trades:
                d = str(pd.Timestamp(t.exit_time).date())
                bl_daily[d] += t.pnl
            bl_arr = np.array(list(bl_daily.values()))
            bl_sh = float(bl_arr.mean() / bl_arr.std(ddof=1) * np.sqrt(252)) if len(bl_arr) > 1 and bl_arr.std(ddof=1) > 0 else 0

            era_results.append({'era': era_name, 'sharpe': round(dd_result['sharpe'], 3),
                                'bl_sharpe': round(bl_sh, 3), 'n': len(era_trades)})
            print(f'      {era_name}: Sharpe={dd_result["sharpe"]:.3f} (bl={bl_sh:.3f})  n={len(era_trades)}')

        era_sharpes = [e['sharpe'] for e in era_results]
        era_pass = all(s > 0 for s in era_sharpes) and min(era_sharpes) > 2.0
        print(f'    Era: min={min(era_sharpes):.3f}  PASS={era_pass}')

        overall = kf_pass and wf_pass and era_pass
        gate_results[label] = {
            'params': {'dd_thresh': dd_t, 'lot_mult': lm, 'recovery': rec},
            'kfold': {'sharpes': [round(s, 3) for s in kf_sharpes], 'wins': kf_wins, 'pass': kf_pass},
            'walk_forward': {'wins': wf_wins, 'total': len(WF_WINDOWS), 'pass': wf_pass, 'details': wf_details},
            'era': {'results': era_results, 'pass': era_pass},
            'overall_pass': overall,
        }
        tag = '[GO]' if overall else '[NO-GO]'
        print(f'    Overall: {tag}')

    save('phase2_three_gate', gate_results)

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: 5/11-5/12 Scenario Replay
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 3: 5/11-5/12 Scenario Replay')
    print('=' * 80)

    # Simulate a sequence mimicking the real drawdown: 4 Cap trades
    # Use actual backtest trades from late April / May period
    may_start = pd.Timestamp('2026-04-01', tz='UTC')
    may_trades = [t for t in trades if pd.Timestamp(t.exit_time) >= may_start]
    print(f'  Trades from 2026-04-01 onward: {len(may_trades)}')

    if may_trades:
        scenario_results = {}
        for cand in top_candidates:
            dd_t = cand['dd_thresh']
            lm = cand['lot_mult']
            rec = cand['recovery']
            label = cand['label']

            result = apply_dd_responsive(may_trades, dd_t, lm, rec)
            bl_pnl = sum(t.pnl for t in may_trades)

            scenario_results[label] = {
                'dd_pnl': result['total_pnl'],
                'baseline_pnl': round(bl_pnl, 2),
                'saved': round(bl_pnl - result['total_pnl'], 2) if bl_pnl < 0 else 0,
                'n_reduced': result['n_reduced'],
                'max_dd': result['max_dd'],
            }
            print(f'  {label}: PnL ${result["total_pnl"]:.0f} (bl: ${bl_pnl:.0f})  '
                  f'reduced={result["n_reduced"]} trades  MaxDD=${result["max_dd"]:.0f}')

        save('phase3_scenario_replay', scenario_results)
    else:
        print('  No May trades in backtest (data may not extend that far)')
        save('phase3_scenario_replay', {'status': 'no_may_trades'})

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Interaction with R202 trail params
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 4: R202 Trail Param Interaction')
    print('=' * 80)

    # Test dd_responsive with current production trail (0.06/0.015)
    # vs the original LIVE_PARITY trail (regime-based)
    trail_configs = {
        'production_0.06_0.015': {
            'trailing_activate_atr': 0.06,
            'trailing_distance_atr': 0.015,
            'regime_config': None,
        },
        'baseline_regime': {},  # Uses LIVE_PARITY_KWARGS defaults
    }

    phase4 = {}
    for trail_label, trail_kw in trail_configs.items():
        print(f'\n  --- {trail_label} ---')
        kw_override = dict(LIVE_PARITY_KWARGS)
        if trail_kw:
            kw_override.update(trail_kw)
        if trail_kw.get('regime_config') is None and 'regime_config' in trail_kw:
            kw_override.pop('regime_config', None)

        get_orb_strategy().reset_daily()
        signals_mod._friday_close_price = None
        signals_mod._gap_traded_today = False
        engine = BacktestEngine(data.m15_df, data.h1_df, **kw_override)
        trail_trades = engine.run()

        bl_daily = defaultdict(float)
        for t in trail_trades:
            d = str(pd.Timestamp(t.exit_time).date())
            bl_daily[d] += t.pnl
        bl_arr = np.array(list(bl_daily.values()))
        bl_sh = float(bl_arr.mean() / bl_arr.std(ddof=1) * np.sqrt(252)) if len(bl_arr) > 1 and bl_arr.std(ddof=1) > 0 else 0

        trail_results = {'baseline_sharpe': round(bl_sh, 3), 'n_trades': len(trail_trades)}

        # Apply best dd_responsive candidate
        if top_candidates:
            best = top_candidates[0]
            dd_result = apply_dd_responsive(trail_trades, best['dd_thresh'],
                                            best['lot_mult'], best['recovery'])
            trail_results['dd_sharpe'] = dd_result['sharpe']
            trail_results['dd_pnl'] = dd_result['total_pnl']
            trail_results['dd_max_dd'] = dd_result['max_dd']
            trail_results['delta_sharpe'] = round(dd_result['sharpe'] - bl_sh, 3)

            print(f'    Baseline Sharpe: {bl_sh:.3f}  DD-responsive: {dd_result["sharpe"]:.3f}  '
                  f'(delta={dd_result["sharpe"] - bl_sh:+.3f})')

        phase4[trail_label] = trail_results

    save('phase4_trail_interaction', phase4)

    # ═══════════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('FINAL SUMMARY')
    print('=' * 80)

    summary = {
        'baseline_sharpe': round(bl_sharpe, 3),
        'baseline_pnl': round(bl_total, 2),
        'best_config': best_config,
        'top_candidates': top_candidates,
        'gate_results': {k: {
            'pass': v['overall_pass'],
            'kf': v['kfold']['wins'],
            'wf': v['walk_forward']['wins'],
            'era_min': min(e['sharpe'] for e in v['era']['results']),
        } for k, v in gate_results.items()},
        'trail_interaction': phase4,
    }

    go_candidates = [k for k, v in gate_results.items() if v['overall_pass']]
    if go_candidates:
        print(f'  GO candidates: {go_candidates}')
        for gc in go_candidates:
            g = gate_results[gc]
            print(f'    {gc}: KF={g["kfold"]["wins"]}/6  WF={g["walk_forward"]["wins"]}/{g["walk_forward"]["total"]}  '
                  f'Era_min={min(e["sharpe"] for e in g["era"]["results"]):.3f}')
        summary['verdict'] = 'GO'
    else:
        print('  No candidates passed 3-Gate.')
        summary['verdict'] = 'NO-GO'

    save('R207_summary', summary)

    elapsed = time.time() - t_start
    print(f'\n  Total runtime: {elapsed:.0f}s')
    print(f'  All results in {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
