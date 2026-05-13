#!/usr/bin/env python3
"""R215: M15 RSI Complete Validation
=====================================
Engine: BacktestEngine (native M15 RSI support)
Filters: ALL active (Choppy Gate, ATR Pctl, regime config, ADX block)

M15 RSI was disabled in live on 2026-04-25 due to:
  "R4-8 PnL=-$339, 实盘持续亏损, 本周4笔WR25% PnL=-$11"

This experiment determines whether M15 RSI has a real edge or not.

Phase 1: Baseline with LIVE_PARITY_KWARGS (all strategies active)
         Extract M15 RSI trades.
Phase 2: M15 RSI isolation — vary RSI thresholds and max_hold
Phase 3: 3-Gate Validation (K-Fold, Walk-Forward, Era Stability)
Phase 4: Sanity Gate — cross-reference with R211 live data
Phase 5: Kill-or-Keep Decision
"""
from __future__ import annotations
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.runner import DataBundle, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine
import indicators as signals_mod
from indicators import get_orb_strategy

OUTPUT_DIR = Path("results/r215_m15_rsi_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LIVE_PERIOD = ("2026-03-25", "2026-04-25")

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
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    print(f'  -> saved {p}')


def calc_stats(trades):
    if not trades:
        return {'n': 0, 'pnl': 0, 'sharpe': 0, 'win_rate': 0, 'avg_pnl': 0, 'max_dd': 0}
    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    cum = np.cumsum(pnls)
    dd = np.maximum.accumulate(cum) - cum
    sharpe = float(pnls.mean() / max(pnls.std(ddof=1), 1e-9) * np.sqrt(252 * 4)) if n > 1 else 0
    return {
        'n': n,
        'pnl': round(float(pnls.sum()), 2),
        'sharpe': round(sharpe, 3),
        'win_rate': round(100 * (pnls > 0).sum() / n, 2),
        'avg_pnl': round(float(pnls.mean()), 2),
        'max_dd': round(float(dd.max()), 2),
    }


def filter_period(trades, start, end, strat=None):
    ts_s = pd.Timestamp(start, tz='UTC')
    ts_e = pd.Timestamp(end, tz='UTC')
    out = [t for t in trades if ts_s <= pd.Timestamp(t.entry_time) < ts_e]
    if strat:
        out = [t for t in out if t.strategy == strat]
    return out


def run_engine(data, **extra_kwargs):
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    get_orb_strategy().reset_daily()
    kwargs = {**LIVE_PARITY_KWARGS, **extra_kwargs}
    engine = BacktestEngine(data.m15_df, data.h1_df, **kwargs)
    return engine.run()


def main():
    t_start = time.time()
    print('=' * 80)
    print('R215: M15 RSI Complete Validation')
    print('=' * 80)
    print('  Engine: BacktestEngine (native)')
    print('  Filters: Choppy Gate, ATR Pctl, regime, ADX block — ALL ACTIVE')

    data = DataBundle.load_default()

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Baseline — extract M15 RSI from full portfolio run
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 1: Baseline (M15 RSI within full portfolio)')
    print('=' * 80)

    all_trades = run_engine(data)
    rsi_trades = [t for t in all_trades if t.strategy == 'm15_rsi']
    kc_trades = [t for t in all_trades if t.strategy == 'keltner']

    rsi_full = calc_stats(rsi_trades)
    kc_full = calc_stats(kc_trades)

    print(f'  Keltner:  n={kc_full["n"]:>5}  Sharpe={kc_full["sharpe"]:.3f}  PnL=${kc_full["pnl"]:.0f}')
    print(f'  M15 RSI:  n={rsi_full["n"]:>5}  Sharpe={rsi_full["sharpe"]:.3f}  PnL=${rsi_full["pnl"]:.0f}  WR={rsi_full["win_rate"]:.1f}%')

    # Era breakdown
    phase1_eras = {}
    for era_name, (es, ee) in ERA_SEGMENTS.items():
        era_rsi = filter_period(rsi_trades, es, ee)
        s = calc_stats(era_rsi)
        phase1_eras[era_name] = s
        print(f'    {era_name:<30} n={s["n"]:>5}  Sharpe={s["sharpe"]:.3f}  PnL=${s["pnl"]:.0f}')

    # Live period
    live_rsi = filter_period(rsi_trades, LIVE_PERIOD[0], LIVE_PERIOD[1])
    live_stats = calc_stats(live_rsi)
    print(f'  Live period ({LIVE_PERIOD[0]}->{LIVE_PERIOD[1]}): n={live_stats["n"]}  PnL=${live_stats["pnl"]:.2f}')

    phase1 = {
        'rsi_full': rsi_full, 'kc_full': kc_full,
        'eras': phase1_eras, 'live_period': live_stats,
    }
    save('phase1_baseline', phase1)

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Parameter sweep (RSI thresholds, max_hold, ADX filter)
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 2: Parameter Sweep')
    print('=' * 80)

    rsi_thresholds = [
        (10, 90), (15, 85), (20, 80), (25, 75), (30, 70),
    ]
    max_holds = [10, 15, 20, 30]
    adx_filters = [0, 30, 40, 50]

    phase2 = []
    best_sharpe = -999
    best_config = None

    # RSI threshold sweep (with default max_hold=15, adx=40)
    print(f'\n  RSI threshold sweep:')
    for buy_th, sell_th in rsi_thresholds:
        trades = run_engine(data,
                            rsi_buy_threshold=buy_th,
                            rsi_sell_threshold=sell_th,
                            rsi_adx_filter=40,
                            rsi_max_hold_m15=15)
        rsi_t = [t for t in trades if t.strategy == 'm15_rsi']
        s = calc_stats(rsi_t)
        label = f'RSI_{buy_th}_{sell_th}'
        print(f'    {label:<20} n={s["n"]:>5}  Sharpe={s["sharpe"]:.3f}  '
              f'PnL=${s["pnl"]:.0f}  WR={s["win_rate"]:.1f}%')
        phase2.append({'label': label, **s})
        if s['sharpe'] > best_sharpe and s['n'] >= 30:
            best_sharpe = s['sharpe']
            best_config = {'buy': buy_th, 'sell': sell_th, 'mh': 15, 'adx': 40}

    # Max hold sweep (with best RSI or default 15/85)
    print(f'\n  Max hold sweep (RSI 15/85, ADX 40):')
    for mh in max_holds:
        trades = run_engine(data,
                            rsi_buy_threshold=15,
                            rsi_sell_threshold=85,
                            rsi_adx_filter=40,
                            rsi_max_hold_m15=mh)
        rsi_t = [t for t in trades if t.strategy == 'm15_rsi']
        s = calc_stats(rsi_t)
        label = f'MH_{mh}'
        print(f'    {label:<20} n={s["n"]:>5}  Sharpe={s["sharpe"]:.3f}  '
              f'PnL=${s["pnl"]:.0f}  WR={s["win_rate"]:.1f}%')
        phase2.append({'label': label, **s})
        if s['sharpe'] > best_sharpe and s['n'] >= 30:
            best_sharpe = s['sharpe']
            best_config = {'buy': 15, 'sell': 85, 'mh': mh, 'adx': 40}

    # ADX filter sweep
    print(f'\n  ADX filter sweep (RSI 15/85, MH 15):')
    for adx in adx_filters:
        trades = run_engine(data,
                            rsi_buy_threshold=15,
                            rsi_sell_threshold=85,
                            rsi_adx_filter=adx,
                            rsi_max_hold_m15=15)
        rsi_t = [t for t in trades if t.strategy == 'm15_rsi']
        s = calc_stats(rsi_t)
        label = f'ADX_{adx}'
        print(f'    {label:<20} n={s["n"]:>5}  Sharpe={s["sharpe"]:.3f}  '
              f'PnL=${s["pnl"]:.0f}  WR={s["win_rate"]:.1f}%')
        phase2.append({'label': label, **s})
        if s['sharpe'] > best_sharpe and s['n'] >= 30:
            best_sharpe = s['sharpe']
            best_config = {'buy': 15, 'sell': 85, 'mh': 15, 'adx': adx}

    print(f'\n  Best config: {best_config}  Sharpe={best_sharpe:.3f}')
    save('phase2_param_sweep', {'results': phase2, 'best': best_config, 'best_sharpe': best_sharpe})

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: 3-Gate Validation on best config
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 3: 3-Gate Validation')
    print('=' * 80)

    if best_config is None:
        print('  No valid config found — skipping 3-Gate')
        phase3 = {'error': 'no valid config'}
        save('phase3_three_gate', phase3)
    else:
        bc = best_config
        print(f'  Testing: RSI {bc["buy"]}/{bc["sell"]}, MH={bc["mh"]}, ADX={bc["adx"]}')

        # Gate 1: 6-Fold CV
        print(f'\n  Gate 1: 6-Fold Cross-Validation')
        all_rsi = run_engine(data,
                             rsi_buy_threshold=bc['buy'],
                             rsi_sell_threshold=bc['sell'],
                             rsi_adx_filter=bc['adx'],
                             rsi_max_hold_m15=bc['mh'])
        rsi_only = [t for t in all_rsi if t.strategy == 'm15_rsi']
        rsi_pnls = np.array([t.pnl for t in rsi_only])

        n_folds = 6
        fold_size = len(rsi_pnls) // n_folds
        fold_results = []
        kf_pass = 0

        for fold in range(n_folds):
            start = fold * fold_size
            end = start + fold_size if fold < n_folds - 1 else len(rsi_pnls)
            fold_pnls = rsi_pnls[start:end]
            if len(fold_pnls) < 5:
                continue
            mu = fold_pnls.mean()
            std = fold_pnls.std(ddof=1)
            s = mu / max(std, 1e-9) * np.sqrt(252 * 4)
            fold_results.append({'fold': fold + 1, 'n': len(fold_pnls), 'sharpe': round(s, 3)})
            print(f'    Fold {fold+1}: n={len(fold_pnls):>4}  Sharpe={s:.3f}')
            if s > 0:
                kf_pass += 1

        kf_pass_rate = kf_pass / max(len(fold_results), 1)
        gate1_pass = kf_pass_rate >= 0.67
        print(f'    KF: {kf_pass}/{len(fold_results)} positive  PASS={gate1_pass}')

        # Gate 2: Walk-Forward
        print(f'\n  Gate 2: Walk-Forward')
        wf_pass_count = 0
        wf_results = []
        for train_s, train_e, test_s, test_e in WF_WINDOWS:
            test_trades = filter_period(rsi_only, test_s, test_e)
            if len(test_trades) < 5:
                wf_results.append({'test': f'{test_s}->{test_e}', 'n': len(test_trades), 'sharpe': 0, 'skip': True})
                continue
            s = calc_stats(test_trades)
            wf_results.append({'test': f'{test_s}->{test_e}', **s})
            if s['sharpe'] > 0:
                wf_pass_count += 1

        valid_wf = [w for w in wf_results if not w.get('skip')]
        gate2_pass = wf_pass_count / max(len(valid_wf), 1) >= 0.60
        print(f'    WF: {wf_pass_count}/{len(valid_wf)} positive  PASS={gate2_pass}')

        # Gate 3: Era Stability
        print(f'\n  Gate 3: Era Stability')
        era_results = {}
        era_min_sharpe = 999
        for era_name, (es, ee) in ERA_SEGMENTS.items():
            era_t = filter_period(rsi_only, es, ee)
            s = calc_stats(era_t)
            era_results[era_name] = s
            if s['n'] >= 20:
                era_min_sharpe = min(era_min_sharpe, s['sharpe'])
            print(f'    {era_name:<30} n={s["n"]:>4}  Sharpe={s["sharpe"]:.3f}')

        gate3_pass = era_min_sharpe > 0 if era_min_sharpe != 999 else False
        print(f'    Era min Sharpe: {era_min_sharpe:.3f}  PASS={gate3_pass}')

        overall_pass = gate1_pass and gate2_pass and gate3_pass
        print(f'\n  Overall: {"[GO]" if overall_pass else "[NO-GO]"}')
        print(f'    Gate 1 (K-Fold): {"PASS" if gate1_pass else "FAIL"}')
        print(f'    Gate 2 (Walk-Forward): {"PASS" if gate2_pass else "FAIL"}')
        print(f'    Gate 3 (Era Stability): {"PASS" if gate3_pass else "FAIL"}')

        phase3 = {
            'config': bc,
            'gate1_kfold': {'folds': fold_results, 'pass': gate1_pass},
            'gate2_wf': {'results': wf_results, 'pass': gate2_pass, 'positive': wf_pass_count, 'total': len(valid_wf)},
            'gate3_era': {'results': era_results, 'min_sharpe': era_min_sharpe, 'pass': gate3_pass},
            'overall_pass': overall_pass,
        }
        save('phase3_three_gate', phase3)

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Sanity Gate
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 4: Sanity Gate')
    print('=' * 80)

    r211_m15_rsi = {'n': 19, 'pnl': 49.97, 'wr': 63.2}

    # BT live period includes when M15 RSI was ON (before 04-25)
    bt_live = filter_period(rsi_trades, '2026-03-25', '2026-04-25')
    bt_live_stats = calc_stats(bt_live)

    ratio = bt_live_stats['n'] / r211_m15_rsi['n'] if r211_m15_rsi['n'] > 0 else 0
    print(f'  Live (03-25 to 04-25): n={r211_m15_rsi["n"]}  PnL=${r211_m15_rsi["pnl"]:.2f}')
    print(f'  BT same period:        n={bt_live_stats["n"]}  PnL=${bt_live_stats["pnl"]:.2f}')
    print(f'  Ratio: {ratio:.1f}x')

    # Sanity checks
    alerts = []
    if rsi_full['sharpe'] > 5.0:
        alerts.append(f'Sharpe {rsi_full["sharpe"]:.1f} > 5.0')
    if rsi_full['win_rate'] > 85:
        alerts.append(f'WR {rsi_full["win_rate"]:.1f}% > 85%')
    if rsi_full['n'] > 50000:
        alerts.append(f'N={rsi_full["n"]} suspiciously high')

    if alerts:
        print(f'  ALERTS: {"; ".join(alerts)}')
    else:
        print(f'  No sanity alerts. Sharpe={rsi_full["sharpe"]:.2f}, WR={rsi_full["win_rate"]:.1f}%, N={rsi_full["n"]}')

    phase4 = {
        'live': r211_m15_rsi,
        'bt_live_period': bt_live_stats,
        'ratio': round(ratio, 2),
        'alerts': alerts,
    }
    save('phase4_sanity_gate', phase4)

    # ═══════════════════════════════════════════════════════════════
    # Phase 5: Kill-or-Keep Decision
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 5: Kill-or-Keep Decision')
    print('=' * 80)

    decision = 'KILL'
    reasons = []

    if rsi_full['sharpe'] > 1.0:
        reasons.append(f'Full Sharpe {rsi_full["sharpe"]:.2f} > 1.0 (positive)')
    else:
        reasons.append(f'Full Sharpe {rsi_full["sharpe"]:.2f} <= 1.0 (weak/negative)')

    if phase3.get('overall_pass'):
        reasons.append('3-Gate: PASS')
        if rsi_full['sharpe'] > 1.0:
            decision = 'KEEP'
    else:
        gate_fails = []
        if not phase3.get('gate1_kfold', {}).get('pass', False):
            gate_fails.append('K-Fold')
        if not phase3.get('gate2_wf', {}).get('pass', False):
            gate_fails.append('Walk-Forward')
        if not phase3.get('gate3_era', {}).get('pass', False):
            gate_fails.append('Era')
        reasons.append(f'3-Gate: FAIL ({", ".join(gate_fails)})')

    if r211_m15_rsi['pnl'] > 0:
        reasons.append(f'Live PnL positive: ${r211_m15_rsi["pnl"]:.2f}')
    else:
        reasons.append(f'Live PnL negative: ${r211_m15_rsi["pnl"]:.2f}')

    print(f'  Decision: {decision}')
    for r in reasons:
        print(f'    - {r}')

    phase5 = {'decision': decision, 'reasons': reasons}
    save('phase5_decision', phase5)

    # ═══════════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('FINAL SUMMARY')
    print('=' * 80)

    summary = {
        'engine': 'BacktestEngine (native M15 RSI)',
        'filters': 'Choppy Gate + ATR Pctl + regime + ADX block',
        'full_stats': rsi_full,
        'eras': phase1_eras,
        'best_params': best_config,
        'three_gate_pass': phase3.get('overall_pass', False),
        'decision': decision,
        'reasons': reasons,
    }
    save('R215_summary', summary)

    elapsed = time.time() - t_start
    print(f'\n  Total runtime: {elapsed:.0f}s')


if __name__ == '__main__':
    main()
