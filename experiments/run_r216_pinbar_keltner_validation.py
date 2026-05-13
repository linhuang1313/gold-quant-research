#!/usr/bin/env python3
"""R216: Pinbar+Keltner Complete Validation
=============================================
Engine: BacktestEngine (native pinbar_confirmation + pinbar_sr_strategy)
Filters: ALL active via LIVE_PARITY_KWARGS

R11 tested Pinbar using run_variant with kc_ema=25/kc_mult=1.2 (old params).
R216 re-validates with current LIVE_PARITY_KWARGS and full filter stack.

Phase 1: Baseline Keltner (no Pinbar) — reference
Phase 2: Pinbar Confirmation filter on Keltner
         (only enter Keltner when prior H1 bar has aligned Pinbar)
Phase 3: PinbarSR independent strategy
         (Pinbar at S/R zones, independent of Keltner signal)
Phase 4: Signal overlap analysis — Pinbar frequency + Keltner coincidence
Phase 5: 6-Fold K-Fold on any promising config
Phase 6: Sanity Gate
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

OUTPUT_DIR = Path("results/r216_pinbar_keltner")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERA_SEGMENTS = {
    "Pre-COVID (2015-2019)":      ("2015-01-01", "2020-01-01"),
    "COVID+Recovery (2020-2021)": ("2020-01-01", "2022-01-01"),
    "Tightening (2022-2023)":     ("2022-01-01", "2024-01-01"),
    "Recent (2024-2026)":         ("2024-01-01", "2026-06-01"),
}


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
    return engine.run(), engine


def main():
    t_start = time.time()
    print('=' * 80)
    print('R216: Pinbar+Keltner Complete Validation')
    print('=' * 80)
    print('  Engine: BacktestEngine (native)')
    print('  Filters: LIVE_PARITY_KWARGS — ALL ACTIVE')

    data = DataBundle.load_default()

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Baseline — vanilla Keltner, no Pinbar
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 1: Baseline (Keltner only, no Pinbar)')
    print('=' * 80)

    trades_base, eng_base = run_engine(data)
    kc_base = [t for t in trades_base if t.strategy == 'keltner']
    s_base = calc_stats(kc_base)
    print(f'  Keltner baseline: n={s_base["n"]:>5}  Sharpe={s_base["sharpe"]:.3f}  '
          f'PnL=${s_base["pnl"]:.0f}  WR={s_base["win_rate"]:.1f}%  MaxDD=${s_base["max_dd"]:.0f}')

    base_eras = {}
    for era_name, (es, ee) in ERA_SEGMENTS.items():
        era_t = filter_period(kc_base, es, ee)
        s = calc_stats(era_t)
        base_eras[era_name] = s
        print(f'    {era_name:<30} n={s["n"]:>5}  Sharpe={s["sharpe"]:.3f}  PnL=${s["pnl"]:.0f}')

    phase1 = {'baseline': s_base, 'eras': base_eras}
    save('phase1_baseline', phase1)

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Pinbar Confirmation on Keltner
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 2: Pinbar Confirmation (Keltner + prior H1 Pinbar aligned)')
    print('=' * 80)

    trades_pb, eng_pb = run_engine(data, pinbar_confirmation=True)
    kc_pb = [t for t in trades_pb if t.strategy == 'keltner']
    s_pb = calc_stats(kc_pb)
    print(f'  Pinbar-confirmed Keltner: n={s_pb["n"]:>5}  Sharpe={s_pb["sharpe"]:.3f}  '
          f'PnL=${s_pb["pnl"]:.0f}  WR={s_pb["win_rate"]:.1f}%  MaxDD=${s_pb["max_dd"]:.0f}')
    print(f'  Skipped by Pinbar filter: {eng_pb.skipped_pinbar}')
    print(f'  Filter pass rate: {s_pb["n"]}/{s_base["n"]} = '
          f'{100*s_pb["n"]/max(s_base["n"],1):.1f}%')

    delta_sharpe = s_pb['sharpe'] - s_base['sharpe']
    print(f'  Sharpe delta: {delta_sharpe:+.3f}')

    pb_eras = {}
    for era_name, (es, ee) in ERA_SEGMENTS.items():
        era_t = filter_period(kc_pb, es, ee)
        s = calc_stats(era_t)
        pb_eras[era_name] = s
        b = base_eras[era_name]
        print(f'    {era_name:<30} n={s["n"]:>5}  Sharpe={s["sharpe"]:.3f}  '
              f'delta={s["sharpe"]-b["sharpe"]:+.3f}')

    phase2 = {
        'pinbar_confirmed': s_pb,
        'eras': pb_eras,
        'skipped_pinbar': eng_pb.skipped_pinbar,
        'delta_sharpe': round(delta_sharpe, 3),
        'pass_rate': round(100 * s_pb['n'] / max(s_base['n'], 1), 2),
    }
    save('phase2_pinbar_confirmation', phase2)

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: PinbarSR Independent Strategy
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 3: PinbarSR Independent Strategy')
    print('=' * 80)

    sr_atr_zones = [1.0, 1.5, 2.0, 2.5, 3.0]
    phase3_results = []
    best_sr_sharpe = -999
    best_sr_zone = None

    for zone in sr_atr_zones:
        trades_sr, eng_sr = run_engine(data,
                                       pinbar_sr_strategy=True,
                                       pinbar_sr_atr_zone=zone)
        sr_trades = [t for t in trades_sr if t.strategy == 'pinbar_sr']
        kc_trades = [t for t in trades_sr if t.strategy == 'keltner']
        s_sr = calc_stats(sr_trades)
        s_kc = calc_stats(kc_trades)
        print(f'  Zone={zone:.1f}ATR: PinbarSR n={s_sr["n"]:>4}  Sharpe={s_sr["sharpe"]:.3f}  '
              f'PnL=${s_sr["pnl"]:.0f}  |  Keltner n={s_kc["n"]:>5}  Sharpe={s_kc["sharpe"]:.3f}')

        phase3_results.append({
            'zone': zone, 'pinbar_sr': s_sr, 'keltner': s_kc,
            'pinbar_sr_entries': eng_sr.pinbar_sr_entries,
        })
        if s_sr['sharpe'] > best_sr_sharpe and s_sr['n'] >= 20:
            best_sr_sharpe = s_sr['sharpe']
            best_sr_zone = zone

    print(f'\n  Best PinbarSR zone: {best_sr_zone}  Sharpe={best_sr_sharpe:.3f}')
    save('phase3_pinbar_sr', {'results': phase3_results, 'best_zone': best_sr_zone,
                               'best_sharpe': best_sr_sharpe})

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Signal Overlap Analysis
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 4: Signal Overlap Analysis')
    print('=' * 80)

    h1 = data.h1_df
    total_bars = len(h1)
    pinbar_bull_bars = int((h1['pinbar_bull'] > 0).sum()) if 'pinbar_bull' in h1.columns else 0
    pinbar_bear_bars = int((h1['pinbar_bear'] > 0).sum()) if 'pinbar_bear' in h1.columns else 0
    total_pinbar = pinbar_bull_bars + pinbar_bear_bars

    print(f'  Total H1 bars: {total_bars}')
    print(f'  Pinbar bull: {pinbar_bull_bars} ({100*pinbar_bull_bars/total_bars:.2f}%)')
    print(f'  Pinbar bear: {pinbar_bear_bars} ({100*pinbar_bear_bars/total_bars:.2f}%)')
    print(f'  Total pinbar: {total_pinbar} ({100*total_pinbar/total_bars:.2f}%)')

    # Overlap with Keltner entries
    kc_entry_times = set()
    for t in kc_base:
        et = pd.Timestamp(t.entry_time)
        kc_entry_times.add(et.floor('h'))

    overlap = 0
    for i in range(1, len(h1)):
        bar_time = h1.index[i]
        prev = h1.iloc[i - 1]
        if bar_time in kc_entry_times:
            pb = float(prev.get('pinbar_bull', 0)) + float(prev.get('pinbar_bear', 0))
            if pb > 0:
                overlap += 1

    overlap_rate = 100 * overlap / max(len(kc_entry_times), 1)
    print(f'\n  Keltner entries: {len(kc_entry_times)}')
    print(f'  Keltner entries WITH prior-bar Pinbar: {overlap}')
    print(f'  Overlap rate: {overlap_rate:.2f}%')

    phase4 = {
        'total_bars': total_bars,
        'pinbar_bull': pinbar_bull_bars,
        'pinbar_bear': pinbar_bear_bars,
        'total_pinbar': total_pinbar,
        'pinbar_rate_pct': round(100 * total_pinbar / total_bars, 3),
        'kc_entries': len(kc_entry_times),
        'overlap': overlap,
        'overlap_rate_pct': round(overlap_rate, 2),
    }
    save('phase4_overlap', phase4)

    # ═══════════════════════════════════════════════════════════════
    # Phase 5: 6-Fold K-Fold on Pinbar Confirmation
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 5: 6-Fold K-Fold Validation')
    print('=' * 80)

    # 5a: Pinbar Confirmation K-Fold
    print('\n  5a: Pinbar Confirmation on Keltner')
    if s_pb['n'] < 30:
        print(f'    Skipping: only {s_pb["n"]} trades (need >=30)')
        kf_pb = {'skip': True, 'reason': f'n={s_pb["n"]} < 30'}
    else:
        pnls_pb = np.array([t.pnl for t in kc_pb])
        n_folds = 6
        fold_size = len(pnls_pb) // n_folds
        fold_results_pb = []
        kf_pass_pb = 0

        for fold in range(n_folds):
            start = fold * fold_size
            end = start + fold_size if fold < n_folds - 1 else len(pnls_pb)
            fp = pnls_pb[start:end]
            if len(fp) < 5:
                continue
            s = float(fp.mean() / max(fp.std(ddof=1), 1e-9) * np.sqrt(252 * 4))
            fold_results_pb.append({'fold': fold + 1, 'n': len(fp), 'sharpe': round(s, 3)})
            print(f'    Fold {fold+1}: n={len(fp):>4}  Sharpe={s:.3f}')
            if s > 0:
                kf_pass_pb += 1

        kf_rate_pb = kf_pass_pb / max(len(fold_results_pb), 1)
        kf_pb = {'folds': fold_results_pb, 'pass_count': kf_pass_pb,
                 'total_folds': len(fold_results_pb), 'pass_rate': round(kf_rate_pb, 3),
                 'verdict': 'PASS' if kf_rate_pb >= 0.67 else 'FAIL'}
        print(f'    K-Fold: {kf_pass_pb}/{len(fold_results_pb)} positive -> {kf_pb["verdict"]}')

    # 5b: PinbarSR K-Fold (if best_sr_zone found)
    print(f'\n  5b: PinbarSR Independent (zone={best_sr_zone})')
    if best_sr_zone is None or best_sr_sharpe <= 0:
        print(f'    Skipping: no viable PinbarSR config')
        kf_sr = {'skip': True, 'reason': 'no viable config'}
    else:
        trades_best_sr, _ = run_engine(data,
                                       pinbar_sr_strategy=True,
                                       pinbar_sr_atr_zone=best_sr_zone)
        sr_only = [t for t in trades_best_sr if t.strategy == 'pinbar_sr']
        if len(sr_only) < 30:
            print(f'    Skipping: only {len(sr_only)} trades')
            kf_sr = {'skip': True, 'reason': f'n={len(sr_only)} < 30'}
        else:
            pnls_sr = np.array([t.pnl for t in sr_only])
            fold_results_sr = []
            kf_pass_sr = 0
            fold_size = len(pnls_sr) // 6

            for fold in range(6):
                start = fold * fold_size
                end = start + fold_size if fold < 5 else len(pnls_sr)
                fp = pnls_sr[start:end]
                if len(fp) < 5:
                    continue
                s = float(fp.mean() / max(fp.std(ddof=1), 1e-9) * np.sqrt(252 * 4))
                fold_results_sr.append({'fold': fold + 1, 'n': len(fp), 'sharpe': round(s, 3)})
                print(f'    Fold {fold+1}: n={len(fp):>4}  Sharpe={s:.3f}')
                if s > 0:
                    kf_pass_sr += 1

            kf_rate_sr = kf_pass_sr / max(len(fold_results_sr), 1)
            kf_sr = {'folds': fold_results_sr, 'pass_count': kf_pass_sr,
                     'total_folds': len(fold_results_sr), 'pass_rate': round(kf_rate_sr, 3),
                     'verdict': 'PASS' if kf_rate_sr >= 0.67 else 'FAIL'}
            print(f'    K-Fold: {kf_pass_sr}/{len(fold_results_sr)} positive -> {kf_sr["verdict"]}')

    # 5c: K-Fold on baseline Keltner (for comparison)
    print(f'\n  5c: Baseline Keltner K-Fold (reference)')
    pnls_base = np.array([t.pnl for t in kc_base])
    fold_results_base = []
    kf_pass_base = 0
    fold_size = len(pnls_base) // 6

    for fold in range(6):
        start = fold * fold_size
        end = start + fold_size if fold < 5 else len(pnls_base)
        fp = pnls_base[start:end]
        if len(fp) < 5:
            continue
        s = float(fp.mean() / max(fp.std(ddof=1), 1e-9) * np.sqrt(252 * 4))
        fold_results_base.append({'fold': fold + 1, 'n': len(fp), 'sharpe': round(s, 3)})
        print(f'    Fold {fold+1}: n={len(fp):>4}  Sharpe={s:.3f}')
        if s > 0:
            kf_pass_base += 1

    kf_base = {'folds': fold_results_base, 'pass_count': kf_pass_base,
               'total_folds': len(fold_results_base)}

    phase5 = {
        'pinbar_confirmation_kfold': kf_pb,
        'pinbar_sr_kfold': kf_sr,
        'baseline_kfold': kf_base,
    }
    save('phase5_kfold', phase5)

    # ═══════════════════════════════════════════════════════════════
    # Phase 6: Sanity Gate + Final Verdict
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 6: Sanity Gate + Final Verdict')
    print('=' * 80)

    alerts = []
    if s_pb['sharpe'] > 15:
        alerts.append(f'Pinbar-confirmed Sharpe {s_pb["sharpe"]:.1f} suspiciously high')
    if s_pb['win_rate'] > 95:
        alerts.append(f'Pinbar-confirmed WR {s_pb["win_rate"]:.1f}% suspiciously high')

    if alerts:
        print(f'  ALERTS: {"; ".join(alerts)}')
    else:
        print(f'  No sanity alerts.')

    # Final verdict
    print('\n  --- FINAL VERDICT ---')
    verdict = {
        'pinbar_confirmation': {
            'sharpe': s_pb['sharpe'],
            'delta_vs_baseline': round(delta_sharpe, 3),
            'kfold_verdict': kf_pb.get('verdict', 'SKIP'),
            'recommendation': 'UNKNOWN',
        },
        'pinbar_sr_strategy': {
            'best_zone': best_sr_zone,
            'best_sharpe': round(best_sr_sharpe, 3),
            'kfold_verdict': kf_sr.get('verdict', 'SKIP'),
            'recommendation': 'UNKNOWN',
        },
    }

    # Pinbar confirmation decision
    if delta_sharpe <= 0:
        verdict['pinbar_confirmation']['recommendation'] = 'REJECT'
        print(f'  Pinbar Confirmation: REJECT (Sharpe delta {delta_sharpe:+.3f} <= 0)')
    elif kf_pb.get('verdict') == 'FAIL':
        verdict['pinbar_confirmation']['recommendation'] = 'REJECT'
        print(f'  Pinbar Confirmation: REJECT (K-Fold FAIL)')
    elif kf_pb.get('verdict') == 'PASS' and delta_sharpe > 0.5:
        verdict['pinbar_confirmation']['recommendation'] = 'CONSIDER'
        print(f'  Pinbar Confirmation: CONSIDER (K-Fold PASS, delta={delta_sharpe:+.3f})')
    elif kf_pb.get('skip'):
        verdict['pinbar_confirmation']['recommendation'] = 'REJECT'
        print(f'  Pinbar Confirmation: REJECT (too few trades for K-Fold)')
    else:
        verdict['pinbar_confirmation']['recommendation'] = 'WEAK'
        print(f'  Pinbar Confirmation: WEAK (delta={delta_sharpe:+.3f})')

    # PinbarSR decision
    if best_sr_sharpe <= 0:
        verdict['pinbar_sr_strategy']['recommendation'] = 'REJECT'
        print(f'  PinbarSR Strategy:   REJECT (Sharpe {best_sr_sharpe:.3f} <= 0)')
    elif kf_sr.get('verdict') == 'FAIL':
        verdict['pinbar_sr_strategy']['recommendation'] = 'REJECT'
        print(f'  PinbarSR Strategy:   REJECT (K-Fold FAIL)')
    elif kf_sr.get('verdict') == 'PASS':
        verdict['pinbar_sr_strategy']['recommendation'] = 'CONSIDER'
        print(f'  PinbarSR Strategy:   CONSIDER (K-Fold PASS)')
    elif kf_sr.get('skip'):
        verdict['pinbar_sr_strategy']['recommendation'] = 'REJECT'
        print(f'  PinbarSR Strategy:   REJECT (insufficient data)')
    else:
        verdict['pinbar_sr_strategy']['recommendation'] = 'WEAK'
        print(f'  PinbarSR Strategy:   WEAK')

    phase6 = {'alerts': alerts, 'verdict': verdict}
    save('phase6_verdict', phase6)

    # Summary
    summary = {
        'engine': 'BacktestEngine (LIVE_PARITY_KWARGS)',
        'filters': 'Choppy Gate + ATR Pctl + regime + ADX — ALL ACTIVE',
        'baseline': s_base,
        'pinbar_confirmation': s_pb,
        'pinbar_sr_best': {'zone': best_sr_zone, 'sharpe': best_sr_sharpe},
        'overlap_rate_pct': phase4['overlap_rate_pct'],
        'kfold_pb': kf_pb.get('verdict', 'SKIP'),
        'kfold_sr': kf_sr.get('verdict', 'SKIP'),
        'verdict': verdict,
    }
    save('R216_summary', summary)

    elapsed = time.time() - t_start
    print(f'\n  Total runtime: {elapsed:.0f}s')


if __name__ == '__main__':
    main()
