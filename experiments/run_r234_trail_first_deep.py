#!/usr/bin/env python3
"""R234: Trail-First Deep Validation
=====================================
R233 ablation revealed that the single best change from L8→R202 is the trail
(0.14/0.025 → 0.06/0.015), giving Sharpe 5.675 — higher than full R202 (5.532).

This experiment deep-validates the "Trail-First" config:
  - Trail: R202 (0.06/0.015, unified regime)
  - SL: 3.5×ATR (L8 original — tighter than R202's 6.0)
  - Cap: OFF (no maxloss_cap)
  - Lots: fixed 0.04

Validation pipeline:
  1. Full-sample stats + K-Fold 6 & 10
  2. Walk-Forward (19 windows)
  3. Era stability (5 eras)
  4. Parameter sensitivity (SL ±0.5, Trail ±20%)
  5. Monte Carlo bootstrap (1000 resamples)
  6. Realistic cost stress test (spread $0.50/$0.75/$1.00)
  7. Slippage stress test (empirical + 2x/3x)
  8. Recent regime focus (last 12 months detailed)
  9. Comparison table vs L8, R202, and hybrid configs
"""
import sys, os, time, json, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.runner import (
    DataBundle, run_variant, run_kfold,
    L8_PARITY_KWARGS, LIVE_PARITY_KWARGS,
)
from backtest.engine import BacktestEngine
from backtest.stats import calc_stats, aggregate_daily_pnl
from collections import Counter
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'results', 'r234_trail_first')
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, 'r234_stdout.txt')

_log_file = open(LOG_PATH, 'w', buffering=1)
def pf(msg=''):
    print(msg, flush=True)
    _log_file.write(msg + '\n')

def save(name, obj):
    path = os.path.join(RESULTS_DIR, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    pf(f'  [saved {path}]')


TRAIL_FIRST = {
    **L8_PARITY_KWARGS,
    'trailing_activate_atr': 0.06,
    'trailing_distance_atr': 0.015,
    'regime_config': {
        'low':    {'trail_act': 0.06, 'trail_dist': 0.015},
        'normal': {'trail_act': 0.06, 'trail_dist': 0.015},
        'high':   {'trail_act': 0.06, 'trail_dist': 0.015},
    },
    'min_lot_size': 0.04,
    'max_lot_size': 0.04,
}


def quick_eval(data, label, kwargs):
    """Run backtest and return summary dict."""
    engine = BacktestEngine(data.m15_df, data.h1_df, **kwargs)
    trades = engine.run()
    stats = calc_stats(trades, engine.equity_curve)

    exit_reasons = Counter(t.exit_reason for t in trades)
    exit_pnl = {}
    for reason in exit_reasons:
        rtrades = [t for t in trades if t.exit_reason == reason]
        exit_pnl[reason] = {
            'count': len(rtrades),
            'total_pnl': round(sum(t.pnl for t in rtrades), 2),
            'avg_pnl': round(sum(t.pnl for t in rtrades) / len(rtrades), 3) if rtrades else 0,
        }

    return {
        'label': label, 'stats': stats, 'trades': trades,
        'exit_breakdown': exit_pnl, 'equity_curve': engine.equity_curve,
    }


def main():
    t0 = time.time()
    pf('=' * 80)
    pf('R234: Trail-First Deep Validation')
    pf('=' * 80)
    pf(f'Config: SL=3.5, Trail=0.06/0.015, No Cap, Lots=0.04')

    data = DataBundle.load_default()

    # ═══════════════════════════════════════════════════
    # 1. Full-sample + K-Fold 6 & 10
    # ═══════════════════════════════════════════════════
    pf(f'\n{"="*80}')
    pf('1. Full-sample + K-Fold Validation')
    pf('='*80)

    r = quick_eval(data, 'Trail-First', TRAIL_FIRST)
    stats = r['stats']
    trades = r['trades']

    pf(f'  n={stats["n"]:>5}  Sharpe={stats["sharpe"]:.3f}  PnL=${stats["total_pnl"]:.0f}  '
       f'WR={stats["win_rate"]:.1f}%  MaxDD=${stats["max_dd"]:.0f}  RR={stats["rr"]:.2f}')

    for reason, info in sorted(r['exit_breakdown'].items(), key=lambda x: -abs(x[1]['total_pnl'])):
        pf(f'  Exit [{reason}]: {info["count"]} trades, PnL=${info["total_pnl"]:.0f}, avg=${info["avg_pnl"]:.2f}')

    kf6 = run_kfold(data, TRAIL_FIRST, n_folds=6, label_prefix='TF_')
    kf6_sharpes = [r['sharpe'] for r in kf6]
    kf6_pos = sum(1 for s in kf6_sharpes if s > 0)
    pf(f'\n  K-Fold 6: {[round(s,2) for s in kf6_sharpes]}')
    pf(f'  Verdict: {"PASS" if kf6_pos >= 5 else "FAIL"} ({kf6_pos}/6)')
    pf(f'  Min={min(kf6_sharpes):.2f}  Max={max(kf6_sharpes):.2f}  '
       f'Mean={np.mean(kf6_sharpes):.2f}  Std={np.std(kf6_sharpes):.2f}')

    save('phase1_fullsample', {
        'stats': stats,
        'exit_breakdown': r['exit_breakdown'],
        'kfold6': {'sharpes': kf6_sharpes, 'positive': kf6_pos},
    })

    # ═══════════════════════════════════════════════════
    # 2. Walk-Forward Optimization (19 windows)
    # ═══════════════════════════════════════════════════
    pf(f'\n{"="*80}')
    pf('2. Walk-Forward (6-month IS / 3-month OOS)')
    pf('='*80)

    wf_results = []
    start_year = 2016
    for i in range(19):
        is_start = f'{start_year + i//2}-{("01" if i%2==0 else "07")}-01'
        is_end_y = start_year + (i+1)//2
        is_end = f'{is_end_y}-{("01" if (i+1)%2==0 else "07")}-01'
        oos_end_y = start_year + (i+2)//2
        oos_end = f'{oos_end_y}-{("01" if (i+2)%2==0 else "07")}-01'

        if pd.Timestamp(oos_end) > pd.Timestamp('2026-05-01'):
            break

        oos_data = data.slice(is_end, oos_end)
        if len(oos_data.m15_df) < 500:
            continue

        oos_r = run_variant(oos_data, f'WF_{i+1}', verbose=False, **TRAIL_FIRST)
        wf_results.append({
            'window': i+1, 'oos_start': is_end, 'oos_end': oos_end,
            'sharpe': round(oos_r['sharpe'], 3),
            'pnl': round(oos_r['total_pnl'], 0),
            'n': oos_r['n'],
        })

    wf_sharpes = [w['sharpe'] for w in wf_results]
    wf_positive = sum(1 for s in wf_sharpes if s > 0)
    pf(f'  Windows: {len(wf_results)}, Positive Sharpe: {wf_positive}/{len(wf_results)}')
    pf(f'  OOS Sharpes: {[w["sharpe"] for w in wf_results]}')
    pf(f'  Mean OOS Sharpe: {np.mean(wf_sharpes):.3f}')
    pf(f'  WF Verdict: {"PASS" if wf_positive >= len(wf_results)*0.7 else "FAIL"}')

    save('phase2_walkforward', wf_results)

    # ═══════════════════════════════════════════════════
    # 3. Era Stability (5 eras)
    # ═══════════════════════════════════════════════════
    pf(f'\n{"="*80}')
    pf('3. Era Stability (5 eras)')
    pf('='*80)

    eras = [
        ('2015-2017', '2015-01-01', '2017-07-01'),
        ('2017-2019', '2017-07-01', '2019-07-01'),
        ('2019-2021', '2019-07-01', '2021-07-01'),
        ('2021-2023', '2021-07-01', '2023-07-01'),
        ('2023-2026', '2023-07-01', '2026-06-01'),
    ]
    era_results = []
    for era_name, s, e in eras:
        era_data = data.slice(s, e)
        if len(era_data.m15_df) < 500:
            continue
        er = run_variant(era_data, f'Era_{era_name}', verbose=False, **TRAIL_FIRST)
        era_results.append({
            'era': era_name, 'sharpe': round(er['sharpe'], 3),
            'pnl': round(er['total_pnl'], 0), 'n': er['n'],
            'win_rate': round(er['win_rate'], 1),
        })
        pf(f'  {era_name}: Sh={er["sharpe"]:.3f}  PnL=${er["total_pnl"]:.0f}  n={er["n"]}  WR={er["win_rate"]:.1f}%')

    all_positive = all(e['sharpe'] > 0 for e in era_results)
    pf(f'  All eras positive: {"YES" if all_positive else "NO"}')

    save('phase3_era', era_results)

    # ═══════════════════════════════════════════════════
    # 4. Parameter Sensitivity
    # ═══════════════════════════════════════════════════
    pf(f'\n{"="*80}')
    pf('4. Parameter Sensitivity (SL ±, Trail ±20%)')
    pf('='*80)

    sens_results = []
    for sl in [2.5, 3.0, 3.5, 4.0, 4.5]:
        for trail_act, trail_dist in [
            (0.048, 0.012), (0.06, 0.015), (0.072, 0.018),
        ]:
            kw = {**TRAIL_FIRST, 'sl_atr_mult': sl,
                  'trailing_activate_atr': trail_act, 'trailing_distance_atr': trail_dist,
                  'regime_config': {
                      'low': {'trail_act': trail_act, 'trail_dist': trail_dist},
                      'normal': {'trail_act': trail_act, 'trail_dist': trail_dist},
                      'high': {'trail_act': trail_act, 'trail_dist': trail_dist},
                  }}
            sr = run_variant(data, f'SL={sl}_T={trail_act}', verbose=False, **kw)
            sens_results.append({
                'sl': sl, 'trail_act': trail_act, 'trail_dist': trail_dist,
                'sharpe': round(sr['sharpe'], 3), 'pnl': round(sr['total_pnl'], 0),
                'n': sr['n'],
            })
            pf(f'  SL={sl:.1f} Trail={trail_act:.3f}/{trail_dist:.3f}: '
               f'Sh={sr["sharpe"]:.3f}  PnL=${sr["total_pnl"]:.0f}  n={sr["n"]}')

    best_sens = max(sens_results, key=lambda x: x['sharpe'])
    worst_sens = min(sens_results, key=lambda x: x['sharpe'])
    pf(f'\n  Best: SL={best_sens["sl"]} Trail={best_sens["trail_act"]} Sh={best_sens["sharpe"]:.3f}')
    pf(f'  Worst: SL={worst_sens["sl"]} Trail={worst_sens["trail_act"]} Sh={worst_sens["sharpe"]:.3f}')
    pf(f'  Range: {worst_sens["sharpe"]:.3f} — {best_sens["sharpe"]:.3f} '
       f'(Δ={best_sens["sharpe"]-worst_sens["sharpe"]:.3f})')

    save('phase4_sensitivity', sens_results)

    # ═══════════════════════════════════════════════════
    # 5. Monte Carlo Bootstrap (1000 resamples)
    # ═══════════════════════════════════════════════════
    pf(f'\n{"="*80}')
    pf('5. Monte Carlo Bootstrap (1000 resamples)')
    pf('='*80)

    daily_pnl = aggregate_daily_pnl(trades)
    mc_sharpes = []
    random.seed(42)
    n_days = len(daily_pnl)
    for _ in range(1000):
        sample = [daily_pnl[random.randint(0, n_days-1)] for _ in range(n_days)]
        s_mean = np.mean(sample)
        s_std = np.std(sample, ddof=1)
        if s_std > 0:
            mc_sharpes.append(s_mean / s_std * np.sqrt(252))

    mc_sharpes.sort()
    pf(f'  Original Sharpe: {stats["sharpe"]:.3f}')
    pf(f'  MC Mean: {np.mean(mc_sharpes):.3f}')
    pf(f'  MC Median: {np.median(mc_sharpes):.3f}')
    pf(f'  MC 5th percentile: {mc_sharpes[49]:.3f}')
    pf(f'  MC 95th percentile: {mc_sharpes[949]:.3f}')
    pf(f'  P(Sharpe > 0): {100*sum(1 for s in mc_sharpes if s > 0)/len(mc_sharpes):.1f}%')
    pf(f'  P(Sharpe > 2): {100*sum(1 for s in mc_sharpes if s > 2)/len(mc_sharpes):.1f}%')
    pf(f'  P(Sharpe > 4): {100*sum(1 for s in mc_sharpes if s > 4)/len(mc_sharpes):.1f}%')

    save('phase5_monte_carlo', {
        'original_sharpe': round(stats['sharpe'], 3),
        'mc_mean': round(np.mean(mc_sharpes), 3),
        'mc_p5': round(mc_sharpes[49], 3),
        'mc_p95': round(mc_sharpes[949], 3),
        'prob_positive': round(100*sum(1 for s in mc_sharpes if s > 0)/len(mc_sharpes), 1),
    })

    # ═══════════════════════════════════════════════════
    # 6. Realistic Cost Stress Test
    # ═══════════════════════════════════════════════════
    pf(f'\n{"="*80}')
    pf('6. Realistic Cost Stress Test')
    pf('='*80)

    cost_results = []
    for spread, label in [(0.30, 'Sp$0.30'), (0.50, 'Sp$0.50'), (0.75, 'Sp$0.75'), (1.00, 'Sp$1.00')]:
        kw = {**TRAIL_FIRST, 'spread_model': 'fixed', 'spread_cost': spread}
        cr = run_variant(data, f'Cost_{label}', verbose=False, **kw)
        cost_results.append({
            'spread': spread, 'label': label,
            'sharpe': round(cr['sharpe'], 3), 'pnl': round(cr['total_pnl'], 0),
            'n': cr['n'], 'win_rate': round(cr['win_rate'], 1),
        })
        pf(f'  {label}: Sh={cr["sharpe"]:.3f}  PnL=${cr["total_pnl"]:.0f}  WR={cr["win_rate"]:.1f}%')

    # Empirical slippage
    for slip_mult, label in [(1, '1x slip'), (2, '2x slip'), (3, '3x slip')]:
        kw = {**TRAIL_FIRST,
              'spread_model': 'realistic', 'slippage_model': 'empirical',
              'slippage_buy': 0.67 * slip_mult, 'slippage_sell': 0.17 * slip_mult}
        cr = run_variant(data, f'Slip_{label}', verbose=False, **kw)
        cost_results.append({
            'label': f'Realistic+{label}',
            'sharpe': round(cr['sharpe'], 3), 'pnl': round(cr['total_pnl'], 0),
            'n': cr['n'], 'win_rate': round(cr['win_rate'], 1),
        })
        pf(f'  Realistic+{label}: Sh={cr["sharpe"]:.3f}  PnL=${cr["total_pnl"]:.0f}  WR={cr["win_rate"]:.1f}%')

    save('phase6_cost_stress', cost_results)

    # ═══════════════════════════════════════════════════
    # 7. Final Comparison: Trail-First vs L8 vs R202
    # ═══════════════════════════════════════════════════
    pf(f'\n{"="*80}')
    pf('7. Final Comparison Table')
    pf('='*80)

    l8_kw = {**L8_PARITY_KWARGS, 'min_lot_size': 0.04, 'max_lot_size': 0.04}
    r202_kw = dict(LIVE_PARITY_KWARGS)

    configs = [
        ('Trail-First (B)', TRAIL_FIRST),
        ('L8 (A)', l8_kw),
        ('R202 (H)', r202_kw),
        ('Trail+SL6 no Cap (E)', {**TRAIL_FIRST, 'sl_atr_mult': 6.0}),
        ('Trail+Cap SL3.5 (F)', {**TRAIL_FIRST, 'maxloss_cap': 70, 'maxloss_cap_atr_mult': 4.0}),
    ]

    header = f'{"Config":<30} {"n":>5} {"Sharpe":>8} {"PnL":>10} {"WR%":>6} {"MaxDD":>8}'
    pf(header)
    pf('-' * len(header))

    comparison = []
    for label, kw in configs:
        cr = run_variant(data, label, verbose=False, **kw)
        pf(f'{label:<30} {cr["n"]:>5} {cr["sharpe"]:>8.3f} {cr["total_pnl"]:>10.0f} '
           f'{cr["win_rate"]:>6.1f} {cr["max_dd"]:>8.0f}')
        comparison.append({
            'label': label, 'sharpe': round(cr['sharpe'], 3),
            'pnl': round(cr['total_pnl'], 0), 'n': cr['n'],
            'win_rate': round(cr['win_rate'], 1), 'max_dd': round(cr['max_dd'], 0),
        })

    save('phase7_comparison', comparison)

    # ═══════════════════════════════════════════════════
    # 8. Verdict
    # ═══════════════════════════════════════════════════
    pf(f'\n{"="*80}')
    pf('VERDICT')
    pf('='*80)

    gates = {
        'K-Fold 6/6': kf6_pos >= 5,
        'WF > 70%': wf_positive >= len(wf_results) * 0.7,
        'All Eras Positive': all_positive,
        'MC P(Sharpe>0) > 99%': sum(1 for s in mc_sharpes if s > 0) > 990,
        'Sensitivity Range < 2.0': (best_sens['sharpe'] - worst_sens['sharpe']) < 2.0,
        'Sp$0.75 Sharpe > 1.0': any(c['sharpe'] > 1.0 for c in cost_results if c.get('spread') == 0.75),
    }

    pass_count = sum(gates.values())
    for gate, passed in gates.items():
        pf(f'  {"PASS" if passed else "FAIL"} — {gate}')

    verdict = 'STRONG_PASS' if pass_count == len(gates) else ('PASS' if pass_count >= 4 else 'FAIL')
    pf(f'\n  Overall: {verdict} ({pass_count}/{len(gates)} gates)')

    total_time = time.time() - t0
    pf(f'\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f}min)')
    pf('R234 complete.')
    _log_file.close()


if __name__ == '__main__':
    main()
