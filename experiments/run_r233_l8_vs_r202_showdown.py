#!/usr/bin/env python3
"""R233: L8 vs R202 Parameter Showdown
========================================
Systematic comparison of two parameter regimes:
  - L8  (2026-04-28): trail=0.14/0.025, SL=3.5, no Cap, ATR lots, regime trail
  - R202 (2026-05-14): trail=0.06/0.015, SL=6.0, Cap=$70, 0.04 lots, unified trail

Both tested under identical conditions:
  1. Full-sample backtest + stats
  2. 6-Fold cross validation (stability)
  3. Era analysis (regime robustness)
  4. Recent vs historical performance split
  5. Exit reason breakdown (how trades end)
  6. Drawdown profile comparison
  7. Monthly PnL consistency
  8. Realistic cost version (slippage + spread)

Also tests hybrid configs to isolate which individual change helps/hurts most.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.runner import (
    DataBundle, run_variant, run_kfold,
    L8_PARITY_KWARGS, LIVE_PARITY_KWARGS, REALISTIC_COST_KWARGS,
)
from backtest.engine import BacktestEngine
from backtest.stats import calc_stats, aggregate_daily_pnl
from collections import Counter
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'results', 'r233_l8_vs_r202')
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, 'r233_stdout.txt')

_log_file = open(LOG_PATH, 'w', buffering=1)
def pf(msg=''):
    print(msg, flush=True)
    _log_file.write(msg + '\n')

def save(name, obj):
    path = os.path.join(RESULTS_DIR, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    pf(f'  [saved {path}]')


def run_full_eval(data, label, kwargs):
    """Run comprehensive evaluation for one parameter set."""
    pf(f'\n--- {label} ---')
    t0 = time.time()

    engine = BacktestEngine(data.m15_df, data.h1_df, **kwargs)
    trades = engine.run()
    stats = calc_stats(trades, engine.equity_curve)

    # K-Fold
    kfold_results = run_kfold(data, kwargs, n_folds=6, label_prefix=label)
    kfold_sharpes = [r['sharpe'] for r in kfold_results]
    kfold_positive = sum(1 for s in kfold_sharpes if s > 0)
    kfold_verdict = 'PASS' if kfold_positive >= 5 else ('WEAK' if kfold_positive >= 4 else 'FAIL')

    # Exit reasons
    exit_reasons = Counter(t.exit_reason for t in trades)
    exit_pnl = {}
    for reason in exit_reasons:
        rtrades = [t for t in trades if t.exit_reason == reason]
        exit_pnl[reason] = {
            'count': len(rtrades),
            'total_pnl': round(sum(t.pnl for t in rtrades), 2),
            'avg_pnl': round(sum(t.pnl for t in rtrades) / len(rtrades), 3) if rtrades else 0,
        }

    # Era analysis: split into 3 eras
    if trades:
        all_times = sorted(t.entry_time for t in trades)
        n = len(all_times)
        era_cuts = [all_times[0], all_times[n//3], all_times[2*n//3], all_times[-1]]
        era_stats = []
        for i in range(3):
            era_trades = [t for t in trades if era_cuts[i] <= t.entry_time < era_cuts[i+1]]
            if era_trades:
                era_pnl = sum(t.pnl for t in era_trades)
                era_wr = 100 * sum(1 for t in era_trades if t.pnl > 0) / len(era_trades)
                era_stats.append({
                    'era': f'E{i+1}',
                    'start': str(era_cuts[i])[:10],
                    'end': str(era_cuts[i+1])[:10],
                    'n': len(era_trades),
                    'pnl': round(era_pnl, 2),
                    'win_rate': round(era_wr, 1),
                })
    else:
        era_stats = []

    # Recent vs historical (last 6 months vs rest)
    if trades:
        cutoff = pd.Timestamp(all_times[-1]) - pd.Timedelta(days=180)
        recent = [t for t in trades if pd.Timestamp(t.entry_time) >= cutoff]
        historical = [t for t in trades if pd.Timestamp(t.entry_time) < cutoff]
        recent_pnl = sum(t.pnl for t in recent)
        hist_pnl = sum(t.pnl for t in historical)
        recent_wr = 100 * sum(1 for t in recent if t.pnl > 0) / len(recent) if recent else 0
        hist_wr = 100 * sum(1 for t in historical if t.pnl > 0) / len(historical) if historical else 0
    else:
        recent_pnl = hist_pnl = recent_wr = hist_wr = 0
        recent = historical = []

    # Monthly PnL
    if trades:
        monthly = {}
        for t in trades:
            key = str(t.entry_time)[:7]
            monthly[key] = monthly.get(key, 0) + t.pnl
        months_positive = sum(1 for v in monthly.values() if v > 0)
        months_total = len(monthly)
        monthly_consistency = round(100 * months_positive / months_total, 1) if months_total else 0
    else:
        monthly = {}
        monthly_consistency = 0

    elapsed = time.time() - t0

    result = {
        'label': label,
        'stats': stats,
        'kfold': {
            'sharpes': [round(s, 3) for s in kfold_sharpes],
            'positive': kfold_positive,
            'verdict': kfold_verdict,
        },
        'exit_breakdown': exit_pnl,
        'era_stats': era_stats,
        'recent_6m': {
            'n': len(recent), 'pnl': round(recent_pnl, 2), 'wr': round(recent_wr, 1),
        },
        'historical': {
            'n': len(historical), 'pnl': round(hist_pnl, 2), 'wr': round(hist_wr, 1),
        },
        'monthly_consistency': monthly_consistency,
        'elapsed_s': round(elapsed, 1),
    }

    pf(f'  n={stats["n"]:>5}  Sharpe={stats["sharpe"]:.3f}  PnL=${stats["total_pnl"]:.0f}  '
       f'WR={stats["win_rate"]:.1f}%  MaxDD=${stats["max_dd"]:.0f}')
    pf(f'  KFold: {kfold_sharpes} → {kfold_verdict} ({kfold_positive}/6)')
    pf(f'  Monthly consistency: {monthly_consistency}%')
    pf(f'  Recent 6m: n={len(recent)} PnL=${recent_pnl:.0f} WR={recent_wr:.1f}%')
    for reason, info in sorted(exit_pnl.items(), key=lambda x: -abs(x[1]['total_pnl'])):
        pf(f'  Exit [{reason}]: {info["count"]} trades, PnL=${info["total_pnl"]:.0f}, '
           f'avg=${info["avg_pnl"]:.2f}')
    pf(f'  [{elapsed:.1f}s]')

    return result


def main():
    t0 = time.time()
    pf('=' * 80)
    pf('R233: L8 vs R202 Parameter Showdown')
    pf('=' * 80)

    data = DataBundle.load_default()

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Head-to-head comparison (both at fixed 0.04 lots for fair comparison)
    # ═══════════════════════════════════════════════════════════
    pf(f'\n{"="*80}')
    pf('PHASE 1: Head-to-Head (fixed 0.04 lots)')
    pf('='*80)

    l8_fixed = dict(L8_PARITY_KWARGS)
    l8_fixed['min_lot_size'] = 0.04
    l8_fixed['max_lot_size'] = 0.04

    r202 = dict(LIVE_PARITY_KWARGS)

    l8_result = run_full_eval(data, 'L8 (fixed 0.04)', l8_fixed)
    r202_result = run_full_eval(data, 'R202 (live-actual)', r202)

    save('phase1_head_to_head', {
        'L8': l8_result,
        'R202': r202_result,
    })

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Ablation — isolate each change
    # ═══════════════════════════════════════════════════════════
    pf(f'\n{"="*80}')
    pf('PHASE 2: Ablation — which change helps most?')
    pf('='*80)
    pf('Starting from L8 baseline (fixed 0.04), apply one R202 change at a time:')

    base = dict(L8_PARITY_KWARGS)
    base['min_lot_size'] = 0.04
    base['max_lot_size'] = 0.04

    ablations = {
        'A: L8 baseline (control)': {},

        'B: +R202 trail only': {
            'trailing_activate_atr': 0.06,
            'trailing_distance_atr': 0.015,
            'regime_config': {
                'low': {'trail_act': 0.06, 'trail_dist': 0.015},
                'normal': {'trail_act': 0.06, 'trail_dist': 0.015},
                'high': {'trail_act': 0.06, 'trail_dist': 0.015},
            },
        },

        'C: +SL 6.0 only': {
            'sl_atr_mult': 6.0,
        },

        'D: +Cap $70 only': {
            'maxloss_cap': 70,
            'maxloss_cap_atr_mult': 4.0,
        },

        'E: +Trail + SL 6.0 (no Cap)': {
            'trailing_activate_atr': 0.06,
            'trailing_distance_atr': 0.015,
            'regime_config': {
                'low': {'trail_act': 0.06, 'trail_dist': 0.015},
                'normal': {'trail_act': 0.06, 'trail_dist': 0.015},
                'high': {'trail_act': 0.06, 'trail_dist': 0.015},
            },
            'sl_atr_mult': 6.0,
        },

        'F: +Trail + Cap (SL=3.5)': {
            'trailing_activate_atr': 0.06,
            'trailing_distance_atr': 0.015,
            'regime_config': {
                'low': {'trail_act': 0.06, 'trail_dist': 0.015},
                'normal': {'trail_act': 0.06, 'trail_dist': 0.015},
                'high': {'trail_act': 0.06, 'trail_dist': 0.015},
            },
            'maxloss_cap': 70,
            'maxloss_cap_atr_mult': 4.0,
        },

        'G: +SL 6.0 + Cap (L8 trail)': {
            'sl_atr_mult': 6.0,
            'maxloss_cap': 70,
            'maxloss_cap_atr_mult': 4.0,
        },

        'H: All R202 changes (= R202)': {
            'trailing_activate_atr': 0.06,
            'trailing_distance_atr': 0.015,
            'regime_config': {
                'low': {'trail_act': 0.06, 'trail_dist': 0.015},
                'normal': {'trail_act': 0.06, 'trail_dist': 0.015},
                'high': {'trail_act': 0.06, 'trail_dist': 0.015},
            },
            'sl_atr_mult': 6.0,
            'maxloss_cap': 70,
            'maxloss_cap_atr_mult': 4.0,
        },
    }

    ablation_results = {}
    for label, overrides in ablations.items():
        kw = {**base, **overrides}
        r = run_full_eval(data, label, kw)
        ablation_results[label] = r

    save('phase2_ablation', ablation_results)

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: Realistic costs comparison
    # ═══════════════════════════════════════════════════════════
    pf(f'\n{"="*80}')
    pf('PHASE 3: Realistic Costs (slippage + spread)')
    pf('='*80)

    l8_cost = dict(L8_PARITY_KWARGS)
    l8_cost['min_lot_size'] = 0.04
    l8_cost['max_lot_size'] = 0.04
    l8_cost['spread_model'] = 'realistic'
    l8_cost['slippage_model'] = 'empirical'
    l8_cost['slippage_buy'] = 0.67
    l8_cost['slippage_sell'] = 0.17

    r202_cost = dict(REALISTIC_COST_KWARGS)

    l8c = run_full_eval(data, 'L8 + realistic costs', l8_cost)
    r202c = run_full_eval(data, 'R202 + realistic costs', r202_cost)

    save('phase3_realistic_costs', {
        'L8_cost': l8c,
        'R202_cost': r202c,
    })

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: Summary comparison table
    # ═══════════════════════════════════════════════════════════
    pf(f'\n{"="*80}')
    pf('FINAL COMPARISON TABLE')
    pf('='*80)

    all_results = [
        ('L8 (no cost)', l8_result),
        ('R202 (no cost)', r202_result),
        ('L8 (w/ cost)', l8c),
        ('R202 (w/ cost)', r202c),
    ]

    header = f'{"Config":<25} {"n":>5} {"Sharpe":>8} {"PnL":>10} {"WR%":>6} {"MaxDD":>8} {"KFold":>6} {"Mon%":>5} {"Recent$":>9}'
    pf(header)
    pf('-' * len(header))
    for label, r in all_results:
        s = r['stats']
        pf(f'{label:<25} {s["n"]:>5} {s["sharpe"]:>8.3f} {s["total_pnl"]:>10.0f} '
           f'{s["win_rate"]:>6.1f} {s["max_dd"]:>8.0f} {r["kfold"]["verdict"]:>6} '
           f'{r["monthly_consistency"]:>5.1f} {r["recent_6m"]["pnl"]:>9.0f}')

    pf(f'\n{"="*80}')
    pf('ABLATION SUMMARY')
    pf('='*80)
    header2 = f'{"Config":<45} {"Sharpe":>8} {"PnL":>10} {"KFold":>6} {"CapHits":>7}'
    pf(header2)
    pf('-' * len(header2))
    for label, r in ablation_results.items():
        s = r['stats']
        cap_hits = r['exit_breakdown'].get('MaxLossCap', {}).get('count', 0)
        pf(f'{label:<45} {s["sharpe"]:>8.3f} {s["total_pnl"]:>10.0f} {r["kfold"]["verdict"]:>6} {cap_hits:>7}')

    total_time = time.time() - t0
    pf(f'\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f}min)')
    pf('R233 complete.')
    _log_file.close()


if __name__ == '__main__':
    main()
