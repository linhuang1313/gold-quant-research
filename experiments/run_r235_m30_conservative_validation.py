#!/usr/bin/env python3
"""R235: M30 Conservative (No-Trail) Validation
=================================================
R230C gave 11/11 STRONG_PASS for M30 strategies — but ALL used trail mode.
R221→R225 showed H4 strategies collapsed from Sharpe 4.5 to 0.08 without trail.

This experiment tests each M30 strategy under conservative conditions:
  - Trail = OFF (0.0/0.0) — the critical test
  - SL/TP from R230C best params (trail portion zeroed out)
  - Also test with reduced trail (0.08/0.02) as middle ground
  - K-Fold 6, Walk-Forward, Era stability, Slippage for each

Only strategies that maintain Sharpe > 1.0 without trail are real signals.
"""
import sys, json, time, os
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.m30_engine import M30BacktestEngine, load_m30_with_indicators
from backtest.engine import TradeRecord

OUTPUT_DIR = Path("results/r235_m30_conservative")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTPUT_DIR / "r235_stdout.txt"

_log_file = open(LOG_PATH, 'w', buffering=1)
def pf(msg=''):
    print(msg, flush=True)
    _log_file.write(msg + '\n')

def save(name, data):
    p = OUTPUT_DIR / f'{name}.json'
    with open(p, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    pf(f'  [saved {p}]')


# R230C best params for each strategy
R230C_BEST = {
    'm30_ema_fast':   {'sl': 2.5, 'tp': 3.0,  'trail_act': 0.15, 'trail_dist': 0.04, 'max_hold': 16},
    'm30_ema_cross':  {'sl': 8.0, 'tp': 10.0, 'trail_act': 0.15, 'trail_dist': 0.04, 'max_hold': 8},
    'm30_macd':       {'sl': 2.5, 'tp': 10.0, 'trail_act': 0.15, 'trail_dist': 0.04, 'max_hold': 16},
    'm30_rsi6':       {'sl': 6.0, 'tp': 6.0,  'trail_act': 0.15, 'trail_dist': 0.04, 'max_hold': 8},
    'm30_rsi14':      {'sl': 8.0, 'tp': 8.0,  'trail_act': 0.3,  'trail_dist': 0.08, 'max_hold': 48},
    'm30_cci':        {'sl': 4.0, 'tp': 6.0,  'trail_act': 0.15, 'trail_dist': 0.04, 'max_hold': 24},
    'm30_stoch':      {'sl': 2.5, 'tp': 6.0,  'trail_act': 0.15, 'trail_dist': 0.04, 'max_hold': 24},
    'm30_squeeze':    {'sl': 2.0, 'tp': 5.0,  'trail_act': 0.15, 'trail_dist': 0.04, 'max_hold': 12},
    'm30_mean_rev':   {'sl': 8.0, 'tp': 10.0, 'trail_act': 0.15, 'trail_dist': 0.04, 'max_hold': 8},
    'm30_inside_bar': {'sl': 6.0, 'tp': 12.0, 'trail_act': 0.15, 'trail_dist': 0.04, 'max_hold': 8},
    'm30_engulf':     {'sl': 8.0, 'tp': 8.0,  'trail_act': 0.15, 'trail_dist': 0.04, 'max_hold': 8},
}

# Import signal functions from R230C
from experiments.run_r230c_m30_parallel_sweep import (
    m30_sig_ema_fast_cross, m30_sig_ema_cross, m30_sig_macd_cross,
    m30_sig_rsi6_extreme, m30_sig_rsi14_trend, m30_sig_cci_momentum,
    m30_sig_stochastic, m30_sig_bb_squeeze,
    m30_sig_mean_revert, m30_sig_inside_bar, m30_sig_engulfing,
)

SIGNAL_MAP = {
    'm30_ema_fast': m30_sig_ema_fast_cross,
    'm30_ema_cross': m30_sig_ema_cross,
    'm30_macd': m30_sig_macd_cross,
    'm30_rsi6': m30_sig_rsi6_extreme,
    'm30_rsi14': m30_sig_rsi14_trend,
    'm30_cci': m30_sig_cci_momentum,
    'm30_stoch': m30_sig_stochastic,
    'm30_squeeze': m30_sig_bb_squeeze,
    'm30_mean_rev': m30_sig_mean_revert,
    'm30_inside_bar': m30_sig_inside_bar,
    'm30_engulf': m30_sig_engulfing,
}

SPREAD = 0.30

ERA_SEGMENTS = {
    "Pre-COVID (2015-2019)":      ("2015-01-01", "2020-01-01"),
    "COVID+Recovery (2020-2021)": ("2020-01-01", "2022-01-01"),
    "Tightening (2022-2023)":     ("2022-01-01", "2024-01-01"),
    "Recent (2024-2026)":         ("2024-01-01", "2026-06-01"),
}


def calc_stats(trades):
    if not trades:
        return {'n': 0, 'pnl': 0, 'sharpe': 0, 'win_rate': 0, 'max_dd': 0}
    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    cum = np.cumsum(pnls)
    dd = np.maximum.accumulate(cum) - cum
    sharpe = float(pnls.mean() / max(pnls.std(ddof=1), 1e-9) * np.sqrt(252)) if n > 1 else 0
    return {'n': n, 'pnl': round(float(pnls.sum()), 2), 'sharpe': round(sharpe, 3),
            'win_rate': round(100 * (pnls > 0).sum() / n, 2), 'max_dd': round(float(dd.max()), 2)}


def filter_period(trades, start, end):
    ts_s, ts_e = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    return [t for t in trades if ts_s <= pd.Timestamp(t.entry_time) < ts_e]


def kfold(trades, k=6):
    if len(trades) < k * 5:
        return {'verdict': 'SKIP', 'pass_count': 0, 'total': 0}
    pnls = np.array([t.pnl for t in trades])
    fold_size = len(pnls) // k
    kf_pass = 0
    sharpes = []
    for fold in range(k):
        s = fold * fold_size
        e = s + fold_size if fold < k - 1 else len(pnls)
        fp = pnls[s:e]
        if len(fp) < 3:
            continue
        sh = float(fp.mean() / max(fp.std(ddof=1), 1e-9) * np.sqrt(252))
        sharpes.append(round(sh, 3))
        if sh > 0:
            kf_pass += 1
    verdict = 'PASS' if kf_pass >= k * 0.67 else 'FAIL'
    return {'verdict': verdict, 'pass_count': kf_pass, 'total': len(sharpes),
            'sharpes': sharpes}


def run_one_config(m30_df, strategy, signal_fn, params, label, spread=SPREAD):
    """Run a single backtest config."""
    engine = M30BacktestEngine(
        m30_df, signal_funcs=[(strategy, signal_fn)],
        sl_atr_mult=params['sl'], tp_atr_mult=params['tp'],
        trailing_activate_atr=params.get('trail_act', 0.0),
        trailing_distance_atr=params.get('trail_dist', 0.0),
        max_hold=params['max_hold'],
        cooldown_bars=2, spread_cost=spread,
    )
    trades = engine.run()
    stats = calc_stats(trades)
    return trades, stats


def main():
    t0 = time.time()
    pf('=' * 80)
    pf('R235: M30 Conservative (No-Trail) Validation')
    pf('=' * 80)
    pf('Testing if M30 strategies survive without trail stop')
    pf('')

    m30_df = load_m30_with_indicators()
    pf(f'M30 data: {len(m30_df)} bars')

    all_results = {}

    for strategy in R230C_BEST:
        pf(f'\n{"="*80}')
        pf(f'{strategy}')
        pf('='*80)

        signal_fn = SIGNAL_MAP[strategy]
        best = R230C_BEST[strategy]

        # ── Config A: Original R230C best (with trail) ──
        trades_a, stats_a = run_one_config(m30_df, strategy, signal_fn, best, 'R230C best')
        pf(f'  A: R230C best (trail={best["trail_act"]}/{best["trail_dist"]}):')
        pf(f'     n={stats_a["n"]}  Sh={stats_a["sharpe"]:.3f}  PnL=${stats_a["pnl"]:.0f}  WR={stats_a["win_rate"]:.1f}%')

        # ── Config B: No trail (conservative) ──
        no_trail = {**best, 'trail_act': 0.0, 'trail_dist': 0.0}
        trades_b, stats_b = run_one_config(m30_df, strategy, signal_fn, no_trail, 'No trail')
        pf(f'  B: No trail:')
        pf(f'     n={stats_b["n"]}  Sh={stats_b["sharpe"]:.3f}  PnL=${stats_b["pnl"]:.0f}  WR={stats_b["win_rate"]:.1f}%')

        # ── Config C: Minimal trail (0.08/0.02) ──
        min_trail = {**best, 'trail_act': 0.08, 'trail_dist': 0.02}
        trades_c, stats_c = run_one_config(m30_df, strategy, signal_fn, min_trail, 'Min trail')
        pf(f'  C: Minimal trail (0.08/0.02):')
        pf(f'     n={stats_c["n"]}  Sh={stats_c["sharpe"]:.3f}  PnL=${stats_c["pnl"]:.0f}  WR={stats_c["win_rate"]:.1f}%')

        # ── Config D: No trail, tighter SL ──
        tight_sl = {**best, 'trail_act': 0.0, 'trail_dist': 0.0, 'sl': min(best['sl'], 3.0)}
        trades_d, stats_d = run_one_config(m30_df, strategy, signal_fn, tight_sl, 'No trail + tight SL')
        pf(f'  D: No trail + tight SL={tight_sl["sl"]}:')
        pf(f'     n={stats_d["n"]}  Sh={stats_d["sharpe"]:.3f}  PnL=${stats_d["pnl"]:.0f}  WR={stats_d["win_rate"]:.1f}%')

        # ── Degradation analysis ──
        if stats_a['sharpe'] > 0:
            deg_b = round(100 * (1 - stats_b['sharpe'] / stats_a['sharpe']), 1)
            deg_c = round(100 * (1 - stats_c['sharpe'] / stats_a['sharpe']), 1)
        else:
            deg_b = deg_c = 0

        pf(f'\n  Degradation:')
        pf(f'    No trail: {deg_b}% (A→B)')
        pf(f'    Min trail: {deg_c}% (A→C)')

        # ── Deep validation on best conservative config ──
        # Pick the best between B and D (no trail variants)
        best_conservative = stats_b if stats_b['sharpe'] >= stats_d['sharpe'] else stats_d
        best_cons_trades = trades_b if stats_b['sharpe'] >= stats_d['sharpe'] else trades_d
        best_cons_label = 'B (no trail)' if stats_b['sharpe'] >= stats_d['sharpe'] else 'D (no trail + tight SL)'

        pf(f'\n  Deep validation on {best_cons_label}:')

        # K-Fold
        kf = kfold(best_cons_trades)
        pf(f'    K-Fold 6: {kf["verdict"]} ({kf["pass_count"]}/{kf["total"]}) {kf.get("sharpes", [])}')

        # Era stability
        era_results = {}
        era_all_positive = True
        for era_name, (es, ee) in ERA_SEGMENTS.items():
            era_trades = filter_period(best_cons_trades, es, ee)
            era_stats = calc_stats(era_trades)
            era_results[era_name] = era_stats
            if era_stats['sharpe'] <= 0:
                era_all_positive = False
            pf(f'    Era [{era_name}]: n={era_stats["n"]} Sh={era_stats["sharpe"]:.3f}')

        era_verdict = 'PASS' if era_all_positive else 'FAIL'
        pf(f'    Era verdict: {era_verdict}')

        # Slippage test (realistic spread $0.75)
        slip_trades, slip_stats = run_one_config(
            m30_df, strategy, signal_fn, no_trail, 'Slippage', spread=0.75)
        slip_deg = round(100 * (1 - slip_stats['sharpe'] / max(best_conservative['sharpe'], 0.01)), 1) if best_conservative['sharpe'] > 0 else 0
        pf(f'    Slippage (Sp$0.75): Sh={slip_stats["sharpe"]:.3f} (deg={slip_deg}%)')

        # ── Verdict ──
        conservative_sharpe = best_conservative['sharpe']
        is_real = conservative_sharpe > 1.0
        kf_pass = kf['verdict'] == 'PASS'
        era_pass = era_verdict == 'PASS'
        slip_pass = slip_stats['sharpe'] > 0.5

        if is_real and kf_pass and era_pass and slip_pass:
            verdict = 'REAL_SIGNAL'
        elif is_real and kf_pass:
            verdict = 'LIKELY_REAL'
        elif conservative_sharpe > 0.5:
            verdict = 'WEAK_SIGNAL'
        else:
            verdict = 'TRAIL_DEPENDENT'

        pf(f'\n  >>> {strategy}: {verdict}')
        pf(f'      Trail Sharpe={stats_a["sharpe"]:.3f} → No-trail Sharpe={stats_b["sharpe"]:.3f} '
           f'({"SURVIVED" if is_real else "COLLAPSED"})')

        all_results[strategy] = {
            'r230c_sharpe': stats_a['sharpe'],
            'r230c_pnl': stats_a['pnl'],
            'no_trail_sharpe': stats_b['sharpe'],
            'no_trail_pnl': stats_b['pnl'],
            'min_trail_sharpe': stats_c['sharpe'],
            'tight_sl_sharpe': stats_d['sharpe'],
            'degradation_pct': deg_b,
            'kfold': kf['verdict'],
            'era': era_verdict,
            'slip_sharpe': slip_stats['sharpe'],
            'verdict': verdict,
            'best_conservative_params': no_trail if stats_b['sharpe'] >= stats_d['sharpe'] else tight_sl,
        }

    # ── Final Summary ──
    pf(f'\n{"="*80}')
    pf('FINAL SUMMARY')
    pf('='*80)

    header = f'{"Strategy":<18} {"Trail Sh":>9} {"NoTrail Sh":>11} {"Deg%":>6} {"KF":>5} {"Era":>5} {"Slip":>6} {"Verdict":<16}'
    pf(header)
    pf('-' * len(header))

    real_count = 0
    for strategy, r in sorted(all_results.items(), key=lambda x: -x[1]['no_trail_sharpe']):
        pf(f'{strategy:<18} {r["r230c_sharpe"]:>9.3f} {r["no_trail_sharpe"]:>11.3f} '
           f'{r["degradation_pct"]:>6.1f} {r["kfold"]:>5} {r["era"]:>5} '
           f'{r["slip_sharpe"]:>6.3f} {r["verdict"]:<16}')
        if r['verdict'] in ('REAL_SIGNAL', 'LIKELY_REAL'):
            real_count += 1

    pf(f'\nReal signals: {real_count}/{len(all_results)}')
    pf(f'Trail-dependent (fake): {sum(1 for r in all_results.values() if r["verdict"] == "TRAIL_DEPENDENT")}/{len(all_results)}')

    save('r235_results', all_results)

    total = time.time() - t0
    pf(f'\nTotal runtime: {total:.0f}s ({total/60:.1f}min)')
    pf('R235 complete.')
    _log_file.close()


if __name__ == '__main__':
    main()
