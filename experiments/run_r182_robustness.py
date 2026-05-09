#!/usr/bin/env python3
"""
R182 — Robustness Validation for R181 Proposed Changes
=======================================================
Before changing live config, validate each R181 recommendation with:
  Phase 1: 6-Fold Time-Series Cross-Validation
  Phase 2: Monte Carlo Bootstrap (1000 resamples)
  Phase 3: 6-Strategy Portfolio Interaction Test (current vs proposed)

Proposed changes under test:
  A. Keltner max_hold: 2 -> 5
  B. Chandelier ATR period for lines: 14 -> 22 (with RSI filter)
  C. Dual Thrust: confirmed-bar (current live) vs current-bar (research)

Pass criteria:
  - K-Fold: Proposed config wins >= 4/6 folds on Sharpe
  - Monte Carlo: P(proposed Sharpe > current Sharpe) >= 70%, delta 95% CI excludes 0
  - Portfolio: Combined proposed Sharpe > current Sharpe in >= 3/4 eras
"""
import sys, os, time, json
import numpy as np
import pandas as pd
from pathlib import Path
import glob as _glob

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r182_robustness")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30

FOLDS = [
    ("F1_2015_2017", "2015-01-01", "2017-01-01"),
    ("F2_2017_2019", "2017-01-01", "2019-01-01"),
    ("F3_2019_2021", "2019-01-01", "2021-01-01"),
    ("F4_2021_2023", "2021-01-01", "2023-01-01"),
    ("F5_2023_2025", "2023-01-01", "2025-01-01"),
    ("F6_2025_2026", "2025-01-01", "2026-06-01"),
]

ERA_SEGMENTS = {
    'full':      None,
    'hike':      [("2015-12-01", "2019-01-01"), ("2022-03-01", "2023-08-01")],
    'cut':       [("2019-07-01", "2022-03-01"), ("2024-09-01", "2026-06-01")],
    'recent_3y': [("2023-06-01", "2026-06-01")],
}

LIVE_CONFIG = {
    'L8_MAX':      {'lot': 0.02, 'cap_atr_mult': 4.0},
    'PSAR':        {'lot': 0.09, 'cap_atr_mult': 4.5},
    'TSMOM':       {'lot': 0.15, 'cap_atr_mult': 6.5},
    'SESS_BO':     {'lot': 0.13, 'cap_atr_mult': 5.0},
    'DUAL_THRUST': {'lot': 0.04, 'cap_atr_mult': 5.0},
    'CHANDELIER':  {'lot': 0.08, 'cap_atr_mult': 5.0},
}

MC_ITERATIONS = 1000
MC_SEED = 42


# ═══════════════════════════════════════════════════════════════
# Import backtest functions from R181 (same directory)
# ═══════════════════════════════════════════════════════════════
from run_r181_full_audit import (
    compute_atr, compute_adx, compute_rsi, add_psar,
    _mk, _run_exit, SESSION_ADX_MAP, _get_utc_hour,
    bt_l8_max, bt_psar, bt_tsmom, bt_sess_bo,
    bt_dual_thrust, bt_chandelier,
    filter_trades_by_era,
)


# ═══════════════════════════════════════════════════════════════
# Stats helpers
# ═══════════════════════════════════════════════════════════════

def _trades_to_daily(trades):
    if not trades: return pd.Series(dtype=float)
    df = pd.DataFrame(trades)
    df['date'] = pd.to_datetime(df['entry_time']).dt.date
    return df.groupby('date')['pnl'].sum()


def _sharpe(daily):
    if len(daily) < 10: return 0.0
    return float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0


def _max_dd(daily):
    if len(daily) == 0: return 0.0
    eq = daily.cumsum()
    return float((np.maximum.accumulate(eq) - eq).max())


def compute_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0.0, 'pnl': 0.0, 'wr': 0.0, 'max_dd': 0.0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    return {
        'n': n,
        'sharpe': round(_sharpe(daily), 3),
        'pnl': round(sum(pnls), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'max_dd': round(_max_dd(daily), 2),
    }


def run_era_stats(trades):
    result = {}
    for era in ['full', 'hike', 'cut', 'recent_3y']:
        era_trades = filter_trades_by_era(trades, era)
        result[era] = compute_stats(era_trades)
    return result


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_h1():
    candidates = sorted(_glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
    if not candidates:
        raise FileNotFoundError("No H1 data found in data/download/")
    csv_path = candidates[-1]
    print(f"  Loading H1: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                        'close': 'Close', 'volume': 'Volume'}, inplace=True)
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    print(f"  {len(df)} bars ({df.index[0]} ~ {df.index[-1]})", flush=True)
    return df


# ═══════════════════════════════════════════════════════════════
# Phase 1: K-Fold Cross-Validation
# ═══════════════════════════════════════════════════════════════

def run_kfold_test(h1_df, test_name, bt_func, current_kwargs, proposed_kwargs):
    """Run 6-fold time-series CV comparing current vs proposed config.
    Returns dict with per-fold results and verdict.
    """
    print(f"\n  K-Fold: {test_name}", flush=True)
    print(f"  {'Fold':<16} {'Cur_N':>6} {'Cur_Sh':>8} {'Pro_N':>6} {'Pro_Sh':>8} {'Delta':>7} {'Winner':>8}", flush=True)
    print(f"  {'-'*16} {'-'*6} {'-'*8} {'-'*6} {'-'*8} {'-'*7} {'-'*8}", flush=True)

    fold_results = []
    proposed_wins = 0

    for fold_name, start, end in FOLDS:
        fold_df = h1_df.loc[start:end].copy()
        if len(fold_df) < 200:
            print(f"  {fold_name:<16} SKIP (only {len(fold_df)} bars)", flush=True)
            continue

        cur_trades = bt_func(fold_df, spread=SPREAD, **current_kwargs)
        pro_trades = bt_func(fold_df, spread=SPREAD, **proposed_kwargs)

        cur_s = compute_stats(cur_trades)
        pro_s = compute_stats(pro_trades)
        delta = pro_s['sharpe'] - cur_s['sharpe']
        winner = "PROPOSED" if delta > 0 else ("CURRENT" if delta < 0 else "TIE")
        if delta > 0:
            proposed_wins += 1

        fold_results.append({
            'fold': fold_name, 'current': cur_s, 'proposed': pro_s,
            'delta': round(delta, 3), 'winner': winner,
        })

        print(f"  {fold_name:<16} {cur_s['n']:>6} {cur_s['sharpe']:>8.3f} "
              f"{pro_s['n']:>6} {pro_s['sharpe']:>8.3f} {delta:>+7.3f} {winner:>8}", flush=True)

    total_folds = len(fold_results)
    pass_threshold = max(1, int(total_folds * 2 / 3))
    passed = proposed_wins >= pass_threshold
    verdict = f"PASS ({proposed_wins}/{total_folds} folds)" if passed else f"FAIL ({proposed_wins}/{total_folds} folds)"

    print(f"\n  K-Fold Verdict: {verdict} (need >= {pass_threshold}/{total_folds})", flush=True)

    return {
        'test_name': test_name,
        'folds': fold_results,
        'proposed_wins': proposed_wins,
        'total_folds': total_folds,
        'passed': passed,
        'verdict': verdict,
    }


# ═══════════════════════════════════════════════════════════════
# Phase 2: Monte Carlo Bootstrap
# ═══════════════════════════════════════════════════════════════

def run_monte_carlo(h1_df, test_name, bt_func, current_kwargs, proposed_kwargs):
    """Bootstrap resample trade PnLs 1000x to get Sharpe confidence intervals."""
    print(f"\n  Monte Carlo ({MC_ITERATIONS}x): {test_name}", flush=True)

    cur_trades = bt_func(h1_df, spread=SPREAD, **current_kwargs)
    pro_trades = bt_func(h1_df, spread=SPREAD, **proposed_kwargs)

    cur_pnls = np.array([t['pnl'] for t in cur_trades])
    pro_pnls = np.array([t['pnl'] for t in pro_trades])

    if len(cur_pnls) < 20 or len(pro_pnls) < 20:
        print(f"  SKIP: insufficient trades (current={len(cur_pnls)}, proposed={len(pro_pnls)})", flush=True)
        return {'test_name': test_name, 'passed': False, 'verdict': 'SKIP (insufficient trades)'}

    rng = np.random.RandomState(MC_SEED)
    cur_sharpes = np.zeros(MC_ITERATIONS)
    pro_sharpes = np.zeros(MC_ITERATIONS)
    delta_sharpes = np.zeros(MC_ITERATIONS)

    for i in range(MC_ITERATIONS):
        cur_sample = rng.choice(cur_pnls, size=len(cur_pnls), replace=True)
        pro_sample = rng.choice(pro_pnls, size=len(pro_pnls), replace=True)

        cur_std = cur_sample.std()
        pro_std = pro_sample.std()
        cur_sharpes[i] = cur_sample.mean() / cur_std * np.sqrt(252) if cur_std > 0 else 0
        pro_sharpes[i] = pro_sample.mean() / pro_std * np.sqrt(252) if pro_std > 0 else 0
        delta_sharpes[i] = pro_sharpes[i] - cur_sharpes[i]

    prob_better = float(np.mean(delta_sharpes > 0))
    delta_mean = float(np.mean(delta_sharpes))
    delta_ci_low = float(np.percentile(delta_sharpes, 2.5))
    delta_ci_high = float(np.percentile(delta_sharpes, 97.5))
    ci_excludes_zero = delta_ci_low > 0 or delta_ci_high < 0

    passed = prob_better >= 0.70 and delta_ci_low > 0
    verdict = "PASS" if passed else "FAIL"

    if prob_better >= 0.70 and not ci_excludes_zero:
        verdict = "MARGINAL"

    print(f"  Current  Sharpe: {np.mean(cur_sharpes):.3f} [{np.percentile(cur_sharpes, 2.5):.3f}, {np.percentile(cur_sharpes, 97.5):.3f}]", flush=True)
    print(f"  Proposed Sharpe: {np.mean(pro_sharpes):.3f} [{np.percentile(pro_sharpes, 2.5):.3f}, {np.percentile(pro_sharpes, 97.5):.3f}]", flush=True)
    print(f"  Delta:           {delta_mean:+.3f} [{delta_ci_low:+.3f}, {delta_ci_high:+.3f}]", flush=True)
    print(f"  P(proposed > current): {prob_better:.1%}", flush=True)
    print(f"  95% CI excludes 0: {'YES' if ci_excludes_zero else 'NO'}", flush=True)
    print(f"  MC Verdict: {verdict}", flush=True)

    return {
        'test_name': test_name,
        'current_sharpe_mean': round(float(np.mean(cur_sharpes)), 3),
        'proposed_sharpe_mean': round(float(np.mean(pro_sharpes)), 3),
        'delta_mean': round(delta_mean, 3),
        'delta_ci_95': [round(delta_ci_low, 3), round(delta_ci_high, 3)],
        'prob_better': round(prob_better, 3),
        'ci_excludes_zero': ci_excludes_zero,
        'passed': passed,
        'verdict': verdict,
    }


# ═══════════════════════════════════════════════════════════════
# Phase 3: Portfolio Interaction Test
# ═══════════════════════════════════════════════════════════════

def run_portfolio_comparison(h1_df):
    """Compare current live portfolio vs proposed changes at portfolio level."""
    print(f"\n{'=' * 90}", flush=True)
    print(f"  Phase 3: 6-Strategy Portfolio Interaction Test", flush=True)
    print(f"{'=' * 90}", flush=True)

    lc = LIVE_CONFIG

    # Current live config (what's deployed now)
    current_portfolio = [
        ("L8_MAX", bt_l8_max, dict(lot=lc['L8_MAX']['lot'], cap_atr_mult=lc['L8_MAX']['cap_atr_mult'],
                                    max_hold=2, session_adx=True)),
        ("PSAR", bt_psar, dict(lot=lc['PSAR']['lot'], cap_atr_mult=lc['PSAR']['cap_atr_mult'],
                                sl_atr=4.0, tp_atr=6.0, trail_act=0.08, trail_dist=0.015, max_hold=15,
                                skip_hours={3, 7, 22})),
        ("TSMOM", bt_tsmom, dict(lot=lc['TSMOM']['lot'], cap_atr_mult=lc['TSMOM']['cap_atr_mult'],
                                  sl_atr=6.0, tp_atr=8.0, max_hold=12)),
        ("SESS_BO", bt_sess_bo, dict(lot=lc['SESS_BO']['lot'], cap_atr_mult=lc['SESS_BO']['cap_atr_mult'],
                                      d1_ema20_filter=True)),
        ("DUAL_THRUST", bt_dual_thrust, dict(lot=lc['DUAL_THRUST']['lot'], cap_atr_mult=lc['DUAL_THRUST']['cap_atr_mult'],
                                              confirmed_bar=True)),
        ("CHANDELIER", bt_chandelier, dict(lot=lc['CHANDELIER']['lot'], cap_atr_mult=lc['CHANDELIER']['cap_atr_mult'],
                                            filter_type='rsi3070', atr_period_for_lines=14)),
    ]

    # Proposed changes: KC MH5, CH ATR22, DT keep confirmed_bar (conservative)
    proposed_portfolio = [
        ("L8_MAX", bt_l8_max, dict(lot=lc['L8_MAX']['lot'], cap_atr_mult=lc['L8_MAX']['cap_atr_mult'],
                                    max_hold=5, session_adx=True)),
        ("PSAR", bt_psar, dict(lot=lc['PSAR']['lot'], cap_atr_mult=lc['PSAR']['cap_atr_mult'],
                                sl_atr=4.0, tp_atr=6.0, trail_act=0.08, trail_dist=0.015, max_hold=15,
                                skip_hours={3, 7, 22})),
        ("TSMOM", bt_tsmom, dict(lot=lc['TSMOM']['lot'], cap_atr_mult=lc['TSMOM']['cap_atr_mult'],
                                  sl_atr=6.0, tp_atr=8.0, max_hold=12)),
        ("SESS_BO", bt_sess_bo, dict(lot=lc['SESS_BO']['lot'], cap_atr_mult=lc['SESS_BO']['cap_atr_mult'],
                                      d1_ema20_filter=True)),
        ("DUAL_THRUST", bt_dual_thrust, dict(lot=lc['DUAL_THRUST']['lot'], cap_atr_mult=lc['DUAL_THRUST']['cap_atr_mult'],
                                              confirmed_bar=True)),
        ("CHANDELIER", bt_chandelier, dict(lot=lc['CHANDELIER']['lot'], cap_atr_mult=lc['CHANDELIER']['cap_atr_mult'],
                                            filter_type='rsi3070', atr_period_for_lines=22)),
    ]

    results = {}
    for label, portfolio in [("CURRENT", current_portfolio), ("PROPOSED", proposed_portfolio)]:
        print(f"\n  Portfolio {label}:", flush=True)
        all_trades = []
        for sname, bt_func, kwargs in portfolio:
            trades = bt_func(h1_df, spread=SPREAD, **kwargs)
            st = compute_stats(trades)
            print(f"    {sname:<14} N={st['n']:>5}  Sharpe={st['sharpe']:>7.3f}  "
                  f"PnL=${st['pnl']:>9,.0f}  WR={st['wr']:>5.1f}%  MaxDD=${st['max_dd']:>7,.0f}", flush=True)
            all_trades.extend(trades)
        all_trades.sort(key=lambda t: t['entry_time'])
        era_stats = run_era_stats(all_trades)
        results[label] = era_stats

        print(f"\n  {label} COMBINED:", flush=True)
        print(f"  {'Era':<12} {'N':>5} {'Sharpe':>8} {'PnL':>10} {'WR%':>6} {'MaxDD':>8}", flush=True)
        print(f"  {'-'*12} {'-'*5} {'-'*8} {'-'*10} {'-'*6} {'-'*8}", flush=True)
        for era in ['full', 'hike', 'cut', 'recent_3y']:
            s = era_stats[era]
            pnl_str = f"${s['pnl']:>9,.0f}" if s['pnl'] >= 0 else f"-${abs(s['pnl']):>8,.0f}"
            dd_str = f"${s['max_dd']:>7,.0f}"
            print(f"  {era:<12} {s['n']:>5} {s['sharpe']:>8.3f} {pnl_str} {s['wr']:>5.1f}% {dd_str}", flush=True)

    # Compare
    print(f"\n  Portfolio Delta (Proposed - Current):", flush=True)
    print(f"  {'Era':<12} {'Cur_Sh':>8} {'Pro_Sh':>8} {'Delta':>7} {'Winner':>10}", flush=True)
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*7} {'-'*10}", flush=True)
    era_wins = 0
    for era in ['full', 'hike', 'cut', 'recent_3y']:
        cur = results['CURRENT'][era]['sharpe']
        pro = results['PROPOSED'][era]['sharpe']
        d = pro - cur
        w = "PROPOSED" if d > 0 else ("CURRENT" if d < 0 else "TIE")
        if d > 0: era_wins += 1
        print(f"  {era:<12} {cur:>8.3f} {pro:>8.3f} {d:>+7.3f} {w:>10}", flush=True)

    passed = era_wins >= 3
    verdict = f"PASS ({era_wins}/4 eras)" if passed else f"FAIL ({era_wins}/4 eras)"
    print(f"\n  Portfolio Verdict: {verdict} (need >= 3/4 eras)", flush=True)

    return {
        'current': {k: v for k, v in results['CURRENT'].items()},
        'proposed': {k: v for k, v in results['PROPOSED'].items()},
        'era_wins': era_wins,
        'passed': passed,
        'verdict': verdict,
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 90, flush=True)
    print("  R182 -- Robustness Validation for R181 Proposed Changes", flush=True)
    print("=" * 90, flush=True)

    h1_df = load_h1()

    all_results = {}

    # ──────────────────────────────────────────────
    # Define the 3 proposed changes
    # ──────────────────────────────────────────────
    lc = LIVE_CONFIG

    proposals = [
        {
            'name': 'KC_MH2_vs_MH5',
            'desc': 'Keltner max_hold 2 -> 5',
            'bt_func': bt_l8_max,
            'current': dict(lot=lc['L8_MAX']['lot'], cap_atr_mult=lc['L8_MAX']['cap_atr_mult'], max_hold=2),
            'proposed': dict(lot=lc['L8_MAX']['lot'], cap_atr_mult=lc['L8_MAX']['cap_atr_mult'], max_hold=5),
        },
        {
            'name': 'CH_ATR14_vs_ATR22',
            'desc': 'Chandelier ATR period 14 -> 22 (RSI filter)',
            'bt_func': bt_chandelier,
            'current': dict(lot=lc['CHANDELIER']['lot'], cap_atr_mult=lc['CHANDELIER']['cap_atr_mult'],
                           filter_type='rsi3070', atr_period_for_lines=14),
            'proposed': dict(lot=lc['CHANDELIER']['lot'], cap_atr_mult=lc['CHANDELIER']['cap_atr_mult'],
                            filter_type='rsi3070', atr_period_for_lines=22),
        },
        {
            'name': 'DT_Confirmed_vs_Current',
            'desc': 'Dual Thrust confirmed-bar vs current-bar',
            'bt_func': bt_dual_thrust,
            'current': dict(lot=lc['DUAL_THRUST']['lot'], cap_atr_mult=lc['DUAL_THRUST']['cap_atr_mult'],
                           confirmed_bar=True),
            'proposed': dict(lot=lc['DUAL_THRUST']['lot'], cap_atr_mult=lc['DUAL_THRUST']['cap_atr_mult'],
                            confirmed_bar=False),
        },
    ]

    # ──────────────────────────────────────────────
    # Phase 1: K-Fold Cross-Validation
    # ──────────────────────────────────────────────
    print(f"\n{'=' * 90}", flush=True)
    print(f"  PHASE 1: 6-Fold Time-Series Cross-Validation", flush=True)
    print(f"{'=' * 90}", flush=True)

    kfold_results = {}
    for p in proposals:
        kfold_results[p['name']] = run_kfold_test(
            h1_df, p['name'], p['bt_func'], p['current'], p['proposed'])

    all_results['kfold'] = kfold_results

    # ──────────────────────────────────────────────
    # Phase 2: Monte Carlo Bootstrap
    # ──────────────────────────────────────────────
    print(f"\n{'=' * 90}", flush=True)
    print(f"  PHASE 2: Monte Carlo Bootstrap ({MC_ITERATIONS} iterations)", flush=True)
    print(f"{'=' * 90}", flush=True)

    mc_results = {}
    for p in proposals:
        mc_results[p['name']] = run_monte_carlo(
            h1_df, p['name'], p['bt_func'], p['current'], p['proposed'])

    all_results['monte_carlo'] = mc_results

    # ──────────────────────────────────────────────
    # Phase 3: Portfolio Interaction
    # ──────────────────────────────────────────────
    portfolio_result = run_portfolio_comparison(h1_df)
    all_results['portfolio'] = portfolio_result

    # ──────────────────────────────────────────────
    # Final Summary
    # ──────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n\n{'=' * 90}", flush=True)
    print(f"  FINAL SUMMARY -- R182 Robustness Validation", flush=True)
    print(f"{'=' * 90}", flush=True)

    print(f"\n  {'Proposal':<28} {'K-Fold':>16} {'Monte Carlo':>16} {'Overall':>10}", flush=True)
    print(f"  {'-'*28} {'-'*16} {'-'*16} {'-'*10}", flush=True)

    final_decisions = {}
    for p in proposals:
        kf = kfold_results[p['name']]
        mc = mc_results[p['name']]

        kf_str = kf['verdict']
        mc_str = mc['verdict']

        tests_passed = sum([kf['passed'], mc['passed']])
        if tests_passed == 2:
            overall = "GO"
        elif tests_passed == 1:
            overall = "CAUTION"
        else:
            overall = "NO-GO"

        final_decisions[p['name']] = {
            'description': p['desc'],
            'kfold': kf_str,
            'monte_carlo': mc_str,
            'overall': overall,
        }

        print(f"  {p['desc']:<28} {kf_str:>16} {mc_str:>16} {overall:>10}", flush=True)

    pf_str = portfolio_result['verdict']
    print(f"\n  Portfolio (combined changes): {pf_str}", flush=True)

    print(f"\n  Decision Key:", flush=True)
    print(f"    GO      = Both K-Fold and MC pass -> safe to deploy", flush=True)
    print(f"    CAUTION = 1 of 2 pass -> deploy with monitoring", flush=True)
    print(f"    NO-GO   = Neither pass -> do NOT deploy", flush=True)

    all_results['final_decisions'] = final_decisions
    all_results['portfolio_verdict'] = pf_str
    all_results['runtime_seconds'] = round(elapsed, 1)

    print(f"\n  Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    out_path = OUTPUT_DIR / "r182_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {out_path}", flush=True)
    print(f"{'=' * 90}", flush=True)


if __name__ == "__main__":
    main()
