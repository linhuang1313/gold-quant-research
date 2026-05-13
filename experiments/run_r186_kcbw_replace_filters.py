#!/usr/bin/env python3
"""
R186 — KCBW Replace-Not-Stack: Engine Filter Substitution Test
================================================================
R185 found KCBW works in standalone but HURTS in full Engine (4.30→4.02).
Hypothesis: Engine's existing filters (choppy gate, RSI-ADX filter) overlap
with KCBW, so stacking them is redundant. Test replacing one with KCBW.

Configs:
  A. Baseline:          choppy=0.50, rsi_adx=40, KCBW=OFF
  B. +KCBW stacked:     choppy=0.50, rsi_adx=40, KCBW=5   (R185 Phase 3)
  C. KCBW replaces choppy:  choppy=OFF, rsi_adx=40, KCBW=5
  D. KCBW replaces RSI-ADX: choppy=0.50, rsi_adx=OFF, KCBW=5
  E. KCBW replaces both:    choppy=OFF, rsi_adx=OFF, KCBW=5
  F. No filters at all:     choppy=OFF, rsi_adx=OFF, KCBW=OFF
  G-J. Vary KCBW lookback (3,8) for best combos

Phase 1: Full-period comparison (all configs)
Phase 2: K-Fold 6 folds on top candidates
Phase 3: Era breakdown (hike / cut / recent_3y)
Phase 4: Walk-Forward OOS (5 windows)
Phase 5: Cost sensitivity ($0.30 — $1.00)
Phase 6: Yearly stability
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS

OUTPUT_DIR = Path("results/r186_kcbw_replace_filters")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
LOT = 0.02

ERA_SEGMENTS = {
    'full':      None,
    'hike':      [("2015-12-01", "2019-01-01"), ("2022-03-01", "2023-08-01")],
    'cut':       [("2019-07-01", "2022-03-01"), ("2024-09-01", "2026-06-01")],
    'recent_3y': [("2023-06-01", "2026-06-01")],
}

FOLDS = [
    ("F1_2015_2017", "2015-01-01", "2017-01-01"),
    ("F2_2017_2019", "2017-01-01", "2019-01-01"),
    ("F3_2019_2021", "2019-01-01", "2021-01-01"),
    ("F4_2021_2023", "2021-01-01", "2023-01-01"),
    ("F5_2023_2025", "2023-01-01", "2025-01-01"),
    ("F6_2025_2026", "2025-01-01", "2026-06-01"),
]

WF_WINDOWS = [
    ("WF1", "2015-01-01", "2019-01-01", "2019-01-01", "2021-01-01"),
    ("WF2", "2016-01-01", "2020-01-01", "2020-01-01", "2022-01-01"),
    ("WF3", "2017-01-01", "2021-01-01", "2021-01-01", "2023-01-01"),
    ("WF4", "2018-01-01", "2022-01-01", "2022-01-01", "2024-01-01"),
    ("WF5", "2019-01-01", "2023-01-01", "2023-01-01", "2025-01-01"),
]

t0 = time.time()


# ═══════════════════════════════════════════════════════════════
# Config definitions
# ═══════════════════════════════════════════════════════════════

BASE_KW = {
    **LIVE_PARITY_KWARGS,
    'spread_cost': SPREAD,
    'min_lot_size': LOT,
    'max_lot_size': LOT,
    'initial_capital': 5000,
    'maxloss_cap': 0,
}

CONFIGS = {
    'A_baseline': {
        **BASE_KW,
        'intraday_adaptive': True,
        'choppy_threshold': 0.50,
        'rsi_adx_filter': 40,
        'kc_bw_filter_bars': 0,
    },
    'B_kcbw_stacked': {
        **BASE_KW,
        'intraday_adaptive': True,
        'choppy_threshold': 0.50,
        'rsi_adx_filter': 40,
        'kc_bw_filter_bars': 5,
    },
    'C_kcbw_replace_choppy': {
        **BASE_KW,
        'intraday_adaptive': False,
        'choppy_threshold': 0.50,
        'rsi_adx_filter': 40,
        'kc_bw_filter_bars': 5,
    },
    'D_kcbw_replace_rsi_adx': {
        **BASE_KW,
        'intraday_adaptive': True,
        'choppy_threshold': 0.50,
        'rsi_adx_filter': 0,
        'kc_bw_filter_bars': 5,
    },
    'E_kcbw_replace_both': {
        **BASE_KW,
        'intraday_adaptive': False,
        'choppy_threshold': 0.50,
        'rsi_adx_filter': 0,
        'kc_bw_filter_bars': 5,
    },
    'F_no_filters': {
        **BASE_KW,
        'intraday_adaptive': False,
        'choppy_threshold': 0.50,
        'rsi_adx_filter': 0,
        'kc_bw_filter_bars': 0,
    },
}

KCBW_VARIANTS = {
    'C3_kcbw3_replace_choppy': {
        **BASE_KW, 'intraday_adaptive': False,
        'rsi_adx_filter': 40, 'kc_bw_filter_bars': 3,
    },
    'C8_kcbw8_replace_choppy': {
        **BASE_KW, 'intraday_adaptive': False,
        'rsi_adx_filter': 40, 'kc_bw_filter_bars': 8,
    },
    'D3_kcbw3_replace_rsi': {
        **BASE_KW, 'intraday_adaptive': True, 'choppy_threshold': 0.50,
        'rsi_adx_filter': 0, 'kc_bw_filter_bars': 3,
    },
    'D8_kcbw8_replace_rsi': {
        **BASE_KW, 'intraday_adaptive': True, 'choppy_threshold': 0.50,
        'rsi_adx_filter': 0, 'kc_bw_filter_bars': 8,
    },
    'E3_kcbw3_replace_both': {
        **BASE_KW, 'intraday_adaptive': False,
        'rsi_adx_filter': 0, 'kc_bw_filter_bars': 3,
    },
    'E8_kcbw8_replace_both': {
        **BASE_KW, 'intraday_adaptive': False,
        'rsi_adx_filter': 0, 'kc_bw_filter_bars': 8,
    },
}


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def extract_stats(result):
    trades = result.get('_trades', [])
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    r_mult = avg_win / avg_loss if avg_loss > 0 else 0
    breakeven = 1.0 / (1.0 + r_mult) if r_mult > 0 else 1.0
    wr = result.get('win_rate', 0)
    safety_margin = wr - breakeven * 100
    return {
        'n': result['n'],
        'sharpe': round(result['sharpe'], 3),
        'pnl': round(result['total_pnl'], 0),
        'wr': round(wr, 1),
        'max_dd': round(result.get('max_dd', 0), 0),
        'r_mult': round(r_mult, 3),
        'safety_margin': round(safety_margin, 1),
        'skipped_choppy': result.get('skipped_choppy', 0),
        'skipped_kc_bw': result.get('skipped_kc_bw', 0),
        'rsi_filtered': result.get('rsi_filtered', 0),
    }


def fmt_row(label, s):
    pnl_s = f"${s['pnl']:>9,.0f}" if s['pnl'] >= 0 else f"-${abs(s['pnl']):>8,.0f}"
    return (f"  {label:<32} {s['n']:>6} {s['sharpe']:>7.3f} {pnl_s} {s['wr']:>5.1f}% "
            f"{s['r_mult']:>6.3f} {s['safety_margin']:>6.1f}pp "
            f"chp={s['skipped_choppy']:>5} bw={s['skipped_kc_bw']:>5} rsi={s['rsi_filtered']:>5}")


HEADER = (f"  {'Config':<32} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} "
          f"{'R_mult':>6} {'Margin':>8} {'Choppy':>9} {'KCBW':>8} {'RSI':>8}")


def run_config(data, label, kw, verbose=True):
    result = run_variant(data, label, verbose=verbose, **kw)
    return result


# ═══════════════════════════════════════════════════════════════
# Phase 1: Full-period comparison
# ═══════════════════════════════════════════════════════════════

def phase1_full_comparison(data):
    print(f"\n{'='*130}")
    print(f"  PHASE 1: Full-Period Filter Substitution Comparison")
    print(f"{'='*130}")
    print(HEADER, flush=True)

    results = {}

    all_configs = {**CONFIGS, **KCBW_VARIANTS}
    for name, kw in all_configs.items():
        r = run_config(data, name, kw)
        s = extract_stats(r)
        print(fmt_row(name, s), flush=True)
        results[name] = {'stats': s, '_result': r}

    baseline_sh = results['A_baseline']['stats']['sharpe']
    print(f"\n  Baseline Sharpe: {baseline_sh:.3f}")
    print(f"\n  Delta vs baseline:")
    for name, data_r in sorted(results.items(), key=lambda x: x[1]['stats']['sharpe'], reverse=True):
        s = data_r['stats']
        delta = s['sharpe'] - baseline_sh
        marker = " <-- BEST" if s['sharpe'] == max(r['stats']['sharpe'] for r in results.values()) else ""
        print(f"    {name:<32} Sharpe={s['sharpe']:.3f} ({delta:+.3f}){marker}")

    return results


# ═══════════════════════════════════════════════════════════════
# Phase 2: K-Fold on top candidates
# ═══════════════════════════════════════════════════════════════

def phase2_kfold(data, p1_results):
    print(f"\n{'='*130}")
    print(f"  PHASE 2: K-Fold 6-Fold Validation (top candidates vs baseline)")
    print(f"{'='*130}")

    baseline_name = 'A_baseline'
    baseline_kw = CONFIGS[baseline_name]

    all_configs = {**CONFIGS, **KCBW_VARIANTS}
    candidates = [(k, v) for k, v in p1_results.items()
                  if k != baseline_name and v['stats']['sharpe'] > p1_results[baseline_name]['stats']['sharpe']]
    candidates.sort(key=lambda x: x[1]['stats']['sharpe'], reverse=True)
    top_candidates = candidates[:6]

    if not top_candidates:
        print("  No candidate beats baseline. Selecting top 3 by Sharpe anyway.")
        candidates = [(k, v) for k, v in p1_results.items() if k != baseline_name]
        candidates.sort(key=lambda x: x[1]['stats']['sharpe'], reverse=True)
        top_candidates = candidates[:3]

    kf_results = {}

    for cand_name, _ in top_candidates:
        cand_kw = all_configs.get(cand_name)
        if cand_kw is None:
            continue

        print(f"\n  --- {cand_name} vs {baseline_name} ---")
        print(f"  {'Fold':<16} {'Base_Sh':>8} {'Cand_Sh':>8} {'Delta':>7} {'Win':>4}")

        wins = 0
        fold_details = []
        for fold_name, start, end in FOLDS:
            fold_data = data.slice(start, end)
            if len(fold_data.m15_df) < 1000:
                continue

            br = run_variant(fold_data, f"{baseline_name}_{fold_name}", verbose=False, **baseline_kw)
            cr = run_variant(fold_data, f"{cand_name}_{fold_name}", verbose=False, **cand_kw)
            delta = cr['sharpe'] - br['sharpe']
            w = "+" if delta > 0 else "-"
            if delta > 0: wins += 1
            print(f"  {fold_name:<16} {br['sharpe']:>8.2f} {cr['sharpe']:>8.2f} {delta:>+7.3f} {w:>4}")
            fold_details.append({'fold': fold_name, 'base': br['sharpe'], 'cand': cr['sharpe'], 'delta': delta})

        total = len(fold_details)
        passed = wins >= max(1, int(total * 2 / 3))
        print(f"  K-Fold: {wins}/{total} {'PASS' if passed else 'FAIL'}")
        kf_results[cand_name] = {'wins': wins, 'total': total, 'pass': passed, 'folds': fold_details}

    return kf_results


# ═══════════════════════════════════════════════════════════════
# Phase 3: Era breakdown
# ═══════════════════════════════════════════════════════════════

def phase3_era(data, p1_results):
    print(f"\n{'='*130}")
    print(f"  PHASE 3: Era Breakdown (hike / cut / recent_3y)")
    print(f"{'='*130}")

    all_configs = {**CONFIGS, **KCBW_VARIANTS}
    test_configs = ['A_baseline', 'C_kcbw_replace_choppy', 'D_kcbw_replace_rsi_adx', 'E_kcbw_replace_both']
    test_configs = [c for c in test_configs if c in p1_results]

    era_results = {}

    for era_name, periods in ERA_SEGMENTS.items():
        if periods is None:
            continue
        print(f"\n  --- Era: {era_name} ---")
        print(f"  {'Config':<32} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'WR%':>6}")

        for cfg_name in test_configs:
            result = p1_results[cfg_name]['_result']
            trades = result.get('_trades', [])
            era_trades = []
            for t in trades:
                entry = pd.Timestamp(t.entry_time)
                if entry.tzinfo is not None:
                    entry = entry.tz_localize(None)
                for ps, pe in periods:
                    if pd.Timestamp(ps) <= entry < pd.Timestamp(pe):
                        era_trades.append(t); break

            pnls = [t.pnl for t in era_trades]
            n = len(era_trades)
            if n == 0:
                print(f"  {cfg_name:<32} {'—':>6}")
                continue
            wins = [p for p in pnls if p > 0]
            wr = len(wins) / n * 100
            daily = {}
            for t in era_trades:
                d = pd.Timestamp(t.exit_time).date()
                daily[d] = daily.get(d, 0) + t.pnl
            daily_arr = list(daily.values())
            sharpe = (np.mean(daily_arr) / np.std(daily_arr) * np.sqrt(252)
                      if len(daily_arr) > 1 and np.std(daily_arr) > 0 else 0)
            pnl = sum(pnls)
            pnl_s = f"${pnl:>9,.0f}" if pnl >= 0 else f"-${abs(pnl):>8,.0f}"
            print(f"  {cfg_name:<32} {n:>6} {sharpe:>7.2f} {pnl_s} {wr:>5.1f}%")

            if cfg_name not in era_results:
                era_results[cfg_name] = {}
            era_results[cfg_name][era_name] = {'n': n, 'sharpe': round(sharpe, 3), 'pnl': round(pnl, 0), 'wr': round(wr, 1)}

    return era_results


# ═══════════════════════════════════════════════════════════════
# Phase 4: Walk-Forward OOS
# ═══════════════════════════════════════════════════════════════

def phase4_walkforward(data, p1_results):
    print(f"\n{'='*130}")
    print(f"  PHASE 4: Walk-Forward OOS Validation")
    print(f"{'='*130}")

    all_configs = {**CONFIGS, **KCBW_VARIANTS}
    baseline_kw = CONFIGS['A_baseline']

    candidates = [(k, v) for k, v in p1_results.items() if k != 'A_baseline']
    candidates.sort(key=lambda x: x[1]['stats']['sharpe'], reverse=True)
    top3 = candidates[:3]

    wf_results = {}

    for cand_name, _ in top3:
        cand_kw = all_configs.get(cand_name)
        if cand_kw is None:
            continue

        print(f"\n  --- {cand_name} vs A_baseline ---")
        print(f"  {'Window':<10} {'OOS_Base':>8} {'OOS_Cand':>8} {'Delta':>8} {'Win':>4}")

        wins = 0
        details = []
        for wf_name, is_start, is_end, oos_start, oos_end in WF_WINDOWS:
            oos_data = data.slice(oos_start, oos_end)
            if len(oos_data.m15_df) < 1000:
                continue
            br = run_variant(oos_data, f"base_{wf_name}", verbose=False, **baseline_kw)
            cr = run_variant(oos_data, f"{cand_name}_{wf_name}", verbose=False, **cand_kw)
            delta = cr['sharpe'] - br['sharpe']
            w = "+" if delta > 0 else "-"
            if delta > 0: wins += 1
            print(f"  {wf_name:<10} {br['sharpe']:>8.2f} {cr['sharpe']:>8.2f} {delta:>+8.3f} {w:>4}")
            details.append({'window': wf_name, 'base': br['sharpe'], 'cand': cr['sharpe'], 'delta': delta})

        total = len(details)
        passed = wins >= max(1, int(total * 0.6))
        print(f"  WF: {wins}/{total} {'PASS' if passed else 'FAIL'}")
        wf_results[cand_name] = {'wins': wins, 'total': total, 'pass': passed, 'details': details}

    return wf_results


# ═══════════════════════════════════════════════════════════════
# Phase 5: Cost sensitivity
# ═══════════════════════════════════════════════════════════════

def phase5_cost(data, p1_results):
    print(f"\n{'='*130}")
    print(f"  PHASE 5: Cost Sensitivity")
    print(f"{'='*130}")

    all_configs = {**CONFIGS, **KCBW_VARIANTS}
    baseline_kw = CONFIGS['A_baseline']

    candidates = [(k, v) for k, v in p1_results.items() if k != 'A_baseline']
    candidates.sort(key=lambda x: x[1]['stats']['sharpe'], reverse=True)
    best_name = candidates[0][0] if candidates else None
    if best_name is None:
        print("  No candidates.")
        return {}

    best_kw = all_configs[best_name]

    print(f"\n  {best_name} vs A_baseline at different spreads:")
    print(f"  {'Spread':>8} {'Base_Sh':>8} {'Cand_Sh':>8} {'Delta':>8} {'Base_PnL':>10} {'Cand_PnL':>10}")

    results = {}
    for sp in [0.20, 0.30, 0.50, 0.80, 1.00]:
        kw_b = {**baseline_kw, 'spread_cost': sp}
        kw_c = {**best_kw, 'spread_cost': sp}
        br = run_variant(data, f"base_sp{sp}", verbose=False, **kw_b)
        cr = run_variant(data, f"{best_name}_sp{sp}", verbose=False, **kw_c)
        delta = cr['sharpe'] - br['sharpe']
        print(f"  ${sp:<7.2f} {br['sharpe']:>8.2f} {cr['sharpe']:>8.2f} {delta:>+8.3f} "
              f"${br['total_pnl']:>9,.0f} ${cr['total_pnl']:>9,.0f}")
        results[f"sp_{sp}"] = {'spread': sp, 'base_sharpe': br['sharpe'], 'cand_sharpe': cr['sharpe'],
                                'delta': round(delta, 3)}

    return results


# ═══════════════════════════════════════════════════════════════
# Phase 6: Yearly stability
# ═══════════════════════════════════════════════════════════════

def phase6_yearly(p1_results):
    print(f"\n{'='*130}")
    print(f"  PHASE 6: Yearly Stability")
    print(f"{'='*130}")

    all_configs_keys = ['A_baseline', 'C_kcbw_replace_choppy', 'D_kcbw_replace_rsi_adx', 'E_kcbw_replace_both']
    all_configs_keys = [k for k in all_configs_keys if k in p1_results]

    yearly_data = {}
    for cfg_name in all_configs_keys:
        result = p1_results[cfg_name]['_result']
        trades = result.get('_trades', [])
        years = sorted(set(pd.Timestamp(t.entry_time).year for t in trades))

        yearly_data[cfg_name] = {}
        for yr in years:
            yr_trades = [t for t in trades if pd.Timestamp(t.entry_time).year == yr]
            pnls = [t.pnl for t in yr_trades]
            if not pnls:
                continue
            daily = {}
            for t in yr_trades:
                d = pd.Timestamp(t.exit_time).date()
                daily[d] = daily.get(d, 0) + t.pnl
            daily_arr = list(daily.values())
            sharpe = (np.mean(daily_arr) / np.std(daily_arr) * np.sqrt(252)
                      if len(daily_arr) > 1 and np.std(daily_arr) > 0 else 0)
            yearly_data[cfg_name][yr] = round(sharpe, 2)

    all_years = sorted(set(yr for d in yearly_data.values() for yr in d))
    header = f"  {'Year':>6}" + "".join(f" {c[:20]:>22}" for c in all_configs_keys)
    print(header)
    for yr in all_years:
        row = f"  {yr:>6}"
        for cfg in all_configs_keys:
            sh = yearly_data.get(cfg, {}).get(yr, 0)
            row += f" {sh:>22.2f}"
        print(row)

    if 'A_baseline' in yearly_data:
        for cfg in all_configs_keys:
            if cfg == 'A_baseline':
                continue
            wins = sum(1 for yr in all_years
                       if yearly_data.get(cfg, {}).get(yr, 0) > yearly_data['A_baseline'].get(yr, 0))
            print(f"  {cfg}: {wins}/{len(all_years)} years beat baseline")

    return yearly_data


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 130, flush=True)
    print("  R186 — KCBW Replace-Not-Stack: Engine Filter Substitution Test", flush=True)
    print("=" * 130, flush=True)

    data = DataBundle.load_default()

    all_results = {}
    p1 = phase1_full_comparison(data)

    p1_clean = {k: {'stats': v['stats']} for k, v in p1.items()}
    all_results['phase1'] = p1_clean

    all_results['phase2_kfold'] = phase2_kfold(data, p1)
    all_results['phase3_era'] = phase3_era(data, p1)
    all_results['phase4_wf'] = phase4_walkforward(data, p1)
    all_results['phase5_cost'] = phase5_cost(data, p1)
    all_results['phase6_yearly'] = phase6_yearly(p1)

    elapsed = time.time() - t0

    # ── Final Summary ──
    print(f"\n\n{'='*130}")
    print(f"  R186 FINAL SUMMARY")
    print(f"{'='*130}")

    baseline_sh = p1['A_baseline']['stats']['sharpe']
    print(f"\n  Baseline (A): Sharpe={baseline_sh:.3f}")

    print(f"\n  {'Config':<32} {'Sharpe':>7} {'Delta':>7} {'KFold':>10} {'WF':>10}")
    print(f"  {'-'*32} {'-'*7} {'-'*7} {'-'*10} {'-'*10}")

    all_configs = {**CONFIGS, **KCBW_VARIANTS}
    for name in sorted(p1.keys(), key=lambda x: p1[x]['stats']['sharpe'], reverse=True):
        s = p1[name]['stats']
        delta = s['sharpe'] - baseline_sh
        kf = all_results.get('phase2_kfold', {}).get(name, {})
        kf_str = f"{kf['wins']}/{kf['total']} {'P' if kf['pass'] else 'F'}" if kf else "—"
        wf = all_results.get('phase4_wf', {}).get(name, {})
        wf_str = f"{wf['wins']}/{wf['total']} {'P' if wf['pass'] else 'F'}" if wf else "—"
        print(f"  {name:<32} {s['sharpe']:>7.3f} {delta:>+7.3f} {kf_str:>10} {wf_str:>10}")

    best = max(p1.items(), key=lambda x: x[1]['stats']['sharpe'])
    best_name = best[0]
    best_sh = best[1]['stats']['sharpe']
    print(f"\n  Best config: {best_name} (Sharpe={best_sh:.3f}, delta={best_sh - baseline_sh:+.3f})")

    kf_pass = all_results.get('phase2_kfold', {}).get(best_name, {}).get('pass', False)
    wf_pass = all_results.get('phase4_wf', {}).get(best_name, {}).get('pass', False)

    if best_sh > baseline_sh and kf_pass and wf_pass:
        print(f"  VERDICT: GO — {best_name} improves on baseline with robust validation")
    elif best_sh > baseline_sh:
        kf_s = "PASS" if kf_pass else "FAIL"
        wf_s = "PASS" if wf_pass else "FAIL"
        print(f"  VERDICT: CAUTION — {best_name} higher Sharpe but KFold={kf_s}, WF={wf_s}")
    else:
        print(f"  VERDICT: NO-GO — no substitution improves on baseline")

    print(f"\n  Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    out_path = OUTPUT_DIR / "r186_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {out_path}")
    print(f"{'='*130}")


if __name__ == "__main__":
    main()
