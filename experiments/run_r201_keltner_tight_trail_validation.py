#!/usr/bin/env python3
"""
R201: Keltner ta=0.06/td=0.015 M15 Engine Validation
=====================================================
Validates the D2 top candidate (ta=0.06/td=0.015) before live deployment.

Compares:
  1. Current production (ta=0.14/td=0.025) — baseline
  2. D2 candidate (ta=0.06/td=0.015) — tight trail

For each, runs with and without regime_config:
  - regime_config=None: true sensitivity to trail params
  - regime_config=LIVE_PARITY: simulates actual live behavior (regime adapts)

Plus:
  - 3-Gate validation on candidate (K-Fold + WF + Era)
  - Exit reason breakdown (Trail%, SL%, TP%, Cap%, Timeout%)
  - Bootstrap CI on Sharpe
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.runner import (
    DataBundle, run_variant, run_kfold, run_variants_parallel,
    LIVE_PARITY_KWARGS,
)
from backtest.runner import _worker_run_variant as _runner_worker

MAX_WORKERS = min(mp.cpu_count(), 64)

OUTPUT_DIR = Path("results/r201_keltner_tight_trail")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERA_SEGMENTS = {
    "Full (2015-2026)":         ("2015-01-01", "2026-05-06"),
    "Pre-COVID (2015-2019)":    ("2015-01-01", "2020-01-01"),
    "COVID+Recovery (2020-2021)": ("2020-01-01", "2022-01-01"),
    "Tightening (2022-2023)":   ("2022-01-01", "2024-01-01"),
    "Recent (2024-2026)":       ("2024-01-01", "2026-05-06"),
}

# Build walk-forward windows (19 windows)
WF_WINDOWS = []
for yr in range(2017, 2027):
    train_s = f"{yr-2}-01-01"
    train_e = f"{yr}-01-01"
    test_s = f"{yr}-01-01"
    test_e = f"{yr}-07-01" if yr < 2026 else "2026-05-06"
    WF_WINDOWS.append((train_s, train_e, test_s, test_e))
    if yr < 2026:
        WF_WINDOWS.append((f"{yr-2}-07-01", f"{yr}-07-01", f"{yr}-07-01", f"{yr+1}-01-01"))

# Candidate & baseline configurations
BASELINE = {'ta': 0.14, 'td': 0.025, 'label': 'baseline'}
CANDIDATE = {'ta': 0.06, 'td': 0.015, 'label': 'candidate_D2top'}


def build_variant(ta, td, label, disable_regime=False):
    """Build engine kwargs for a given trail setting."""
    v = dict(LIVE_PARITY_KWARGS)
    v['trailing_activate_atr'] = ta
    v['trailing_distance_atr'] = td
    if disable_regime:
        v['regime_config'] = None
    v['label'] = label
    return v


def print_header(s):
    print(f"\n{'='*100}")
    print(f"  {s}")
    print(f"{'='*100}")


def bootstrap_sharpe_ci(daily_pnls, n_boot=2000, ci=0.95):
    """Bootstrap CI for annualized Sharpe."""
    if len(daily_pnls) < 10:
        return {'mean': 0, 'ci_lo': 0, 'ci_hi': 0, 'se': 0}
    rng = np.random.default_rng(42)
    d = np.asarray(daily_pnls)
    sharpes = []
    for _ in range(n_boot):
        sample = rng.choice(d, size=len(d), replace=True)
        if np.std(sample, ddof=1) > 0:
            sharpes.append(np.mean(sample) / np.std(sample, ddof=1) * np.sqrt(252))
    lo = np.percentile(sharpes, (1 - ci) / 2 * 100)
    hi = np.percentile(sharpes, (1 + ci) / 2 * 100)
    return {
        'mean': round(np.mean(sharpes), 3),
        'ci_lo': round(lo, 3), 'ci_hi': round(hi, 3),
        'se': round(np.std(sharpes), 3),
    }


def exit_reasons_from_trades(trades):
    """Aggregate exit-reason counts + PnL per reason."""
    reasons = {}
    total_n = len(trades)
    for t in trades:
        r = getattr(t, 'reason', None) or getattr(t, 'exit_reason', None) or 'Unknown'
        if r not in reasons:
            reasons[r] = {'n': 0, 'pnl': 0.0}
        pnl = getattr(t, 'pnl', 0) or 0
        reasons[r]['n'] += 1
        reasons[r]['pnl'] += pnl
    for r, v in reasons.items():
        v['pct'] = round(v['n'] / total_n * 100, 1) if total_n > 0 else 0
        v['avg_pnl'] = round(v['pnl'] / v['n'], 2) if v['n'] > 0 else 0
        v['pnl'] = round(v['pnl'], 2)
    return reasons


def daily_pnls_from_trades(trades):
    by_day = {}
    for t in trades:
        et = getattr(t, 'exit_time', None)
        if et is None:
            continue
        d = pd.Timestamp(et).date()
        by_day[d] = by_day.get(d, 0) + (getattr(t, 'pnl', 0) or 0)
    return list(by_day.values())


def main():
    t0 = time.time()
    print_header("R201: Keltner ta=0.06/td=0.015 M15 Engine Validation")

    print("\nLoading data...")
    data = DataBundle.load_default()

    # ─────────────────────────────────────────────────────────────
    # 1. Direct comparison: baseline vs candidate, with / without regime_config
    # ─────────────────────────────────────────────────────────────
    print_header("1. Direct Comparison (Full 2015-2026)")

    variants = [
        build_variant(BASELINE['ta'], BASELINE['td'], 'baseline_regOFF', disable_regime=True),
        build_variant(BASELINE['ta'], BASELINE['td'], 'baseline_regON', disable_regime=False),
        build_variant(CANDIDATE['ta'], CANDIDATE['td'], 'candidate_regOFF', disable_regime=True),
        build_variant(CANDIDATE['ta'], CANDIDATE['td'], 'candidate_regON', disable_regime=False),
    ]

    # Run sequentially to capture _trades (parallel drops them)
    direct_results = {}
    for v in variants:
        label = v.pop('label')
        print(f"\n  Running {label}...")
        stats = run_variant(data, label, verbose=False, **v)
        trades = stats.get('_trades', [])
        reasons = exit_reasons_from_trades(trades)
        dpnl = daily_pnls_from_trades(trades)
        boot = bootstrap_sharpe_ci(dpnl)
        direct_results[label] = {
            'sharpe': round(stats.get('sharpe', 0), 3),
            'sharpe_ci': boot,
            'pnl': round(stats.get('total_pnl', 0), 2),
            'n_trades': len(trades),
            'max_dd': round(stats.get('max_dd', 0), 2),
            'win_rate': round(stats.get('win_rate', 0), 2),
            'exit_reasons': reasons,
        }
        print(f"    Sharpe={direct_results[label]['sharpe']:.3f} "
              f"(CI: {boot['ci_lo']:.3f} to {boot['ci_hi']:.3f})  "
              f"PnL=${direct_results[label]['pnl']:.0f}  "
              f"N={len(trades)}  WR={direct_results[label]['win_rate']:.1f}%  "
              f"MaxDD=${direct_results[label]['max_dd']:.0f}")
        trail_info = reasons.get('Trail', {}) or reasons.get('trail', {})
        sl_info = reasons.get('SL', {})
        tp_info = reasons.get('TP', {})
        cap_info = reasons.get('MaxLossCap', reasons.get('Cap', {}))
        to_info = reasons.get('Timeout', reasons.get('MaxHold', {}))
        print(f"    Exit breakdown: "
              f"Trail={trail_info.get('pct',0):.1f}%(${trail_info.get('avg_pnl',0):.2f}/t)  "
              f"SL={sl_info.get('pct',0):.1f}%  TP={tp_info.get('pct',0):.1f}%  "
              f"Cap={cap_info.get('pct',0):.1f}%  Timeout={to_info.get('pct',0):.1f}%")

    # ─────────────────────────────────────────────────────────────
    # 2. Era-segmented performance for candidate
    # ─────────────────────────────────────────────────────────────
    print_header("2. Era-Segmented Performance (candidate vs baseline, regime_config=ON)")

    era_variants_baseline = []
    era_variants_candidate = []
    era_names = []
    for era_name, (s, e) in ERA_SEGMENTS.items():
        if era_name == 'Full (2015-2026)':
            continue
        era_data = data.slice(s, e)
        if len(era_data.m15_df) < 1000:
            continue
        era_names.append(era_name)
        vb = build_variant(BASELINE['ta'], BASELINE['td'], f'bl_{era_name}', disable_regime=False)
        vc = build_variant(CANDIDATE['ta'], CANDIDATE['td'], f'cd_{era_name}', disable_regime=False)
        era_variants_baseline.append((era_data, vb))
        era_variants_candidate.append((era_data, vc))

    era_results = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for (ed, v), name in zip(era_variants_baseline, era_names):
            label = v.pop('label')
            fut = pool.submit(_runner_worker, (ed.m15_df, ed.h1_df, label, v))
            futures[fut] = ('baseline', name)
        for (ed, v), name in zip(era_variants_candidate, era_names):
            label = v.pop('label')
            fut = pool.submit(_runner_worker, (ed.m15_df, ed.h1_df, label, v))
            futures[fut] = ('candidate', name)
        for fut in as_completed(futures):
            tag, name = futures[fut]
            stats = fut.result()
            era_results.setdefault(name, {})[tag] = {
                'sharpe': round(stats.get('sharpe', 0), 3),
                'pnl': round(stats.get('total_pnl', 0), 2),
                'n': stats.get('n', 0),
            }

    print(f"\n  {'Era':<30} {'Baseline':>20} {'Candidate':>20} {'Delta':>12}")
    for era_name in era_names:
        r = era_results.get(era_name, {})
        bl = r.get('baseline', {})
        cd = r.get('candidate', {})
        delta = cd.get('sharpe', 0) - bl.get('sharpe', 0)
        print(f"  {era_name:<30} "
              f"{bl.get('sharpe',0):>8.3f}/${bl.get('pnl',0):>8.0f} "
              f"{cd.get('sharpe',0):>8.3f}/${cd.get('pnl',0):>8.0f} "
              f"{delta:>+8.3f}")

    # ─────────────────────────────────────────────────────────────
    # 3. 3-Gate Validation on candidate (regime_config=ON, live-like)
    # ─────────────────────────────────────────────────────────────
    print_header("3. 3-Gate Validation on Candidate (regime_config=ON)")

    cand_kw = build_variant(CANDIDATE['ta'], CANDIDATE['td'], 'gate', disable_regime=False)
    cand_kw.pop('label', None)

    print("\n  K-Fold (6 folds, parallel)...")
    kf_results = run_kfold(data, cand_kw, n_folds=6, parallel=True)
    kf_sharpes = [r['sharpe'] for r in kf_results]
    kf_wins = sum(1 for s in kf_sharpes if s > 0)
    print(f"    Wins: {kf_wins}/6  Sharpes: {[round(s,2) for s in kf_sharpes]}")

    print("\n  Walk-Forward (parallel)...")
    wf_tasks = []
    for train_s, train_e, test_s, test_e in WF_WINDOWS:
        test_data = data.slice(test_s, test_e)
        if len(test_data.m15_df) < 500:
            continue
        v = dict(cand_kw)
        label = f"WF_{test_s}"
        wf_tasks.append((test_data.m15_df, test_data.h1_df, label, v))

    wf_wins = 0
    wf_total = len(wf_tasks)
    wf_details = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_runner_worker, args): args[2] for args in wf_tasks}
        for fut in as_completed(futures):
            label = futures[fut]
            stats = fut.result()
            sharpe = stats.get('sharpe', 0)
            wf_details.append({'window': label, 'sharpe': round(sharpe, 3)})
            if sharpe > 0:
                wf_wins += 1
    print(f"    Wins: {wf_wins}/{wf_total}")

    kfold_pass = kf_wins >= 4
    wf_pass = wf_wins >= 13
    era_pass = all(era_results[n]['candidate']['sharpe'] > 0 for n in era_names)
    gate_pass = kfold_pass and wf_pass and era_pass

    print(f"\n  === 3-Gate Verdict ===")
    print(f"    K-Fold: {kf_wins}/6  {'[PASS]' if kfold_pass else '[FAIL]'} (need >=4)")
    print(f"    Walk-Forward: {wf_wins}/{wf_total}  "
          f"{'[PASS]' if wf_pass else '[FAIL]'} (need >=13/19)")
    print(f"    Era: {'all positive' if era_pass else 'some negative'}  "
          f"{'[PASS]' if era_pass else '[FAIL]'}")
    print(f"    Overall: {'[GO]' if gate_pass else '[NO-GO]'}")

    # ─────────────────────────────────────────────────────────────
    # 4. Save results
    # ─────────────────────────────────────────────────────────────
    final = {
        'candidate': CANDIDATE,
        'baseline': BASELINE,
        'direct_comparison': direct_results,
        'era_segmented': era_results,
        'three_gate': {
            'kfold_wins': kf_wins,
            'kfold_sharpes': [round(s, 3) for s in kf_sharpes],
            'wf_wins': wf_wins,
            'wf_total': wf_total,
            'wf_details': wf_details,
            'era_pass': era_pass,
            'overall_pass': gate_pass,
        },
        'total_runtime_sec': round(time.time() - t0, 1),
    }
    out_path = OUTPUT_DIR / 'R201_keltner_tight_trail_validation.json'
    with open(out_path, 'w') as f:
        json.dump(final, f, indent=2, default=str)
    print_header(f"Saved to {out_path}")
    print(f"  Total runtime: {final['total_runtime_sec']:.0f}s "
          f"({final['total_runtime_sec']/60:.1f} min)")


if __name__ == '__main__':
    main()
