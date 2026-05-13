#!/usr/bin/env python3
"""
R197 — Trend Score (Choppy Threshold) Comprehensive Sweep
==========================================================
Tests choppy_threshold from 0.40 to 0.55 (16 values, step 0.01).
Also tests 0.00 (disabled) as baseline.

For EACH threshold:
  Phase 1: Full-sample Sharpe/PnL/WR/MaxDD/N
  Phase 2: 6-Fold Cross-Validation (PASS = >= 4/6 folds win vs OFF)
  Phase 3: 19-Window Walk-Forward (PASS = >= 13/19 OOS wins vs OFF)
  Phase 4: Era Segmentation (PASS = all eras positive, no degradation > 0.3)

Uses the official backtest engine to ensure parity with live system.
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.engine import BacktestEngine
from backtest.runner import DataBundle, run_variant, calc_stats

OUTPUT_DIR = Path("results/r197_trend_score")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
t0 = time.time()

THRESHOLDS = [round(x, 2) for x in np.arange(0.40, 0.551, 0.01)]
SPREAD = 0.30
LOT = 0.04
CAP = 70.0

def phase_done(name):
    return (OUTPUT_DIR / f"{name}.json").exists()

def save_phase(name, data):
    with open(OUTPUT_DIR / f"{name}.json", 'w') as f:
        json.dump(data, f, indent=2, default=str)

def elapsed():
    return f"[{time.time()-t0:.0f}s]"


def run_keltner_with_threshold(data: DataBundle, choppy_thr, label=""):
    """Run Keltner backtest with a specific choppy threshold.
    choppy_thr=0 means intraday_adaptive=False (disabled)."""
    kwargs = {
        'trailing_activate_atr': 0.06,
        'trailing_distance_atr': 0.01,
        'sl_atr_mult': 3.5,
        'tp_atr_mult': 8.0,
        'keltner_adx_threshold': 14,
        'max_positions': 1,
        'live_atr_percentile': True,
        'spread_cost': SPREAD,
    }

    if choppy_thr > 0:
        kwargs['intraday_adaptive'] = True
        kwargs['choppy_threshold'] = choppy_thr
        kwargs['kc_only_threshold'] = 0.60
    else:
        kwargs['intraday_adaptive'] = False

    import io, contextlib
    eng = BacktestEngine(data.m15_df, data.h1_df, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        trades = eng.run()

    skipped = getattr(eng, 'skipped_choppy', 0)
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0, 'skipped_choppy': skipped}

    pnls = []
    for t in trades:
        pnl_usd = t.pnl * LOT * 100
        if CAP > 0 and pnl_usd < -CAP:
            pnl_usd = -CAP
        pnls.append(pnl_usd)

    n = len(pnls)
    total_pnl = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n * 100 if n > 0 else 0

    daily = {}
    for t, p in zip(trades, pnls):
        d = pd.Timestamp(t.exit_time).normalize()
        daily[d] = daily.get(d, 0) + p
    ds = pd.Series(daily).sort_index()
    sharpe = float(ds.mean() / ds.std() * np.sqrt(252)) if len(ds) > 10 and ds.std() > 0 else 0
    eq = ds.cumsum()
    max_dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 1 else 0

    return {
        'n': n, 'sharpe': round(sharpe, 3), 'pnl': round(total_pnl, 2),
        'wr': round(wr, 1), 'max_dd': round(max_dd, 2),
        'skipped_choppy': skipped,
    }


# ═══════════════ Load Data ═══════════════
print(f"{'='*90}")
print(f"  R197 — Trend Score Choppy Threshold Comprehensive Sweep")
print(f"  Thresholds: {THRESHOLDS[0]} to {THRESHOLDS[-1]} (step 0.01) + OFF baseline")
print(f"{'='*90}")

print(f"\n{elapsed()} Loading data...")
data_full = DataBundle.load_custom()
print(f"{elapsed()} Data loaded: {len(data_full.h1_df)} H1, {len(data_full.m15_df)} M15")


# ═══════════════ Phase 1: Full-Sample Sweep ═══════════════
if not phase_done("phase1_sweep"):
    print(f"\n{'─'*90}")
    print(f"  Phase 1: Full-Sample Sweep ({len(THRESHOLDS)} thresholds + OFF)")
    print(f"{'─'*90}")

    results = {}

    r_off = run_keltner_with_threshold(data_full, 0.0, "OFF")
    results['OFF'] = r_off
    print(f"  {'OFF (disabled)':<20} Sharpe={r_off['sharpe']:>7.3f}  PnL=${r_off['pnl']:>10,.0f}  WR={r_off['wr']:>5.1f}%  MaxDD=${r_off['max_dd']:>7,.0f}  N={r_off['n']:>6}  Skipped=0")

    for thr in THRESHOLDS:
        r = run_keltner_with_threshold(data_full, thr, f"thr={thr}")
        results[str(thr)] = r
        delta_s = r['sharpe'] - r_off['sharpe']
        print(f"  thr={thr:.2f}          Sharpe={r['sharpe']:>7.3f}  PnL=${r['pnl']:>10,.0f}  WR={r['wr']:>5.1f}%  MaxDD=${r['max_dd']:>7,.0f}  N={r['n']:>6}  Skipped={r['skipped_choppy']:>5}  dSharpe={delta_s:>+.3f}")

    save_phase("phase1_sweep", results)
    print(f"\n{elapsed()} Phase 1 complete.")
else:
    print(f"\n{elapsed()} Phase 1 already done, loading...")
    with open(OUTPUT_DIR / "phase1_sweep.json") as f:
        results = json.load(f)
    r_off = results['OFF']


# ═══════════════ Phase 2: 6-Fold Cross-Validation ═══════════════
if not phase_done("phase2_kfold"):
    print(f"\n{'─'*90}")
    print(f"  Phase 2: 6-Fold Cross-Validation (each threshold vs OFF)")
    print(f"{'─'*90}")

    kf_results = {}
    n_folds = 6
    h1_idx = data_full.h1_df.index
    fold_size_h1 = len(h1_idx) // n_folds

    for thr in THRESHOLDS:
        wins = 0
        fold_details = []
        for fold in range(n_folds):
            fs = fold * fold_size_h1
            fe = min((fold + 1) * fold_size_h1, len(h1_idx))
            start_ts = str(h1_idx[fs])
            end_ts = str(h1_idx[fe - 1])

            data_fold = data_full.slice(start_ts, end_ts)
            if len(data_fold.h1_df) < 500:
                continue

            r_new = run_keltner_with_threshold(data_fold, thr)
            r_base = run_keltner_with_threshold(data_fold, 0.0)

            win = 1 if r_new['sharpe'] > r_base['sharpe'] else 0
            wins += win
            fold_details.append({
                'fold': fold, 'new_sharpe': r_new['sharpe'], 'base_sharpe': r_base['sharpe'],
                'delta': round(r_new['sharpe'] - r_base['sharpe'], 3), 'win': win
            })

        passed = wins >= 4
        kf_results[str(thr)] = {
            'wins': wins, 'total': n_folds, 'passed': passed, 'folds': fold_details
        }
        status = "PASS" if passed else "FAIL"
        print(f"  thr={thr:.2f}  K-Fold: {wins}/{n_folds} wins  [{status}]  deltas={[f['delta'] for f in fold_details]}")

    save_phase("phase2_kfold", kf_results)
    print(f"\n{elapsed()} Phase 2 complete.")
else:
    print(f"\n{elapsed()} Phase 2 already done, loading...")
    with open(OUTPUT_DIR / "phase2_kfold.json") as f:
        kf_results = json.load(f)


# ═══════════════ Phase 3: Walk-Forward Validation ═══════════════
if not phase_done("phase3_walkforward"):
    print(f"\n{'─'*90}")
    print(f"  Phase 3: 19-Window Walk-Forward (each threshold vs OFF)")
    print(f"{'─'*90}")

    wf_results = {}
    n_windows = 19
    total_h1 = len(data_full.h1_df)
    window_size = total_h1 // (n_windows + 1)

    for thr in THRESHOLDS:
        wins = 0
        wf_details = []

        for w in range(n_windows):
            oos_start = total_h1 - (n_windows - w) * window_size
            oos_end = min(oos_start + window_size, total_h1)
            if oos_start < 0 or oos_end - oos_start < 300:
                continue

            start_ts = str(h1_idx[oos_start])
            end_ts = str(h1_idx[oos_end - 1])
            data_oos = data_full.slice(start_ts, end_ts)
            if len(data_oos.m15_df) < 300:
                continue

            r_new = run_keltner_with_threshold(data_oos, thr)
            r_base = run_keltner_with_threshold(data_oos, 0.0)

            win = 1 if r_new['sharpe'] > r_base['sharpe'] else 0
            wins += win
            wf_details.append({
                'window': w, 'new_sharpe': r_new['sharpe'], 'base_sharpe': r_base['sharpe'],
                'delta': round(r_new['sharpe'] - r_base['sharpe'], 3), 'win': win
            })

        passed = wins >= 13
        wf_results[str(thr)] = {
            'wins': wins, 'total': len(wf_details), 'passed': passed, 'windows': wf_details
        }
        status = "PASS" if passed else "FAIL"
        print(f"  thr={thr:.2f}  WF: {wins}/{len(wf_details)} OOS wins  [{status}]")

    save_phase("phase3_walkforward", wf_results)
    print(f"\n{elapsed()} Phase 3 complete.")
else:
    print(f"\n{elapsed()} Phase 3 already done, loading...")
    with open(OUTPUT_DIR / "phase3_walkforward.json") as f:
        wf_results = json.load(f)


# ═══════════════ Phase 4: Era Segmentation ═══════════════
if not phase_done("phase4_era"):
    print(f"\n{'─'*90}")
    print(f"  Phase 4: Era Segmentation (each threshold vs OFF)")
    print(f"{'─'*90}")

    eras = {
        'Pre-COVID (2015-2019)': ('2015-01-01', '2019-12-31'),
        'COVID (2020-2021)': ('2020-01-01', '2021-12-31'),
        'Hike Cycle (2022-2023)': ('2022-01-01', '2023-12-31'),
        'Recent (2024-2026)': ('2024-01-01', '2026-12-31'),
    }

    era_results = {}

    for thr in THRESHOLDS:
        era_detail = {}
        all_positive = True
        no_degrade = True

        for era_name, (start, end) in eras.items():
            data_era = data_full.slice(start, end)
            if len(data_era.h1_df) < 500 or len(data_era.m15_df) < 300:
                continue

            r_new = run_keltner_with_threshold(data_era, thr)
            r_base = run_keltner_with_threshold(data_era, 0.0)

            delta = round(r_new['sharpe'] - r_base['sharpe'], 3)
            if r_new['sharpe'] <= 0:
                all_positive = False
            if delta < -0.3:
                no_degrade = False

            era_detail[era_name] = {
                'new_sharpe': r_new['sharpe'], 'base_sharpe': r_base['sharpe'],
                'delta': delta, 'new_pnl': r_new['pnl'], 'base_pnl': r_base['pnl'],
            }

        passed = all_positive and no_degrade
        era_results[str(thr)] = {
            'passed': passed, 'all_positive': all_positive, 'no_degrade': no_degrade,
            'eras': era_detail
        }
        status = "PASS" if passed else "FAIL"
        era_deltas = {k: v['delta'] for k, v in era_detail.items()}
        print(f"  thr={thr:.2f}  Era: [{status}]  deltas={era_deltas}")

    save_phase("phase4_era", era_results)
    print(f"\n{elapsed()} Phase 4 complete.")
else:
    print(f"\n{elapsed()} Phase 4 already done, loading...")
    with open(OUTPUT_DIR / "phase4_era.json") as f:
        era_results = json.load(f)


# ═══════════════ Final Summary ═══════════════
print(f"\n{'='*90}")
print(f"  FINAL SUMMARY — R197 Trend Score Threshold Sweep")
print(f"{'='*90}")

header = f"  {'Threshold':<10} {'Sharpe':>8} {'PnL':>10} {'WR':>6} {'MaxDD':>8} {'N':>6} {'dSharpe':>8} {'K-Fold':>8} {'WF':>8} {'Era':>6} {'Verdict':>8}"
print(header)
print(f"  {'-'*98}")

r_off = results.get('OFF', {})
print(f"  {'OFF':<10} {r_off.get('sharpe',0):>8.3f} {r_off.get('pnl',0):>10,.0f} {r_off.get('wr',0):>5.1f}% {r_off.get('max_dd',0):>8,.0f} {r_off.get('n',0):>6}  {'baseline':>8} {'---':>8} {'---':>8} {'---':>6} {'BASE':>8}")

best_thr = None
best_sharpe = -999

for thr in THRESHOLDS:
    key = str(thr)
    r = results.get(key, {})
    kf = kf_results.get(key, {})
    wf = wf_results.get(key, {})
    era = era_results.get(key, {})

    delta_s = r.get('sharpe', 0) - r_off.get('sharpe', 0)
    kf_str = f"{kf.get('wins',0)}/{kf.get('total',6)}"
    wf_str = f"{wf.get('wins',0)}/{wf.get('total',19)}"
    era_str = "PASS" if era.get('passed', False) else "FAIL"

    kf_pass = kf.get('passed', False)
    wf_pass = wf.get('passed', False)
    era_pass = era.get('passed', False)
    all_pass = kf_pass and wf_pass and era_pass
    verdict = "GO" if all_pass else "NO-GO"

    if all_pass and r.get('sharpe', 0) > best_sharpe:
        best_sharpe = r.get('sharpe', 0)
        best_thr = thr

    kf_mark = "PASS" if kf_pass else "FAIL"
    wf_mark = "PASS" if wf_pass else "FAIL"

    print(f"  {thr:<10.2f} {r.get('sharpe',0):>8.3f} {r.get('pnl',0):>10,.0f} {r.get('wr',0):>5.1f}% {r.get('max_dd',0):>8,.0f} {r.get('n',0):>6}  {delta_s:>+8.3f} {kf_str:>5} {kf_mark:<4} {wf_str:>5} {wf_mark:<4} {era_str:>6} {verdict:>8}")

print(f"\n  {'='*98}")
if best_thr is not None:
    print(f"  RECOMMENDATION: choppy_threshold = {best_thr:.2f} (Sharpe {best_sharpe:.3f})")
    print(f"  Current live setting: 0.50")
    if abs(best_thr - 0.50) < 0.001:
        print(f"  -> Current setting IS optimal. No change needed.")
    else:
        print(f"  -> Consider changing from 0.50 to {best_thr:.2f}")
else:
    go_count = sum(1 for thr in THRESHOLDS if
                   kf_results.get(str(thr), {}).get('passed', False) and
                   wf_results.get(str(thr), {}).get('passed', False) and
                   era_results.get(str(thr), {}).get('passed', False))
    if go_count == 0:
        print(f"  No threshold passed all 3 gates. Consider disabling trend_score gating.")
    else:
        print(f"  {go_count} thresholds passed all gates.")

total_time = time.time() - t0
print(f"\n  Total runtime: {total_time:.0f}s ({total_time/60:.1f}min)")
print(f"{'='*90}")
