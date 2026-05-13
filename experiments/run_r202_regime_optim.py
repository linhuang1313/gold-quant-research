#!/usr/bin/env python3
"""
R202: Keltner Regime-Config Optimization
=========================================
R201 revealed that D2's top candidate (ta=0.06/td=0.015) is NEUTRALIZED by
regime_config at runtime — in live mode (regime_config=ON), it produces
identical results to baseline.

R202 tests 8 regime_config variants to find a configuration that:
  1. Keeps regime adaptation (+0.19 Sharpe vs no-regime baseline)
  2. Realizes the tight-trail improvement D2 identified (+0.79 Sharpe)

For each variant:
  - Full 2015-2026 Sharpe + exit reason breakdown
  - 4 era-segment Sharpes (Pre-COVID, COVID+Recovery, Tightening, Recent)

Then:
  - 3-Gate validation on top 2-3 candidates
  - Final GO/NO-GO verdict vs current production baseline
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
    DataBundle, run_variant, run_kfold,
    LIVE_PARITY_KWARGS,
)
from backtest.runner import _worker_run_variant as _runner_worker

MAX_WORKERS = min(mp.cpu_count(), 64)

OUTPUT_DIR = Path("results/r202_regime_optim")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERA_SEGMENTS = {
    "Pre-COVID (2015-2019)":      ("2015-01-01", "2020-01-01"),
    "COVID+Recovery (2020-2021)": ("2020-01-01", "2022-01-01"),
    "Tightening (2022-2023)":     ("2022-01-01", "2024-01-01"),
    "Recent (2024-2026)":         ("2024-01-01", "2026-05-06"),
}

# Build WF windows (19 total)
WF_WINDOWS = []
for yr in range(2017, 2027):
    train_s = f"{yr-2}-01-01"
    train_e = f"{yr}-01-01"
    test_s = f"{yr}-01-01"
    test_e = f"{yr}-07-01" if yr < 2026 else "2026-05-06"
    WF_WINDOWS.append((train_s, train_e, test_s, test_e))
    if yr < 2026:
        WF_WINDOWS.append((f"{yr-2}-07-01", f"{yr}-07-01", f"{yr}-07-01", f"{yr+1}-01-01"))


# ══════════════════════════════════════════════════════════════════
# 8 configurations to test
# Base values: trailing_activate_atr / trailing_distance_atr are the
# "default" values used when regime_config=None or fallback. When
# regime_config is ON, the runtime rc.get(regime, {}) overrides them.
# ══════════════════════════════════════════════════════════════════
ORIGINAL_REGIME = {
    'low':    {'trail_act': 0.22, 'trail_dist': 0.04},
    'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
    'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
}

CONFIGS = [
    # 1. Baseline (current production)
    {'label': 'baseline',
     'ta': 0.14, 'td': 0.025,
     'regime_config': dict(ORIGINAL_REGIME),
     'desc': 'Current production (regime ON, normal=0.14/0.025)'},

    # 2. Option A: D2 global tight, no regime adaptation
    {'label': 'A_global_tight',
     'ta': 0.06, 'td': 0.015,
     'regime_config': None,
     'desc': 'Global tight 0.06/0.015, regime OFF'},

    # 3. Option B: Regime ON, tighten normal to D2-top
    {'label': 'B_tight_normal',
     'ta': 0.14, 'td': 0.025,
     'regime_config': {
         'low':    {'trail_act': 0.22, 'trail_dist': 0.04},
         'normal': {'trail_act': 0.06, 'trail_dist': 0.015},
         'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
     },
     'desc': 'Only normal regime tightened to 0.06/0.015'},

    # 4. Option C: Tighten low+normal, keep high
    {'label': 'C_tight_lownormal',
     'ta': 0.14, 'td': 0.025,
     'regime_config': {
         'low':    {'trail_act': 0.10, 'trail_dist': 0.025},
         'normal': {'trail_act': 0.06, 'trail_dist': 0.015},
         'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
     },
     'desc': 'low tightened to 0.10/0.025, normal to 0.06/0.015'},

    # 5. Option D: All regimes tight
    {'label': 'D_all_tight',
     'ta': 0.06, 'td': 0.015,
     'regime_config': {
         'low':    {'trail_act': 0.06, 'trail_dist': 0.015},
         'normal': {'trail_act': 0.06, 'trail_dist': 0.015},
         'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
     },
     'desc': 'All regimes tight (effectively disables adaptation)'},

    # 6. Option E: Graduated tighter
    {'label': 'E_graduated_tight',
     'ta': 0.14, 'td': 0.025,
     'regime_config': {
         'low':    {'trail_act': 0.14, 'trail_dist': 0.025},
         'normal': {'trail_act': 0.08, 'trail_dist': 0.015},
         'high':   {'trail_act': 0.04, 'trail_dist': 0.008},
     },
     'desc': 'Graduated: low=0.14, normal=0.08, high=0.04'},

    # 7. Option F: Tightest D2 point globally
    {'label': 'F_global_tightest',
     'ta': 0.06, 'td': 0.008,
     'regime_config': None,
     'desc': 'D2 absolute top 0.06/0.008, regime OFF'},

    # 8. Option G: Tighten low only (keep normal & high)
    {'label': 'G_tight_low',
     'ta': 0.14, 'td': 0.025,
     'regime_config': {
         'low':    {'trail_act': 0.06, 'trail_dist': 0.015},
         'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
         'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
     },
     'desc': 'Only low regime tightened (isolate low-vol effect)'},
]


def build_kwargs(cfg):
    v = dict(LIVE_PARITY_KWARGS)
    v['trailing_activate_atr'] = cfg['ta']
    v['trailing_distance_atr'] = cfg['td']
    v['regime_config'] = cfg['regime_config']
    return v


def print_header(s):
    print(f"\n{'='*100}")
    print(f"  {s}")
    print(f"{'='*100}")


def exit_reasons_from_trades(trades):
    reasons = {}
    total = len(trades)
    for t in trades:
        r = getattr(t, 'exit_reason', None) or getattr(t, 'reason', None) or 'Unknown'
        r = str(r)
        if r.startswith('M15 RSI') or r.startswith('Timeout'):
            r = r.split(':')[0].strip()
        if r not in reasons:
            reasons[r] = {'n': 0, 'pnl': 0.0}
        pnl = getattr(t, 'pnl', 0) or 0
        reasons[r]['n'] += 1
        reasons[r]['pnl'] += pnl
    for r, v in reasons.items():
        v['pct'] = round(v['n'] / total * 100, 1) if total > 0 else 0
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


def bootstrap_ci(daily_pnls, n_boot=1000, ci=0.95):
    if len(daily_pnls) < 10:
        return {'mean': 0, 'ci_lo': 0, 'ci_hi': 0}
    rng = np.random.default_rng(42)
    d = np.asarray(daily_pnls)
    sharpes = []
    for _ in range(n_boot):
        s = rng.choice(d, size=len(d), replace=True)
        if np.std(s, ddof=1) > 0:
            sharpes.append(np.mean(s) / np.std(s, ddof=1) * np.sqrt(252))
    return {
        'mean': round(np.mean(sharpes), 3),
        'ci_lo': round(np.percentile(sharpes, (1 - ci) / 2 * 100), 3),
        'ci_hi': round(np.percentile(sharpes, (1 + ci) / 2 * 100), 3),
    }


def main():
    t0 = time.time()
    print_header("R202: Keltner Regime-Config Optimization")
    print("Loading data...")
    data = DataBundle.load_default()

    # ──────────────────────────────────────────────────────────
    # 1. Full 2015-2026 comparison (serial, captures _trades)
    # ──────────────────────────────────────────────────────────
    print_header("1. Full 2015-2026 Direct Comparison")

    full_results = {}
    for cfg in CONFIGS:
        label = cfg['label']
        kw = build_kwargs(cfg)
        print(f"\n  Running {label} ({cfg['desc']})...")
        stats = run_variant(data, label, verbose=False, **kw)
        trades = stats.get('_trades', [])
        reasons = exit_reasons_from_trades(trades)
        dpnl = daily_pnls_from_trades(trades)
        ci = bootstrap_ci(dpnl)
        full_results[label] = {
            'sharpe': round(stats.get('sharpe', 0), 3),
            'sharpe_ci': ci,
            'pnl': round(stats.get('total_pnl', 0), 2),
            'n_trades': len(trades),
            'max_dd': round(stats.get('max_dd', 0), 2),
            'win_rate': round(stats.get('win_rate', 0), 2),
            'exit_reasons': reasons,
            'config_desc': cfg['desc'],
        }
        trail_info = reasons.get('Trailing', {})
        to_info = reasons.get('Timeout', {})
        sl_info = reasons.get('SL', {})
        tp_info = reasons.get('TP', {})
        cap_info = reasons.get('MaxLossCap', {})
        print(f"    Sharpe={full_results[label]['sharpe']:.3f} "
              f"(CI: {ci['ci_lo']:.2f}-{ci['ci_hi']:.2f})  "
              f"PnL=${full_results[label]['pnl']:.0f}  N={len(trades)}  "
              f"WR={full_results[label]['win_rate']:.1f}%  MaxDD=${full_results[label]['max_dd']:.0f}")
        print(f"    Exit: Trail={trail_info.get('pct',0):.1f}%"
              f"(${trail_info.get('avg_pnl',0):+.2f}/t)  "
              f"Timeout={to_info.get('pct',0):.1f}%"
              f"(${to_info.get('avg_pnl',0):+.2f}/t)  "
              f"SL={sl_info.get('pct',0):.1f}%  TP={tp_info.get('pct',0):.1f}%  "
              f"Cap={cap_info.get('pct',0):.1f}%")

    # Summary table
    print("\n  " + "─" * 98)
    print(f"  {'Config':<22} {'Sharpe':>8} {'95% CI':>18} {'PnL':>10} {'N':>7} {'Trail%':>8} {'Delta':>7}")
    base_sharpe = full_results['baseline']['sharpe']
    for cfg in CONFIGS:
        label = cfg['label']
        r = full_results[label]
        ci = r['sharpe_ci']
        trail_pct = r['exit_reasons'].get('Trailing', {}).get('pct', 0)
        delta = r['sharpe'] - base_sharpe
        print(f"  {label:<22} {r['sharpe']:>8.3f} "
              f"{ci['ci_lo']:>6.2f}-{ci['ci_hi']:<6.2f}  "
              f"${r['pnl']:>8.0f} {r['n_trades']:>7} {trail_pct:>7.1f}% "
              f"{delta:>+7.3f}")
    print("  " + "─" * 98)

    # ──────────────────────────────────────────────────────────
    # 2. Era-segmented (parallel)
    # ──────────────────────────────────────────────────────────
    print_header("2. Era-Segmented Performance (parallel)")

    era_tasks = []
    for cfg in CONFIGS:
        for era_name, (s, e) in ERA_SEGMENTS.items():
            era_data = data.slice(s, e)
            if len(era_data.m15_df) < 1000:
                continue
            kw = build_kwargs(cfg)
            kw.pop('label', None)
            era_tasks.append((cfg['label'], era_name, era_data.m15_df, era_data.h1_df,
                              f"{cfg['label']}_{era_name}", kw))

    print(f"  Running {len(era_tasks)} era backtests in parallel ({MAX_WORKERS} workers)...")
    era_results = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for task in era_tasks:
            cfg_label, era_name, m15, h1, run_label, kw = task
            fut = pool.submit(_runner_worker, (m15, h1, run_label, kw))
            futures[fut] = (cfg_label, era_name)
        for fut in as_completed(futures):
            cfg_label, era_name = futures[fut]
            stats = fut.result()
            era_results.setdefault(cfg_label, {})[era_name] = {
                'sharpe': round(stats.get('sharpe', 0), 3),
                'pnl': round(stats.get('total_pnl', 0), 2),
                'n': stats.get('n', 0),
            }

    # Era table
    print(f"\n  {'Config':<22}", end='')
    for era_name in ERA_SEGMENTS.keys():
        short = era_name.split(' ')[0][:10]
        print(f"{short:>12}", end='')
    print(f"{'All+?':>8}")
    for cfg in CONFIGS:
        label = cfg['label']
        row = era_results.get(label, {})
        print(f"  {label:<22}", end='')
        all_pos = True
        for era_name in ERA_SEGMENTS.keys():
            s = row.get(era_name, {}).get('sharpe', 0)
            if s <= 0:
                all_pos = False
            print(f"{s:>12.3f}", end='')
        print(f"{'YES' if all_pos else 'NO':>8}")

    # ──────────────────────────────────────────────────────────
    # 3. Pick top 2 candidates and run 3-Gate
    # ──────────────────────────────────────────────────────────
    print_header("3. 3-Gate Validation (top-2 candidates vs baseline)")

    # Score = full Sharpe + min era Sharpe (rewards consistency)
    scores = {}
    for cfg in CONFIGS:
        label = cfg['label']
        if label == 'baseline':
            continue
        full_s = full_results[label]['sharpe']
        era_ss = [era_results[label][e]['sharpe'] for e in ERA_SEGMENTS.keys()]
        min_era = min(era_ss) if era_ss else 0
        scores[label] = full_s + min_era * 0.3
    top2 = sorted(scores.items(), key=lambda x: -x[1])[:2]
    print(f"  Top candidates: {[t[0] for t in top2]}")

    three_gate_results = {}
    for cand_label, _score in top2:
        cfg = next(c for c in CONFIGS if c['label'] == cand_label)
        kw = build_kwargs(cfg)
        print(f"\n  --- {cand_label} ---")

        # K-Fold (parallel inside)
        kf = run_kfold(data, kw, n_folds=6, parallel=True)
        kf_sharpes = [r['sharpe'] for r in kf]
        kf_wins = sum(1 for s in kf_sharpes if s > 0)
        print(f"    K-Fold: {kf_wins}/6  {[round(s,2) for s in kf_sharpes]}")

        # Walk-Forward (parallel)
        wf_tasks = []
        for train_s, train_e, test_s, test_e in WF_WINDOWS:
            test_data = data.slice(test_s, test_e)
            if len(test_data.m15_df) < 500:
                continue
            label = f"{cand_label}_WF_{test_s}"
            wf_tasks.append((test_data.m15_df, test_data.h1_df, label, dict(kw)))
        wf_wins = 0
        wf_total = len(wf_tasks)
        wf_details = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {pool.submit(_runner_worker, args): args[2] for args in wf_tasks}
            for fut in as_completed(futs):
                label = futs[fut]
                stats = fut.result()
                s = stats.get('sharpe', 0)
                wf_details.append({'window': label.rsplit('_', 1)[-1] if '_' in label else label, 'sharpe': round(s, 3)})
                if s > 0:
                    wf_wins += 1
        print(f"    Walk-Forward: {wf_wins}/{wf_total}")

        # Era
        era_ss = era_results[cand_label]
        era_pass = all(v['sharpe'] > 0 for v in era_ss.values())
        print(f"    Era: {'PASS' if era_pass else 'FAIL'}  {[(k[:10], round(v['sharpe'],2)) for k,v in era_ss.items()]}")

        kfold_pass = kf_wins >= 4
        wf_pass = wf_wins >= 13
        overall = kfold_pass and wf_pass and era_pass
        three_gate_results[cand_label] = {
            'kfold_wins': kf_wins, 'kfold_sharpes': [round(s, 3) for s in kf_sharpes],
            'wf_wins': wf_wins, 'wf_total': wf_total, 'wf_details': wf_details,
            'era_pass': era_pass,
            'kfold_pass': kfold_pass, 'wf_pass': wf_pass,
            'overall_pass': overall,
        }
        print(f"    Overall: {'[GO]' if overall else '[NO-GO]'}")

    # ──────────────────────────────────────────────────────────
    # 4. Save + verdict
    # ──────────────────────────────────────────────────────────
    print_header("4. Final Summary vs Baseline")

    base = full_results['baseline']
    print(f"  Baseline: Sharpe={base['sharpe']:.3f}  PnL=${base['pnl']:.0f}  "
          f"MaxDD=${base['max_dd']:.0f}")

    for cand_label, _ in top2:
        r = full_results[cand_label]
        g = three_gate_results[cand_label]
        delta_s = r['sharpe'] - base['sharpe']
        delta_p = r['pnl'] - base['pnl']
        print(f"\n  {cand_label}:")
        print(f"    Sharpe: {r['sharpe']:.3f} ({delta_s:+.3f})  "
              f"PnL: ${r['pnl']:.0f} ({delta_p:+.0f})  "
              f"MaxDD: ${r['max_dd']:.0f}")
        print(f"    3-Gate: KF={g['kfold_wins']}/6 WF={g['wf_wins']}/{g['wf_total']} "
              f"Era={'PASS' if g['era_pass'] else 'FAIL'} -> "
              f"{'[GO]' if g['overall_pass'] else '[NO-GO]'}")

    # Save
    final = {
        'configs': [{'label': c['label'], 'ta': c['ta'], 'td': c['td'],
                     'regime_config': c['regime_config'], 'desc': c['desc']}
                    for c in CONFIGS],
        'full_results': full_results,
        'era_results': era_results,
        'top2': [t[0] for t in top2],
        'three_gate': three_gate_results,
        'runtime_sec': round(time.time() - t0, 1),
    }
    out = OUTPUT_DIR / 'R202_regime_optim.json'
    with open(out, 'w') as f:
        json.dump(final, f, indent=2, default=str)
    print_header(f"Saved to {out}  (runtime: {final['runtime_sec']:.0f}s)")


if __name__ == '__main__':
    main()
