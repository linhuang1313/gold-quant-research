#!/usr/bin/env python3
"""R210: Entry Hour Filter Analysis (Keltner)
=============================================
R200 B2 produced per-hour PnL data for Keltner, but never tested "what if we
block certain hours?" with proper 3-gate validation.

Key question: Are there UTC hours that consistently lose money, and does
blocking them survive 3-gate validation?

Phase 1 — Per-Hour PnL Analysis
  Run Keltner baseline with LIVE_PARITY_KWARGS, extract entry hour (UTC),
  build hour × PnL matrix (n_trades, total_pnl, avg_pnl, win_rate).
  Full period + 2022-2026 focus.

Phase 2 — Hour Block Sweep
  For each hour 0-23: "block this hour only" vs baseline.
  For combinations: block worst-2, worst-3 hours.
  Measure Sharpe delta.

  Implementation: POST-HOC filtering on trade list (approximation that
  doesn't account for capital reallocation, but sufficient for analysis).

Phase 3 — 3-Gate Validation on best hour filter
  6-fold CV, 19-window walk-forward, 4-era stability.

Run: python experiments/run_r210_entry_hour_filter.py
"""
from __future__ import annotations
import sys
import json
import time
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine
from backtest.stats import calc_stats
import research_config as config
import indicators as signals_mod
from indicators import get_orb_strategy

OUTPUT_DIR = Path("results/r210_entry_hour_filter")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    with open(p, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f'  -> saved {p}')


def run_keltner(data: DataBundle, label: str = "baseline", **extra_kw):
    """Run Keltner backtest, return (trades, equity_curve)."""
    get_orb_strategy().reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False

    kw = dict(LIVE_PARITY_KWARGS)
    kw.update(extra_kw)
    engine = BacktestEngine(data.m15_df, data.h1_df, label=label, **kw)
    trades = engine.run()
    return trades, engine.equity_curve


def filter_trades_by_hour(trades, blocked_hours: set):
    """Post-hoc filter: remove trades whose entry hour is in blocked_hours."""
    return [t for t in trades if pd.Timestamp(t.entry_time).hour not in blocked_hours]


def trade_stats(trades, equity_curve=None):
    """Lightweight stats from a list of TradeRecord (uses calc_stats if equity_curve available)."""
    if equity_curve is not None:
        return calc_stats(trades, equity_curve)
    if not trades:
        return {'n': 0, 'total_pnl': 0, 'sharpe': 0, 'win_rate': 0, 'max_dd': 0}
    pnls = [t.pnl for t in trades]
    n = len(pnls)
    total = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)

    daily: Dict = {}
    for t in trades:
        d = pd.Timestamp(t.exit_time).date()
        daily[d] = daily.get(d, 0) + t.pnl
    daily_pnl = list(daily.values())
    sharpe = 0.0
    if len(daily_pnl) > 1 and np.std(daily_pnl, ddof=1) > 0:
        sharpe = np.mean(daily_pnl) / np.std(daily_pnl, ddof=1) * np.sqrt(252)

    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(dd.max()) if len(dd) > 0 else 0

    return {
        'n': n,
        'total_pnl': round(total, 2),
        'sharpe': round(sharpe, 3),
        'win_rate': round(100 * wins / n, 2) if n > 0 else 0,
        'max_dd': round(max_dd, 2),
        'avg_pnl': round(total / n, 4) if n > 0 else 0,
    }


def hour_pnl_matrix(trades):
    """Build per-hour stats from trade list."""
    by_hour = defaultdict(list)
    for t in trades:
        h = pd.Timestamp(t.entry_time).hour
        by_hour[h].append(t.pnl)
    result = {}
    for h in range(24):
        pnls = by_hour.get(h, [])
        n = len(pnls)
        if n == 0:
            result[h] = {'n': 0, 'total_pnl': 0, 'avg_pnl': 0, 'win_rate': 0}
        else:
            result[h] = {
                'n': n,
                'total_pnl': round(sum(pnls), 2),
                'avg_pnl': round(sum(pnls) / n, 4),
                'win_rate': round(100 * sum(1 for p in pnls if p > 0) / n, 2),
            }
    return result


def main():
    t_start = time.time()
    print('=' * 80)
    print('R210: Entry Hour Filter Analysis (Keltner)')
    print('=' * 80)

    # ─── Load data ─────────────────────────────────────────────
    data = DataBundle.load_default()

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Per-Hour PnL Analysis
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 1: Per-Hour PnL Analysis')
    print('=' * 80)

    print('\n  Running baseline Keltner backtest...')
    all_trades, eq_curve = run_keltner(data, label="R210_baseline")
    bl_stats = calc_stats(all_trades, eq_curve)
    print(f'  Baseline: n={bl_stats["n"]}  PnL=${bl_stats["total_pnl"]:.0f}  '
          f'Sharpe={bl_stats["sharpe"]:.3f}  WR={bl_stats["win_rate"]:.1f}%')

    # Full-period hour matrix
    print('\n  Full period per-hour breakdown:')
    full_matrix = hour_pnl_matrix(all_trades)
    print(f'  {"Hour":>4} {"N":>6} {"TotalPnL":>12} {"AvgPnL":>10} {"WR%":>6}')
    print(f'  {"-"*4} {"-"*6} {"-"*12} {"-"*10} {"-"*6}')
    for h in range(24):
        r = full_matrix[h]
        print(f'  {h:>4} {r["n"]:>6} ${r["total_pnl"]:>10.2f} ${r["avg_pnl"]:>9.4f} {r["win_rate"]:>5.1f}%')

    # 2022-2026 subset
    recent_trades = [t for t in all_trades
                     if pd.Timestamp(t.entry_time) >= pd.Timestamp("2022-01-01", tz="UTC")]
    recent_matrix = hour_pnl_matrix(recent_trades)
    print(f'\n  Recent (2022-2026) per-hour breakdown:')
    print(f'  {"Hour":>4} {"N":>6} {"TotalPnL":>12} {"AvgPnL":>10} {"WR%":>6}')
    print(f'  {"-"*4} {"-"*6} {"-"*12} {"-"*10} {"-"*6}')
    for h in range(24):
        r = recent_matrix[h]
        print(f'  {h:>4} {r["n"]:>6} ${r["total_pnl"]:>10.2f} ${r["avg_pnl"]:>9.4f} {r["win_rate"]:>5.1f}%')

    save('phase1_hour_matrix', {
        'baseline_stats': {k: v for k, v in bl_stats.items() if not k.startswith('_')},
        'full_period': {str(k): v for k, v in full_matrix.items()},
        'recent_2022_2026': {str(k): v for k, v in recent_matrix.items()},
    })

    # Identify worst hours (negative avg_pnl in full period)
    hours_ranked = sorted(full_matrix.items(), key=lambda x: x[1]['avg_pnl'])
    worst_hours = [h for h, r in hours_ranked if r['avg_pnl'] < 0 and r['n'] >= 5]
    print(f'\n  Hours with negative avg_pnl (n>=5): {worst_hours}')
    if len(worst_hours) < 2:
        worst_hours_full = [h for h, _ in hours_ranked[:3]]
        print(f'  (Too few negative hours; using bottom-3: {worst_hours_full})')
    else:
        worst_hours_full = worst_hours

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Hour Block Sweep
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 2: Hour Block Sweep')
    print('=' * 80)

    bl_sharpe = bl_stats['sharpe']
    bl_pnl = bl_stats['total_pnl']
    bl_n = bl_stats['n']

    # Single-hour blocks
    print(f'\n  Single-hour blocks (vs baseline Sharpe={bl_sharpe:.3f}):')
    print(f'  {"Blocked":>8} {"N":>6} {"PnL":>12} {"Sharpe":>8} {"dSharpe":>8} {"dPnL":>10} {"WR%":>6}')
    single_results = []
    for h in range(24):
        filtered = filter_trades_by_hour(all_trades, {h})
        st = trade_stats(filtered)
        delta_s = st['sharpe'] - bl_sharpe
        delta_p = st['total_pnl'] - bl_pnl
        single_results.append({
            'blocked_hour': h, **st,
            'delta_sharpe': round(delta_s, 3),
            'delta_pnl': round(delta_p, 2),
            'removed_n': bl_n - st['n'],
        })
        tag = ' <<<' if delta_s > 0.1 else ''
        print(f'  {h:>8} {st["n"]:>6} ${st["total_pnl"]:>10.0f} {st["sharpe"]:>8.3f} '
              f'{delta_s:>+7.3f} ${delta_p:>+9.0f} {st["win_rate"]:>5.1f}%{tag}')

    # Sort by sharpe improvement
    single_sorted = sorted(single_results, key=lambda x: x['delta_sharpe'], reverse=True)
    best_singles = [r['blocked_hour'] for r in single_sorted if r['delta_sharpe'] > 0]
    print(f'\n  Hours that improve Sharpe when blocked: {best_singles[:8]}')

    # Multi-hour combos: worst-2, worst-3
    worst_1 = [hours_ranked[0][0]] if hours_ranked else []
    worst_2 = [h for h, _ in hours_ranked[:2]]
    worst_3 = [h for h, _ in hours_ranked[:3]]

    best_single_hours = [r['blocked_hour'] for r in single_sorted[:3]]
    best_2 = best_single_hours[:2]
    best_3 = best_single_hours[:3]

    combos = [
        ('worst_1', set(worst_1)),
        ('worst_2', set(worst_2)),
        ('worst_3', set(worst_3)),
        ('best_single_1', {best_single_hours[0]} if best_single_hours else set()),
        ('best_single_2', set(best_2)),
        ('best_single_3', set(best_3)),
    ]

    # Also test all 2-hour combos from top-5 improving hours
    top5_improving = [r['blocked_hour'] for r in single_sorted[:5] if r['delta_sharpe'] > 0]
    for pair in itertools.combinations(top5_improving, 2):
        combos.append((f'block_{pair[0]}_{pair[1]}', set(pair)))

    print(f'\n  Multi-hour combos:')
    print(f'  {"Label":<25} {"Hours":<20} {"N":>6} {"PnL":>12} {"Sharpe":>8} '
          f'{"dSharpe":>8} {"dPnL":>10}')
    combo_results = []
    for label, hours in combos:
        if not hours:
            continue
        filtered = filter_trades_by_hour(all_trades, hours)
        st = trade_stats(filtered)
        delta_s = st['sharpe'] - bl_sharpe
        delta_p = st['total_pnl'] - bl_pnl
        row = {
            'label': label, 'blocked_hours': sorted(hours), **st,
            'delta_sharpe': round(delta_s, 3),
            'delta_pnl': round(delta_p, 2),
        }
        combo_results.append(row)
        print(f'  {label:<25} {str(sorted(hours)):<20} {st["n"]:>6} ${st["total_pnl"]:>10.0f} '
              f'{st["sharpe"]:>8.3f} {delta_s:>+7.3f} ${delta_p:>+9.0f}')

    # Combine all and rank
    all_candidates = single_results + combo_results
    all_candidates.sort(key=lambda x: x.get('sharpe', 0), reverse=True)

    save('phase2_hour_sweep', {
        'baseline': {'n': bl_n, 'pnl': bl_pnl, 'sharpe': bl_sharpe},
        'single_hour_blocks': single_results,
        'combo_blocks': combo_results,
        'worst_hours_by_avg_pnl': [h for h, _ in hours_ranked],
    })

    # Select top 3 candidates for 3-gate
    top3_for_gate = []
    seen_labels = set()
    for c in all_candidates:
        lbl = c.get('label', f'block_{c.get("blocked_hour", "?")}')
        if lbl in seen_labels:
            continue
        if c.get('delta_sharpe', 0) <= 0:
            continue
        seen_labels.add(lbl)
        hours = c.get('blocked_hours', [c['blocked_hour']] if 'blocked_hour' in c else [])
        top3_for_gate.append({
            'label': lbl,
            'blocked_hours': set(hours) if isinstance(hours, list) else hours,
        })
        if len(top3_for_gate) >= 3:
            break

    if not top3_for_gate:
        print('\n  WARNING: No hour filter improves Sharpe. Skipping Phase 3.')
        summary = {
            'verdict': 'NO-FILTER',
            'reason': 'No single-hour or combo block improves Sharpe over baseline',
            'baseline': {'n': bl_n, 'pnl': bl_pnl, 'sharpe': bl_sharpe},
        }
        save('R210_summary', summary)
        elapsed = time.time() - t_start
        print(f'\n  Total runtime: {elapsed:.0f}s')
        return

    print(f'\n  Top candidates for 3-Gate:')
    for c in top3_for_gate:
        print(f'    {c["label"]}: blocked_hours={sorted(c["blocked_hours"])}')

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: 3-Gate Validation
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 3: 3-Gate Validation')
    print('=' * 80)

    gate_results = {}

    for candidate in top3_for_gate:
        label = candidate['label']
        blocked = candidate['blocked_hours']
        print(f'\n  === {label} (block hours: {sorted(blocked)}) ===')

        # Gate 1: 6-Fold CV
        print('    Gate 1: 6-Fold CV...')
        folds = [
            ("Fold1", "2015-01-01", "2017-01-01"),
            ("Fold2", "2017-01-01", "2019-01-01"),
            ("Fold3", "2019-01-01", "2021-01-01"),
            ("Fold4", "2021-01-01", "2023-01-01"),
            ("Fold5", "2023-01-01", "2025-01-01"),
            ("Fold6", "2025-01-01", "2026-04-01"),
        ]
        kf_wins = 0
        kf_details = []
        for fold_name, fs, fe in folds:
            fold_data = data.slice(fs, fe)
            if len(fold_data.m15_df) < 1000:
                continue
            trades_fold, eq_fold = run_keltner(fold_data, label=f"{label}_{fold_name}")
            bl_st = trade_stats(trades_fold)
            filtered_fold = filter_trades_by_hour(trades_fold, blocked)
            filt_st = trade_stats(filtered_fold)

            win = filt_st['sharpe'] >= bl_st['sharpe']
            if win:
                kf_wins += 1
            kf_details.append({
                'fold': fold_name, 'bl_sharpe': bl_st['sharpe'],
                'filt_sharpe': filt_st['sharpe'],
                'bl_n': bl_st['n'], 'filt_n': filt_st['n'], 'win': win,
            })
            print(f'      {fold_name}: filt={filt_st["sharpe"]:.3f} bl={bl_st["sharpe"]:.3f} '
                  f'n={filt_st["n"]}/{bl_st["n"]} {"WIN" if win else ""}')

        kf_total = len(kf_details)
        kf_pass = kf_wins >= 4
        print(f'    KF: {kf_wins}/{kf_total}  PASS={kf_pass}')

        # Gate 2: Walk-Forward
        print('    Gate 2: Walk-Forward...')
        wf_wins = 0
        wf_details = []
        for train_s, train_e, test_s, test_e in WF_WINDOWS:
            wf_data = data.slice(test_s, test_e)
            if len(wf_data.m15_df) < 500:
                continue
            trades_wf, _ = run_keltner(wf_data, label=f"{label}_WF")
            bl_st = trade_stats(trades_wf)
            filtered_wf = filter_trades_by_hour(trades_wf, blocked)
            filt_st = trade_stats(filtered_wf)

            win = filt_st['sharpe'] >= bl_st['sharpe']
            if win:
                wf_wins += 1
            wf_details.append({
                'test_window': f'{test_s}_{test_e}',
                'bl_sharpe': bl_st['sharpe'], 'filt_sharpe': filt_st['sharpe'],
                'win': win,
            })

        wf_total = len(wf_details)
        wf_pass = wf_wins >= wf_total * 0.6
        print(f'    WF: {wf_wins}/{wf_total}  PASS={wf_pass}')

        # Gate 3: Era Stability
        print('    Gate 3: Era Stability...')
        era_details = []
        for era_name, (es, ee) in ERA_SEGMENTS.items():
            era_data = data.slice(es, ee)
            if len(era_data.m15_df) < 1000:
                continue
            trades_era, eq_era = run_keltner(era_data, label=f"{label}_{era_name[:8]}")
            bl_st = trade_stats(trades_era)
            filtered_era = filter_trades_by_hour(trades_era, blocked)
            filt_st = trade_stats(filtered_era)
            delta = filt_st['sharpe'] - bl_st['sharpe']
            era_details.append({
                'era': era_name,
                'bl_sharpe': bl_st['sharpe'], 'filt_sharpe': filt_st['sharpe'],
                'delta': round(delta, 3),
                'bl_n': bl_st['n'], 'filt_n': filt_st['n'],
            })
            print(f'      {era_name}: filt={filt_st["sharpe"]:.3f} bl={bl_st["sharpe"]:.3f} '
                  f'delta={delta:+.3f}  n={filt_st["n"]}/{bl_st["n"]}')

        era_sharpes = [e['filt_sharpe'] for e in era_details]
        era_pass = len(era_sharpes) >= 3 and all(s > 0 for s in era_sharpes) and min(era_sharpes) > 0.5
        print(f'    Era: min_sharpe={min(era_sharpes):.3f}  PASS={era_pass}')

        overall = kf_pass and wf_pass and era_pass
        gate_results[label] = {
            'blocked_hours': sorted(blocked),
            'kfold': {'wins': kf_wins, 'total': kf_total, 'pass': kf_pass, 'details': kf_details},
            'walk_forward': {'wins': wf_wins, 'total': wf_total, 'pass': wf_pass, 'details': wf_details},
            'era': {'results': era_details, 'pass': era_pass},
            'overall_pass': overall,
        }
        tag = '[GO]' if overall else '[NO-GO]'
        print(f'    Overall: {tag}')

    save('phase3_three_gate', gate_results)

    # ═══════════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('FINAL SUMMARY')
    print('=' * 80)

    go_candidates = [k for k, v in gate_results.items() if v['overall_pass']]
    summary = {
        'baseline': {
            'n': bl_stats['n'], 'pnl': bl_stats['total_pnl'],
            'sharpe': bl_stats['sharpe'], 'win_rate': bl_stats['win_rate'],
        },
        'hour_analysis': {
            'worst_hours_by_avg_pnl': [h for h, _ in hours_ranked[:5]],
            'hours_improving_sharpe_when_blocked': best_singles[:5],
        },
        'candidates_tested': {
            k: {
                'blocked_hours': v['blocked_hours'],
                'kf': f'{v["kfold"]["wins"]}/{v["kfold"]["total"]}',
                'wf': f'{v["walk_forward"]["wins"]}/{v["walk_forward"]["total"]}',
                'era_pass': v['era']['pass'],
                'overall': v['overall_pass'],
            }
            for k, v in gate_results.items()
        },
    }

    if go_candidates:
        summary['verdict'] = 'GO'
        summary['recommended'] = go_candidates
        print(f'  VERDICT: GO — Hour filter(s) validated')
        for gc in go_candidates:
            g = gate_results[gc]
            print(f'    {gc}: hours={g["blocked_hours"]}  '
                  f'KF={g["kfold"]["wins"]}/{g["kfold"]["total"]}  '
                  f'WF={g["walk_forward"]["wins"]}/{g["walk_forward"]["total"]}  '
                  f'Era={g["era"]["pass"]}')
    else:
        summary['verdict'] = 'NO-GO'
        summary['reason'] = 'Hour filters failed 3-gate validation; keep all hours active'
        print(f'  VERDICT: NO-GO — No hour filter passes 3-gate validation')
        print(f'  Recommendation: Keep all hours active in production')

    save('R210_summary', summary)

    elapsed = time.time() - t_start
    print(f'\n  Total runtime: {elapsed:.0f}s')
    print(f'  All results in {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
