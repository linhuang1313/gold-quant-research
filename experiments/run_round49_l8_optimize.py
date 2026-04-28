#!/usr/bin/env python3
"""
Round 49 — L8_BASE Overlay Validation
======================================
Systematically re-validate optimization layers discovered in R31-R36 (on L7)
against the current live strategy L8_BASE+Cap80.

Phase 1: Single-layer marginal tests (vs naked L8_BASE)
  P1-1: H1 KC same-direction filter (EMA15/20/25 x Mult1.2/1.5/2.0)
  P1-2: EqCurve (LB 5/10/20/30, Cut 0/-1, Red 0/0.5)
  P1-3: MaxHold sweep (8/12/16/20/30)
  P1-4: Session filter (skip 22-1 UTC / skip Asian)
  P1-5: TATrail ON vs OFF under L8 tight trail
  P1-6: Cap sweep (30/50/80/120/OFF)

Phase 2: K-Fold 6-Fold for layers with positive delta

Phase 3: Best combo L8_MAX stress test + final comparison

Usage:
  python -m experiments.run_round49_l8_optimize
"""
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from backtest.runner import DataBundle, run_variant, run_kfold, LIVE_PARITY_KWARGS
from backtest.engine import TradeRecord
from backtest.stats import calc_stats

EXPERIMENT_NAME = "round49_l8_optimize"
OUT_DIR = ROOT / "results" / EXPERIMENT_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPREAD = 0.50

L8_BASE = {
    **LIVE_PARITY_KWARGS,
    'keltner_adx_threshold': 14,
    'regime_config': {
        'low':    {'trail_act': 0.22, 'trail_dist': 0.04},
        'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
        'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
    },
    'min_entry_gap_hours': 1.0,
    'keltner_max_hold_m15': 20,
    'spread_cost': SPREAD,
}


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


# ═══════════════════════════════════════════════════════════════
# H1 KC Direction Helpers (from R32)
# ═══════════════════════════════════════════════════════════════

def add_h1_kc_dir(h1_df: pd.DataFrame, ema_period: int = 20, mult: float = 2.0) -> pd.DataFrame:
    h1 = h1_df.copy()
    h1['EMA_kc'] = h1['Close'].ewm(span=ema_period, adjust=False).mean()
    tr = pd.DataFrame({
        'hl': h1['High'] - h1['Low'],
        'hc': (h1['High'] - h1['Close'].shift(1)).abs(),
        'lc': (h1['Low'] - h1['Close'].shift(1)).abs(),
    }).max(axis=1)
    h1['ATR_kc'] = tr.rolling(14).mean()
    h1['KC_U'] = h1['EMA_kc'] + mult * h1['ATR_kc']
    h1['KC_L'] = h1['EMA_kc'] - mult * h1['ATR_kc']
    h1['kc_dir'] = 'NEUTRAL'
    h1.loc[h1['Close'] > h1['KC_U'], 'kc_dir'] = 'BULL'
    h1.loc[h1['Close'] < h1['KC_L'], 'kc_dir'] = 'BEAR'
    return h1


def filter_trades_by_h1_kc(trades: List, h1_kc: pd.DataFrame) -> Tuple[List, int]:
    kept = []
    skipped = 0
    for t in trades:
        et = t.entry_time if hasattr(t, 'entry_time') else t['entry_time']
        td = t.direction if hasattr(t, 'direction') else t.get('dir', '')
        et_ts = pd.Timestamp(et)
        h1_mask = h1_kc.index <= et_ts
        if not h1_mask.any():
            skipped += 1
            continue
        kc_d = h1_kc.loc[h1_kc.index[h1_mask][-1], 'kc_dir']
        if (td == 'BUY' and kc_d == 'BULL') or (td == 'SELL' and kc_d == 'BEAR'):
            kept.append(t)
        else:
            skipped += 1
    return kept, skipped


def stats_from_trades(trades: List, label: str = "") -> Dict:
    if not trades:
        return {'n': 0, 'total_pnl': 0, 'sharpe': 0, 'win_rate': 0, 'max_dd': 0}
    pnls = [t.pnl for t in trades]
    daily: Dict = {}
    for t in trades:
        d = pd.Timestamp(t.exit_time).date()
        daily[d] = daily.get(d, 0) + t.pnl
    daily_pnls = list(daily.values())
    sharpe = (np.mean(daily_pnls) / np.std(daily_pnls, ddof=1) * np.sqrt(252)
              if len(daily_pnls) > 1 and np.std(daily_pnls, ddof=1) > 0 else 0)
    wins = [p for p in pnls if p > 0]
    wr = len(wins) / len(pnls) * 100 if pnls else 0
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0
    return {
        'label': label,
        'n': len(trades),
        'total_pnl': sum(pnls),
        'sharpe': round(sharpe, 2),
        'win_rate': round(wr, 1),
        'max_dd': round(max_dd, 2),
    }


def apply_cap(trades: List, cap_usd: float) -> List:
    """Simulate MaxLoss cap: cap individual trade loss at -cap_usd."""
    if cap_usd <= 0:
        return trades
    capped = []
    for t in trades:
        if t.pnl < -cap_usd:
            capped.append(TradeRecord(
                entry_time=t.entry_time, exit_time=t.exit_time,
                direction=t.direction, entry_price=t.entry_price,
                exit_price=t.exit_price, pnl=-cap_usd,
                exit_reason=f"MaxLossCap${cap_usd}",
                strategy=t.strategy, bars_held=t.bars_held,
            ))
        else:
            capped.append(t)
    return capped


# ═══════════════════════════════════════════════════════════════
# Phase 1: Single-Layer Marginal Tests
# ═══════════════════════════════════════════════════════════════

def phase_1(data: DataBundle) -> Dict:
    print("\n" + "=" * 80)
    print("  PHASE 1: Single-Layer Marginal Validation (vs naked L8_BASE)")
    print(f"  Spread: ${SPREAD}")
    print("=" * 80)

    t0 = time.time()

    # --- Baseline: naked L8_BASE ---
    print("\n  [Baseline] L8_BASE naked...")
    base = run_variant(data, "L8_BASE_naked", **L8_BASE)
    base_trades = base['_trades']
    base_sh = base['sharpe']
    base_pnl = base['total_pnl']
    print(f"    Sharpe={base_sh:.2f}, PnL=${base_pnl:.0f}, "
          f"N={base['n']}, MaxDD=${base['max_dd']:.0f}")

    results = {
        'baseline': {
            'sharpe': base_sh, 'pnl': base_pnl,
            'n': base['n'], 'max_dd': base['max_dd'],
        },
        'tests': {},
    }

    # --- P1-1: H1 KC Same-Direction Filter ---
    print("\n  --- P1-1: H1 KC Same-Direction Filter ---")
    print(f"  {'Config':>20} {'Sharpe':>8} {'Delta':>7} {'N':>6} {'PnL':>10} {'MaxDD':>8} {'Skipped':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*7} {'-'*6} {'-'*10} {'-'*8} {'-'*8}")

    h1kc_results = []
    for ema_p in [15, 20, 25]:
        for mult in [1.2, 1.5, 2.0]:
            tag = f"E{ema_p}_M{mult}"
            h1_kc = add_h1_kc_dir(data.h1_df, ema_period=ema_p, mult=mult)
            kept, skipped = filter_trades_by_h1_kc(base_trades, h1_kc)
            st = stats_from_trades(kept, tag)
            delta = st['sharpe'] - base_sh
            h1kc_results.append({**st, 'ema': ema_p, 'mult': mult, 'delta': delta, 'skipped': skipped})
            print(f"  {tag:>20} {st['sharpe']:>8.2f} {delta:>+7.2f} "
                  f"{st['n']:>6} ${st['total_pnl']:>9.0f} ${st['max_dd']:>7.0f} {skipped:>8}")

    best_h1kc = max(h1kc_results, key=lambda x: x['sharpe'])
    results['tests']['h1_kc_filter'] = {
        'all': h1kc_results,
        'best': best_h1kc,
        'effective': best_h1kc['delta'] > 0,
    }

    # --- P1-2: EqCurve ---
    print("\n  --- P1-2: EqCurve Filter ---")
    print(f"  {'Config':>20} {'Sharpe':>8} {'Delta':>7} {'N':>6} {'PnL':>10} {'MaxDD':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*7} {'-'*6} {'-'*10} {'-'*8}")

    eq_results = []
    for lb in [5, 10, 20, 30]:
        for cut in [0, -1]:
            for red in [0.0, 0.5]:
                tag = f"LB{lb}_C{cut}_R{red}"
                kw = {**L8_BASE,
                      'equity_curve_filter': True,
                      'equity_ma_period': lb}
                st = run_variant(data, f"EQ_{tag}", verbose=False, **kw)
                delta = st['sharpe'] - base_sh
                eq_results.append({
                    'label': tag, 'lb': lb, 'cut': cut, 'red': red,
                    'sharpe': st['sharpe'], 'pnl': st['total_pnl'],
                    'n': st['n'], 'max_dd': st['max_dd'], 'delta': delta,
                })
                print(f"  {tag:>20} {st['sharpe']:>8.2f} {delta:>+7.2f} "
                      f"{st['n']:>6} ${st['total_pnl']:>9.0f} ${st['max_dd']:>7.0f}")

    best_eq = max(eq_results, key=lambda x: x['sharpe'])
    results['tests']['eqcurve'] = {
        'all': eq_results, 'best': best_eq, 'effective': best_eq['delta'] > 0,
    }

    # --- P1-3: MaxHold Sweep ---
    print("\n  --- P1-3: MaxHold Sweep ---")
    print(f"  {'MH':>6} {'Sharpe':>8} {'Delta':>7} {'N':>6} {'PnL':>10} {'MaxDD':>8}")
    print(f"  {'-'*6} {'-'*8} {'-'*7} {'-'*6} {'-'*10} {'-'*8}")

    mh_results = []
    for mh in [8, 12, 16, 20, 30]:
        kw = {**L8_BASE, 'keltner_max_hold_m15': mh}
        st = run_variant(data, f"MH{mh}", verbose=False, **kw)
        delta = st['sharpe'] - base_sh
        mh_results.append({
            'mh': mh, 'sharpe': st['sharpe'], 'pnl': st['total_pnl'],
            'n': st['n'], 'max_dd': st['max_dd'], 'delta': delta,
        })
        print(f"  {mh:>6} {st['sharpe']:>8.2f} {delta:>+7.2f} "
              f"{st['n']:>6} ${st['total_pnl']:>9.0f} ${st['max_dd']:>7.0f}")

    best_mh = max(mh_results, key=lambda x: x['sharpe'])
    results['tests']['maxhold'] = {
        'all': mh_results, 'best': best_mh, 'effective': best_mh['delta'] > 0,
    }

    # --- P1-4: Session Filter ---
    print("\n  --- P1-4: Session Filter ---")
    print(f"  {'Filter':>20} {'Sharpe':>8} {'Delta':>7} {'N':>6} {'PnL':>10} {'MaxDD':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*7} {'-'*6} {'-'*10} {'-'*8}")

    sess_results = []
    session_configs = {
        'NoFilter': None,
        'Skip22-1UTC': list(range(1, 22)),
        'SkipAsian': [i for i in range(24) if i not in range(22, 24) and i not in range(0, 5)],
        'LDN+NY_only': list(range(7, 21)),
    }
    for name, hours in session_configs.items():
        kw = {**L8_BASE}
        if hours is not None:
            kw['h1_allowed_sessions'] = hours
        st = run_variant(data, f"Sess_{name}", verbose=False, **kw)
        delta = st['sharpe'] - base_sh
        sess_results.append({
            'label': name, 'sharpe': st['sharpe'], 'pnl': st['total_pnl'],
            'n': st['n'], 'max_dd': st['max_dd'], 'delta': delta,
        })
        print(f"  {name:>20} {st['sharpe']:>8.2f} {delta:>+7.2f} "
              f"{st['n']:>6} ${st['total_pnl']:>9.0f} ${st['max_dd']:>7.0f}")

    best_sess = max(sess_results, key=lambda x: x['sharpe'])
    results['tests']['session'] = {
        'all': sess_results, 'best': best_sess,
        'effective': best_sess['delta'] > 0 and best_sess['label'] != 'NoFilter',
    }

    # --- P1-5: TATrail ---
    print("\n  --- P1-5: TATrail (ON vs OFF under L8 tight trail) ---")
    print(f"  {'Config':>20} {'Sharpe':>8} {'Delta':>7} {'N':>6} {'PnL':>10} {'MaxDD':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*7} {'-'*6} {'-'*10} {'-'*8}")

    ta_results = []
    ta_configs = [
        ("OFF (baseline)", {}),
        ("s2/d0.75/f0.003", {'time_adaptive_trail': {'start': 2, 'decay': 0.75, 'floor': 0.003}}),
        ("s2/d0.80/f0.005", {'time_adaptive_trail': {'start': 2, 'decay': 0.80, 'floor': 0.005}}),
        ("s3/d0.70/f0.003", {'time_adaptive_trail': {'start': 3, 'decay': 0.70, 'floor': 0.003}}),
    ]
    for name, extra_kw in ta_configs:
        kw = {**L8_BASE, **extra_kw}
        st = run_variant(data, f"TA_{name}", verbose=False, **kw)
        delta = st['sharpe'] - base_sh
        ta_results.append({
            'label': name, 'sharpe': st['sharpe'], 'pnl': st['total_pnl'],
            'n': st['n'], 'max_dd': st['max_dd'], 'delta': delta,
        })
        print(f"  {name:>20} {st['sharpe']:>8.2f} {delta:>+7.2f} "
              f"{st['n']:>6} ${st['total_pnl']:>9.0f} ${st['max_dd']:>7.0f}")

    best_ta = max(ta_results, key=lambda x: x['sharpe'])
    results['tests']['tatrail'] = {
        'all': ta_results, 'best': best_ta,
        'effective': best_ta['delta'] > 0 and best_ta['label'] != 'OFF (baseline)',
    }

    # --- P1-6: Cap Sweep ---
    print("\n  --- P1-6: MaxLoss Cap Sweep ---")
    print(f"  {'Cap':>8} {'Sharpe':>8} {'Delta':>7} {'N':>6} {'PnL':>10} {'MaxDD':>8} {'Capped':>7}")
    print(f"  {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*10} {'-'*8} {'-'*7}")

    cap_results = []
    for cap in [30, 50, 80, 120, 0]:
        cap_label = f"${cap}" if cap > 0 else "OFF"
        capped_trades = apply_cap(base_trades, cap) if cap > 0 else base_trades
        n_capped = sum(1 for a, b in zip(base_trades, capped_trades) if a.pnl != b.pnl) if cap > 0 else 0
        st = stats_from_trades(capped_trades, cap_label)
        delta = st['sharpe'] - base_sh
        cap_results.append({
            'cap': cap, 'label': cap_label, 'sharpe': st['sharpe'],
            'pnl': st['total_pnl'], 'n': st['n'], 'max_dd': st['max_dd'],
            'delta': delta, 'n_capped': n_capped,
        })
        print(f"  {cap_label:>8} {st['sharpe']:>8.2f} {delta:>+7.2f} "
              f"{st['n']:>6} ${st['total_pnl']:>9.0f} ${st['max_dd']:>7.0f} {n_capped:>7}")

    best_cap = max(cap_results, key=lambda x: x['sharpe'])
    results['tests']['cap'] = {
        'all': cap_results, 'best': best_cap, 'effective': best_cap['delta'] > 0,
    }

    elapsed = time.time() - t0

    # Summary
    print(f"\n  {'='*60}")
    print(f"  Phase 1 Summary (elapsed: {elapsed:.0f}s)")
    print(f"  {'='*60}")
    print(f"  {'Layer':>20} {'Best Config':>20} {'Sharpe':>8} {'Delta':>7} {'Effective':>10}")
    print(f"  {'-'*20} {'-'*20} {'-'*8} {'-'*7} {'-'*10}")
    for layer_name, layer_data in results['tests'].items():
        b = layer_data['best']
        lbl = b.get('label', str(b.get('mh', b.get('cap', ''))))
        eff = "YES" if layer_data['effective'] else "no"
        print(f"  {layer_name:>20} {lbl:>20} {b['sharpe']:>8.2f} {b.get('delta',0):>+7.2f} {eff:>10}")

    with open(OUT_DIR / "phase1_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {OUT_DIR / 'phase1_results.json'}")

    return results


# ═══════════════════════════════════════════════════════════════
# Phase 2: K-Fold Validation for effective layers
# ═══════════════════════════════════════════════════════════════

def phase_2(data: DataBundle, p1: Dict) -> Dict:
    print("\n" + "=" * 80)
    print("  PHASE 2: K-Fold 6-Fold Validation (layers with positive delta)")
    print("=" * 80)

    t0 = time.time()
    kfold_results = {}
    base_sh = p1['baseline']['sharpe']

    folds = [
        ("Fold1", "2015-01-01", "2017-01-01"),
        ("Fold2", "2017-01-01", "2019-01-01"),
        ("Fold3", "2019-01-01", "2021-01-01"),
        ("Fold4", "2021-01-01", "2023-01-01"),
        ("Fold5", "2023-01-01", "2025-01-01"),
        ("Fold6", "2025-01-01", "2026-04-01"),
    ]

    # H1 KC filter — needs per-fold manual filtering
    h1kc_test = p1['tests'].get('h1_kc_filter', {})
    if h1kc_test.get('effective'):
        best = h1kc_test['best']
        ema_p, mult = best['ema'], best['mult']
        tag = f"H1KC_E{ema_p}_M{mult}"
        print(f"\n  --- K-Fold: {tag} ---")
        print(f"  {'Fold':>6} {'Base':>8} {'Filter':>8} {'Delta':>7} {'N_base':>7} {'N_filt':>7}")

        h1_kc = add_h1_kc_dir(data.h1_df, ema_period=ema_p, mult=mult)
        fold_sharpes = []
        fold_deltas = []

        for fname, start, end in folds:
            fold_data = data.slice(start, end)
            if len(fold_data.m15_df) < 1000:
                continue
            base_stats = run_variant(fold_data, f"base_{fname}", verbose=False, **L8_BASE)
            base_trades = base_stats['_trades']
            kept, _ = filter_trades_by_h1_kc(base_trades, h1_kc)
            filt_st = stats_from_trades(kept)
            delta = filt_st['sharpe'] - base_stats['sharpe']
            fold_sharpes.append(filt_st['sharpe'])
            fold_deltas.append(delta)
            print(f"  {fname:>6} {base_stats['sharpe']:>8.2f} {filt_st['sharpe']:>8.2f} "
                  f"{delta:>+7.2f} {base_stats['n']:>7} {filt_st['n']:>7}")

        n_pass = sum(1 for d in fold_deltas if d > 0)
        kfold_results['h1_kc_filter'] = {
            'config': {'ema': ema_p, 'mult': mult},
            'fold_sharpes': fold_sharpes,
            'fold_deltas': fold_deltas,
            'mean_delta': round(np.mean(fold_deltas), 2) if fold_deltas else 0,
            'pass': f"{n_pass}/{len(fold_deltas)}",
            'all_positive': all(d > 0 for d in fold_deltas),
        }
        print(f"  K-Fold: {n_pass}/{len(fold_deltas)} positive delta, "
              f"mean delta={np.mean(fold_deltas):+.2f}")

    # Engine-based layers: EqCurve, MaxHold, Session, TATrail
    engine_layers = {
        'eqcurve': lambda best: {
            **L8_BASE,
            'equity_curve_filter': True,
            'equity_ma_period': best['lb'],
        },
        'maxhold': lambda best: {
            **L8_BASE,
            'keltner_max_hold_m15': best['mh'],
        },
        'session': lambda best: {
            **L8_BASE,
            'h1_allowed_sessions': session_configs_map().get(best['label']),
        } if best['label'] != 'NoFilter' else L8_BASE,
        'tatrail': lambda best: {
            **L8_BASE,
            **tatrail_configs_map().get(best['label'], {}),
        },
    }

    for layer_name, kw_fn in engine_layers.items():
        layer_test = p1['tests'].get(layer_name, {})
        if not layer_test.get('effective'):
            print(f"\n  --- K-Fold: {layer_name} — SKIPPED (not effective) ---")
            continue

        best = layer_test['best']
        lbl = best.get('label', str(best.get('mh', '')))
        print(f"\n  --- K-Fold: {layer_name} ({lbl}) ---")
        print(f"  {'Fold':>6} {'Base':>8} {'Layer':>8} {'Delta':>7}")

        kw = kw_fn(best)
        fold_results = run_kfold(data, kw, label_prefix=f"KF_{layer_name}_")
        base_folds = run_kfold(data, L8_BASE, label_prefix=f"KFbase_{layer_name}_")

        fold_deltas = []
        for bf, lf in zip(base_folds, fold_results):
            delta = lf['sharpe'] - bf['sharpe']
            fold_deltas.append(delta)
            print(f"  {bf['fold']:>6} {bf['sharpe']:>8.2f} {lf['sharpe']:>8.2f} {delta:>+7.2f}")

        n_pass = sum(1 for d in fold_deltas if d > 0)
        kfold_results[layer_name] = {
            'config': lbl,
            'fold_sharpes': [r['sharpe'] for r in fold_results],
            'fold_deltas': fold_deltas,
            'mean_delta': round(np.mean(fold_deltas), 2) if fold_deltas else 0,
            'pass': f"{n_pass}/{len(fold_deltas)}",
            'all_positive': all(d > 0 for d in fold_deltas),
        }
        print(f"  K-Fold: {n_pass}/{len(fold_deltas)} positive delta, "
              f"mean delta={np.mean(fold_deltas):+.2f}")

    # Cap — use trade-level filtering per fold
    cap_test = p1['tests'].get('cap', {})
    if cap_test.get('effective'):
        best_cap = cap_test['best']['cap']
        print(f"\n  --- K-Fold: Cap ${best_cap} ---")
        print(f"  {'Fold':>6} {'Base':>8} {'Capped':>8} {'Delta':>7}")

        fold_deltas = []
        for fname, start, end in folds:
            fold_data = data.slice(start, end)
            if len(fold_data.m15_df) < 1000:
                continue
            base_stats = run_variant(fold_data, f"capbase_{fname}", verbose=False, **L8_BASE)
            base_trades = base_stats['_trades']
            capped = apply_cap(base_trades, best_cap)
            cap_st = stats_from_trades(capped)
            delta = cap_st['sharpe'] - base_stats['sharpe']
            fold_deltas.append(delta)
            print(f"  {fname:>6} {base_stats['sharpe']:>8.2f} {cap_st['sharpe']:>8.2f} {delta:>+7.2f}")

        n_pass = sum(1 for d in fold_deltas if d > 0)
        kfold_results['cap'] = {
            'config': f"${best_cap}",
            'fold_deltas': fold_deltas,
            'mean_delta': round(np.mean(fold_deltas), 2) if fold_deltas else 0,
            'pass': f"{n_pass}/{len(fold_deltas)}",
            'all_positive': all(d > 0 for d in fold_deltas),
        }
        print(f"  K-Fold: {n_pass}/{len(fold_deltas)} positive delta, "
              f"mean delta={np.mean(fold_deltas):+.2f}")

    elapsed = time.time() - t0

    # Summary
    print(f"\n  {'='*60}")
    print(f"  Phase 2 Summary (elapsed: {elapsed:.0f}s)")
    print(f"  {'='*60}")
    print(f"  {'Layer':>20} {'Config':>20} {'Pass':>8} {'MeanDelta':>10} {'AllPos':>8}")
    print(f"  {'-'*20} {'-'*20} {'-'*8} {'-'*10} {'-'*8}")
    for layer_name, kf in kfold_results.items():
        cfg = str(kf.get('config', ''))[:20]
        all_pos = "YES" if kf['all_positive'] else "no"
        print(f"  {layer_name:>20} {cfg:>20} {kf['pass']:>8} {kf['mean_delta']:>+10.2f} {all_pos:>8}")

    with open(OUT_DIR / "phase2_kfold.json", 'w') as f:
        json.dump(kfold_results, f, indent=2, default=str)
    print(f"\n  Saved: {OUT_DIR / 'phase2_kfold.json'}")

    return kfold_results


# ═══════════════════════════════════════════════════════════════
# Phase 3: Best Combo Stress Test
# ═══════════════════════════════════════════════════════════════

def phase_3(data: DataBundle, p1: Dict, p2: Dict) -> Dict:
    print("\n" + "=" * 80)
    print("  PHASE 3: L8_MAX Combo + Stress Test")
    print("=" * 80)

    t0 = time.time()

    passed_layers = [k for k, v in p2.items() if v.get('all_positive')]
    print(f"\n  Passed layers: {passed_layers if passed_layers else 'NONE'}")

    # Build L8_MAX: stack all passed engine layers on L8_BASE
    l8_max = {**L8_BASE}
    combo_desc = ["L8_BASE"]

    for layer in passed_layers:
        if layer == 'eqcurve':
            best = p1['tests']['eqcurve']['best']
            l8_max['equity_curve_filter'] = True
            l8_max['equity_ma_period'] = best['lb']
            combo_desc.append(f"EqCurve(LB={best['lb']})")
        elif layer == 'maxhold':
            best = p1['tests']['maxhold']['best']
            l8_max['keltner_max_hold_m15'] = best['mh']
            combo_desc.append(f"MH={best['mh']}")
        elif layer == 'session':
            best = p1['tests']['session']['best']
            hours = session_configs_map().get(best['label'])
            if hours:
                l8_max['h1_allowed_sessions'] = hours
                combo_desc.append(f"Session({best['label']})")
        elif layer == 'tatrail':
            best = p1['tests']['tatrail']['best']
            extra = tatrail_configs_map().get(best['label'], {})
            l8_max.update(extra)
            combo_desc.append(f"TATrail({best['label']})")

    combo_label = " + ".join(combo_desc)
    print(f"  L8_MAX = {combo_label}")

    # Run L8_MAX full sample
    max_stats = run_variant(data, "L8_MAX", **l8_max)
    max_trades = max_stats['_trades']
    print(f"\n  L8_MAX full sample: Sharpe={max_stats['sharpe']:.2f}, "
          f"PnL=${max_stats['total_pnl']:.0f}, N={max_stats['n']}, MaxDD=${max_stats['max_dd']:.0f}")

    # Apply H1 KC filter + Cap if passed
    h1kc_passed = 'h1_kc_filter' in passed_layers
    cap_passed = 'cap' in passed_layers

    final_trades = max_trades
    if h1kc_passed:
        cfg = p2['h1_kc_filter']['config']
        h1_kc = add_h1_kc_dir(data.h1_df, ema_period=cfg['ema'], mult=cfg['mult'])
        final_trades, skipped = filter_trades_by_h1_kc(final_trades, h1_kc)
        combo_desc.append(f"H1KC(E{cfg['ema']}/M{cfg['mult']})")
        st = stats_from_trades(final_trades)
        print(f"  + H1 KC filter: Sharpe={st['sharpe']:.2f}, N={st['n']} (skipped {skipped})")

    if cap_passed:
        best_cap = p1['tests']['cap']['best']['cap']
        final_trades = apply_cap(final_trades, best_cap)
        combo_desc.append(f"Cap${best_cap}")
        st = stats_from_trades(final_trades)
        print(f"  + Cap ${best_cap}: Sharpe={st['sharpe']:.2f}")

    final_st = stats_from_trades(final_trades, "L8_MAX_final")
    combo_label = " + ".join(combo_desc)
    print(f"\n  FINAL L8_MAX = {combo_label}")
    print(f"  Sharpe={final_st['sharpe']:.2f}, PnL=${final_st['total_pnl']:.0f}, "
          f"N={final_st['n']}, MaxDD=${final_st['max_dd']:.0f}")

    # Spread sensitivity
    print(f"\n  --- Spread Sensitivity ---")
    spread_results = []
    for sp in [0.30, 0.50, 0.80, 1.00]:
        kw = {**l8_max, 'spread_cost': sp}
        st = run_variant(data, f"L8MAX_sp{sp}", verbose=False, **kw)
        trades_sp = st['_trades']
        if h1kc_passed:
            trades_sp, _ = filter_trades_by_h1_kc(trades_sp, h1_kc)
        if cap_passed:
            trades_sp = apply_cap(trades_sp, best_cap)
        sp_st = stats_from_trades(trades_sp)
        spread_results.append({'spread': sp, 'sharpe': sp_st['sharpe'], 'pnl': sp_st['total_pnl']})
        print(f"    ${sp:.2f}: Sharpe={sp_st['sharpe']:.2f}, PnL=${sp_st['total_pnl']:.0f}")

    # Yearly breakdown
    print(f"\n  --- Yearly Breakdown ---")
    year_pnl = {}
    for t in final_trades:
        yr = pd.Timestamp(t.exit_time).year
        year_pnl[yr] = year_pnl.get(yr, 0) + t.pnl
    for yr in sorted(year_pnl):
        status = "OK" if year_pnl[yr] > 0 else "LOSS"
        print(f"    {yr}: ${year_pnl[yr]:>8.0f} ({status})")
    positive_years = sum(1 for v in year_pnl.values() if v > 0)
    print(f"    Positive years: {positive_years}/{len(year_pnl)}")

    # Comparison with baseline
    base_sh = p1['baseline']['sharpe']
    print(f"\n  --- Final Comparison ---")
    print(f"  {'Version':>20} {'Sharpe':>8} {'PnL':>10} {'N':>6} {'MaxDD':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*6} {'-'*8}")
    print(f"  {'L8_BASE naked':>20} {base_sh:>8.2f} ${p1['baseline']['pnl']:>9.0f} "
          f"{p1['baseline']['n']:>6} ${p1['baseline']['max_dd']:>7.0f}")
    print(f"  {'L8_MAX':>20} {final_st['sharpe']:>8.2f} ${final_st['total_pnl']:>9.0f} "
          f"{final_st['n']:>6} ${final_st['max_dd']:>7.0f}")
    improvement = final_st['sharpe'] - base_sh
    print(f"  Delta: {improvement:+.2f} Sharpe")

    elapsed = time.time() - t0

    result = {
        'combo': combo_label,
        'passed_layers': passed_layers,
        'final_stats': final_st,
        'spread_sensitivity': spread_results,
        'year_pnl': {str(k): v for k, v in year_pnl.items()},
        'positive_years': f"{positive_years}/{len(year_pnl)}",
        'improvement_vs_baseline': improvement,
        'elapsed_s': round(elapsed),
    }

    with open(OUT_DIR / "phase3_stress.json", 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Saved: {OUT_DIR / 'phase3_stress.json'}")
    print(f"  Phase 3 elapsed: {elapsed:.0f}s")

    return result


# ═══════════════════════════════════════════════════════════════
# Config Maps (for K-Fold reconstruction)
# ═══════════════════════════════════════════════════════════════

def session_configs_map():
    return {
        'NoFilter': None,
        'Skip22-1UTC': list(range(1, 22)),
        'SkipAsian': [i for i in range(24) if i not in range(22, 24) and i not in range(0, 5)],
        'LDN+NY_only': list(range(7, 21)),
    }

def tatrail_configs_map():
    return {
        'OFF (baseline)': {},
        's2/d0.75/f0.003': {'time_adaptive_trail': {'start': 2, 'decay': 0.75, 'floor': 0.003}},
        's2/d0.80/f0.005': {'time_adaptive_trail': {'start': 2, 'decay': 0.80, 'floor': 0.005}},
        's3/d0.70/f0.003': {'time_adaptive_trail': {'start': 3, 'decay': 0.70, 'floor': 0.003}},
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t_total = time.time()

    out_path = OUT_DIR / "R49_output.txt"
    f_out = open(out_path, 'w', encoding='utf-8')
    tee = Tee(sys.stdout, f_out)
    sys.stdout = tee

    print("=" * 80)
    print(f"  R49: L8_BASE Overlay Validation")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output: {OUT_DIR}")
    print("=" * 80)

    print("\n  Loading data...")
    data = DataBundle.load_default()
    print(f"  M15: {len(data.m15_df)} bars, H1: {len(data.h1_df)} bars")

    p1 = phase_1(data)
    print(f"\n  [Checkpoint] Phase 1 done, elapsed: {(time.time()-t_total)/60:.1f} min")

    p2 = phase_2(data, p1)
    print(f"\n  [Checkpoint] Phase 2 done, elapsed: {(time.time()-t_total)/60:.1f} min")

    p3 = phase_3(data, p1, p2)
    print(f"\n  [Checkpoint] Phase 3 done, elapsed: {(time.time()-t_total)/60:.1f} min")

    total = time.time() - t_total
    print(f"\n{'='*80}")
    print(f"  R49 COMPLETE — Total: {total/60:.1f} min ({total:.0f}s)")
    print(f"{'='*80}")

    sys.stdout = sys.__stdout__
    f_out.close()


if __name__ == '__main__':
    main()
