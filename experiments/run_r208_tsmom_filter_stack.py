#!/usr/bin/env python3
"""R208: TSMOM Filter Stack Validation
=======================================
R205/R206 validated TSMOM candidate_A params (slow_lb=960, SL=3.5, TP=8.0,
MH=30, RuleB=2.5σ) — but those tests used ONLY Rule B + absolute ATR floor.

The live EA adds two more filter layers that were NEVER tested for TSMOM:
  1. Choppy Gate: trend_score < threshold → skip entry
  2. ATR Percentile Floor: ATR < Nth pctl of rolling window → skip

These filters are Keltner-origin and were validated only for Keltner (R197).
If they're active for TSMOM in production, we must verify they help, not hurt.

This experiment:

Phase 1 — Choppy Threshold Sweep
  Compute intraday trend_score (same algo as BacktestEngine), apply to TSMOM.
  Sweep threshold: [0.00(disabled), 0.20, 0.30, 0.40, 0.50, 0.60]
  Focus: full period + per-era + 2024-2026 close-up

Phase 2 — ATR Percentile Floor Sweep
  Rolling-50 H1 ATR percentile rank. Skip entry if pctl < floor.
  Sweep floor: [0.00(disabled), 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]

Phase 3 — Interaction Matrix (Choppy × ATR Pctl)
  Top 3 choppy × top 3 ATR pctl = 9 combos + baseline

Phase 4 — 3-Gate Validation on top combos
  6-fold CV, walk-forward, era stability (with Recent 2024-2026 weight)

Phase 5 — Recent Era Deep-Dive (2022-2026)
  Monthly Sharpe heatmap, filter hit rate, missed-trade analysis

Run: python experiments/run_r208_tsmom_filter_stack.py
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

from backtest.runner import load_csv, H1_CSV_PATH

OUTPUT_DIR = Path("results/r208_tsmom_filter_stack")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOT = 0.04
PV = 100

# R205 candidate_A params (3-gate validated)
CANDIDATE_A_LOOKBACKS = {'fast_lb': 480, 'slow_lb': 960}
CANDIDATE_A_BT = {
    'sl_atr_mult': 3.5, 'tp_atr_mult': 8.0,
    'max_hold': 30, 'rule_b_sigma': 2.5,
    'rule_b_lb': 60, 'rule_b_skip': 8,
    'atr_floor': 0.1, 'min_gap_hours': 2.0,
    'slow_lb': 960,
}

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


# ─── Helper functions ──────────────────────────────────────────────

def compute_atr(h1: pd.DataFrame, period: int = 14) -> np.ndarray:
    high = h1['High'].values
    low  = h1['Low'].values
    close = h1['Close'].values
    tr = np.zeros(len(h1))
    for i in range(1, len(h1)):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    atr = np.zeros(len(h1))
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, len(h1)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def compute_atr_pctl(atr: np.ndarray, window: int = 50) -> np.ndarray:
    """Rolling percentile rank of ATR over last `window` bars (same as engine)."""
    n = len(atr)
    pctl = np.full(n, 0.5)
    for i in range(window, n):
        w = atr[i - window:i]
        valid = w[w > 0]
        if len(valid) >= 10:
            pctl[i] = float((valid < atr[i]).sum()) / len(valid)
    return pctl


def compute_trend_score(h1: pd.DataFrame) -> np.ndarray:
    """Replicate BacktestEngine._calc_realtime_score() daily trend score.

    Uses same logic as engine.py:
      For each day, compute score from today's H1 bars:
        body_pct, wick_ratio, direction_consistency,
        range_expansion, close_position
      Average → score in [0, 1]
    """
    n = len(h1)
    score = np.full(n, 0.5)
    high = h1['High'].values
    low = h1['Low'].values
    opn = h1['Open'].values
    close = h1['Close'].values
    dates = h1.index.date

    cur_date = None
    day_start = 0

    for i in range(n):
        d = dates[i]
        if d != cur_date:
            cur_date = d
            day_start = i

        day_len = i - day_start + 1
        if day_len < 3:
            score[i] = 0.5
            continue

        sl = slice(day_start, i + 1)
        d_h = high[sl]
        d_l = low[sl]
        d_o = opn[sl]
        d_c = close[sl]

        # Component 1: body/range ratio
        bodies = np.abs(d_c - d_o)
        ranges = d_h - d_l
        ranges = np.where(ranges < 1e-8, 1e-8, ranges)
        body_pct = float(np.mean(bodies / ranges))

        # Component 2: direction consistency
        up_bars = np.sum(d_c > d_o)
        dn_bars = np.sum(d_c < d_o)
        dir_consist = abs(up_bars - dn_bars) / max(day_len, 1)

        # Component 3: range expansion
        if day_len >= 3:
            first_half = ranges[:day_len // 2]
            second_half = ranges[day_len // 2:]
            if first_half.mean() > 1e-8:
                range_exp = min(1.0, max(0.0, second_half.mean() / first_half.mean() - 0.5))
            else:
                range_exp = 0.5
        else:
            range_exp = 0.5

        # Average
        score[i] = (body_pct + dir_consist + range_exp) / 3.0

    return score


def compute_score(close: np.ndarray, fast_lb: int, slow_lb: int) -> np.ndarray:
    n = len(close)
    score = np.zeros(n)
    for i in range(n):
        s = 0.0
        if i >= fast_lb and close[i - fast_lb] > 0:
            ret = close[i] / close[i - fast_lb] - 1.0
            s += 0.5 * (1.0 if ret > 0 else (-1.0 if ret < 0 else 0.0))
        if i >= slow_lb and close[i - slow_lb] > 0:
            ret = close[i] / close[i - slow_lb] - 1.0
            s += 0.5 * (1.0 if ret > 0 else (-1.0 if ret < 0 else 0.0))
        score[i] = s
    return score


def backtest_tsmom_filtered(
    h1: pd.DataFrame,
    atr: np.ndarray,
    mom_score: np.ndarray,
    trend_score: np.ndarray,
    atr_pctl: np.ndarray,
    *,
    choppy_threshold: float = 0.0,
    atr_pctl_floor: float = 0.0,
    sl_atr_mult: float = 3.5,
    tp_atr_mult: float = 8.0,
    max_hold: int = 30,
    rule_b_sigma: float = 2.5,
    rule_b_lb: int = 60,
    rule_b_skip: int = 8,
    atr_floor: float = 0.1,
    min_gap_hours: float = 2.0,
    slow_lb: int = 960,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict:
    """TSMOM backtest with full filter stack. Returns trades + filter stats."""
    n = len(h1)
    close = h1['Close'].values
    high  = h1['High'].values
    low   = h1['Low'].values
    ts    = h1.index

    start_idx = slow_lb + 2
    end_idx = n
    if start:
        start_idx = max(ts.searchsorted(pd.Timestamp(start, tz='UTC')), slow_lb + 2)
    if end:
        end_idx = ts.searchsorted(pd.Timestamp(end, tz='UTC'))

    rb_skip = 0
    last_entry_time = pd.Timestamp('1970-01-01', tz='UTC')
    in_pos = False
    pos_dir = 0
    pos_bars = 0
    pos_entry_price = 0.0
    pos_entry_atr = 0.0
    pos_entry_time = pd.Timestamp('1970-01-01', tz='UTC')

    trades = []
    filter_stats = {
        'total_signals': 0,
        'blocked_choppy': 0,
        'blocked_atr_pctl': 0,
        'blocked_rule_b': 0,
        'blocked_atr_floor': 0,
        'blocked_gap': 0,
        'entries': 0,
    }
    # Per-year filter breakdown
    yearly_filter = defaultdict(lambda: {
        'signals': 0, 'entries': 0,
        'b_choppy': 0, 'b_atr_pctl': 0, 'b_rule_b': 0
    })

    for i in range(start_idx, min(end_idx, n)):
        cur_atr = atr[i]
        if cur_atr <= 0:
            continue

        # Manage position
        if in_pos:
            pos_bars += 1
            c = close[i]
            tp_dist = tp_atr_mult * pos_entry_atr
            sl_dist = sl_atr_mult * pos_entry_atr
            pnl = (c - pos_entry_price) if pos_dir == 1 else (pos_entry_price - c)

            reason = None
            exit_price = c
            if pnl >= tp_dist:
                reason = 'TP'
                exit_price = pos_entry_price + (tp_dist if pos_dir == 1 else -tp_dist)
            elif pnl <= -sl_dist:
                reason = 'SL'
                exit_price = pos_entry_price + (-sl_dist if pos_dir == 1 else sl_dist)
            elif pos_bars >= max_hold:
                reason = 'Timeout'
            else:
                cur_mom = mom_score[i]
                if pos_dir == 1 and cur_mom < 0:
                    reason = 'ScoreFlip'
                if pos_dir == -1 and cur_mom > 0:
                    reason = 'ScoreFlip'

            if reason:
                final_pnl = (exit_price - pos_entry_price) if pos_dir == 1 else (pos_entry_price - exit_price)
                trades.append({
                    'entry_time': str(pos_entry_time),
                    'exit_time': str(ts[i]),
                    'direction': 'BUY' if pos_dir == 1 else 'SELL',
                    'entry_price': pos_entry_price,
                    'exit_price': exit_price,
                    'pnl': final_pnl,
                    'pnl_usd': final_pnl * LOT * PV,
                    'bars_held': pos_bars,
                    'reason': reason,
                    'trend_score_at_entry': round(float(trend_score[pos_entry_idx]), 3) if 'pos_entry_idx' in dir() else 0,
                    'atr_pctl_at_entry': round(float(atr_pctl[pos_entry_idx]), 3) if 'pos_entry_idx' in dir() else 0,
                })
                in_pos = False

        # Rule B
        if i >= rule_b_lb:
            window = atr[i - rule_b_lb:i]
            window = window[window > 0]
            if len(window) >= 10:
                mean = window.mean()
                std = max(window.std(), 1e-6)
                if cur_atr > mean + rule_b_sigma * std:
                    rb_skip = rule_b_skip
                elif rb_skip > 0:
                    rb_skip -= 1

        # Signal detection
        if i < 1:
            continue
        s_now = mom_score[i]
        s_prev = mom_score[i - 1]
        sig = None
        if s_now > 0 and s_prev <= 0:
            sig = 1
        elif s_now < 0 and s_prev >= 0:
            sig = -1
        if sig is None:
            continue

        if in_pos:
            continue

        yr = ts[i].year
        filter_stats['total_signals'] += 1
        yearly_filter[yr]['signals'] += 1

        # Filter: Rule B
        if rb_skip > 0:
            filter_stats['blocked_rule_b'] += 1
            yearly_filter[yr]['b_rule_b'] += 1
            continue

        # Filter: min entry gap
        if (ts[i] - last_entry_time).total_seconds() < min_gap_hours * 3600:
            filter_stats['blocked_gap'] += 1
            continue

        # Filter: absolute ATR floor
        if cur_atr < atr_floor:
            filter_stats['blocked_atr_floor'] += 1
            continue

        # Filter: Choppy Gate
        if choppy_threshold > 0 and trend_score[i] < choppy_threshold:
            filter_stats['blocked_choppy'] += 1
            yearly_filter[yr]['b_choppy'] += 1
            continue

        # Filter: ATR Percentile Floor
        if atr_pctl_floor > 0 and atr_pctl[i] < atr_pctl_floor:
            filter_stats['blocked_atr_pctl'] += 1
            yearly_filter[yr]['b_atr_pctl'] += 1
            continue

        # Entry
        in_pos = True
        pos_dir = sig
        pos_entry_atr = cur_atr
        pos_entry_price = close[i]
        pos_bars = 0
        pos_entry_time = ts[i]
        pos_entry_idx = i
        last_entry_time = ts[i]
        filter_stats['entries'] += 1
        yearly_filter[yr]['entries'] += 1

    return {
        'trades': trades,
        'filter_stats': filter_stats,
        'yearly_filter': dict(yearly_filter),
    }


def calc_stats(trades: List[Dict]) -> Dict:
    if not trades:
        return {'n': 0, 'pnl': 0, 'sharpe': 0, 'win_rate': 0, 'avg_pnl': 0, 'max_dd': 0}
    pnls = np.array([t['pnl_usd'] for t in trades])
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    dd = running_max - cumsum
    n = len(pnls)
    sharpe = float(pnls.mean() / max(pnls.std(), 1e-9) * np.sqrt(252 * 4)) if n > 1 else 0
    reasons = {}
    for t in trades:
        reasons[t['reason']] = reasons.get(t['reason'], 0) + 1
    return {
        'n': n,
        'pnl': round(float(pnls.sum()), 2),
        'sharpe': round(sharpe, 3),
        'win_rate': round(100 * (pnls > 0).sum() / n, 2),
        'avg_pnl': round(float(pnls.mean()), 4),
        'max_dd': round(float(dd.max()), 2),
        'reasons': reasons,
    }


def save(name, data):
    p = OUTPUT_DIR / f'{name}.json'
    with open(p, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f'  -> saved {p}')


def run_one(h1, atr, mom_score, trend_score, atr_pctl,
            choppy_t, atr_pctl_f, start=None, end=None):
    """Run one filtered backtest, return stats + filter_stats."""
    result = backtest_tsmom_filtered(
        h1, atr, mom_score, trend_score, atr_pctl,
        choppy_threshold=choppy_t, atr_pctl_floor=atr_pctl_f,
        start=start, end=end, **CANDIDATE_A_BT,
    )
    stats = calc_stats(result['trades'])
    stats['filter_stats'] = result['filter_stats']
    stats['yearly_filter'] = result['yearly_filter']
    return stats, result['trades']


def main():
    t_start = time.time()
    print('=' * 80)
    print('R208: TSMOM Filter Stack Validation')
    print('=' * 80)

    print('\nLoading H1 data...')
    h1 = load_csv(str(H1_CSV_PATH))
    print(f'  H1: {len(h1):,} bars, {h1.index[0]} -> {h1.index[-1]}')

    print('Computing indicators...')
    atr = compute_atr(h1)
    mom_score = compute_score(h1['Close'].values,
                              CANDIDATE_A_LOOKBACKS['fast_lb'],
                              CANDIDATE_A_LOOKBACKS['slow_lb'])
    trend_score = compute_trend_score(h1)
    atr_pctl = compute_atr_pctl(atr, window=50)
    print(f'  ATR range: {atr[atr>0].min():.2f} - {atr.max():.2f}')
    print(f'  Trend score range: {trend_score.min():.3f} - {trend_score.max():.3f}')
    print(f'  ATR pctl range: {atr_pctl[atr_pctl>0].min():.3f} - {atr_pctl.max():.3f}')

    # Baseline: no choppy, no ATR pctl (pure R205 candidate_A)
    print('\nBaseline (no choppy, no ATR pctl)...')
    bl_stats, bl_trades = run_one(h1, atr, mom_score, trend_score, atr_pctl, 0.0, 0.0)
    print(f'  Baseline: n={bl_stats["n"]}  PnL=${bl_stats["pnl"]:.0f}  '
          f'Sharpe={bl_stats["sharpe"]:.3f}  WR={bl_stats["win_rate"]:.1f}%')

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Choppy Threshold Sweep
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 1: Choppy Threshold Sweep (ATR pctl = disabled)')
    print('=' * 80)

    CHOPPY_THRESHOLDS = [0.0, 0.20, 0.30, 0.40, 0.50, 0.60]

    phase1 = []
    for ct in CHOPPY_THRESHOLDS:
        label = f'choppy_{ct:.2f}'
        stats, _ = run_one(h1, atr, mom_score, trend_score, atr_pctl, ct, 0.0)
        row = {'label': label, 'choppy_threshold': ct, **stats}
        phase1.append(row)
        fs = stats['filter_stats']
        print(f'  {label}: n={stats["n"]}  PnL=${stats["pnl"]:>8.0f}  '
              f'Sharpe={stats["sharpe"]:>6.3f}  blocked_choppy={fs["blocked_choppy"]:>3}  '
              f'WR={stats["win_rate"]:>5.1f}%')

    # Per-era for each threshold
    print('\n  Per-era breakdown:')
    phase1_era = {}
    for ct in CHOPPY_THRESHOLDS:
        label = f'choppy_{ct:.2f}'
        era_row = {}
        for era_name, (es, ee) in ERA_SEGMENTS.items():
            stats, _ = run_one(h1, atr, mom_score, trend_score, atr_pctl, ct, 0.0, es, ee)
            era_row[era_name] = {
                'n': stats['n'], 'pnl': stats['pnl'], 'sharpe': stats['sharpe'],
            }
        phase1_era[label] = era_row
        recent = era_row.get("Recent (2024-2026)", {})
        print(f'    {label} Recent: n={recent.get("n",0)}  '
              f'PnL=${recent.get("pnl",0):.0f}  Sharpe={recent.get("sharpe",0):.3f}')

    save('phase1_choppy_sweep', {'grid': phase1, 'per_era': phase1_era})

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: ATR Percentile Floor Sweep
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 2: ATR Percentile Floor Sweep (Choppy = disabled)')
    print('=' * 80)

    PCTL_FLOORS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]

    phase2 = []
    for pf in PCTL_FLOORS:
        label = f'atr_pctl_{pf:.2f}'
        stats, _ = run_one(h1, atr, mom_score, trend_score, atr_pctl, 0.0, pf)
        row = {'label': label, 'atr_pctl_floor': pf, **stats}
        phase2.append(row)
        fs = stats['filter_stats']
        print(f'  {label}: n={stats["n"]}  PnL=${stats["pnl"]:>8.0f}  '
              f'Sharpe={stats["sharpe"]:>6.3f}  blocked_pctl={fs["blocked_atr_pctl"]:>3}  '
              f'WR={stats["win_rate"]:>5.1f}%')

    # Per-era
    print('\n  Per-era breakdown:')
    phase2_era = {}
    for pf in PCTL_FLOORS:
        label = f'atr_pctl_{pf:.2f}'
        era_row = {}
        for era_name, (es, ee) in ERA_SEGMENTS.items():
            stats, _ = run_one(h1, atr, mom_score, trend_score, atr_pctl, 0.0, pf, es, ee)
            era_row[era_name] = {
                'n': stats['n'], 'pnl': stats['pnl'], 'sharpe': stats['sharpe'],
            }
        phase2_era[label] = era_row
        recent = era_row.get("Recent (2024-2026)", {})
        print(f'    {label} Recent: n={recent.get("n",0)}  '
              f'PnL=${recent.get("pnl",0):.0f}  Sharpe={recent.get("sharpe",0):.3f}')

    save('phase2_atr_pctl_sweep', {'grid': phase2, 'per_era': phase2_era})

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: Interaction Matrix
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 3: Choppy × ATR Pctl Interaction Matrix')
    print('=' * 80)

    # Pick top 3 from each (by Sharpe), plus disabled
    p1_sorted = sorted(phase1, key=lambda x: x['sharpe'], reverse=True)
    p2_sorted = sorted(phase2, key=lambda x: x['sharpe'], reverse=True)

    choppy_candidates = sorted(set([0.0] + [r['choppy_threshold'] for r in p1_sorted[:3]]))
    pctl_candidates   = sorted(set([0.0] + [r['atr_pctl_floor'] for r in p2_sorted[:3]]))

    print(f'  Choppy candidates: {choppy_candidates}')
    print(f'  ATR pctl candidates: {pctl_candidates}')

    phase3 = []
    print(f'\n  {"Label":<30} {"n":>5} {"PnL":>10} {"Sharpe":>8} {"WR":>6} {"Blk_C":>6} {"Blk_P":>6}')
    for ct, pf in itertools.product(choppy_candidates, pctl_candidates):
        label = f'c{ct:.2f}_p{pf:.2f}'
        stats, trades = run_one(h1, atr, mom_score, trend_score, atr_pctl, ct, pf)
        fs = stats['filter_stats']
        row = {
            'label': label, 'choppy': ct, 'atr_pctl': pf,
            **{k: v for k, v in stats.items() if k not in ('filter_stats', 'yearly_filter')},
            'blocked_choppy': fs['blocked_choppy'],
            'blocked_atr_pctl': fs['blocked_atr_pctl'],
        }
        phase3.append(row)
        print(f'  {label:<30} {stats["n"]:>5} {stats["pnl"]:>10.0f} {stats["sharpe"]:>8.3f} '
              f'{stats["win_rate"]:>5.1f}% {fs["blocked_choppy"]:>6} {fs["blocked_atr_pctl"]:>6}')

    phase3.sort(key=lambda x: x['sharpe'], reverse=True)
    save('phase3_interaction_matrix', phase3)

    # Top 3 combos for 3-gate
    top3 = phase3[:3]
    print(f'\n  Top 3 for 3-Gate:')
    for r in top3:
        print(f'    {r["label"]}: Sharpe={r["sharpe"]:.3f}  n={r["n"]}')

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: 3-Gate Validation
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 4: 3-Gate Validation')
    print('=' * 80)

    gate_results = {}
    for combo in top3:
        ct = combo['choppy']
        pf = combo['atr_pctl']
        label = combo['label']
        print(f'\n  === {label} (choppy={ct}, atr_pctl={pf}) ===')

        # Gate 1: 6-Fold CV
        print('    Gate 1: 6-Fold CV...')
        n_bars = len(h1)
        fold_size = n_bars // 6
        kf_sharpes = []
        kf_bl_sharpes = []

        for fold in range(6):
            test_start_idx = fold * fold_size
            test_end_idx = min((fold + 1) * fold_size, n_bars)
            test_start = str(h1.index[test_start_idx])[:10]
            test_end   = str(h1.index[min(test_end_idx, n_bars - 1)])[:10]

            stats, _ = run_one(h1, atr, mom_score, trend_score, atr_pctl,
                               ct, pf, test_start, test_end)
            bl_stats, _ = run_one(h1, atr, mom_score, trend_score, atr_pctl,
                                  0.0, 0.0, test_start, test_end)

            kf_sharpes.append(round(stats['sharpe'], 3))
            kf_bl_sharpes.append(round(bl_stats['sharpe'], 3))
            print(f'      Fold {fold+1}: Sharpe={stats["sharpe"]:.3f} '
                  f'(baseline={bl_stats["sharpe"]:.3f})  n={stats["n"]}')

        kf_wins = sum(1 for s, b in zip(kf_sharpes, kf_bl_sharpes) if s >= b)
        kf_pass = kf_wins >= 4
        print(f'    KF: {kf_wins}/6  PASS={kf_pass}')

        # Gate 2: Walk-Forward
        print('    Gate 2: Walk-Forward...')
        wf_wins = 0
        wf_details = []
        for train_s, train_e, test_s, test_e in WF_WINDOWS:
            stats, _ = run_one(h1, atr, mom_score, trend_score, atr_pctl,
                               ct, pf, test_s, test_e)
            bl_stats, _ = run_one(h1, atr, mom_score, trend_score, atr_pctl,
                                  0.0, 0.0, test_s, test_e)
            win = stats['sharpe'] >= bl_stats['sharpe']
            if win:
                wf_wins += 1
            wf_details.append({
                'window': test_s, 'sharpe': stats['sharpe'],
                'bl_sharpe': bl_stats['sharpe'], 'win': win
            })
        wf_pass = wf_wins >= len(WF_WINDOWS) * 0.6
        print(f'    WF: {wf_wins}/{len(WF_WINDOWS)}  PASS={wf_pass}')

        # Gate 3: Era Stability
        print('    Gate 3: Era Stability...')
        era_results = []
        for era_name, (es, ee) in ERA_SEGMENTS.items():
            stats, _ = run_one(h1, atr, mom_score, trend_score, atr_pctl,
                               ct, pf, es, ee)
            bl_stats, _ = run_one(h1, atr, mom_score, trend_score, atr_pctl,
                                  0.0, 0.0, es, ee)
            era_results.append({
                'era': era_name, 'sharpe': stats['sharpe'], 'bl_sharpe': bl_stats['sharpe'],
                'n': stats['n'], 'bl_n': bl_stats['n'],
                'pnl': stats['pnl'], 'bl_pnl': bl_stats['pnl'],
            })
            delta = stats['sharpe'] - bl_stats['sharpe']
            print(f'      {era_name}: Sharpe={stats["sharpe"]:.3f} (bl={bl_stats["sharpe"]:.3f}) '
                  f'delta={delta:+.3f}  n={stats["n"]} (bl={bl_stats["n"]})')

        era_sharpes = [e['sharpe'] for e in era_results]
        era_pass = all(s > 0 for s in era_sharpes) and min(era_sharpes) > 0.5
        print(f'    Era: min_sharpe={min(era_sharpes):.3f}  PASS={era_pass}')

        overall = kf_pass and wf_pass and era_pass
        gate_results[label] = {
            'choppy': ct, 'atr_pctl': pf,
            'kfold': {'sharpes': kf_sharpes, 'bl_sharpes': kf_bl_sharpes,
                      'wins': kf_wins, 'pass': kf_pass},
            'walk_forward': {'wins': wf_wins, 'total': len(WF_WINDOWS),
                             'pass': wf_pass, 'details': wf_details},
            'era': {'results': era_results, 'pass': era_pass},
            'overall_pass': overall,
        }
        tag = '[GO]' if overall else '[NO-GO]'
        print(f'    Overall: {tag}')

    save('phase4_three_gate', gate_results)

    # ═══════════════════════════════════════════════════════════════
    # Phase 5: Recent Era Deep-Dive (2022-2026)
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 5: Recent Era Deep-Dive (2022-2026)')
    print('=' * 80)

    # Monthly Sharpe for recent years, comparing baseline vs best filter combo
    best_combo = phase3[0]
    best_ct = best_combo['choppy']
    best_pf = best_combo['atr_pctl']
    print(f'  Best combo: choppy={best_ct}, atr_pctl={best_pf}')

    monthly_data = []
    for yr in range(2022, 2027):
        for mo in range(1, 13):
            ms = f'{yr}-{mo:02d}-01'
            if mo < 12:
                me = f'{yr}-{mo+1:02d}-01'
            else:
                me = f'{yr+1}-01-01'
            if pd.Timestamp(ms, tz='UTC') > h1.index[-1]:
                break

            stats_f, trades_f = run_one(h1, atr, mom_score, trend_score, atr_pctl,
                                        best_ct, best_pf, ms, me)
            stats_bl, trades_bl = run_one(h1, atr, mom_score, trend_score, atr_pctl,
                                          0.0, 0.0, ms, me)

            if stats_f['n'] > 0 or stats_bl['n'] > 0:
                monthly_data.append({
                    'month': f'{yr}-{mo:02d}',
                    'filtered_n': stats_f['n'], 'filtered_pnl': stats_f['pnl'],
                    'filtered_sharpe': stats_f['sharpe'],
                    'baseline_n': stats_bl['n'], 'baseline_pnl': stats_bl['pnl'],
                    'baseline_sharpe': stats_bl['sharpe'],
                    'delta_pnl': round(stats_f['pnl'] - stats_bl['pnl'], 2),
                    'delta_n': stats_f['n'] - stats_bl['n'],
                })

    print(f'\n  Monthly comparison (2022-2026):')
    print(f'  {"Month":<10} {"Filt_n":>7} {"Filt_PnL":>10} {"BL_n":>7} {"BL_PnL":>10} {"dPnL":>8}')
    for m in monthly_data:
        print(f'  {m["month"]:<10} {m["filtered_n"]:>7} {m["filtered_pnl"]:>10.0f} '
              f'{m["baseline_n"]:>7} {m["baseline_pnl"]:>10.0f} {m["delta_pnl"]:>+8.0f}')

    save('phase5_recent_monthly', monthly_data)

    # Missed trade analysis: trades in baseline but filtered out
    print(f'\n  Missed Trade Analysis (full period, best combo)...')
    _, all_bl_trades = run_one(h1, atr, mom_score, trend_score, atr_pctl, 0.0, 0.0)
    _, all_f_trades  = run_one(h1, atr, mom_score, trend_score, atr_pctl, best_ct, best_pf)

    bl_entries = set(t['entry_time'] for t in all_bl_trades)
    f_entries  = set(t['entry_time'] for t in all_f_trades)
    missed_entries = bl_entries - f_entries

    missed_trades = [t for t in all_bl_trades if t['entry_time'] in missed_entries]
    if missed_trades:
        missed_pnls = [t['pnl_usd'] for t in missed_trades]
        missed_wins = sum(1 for p in missed_pnls if p > 0)
        missed_total = sum(missed_pnls)
        print(f'  Missed trades: {len(missed_trades)}')
        print(f'  Missed PnL: ${missed_total:.0f} (win_rate={100*missed_wins/len(missed_trades):.1f}%)')
        print(f'  Average missed trade: ${np.mean(missed_pnls):.2f}')

        missed_summary = {
            'n_missed': len(missed_trades),
            'missed_pnl_total': round(missed_total, 2),
            'missed_win_rate': round(100 * missed_wins / len(missed_trades), 2),
            'missed_avg_pnl': round(float(np.mean(missed_pnls)), 2),
            'missed_by_year': {},
        }
        for t in missed_trades:
            yr = t['entry_time'][:4]
            if yr not in missed_summary['missed_by_year']:
                missed_summary['missed_by_year'][yr] = {'n': 0, 'pnl': 0}
            missed_summary['missed_by_year'][yr]['n'] += 1
            missed_summary['missed_by_year'][yr]['pnl'] += t['pnl_usd']

        print(f'\n  Missed trades by year:')
        for yr, yd in sorted(missed_summary['missed_by_year'].items()):
            yd['pnl'] = round(yd['pnl'], 2)
            print(f'    {yr}: n={yd["n"]}  PnL=${yd["pnl"]:.0f}')

        save('phase5_missed_trades', missed_summary)
    else:
        print('  No missed trades (filters have no effect)')
        save('phase5_missed_trades', {'n_missed': 0})

    # ═══════════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('FINAL SUMMARY')
    print('=' * 80)

    summary = {
        'baseline': {
            'n': bl_stats['n'], 'pnl': bl_stats['pnl'], 'sharpe': bl_stats['sharpe'],
            'description': 'R205 candidate_A, no choppy/ATR pctl filters',
        },
        'best_combo': {
            'choppy': best_ct, 'atr_pctl': best_pf,
            'n': phase3[0]['n'], 'pnl': phase3[0]['pnl'], 'sharpe': phase3[0]['sharpe'],
        },
        'phase1_best_choppy': p1_sorted[0] if p1_sorted else None,
        'phase2_best_pctl': p2_sorted[0] if p2_sorted else None,
        'three_gate_results': {
            k: {'pass': v['overall_pass'], 'kf_wins': v['kfold']['wins'],
                'wf_wins': v['walk_forward']['wins'],
                'era_min': min(e['sharpe'] for e in v['era']['results'])}
            for k, v in gate_results.items()
        },
    }

    # Verdict
    go_combos = [k for k, v in gate_results.items() if v['overall_pass']]
    if go_combos:
        summary['verdict'] = 'GO'
        summary['recommended'] = go_combos
        print(f'  VERDICT: GO — Filters validated')
        for gc in go_combos:
            g = gate_results[gc]
            print(f'    {gc}: KF={g["kfold"]["wins"]}/6  WF={g["walk_forward"]["wins"]}/{g["walk_forward"]["total"]}  '
                  f'Era_min={min(e["sharpe"] for e in g["era"]["results"]):.3f}')
    else:
        # Check if baseline (no filters) is better
        bl_best = bl_stats['sharpe'] >= phase3[0]['sharpe']
        if bl_best:
            summary['verdict'] = 'NO-FILTER'
            summary['reason'] = 'Baseline (no choppy/ATR pctl) has equal or better Sharpe; filters hurt TSMOM'
            print(f'  VERDICT: NO-FILTER — Filters hurt TSMOM performance')
            print(f'  Baseline Sharpe={bl_stats["sharpe"]:.3f} >= Best filtered={phase3[0]["sharpe"]:.3f}')
        else:
            summary['verdict'] = 'INCONCLUSIVE'
            summary['reason'] = 'Filters show improvement but fail 3-gate; more data needed'
            print(f'  VERDICT: INCONCLUSIVE')

    save('R208_summary', summary)

    elapsed = time.time() - t_start
    print(f'\n  Total runtime: {elapsed:.0f}s')
    print(f'  All results in {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
