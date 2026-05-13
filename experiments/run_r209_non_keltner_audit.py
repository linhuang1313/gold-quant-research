#!/usr/bin/env python3
"""R209: Non-Keltner Strategy Deep Audit
=========================================
Live reconciliation (R200 E2) shows:
  PSAR: 3 trades, all losses (-$85)
  Chandelier: 3 trades, net loss (-$41)
  DualThrust: 11 trades, barely positive ($25)
  Session BO: 1 trade, +$35

These strategies have NEVER had R205-level validation. This experiment:

Phase 1 — Baseline per-strategy performance (full + per-era + recent)
Phase 2 — Parameter sweep for each strategy (SL/TP/MH/trail)
Phase 3 — 3-Gate Validation on best params vs current params
Phase 4 — Kill-or-Keep decision: is each strategy worth running?

Uses the same bt_h1_strategy() approach as R200 for consistency.

Run: python experiments/run_r209_non_keltner_audit.py
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

from backtest.runner import load_csv, H1_CSV_PATH, M15_CSV_PATH

OUTPUT_DIR = Path("results/r209_non_keltner_audit")
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

# Current production configs (from R200 STRAT_CONFIGS)
STRAT_CONFIGS = {
    'psar': {
        'lot': 0.09, 'sl_atr': 6.0, 'tp_atr': 6.0, 'cap': 60,
        'trail_act_atr': 0.06, 'trail_dist_atr': 0.01, 'max_hold_h': 15,
        'cooldown_h': 2,
    },
    'sess_bo': {
        'lot': 0.04, 'sl_atr': 4.5, 'tp_atr': 4.0, 'cap': 60,
        'trail_act_atr': 0.06, 'trail_dist_atr': 0.01, 'max_hold_h': 20,
        'cooldown_h': 2,
    },
    'dual_thrust': {
        'lot': 0.04, 'sl_atr': 6.0, 'tp_atr': 8.0, 'cap': 18,
        'trail_act_atr': 0.06, 'trail_dist_atr': 0.01, 'max_hold_h': 20,
        'cooldown_h': 2,
    },
    'chandelier': {
        'lot': 0.03, 'sl_atr': 4.5, 'tp_atr': 8.0, 'cap': 25,
        'trail_act_atr': 0.06, 'trail_dist_atr': 0.01, 'max_hold_h': 20,
        'cooldown_h': 2,
    },
}

PV = 100


# ─── Indicator computation ──────────────────────────────────────────

def add_indicators(h1: pd.DataFrame) -> pd.DataFrame:
    """Add ATR, PSAR, Chandelier, DualThrust range, RSI, and atr_percentile."""
    h1 = h1.copy()
    high = h1['High'].values
    low = h1['Low'].values
    close = h1['Close'].values
    n = len(h1)

    # ATR (Wilder-14)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i-1]),
                     abs(low[i] - close[i-1]))
    atr = np.zeros(n)
    atr[14] = np.mean(tr[1:15])
    for i in range(15, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    h1['ATR'] = atr

    # ATR percentile (rolling 500)
    atr_s = pd.Series(atr, index=h1.index)
    h1['atr_percentile'] = atr_s.rolling(500, min_periods=50).apply(
        lambda x: (x[:-1] < x.iloc[-1]).mean() if len(x) > 1 else 0.5, raw=False
    ).fillna(0.5)

    # PSAR
    psar = np.zeros(n)
    bull = True
    af_step, af_max = 0.02, 0.20
    af = af_step
    ep = high[0]
    psar[0] = low[0]
    for i in range(1, n):
        psar[i] = psar[i-1] + af * (ep - psar[i-1])
        if bull:
            psar[i] = min(psar[i], low[i-1], low[max(0, i-2)])
            if low[i] < psar[i]:
                bull = False
                psar[i] = ep
                af = af_step
                ep = low[i]
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            psar[i] = max(psar[i], high[i-1], high[max(0, i-2)])
            if high[i] > psar[i]:
                bull = True
                psar[i] = ep
                af = af_step
                ep = high[i]
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)
    h1['PSAR'] = psar

    # Chandelier (ATR-22 based)
    period = 22
    mult = 3.0
    chand_long = np.full(n, np.nan)
    chand_short = np.full(n, np.nan)
    for i in range(period, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        a = atr[i]
        chand_long[i] = hh - mult * a
        chand_short[i] = ll + mult * a
    h1['Chand_long'] = chand_long
    h1['Chand_short'] = chand_short

    # DualThrust range
    dt_range = np.zeros(n)
    for i in range(1, n):
        dt_range[i] = max(high[i-1] - close[i-1], close[i-1] - low[i-1])
    h1['DT_range'] = dt_range

    # RSI-14
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    h1['RSI14'] = 100 - (100 / (1 + rs))

    return h1


# ─── Entry signal logic (mirrors R200 _check_entry) ─────────────────

def check_entry(h1, i, strategy, extra=None):
    if extra is None:
        extra = {}
    row = h1.iloc[i]
    prev = h1.iloc[i-1] if i > 0 else row
    pprev = h1.iloc[i-2] if i > 1 else prev

    if strategy == 'psar':
        psar_now = row.get('PSAR', 0)
        psar_prev = prev.get('PSAR', 0)
        c = row['Close']
        if psar_prev > prev['Close'] and psar_now < c:
            return 'BUY'
        elif psar_prev < prev['Close'] and psar_now > c:
            return 'SELL'

    elif strategy == 'sess_bo':
        hour = h1.index[i].hour
        if hour != 12:
            return None
        c = row['Close']
        rng = h1.iloc[max(0, i-4):i]
        if len(rng) < 4:
            return None
        hi = rng['High'].max()
        lo = rng['Low'].min()
        if c > hi:
            return 'BUY'
        elif c < lo:
            return 'SELL'

    elif strategy == 'dual_thrust':
        c = prev['Close']
        dt_range = prev.get('DT_range', 0)
        if dt_range <= 0:
            return None
        k = extra.get('k_up', 0.5)
        daily_open = h1.iloc[max(0, i-1)]['Open']
        if c > daily_open + k * dt_range:
            return 'BUY'
        elif c < daily_open - k * dt_range:
            return 'SELL'

    elif strategy == 'chandelier':
        clong_now = row.get('Chand_long', 0)
        cshort_now = row.get('Chand_short', 0)
        clong_prev = prev.get('Chand_long', 0)
        cshort_prev = prev.get('Chand_short', 0)
        c = prev['Close']
        rsi = row.get('RSI14', 50)
        if c > clong_prev and pprev['Close'] <= pprev.get('Chand_long', c):
            if rsi < 70:
                return 'BUY'
        elif c < cshort_prev and pprev['Close'] >= pprev.get('Chand_short', c):
            if rsi > 30:
                return 'SELL'

    return None


# ─── Generic H1 backtest (mirrors R200 bt_h1_strategy) ──────────────

def bt_h1_strategy(h1, cfg, strategy='psar', trail_act=None, trail_dist=None,
                    cooldown=None, max_hold=None, cap=None, sl_atr=None,
                    tp_atr=None, start=None, end=None, **extra):
    ta = trail_act if trail_act is not None else cfg['trail_act_atr']
    td = trail_dist if trail_dist is not None else cfg['trail_dist_atr']
    cd = cooldown if cooldown is not None else cfg.get('cooldown_h', 2)
    mh = max_hold if max_hold is not None else cfg['max_hold_h']
    cap_usd = cap if cap is not None else cfg.get('cap', 60)
    lot = cfg['lot']
    sl_mult = sl_atr if sl_atr is not None else cfg['sl_atr']
    tp_mult = tp_atr if tp_atr is not None else cfg['tp_atr']

    h1_use = h1
    if start or end:
        mask = pd.Series(True, index=h1.index)
        if start:
            mask &= h1.index >= pd.Timestamp(start, tz='UTC')
        if end:
            mask &= h1.index < pd.Timestamp(end, tz='UTC')
        h1_use = h1[mask]

    close = h1_use['Close'].values
    high_arr = h1_use['High'].values
    low_arr = h1_use['Low'].values
    atr_arr = h1_use['ATR'].values
    pctl_arr = h1_use['atr_percentile'].values

    trades = []
    pos = None
    last_exit_bar = -999

    for i in range(100, len(h1_use)):
        if np.isnan(atr_arr[i]) or atr_arr[i] <= 0:
            continue

        # Position management
        if pos is not None:
            pos['bars'] += 1
            c = close[i]
            h = high_arr[i]
            l = low_arr[i]
            pnl_raw = (c - pos['entry']) if pos['dir'] == 'BUY' else (pos['entry'] - c)
            pnl_usd = pnl_raw * lot * PV

            # Trail
            act_dist = ta * pos['entry_atr']
            trail_d = td * pos['entry_atr']
            if pos['dir'] == 'BUY':
                pos['extreme'] = max(pos['extreme'], h)
                if h - pos['entry'] >= act_dist:
                    trail_sl = pos['extreme'] - trail_d
                    pos['trail'] = max(pos.get('trail', 0), trail_sl)
            else:
                pos['extreme'] = min(pos['extreme'], l)
                if pos['entry'] - l >= act_dist:
                    trail_sl = pos['extreme'] + trail_d
                    if pos.get('trail', 0) == 0:
                        pos['trail'] = trail_sl
                    else:
                        pos['trail'] = min(pos['trail'], trail_sl)

            reason = None
            exit_p = c
            # SL
            if pos['dir'] == 'BUY' and l <= pos['sl']:
                reason = 'SL'
                exit_p = pos['sl']
            elif pos['dir'] == 'SELL' and h >= pos['sl']:
                reason = 'SL'
                exit_p = pos['sl']
            # TP
            if reason is None:
                if pos['dir'] == 'BUY' and h >= pos['tp']:
                    reason = 'TP'
                    exit_p = pos['tp']
                elif pos['dir'] == 'SELL' and l <= pos['tp']:
                    reason = 'TP'
                    exit_p = pos['tp']
            # Trail
            if reason is None and pos.get('trail', 0) != 0:
                if pos['dir'] == 'BUY' and l <= pos['trail']:
                    reason = 'Trail'
                    exit_p = pos['trail']
                elif pos['dir'] == 'SELL' and h >= pos['trail']:
                    reason = 'Trail'
                    exit_p = pos['trail']
            # Max hold
            if reason is None and pos['bars'] >= mh:
                reason = 'Timeout'
            # Cap
            if reason is None and -pnl_usd >= cap_usd:
                reason = 'Cap'
                exit_p = c

            if reason:
                final_pnl_raw = (exit_p - pos['entry']) if pos['dir'] == 'BUY' else (pos['entry'] - exit_p)
                final_pnl = final_pnl_raw * lot * PV
                final_pnl = max(final_pnl, -cap_usd)
                trades.append({
                    'entry_time': str(pos['entry_time']),
                    'exit_time': str(h1_use.index[i]),
                    'dir': pos['dir'],
                    'entry': pos['entry'],
                    'exit': exit_p,
                    'pnl': round(final_pnl, 2),
                    'reason': reason,
                    'bars': pos['bars'],
                })
                last_exit_bar = i
                pos = None

        # Entry
        if pos is None and i - last_exit_bar >= cd:
            sig = check_entry(h1_use, i, strategy, extra)
            if sig:
                atr = atr_arr[i]
                sl_dist = sl_mult * atr
                tp_dist = tp_mult * atr
                entry_p = close[i]
                if sig == 'BUY':
                    sl_p = entry_p - sl_dist
                    tp_p = entry_p + tp_dist
                else:
                    sl_p = entry_p + sl_dist
                    tp_p = entry_p - tp_dist
                pos = {
                    'entry': entry_p, 'sl': sl_p, 'tp': tp_p, 'dir': sig,
                    'bars': 0, 'extreme': entry_p, 'trail': 0,
                    'entry_time': h1_use.index[i], 'entry_atr': atr,
                }

    return trades


def calc_stats(trades: List[Dict]) -> Dict:
    if not trades:
        return {'n': 0, 'pnl': 0, 'sharpe': 0, 'win_rate': 0, 'avg_pnl': 0, 'max_dd': 0, 'reasons': {}}
    pnls = np.array([t['pnl'] for t in trades])
    n = len(pnls)
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    dd = running_max - cumsum
    sharpe = float(pnls.mean() / max(pnls.std(ddof=1), 1e-9) * np.sqrt(252 * 4)) if n > 1 else 0
    reasons = {}
    for t in trades:
        reasons[t['reason']] = reasons.get(t['reason'], 0) + 1
    return {
        'n': n,
        'pnl': round(float(pnls.sum()), 2),
        'sharpe': round(sharpe, 3),
        'win_rate': round(100 * (pnls > 0).sum() / n, 2),
        'avg_pnl': round(float(pnls.mean()), 2),
        'max_dd': round(float(dd.max()), 2),
        'reasons': reasons,
    }


def save(name, data):
    p = OUTPUT_DIR / f'{name}.json'
    with open(p, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f'  -> saved {p}')


def main():
    t_start = time.time()
    print('=' * 80)
    print('R209: Non-Keltner Strategy Deep Audit')
    print('=' * 80)

    print('\nLoading H1 data...')
    h1 = load_csv(str(H1_CSV_PATH))
    print(f'  H1: {len(h1):,} bars, {h1.index[0]} -> {h1.index[-1]}')

    print('Computing indicators...')
    h1 = add_indicators(h1)
    print(f'  ATR, PSAR, Chandelier, DualThrust, RSI computed')

    strategies = ['psar', 'sess_bo', 'dual_thrust', 'chandelier']

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Baseline Performance
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 1: Baseline Performance')
    print('=' * 80)

    phase1 = {}
    for strat in strategies:
        cfg = STRAT_CONFIGS[strat]
        print(f'\n  === {strat.upper()} ===')

        # Full period
        trades = bt_h1_strategy(h1, cfg, strategy=strat)
        full_stats = calc_stats(trades)
        print(f'    Full: n={full_stats["n"]}  PnL=${full_stats["pnl"]:.0f}  '
              f'Sharpe={full_stats["sharpe"]:.3f}  WR={full_stats["win_rate"]:.1f}%  '
              f'MaxDD=${full_stats["max_dd"]:.0f}')

        # Per era
        era_stats = {}
        for era_name, (es, ee) in ERA_SEGMENTS.items():
            era_trades = bt_h1_strategy(h1, cfg, strategy=strat, start=es, end=ee)
            est = calc_stats(era_trades)
            era_stats[era_name] = est
            print(f'    {era_name}: n={est["n"]}  PnL=${est["pnl"]:.0f}  '
                  f'Sharpe={est["sharpe"]:.3f}  WR={est["win_rate"]:.1f}%')

        # Recent year breakdown
        yearly = {}
        for yr in range(2022, 2027):
            yr_trades = bt_h1_strategy(h1, cfg, strategy=strat,
                                       start=f'{yr}-01-01', end=f'{yr+1}-01-01')
            yr_st = calc_stats(yr_trades)
            yearly[str(yr)] = yr_st
            print(f'    {yr}: n={yr_st["n"]}  PnL=${yr_st["pnl"]:.0f}  Sharpe={yr_st["sharpe"]:.3f}')

        phase1[strat] = {
            'full': full_stats,
            'era': era_stats,
            'yearly': yearly,
        }

    save('phase1_baseline', phase1)

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Parameter Sweep
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 2: Parameter Sweep')
    print('=' * 80)

    SL_GRID = [3.0, 4.5, 6.0, 8.0]
    TP_GRID = [4.0, 6.0, 8.0, 10.0]
    MH_GRID = [10, 15, 20, 30]
    TRAIL_GRID = [
        (0.02, 0.005), (0.06, 0.01), (0.06, 0.015),
        (0.10, 0.02), (0.14, 0.025),
    ]

    phase2 = {}
    for strat in strategies:
        cfg = STRAT_CONFIGS[strat]
        print(f'\n  === {strat.upper()} param sweep ===')

        strat_results = []
        baseline_trades = bt_h1_strategy(h1, cfg, strategy=strat)
        bl_stats = calc_stats(baseline_trades)
        strat_results.append({
            'label': 'baseline', 'sharpe': bl_stats['sharpe'], 'pnl': bl_stats['pnl'],
            'n': bl_stats['n'], 'wr': bl_stats['win_rate'],
        })

        # SL/TP sweep
        for sl, tp in itertools.product(SL_GRID, TP_GRID):
            if tp <= sl:
                continue
            label = f'SL{sl}_TP{tp}'
            trades = bt_h1_strategy(h1, cfg, strategy=strat, sl_atr=sl, tp_atr=tp)
            st = calc_stats(trades)
            strat_results.append({
                'label': label, 'sharpe': st['sharpe'], 'pnl': st['pnl'],
                'n': st['n'], 'wr': st['win_rate'],
            })

        # MH sweep
        for mh in MH_GRID:
            label = f'MH{mh}'
            trades = bt_h1_strategy(h1, cfg, strategy=strat, max_hold=mh)
            st = calc_stats(trades)
            strat_results.append({
                'label': label, 'sharpe': st['sharpe'], 'pnl': st['pnl'],
                'n': st['n'], 'wr': st['win_rate'],
            })

        # Trail sweep
        for ta, td in TRAIL_GRID:
            label = f'T{ta}_{td}'
            trades = bt_h1_strategy(h1, cfg, strategy=strat, trail_act=ta, trail_dist=td)
            st = calc_stats(trades)
            strat_results.append({
                'label': label, 'sharpe': st['sharpe'], 'pnl': st['pnl'],
                'n': st['n'], 'wr': st['win_rate'],
            })

        strat_results.sort(key=lambda x: x['sharpe'], reverse=True)
        print(f'    Top 5:')
        for r in strat_results[:5]:
            print(f'      {r["label"]:<20} Sharpe={r["sharpe"]:>7.3f}  PnL=${r["pnl"]:>8.0f}  '
                  f'n={r["n"]:>5}  WR={r["wr"]:.1f}%')
        print(f'    Baseline: Sharpe={bl_stats["sharpe"]:.3f}')

        phase2[strat] = strat_results

    save('phase2_param_sweep', phase2)

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: 3-Gate Validation
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 3: 3-Gate Validation')
    print('=' * 80)

    phase3 = {}
    for strat in strategies:
        cfg = STRAT_CONFIGS[strat]
        sweep = phase2[strat]
        best = sweep[0]
        bl = next(r for r in sweep if r['label'] == 'baseline')

        print(f'\n  === {strat.upper()}: best={best["label"]} (Sharpe={best["sharpe"]:.3f}) '
              f'vs baseline (Sharpe={bl["sharpe"]:.3f}) ===')

        # Parse best params
        best_kw = {}
        label = best['label']
        if label.startswith('SL'):
            parts = label.split('_')
            best_kw['sl_atr'] = float(parts[0][2:])
            best_kw['tp_atr'] = float(parts[1][2:])
        elif label.startswith('MH'):
            best_kw['max_hold'] = int(label[2:])
        elif label.startswith('T'):
            parts = label[1:].split('_')
            best_kw['trail_act'] = float(parts[0])
            best_kw['trail_dist'] = float(parts[1])

        # Gate 1: 6-Fold CV
        print(f'    Gate 1: 6-Fold CV...')
        n_bars = len(h1)
        fold_size = n_bars // 6
        kf_sharpes = []
        kf_bl_sharpes = []

        for fold in range(6):
            ts = str(h1.index[fold * fold_size])[:10]
            te = str(h1.index[min((fold + 1) * fold_size, n_bars - 1)])[:10]

            trades = bt_h1_strategy(h1, cfg, strategy=strat, start=ts, end=te, **best_kw)
            st = calc_stats(trades)
            kf_sharpes.append(round(st['sharpe'], 3))

            bl_trades = bt_h1_strategy(h1, cfg, strategy=strat, start=ts, end=te)
            bl_st = calc_stats(bl_trades)
            kf_bl_sharpes.append(round(bl_st['sharpe'], 3))

            print(f'      Fold {fold+1}: best={st["sharpe"]:.3f} bl={bl_st["sharpe"]:.3f}')

        kf_wins = sum(1 for s, b in zip(kf_sharpes, kf_bl_sharpes) if s >= b)
        kf_pass = kf_wins >= 4
        print(f'    KF: {kf_wins}/6  PASS={kf_pass}')

        # Gate 2: Walk-Forward
        print(f'    Gate 2: Walk-Forward...')
        wf_wins = 0
        wf_details = []
        for _, _, test_s, test_e in WF_WINDOWS:
            trades = bt_h1_strategy(h1, cfg, strategy=strat, start=test_s, end=test_e, **best_kw)
            st = calc_stats(trades)
            bl_trades = bt_h1_strategy(h1, cfg, strategy=strat, start=test_s, end=test_e)
            bl_st = calc_stats(bl_trades)
            win = st['sharpe'] >= bl_st['sharpe']
            if win:
                wf_wins += 1
            wf_details.append({'window': test_s, 'sharpe': st['sharpe'], 'bl': bl_st['sharpe'], 'win': win})

        wf_pass = wf_wins >= len(WF_WINDOWS) * 0.6
        print(f'    WF: {wf_wins}/{len(WF_WINDOWS)}  PASS={wf_pass}')

        # Gate 3: Era Stability
        print(f'    Gate 3: Era Stability...')
        era_results = []
        for era_name, (es, ee) in ERA_SEGMENTS.items():
            trades = bt_h1_strategy(h1, cfg, strategy=strat, start=es, end=ee, **best_kw)
            st = calc_stats(trades)
            bl_trades = bt_h1_strategy(h1, cfg, strategy=strat, start=es, end=ee)
            bl_st = calc_stats(bl_trades)
            era_results.append({
                'era': era_name, 'sharpe': st['sharpe'], 'bl_sharpe': bl_st['sharpe'],
                'n': st['n'], 'pnl': st['pnl'],
            })
            print(f'      {era_name}: best={st["sharpe"]:.3f} bl={bl_st["sharpe"]:.3f}  n={st["n"]}')

        era_sharpes = [e['sharpe'] for e in era_results]
        era_pass = all(s > 0 for s in era_sharpes) and min(era_sharpes) > 0.5
        print(f'    Era: min={min(era_sharpes):.3f}  PASS={era_pass}')

        overall = kf_pass and wf_pass and era_pass
        phase3[strat] = {
            'best_label': best['label'],
            'best_kw': best_kw,
            'kfold': {'wins': kf_wins, 'sharpes': kf_sharpes, 'bl_sharpes': kf_bl_sharpes, 'pass': kf_pass},
            'walk_forward': {'wins': wf_wins, 'total': len(WF_WINDOWS), 'pass': wf_pass},
            'era': {'results': era_results, 'pass': era_pass},
            'overall_pass': overall,
        }
        tag = '[GO]' if overall else '[NO-GO]'
        print(f'    Overall: {tag}')

    save('phase3_three_gate', phase3)

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Kill-or-Keep Decision
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 4: Kill-or-Keep Decision')
    print('=' * 80)

    phase4 = {}
    for strat in strategies:
        bl = phase1[strat]
        recent = bl['era'].get('Recent (2024-2026)', {})
        full = bl['full']
        gate = phase3[strat]

        # Decision criteria:
        # KILL if: recent Sharpe < 0, OR full Sharpe < 1.0, OR consistently losing
        # TUNE if: 3-gate pass for new params
        # KEEP if: current params fine

        recent_sharpe = recent.get('sharpe', 0)
        full_sharpe = full.get('sharpe', 0)
        recent_pnl = recent.get('pnl', 0)
        full_pnl = full.get('pnl', 0)

        if full_sharpe < 0.5 or (recent_sharpe < 0 and recent_pnl < -100):
            decision = 'KILL'
            reason = f'Poor performance: full Sharpe={full_sharpe:.3f}, recent Sharpe={recent_sharpe:.3f}'
        elif gate['overall_pass']:
            decision = 'TUNE'
            reason = f'Better params found: {gate["best_label"]} passes 3-gate'
        elif full_sharpe > 2.0 and recent_sharpe > 0:
            decision = 'KEEP'
            reason = f'Current params adequate: full Sharpe={full_sharpe:.3f}'
        else:
            decision = 'REVIEW'
            reason = f'Marginal: full Sharpe={full_sharpe:.3f}, recent={recent_sharpe:.3f}'

        phase4[strat] = {
            'decision': decision,
            'reason': reason,
            'full_sharpe': full_sharpe,
            'recent_sharpe': recent_sharpe,
            'full_pnl': full_pnl,
            'recent_pnl': recent_pnl,
            'gate_pass': gate['overall_pass'],
        }
        print(f'  {strat.upper():<15} {decision:<8} {reason}')

    save('phase4_kill_or_keep', phase4)

    # ═══════════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('FINAL SUMMARY')
    print('=' * 80)

    summary = {
        'baseline': {s: {'sharpe': phase1[s]['full']['sharpe'], 'pnl': phase1[s]['full']['pnl'],
                         'n': phase1[s]['full']['n']}
                     for s in strategies},
        'decisions': phase4,
        'gate_results': {s: {'pass': phase3[s]['overall_pass'], 'best': phase3[s]['best_label']}
                         for s in strategies},
    }

    for strat in strategies:
        d = phase4[strat]
        print(f'  {strat.upper():<15} {d["decision"]:<8} '
              f'Full Sharpe={d["full_sharpe"]:.3f}  Recent={d["recent_sharpe"]:.3f}  '
              f'PnL=${d["full_pnl"]:.0f}')

    save('R209_summary', summary)

    elapsed = time.time() - t_start
    print(f'\n  Total runtime: {elapsed:.0f}s')
    print(f'  All results in {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
