#!/usr/bin/env python3
"""R205: TSMOM Standalone Parameter Deep-Validation
=====================================================
R203/R203b diagnosed that TSMOM signals fire at ~7/month rate and 53% pass
the filter cascade. But the core parameters have NEVER been validated with
the same rigor we apply to Keltner (kfold + walk-forward + era).

This experiment:
  Phase 1 — Grid Search over key TSMOM parameters:
    - fast_lb / slow_lb: momentum lookback (current 480/720 H1 bars)
    - sl_atr_mult / tp_atr_mult: SL/TP sizing (current 4.5/6.0)
    - max_hold_bars: max hold (current 20 H1 bars)
    - rule_b_sigma: ATR spike filter threshold (current 3.0)
    ~45 combos, evaluate PnL/trade + Sharpe approx on full 2015-2026

  Phase 2 — 3-Gate Validation on top-N candidates:
    - 6-fold cross-validation
    - 19-window walk-forward
    - 4-era stability

  Phase 3 — Sensitivity analysis: how robust are top params to ±10% perturbation

Run: python experiments/run_r205_tsmom_param_validation.py
"""
from __future__ import annotations
import sys
import json
import time
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.runner import load_csv, H1_CSV_PATH

OUTPUT_DIR = Path("results/r205_tsmom_params")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOT = 0.04
PV = 100  # $100 per 1.0 price unit for lot=0.04

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


def backtest_tsmom(
    h1: pd.DataFrame,
    atr: np.ndarray,
    score: np.ndarray,
    *,
    sl_atr_mult: float,
    tp_atr_mult: float,
    max_hold: int,
    rule_b_sigma: float,
    rule_b_lb: int = 60,
    rule_b_skip: int = 8,
    atr_floor: float = 0.1,
    min_gap_hours: float = 2.0,
    slow_lb: int = 720,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> List[Dict]:
    """Standalone TSMOM backtest, returns list of trades."""
    n = len(h1)
    close = h1['Close'].values
    high  = h1['High'].values
    low   = h1['Low'].values
    ts    = h1.index

    start_idx = 0
    end_idx = n
    if start:
        start_idx = max(ts.searchsorted(pd.Timestamp(start, tz='UTC')), slow_lb + 2)
    else:
        start_idx = slow_lb + 2
    if end:
        end_idx = ts.searchsorted(pd.Timestamp(end, tz='UTC'))

    rb_skip = 0
    last_entry_time = pd.Timestamp('1970-01-01', tz='UTC')
    in_pos = False
    pos_dir = 0
    pos_bars = 0
    pos_entry_price = 0.0
    pos_entry_atr = 0.0
    pos_extreme = 0.0
    pos_entry_time = pd.Timestamp('1970-01-01', tz='UTC')

    trades = []

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
                # Score-flip exit
                cur_score = score[i]
                if pos_dir == 1 and cur_score < 0:
                    reason = 'ScoreFlip'
                if pos_dir == -1 and cur_score > 0:
                    reason = 'ScoreFlip'

            if reason:
                final_pnl = (exit_price - pos_entry_price) if pos_dir == 1 else (pos_entry_price - exit_price)
                trades.append({
                    'entry_time': pos_entry_time,
                    'exit_time': ts[i],
                    'direction': 'BUY' if pos_dir == 1 else 'SELL',
                    'entry_price': pos_entry_price,
                    'exit_price': exit_price,
                    'pnl': final_pnl,
                    'pnl_usd': final_pnl * LOT * PV,
                    'bars_held': pos_bars,
                    'reason': reason,
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

        # Signal
        if i < 1:
            continue
        s_now = score[i]
        s_prev = score[i - 1]
        sig = None
        if s_now > 0 and s_prev <= 0:
            sig = 1
        elif s_now < 0 and s_prev >= 0:
            sig = -1
        if sig is None:
            continue

        # Filters
        if in_pos:
            continue
        if rb_skip > 0:
            continue
        if (ts[i] - last_entry_time).total_seconds() < min_gap_hours * 3600:
            continue
        if cur_atr < atr_floor:
            continue

        # Entry
        in_pos = True
        pos_dir = sig
        pos_entry_atr = cur_atr
        pos_extreme = close[i]
        pos_entry_price = close[i]
        pos_bars = 0
        pos_entry_time = ts[i]
        last_entry_time = ts[i]

    return trades


def calc_sharpe(trades: List[Dict], annualize: float = 252.0 * 4.0) -> float:
    """Approx Sharpe from per-trade PnL in USD."""
    if len(trades) < 5:
        return 0.0
    pnls = np.array([t['pnl_usd'] for t in trades])
    std = pnls.std()
    if std < 1e-9:
        return 0.0
    return float(pnls.mean() / std * np.sqrt(annualize))


def calc_stats_from_trades(trades: List[Dict]) -> Dict:
    if not trades:
        return {'n': 0, 'pnl': 0, 'sharpe': 0, 'win_rate': 0, 'avg_pnl': 0, 'max_dd': 0}
    pnls = np.array([t['pnl_usd'] for t in trades])
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    dd = running_max - cumsum
    reasons = {}
    for t in trades:
        reasons[t['reason']] = reasons.get(t['reason'], 0) + 1
    return {
        'n': len(trades),
        'pnl': round(float(pnls.sum()), 2),
        'sharpe': round(calc_sharpe(trades), 3),
        'win_rate': round(100 * (pnls > 0).sum() / len(pnls), 2),
        'avg_pnl': round(float(pnls.mean()), 4),
        'max_dd': round(float(dd.max()), 2),
        'reasons': reasons,
    }


def save(name: str, data):
    p = OUTPUT_DIR / f'{name}.json'
    with open(p, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f'  -> saved {p}')


def main():
    t_start = time.time()
    print('=' * 80)
    print('R205: TSMOM Standalone Parameter Deep-Validation')
    print('=' * 80)

    print('\nLoading H1 data...')
    h1 = load_csv(str(H1_CSV_PATH))
    print(f'  H1: {len(h1):,} bars, {h1.index[0]} -> {h1.index[-1]}')

    print('Computing ATR (Wilder-14)...')
    atr = compute_atr(h1)

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Grid Search
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 1: Parameter Grid Search')
    print('=' * 80)

    FAST_LBS  = [360, 480, 600]        # 15d, 20d, 25d
    SLOW_LBS  = [600, 720, 960]        # 25d, 30d, 40d
    SL_MULTS  = [3.5, 4.5, 5.5]
    TP_MULTS  = [5.0, 6.0, 8.0]
    MAX_HOLDS = [15, 20, 30]
    RB_SIGMAS = [2.5, 3.0, 3.5]

    # We'll do a structured grid: lookbacks x exit_params x filters
    # But full cartesian is 3^6=729, too many. Use layered approach:
    # Layer A: sweep lookbacks with baseline exit params
    # Layer B: sweep exit params (SL/TP/max_hold) with best lookbacks
    # Layer C: sweep Rule B sigma with best from A+B

    # --- Layer A: Lookback sweep ---
    print('\n  --- Layer A: Lookback Sweep (9 combos) ---')
    base_params = {'sl_atr_mult': 4.5, 'tp_atr_mult': 6.0,
                   'max_hold': 20, 'rule_b_sigma': 3.0}

    layer_a_results = []
    for fast, slow in itertools.product(FAST_LBS, SLOW_LBS):
        if fast >= slow:
            continue
        label = f'LB_{fast}_{slow}'
        print(f'    {label}...', end=' ', flush=True)
        score = compute_score(h1['Close'].values, fast, slow)
        trades = backtest_tsmom(h1, atr, score, slow_lb=slow, **base_params)
        st = calc_stats_from_trades(trades)
        st['label'] = label
        st['fast_lb'] = fast
        st['slow_lb'] = slow
        layer_a_results.append(st)
        print(f'n={st["n"]}  PnL=${st["pnl"]:.0f}  Sharpe={st["sharpe"]:.3f}  WR={st["win_rate"]:.1f}%')

    layer_a_results.sort(key=lambda x: x['sharpe'], reverse=True)
    best_fast = layer_a_results[0]['fast_lb']
    best_slow = layer_a_results[0]['slow_lb']
    print(f'\n  Best lookback: fast={best_fast}, slow={best_slow}  '
          f'(Sharpe={layer_a_results[0]["sharpe"]:.3f})')

    save('phase1_layer_a_lookbacks', layer_a_results)

    # --- Layer B: Exit param sweep with best lookbacks ---
    print(f'\n  --- Layer B: Exit Params Sweep (LB={best_fast}/{best_slow}) ---')
    score_best = compute_score(h1['Close'].values, best_fast, best_slow)

    layer_b_results = []
    for sl, tp, mh in itertools.product(SL_MULTS, TP_MULTS, MAX_HOLDS):
        if tp <= sl:
            continue
        label = f'SL{sl}_TP{tp}_MH{mh}'
        print(f'    {label}...', end=' ', flush=True)
        trades = backtest_tsmom(h1, atr, score_best, slow_lb=best_slow,
                                sl_atr_mult=sl, tp_atr_mult=tp,
                                max_hold=mh, rule_b_sigma=3.0)
        st = calc_stats_from_trades(trades)
        st['label'] = label
        st['sl_atr'] = sl
        st['tp_atr'] = tp
        st['max_hold'] = mh
        layer_b_results.append(st)
        print(f'n={st["n"]}  PnL=${st["pnl"]:.0f}  Sharpe={st["sharpe"]:.3f}  WR={st["win_rate"]:.1f}%')

    layer_b_results.sort(key=lambda x: x['sharpe'], reverse=True)
    best_sl = layer_b_results[0]['sl_atr']
    best_tp = layer_b_results[0]['tp_atr']
    best_mh = layer_b_results[0]['max_hold']
    print(f'\n  Best exit: SL={best_sl}, TP={best_tp}, MH={best_mh}  '
          f'(Sharpe={layer_b_results[0]["sharpe"]:.3f})')

    save('phase1_layer_b_exits', layer_b_results)

    # --- Layer C: Rule B sigma sweep ---
    print(f'\n  --- Layer C: Rule B Sigma Sweep ---')
    layer_c_results = []
    for rbs in RB_SIGMAS:
        label = f'RB{rbs}'
        print(f'    {label}...', end=' ', flush=True)
        trades = backtest_tsmom(h1, atr, score_best, slow_lb=best_slow,
                                sl_atr_mult=best_sl, tp_atr_mult=best_tp,
                                max_hold=best_mh, rule_b_sigma=rbs)
        st = calc_stats_from_trades(trades)
        st['label'] = label
        st['rule_b_sigma'] = rbs
        layer_c_results.append(st)
        print(f'n={st["n"]}  PnL=${st["pnl"]:.0f}  Sharpe={st["sharpe"]:.3f}')

    layer_c_results.sort(key=lambda x: x['sharpe'], reverse=True)
    best_rbs = layer_c_results[0]['rule_b_sigma']
    print(f'\n  Best Rule B sigma: {best_rbs}')

    save('phase1_layer_c_ruleb', layer_c_results)

    # Assemble top candidate
    top_params = {
        'fast_lb': best_fast, 'slow_lb': best_slow,
        'sl_atr_mult': best_sl, 'tp_atr_mult': best_tp,
        'max_hold': best_mh, 'rule_b_sigma': best_rbs,
    }
    baseline_params = {
        'fast_lb': 480, 'slow_lb': 720,
        'sl_atr_mult': 4.5, 'tp_atr_mult': 6.0,
        'max_hold': 20, 'rule_b_sigma': 3.0,
    }

    # Also grab the 2nd-best lookback for 2nd candidate
    alt_fast = layer_a_results[1]['fast_lb'] if len(layer_a_results) > 1 else best_fast
    alt_slow = layer_a_results[1]['slow_lb'] if len(layer_a_results) > 1 else best_slow
    alt_params = dict(top_params)
    alt_params['fast_lb'] = alt_fast
    alt_params['slow_lb'] = alt_slow

    candidates = {
        'baseline': baseline_params,
        'candidate_A': top_params,
        'candidate_B': alt_params,
    }

    print('\n  Candidates for 3-Gate:')
    for name, p in candidates.items():
        print(f'    {name}: {p}')

    save('phase1_candidates', candidates)

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: 3-Gate Validation
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 2: 3-Gate Validation')
    print('=' * 80)

    gate_results = {}
    for cand_name, params in candidates.items():
        print(f'\n  === {cand_name} ===')
        score_c = compute_score(h1['Close'].values, params['fast_lb'], params['slow_lb'])

        # Gate 1: 6-Fold Cross-Validation
        print('    Gate 1: 6-Fold CV...')
        n_bars = len(h1)
        fold_size = n_bars // 6
        kf_sharpes = []
        kf_baseline_sharpes = []
        for fold in range(6):
            test_start_idx = fold * fold_size
            test_end_idx = min((fold + 1) * fold_size, n_bars)
            test_start = str(h1.index[test_start_idx])[:10]
            test_end = str(h1.index[min(test_end_idx, n_bars - 1)])[:10]

            trades = backtest_tsmom(h1, atr, score_c, slow_lb=params['slow_lb'],
                                    sl_atr_mult=params['sl_atr_mult'],
                                    tp_atr_mult=params['tp_atr_mult'],
                                    max_hold=params['max_hold'],
                                    rule_b_sigma=params['rule_b_sigma'],
                                    start=test_start, end=test_end)
            sh = calc_sharpe(trades)
            kf_sharpes.append(round(sh, 3))

            # Baseline comparison
            score_bl = compute_score(h1['Close'].values, 480, 720)
            bl_trades = backtest_tsmom(h1, atr, score_bl, slow_lb=720,
                                       sl_atr_mult=4.5, tp_atr_mult=6.0,
                                       max_hold=20, rule_b_sigma=3.0,
                                       start=test_start, end=test_end)
            bl_sh = calc_sharpe(bl_trades)
            kf_baseline_sharpes.append(round(bl_sh, 3))

            print(f'      Fold {fold+1}: Sharpe={sh:.3f} (baseline={bl_sh:.3f})')

        kf_wins = sum(1 for s, b in zip(kf_sharpes, kf_baseline_sharpes) if s >= b)
        kf_pass = kf_wins >= 4
        print(f'    KF result: wins={kf_wins}/6  PASS={kf_pass}')

        # Gate 2: Walk-Forward
        print('    Gate 2: Walk-Forward...')
        wf_wins = 0
        wf_details = []
        for train_s, train_e, test_s, test_e in WF_WINDOWS:
            trades = backtest_tsmom(h1, atr, score_c, slow_lb=params['slow_lb'],
                                    sl_atr_mult=params['sl_atr_mult'],
                                    tp_atr_mult=params['tp_atr_mult'],
                                    max_hold=params['max_hold'],
                                    rule_b_sigma=params['rule_b_sigma'],
                                    start=test_s, end=test_e)
            sh = calc_sharpe(trades)

            score_bl = compute_score(h1['Close'].values, 480, 720)
            bl_trades = backtest_tsmom(h1, atr, score_bl, slow_lb=720,
                                       sl_atr_mult=4.5, tp_atr_mult=6.0,
                                       max_hold=20, rule_b_sigma=3.0,
                                       start=test_s, end=test_e)
            bl_sh = calc_sharpe(bl_trades)

            win = sh >= bl_sh
            if win:
                wf_wins += 1
            wf_details.append({'window': test_s, 'sharpe': round(sh, 3),
                               'baseline_sharpe': round(bl_sh, 3), 'win': win})

        wf_pass = wf_wins >= len(WF_WINDOWS) * 0.6
        print(f'    WF result: wins={wf_wins}/{len(WF_WINDOWS)}  PASS={wf_pass}')

        # Gate 3: Era Stability
        print('    Gate 3: Era Stability...')
        era_results = []
        for era_name, (es, ee) in ERA_SEGMENTS.items():
            trades = backtest_tsmom(h1, atr, score_c, slow_lb=params['slow_lb'],
                                    sl_atr_mult=params['sl_atr_mult'],
                                    tp_atr_mult=params['tp_atr_mult'],
                                    max_hold=params['max_hold'],
                                    rule_b_sigma=params['rule_b_sigma'],
                                    start=es, end=ee)
            sh = calc_sharpe(trades)
            era_results.append({'era': era_name, 'sharpe': round(sh, 3),
                                'n_trades': len(trades)})
            print(f'      {era_name}: Sharpe={sh:.3f}  n={len(trades)}')

        era_sharpes = [e['sharpe'] for e in era_results]
        era_pass = all(s > 0 for s in era_sharpes) and min(era_sharpes) > 0.5
        print(f'    Era result: min_sharpe={min(era_sharpes):.3f}  PASS={era_pass}')

        overall = kf_pass and wf_pass and era_pass
        gate_results[cand_name] = {
            'params': params,
            'kfold': {'sharpes': kf_sharpes, 'baseline_sharpes': kf_baseline_sharpes,
                      'wins': kf_wins, 'pass': kf_pass},
            'walk_forward': {'wins': wf_wins, 'total': len(WF_WINDOWS),
                             'details': wf_details, 'pass': wf_pass},
            'era': {'results': era_results, 'pass': era_pass},
            'overall_pass': overall,
        }
        tag = '[GO]' if overall else '[NO-GO]'
        print(f'    Overall: {tag}  (KF={kf_pass} WF={wf_pass} Era={era_pass})')

    save('phase2_three_gate', gate_results)

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: Sensitivity Analysis on top candidate
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 3: Sensitivity Analysis')
    print('=' * 80)

    # Pick best passing candidate
    go_candidates = [k for k, v in gate_results.items() if v['overall_pass'] and k != 'baseline']
    if not go_candidates:
        print('  No [GO] candidates, testing top candidate anyway')
        go_candidates = [k for k in gate_results if k != 'baseline']

    if go_candidates:
        best_name = go_candidates[0]
        best_p = gate_results[best_name]['params']
        print(f'  Testing sensitivity of: {best_name}')

        perturb_results = []
        perturb_keys = ['sl_atr_mult', 'tp_atr_mult', 'max_hold']
        for key in perturb_keys:
            base_val = best_p[key]
            for delta_pct in [-0.20, -0.10, 0, 0.10, 0.20]:
                perturbed = dict(best_p)
                if key == 'max_hold':
                    perturbed[key] = max(5, int(base_val * (1 + delta_pct)))
                else:
                    perturbed[key] = round(base_val * (1 + delta_pct), 3)

                score_p = compute_score(h1['Close'].values,
                                        perturbed['fast_lb'], perturbed['slow_lb'])
                trades = backtest_tsmom(h1, atr, score_p,
                                        slow_lb=perturbed['slow_lb'],
                                        sl_atr_mult=perturbed['sl_atr_mult'],
                                        tp_atr_mult=perturbed['tp_atr_mult'],
                                        max_hold=perturbed['max_hold'],
                                        rule_b_sigma=perturbed['rule_b_sigma'])
                st = calc_stats_from_trades(trades)
                st['param'] = key
                st['delta_pct'] = delta_pct
                st['value'] = perturbed[key]
                perturb_results.append(st)
                print(f'    {key}={perturbed[key]} ({delta_pct:+.0%}): '
                      f'Sharpe={st["sharpe"]:.3f}  PnL=${st["pnl"]:.0f}')

        save('phase3_sensitivity', perturb_results)
    else:
        print('  No candidates to test')

    # ═══════════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Final Summary')
    print('=' * 80)

    # Full-period stats for each candidate
    summary = {}
    for cand_name, params in candidates.items():
        score_c = compute_score(h1['Close'].values, params['fast_lb'], params['slow_lb'])
        trades = backtest_tsmom(h1, atr, score_c, slow_lb=params['slow_lb'],
                                sl_atr_mult=params['sl_atr_mult'],
                                tp_atr_mult=params['tp_atr_mult'],
                                max_hold=params['max_hold'],
                                rule_b_sigma=params['rule_b_sigma'])
        st = calc_stats_from_trades(trades)
        st['params'] = params
        st['three_gate'] = gate_results.get(cand_name, {}).get('overall_pass', False)
        summary[cand_name] = st

        tag = '[GO]' if st['three_gate'] else '[NO-GO]'
        print(f'  {cand_name} {tag}:')
        print(f'    Params: LB={params["fast_lb"]}/{params["slow_lb"]}  '
              f'SL={params["sl_atr_mult"]}  TP={params["tp_atr_mult"]}  '
              f'MH={params["max_hold"]}  RB={params["rule_b_sigma"]}')
        print(f'    Trades={st["n"]}  PnL=${st["pnl"]:.0f}  Sharpe={st["sharpe"]:.3f}  '
              f'WR={st["win_rate"]:.1f}%  MaxDD=${st["max_dd"]:.0f}')

    save('R205_summary', summary)

    elapsed = time.time() - t_start
    print(f'\n  Total runtime: {elapsed:.0f}s')
    print(f'  All results in {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
