#!/usr/bin/env python3
"""R206: TSMOM R205-Candidate Robustness Verification
======================================================
R205 found candidate_A (LB=480/960, SL=3.5, TP=8.0, MH=30, RB=2.5)
passes 3-gate with Sharpe 4.16 vs baseline 1.08.

Three concerns remain before shadow deployment:

Phase 1 — M1-Resolution SL/TP Verification
  H1 backtest has OHLC ordering bias just like M15: when both SL and TP
  are reachable within one H1 bar, the backtest must guess which triggers
  first. This systematically overstates TP hits (or SL hits, depending on
  implementation). Replay each trade at M1 resolution to get true exit.

Phase 2 — Monte Carlo Bootstrap Confidence
  With only 623 trades (vs 785 baseline), is the Sharpe improvement
  statistically significant? Bootstrap 10,000 resamples of trade PnL to
  get CI on Sharpe and test P(candidate > baseline).

Phase 3 — Warmup Sensitivity
  slow_lb=960 H1 bars = 40 trading days. Test:
  a) Does the strategy survive if we start from 2016 instead of 2015?
  b) What happens with slow_lb=900, 960, 1020 (±5%) at each era?
  c) Score correlation between slow_lb=720 and slow_lb=960 — how often
     do they agree on direction?

Run: python experiments/run_r206_tsmom_robustness.py
"""
from __future__ import annotations
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.runner import load_csv, H1_CSV_PATH

OUTPUT_DIR = Path("results/r206_tsmom_robustness")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOT = 0.04
PV = 100

M1_CANDIDATES = [
    Path('data/download/xauusd-m1-bid-2015-01-01-2026-05-06.csv'),
    Path('data/download/xauusd-m1-bid-2015-01-01-2026-04-27.csv'),
    Path('data/download/xauusd-m1-bid-2015-01-01-2026-04-10.csv'),
]

BASELINE = {'fast_lb': 480, 'slow_lb': 720, 'sl_atr_mult': 4.5,
            'tp_atr_mult': 6.0, 'max_hold': 20, 'rule_b_sigma': 3.0}
CANDIDATE = {'fast_lb': 480, 'slow_lb': 960, 'sl_atr_mult': 3.5,
             'tp_atr_mult': 8.0, 'max_hold': 30, 'rule_b_sigma': 2.5}

ERA_SEGMENTS = {
    "Pre-COVID (2015-2019)":      ("2015-01-01", "2020-01-01"),
    "COVID+Recovery (2020-2021)": ("2020-01-01", "2022-01-01"),
    "Tightening (2022-2023)":     ("2022-01-01", "2024-01-01"),
    "Recent (2024-2026)":         ("2024-01-01", "2026-06-01"),
}


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


def backtest_tsmom(h1, atr, score, *, sl_atr_mult, tp_atr_mult, max_hold,
                   rule_b_sigma, rule_b_lb=60, rule_b_skip=8, atr_floor=0.1,
                   min_gap_hours=2.0, slow_lb=720, start=None, end=None):
    n = len(h1)
    close = h1['Close'].values
    high  = h1['High'].values
    low   = h1['Low'].values
    ts    = h1.index

    start_idx = max(ts.searchsorted(pd.Timestamp(start, tz='UTC')), slow_lb + 2) if start else slow_lb + 2
    end_idx = ts.searchsorted(pd.Timestamp(end, tz='UTC')) if end else n

    rb_skip = 0
    last_entry_time = pd.Timestamp('1970-01-01', tz='UTC')
    in_pos = False
    pos_dir = 0; pos_bars = 0; pos_entry_price = 0.0
    pos_entry_atr = 0.0; pos_entry_time = pd.Timestamp('1970-01-01', tz='UTC')
    trades = []

    for i in range(start_idx, min(end_idx, n)):
        cur_atr = atr[i]
        if cur_atr <= 0:
            continue

        if in_pos:
            pos_bars += 1
            c = close[i]
            tp_dist = tp_atr_mult * pos_entry_atr
            sl_dist = sl_atr_mult * pos_entry_atr
            pnl = (c - pos_entry_price) if pos_dir == 1 else (pos_entry_price - c)
            reason = None; exit_price = c
            if pnl >= tp_dist:
                reason = 'TP'
                exit_price = pos_entry_price + (tp_dist if pos_dir == 1 else -tp_dist)
            elif pnl <= -sl_dist:
                reason = 'SL'
                exit_price = pos_entry_price + (-sl_dist if pos_dir == 1 else sl_dist)
            elif pos_bars >= max_hold:
                reason = 'Timeout'
            else:
                cur_score = score[i]
                if pos_dir == 1 and cur_score < 0:
                    reason = 'ScoreFlip'
                if pos_dir == -1 and cur_score > 0:
                    reason = 'ScoreFlip'
            if reason:
                final_pnl = (exit_price - pos_entry_price) if pos_dir == 1 else (pos_entry_price - exit_price)
                trades.append({
                    'entry_time': pos_entry_time, 'exit_time': ts[i],
                    'direction': 'BUY' if pos_dir == 1 else 'SELL',
                    'entry_price': pos_entry_price, 'exit_price': exit_price,
                    'pnl': final_pnl, 'pnl_usd': final_pnl * LOT * PV,
                    'bars_held': pos_bars, 'reason': reason,
                    'entry_atr': pos_entry_atr,
                })
                in_pos = False

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

        if i < 1:
            continue
        s_now = score[i]; s_prev = score[i - 1]
        sig = None
        if s_now > 0 and s_prev <= 0: sig = 1
        elif s_now < 0 and s_prev >= 0: sig = -1
        if sig is None: continue
        if in_pos: continue
        if rb_skip > 0: continue
        if (ts[i] - last_entry_time).total_seconds() < min_gap_hours * 3600: continue
        if cur_atr < atr_floor: continue

        in_pos = True; pos_dir = sig
        pos_entry_atr = cur_atr; pos_entry_price = close[i]
        pos_bars = 0; pos_entry_time = ts[i]; last_entry_time = ts[i]
    return trades


def calc_sharpe(trades, annualize=252.0 * 4.0):
    if len(trades) < 5: return 0.0
    pnls = np.array([t['pnl_usd'] for t in trades])
    std = pnls.std()
    if std < 1e-9: return 0.0
    return float(pnls.mean() / std * np.sqrt(annualize))


def calc_stats(trades):
    if not trades:
        return {'n': 0, 'pnl': 0, 'sharpe': 0, 'win_rate': 0, 'max_dd': 0}
    pnls = np.array([t['pnl_usd'] for t in trades])
    cumsum = np.cumsum(pnls)
    dd = np.maximum.accumulate(cumsum) - cumsum
    reasons = {}
    for t in trades: reasons[t['reason']] = reasons.get(t['reason'], 0) + 1
    return {
        'n': len(trades), 'pnl': round(float(pnls.sum()), 2),
        'sharpe': round(calc_sharpe(trades), 3),
        'win_rate': round(100 * (pnls > 0).sum() / len(pnls), 2),
        'avg_pnl': round(float(pnls.mean()), 4),
        'max_dd': round(float(dd.max()), 2),
        'reasons': reasons,
    }


def save(name, data):
    p = OUTPUT_DIR / f'{name}.json'
    with open(p, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f'  -> saved {p}')


def find_m1():
    for p in M1_CANDIDATES:
        if p.exists():
            return p
    return None


def load_m1(path):
    with open(path, 'r') as f:
        header = f.readline().strip()
    if 'timestamp' in header.lower():
        return load_csv(str(path))
    print(f'  Detected Dukascopy format')
    df = pd.read_csv(path)
    if 'Gmt time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f', utc=True)
        df.drop(columns=['Gmt time'], inplace=True)
    elif 'GMT Time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['GMT Time'], format='%d.%m.%Y %H:%M:%S.%f', utc=True)
        df.drop(columns=['GMT Time'], inplace=True)
    else:
        raise ValueError(f'Unknown M1 format: {df.columns.tolist()}')
    df.set_index('timestamp', inplace=True)
    if 'Volume' not in df.columns: df['Volume'] = 0
    df['is_flat'] = (df['Open'] == df['High']) & (df['High'] == df['Low']) & (df['Low'] == df['Close'])
    return df


def replay_trade_m1(m1, trade, max_m1_bars=1800):
    """Replay a single TSMOM trade at M1 resolution for true SL/TP ordering."""
    entry_time = trade['entry_time']
    entry_price = trade['entry_price']
    entry_atr = trade['entry_atr']
    is_buy = trade['direction'] == 'BUY'
    sl_dist = trade['entry_atr'] * (CANDIDATE['sl_atr_mult'] if '_cand' in str(trade.get('config','')) else trade.get('sl_mult', CANDIDATE['sl_atr_mult']))
    tp_dist = trade['entry_atr'] * (CANDIDATE['tp_atr_mult'] if '_cand' in str(trade.get('config','')) else trade.get('tp_mult', CANDIDATE['tp_atr_mult']))
    max_hold_h1 = trade.get('max_hold_bars', 30)

    sl_price = entry_price - sl_dist if is_buy else entry_price + sl_dist
    tp_price = entry_price + tp_dist if is_buy else entry_price - tp_dist

    sub = m1[m1.index > entry_time]
    if len(sub) == 0:
        return {'m1_reason': 'no_data', 'm1_pnl': 0, 'm1_bars': 0}
    sub = sub.iloc[:max_m1_bars]
    high = sub['High'].values; low = sub['Low'].values; close = sub['Close'].values

    for j in range(len(sub)):
        if is_buy:
            if low[j] <= sl_price:
                return {'m1_reason': 'SL', 'm1_pnl': (sl_price - entry_price) * LOT * PV,
                        'm1_bars': j+1, 'm1_exit_price': sl_price}
            if high[j] >= tp_price:
                return {'m1_reason': 'TP', 'm1_pnl': (tp_price - entry_price) * LOT * PV,
                        'm1_bars': j+1, 'm1_exit_price': tp_price}
        else:
            if high[j] >= sl_price:
                return {'m1_reason': 'SL', 'm1_pnl': (entry_price - sl_price) * LOT * PV,
                        'm1_bars': j+1, 'm1_exit_price': sl_price}
            if low[j] <= tp_price:
                return {'m1_reason': 'TP', 'm1_pnl': (entry_price - tp_price) * LOT * PV,
                        'm1_bars': j+1, 'm1_exit_price': tp_price}

    # Neither SL nor TP hit within window — use original H1 exit
    return {'m1_reason': trade['reason'], 'm1_pnl': trade['pnl_usd'],
            'm1_bars': len(sub), 'm1_exit_price': trade['exit_price']}


def main():
    t_start = time.time()
    print('=' * 80)
    print('R206: TSMOM R205-Candidate Robustness Verification')
    print('=' * 80)

    print('\nLoading H1 data...')
    h1 = load_csv(str(H1_CSV_PATH))
    print(f'  H1: {len(h1):,} bars, {h1.index[0]} -> {h1.index[-1]}')
    atr = compute_atr(h1)

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: M1 SL/TP Resolution Verification
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 1: M1-Resolution SL/TP Verification')
    print('=' * 80)

    m1_path = find_m1()
    phase1 = {}

    if m1_path is None:
        print('  No M1 data available, skipping Phase 1')
        phase1 = {'status': 'skipped', 'reason': 'no M1 data'}
    else:
        print(f'  Loading M1: {m1_path}')
        m1 = load_m1(m1_path)
        print(f'  M1: {len(m1):,} bars, {m1.index[0]} -> {m1.index[-1]}')

        for config_label, params in [('baseline', BASELINE), ('candidate', CANDIDATE)]:
            print(f'\n  --- {config_label} ---')
            score = compute_score(h1['Close'].values, params['fast_lb'], params['slow_lb'])
            trades = backtest_tsmom(h1, atr, score, slow_lb=params['slow_lb'], **{k: v for k, v in params.items() if k != 'fast_lb' and k != 'slow_lb'})
            h1_stats = calc_stats(trades)
            print(f'    H1 backtest: n={h1_stats["n"]}  Sharpe={h1_stats["sharpe"]}  PnL=${h1_stats["pnl"]:.0f}')

            # Filter to M1 coverage
            m1_start = m1.index[0]; m1_end = m1.index[-1]
            m1_trades = [t for t in trades if m1_start <= t['entry_time'] <= m1_end]
            print(f'    Trades in M1 range: {len(m1_trades)}')

            # Only replay SL and TP trades (Timeout/ScoreFlip aren't affected by ordering)
            sltp_trades = [t for t in m1_trades if t['reason'] in ('SL', 'TP')]
            print(f'    SL/TP trades to replay: {len(sltp_trades)}')

            # Add params to trades for replay
            for t in sltp_trades:
                t['sl_mult'] = params['sl_atr_mult']
                t['tp_mult'] = params['tp_atr_mult']
                t['max_hold_bars'] = params['max_hold']

            m1_results = []
            for k, t in enumerate(sltp_trades):
                if (k + 1) % 50 == 0:
                    print(f'      replay {k+1}/{len(sltp_trades)}', flush=True)
                r = replay_trade_m1(m1, t, max_m1_bars=params['max_hold'] * 60)
                r['h1_reason'] = t['reason']
                r['h1_pnl'] = t['pnl_usd']
                m1_results.append(r)

            # Analysis: how often does M1 disagree with H1?
            flips = sum(1 for r in m1_results if r['m1_reason'] != r['h1_reason'])
            h1_tp_m1_sl = sum(1 for r in m1_results if r['h1_reason'] == 'TP' and r['m1_reason'] == 'SL')
            h1_sl_m1_tp = sum(1 for r in m1_results if r['h1_reason'] == 'SL' and r['m1_reason'] == 'TP')
            h1_pnl_sltp = sum(r['h1_pnl'] for r in m1_results)
            m1_pnl_sltp = sum(r['m1_pnl'] for r in m1_results)

            # Reconstruct full PnL using M1-corrected SL/TP trades
            non_sltp = [t for t in m1_trades if t['reason'] not in ('SL', 'TP')]
            full_m1_pnl = sum(t['pnl_usd'] for t in non_sltp) + m1_pnl_sltp
            full_h1_pnl = sum(t['pnl_usd'] for t in non_sltp) + h1_pnl_sltp

            result = {
                'h1_stats': h1_stats,
                'n_m1_trades': len(m1_trades),
                'n_sltp_replayed': len(sltp_trades),
                'flip_count': flips,
                'flip_pct': round(100 * flips / max(len(m1_results), 1), 2),
                'h1_tp_became_m1_sl': h1_tp_m1_sl,
                'h1_sl_became_m1_tp': h1_sl_m1_tp,
                'sltp_pnl_h1': round(h1_pnl_sltp, 2),
                'sltp_pnl_m1': round(m1_pnl_sltp, 2),
                'sltp_pnl_delta': round(m1_pnl_sltp - h1_pnl_sltp, 2),
                'full_pnl_h1': round(full_h1_pnl, 2),
                'full_pnl_m1': round(full_m1_pnl, 2),
                'full_pnl_delta': round(full_m1_pnl - full_h1_pnl, 2),
                'bias_direction': 'H1_optimistic' if full_h1_pnl > full_m1_pnl else 'H1_pessimistic',
            }
            phase1[config_label] = result

            print(f'    SL/TP flip rate: {result["flip_pct"]:.1f}% ({flips}/{len(m1_results)})')
            print(f'    H1 TP -> M1 SL: {h1_tp_m1_sl}   H1 SL -> M1 TP: {h1_sl_m1_tp}')
            print(f'    Full PnL H1: ${full_h1_pnl:.0f}  M1-corrected: ${full_m1_pnl:.0f}  delta: ${full_m1_pnl - full_h1_pnl:+.0f}')
            print(f'    Bias: {result["bias_direction"]}')

    save('phase1_m1_verification', phase1)

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Monte Carlo Bootstrap
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 2: Monte Carlo Bootstrap Confidence')
    print('=' * 80)

    N_BOOTSTRAP = 10000
    rng = np.random.RandomState(42)

    phase2 = {}
    for config_label, params in [('baseline', BASELINE), ('candidate', CANDIDATE)]:
        score = compute_score(h1['Close'].values, params['fast_lb'], params['slow_lb'])
        trades = backtest_tsmom(h1, atr, score, slow_lb=params['slow_lb'],
                                **{k: v for k, v in params.items() if k not in ('fast_lb', 'slow_lb')})
        pnls = np.array([t['pnl_usd'] for t in trades])
        n = len(pnls)
        orig_sharpe = calc_sharpe(trades)

        boot_sharpes = []
        boot_pnls = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.randint(0, n, size=n)
            sample = pnls[idx]
            std = sample.std()
            sh = sample.mean() / std * np.sqrt(252 * 4) if std > 1e-9 else 0.0
            boot_sharpes.append(sh)
            boot_pnls.append(sample.sum())
        boot_sharpes = np.array(boot_sharpes)
        boot_pnls = np.array(boot_pnls)

        result = {
            'n_trades': n,
            'orig_sharpe': round(orig_sharpe, 3),
            'orig_pnl': round(float(pnls.sum()), 2),
            'sharpe_mean': round(float(boot_sharpes.mean()), 3),
            'sharpe_ci_5': round(float(np.percentile(boot_sharpes, 2.5)), 3),
            'sharpe_ci_95': round(float(np.percentile(boot_sharpes, 97.5)), 3),
            'sharpe_std': round(float(boot_sharpes.std()), 3),
            'pnl_ci_5': round(float(np.percentile(boot_pnls, 2.5)), 2),
            'pnl_ci_95': round(float(np.percentile(boot_pnls, 97.5)), 2),
            'prob_positive_sharpe': round(float((boot_sharpes > 0).mean()), 4),
        }
        phase2[config_label] = result
        print(f'  {config_label}: Sharpe={orig_sharpe:.3f}  '
              f'95% CI=[{result["sharpe_ci_5"]:.3f}, {result["sharpe_ci_95"]:.3f}]  '
              f'P(Sharpe>0)={result["prob_positive_sharpe"]:.4f}')

    # Paired bootstrap: P(candidate > baseline)
    print('\n  Paired bootstrap: P(candidate Sharpe > baseline Sharpe)...')
    bl_score = compute_score(h1['Close'].values, BASELINE['fast_lb'], BASELINE['slow_lb'])
    bl_trades = backtest_tsmom(h1, atr, bl_score, slow_lb=BASELINE['slow_lb'],
                               **{k: v for k, v in BASELINE.items() if k not in ('fast_lb', 'slow_lb')})
    ca_score = compute_score(h1['Close'].values, CANDIDATE['fast_lb'], CANDIDATE['slow_lb'])
    ca_trades = backtest_tsmom(h1, atr, ca_score, slow_lb=CANDIDATE['slow_lb'],
                               **{k: v for k, v in CANDIDATE.items() if k not in ('fast_lb', 'slow_lb')})
    bl_pnls = np.array([t['pnl_usd'] for t in bl_trades])
    ca_pnls = np.array([t['pnl_usd'] for t in ca_trades])

    wins = 0
    for _ in range(N_BOOTSTRAP):
        bl_s = bl_pnls[rng.randint(0, len(bl_pnls), size=len(bl_pnls))]
        ca_s = ca_pnls[rng.randint(0, len(ca_pnls), size=len(ca_pnls))]
        bl_sh = bl_s.mean() / max(bl_s.std(), 1e-9) * np.sqrt(252 * 4)
        ca_sh = ca_s.mean() / max(ca_s.std(), 1e-9) * np.sqrt(252 * 4)
        if ca_sh > bl_sh:
            wins += 1
    p_better = wins / N_BOOTSTRAP
    phase2['paired_prob_candidate_wins'] = round(p_better, 4)
    print(f'  P(candidate > baseline) = {p_better:.4f}')

    save('phase2_bootstrap', phase2)

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: Warmup Sensitivity
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 3: Warmup & slow_lb Sensitivity')
    print('=' * 80)

    # 3a: Start date sensitivity (warmup impact)
    print('\n  --- 3a: Start date sensitivity ---')
    start_dates = ['2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01']
    start_results = []
    for sd in start_dates:
        for config_label, params in [('baseline', BASELINE), ('candidate', CANDIDATE)]:
            score = compute_score(h1['Close'].values, params['fast_lb'], params['slow_lb'])
            trades = backtest_tsmom(h1, atr, score, slow_lb=params['slow_lb'],
                                    start=sd,
                                    **{k: v for k, v in params.items() if k not in ('fast_lb', 'slow_lb')})
            st = calc_stats(trades)
            start_results.append({
                'start': sd, 'config': config_label,
                'n': st['n'], 'sharpe': st['sharpe'], 'pnl': st['pnl'],
            })
            print(f'    start={sd}  {config_label}: n={st["n"]}  Sharpe={st["sharpe"]:.3f}  PnL=${st["pnl"]:.0f}')

    # 3b: slow_lb perturbation per era
    print('\n  --- 3b: slow_lb perturbation ---')
    SLOW_LBS_TEST = [864, 912, 960, 1008, 1056]  # ±10% around 960
    slow_lb_results = []
    for sl in SLOW_LBS_TEST:
        score = compute_score(h1['Close'].values, 480, sl)
        trades = backtest_tsmom(h1, atr, score, slow_lb=sl,
                                sl_atr_mult=CANDIDATE['sl_atr_mult'],
                                tp_atr_mult=CANDIDATE['tp_atr_mult'],
                                max_hold=CANDIDATE['max_hold'],
                                rule_b_sigma=CANDIDATE['rule_b_sigma'])
        full = calc_stats(trades)
        era_sharpes = {}
        for era_name, (es, ee) in ERA_SEGMENTS.items():
            et = backtest_tsmom(h1, atr, score, slow_lb=sl,
                                sl_atr_mult=CANDIDATE['sl_atr_mult'],
                                tp_atr_mult=CANDIDATE['tp_atr_mult'],
                                max_hold=CANDIDATE['max_hold'],
                                rule_b_sigma=CANDIDATE['rule_b_sigma'],
                                start=es, end=ee)
            era_sharpes[era_name] = calc_sharpe(et)
        row = {'slow_lb': sl, 'n': full['n'], 'sharpe': full['sharpe'],
               'pnl': full['pnl'], 'era_sharpes': {k: round(v, 3) for k, v in era_sharpes.items()}}
        slow_lb_results.append(row)
        min_era = min(era_sharpes.values())
        print(f'    slow_lb={sl}: Sharpe={full["sharpe"]:.3f}  PnL=${full["pnl"]:.0f}  min_era={min_era:.3f}')

    # 3c: Score correlation between 720 and 960
    print('\n  --- 3c: Score direction agreement ---')
    s720 = compute_score(h1['Close'].values, 480, 720)
    s960 = compute_score(h1['Close'].values, 480, 960)
    valid = (s720 != 0) & (s960 != 0)
    agree = ((s720 > 0) & (s960 > 0)) | ((s720 < 0) & (s960 < 0))
    agree_pct = 100 * agree[valid].sum() / valid.sum()
    corr = float(np.corrcoef(s720[valid], s960[valid])[0, 1])
    score_agreement = {
        'agree_pct': round(agree_pct, 2),
        'correlation': round(corr, 4),
        'n_valid': int(valid.sum()),
    }
    print(f'    Direction agreement: {agree_pct:.1f}%  Correlation: {corr:.4f}')

    phase3 = {
        'start_date_sensitivity': start_results,
        'slow_lb_perturbation': slow_lb_results,
        'score_agreement_720_vs_960': score_agreement,
    }
    save('phase3_warmup_sensitivity', phase3)

    # ═══════════════════════════════════════════════════════════════
    # Final Verdict
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('FINAL VERDICT')
    print('=' * 80)

    verdict = {
        'phase1_m1_bias': phase1,
        'phase2_confidence': {
            'candidate_sharpe_ci': f"[{phase2['candidate']['sharpe_ci_5']}, {phase2['candidate']['sharpe_ci_95']}]",
            'p_candidate_better': phase2.get('paired_prob_candidate_wins', 'N/A'),
        },
        'phase3_robustness': {
            'min_era_sharpe_across_slow_lbs': min(min(r['era_sharpes'].values()) for r in slow_lb_results),
            'score_agreement_720_960': score_agreement['agree_pct'],
        },
    }

    m1_ok = True
    if isinstance(phase1, dict) and 'candidate' in phase1:
        bias = phase1['candidate'].get('bias_direction', '')
        delta = phase1['candidate'].get('full_pnl_delta', 0)
        pct_impact = abs(delta) / max(abs(phase1['candidate'].get('full_pnl_h1', 1)), 1) * 100
        verdict['m1_bias_pct'] = round(pct_impact, 2)
        m1_ok = pct_impact < 15  # <15% degradation is acceptable

    bootstrap_ok = phase2.get('paired_prob_candidate_wins', 0) > 0.80
    warmup_ok = all(min(r['era_sharpes'].values()) > 0 for r in slow_lb_results)

    overall = m1_ok and bootstrap_ok and warmup_ok
    verdict['m1_ok'] = m1_ok
    verdict['bootstrap_ok'] = bootstrap_ok
    verdict['warmup_ok'] = warmup_ok
    verdict['overall'] = 'GO_SHADOW' if overall else 'HOLD'

    tag = verdict['overall']
    print(f'  M1 bias check:     {"PASS" if m1_ok else "FAIL"}')
    print(f'  Bootstrap:         {"PASS" if bootstrap_ok else "FAIL"} (P={phase2.get("paired_prob_candidate_wins", "?")})')
    print(f'  Warmup robustness: {"PASS" if warmup_ok else "FAIL"}')
    print(f'\n  >>> VERDICT: {tag} <<<')

    save('R206_verdict', verdict)

    elapsed = time.time() - t_start
    print(f'\n  Total runtime: {elapsed:.0f}s')


if __name__ == '__main__':
    main()
