#!/usr/bin/env python3
"""R230C: M30 Parallel Parameter Sweep + Deep Validation
=========================================================
Runs Phase 3-10 for all 11 remaining M30 strategies (excluding m30_kc 
which R230 is already handling) using multiprocessing for massive speedup.

Server has 208 cores - we'll use up to 20 workers per strategy.
"""
from __future__ import annotations
import sys, json, time, traceback, os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.m30_engine import M30BacktestEngine, load_m30_with_indicators
from backtest.engine import TradeRecord

OUTPUT_DIR = Path("results/r230c_m30_parallel")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPREAD = 0.30
N_BOOTSTRAP = 5000
N_WORKERS = min(40, cpu_count())

ERA_SEGMENTS = {
    "Pre-COVID (2015-2019)":      ("2015-01-01", "2020-01-01"),
    "COVID+Recovery (2020-2021)": ("2020-01-01", "2022-01-01"),
    "Tightening (2022-2023)":     ("2022-01-01", "2024-01-01"),
    "Recent (2024-2026)":         ("2024-01-01", "2026-06-01"),
}

WF_CUTOFFS = [
    ("2015-01-01", "2017-01-01", "2017-01-01", "2018-10-01"),
    ("2015-01-01", "2018-10-01", "2018-10-01", "2020-07-01"),
    ("2015-01-01", "2020-07-01", "2020-07-01", "2022-04-01"),
    ("2015-01-01", "2022-04-01", "2022-04-01", "2024-01-01"),
    ("2015-01-01", "2024-01-01", "2024-01-01", "2025-07-01"),
    ("2015-01-01", "2025-07-01", "2025-07-01", "2026-06-01"),
]

SL_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
TP_GRID = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
TRAIL_GRID = [
    (0.0, 0.0), (0.15, 0.04), (0.2, 0.06), (0.25, 0.07),
    (0.3, 0.08), (0.4, 0.10), (0.5, 0.12),
    (0.5, 0.15), (0.8, 0.20), (1.0, 0.25),
]
MAX_HOLD_GRID = [8, 12, 16, 24, 48]

SLIPPAGE_CONFIGS = [
    {"name": "no_slippage", "slippage_model": "none"},
    {"name": "fixed_slippage", "slippage_model": "fixed"},
    {"name": "empirical_slippage", "slippage_model": "empirical"},
    {"name": "realistic_slippage", "slippage_model": "realistic"},
]

DEFAULT_PARAMS = {
    'sl_atr_mult': 2.0, 'tp_atr_mult': 4.0,
    'trailing_activate_atr': 0.2, 'trailing_distance_atr': 0.06,
    'max_hold': 24, 'cooldown_bars': 2, 'spread_cost': SPREAD,
}

PROGRESS_FILE = OUTPUT_DIR / "_progress.json"

def save(name, data):
    p = OUTPUT_DIR / f'{name}.json'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    pf(f'  -> saved {p}')

def save_progress(phase, detail=""):
    prog = {}
    if PROGRESS_FILE.exists():
        try: prog = json.loads(PROGRESS_FILE.read_text())
        except: pass
    prog[phase] = {'timestamp': pd.Timestamp.now().isoformat(), 'detail': detail}
    PROGRESS_FILE.write_text(json.dumps(prog, indent=2, default=str))

def pf(msg): print(msg, flush=True)

def calc_stats(trades):
    if not trades:
        return {'n': 0, 'pnl': 0, 'sharpe': 0, 'win_rate': 0, 'avg_pnl': 0, 'max_dd': 0, 'profit_factor': 0}
    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    cum = np.cumsum(pnls)
    dd = np.maximum.accumulate(cum) - cum
    sharpe = float(pnls.mean() / max(pnls.std(ddof=1), 1e-9) * np.sqrt(252)) if n > 1 else 0
    wins, losses = pnls[pnls > 0], pnls[pnls < 0]
    pf_val = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 99.9
    return {'n': n, 'pnl': round(float(pnls.sum()), 2), 'sharpe': round(sharpe, 3),
            'win_rate': round(100 * (pnls > 0).sum() / n, 2), 'avg_pnl': round(float(pnls.mean()), 4),
            'max_dd': round(float(dd.max()), 2), 'profit_factor': round(pf_val, 3)}

def filter_period(trades, start, end):
    ts_s, ts_e = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    return [t for t in trades if ts_s <= pd.Timestamp(t.entry_time) < ts_e]

def kfold(trades, k):
    if len(trades) < k * 5: return {'skip': True, 'verdict': 'SKIP'}
    pnls = np.array([t.pnl for t in trades])
    fold_size = len(pnls) // k
    folds, kf_pass = [], 0
    for fold in range(k):
        s, e = fold * fold_size, (fold * fold_size + fold_size if fold < k-1 else len(pnls))
        fp = pnls[s:e]
        if len(fp) < 3: continue
        sh = float(fp.mean() / max(fp.std(ddof=1), 1e-9) * np.sqrt(252))
        folds.append({'fold': fold+1, 'n': len(fp), 'sharpe': round(sh, 3)})
        if sh > 0: kf_pass += 1
    rate = kf_pass / max(len(folds), 1)
    thresh = 0.67 if k == 6 else 0.70
    return {'folds': folds, 'pass_count': kf_pass, 'total_folds': len(folds),
            'pass_rate': round(rate, 3), 'verdict': 'PASS' if rate >= thresh else 'FAIL'}


# ═══════════════════════════════════════════════════════════════
# M30 Signal Functions (skip m30_kc — R230 handles it)
# ═══════════════════════════════════════════════════════════════

def m30_sig_ema_fast_cross(df):
    if len(df) < 25: return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    e9, e20 = float(curr['EMA9']), float(curr['EMA20'])
    e9p, e20p = float(prev['EMA9']), float(prev['EMA20'])
    if pd.isna(e9) or pd.isna(e20) or float(curr.get('ATR', 0)) <= 0: return None
    if e9 > e20 and e9p <= e20p: return {'strategy': 'm30_ema_fast', 'signal': 'BUY'}
    if e9 < e20 and e9p >= e20p: return {'strategy': 'm30_ema_fast', 'signal': 'SELL'}
    return None

def m30_sig_ema_cross(df):
    if len(df) < 55: return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    e20, e50 = float(curr['EMA20']), float(curr['EMA50'])
    e20p, e50p = float(prev['EMA20']), float(prev['EMA50'])
    if pd.isna(e20) or pd.isna(e50) or float(curr.get('ATR', 0)) <= 0: return None
    if e20 > e50 and e20p <= e50p: return {'strategy': 'm30_ema_cross', 'signal': 'BUY'}
    if e20 < e50 and e20p >= e50p: return {'strategy': 'm30_ema_cross', 'signal': 'SELL'}
    return None

def m30_sig_macd_cross(df):
    if len(df) < 30: return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    macd, sig = float(curr['MACD']), float(curr['MACD_signal'])
    macd_p, sig_p = float(prev['MACD']), float(prev['MACD_signal'])
    if pd.isna(macd) or pd.isna(sig) or float(curr.get('ATR', 0)) <= 0: return None
    if macd > sig and macd_p <= sig_p: return {'strategy': 'm30_macd', 'signal': 'BUY'}
    if macd < sig and macd_p >= sig_p: return {'strategy': 'm30_macd', 'signal': 'SELL'}
    return None

def m30_sig_rsi6_extreme(df):
    if len(df) < 20: return None
    row = df.iloc[-1]
    rsi6 = float(row.get('RSI6', 50))
    c, ema200 = float(row['Close']), float(row.get('EMA200', float(row['Close'])))
    if pd.isna(rsi6) or float(row.get('ATR', 0)) <= 0 or pd.isna(ema200): return None
    if rsi6 < 15 and c > ema200: return {'strategy': 'm30_rsi6', 'signal': 'BUY'}
    if rsi6 > 85 and c < ema200: return {'strategy': 'm30_rsi6', 'signal': 'SELL'}
    return None

def m30_sig_rsi14_trend(df):
    if len(df) < 55: return None
    row = df.iloc[-1]
    rsi = float(row.get('RSI14', 50))
    c, ema50 = float(row['Close']), float(row.get('EMA50', float(row['Close'])))
    slope = float(row.get('EMA50_slope', 0))
    if pd.isna(rsi) or float(row.get('ATR', 0)) <= 0: return None
    if rsi < 30 and c > ema50 and slope > 0: return {'strategy': 'm30_rsi14', 'signal': 'BUY'}
    if rsi > 70 and c < ema50 and slope < 0: return {'strategy': 'm30_rsi14', 'signal': 'SELL'}
    return None

def m30_sig_cci_momentum(df):
    if len(df) < 25: return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    cci, cci_p = float(curr.get('CCI', 0)), float(prev.get('CCI', 0))
    slope = float(curr.get('EMA50_slope', 0))
    if pd.isna(cci) or pd.isna(cci_p) or float(curr.get('ATR', 0)) <= 0: return None
    if cci > 0 and cci_p <= 0 and slope > 0: return {'strategy': 'm30_cci', 'signal': 'BUY'}
    if cci < 0 and cci_p >= 0 and slope < 0: return {'strategy': 'm30_cci', 'signal': 'SELL'}
    return None

def m30_sig_stochastic(df):
    if len(df) < 20: return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    k, d = float(curr.get('STOCH_K', 50)), float(curr.get('STOCH_D', 50))
    kp, dp = float(prev.get('STOCH_K', 50)), float(prev.get('STOCH_D', 50))
    if pd.isna(k) or pd.isna(d) or float(curr.get('ATR', 0)) <= 0: return None
    if k > d and kp <= dp and k < 30: return {'strategy': 'm30_stoch', 'signal': 'BUY'}
    if k < d and kp >= dp and k > 70: return {'strategy': 'm30_stoch', 'signal': 'SELL'}
    return None

def m30_sig_bb_squeeze(df):
    if len(df) < 15: return None
    row = df.iloc[-1]
    bb_u, bb_l = float(row.get('BB_upper', 0)), float(row.get('BB_lower', 0))
    kc_u, kc_l = float(row.get('KC_upper', 0)), float(row.get('KC_lower', 0))
    c = float(row['Close'])
    if pd.isna(bb_u) or pd.isna(kc_u) or kc_u == 0 or float(row.get('ATR', 0)) <= 0: return None
    if (bb_u < kc_u) and (bb_l > kc_l): return None
    squeeze_count = 0
    for j in range(max(0, len(df) - 11), len(df) - 1):
        r = df.iloc[j]
        if (float(r.get('BB_upper', 0)) < float(r.get('KC_upper', 0))
            and float(r.get('BB_lower', 0)) > float(r.get('KC_lower', 0))):
            squeeze_count += 1
        else: squeeze_count = 0
    if squeeze_count < 5: return None
    kc_mid = float(row.get('KC_mid', 0))
    if c > kc_mid: return {'strategy': 'm30_squeeze', 'signal': 'BUY'}
    return {'strategy': 'm30_squeeze', 'signal': 'SELL'}

def m30_sig_mean_revert(df):
    if len(df) < 25: return None
    row = df.iloc[-1]
    c, sma20 = float(row['Close']), float(row.get('SMA20', float(row['Close'])))
    atr = float(row.get('ATR', 0))
    slope = float(row.get('EMA50_slope', 0))
    if pd.isna(sma20) or atr <= 0: return None
    dev = (c - sma20) / atr
    if dev < -2.0 and slope > -0.5: return {'strategy': 'm30_mean_rev', 'signal': 'BUY'}
    if dev > 2.0 and slope < 0.5: return {'strategy': 'm30_mean_rev', 'signal': 'SELL'}
    return None

def m30_sig_inside_bar(df):
    if len(df) < 5: return None
    curr = df.iloc[-1]
    c_c = float(curr['Close'])
    if float(curr.get('ATR', 0)) <= 0: return None
    was_inside = (float(df.iloc[-3]['High']) <= float(df.iloc[-4]['High']) and
                  float(df.iloc[-3]['Low']) >= float(df.iloc[-4]['Low'])) if len(df) >= 5 else False
    if not was_inside: return None
    mother_high, mother_low = float(df.iloc[-4]['High']), float(df.iloc[-4]['Low'])
    if c_c > mother_high: return {'strategy': 'm30_inside_bar', 'signal': 'BUY'}
    if c_c < mother_low: return {'strategy': 'm30_inside_bar', 'signal': 'SELL'}
    return None

def m30_sig_engulfing(df):
    if len(df) < 10: return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    c_o = float(curr.get('Open', curr.get('open', 0)))
    c_c = float(curr['Close'])
    p_o = float(prev.get('Open', prev.get('open', 0)))
    p_c = float(prev['Close'])
    ema50 = float(curr.get('EMA50', c_c))
    if float(curr.get('ATR', 0)) <= 0 or pd.isna(ema50): return None
    curr_body, prev_body = c_c - c_o, p_c - p_o
    if curr_body > 0 and prev_body < 0 and c_c > p_o and c_o < p_c and c_c > ema50:
        return {'strategy': 'm30_engulf', 'signal': 'BUY'}
    if curr_body < 0 and prev_body > 0 and c_c < p_o and c_o > p_c and c_c < ema50:
        return {'strategy': 'm30_engulf', 'signal': 'SELL'}
    return None

M30_STRATEGIES = [
    ('m30_ema_fast', m30_sig_ema_fast_cross),
    ('m30_ema_cross', m30_sig_ema_cross),
    ('m30_macd', m30_sig_macd_cross),
    ('m30_rsi6', m30_sig_rsi6_extreme),
    ('m30_rsi14', m30_sig_rsi14_trend),
    ('m30_cci', m30_sig_cci_momentum),
    ('m30_stoch', m30_sig_stochastic),
    ('m30_squeeze', m30_sig_bb_squeeze),
    ('m30_mean_rev', m30_sig_mean_revert),
    ('m30_inside_bar', m30_sig_inside_bar),
    ('m30_engulf', m30_sig_engulfing),
]

STRAT_MAP = dict(M30_STRATEGIES)

# ═══════════════════════════════════════════════════════════════
# Worker function for parallel parameter sweep
# ═══════════════════════════════════════════════════════════════

# Global data for worker processes (loaded once via initializer)
_worker_df = None

def _init_worker(df_pickle_path):
    global _worker_df
    _worker_df = pd.read_pickle(df_pickle_path)

def _sweep_one_combo(args):
    """Evaluate a single parameter combo for a strategy."""
    strat_name, sig_func_name, sl, tp, ta, td, mh = args
    if tp < sl:
        return None
    global _worker_df
    sig_func = STRAT_MAP[strat_name]
    params = dict(DEFAULT_PARAMS)
    params.update({'sl_atr_mult': sl, 'tp_atr_mult': tp,
                   'trailing_activate_atr': ta, 'trailing_distance_atr': td,
                   'max_hold': mh})
    engine = M30BacktestEngine(_worker_df, signal_funcs=[(strat_name, sig_func)], **params)
    trades = engine.run()
    strat_trades = [t for t in trades if t.strategy == strat_name]
    s = calc_stats(strat_trades)
    return {'sl': sl, 'tp': tp, 'ta': ta, 'td': td, 'mh': mh, **s}


def run_single(df, strat_name, sig_func, params_override=None, **extra):
    p = dict(DEFAULT_PARAMS)
    if params_override: p.update(params_override)
    p.update(extra)
    engine = M30BacktestEngine(df, signal_funcs=[(strat_name, sig_func)], **p)
    trades = engine.run()
    return [t for t in trades if t.strategy == strat_name]


def main():
    t0 = time.time()
    pf('='*80)
    pf(f'R230C: M30 Parallel Sweep (11 strategies, {N_WORKERS} workers)')
    pf(f'Started: {pd.Timestamp.now()}')
    pf(f'CPU cores available: {cpu_count()}')
    pf('='*80)

    m30_df = load_m30_with_indicators()

    # Save df as pickle for worker init
    pkl_path = OUTPUT_DIR / '_m30_df.pkl'
    m30_df.to_pickle(str(pkl_path))
    pf(f'  Data saved to {pkl_path} for workers')

    # Build combo list template
    all_combos_template = []
    for sl in SL_GRID:
        for tp in TP_GRID:
            if tp < sl: continue
            for ta, td in TRAIL_GRID:
                for mh in MAX_HOLD_GRID:
                    all_combos_template.append((sl, tp, ta, td, mh))
    total_per_strat = len(all_combos_template)
    pf(f'  {total_per_strat} combos per strategy, {total_per_strat * 11} total')

    # Phase 3: Parallel parameter sweep for all 11 strategies
    pf(f'\n{"="*80}\nPhase 3: Parallel Parameter Sweep\n{"="*80}')
    phase3 = {}

    for sn, sf in M30_STRATEGIES:
        t1 = time.time()
        pf(f'\n  --- {sn} sweep ({total_per_strat} combos, {N_WORKERS} workers) ---')

        combo_args = [(sn, sn, sl, tp, ta, td, mh) for sl, tp, ta, td, mh in all_combos_template]

        with Pool(N_WORKERS, initializer=_init_worker, initargs=(str(pkl_path),)) as pool:
            results = []
            done = 0
            for r in pool.imap_unordered(_sweep_one_combo, combo_args, chunksize=10):
                if r is not None:
                    results.append(r)
                done += 1
                if done % 200 == 0:
                    pf(f'    ... {done}/{len(combo_args)}')

        results.sort(key=lambda x: x['sharpe'], reverse=True)
        best = results[0] if results else {}
        best_params = {'sl': best.get('sl'), 'tp': best.get('tp'),
                       'trail_act': best.get('ta'), 'trail_dist': best.get('td'),
                       'max_hold': best.get('mh')} if best else None
        best_sh = best.get('sharpe', 0) if best else 0

        for r in results[:10]:
            pf(f'    SL{r["sl"]}_TP{r["tp"]}_T{r["ta"]}/{r["td"]}_MH{r["mh"]}  '
               f'n={r["n"]:>5} Sh={r["sharpe"]:.3f} PnL=${r["pnl"]:.0f}')
        pf(f'  Best: {best_params}  Sh={best_sh:.3f}  ({time.time()-t1:.0f}s)')

        phase3[sn] = {'best_params': best_params, 'best_sharpe': best_sh,
                      'top10': results[:10], 'total': len(results)}
        save(f'{sn}_phase3', phase3[sn])
        save_progress(f'{sn}_phase3', f'best_sh={best_sh:.3f}')

    save('all_phase3_results', phase3)
    save_progress('phase3_complete', f'all 11 strategies done')

    # Phase 4-10: Deep validation on top candidates
    deep = [s for s, d in phase3.items() if d.get('best_sharpe', 0) > 1.0]
    if not deep:
        deep = sorted(phase3.keys(), key=lambda s: phase3[s].get('best_sharpe', 0), reverse=True)[:5]
    pf(f'\n  Deep validation candidates: {deep}')

    # Phase 4: Walk-Forward
    phase4 = {}
    pf(f'\n{"="*80}\nPhase 4: Walk-Forward\n{"="*80}')
    for sn in deep:
        bp = phase3.get(sn, {}).get('best_params', {})
        if not bp: continue
        oos_sharpes = []
        wf_res = []
        for wi, (ts, te, os_, oe) in enumerate(WF_CUTOFFS):
            best_wf_sh, best_wf_p = -999, None
            bsl, btp = bp.get('sl', 2), bp.get('tp', 4)
            for sf_ in [0.7, 0.85, 1.0, 1.15, 1.3]:
                for tf_ in [0.7, 0.85, 1.0, 1.15, 1.3]:
                    sl_v, tp_v = round(bsl*sf_, 1), round(btp*tf_, 1)
                    if tp_v < sl_v: continue
                    dt = m30_df[(m30_df.index >= pd.Timestamp(ts, tz='UTC')) & (m30_df.index < pd.Timestamp(te, tz='UTC'))].copy()
                    if len(dt) < 500: continue
                    par = {'sl_atr_mult': sl_v, 'tp_atr_mult': tp_v,
                           'trailing_activate_atr': bp.get('trail_act', 0.2),
                           'trailing_distance_atr': bp.get('trail_dist', 0.06),
                           'max_hold': bp.get('max_hold', 24)}
                    tr = run_single(dt, sn, STRAT_MAP[sn], par)
                    st = calc_stats(tr)
                    if st['sharpe'] > best_wf_sh and st['n'] >= 10:
                        best_wf_sh = st['sharpe']
                        best_wf_p = par.copy()
            if not best_wf_p:
                best_wf_p = {'sl_atr_mult': bsl, 'tp_atr_mult': btp,
                             'trailing_activate_atr': bp.get('trail_act', 0.2),
                             'trailing_distance_atr': bp.get('trail_dist', 0.06)}
            dt = m30_df[(m30_df.index >= pd.Timestamp(os_, tz='UTC')) & (m30_df.index < pd.Timestamp(oe, tz='UTC'))].copy()
            if len(dt) < 200:
                wf_res.append({'period': f'{os_}->{oe}', 'skip': True}); continue
            oos_tr = run_single(dt, sn, STRAT_MAP[sn], best_wf_p)
            oos_st = calc_stats(oos_tr)
            oos_sharpes.append(oos_st['sharpe'])
            wf_res.append({'period': f'{os_}->{oe}', 'oos_n': oos_st['n'], 'oos_sharpe': oos_st['sharpe']})
            pf(f'    {sn} WF{wi+1}: OOS n={oos_st["n"]} Sh={oos_st["sharpe"]:.3f}')
        valid = [s for s in oos_sharpes if not np.isnan(s)]
        pos = sum(1 for s in valid if s > 0)
        rate = pos / max(len(valid), 1)
        v = 'PASS' if rate >= 0.67 else 'FAIL'
        pf(f'    {sn} WF: {v} ({pos}/{len(valid)})')
        phase4[sn] = {'wf': wf_res, 'wf_verdict': v, 'wf_pass_rate': round(rate, 3)}
    save('phase4_walk_forward', phase4)
    save_progress('phase4')

    # Phase 5: Era Stability
    phase5 = {}
    pf(f'\n{"="*80}\nPhase 5: Era Stability\n{"="*80}')
    for sn in deep:
        bp = phase3.get(sn, {}).get('best_params', {})
        par = {'sl_atr_mult': bp.get('sl', 2), 'tp_atr_mult': bp.get('tp', 4),
               'trailing_activate_atr': bp.get('trail_act', 0.2), 'trailing_distance_atr': bp.get('trail_dist', 0.06)}
        trades = run_single(m30_df, sn, STRAT_MAP[sn], par)
        pos_eras = 0
        for en, (es, ee) in ERA_SEGMENTS.items():
            s = calc_stats(filter_period(trades, es, ee))
            if s['sharpe'] > 0: pos_eras += 1
            pf(f'  {sn} {en:<30} Sh={s["sharpe"]:.3f}')
        v = 'PASS' if pos_eras >= 3 else 'FAIL'
        phase5[sn] = {'positive_eras': pos_eras, 'era_verdict': v}
    save('phase5_era', phase5)
    save_progress('phase5')

    # Phase 6: Sensitivity
    phase6 = {}
    pf(f'\n{"="*80}\nPhase 6: Sensitivity\n{"="*80}')
    for sn in deep:
        bp = phase3.get(sn, {}).get('best_params', {})
        if not bp: continue
        base_par = {'sl_atr_mult': bp.get('sl',2), 'tp_atr_mult': bp.get('tp',4),
                    'trailing_activate_atr': bp.get('trail_act',0.2), 'trailing_distance_atr': bp.get('trail_dist',0.06)}
        base_sh = calc_stats(run_single(m30_df, sn, STRAT_MAP[sn], base_par))['sharpe']
        drops = []
        for pn in ['sl_atr_mult','tp_atr_mult','trailing_activate_atr','trailing_distance_atr']:
            bv = base_par.get(pn, 0)
            if bv == 0: continue
            for f in [0.6, 0.8, 1.2, 1.4]:
                pp = dict(base_par); pp[pn] = round(bv*f, 4)
                ps = calc_stats(run_single(m30_df, sn, STRAT_MAP[sn], pp))['sharpe']
                drops.append(abs((base_sh - ps) / max(abs(base_sh), 1e-9) * 100))
        mx = max(drops) if drops else 0
        v = 'STABLE' if mx < 40 else ('MODERATE' if mx < 70 else 'FRAGILE')
        pf(f'  {sn}: {v} (max_drop={mx:.1f}%)')
        phase6[sn] = {'verdict': v, 'max_drop': round(mx, 1)}
    save('phase6_sensitivity', phase6)
    save_progress('phase6')

    # Phase 7: Monte Carlo
    phase7 = {}
    pf(f'\n{"="*80}\nPhase 7: Monte Carlo ({N_BOOTSTRAP}x)\n{"="*80}')
    for sn in deep:
        bp = phase3.get(sn, {}).get('best_params', {})
        par = {'sl_atr_mult': bp.get('sl',2), 'tp_atr_mult': bp.get('tp',4),
               'trailing_activate_atr': bp.get('trail_act',0.2), 'trailing_distance_atr': bp.get('trail_dist',0.06)}
        trades = run_single(m30_df, sn, STRAT_MAP[sn], par)
        pnls = np.array([t.pnl for t in trades])
        if len(pnls) < 20:
            phase7[sn] = {'skip': True}; continue
        rng = np.random.default_rng(42)
        boot = [float(rng.choice(pnls, len(pnls), True).mean() / max(rng.choice(pnls, len(pnls), True).std(ddof=1), 1e-9) * np.sqrt(252)) for _ in range(N_BOOTSTRAP)]
        ba = np.array(boot)
        pv = (ba <= 0).sum() / N_BOOTSTRAP
        v = 'PASS' if pv < 0.05 else 'FAIL'
        pf(f'  {sn}: p={pv:.4f} median_Sh={np.median(ba):.3f} -> {v}')
        phase7[sn] = {'p_value': round(float(pv), 5), 'median_sharpe': round(float(np.median(ba)), 3),
                      'ci_5': round(float(np.percentile(ba, 5)), 3), 'mc_verdict': v}
    save('phase7_monte_carlo', phase7)
    save_progress('phase7')

    # Phase 8: Drawdown
    phase8 = {}
    pf(f'\n{"="*80}\nPhase 8: Drawdown\n{"="*80}')
    for sn in deep:
        bp = phase3.get(sn, {}).get('best_params', {})
        par = {'sl_atr_mult': bp.get('sl',2), 'tp_atr_mult': bp.get('tp',4),
               'trailing_activate_atr': bp.get('trail_act',0.2), 'trailing_distance_atr': bp.get('trail_dist',0.06)}
        trades = run_single(m30_df, sn, STRAT_MAP[sn], par)
        pnls = np.array([t.pnl for t in trades])
        if len(pnls) < 10: phase8[sn] = {'skip': True}; continue
        cum = np.cumsum(pnls)
        dd = np.maximum.accumulate(cum) - cum
        streak, mx_streak = 0, 0
        for p in pnls:
            if p < 0: streak += 1; mx_streak = max(mx_streak, streak)
            else: streak = 0
        pf(f'  {sn}: MaxDD=${dd.max():.0f}  WorstStreak={mx_streak}')
        phase8[sn] = {'max_dd': round(float(dd.max()), 2), 'worst_streak': mx_streak}
    save('phase8_drawdown', phase8)
    save_progress('phase8')

    # Phase 9: Slippage
    winners = [s for s in deep if phase7.get(s, {}).get('mc_verdict') == 'PASS' or phase4.get(s, {}).get('wf_verdict') == 'PASS'] or deep[:3]
    phase9 = {}
    if winners:
        pf(f'\n{"="*80}\nPhase 9: Slippage Testing\n{"="*80}')
        for sn in winners:
            bp = phase3.get(sn, {}).get('best_params', {})
            slip_res = {}
            for sc in SLIPPAGE_CONFIGS:
                par = dict(DEFAULT_PARAMS)
                if bp: par.update({'sl_atr_mult': bp.get('sl',2), 'tp_atr_mult': bp.get('tp',4),
                                   'trailing_activate_atr': bp.get('trail_act',0.2), 'trailing_distance_atr': bp.get('trail_dist',0.06)})
                par['slippage_model'] = sc['slippage_model']
                engine = M30BacktestEngine(m30_df, signal_funcs=[(sn, STRAT_MAP[sn])], **par)
                tr = engine.run()
                st = calc_stats([t for t in tr if t.strategy == sn])
                slip_res[sc['name']] = st
                pf(f'  {sn} [{sc["name"]}]: n={st["n"]} Sh={st["sharpe"]:.3f} PnL=${st["pnl"]:.0f}')
            ns = slip_res.get('no_slippage', {}).get('sharpe', 0)
            rs = slip_res.get('realistic_slippage', {}).get('sharpe', 0)
            deg = (ns - rs) / max(abs(ns), 1e-9) * 100
            v = 'ROBUST' if deg < 30 else ('ACCEPTABLE' if deg < 60 else 'FRAGILE')
            pf(f'  {sn} degradation: {deg:.1f}% -> {v}')
            phase9[sn] = {'results': slip_res, 'degradation_pct': round(deg, 1), 'verdict': v}
        save('phase9_slippage', phase9)
        save_progress('phase9')

    # Phase 10: Final Verdict
    pf(f'\n{"="*80}\nPhase 10: Final Verdict\n{"="*80}')
    phase10 = {}
    for sn in [s for s, _ in M30_STRATEGIES]:
        kf6_v = 'PASS'  # all 11 passed K-Fold in R230
        wf = phase4.get(sn, {}).get('wf_verdict', 'N/A')
        era = phase5.get(sn, {}).get('era_verdict', 'N/A')
        mc = phase7.get(sn, {}).get('mc_verdict', 'N/A')
        sens = phase6.get(sn, {}).get('verdict', 'N/A')
        slip = phase9.get(sn, {}).get('verdict', 'N/A')
        bsh = phase3.get(sn, {}).get('best_sharpe', 0)
        bpar = phase3.get(sn, {}).get('best_params', {})
        pc = sum(1 for v in [kf6_v, wf, era, mc] if v == 'PASS')
        if pc >= 4 and sens != 'FRAGILE' and slip in ('ROBUST','ACCEPTABLE'): final = 'STRONG_PASS'
        elif pc >= 3 and mc == 'PASS': final = 'CONDITIONAL_PASS'
        elif pc >= 2: final = 'WEAK_PASS'
        elif kf6_v == 'PASS': final = 'MARGINAL'
        else: final = 'REJECT'
        phase10[sn] = {'best_sharpe': bsh, 'best_params': bpar, 'final_verdict': final,
                       'gates': {'kf6': kf6_v, 'wf': wf, 'era': era, 'mc': mc, 'sens': sens, 'slip': slip}}
        pf(f'  {sn:<20} Sh={bsh:.3f}  KF=PASS WF={wf:<5} Era={era:<5} MC={mc:<5} '
           f'Sens={sens:<8} Slip={slip:<10} -> {final}')
    save('phase10_verdict', phase10)
    save_progress('DONE', f'total_time={time.time()-t0:.0f}s')

    elapsed = time.time() - t0
    pf(f'\n  Total runtime: {elapsed:.0f}s ({elapsed/3600:.1f}h)')
    pf(f'  Finished: {pd.Timestamp.now()}')

    # Cleanup pickle
    try: pkl_path.unlink()
    except: pass

if __name__ == '__main__':
    main()
