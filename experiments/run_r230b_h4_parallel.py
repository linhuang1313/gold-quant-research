#!/usr/bin/env python3
"""R230B: H4 Parallel Pipeline (runs alongside R230 M30)
=========================================================
Identical 10-phase pipeline as R230 but for H4 only.
Runs in parallel with R230's M30 pipeline to utilize server's 208 cores.
"""
from __future__ import annotations
import sys, json, time, traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.h4_engine import H4BacktestEngine, load_h4_with_indicators
from backtest.engine import TradeRecord

OUTPUT_DIR = Path("results/r230b_h4_parallel")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPREAD = 0.30
N_BOOTSTRAP = 5000

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
MAX_HOLD_GRID = [15, 20, 30, 45, 60]

SLIPPAGE_CONFIGS = [
    {"name": "no_slippage", "slippage_model": "none"},
    {"name": "fixed_slippage", "slippage_model": "fixed"},
    {"name": "empirical_slippage", "slippage_model": "empirical"},
    {"name": "realistic_slippage", "slippage_model": "realistic"},
]

PROGRESS_FILE = OUTPUT_DIR / "_progress.json"
DEFAULT_PARAMS = {
    'sl_atr_mult': 3.0, 'tp_atr_mult': 6.0,
    'trailing_activate_atr': 0.3, 'trailing_distance_atr': 0.08,
    'max_hold': 30, 'cooldown_bars': 2, 'spread_cost': SPREAD,
}


def save(name, data):
    p = OUTPUT_DIR / f'{name}.json'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    print(f'  -> saved {p}', flush=True)

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

def kfold_6(trades):
    if len(trades) < 30: return {'skip': True, 'verdict': 'SKIP'}
    pnls = np.array([t.pnl for t in trades])
    fold_size = len(pnls) // 6
    folds, kf_pass = [], 0
    for fold in range(6):
        s, e = fold * fold_size, (fold * fold_size + fold_size if fold < 5 else len(pnls))
        fp = pnls[s:e]
        if len(fp) < 5: continue
        sh = float(fp.mean() / max(fp.std(ddof=1), 1e-9) * np.sqrt(252))
        folds.append({'fold': fold+1, 'n': len(fp), 'sharpe': round(sh, 3)})
        if sh > 0: kf_pass += 1
    rate = kf_pass / max(len(folds), 1)
    return {'folds': folds, 'pass_count': kf_pass, 'total_folds': len(folds),
            'pass_rate': round(rate, 3), 'verdict': 'PASS' if rate >= 0.67 else 'FAIL'}

def kfold_10(trades):
    if len(trades) < 50: return {'skip': True, 'verdict': 'SKIP'}
    pnls = np.array([t.pnl for t in trades])
    fold_size = len(pnls) // 10
    folds, kf_pass = [], 0
    for fold in range(10):
        s, e = fold * fold_size, (fold * fold_size + fold_size if fold < 9 else len(pnls))
        fp = pnls[s:e]
        if len(fp) < 3: continue
        sh = float(fp.mean() / max(fp.std(ddof=1), 1e-9) * np.sqrt(252))
        folds.append({'fold': fold+1, 'n': len(fp), 'sharpe': round(sh, 3)})
        if sh > 0: kf_pass += 1
    rate = kf_pass / max(len(folds), 1)
    return {'folds': folds, 'pass_count': kf_pass, 'total_folds': len(folds),
            'pass_rate': round(rate, 3), 'verdict': 'PASS' if rate >= 0.70 else 'FAIL'}


# ═══════════════════════════════════════════════════════════════
# H4 Signal Functions (15 strategies)
# ═══════════════════════════════════════════════════════════════

def h4_sig_kc(df):
    if len(df) < 30: return None
    r = df.iloc[-1]
    c, u, l = float(r['Close']), float(r.get('KC_upper', 0)), float(r.get('KC_lower', 0))
    if pd.isna(u) or u == 0 or float(r.get('ATR', 0)) <= 0: return None
    if c > u: return {'strategy': 'h4_kc', 'signal': 'BUY'}
    if c < l: return {'strategy': 'h4_kc', 'signal': 'SELL'}
    return None

def h4_sig_ema_cross(df):
    if len(df) < 55: return None
    c, p = df.iloc[-1], df.iloc[-2]
    e20, e50 = float(c['EMA20']), float(c['EMA50'])
    e20p, e50p = float(p['EMA20']), float(p['EMA50'])
    if pd.isna(e20) or pd.isna(e50) or float(c.get('ATR', 0)) <= 0: return None
    if e20 > e50 and e20p <= e50p: return {'strategy': 'h4_ema_cross', 'signal': 'BUY'}
    if e20 < e50 and e20p >= e50p: return {'strategy': 'h4_ema_cross', 'signal': 'SELL'}
    return None

def h4_sig_macd(df):
    if len(df) < 30: return None
    c, p = df.iloc[-1], df.iloc[-2]
    m, s = float(c['MACD']), float(c['MACD_signal'])
    mp, sp = float(p['MACD']), float(p['MACD_signal'])
    if pd.isna(m) or pd.isna(s) or float(c.get('ATR', 0)) <= 0: return None
    if m > s and mp <= sp: return {'strategy': 'h4_macd', 'signal': 'BUY'}
    if m < s and mp >= sp: return {'strategy': 'h4_macd', 'signal': 'SELL'}
    return None

def h4_sig_rsi(df):
    if len(df) < 30: return None
    r = df.iloc[-1]
    rsi = float(r.get('RSI14', 50))
    c, e100 = float(r['Close']), float(r.get('EMA100', float(r['Close'])))
    if pd.isna(rsi) or float(r.get('ATR', 0)) <= 0: return None
    if rsi < 25 and c > e100: return {'strategy': 'h4_rsi', 'signal': 'BUY'}
    if rsi > 75 and c < e100: return {'strategy': 'h4_rsi', 'signal': 'SELL'}
    return None

def h4_sig_cci(df):
    if len(df) < 25: return None
    c, p = df.iloc[-1], df.iloc[-2]
    cci, cci_p = float(c.get('CCI', 0)), float(p.get('CCI', 0))
    slope = float(c.get('EMA50_slope', 0))
    if pd.isna(cci) or pd.isna(cci_p) or float(c.get('ATR', 0)) <= 0: return None
    if cci > 0 and cci_p <= 0 and slope > 0: return {'strategy': 'h4_cci', 'signal': 'BUY'}
    if cci < 0 and cci_p >= 0 and slope < 0: return {'strategy': 'h4_cci', 'signal': 'SELL'}
    return None

def h4_sig_squeeze(df):
    if len(df) < 15: return None
    r = df.iloc[-1]
    bb_u, bb_l = float(r.get('BB_upper', 0)), float(r.get('BB_lower', 0))
    kc_u, kc_l = float(r.get('KC_upper', 0)), float(r.get('KC_lower', 0))
    c = float(r['Close'])
    if pd.isna(bb_u) or pd.isna(kc_u) or kc_u == 0 or float(r.get('ATR', 0)) <= 0: return None
    if (bb_u < kc_u) and (bb_l > kc_l): return None
    sq = 0
    for j in range(max(0, len(df)-11), len(df)-1):
        rr = df.iloc[j]
        if float(rr.get('BB_upper',0)) < float(rr.get('KC_upper',0)) and float(rr.get('BB_lower',0)) > float(rr.get('KC_lower',0)):
            sq += 1
        else: sq = 0
    if sq < 5: return None
    km = float(r.get('KC_mid', 0))
    if c > km: return {'strategy': 'h4_squeeze', 'signal': 'BUY'}
    return {'strategy': 'h4_squeeze', 'signal': 'SELL'}

def h4_sig_donchian(df):
    if len(df) < 55: return None
    r = df.iloc[-1]
    c = float(r['Close'])
    h50 = float(df['High'].iloc[-51:-1].max())
    l50 = float(df['Low'].iloc[-51:-1].min())
    if float(r.get('ATR', 0)) <= 0: return None
    if c > h50: return {'strategy': 'h4_donchian', 'signal': 'BUY'}
    if c < l50: return {'strategy': 'h4_donchian', 'signal': 'SELL'}
    return None

def h4_sig_ema_fast(df):
    if len(df) < 25: return None
    c, p = df.iloc[-1], df.iloc[-2]
    if 'EMA9' not in df.columns:
        ema9 = df['Close'].ewm(span=9, adjust=False).mean()
        e9, e9p = float(ema9.iloc[-1]), float(ema9.iloc[-2])
    else:
        e9, e9p = float(c.get('EMA9', 0)), float(p.get('EMA9', 0))
    e20, e20p = float(c['EMA20']), float(p['EMA20'])
    if pd.isna(e9) or pd.isna(e20) or float(c.get('ATR', 0)) <= 0: return None
    if e9 > e20 and e9p <= e20p: return {'strategy': 'h4_ema_fast', 'signal': 'BUY'}
    if e9 < e20 and e9p >= e20p: return {'strategy': 'h4_ema_fast', 'signal': 'SELL'}
    return None

def h4_sig_rsi_div(df):
    if len(df) < 30: return None
    c = df.iloc[-1]
    if float(c.get('ATR', 0)) <= 0: return None
    pr = df['Close'].values[-20:]
    rv = df['RSI14'].values[-20:] if 'RSI14' in df.columns else None
    if rv is None or np.any(np.isnan(rv)) or len(pr) < 10: return None
    if pr[10:].min() < pr[:10].min() and rv[10:].min() > rv[:10].min() and float(c['RSI14']) < 40:
        return {'strategy': 'h4_rsi_div', 'signal': 'BUY'}
    if pr[10:].max() > pr[:10].max() and rv[10:].max() < rv[:10].max() and float(c['RSI14']) > 60:
        return {'strategy': 'h4_rsi_div', 'signal': 'SELL'}
    return None

def h4_sig_adx_di(df):
    if len(df) < 20: return None
    c, p = df.iloc[-1], df.iloc[-2]
    adx = float(c.get('ADX', 0))
    if pd.isna(adx) or adx < 25: return None
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    pdm = df['High'].diff().clip(lower=0)
    mdm = (-df['Low'].diff()).clip(lower=0)
    pdm[df['High'].diff() <= (-df['Low'].diff())] = 0
    mdm[(-df['Low'].diff()) <= df['High'].diff()] = 0
    atr_s = tr.ewm(span=14, adjust=False).mean()
    pdi = 100 * pdm.ewm(span=14, adjust=False).mean() / atr_s.replace(0, np.nan)
    mdi = 100 * mdm.ewm(span=14, adjust=False).mean() / atr_s.replace(0, np.nan)
    pc, pp = float(pdi.iloc[-1]), float(pdi.iloc[-2])
    mc, mp = float(mdi.iloc[-1]), float(mdi.iloc[-2])
    if pc > mc and pp <= mp: return {'strategy': 'h4_adx_di', 'signal': 'BUY'}
    if pc < mc and pp >= mp: return {'strategy': 'h4_adx_di', 'signal': 'SELL'}
    return None

def h4_sig_stoch(df):
    if len(df) < 20: return None
    l14 = df['Low'].rolling(14).min()
    h14 = df['High'].rolling(14).max()
    sk = 100 * (df['Close'] - l14) / (h14 - l14).replace(0, np.nan)
    sd = sk.rolling(3).mean()
    kc, dc = float(sk.iloc[-1]), float(sd.iloc[-1])
    kp, dp = float(sk.iloc[-2]), float(sd.iloc[-2])
    if pd.isna(kc) or pd.isna(dc): return None
    if kc > dc and kp <= dp and kc < 30: return {'strategy': 'h4_stoch', 'signal': 'BUY'}
    if kc < dc and kp >= dp and kc > 70: return {'strategy': 'h4_stoch', 'signal': 'SELL'}
    return None

def h4_sig_ema_ribbon(df):
    if len(df) < 105: return None
    c, p = df.iloc[-1], df.iloc[-2]
    e20, e50, e100 = float(c['EMA20']), float(c['EMA50']), float(c.get('EMA100', 0))
    e20p, e50p, e100p = float(p['EMA20']), float(p['EMA50']), float(p.get('EMA100', 0))
    if pd.isna(e100) or e100 == 0: return None
    if (e20>e50>e100) and not (e20p>e50p>e100p): return {'strategy': 'h4_ema_ribbon', 'signal': 'BUY'}
    if (e20<e50<e100) and not (e20p<e50p<e100p): return {'strategy': 'h4_ema_ribbon', 'signal': 'SELL'}
    return None

def h4_sig_mean_rev(df):
    if len(df) < 55: return None
    c = df.iloc[-1]
    cl, e50 = float(c['Close']), float(c['EMA50'])
    atr = float(c.get('ATR', 0))
    if pd.isna(e50) or atr <= 0: return None
    d = (cl - e50) / atr
    if d < -2.0: return {'strategy': 'h4_mean_rev', 'signal': 'BUY'}
    if d > 2.0: return {'strategy': 'h4_mean_rev', 'signal': 'SELL'}
    return None

def h4_sig_momentum(df):
    if len(df) < 15: return None
    c = df.iloc[-1]
    cl, atr = float(c['Close']), float(c.get('ATR', 0))
    if atr <= 0: return None
    h10 = float(df['High'].iloc[-11:-1].max())
    l10 = float(df['Low'].iloc[-11:-1].min())
    if cl > h10 + 0.5*atr: return {'strategy': 'h4_momentum', 'signal': 'BUY'}
    if cl < l10 - 0.5*atr: return {'strategy': 'h4_momentum', 'signal': 'SELL'}
    return None

def h4_sig_inside_bar(df):
    if len(df) < 5: return None
    c, p1, p2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    cl = float(c['Close'])
    if not (float(p1['High']) <= float(p2['High']) and float(p1['Low']) >= float(p2['Low'])): return None
    if cl > float(p2['High']): return {'strategy': 'h4_inside_bar', 'signal': 'BUY'}
    if cl < float(p2['Low']): return {'strategy': 'h4_inside_bar', 'signal': 'SELL'}
    return None

H4_STRATEGIES = [
    ('h4_kc', h4_sig_kc), ('h4_ema_cross', h4_sig_ema_cross),
    ('h4_macd', h4_sig_macd), ('h4_rsi', h4_sig_rsi),
    ('h4_cci', h4_sig_cci), ('h4_squeeze', h4_sig_squeeze),
    ('h4_donchian', h4_sig_donchian), ('h4_ema_fast', h4_sig_ema_fast),
    ('h4_rsi_div', h4_sig_rsi_div), ('h4_adx_di', h4_sig_adx_di),
    ('h4_stoch', h4_sig_stoch), ('h4_ema_ribbon', h4_sig_ema_ribbon),
    ('h4_mean_rev', h4_sig_mean_rev), ('h4_momentum', h4_sig_momentum),
    ('h4_inside_bar', h4_sig_inside_bar),
]

STRAT_MAP = dict(H4_STRATEGIES)


def run_single(h4_df, strat_name, sig_func, params_override=None, **extra):
    p = dict(DEFAULT_PARAMS)
    if params_override: p.update(params_override)
    p.update(extra)
    engine = H4BacktestEngine(h4_df, signal_funcs=[(strat_name, sig_func)], **p)
    trades = engine.run()
    return [t for t in trades if t.strategy == strat_name]


def main():
    t0 = time.time()
    pf('='*80)
    pf('R230B: H4 Parallel Pipeline (15 strategies, 10 phases)')
    pf(f'Started: {pd.Timestamp.now()}')
    pf('='*80)

    h4_df = load_h4_with_indicators()

    # ── Phase 1: Screening ──
    pf(f'\n{"="*80}\nPhase 1: Strategy Screening\n{"="*80}')
    phase1, viable = {}, []
    for sn, sf in H4_STRATEGIES:
        pf(f'\n  --- {sn} ---')
        trades = run_single(h4_df, sn, sf)
        s = calc_stats(trades)
        pf(f'  {sn:<20} n={s["n"]:>5}  Sh={s["sharpe"]:.3f}  PnL=${s["pnl"]:.0f}  WR={s["win_rate"]:.1f}%  PF={s["profit_factor"]:.2f}')
        eras = {}
        for en, (es, ee) in ERA_SEGMENTS.items():
            et = filter_period(trades, es, ee)
            era_s = calc_stats(et)
            eras[en] = era_s
            pf(f'    {en:<30} n={era_s["n"]:>4}  Sh={era_s["sharpe"]:.3f}')
        phase1[sn] = {'stats': s, 'eras': eras}
        if s['sharpe'] > 0.3 and s['n'] >= 30: viable.append(sn)
    save('phase1_screening', phase1)
    save_progress('phase1', f'viable={viable}')

    # ── Phase 2: K-Fold ──
    pf(f'\n{"="*80}\nPhase 2: K-Fold Validation\n{"="*80}')
    phase2 = {}
    for sn in viable:
        trades = run_single(h4_df, sn, STRAT_MAP[sn])
        kf6, kf10 = kfold_6(trades), kfold_10(trades)
        pf(f'  {sn}: 6F={kf6["verdict"]}  10F={kf10["verdict"]}')
        phase2[sn] = {'kfold_6': kf6, 'kfold_10': kf10}
    save('phase2_kfold', phase2)
    save_progress('phase2')
    kf_passers = [s for s in viable if phase2.get(s, {}).get('kfold_6', {}).get('verdict') == 'PASS']
    pf(f'  K-Fold passers: {kf_passers}')

    # ── Phase 3: Extended Param Sweep ──
    phase3 = {}
    if kf_passers:
        pf(f'\n{"="*80}\nPhase 3: Extended Parameter Sweep\n{"="*80}')
        for sn in kf_passers:
            sf = STRAT_MAP[sn]
            pf(f'\n  --- {sn} sweep ---')
            best_sh, best_p, results = -999, None, []
            combo, total = 0, len(SL_GRID)*len(TP_GRID)*len(TRAIL_GRID)*len(MAX_HOLD_GRID)
            for sl in SL_GRID:
                for tp in TP_GRID:
                    if tp < sl: continue
                    for ta, td in TRAIL_GRID:
                        for mh in MAX_HOLD_GRID:
                            combo += 1
                            if combo % 100 == 0: pf(f'    ... {combo}/{total}')
                            tr = run_single(h4_df, sn, sf, {'sl_atr_mult': sl, 'tp_atr_mult': tp,
                                'trailing_activate_atr': ta, 'trailing_distance_atr': td, 'max_hold': mh})
                            st = calc_stats(tr)
                            results.append({'sl': sl, 'tp': tp, 'ta': ta, 'td': td, 'mh': mh, **st})
                            if st['sharpe'] > best_sh and st['n'] >= 20:
                                best_sh = st['sharpe']
                                best_p = {'sl': sl, 'tp': tp, 'trail_act': ta, 'trail_dist': td, 'max_hold': mh}
            results.sort(key=lambda x: x['sharpe'], reverse=True)
            for r in results[:10]:
                pf(f'    SL{r["sl"]}_TP{r["tp"]}_T{r["ta"]}/{r["td"]}_MH{r["mh"]}  n={r["n"]:>4} Sh={r["sharpe"]:.3f} PnL=${r["pnl"]:.0f}')
            pf(f'  Best: {best_p}  Sh={best_sh:.3f}')
            phase3[sn] = {'best_params': best_p, 'best_sharpe': best_sh, 'top10': results[:10], 'total': combo}
        save('phase3_param_sweep', phase3)
        save_progress('phase3')

    # ── Phase 4-10: Deep validation on top candidates ──
    deep = [s for s in kf_passers if phase3.get(s, {}).get('best_sharpe', 0) > 1.0] or kf_passers[:5]

    # Phase 4: Walk-Forward
    phase4 = {}
    if deep:
        pf(f'\n{"="*80}\nPhase 4: Walk-Forward\n{"="*80}')
        for sn in deep:
            sf = STRAT_MAP[sn]
            bp = phase3.get(sn, {}).get('best_params', {})
            oos_sharpes = []
            wf_res = []
            for wi, (ts, te, os_, oe) in enumerate(WF_CUTOFFS):
                best_wf_sh, best_wf_p = -999, None
                bsl = bp.get('sl', 3.0)
                btp = bp.get('tp', 6.0)
                for sf_ in [0.7, 0.85, 1.0, 1.15, 1.3]:
                    for tf_ in [0.7, 0.85, 1.0, 1.15, 1.3]:
                        sl_v, tp_v = round(bsl*sf_, 1), round(btp*tf_, 1)
                        if tp_v < sl_v: continue
                        dt = h4_df[(h4_df.index >= pd.Timestamp(ts, tz='UTC')) & (h4_df.index < pd.Timestamp(te, tz='UTC'))].copy()
                        if len(dt) < 100: continue
                        par = {'sl_atr_mult': sl_v, 'tp_atr_mult': tp_v,
                               'trailing_activate_atr': bp.get('trail_act', 0.3),
                               'trailing_distance_atr': bp.get('trail_dist', 0.08),
                               'max_hold': bp.get('max_hold', 30)}
                        tr = run_single(dt, sn, sf, par)
                        st = calc_stats(tr)
                        if st['sharpe'] > best_wf_sh and st['n'] >= 5:
                            best_wf_sh = st['sharpe']
                            best_wf_p = par.copy()
                if not best_wf_p:
                    best_wf_p = {'sl_atr_mult': bsl, 'tp_atr_mult': btp,
                                 'trailing_activate_atr': bp.get('trail_act', 0.3),
                                 'trailing_distance_atr': bp.get('trail_dist', 0.08)}
                dt = h4_df[(h4_df.index >= pd.Timestamp(os_, tz='UTC')) & (h4_df.index < pd.Timestamp(oe, tz='UTC'))].copy()
                if len(dt) < 50:
                    wf_res.append({'period': f'{os_}->{oe}', 'skip': True})
                    continue
                oos_tr = run_single(dt, sn, sf, best_wf_p)
                oos_st = calc_stats(oos_tr)
                oos_sharpes.append(oos_st['sharpe'])
                wf_res.append({'period': f'{os_}->{oe}', 'oos_n': oos_st['n'], 'oos_sharpe': oos_st['sharpe']})
                pf(f'    {sn} WF{wi+1}: OOS n={oos_st["n"]} Sh={oos_st["sharpe"]:.3f}')
            valid = [s for s in oos_sharpes if not np.isnan(s)]
            pos = sum(1 for s in valid if s > 0)
            rate = pos / max(len(valid), 1)
            verdict = 'PASS' if rate >= 0.67 else 'FAIL'
            pf(f'    {sn} WF: {verdict} ({pos}/{len(valid)})')
            phase4[sn] = {'wf': wf_res, 'wf_verdict': verdict, 'wf_pass_rate': round(rate, 3)}
        save('phase4_walk_forward', phase4)
        save_progress('phase4')

    # Phase 5: Era Stability
    phase5 = {}
    if deep:
        pf(f'\n{"="*80}\nPhase 5: Era Stability\n{"="*80}')
        for sn in deep:
            bp = phase3.get(sn, {}).get('best_params', {})
            par = {'sl_atr_mult': bp.get('sl', 3.0), 'tp_atr_mult': bp.get('tp', 6.0),
                   'trailing_activate_atr': bp.get('trail_act', 0.3), 'trailing_distance_atr': bp.get('trail_dist', 0.08)}
            trades = run_single(h4_df, sn, STRAT_MAP[sn], par)
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
    if deep:
        pf(f'\n{"="*80}\nPhase 6: Sensitivity\n{"="*80}')
        for sn in deep:
            bp = phase3.get(sn, {}).get('best_params', {})
            if not bp: continue
            base_par = {'sl_atr_mult': bp.get('sl',3), 'tp_atr_mult': bp.get('tp',6),
                        'trailing_activate_atr': bp.get('trail_act',0.3), 'trailing_distance_atr': bp.get('trail_dist',0.08)}
            base_sh = calc_stats(run_single(h4_df, sn, STRAT_MAP[sn], base_par))['sharpe']
            drops = []
            for pn in ['sl_atr_mult','tp_atr_mult','trailing_activate_atr','trailing_distance_atr']:
                bv = base_par.get(pn, 0)
                if bv == 0: continue
                for f in [0.6, 0.8, 1.2, 1.4]:
                    pp = dict(base_par); pp[pn] = round(bv*f, 4)
                    ps = calc_stats(run_single(h4_df, sn, STRAT_MAP[sn], pp))['sharpe']
                    drops.append(abs((base_sh - ps) / max(abs(base_sh), 1e-9) * 100))
            mx = max(drops) if drops else 0
            v = 'STABLE' if mx < 40 else ('MODERATE' if mx < 70 else 'FRAGILE')
            pf(f'  {sn}: {v} (max_drop={mx:.1f}%)')
            phase6[sn] = {'verdict': v, 'max_drop': round(mx, 1)}
        save('phase6_sensitivity', phase6)
        save_progress('phase6')

    # Phase 7: Monte Carlo
    phase7 = {}
    if deep:
        pf(f'\n{"="*80}\nPhase 7: Monte Carlo ({N_BOOTSTRAP}x)\n{"="*80}')
        for sn in deep:
            bp = phase3.get(sn, {}).get('best_params', {})
            par = {'sl_atr_mult': bp.get('sl',3), 'tp_atr_mult': bp.get('tp',6),
                   'trailing_activate_atr': bp.get('trail_act',0.3), 'trailing_distance_atr': bp.get('trail_dist',0.08)}
            trades = run_single(h4_df, sn, STRAT_MAP[sn], par)
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
    if deep:
        pf(f'\n{"="*80}\nPhase 8: Drawdown\n{"="*80}')
        for sn in deep:
            bp = phase3.get(sn, {}).get('best_params', {})
            par = {'sl_atr_mult': bp.get('sl',3), 'tp_atr_mult': bp.get('tp',6),
                   'trailing_activate_atr': bp.get('trail_act',0.3), 'trailing_distance_atr': bp.get('trail_dist',0.08)}
            trades = run_single(h4_df, sn, STRAT_MAP[sn], par)
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

    # Phase 9: Slippage Testing
    winners = [s for s in deep if phase7.get(s, {}).get('mc_verdict') == 'PASS' or phase4.get(s, {}).get('wf_verdict') == 'PASS'] or deep[:3]
    phase9 = {}
    if winners:
        pf(f'\n{"="*80}\nPhase 9: Slippage Testing\n{"="*80}')
        for sn in winners:
            bp = phase3.get(sn, {}).get('best_params', {})
            slip_res = {}
            for sc in SLIPPAGE_CONFIGS:
                par = dict(DEFAULT_PARAMS)
                if bp: par.update({'sl_atr_mult': bp.get('sl',3), 'tp_atr_mult': bp.get('tp',6),
                                   'trailing_activate_atr': bp.get('trail_act',0.3), 'trailing_distance_atr': bp.get('trail_dist',0.08)})
                par['slippage_model'] = sc['slippage_model']
                engine = H4BacktestEngine(h4_df, signal_funcs=[(sn, STRAT_MAP[sn])], **par)
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
    for sn in viable:
        p1_sh = phase1.get(sn, {}).get('stats', {}).get('sharpe', 0)
        kf6 = phase2.get(sn, {}).get('kfold_6', {}).get('verdict', 'SKIP')
        wf = phase4.get(sn, {}).get('wf_verdict', 'N/A')
        era = phase5.get(sn, {}).get('era_verdict', 'N/A')
        mc = phase7.get(sn, {}).get('mc_verdict', 'N/A')
        sens = phase6.get(sn, {}).get('verdict', 'N/A')
        slip = phase9.get(sn, {}).get('verdict', 'N/A')
        bsh = phase3.get(sn, {}).get('best_sharpe', p1_sh)
        bpar = phase3.get(sn, {}).get('best_params', {})
        pc = sum(1 for v in [kf6, wf, era, mc] if v == 'PASS')
        if pc >= 4 and sens != 'FRAGILE' and slip in ('ROBUST','ACCEPTABLE'): final = 'STRONG_PASS'
        elif pc >= 3 and mc == 'PASS': final = 'CONDITIONAL_PASS'
        elif pc >= 2: final = 'WEAK_PASS'
        elif kf6 == 'PASS': final = 'MARGINAL'
        else: final = 'REJECT'
        phase10[sn] = {'best_sharpe': bsh, 'best_params': bpar, 'final_verdict': final,
                       'gates': {'kf6': kf6, 'wf': wf, 'era': era, 'mc': mc, 'sens': sens, 'slip': slip}}
        pf(f'  {sn:<20} Sh={bsh:.3f}  KF={kf6:<5} WF={wf:<5} Era={era:<5} MC={mc:<5} Sens={sens:<8} Slip={slip:<10} -> {final}')
    save('phase10_verdict', phase10)
    save_progress('phase10_DONE')

    elapsed = time.time() - t0
    pf(f'\n  Total runtime: {elapsed:.0f}s ({elapsed/3600:.1f}h)')
    pf(f'  Finished: {pd.Timestamp.now()}')

if __name__ == '__main__':
    main()
