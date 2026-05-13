#!/usr/bin/env python3
"""R223: H1 New Strategy Exploration (Beyond Keltner/TSMOM)
=============================================================
Engine: BacktestEngine (existing M15+H1 framework)
Data: Dukascopy H1+M15 XAUUSD 2015-2026

Explores new H1-timeframe strategies that use the existing BacktestEngine
by monkey-patching scan_all_signals. Each strategy is tested individually.

Strategies tested:
  1. H1 EMA20/EMA50 crossover (trend following)
  2. H1 MACD crossover (momentum)
  3. H1 CCI zero-cross + trend filter
  4. H1 RSI mean reversion (oversold in uptrend)
  5. H1 Donchian breakout (20-period)
  6. H1 Inside Bar breakout
  7. H1 BB squeeze release
  8. H1 ADX + DI crossover (directional movement)
  9. H1 Engulfing + EMA trend

Each strategy: baseline, era breakdown, 6-Fold K-Fold.
K-Fold passers get parameter sweep.
"""
from __future__ import annotations
import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.runner import DataBundle, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine, TradeRecord
import indicators as signals_mod

OUTPUT_DIR = Path("results/r223_h1_new_strategies")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERA_SEGMENTS = {
    "Pre-COVID (2015-2019)":      ("2015-01-01", "2020-01-01"),
    "COVID+Recovery (2020-2021)": ("2020-01-01", "2022-01-01"),
    "Tightening (2022-2023)":     ("2022-01-01", "2024-01-01"),
    "Recent (2024-2026)":         ("2024-01-01", "2026-06-01"),
}


def save(name, data):
    p = OUTPUT_DIR / f'{name}.json'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    print(f'  -> saved {p}')


def calc_stats(trades):
    if not trades:
        return {'n': 0, 'pnl': 0, 'sharpe': 0, 'win_rate': 0, 'avg_pnl': 0, 'max_dd': 0}
    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    cum = np.cumsum(pnls)
    dd = np.maximum.accumulate(cum) - cum
    sharpe = float(pnls.mean() / max(pnls.std(ddof=1), 1e-9) * np.sqrt(252)) if n > 1 else 0
    return {
        'n': n, 'pnl': round(float(pnls.sum()), 2),
        'sharpe': round(sharpe, 3),
        'win_rate': round(100 * (pnls > 0).sum() / n, 2),
        'avg_pnl': round(float(pnls.mean()), 2),
        'max_dd': round(float(dd.max()), 2),
    }


def filter_period(trades, start, end, strat=None):
    ts_s, ts_e = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    out = [t for t in trades if ts_s <= pd.Timestamp(t.entry_time) < ts_e]
    if strat:
        out = [t for t in out if t.strategy == strat]
    return out


def kfold_6(trades):
    if len(trades) < 30:
        return {'skip': True, 'verdict': 'SKIP'}
    pnls = np.array([t.pnl for t in trades])
    fold_size = len(pnls) // 6
    folds, kf_pass = [], 0
    for fold in range(6):
        s_idx = fold * fold_size
        e_idx = s_idx + fold_size if fold < 5 else len(pnls)
        fp = pnls[s_idx:e_idx]
        if len(fp) < 5:
            continue
        sh = float(fp.mean() / max(fp.std(ddof=1), 1e-9) * np.sqrt(252))
        folds.append({'fold': fold + 1, 'n': len(fp), 'sharpe': round(sh, 3)})
        if sh > 0:
            kf_pass += 1
    rate = kf_pass / max(len(folds), 1)
    return {'folds': folds, 'pass_count': kf_pass, 'total_folds': len(folds),
            'pass_rate': round(rate, 3), 'verdict': 'PASS' if rate >= 0.67 else 'FAIL'}


# ═══════════════════════════════════════════════════════════════
# H1 Signal functions (to be monkey-patched into scan_all_signals)
# ═══════════════════════════════════════════════════════════════

_current_signal_func = None
_current_strategy_name = None

_original_scan = signals_mod.scan_all_signals


def _check_h1_signal(df):
    """Generic wrapper: calls the currently-set signal function."""
    if _current_signal_func is None:
        return None
    return _current_signal_func(df)


def patched_scan_all_signals(df, timeframe='H1', h1_adx=None):
    """Replace scan_all_signals to only emit our custom strategy signal."""
    if timeframe != 'H1':
        return _original_scan(df, timeframe=timeframe, h1_adx=h1_adx)
    sig = _check_h1_signal(df)
    if sig:
        return [sig]
    return []


# ── Individual H1 signal functions ──

def h1_ema_cross(df):
    if len(df) < 55:
        return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    e20 = float(curr.get('ema_20', curr.get('EMA20', 0)))
    e50 = float(curr.get('ema_50', curr.get('EMA50', 0)))
    e20p = float(prev.get('ema_20', prev.get('EMA20', 0)))
    e50p = float(prev.get('ema_50', prev.get('EMA50', 0)))
    if pd.isna(e20) or pd.isna(e50) or e20 == 0 or e50 == 0:
        return None
    if e20 > e50 and e20p <= e50p:
        return {'strategy': 'h1_ema_cross', 'signal': 'BUY', 'direction': 'BUY'}
    if e20 < e50 and e20p >= e50p:
        return {'strategy': 'h1_ema_cross', 'signal': 'SELL', 'direction': 'SELL'}
    return None


def h1_macd_cross(df):
    if len(df) < 30:
        return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    macd = float(curr.get('macd', curr.get('MACD', 0)))
    sig = float(curr.get('macd_signal', curr.get('MACD_signal', 0)))
    macd_p = float(prev.get('macd', prev.get('MACD', 0)))
    sig_p = float(prev.get('macd_signal', prev.get('MACD_signal', 0)))
    if pd.isna(macd) or pd.isna(sig) or macd == 0:
        return None
    if macd > sig and macd_p <= sig_p:
        return {'strategy': 'h1_macd', 'signal': 'BUY', 'direction': 'BUY'}
    if macd < sig and macd_p >= sig_p:
        return {'strategy': 'h1_macd', 'signal': 'SELL', 'direction': 'SELL'}
    return None


def h1_cci_cross(df):
    if len(df) < 25:
        return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    cci = float(curr.get('cci', curr.get('CCI', 0)))
    cci_p = float(prev.get('cci', prev.get('CCI', 0)))
    slope = float(curr.get('ema_50_slope', curr.get('EMA50_slope', 0)))
    if pd.isna(cci) or pd.isna(cci_p):
        return None
    if cci > 0 and cci_p <= 0 and slope > 0:
        return {'strategy': 'h1_cci', 'signal': 'BUY', 'direction': 'BUY'}
    if cci < 0 and cci_p >= 0 and slope < 0:
        return {'strategy': 'h1_cci', 'signal': 'SELL', 'direction': 'SELL'}
    return None


def h1_rsi_revert(df):
    if len(df) < 55:
        return None
    curr = df.iloc[-1]
    rsi = float(curr.get('RSI14', curr.get('rsi_14', 50)))
    c = float(curr.get('Close', curr.get('close', 0)))
    ema50 = float(curr.get('EMA50', curr.get('ema_50', c)))
    if pd.isna(rsi) or pd.isna(ema50) or c == 0:
        return None
    if rsi < 30 and c > ema50:
        return {'strategy': 'h1_rsi', 'signal': 'BUY', 'direction': 'BUY'}
    if rsi > 70 and c < ema50:
        return {'strategy': 'h1_rsi', 'signal': 'SELL', 'direction': 'SELL'}
    return None


def h1_donchian(df):
    if len(df) < 25:
        return None
    curr = df.iloc[-1]
    c = float(curr.get('Close', curr.get('close', 0)))
    h_col = 'High' if 'High' in df.columns else 'high'
    l_col = 'Low' if 'Low' in df.columns else 'low'
    dc_u = df[h_col].iloc[-20:].max()
    dc_l = df[l_col].iloc[-20:].min()
    if c >= dc_u:
        return {'strategy': 'h1_donchian', 'signal': 'BUY', 'direction': 'BUY'}
    if c <= dc_l:
        return {'strategy': 'h1_donchian', 'signal': 'SELL', 'direction': 'SELL'}
    return None


def h1_inside_bar(df):
    if len(df) < 5:
        return None
    curr = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3]
    c = float(curr.get('Close', curr.get('close', 0)))
    h_k, l_k = ('High', 'Low') if 'High' in df.columns else ('high', 'low')
    was_inside = (float(prev1[h_k]) <= float(prev2[h_k]) and
                  float(prev1[l_k]) >= float(prev2[l_k]))
    if not was_inside:
        return None
    if c > float(prev2[h_k]):
        return {'strategy': 'h1_inside_bar', 'signal': 'BUY', 'direction': 'BUY'}
    if c < float(prev2[l_k]):
        return {'strategy': 'h1_inside_bar', 'signal': 'SELL', 'direction': 'SELL'}
    return None


def h1_bb_squeeze(df):
    if len(df) < 25:
        return None
    curr = df.iloc[-1]
    c = float(curr.get('Close', curr.get('close', 0)))
    bb_u = float(curr.get('BB_upper', curr.get('bb_upper', 0)))
    bb_l = float(curr.get('BB_lower', curr.get('bb_lower', 0)))
    kc_u = float(curr.get('KC_upper', curr.get('kc_upper', 0)))
    kc_l = float(curr.get('KC_lower', curr.get('kc_lower', 0)))
    if pd.isna(bb_u) or bb_u == 0 or pd.isna(kc_u) or kc_u == 0:
        return None
    is_squeeze = (bb_u < kc_u) and (bb_l > kc_l)
    if is_squeeze:
        return None
    sq_count = 0
    for j in range(max(0, len(df) - 11), len(df) - 1):
        r = df.iloc[j]
        bb_u_j = float(r.get('BB_upper', r.get('bb_upper', 0)))
        kc_u_j = float(r.get('KC_upper', r.get('kc_upper', 0)))
        bb_l_j = float(r.get('BB_lower', r.get('bb_lower', 0)))
        kc_l_j = float(r.get('KC_lower', r.get('kc_lower', 0)))
        if bb_u_j < kc_u_j and bb_l_j > kc_l_j:
            sq_count += 1
        else:
            sq_count = 0
    if sq_count < 5:
        return None
    kc_mid = float(curr.get('KC_mid', curr.get('kc_mid', 0)))
    if kc_mid == 0:
        return None
    if c > kc_mid:
        return {'strategy': 'h1_bb_squeeze', 'signal': 'BUY', 'direction': 'BUY'}
    else:
        return {'strategy': 'h1_bb_squeeze', 'signal': 'SELL', 'direction': 'SELL'}


def h1_adx_di_cross(df):
    """ADX > 25 + DI cross: +DI crosses above -DI -> BUY, vice versa."""
    if len(df) < 20:
        return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    adx = float(curr.get('adx', curr.get('ADX', 0)))
    plus_di = float(curr.get('plus_di', curr.get('PLUS_DI', 0)))
    minus_di = float(curr.get('minus_di', curr.get('MINUS_DI', 0)))
    plus_di_p = float(prev.get('plus_di', prev.get('PLUS_DI', 0)))
    minus_di_p = float(prev.get('minus_di', prev.get('MINUS_DI', 0)))
    if pd.isna(adx) or adx < 25:
        return None
    if plus_di > minus_di and plus_di_p <= minus_di_p:
        return {'strategy': 'h1_adx_di', 'signal': 'BUY', 'direction': 'BUY'}
    if plus_di < minus_di and plus_di_p >= minus_di_p:
        return {'strategy': 'h1_adx_di', 'signal': 'SELL', 'direction': 'SELL'}
    return None


def h1_engulfing(df):
    if len(df) < 10:
        return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    c_o = float(curr.get('Open', curr.get('open', 0)))
    c_c = float(curr.get('Close', curr.get('close', 0)))
    p_o = float(prev.get('Open', prev.get('open', 0)))
    p_c = float(prev.get('Close', prev.get('close', 0)))
    ema50 = float(curr.get('EMA50', curr.get('ema_50', c_c)))
    if pd.isna(ema50):
        return None
    curr_body = c_c - c_o
    prev_body = p_c - p_o
    if curr_body > 0 and prev_body < 0:
        if c_c > p_o and c_o < p_c and c_c > ema50:
            return {'strategy': 'h1_engulfing', 'signal': 'BUY', 'direction': 'BUY'}
    if curr_body < 0 and prev_body > 0:
        if c_c < p_o and c_o > p_c and c_c < ema50:
            return {'strategy': 'h1_engulfing', 'signal': 'SELL', 'direction': 'SELL'}
    return None


ALL_H1_STRATEGIES = [
    ('h1_ema_cross', h1_ema_cross),
    ('h1_macd', h1_macd_cross),
    ('h1_cci', h1_cci_cross),
    ('h1_rsi', h1_rsi_revert),
    ('h1_donchian', h1_donchian),
    ('h1_inside_bar', h1_inside_bar),
    ('h1_bb_squeeze', h1_bb_squeeze),
    ('h1_adx_di', h1_adx_di_cross),
    ('h1_engulfing', h1_engulfing),
]


def run_engine(data, strat_name, sig_func, **extra_kwargs):
    """Run BacktestEngine with monkey-patched signal for one strategy."""
    global _current_signal_func, _current_strategy_name
    _current_signal_func = sig_func
    _current_strategy_name = strat_name

    signals_mod.scan_all_signals = patched_scan_all_signals
    try:
        kwargs = dict(LIVE_PARITY_KWARGS)
        kwargs.update(extra_kwargs)
        engine = BacktestEngine(data.m15_df, data.h1_df, **kwargs)
        engine.run()
        return engine.trades
    finally:
        signals_mod.scan_all_signals = _original_scan
        _current_signal_func = None


def main():
    t_start = time.time()
    print('=' * 80)
    print('R223: H1 New Strategy Exploration')
    print('=' * 80)

    print('\nLoading data...')
    data = DataBundle.load_default()

    # ── Phase 1: Individual screening ──
    print('\n' + '=' * 80)
    print('Phase 1: Individual Strategy Screening')
    print('=' * 80)

    phase1 = {}
    viable = []

    for strat_name, sig_func in ALL_H1_STRATEGIES:
        print(f'\n  --- {strat_name} ---')
        trades = run_engine(data, strat_name, sig_func)
        st = [t for t in trades if t.strategy == strat_name]
        s = calc_stats(st)
        print(f'  {strat_name:<20} n={s["n"]:>5}  Sharpe={s["sharpe"]:.3f}  '
              f'PnL=${s["pnl"]:.0f}  WR={s["win_rate"]:.1f}%')

        eras = {}
        for era_name, (es, ee) in ERA_SEGMENTS.items():
            era_t = filter_period(st, es, ee, strat=strat_name)
            era_s = calc_stats(era_t)
            eras[era_name] = era_s
            print(f'    {era_name:<30} n={era_s["n"]:>4}  Sharpe={era_s["sharpe"]:.3f}')

        phase1[strat_name] = {'stats': s, 'eras': eras}
        if s['sharpe'] > 0.3 and s['n'] >= 30:
            viable.append(strat_name)

    save('phase1_screening', phase1)
    print(f'\n  Viable: {viable}')

    # ── Phase 2: K-Fold ──
    print('\n' + '=' * 80)
    print('Phase 2: 6-Fold K-Fold')
    print('=' * 80)

    phase2 = {}
    for strat_name in viable:
        sig_func = dict(ALL_H1_STRATEGIES)[strat_name]
        trades = run_engine(data, strat_name, sig_func)
        st = [t for t in trades if t.strategy == strat_name]
        print(f'\n  {strat_name}:')
        kf = kfold_6(st)
        if kf.get('folds'):
            for f in kf['folds']:
                print(f'    Fold {f["fold"]}: n={f["n"]:>4}  Sharpe={f["sharpe"]:.3f}')
        print(f'    Verdict: {kf.get("verdict", "SKIP")}')
        phase2[strat_name] = kf

    save('phase2_kfold', phase2)

    # ── Final Summary ──
    print('\n' + '=' * 80)
    print('FINAL SUMMARY')
    print('=' * 80)

    kf_passers = [s for s in viable if phase2.get(s, {}).get('verdict') == 'PASS']
    summary = []
    for strat_name, _ in ALL_H1_STRATEGIES:
        s = phase1[strat_name]['stats']
        kf = phase2.get(strat_name, {}).get('verdict', 'SKIP')
        if kf == 'PASS' and s['sharpe'] > 1.0:
            decision = 'CONSIDER'
        elif kf == 'PASS':
            decision = 'WEAK'
        elif s['sharpe'] <= 0:
            decision = 'REJECT'
        else:
            decision = 'INSUFFICIENT'
        summary.append({'strategy': strat_name, 'n': s['n'], 'sharpe': s['sharpe'],
                        'pnl': s['pnl'], 'kfold': kf, 'decision': decision})
        print(f'  {strat_name:<20} n={s["n"]:>5}  Sharpe={s["sharpe"]:.3f}  '
              f'KF={kf:<5}  -> {decision}')

    save('R223_summary', {
        'engine': 'BacktestEngine (LIVE_PARITY_KWARGS)',
        'strategies': summary, 'viable': viable, 'kf_passers': kf_passers,
    })

    elapsed = time.time() - t_start
    print(f'\n  Total runtime: {elapsed:.0f}s')


if __name__ == '__main__':
    main()
