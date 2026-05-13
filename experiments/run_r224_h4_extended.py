#!/usr/bin/env python3
"""R224: H4 Extended Strategy Exploration
===========================================
Engine: H4BacktestEngine
Data: Dukascopy H4 XAUUSD 2015-2026

Tests additional H4 strategies beyond R220:
  1. EMA9/EMA20 fast cross (shorter EMA pair)
  2. RSI(14) divergence (price vs RSI)
  3. ADX + DI crossover
  4. Stochastic K/D crossover in zones
  5. Double EMA ribbon (EMA20>EMA50>EMA100 alignment)
  6. Mean reversion (2 ATR deviation from EMA50)
  7. Momentum breakout (close > N-bar high by 1 ATR)
  8. Inside bar breakout on H4

Each: baseline, era breakdown, K-Fold. Passers get param sweep.
"""
from __future__ import annotations
import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.h4_engine import H4BacktestEngine, prepare_h4_indicators, load_h4_with_indicators
from backtest.engine import TradeRecord

OUTPUT_DIR = Path("results/r224_h4_extended")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERA_SEGMENTS = {
    "Pre-COVID (2015-2019)":      ("2015-01-01", "2020-01-01"),
    "COVID+Recovery (2020-2021)": ("2020-01-01", "2022-01-01"),
    "Tightening (2022-2023)":     ("2022-01-01", "2024-01-01"),
    "Recent (2024-2026)":         ("2024-01-01", "2026-06-01"),
}
SPREAD = 0.30


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
    return {'n': n, 'pnl': round(float(pnls.sum()), 2), 'sharpe': round(sharpe, 3),
            'win_rate': round(100 * (pnls > 0).sum() / n, 2),
            'avg_pnl': round(float(pnls.mean()), 2), 'max_dd': round(float(dd.max()), 2)}


def filter_period(trades, start, end):
    ts_s, ts_e = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    return [t for t in trades if ts_s <= pd.Timestamp(t.entry_time) < ts_e]


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
# Signal functions
# ═══════════════════════════════════════════════════════════════

def sig_ema_fast(df):
    """EMA9/EMA20 fast crossover."""
    if len(df) < 25:
        return None
    # Need EMA9 — compute on-the-fly if missing
    curr, prev = df.iloc[-1], df.iloc[-2]
    if 'EMA9' not in df.columns:
        ema9 = df['Close'].ewm(span=9, adjust=False).mean()
        e9, e9p = float(ema9.iloc[-1]), float(ema9.iloc[-2])
    else:
        e9, e9p = float(curr.get('EMA9', 0)), float(prev.get('EMA9', 0))
    e20, e20p = float(curr['EMA20']), float(prev['EMA20'])
    atr = float(curr.get('ATR', 0))
    if pd.isna(e9) or pd.isna(e20) or atr <= 0:
        return None
    if e9 > e20 and e9p <= e20p:
        return {'strategy': 'h4_ema_fast', 'signal': 'BUY'}
    if e9 < e20 and e9p >= e20p:
        return {'strategy': 'h4_ema_fast', 'signal': 'SELL'}
    return None


def sig_rsi_divergence(df):
    """Price makes lower low but RSI makes higher low (bullish divergence) and vice versa."""
    if len(df) < 30:
        return None
    curr = df.iloc[-1]
    atr = float(curr.get('ATR', 0))
    if atr <= 0:
        return None
    prices = df['Close'].values[-20:]
    rsi_vals = df['RSI14'].values[-20:] if 'RSI14' in df.columns else None
    if rsi_vals is None or np.any(np.isnan(rsi_vals)):
        return None
    # Find local lows
    if len(prices) < 10:
        return None
    # Bullish: price lower low, RSI higher low
    p_min1 = prices[:10].min()
    p_min2 = prices[10:].min()
    r_min1 = rsi_vals[:10].min()
    r_min2 = rsi_vals[10:].min()
    if p_min2 < p_min1 and r_min2 > r_min1 and float(curr['RSI14']) < 40:
        return {'strategy': 'h4_rsi_div', 'signal': 'BUY'}
    # Bearish: price higher high, RSI lower high
    p_max1 = prices[:10].max()
    p_max2 = prices[10:].max()
    r_max1 = rsi_vals[:10].max()
    r_max2 = rsi_vals[10:].max()
    if p_max2 > p_max1 and r_max2 < r_max1 and float(curr['RSI14']) > 60:
        return {'strategy': 'h4_rsi_div', 'signal': 'SELL'}
    return None


def sig_adx_di_cross(df):
    """ADX>25 + DI crossover."""
    if len(df) < 20:
        return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    adx = float(curr.get('ADX', 0))
    if pd.isna(adx) or adx < 25:
        return None
    # Compute +DI/-DI from raw data
    tr = pd.concat([df['High'] - df['Low'],
                    (df['High'] - df['Close'].shift(1)).abs(),
                    (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = (-df['Low'].diff()).clip(lower=0)
    plus_dm[df['High'].diff() <= (-df['Low'].diff())] = 0
    minus_dm[(-df['Low'].diff()) <= df['High'].diff()] = 0
    atr_s = tr.ewm(span=14, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr_s.replace(0, np.nan)
    mdi = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr_s.replace(0, np.nan)
    pdi_c, pdi_p = float(pdi.iloc[-1]), float(pdi.iloc[-2])
    mdi_c, mdi_p = float(mdi.iloc[-1]), float(mdi.iloc[-2])
    if pdi_c > mdi_c and pdi_p <= mdi_p:
        return {'strategy': 'h4_adx_di', 'signal': 'BUY'}
    if pdi_c < mdi_c and pdi_p >= mdi_p:
        return {'strategy': 'h4_adx_di', 'signal': 'SELL'}
    return None


def sig_stochastic(df):
    if len(df) < 20:
        return None
    # Compute Stochastic on-the-fly
    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    stoch_k = 100 * (df['Close'] - low14) / (high14 - low14).replace(0, np.nan)
    stoch_d = stoch_k.rolling(3).mean()
    k_c, d_c = float(stoch_k.iloc[-1]), float(stoch_d.iloc[-1])
    k_p, d_p = float(stoch_k.iloc[-2]), float(stoch_d.iloc[-2])
    if pd.isna(k_c) or pd.isna(d_c):
        return None
    if k_c > d_c and k_p <= d_p and k_c < 30:
        return {'strategy': 'h4_stoch', 'signal': 'BUY'}
    if k_c < d_c and k_p >= d_p and k_c > 70:
        return {'strategy': 'h4_stoch', 'signal': 'SELL'}
    return None


def sig_ema_ribbon(df):
    """Triple EMA alignment: EMA20>EMA50>EMA100 for BUY (and reverse SELL)."""
    if len(df) < 105:
        return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    e20 = float(curr['EMA20'])
    e50 = float(curr['EMA50'])
    e100 = float(curr.get('EMA100', 0))
    e20p = float(prev['EMA20'])
    e50p = float(prev['EMA50'])
    e100p = float(prev.get('EMA100', 0))
    if pd.isna(e100) or e100 == 0:
        return None
    # Just aligned now, wasn't before
    aligned_bull = (e20 > e50 > e100)
    aligned_bull_prev = (e20p > e50p > e100p)
    aligned_bear = (e20 < e50 < e100)
    aligned_bear_prev = (e20p < e50p < e100p)
    if aligned_bull and not aligned_bull_prev:
        return {'strategy': 'h4_ema_ribbon', 'signal': 'BUY'}
    if aligned_bear and not aligned_bear_prev:
        return {'strategy': 'h4_ema_ribbon', 'signal': 'SELL'}
    return None


def sig_mean_revert(df):
    """Mean reversion: 2 ATR deviation from EMA50."""
    if len(df) < 55:
        return None
    curr = df.iloc[-1]
    c = float(curr['Close'])
    ema50 = float(curr['EMA50'])
    atr = float(curr.get('ATR', 0))
    if pd.isna(ema50) or atr <= 0:
        return None
    dev = (c - ema50) / atr
    if dev < -2.0:
        return {'strategy': 'h4_mean_rev', 'signal': 'BUY'}
    if dev > 2.0:
        return {'strategy': 'h4_mean_rev', 'signal': 'SELL'}
    return None


def sig_momentum_breakout(df):
    """Close > 10-bar high + 0.5 ATR momentum breakout."""
    if len(df) < 15:
        return None
    curr = df.iloc[-1]
    c = float(curr['Close'])
    atr = float(curr.get('ATR', 0))
    if atr <= 0:
        return None
    high10 = float(df['High'].iloc[-11:-1].max())
    low10 = float(df['Low'].iloc[-11:-1].min())
    if c > high10 + 0.5 * atr:
        return {'strategy': 'h4_momentum', 'signal': 'BUY'}
    if c < low10 - 0.5 * atr:
        return {'strategy': 'h4_momentum', 'signal': 'SELL'}
    return None


def sig_inside_bar(df):
    if len(df) < 5:
        return None
    curr = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3]
    c = float(curr['Close'])
    was_inside = (float(prev1['High']) <= float(prev2['High']) and
                  float(prev1['Low']) >= float(prev2['Low']))
    if not was_inside:
        return None
    if c > float(prev2['High']):
        return {'strategy': 'h4_inside_bar', 'signal': 'BUY'}
    if c < float(prev2['Low']):
        return {'strategy': 'h4_inside_bar', 'signal': 'SELL'}
    return None


ALL_STRATEGIES = [
    ('h4_ema_fast', sig_ema_fast),
    ('h4_rsi_div', sig_rsi_divergence),
    ('h4_adx_di', sig_adx_di_cross),
    ('h4_stoch', sig_stochastic),
    ('h4_ema_ribbon', sig_ema_ribbon),
    ('h4_mean_rev', sig_mean_revert),
    ('h4_momentum', sig_momentum_breakout),
    ('h4_inside_bar', sig_inside_bar),
]


def main():
    t_start = time.time()
    print('=' * 80)
    print('R224: H4 Extended Strategy Exploration')
    print('=' * 80)

    h4_df = load_h4_with_indicators()

    # Add EMA9 if not present
    if 'EMA9' not in h4_df.columns:
        h4_df['EMA9'] = h4_df['Close'].ewm(span=9, adjust=False).mean()

    # ── Phase 1: Individual screening ──
    print('\n' + '=' * 80)
    print('Phase 1: Individual Strategy Screening')
    print('=' * 80)

    phase1 = {}
    viable = []

    for strat_name, sig_func in ALL_STRATEGIES:
        print(f'\n  --- {strat_name} ---')
        engine = H4BacktestEngine(
            h4_df, signal_funcs=[(strat_name, sig_func)],
            sl_atr_mult=3.0, tp_atr_mult=6.0,
            trailing_activate_atr=0.3, trailing_distance_atr=0.08,
            max_hold=30, cooldown_bars=2, spread_cost=SPREAD,
        )
        trades = engine.run()
        st = [t for t in trades if t.strategy == strat_name]
        s = calc_stats(st)
        print(f'  {strat_name:<20} n={s["n"]:>5}  Sharpe={s["sharpe"]:.3f}  '
              f'PnL=${s["pnl"]:.0f}  WR={s["win_rate"]:.1f}%')

        eras = {}
        for era_name, (es, ee) in ERA_SEGMENTS.items():
            era_t = filter_period(st, es, ee)
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
        sig_func = dict(ALL_STRATEGIES)[strat_name]
        engine = H4BacktestEngine(
            h4_df, signal_funcs=[(strat_name, sig_func)],
            sl_atr_mult=3.0, tp_atr_mult=6.0,
            trailing_activate_atr=0.3, trailing_distance_atr=0.08,
            max_hold=30, cooldown_bars=2, spread_cost=SPREAD,
        )
        trades = engine.run()
        st = [t for t in trades if t.strategy == strat_name]
        print(f'\n  {strat_name}:')
        kf = kfold_6(st)
        if kf.get('folds'):
            for f in kf['folds']:
                print(f'    Fold {f["fold"]}: n={f["n"]:>4}  Sharpe={f["sharpe"]:.3f}')
        print(f'    Verdict: {kf.get("verdict", "SKIP")}')
        phase2[strat_name] = kf

    save('phase2_kfold', phase2)

    # ── Phase 3: Param sweep on passers ──
    kf_passers = [s for s in viable if phase2.get(s, {}).get('verdict') == 'PASS']
    phase3 = {}

    if kf_passers:
        print('\n' + '=' * 80)
        print('Phase 3: Parameter Sweep')
        print('=' * 80)

        for strat_name in kf_passers:
            sig_func = dict(ALL_STRATEGIES)[strat_name]
            print(f'\n  --- {strat_name} sweep ---')
            best_sharpe, best_params = -999, None
            sweep = []

            for sl_m in [2.0, 3.0, 4.0, 5.0]:
                for tp_m in [4.0, 6.0, 8.0, 10.0]:
                    for trail_a, trail_d in [(0.0, 0.0), (0.3, 0.08), (0.5, 0.15)]:
                        engine = H4BacktestEngine(
                            h4_df, signal_funcs=[(strat_name, sig_func)],
                            sl_atr_mult=sl_m, tp_atr_mult=tp_m,
                            trailing_activate_atr=trail_a, trailing_distance_atr=trail_d,
                            max_hold=30, cooldown_bars=2, spread_cost=SPREAD,
                        )
                        trades = engine.run()
                        st = [t for t in trades if t.strategy == strat_name]
                        s = calc_stats(st)
                        sweep.append({'sl': sl_m, 'tp': tp_m, 'trail_a': trail_a, 'trail_d': trail_d, **s})
                        if s['sharpe'] > best_sharpe and s['n'] >= 20:
                            best_sharpe = s['sharpe']
                            best_params = {'sl': sl_m, 'tp': tp_m, 'trail_act': trail_a, 'trail_dist': trail_d}

            sweep.sort(key=lambda x: x['sharpe'], reverse=True)
            for r in sweep[:5]:
                print(f'    SL{r["sl"]}_TP{r["tp"]}_T{r["trail_a"]}/{r["trail_d"]}  '
                      f'n={r["n"]:>4} Sh={r["sharpe"]:.3f} PnL=${r["pnl"]:.0f}')
            phase3[strat_name] = {'best_params': best_params, 'best_sharpe': best_sharpe, 'top5': sweep[:5]}

        save('phase3_param_sweep', phase3)

    # ── Final Summary ──
    print('\n' + '=' * 80)
    print('FINAL SUMMARY')
    print('=' * 80)

    summary = []
    for strat_name, _ in ALL_STRATEGIES:
        s = phase1[strat_name]['stats']
        kf = phase2.get(strat_name, {}).get('verdict', 'SKIP')
        ps = phase3.get(strat_name, {}).get('best_sharpe', None)
        if kf == 'PASS' and s['sharpe'] > 1.0:
            decision = 'CONSIDER'
        elif kf == 'PASS':
            decision = 'WEAK'
        elif s['sharpe'] <= 0:
            decision = 'REJECT'
        else:
            decision = 'INSUFFICIENT'
        summary.append({'strategy': strat_name, 'n': s['n'], 'sharpe': s['sharpe'],
                        'pnl': s['pnl'], 'wr': s['win_rate'], 'kfold': kf,
                        'decision': decision, 'best_sweep_sharpe': ps})
        print(f'  {strat_name:<20} n={s["n"]:>5}  Sharpe={s["sharpe"]:.3f}  '
              f'KF={kf:<5}  -> {decision}')

    save('R224_summary', {
        'engine': 'H4BacktestEngine', 'data': 'XAUUSD H4 2015-2026',
        'strategies': summary, 'viable': viable, 'kf_passers': kf_passers,
        'phase3_best': {k: v['best_params'] for k, v in phase3.items()},
    })

    elapsed = time.time() - t_start
    print(f'\n  Total runtime: {elapsed:.0f}s')


if __name__ == '__main__':
    main()
