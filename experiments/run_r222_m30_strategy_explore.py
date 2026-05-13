#!/usr/bin/env python3
"""R222: M30 Strategy Exploration
====================================
Engine: M30BacktestEngine (new)
Data: Dukascopy M30 XAUUSD 2015-2026 (134K bars)

Tests M30-specific strategy families:
  1. Keltner Channel breakout
  2. EMA9/EMA20 fast crossover
  3. EMA20/EMA50 crossover
  4. MACD crossover
  5. RSI(6) extreme mean reversion
  6. RSI(14) + EMA trend filter
  7. CCI zero-line momentum
  8. Stochastic oversold/overbought
  9. Bollinger Band squeeze release
  10. VWAP-like mean reversion (close vs SMA20)
  11. Inside Bar breakout
  12. Engulfing candle pattern

Each strategy: baseline stats, era breakdown, 6-Fold K-Fold.
K-Fold passers get parameter sweep.
"""
from __future__ import annotations
import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.m30_engine import M30BacktestEngine, load_m30_with_indicators
from backtest.engine import TradeRecord

OUTPUT_DIR = Path("results/r222_m30_explore")
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
    return {
        'n': n, 'pnl': round(float(pnls.sum()), 2),
        'sharpe': round(sharpe, 3),
        'win_rate': round(100 * (pnls > 0).sum() / n, 2),
        'avg_pnl': round(float(pnls.mean()), 2),
        'max_dd': round(float(dd.max()), 2),
    }


def filter_period(trades, start, end):
    ts_s, ts_e = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    return [t for t in trades if ts_s <= pd.Timestamp(t.entry_time) < ts_e]


def kfold_6(trades):
    if len(trades) < 30:
        return {'skip': True, 'reason': f'n={len(trades)} < 30', 'verdict': 'SKIP'}
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
# Signal functions — M30-specific
# ═══════════════════════════════════════════════════════════════

def sig_kc_breakout(df):
    if len(df) < 30:
        return None
    row = df.iloc[-1]
    c, kc_u, kc_l = float(row['Close']), float(row.get('KC_upper', 0)), float(row.get('KC_lower', 0))
    atr = float(row.get('ATR', 0))
    if pd.isna(kc_u) or kc_u == 0 or atr <= 0:
        return None
    if c > kc_u:
        return {'strategy': 'm30_kc', 'signal': 'BUY'}
    if c < kc_l:
        return {'strategy': 'm30_kc', 'signal': 'SELL'}
    return None


def sig_ema_fast_cross(df):
    """EMA9 x EMA20 fast crossover."""
    if len(df) < 25:
        return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    e9, e20 = float(curr['EMA9']), float(curr['EMA20'])
    e9p, e20p = float(prev['EMA9']), float(prev['EMA20'])
    atr = float(curr.get('ATR', 0))
    if pd.isna(e9) or pd.isna(e20) or atr <= 0:
        return None
    if e9 > e20 and e9p <= e20p:
        return {'strategy': 'm30_ema_fast', 'signal': 'BUY'}
    if e9 < e20 and e9p >= e20p:
        return {'strategy': 'm30_ema_fast', 'signal': 'SELL'}
    return None


def sig_ema_cross(df):
    """EMA20 x EMA50 crossover."""
    if len(df) < 55:
        return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    e20, e50 = float(curr['EMA20']), float(curr['EMA50'])
    e20p, e50p = float(prev['EMA20']), float(prev['EMA50'])
    atr = float(curr.get('ATR', 0))
    if pd.isna(e20) or pd.isna(e50) or atr <= 0:
        return None
    if e20 > e50 and e20p <= e50p:
        return {'strategy': 'm30_ema_cross', 'signal': 'BUY'}
    if e20 < e50 and e20p >= e50p:
        return {'strategy': 'm30_ema_cross', 'signal': 'SELL'}
    return None


def sig_macd_cross(df):
    if len(df) < 30:
        return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    macd, sig = float(curr['MACD']), float(curr['MACD_signal'])
    macd_p, sig_p = float(prev['MACD']), float(prev['MACD_signal'])
    atr = float(curr.get('ATR', 0))
    if pd.isna(macd) or pd.isna(sig) or atr <= 0:
        return None
    if macd > sig and macd_p <= sig_p:
        return {'strategy': 'm30_macd', 'signal': 'BUY'}
    if macd < sig and macd_p >= sig_p:
        return {'strategy': 'm30_macd', 'signal': 'SELL'}
    return None


def sig_rsi6_extreme(df):
    """RSI(6) extreme reversal: <15 BUY, >85 SELL with EMA200 trend."""
    if len(df) < 20:
        return None
    row = df.iloc[-1]
    rsi6 = float(row.get('RSI6', 50))
    c = float(row['Close'])
    ema200 = float(row.get('EMA200', c))
    atr = float(row.get('ATR', 0))
    if pd.isna(rsi6) or atr <= 0 or pd.isna(ema200):
        return None
    if rsi6 < 15 and c > ema200:
        return {'strategy': 'm30_rsi6', 'signal': 'BUY'}
    if rsi6 > 85 and c < ema200:
        return {'strategy': 'm30_rsi6', 'signal': 'SELL'}
    return None


def sig_rsi14_trend(df):
    """RSI(14) with EMA50 trend: RSI<30 BUY in uptrend, RSI>70 SELL in downtrend."""
    if len(df) < 55:
        return None
    row = df.iloc[-1]
    rsi = float(row.get('RSI14', 50))
    c = float(row['Close'])
    ema50 = float(row.get('EMA50', c))
    slope = float(row.get('EMA50_slope', 0))
    atr = float(row.get('ATR', 0))
    if pd.isna(rsi) or atr <= 0:
        return None
    if rsi < 30 and c > ema50 and slope > 0:
        return {'strategy': 'm30_rsi14', 'signal': 'BUY'}
    if rsi > 70 and c < ema50 and slope < 0:
        return {'strategy': 'm30_rsi14', 'signal': 'SELL'}
    return None


def sig_cci_momentum(df):
    if len(df) < 25:
        return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    cci = float(curr.get('CCI', 0))
    cci_p = float(prev.get('CCI', 0))
    slope = float(curr.get('EMA50_slope', 0))
    atr = float(curr.get('ATR', 0))
    if pd.isna(cci) or pd.isna(cci_p) or atr <= 0:
        return None
    if cci > 0 and cci_p <= 0 and slope > 0:
        return {'strategy': 'm30_cci', 'signal': 'BUY'}
    if cci < 0 and cci_p >= 0 and slope < 0:
        return {'strategy': 'm30_cci', 'signal': 'SELL'}
    return None


def sig_stochastic(df):
    """Stochastic K/D crossover in oversold/overbought zones."""
    if len(df) < 20:
        return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    k, d = float(curr.get('STOCH_K', 50)), float(curr.get('STOCH_D', 50))
    kp, dp = float(prev.get('STOCH_K', 50)), float(prev.get('STOCH_D', 50))
    atr = float(curr.get('ATR', 0))
    if pd.isna(k) or pd.isna(d) or atr <= 0:
        return None
    if k > d and kp <= dp and k < 30:
        return {'strategy': 'm30_stoch', 'signal': 'BUY'}
    if k < d and kp >= dp and k > 70:
        return {'strategy': 'm30_stoch', 'signal': 'SELL'}
    return None


def sig_bb_squeeze(df):
    if len(df) < 15:
        return None
    row = df.iloc[-1]
    bb_u, bb_l = float(row.get('BB_upper', 0)), float(row.get('BB_lower', 0))
    kc_u, kc_l = float(row.get('KC_upper', 0)), float(row.get('KC_lower', 0))
    c, atr = float(row['Close']), float(row.get('ATR', 0))
    if pd.isna(bb_u) or pd.isna(kc_u) or kc_u == 0 or atr <= 0:
        return None
    if (bb_u < kc_u) and (bb_l > kc_l):
        return None
    squeeze_count = 0
    for j in range(max(0, len(df) - 11), len(df) - 1):
        r = df.iloc[j]
        if (float(r.get('BB_upper', 0)) < float(r.get('KC_upper', 0))
            and float(r.get('BB_lower', 0)) > float(r.get('KC_lower', 0))):
            squeeze_count += 1
        else:
            squeeze_count = 0
    if squeeze_count < 5:
        return None
    kc_mid = float(row.get('KC_mid', 0))
    if c > kc_mid:
        return {'strategy': 'm30_squeeze', 'signal': 'BUY'}
    else:
        return {'strategy': 'm30_squeeze', 'signal': 'SELL'}


def sig_mean_revert_sma(df):
    """Mean reversion: close deviates >2 ATR from SMA20, snap back."""
    if len(df) < 25:
        return None
    row = df.iloc[-1]
    c = float(row['Close'])
    sma20 = float(row.get('SMA20', c))
    atr = float(row.get('ATR', 0))
    ema50_slope = float(row.get('EMA50_slope', 0))
    if pd.isna(sma20) or atr <= 0:
        return None
    deviation = (c - sma20) / atr
    if deviation < -2.0 and ema50_slope > -0.5:
        return {'strategy': 'm30_mean_rev', 'signal': 'BUY'}
    if deviation > 2.0 and ema50_slope < 0.5:
        return {'strategy': 'm30_mean_rev', 'signal': 'SELL'}
    return None


def sig_inside_bar(df):
    """Inside bar breakout: bar fully inside previous bar, enter on break."""
    if len(df) < 5:
        return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    c_h, c_l, c_c = float(curr['High']), float(curr['Low']), float(curr['Close'])
    p_h, p_l = float(prev['High']), float(prev['Low'])
    atr = float(curr.get('ATR', 0))
    if atr <= 0:
        return None
    was_inside = (float(df.iloc[-3]['High']) <= float(df.iloc[-4]['High']) and
                  float(df.iloc[-3]['Low']) >= float(df.iloc[-4]['Low'])) if len(df) >= 5 else False
    if not was_inside:
        return None
    mother_high = float(df.iloc[-4]['High'])
    mother_low = float(df.iloc[-4]['Low'])
    if c_c > mother_high:
        return {'strategy': 'm30_inside_bar', 'signal': 'BUY'}
    if c_c < mother_low:
        return {'strategy': 'm30_inside_bar', 'signal': 'SELL'}
    return None


def sig_engulfing(df):
    """Engulfing candle pattern with trend confirmation."""
    if len(df) < 10:
        return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    c_o, c_c = float(curr['Open']) if 'Open' in curr.index else float(curr.get('open', 0)), float(curr['Close'])
    p_o, p_c = float(prev['Open']) if 'Open' in prev.index else float(prev.get('open', 0)), float(prev['Close'])
    atr = float(curr.get('ATR', 0))
    ema50 = float(curr.get('EMA50', c_c))
    if atr <= 0 or pd.isna(ema50):
        return None
    curr_body = c_c - c_o
    prev_body = p_c - p_o
    if curr_body > 0 and prev_body < 0:
        if c_c > p_o and c_o < p_c and c_c > ema50:
            return {'strategy': 'm30_engulf', 'signal': 'BUY'}
    if curr_body < 0 and prev_body > 0:
        if c_c < p_o and c_o > p_c and c_c < ema50:
            return {'strategy': 'm30_engulf', 'signal': 'SELL'}
    return None


ALL_STRATEGIES = [
    ('m30_kc', sig_kc_breakout),
    ('m30_ema_fast', sig_ema_fast_cross),
    ('m30_ema_cross', sig_ema_cross),
    ('m30_macd', sig_macd_cross),
    ('m30_rsi6', sig_rsi6_extreme),
    ('m30_rsi14', sig_rsi14_trend),
    ('m30_cci', sig_cci_momentum),
    ('m30_stoch', sig_stochastic),
    ('m30_squeeze', sig_bb_squeeze),
    ('m30_mean_rev', sig_mean_revert_sma),
    ('m30_inside_bar', sig_inside_bar),
    ('m30_engulf', sig_engulfing),
]


def main():
    t_start = time.time()
    print('=' * 80)
    print('R222: M30 Strategy Exploration')
    print('=' * 80)

    m30_df = load_m30_with_indicators()

    # ── Phase 1: Individual Screening ──
    print('\n' + '=' * 80)
    print('Phase 1: Individual Strategy Screening')
    print('=' * 80)

    phase1 = {}
    viable = []

    for strat_name, sig_func in ALL_STRATEGIES:
        print(f'\n  --- {strat_name} ---')
        engine = M30BacktestEngine(
            m30_df, signal_funcs=[(strat_name, sig_func)],
            sl_atr_mult=2.0, tp_atr_mult=4.0,
            trailing_activate_atr=0.3, trailing_distance_atr=0.08,
            max_hold=48, cooldown_bars=4, spread_cost=SPREAD,
        )
        trades = engine.run()
        st = [t for t in trades if t.strategy == strat_name]
        s = calc_stats(st)
        print(f'  {strat_name:<20} n={s["n"]:>5}  Sharpe={s["sharpe"]:.3f}  '
              f'PnL=${s["pnl"]:.0f}  WR={s["win_rate"]:.1f}%  MaxDD=${s["max_dd"]:.0f}')

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
    print(f'\n  Viable (Sharpe>0.3, n>=30): {viable}')

    # ── Phase 2: K-Fold on viable ──
    print('\n' + '=' * 80)
    print('Phase 2: 6-Fold K-Fold')
    print('=' * 80)

    phase2 = {}
    for strat_name in viable:
        sig_func = dict(ALL_STRATEGIES)[strat_name]
        engine = M30BacktestEngine(
            m30_df, signal_funcs=[(strat_name, sig_func)],
            sl_atr_mult=2.0, tp_atr_mult=4.0,
            trailing_activate_atr=0.3, trailing_distance_atr=0.08,
            max_hold=48, cooldown_bars=4, spread_cost=SPREAD,
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

    # ── Phase 3: Parameter sweep on K-Fold passers ──
    kf_passers = [s for s in viable if phase2.get(s, {}).get('verdict') == 'PASS']
    print(f'\n  K-Fold passers: {kf_passers}')

    if kf_passers:
        print('\n' + '=' * 80)
        print('Phase 3: Parameter Sweep')
        print('=' * 80)

        phase3 = {}
        for strat_name in kf_passers:
            sig_func = dict(ALL_STRATEGIES)[strat_name]
            print(f'\n  --- {strat_name} sweep ---')
            best_sharpe, best_params = -999, None
            sweep_results = []

            for sl_m in [1.5, 2.0, 3.0, 4.0]:
                for tp_m in [3.0, 4.0, 6.0, 8.0]:
                    for trail_a, trail_d in [(0.0, 0.0), (0.2, 0.06), (0.3, 0.08), (0.5, 0.15)]:
                        engine = M30BacktestEngine(
                            m30_df, signal_funcs=[(strat_name, sig_func)],
                            sl_atr_mult=sl_m, tp_atr_mult=tp_m,
                            trailing_activate_atr=trail_a, trailing_distance_atr=trail_d,
                            max_hold=48, cooldown_bars=4, spread_cost=SPREAD,
                        )
                        trades = engine.run()
                        st = [t for t in trades if t.strategy == strat_name]
                        s = calc_stats(st)
                        sweep_results.append({'sl': sl_m, 'tp': tp_m, 'trail_a': trail_a, 'trail_d': trail_d, **s})
                        if s['sharpe'] > best_sharpe and s['n'] >= 20:
                            best_sharpe = s['sharpe']
                            best_params = {'sl': sl_m, 'tp': tp_m, 'trail_act': trail_a, 'trail_dist': trail_d}

            sweep_results.sort(key=lambda x: x['sharpe'], reverse=True)
            for r in sweep_results[:5]:
                print(f'    SL{r["sl"]}_TP{r["tp"]}_T{r["trail_a"]}/{r["trail_d"]}  '
                      f'n={r["n"]:>4} Sharpe={r["sharpe"]:.3f} PnL=${r["pnl"]:.0f}')
            print(f'  Best: {best_params}  Sharpe={best_sharpe:.3f}')
            phase3[strat_name] = {'best_params': best_params, 'best_sharpe': best_sharpe,
                                  'top5': sweep_results[:5]}

        save('phase3_param_sweep', phase3)
    else:
        phase3 = {}

    # ── Final Summary ──
    print('\n' + '=' * 80)
    print('FINAL SUMMARY')
    print('=' * 80)

    summary_table = []
    for strat_name, _ in ALL_STRATEGIES:
        s = phase1[strat_name]['stats']
        kf = phase2.get(strat_name, {}).get('verdict', 'SKIP')
        ps = phase3.get(strat_name, {}).get('best_sharpe', None)
        if kf == 'PASS' and s['sharpe'] > 1.0:
            decision = 'CONSIDER'
        elif kf == 'PASS' and s['sharpe'] > 0.5:
            decision = 'WEAK'
        elif s['sharpe'] <= 0:
            decision = 'REJECT'
        else:
            decision = 'INSUFFICIENT'
        summary_table.append({'strategy': strat_name, 'n': s['n'], 'sharpe': s['sharpe'],
                              'pnl': s['pnl'], 'wr': s['win_rate'], 'kfold': kf,
                              'decision': decision, 'best_sweep_sharpe': ps})
        print(f'  {strat_name:<20} n={s["n"]:>5}  Sharpe={s["sharpe"]:.3f}  '
              f'KF={kf:<5}  -> {decision}')

    save('R222_summary', {
        'engine': 'M30BacktestEngine', 'data': 'XAUUSD M30 2015-2026',
        'strategies': summary_table, 'viable': viable, 'kf_passers': kf_passers,
        'phase3_best': {k: v['best_params'] for k, v in phase3.items()},
    })

    elapsed = time.time() - t_start
    print(f'\n  Total runtime: {elapsed:.0f}s')


if __name__ == '__main__':
    main()
