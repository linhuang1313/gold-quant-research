#!/usr/bin/env python3
"""R220: H4 Strategy Exploration
==================================
Engine: H4BacktestEngine (new)
Data: Dukascopy H4 XAUUSD 2015-2026

Tests multiple H4 strategy families:
  1. Keltner Channel breakout (H4 KC)
  2. Donchian Channel breakout
  3. EMA crossover (EMA20 x EMA50)
  4. MACD crossover
  5. RSI mean reversion (extreme levels)
  6. CCI momentum / mean reversion
  7. Bollinger Band squeeze release

Each strategy gets: baseline stats, era breakdown, 6-Fold K-Fold.
Best candidates go through parameter sweep.
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

from backtest.h4_engine import (
    H4BacktestEngine, prepare_h4_indicators, load_h4_with_indicators
)
from backtest.engine import TradeRecord

OUTPUT_DIR = Path("results/r220_h4_explore")
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
        'n': n,
        'pnl': round(float(pnls.sum()), 2),
        'sharpe': round(sharpe, 3),
        'win_rate': round(100 * (pnls > 0).sum() / n, 2),
        'avg_pnl': round(float(pnls.mean()), 2),
        'max_dd': round(float(dd.max()), 2),
    }


def filter_period(trades, start, end):
    ts_s = pd.Timestamp(start, tz='UTC')
    ts_e = pd.Timestamp(end, tz='UTC')
    return [t for t in trades if ts_s <= pd.Timestamp(t.entry_time) < ts_e]


def kfold_6(trades):
    if len(trades) < 30:
        return {'skip': True, 'reason': f'n={len(trades)} < 30'}
    pnls = np.array([t.pnl for t in trades])
    fold_size = len(pnls) // 6
    folds = []
    kf_pass = 0
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
    return {
        'folds': folds, 'pass_count': kf_pass,
        'total_folds': len(folds), 'pass_rate': round(rate, 3),
        'verdict': 'PASS' if rate >= 0.67 else 'FAIL',
    }


# ═══════════════════════════════════════════════════════════════
# Signal functions
# ═══════════════════════════════════════════════════════════════

def sig_kc_breakout(df: pd.DataFrame) -> Optional[Dict]:
    """H4 Keltner Channel breakout."""
    if len(df) < 30:
        return None
    row = df.iloc[-1]
    c = float(row['Close'])
    kc_u = float(row.get('KC_upper', 0))
    kc_l = float(row.get('KC_lower', 0))
    atr = float(row.get('ATR', 0))
    if pd.isna(kc_u) or kc_u == 0 or atr <= 0:
        return None
    if c > kc_u:
        return {'strategy': 'h4_kc', 'signal': 'BUY', 'sl': atr * 3, 'tp': atr * 6}
    if c < kc_l:
        return {'strategy': 'h4_kc', 'signal': 'SELL', 'sl': atr * 3, 'tp': atr * 6}
    return None


def sig_donchian_breakout(df: pd.DataFrame) -> Optional[Dict]:
    """H4 Donchian Channel breakout."""
    if len(df) < 25:
        return None
    row = df.iloc[-1]
    c = float(row['Close'])
    dc_u = float(row.get('DC_upper', 0))
    dc_l = float(row.get('DC_lower', 0))
    atr = float(row.get('ATR', 0))
    if pd.isna(dc_u) or dc_u == 0 or atr <= 0:
        return None
    if c >= dc_u:
        return {'strategy': 'h4_donchian', 'signal': 'BUY', 'sl': atr * 3, 'tp': atr * 6}
    if c <= dc_l:
        return {'strategy': 'h4_donchian', 'signal': 'SELL', 'sl': atr * 3, 'tp': atr * 6}
    return None


def sig_ema_cross(df: pd.DataFrame) -> Optional[Dict]:
    """EMA20 / EMA50 crossover."""
    if len(df) < 55:
        return None
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    ema20 = float(curr['EMA20'])
    ema50 = float(curr['EMA50'])
    ema20_prev = float(prev['EMA20'])
    ema50_prev = float(prev['EMA50'])
    atr = float(curr.get('ATR', 0))
    if pd.isna(ema20) or pd.isna(ema50) or atr <= 0:
        return None
    if ema20 > ema50 and ema20_prev <= ema50_prev:
        return {'strategy': 'h4_ema_cross', 'signal': 'BUY', 'sl': atr * 3, 'tp': atr * 6}
    if ema20 < ema50 and ema20_prev >= ema50_prev:
        return {'strategy': 'h4_ema_cross', 'signal': 'SELL', 'sl': atr * 3, 'tp': atr * 6}
    return None


def sig_macd_cross(df: pd.DataFrame) -> Optional[Dict]:
    """MACD line crosses signal line."""
    if len(df) < 30:
        return None
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    macd = float(curr['MACD'])
    sig_line = float(curr['MACD_signal'])
    macd_prev = float(prev['MACD'])
    sig_prev = float(prev['MACD_signal'])
    atr = float(curr.get('ATR', 0))
    if pd.isna(macd) or pd.isna(sig_line) or atr <= 0:
        return None
    if macd > sig_line and macd_prev <= sig_prev:
        return {'strategy': 'h4_macd', 'signal': 'BUY', 'sl': atr * 3, 'tp': atr * 6}
    if macd < sig_line and macd_prev >= sig_prev:
        return {'strategy': 'h4_macd', 'signal': 'SELL', 'sl': atr * 3, 'tp': atr * 6}
    return None


def sig_rsi_extreme(df: pd.DataFrame) -> Optional[Dict]:
    """RSI(14) extreme mean reversion: <25 BUY, >75 SELL."""
    if len(df) < 20:
        return None
    row = df.iloc[-1]
    rsi = float(row.get('RSI14', 50))
    atr = float(row.get('ATR', 0))
    ema100 = float(row.get('EMA100', 0))
    c = float(row['Close'])
    if pd.isna(rsi) or atr <= 0 or pd.isna(ema100):
        return None
    if rsi < 25 and c > ema100:
        return {'strategy': 'h4_rsi', 'signal': 'BUY', 'sl': atr * 2, 'tp': atr * 4}
    if rsi > 75 and c < ema100:
        return {'strategy': 'h4_rsi', 'signal': 'SELL', 'sl': atr * 2, 'tp': atr * 4}
    return None


def sig_cci_momentum(df: pd.DataFrame) -> Optional[Dict]:
    """CCI(20) zero-line cross with trend confirmation."""
    if len(df) < 25:
        return None
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    cci = float(curr.get('CCI', 0))
    cci_prev = float(prev.get('CCI', 0))
    atr = float(curr.get('ATR', 0))
    ema50_slope = float(curr.get('EMA50_slope', 0))
    if pd.isna(cci) or pd.isna(cci_prev) or atr <= 0:
        return None
    if cci > 0 and cci_prev <= 0 and ema50_slope > 0:
        return {'strategy': 'h4_cci', 'signal': 'BUY', 'sl': atr * 3, 'tp': atr * 6}
    if cci < 0 and cci_prev >= 0 and ema50_slope < 0:
        return {'strategy': 'h4_cci', 'signal': 'SELL', 'sl': atr * 3, 'tp': atr * 6}
    return None


def sig_bb_squeeze(df: pd.DataFrame) -> Optional[Dict]:
    """BB squeeze release: BB inside KC for 5+ bars, then release."""
    if len(df) < 15:
        return None
    row = df.iloc[-1]
    bb_u = float(row.get('BB_upper', 0))
    bb_l = float(row.get('BB_lower', 0))
    kc_u = float(row.get('KC_upper', 0))
    kc_l = float(row.get('KC_lower', 0))
    c = float(row['Close'])
    atr = float(row.get('ATR', 0))

    if pd.isna(bb_u) or pd.isna(kc_u) or kc_u == 0 or atr <= 0:
        return None

    is_squeeze = (bb_u < kc_u) and (bb_l > kc_l)
    if is_squeeze:
        return None

    # Check previous bars for squeeze
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
        return {'strategy': 'h4_squeeze', 'signal': 'BUY', 'sl': atr * 2.5, 'tp': atr * 5}
    else:
        return {'strategy': 'h4_squeeze', 'signal': 'SELL', 'sl': atr * 2.5, 'tp': atr * 5}


ALL_STRATEGIES = [
    ('h4_kc', sig_kc_breakout),
    ('h4_donchian', sig_donchian_breakout),
    ('h4_ema_cross', sig_ema_cross),
    ('h4_macd', sig_macd_cross),
    ('h4_rsi', sig_rsi_extreme),
    ('h4_cci', sig_cci_momentum),
    ('h4_squeeze', sig_bb_squeeze),
]


def main():
    t_start = time.time()
    print('=' * 80)
    print('R220: H4 Strategy Exploration')
    print('=' * 80)
    print('  Engine: H4BacktestEngine (new)')
    print('  Data: XAUUSD H4 2015-2026')

    h4_df = load_h4_with_indicators()

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Baseline — test each strategy individually
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 1: Individual Strategy Screening')
    print('=' * 80)

    phase1 = {}
    viable_strategies = []

    for strat_name, sig_func in ALL_STRATEGIES:
        print(f'\n  --- {strat_name} ---')
        engine = H4BacktestEngine(
            h4_df,
            signal_funcs=[(strat_name, sig_func)],
            sl_atr_mult=3.0,
            tp_atr_mult=6.0,
            trailing_activate_atr=0.5,
            trailing_distance_atr=0.15,
            max_hold=30,
            cooldown_bars=2,
            spread_cost=SPREAD,
        )
        trades = engine.run()
        strat_trades = [t for t in trades if t.strategy == strat_name]
        s = calc_stats(strat_trades)
        print(f'  {strat_name:<20} n={s["n"]:>5}  Sharpe={s["sharpe"]:.3f}  '
              f'PnL=${s["pnl"]:.0f}  WR={s["win_rate"]:.1f}%  MaxDD=${s["max_dd"]:.0f}')
        print(f'    Signals: {engine.total_signals}  Filtered(ADX): {engine.filtered_adx}')

        # Era breakdown
        eras = {}
        for era_name, (es, ee) in ERA_SEGMENTS.items():
            era_t = filter_period(strat_trades, es, ee)
            era_s = calc_stats(era_t)
            eras[era_name] = era_s
            print(f'    {era_name:<30} n={era_s["n"]:>4}  Sharpe={era_s["sharpe"]:.3f}')

        phase1[strat_name] = {'stats': s, 'eras': eras}

        if s['sharpe'] > 0.3 and s['n'] >= 30:
            viable_strategies.append(strat_name)

    save('phase1_screening', phase1)
    print(f'\n  Viable strategies (Sharpe>0.3, n>=30): {viable_strategies}')

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: K-Fold on viable strategies
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 2: 6-Fold K-Fold on Viable Strategies')
    print('=' * 80)

    phase2 = {}
    for strat_name in viable_strategies:
        sig_func = dict(ALL_STRATEGIES)[strat_name]
        engine = H4BacktestEngine(
            h4_df,
            signal_funcs=[(strat_name, sig_func)],
            sl_atr_mult=3.0, tp_atr_mult=6.0,
            trailing_activate_atr=0.5, trailing_distance_atr=0.15,
            max_hold=30, cooldown_bars=2, spread_cost=SPREAD,
        )
        trades = engine.run()
        strat_trades = [t for t in trades if t.strategy == strat_name]

        print(f'\n  {strat_name}:')
        kf = kfold_6(strat_trades)
        if kf.get('folds'):
            for f in kf['folds']:
                print(f'    Fold {f["fold"]}: n={f["n"]:>4}  Sharpe={f["sharpe"]:.3f}')
        print(f'    Verdict: {kf.get("verdict", "SKIP")}')
        phase2[strat_name] = kf

    save('phase2_kfold', phase2)

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: Parameter sweep on K-Fold passers
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 3: Parameter Sweep on K-Fold Passers')
    print('=' * 80)

    kf_passers = [s for s in viable_strategies if phase2.get(s, {}).get('verdict') == 'PASS']
    print(f'  K-Fold passers: {kf_passers}')

    phase3 = {}
    for strat_name in kf_passers:
        sig_func = dict(ALL_STRATEGIES)[strat_name]
        print(f'\n  --- {strat_name} parameter sweep ---')
        best_sharpe = -999
        best_params = None
        sweep_results = []

        for sl_m in [2.0, 3.0, 4.0, 5.0]:
            for tp_m in [4.0, 6.0, 8.0, 10.0]:
                for trail_a, trail_d in [(0.0, 0.0), (0.3, 0.08), (0.5, 0.15), (1.0, 0.3)]:
                    engine = H4BacktestEngine(
                        h4_df,
                        signal_funcs=[(strat_name, sig_func)],
                        sl_atr_mult=sl_m, tp_atr_mult=tp_m,
                        trailing_activate_atr=trail_a, trailing_distance_atr=trail_d,
                        max_hold=30, cooldown_bars=2, spread_cost=SPREAD,
                    )
                    trades = engine.run()
                    st = [t for t in trades if t.strategy == strat_name]
                    s = calc_stats(st)
                    label = f'SL{sl_m}_TP{tp_m}_T{trail_a}/{trail_d}'
                    sweep_results.append({'label': label, **s})
                    if s['sharpe'] > best_sharpe and s['n'] >= 20:
                        best_sharpe = s['sharpe']
                        best_params = {'sl': sl_m, 'tp': tp_m,
                                       'trail_act': trail_a, 'trail_dist': trail_d}

        # Print top 5
        sweep_results.sort(key=lambda x: x['sharpe'], reverse=True)
        for r in sweep_results[:5]:
            print(f'    {r["label"]:<30} n={r["n"]:>4}  Sharpe={r["sharpe"]:.3f}  PnL=${r["pnl"]:.0f}')

        print(f'  Best: {best_params}  Sharpe={best_sharpe:.3f}')
        phase3[strat_name] = {
            'best_params': best_params, 'best_sharpe': best_sharpe,
            'top5': sweep_results[:5],
        }

    save('phase3_param_sweep', phase3)

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Final Summary
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('FINAL SUMMARY')
    print('=' * 80)

    summary_table = []
    for strat_name, sig_func in ALL_STRATEGIES:
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

        summary_table.append({
            'strategy': strat_name,
            'n': s['n'], 'sharpe': s['sharpe'], 'pnl': s['pnl'],
            'wr': s['win_rate'], 'kfold': kf, 'decision': decision,
            'best_sweep_sharpe': ps,
        })
        print(f'  {strat_name:<20} n={s["n"]:>5}  Sharpe={s["sharpe"]:.3f}  '
              f'KF={kf:<5}  -> {decision}')

    summary = {
        'engine': 'H4BacktestEngine',
        'data': 'XAUUSD H4 2015-2026',
        'strategies': summary_table,
        'viable': viable_strategies,
        'kf_passers': kf_passers,
        'phase3_best': {k: v['best_params'] for k, v in phase3.items()},
    }
    save('R220_summary', summary)

    elapsed = time.time() - t_start
    print(f'\n  Total runtime: {elapsed:.0f}s')


if __name__ == '__main__':
    main()
