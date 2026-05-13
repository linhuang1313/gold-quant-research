#!/usr/bin/env python3
"""R217: MACD Divergence Strategy — BacktestEngine Validation
===============================================================
Engine: BacktestEngine (monkey-patch scan_all_signals)
Filters: ALL active via LIVE_PARITY_KWARGS

MACD crossover was tested (R80) and used as entry signal, but
MACD DIVERGENCE (price vs MACD disagreement) has NEVER been tested
as an independent strategy.

Divergence types:
  Bullish: price makes lower low, but MACD histogram makes higher low
  Bearish: price makes higher high, but MACD histogram makes lower high

Phase 1: Baseline divergence signal (default lookback)
Phase 2: Parameter sweep (lookback window, min divergence bars, SL/TP)
Phase 3: 6-Fold K-Fold on best config
Phase 4: Keltner correlation analysis
Phase 5: Sanity Gate + Decision
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

from backtest.runner import DataBundle, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine
import indicators as signals_mod
from indicators import get_orb_strategy

OUTPUT_DIR = Path("results/r217_macd_divergence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERA_SEGMENTS = {
    "Pre-COVID (2015-2019)":      ("2015-01-01", "2020-01-01"),
    "COVID+Recovery (2020-2021)": ("2020-01-01", "2022-01-01"),
    "Tightening (2022-2023)":     ("2022-01-01", "2024-01-01"),
    "Recent (2024-2026)":         ("2024-01-01", "2026-06-01"),
}

_div_lookback = 20
_div_min_bars = 5


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
    sharpe = float(pnls.mean() / max(pnls.std(ddof=1), 1e-9) * np.sqrt(252 * 4)) if n > 1 else 0
    return {
        'n': n,
        'pnl': round(float(pnls.sum()), 2),
        'sharpe': round(sharpe, 3),
        'win_rate': round(100 * (pnls > 0).sum() / n, 2),
        'avg_pnl': round(float(pnls.mean()), 2),
        'max_dd': round(float(dd.max()), 2),
    }


def filter_period(trades, start, end, strat=None):
    ts_s = pd.Timestamp(start, tz='UTC')
    ts_e = pd.Timestamp(end, tz='UTC')
    out = [t for t in trades if ts_s <= pd.Timestamp(t.entry_time) < ts_e]
    if strat:
        out = [t for t in out if t.strategy == strat]
    return out


def _find_swing_lows(prices, window=5):
    """Find indices of local minima."""
    lows = []
    for i in range(window, len(prices) - window):
        if prices[i] == min(prices[i - window:i + window + 1]):
            lows.append(i)
    return lows


def _find_swing_highs(prices, window=5):
    """Find indices of local maxima."""
    highs = []
    for i in range(window, len(prices) - window):
        if prices[i] == max(prices[i - window:i + window + 1]):
            highs.append(i)
    return highs


def check_macd_divergence_signal(df: pd.DataFrame) -> Optional[Dict]:
    """Detect MACD divergence on the last N bars of H1 data."""
    if len(df) < _div_lookback + 10:
        return None

    window = df.iloc[-_div_lookback:]
    close = window['Close'].values.astype(float)
    macd_hist = window['MACD_hist'].values.astype(float)
    atr_val = float(df.iloc[-1].get('ATR', 0))

    if atr_val <= 0 or np.any(np.isnan(macd_hist)):
        return None

    swing_window = max(2, _div_min_bars // 2)

    # Bullish divergence: price lower low + MACD hist higher low
    price_lows = _find_swing_lows(close, swing_window)
    hist_lows = _find_swing_lows(macd_hist, swing_window)

    if len(price_lows) >= 2 and len(hist_lows) >= 2:
        p_prev, p_curr = price_lows[-2], price_lows[-1]
        h_prev, h_curr = hist_lows[-2], hist_lows[-1]

        if (p_curr - p_prev >= _div_min_bars
            and close[p_curr] < close[p_prev]
            and macd_hist[h_curr] > macd_hist[h_prev]
            and p_curr >= len(close) - 3):
            entry = float(df.iloc[-1]['Close'])
            return {
                'strategy': 'macd_div',
                'signal': 'BUY',
                'reason': f"MACD看涨背离: 价格低低+MACD柱高低",
                'close': entry,
                'sl': round(atr_val * 2.5, 2),
                'tp': round(atr_val * 5.0, 2),
            }

    # Bearish divergence: price higher high + MACD hist lower high
    price_highs = _find_swing_highs(close, swing_window)
    hist_highs = _find_swing_highs(macd_hist, swing_window)

    if len(price_highs) >= 2 and len(hist_highs) >= 2:
        p_prev, p_curr = price_highs[-2], price_highs[-1]
        h_prev, h_curr = hist_highs[-2], hist_highs[-1]

        if (p_curr - p_prev >= _div_min_bars
            and close[p_curr] > close[p_prev]
            and macd_hist[h_curr] < macd_hist[h_prev]
            and p_curr >= len(close) - 3):
            entry = float(df.iloc[-1]['Close'])
            return {
                'strategy': 'macd_div',
                'signal': 'SELL',
                'reason': f"MACD看跌背离: 价格高高+MACD柱低高",
                'close': entry,
                'sl': round(atr_val * 2.5, 2),
                'tp': round(atr_val * 5.0, 2),
            }

    return None


_original_scan = signals_mod.scan_all_signals


def patched_scan_all_signals(df, timeframe='H1', h1_adx=None):
    signals = _original_scan(df, timeframe, h1_adx)
    if timeframe == 'H1':
        sig = check_macd_divergence_signal(df)
        if sig:
            signals.append(sig)
    return signals


def run_engine(data, **extra_kwargs):
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    get_orb_strategy().reset_daily()
    kwargs = {**LIVE_PARITY_KWARGS, **extra_kwargs}
    engine = BacktestEngine(data.m15_df, data.h1_df, **kwargs)
    return engine.run(), engine


def main():
    global _div_lookback, _div_min_bars
    t_start = time.time()
    print('=' * 80)
    print('R217: MACD Divergence Strategy — BacktestEngine Validation')
    print('=' * 80)

    data = DataBundle.load_default()
    signals_mod.scan_all_signals = patched_scan_all_signals

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Baseline
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 1: Baseline MACD Divergence')
    print('=' * 80)

    _div_lookback = 20
    _div_min_bars = 5
    trades_all, eng = run_engine(data, sl_atr_mult=0, tp_atr_mult=0)
    div_trades = [t for t in trades_all if t.strategy == 'macd_div']
    kc_trades = [t for t in trades_all if t.strategy == 'keltner']

    div_stats = calc_stats(div_trades)
    kc_stats = calc_stats(kc_trades)

    print(f'  Keltner:        n={kc_stats["n"]:>5}  Sharpe={kc_stats["sharpe"]:.3f}  PnL=${kc_stats["pnl"]:.0f}')
    print(f'  MACD Divergence: n={div_stats["n"]:>5}  Sharpe={div_stats["sharpe"]:.3f}  '
          f'PnL=${div_stats["pnl"]:.0f}  WR={div_stats["win_rate"]:.1f}%')

    div_eras = {}
    for era_name, (es, ee) in ERA_SEGMENTS.items():
        era_t = filter_period(div_trades, es, ee)
        s = calc_stats(era_t)
        div_eras[era_name] = s
        print(f'    {era_name:<30} n={s["n"]:>4}  Sharpe={s["sharpe"]:.3f}  PnL=${s["pnl"]:.0f}')

    phase1 = {'divergence': div_stats, 'keltner': kc_stats, 'eras': div_eras}
    save('phase1_baseline', phase1)

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Parameter Sweep
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 2: Parameter Sweep')
    print('=' * 80)

    phase2 = []
    best_sharpe = -999
    best_config = None

    lookbacks = [15, 20, 30, 40]
    min_bars_options = [3, 5, 8]

    for lb in lookbacks:
        for mb in min_bars_options:
            _div_lookback = lb
            _div_min_bars = mb
            trades, _ = run_engine(data, sl_atr_mult=0, tp_atr_mult=0)
            div = [t for t in trades if t.strategy == 'macd_div']
            s = calc_stats(div)
            label = f'LB{lb}_MB{mb}'
            print(f'    {label:<20} n={s["n"]:>4}  Sharpe={s["sharpe"]:.3f}  '
                  f'PnL=${s["pnl"]:.0f}  WR={s["win_rate"]:.1f}%')
            phase2.append({'label': label, **s})
            if s['sharpe'] > best_sharpe and s['n'] >= 20:
                best_sharpe = s['sharpe']
                best_config = {'lookback': lb, 'min_bars': mb}

    # SL sweep with best divergence params
    if best_config:
        _div_lookback = best_config['lookback']
        _div_min_bars = best_config['min_bars']
        print(f'\n  SL sweep (LB={best_config["lookback"]}, MB={best_config["min_bars"]}):')
        for sl_m in [1.5, 2.5, 3.5, 5.0]:
            trades, _ = run_engine(data, sl_atr_mult=sl_m, tp_atr_mult=0)
            div = [t for t in trades if t.strategy == 'macd_div']
            s = calc_stats(div)
            label = f'SL_{sl_m}'
            print(f'    {label:<20} n={s["n"]:>4}  Sharpe={s["sharpe"]:.3f}  '
                  f'PnL=${s["pnl"]:.0f}')
            phase2.append({'label': label, **s})
            if s['sharpe'] > best_sharpe and s['n'] >= 20:
                best_sharpe = s['sharpe']
                best_config['sl'] = sl_m

    print(f'\n  Best config: {best_config}  Sharpe={best_sharpe:.3f}')
    save('phase2_sweep', {'results': phase2, 'best': best_config, 'best_sharpe': best_sharpe})

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: 6-Fold K-Fold
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 3: 6-Fold K-Fold Validation')
    print('=' * 80)

    if best_config is None or best_sharpe <= 0:
        print('  No viable config — SKIP')
        phase3 = {'skip': True, 'reason': 'no viable config'}
    else:
        _div_lookback = best_config['lookback']
        _div_min_bars = best_config['min_bars']
        sl_kw = {'sl_atr_mult': best_config.get('sl', 0), 'tp_atr_mult': 0}
        trades, _ = run_engine(data, **sl_kw)
        div_only = [t for t in trades if t.strategy == 'macd_div']

        if len(div_only) < 30:
            print(f'  Only {len(div_only)} trades — insufficient')
            phase3 = {'skip': True, 'reason': f'n={len(div_only)} < 30'}
        else:
            pnls = np.array([t.pnl for t in div_only])
            fold_results = []
            kf_pass = 0
            fold_size = len(pnls) // 6

            for fold in range(6):
                start = fold * fold_size
                end = start + fold_size if fold < 5 else len(pnls)
                fp = pnls[start:end]
                if len(fp) < 5:
                    continue
                s = float(fp.mean() / max(fp.std(ddof=1), 1e-9) * np.sqrt(252 * 4))
                fold_results.append({'fold': fold + 1, 'n': len(fp), 'sharpe': round(s, 3)})
                print(f'    Fold {fold+1}: n={len(fp):>4}  Sharpe={s:.3f}')
                if s > 0:
                    kf_pass += 1

            kf_rate = kf_pass / max(len(fold_results), 1)
            phase3 = {
                'config': best_config,
                'folds': fold_results,
                'pass_count': kf_pass,
                'total_folds': len(fold_results),
                'pass_rate': round(kf_rate, 3),
                'verdict': 'PASS' if kf_rate >= 0.67 else 'FAIL',
            }
            print(f'    K-Fold: {kf_pass}/{len(fold_results)} positive -> {phase3["verdict"]}')

    save('phase3_kfold', phase3)

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Keltner Correlation
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 4: Keltner Correlation Analysis')
    print('=' * 80)

    if best_config:
        _div_lookback = best_config['lookback']
        _div_min_bars = best_config['min_bars']
        trades, _ = run_engine(data, sl_atr_mult=0, tp_atr_mult=0)
        div_only = [t for t in trades if t.strategy == 'macd_div']
        kc_only = [t for t in trades if t.strategy == 'keltner']

        def daily_pnl(tl):
            d = {}
            for t in tl:
                day = pd.Timestamp(t.entry_time).strftime('%Y-%m-%d')
                d[day] = d.get(day, 0) + t.pnl
            return pd.Series(d).sort_index()

        div_daily = daily_pnl(div_only)
        kc_daily = daily_pnl(kc_only)
        common = div_daily.index.intersection(kc_daily.index)

        corr = float(div_daily.loc[common].corr(kc_daily.loc[common])) if len(common) >= 10 else 0.0
        diversification = 'GOOD' if corr < 0.3 else ('MODERATE' if corr < 0.5 else 'HIGH')

        print(f'  Daily PnL correlation: {corr:.3f}  -> {diversification}')
        phase4 = {'correlation': round(corr, 3), 'diversification': diversification}
    else:
        phase4 = {'skip': True}
        print('  SKIP')

    save('phase4_correlation', phase4)

    # ═══════════════════════════════════════════════════════════════
    # Phase 5: Final Verdict
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 5: Final Verdict')
    print('=' * 80)

    decision = 'REJECT'
    reasons = []

    if div_stats['sharpe'] > 1.0:
        reasons.append(f'Full Sharpe {div_stats["sharpe"]:.2f} > 1.0')
    else:
        reasons.append(f'Full Sharpe {div_stats["sharpe"]:.2f} <= 1.0')

    kf_verdict = phase3.get('verdict', 'SKIP')
    reasons.append(f'K-Fold: {kf_verdict}')

    if phase4.get('diversification'):
        reasons.append(f'KC correlation: {phase4.get("correlation", "?")} ({phase4.get("diversification")})')

    if div_stats['sharpe'] > 1.0 and kf_verdict == 'PASS':
        decision = 'CONSIDER'
    elif div_stats['sharpe'] > 0.5 and kf_verdict == 'PASS':
        decision = 'WEAK-CONSIDER'

    print(f'  Decision: {decision}')
    for r in reasons:
        print(f'    - {r}')

    summary = {
        'engine': 'BacktestEngine (LIVE_PARITY_KWARGS)',
        'divergence_baseline': div_stats,
        'best_config': best_config,
        'kfold_verdict': kf_verdict,
        'kc_correlation': phase4.get('correlation'),
        'decision': decision,
        'reasons': reasons,
    }
    save('R217_summary', summary)

    signals_mod.scan_all_signals = _original_scan

    elapsed = time.time() - t_start
    print(f'\n  Total runtime: {elapsed:.0f}s')


if __name__ == '__main__':
    main()
