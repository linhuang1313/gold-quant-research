#!/usr/bin/env python3
"""R209v2: Non-Keltner Strategy Audit — Using BacktestEngine
=============================================================
Replaces the INVALID R209 which used a standalone backtest loop.

This version:
  - Uses the REAL BacktestEngine with all filters (Choppy Gate, ATR Pctl, etc.)
  - Monkey-patches indicators.scan_all_signals to include PSAR, Sess BO,
    DualThrust, Chandelier from the live system's signal logic
  - Compares backtest vs live trade counts for sanity check

Phase 1: All strategies together (like live — slot contention, filters active)
Phase 2: Per-strategy isolation (disable Keltner, run each alone)
Phase 3: Sanity gate — cross-reference with R211 live data

Engine: BacktestEngine (backtest.engine)
Filters: Choppy Gate, ATR Pctl, Rule B, regime config — ALL ACTIVE
"""
from __future__ import annotations
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.runner import DataBundle, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine
import indicators as signals_mod
import research_config as config

OUTPUT_DIR = Path("results/r209v2_non_keltner_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LIVE_PERIOD = ("2026-03-25", "2026-05-13")

LIVE_STRAT_CONFIGS = {
    'psar': {
        'enabled': True, 'psar_af_step': 0.01, 'psar_af_max': 0.05,
        'min_atr': 0.1, 'sl_atr': 4.0, 'tp_atr': 6.0,
        'trail_act_atr': 0.08, 'trail_dist_atr': 0.015,
    },
    'sess_bo': {
        'enabled': True, 'broker_gmt_offset': 0, 'session_hour_gmt': 12,
        'lookback_bars': 4, 'sl_atr': 4.5, 'tp_atr': 4.0,
    },
    'dual_thrust': {
        'enabled': True, 'n_bars': 6, 'k_up': 0.5, 'k_down': 0.5,
        'sl_atr': 4.5, 'tp_atr': 8.0,
        'trail_act_atr': 0.14, 'trail_dist_atr': 0.025,
    },
    'chandelier': {
        'enabled': True, 'chand_period': 22, 'chand_mult': 3.0,
        'rsi_filter': True, 'sl_atr': 4.5, 'tp_atr': 8.0,
        'trail_act_atr': 0.14, 'trail_dist_atr': 0.025,
    },
}


# ═══════════════════════════════════════════════════════════════
# Signal functions ported from gold-quant-trading/strategies/signals.py
# ═══════════════════════════════════════════════════════════════

def _compute_psar_series(highs, lows, af_step=0.01, af_max=0.05):
    n = len(highs)
    psar = np.zeros(n)
    direction = np.ones(n)
    bull = True
    af = af_step
    ep = highs[0]
    psar[0] = lows[0]
    for i in range(1, n):
        psar[i] = psar[i-1] + af * (ep - psar[i-1])
        if bull:
            psar[i] = min(psar[i], lows[i-1], lows[max(0, i-2)])
            if lows[i] < psar[i]:
                bull = False
                psar[i] = ep
                af = af_step
                ep = lows[i]
            else:
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + af_step, af_max)
        else:
            psar[i] = max(psar[i], highs[i-1], highs[max(0, i-2)])
            if highs[i] > psar[i]:
                bull = True
                psar[i] = ep
                af = af_step
                ep = highs[i]
            else:
                if lows[i] < ep:
                    ep = lows[i]
                    af = min(af + af_step, af_max)
        direction[i] = 1 if bull else -1
    return psar, direction


def check_psar_signal(df: pd.DataFrame) -> Optional[Dict]:
    cfg = LIVE_STRAT_CONFIGS['psar']
    if not cfg.get('enabled'):
        return None
    if df is None or len(df) < 30:
        return None
    lookback = min(200, len(df))
    sub = df.iloc[-lookback:]
    highs = sub['High'].values.astype(float)
    lows = sub['Low'].values.astype(float)
    _, direction = _compute_psar_series(highs, lows, cfg['psar_af_step'], cfg['psar_af_max'])
    cur_dir = int(direction[-1])
    prev_dir = int(direction[-2])
    if prev_dir == -1 and cur_dir == 1:
        signal = 'BUY'
    elif prev_dir == 1 and cur_dir == -1:
        signal = 'SELL'
    else:
        return None
    atr = float(df.iloc[-1].get('ATR', 0))
    if pd.isna(atr) or atr < cfg['min_atr']:
        return None
    sl = round(atr * cfg['sl_atr'], 2)
    tp = round(atr * cfg['tp_atr'], 2)
    return {
        'strategy': 'psar', 'signal': signal,
        'close': float(df.iloc[-1]['Close']),
        'sl': sl, 'tp': tp,
        'reason': f"PSAR flip {signal}",
    }


def check_sess_bo_signal(df: pd.DataFrame) -> Optional[Dict]:
    cfg = LIVE_STRAT_CONFIGS['sess_bo']
    if not cfg.get('enabled'):
        return None
    if df is None or len(df) < 10:
        return None
    bar_time = df.index[-1]
    hour = pd.Timestamp(bar_time).hour if hasattr(bar_time, 'hour') else 0
    broker_offset = cfg.get('broker_gmt_offset', 0)
    gmt_hour = (hour - broker_offset) % 24
    if gmt_hour != cfg['session_hour_gmt']:
        return None
    if len(df) < 2:
        return None
    prev_hour = pd.Timestamp(df.index[-2]).hour if hasattr(df.index[-2], 'hour') else 0
    prev_gmt = (prev_hour - broker_offset) % 24
    if prev_gmt == cfg['session_hour_gmt']:
        return None
    lookback = cfg['lookback_bars']
    if len(df) < lookback + 1:
        return None
    range_bars = df.iloc[-(lookback + 1):-1]
    range_high = float(range_bars['High'].max())
    range_low = float(range_bars['Low'].min())
    cur_close = float(df.iloc[-1]['Close'])
    if cur_close > range_high:
        signal = 'BUY'
    elif cur_close < range_low:
        signal = 'SELL'
    else:
        return None
    atr = float(df.iloc[-1].get('ATR', 0))
    if pd.isna(atr) or atr < 0.1:
        return None
    sl = round(atr * cfg['sl_atr'], 2)
    tp = round(atr * cfg['tp_atr'], 2)
    return {
        'strategy': 'sess_bo', 'signal': signal,
        'close': cur_close, 'sl': sl, 'tp': tp,
        'reason': f"SESS_BO {signal}: range [{range_low:.2f}-{range_high:.2f}]",
    }


def check_dual_thrust_signal(df: pd.DataFrame) -> Optional[Dict]:
    cfg = LIVE_STRAT_CONFIGS['dual_thrust']
    if not cfg.get('enabled'):
        return None
    if df is None or len(df) < 30:
        return None
    n_bars = cfg['n_bars']
    k_up = cfg['k_up']
    k_down = cfg['k_down']
    if len(df) < n_bars + 3:
        return None
    hh = df['High'].rolling(n_bars).max()
    lc = df['Close'].rolling(n_bars).min()
    hc = df['Close'].rolling(n_bars).max()
    ll = df['Low'].rolling(n_bars).min()
    dt_range = pd.concat([hh - lc, hc - ll], axis=1).max(axis=1)
    _date_series = pd.Series(df.index.date, index=df.index)
    daily_open = df.groupby(_date_series)['Open'].transform('first')
    cur = df.iloc[-2]
    prev = df.iloc[-3]
    close = float(cur['Close'])
    close_prev = float(prev['Close'])
    r_cur = float(dt_range.iloc[-2]) if not pd.isna(dt_range.iloc[-2]) else 0
    r_prev = float(dt_range.iloc[-3]) if not pd.isna(dt_range.iloc[-3]) else 0
    do_cur = float(daily_open.iloc[-2]) if not pd.isna(daily_open.iloc[-2]) else 0
    do_prev = float(daily_open.iloc[-3]) if not pd.isna(daily_open.iloc[-3]) else 0
    if r_cur <= 0 or do_cur <= 0:
        return None
    up_now = close > do_cur + k_up * r_cur
    up_prev = close_prev > do_prev + k_up * r_prev if r_prev > 0 and do_prev > 0 else False
    dn_now = close < do_cur - k_down * r_cur
    dn_prev = close_prev < do_prev - k_down * r_prev if r_prev > 0 and do_prev > 0 else False
    signal = None
    if up_now and not up_prev:
        signal = 'BUY'
    elif dn_now and not dn_prev:
        signal = 'SELL'
    if signal is None:
        return None
    atr = float(df.iloc[-1].get('ATR', 0))
    if pd.isna(atr) or atr < 0.1:
        return None
    sl = round(atr * cfg['sl_atr'], 2)
    tp = round(atr * cfg['tp_atr'], 2)
    return {
        'strategy': 'dual_thrust', 'signal': signal,
        'close': float(df.iloc[-1]['Close']),
        'sl': sl, 'tp': tp,
        'reason': f"DualThrust {signal}",
    }


def check_chandelier_signal(df: pd.DataFrame) -> Optional[Dict]:
    cfg = LIVE_STRAT_CONFIGS['chandelier']
    if not cfg.get('enabled'):
        return None
    if df is None or len(df) < 105:
        return None
    period = cfg['chand_period']
    mult = cfg['chand_mult']
    atr_raw = (df['High'] - df['Low']).rolling(14).mean()
    hh = df['High'].rolling(period).max()
    ll = df['Low'].rolling(period).min()
    chand_long = hh - mult * atr_raw
    chand_short = ll + mult * atr_raw
    cur_close = float(df.iloc[-2]['Close'])
    prev_close = float(df.iloc[-3]['Close'])
    cur_cl = float(chand_long.iloc[-2]) if not pd.isna(chand_long.iloc[-2]) else 0
    prev_cl = float(chand_long.iloc[-3]) if not pd.isna(chand_long.iloc[-3]) else 0
    cur_cs = float(chand_short.iloc[-2]) if not pd.isna(chand_short.iloc[-2]) else 0
    prev_cs = float(chand_short.iloc[-3]) if not pd.isna(chand_short.iloc[-3]) else 0
    if cur_cl <= 0 or cur_cs <= 0:
        return None
    flip_bull = (cur_close > cur_cl) and not (prev_close > prev_cl)
    flip_bear = (cur_close < cur_cs) and not (prev_close < prev_cs)
    signal = None
    if flip_bull:
        signal = 'BUY'
    elif flip_bear:
        signal = 'SELL'
    if signal is None:
        return None
    if cfg.get('rsi_filter', True):
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float('nan'))
        rsi = 100 - (100 / (1 + rs))
        cur_rsi = rsi.iloc[-2]
        if not pd.isna(cur_rsi):
            if signal == 'BUY' and cur_rsi > 70:
                return None
            if signal == 'SELL' and cur_rsi < 30:
                return None
    atr = float(df.iloc[-1].get('ATR', 0))
    if pd.isna(atr) or atr < 0.1:
        return None
    sl = round(atr * cfg['sl_atr'], 2)
    tp = round(atr * cfg['tp_atr'], 2)
    return {
        'strategy': 'chandelier', 'signal': signal,
        'close': float(df.iloc[-1]['Close']),
        'sl': sl, 'tp': tp,
        'reason': f"Chandelier {signal}",
    }


# ═══════════════════════════════════════════════════════════════
# Monkey-patch scan_all_signals to include non-Keltner strategies
# ═══════════════════════════════════════════════════════════════

_original_scan = signals_mod.scan_all_signals

def patched_scan_all_signals(df, timeframe='H1', h1_adx=None):
    signals = _original_scan(df, timeframe, h1_adx)
    if timeframe == 'H1':
        for check_fn in [check_psar_signal, check_sess_bo_signal,
                         check_dual_thrust_signal, check_chandelier_signal]:
            sig = check_fn(df)
            if sig:
                signals.append(sig)
    return signals


def save(name, data):
    p = OUTPUT_DIR / f'{name}.json'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    print(f'  -> saved {p}')


def calc_stats(trades, strategy_filter=None):
    if strategy_filter:
        trades = [t for t in trades if t.strategy == strategy_filter]
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


def calc_stats_period(trades, start, end, strategy_filter=None):
    ts_start = pd.Timestamp(start, tz='UTC')
    ts_end = pd.Timestamp(end, tz='UTC')
    filtered = [t for t in trades
                if ts_start <= pd.Timestamp(t.entry_time) < ts_end]
    return calc_stats(filtered, strategy_filter)


def main():
    t_start = time.time()
    print('=' * 80)
    print('R209v2: Non-Keltner Strategy Audit (BacktestEngine)')
    print('=' * 80)
    print('  Engine: BacktestEngine with ALL filters')
    print('  Filters: Choppy Gate, ATR Pctl, regime config, ADX — ALL ACTIVE')

    # Patch scan_all_signals
    signals_mod.scan_all_signals = patched_scan_all_signals
    print('  Patched scan_all_signals with PSAR/SESS_BO/DualThrust/Chandelier')

    data = DataBundle.load_default()

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: All strategies together (like live)
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 1: All Strategies Together (Keltner + non-Keltner)')
    print('  Using LIVE_PARITY_KWARGS — sl/tp override applies to ALL')
    print('=' * 80)

    # NOTE: LIVE_PARITY_KWARGS has sl_atr_mult=3.5, tp_atr_mult=8.0
    # which OVERRIDES all signal SL/TP. This matches Keltner live config
    # but NOT the non-Keltner live configs. We run this first for the
    # "portfolio" view, then run per-strategy with proper SL/TP.

    kwargs_all = {**LIVE_PARITY_KWARGS}
    # Allow more slots so non-Keltner strategies can actually enter
    kwargs_all['max_positions'] = 4

    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    from indicators import get_orb_strategy
    get_orb_strategy().reset_daily()

    engine = BacktestEngine(data.m15_df, data.h1_df, **kwargs_all)
    all_trades = engine.run()

    strategies_seen = set(t.strategy for t in all_trades)
    print(f'\n  Total trades: {len(all_trades)}')
    print(f'  Strategies seen: {sorted(strategies_seen)}')

    phase1 = {}
    print(f'\n  {"Strategy":<15} {"N":>6} {"PnL":>10} {"Sharpe":>8} {"WR%":>7} {"AvgPnL":>8} {"MaxDD":>8}')
    for strat in sorted(strategies_seen):
        s = calc_stats(all_trades, strat)
        print(f'  {strat:<15} {s["n"]:>6} {s["pnl"]:>10.2f} {s["sharpe"]:>8.3f} '
              f'{s["win_rate"]:>6.1f}% {s["avg_pnl"]:>8.2f} {s["max_dd"]:>8.2f}')
        phase1[strat] = s

    # Live period only
    print(f'\n  === Live Period ({LIVE_PERIOD[0]} -> {LIVE_PERIOD[1]}) ===')
    print(f'  {"Strategy":<15} {"N":>6} {"PnL":>10} {"Sharpe":>8} {"WR%":>7}')
    phase1_live = {}
    for strat in sorted(strategies_seen):
        s = calc_stats_period(all_trades, LIVE_PERIOD[0], LIVE_PERIOD[1], strat)
        if s['n'] > 0:
            print(f'  {strat:<15} {s["n"]:>6} {s["pnl"]:>10.2f} {s["sharpe"]:>8.3f} {s["win_rate"]:>6.1f}%')
            phase1_live[strat] = s

    phase1['_live_period'] = phase1_live
    save('phase1_all_together', phase1)

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Per-strategy isolation (proper SL/TP per strategy)
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 2: Per-Strategy Isolation')
    print('  Disabling Keltner SL/TP override, using signal-provided SL/TP')
    print('=' * 80)

    phase2 = {}
    target_strategies = ['psar', 'sess_bo', 'dual_thrust', 'chandelier']

    for target in target_strategies:
        print(f'\n  === {target.upper()} (isolated) ===')

        # Disable all except the target
        for s in LIVE_STRAT_CONFIGS:
            LIVE_STRAT_CONFIGS[s]['enabled'] = (s == target)

        # No SL/TP override — let signal provide its own
        kwargs_iso = {**LIVE_PARITY_KWARGS}
        kwargs_iso['sl_atr_mult'] = 0
        kwargs_iso['tp_atr_mult'] = 0
        kwargs_iso['max_positions'] = 1

        signals_mod._friday_close_price = None
        signals_mod._gap_traded_today = False
        get_orb_strategy().reset_daily()

        engine = BacktestEngine(data.m15_df, data.h1_df, **kwargs_iso)
        trades = engine.run()

        # Only count target strategy trades
        target_trades = [t for t in trades if t.strategy == target]
        other_trades = [t for t in trades if t.strategy != target]

        full_stats = calc_stats(target_trades)
        live_stats = calc_stats_period(target_trades, LIVE_PERIOD[0], LIVE_PERIOD[1])

        # Per-era
        eras = {
            'Pre-COVID (2015-2019)': ('2015-01-01', '2020-01-01'),
            'COVID+Recovery (2020-2021)': ('2020-01-01', '2022-01-01'),
            'Tightening (2022-2023)': ('2022-01-01', '2024-01-01'),
            'Recent (2024-2026)': ('2024-01-01', '2026-06-01'),
        }
        era_stats = {}
        for era_name, (era_start, era_end) in eras.items():
            era_stats[era_name] = calc_stats_period(target_trades, era_start, era_end)

        print(f'    Full:  n={full_stats["n"]:>5}  PnL=${full_stats["pnl"]:>10.2f}  '
              f'Sharpe={full_stats["sharpe"]:.3f}  WR={full_stats["win_rate"]:.1f}%')
        print(f'    Live:  n={live_stats["n"]:>5}  PnL=${live_stats["pnl"]:>10.2f}  '
              f'Sharpe={live_stats["sharpe"]:.3f}  WR={live_stats["win_rate"]:.1f}%')
        for era_name, es in era_stats.items():
            print(f'    {era_name:<30} n={es["n"]:>5}  PnL=${es["pnl"]:>8.0f}  Sharpe={es["sharpe"]:.3f}')

        if other_trades:
            print(f'    (also generated {len(other_trades)} Keltner/M15 trades — these use the engine\'s original signals)')

        phase2[target] = {
            'full': full_stats,
            'live_period': live_stats,
            'eras': era_stats,
            'other_trades_n': len(other_trades),
        }

    # Re-enable all for further use
    for s in LIVE_STRAT_CONFIGS:
        LIVE_STRAT_CONFIGS[s]['enabled'] = True

    save('phase2_per_strategy', phase2)

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: Sanity Gate — cross-reference with R211 live data
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 3: Sanity Gate')
    print('=' * 80)

    r211_live = {
        'keltner': {'n': 114, 'pnl': 386.84, 'wr': 79.8},
        'dual_thrust': {'n': 12, 'pnl': 34.52, 'wr': 58.3},
        'psar': {'n': 3, 'pnl': -85.14, 'wr': 0.0},
        'chandelier': {'n': 4, 'pnl': -36.26, 'wr': 50.0},
        'sess_bo': {'n': 1, 'pnl': 34.84, 'wr': 100.0},
        'm15_rsi': {'n': 19, 'pnl': 49.97, 'wr': 63.2},
    }

    sanity = {}
    print(f'\n  {"Strategy":<15} {"Live_N":>7} {"BT_N":>7} {"Ratio":>7} '
          f'{"Live_PnL":>10} {"BT_PnL":>10} {"Flag":>8}')

    for strat in sorted(r211_live.keys()):
        live = r211_live[strat]
        bt = phase1_live.get(strat, {'n': 0, 'pnl': 0})
        ratio = bt['n'] / live['n'] if live['n'] > 0 else float('inf')

        flag = 'OK'
        if ratio > 5 or ratio < 0.2:
            flag = 'MISMATCH'
        elif bt['n'] == 0 and live['n'] > 0:
            flag = 'MISSING'

        print(f'  {strat:<15} {live["n"]:>7} {bt["n"]:>7} {ratio:>7.1f}x '
              f'{live["pnl"]:>10.2f} {bt.get("pnl", 0):>10.2f} {flag:>8}')

        sanity[strat] = {
            'live_n': live['n'], 'bt_n': bt['n'], 'ratio': round(ratio, 2),
            'live_pnl': live['pnl'], 'bt_pnl': bt.get('pnl', 0),
            'flag': flag,
        }

    save('phase3_sanity_gate', sanity)

    # ═══════════════════════════════════════════════════════════════
    # Sanity check: are the numbers plausible?
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('SANITY CHECK (per backtest-engine-mandatory rule)')
    print('=' * 80)

    for strat, s in phase1.items():
        if strat.startswith('_'):
            continue
        alerts = []
        if s.get('sharpe', 0) > 5.0:
            alerts.append(f'Sharpe {s["sharpe"]:.1f} > 5.0 — SUSPICIOUS')
        if s.get('win_rate', 0) > 85:
            alerts.append(f'WR {s["win_rate"]:.1f}% > 85% — SUSPICIOUS')
        if s.get('n', 0) > 5000 * 11:
            alerts.append(f'N={s["n"]} > 55000 — SUSPICIOUS (too many trades)')
        if alerts:
            print(f'  {strat}: ' + '; '.join(alerts))
        else:
            print(f'  {strat}: OK (Sharpe={s.get("sharpe", 0):.2f}, WR={s.get("win_rate", 0):.1f}%, N={s.get("n", 0)})')

    # ═══════════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('FINAL SUMMARY')
    print('=' * 80)

    summary = {
        'engine': 'BacktestEngine (full filter stack)',
        'filters_active': ['choppy_gate', 'atr_percentile', 'regime_config', 'adx_threshold', 'cooldown'],
        'phase1_all_together': phase1,
        'phase2_isolated': phase2,
        'phase3_sanity': sanity,
    }

    for strat in target_strategies:
        iso = phase2.get(strat, {})
        full = iso.get('full', {})
        live = iso.get('live_period', {})
        gate = sanity.get(strat, {})
        print(f'  {strat:<15} Full: n={full.get("n", 0):>5} Sharpe={full.get("sharpe", 0):.3f}  '
              f'Live: n={live.get("n", 0):>3} PnL=${live.get("pnl", 0):.0f}  '
              f'Sanity: {gate.get("flag", "N/A")}')

    save('R209v2_summary', summary)

    elapsed = time.time() - t_start
    print(f'\n  Total runtime: {elapsed:.0f}s')

    # Restore original scan
    signals_mod.scan_all_signals = _original_scan


if __name__ == '__main__':
    main()
