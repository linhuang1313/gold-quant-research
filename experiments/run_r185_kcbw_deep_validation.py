#!/usr/bin/env python3
"""
R185 — KCBW Deep Validation
=============================
Follow-up to R184's KCBW discovery (Sharpe +16.6%, WR 71.8%→80.4%).
Before deploying KCBW, run comprehensive validation:

Phase 1: KCBW Lookback Sensitivity (is 5 a plateau or a spike?)
Phase 2: KCBW Definition Consistency (R184 standalone vs Engine vs MT4-equiv)
Phase 3: Engine Cross-Validation (full M15 engine with kc_bw_filter_bars)
Phase 4: Walk-Forward OOS (5 anchored windows)
Phase 5: Parameter Perturbation MC (joint perturbation of KCBW + exit params)
Phase 6: Cost Sensitivity (spread 0.30 → 1.00)
Phase 7: Yearly Stability (per-year Sharpe delta)
Phase 8: PBO — Probability of Backtest Overfitting (combinatorial split)

Target: ~5-10 min on local machine.
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r185_kcbw_deep_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
LOT = 0.02
CAP_ATR_MULT = 4.0

SL_ATR = 3.5
TP_ATR = 8.0
TRAIL_ACT = 0.14
TRAIL_DIST = 0.025
MAX_HOLD = 2

ERA_SEGMENTS = {
    'full':      None,
    'hike':      [("2015-12-01", "2019-01-01"), ("2022-03-01", "2023-08-01")],
    'cut':       [("2019-07-01", "2022-03-01"), ("2024-09-01", "2026-06-01")],
    'recent_3y': [("2023-06-01", "2026-06-01")],
}

FOLDS = [
    ("F1_2015_2017", "2015-01-01", "2017-01-01"),
    ("F2_2017_2019", "2017-01-01", "2019-01-01"),
    ("F3_2019_2021", "2019-01-01", "2021-01-01"),
    ("F4_2021_2023", "2021-01-01", "2023-01-01"),
    ("F5_2023_2025", "2023-01-01", "2025-01-01"),
    ("F6_2025_2026", "2025-01-01", "2026-06-01"),
]

WF_WINDOWS = [
    ("WF1", "2015-01-01", "2019-01-01", "2019-01-01", "2021-01-01"),
    ("WF2", "2016-01-01", "2020-01-01", "2020-01-01", "2022-01-01"),
    ("WF3", "2017-01-01", "2021-01-01", "2021-01-01", "2023-01-01"),
    ("WF4", "2018-01-01", "2022-01-01", "2022-01-01", "2024-01-01"),
    ("WF5", "2019-01-01", "2023-01-01", "2023-01-01", "2025-01-01"),
]

MC_ITERATIONS = 1000
MC_SEED = 42

import glob as _glob

t0 = time.time()


# ═══════════════════════════════════════════════════════════════
# Shared helpers (from R184)
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


def compute_adx(df, period=14):
    high = df['High']; low = df['Low']; close = df['Close']
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.DataFrame({'hl': high - low, 'hc': (high - close.shift(1)).abs(),
                        'lc': (low - close.shift(1)).abs()}).max(axis=1)
    atr_s = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_s)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_s)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.rolling(period).mean()


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar'],
            'atr': pos['atr']}


def _run_exit(pos, i, h, lo_v, c, spread, lot, pv, times,
              sl_atr, tp_atr, trail_act_atr, trail_dist_atr,
              max_hold, cap_atr_mult=0):
    held = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_h = (h - pos['entry'] - spread) * lot * pv
        pnl_l = (lo_v - pos['entry'] - spread) * lot * pv
        pnl_c = (c - pos['entry'] - spread) * lot * pv
    else:
        pnl_h = (pos['entry'] - lo_v - spread) * lot * pv
        pnl_l = (pos['entry'] - h - spread) * lot * pv
        pnl_c = (pos['entry'] - c - spread) * lot * pv
    tp_val = tp_atr * pos['atr'] * lot * pv
    sl_val = sl_atr * pos['atr'] * lot * pv
    if pnl_h >= tp_val:
        return _mk(pos, c, times[i], "TP", i, tp_val)
    if pnl_l <= -sl_val:
        return _mk(pos, c, times[i], "SL", i, -sl_val)
    if cap_atr_mult > 0 and cap_atr_mult < 900:
        cap_dollar = lot * pv * cap_atr_mult * pos['atr']
        if pnl_c < -cap_dollar:
            return _mk(pos, c, times[i], "Cap", i, -cap_dollar)
    if pos['dir'] == 'BUY' and h - pos['entry'] >= trail_act_atr * pos['atr']:
        ts_p = h - trail_dist_atr * pos['atr']
        if lo_v <= ts_p:
            return _mk(pos, c, times[i], "Trail", i, (ts_p - pos['entry'] - spread) * lot * pv)
    elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= trail_act_atr * pos['atr']:
        ts_p = lo_v + trail_dist_atr * pos['atr']
        if h >= ts_p:
            return _mk(pos, c, times[i], "Trail", i, (pos['entry'] - ts_p - spread) * lot * pv)
    if held >= max_hold:
        return _mk(pos, c, times[i], "Timeout", i, pnl_c)
    return None


def bt_keltner_filtered(h1_df, adx_th=14, ema_type='ema100', kc_mult=1.2,
                         session_hours=None, kcbw_filter=False, kcbw_lookback=5,
                         kcbw_mode='rolling_mean',
                         spread_override=None,
                         sl_atr=None, tp_atr=None,
                         trail_act=None, trail_dist=None, max_hold_override=None):
    """Keltner backtest with configurable KCBW definition.

    kcbw_mode:
      'rolling_mean' — R184 style: KCBW > KCBW_rolling_mean(lookback)
      'lag_compare'  — Engine style: bw_now > bw(N bars ago)
      'rolling_min'  — MT4/R42 style: bw > rolling_min(lookback).shift(1)
    """
    _spread = spread_override if spread_override is not None else SPREAD
    _sl = sl_atr if sl_atr is not None else SL_ATR
    _tp = tp_atr if tp_atr is not None else TP_ATR
    _trail_act = trail_act if trail_act is not None else TRAIL_ACT
    _trail_dist = trail_dist if trail_dist is not None else TRAIL_DIST
    _max_hold = max_hold_override if max_hold_override is not None else MAX_HOLD

    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df['ADX'] = compute_adx(df)
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['KC_mid'] = df['Close'].ewm(span=25, adjust=False).mean()
    df['KC_upper'] = df['KC_mid'] + kc_mult * df['ATR']
    df['KC_lower'] = df['KC_mid'] - kc_mult * df['ATR']

    if kcbw_filter:
        df['KCBW'] = (df['KC_upper'] - df['KC_lower']) / df['KC_mid']
        if kcbw_mode == 'rolling_mean':
            df['KCBW_ref'] = df['KCBW'].rolling(kcbw_lookback).mean()
        elif kcbw_mode == 'lag_compare':
            df['KCBW_ref'] = df['KCBW'].shift(kcbw_lookback)
        elif kcbw_mode == 'rolling_min':
            df['KCBW_ref'] = df['KCBW'].rolling(kcbw_lookback).min().shift(1)
        else:
            df['KCBW_ref'] = df['KCBW'].rolling(kcbw_lookback).mean()

    df = df.dropna(subset=['ATR', 'ADX', 'EMA100', 'KC_upper'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; adx = df['ADX'].values
    ema100 = df['EMA100'].values
    kc_u = df['KC_upper'].values; kc_l = df['KC_lower'].values
    hours = df.index.hour
    kcbw = df['KCBW'].values if kcbw_filter else None
    kcbw_ref = df['KCBW_ref'].values if kcbw_filter else None
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999

    for i in range(1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], _spread, LOT, PV, times,
                               _sl, _tp, _trail_act, _trail_dist, _max_hold, CAP_ATR_MULT)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue

        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue

        if adx_th > 0:
            if np.isnan(adx[i]) or adx[i] < adx_th: continue

        if session_hours is not None:
            if hours[i] not in session_hours: continue

        if kcbw_filter and kcbw is not None and kcbw_ref is not None:
            if np.isnan(kcbw[i]) or np.isnan(kcbw_ref[i]): continue
            if kcbw[i] <= kcbw_ref[i]: continue

        if ema_type == 'ema100':
            buy_ok = c[i] > ema100[i]
            sell_ok = c[i] < ema100[i]
        elif ema_type == 'none':
            buy_ok = True; sell_ok = True
        else:
            buy_ok = c[i] > ema100[i]
            sell_ok = c[i] < ema100[i]

        if c[i] > kc_u[i] and buy_ok:
            pos = {'dir': 'BUY', 'entry': c[i] + _spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < kc_l[i] and sell_ok:
            pos = {'dir': 'SELL', 'entry': c[i] - _spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}

    return trades


# ═══════════════════════════════════════════════════════════════
# Stats helpers
# ═══════════════════════════════════════════════════════════════

def _trades_to_daily(trades):
    if not trades: return pd.Series(dtype=float)
    df = pd.DataFrame(trades)
    df['date'] = pd.to_datetime(df['entry_time']).dt.date
    return df.groupby('date')['pnl'].sum()


def _sharpe(daily):
    if len(daily) < 10: return 0.0
    return float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0


def compute_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'r_mult': 0, 'safety_margin': 0, 'max_dd': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    r_mult = avg_win / avg_loss if avg_loss > 0 else float('inf')
    breakeven = 1.0 / (1.0 + r_mult) if r_mult > 0 else 1.0
    wr = len(wins) / n * 100
    eq = daily.cumsum()
    max_dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 0 else 0
    return {
        'n': n, 'sharpe': round(_sharpe(daily), 3), 'pnl': round(sum(pnls), 2),
        'wr': round(wr, 1), 'r_mult': round(r_mult, 3),
        'safety_margin': round(wr - breakeven * 100, 1), 'max_dd': round(max_dd, 2),
    }


def filter_trades_by_era(trades, era_name):
    if era_name == 'full' or ERA_SEGMENTS[era_name] is None:
        return trades
    periods = ERA_SEGMENTS[era_name]
    filtered = []
    for t in trades:
        entry = pd.Timestamp(t['entry_time'])
        for start, end in periods:
            if pd.Timestamp(start) <= entry < pd.Timestamp(end):
                filtered.append(t); break
    return filtered


def load_h1():
    candidates = sorted(_glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
    if not candidates:
        raise FileNotFoundError("No H1 data found in data/download/")
    csv_path = candidates[-1]
    print(f"  Loading H1: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                        'close': 'Close', 'volume': 'Volume'}, inplace=True)
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    print(f"  {len(df)} bars ({df.index[0]} ~ {df.index[-1]})", flush=True)
    return df


def fmt(label, s):
    return (f"  {label:<40} {s['n']:>6} {s['sharpe']:>7.3f} ${s['pnl']:>9,.0f} "
            f"{s['wr']:>5.1f}% {s['r_mult']:>6.3f} {s['safety_margin']:>6.1f}pp")


HEADER = (f"  {'Config':<40} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} "
          f"{'R_mult':>6} {'Margin':>8}")


# ═══════════════════════════════════════════════════════════════
# Phase 1: KCBW Lookback Sensitivity
# ═══════════════════════════════════════════════════════════════

def phase1_lookback(h1_df):
    print(f"\n{'='*110}")
    print(f"  PHASE 1: KCBW Lookback Sensitivity (is 5 a plateau?)")
    print(f"{'='*110}")
    print(HEADER, flush=True)

    results = {}
    base_trades = bt_keltner_filtered(h1_df, kcbw_filter=False)
    base_s = compute_stats(base_trades)
    print(fmt("KCBW_OFF (baseline)", base_s), flush=True)
    results['KCBW_OFF'] = base_s

    for lb in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
        trades = bt_keltner_filtered(h1_df, kcbw_filter=True, kcbw_lookback=lb)
        s = compute_stats(trades)
        label = f"KCBW_lb{lb}"
        print(fmt(label, s), flush=True)
        results[label] = s

    sharpes = {k: v['sharpe'] for k, v in results.items() if k.startswith('KCBW_lb')}
    best_lb = max(sharpes, key=sharpes.get)
    worst_lb = min(sharpes, key=sharpes.get)
    spread_sh = max(sharpes.values()) - min(sharpes.values())

    print(f"\n  Plateau check: best={best_lb} ({sharpes[best_lb]:.3f}), "
          f"worst={worst_lb} ({sharpes[worst_lb]:.3f}), spread={spread_sh:.3f}")
    plateau = spread_sh < 1.0 and all(v > base_s['sharpe'] for v in sharpes.values())
    print(f"  All lookbacks beat baseline: {all(v > base_s['sharpe'] for v in sharpes.values())}")
    print(f"  Plateau (spread < 1.0 and all beat base): {'YES' if plateau else 'NO'}")

    return results


# ═══════════════════════════════════════════════════════════════
# Phase 2: KCBW Definition Consistency
# ═══════════════════════════════════════════════════════════════

def phase2_definitions(h1_df):
    print(f"\n{'='*110}")
    print(f"  PHASE 2: KCBW Definition Consistency")
    print(f"{'='*110}")
    print(HEADER, flush=True)

    results = {}
    for mode_name, mode_val in [('rolling_mean (R184)', 'rolling_mean'),
                                 ('lag_compare (Engine)', 'lag_compare'),
                                 ('rolling_min (MT4/R42)', 'rolling_min')]:
        for lb in [3, 5, 8, 10]:
            trades = bt_keltner_filtered(h1_df, kcbw_filter=True,
                                          kcbw_lookback=lb, kcbw_mode=mode_val)
            s = compute_stats(trades)
            label = f"{mode_name}_lb{lb}"
            print(fmt(label, s), flush=True)
            results[label] = {'mode': mode_val, 'lookback': lb, **s}

    return results


# ═══════════════════════════════════════════════════════════════
# Phase 3: Engine Cross-Validation (full M15 path)
# ═══════════════════════════════════════════════════════════════

def phase3_engine(h1_df):
    print(f"\n{'='*110}")
    print(f"  PHASE 3: Engine Cross-Validation (M15 execution path)")
    print(f"{'='*110}")

    try:
        from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS
    except ImportError as e:
        print(f"  SKIP: cannot import engine ({e})", flush=True)
        return None

    data = DataBundle.load_default()

    print(f"\n  Running baseline (kc_bw_filter_bars=0)...")
    kw_base = {**LIVE_PARITY_KWARGS, 'spread_cost': SPREAD,
               'min_lot_size': LOT, 'max_lot_size': LOT,
               'initial_capital': 5000, 'maxloss_cap': 0,
               'kc_bw_filter_bars': 0}
    base = run_variant(data, "Engine_KCBW_OFF", **kw_base)

    engine_results = {'KCBW_OFF': {
        'n': base['n'], 'sharpe': base['sharpe'], 'pnl': base['total_pnl'],
        'wr': base['win_rate'], 'skipped_kc_bw': base['skipped_kc_bw'],
    }}

    for bw_bars in [3, 5, 8, 10]:
        print(f"\n  Running kc_bw_filter_bars={bw_bars}...")
        kw = {**kw_base, 'kc_bw_filter_bars': bw_bars}
        r = run_variant(data, f"Engine_KCBW_{bw_bars}", **kw)
        engine_results[f'KCBW_{bw_bars}'] = {
            'n': r['n'], 'sharpe': r['sharpe'], 'pnl': r['total_pnl'],
            'wr': r['win_rate'], 'skipped_kc_bw': r['skipped_kc_bw'],
        }

    print(f"\n  Engine results:")
    print(f"  {'Config':<25} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'Skipped':>8}")
    for k, v in engine_results.items():
        pnl_s = f"${v['pnl']:>9,.0f}" if v['pnl'] >= 0 else f"-${abs(v['pnl']):>8,.0f}"
        print(f"  {k:<25} {v['n']:>6} {v['sharpe']:>7.2f} {pnl_s} {v['wr']:>5.1f}% {v['skipped_kc_bw']:>8}")

    return engine_results


# ═══════════════════════════════════════════════════════════════
# Phase 4: Walk-Forward OOS
# ═══════════════════════════════════════════════════════════════

def phase4_walkforward(h1_df):
    print(f"\n{'='*110}")
    print(f"  PHASE 4: Walk-Forward Out-of-Sample Validation")
    print(f"{'='*110}")

    print(f"\n  {'Window':<10} {'IS_Base':>8} {'IS_KCBW':>8} {'OOS_Base':>8} {'OOS_KCBW':>8} "
          f"{'OOS_delta':>9} {'OOS/IS':>7} {'Win':>4}")

    wf_results = []
    for wf_name, is_start, is_end, oos_start, oos_end in WF_WINDOWS:
        is_df = h1_df.loc[is_start:is_end].copy()
        oos_df = h1_df.loc[oos_start:oos_end].copy()
        if len(is_df) < 200 or len(oos_df) < 200:
            print(f"  {wf_name:<10} SKIP (insufficient data)")
            continue

        is_base = compute_stats(bt_keltner_filtered(is_df, kcbw_filter=False))
        is_kcbw = compute_stats(bt_keltner_filtered(is_df, kcbw_filter=True, kcbw_lookback=5))
        oos_base = compute_stats(bt_keltner_filtered(oos_df, kcbw_filter=False))
        oos_kcbw = compute_stats(bt_keltner_filtered(oos_df, kcbw_filter=True, kcbw_lookback=5))

        oos_delta = oos_kcbw['sharpe'] - oos_base['sharpe']
        is_delta = is_kcbw['sharpe'] - is_base['sharpe']
        oos_is_ratio = oos_delta / is_delta if is_delta != 0 else 0
        win = "+" if oos_delta > 0 else "-"

        print(f"  {wf_name:<10} {is_base['sharpe']:>8.3f} {is_kcbw['sharpe']:>8.3f} "
              f"{oos_base['sharpe']:>8.3f} {oos_kcbw['sharpe']:>8.3f} "
              f"{oos_delta:>+9.3f} {oos_is_ratio:>7.2f} {win:>4}")

        wf_results.append({
            'window': wf_name,
            'is_base': is_base['sharpe'], 'is_kcbw': is_kcbw['sharpe'],
            'oos_base': oos_base['sharpe'], 'oos_kcbw': oos_kcbw['sharpe'],
            'oos_delta': oos_delta, 'oos_is_ratio': oos_is_ratio, 'win': oos_delta > 0,
        })

    wins = sum(1 for r in wf_results if r['win'])
    total = len(wf_results)
    avg_oos_delta = np.mean([r['oos_delta'] for r in wf_results]) if wf_results else 0
    avg_ratio = np.mean([r['oos_is_ratio'] for r in wf_results]) if wf_results else 0
    wf_pass = wins >= max(1, int(total * 0.6))

    print(f"\n  WF OOS wins: {wins}/{total} {'PASS' if wf_pass else 'FAIL'}")
    print(f"  Avg OOS Sharpe delta: {avg_oos_delta:+.3f}")
    print(f"  Avg OOS/IS ratio: {avg_ratio:.2f} (ideal > 0.5)")

    return {'windows': wf_results, 'wins': wins, 'total': total,
            'pass': wf_pass, 'avg_oos_delta': avg_oos_delta}


# ═══════════════════════════════════════════════════════════════
# Phase 5: Parameter Perturbation MC
# ═══════════════════════════════════════════════════════════════

def phase5_param_perturb_mc(h1_df):
    print(f"\n{'='*110}")
    print(f"  PHASE 5: Parameter Perturbation Monte Carlo ({MC_ITERATIONS} sims)")
    print(f"{'='*110}")

    rng = np.random.RandomState(MC_SEED)
    base_params = {'sl_atr': SL_ATR, 'tp_atr': TP_ATR,
                   'trail_act': TRAIL_ACT, 'trail_dist': TRAIL_DIST}
    perturb_ranges = {
        'sl_atr': (2.5, 5.0), 'tp_atr': (5.0, 12.0),
        'trail_act': (0.08, 0.22), 'trail_dist': (0.015, 0.045),
        'kcbw_lookback': (3, 10),
    }

    base_sharpes = []
    kcbw_sharpes = []
    n_sims = min(MC_ITERATIONS, 500)

    for it in range(n_sims):
        params = {}
        for k, (lo, hi) in perturb_ranges.items():
            params[k] = rng.uniform(lo, hi)

        kcbw_lb = max(2, int(round(params.pop('kcbw_lookback'))))

        bt_base = bt_keltner_filtered(h1_df, kcbw_filter=False,
                                       sl_atr=params['sl_atr'], tp_atr=params['tp_atr'],
                                       trail_act=params['trail_act'], trail_dist=params['trail_dist'])
        bt_kcbw = bt_keltner_filtered(h1_df, kcbw_filter=True, kcbw_lookback=kcbw_lb,
                                       sl_atr=params['sl_atr'], tp_atr=params['tp_atr'],
                                       trail_act=params['trail_act'], trail_dist=params['trail_dist'])

        base_sharpes.append(compute_stats(bt_base)['sharpe'])
        kcbw_sharpes.append(compute_stats(bt_kcbw)['sharpe'])

        if (it + 1) % 100 == 0:
            print(f"    {it+1}/{n_sims} sims done...", flush=True)

    base_sharpes = np.array(base_sharpes)
    kcbw_sharpes = np.array(kcbw_sharpes)
    deltas = kcbw_sharpes - base_sharpes

    prob_better = float(np.mean(deltas > 0))
    delta_5th = float(np.percentile(deltas, 5))
    delta_50th = float(np.percentile(deltas, 50))
    delta_95th = float(np.percentile(deltas, 95))
    kcbw_5th = float(np.percentile(kcbw_sharpes, 5))
    base_median = float(np.median(base_sharpes))

    mc_pass = prob_better >= 0.65 and delta_5th > -0.5

    print(f"\n  P(KCBW > base): {prob_better:.1%}")
    print(f"  Delta Sharpe: 5th={delta_5th:+.3f}, median={delta_50th:+.3f}, 95th={delta_95th:+.3f}")
    print(f"  KCBW 5th pctl Sharpe: {kcbw_5th:.3f} vs Base median: {base_median:.3f}")
    print(f"  Robust: {'PASS' if mc_pass else 'FAIL'} "
          f"(KCBW 5th > base median: {'YES' if kcbw_5th > base_median else 'NO'})")

    return {
        'n_sims': n_sims, 'prob_better': round(prob_better, 3),
        'delta_5th': round(delta_5th, 3), 'delta_median': round(delta_50th, 3),
        'delta_95th': round(delta_95th, 3),
        'kcbw_5th': round(kcbw_5th, 3), 'base_median': round(base_median, 3),
        'pass': mc_pass,
    }


# ═══════════════════════════════════════════════════════════════
# Phase 6: Cost Sensitivity
# ═══════════════════════════════════════════════════════════════

def phase6_cost_sensitivity(h1_df):
    print(f"\n{'='*110}")
    print(f"  PHASE 6: Cost Sensitivity (varying spread)")
    print(f"{'='*110}")

    print(f"\n  {'Spread':>8} {'Base_Sh':>8} {'KCBW_Sh':>8} {'Delta':>8} {'Base_PnL':>10} {'KCBW_PnL':>10}")

    results = {}
    for sp in [0.20, 0.30, 0.50, 0.80, 1.00]:
        bt_b = bt_keltner_filtered(h1_df, kcbw_filter=False, spread_override=sp)
        bt_k = bt_keltner_filtered(h1_df, kcbw_filter=True, kcbw_lookback=5, spread_override=sp)
        sb = compute_stats(bt_b)
        sk = compute_stats(bt_k)
        delta = sk['sharpe'] - sb['sharpe']
        print(f"  ${sp:<7.2f} {sb['sharpe']:>8.3f} {sk['sharpe']:>8.3f} {delta:>+8.3f} "
              f"${sb['pnl']:>9,.0f} ${sk['pnl']:>9,.0f}")
        results[f"sp_{sp}"] = {'spread': sp, 'base': sb, 'kcbw': sk, 'delta': delta}

    return results


# ═══════════════════════════════════════════════════════════════
# Phase 7: Yearly Stability
# ═══════════════════════════════════════════════════════════════

def phase7_yearly(h1_df):
    print(f"\n{'='*110}")
    print(f"  PHASE 7: Yearly Stability (per-year Sharpe delta)")
    print(f"{'='*110}")

    base_trades = bt_keltner_filtered(h1_df, kcbw_filter=False)
    kcbw_trades = bt_keltner_filtered(h1_df, kcbw_filter=True, kcbw_lookback=5)

    years = sorted(set(pd.Timestamp(t['entry_time']).year for t in base_trades))

    print(f"\n  {'Year':>6} {'Base_Sh':>8} {'KCBW_Sh':>8} {'Delta':>8} {'Base_N':>7} {'KCBW_N':>7} {'Win':>4}")

    results = {}
    wins = 0
    for yr in years:
        yr_base = [t for t in base_trades if pd.Timestamp(t['entry_time']).year == yr]
        yr_kcbw = [t for t in kcbw_trades if pd.Timestamp(t['entry_time']).year == yr]
        sb = compute_stats(yr_base)
        sk = compute_stats(yr_kcbw)
        delta = sk['sharpe'] - sb['sharpe']
        w = "+" if delta > 0 else "-"
        if delta > 0: wins += 1
        print(f"  {yr:>6} {sb['sharpe']:>8.3f} {sk['sharpe']:>8.3f} {delta:>+8.3f} "
              f"{sb['n']:>7} {sk['n']:>7} {w:>4}")
        results[yr] = {'base': sb['sharpe'], 'kcbw': sk['sharpe'], 'delta': delta}

    total = len(years)
    print(f"\n  Year wins: {wins}/{total} ({wins/total*100:.0f}%)")
    print(f"  Concentrated: {'NO (>60% years positive)' if wins/total > 0.6 else 'YES — WARNING'}")

    return {'years': results, 'wins': wins, 'total': total}


# ═══════════════════════════════════════════════════════════════
# Phase 8: PBO (Probability of Backtest Overfitting)
# ═══════════════════════════════════════════════════════════════

def phase8_pbo(h1_df):
    print(f"\n{'='*110}")
    print(f"  PHASE 8: PBO — Probability of Backtest Overfitting")
    print(f"{'='*110}")

    n_segments = 8
    df_len = len(h1_df)
    seg_size = df_len // n_segments
    segments = []
    for s in range(n_segments):
        start_idx = s * seg_size
        end_idx = (s + 1) * seg_size if s < n_segments - 1 else df_len
        segments.append(h1_df.iloc[start_idx:end_idx].copy())

    n_is = n_segments // 2
    all_combos = list(combinations(range(n_segments), n_is))
    n_combos = len(all_combos)
    print(f"  {n_segments} segments, C({n_segments},{n_is}) = {n_combos} combinations")

    overfit_count = 0
    for combo_idx, is_indices in enumerate(all_combos):
        oos_indices = [i for i in range(n_segments) if i not in is_indices]

        is_df = pd.concat([segments[i] for i in is_indices])
        oos_df = pd.concat([segments[i] for i in oos_indices])

        is_base = compute_stats(bt_keltner_filtered(is_df, kcbw_filter=False))
        is_kcbw = compute_stats(bt_keltner_filtered(is_df, kcbw_filter=True, kcbw_lookback=5))

        is_pick_kcbw = is_kcbw['sharpe'] > is_base['sharpe']

        oos_base = compute_stats(bt_keltner_filtered(oos_df, kcbw_filter=False))
        oos_kcbw = compute_stats(bt_keltner_filtered(oos_df, kcbw_filter=True, kcbw_lookback=5))

        if is_pick_kcbw:
            if oos_kcbw['sharpe'] <= oos_base['sharpe']:
                overfit_count += 1
        else:
            if oos_base['sharpe'] <= oos_kcbw['sharpe']:
                overfit_count += 1

        if (combo_idx + 1) % 20 == 0:
            print(f"    {combo_idx+1}/{n_combos} combos done...", flush=True)

    pbo = overfit_count / n_combos
    print(f"\n  PBO = {pbo:.3f} ({overfit_count}/{n_combos})")
    print(f"  Interpretation: {'GOOD (< 0.30)' if pbo < 0.30 else ('MARGINAL (0.30-0.50)' if pbo < 0.50 else 'HIGH RISK (>= 0.50)')}")

    return {'pbo': round(pbo, 3), 'overfit_count': overfit_count, 'n_combos': n_combos}


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 110, flush=True)
    print("  R185 — KCBW Deep Validation", flush=True)
    print("=" * 110, flush=True)

    h1_df = load_h1()

    all_results = {}

    all_results['phase1_lookback'] = phase1_lookback(h1_df)
    all_results['phase2_definitions'] = phase2_definitions(h1_df)
    all_results['phase3_engine'] = phase3_engine(h1_df)
    all_results['phase4_walkforward'] = phase4_walkforward(h1_df)
    all_results['phase5_param_mc'] = phase5_param_perturb_mc(h1_df)
    all_results['phase6_cost'] = phase6_cost_sensitivity(h1_df)
    all_results['phase7_yearly'] = phase7_yearly(h1_df)
    all_results['phase8_pbo'] = phase8_pbo(h1_df)

    elapsed = time.time() - t0

    # ── Final Summary ──
    print(f"\n\n{'='*110}")
    print(f"  R185 FINAL SUMMARY")
    print(f"{'='*110}")

    p1 = all_results['phase1_lookback']
    lb_sharpes = {k: v['sharpe'] for k, v in p1.items() if k.startswith('KCBW_lb')}
    all_beat = all(v > p1['KCBW_OFF']['sharpe'] for v in lb_sharpes.values())
    print(f"\n  1. Lookback Plateau: {'YES' if all_beat else 'NO'} "
          f"(range: {min(lb_sharpes.values()):.3f} — {max(lb_sharpes.values()):.3f})")

    p2 = all_results['phase2_definitions']
    for mode in ['rolling_mean', 'lag_compare', 'rolling_min']:
        sub = {k: v for k, v in p2.items() if mode in k}
        sharpes_m = [v['sharpe'] for v in sub.values()]
        print(f"  2. {mode}: Sharpe range {min(sharpes_m):.3f} — {max(sharpes_m):.3f}")

    p3 = all_results.get('phase3_engine')
    if p3:
        base_eng = p3.get('KCBW_OFF', {})
        best_eng = max([v for k, v in p3.items() if k != 'KCBW_OFF'],
                       key=lambda x: x['sharpe'], default=base_eng)
        print(f"  3. Engine: base Sharpe={base_eng.get('sharpe', 0):.2f}, "
              f"best KCBW Sharpe={best_eng.get('sharpe', 0):.2f}")

    p4 = all_results['phase4_walkforward']
    print(f"  4. Walk-Forward: {p4['wins']}/{p4['total']} OOS wins, "
          f"avg delta={p4['avg_oos_delta']:+.3f} {'PASS' if p4['pass'] else 'FAIL'}")

    p5 = all_results['phase5_param_mc']
    print(f"  5. Param MC: P(better)={p5['prob_better']:.1%}, "
          f"delta 5th={p5['delta_5th']:+.3f} {'PASS' if p5['pass'] else 'FAIL'}")

    p6 = all_results['phase6_cost']
    deltas_cost = [v['delta'] for v in p6.values()]
    print(f"  6. Cost: KCBW advantage at all spreads: "
          f"{'YES' if all(d > 0 for d in deltas_cost) else 'NO'} "
          f"(range: {min(deltas_cost):+.3f} — {max(deltas_cost):+.3f})")

    p7 = all_results['phase7_yearly']
    print(f"  7. Yearly: {p7['wins']}/{p7['total']} years positive "
          f"({'NOT concentrated' if p7['wins']/p7['total'] > 0.6 else 'CONCENTRATED — risk'})")

    p8 = all_results['phase8_pbo']
    print(f"  8. PBO: {p8['pbo']:.3f} "
          f"({'GOOD' if p8['pbo'] < 0.30 else ('MARGINAL' if p8['pbo'] < 0.50 else 'HIGH RISK')})")

    tests = [
        all_beat,
        p4['pass'],
        p5['pass'],
        all(d > 0 for d in deltas_cost),
        p7['wins']/p7['total'] > 0.6,
        p8['pbo'] < 0.30,
    ]
    passed = sum(tests)
    print(f"\n  Overall: {passed}/6 tests passed")
    if passed >= 5:
        print(f"  VERDICT: STRONG GO — deploy KCBW to live")
    elif passed >= 4:
        print(f"  VERDICT: GO with monitoring")
    elif passed >= 3:
        print(f"  VERDICT: CAUTION — review failing tests before deploying")
    else:
        print(f"  VERDICT: NO-GO — KCBW may be overfit")

    print(f"\n  Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    out_path = OUTPUT_DIR / "r185_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {out_path}")
    print(f"{'='*110}")


if __name__ == "__main__":
    main()
