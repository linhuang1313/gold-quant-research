#!/usr/bin/env python3
"""
R167-R169 — Batch 3: Multi-Timeframe, Session, & Volatility Regime
====================================================================
Three exploratory experiments for Keltner (L8_MAX) enhancement:

R167: Multi-Timeframe Confirmation (H4 direction + H1 entry)
  - Construct H4 bars from H1 data (resample 4 H1 bars)
  - H4 EMA20 slope, Price vs EMA50, HH/HL filter
  - Only allow Keltner entries when H4 trend agrees with trade direction
  - K-Fold on best filter

R168: Intraday Session Optimization
  - Group L8_MAX trades by entry hour (UTC 0-23)
  - Identify "bad hours" (negative avg PnL or WR < 50%)
  - Post-filter by allowed hours, day-of-week, combined
  - K-Fold on best session filter

R169: Volatility Regime Prediction Model
  - Feature engineering: ATR ratio, daily range ratio, spike count, etc.
  - Define regimes: LOW / NORMAL / HIGH / EXTREME
  - Test regime-based lot sizing (scale PnL post-hoc)
  - Test regime-based SL adjustment (clip PnL post-hoc)
  - K-Fold on best regime approach
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, parent_dir)
warnings.filterwarnings('ignore')

from backtest.runner import (
    DataBundle, run_variant, LIVE_PARITY_KWARGS,
    load_m15, load_h1_aligned, H1_CSV_PATH,
)

OUTPUT_DIR = Path("results/r167_r169_batch3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01
CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.09, 'SESS_BO': 0.09}

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]

t0 = time.time()


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


def add_psar(df, af_start=0.01, af_max=0.05):
    df = df.copy(); n = len(df)
    psar = np.zeros(n); direction = np.ones(n)
    af = af_start; ep = df['High'].iloc[0]; psar[0] = df['Low'].iloc[0]
    for i in range(1, n):
        prev = psar[i-1]
        if direction[i-1] == 1:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], df['Low'].iloc[i-1], df['Low'].iloc[max(0, i-2)])
            if df['Low'].iloc[i] < psar[i]:
                direction[i] = -1; psar[i] = ep; ep = df['Low'].iloc[i]; af = af_start
            else:
                direction[i] = 1
                if df['High'].iloc[i] > ep:
                    ep = df['High'].iloc[i]; af = min(af+af_start, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], df['High'].iloc[i-1], df['High'].iloc[max(0, i-2)])
            if df['High'].iloc[i] > psar[i]:
                direction[i] = 1; psar[i] = ep; ep = df['High'].iloc[i]; af = af_start
            else:
                direction[i] = -1
                if df['Low'].iloc[i] < ep:
                    ep = df['Low'].iloc[i]; af = min(af+af_start, af_max)
    df['PSAR_dir'] = direction; df['ATR'] = compute_atr(df)
    return df


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


def _run_exit_with_cap(pos, i, h, lo_v, c, spread, lot, pv, times,
                       sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold,
                       maxloss_cap=0):
    held = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_c = (c - pos['entry'] - spread) * lot * pv
    else:
        pnl_c = (pos['entry'] - c - spread) * lot * pv
    tp_val = tp_atr * pos['atr'] * lot * pv
    sl_val = sl_atr * pos['atr'] * lot * pv
    if pos['dir'] == 'BUY':
        pnl_h = (h - pos['entry'] - spread) * lot * pv
        pnl_l = (lo_v - pos['entry'] - spread) * lot * pv
    else:
        pnl_h = (pos['entry'] - lo_v - spread) * lot * pv
        pnl_l = (pos['entry'] - h - spread) * lot * pv
    if pnl_h >= tp_val:
        return _mk(pos, c, times[i], "TP", i, tp_val)
    if pnl_l <= -sl_val:
        return _mk(pos, c, times[i], "SL", i, -sl_val)
    if maxloss_cap > 0 and pnl_c < -maxloss_cap:
        return _mk(pos, c, times[i], "MaxLossCap", i, -maxloss_cap)
    ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
    if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
        ts_p = h - td
        if lo_v <= ts_p:
            return _mk(pos, c, times[i], "Trail", i, (ts_p - pos['entry'] - spread) * lot * pv)
    elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
        ts_p = lo_v + td
        if h >= ts_p:
            return _mk(pos, c, times[i], "Trail", i, (pos['entry'] - ts_p - spread) * lot * pv)
    if held >= max_hold:
        return _mk(pos, c, times[i], "Timeout", i, pnl_c)
    return None


def bt_psar(h1_df, spread, lot, maxloss_cap=0):
    sl_atr=4.5; tp_atr=16.0; trail_act=0.20; trail_dist=0.04; max_hold=20
    df = add_psar(h1_df).dropna(subset=['PSAR_dir', 'ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    pdir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pdir[i-1] == -1 and pdir[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif pdir[i-1] == 1 and pdir[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom(h1_df, spread, lot, maxloss_cap=0):
    fast=480; slow=720; sl_atr=4.5; tp_atr=6.0
    trail_act=0.14; trail_dist=0.025; max_hold=20
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    max_lb = max(fast, slow)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        if c[i-fast] > 0: s += 0.5 * np.sign(c[i]/c[i-fast] - 1.0)
        if c[i-slow] > 0: s += 0.5 * np.sign(c[i]/c[i-slow] - 1.0)
        score[i] = s
    trades = []; pos = None; last_exit = -999
    for i in range(max_lb+1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and score[i] < 0:
                pnl = (c[i]-pos['entry']-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and score[i] > 0:
                pnl = (pos['entry']-c[i]-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(score[i]) or np.isnan(score[i-1]): continue
        if score[i] > 0 and score[i-1] <= 0:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_sess_bo(h1_df, spread, lot, maxloss_cap=0):
    session_hour=12; lookback=4; sl_atr=4.5; tp_atr=4.0
    trail_act=0.14; trail_dist=0.025; max_hold=20
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(lookback, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if hours[i] != session_hour: continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_l8_max(data_bundle, spread, lot, maxloss_cap=35):
    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': maxloss_cap,
          'spread_cost': spread, 'initial_capital': 2000,
          'keltner_max_hold_m15': 8,
          'min_lot_size': lot, 'max_lot_size': lot}
    result = run_variant(data_bundle, "L8_MAX", verbose=False, **kw)
    raw = result.get('_trades', [])
    return [
        {'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
         'entry_time': t.entry_time, 'exit_time': t.exit_time,
         'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0}
        for t in raw
    ]


def _normalize_ts(ts):
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    return t


def merge_portfolio_trades(strat_trades_dict, lot_scale=None):
    if lot_scale is None:
        lot_scale = R89_LOTS
    all_trades = []
    for strat_name, trades in strat_trades_dict.items():
        mult = lot_scale.get(strat_name, UNIT_LOT) / UNIT_LOT
        for t in trades:
            all_trades.append({
                'strategy': strat_name,
                'dir': t['dir'], 'entry': t['entry'], 'exit': t['exit'],
                'entry_time': _normalize_ts(t['entry_time']),
                'exit_time': _normalize_ts(t['exit_time']),
                'pnl': t['pnl'] * mult,
                'pnl_unit': t['pnl'],
                'reason': t['reason'],
            })
    all_trades.sort(key=lambda x: x['exit_time'])
    return all_trades


def trades_to_daily(trades):
    if not trades:
        return np.array([0.0])
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).normalize()
        if hasattr(d, 'tz') and d.tzinfo is not None:
            d = d.tz_localize(None)
        daily[d] = daily.get(d, 0) + t['pnl']
    if not daily:
        return np.array([0.0])
    return np.array([daily[k] for k in sorted(daily.keys())])


def sharpe(arr):
    if len(arr) < 10 or np.std(arr, ddof=1) == 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))


def max_dd(arr):
    if len(arr) < 2:
        return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def compute_stats(trades, label=""):
    n = len(trades)
    if n == 0:
        return {'label': label, 'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0,
                'max_dd': 0, 'avg_pnl': 0, 'worst_trade': 0}
    pnls = np.array([t['pnl'] for t in trades])
    daily = trades_to_daily(trades)
    return {
        'label': label, 'n': n,
        'sharpe': round(sharpe(daily), 3),
        'pnl': round(float(pnls.sum()), 2),
        'wr': round(float((pnls > 0).sum()) / n * 100, 1),
        'max_dd': round(max_dd(daily), 2),
        'avg_pnl': round(float(pnls.mean()), 4),
        'worst_trade': round(float(pnls.min()), 2),
    }


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


# ═══════════════════════════════════════════════════════════════
# R167: Multi-Timeframe Confirmation (H4 direction + H1 entry)
# ═══════════════════════════════════════════════════════════════

def build_h4_from_h1(h1_df):
    """Construct H4 bars by resampling 4 H1 bars."""
    h4_df = h1_df.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    h4_df['EMA20'] = h4_df['Close'].ewm(span=20).mean()
    h4_df['EMA50'] = h4_df['Close'].ewm(span=50).mean()
    h4_df['EMA20_slope'] = h4_df['EMA20'].diff(3)
    return h4_df


def get_h4_signals_at_time(h4_df, entry_time):
    """Find latest H4 bar before entry_time, return filter signals."""
    ts = pd.Timestamp(entry_time)
    if h4_df.index.tz is not None and ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    elif h4_df.index.tz is None and ts.tzinfo is not None:
        ts = ts.tz_localize(None)

    mask = h4_df.index <= ts
    if not mask.any():
        return None

    row = h4_df.loc[mask].iloc[-1]
    return {
        'ema20_slope': row['EMA20_slope'],
        'close_vs_ema50': row['Close'] - row['EMA50'],
        'close': row['Close'],
        'ema50': row['EMA50'],
    }


def filter_trades_h4(trades, h4_df, filter_type='A'):
    """Post-filter L8_MAX trades using H4 trend confirmation.

    Filter A: EMA20 slope direction
    Filter B: Price vs EMA50
    Filter C: Both A and B must agree
    """
    kept = []
    skipped = 0
    for t in trades:
        sig = get_h4_signals_at_time(h4_df, t['entry_time'])
        if sig is None:
            kept.append(t)
            continue

        is_buy = t['dir'] == 'BUY'

        if filter_type == 'A':
            if is_buy and sig['ema20_slope'] <= 0:
                skipped += 1; continue
            if not is_buy and sig['ema20_slope'] >= 0:
                skipped += 1; continue
        elif filter_type == 'B':
            if is_buy and sig['close_vs_ema50'] <= 0:
                skipped += 1; continue
            if not is_buy and sig['close_vs_ema50'] >= 0:
                skipped += 1; continue
        elif filter_type == 'C':
            slope_ok = (is_buy and sig['ema20_slope'] > 0) or (not is_buy and sig['ema20_slope'] < 0)
            ema50_ok = (is_buy and sig['close_vs_ema50'] > 0) or (not is_buy and sig['close_vs_ema50'] < 0)
            if not (slope_ok and ema50_ok):
                skipped += 1; continue

        kept.append(t)
    return kept, skipped


def run_r167(bundle, h1_df):
    print("\n" + "=" * 80)
    print("  R167 — Multi-Timeframe Confirmation (H4 direction + H1 entry)")
    print("=" * 80, flush=True)

    results = {}

    # Phase 1: Build H4
    print("\n  Phase 1: Building H4 bars from H1...", flush=True)
    h4_df = build_h4_from_h1(h1_df)
    print(f"    H4 bars: {len(h4_df)} ({h4_df.index[0]} -> {h4_df.index[-1]})", flush=True)
    results['h4_bars'] = len(h4_df)

    # Phase 2: H4 trend signal coverage
    print("\n  Phase 2: H4 trend signal summary...", flush=True)
    slope_pos = (h4_df['EMA20_slope'] > 0).sum()
    slope_neg = (h4_df['EMA20_slope'] < 0).sum()
    above_ema50 = (h4_df['Close'] > h4_df['EMA50']).sum()
    below_ema50 = (h4_df['Close'] < h4_df['EMA50']).sum()
    print(f"    EMA20 slope: {slope_pos} rising, {slope_neg} falling", flush=True)
    print(f"    Price vs EMA50: {above_ema50} above, {below_ema50} below", flush=True)
    results['h4_signals'] = {
        'ema20_slope_rising': int(slope_pos), 'ema20_slope_falling': int(slope_neg),
        'above_ema50': int(above_ema50), 'below_ema50': int(below_ema50),
    }

    # Phase 3: Baseline L8_MAX
    print("\n  Phase 3: Running baseline L8_MAX...", flush=True)
    baseline_trades = bt_l8_max(bundle, SPREAD, UNIT_LOT, CAPS['L8_MAX'])
    base_stats = compute_stats(baseline_trades, "Baseline")
    print(f"    Baseline: n={base_stats['n']}, Sharpe={base_stats['sharpe']:.3f}, "
          f"PnL={fmt(base_stats['pnl'])}", flush=True)
    results['baseline'] = base_stats

    # Phase 4: Test each H4 filter
    print("\n  Phase 4: Testing H4 filters...", flush=True)
    print(f"\n  {'Filter':<20} {'N':>6} {'Skip':>6} {'Sharpe':>7} {'PnL':>10} "
          f"{'MaxDD':>8} {'WR':>6} {'AvgPnL':>8}")
    print("  " + "-" * 75)

    filter_results = {}
    for ftype, fdesc in [('A', 'EMA20 slope'), ('B', 'Price vs EMA50'), ('C', 'Combined A+B')]:
        filtered, skipped = filter_trades_h4(baseline_trades, h4_df, ftype)
        st = compute_stats(filtered, f"H4_{ftype}")
        filter_results[ftype] = {**st, 'skipped': skipped, 'desc': fdesc}
        print(f"  H4_{ftype} ({fdesc:<14}) {st['n']:>6} {skipped:>6} {st['sharpe']:>7.3f} "
              f"{st['pnl']:>10.1f} {st['max_dd']:>8.1f} {st['wr']:>5.1f}% {st['avg_pnl']:>8.4f}",
              flush=True)

    results['filters'] = filter_results

    best_filter = max(filter_results.items(), key=lambda x: x[1]['sharpe'])
    best_ftype = best_filter[0]
    print(f"\n  >>> Best H4 filter: {best_ftype} ({best_filter[1]['desc']}) "
          f"— Sharpe={best_filter[1]['sharpe']:.3f} (baseline: {base_stats['sharpe']:.3f})",
          flush=True)
    results['best_filter'] = best_ftype

    # Phase 5: K-Fold on best filter
    print(f"\n  Phase 5: K-Fold validation on filter {best_ftype}...", flush=True)
    fold_sharpes = []
    for fold_name, start, end in FOLDS:
        try:
            fold_b = bundle.slice(start, end)
            fold_trades = bt_l8_max(fold_b, SPREAD, UNIT_LOT, CAPS['L8_MAX'])
        except Exception:
            fold_sharpes.append(0.0)
            continue
        h1_fold = h1_df[(h1_df.index >= start) & (h1_df.index < end)]
        if len(h1_fold) < 100:
            fold_sharpes.append(0.0)
            continue
        h4_fold = build_h4_from_h1(h1_fold)
        if len(h4_fold) < 50:
            fold_sharpes.append(0.0)
            continue
        filtered, _ = filter_trades_h4(fold_trades, h4_fold, best_ftype)
        daily = trades_to_daily(filtered)
        fold_sharpes.append(sharpe(daily))

    pos = sum(1 for s in fold_sharpes if s > 0)
    print(f"    Folds: {[round(s, 2) for s in fold_sharpes]}, {pos}/6 positive", flush=True)
    results['kfold'] = {
        'filter': best_ftype,
        'fold_sharpes': [round(s, 3) for s in fold_sharpes],
        'positive_folds': pos,
        'mean_sharpe': round(float(np.mean(fold_sharpes)), 3),
    }

    # Also K-Fold baseline for comparison
    print(f"  K-Fold baseline (no filter)...", flush=True)
    fold_sharpes_base = []
    for fold_name, start, end in FOLDS:
        try:
            fold_b = bundle.slice(start, end)
            fold_trades = bt_l8_max(fold_b, SPREAD, UNIT_LOT, CAPS['L8_MAX'])
        except Exception:
            fold_sharpes_base.append(0.0)
            continue
        daily = trades_to_daily(fold_trades)
        fold_sharpes_base.append(sharpe(daily))
    pos_base = sum(1 for s in fold_sharpes_base if s > 0)
    print(f"    Baseline folds: {[round(s, 2) for s in fold_sharpes_base]}, "
          f"{pos_base}/6 positive", flush=True)
    results['kfold_baseline'] = {
        'fold_sharpes': [round(s, 3) for s in fold_sharpes_base],
        'positive_folds': pos_base,
        'mean_sharpe': round(float(np.mean(fold_sharpes_base)), 3),
    }

    elapsed = time.time() - t0
    print(f"\n  R167 done — {elapsed:.0f}s", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# R168: Intraday Session Optimization
# ═══════════════════════════════════════════════════════════════

def filter_by_hours(trades, allowed_hours):
    return [t for t in trades if pd.Timestamp(t['entry_time']).hour in allowed_hours]


def filter_by_dow(trades, allowed_dow):
    return [t for t in trades if pd.Timestamp(t['entry_time']).dayofweek in allowed_dow]


def filter_by_hours_and_dow(trades, allowed_hours, allowed_dow):
    return [t for t in trades
            if pd.Timestamp(t['entry_time']).hour in allowed_hours
            and pd.Timestamp(t['entry_time']).dayofweek in allowed_dow]


def run_r168(bundle, h1_df):
    print("\n" + "=" * 80)
    print("  R168 — Intraday Session Optimization")
    print("=" * 80, flush=True)

    results = {}

    # Phase 1: Baseline trades with hour annotation
    print("\n  Phase 1: Baseline L8_MAX with hour/DOW annotation...", flush=True)
    baseline_trades = bt_l8_max(bundle, SPREAD, UNIT_LOT, CAPS['L8_MAX'])
    base_stats = compute_stats(baseline_trades, "Baseline")
    print(f"    Baseline: n={base_stats['n']}, Sharpe={base_stats['sharpe']:.3f}, "
          f"PnL={fmt(base_stats['pnl'])}", flush=True)
    results['baseline'] = base_stats

    # Phase 2: Per-hour stats
    print("\n  Phase 2: Per-hour performance (UTC)...", flush=True)
    print(f"\n  {'Hour':>6} {'N':>6} {'WR':>6} {'Sharpe':>7} {'AvgPnL':>8} {'TotalPnL':>10}")
    print("  " + "-" * 55)

    hour_groups = defaultdict(list)
    for t in baseline_trades:
        h = pd.Timestamp(t['entry_time']).hour
        hour_groups[h].append(t)

    hour_stats = {}
    bad_hours = []
    for h in range(24):
        ht = hour_groups.get(h, [])
        if not ht:
            hour_stats[h] = {'n': 0, 'wr': 0, 'sharpe': 0, 'avg_pnl': 0, 'total_pnl': 0}
            continue
        pnls = [t['pnl'] for t in ht]
        daily = trades_to_daily(ht)
        n_h = len(ht)
        wr = sum(1 for p in pnls if p > 0) / n_h * 100
        sh = sharpe(daily)
        avg_p = float(np.mean(pnls))
        total_p = float(sum(pnls))
        hour_stats[h] = {'n': n_h, 'wr': round(wr, 1), 'sharpe': round(sh, 3),
                         'avg_pnl': round(avg_p, 4), 'total_pnl': round(total_p, 2)}
        is_bad = avg_p < 0 or wr < 50
        marker = " <<< BAD" if is_bad else ""
        if is_bad and n_h >= 5:
            bad_hours.append(h)
        print(f"  {h:>4}h {n_h:>6} {wr:>5.1f}% {sh:>7.3f} {avg_p:>8.4f} {total_p:>10.1f}{marker}",
              flush=True)

    results['hour_stats'] = {str(k): v for k, v in hour_stats.items()}
    results['bad_hours'] = bad_hours
    print(f"\n  Bad hours (neg avg PnL or WR<50%, n>=5): {bad_hours}", flush=True)

    # Phase 3: Identify good hours
    good_hours = [h for h in range(24) if h not in bad_hours]
    print(f"  Good hours: {good_hours}", flush=True)

    # Phase 4: Test blocking bad hours
    print("\n  Phase 4: Session filter sweep...", flush=True)
    print(f"\n  {'Config':<35} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'MaxDD':>8} {'WR':>6}")
    print("  " + "-" * 75)

    session_configs = [
        ("All hours (baseline)", list(range(24))),
        ("Block bad hours", good_hours),
        ("Asia only (00-07)", list(range(0, 8))),
        ("Asia+London (00-12)", list(range(0, 13))),
        ("Asia+London+earlyUS (00-16)", list(range(0, 17))),
        ("No US core (block 13-20)", [h for h in range(24) if h not in range(13, 21)]),
        ("Block late US (no 17-20)", [h for h in range(24) if h not in [17, 18, 19, 20]]),
    ]

    session_results = {}
    for label, hours in session_configs:
        filtered = filter_by_hours(baseline_trades, hours)
        st = compute_stats(filtered, label)
        session_results[label] = {**st, 'hours': hours}
        print(f"  {label:<35} {st['n']:>6} {st['sharpe']:>7.3f} {st['pnl']:>10.1f} "
              f"{st['max_dd']:>8.1f} {st['wr']:>5.1f}%", flush=True)

    results['session_sweep'] = session_results

    best_session = max(session_results.items(), key=lambda x: x[1]['sharpe'])
    best_session_label = best_session[0]
    best_session_hours = best_session[1]['hours']
    print(f"\n  >>> Best session: {best_session_label} — "
          f"Sharpe={best_session[1]['sharpe']:.3f}", flush=True)

    # Phase 5: Day-of-week analysis
    print("\n  Phase 5: Day-of-week analysis...", flush=True)
    print(f"\n  {'DOW':<12} {'N':>6} {'WR':>6} {'Sharpe':>7} {'AvgPnL':>8} {'TotalPnL':>10}")
    print("  " + "-" * 55)

    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    dow_groups = defaultdict(list)
    for t in baseline_trades:
        dow = pd.Timestamp(t['entry_time']).dayofweek
        dow_groups[dow].append(t)

    dow_stats = {}
    bad_days = []
    for d in range(5):
        dt = dow_groups.get(d, [])
        if not dt:
            dow_stats[d] = {'n': 0, 'wr': 0, 'sharpe': 0, 'avg_pnl': 0, 'total_pnl': 0}
            continue
        pnls = [t['pnl'] for t in dt]
        daily = trades_to_daily(dt)
        n_d = len(dt)
        wr = sum(1 for p in pnls if p > 0) / n_d * 100
        sh = sharpe(daily)
        avg_p = float(np.mean(pnls))
        total_p = float(sum(pnls))
        dow_stats[d] = {'n': n_d, 'wr': round(wr, 1), 'sharpe': round(sh, 3),
                        'avg_pnl': round(avg_p, 4), 'total_pnl': round(total_p, 2)}
        is_bad = avg_p < 0 or wr < 50
        marker = " <<< BAD" if is_bad else ""
        if is_bad and n_d >= 10:
            bad_days.append(d)
        print(f"  {dow_names[d]:<12} {n_d:>6} {wr:>5.1f}% {sh:>7.3f} {avg_p:>8.4f} "
              f"{total_p:>10.1f}{marker}", flush=True)

    results['dow_stats'] = {str(k): v for k, v in dow_stats.items()}
    results['bad_days'] = bad_days
    good_days = [d for d in range(5) if d not in bad_days]
    print(f"\n  Bad days: {[dow_names[d] for d in bad_days]}", flush=True)
    print(f"  Good days: {[dow_names[d] for d in good_days]}", flush=True)

    # Phase 6: Combined hour + day filter
    print("\n  Phase 6: Combined hour + DOW filter...", flush=True)
    print(f"\n  {'Config':<40} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'MaxDD':>8} {'WR':>6}")
    print("  " + "-" * 80)

    combined_configs = [
        ("Baseline", list(range(24)), list(range(5))),
        ("Good hours only", good_hours, list(range(5))),
        ("Good days only", list(range(24)), good_days),
        ("Good hours + good days", good_hours, good_days),
        (f"Best session hours", best_session_hours, list(range(5))),
        (f"Best session + good days", best_session_hours, good_days),
    ]

    combined_results = {}
    for label, hours, days in combined_configs:
        filtered = filter_by_hours_and_dow(baseline_trades, hours, days)
        st = compute_stats(filtered, label)
        combined_results[label] = {**st, 'hours': hours, 'days': days}
        print(f"  {label:<40} {st['n']:>6} {st['sharpe']:>7.3f} {st['pnl']:>10.1f} "
              f"{st['max_dd']:>8.1f} {st['wr']:>5.1f}%", flush=True)

    results['combined_sweep'] = {k: {kk: vv for kk, vv in v.items()
                                      if kk not in ('hours', 'days')}
                                  for k, v in combined_results.items()}

    best_combined = max(combined_results.items(), key=lambda x: x[1]['sharpe'])
    best_combined_label = best_combined[0]
    best_combined_hours = best_combined[1]['hours']
    best_combined_days = best_combined[1]['days']
    print(f"\n  >>> Best combined: {best_combined_label} — "
          f"Sharpe={best_combined[1]['sharpe']:.3f}", flush=True)
    results['best_combined'] = best_combined_label

    # Phase 7: K-Fold on best session filter
    print(f"\n  Phase 7: K-Fold on best filter ({best_combined_label})...", flush=True)
    fold_sharpes = []
    for fold_name, start, end in FOLDS:
        try:
            fold_b = bundle.slice(start, end)
            fold_trades = bt_l8_max(fold_b, SPREAD, UNIT_LOT, CAPS['L8_MAX'])
        except Exception:
            fold_sharpes.append(0.0)
            continue
        filtered = filter_by_hours_and_dow(fold_trades, best_combined_hours, best_combined_days)
        daily = trades_to_daily(filtered)
        fold_sharpes.append(sharpe(daily))

    pos = sum(1 for s in fold_sharpes if s > 0)
    print(f"    Folds: {[round(s, 2) for s in fold_sharpes]}, {pos}/6 positive", flush=True)
    results['kfold'] = {
        'config': best_combined_label,
        'fold_sharpes': [round(s, 3) for s in fold_sharpes],
        'positive_folds': pos,
        'mean_sharpe': round(float(np.mean(fold_sharpes)), 3),
    }

    elapsed = time.time() - t0
    print(f"\n  R168 done — {elapsed:.0f}s", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# R169: Volatility Regime Prediction Model
# ═══════════════════════════════════════════════════════════════

def compute_daily_features(h1_df):
    """Compute daily volatility features from H1 data."""
    df = h1_df.copy()
    df['ATR14'] = compute_atr(df, 14)
    df['ATR60'] = compute_atr(df, 60)
    df['atr_ratio'] = df['ATR14'] / df['ATR60'].replace(0, np.nan)

    dates = df.index.normalize()
    daily_groups = df.groupby(dates)

    daily_feats = []
    for date, group in daily_groups:
        if len(group) < 4:
            continue
        atr14_last = group['ATR14'].iloc[-1]
        atr60_last = group['ATR60'].iloc[-1]
        if pd.isna(atr14_last) or pd.isna(atr60_last) or atr60_last == 0:
            continue

        atr_ratio = atr14_last / atr60_last
        daily_range = group['High'].max() - group['Low'].min()
        avg_range_20 = df['High'].rolling(20 * 24).max() - df['Low'].rolling(20 * 24).min()
        idx = df.index.get_indexer([group.index[-1]], method='nearest')[0]
        avg_r = avg_range_20.iloc[idx] if idx < len(avg_range_20) else daily_range
        range_ratio = daily_range / avg_r if avg_r > 0 else 1.0

        cc_change = abs(group['Close'].iloc[-1] - group['Close'].iloc[0])
        cc_atr = cc_change / atr14_last if atr14_last > 0 else 0

        spike_threshold = 2 * atr14_last
        h1_ranges = group['High'] - group['Low']
        spike_count = (h1_ranges > spike_threshold).sum()

        dow = pd.Timestamp(date).dayofweek

        daily_feats.append({
            'date': date,
            'atr_ratio': round(float(atr_ratio), 4),
            'range_ratio': round(float(range_ratio), 4),
            'cc_atr': round(float(cc_atr), 4),
            'spike_count': int(spike_count),
            'dow': int(dow),
            'atr14': round(float(atr14_last), 4),
        })

    return pd.DataFrame(daily_feats)


def classify_regime(atr_ratio):
    if atr_ratio < 0.8:
        return 'LOW'
    elif atr_ratio < 1.3:
        return 'NORMAL'
    elif atr_ratio < 2.0:
        return 'HIGH'
    else:
        return 'EXTREME'


LOT_MULTIPLIERS = {'LOW': 1.5, 'NORMAL': 1.0, 'HIGH': 0.75, 'EXTREME': 0.5}
SL_ATR_MULTS = {'LOW': 3.0, 'NORMAL': 3.5, 'HIGH': 4.0, 'EXTREME': None}


def apply_regime_lot_sizing(trades, daily_feats_df, h1_df):
    """Post-process trades: scale PnL by regime-based lot multiplier."""
    feat_dates = set(daily_feats_df['date'].values)
    date_to_regime = {}
    for _, row in daily_feats_df.iterrows():
        date_to_regime[row['date']] = classify_regime(row['atr_ratio'])

    adjusted = []
    regime_counts = defaultdict(int)
    for t in trades:
        entry_date = pd.Timestamp(t['entry_time']).normalize()
        if hasattr(entry_date, 'tz') and entry_date.tzinfo is not None:
            entry_date = entry_date.tz_localize(None)
        regime = date_to_regime.get(entry_date, 'NORMAL')
        mult = LOT_MULTIPLIERS.get(regime, 1.0)
        adj_t = dict(t)
        adj_t['pnl'] = t['pnl'] * mult
        adj_t['regime'] = regime
        adjusted.append(adj_t)
        regime_counts[regime] += 1

    return adjusted, dict(regime_counts)


def apply_regime_sl_adjustment(trades, daily_feats_df, h1_df):
    """Post-process trades: clip losses by regime-based SL, skip EXTREME."""
    date_to_regime = {}
    date_to_atr = {}
    for _, row in daily_feats_df.iterrows():
        date_to_regime[row['date']] = classify_regime(row['atr_ratio'])
        date_to_atr[row['date']] = row['atr14']

    adjusted = []
    skipped_extreme = 0
    regime_counts = defaultdict(int)
    for t in trades:
        entry_date = pd.Timestamp(t['entry_time']).normalize()
        if hasattr(entry_date, 'tz') and entry_date.tzinfo is not None:
            entry_date = entry_date.tz_localize(None)
        regime = date_to_regime.get(entry_date, 'NORMAL')
        atr = date_to_atr.get(entry_date, 0)

        sl_mult = SL_ATR_MULTS.get(regime)
        if sl_mult is None:
            skipped_extreme += 1
            continue

        adj_t = dict(t)
        if atr > 0 and t['pnl'] < 0:
            max_loss = sl_mult * atr * UNIT_LOT * PV
            adj_t['pnl'] = max(t['pnl'], -max_loss)
        adj_t['regime'] = regime
        adjusted.append(adj_t)
        regime_counts[regime] += 1

    return adjusted, skipped_extreme, dict(regime_counts)


def apply_regime_combined(trades, daily_feats_df, h1_df):
    """Combine lot sizing + SL adjustment."""
    date_to_regime = {}
    date_to_atr = {}
    for _, row in daily_feats_df.iterrows():
        date_to_regime[row['date']] = classify_regime(row['atr_ratio'])
        date_to_atr[row['date']] = row['atr14']

    adjusted = []
    skipped_extreme = 0
    regime_counts = defaultdict(int)
    for t in trades:
        entry_date = pd.Timestamp(t['entry_time']).normalize()
        if hasattr(entry_date, 'tz') and entry_date.tzinfo is not None:
            entry_date = entry_date.tz_localize(None)
        regime = date_to_regime.get(entry_date, 'NORMAL')
        atr = date_to_atr.get(entry_date, 0)

        sl_mult = SL_ATR_MULTS.get(regime)
        if sl_mult is None:
            skipped_extreme += 1
            continue

        lot_mult = LOT_MULTIPLIERS.get(regime, 1.0)
        adj_t = dict(t)
        adj_t['pnl'] = t['pnl'] * lot_mult
        if atr > 0 and adj_t['pnl'] < 0:
            max_loss = sl_mult * atr * UNIT_LOT * PV * lot_mult
            adj_t['pnl'] = max(adj_t['pnl'], -max_loss)
        adj_t['regime'] = regime
        adjusted.append(adj_t)
        regime_counts[regime] += 1

    return adjusted, skipped_extreme, dict(regime_counts)


def run_r169(bundle, h1_df):
    print("\n" + "=" * 80)
    print("  R169 — Volatility Regime Prediction Model")
    print("=" * 80, flush=True)

    results = {}

    # Phase 1: Feature engineering
    print("\n  Phase 1: Computing daily volatility features...", flush=True)
    daily_feats = compute_daily_features(h1_df)
    print(f"    {len(daily_feats)} daily feature rows", flush=True)
    print(f"    ATR ratio: mean={daily_feats['atr_ratio'].mean():.3f}, "
          f"std={daily_feats['atr_ratio'].std():.3f}, "
          f"min={daily_feats['atr_ratio'].min():.3f}, "
          f"max={daily_feats['atr_ratio'].max():.3f}", flush=True)
    results['n_daily_features'] = len(daily_feats)
    results['atr_ratio_summary'] = {
        'mean': round(float(daily_feats['atr_ratio'].mean()), 4),
        'std': round(float(daily_feats['atr_ratio'].std()), 4),
        'min': round(float(daily_feats['atr_ratio'].min()), 4),
        'max': round(float(daily_feats['atr_ratio'].max()), 4),
    }

    # Phase 2: Regime classification
    print("\n  Phase 2: Regime distribution...", flush=True)
    daily_feats['regime'] = daily_feats['atr_ratio'].apply(classify_regime)
    regime_dist = daily_feats['regime'].value_counts()
    for r in ['LOW', 'NORMAL', 'HIGH', 'EXTREME']:
        cnt = regime_dist.get(r, 0)
        pct = cnt / len(daily_feats) * 100 if len(daily_feats) > 0 else 0
        print(f"    {r:<10}: {cnt:>5} days ({pct:>5.1f}%)", flush=True)
    results['regime_distribution'] = {r: int(regime_dist.get(r, 0))
                                       for r in ['LOW', 'NORMAL', 'HIGH', 'EXTREME']}

    # Phase 3: Per-regime Keltner performance
    print("\n  Phase 3: Per-regime L8_MAX performance...", flush=True)
    baseline_trades = bt_l8_max(bundle, SPREAD, UNIT_LOT, CAPS['L8_MAX'])
    base_stats = compute_stats(baseline_trades, "Baseline")
    print(f"    Baseline: n={base_stats['n']}, Sharpe={base_stats['sharpe']:.3f}, "
          f"PnL={fmt(base_stats['pnl'])}", flush=True)
    results['baseline'] = base_stats

    date_to_regime = {}
    for _, row in daily_feats.iterrows():
        date_to_regime[row['date']] = row['regime']

    regime_trades = defaultdict(list)
    for t in baseline_trades:
        entry_date = pd.Timestamp(t['entry_time']).normalize()
        if hasattr(entry_date, 'tz') and entry_date.tzinfo is not None:
            entry_date = entry_date.tz_localize(None)
        regime = date_to_regime.get(entry_date, 'NORMAL')
        regime_trades[regime].append(t)

    print(f"\n  {'Regime':<10} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'MaxDD':>8} "
          f"{'WR':>6} {'AvgPnL':>8}")
    print("  " + "-" * 60)

    regime_perf = {}
    for r in ['LOW', 'NORMAL', 'HIGH', 'EXTREME']:
        rt = regime_trades.get(r, [])
        st = compute_stats(rt, r)
        regime_perf[r] = st
        print(f"  {r:<10} {st['n']:>6} {st['sharpe']:>7.3f} {st['pnl']:>10.1f} "
              f"{st['max_dd']:>8.1f} {st['wr']:>5.1f}% {st['avg_pnl']:>8.4f}", flush=True)

    results['regime_performance'] = regime_perf

    # Phase 4: Regime-based lot sizing
    print("\n  Phase 4: Regime-based lot sizing (scale PnL)...", flush=True)
    print(f"    LOW: 1.5x | NORMAL: 1.0x | HIGH: 0.75x | EXTREME: 0.5x", flush=True)
    lot_adjusted, lot_regime_counts = apply_regime_lot_sizing(
        baseline_trades, daily_feats, h1_df)
    lot_stats = compute_stats(lot_adjusted, "LotSizing")
    print(f"    Result: n={lot_stats['n']}, Sharpe={lot_stats['sharpe']:.3f}, "
          f"PnL={fmt(lot_stats['pnl'])}", flush=True)
    print(f"    Regime counts: {dict(lot_regime_counts)}", flush=True)
    results['lot_sizing'] = {**lot_stats, 'regime_counts': lot_regime_counts}

    # Phase 5: Regime-based SL adjustment
    print("\n  Phase 5: Regime-based SL adjustment...", flush=True)
    print(f"    LOW: 3.0*ATR | NORMAL: 3.5*ATR | HIGH: 4.0*ATR | EXTREME: skip", flush=True)
    sl_adjusted, sl_skipped, sl_regime_counts = apply_regime_sl_adjustment(
        baseline_trades, daily_feats, h1_df)
    sl_stats = compute_stats(sl_adjusted, "SL_Adjust")
    print(f"    Result: n={sl_stats['n']}, Sharpe={sl_stats['sharpe']:.3f}, "
          f"PnL={fmt(sl_stats['pnl'])}, skipped_extreme={sl_skipped}", flush=True)
    print(f"    Regime counts: {dict(sl_regime_counts)}", flush=True)
    results['sl_adjustment'] = {**sl_stats, 'skipped_extreme': sl_skipped,
                                 'regime_counts': sl_regime_counts}

    # Combined lot + SL
    print("\n  Phase 5b: Combined lot sizing + SL adjustment...", flush=True)
    combo_adjusted, combo_skipped, combo_regime_counts = apply_regime_combined(
        baseline_trades, daily_feats, h1_df)
    combo_stats = compute_stats(combo_adjusted, "Combined")
    print(f"    Result: n={combo_stats['n']}, Sharpe={combo_stats['sharpe']:.3f}, "
          f"PnL={fmt(combo_stats['pnl'])}, skipped_extreme={combo_skipped}", flush=True)
    results['combined'] = {**combo_stats, 'skipped_extreme': combo_skipped,
                            'regime_counts': combo_regime_counts}

    # Find best approach
    approaches = {
        'baseline': base_stats,
        'lot_sizing': lot_stats,
        'sl_adjustment': sl_stats,
        'combined': combo_stats,
    }
    best_approach = max(approaches.items(), key=lambda x: x[1]['sharpe'])
    best_approach_name = best_approach[0]
    print(f"\n  >>> Best regime approach: {best_approach_name} "
          f"(Sharpe={best_approach[1]['sharpe']:.3f})", flush=True)
    results['best_approach'] = best_approach_name

    # Phase 6: K-Fold on best regime approach
    print(f"\n  Phase 6: K-Fold on {best_approach_name}...", flush=True)
    fold_sharpes = []
    for fold_name, start, end in FOLDS:
        try:
            fold_b = bundle.slice(start, end)
            fold_trades = bt_l8_max(fold_b, SPREAD, UNIT_LOT, CAPS['L8_MAX'])
        except Exception:
            fold_sharpes.append(0.0)
            continue
        h1_fold = h1_df[(h1_df.index >= start) & (h1_df.index < end)]
        if len(h1_fold) < 100:
            fold_sharpes.append(0.0)
            continue
        fold_feats = compute_daily_features(h1_fold)
        if len(fold_feats) < 10:
            fold_sharpes.append(0.0)
            continue

        if best_approach_name == 'lot_sizing':
            adj, _ = apply_regime_lot_sizing(fold_trades, fold_feats, h1_fold)
        elif best_approach_name == 'sl_adjustment':
            adj, _, _ = apply_regime_sl_adjustment(fold_trades, fold_feats, h1_fold)
        elif best_approach_name == 'combined':
            adj, _, _ = apply_regime_combined(fold_trades, fold_feats, h1_fold)
        else:
            adj = fold_trades

        daily = trades_to_daily(adj)
        fold_sharpes.append(sharpe(daily))

    pos = sum(1 for s in fold_sharpes if s > 0)
    print(f"    Folds: {[round(s, 2) for s in fold_sharpes]}, {pos}/6 positive", flush=True)
    results['kfold'] = {
        'approach': best_approach_name,
        'fold_sharpes': [round(s, 3) for s in fold_sharpes],
        'positive_folds': pos,
        'mean_sharpe': round(float(np.mean(fold_sharpes)), 3),
    }

    # K-Fold baseline
    print(f"  K-Fold baseline (no regime adjustment)...", flush=True)
    fold_sharpes_base = []
    for fold_name, start, end in FOLDS:
        try:
            fold_b = bundle.slice(start, end)
            fold_trades = bt_l8_max(fold_b, SPREAD, UNIT_LOT, CAPS['L8_MAX'])
        except Exception:
            fold_sharpes_base.append(0.0)
            continue
        daily = trades_to_daily(fold_trades)
        fold_sharpes_base.append(sharpe(daily))
    pos_base = sum(1 for s in fold_sharpes_base if s > 0)
    print(f"    Baseline folds: {[round(s, 2) for s in fold_sharpes_base]}, "
          f"{pos_base}/6 positive", flush=True)
    results['kfold_baseline'] = {
        'fold_sharpes': [round(s, 3) for s in fold_sharpes_base],
        'positive_folds': pos_base,
        'mean_sharpe': round(float(np.mean(fold_sharpes_base)), 3),
    }

    elapsed = time.time() - t0
    print(f"\n  R169 done — {elapsed:.0f}s", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  R167-R169 — Batch 3: MTF Confirmation, Session, Volatility Regime")
    print("  R167: H4 trend filter | R168: Session optimization | R169: Vol regime")
    print("=" * 80, flush=True)

    print("\n  Loading data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    bundle = DataBundle.load_custom()
    print(f"  H1: {len(h1_df)} bars, M15: {len(m15_raw)} bars", flush=True)

    all_results = {
        'experiment': 'R167-R169 Batch 3',
        'description': 'MTF Confirmation + Session Optimization + Volatility Regime',
    }

    # R167
    r167_results = run_r167(bundle, h1_df)
    all_results['r167_mtf_confirmation'] = r167_results
    with open(OUTPUT_DIR / "r167_results.json", 'w') as f:
        json.dump({'experiment': 'R167', **r167_results}, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r167_results.json", flush=True)

    # R168
    r168_results = run_r168(bundle, h1_df)
    all_results['r168_session_optimization'] = r168_results
    with open(OUTPUT_DIR / "r168_results.json", 'w') as f:
        json.dump({'experiment': 'R168', **r168_results}, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r168_results.json", flush=True)

    # R169
    r169_results = run_r169(bundle, h1_df)
    all_results['r169_volatility_regime'] = r169_results
    with open(OUTPUT_DIR / "r169_results.json", 'w') as f:
        json.dump({'experiment': 'R169', **r169_results}, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r169_results.json", flush=True)

    # Combined JSON
    elapsed = time.time() - t0
    all_results['elapsed_s'] = round(elapsed, 1)
    all_results['elapsed_min'] = round(elapsed / 60, 1)

    with open(OUTPUT_DIR / "r167_r169_combined.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 80}")
    print(f"  R167-R169 BATCH 3 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"{'=' * 80}", flush=True)

    # Summary
    print("\n  === SUMMARY ===", flush=True)
    r167_best = r167_results.get('best_filter', '?')
    r167_sh = r167_results.get('filters', {}).get(r167_best, {}).get('sharpe', 0)
    r167_base_sh = r167_results.get('baseline', {}).get('sharpe', 0)
    r167_kf = r167_results.get('kfold', {}).get('positive_folds', 0)
    print(f"  R167 MTF: best={r167_best}, Sharpe {r167_base_sh:.3f} -> {r167_sh:.3f}, "
          f"K-Fold {r167_kf}/6", flush=True)

    r168_best = r168_results.get('best_combined', '?')
    r168_best_sh = 0
    for v in r168_results.get('combined_sweep', {}).values():
        if v.get('sharpe', 0) > r168_best_sh:
            r168_best_sh = v['sharpe']
    r168_base_sh = r168_results.get('baseline', {}).get('sharpe', 0)
    r168_kf = r168_results.get('kfold', {}).get('positive_folds', 0)
    print(f"  R168 Session: best={r168_best}, Sharpe {r168_base_sh:.3f} -> {r168_best_sh:.3f}, "
          f"K-Fold {r168_kf}/6", flush=True)

    r169_best = r169_results.get('best_approach', '?')
    r169_sh = r169_results.get(r169_best, r169_results.get('lot_sizing', {})).get('sharpe', 0)
    r169_base_sh = r169_results.get('baseline', {}).get('sharpe', 0)
    r169_kf = r169_results.get('kfold', {}).get('positive_folds', 0)
    print(f"  R169 Regime: best={r169_best}, Sharpe {r169_base_sh:.3f} -> {r169_sh:.3f}, "
          f"K-Fold {r169_kf}/6", flush=True)


if __name__ == "__main__":
    main()
