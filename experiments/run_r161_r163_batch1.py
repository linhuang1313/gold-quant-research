#!/usr/bin/env python3
"""
R161-R163 Batch 1 — Adaptive Lot + Combined Improvements + Donchian Confirmation
==================================================================================
Three sequential experiments:

R161: Adaptive Lot Sizing K-Fold Deep Validation
  R160 found ATR > 2x mean → 50% lot improved Keltner Sharpe +15%.
  Full-sample reproduce, K-Fold on L8_MAX alone, portfolio-level, portfolio K-Fold.

R162: All Improvements Combined (RuleB + CB + Adaptive Lot)
  Test the combination of all three validated improvements.
  4 variants × full sample + K-Fold on Baseline vs ALL THREE.

R163: Donchian50 Pre-deployment Confirmation
  Confirm Donchian channel breakout (channel=60, sl_atr=4, tp_atr=3, max_hold=20)
  still works; test as 5th strategy in portfolio.
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import (
    DataBundle, run_variant, LIVE_PARITY_KWARGS,
    load_m15, load_h1_aligned, H1_CSV_PATH,
)

OUTPUT_DIR = Path("results/r161_r163_batch1")
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

CUSUM_SIGMA = 3.0
EXTREME_WINDOW = 12
SKIP_BARS = 8
CB_STREAK = 3
CB_PAUSE_HOURS = 1
ATR_MULT_THRESHOLD = 2.0
REDUCED_LOT_RATIO = 0.5

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


def _normalize_ts(ts):
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    return t


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


# ═══════════════════════════════════════════════════════════════
# Strategy engines
# ═══════════════════════════════════════════════════════════════

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


def bt_donchian(h1_df, spread, lot, maxloss_cap=0,
                channel=60, sl_atr=4.0, tp_atr=3.0, max_hold=20):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(channel, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, 0.14, 0.025, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        upper = max(h[i-j] for j in range(1, channel+1))
        lower = min(lo[i-j] for j in range(1, channel+1))
        if c[i] > upper:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < lower:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


# ═══════════════════════════════════════════════════════════════
# Portfolio / filter helpers
# ═══════════════════════════════════════════════════════════════

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


def build_extreme_mask(h1_df, cusum_sigma=3.0, extreme_window=12):
    n = len(h1_df)
    extreme = np.zeros(n, dtype=bool)
    atr = compute_atr(h1_df).values
    atr_clean = np.nan_to_num(atr, nan=0.0)
    atr_mean = pd.Series(atr_clean).rolling(60, min_periods=20).mean().values
    atr_std = pd.Series(atr_clean).rolling(60, min_periods=20).std().values
    atr_std = np.maximum(atr_std, 1e-6)
    cusum_trigger = atr_clean > (atr_mean + cusum_sigma * atr_std)
    for i in range(n):
        if cusum_trigger[i]:
            end_i = min(i + extreme_window, n)
            extreme[i:end_i] = True
    return extreme


def apply_rule_b(trades, h1_df, extreme_mask, skip_bars=8):
    if not trades:
        return [], 0
    times_idx = h1_df.index
    n_bars = len(times_idx)
    idx_is_tz_aware = times_idx.tz is not None

    def _find_bar(ts):
        ts = pd.Timestamp(ts)
        if idx_is_tz_aware and ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        elif not idx_is_tz_aware and ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        idx = times_idx.searchsorted(ts)
        return min(idx, n_bars - 1) if idx < n_bars else n_bars - 1

    protected = []; skipped = 0
    for t in trades:
        entry_bar = _find_bar(t['entry_time'])
        skip_end = entry_bar
        for j in range(max(0, entry_bar - skip_bars), entry_bar):
            if j < n_bars and extreme_mask[j]:
                skip_end = j + skip_bars
                break
        if entry_bar < skip_end:
            skipped += 1
            continue
        protected.append(t)
    return protected, skipped


def apply_circuit_breaker(portfolio_trades, streak_thresh=3, pause_hours=1):
    taken = []; skipped = 0
    consec_losses = 0; pause_until = None
    for trade in portfolio_trades:
        entry_t = trade['entry_time']
        exit_t = trade['exit_time']
        if pause_until is not None and entry_t < pause_until:
            skipped += 1
            continue
        if pause_until is not None and entry_t >= pause_until:
            pause_until = None
            consec_losses = 0
        taken.append(trade)
        if trade['pnl'] < 0:
            consec_losses += 1
            if consec_losses >= streak_thresh:
                pause_until = exit_t + timedelta(hours=pause_hours)
        else:
            consec_losses = 0
    return taken, skipped


# ═══════════════════════════════════════════════════════════════
# Adaptive lot helper
# ═══════════════════════════════════════════════════════════════

def apply_adaptive_lot_to_l8(l8_trades, h1_df,
                             atr_mult_threshold=2.0, reduced_ratio=0.5):
    """Post-process L8_MAX trades: scale PnL by reduced_ratio when entry ATR
    exceeds atr_mult_threshold * rolling mean ATR."""
    atr_series = compute_atr(h1_df)
    atr_mean = atr_series.rolling(60*24, min_periods=100).mean()
    times_idx = h1_df.index
    idx_is_tz_aware = times_idx.tz is not None

    adjusted = []; n_reduced = 0
    for t in l8_trades:
        ts = pd.Timestamp(t['entry_time'])
        if idx_is_tz_aware and ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        elif not idx_is_tz_aware and ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        idx = min(int(times_idx.searchsorted(ts)), len(times_idx)-1)
        current_atr = atr_series.iloc[idx] if idx < len(atr_series) else 0
        mean_atr = atr_mean.iloc[idx] if idx < len(atr_mean) else 0
        if pd.notna(mean_atr) and mean_atr > 0 and current_atr > atr_mult_threshold * mean_atr:
            adj_t = dict(t)
            adj_t['pnl'] = t['pnl'] * reduced_ratio
            adjusted.append(adj_t)
            n_reduced += 1
        else:
            adjusted.append(t)
    return adjusted, n_reduced


# ═══════════════════════════════════════════════════════════════
# Strategy runners
# ═══════════════════════════════════════════════════════════════

def run_all_4_strategies(h1_df, l8_bundle):
    strat = {}
    strat['L8_MAX'] = bt_l8_max(l8_bundle, SPREAD, UNIT_LOT, CAPS['L8_MAX'])
    strat['PSAR'] = bt_psar(h1_df, SPREAD, UNIT_LOT, CAPS['PSAR'])
    strat['TSMOM'] = bt_tsmom(h1_df, SPREAD, UNIT_LOT, CAPS['TSMOM'])
    strat['SESS_BO'] = bt_sess_bo(h1_df, SPREAD, UNIT_LOT, CAPS['SESS_BO'])
    return strat


def print_stats_row(st, prefix="    "):
    print(f"{prefix}N={st['n']}, Sharpe={st['sharpe']:.3f}, PnL=${st['pnl']:.1f}, "
          f"MaxDD=${st['max_dd']:.1f}, WR={st['wr']:.1f}%", flush=True)


# ═══════════════════════════════════════════════════════════════
# R161: Adaptive Lot Sizing K-Fold Deep Validation
# ═══════════════════════════════════════════════════════════════

def run_r161(h1_df, l8_bundle):
    print("\n" + "=" * 80)
    print("  R161 — Adaptive Lot Sizing K-Fold Deep Validation")
    print("  R160 found: ATR > 2x mean → 50% lot improved Keltner Sharpe +15%")
    print("=" * 80, flush=True)

    results = {}

    # Phase 1: Reproduce full-sample
    print("\n  Phase 1: Reproduce full-sample result", flush=True)
    l8_base = bt_l8_max(l8_bundle, SPREAD, UNIT_LOT, CAPS['L8_MAX'])
    l8_adaptive, n_reduced = apply_adaptive_lot_to_l8(l8_base, h1_df)

    st_base = compute_stats(l8_base, "L8_Base")
    st_adapt = compute_stats(l8_adaptive, "L8_Adaptive")

    print(f"    L8 Baseline:  ", end=""); print_stats_row(st_base, "")
    print(f"    L8 Adaptive:  ", end=""); print_stats_row(st_adapt, "")
    print(f"    Reduced trades: {n_reduced}/{len(l8_base)} ({n_reduced/max(len(l8_base),1)*100:.1f}%)")
    delta_sharpe = st_adapt['sharpe'] - st_base['sharpe']
    print(f"    Sharpe delta: {delta_sharpe:+.3f} ({delta_sharpe/max(abs(st_base['sharpe']),0.001)*100:+.1f}%)", flush=True)

    results['phase1'] = {
        'baseline': st_base, 'adaptive': st_adapt,
        'n_reduced': n_reduced, 'delta_sharpe': round(delta_sharpe, 4),
    }

    # Phase 2: K-Fold on L8_MAX alone (adaptive vs baseline)
    print("\n  Phase 2: K-Fold on L8_MAX (adaptive vs baseline)", flush=True)
    kfold_base = []; kfold_adapt = []

    print(f"    {'Fold':<8} {'Base_S':>8} {'Adapt_S':>8} {'Delta':>8} {'N_red':>6}")
    print("    " + "-" * 45)

    for fold_name, start, end in FOLDS:
        h1_fold = h1_df[(h1_df.index >= start) & (h1_df.index < end)]
        if len(h1_fold) < 100:
            kfold_base.append(0.0); kfold_adapt.append(0.0)
            print(f"    {fold_name:<8} {'skip':>8} {'skip':>8}", flush=True)
            continue
        try:
            fold_b = l8_bundle.slice(start, end)
            l8_fold = bt_l8_max(fold_b, SPREAD, UNIT_LOT, CAPS['L8_MAX'])
        except Exception:
            kfold_base.append(0.0); kfold_adapt.append(0.0)
            continue
        l8_fold_adapt, nr = apply_adaptive_lot_to_l8(l8_fold, h1_fold)
        d_base = trades_to_daily(l8_fold)
        d_adapt = trades_to_daily(l8_fold_adapt)
        sb = sharpe(d_base); sa = sharpe(d_adapt)
        kfold_base.append(sb); kfold_adapt.append(sa)
        print(f"    {fold_name:<8} {sb:>8.3f} {sa:>8.3f} {sa-sb:>+8.3f} {nr:>6}", flush=True)

    adapt_wins = sum(1 for a, b in zip(kfold_adapt, kfold_base) if a > b)
    print(f"    Adaptive wins: {adapt_wins}/6 folds")
    print(f"    Base  mean: {np.mean(kfold_base):.3f}")
    print(f"    Adapt mean: {np.mean(kfold_adapt):.3f}", flush=True)

    results['phase2_kfold_l8'] = {
        'base_sharpes': [round(s, 3) for s in kfold_base],
        'adapt_sharpes': [round(s, 3) for s in kfold_adapt],
        'adapt_wins': adapt_wins,
        'base_mean': round(float(np.mean(kfold_base)), 3),
        'adapt_mean': round(float(np.mean(kfold_adapt)), 3),
    }

    # Phase 3: Portfolio-level test
    print("\n  Phase 3: Portfolio-level (4 strats, adaptive on L8_MAX)", flush=True)
    strat_trades = run_all_4_strategies(h1_df, l8_bundle)
    l8_adaptive_full, nr_full = apply_adaptive_lot_to_l8(strat_trades['L8_MAX'], h1_df)

    port_base = merge_portfolio_trades(strat_trades)
    strat_adaptive = dict(strat_trades)
    strat_adaptive['L8_MAX'] = l8_adaptive_full
    port_adapt = merge_portfolio_trades(strat_adaptive)

    st_port_base = compute_stats(port_base, "Portfolio_Base")
    st_port_adapt = compute_stats(port_adapt, "Portfolio_Adaptive")

    print(f"    Portfolio Base:     ", end=""); print_stats_row(st_port_base, "")
    print(f"    Portfolio Adaptive: ", end=""); print_stats_row(st_port_adapt, "")
    delta_p = st_port_adapt['sharpe'] - st_port_base['sharpe']
    print(f"    Portfolio Sharpe delta: {delta_p:+.3f}", flush=True)

    results['phase3_portfolio'] = {
        'baseline': st_port_base, 'adaptive': st_port_adapt,
        'delta_sharpe': round(delta_p, 4), 'n_reduced_l8': nr_full,
    }

    # Phase 4: K-Fold on portfolio level
    print("\n  Phase 4: K-Fold on portfolio (adaptive vs baseline)", flush=True)
    kfold_port_base = []; kfold_port_adapt = []

    print(f"    {'Fold':<8} {'Base_S':>8} {'Adapt_S':>8} {'Delta':>8}")
    print("    " + "-" * 40)

    for fold_name, start, end in FOLDS:
        h1_fold = h1_df[(h1_df.index >= start) & (h1_df.index < end)]
        if len(h1_fold) < 100:
            kfold_port_base.append(0.0); kfold_port_adapt.append(0.0)
            print(f"    {fold_name:<8} {'skip':>8} {'skip':>8}", flush=True)
            continue
        fold_strat = {}
        fold_strat['PSAR'] = bt_psar(h1_fold, SPREAD, UNIT_LOT, CAPS['PSAR'])
        fold_strat['TSMOM'] = bt_tsmom(h1_fold, SPREAD, UNIT_LOT, CAPS['TSMOM'])
        fold_strat['SESS_BO'] = bt_sess_bo(h1_fold, SPREAD, UNIT_LOT, CAPS['SESS_BO'])
        try:
            l8_fold = l8_bundle.slice(start, end)
            fold_strat['L8_MAX'] = bt_l8_max(l8_fold, SPREAD, UNIT_LOT, CAPS['L8_MAX'])
        except Exception:
            kfold_port_base.append(0.0); kfold_port_adapt.append(0.0)
            continue

        port_b = merge_portfolio_trades(fold_strat)
        l8_fold_adapt, _ = apply_adaptive_lot_to_l8(fold_strat['L8_MAX'], h1_fold)
        fold_strat_a = dict(fold_strat); fold_strat_a['L8_MAX'] = l8_fold_adapt
        port_a = merge_portfolio_trades(fold_strat_a)

        sb = sharpe(trades_to_daily(port_b)); sa = sharpe(trades_to_daily(port_a))
        kfold_port_base.append(sb); kfold_port_adapt.append(sa)
        print(f"    {fold_name:<8} {sb:>8.3f} {sa:>8.3f} {sa-sb:>+8.3f}", flush=True)

    adapt_port_wins = sum(1 for a, b in zip(kfold_port_adapt, kfold_port_base) if a > b)
    print(f"    Adaptive wins: {adapt_port_wins}/6 folds (portfolio)")
    print(f"    Base  mean: {np.mean(kfold_port_base):.3f}")
    print(f"    Adapt mean: {np.mean(kfold_port_adapt):.3f}", flush=True)

    results['phase4_kfold_portfolio'] = {
        'base_sharpes': [round(s, 3) for s in kfold_port_base],
        'adapt_sharpes': [round(s, 3) for s in kfold_port_adapt],
        'adapt_wins': adapt_port_wins,
        'base_mean': round(float(np.mean(kfold_port_base)), 3),
        'adapt_mean': round(float(np.mean(kfold_port_adapt)), 3),
    }

    verdict = "VALIDATED" if adapt_wins >= 4 and adapt_port_wins >= 4 else "MIXED"
    if adapt_wins < 3:
        verdict = "WEAK"
    results['verdict'] = verdict
    print(f"\n  R161 Verdict: {verdict}", flush=True)

    return results


# ═══════════════════════════════════════════════════════════════
# R162: All Improvements Combined (RuleB + CB + Adaptive Lot)
# ═══════════════════════════════════════════════════════════════

def run_r162(h1_df, l8_bundle):
    print("\n" + "=" * 80)
    print("  R162 — All Improvements Combined (RuleB + CB + Adaptive Lot)")
    print("  Processing: run → adaptive lot on L8 → merge → Rule B → CB")
    print("=" * 80, flush=True)

    results = {}

    # Run all strategies once
    print("\n  Running all 4 strategies...", flush=True)
    strat_trades = run_all_4_strategies(h1_df, l8_bundle)
    for s, tr in strat_trades.items():
        print(f"    {s}: {len(tr)} trades", flush=True)

    extreme_mask = build_extreme_mask(h1_df, CUSUM_SIGMA, EXTREME_WINDOW)
    pct_extreme = extreme_mask.sum() / len(h1_df) * 100
    print(f"    Extreme mask: {extreme_mask.sum()}/{len(h1_df)} ({pct_extreme:.1f}%)", flush=True)

    # Prepare L8 adaptive trades
    l8_adaptive, n_reduced = apply_adaptive_lot_to_l8(strat_trades['L8_MAX'], h1_df)
    print(f"    L8 adaptive: {n_reduced}/{len(strat_trades['L8_MAX'])} reduced", flush=True)

    # Build 4 variant portfolios
    def _build_portfolio(use_adaptive_l8=False):
        sd = dict(strat_trades)
        if use_adaptive_l8:
            sd['L8_MAX'] = l8_adaptive
        return merge_portfolio_trades(sd)

    VARIANTS = {
        'A_Baseline':     {'adaptive': False, 'rule_b': False, 'cb': False},
        'B_RuleB_CB':     {'adaptive': False, 'rule_b': True,  'cb': True},
        'C_AdaptiveLot':  {'adaptive': True,  'rule_b': False, 'cb': False},
        'D_ALL_THREE':    {'adaptive': True,  'rule_b': True,  'cb': True},
    }

    print(f"\n  Full-sample comparison:")
    print(f"    {'Variant':<20} {'N':>6} {'Sharpe':>8} {'PnL':>10} "
          f"{'MaxDD':>8} {'WR':>6} {'RB_skip':>8} {'CB_skip':>8}")
    print("    " + "-" * 80)

    phase1 = {}
    for vname, cfg in VARIANTS.items():
        port = _build_portfolio(use_adaptive_l8=cfg['adaptive'])
        rb_skip = 0; cb_skip = 0

        if cfg['rule_b']:
            port, rb_skip = apply_rule_b(port, h1_df, extreme_mask, SKIP_BARS)
        if cfg['cb']:
            port, cb_skip = apply_circuit_breaker(port, CB_STREAK, CB_PAUSE_HOURS)

        st = compute_stats(port, vname)
        st['rb_skipped'] = rb_skip; st['cb_skipped'] = cb_skip
        phase1[vname] = st
        print(f"    {vname:<20} {st['n']:>6} {st['sharpe']:>8.3f} {st['pnl']:>10.1f} "
              f"{st['max_dd']:>8.1f} {st['wr']:>5.1f}% {rb_skip:>8} {cb_skip:>8}", flush=True)

    # Interaction analysis
    sa = phase1['A_Baseline']['sharpe']
    sb = phase1['B_RuleB_CB']['sharpe']
    sc = phase1['C_AdaptiveLot']['sharpe']
    sd = phase1['D_ALL_THREE']['sharpe']

    print(f"\n  Interaction analysis:")
    print(f"    RuleB+CB standalone:    {sb - sa:+.3f}")
    print(f"    AdaptiveLot standalone: {sc - sa:+.3f}")
    print(f"    ALL THREE combined:     {sd - sa:+.3f}")
    synergy = sd - sa - (sb - sa) - (sc - sa)
    print(f"    Synergy:                {synergy:+.3f}")
    print(f"    Combined > best single? {sd >= max(sb, sc)}", flush=True)

    results['phase1'] = phase1
    results['interaction'] = {
        'rb_cb_delta': round(sb - sa, 4),
        'adaptive_delta': round(sc - sa, 4),
        'all_three_delta': round(sd - sa, 4),
        'synergy': round(synergy, 4),
        'combined_best': sd >= max(sb, sc),
    }

    # K-Fold: Baseline vs ALL THREE
    print(f"\n  K-Fold: Baseline vs ALL THREE", flush=True)
    kfold_results = {}

    for vname in ['A_Baseline', 'D_ALL_THREE']:
        cfg = VARIANTS[vname]
        fold_sharpes = []
        print(f"\n    K-Fold: {vname}", flush=True)

        for fold_name, start, end in FOLDS:
            h1_fold = h1_df[(h1_df.index >= start) & (h1_df.index < end)]
            if len(h1_fold) < 100:
                fold_sharpes.append(0.0)
                print(f"      {fold_name}: skipped (too few bars)", flush=True)
                continue

            fold_strat = {}
            fold_strat['PSAR'] = bt_psar(h1_fold, SPREAD, UNIT_LOT, CAPS['PSAR'])
            fold_strat['TSMOM'] = bt_tsmom(h1_fold, SPREAD, UNIT_LOT, CAPS['TSMOM'])
            fold_strat['SESS_BO'] = bt_sess_bo(h1_fold, SPREAD, UNIT_LOT, CAPS['SESS_BO'])
            try:
                l8_fold = l8_bundle.slice(start, end)
                fold_strat['L8_MAX'] = bt_l8_max(l8_fold, SPREAD, UNIT_LOT, CAPS['L8_MAX'])
            except Exception:
                fold_sharpes.append(0.0)
                continue

            if cfg['adaptive']:
                fold_strat['L8_MAX'], _ = apply_adaptive_lot_to_l8(
                    fold_strat['L8_MAX'], h1_fold)

            fold_port = merge_portfolio_trades(fold_strat)
            if not fold_port:
                fold_sharpes.append(0.0)
                continue

            if cfg['rule_b']:
                fold_mask = build_extreme_mask(h1_fold, CUSUM_SIGMA, EXTREME_WINDOW)
                fold_port, _ = apply_rule_b(fold_port, h1_fold, fold_mask, SKIP_BARS)
            if cfg['cb']:
                fold_port, _ = apply_circuit_breaker(fold_port, CB_STREAK, CB_PAUSE_HOURS)

            daily = trades_to_daily(fold_port)
            s = sharpe(daily)
            fold_sharpes.append(s)
            print(f"      {fold_name}: Sharpe={s:.3f}, N={len(fold_port)}", flush=True)

        pos = sum(1 for s in fold_sharpes if s > 0)
        kfold_results[vname] = {
            'fold_sharpes': [round(s, 3) for s in fold_sharpes],
            'positive_folds': pos,
            'mean_sharpe': round(float(np.mean(fold_sharpes)), 3),
        }
        print(f"    {vname}: {pos}/6 positive, mean={np.mean(fold_sharpes):.3f}", flush=True)

    d_wins = sum(1 for a_s, d_s in zip(
        kfold_results['A_Baseline']['fold_sharpes'],
        kfold_results['D_ALL_THREE']['fold_sharpes']) if d_s > a_s)
    print(f"\n    ALL THREE wins over Baseline in {d_wins}/6 folds", flush=True)
    kfold_results['d_wins_over_a'] = d_wins

    results['kfold'] = kfold_results

    verdict = "DEPLOY"
    if sd < sa:
        verdict = "REJECT"
    elif d_wins < 4:
        verdict = "CAUTIOUS"
    results['verdict'] = verdict
    print(f"\n  R162 Verdict: {verdict}", flush=True)

    return results


# ═══════════════════════════════════════════════════════════════
# R163: Donchian50 Pre-deployment Confirmation
# ═══════════════════════════════════════════════════════════════

def run_r163(h1_df, l8_bundle):
    print("\n" + "=" * 80)
    print("  R163 — Donchian50 Pre-deployment Confirmation")
    print("  Channel=60, SL=4 ATR, TP=3 ATR, MaxHold=20, Lot=0.03")
    print("=" * 80, flush=True)

    DONCHIAN_LOT = 0.03
    results = {}

    # Phase 1: Donchian standalone
    print("\n  Phase 1: Donchian standalone", flush=True)
    don_trades = bt_donchian(h1_df, SPREAD, UNIT_LOT, maxloss_cap=0,
                             channel=60, sl_atr=4.0, tp_atr=3.0, max_hold=20)
    st_don = compute_stats(don_trades, "Donchian60")
    print(f"    Donchian: ", end=""); print_stats_row(st_don, "")

    buys = [t for t in don_trades if t['dir'] == 'BUY']
    sells = [t for t in don_trades if t['dir'] == 'SELL']
    print(f"    BUY: {len(buys)}, SELL: {len(sells)}")
    reasons = {}
    for t in don_trades:
        reasons[t['reason']] = reasons.get(t['reason'], 0) + 1
    print(f"    Exit reasons: {reasons}", flush=True)

    results['phase1_standalone'] = {**st_don, 'n_buy': len(buys), 'n_sell': len(sells),
                                    'exit_reasons': reasons}

    # Phase 2: 4-strategy vs 5-strategy portfolio
    print("\n  Phase 2: 4-strat vs 5-strat portfolio", flush=True)
    strat_trades = run_all_4_strategies(h1_df, l8_bundle)
    for s, tr in strat_trades.items():
        print(f"    {s}: {len(tr)} trades", flush=True)

    port_4 = merge_portfolio_trades(strat_trades)
    st_port4 = compute_stats(port_4, "4Strat_Portfolio")
    print(f"    4-Strat: ", end=""); print_stats_row(st_port4, "")

    # 5-strategy: add Donchian
    r89_lots_5 = dict(R89_LOTS)
    r89_lots_5['DONCH'] = DONCHIAN_LOT
    strat_trades_5 = dict(strat_trades)
    strat_trades_5['DONCH'] = don_trades
    port_5 = merge_portfolio_trades(strat_trades_5, lot_scale=r89_lots_5)
    st_port5 = compute_stats(port_5, "5Strat_Portfolio")
    print(f"    5-Strat: ", end=""); print_stats_row(st_port5, "")

    delta_p = st_port5['sharpe'] - st_port4['sharpe']
    print(f"    Sharpe delta: {delta_p:+.3f}", flush=True)

    results['phase2_portfolio'] = {
        '4strat': st_port4, '5strat': st_port5,
        'delta_sharpe': round(delta_p, 4),
    }

    # Phase 3: K-Fold on 5-strategy portfolio
    print("\n  Phase 3: K-Fold on 5-strategy portfolio", flush=True)
    kfold_4 = []; kfold_5 = []

    print(f"    {'Fold':<8} {'4S_Sharpe':>10} {'5S_Sharpe':>10} {'Delta':>8}")
    print("    " + "-" * 45)

    for fold_name, start, end in FOLDS:
        h1_fold = h1_df[(h1_df.index >= start) & (h1_df.index < end)]
        if len(h1_fold) < 100:
            kfold_4.append(0.0); kfold_5.append(0.0)
            print(f"    {fold_name:<8} {'skip':>10} {'skip':>10}", flush=True)
            continue

        fold_strat = {}
        fold_strat['PSAR'] = bt_psar(h1_fold, SPREAD, UNIT_LOT, CAPS['PSAR'])
        fold_strat['TSMOM'] = bt_tsmom(h1_fold, SPREAD, UNIT_LOT, CAPS['TSMOM'])
        fold_strat['SESS_BO'] = bt_sess_bo(h1_fold, SPREAD, UNIT_LOT, CAPS['SESS_BO'])
        try:
            l8_fold = l8_bundle.slice(start, end)
            fold_strat['L8_MAX'] = bt_l8_max(l8_fold, SPREAD, UNIT_LOT, CAPS['L8_MAX'])
        except Exception:
            kfold_4.append(0.0); kfold_5.append(0.0)
            continue

        don_fold = bt_donchian(h1_fold, SPREAD, UNIT_LOT, maxloss_cap=0,
                               channel=60, sl_atr=4.0, tp_atr=3.0, max_hold=20)

        port_4f = merge_portfolio_trades(fold_strat)
        fold_strat_5 = dict(fold_strat)
        fold_strat_5['DONCH'] = don_fold
        port_5f = merge_portfolio_trades(fold_strat_5, lot_scale=r89_lots_5)

        s4 = sharpe(trades_to_daily(port_4f))
        s5 = sharpe(trades_to_daily(port_5f))
        kfold_4.append(s4); kfold_5.append(s5)
        print(f"    {fold_name:<8} {s4:>10.3f} {s5:>10.3f} {s5-s4:>+8.3f}", flush=True)

    wins_5 = sum(1 for a, b in zip(kfold_5, kfold_4) if a > b)
    pos_5 = sum(1 for s in kfold_5 if s > 0)
    print(f"    5-strat wins: {wins_5}/6 folds")
    print(f"    5-strat positive: {pos_5}/6 folds")
    print(f"    4-strat mean: {np.mean(kfold_4):.3f}, 5-strat mean: {np.mean(kfold_5):.3f}", flush=True)

    results['phase3_kfold'] = {
        '4strat_sharpes': [round(s, 3) for s in kfold_4],
        '5strat_sharpes': [round(s, 3) for s in kfold_5],
        '5strat_wins': wins_5,
        '5strat_positive': pos_5,
        '4strat_mean': round(float(np.mean(kfold_4)), 3),
        '5strat_mean': round(float(np.mean(kfold_5)), 3),
    }

    # Phase 4: Correlation between Donchian and existing strategies
    print("\n  Phase 4: Strategy correlation analysis", flush=True)
    strat_daily = {}
    for sname, trades in strat_trades.items():
        strat_daily[sname] = trades_to_daily(
            [{'exit_time': _normalize_ts(t['exit_time']), 'pnl': t['pnl']} for t in trades]
        )
    strat_daily['DONCH'] = trades_to_daily(
        [{'exit_time': _normalize_ts(t['exit_time']), 'pnl': t['pnl']} for t in don_trades]
    )

    all_dates = set()
    strat_by_date = {}
    for sname in strat_daily:
        strat_by_date[sname] = {}

    for sname, trades_list in [('L8_MAX', strat_trades['L8_MAX']),
                                ('PSAR', strat_trades['PSAR']),
                                ('TSMOM', strat_trades['TSMOM']),
                                ('SESS_BO', strat_trades['SESS_BO']),
                                ('DONCH', don_trades)]:
        for t in trades_list:
            d = _normalize_ts(t['exit_time']).normalize()
            if hasattr(d, 'tz') and d.tzinfo is not None:
                d = d.tz_localize(None)
            all_dates.add(d)
            strat_by_date[sname][d] = strat_by_date[sname].get(d, 0) + t['pnl']

    all_dates = sorted(all_dates)
    if len(all_dates) > 10:
        df_corr = pd.DataFrame(index=all_dates)
        for sname in ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO', 'DONCH']:
            df_corr[sname] = [strat_by_date[sname].get(d, 0.0) for d in all_dates]

        corr_matrix = df_corr.corr()
        don_corrs = {}
        for sname in ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']:
            don_corrs[sname] = round(float(corr_matrix.loc['DONCH', sname]), 4)
        print(f"    Donchian daily PnL correlations:")
        for sname, corr_val in don_corrs.items():
            print(f"      DONCH vs {sname}: {corr_val:+.4f}", flush=True)

        avg_corr = np.mean(list(don_corrs.values()))
        print(f"    Average correlation: {avg_corr:+.4f}")
        diversification = "GOOD" if abs(avg_corr) < 0.15 else ("OK" if abs(avg_corr) < 0.30 else "POOR")
        print(f"    Diversification: {diversification}", flush=True)

        results['phase4_correlation'] = {
            'don_vs_strats': don_corrs,
            'avg_correlation': round(avg_corr, 4),
            'diversification': diversification,
        }
    else:
        results['phase4_correlation'] = {'error': 'insufficient dates'}

    verdict = "ADD_DONCHIAN" if wins_5 >= 4 and pos_5 >= 5 else "CAUTIOUS"
    if wins_5 < 3 or pos_5 < 4:
        verdict = "SKIP"
    results['verdict'] = verdict
    print(f"\n  R163 Verdict: {verdict}", flush=True)

    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  R161-R163 Batch 1")
    print("  R161: Adaptive Lot K-Fold | R162: Combined Improvements | R163: Donchian")
    print("=" * 80, flush=True)

    # Load data
    print("\n  Loading data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    l8_bundle = DataBundle.load_custom()
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})")
    print(f"  M15: {len(m15_raw)} bars", flush=True)

    # R161
    t1 = time.time()
    r161 = run_r161(h1_df, l8_bundle)
    t161 = time.time() - t1
    print(f"\n  R161 completed in {t161:.0f}s ({t161/60:.1f}min)", flush=True)

    # R162
    t2 = time.time()
    r162 = run_r162(h1_df, l8_bundle)
    t162 = time.time() - t2
    print(f"\n  R162 completed in {t162:.0f}s ({t162/60:.1f}min)", flush=True)

    # R163
    t3 = time.time()
    r163 = run_r163(h1_df, l8_bundle)
    t163 = time.time() - t3
    print(f"\n  R163 completed in {t163:.0f}s ({t163/60:.1f}min)", flush=True)

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  R161-R163 BATCH COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"    R161 Adaptive Lot:    {r161['verdict']}")
    print(f"    R162 Combined:        {r162['verdict']}")
    print(f"    R163 Donchian:        {r163['verdict']}")
    print(f"{'='*80}", flush=True)

    # Save combined JSON
    output = {
        'experiment': 'R161-R163 Batch 1',
        'config': {
            'pv': PV, 'spread': SPREAD, 'unit_lot': UNIT_LOT,
            'r89_lots': R89_LOTS, 'caps': CAPS,
            'cusum_sigma': CUSUM_SIGMA, 'extreme_window': EXTREME_WINDOW,
            'skip_bars': SKIP_BARS, 'cb_streak': CB_STREAK,
            'cb_pause_hours': CB_PAUSE_HOURS,
            'atr_mult_threshold': ATR_MULT_THRESHOLD,
            'reduced_lot_ratio': REDUCED_LOT_RATIO,
        },
        'r161_adaptive_lot': r161,
        'r162_combined': r162,
        'r163_donchian': r163,
        'timing': {
            'r161_s': round(t161, 1),
            'r162_s': round(t162, 1),
            'r163_s': round(t163, 1),
            'total_s': round(elapsed, 1),
        },
    }

    with open(OUTPUT_DIR / "r161_r163_results.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r161_r163_results.json", flush=True)


if __name__ == "__main__":
    main()
