#!/usr/bin/env python3
"""
R177 — Multi-Strategy Resonance
================================
Tests whether multiple strategies signaling the same direction on the same H1
bar improves win rate and Sharpe ratio.
"""
import sys
import os
import json
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r177_resonance")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]

RESONANCE_THRESHOLD_OPTIONS = [2, 3]
BOOST_MULT = 1.5

EXIT_PARAMS = dict(
    sl_atr=3.5, tp_atr=8.0,
    trail_act=0.14, trail_dist=0.025,
    max_hold=20, maxloss_cap=35,
)


# ═══════════════════════════════════════════════════════════════
# Helpers
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
                    ep = df['High'].iloc[i]; af = min(af + af_start, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], df['High'].iloc[i-1], df['High'].iloc[max(0, i-2)])
            if df['High'].iloc[i] > psar[i]:
                direction[i] = 1; psar[i] = ep; ep = df['High'].iloc[i]; af = af_start
            else:
                direction[i] = -1
                if df['Low'].iloc[i] < ep:
                    ep = df['Low'].iloc[i]; af = min(af + af_start, af_max)
    df['PSAR_dir'] = direction; df['ATR'] = compute_atr(df)
    return df


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': str(pos['time']), 'exit_time': str(exit_time),
            'entry_bar': pos['bar'],
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


def _trades_to_daily(trades):
    if not trades:
        return np.array([0.0])
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    if not daily:
        return np.array([0.0])
    return np.array([daily[k] for k in sorted(daily.keys())])


def _sharpe(arr):
    if len(arr) < 10 or np.std(arr, ddof=1) == 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))


def _max_dd(arr):
    if len(arr) == 0:
        return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def _compute_stats(trades):
    if not trades:
        return {'n_trades': 0, 'sharpe': 0, 'pnl': 0, 'win_rate': 0, 'max_dd': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    return {
        'n_trades': n,
        'sharpe': round(_sharpe(daily), 3),
        'pnl': round(sum(pnls), 2),
        'win_rate': round(sum(1 for p in pnls if p > 0) / n * 100, 1) if n > 0 else 0,
        'max_dd': round(_max_dd(daily), 2),
    }


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_h1():
    candidates = sorted(glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
    if not candidates:
        raise FileNotFoundError("No xauusd H1 CSV found")
    csv_path = candidates[-1]
    print(f"  Loading H1 from: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.set_index('time')[['open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[df['Close'] > 100]
    return df


# ═══════════════════════════════════════════════════════════════
# Signal extraction (per-bar signal arrays)
# ═══════════════════════════════════════════════════════════════

def extract_keltner_signals(df, adx_th=14, ema_period=25, kc_mult=1.2):
    d = df.copy()
    d['ATR'] = compute_atr(d)
    d['ADX'] = compute_adx(d)
    d['EMA100'] = d['Close'].ewm(span=100, adjust=False).mean()
    d['KC_mid'] = d['Close'].ewm(span=ema_period, adjust=False).mean()
    d['KC_upper'] = d['KC_mid'] + kc_mult * d['ATR']
    d['KC_lower'] = d['KC_mid'] - kc_mult * d['ATR']

    n = len(d)
    signals = np.zeros(n, dtype=np.int8)
    c = d['Close'].values
    adx = d['ADX'].values
    ema100 = d['EMA100'].values
    kc_u = d['KC_upper'].values
    kc_l = d['KC_lower'].values

    for i in range(n):
        if np.isnan(adx[i]) or adx[i] < adx_th:
            continue
        if np.isnan(kc_u[i]) or np.isnan(ema100[i]):
            continue
        if c[i] > kc_u[i] and c[i] > ema100[i]:
            signals[i] = 1
        elif c[i] < kc_l[i] and c[i] < ema100[i]:
            signals[i] = -1
    return signals


def extract_tsmom_signals(df, fast=480, slow=720):
    c = df['Close'].values
    n = len(df)
    max_lb = max(fast, slow)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        if c[i - fast] > 0:
            s += 0.5 * np.sign(c[i] / c[i - fast] - 1.0)
        if c[i - slow] > 0:
            s += 0.5 * np.sign(c[i] / c[i - slow] - 1.0)
        score[i] = s

    signals = np.zeros(n, dtype=np.int8)
    for i in range(max_lb + 1, n):
        if np.isnan(score[i]) or np.isnan(score[i-1]):
            continue
        if score[i] > 0 and score[i-1] <= 0:
            signals[i] = 1
        elif score[i] < 0 and score[i-1] >= 0:
            signals[i] = -1
    return signals


def extract_psar_signals(df):
    d = add_psar(df)
    pdir = d['PSAR_dir'].values
    n = len(d)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if pdir[i-1] == -1 and pdir[i] == 1:
            signals[i] = 1
        elif pdir[i-1] == 1 and pdir[i] == -1:
            signals[i] = -1
    return signals


def extract_sess_bo_signals(df, session_hour=12, lookback=4):
    c = df['Close'].values
    h = df['High'].values
    lo = df['Low'].values
    hours = df.index.hour
    n = len(df)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(lookback, n):
        if hours[i] != session_hour:
            continue
        hh = max(h[i - j] for j in range(1, lookback + 1))
        ll = min(lo[i - j] for j in range(1, lookback + 1))
        if c[i] > hh:
            signals[i] = 1
        elif c[i] < ll:
            signals[i] = -1
    return signals


# ═══════════════════════════════════════════════════════════════
# Resonance detection
# ═══════════════════════════════════════════════════════════════

def compute_resonance(signal_arrays):
    n = len(signal_arrays[0])
    resonance_dir = np.zeros(n, dtype=np.int8)
    resonance_count = np.zeros(n, dtype=np.int8)

    for i in range(n):
        buy_count = sum(1 for s in signal_arrays if s[i] == 1)
        sell_count = sum(1 for s in signal_arrays if s[i] == -1)
        if buy_count > sell_count and buy_count >= 2:
            resonance_dir[i] = 1
            resonance_count[i] = buy_count
        elif sell_count > buy_count and sell_count >= 2:
            resonance_dir[i] = -1
            resonance_count[i] = sell_count
        elif buy_count == sell_count and buy_count >= 2:
            pass
        elif buy_count >= 2:
            resonance_dir[i] = 1
            resonance_count[i] = buy_count
        elif sell_count >= 2:
            resonance_dir[i] = -1
            resonance_count[i] = sell_count

    return resonance_dir, resonance_count


# ═══════════════════════════════════════════════════════════════
# Backtests
# ═══════════════════════════════════════════════════════════════

def bt_resonance(h1_df, resonance_dir, resonance_count, threshold, spread, lot):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)

    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(
                pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                EXIT_PARAMS['sl_atr'], EXIT_PARAMS['tp_atr'],
                EXIT_PARAMS['trail_act'], EXIT_PARAMS['trail_dist'],
                EXIT_PARAMS['max_hold'], EXIT_PARAMS['maxloss_cap'])
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
        if resonance_count[i] >= threshold:
            if resonance_dir[i] == 1:
                pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
            elif resonance_dir[i] == -1:
                pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_keltner_boosted(h1_df, keltner_signals, signal_arrays, spread, base_lot):
    """Keltner entry with 1.5x lot when another strategy confirms."""
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)

    other_signals = [s for s in signal_arrays if not np.array_equal(s, keltner_signals)]

    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(
                pos, i, h[i], lo[i], c[i], spread, pos['lot'], PV, times,
                EXIT_PARAMS['sl_atr'], EXIT_PARAMS['tp_atr'],
                EXIT_PARAMS['trail_act'], EXIT_PARAMS['trail_dist'],
                EXIT_PARAMS['max_hold'], EXIT_PARAMS['maxloss_cap'])
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
        if keltner_signals[i] == 0:
            continue

        direction = keltner_signals[i]
        confirmed = any(s[i] == direction for s in other_signals)
        lot = base_lot * BOOST_MULT if confirmed else base_lot

        if direction == 1:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i], 'lot': lot}
        else:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i], 'lot': lot}
    return trades


def bt_keltner_only(h1_df, keltner_signals, spread, lot):
    """Baseline: trade only on keltner signals using standard exit."""
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)

    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(
                pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                EXIT_PARAMS['sl_atr'], EXIT_PARAMS['tp_atr'],
                EXIT_PARAMS['trail_act'], EXIT_PARAMS['trail_dist'],
                EXIT_PARAMS['max_hold'], EXIT_PARAMS['maxloss_cap'])
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
        if keltner_signals[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif keltner_signals[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


# ═══════════════════════════════════════════════════════════════
# K-Fold validation
# ═══════════════════════════════════════════════════════════════

def run_kfold(h1_df, signal_arrays, keltner_signals, resonance_dir, resonance_count, best_variant):
    print("\n  K-Fold 6-Fold Validation:", flush=True)
    fold_results = []

    for fold_name, start, end in FOLDS:
        mask = (h1_df.index >= start) & (h1_df.index < end)
        fold_df = h1_df[mask]
        if len(fold_df) < 100:
            fold_results.append({'fold': fold_name, 'n_bars': len(fold_df), 'stats': _compute_stats([])})
            continue

        fold_start_idx = h1_df.index.get_loc(fold_df.index[0])
        fold_end_idx = h1_df.index.get_loc(fold_df.index[-1]) + 1
        fold_res_dir = resonance_dir[fold_start_idx:fold_end_idx]
        fold_res_cnt = resonance_count[fold_start_idx:fold_end_idx]
        fold_kelt = keltner_signals[fold_start_idx:fold_end_idx]
        fold_signals = [s[fold_start_idx:fold_end_idx] for s in signal_arrays]

        if best_variant == 'resonance_2':
            trades = bt_resonance(fold_df, fold_res_dir, fold_res_cnt, 2, SPREAD, UNIT_LOT)
        elif best_variant == 'resonance_3':
            trades = bt_resonance(fold_df, fold_res_dir, fold_res_cnt, 3, SPREAD, UNIT_LOT)
        elif best_variant == 'keltner_boosted':
            trades = bt_keltner_boosted(fold_df, fold_kelt, fold_signals, SPREAD, UNIT_LOT)
        else:
            trades = bt_keltner_only(fold_df, fold_kelt, SPREAD, UNIT_LOT)

        stats = _compute_stats(trades)
        fold_results.append({'fold': fold_name, 'n_bars': len(fold_df), 'stats': stats})
        print(f"    {fold_name}: n={stats['n_trades']:>4d}  sharpe={stats['sharpe']:>6.3f}  "
              f"pnl=${stats['pnl']:>8.2f}  wr={stats['win_rate']:.1f}%  dd=${stats['max_dd']:.2f}", flush=True)

    return fold_results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("R177 — Multi-Strategy Resonance Experiment", flush=True)
    print("=" * 70, flush=True)

    h1_df = load_h1()
    n_bars = len(h1_df)
    print(f"  H1 bars: {n_bars} ({h1_df.index[0]} -> {h1_df.index[-1]})", flush=True)

    print("\n  Extracting signals...", flush=True)
    sig_keltner = extract_keltner_signals(h1_df)
    print(f"    Keltner: {np.sum(sig_keltner == 1)} BUY, {np.sum(sig_keltner == -1)} SELL", flush=True)

    sig_tsmom = extract_tsmom_signals(h1_df)
    print(f"    TSMOM:   {np.sum(sig_tsmom == 1)} BUY, {np.sum(sig_tsmom == -1)} SELL", flush=True)

    sig_psar = extract_psar_signals(h1_df)
    print(f"    PSAR:    {np.sum(sig_psar == 1)} BUY, {np.sum(sig_psar == -1)} SELL", flush=True)

    sig_sess = extract_sess_bo_signals(h1_df)
    print(f"    SESS_BO: {np.sum(sig_sess == 1)} BUY, {np.sum(sig_sess == -1)} SELL", flush=True)

    signal_arrays = [sig_keltner, sig_tsmom, sig_psar, sig_sess]

    print("\n  Computing resonance...", flush=True)
    resonance_dir, resonance_count = compute_resonance(signal_arrays)

    freq_analysis = {}
    for thr in [2, 3, 4]:
        cnt = int(np.sum(resonance_count >= thr))
        freq_analysis[f'bars_with_{thr}_or_more'] = cnt
        freq_analysis[f'pct_with_{thr}_or_more'] = round(cnt / n_bars * 100, 3)
    print(f"    Resonance >= 2: {freq_analysis['bars_with_2_or_more']} bars "
          f"({freq_analysis['pct_with_2_or_more']:.3f}%)", flush=True)
    print(f"    Resonance >= 3: {freq_analysis['bars_with_3_or_more']} bars "
          f"({freq_analysis['pct_with_3_or_more']:.3f}%)", flush=True)
    print(f"    Resonance >= 4: {freq_analysis['bars_with_4_or_more']} bars "
          f"({freq_analysis['pct_with_4_or_more']:.3f}%)", flush=True)

    # --- Variant backtests ---
    print("\n  Running variant backtests...", flush=True)
    results = {}

    print("    [1/4] Keltner-only (baseline)...", flush=True)
    trades_keltner = bt_keltner_only(h1_df, sig_keltner, SPREAD, UNIT_LOT)
    stats_keltner = _compute_stats(trades_keltner)
    results['keltner_only'] = stats_keltner
    print(f"          n={stats_keltner['n_trades']}  sharpe={stats_keltner['sharpe']:.3f}  "
          f"pnl=${stats_keltner['pnl']:.2f}  wr={stats_keltner['win_rate']:.1f}%", flush=True)

    print("    [2/4] Resonance >= 2...", flush=True)
    trades_res2 = bt_resonance(h1_df, resonance_dir, resonance_count, 2, SPREAD, UNIT_LOT)
    stats_res2 = _compute_stats(trades_res2)
    results['resonance_2'] = stats_res2
    print(f"          n={stats_res2['n_trades']}  sharpe={stats_res2['sharpe']:.3f}  "
          f"pnl=${stats_res2['pnl']:.2f}  wr={stats_res2['win_rate']:.1f}%", flush=True)

    print("    [3/4] Resonance >= 3...", flush=True)
    trades_res3 = bt_resonance(h1_df, resonance_dir, resonance_count, 3, SPREAD, UNIT_LOT)
    stats_res3 = _compute_stats(trades_res3)
    results['resonance_3'] = stats_res3
    print(f"          n={stats_res3['n_trades']}  sharpe={stats_res3['sharpe']:.3f}  "
          f"pnl=${stats_res3['pnl']:.2f}  wr={stats_res3['win_rate']:.1f}%", flush=True)

    print("    [4/4] Keltner + boost on confirmation...", flush=True)
    trades_boost = bt_keltner_boosted(h1_df, sig_keltner, signal_arrays, SPREAD, UNIT_LOT)
    stats_boost = _compute_stats(trades_boost)
    results['keltner_boosted'] = stats_boost
    print(f"          n={stats_boost['n_trades']}  sharpe={stats_boost['sharpe']:.3f}  "
          f"pnl=${stats_boost['pnl']:.2f}  wr={stats_boost['win_rate']:.1f}%", flush=True)

    # --- Determine best variant ---
    best_variant = max(results.keys(), key=lambda k: results[k]['sharpe'])
    print(f"\n  Best variant: {best_variant} (Sharpe={results[best_variant]['sharpe']:.3f})", flush=True)

    # --- K-Fold validation on best variant ---
    fold_results = run_kfold(h1_df, signal_arrays, sig_keltner,
                             resonance_dir, resonance_count, best_variant)

    # --- Summary ---
    print("\n" + "=" * 70, flush=True)
    print("  SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'Variant':<20s} {'N':>5s} {'Sharpe':>8s} {'PnL':>10s} {'WR%':>6s} {'MaxDD':>8s}", flush=True)
    print(f"  {'-'*20} {'-'*5} {'-'*8} {'-'*10} {'-'*6} {'-'*8}", flush=True)
    for name, stats in results.items():
        print(f"  {name:<20s} {stats['n_trades']:>5d} {stats['sharpe']:>8.3f} "
              f"${stats['pnl']:>9.2f} {stats['win_rate']:>5.1f}% ${stats['max_dd']:>7.2f}", flush=True)

    # --- Save results ---
    output = {
        'experiment': 'R177_resonance',
        'h1_bars': n_bars,
        'date_range': f"{h1_df.index[0]} -> {h1_df.index[-1]}",
        'variant_stats': results,
        'resonance_frequency': freq_analysis,
        'best_variant': best_variant,
        'kfold_validation': fold_results,
        'params': {
            'exit_params': EXIT_PARAMS,
            'boost_multiplier': BOOST_MULT,
            'strategies_used': ['keltner', 'tsmom', 'psar', 'sess_bo'],
        },
    }

    out_path = OUTPUT_DIR / "r177_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}", flush=True)
    print("  Done.", flush=True)


if __name__ == '__main__':
    main()
