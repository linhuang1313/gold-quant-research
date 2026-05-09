#!/usr/bin/env python3
"""
R149 — Complete D1 Filter Shootout for SESS_BO & TSMOM
=======================================================
R123 only K-Fold validated D1_EMA20 and Combined.
This experiment gives every filter a fair trial:

Phase 1: Full-sample performance (all 5 filters + no-filter baseline)
Phase 2: K-Fold 5-fold validation for ALL filters (per-fold D1 recalc)
Phase 3: EMA period sweep (EMA10/15/20/25/30/50) with K-Fold
Phase 4: Walk-Forward validation (train 3yr, test 1yr, slide 1yr)
Phase 5: Filter impact on trade frequency (monthly breakdown)
Phase 6: Regime analysis — filter performance in trending vs ranging markets

Strategies tested: SESS_BO, TSMOM
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r149_filter_shootout")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01
CAPS = {'SESS_BO': 35, 'TSMOM': 0}

FOLDS = [
    ("Fold1", "2015-01-01", "2017-03-01"),
    ("Fold2", "2017-03-01", "2019-06-01"),
    ("Fold3", "2019-06-01", "2021-09-01"),
    ("Fold4", "2021-09-01", "2023-12-01"),
    ("Fold5", "2023-12-01", "2027-01-01"),
]

WF_WINDOWS = [
    ("WF1", "2015-01-01", "2018-01-01", "2018-01-01", "2019-01-01"),
    ("WF2", "2016-01-01", "2019-01-01", "2019-01-01", "2020-01-01"),
    ("WF3", "2017-01-01", "2020-01-01", "2020-01-01", "2021-01-01"),
    ("WF4", "2018-01-01", "2021-01-01", "2021-01-01", "2022-01-01"),
    ("WF5", "2019-01-01", "2022-01-01", "2022-01-01", "2023-01-01"),
    ("WF6", "2020-01-01", "2023-01-01", "2023-01-01", "2024-01-01"),
    ("WF7", "2021-01-01", "2024-01-01", "2024-01-01", "2025-01-01"),
    ("WF8", "2022-01-01", "2025-01-01", "2025-01-01", "2026-05-01"),
]

t0 = time.time()

# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_h1():
    import glob
    candidates = sorted(glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
    if not candidates:
        raise FileNotFoundError("No xauusd H1 CSV found")
    csv_path = candidates[-1]
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.set_index('time')[['open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[df['Close'] > 100]
    return df


# ═══════════════════════════════════════════════════════════════
# Helpers (from R123)
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


def _run_exit(pos, i, h, lo_v, c, spread, lot, times,
              sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap):
    held = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_h = (h - pos['entry'] - spread) * lot * PV
        pnl_l = (lo_v - pos['entry'] - spread) * lot * PV
        pnl_c = (c - pos['entry'] - spread) * lot * PV
    else:
        pnl_h = (pos['entry'] - lo_v - spread) * lot * PV
        pnl_l = (pos['entry'] - h - spread) * lot * PV
        pnl_c = (pos['entry'] - c - spread) * lot * PV
    tp_val = tp_atr * pos['atr'] * lot * PV
    sl_val = sl_atr * pos['atr'] * lot * PV
    if pnl_h >= tp_val:
        return _mk(pos, c, times[i], "TP", i, tp_val)
    if pnl_l <= -sl_val:
        return _mk(pos, c, times[i], "SL", i, -sl_val)
    if cap > 0 and pnl_c < -cap:
        return _mk(pos, c, times[i], "MaxLossCap", i, -cap)
    ad = trail_act * pos['atr']; td = trail_dist * pos['atr']
    if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
        ts_p = h - td
        if lo_v <= ts_p:
            return _mk(pos, c, times[i], "Trail", i, (ts_p - pos['entry'] - spread) * lot * PV)
    elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
        ts_p = lo_v + td
        if h >= ts_p:
            return _mk(pos, c, times[i], "Trail", i, (pos['entry'] - ts_p - spread) * lot * PV)
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
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 1 else 0.0


def _stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0.0, 'pnl': 0.0, 'wr': 0.0, 'max_dd': 0.0, 'avg_pnl': 0.0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    eq = np.cumsum(daily)
    dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 0 else 0
    return {
        'n': n,
        'sharpe': round(_sharpe(daily), 3),
        'pnl': round(sum(pnls), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'max_dd': round(dd, 2),
        'avg_pnl': round(sum(pnls) / n, 2),
    }


# ═══════════════════════════════════════════════════════════════
# Strategy backtests
# ═══════════════════════════════════════════════════════════════

def bt_tsmom(h1_df, spread, lot, cap, d1_filter=None,
             fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
             trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    max_lb = max(fast, slow)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        if c[i - fast] > 0: s += 0.5 * np.sign(c[i] / c[i - fast] - 1.0)
        if c[i - slow] > 0: s += 0.5 * np.sign(c[i] / c[i - slow] - 1.0)
        score[i] = s
    trades = []; pos = None; last_exit = -999
    for i in range(max_lb + 1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and score[i] < 0:
                pnl = (c[i] - pos['entry'] - spread) * lot * PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and score[i] > 0:
                pnl = (pos['entry'] - c[i] - spread) * lot * PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(score[i]) or np.isnan(score[i-1]): continue
        entry_date = times[i].date()
        if score[i] > 0 and score[i-1] <= 0:
            if d1_filter is None or _filter_ok(d1_filter, entry_date, 'BUY'):
                pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            if d1_filter is None or _filter_ok(d1_filter, entry_date, 'SELL'):
                pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_sess_bo(h1_df, spread, lot, cap, d1_filter=None,
               session_hour=12, lookback=4, sl_atr=4.5, tp_atr=4.0,
               trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(lookback, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if hours[i] != session_hour: continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        entry_date = times[i].date()
        hh = max(h[i - j] for j in range(1, lookback + 1))
        ll = min(lo[i - j] for j in range(1, lookback + 1))
        if c[i] > hh:
            if d1_filter is None or _filter_ok(d1_filter, entry_date, 'BUY'):
                pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            if d1_filter is None or _filter_ok(d1_filter, entry_date, 'SELL'):
                pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def _filter_ok(d1_filter, entry_date, direction):
    val = d1_filter.get(entry_date, 0)
    if val == 99: return True
    if val == 0: return True
    if direction == 'BUY' and val >= 1: return True
    if direction == 'SELL' and val <= -1: return True
    return False


# ═══════════════════════════════════════════════════════════════
# D1 construction & filters
# ═══════════════════════════════════════════════════════════════

def build_d1(h1_df):
    return h1_df.resample('D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum',
    }).dropna()


def compute_d1_indicators(d1, ema_period=20):
    d1 = d1.copy()
    d1['EMA'] = d1['Close'].ewm(span=ema_period).mean()
    d1['EMA50'] = d1['Close'].ewm(span=50).mean()
    d1['SMA200'] = d1['Close'].rolling(200).mean()

    d1['tenkan'] = (d1['High'].rolling(9).max() + d1['Low'].rolling(9).min()) / 2
    d1['kijun'] = (d1['High'].rolling(26).max() + d1['Low'].rolling(26).min()) / 2
    d1['senkou_a'] = ((d1['tenkan'] + d1['kijun']) / 2).shift(26)
    d1['senkou_b'] = ((d1['High'].rolling(52).max() + d1['Low'].rolling(52).min()) / 2).shift(26)

    tr = pd.concat([d1['High'] - d1['Low'],
                     (d1['High'] - d1['Close'].shift()).abs(),
                     (d1['Low'] - d1['Close'].shift()).abs()], axis=1).max(axis=1)
    d1['D1_ATR14'] = tr.rolling(14).mean()
    d1['KC_upper'] = d1['EMA'] + 2.0 * d1['D1_ATR14']
    d1['KC_lower'] = d1['EMA'] - 2.0 * d1['D1_ATR14']

    h_arr = d1['High'].values; l_arr = d1['Low'].values; c_arr = d1['Close'].values
    n = len(d1)
    plus_dm = np.zeros(n); minus_dm = np.zeros(n); tr_arr = np.zeros(n)
    for i in range(1, n):
        up = h_arr[i] - h_arr[i-1]; down = l_arr[i-1] - l_arr[i]
        plus_dm[i] = up if (up > down and up > 0) else 0
        minus_dm[i] = down if (down > up and down > 0) else 0
        tr_arr[i] = max(h_arr[i] - l_arr[i], abs(h_arr[i] - c_arr[i-1]), abs(l_arr[i] - c_arr[i-1]))
    atr14 = pd.Series(tr_arr).rolling(14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(14).mean().values / np.where(atr14 > 0, atr14, 1)
    minus_di = 100 * pd.Series(minus_dm).rolling(14).mean().values / np.where(atr14 > 0, atr14, 1)
    dx = np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1) * 100
    d1['ADX'] = pd.Series(dx, index=d1.index).rolling(14).mean()

    return d1


def build_filters(d1):
    filters = {}

    ema_dir = pd.Series(0, index=d1.index)
    ema_dir[d1['Close'] > d1['EMA']] = 1
    ema_dir[d1['Close'] < d1['EMA']] = -1
    filters['D1_EMA'] = {k.date() if hasattr(k, 'date') else k: int(v) for k, v in ema_dir.items()}

    sma_dir = pd.Series(0, index=d1.index)
    sma_dir[d1['Close'] > d1['SMA200']] = 1
    sma_dir[d1['Close'] < d1['SMA200']] = -1
    filters['D1_SMA200'] = {k.date() if hasattr(k, 'date') else k: int(v) for k, v in sma_dir.items()}

    ichi_dir = pd.Series(0, index=d1.index)
    cloud_top = d1[['senkou_a', 'senkou_b']].max(axis=1)
    cloud_bot = d1[['senkou_a', 'senkou_b']].min(axis=1)
    ichi_dir[d1['Close'] > cloud_top] = 1
    ichi_dir[d1['Close'] < cloud_bot] = -1
    filters['D1_Ichimoku'] = {k.date() if hasattr(k, 'date') else k: int(v) for k, v in ichi_dir.items()}

    kc_dir = pd.Series(0, index=d1.index)
    kc_dir[d1['Close'] > d1['KC_upper']] = 1
    kc_dir[d1['Close'] < d1['KC_lower']] = -1
    filters['D1_KC'] = {k.date() if hasattr(k, 'date') else k: int(v) for k, v in kc_dir.items()}

    adx_filter = {}
    for idx_val in d1.index:
        dt = idx_val.date() if hasattr(idx_val, 'date') else idx_val
        adx_val = d1.loc[idx_val, 'ADX']
        adx_filter[dt] = 99 if (not np.isnan(adx_val) and adx_val > 20) else 0
    filters['D1_ADX'] = adx_filter

    return filters


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80, flush=True)
    print("  R149 — Complete D1 Filter Shootout", flush=True)
    print("=" * 80, flush=True)

    h1_df = load_h1()
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})", flush=True)

    results = {}
    strats = {
        'SESS_BO': (bt_sess_bo, CAPS['SESS_BO']),
        'TSMOM': (bt_tsmom, CAPS['TSMOM']),
    }
    filter_names = ['D1_EMA', 'D1_SMA200', 'D1_Ichimoku', 'D1_KC', 'D1_ADX']

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Full-sample performance
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 1: Full-Sample Performance (EMA period=20)", flush=True)
    print("=" * 70, flush=True)

    d1 = build_d1(h1_df)
    d1 = compute_d1_indicators(d1, ema_period=20)
    d1_filters = build_filters(d1)

    phase1 = {}
    for sn, (bt_fn, cap) in strats.items():
        print(f"\n  --- {sn} ---", flush=True)
        base = bt_fn(h1_df, SPREAD, UNIT_LOT, cap, d1_filter=None)
        bs = _stats(base)
        print(f"    {'No filter':<15}: n={bs['n']:5d}  Sharpe={bs['sharpe']:7.3f}  "
              f"PnL=${bs['pnl']:9.0f}  WR={bs['wr']:.1f}%  MaxDD=${bs['max_dd']:.0f}", flush=True)
        strat_res = {'baseline': bs}

        for fname in filter_names:
            filt = bt_fn(h1_df, SPREAD, UNIT_LOT, cap, d1_filter=d1_filters[fname])
            fs = _stats(filt)
            delta = fs['sharpe'] - bs['sharpe']
            keep = fs['n'] / bs['n'] * 100 if bs['n'] > 0 else 0
            marker = " ***" if delta > 0.5 else " **" if delta > 0.1 else " *" if delta > 0.05 else ""
            print(f"    {fname:<15}: n={fs['n']:5d}  Sharpe={fs['sharpe']:7.3f}  "
                  f"PnL=${fs['pnl']:9.0f}  WR={fs['wr']:.1f}%  MaxDD=${fs['max_dd']:.0f}  "
                  f"ΔSh={delta:+.3f}  keep={keep:.0f}%{marker}", flush=True)
            strat_res[fname] = {**fs, 'delta_sharpe': round(delta, 3), 'keep_pct': round(keep, 1)}

        phase1[sn] = strat_res
    results['phase1'] = phase1

    # ═══════════════════════════════════════════════════════════
    # Phase 2: K-Fold 5-fold for ALL filters
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: K-Fold 5-Fold Validation — ALL Filters", flush=True)
    print("=" * 70, flush=True)

    phase2 = {}
    for fname in filter_names:
        print(f"\n  === Filter: {fname} ===", flush=True)
        filt_kf = {}

        for sn, (bt_fn, cap) in strats.items():
            fold_base = []; fold_filt = []

            for fold_name, start, end in FOLDS:
                fh1 = h1_df[(h1_df.index >= start) & (h1_df.index < end)]
                if len(fh1) < 500:
                    fold_base.append(0.0); fold_filt.append(0.0); continue

                fd1 = build_d1(fh1)
                if len(fd1) < 60:
                    fold_base.append(0.0); fold_filt.append(0.0); continue
                fd1 = compute_d1_indicators(fd1, ema_period=20)
                ff = build_filters(fd1)

                base_trades = bt_fn(fh1, SPREAD, UNIT_LOT, cap, d1_filter=None)
                filt_trades = bt_fn(fh1, SPREAD, UNIT_LOT, cap, d1_filter=ff.get(fname, {}))

                fold_base.append(_stats(base_trades)['sharpe'])
                fold_filt.append(_stats(filt_trades)['sharpe'])

            wins = sum(1 for a, b in zip(fold_filt, fold_base) if a > b)
            avg_delta = np.mean(fold_filt) - np.mean(fold_base)
            print(f"    {sn:<10}: base={[round(s,2) for s in fold_base]}  "
                  f"filt={[round(s,2) for s in fold_filt]}  wins={wins}/5  "
                  f"avg_Δ={avg_delta:+.2f}", flush=True)

            filt_kf[sn] = {
                'base_folds': [round(s, 3) for s in fold_base],
                'filt_folds': [round(s, 3) for s in fold_filt],
                'base_mean': round(float(np.mean(fold_base)), 3),
                'filt_mean': round(float(np.mean(fold_filt)), 3),
                'wins': wins,
                'avg_delta': round(float(avg_delta), 3),
            }

        phase2[fname] = filt_kf
    results['phase2_kfold'] = phase2

    # ═══════════════════════════════════════════════════════════
    # Phase 3: EMA Period Sweep with K-Fold
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 3: EMA Period Sweep (10/15/20/25/30/50) + K-Fold", flush=True)
    print("=" * 70, flush=True)

    ema_periods = [10, 15, 20, 25, 30, 50]
    phase3 = {}

    for period in ema_periods:
        print(f"\n  === EMA period = {period} ===", flush=True)
        period_res = {}

        for sn, (bt_fn, cap) in strats.items():
            # Full sample
            d1_p = compute_d1_indicators(build_d1(h1_df), ema_period=period)
            ff_p = build_filters(d1_p)
            base = bt_fn(h1_df, SPREAD, UNIT_LOT, cap, d1_filter=None)
            filt = bt_fn(h1_df, SPREAD, UNIT_LOT, cap, d1_filter=ff_p['D1_EMA'])
            bs = _stats(base); fs = _stats(filt)

            # K-Fold
            fold_base = []; fold_filt = []
            for _, start, end in FOLDS:
                fh1 = h1_df[(h1_df.index >= start) & (h1_df.index < end)]
                if len(fh1) < 500:
                    fold_base.append(0.0); fold_filt.append(0.0); continue
                fd1 = build_d1(fh1)
                if len(fd1) < 60:
                    fold_base.append(0.0); fold_filt.append(0.0); continue
                fd1 = compute_d1_indicators(fd1, ema_period=period)
                ff = build_filters(fd1)
                b_trades = bt_fn(fh1, SPREAD, UNIT_LOT, cap, d1_filter=None)
                f_trades = bt_fn(fh1, SPREAD, UNIT_LOT, cap, d1_filter=ff['D1_EMA'])
                fold_base.append(_stats(b_trades)['sharpe'])
                fold_filt.append(_stats(f_trades)['sharpe'])

            wins = sum(1 for a, b in zip(fold_filt, fold_base) if a > b)
            avg_delta = np.mean(fold_filt) - np.mean(fold_base)
            keep = fs['n'] / bs['n'] * 100 if bs['n'] > 0 else 0

            print(f"    {sn:<10}: full ΔSh={fs['sharpe']-bs['sharpe']:+.3f}  "
                  f"kfold wins={wins}/5  avg_Δ={avg_delta:+.2f}  "
                  f"keep={keep:.0f}%  n={fs['n']}", flush=True)

            period_res[sn] = {
                'full_base_sharpe': bs['sharpe'],
                'full_filt_sharpe': fs['sharpe'],
                'full_delta': round(fs['sharpe'] - bs['sharpe'], 3),
                'full_n': fs['n'],
                'keep_pct': round(keep, 1),
                'kfold_base': [round(s, 3) for s in fold_base],
                'kfold_filt': [round(s, 3) for s in fold_filt],
                'kfold_wins': wins,
                'kfold_avg_delta': round(float(avg_delta), 3),
            }

        phase3[str(period)] = period_res
    results['phase3_ema_sweep'] = phase3

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Walk-Forward Validation (ALL filters)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 4: Walk-Forward Validation (train 3yr, test 1yr)", flush=True)
    print("=" * 70, flush=True)

    phase4 = {}
    for fname in filter_names:
        print(f"\n  === Filter: {fname} ===", flush=True)
        wf_res = {}

        for sn, (bt_fn, cap) in strats.items():
            train_sharpes = []; test_sharpes = []
            train_base = []; test_base = []

            for wf_name, tr_start, tr_end, te_start, te_end in WF_WINDOWS:
                train_h1 = h1_df[(h1_df.index >= tr_start) & (h1_df.index < tr_end)]
                test_h1 = h1_df[(h1_df.index >= te_start) & (h1_df.index < te_end)]

                if len(train_h1) < 500 or len(test_h1) < 200:
                    continue

                train_d1 = compute_d1_indicators(build_d1(train_h1), ema_period=20)
                test_d1 = compute_d1_indicators(build_d1(test_h1), ema_period=20)
                train_ff = build_filters(train_d1)
                test_ff = build_filters(test_d1)

                # Train: base vs filtered
                tr_base = bt_fn(train_h1, SPREAD, UNIT_LOT, cap, d1_filter=None)
                tr_filt = bt_fn(train_h1, SPREAD, UNIT_LOT, cap, d1_filter=train_ff.get(fname, {}))
                train_base.append(_stats(tr_base)['sharpe'])
                train_sharpes.append(_stats(tr_filt)['sharpe'])

                # Test: base vs filtered (out-of-sample)
                te_base = bt_fn(test_h1, SPREAD, UNIT_LOT, cap, d1_filter=None)
                te_filt = bt_fn(test_h1, SPREAD, UNIT_LOT, cap, d1_filter=test_ff.get(fname, {}))
                test_base.append(_stats(te_base)['sharpe'])
                test_sharpes.append(_stats(te_filt)['sharpe'])

            if not test_sharpes:
                continue

            oos_wins = sum(1 for a, b in zip(test_sharpes, test_base) if a > b)
            oos_delta = np.mean(test_sharpes) - np.mean(test_base)

            is_delta_mean = np.mean(train_sharpes) - np.mean(train_base) if train_sharpes else 0
            wfe = oos_delta / is_delta_mean if is_delta_mean != 0 else 0

            print(f"    {sn:<10}: OOS wins={oos_wins}/{len(test_sharpes)}  "
                  f"OOS avg_Δ={oos_delta:+.2f}  IS avg_Δ={is_delta_mean:+.2f}  "
                  f"WFE={wfe:.2f}", flush=True)

            wf_res[sn] = {
                'oos_wins': oos_wins,
                'oos_total': len(test_sharpes),
                'oos_delta': round(float(oos_delta), 3),
                'is_delta': round(float(is_delta_mean), 3),
                'wfe': round(float(wfe), 2),
                'oos_base': [round(s, 3) for s in test_base],
                'oos_filt': [round(s, 3) for s in test_sharpes],
            }

        phase4[fname] = wf_res
    results['phase4_walkforward'] = phase4

    # ═══════════════════════════════════════════════════════════
    # Phase 5: Monthly trade frequency impact
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 5: Trade Frequency Impact (monthly)", flush=True)
    print("=" * 70, flush=True)

    phase5 = {}
    d1_full = compute_d1_indicators(build_d1(h1_df), ema_period=20)
    d1_full_filters = build_filters(d1_full)

    for sn, (bt_fn, cap) in strats.items():
        print(f"\n  --- {sn} ---", flush=True)
        base_trades = bt_fn(h1_df, SPREAD, UNIT_LOT, cap, d1_filter=None)

        monthly_base = defaultdict(int)
        for t in base_trades:
            ym = pd.Timestamp(t['entry_time']).strftime('%Y-%m')
            monthly_base[ym] += 1

        sn_res = {'baseline_monthly_avg': round(np.mean(list(monthly_base.values())), 1) if monthly_base else 0}
        print(f"    No filter: avg {sn_res['baseline_monthly_avg']:.1f} trades/month", flush=True)

        for fname in filter_names:
            filt_trades = bt_fn(h1_df, SPREAD, UNIT_LOT, cap, d1_filter=d1_full_filters[fname])
            monthly_filt = defaultdict(int)
            for t in filt_trades:
                ym = pd.Timestamp(t['entry_time']).strftime('%Y-%m')
                monthly_filt[ym] += 1

            avg_filt = round(np.mean(list(monthly_filt.values())), 1) if monthly_filt else 0
            zero_months = sum(1 for ym in monthly_base if monthly_filt.get(ym, 0) == 0)
            print(f"    {fname:<15}: avg {avg_filt:.1f} trades/month  "
                  f"zero-trade months: {zero_months}", flush=True)
            sn_res[fname] = {
                'monthly_avg': avg_filt,
                'zero_months': zero_months,
            }

        phase5[sn] = sn_res
    results['phase5_frequency'] = phase5

    # ═══════════════════════════════════════════════════════════
    # Phase 6: Regime analysis (trending vs ranging)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 6: Regime Analysis (High/Low ADX periods)", flush=True)
    print("=" * 70, flush=True)

    phase6 = {}
    d1_regime = compute_d1_indicators(build_d1(h1_df), ema_period=20)

    adx_median = d1_regime['ADX'].median()
    trending_dates = set(d1_regime[d1_regime['ADX'] > adx_median].index.date)
    ranging_dates = set(d1_regime[d1_regime['ADX'] <= adx_median].index.date)
    print(f"  ADX median: {adx_median:.1f}", flush=True)
    print(f"  Trending days: {len(trending_dates)}, Ranging days: {len(ranging_dates)}", flush=True)

    for sn, (bt_fn, cap) in strats.items():
        print(f"\n  --- {sn} ---", flush=True)
        base_all = bt_fn(h1_df, SPREAD, UNIT_LOT, cap, d1_filter=None)

        base_trending = [t for t in base_all if pd.Timestamp(t['entry_time']).date() in trending_dates]
        base_ranging = [t for t in base_all if pd.Timestamp(t['entry_time']).date() in ranging_dates]

        sn_regime = {
            'baseline_trending': _stats(base_trending),
            'baseline_ranging': _stats(base_ranging),
        }
        print(f"    No filter: trending Sh={_stats(base_trending)['sharpe']:.3f} (n={len(base_trending)}), "
              f"ranging Sh={_stats(base_ranging)['sharpe']:.3f} (n={len(base_ranging)})", flush=True)

        for fname in filter_names:
            filt_all = bt_fn(h1_df, SPREAD, UNIT_LOT, cap, d1_filter=d1_full_filters[fname])
            filt_trending = [t for t in filt_all if pd.Timestamp(t['entry_time']).date() in trending_dates]
            filt_ranging = [t for t in filt_all if pd.Timestamp(t['entry_time']).date() in ranging_dates]

            st_t = _stats(filt_trending); st_r = _stats(filt_ranging)
            dt = st_t['sharpe'] - _stats(base_trending)['sharpe']
            dr = st_r['sharpe'] - _stats(base_ranging)['sharpe']
            print(f"    {fname:<15}: trend Sh={st_t['sharpe']:.3f} (Δ={dt:+.3f}), "
                  f"range Sh={st_r['sharpe']:.3f} (Δ={dr:+.3f})", flush=True)

            sn_regime[fname] = {
                'trending': st_t,
                'ranging': st_r,
                'delta_trending': round(dt, 3),
                'delta_ranging': round(dr, 3),
            }

        phase6[sn] = sn_regime
    results['phase6_regime'] = phase6

    # ═══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    print("\n" + "=" * 80, flush=True)
    print("  R149 FINAL SUMMARY — Filter Ranking", flush=True)
    print("=" * 80, flush=True)

    print(f"\n  {'Filter':<15} {'Full ΔSh':>9} {'KF wins':>8} {'KF avg_Δ':>9} "
          f"{'WF OOS_Δ':>9} {'WFE':>6} {'Keep%':>6}", flush=True)
    print(f"  {'='*62}", flush=True)

    ranking = []
    for fname in filter_names:
        kf = phase2.get(fname, {})
        wf = phase4.get(fname, {})
        p1 = phase1

        total_full_delta = sum(
            p1[sn].get(fname, {}).get('delta_sharpe', 0) for sn in strats
        )
        total_kf_wins = sum(kf.get(sn, {}).get('wins', 0) for sn in strats)
        total_kf_delta = np.mean([kf.get(sn, {}).get('avg_delta', 0) for sn in strats])
        total_oos_delta = np.mean([wf.get(sn, {}).get('oos_delta', 0) for sn in strats])
        avg_wfe = np.mean([wf.get(sn, {}).get('wfe', 0) for sn in strats])
        avg_keep = np.mean([
            p1[sn].get(fname, {}).get('keep_pct', 100) for sn in strats
        ])

        print(f"  {fname:<15} {total_full_delta:>+9.3f} {total_kf_wins:>5d}/10 "
              f"{total_kf_delta:>+9.3f} {total_oos_delta:>+9.3f} {avg_wfe:>6.2f} "
              f"{avg_keep:>5.0f}%", flush=True)

        ranking.append({
            'filter': fname,
            'full_delta': round(total_full_delta, 3),
            'kf_wins': total_kf_wins,
            'kf_delta': round(float(total_kf_delta), 3),
            'oos_delta': round(float(total_oos_delta), 3),
            'wfe': round(float(avg_wfe), 2),
            'keep_pct': round(float(avg_keep), 1),
        })

    ranking.sort(key=lambda x: (-x['kf_wins'], -x['kf_delta']))
    results['ranking'] = ranking

    print(f"\n  Best EMA period (Phase 3):", flush=True)
    for sn in strats:
        best_p = max(phase3.items(), key=lambda x: x[1].get(sn, {}).get('kfold_wins', 0))
        print(f"    {sn}: EMA{best_p[0]} (kfold wins={best_p[1][sn]['kfold_wins']}/5, "
              f"avg_Δ={best_p[1][sn]['kfold_avg_delta']:+.3f})", flush=True)

    results['elapsed_s'] = round(elapsed, 1)
    out_file = OUTPUT_DIR / "r149_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"  Saved: {out_file}", flush=True)
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
