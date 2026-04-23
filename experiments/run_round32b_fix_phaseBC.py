"""
R32-B3/C Fix: Alternative Trend Indicators & Vol TS Independent Strategy
=========================================================================
Phase B3 and C produced 0 trades due to signal_fn closures not capturing
loop variables correctly + dropna() issues. This patch fixes both.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS

OUT_DIR = Path("results/round32_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            try: f.write(data)
            except: pass
            f.flush()
    def flush(self):
        for f in self.files: f.flush()


L7_MH8 = {
    **LIVE_PARITY_KWARGS,
    'time_adaptive_trail': {'start': 2, 'decay': 0.75, 'floor': 0.003},
    'min_entry_gap_hours': 1.0,
    'keltner_max_hold_m15': 8,
}


def make_daily(trades):
    daily = {}
    for t in trades:
        exit_t = t.exit_time if hasattr(t, 'exit_time') else t['exit_time']
        pnl = t.pnl if hasattr(t, 'pnl') else t['pnl']
        d = pd.Timestamp(exit_t).date()
        daily.setdefault(d, 0); daily[d] += pnl
    return pd.Series(daily).sort_index()


def _stats_from_trades(trades, label):
    if not trades:
        return {'label': label, 'n': 0, 'sharpe': 0, 'total_pnl': 0, 'win_rate': 0, 'max_dd': 0,
                '_trades': [], '_daily_pnl': {}}
    pnls = [t['pnl'] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    eq = np.cumsum(pnls); dd = (np.maximum.accumulate(eq + 2000) - (eq + 2000)).max()
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily.setdefault(d, 0); daily[d] += t['pnl']
    da = np.array(list(daily.values()))
    sh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0
    return {'label': label, 'n': len(trades), 'sharpe': sh, 'total_pnl': sum(pnls),
            'win_rate': wins/len(trades)*100, 'max_dd': dd, '_trades': trades, '_daily_pnl': daily}


def _at(trades, equity, pos, close, exit_time, reason, bar_idx, pnl):
    trades.append({'dir': pos['dir'], 'entry': pos['entry'],
                   'entry_time': pos['time'], 'exit_time': exit_time,
                   'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']})
    equity.append(equity[-1] + pnl)


def backtest_generic_v2(close_arr, high_arr, low_arr, atr_arr, times_arr, signals,
                        label, sl_atr=3.5, tp_atr=8.0,
                        trail_act_atr=0.28, trail_dist_atr=0.06, max_hold=50,
                        spread=0.30, lot=0.03):
    """Simplified backtester that takes pre-computed signals array.
    signals[i] = 'BUY', 'SELL', or None
    """
    n = len(close_arr)
    trades = []; pos = None; equity = [2000.0]; last_exit = -999

    for i in range(1, n):
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (high_arr[i] - pos['entry'] - spread) * lot * 100
                pnl_l = (low_arr[i] - pos['entry'] - spread) * lot * 100
                pnl_c = (close_arr[i] - pos['entry'] - spread) * lot * 100
            else:
                pnl_h = (pos['entry'] - low_arr[i] - spread) * lot * 100
                pnl_l = (pos['entry'] - high_arr[i] - spread) * lot * 100
                pnl_c = (pos['entry'] - close_arr[i] - spread) * lot * 100
            tp_val = tp_atr * pos['atr'] * lot * 100
            sl_val = sl_atr * pos['atr'] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                _at(trades, equity, pos, close_arr[i], times_arr[i], "TP", i, tp_val); exited = True
            elif pnl_l <= -sl_val:
                _at(trades, equity, pos, close_arr[i], times_arr[i], "SL", i, -sl_val); exited = True
            else:
                ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and high_arr[i] - pos['entry'] >= ad:
                    ts = high_arr[i] - td
                    if low_arr[i] <= ts:
                        _at(trades, equity, pos, close_arr[i], times_arr[i], "Trail", i,
                            (ts - pos['entry'] - spread) * lot * 100); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - low_arr[i] >= ad:
                    ts = low_arr[i] + td
                    if high_arr[i] >= ts:
                        _at(trades, equity, pos, close_arr[i], times_arr[i], "Trail", i,
                            (pos['entry'] - ts - spread) * lot * 100); exited = True
                if not exited and held >= max_hold:
                    _at(trades, equity, pos, close_arr[i], times_arr[i], "Timeout", i, pnl_c); exited = True
            if exited: pos = None; last_exit = i; continue
        if pos is not None: continue
        if i - last_exit < 2: continue
        if np.isnan(atr_arr[i]) or atr_arr[i] < 0.1: continue
        sig = signals[i]
        if sig == 'BUY':
            pos = {'dir': 'BUY', 'entry': close_arr[i] + spread/2, 'bar': i, 'time': times_arr[i], 'atr': atr_arr[i]}
        elif sig == 'SELL':
            pos = {'dir': 'SELL', 'entry': close_arr[i] - spread/2, 'bar': i, 'time': times_arr[i], 'atr': atr_arr[i]}

    return _stats_from_trades(trades, label)


def compute_supertrend(high, low, close, atr, period=10, factor=3.0):
    """Compute SuperTrend from numpy arrays."""
    n = len(close)
    hl2 = (high + low) / 2
    upper_basic = hl2 + factor * atr
    lower_basic = hl2 - factor * atr

    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()
    direction = np.zeros(n, dtype=int)

    start = max(period, np.argmax(~np.isnan(atr)))
    direction[:start] = -1

    for i in range(start, n):
        if np.isnan(atr[i]): direction[i] = direction[i-1]; continue
        if i == start:
            direction[i] = -1 if close[i] > upper_band[i] else 1
        else:
            if lower_basic[i] > lower_band[i-1] or close[i-1] < lower_band[i-1]:
                lower_band[i] = lower_basic[i]
            else:
                lower_band[i] = lower_band[i-1]
            if upper_basic[i] < upper_band[i-1] or close[i-1] > upper_band[i-1]:
                upper_band[i] = upper_basic[i]
            else:
                upper_band[i] = upper_band[i-1]

            if direction[i-1] == 1:
                direction[i] = -1 if close[i] > upper_band[i] else 1
            else:
                direction[i] = 1 if close[i] < lower_band[i] else -1

    return direction


def compute_ichimoku(high, low, close, tenkan=9, kijun=26, senkou_b_p=52):
    """Compute Ichimoku from numpy arrays."""
    n = len(close)
    tenkan_sen = np.full(n, np.nan)
    kijun_sen = np.full(n, np.nan)
    senkou_a = np.full(n, np.nan)
    senkou_b = np.full(n, np.nan)

    for i in range(senkou_b_p + kijun, n):
        if i >= tenkan - 1:
            tenkan_sen[i] = (np.max(high[i-tenkan+1:i+1]) + np.min(low[i-tenkan+1:i+1])) / 2
        if i >= kijun - 1:
            kijun_sen[i] = (np.max(high[i-kijun+1:i+1]) + np.min(low[i-kijun+1:i+1])) / 2
        if i >= kijun and not np.isnan(tenkan_sen[i-kijun]) and not np.isnan(kijun_sen[i-kijun]):
            senkou_a[i] = (tenkan_sen[i-kijun] + kijun_sen[i-kijun]) / 2
        if i >= kijun + senkou_b_p - 1:
            idx = i - kijun
            senkou_b[i] = (np.max(high[idx-senkou_b_p+1:idx+1]) + np.min(low[idx-senkou_b_p+1:idx+1])) / 2

    return tenkan_sen, kijun_sen, senkou_a, senkou_b


def compute_psar(high, low, close, af_start=0.02, af_step=0.02, af_max=0.20):
    """Compute Parabolic SAR from numpy arrays."""
    n = len(close)
    psar = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)
    af = af_start
    ep = high[0]; psar[0] = low[0]; direction[0] = 1

    for i in range(1, n):
        if direction[i-1] == 1:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            psar[i] = min(psar[i], low[i-1], low[max(0,i-2)])
            if low[i] < psar[i]:
                direction[i] = -1; psar[i] = ep; ep = low[i]; af = af_start
            else:
                direction[i] = 1
                if high[i] > ep: ep = high[i]; af = min(af + af_step, af_max)
        else:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            psar[i] = max(psar[i], high[i-1], high[max(0,i-2)])
            if high[i] > psar[i]:
                direction[i] = 1; psar[i] = ep; ep = high[i]; af = af_start
            else:
                direction[i] = -1
                if low[i] < ep: ep = low[i]; af = min(af + af_step, af_max)

    return psar, direction


def run_phase_B3(h1_df):
    """Vol Term Structure as independent strategy - FIXED."""
    print("\n" + "=" * 80)
    print("Phase B3 FIX: Vol Term Structure as Independent Strategy")
    print("=" * 80)

    h1 = h1_df.copy()
    close = h1['Close'].values; high = h1['High'].values; low = h1['Low'].values
    times = h1.index.values

    tr = np.maximum(high - low,
         np.maximum(np.abs(high - np.roll(close, 1)),
                    np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]

    atr14 = pd.Series(tr).rolling(14).mean().values

    for fast_p, slow_p in [(3, 30), (5, 50), (7, 70), (10, 100)]:
        atr_fast = pd.Series(tr).rolling(fast_p).mean().values
        atr_slow = pd.Series(tr).rolling(slow_p).mean().values
        ratio = np.where(atr_slow > 0, atr_fast / atr_slow, np.nan)

        # Signal: ratio crosses above 1.0 from below → trend start
        signals = [None] * len(close)
        for i in range(1, len(close)):
            if np.isnan(ratio[i]) or np.isnan(ratio[i-1]): continue
            if ratio[i] > 1.0 and ratio[i-1] <= 1.0:
                signals[i] = 'BUY' if close[i] > close[max(0,i-5)] else 'SELL'
            elif ratio[i] > 1.2 and ratio[i-1] <= 1.2:
                signals[i] = 'BUY' if close[i] > close[max(0,i-5)] else 'SELL'

        n_sig = sum(1 for s in signals if s is not None)
        r = backtest_generic_v2(close, high, low, atr14, times, signals,
                                f"VolTS_{fast_p}/{slow_p}_cross1.0",
                                sl_atr=3.5, tp_atr=8.0, max_hold=30)
        print(f"  ATR({fast_p}/{slow_p}) cross 1.0: signals={n_sig}, N={r['n']:>5}, "
              f"Sharpe={r['sharpe']:>7.2f}, PnL=${r['total_pnl']:>8.0f}, WR={r['win_rate']:>5.1f}%")

    # Also test: direction = trend of EMA, entry on vol expansion
    ema20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False).mean().values
    atr5 = pd.Series(tr).rolling(5).mean().values
    atr50 = pd.Series(tr).rolling(50).mean().values
    ratio_5_50 = np.where(atr50 > 0, atr5 / atr50, np.nan)

    for thresh in [1.0, 1.1, 1.2, 1.5]:
        signals = [None] * len(close)
        for i in range(1, len(close)):
            if np.isnan(ratio_5_50[i]) or np.isnan(ratio_5_50[i-1]): continue
            if ratio_5_50[i] > thresh and ratio_5_50[i-1] <= thresh:
                signals[i] = 'BUY' if ema20[i] > ema50[i] else 'SELL'

        n_sig = sum(1 for s in signals if s is not None)
        r = backtest_generic_v2(close, high, low, atr14, times, signals,
                                f"VolTS_EMA_thresh{thresh}",
                                sl_atr=3.5, tp_atr=8.0, max_hold=30)
        print(f"  Vol expand>{thresh:.1f} + EMA trend: signals={n_sig}, N={r['n']:>5}, "
              f"Sharpe={r['sharpe']:>7.2f}, PnL=${r['total_pnl']:>8.0f}, WR={r['win_rate']:>5.1f}%")


def run_phase_C(h1_df):
    """Test SuperTrend, Ichimoku, PSAR as independent strategies - FIXED."""
    print("\n" + "=" * 80)
    print("Phase C FIX: Alternative Trend Indicators as Independent Strategies")
    print("=" * 80)

    h1 = h1_df.copy()
    close = h1['Close'].values; high = h1['High'].values; low = h1['Low'].values
    times = h1.index.values

    tr = np.maximum(high - low,
         np.maximum(np.abs(high - np.roll(close, 1)),
                    np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr14 = pd.Series(tr).rolling(14).mean().values

    # C1: SuperTrend
    print(f"\n  --- C1: SuperTrend ---")
    print(f"  {'Period':>7} {'Factor':>7} {'Signals':>8} {'N':>5} {'Sharpe':>7} {'PnL':>9} {'WR':>5}")
    best_st = None

    for period in [7, 10, 14, 20]:
        for factor in [2.0, 2.5, 3.0, 4.0]:
            atr_p = pd.Series(tr).rolling(period).mean().values
            st_dir = compute_supertrend(high, low, close, atr_p, period=period, factor=factor)

            signals = [None] * len(close)
            for i in range(1, len(close)):
                if st_dir[i] == -1 and st_dir[i-1] == 1: signals[i] = 'BUY'
                elif st_dir[i] == 1 and st_dir[i-1] == -1: signals[i] = 'SELL'

            n_sig = sum(1 for s in signals if s is not None)
            r = backtest_generic_v2(close, high, low, atr14, times, signals,
                                    f"ST_{period}_{factor}",
                                    sl_atr=3.5, tp_atr=8.0, max_hold=50)
            print(f"  {period:>7} {factor:>7.1f} {n_sig:>8} {r['n']:>5} {r['sharpe']:>7.2f} "
                  f"${r['total_pnl']:>8.0f} {r['win_rate']:>4.1f}%")
            if best_st is None or r['sharpe'] > best_st['sharpe']:
                best_st = r

    # Also scan exit parameters for best SuperTrend
    if best_st and best_st['sharpe'] > 0:
        print(f"\n  Best SuperTrend: {best_st['label']}, Sharpe={best_st['sharpe']:.2f}")
        print(f"  Scanning SL/TP/MaxHold for best ST params...")
        best_label = best_st['label']
        parts = best_label.split('_')
        bp = int(parts[1]); bf = float(parts[2])
        atr_p = pd.Series(tr).rolling(bp).mean().values
        st_dir = compute_supertrend(high, low, close, atr_p, period=bp, factor=bf)
        signals = [None] * len(close)
        for i in range(1, len(close)):
            if st_dir[i] == -1 and st_dir[i-1] == 1: signals[i] = 'BUY'
            elif st_dir[i] == 1 and st_dir[i-1] == -1: signals[i] = 'SELL'

        print(f"  {'SL':>4} {'TP':>4} {'MH':>4} {'N':>5} {'Sharpe':>7} {'PnL':>9} {'WR':>5}")
        for sl in [2.0, 3.0, 3.5, 5.0]:
            for tp in [6.0, 8.0, 10.0, 12.0]:
                for mh in [30, 50, 80]:
                    r = backtest_generic_v2(close, high, low, atr14, times, signals,
                                            f"ST_opt", sl_atr=sl, tp_atr=tp, max_hold=mh)
                    if r['sharpe'] > 2.0:
                        print(f"  {sl:>4.1f} {tp:>4.0f} {mh:>4} {r['n']:>5} {r['sharpe']:>7.2f} "
                              f"${r['total_pnl']:>8.0f} {r['win_rate']:>4.1f}%")

    # C2: Ichimoku
    print(f"\n  --- C2: Ichimoku Cloud ---")
    print(f"  {'Tenkan':>7} {'Kijun':>6} {'Signals':>8} {'N':>5} {'Sharpe':>7} {'PnL':>9} {'WR':>5}")

    for tenkan, kijun in [(9, 26), (7, 22), (12, 30), (20, 52), (5, 13)]:
        ts, ks, sa, sb = compute_ichimoku(high, low, close, tenkan=tenkan, kijun=kijun)

        signals = [None] * len(close)
        for i in range(1, len(close)):
            if np.isnan(ts[i]) or np.isnan(ks[i]) or np.isnan(sa[i]) or np.isnan(sb[i]): continue
            if np.isnan(ts[i-1]) or np.isnan(ks[i-1]): continue
            cloud_top = max(sa[i], sb[i])
            cloud_bot = min(sa[i], sb[i])
            if ts[i] > ks[i] and ts[i-1] <= ks[i-1] and close[i] > cloud_top:
                signals[i] = 'BUY'
            elif ts[i] < ks[i] and ts[i-1] >= ks[i-1] and close[i] < cloud_bot:
                signals[i] = 'SELL'

        n_sig = sum(1 for s in signals if s is not None)
        r = backtest_generic_v2(close, high, low, atr14, times, signals,
                                f"Ichi_{tenkan}_{kijun}",
                                sl_atr=3.5, tp_atr=8.0, max_hold=50)
        print(f"  {tenkan:>7} {kijun:>6} {n_sig:>8} {r['n']:>5} {r['sharpe']:>7.2f} "
              f"${r['total_pnl']:>8.0f} {r['win_rate']:>4.1f}%")

    # C2b: Relaxed Ichimoku (no cloud requirement)
    print(f"\n  --- C2b: Ichimoku Relaxed (TK cross only) ---")
    for tenkan, kijun in [(9, 26), (7, 22), (5, 13)]:
        ts, ks, sa, sb = compute_ichimoku(high, low, close, tenkan=tenkan, kijun=kijun)
        signals = [None] * len(close)
        for i in range(1, len(close)):
            if np.isnan(ts[i]) or np.isnan(ks[i]) or np.isnan(ts[i-1]) or np.isnan(ks[i-1]): continue
            if ts[i] > ks[i] and ts[i-1] <= ks[i-1]: signals[i] = 'BUY'
            elif ts[i] < ks[i] and ts[i-1] >= ks[i-1]: signals[i] = 'SELL'

        n_sig = sum(1 for s in signals if s is not None)
        r = backtest_generic_v2(close, high, low, atr14, times, signals,
                                f"Ichi_TK_{tenkan}_{kijun}",
                                sl_atr=3.5, tp_atr=8.0, max_hold=50)
        print(f"  TK({tenkan}/{kijun}): signals={n_sig}, N={r['n']:>5}, "
              f"Sharpe={r['sharpe']:>7.2f}, PnL=${r['total_pnl']:>8.0f}, WR={r['win_rate']:>4.1f}%")

    # C3: Parabolic SAR
    print(f"\n  --- C3: Parabolic SAR ---")
    print(f"  {'AF_start':>9} {'AF_max':>7} {'Signals':>8} {'N':>5} {'Sharpe':>7} {'PnL':>9} {'WR':>5}")
    best_psar = None

    for af_s in [0.005, 0.01, 0.02, 0.03]:
        for af_m in [0.10, 0.20, 0.30, 0.50]:
            _, psar_dir = compute_psar(high, low, close, af_start=af_s, af_max=af_m)

            signals = [None] * len(close)
            for i in range(1, len(close)):
                if psar_dir[i] == 1 and psar_dir[i-1] == -1: signals[i] = 'BUY'
                elif psar_dir[i] == -1 and psar_dir[i-1] == 1: signals[i] = 'SELL'

            n_sig = sum(1 for s in signals if s is not None)
            r = backtest_generic_v2(close, high, low, atr14, times, signals,
                                    f"PSAR_{af_s}_{af_m}",
                                    sl_atr=3.5, tp_atr=8.0, max_hold=50)
            print(f"  {af_s:>9.3f} {af_m:>7.2f} {n_sig:>8} {r['n']:>5} {r['sharpe']:>7.2f} "
                  f"${r['total_pnl']:>8.0f} {r['win_rate']:>4.1f}%")
            if best_psar is None or r['sharpe'] > best_psar['sharpe']:
                best_psar = r

    # C4: Correlation of best indicators with L7
    print(f"\n  --- C4: Best indicator summary ---")
    if best_st: print(f"  Best SuperTrend: {best_st['label']}, Sharpe={best_st['sharpe']:.2f}")
    if best_psar: print(f"  Best PSAR: {best_psar['label']}, Sharpe={best_psar['sharpe']:.2f}")


def run_phase_A_kfold(data):
    """K-Fold validation for the best Multi-TF filter found in Phase A."""
    print("\n" + "=" * 80)
    print("Phase A-KF: K-Fold Validation of Best Multi-TF Filter")
    print("=" * 80)

    from backtest.runner import run_kfold
    h1_df = data.h1_df.copy()

    # Build H1 KC direction for EMA20/M2.0
    h1_kc = h1_df.copy()
    h1_kc['EMA20'] = h1_kc['Close'].ewm(span=20, adjust=False).mean()
    tr = pd.DataFrame({
        'hl': h1_kc['High'] - h1_kc['Low'],
        'hc': (h1_kc['High'] - h1_kc['Close'].shift(1)).abs(),
        'lc': (h1_kc['Low'] - h1_kc['Close'].shift(1)).abs(),
    }).max(axis=1)
    h1_kc['ATR14'] = tr.rolling(14).mean()
    h1_kc['KC_U'] = h1_kc['EMA20'] + 2.0 * h1_kc['ATR14']
    h1_kc['KC_L'] = h1_kc['EMA20'] - 2.0 * h1_kc['ATR14']
    h1_kc['kc_dir'] = 'NEUTRAL'
    h1_kc.loc[h1_kc['Close'] > h1_kc['KC_U'], 'kc_dir'] = 'BULL'
    h1_kc.loc[h1_kc['Close'] < h1_kc['KC_L'], 'kc_dir'] = 'BEAR'

    folds = [
        ("Fold1", "2015-01-01", "2017-01-01"),
        ("Fold2", "2017-01-01", "2019-01-01"),
        ("Fold3", "2019-01-01", "2021-01-01"),
        ("Fold4", "2021-01-01", "2023-01-01"),
        ("Fold5", "2023-01-01", "2025-01-01"),
        ("Fold6", "2025-01-01", "2026-04-01"),
    ]

    print(f"  Running 6-fold validation: H1 KC(EMA20/M2.0) same-dir filter")
    print(f"  {'Fold':>6} {'Base_Sh':>8} {'Filter_Sh':>10} {'Delta':>7} {'N_base':>7} {'N_filt':>7}")

    for fname, start, end in folds:
        base = run_variant(data, f"base_{fname}", verbose=False,
                          start_date=start, end_date=end, **L7_MH8)
        trades_base = base['_trades']
        sh_base = base['sharpe']

        kept = []
        for t in trades_base:
            et = t.entry_time if hasattr(t, 'entry_time') else t['entry_time']
            td = t.direction if hasattr(t, 'direction') else t['dir']
            et_ts = pd.Timestamp(et)
            h1_mask = h1_kc.index <= et_ts
            if not h1_mask.any(): continue
            kc_d = h1_kc.loc[h1_kc.index[h1_mask][-1], 'kc_dir']
            if (td == 'BUY' and kc_d == 'BULL') or (td == 'SELL' and kc_d == 'BEAR'):
                kept.append(t)

        daily = {}
        for t in kept:
            exit_t = t.exit_time if hasattr(t, 'exit_time') else t['exit_time']
            pnl = t.pnl if hasattr(t, 'pnl') else t['pnl']
            d = pd.Timestamp(exit_t).date()
            daily.setdefault(d, 0); daily[d] += pnl
        da = np.array(list(daily.values())) if daily else np.array([0])
        sh_filt = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0

        print(f"  {fname:>6} {sh_base:>8.2f} {sh_filt:>10.2f} {sh_filt-sh_base:>+7.2f} "
              f"{len(trades_base):>7} {len(kept):>7}")


def main():
    t0 = time.time()
    out_path = OUT_DIR / "R32b_output.txt"
    out = open(out_path, 'w', encoding='utf-8', buffering=1)
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, out)

    print(f"# R32-B3/C Fix: Alternative Trend Indicators & Vol TS")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    data = DataBundle.load_default()
    h1_df = data.h1_df.copy()

    for name, fn in [("B3", lambda: run_phase_B3(h1_df)),
                     ("C", lambda: run_phase_C(h1_df)),
                     ("A-KF", lambda: run_phase_A_kfold(data))]:
        try:
            fn()
            print(f"\n# Phase {name} completed at {datetime.now().strftime('%H:%M:%S')}")
            out.flush()
        except Exception as e:
            print(f"\n# Phase {name} FAILED: {e}")
            import traceback; traceback.print_exc()
            out.flush()

    elapsed = time.time() - t0
    print(f"\n# Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Elapsed: {elapsed/60:.1f} minutes")

    sys.stdout = old_stdout
    out.close()
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
