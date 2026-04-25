"""
R37: Higher-Timeframe Strategy Exploration
=============================================
A: H2/H4/D1 Mean Reversion — Bollinger Band & RSI extreme bounce
B: H2/H4/D1 Donchian Channel Breakout — N-period high/low breakout
C: H4/D1 Dual Moving Average Crossover — EMA fast/slow crossover with ADX gate
D: H4/D1 Momentum Regime — ROC + ADX regime-based trend following
E: Multi-TF Top-Down — D1 trend direction + H4 pullback entry
F: K-Fold validation of top strategies from A-E
G: Cross-correlation with existing L7 trades
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_FILE = "R37_output.txt"

class Tee:
    def __init__(self, fname):
        self.file = open(fname, 'w', buffering=1)
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()

sys.stdout = Tee(OUTPUT_FILE)
print(f"R37 started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════

from backtest.runner import load_csv, load_m15, load_h1_aligned

M15_PATH = Path("data/download/xauusd-m15-bid-2015-01-01-2026-04-10.csv")
H1_PATH = Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv")

print("\nLoading data...")
m15_df = load_m15(M15_PATH)
h1_df = load_h1_aligned(H1_PATH, m15_df.index[0])

def resample_ohlcv(df, rule):
    r = df.resample(rule).agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    return r

h2_df = resample_ohlcv(h1_df, '2h')
h4_df = resample_ohlcv(h1_df, '4h')
d1_df = resample_ohlcv(h1_df, '1D')

print(f"  H2: {len(h2_df)} bars, H4: {len(h4_df)} bars, D1: {len(d1_df)} bars")

def add_indicators(df):
    df = df.copy()
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs()
    }).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['RSI14'] = _calc_rsi(df['Close'], 14)
    df['RSI2'] = _calc_rsi(df['Close'], 2)
    df['ADX'] = _calc_adx(df, 14)
    for span in [9, 20, 25, 50, 100, 200]:
        df[f'EMA{span}'] = df['Close'].ewm(span=span).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_upper'] = df['SMA20'] + 2 * bb_std
    df['BB_lower'] = df['SMA20'] - 2 * bb_std
    df['BB_mid'] = df['SMA20']
    df['BB_pctb'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower']).replace(0, np.nan)
    kc_mid = df['Close'].ewm(span=20).mean()
    df['KC_upper'] = kc_mid + 2.0 * df['ATR']
    df['KC_lower'] = kc_mid - 2.0 * df['ATR']
    df['KC_mid'] = kc_mid
    df['squeeze'] = ((df['BB_upper'] < df['KC_upper']) & (df['BB_lower'] > df['KC_lower'])).astype(float)
    for n in [10, 20, 50]:
        df[f'Donchian_high_{n}'] = df['High'].rolling(n).max()
        df[f'Donchian_low_{n}'] = df['Low'].rolling(n).min()
    df['ROC5'] = df['Close'].pct_change(5) * 100
    df['ROC10'] = df['Close'].pct_change(10) * 100
    df['ROC20'] = df['Close'].pct_change(20) * 100
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    return df

def _calc_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _calc_adx(df, period):
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr = pd.DataFrame({
        'hl': high - low, 'hc': (high - close.shift(1)).abs(), 'lc': (low - close.shift(1)).abs()
    }).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(period).mean() / atr.replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.rolling(period).mean()

print("Adding indicators...")
h2_df = add_indicators(h2_df)
h4_df = add_indicators(h4_df)
d1_df = add_indicators(d1_df)
print("Indicators ready.")

# ═══════════════════════════════════════════════════════════════
# Generic backtest engine (single timeframe)
# ═══════════════════════════════════════════════════════════════

def backtest(df, label, signal_fn, sl_atr=3.5, tp_atr=8.0,
             trail_act_atr=0.28, trail_dist_atr=0.06, max_hold=50,
             spread=0.30, lot=0.03, cooldown=2):
    df = df.dropna(subset=['ATR'])
    trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    open_ = df['Open'].values; atr = df['ATR'].values
    times = df.index; n = len(df); last_exit = -999

    for i in range(1, n):
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (high[i] - pos['entry'] - spread) * lot * 100
                pnl_l = (low[i] - pos['entry'] - spread) * lot * 100
                pnl_c = (close[i] - pos['entry'] - spread) * lot * 100
            else:
                pnl_h = (pos['entry'] - low[i] - spread) * lot * 100
                pnl_l = (pos['entry'] - high[i] - spread) * lot * 100
                pnl_c = (pos['entry'] - close[i] - spread) * lot * 100
            tp_val = tp_atr * pos['atr'] * lot * 100
            sl_val = sl_atr * pos['atr'] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                _rec(trades, pos, close[i], times[i], "TP", i, tp_val); exited = True
            elif pnl_l <= -sl_val:
                _rec(trades, pos, close[i], times[i], "SL", i, -sl_val); exited = True
            else:
                ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and high[i] - pos['entry'] >= ad:
                    ts = high[i] - td
                    if low[i] <= ts:
                        _rec(trades, pos, close[i], times[i], "Trail", i,
                             (ts - pos['entry'] - spread) * lot * 100); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - low[i] >= ad:
                    ts = low[i] + td
                    if high[i] >= ts:
                        _rec(trades, pos, close[i], times[i], "Trail", i,
                             (pos['entry'] - ts - spread) * lot * 100); exited = True
                if not exited and held >= max_hold:
                    _rec(trades, pos, close[i], times[i], "Timeout", i, pnl_c); exited = True
            if exited: pos = None; last_exit = i; continue
        if pos is not None: continue
        if i - last_exit < cooldown: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        sig = signal_fn(df, i)
        if sig == 'BUY':
            entry_price = open_[i] + spread / 2 if i + 1 < n else close[i] + spread / 2
            pos = {'dir': 'BUY', 'entry': entry_price, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif sig == 'SELL':
            entry_price = open_[i] - spread / 2 if i + 1 < n else close[i] - spread / 2
            pos = {'dir': 'SELL', 'entry': entry_price, 'bar': i, 'time': times[i], 'atr': atr[i]}

    return _stats(trades, label)


def _rec(trades, pos, exit_price, exit_time, reason, bar_i, pnl):
    trades.append({
        'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_price,
        'entry_time': pos['time'], 'exit_time': exit_time,
        'pnl': pnl, 'exit_reason': reason, 'bars_held': bar_i - pos['bar'],
        'entry_atr': pos['atr']
    })


def _stats(trades, label):
    if not trades:
        return {'label': label, 'n': 0, 'sharpe': 0, 'total_pnl': 0, 'win_rate': 0,
                'max_dd': 0, '_trades': [], '_daily_pnl': {}}
    pnls = [t['pnl'] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    eq = np.cumsum(pnls); dd = (np.maximum.accumulate(eq + 2000) - (eq + 2000)).max()
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily.setdefault(d, 0); daily[d] += t['pnl']
    da = np.array(list(daily.values()))
    sh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0
    avg_pnl = np.mean(pnls)
    med_bars = np.median([t['bars_held'] for t in trades])
    return {'label': label, 'n': len(trades), 'sharpe': sh, 'total_pnl': sum(pnls),
            'win_rate': wins / len(trades) * 100, 'max_dd': dd, 'avg_pnl': avg_pnl,
            'med_bars': med_bars, '_trades': trades, '_daily_pnl': daily}


def print_result(r):
    print(f"  {r['label']:>40}: N={r['n']:>5}, Sharpe={r['sharpe']:>6.2f}, "
          f"PnL=${r['total_pnl']:>8.0f}, WR={r['win_rate']:>5.1f}%, "
          f"MaxDD=${r['max_dd']:>6.0f}, AvgPnL=${r.get('avg_pnl',0):>5.2f}, "
          f"MedBars={r.get('med_bars',0):>4.0f}")


def sharpe_from_daily(daily_dict):
    if not daily_dict: return 0
    da = np.array(list(daily_dict.values()))
    return da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0


# ═══════════════════════════════════════════════════════════════
# Phase A: Mean Reversion — BB + RSI extreme
# ═══════════════════════════════════════════════════════════════

def run_phase_A():
    print("\n" + "=" * 80)
    print("Phase A: Mean Reversion (Bollinger Band + RSI)")
    print("=" * 80)

    results = []
    for tf_name, df in [("H2", h2_df), ("H4", h4_df), ("D1", d1_df)]:
        for rsi_lo, rsi_hi in [(20, 80), (25, 75), (30, 70), (15, 85)]:
            def sig_fn(d, i, _lo=rsi_lo, _hi=rsi_hi):
                if i < 2: return None
                rsi = d['RSI14'].iloc[i]
                close = d['Close'].iloc[i]
                bb_lo = d['BB_lower'].iloc[i]
                bb_hi = d['BB_upper'].iloc[i]
                if np.isnan(rsi) or np.isnan(bb_lo): return None
                if rsi < _lo and close <= bb_lo: return 'BUY'
                if rsi > _hi and close >= bb_hi: return 'SELL'
                return None

            for sl, tp, mh in [(3.0, 4.0, 20), (2.5, 5.0, 30), (2.0, 3.0, 15), (3.5, 6.0, 40)]:
                label = f"{tf_name}_MR_RSI{rsi_lo}/{rsi_hi}_SL{sl}_TP{tp}_MH{mh}"
                r = backtest(df, label, sig_fn, sl_atr=sl, tp_atr=tp, max_hold=mh,
                             trail_act_atr=0.3, trail_dist_atr=0.08)
                if r['n'] > 10:
                    results.append(r)

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  Top 15 Mean Reversion strategies:")
    for r in results[:15]:
        print_result(r)

    # Also test RSI2 extreme (deep mean reversion)
    print(f"\n  --- RSI2 Deep Mean Reversion ---")
    rsi2_results = []
    for tf_name, df in [("H2", h2_df), ("H4", h4_df), ("D1", d1_df)]:
        for lo, hi in [(5, 95), (10, 90), (15, 85)]:
            def sig_fn2(d, i, _lo=lo, _hi=hi):
                if i < 2: return None
                r2 = d['RSI2'].iloc[i]
                if np.isnan(r2): return None
                if r2 < _lo: return 'BUY'
                if r2 > _hi: return 'SELL'
                return None
            for sl, tp, mh in [(2.0, 3.0, 10), (3.0, 5.0, 20), (1.5, 2.5, 8)]:
                label = f"{tf_name}_RSI2_{lo}/{hi}_SL{sl}_TP{tp}_MH{mh}"
                r = backtest(df, label, sig_fn2, sl_atr=sl, tp_atr=tp, max_hold=mh,
                             trail_act_atr=0.2, trail_dist_atr=0.05)
                if r['n'] > 10:
                    rsi2_results.append(r)

    rsi2_results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"  Top 10 RSI2 strategies:")
    for r in rsi2_results[:10]:
        print_result(r)

    return (results[:5] if results else []) + (rsi2_results[:5] if rsi2_results else [])


# ═══════════════════════════════════════════════════════════════
# Phase B: Donchian Channel Breakout
# ═══════════════════════════════════════════════════════════════

def run_phase_B():
    print("\n" + "=" * 80)
    print("Phase B: Donchian Channel Breakout")
    print("=" * 80)

    results = []
    for tf_name, df in [("H2", h2_df), ("H4", h4_df), ("D1", d1_df)]:
        for period in [10, 20, 50]:
            for adx_gate in [0, 18, 25]:
                def sig_fn(d, i, _p=period, _adx=adx_gate):
                    if i < 2: return None
                    ch = d[f'Donchian_high_{_p}'].iloc[i-1]
                    cl = d[f'Donchian_low_{_p}'].iloc[i-1]
                    c = d['Close'].iloc[i]
                    if np.isnan(ch) or np.isnan(cl): return None
                    if _adx > 0:
                        adx_val = d['ADX'].iloc[i]
                        if np.isnan(adx_val) or adx_val < _adx: return None
                    if c > ch: return 'BUY'
                    if c < cl: return 'SELL'
                    return None

                for sl, tp, mh in [(3.5, 8.0, 30), (2.5, 6.0, 20), (4.0, 10.0, 50),
                                    (3.0, 5.0, 15)]:
                    label = f"{tf_name}_Don{period}_ADX{adx_gate}_SL{sl}_TP{tp}_MH{mh}"
                    r = backtest(df, label, sig_fn, sl_atr=sl, tp_atr=tp, max_hold=mh,
                                 trail_act_atr=0.3, trail_dist_atr=0.06)
                    if r['n'] > 10:
                        results.append(r)

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  Top 15 Donchian Breakout strategies:")
    for r in results[:15]:
        print_result(r)
    return results[:5] if results else []


# ═══════════════════════════════════════════════════════════════
# Phase C: Dual EMA Crossover + ADX Gate
# ═══════════════════════════════════════════════════════════════

def run_phase_C():
    print("\n" + "=" * 80)
    print("Phase C: Dual EMA Crossover + ADX Gate")
    print("=" * 80)

    results = []
    for tf_name, df in [("H2", h2_df), ("H4", h4_df), ("D1", d1_df)]:
        for fast, slow in [(9, 25), (9, 50), (20, 50), (20, 100), (25, 100), (50, 200)]:
            for adx_gate in [0, 18, 25]:
                f_col = f'EMA{fast}'; s_col = f'EMA{slow}'
                if f_col not in df.columns or s_col not in df.columns: continue

                def sig_fn(d, i, _fc=f_col, _sc=s_col, _adx=adx_gate):
                    if i < 2: return None
                    f_now = d[_fc].iloc[i]; f_prev = d[_fc].iloc[i-1]
                    s_now = d[_sc].iloc[i]; s_prev = d[_sc].iloc[i-1]
                    if np.isnan(f_now) or np.isnan(s_now): return None
                    if _adx > 0:
                        adx_val = d['ADX'].iloc[i]
                        if np.isnan(adx_val) or adx_val < _adx: return None
                    if f_prev <= s_prev and f_now > s_now: return 'BUY'
                    if f_prev >= s_prev and f_now < s_now: return 'SELL'
                    return None

                for sl, tp, mh in [(3.5, 8.0, 30), (4.0, 10.0, 50), (3.0, 6.0, 20)]:
                    label = f"{tf_name}_EMAx{fast}/{slow}_ADX{adx_gate}_SL{sl}_TP{tp}_MH{mh}"
                    r = backtest(df, label, sig_fn, sl_atr=sl, tp_atr=tp, max_hold=mh,
                                 trail_act_atr=0.28, trail_dist_atr=0.06)
                    if r['n'] > 10:
                        results.append(r)

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  Top 15 EMA Crossover strategies:")
    for r in results[:15]:
        print_result(r)
    return results[:5] if results else []


# ═══════════════════════════════════════════════════════════════
# Phase D: Momentum Regime (ROC + ADX trend following)
# ═══════════════════════════════════════════════════════════════

def run_phase_D():
    print("\n" + "=" * 80)
    print("Phase D: Momentum / ROC Regime Trend Following")
    print("=" * 80)

    results = []
    for tf_name, df in [("H2", h2_df), ("H4", h4_df), ("D1", d1_df)]:
        # D1: ROC-based momentum + EMA trend alignment
        for roc_col, roc_thresh in [('ROC5', 1.0), ('ROC10', 2.0), ('ROC20', 3.0),
                                      ('ROC5', 0.5), ('ROC10', 1.0), ('ROC20', 2.0)]:
            for ema_trend in [50, 100, 200]:
                ema_col = f'EMA{ema_trend}'
                if ema_col not in df.columns: continue

                def sig_fn(d, i, _rc=roc_col, _rt=roc_thresh, _ec=ema_col):
                    if i < 2: return None
                    roc = d[_rc].iloc[i]; c = d['Close'].iloc[i]; ema = d[_ec].iloc[i]
                    if np.isnan(roc) or np.isnan(ema): return None
                    if roc > _rt and c > ema: return 'BUY'
                    if roc < -_rt and c < ema: return 'SELL'
                    return None

                for sl, tp, mh in [(3.5, 8.0, 30), (4.0, 10.0, 50), (3.0, 6.0, 20)]:
                    label = f"{tf_name}_ROC({roc_col}>{roc_thresh})_EMA{ema_trend}_SL{sl}_MH{mh}"
                    r = backtest(df, label, sig_fn, sl_atr=sl, tp_atr=tp, max_hold=mh,
                                 trail_act_atr=0.28, trail_dist_atr=0.06)
                    if r['n'] > 10:
                        results.append(r)

        # MACD histogram regime
        for adx_gate in [0, 18, 25]:
            def sig_macd(d, i, _adx=adx_gate):
                if i < 2: return None
                hist = d['MACD_hist'].iloc[i]; prev = d['MACD_hist'].iloc[i-1]
                if np.isnan(hist) or np.isnan(prev): return None
                if _adx > 0:
                    adx_val = d['ADX'].iloc[i]
                    if np.isnan(adx_val) or adx_val < _adx: return None
                if prev < 0 and hist > 0: return 'BUY'
                if prev > 0 and hist < 0: return 'SELL'
                return None

            for sl, tp, mh in [(3.5, 8.0, 30), (4.0, 10.0, 50)]:
                label = f"{tf_name}_MACDhist_ADX{adx_gate}_SL{sl}_MH{mh}"
                r = backtest(df, label, sig_macd, sl_atr=sl, tp_atr=tp, max_hold=mh,
                             trail_act_atr=0.28, trail_dist_atr=0.06)
                if r['n'] > 10:
                    results.append(r)

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  Top 15 Momentum/ROC strategies:")
    for r in results[:15]:
        print_result(r)
    return results[:5] if results else []


# ═══════════════════════════════════════════════════════════════
# Phase E: Multi-TF Top-Down (D1 trend + H4 pullback entry)
# ═══════════════════════════════════════════════════════════════

def run_phase_E():
    print("\n" + "=" * 80)
    print("Phase E: Multi-TF Top-Down (D1 direction + H4 pullback entry)")
    print("=" * 80)

    results = []
    d1_trend = d1_df[['Close', 'EMA50', 'EMA100', 'EMA200', 'KC_upper', 'KC_lower', 'ADX']].copy()

    for trend_method in ['ema50', 'ema100', 'ema200', 'kc']:
        for entry_method in ['rsi_pullback', 'bb_touch', 'ema_bounce']:
            for sl, tp, mh in [(3.5, 8.0, 20), (3.0, 6.0, 15), (4.0, 10.0, 30)]:

                def sig_fn(d, i, _tm=trend_method, _em=entry_method):
                    if i < 2: return None
                    c = d['Close'].iloc[i]; t = d.index[i]

                    d1_mask = d1_trend.index <= t
                    if not d1_mask.any(): return None
                    d1_row = d1_trend.loc[d1_trend.index[d1_mask][-1]]

                    if _tm == 'ema50':
                        trend = 'BULL' if d1_row['Close'] > d1_row['EMA50'] else 'BEAR'
                    elif _tm == 'ema100':
                        trend = 'BULL' if d1_row['Close'] > d1_row['EMA100'] else 'BEAR'
                    elif _tm == 'ema200':
                        if np.isnan(d1_row['EMA200']): return None
                        trend = 'BULL' if d1_row['Close'] > d1_row['EMA200'] else 'BEAR'
                    elif _tm == 'kc':
                        if d1_row['Close'] > d1_row['KC_upper']: trend = 'BULL'
                        elif d1_row['Close'] < d1_row['KC_lower']: trend = 'BEAR'
                        else: return None

                    if _em == 'rsi_pullback':
                        rsi = d['RSI14'].iloc[i]
                        if np.isnan(rsi): return None
                        if trend == 'BULL' and rsi < 40: return 'BUY'
                        if trend == 'BEAR' and rsi > 60: return 'SELL'
                    elif _em == 'bb_touch':
                        if np.isnan(d['BB_lower'].iloc[i]): return None
                        if trend == 'BULL' and c <= d['BB_lower'].iloc[i]: return 'BUY'
                        if trend == 'BEAR' and c >= d['BB_upper'].iloc[i]: return 'SELL'
                    elif _em == 'ema_bounce':
                        ema20 = d['EMA20'].iloc[i]
                        if np.isnan(ema20): return None
                        prev_c = d['Close'].iloc[i-1]
                        if trend == 'BULL' and prev_c < ema20 and c > ema20: return 'BUY'
                        if trend == 'BEAR' and prev_c > ema20 and c < ema20: return 'SELL'
                    return None

                label = f"D1({trend_method})+H4({entry_method})_SL{sl}_MH{mh}"
                r = backtest(h4_df, label, sig_fn, sl_atr=sl, tp_atr=tp, max_hold=mh,
                             trail_act_atr=0.28, trail_dist_atr=0.06)
                if r['n'] > 10:
                    results.append(r)

    # D1 trend + H2 entry
    for trend_method in ['ema100', 'kc']:
        for entry_method in ['rsi_pullback', 'bb_touch']:
            def sig_fn2(d, i, _tm=trend_method, _em=entry_method):
                if i < 2: return None
                c = d['Close'].iloc[i]; t = d.index[i]
                d1_mask = d1_trend.index <= t
                if not d1_mask.any(): return None
                d1_row = d1_trend.loc[d1_trend.index[d1_mask][-1]]
                if _tm == 'ema100':
                    trend = 'BULL' if d1_row['Close'] > d1_row['EMA100'] else 'BEAR'
                elif _tm == 'kc':
                    if d1_row['Close'] > d1_row['KC_upper']: trend = 'BULL'
                    elif d1_row['Close'] < d1_row['KC_lower']: trend = 'BEAR'
                    else: return None
                if _em == 'rsi_pullback':
                    rsi = d['RSI14'].iloc[i]
                    if np.isnan(rsi): return None
                    if trend == 'BULL' and rsi < 35: return 'BUY'
                    if trend == 'BEAR' and rsi > 65: return 'SELL'
                elif _em == 'bb_touch':
                    if np.isnan(d['BB_lower'].iloc[i]): return None
                    if trend == 'BULL' and c <= d['BB_lower'].iloc[i]: return 'BUY'
                    if trend == 'BEAR' and c >= d['BB_upper'].iloc[i]: return 'SELL'
                return None

            label = f"D1({trend_method})+H2({entry_method})_SL3.5_MH20"
            r = backtest(h2_df, label, sig_fn2, sl_atr=3.5, tp_atr=8.0, max_hold=20,
                         trail_act_atr=0.28, trail_dist_atr=0.06)
            if r['n'] > 10:
                results.append(r)

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  Top 15 Multi-TF Top-Down strategies:")
    for r in results[:15]:
        print_result(r)
    return results[:5] if results else []


# ═══════════════════════════════════════════════════════════════
# Phase F: K-Fold validation of top strategies from A-E
# ═══════════════════════════════════════════════════════════════

def run_phase_F(all_top_results):
    print("\n" + "=" * 80)
    print("Phase F: K-Fold Validation of Top Strategies")
    print("=" * 80)

    folds = [
        ("Fold1", "2015-01-01", "2017-01-01"), ("Fold2", "2017-01-01", "2019-01-01"),
        ("Fold3", "2019-01-01", "2021-01-01"), ("Fold4", "2021-01-01", "2023-01-01"),
        ("Fold5", "2023-01-01", "2025-01-01"), ("Fold6", "2025-01-01", "2026-04-01"),
    ]

    for r in all_top_results[:10]:
        label = r['label']
        trades = r['_trades']
        if not trades: continue

        print(f"\n  --- {label} (Full: Sharpe={r['sharpe']:.2f}, N={r['n']}) ---")
        pass_count = 0
        for fname, start, end in folds:
            start_d = pd.Timestamp(start, tz='UTC')
            end_d = pd.Timestamp(end, tz='UTC')
            fold_trades = [t for t in trades
                           if start_d <= pd.Timestamp(t['exit_time']) < end_d]
            if not fold_trades:
                print(f"    {fname}: no trades")
                continue
            fold_daily = {}
            for t in fold_trades:
                d = pd.Timestamp(t['exit_time']).date()
                fold_daily.setdefault(d, 0); fold_daily[d] += t['pnl']
            da = np.array(list(fold_daily.values()))
            sh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0
            pnl = sum(t['pnl'] for t in fold_trades)
            passed = sh > 0
            if passed: pass_count += 1
            print(f"    {fname}: Sharpe={sh:>6.2f}, PnL=${pnl:>7.0f}, N={len(fold_trades):>4} {'PASS' if passed else 'FAIL'}")
        print(f"    K-Fold: {pass_count}/6")


# ═══════════════════════════════════════════════════════════════
# Phase G: Cross-correlation with L7
# ═══════════════════════════════════════════════════════════════

def run_phase_G(all_top_results):
    print("\n" + "=" * 80)
    print("Phase G: Cross-Correlation with L7 Trades")
    print("=" * 80)

    from backtest.runner import DataBundle
    print("  Running L7(MH=8) baseline...")
    data = DataBundle(m15_df, h1_df, spread_model='fixed', spread_cost=0.30)
    from backtest.runner import run_variant
    LIVE_KWARGS = {
        'trailing_activate_atr': 0.28, 'trailing_distance_atr': 0.06,
        'sl_atr_mult': 3.5, 'tp_atr_mult': 8.0,
        'keltner_adx_threshold': 18, 'max_positions': 1,
        'intraday_adaptive': True, 'choppy_threshold': 0.50,
        'kc_only_threshold': 0.60, 'min_entry_gap_hours': 1.0,
        'regime_config': {
            'low_pct': 25, 'high_pct': 75,
            'low_trail': (0.10, 0.025), 'mid_trail': (0.20, 0.04),
            'high_trail': (0.28, 0.06),
        },
        'time_adaptive_trail': True, 'time_adaptive_trail_start': 2,
        'time_adaptive_trail_decay': 0.75, 'time_adaptive_trail_floor': 0.003,
        'keltner_max_hold_m15': 8,
    }
    l7_result = run_variant(data, "L7_MH8", verbose=False, **LIVE_KWARGS)
    l7_trades = l7_result['trades']
    l7_daily = {}
    for t in l7_trades:
        d = pd.Timestamp(t.exit_time).date()
        l7_daily.setdefault(d, 0); l7_daily[d] += t.pnl
    l7_daily_s = pd.Series(l7_daily).sort_index()
    l7_sh = sharpe_from_daily(l7_daily)
    print(f"  L7(MH=8): N={len(l7_trades)}, Sharpe={l7_sh:.2f}, PnL=${sum(t.pnl for t in l7_trades):.0f}")

    for r in all_top_results[:10]:
        if not r['_daily_pnl']: continue
        new_daily = pd.Series(r['_daily_pnl']).sort_index()
        all_idx = sorted(set(l7_daily_s.index) | set(new_daily.index))
        corr_df = pd.DataFrame({
            'L7': l7_daily_s.reindex(all_idx, fill_value=0),
            r['label']: new_daily.reindex(all_idx, fill_value=0),
        })
        corr = corr_df.corr().iloc[0, 1]

        combo_daily = corr_df['L7'] + corr_df[r['label']]
        combo_sh = combo_daily.mean() / combo_daily.std() * np.sqrt(252) if combo_daily.std() > 0 else 0
        combo_pnl = combo_daily.sum()

        print(f"  {r['label']:>45}: r={corr:>6.3f}, Combo Sharpe={combo_sh:>6.2f}, "
              f"Combo PnL=${combo_pnl:>8.0f}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()

    top_A = run_phase_A()
    print(f"\n  Phase A done: {time.time()-t0:.0f}s")

    top_B = run_phase_B()
    print(f"\n  Phase B done: {time.time()-t0:.0f}s")

    top_C = run_phase_C()
    print(f"\n  Phase C done: {time.time()-t0:.0f}s")

    top_D = run_phase_D()
    print(f"\n  Phase D done: {time.time()-t0:.0f}s")

    top_E = run_phase_E()
    print(f"\n  Phase E done: {time.time()-t0:.0f}s")

    all_top = top_A + top_B + top_C + top_D + top_E
    all_top.sort(key=lambda x: x['sharpe'], reverse=True)

    print("\n" + "=" * 80)
    print(f"GLOBAL TOP 20 (across all phases):")
    print("=" * 80)
    for r in all_top[:20]:
        print_result(r)

    run_phase_F(all_top)
    print(f"\n  Phase F done: {time.time()-t0:.0f}s")

    try:
        run_phase_G(all_top)
        print(f"\n  Phase G done: {time.time()-t0:.0f}s")
    except Exception as e:
        print(f"\n  Phase G skipped: {e}")

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"R37 completed in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}")
