#!/usr/bin/env python3
"""
R123 — Daily Timeframe Overlay Filter
=======================================
Construct D1 bars from H1 data. Use D1 indicators as filter for H1 entries.
Builds on R104 (TSMOM+D1_EMA20 showed Sharpe ~8.02).

Phase 1: Build D1 bars from H1
Phase 2: Compute D1 indicators (Ichimoku, EMA, KC, pivot levels)
Phase 3: Test each D1 filter per strategy
Phase 4: Combined best filters
Phase 5: K-Fold validation (5 folds)
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import run_variant, LIVE_PARITY_KWARGS, DataBundle

OUTPUT_DIR = Path("results/r123_daily_overlay")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.09, 'SESS_BO': 0.09}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']

FOLDS = [
    ("Fold1", "2015-01-01", "2017-03-01"),
    ("Fold2", "2017-03-01", "2019-06-01"),
    ("Fold3", "2019-06-01", "2021-09-01"),
    ("Fold4", "2021-09-01", "2023-12-01"),
    ("Fold5", "2023-12-01", "2027-01-01"),
]

t0 = time.time()

# ═══════════════════════════════════════════════════════════════
# H1 data loading
# ═══════════════════════════════════════════════════════════════

H1_CSV = Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv")

def load_h1():
    csv_path = H1_CSV
    if not csv_path.exists():
        import glob
        candidates = glob.glob("data/download/xauusd-h1-bid-*.csv")
        if candidates:
            csv_path = Path(sorted(candidates)[-1])
        else:
            raise FileNotFoundError("No xauusd H1 CSV found in data/download/")
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.set_index('time')[['open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[df['Close'] > 100]
    return df


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
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


def _compute_stats(trades):
    if not trades:
        return {'n_trades': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    eq = np.cumsum(daily)
    dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 0 else 0
    return {
        'n_trades': n,
        'sharpe': round(_sharpe(daily), 3),
        'pnl': round(sum(pnls), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'max_dd': round(dd, 2),
    }


# ═══════════════════════════════════════════════════════════════
# Strategy backtests (unfiltered)
# ═══════════════════════════════════════════════════════════════

def bt_psar(h1_df, spread, lot, maxloss_cap=0,
            sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04, max_hold=20):
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
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pdir[i-1] == -1 and pdir[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif pdir[i-1] == 1 and pdir[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom(h1_df, spread, lot, maxloss_cap=0,
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
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
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
        if score[i] > 0 and score[i-1] <= 0:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_sess_bo(h1_df, spread, lot, maxloss_cap=0,
               session_hour=12, lookback=4, sl_atr=4.5, tp_atr=4.0,
               trail_act=0.14, trail_dist=0.025, max_hold=20):
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
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i - j] for j in range(1, lookback + 1))
        ll = min(lo[i - j] for j in range(1, lookback + 1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_l8_max(data_bundle, spread, lot, maxloss_cap=35):
    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': maxloss_cap,
          'spread_cost': spread, 'initial_capital': 2000,
          'min_lot_size': lot, 'max_lot_size': lot}
    result = run_variant(data_bundle, "L8_MAX", verbose=False, **kw)
    raw_trades = result.get('_trades', [])
    trades = []
    for t in raw_trades:
        trades.append({
            'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
            'entry_time': t.entry_time, 'exit_time': t.exit_time,
            'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0,
        })
    return trades


# ═══════════════════════════════════════════════════════════════
# D1 bar construction & indicators
# ═══════════════════════════════════════════════════════════════

def build_d1_bars(h1_df):
    d1 = h1_df.resample('D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum',
    }).dropna()
    return d1


def compute_d1_indicators(d1):
    d1 = d1.copy()
    d1['EMA20'] = d1['Close'].ewm(span=20).mean()
    d1['EMA50'] = d1['Close'].ewm(span=50).mean()
    d1['SMA200'] = d1['Close'].rolling(200).mean()

    # Ichimoku
    d1['tenkan'] = (d1['High'].rolling(9).max() + d1['Low'].rolling(9).min()) / 2
    d1['kijun'] = (d1['High'].rolling(26).max() + d1['Low'].rolling(26).min()) / 2
    d1['senkou_a'] = ((d1['tenkan'] + d1['kijun']) / 2).shift(26)
    d1['senkou_b'] = ((d1['High'].rolling(52).max() + d1['Low'].rolling(52).min()) / 2).shift(26)

    # Keltner Channel (EMA20, 2.0*ATR14)
    tr = pd.concat([d1['High'] - d1['Low'],
                     (d1['High'] - d1['Close'].shift()).abs(),
                     (d1['Low'] - d1['Close'].shift()).abs()], axis=1).max(axis=1)
    d1['D1_ATR14'] = tr.rolling(14).mean()
    d1['KC_upper'] = d1['EMA20'] + 2.0 * d1['D1_ATR14']
    d1['KC_lower'] = d1['EMA20'] - 2.0 * d1['D1_ATR14']

    # ADX(14)
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

    # Weekly pivot levels (prior week H/L/C)
    weekly = d1.resample('W-FRI').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).shift(1)
    weekly.columns = ['wk_high', 'wk_low', 'wk_close']
    weekly['pivot'] = (weekly['wk_high'] + weekly['wk_low'] + weekly['wk_close']) / 3
    d1 = d1.join(weekly[['pivot', 'wk_high', 'wk_low']].resample('D').ffill(), how='left')
    d1['pivot'] = d1['pivot'].ffill()
    d1['wk_high'] = d1['wk_high'].ffill()
    d1['wk_low'] = d1['wk_low'].ffill()

    return d1


def build_d1_filter_dicts(d1):
    """Build date->value filter dicts for each D1 indicator."""
    filters = {}

    # D1_EMA20: +1 if close > EMA20, -1 if below
    ema20_dir = pd.Series(0, index=d1.index)
    ema20_dir[d1['Close'] > d1['EMA20']] = 1
    ema20_dir[d1['Close'] < d1['EMA20']] = -1
    filters['D1_EMA20'] = {k.date() if hasattr(k, 'date') else k: int(v) for k, v in ema20_dir.items()}

    # D1_SMA200: +1 if close > SMA200, -1 if below
    sma200_dir = pd.Series(0, index=d1.index)
    mask_up = d1['Close'] > d1['SMA200']
    mask_dn = d1['Close'] < d1['SMA200']
    sma200_dir[mask_up] = 1; sma200_dir[mask_dn] = -1
    filters['D1_SMA200'] = {k.date() if hasattr(k, 'date') else k: int(v) for k, v in sma200_dir.items()}

    # D1_Ichimoku: +1 if close above cloud, -1 if below
    ichi_dir = pd.Series(0, index=d1.index)
    cloud_top = d1[['senkou_a', 'senkou_b']].max(axis=1)
    cloud_bot = d1[['senkou_a', 'senkou_b']].min(axis=1)
    ichi_dir[d1['Close'] > cloud_top] = 1
    ichi_dir[d1['Close'] < cloud_bot] = -1
    filters['D1_Ichimoku'] = {k.date() if hasattr(k, 'date') else k: int(v) for k, v in ichi_dir.items()}

    # D1_KC: +1 if close > KC upper (breakout buy), -1 if close < KC lower
    kc_dir = pd.Series(0, index=d1.index)
    kc_dir[d1['Close'] > d1['KC_upper']] = 1
    kc_dir[d1['Close'] < d1['KC_lower']] = -1
    filters['D1_KC'] = {k.date() if hasattr(k, 'date') else k: int(v) for k, v in kc_dir.items()}

    # D1_ADX: allow trade (+1 neutral = both directions) when ADX > 20, else 0 (block)
    adx_filter = {}
    for idx_val in d1.index:
        dt = idx_val.date() if hasattr(idx_val, 'date') else idx_val
        adx_val = d1.loc[idx_val, 'ADX']
        adx_filter[dt] = 99 if (not np.isnan(adx_val) and adx_val > 20) else 0
    filters['D1_ADX'] = adx_filter

    return filters


# ═══════════════════════════════════════════════════════════════
# Filtered strategy backtests
# ═══════════════════════════════════════════════════════════════

def _apply_direction_filter(d1_filter, entry_date, direction):
    """Return True if trade should be allowed."""
    val = d1_filter.get(entry_date, 0)
    if val == 99:
        return True
    if val == 0:
        return True
    if direction == 'BUY' and val >= 1:
        return True
    if direction == 'SELL' and val <= -1:
        return True
    return False


def bt_psar_filtered(h1_df, spread, lot, d1_filter, maxloss_cap=0,
                     sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04, max_hold=20):
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
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        entry_date = times[i].date()
        if pdir[i-1] == -1 and pdir[i] == 1:
            if _apply_direction_filter(d1_filter, entry_date, 'BUY'):
                pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif pdir[i-1] == 1 and pdir[i] == -1:
            if _apply_direction_filter(d1_filter, entry_date, 'SELL'):
                pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom_filtered(h1_df, spread, lot, d1_filter, maxloss_cap=0,
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
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
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
            if _apply_direction_filter(d1_filter, entry_date, 'BUY'):
                pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            if _apply_direction_filter(d1_filter, entry_date, 'SELL'):
                pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_sess_bo_filtered(h1_df, spread, lot, d1_filter, maxloss_cap=0,
                        session_hour=12, lookback=4, sl_atr=4.5, tp_atr=4.0,
                        trail_act=0.14, trail_dist=0.025, max_hold=20):
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
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        entry_date = times[i].date()
        hh = max(h[i - j] for j in range(1, lookback + 1))
        ll = min(lo[i - j] for j in range(1, lookback + 1))
        if c[i] > hh:
            if _apply_direction_filter(d1_filter, entry_date, 'BUY'):
                pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            if _apply_direction_filter(d1_filter, entry_date, 'SELL'):
                pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80, flush=True)
    print("  R123 — Daily Timeframe Overlay Filter", flush=True)
    print("=" * 80, flush=True)

    # ── Load data ──
    print("\n  Loading H1 data...", flush=True)
    h1_df = load_h1()
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})", flush=True)

    print("  Loading L8_MAX DataBundle...", flush=True)
    bundle = DataBundle.load_custom()

    all_results = {}

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Build D1 bars from H1
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 1: Build D1 Bars from H1 Data", flush=True)
    print("=" * 70, flush=True)

    d1 = build_d1_bars(h1_df)
    print(f"    D1 bars: {len(d1)} ({d1.index[0].date()} ~ {d1.index[-1].date()})", flush=True)
    all_results['phase1'] = {'d1_bars': len(d1), 'start': str(d1.index[0].date()),
                             'end': str(d1.index[-1].date())}

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Compute D1 indicators
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: Compute D1 Indicators", flush=True)
    print("=" * 70, flush=True)

    d1 = compute_d1_indicators(d1)
    d1_filters = build_d1_filter_dicts(d1)

    for fname, fdata in d1_filters.items():
        vals = list(fdata.values())
        if fname == 'D1_ADX':
            active = sum(1 for v in vals if v == 99)
            blocked = sum(1 for v in vals if v == 0)
            print(f"    {fname}: active={active} ({active/len(vals)*100:.0f}%), "
                  f"blocked={blocked} ({blocked/len(vals)*100:.0f}%)", flush=True)
        else:
            up = sum(1 for v in vals if v > 0)
            dn = sum(1 for v in vals if v < 0)
            neutral = sum(1 for v in vals if v == 0)
            print(f"    {fname}: +1={up} ({up/len(vals)*100:.0f}%), "
                  f"-1={dn} ({dn/len(vals)*100:.0f}%), 0={neutral}", flush=True)

    sample_dates = sorted(d1.index)[-5:]
    print(f"\n    Sample (last 5 D1 bars):", flush=True)
    print(f"    {'Date':<12} {'EMA20':>8} {'SMA200':>8} {'Cloud_T':>8} {'KC_U':>8} {'ADX':>6}", flush=True)
    for dt in sample_dates:
        row = d1.loc[dt]
        adx_str = f"{row['ADX']:.1f}" if not np.isnan(row['ADX']) else "N/A"
        print(f"    {dt.date()}  {row['EMA20']:>8.1f} {row.get('SMA200', np.nan):>8.1f} "
              f"{row.get('senkou_a', np.nan):>8.1f} {row['KC_upper']:>8.1f} {adx_str:>6}", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Test each D1 filter per strategy
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 3: D1 Filter × Strategy Grid", flush=True)
    print("=" * 70, flush=True)

    h1_strats = {
        'PSAR': (bt_psar, bt_psar_filtered, CAPS['PSAR']),
        'TSMOM': (bt_tsmom, bt_tsmom_filtered, CAPS['TSMOM']),
        'SESS_BO': (bt_sess_bo, bt_sess_bo_filtered, CAPS['SESS_BO']),
    }

    phase3 = {}
    filter_scores = {f: 0.0 for f in d1_filters.keys()}

    for strat_name, (bt_fn, bt_fn_filt, cap) in h1_strats.items():
        print(f"\n  --- {strat_name} ---", flush=True)
        baseline = bt_fn(h1_df, SPREAD, UNIT_LOT, maxloss_cap=cap)
        bm = _compute_stats(baseline)
        print(f"    {'Unfiltered':<15}: n={bm['n_trades']:5d}  Sharpe={bm['sharpe']:7.3f}  "
              f"PnL=${bm['pnl']:9.0f}  WR={bm['wr']:.1f}%", flush=True)
        strat_res = {'baseline': bm}

        for fname, fdata in d1_filters.items():
            filtered = bt_fn_filt(h1_df, SPREAD, UNIT_LOT, fdata, maxloss_cap=cap)
            fm = _compute_stats(filtered)
            delta = fm['sharpe'] - bm['sharpe']
            marker = " ***" if delta > 0.1 else " *" if delta > 0.05 else ""
            print(f"    {fname:<15}: n={fm['n_trades']:5d}  Sharpe={fm['sharpe']:7.3f}  "
                  f"PnL=${fm['pnl']:9.0f}  WR={fm['wr']:.1f}%  (ΔSh={delta:+.3f}){marker}", flush=True)
            strat_res[fname] = fm
            filter_scores[fname] += delta

        phase3[strat_name] = strat_res

    # L8_MAX baseline (unfiltered only — uses backtest.runner)
    print(f"\n  --- L8_MAX (unfiltered baseline) ---", flush=True)
    l8_trades = bt_l8_max(bundle, SPREAD, UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])
    l8_stats = _compute_stats(l8_trades)
    print(f"    {'Unfiltered':<15}: n={l8_stats['n_trades']:5d}  Sharpe={l8_stats['sharpe']:7.3f}  "
          f"PnL=${l8_stats['pnl']:9.0f}  WR={l8_stats['wr']:.1f}%", flush=True)
    phase3['L8_MAX'] = {'baseline': l8_stats}

    all_results['phase3'] = phase3

    # Rank filters
    print(f"\n  Filter score totals (sum of Sharpe deltas across H1 strategies):", flush=True)
    sorted_filters = sorted(filter_scores.items(), key=lambda x: -x[1])
    best_filter_name = sorted_filters[0][0]
    second_filter_name = sorted_filters[1][0] if len(sorted_filters) > 1 else None
    for fname, score in sorted_filters:
        marker = " <-- BEST" if fname == best_filter_name else ""
        print(f"    {fname:<15}: {score:+.3f}{marker}", flush=True)
    all_results['filter_scores'] = {k: round(v, 3) for k, v in filter_scores.items()}

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Combine top 2 filters (intersection)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print(f"  Phase 4: Combined Top-2 Filters ({best_filter_name} ∩ {second_filter_name})", flush=True)
    print("=" * 70, flush=True)

    filt_a = d1_filters[best_filter_name]
    filt_b = d1_filters[second_filter_name]

    combined = {}
    for dt in set(list(filt_a.keys()) + list(filt_b.keys())):
        va = filt_a.get(dt, 0)
        vb = filt_b.get(dt, 0)
        if va == 99: va_dir = 0
        else: va_dir = va
        if vb == 99: vb_dir = 0
        else: vb_dir = vb
        if va_dir == vb_dir:
            combined[dt] = va_dir
        elif va_dir == 0:
            combined[dt] = vb_dir
        elif vb_dir == 0:
            combined[dt] = va_dir
        else:
            combined[dt] = 0

    phase4 = {}
    for strat_name, (bt_fn, bt_fn_filt, cap) in h1_strats.items():
        baseline = bt_fn(h1_df, SPREAD, UNIT_LOT, maxloss_cap=cap)
        bm = _compute_stats(baseline)
        filtered = bt_fn_filt(h1_df, SPREAD, UNIT_LOT, combined, maxloss_cap=cap)
        fm = _compute_stats(filtered)
        delta = fm['sharpe'] - bm['sharpe']
        print(f"    {strat_name:<10}: base Sharpe={bm['sharpe']:7.3f} -> combined Sharpe={fm['sharpe']:7.3f}  "
              f"(ΔSh={delta:+.3f})  n: {bm['n_trades']}->{fm['n_trades']}", flush=True)
        phase4[strat_name] = {'baseline': bm, 'combined': fm, 'delta_sharpe': round(delta, 3)}

    all_results['phase4'] = phase4

    # ═══════════════════════════════════════════════════════════
    # Phase 5: K-Fold Validation (5 folds)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 5: K-Fold Validation (5 folds)", flush=True)
    print("=" * 70, flush=True)

    test_filters = [best_filter_name, 'combined']
    filter_map = {best_filter_name: d1_filters[best_filter_name], 'combined': combined}

    kfold_results = {}
    for fname in test_filters:
        fdata = filter_map[fname]
        print(f"\n  --- Filter: {fname} ---", flush=True)
        kf_data = {}

        for strat_name, (bt_fn, bt_fn_filt, cap) in h1_strats.items():
            fold_sharpes_base = []
            fold_sharpes_filt = []

            for fold_name, start, end in FOLDS:
                fold_h1 = h1_df[(h1_df.index >= start) & (h1_df.index < end)]
                if len(fold_h1) < 200:
                    fold_sharpes_base.append(0.0); fold_sharpes_filt.append(0.0)
                    continue

                fold_d1 = build_d1_bars(fold_h1)
                if len(fold_d1) < 60:
                    fold_sharpes_base.append(0.0); fold_sharpes_filt.append(0.0)
                    continue

                fold_d1_ind = compute_d1_indicators(fold_d1)
                fold_filters = build_d1_filter_dicts(fold_d1_ind)

                if fname == 'combined':
                    fa = fold_filters.get(best_filter_name, {})
                    fb = fold_filters.get(second_filter_name, {})
                    fold_fdata = {}
                    for dt in set(list(fa.keys()) + list(fb.keys())):
                        va = fa.get(dt, 0); vb = fb.get(dt, 0)
                        if va == 99: va = 0
                        if vb == 99: vb = 0
                        if va == vb: fold_fdata[dt] = va
                        elif va == 0: fold_fdata[dt] = vb
                        elif vb == 0: fold_fdata[dt] = va
                        else: fold_fdata[dt] = 0
                else:
                    fold_fdata = fold_filters.get(fname, {})

                base_trades = bt_fn(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=cap)
                filt_trades = bt_fn_filt(fold_h1, SPREAD, UNIT_LOT, fold_fdata, maxloss_cap=cap)

                fold_sharpes_base.append(_compute_stats(base_trades)['sharpe'])
                fold_sharpes_filt.append(_compute_stats(filt_trades)['sharpe'])

            pos_base = sum(1 for s in fold_sharpes_base if s > 0)
            pos_filt = sum(1 for s in fold_sharpes_filt if s > 0)
            filt_wins = sum(1 for a, b in zip(fold_sharpes_filt, fold_sharpes_base) if a > b)

            print(f"    {strat_name:<10}: base={[round(s,2) for s in fold_sharpes_base]}  "
                  f"filt={[round(s,2) for s in fold_sharpes_filt]}  wins={filt_wins}/5", flush=True)

            kf_data[strat_name] = {
                'base_folds': [round(s, 3) for s in fold_sharpes_base],
                'filt_folds': [round(s, 3) for s in fold_sharpes_filt],
                'base_mean': round(float(np.mean(fold_sharpes_base)), 3),
                'filt_mean': round(float(np.mean(fold_sharpes_filt)), 3),
                'filt_wins': filt_wins,
                'base_positive': pos_base,
                'filt_positive': pos_filt,
            }

        kfold_results[fname] = kf_data

    all_results['phase5_kfold'] = kfold_results

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    print("\n" + "=" * 80, flush=True)
    print("  R123 FINAL SUMMARY", flush=True)
    print("=" * 80, flush=True)

    print(f"\n  Best filter: {best_filter_name}", flush=True)
    print(f"  Second filter: {second_filter_name}", flush=True)

    print(f"\n  Phase 3 — Individual filter Sharpe deltas:", flush=True)
    for fname, score in sorted_filters:
        print(f"    {fname:<15}: {score:+.3f}", flush=True)

    print(f"\n  Phase 4 — Combined filter results:", flush=True)
    for sn, sv in phase4.items():
        print(f"    {sn:<10}: ΔSharpe = {sv['delta_sharpe']:+.3f}", flush=True)

    print(f"\n  Phase 5 — K-Fold:", flush=True)
    for fname, kf_data in kfold_results.items():
        total_wins = sum(v['filt_wins'] for v in kf_data.values())
        total_possible = len(kf_data) * 5
        print(f"    {fname:<15}: filter wins {total_wins}/{total_possible} across all strategies", flush=True)

    all_results['elapsed_s'] = round(elapsed, 1)
    all_results['best_filter'] = best_filter_name
    all_results['second_filter'] = second_filter_name

    out_file = OUTPUT_DIR / "r123_results.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Saved: {out_file}", flush=True)
    print(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)", flush=True)
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
