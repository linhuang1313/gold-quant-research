#!/usr/bin/env python3
"""
R181 — Full Live Strategy Audit
================================
Comprehensive comparison of all 6 live strategies testing:
  - Research baseline (old params) vs Live actual (current config) vs Live+Filters
  - Session ADX (R178) for Keltner
  - PSAR skip hours {3,7,22} UTC
  - SESS_BO D1 EMA20 filter
  - Dual Thrust confirmed-bar + crossover vs current-bar
  - Chandelier RSI 30/70 vs EMA100 filter
  - Chandelier ATR period 14 vs 22
  - Keltner max_hold 2 vs 5 vs 20
  - 6-strategy combined portfolio comparison

All tests produce 4-era segmented results (Full, Hike, Cut, Recent 3Y).
"""
import sys, os, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import glob as _glob

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r181_full_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30

ERA_SEGMENTS = {
    'full':      None,
    'hike':      [("2015-12-01", "2019-01-01"), ("2022-03-01", "2023-08-01")],
    'cut':       [("2019-07-01", "2022-03-01"), ("2024-09-01", "2026-06-01")],
    'recent_3y': [("2023-06-01", "2026-06-01")],
}

LIVE_CONFIG = {
    'L8_MAX':      {'lot': 0.02, 'cap_atr_mult': 4.0},
    'PSAR':        {'lot': 0.09, 'cap_atr_mult': 4.5},
    'TSMOM':       {'lot': 0.15, 'cap_atr_mult': 6.5},
    'SESS_BO':     {'lot': 0.13, 'cap_atr_mult': 5.0},
    'DUAL_THRUST': {'lot': 0.04, 'cap_atr_mult': 5.0},
    'CHANDELIER':  {'lot': 0.08, 'cap_atr_mult': 5.0},
}


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


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


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
            return _mk(pos, c, times[i], "MaxLossCap", i, -cap_dollar)
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


# ═══════════════════════════════════════════════════════════════
# Session ADX map for Keltner (R178)
# ═══════════════════════════════════════════════════════════════
SESSION_ADX_MAP = {
    # UTC hour -> ADX threshold (None = block entry)
    **{h: 14 for h in range(0, 8)},    # Asia: ADX=14
    **{h: 10 for h in range(8, 13)},    # London: ADX=10
    **{h: 16 for h in range(13, 18)},   # NY: ADX=16
    **{h: 14 for h in range(18, 24)},   # Evening: ADX=14
}


def _get_utc_hour(ts):
    """Extract UTC hour from a pandas Timestamp."""
    if hasattr(ts, 'hour'):
        return ts.hour
    return pd.Timestamp(ts).hour


# ═══════════════════════════════════════════════════════════════
# Strategy backtest functions — all variants
# ═══════════════════════════════════════════════════════════════

def bt_l8_max(h1_df, spread, lot, cap_atr_mult=0,
              adx_th=14, ema_period=25, kc_mult=1.2,
              sl_atr=3.5, tp_atr=8.0,
              trail_act=0.14, trail_dist=0.025, max_hold=2,
              session_adx=False):
    """Keltner L8_MAX. session_adx=True uses R178 per-hour ADX thresholds."""
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df['ADX'] = compute_adx(df)
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['KC_mid'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    df['KC_upper'] = df['KC_mid'] + kc_mult * df['ATR']
    df['KC_lower'] = df['KC_mid'] - kc_mult * df['ATR']
    df = df.dropna(subset=['ATR', 'ADX', 'EMA100', 'KC_upper'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; adx = df['ADX'].values
    ema100 = df['EMA100'].values
    kc_u = df['KC_upper'].values; kc_l = df['KC_lower'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap_atr_mult)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1 or np.isnan(adx[i]): continue
        if session_adx:
            utc_h = _get_utc_hour(times[i])
            req_adx = SESSION_ADX_MAP.get(utc_h)
            if req_adx is None or adx[i] < req_adx:
                continue
        else:
            if adx[i] < adx_th: continue
        if c[i] > kc_u[i] and c[i] > ema100[i]:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < kc_l[i] and c[i] < ema100[i]:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_psar(h1_df, spread, lot, cap_atr_mult=0,
            sl_atr=4.0, tp_atr=6.0, trail_act=0.08, trail_dist=0.015, max_hold=15,
            skip_hours=None):
    """PSAR flip. skip_hours: set of UTC hours to block entry (e.g. {3,7,22})."""
    df = add_psar(h1_df).dropna(subset=['PSAR_dir', 'ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    pdir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap_atr_mult)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if skip_hours:
            utc_h = _get_utc_hour(times[i])
            if utc_h in skip_hours:
                continue
        if pdir[i-1] == -1 and pdir[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif pdir[i-1] == 1 and pdir[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom(h1_df, spread, lot, cap_atr_mult=0,
             fast=480, slow=720, sl_atr=6.0, tp_atr=8.0,
             trail_act=0.14, trail_dist=0.025, max_hold=12):
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
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap_atr_mult)
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


def bt_sess_bo(h1_df, spread, lot, cap_atr_mult=0,
               session_hour=12, lookback=4, sl_atr=4.5, tp_atr=4.0,
               trail_act=0.14, trail_dist=0.025, max_hold=20,
               d1_ema20_filter=False):
    """Session Breakout. d1_ema20_filter=True blocks trades against D1 EMA20 direction."""
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    d1_dir = None
    if d1_ema20_filter:
        daily = df['Close'].resample('D').last().dropna()
        d1_ema20 = daily.ewm(span=20, adjust=False).mean()
        d1_direction = pd.Series(np.where(daily > d1_ema20, 1, -1), index=daily.index)
        d1_dir = d1_direction.reindex(df.index, method='ffill')
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(lookback, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap_atr_mult)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if hours[i] != session_hour: continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i-j] for j in range(1, lookback + 1))
        ll = min(lo[i-j] for j in range(1, lookback + 1))
        signal = None
        if c[i] > hh: signal = 'BUY'
        elif c[i] < ll: signal = 'SELL'
        if signal is None: continue
        if d1_ema20_filter and d1_dir is not None:
            d1_val = d1_dir.iloc[i] if i < len(d1_dir) else 0
            if signal == 'BUY' and d1_val == -1: continue
            if signal == 'SELL' and d1_val == 1: continue
        if signal == 'BUY':
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        else:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_dual_thrust(h1_df, spread, lot, cap_atr_mult=0,
                   n_bars=6, k=0.5, sl_atr=4.5, tp_atr=8.0,
                   trail_act=0.14, trail_dist=0.025, max_hold=20,
                   confirmed_bar=False):
    """Dual Thrust. confirmed_bar=True uses crossover on completed bar (like live)."""
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    o = df['Open'].values; atr = df['ATR'].values
    times = df.index; n = len(df)
    dates = pd.Series(df.index.date, index=df.index)
    daily_open = df.groupby(dates)['Open'].transform('first').values
    trades = []; pos = None; last_exit = -999
    for i in range(n_bars + 1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap_atr_mult)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if confirmed_bar:
            hh = np.max(h[i - n_bars - 1:i - 1])
            lc_v = np.min(c[i - n_bars - 1:i - 1])
            hc_v = np.max(c[i - n_bars - 1:i - 1])
            ll = np.min(lo[i - n_bars - 1:i - 1])
            rng = max(hh - lc_v, hc_v - ll)
            do_cur = daily_open[i - 1]
            do_prev = daily_open[i - 2] if i >= 2 else do_cur
            hh_prev = np.max(h[i - n_bars - 2:i - 2]) if i >= n_bars + 2 else 0
            lc_prev = np.min(c[i - n_bars - 2:i - 2]) if i >= n_bars + 2 else 0
            hc_prev = np.max(c[i - n_bars - 2:i - 2]) if i >= n_bars + 2 else 0
            ll_prev = np.min(lo[i - n_bars - 2:i - 2]) if i >= n_bars + 2 else 0
            rng_prev = max(hh_prev - lc_prev, hc_prev - ll_prev) if i >= n_bars + 2 else 0
            up_now = c[i-1] > do_cur + k * rng
            up_prev = c[i-2] > do_prev + k * rng_prev if rng_prev > 0 else False
            dn_now = c[i-1] < do_cur - k * rng
            dn_prev = c[i-2] < do_prev - k * rng_prev if rng_prev > 0 else False
            if up_now and not up_prev:
                pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
            elif dn_now and not dn_prev:
                pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        else:
            hh = np.max(h[i - n_bars:i])
            lc_v = np.min(c[i - n_bars:i])
            hc_v = np.max(c[i - n_bars:i])
            ll = np.min(lo[i - n_bars:i])
            rng = max(hh - lc_v, hc_v - ll)
            buy_line = o[i] + k * rng
            sell_line = o[i] - k * rng
            if c[i] > buy_line:
                pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
            elif c[i] < sell_line:
                pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_chandelier(h1_df, spread, lot, cap_atr_mult=0,
                  period=22, mult=3.0, sl_atr=4.5, tp_atr=8.0,
                  trail_act=0.14, trail_dist=0.025, max_hold=20,
                  filter_type='ema100', atr_period_for_lines=None):
    """Chandelier Exit Flip.
    filter_type: 'ema100' (research default) or 'rsi3070' (live R176e/f)
    atr_period_for_lines: period for ATR used in chandelier lines (None=same as `period`)
    """
    atr_p = atr_period_for_lines if atr_period_for_lines else period
    df = h1_df.copy()
    df['ATR_lines'] = compute_atr(df, period=atr_p)
    df['ATR'] = compute_atr(df, period=14)
    df['EMA'] = df['Close'].ewm(span=100, adjust=False).mean()
    if filter_type == 'rsi3070':
        df['RSI14'] = compute_rsi(df['Close'], 14)
    df = df.dropna(subset=['ATR_lines', 'ATR', 'EMA'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr_lines = df['ATR_lines'].values; atr = df['ATR'].values
    ema = df['EMA'].values
    rsi = df['RSI14'].values if 'RSI14' in df.columns else None
    times = df.index; n = len(df)
    chandelier_long = np.full(n, np.nan)
    chandelier_short = np.full(n, np.nan)
    for i in range(period, n):
        hh = np.max(h[i - period + 1:i + 1])
        ll = np.min(lo[i - period + 1:i + 1])
        chandelier_long[i] = hh - mult * atr_lines[i]
        chandelier_short[i] = ll + mult * atr_lines[i]
    direction = np.zeros(n)
    for i in range(period + 1, n):
        if np.isnan(chandelier_long[i]) or np.isnan(chandelier_short[i]):
            direction[i] = direction[i-1]; continue
        if c[i] > chandelier_short[i-1]:
            direction[i] = 1
        elif c[i] < chandelier_long[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
    trades = []; pos = None; last_exit = -999
    for i in range(period + 2, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap_atr_mult)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        flip_bull = direction[i] == 1 and direction[i-1] != 1
        flip_bear = direction[i] == -1 and direction[i-1] != -1
        if not flip_bull and not flip_bear:
            continue
        if filter_type == 'ema100':
            if flip_bull and c[i] <= ema[i]: continue
            if flip_bear and c[i] >= ema[i]: continue
        elif filter_type == 'rsi3070' and rsi is not None:
            if flip_bull and not np.isnan(rsi[i]) and rsi[i] > 70: continue
            if flip_bear and not np.isnan(rsi[i]) and rsi[i] < 30: continue
        if flip_bull:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif flip_bear:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


# ═══════════════════════════════════════════════════════════════
# Stats + era filtering
# ═══════════════════════════════════════════════════════════════

def _trades_to_daily(trades):
    if not trades: return pd.Series(dtype=float)
    df = pd.DataFrame(trades)
    df['date'] = pd.to_datetime(df['entry_time']).dt.date
    return df.groupby('date')['pnl'].sum()


def _sharpe(daily):
    if len(daily) < 10: return 0
    return float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0


def _max_dd(daily):
    if len(daily) == 0: return 0
    eq = daily.cumsum()
    return float((np.maximum.accumulate(eq) - eq).max())


def compute_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0, 'cap_pct': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    reasons = [t.get('reason', '') for t in trades]
    cap_hits = sum(1 for r in reasons if r == 'MaxLossCap')
    return {
        'n': n,
        'sharpe': round(_sharpe(daily), 3),
        'pnl': round(sum(pnls), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'max_dd': round(_max_dd(daily), 2),
        'cap_pct': round(cap_hits / n * 100, 1) if n > 0 else 0,
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
                filtered.append(t)
                break
    return filtered


def run_era_stats(trades):
    """Return dict of era -> stats."""
    result = {}
    for era in ['full', 'hike', 'cut', 'recent_3y']:
        era_trades = filter_trades_by_era(trades, era)
        result[era] = compute_stats(era_trades)
    return result


def print_era_table(label, era_stats):
    print(f"\n  {label}", flush=True)
    print(f"  {'Era':<12} {'N':>5} {'Sharpe':>8} {'PnL':>10} {'WR%':>6} {'MaxDD':>8} {'Cap%':>5}", flush=True)
    print(f"  {'-'*12} {'-'*5} {'-'*8} {'-'*10} {'-'*6} {'-'*8} {'-'*5}", flush=True)
    for era in ['full', 'hike', 'cut', 'recent_3y']:
        s = era_stats[era]
        pnl_str = f"${s['pnl']:>9,.0f}" if s['pnl'] >= 0 else f"-${abs(s['pnl']):>8,.0f}"
        dd_str = f"${s['max_dd']:>7,.0f}"
        print(f"  {era:<12} {s['n']:>5} {s['sharpe']:>8.3f} {pnl_str} {s['wr']:>5.1f}% {dd_str} {s['cap_pct']:>4.1f}%", flush=True)


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# Test configurations
# ═══════════════════════════════════════════════════════════════

def define_tests():
    """Return list of (test_name, strategy_key, bt_func, kwargs)."""
    tests = []

    # --- Test 1: Keltner Session ADX ---
    kc = LIVE_CONFIG['L8_MAX']
    tests.append(("KC_flat_ADX14", "L8_MAX", bt_l8_max,
                  dict(lot=kc['lot'], cap_atr_mult=kc['cap_atr_mult'],
                       adx_th=14, session_adx=False, max_hold=2)))
    tests.append(("KC_session_ADX_R178", "L8_MAX", bt_l8_max,
                  dict(lot=kc['lot'], cap_atr_mult=kc['cap_atr_mult'],
                       adx_th=14, session_adx=True, max_hold=2)))

    # --- Test 2: Keltner max_hold ---
    tests.append(("KC_MH2", "L8_MAX", bt_l8_max,
                  dict(lot=kc['lot'], cap_atr_mult=kc['cap_atr_mult'], max_hold=2)))
    tests.append(("KC_MH5", "L8_MAX", bt_l8_max,
                  dict(lot=kc['lot'], cap_atr_mult=kc['cap_atr_mult'], max_hold=5)))
    tests.append(("KC_MH20", "L8_MAX", bt_l8_max,
                  dict(lot=kc['lot'], cap_atr_mult=kc['cap_atr_mult'], max_hold=20)))

    # --- Test 3: PSAR skip hours ---
    ps = LIVE_CONFIG['PSAR']
    tests.append(("PSAR_no_skip", "PSAR", bt_psar,
                  dict(lot=ps['lot'], cap_atr_mult=ps['cap_atr_mult'], skip_hours=None)))
    tests.append(("PSAR_skip_3_7_22", "PSAR", bt_psar,
                  dict(lot=ps['lot'], cap_atr_mult=ps['cap_atr_mult'], skip_hours={3, 7, 22})))

    # --- Test 4: PSAR old vs new params ---
    tests.append(("PSAR_old_params", "PSAR", bt_psar,
                  dict(lot=ps['lot'], cap_atr_mult=ps['cap_atr_mult'],
                       sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04, max_hold=20)))
    tests.append(("PSAR_live_params", "PSAR", bt_psar,
                  dict(lot=ps['lot'], cap_atr_mult=ps['cap_atr_mult'],
                       sl_atr=4.0, tp_atr=6.0, trail_act=0.08, trail_dist=0.015, max_hold=15)))

    # --- Test 5: TSMOM old vs new params ---
    ts = LIVE_CONFIG['TSMOM']
    tests.append(("TSMOM_old_params", "TSMOM", bt_tsmom,
                  dict(lot=ts['lot'], cap_atr_mult=ts['cap_atr_mult'],
                       sl_atr=4.5, tp_atr=6.0, max_hold=20)))
    tests.append(("TSMOM_live_params", "TSMOM", bt_tsmom,
                  dict(lot=ts['lot'], cap_atr_mult=ts['cap_atr_mult'],
                       sl_atr=6.0, tp_atr=8.0, max_hold=12)))

    # --- Test 6: SESS_BO D1 EMA20 filter ---
    sb = LIVE_CONFIG['SESS_BO']
    tests.append(("SESSBO_no_d1", "SESS_BO", bt_sess_bo,
                  dict(lot=sb['lot'], cap_atr_mult=sb['cap_atr_mult'], d1_ema20_filter=False)))
    tests.append(("SESSBO_d1_ema20", "SESS_BO", bt_sess_bo,
                  dict(lot=sb['lot'], cap_atr_mult=sb['cap_atr_mult'], d1_ema20_filter=True)))

    # --- Test 7: Dual Thrust signal timing ---
    dt = LIVE_CONFIG['DUAL_THRUST']
    tests.append(("DT_current_bar", "DUAL_THRUST", bt_dual_thrust,
                  dict(lot=dt['lot'], cap_atr_mult=dt['cap_atr_mult'], confirmed_bar=False)))
    tests.append(("DT_confirmed_bar", "DUAL_THRUST", bt_dual_thrust,
                  dict(lot=dt['lot'], cap_atr_mult=dt['cap_atr_mult'], confirmed_bar=True)))

    # --- Test 8: Chandelier filter type ---
    ch = LIVE_CONFIG['CHANDELIER']
    tests.append(("CH_ema100_atr22", "CHANDELIER", bt_chandelier,
                  dict(lot=ch['lot'], cap_atr_mult=ch['cap_atr_mult'],
                       filter_type='ema100', atr_period_for_lines=22)))
    tests.append(("CH_rsi3070_atr22", "CHANDELIER", bt_chandelier,
                  dict(lot=ch['lot'], cap_atr_mult=ch['cap_atr_mult'],
                       filter_type='rsi3070', atr_period_for_lines=22)))

    # --- Test 9: Chandelier ATR period for lines ---
    tests.append(("CH_atr14_ema100", "CHANDELIER", bt_chandelier,
                  dict(lot=ch['lot'], cap_atr_mult=ch['cap_atr_mult'],
                       filter_type='ema100', atr_period_for_lines=14)))
    tests.append(("CH_atr22_ema100", "CHANDELIER", bt_chandelier,
                  dict(lot=ch['lot'], cap_atr_mult=ch['cap_atr_mult'],
                       filter_type='ema100', atr_period_for_lines=22)))

    # --- Test 10: Chandelier live config (RSI + ATR14 for lines, matching trading code) ---
    tests.append(("CH_live_rsi_atr14", "CHANDELIER", bt_chandelier,
                  dict(lot=ch['lot'], cap_atr_mult=ch['cap_atr_mult'],
                       filter_type='rsi3070', atr_period_for_lines=14)))

    return tests


# ═══════════════════════════════════════════════════════════════
# Portfolio test: combine all 6 strategies
# ═══════════════════════════════════════════════════════════════

def run_portfolio_test(h1_df, config_label, strat_configs):
    """Run all 6 strategies and merge trades for portfolio-level stats.
    strat_configs: list of (strat_name, bt_func, kwargs)
    """
    all_trades = []
    per_strat = {}
    for strat_name, bt_func, kwargs in strat_configs:
        trades = bt_func(h1_df, spread=SPREAD, **kwargs)
        per_strat[strat_name] = trades
        all_trades.extend(trades)
    all_trades.sort(key=lambda t: t['entry_time'])
    return all_trades, per_strat


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 90, flush=True)
    print("  R181 — Full Live Strategy Audit", flush=True)
    print("=" * 90, flush=True)

    h1_df = load_h1()

    # ──────────────────────────────────────────────
    # Phase 1: Individual strategy A/B tests
    # ──────────────────────────────────────────────
    print(f"\n{'=' * 90}", flush=True)
    print("  PHASE 1: Individual Strategy A/B Tests", flush=True)
    print(f"{'=' * 90}", flush=True)

    tests = define_tests()
    all_results = {}

    for test_name, strat_key, bt_func, kwargs in tests:
        print(f"\n  Running: {test_name} ...", end="", flush=True)
        trades = bt_func(h1_df, spread=SPREAD, **kwargs)
        era_stats = run_era_stats(trades)
        all_results[test_name] = {'strategy': strat_key, 'era_stats': era_stats, 'kwargs': str(kwargs)}
        s = era_stats['full']
        print(f" N={s['n']}, Sharpe={s['sharpe']:.3f}, PnL=${s['pnl']:,.0f}", flush=True)

    # Print grouped comparison tables
    groups = [
        ("Keltner: Session ADX", ["KC_flat_ADX14", "KC_session_ADX_R178"]),
        ("Keltner: Max Hold", ["KC_MH2", "KC_MH5", "KC_MH20"]),
        ("PSAR: Skip Hours", ["PSAR_no_skip", "PSAR_skip_3_7_22"]),
        ("PSAR: Old vs Live Params", ["PSAR_old_params", "PSAR_live_params"]),
        ("TSMOM: Old vs Live Params", ["TSMOM_old_params", "TSMOM_live_params"]),
        ("SESS_BO: D1 EMA20 Filter", ["SESSBO_no_d1", "SESSBO_d1_ema20"]),
        ("Dual Thrust: Signal Timing", ["DT_current_bar", "DT_confirmed_bar"]),
        ("Chandelier: Filter Type", ["CH_ema100_atr22", "CH_rsi3070_atr22"]),
        ("Chandelier: ATR Period for Lines", ["CH_atr14_ema100", "CH_atr22_ema100"]),
        ("Chandelier: Live Config", ["CH_live_rsi_atr14", "CH_rsi3070_atr22"]),
    ]

    for group_name, test_names in groups:
        print(f"\n{'=' * 90}", flush=True)
        print(f"  {group_name}", flush=True)
        print(f"{'=' * 90}", flush=True)
        for tn in test_names:
            if tn in all_results:
                print_era_table(tn, all_results[tn]['era_stats'])

    # ──────────────────────────────────────────────
    # Phase 2: Portfolio comparison
    # ──────────────────────────────────────────────
    print(f"\n\n{'=' * 90}", flush=True)
    print("  PHASE 2: 6-Strategy Portfolio Comparison", flush=True)
    print(f"{'=' * 90}", flush=True)

    lc = LIVE_CONFIG

    # Portfolio A: Research baseline (old params, no filters)
    portfolio_a = [
        ("L8_MAX", bt_l8_max, dict(lot=lc['L8_MAX']['lot'], cap_atr_mult=lc['L8_MAX']['cap_atr_mult'],
                                    max_hold=20, session_adx=False)),
        ("PSAR", bt_psar, dict(lot=lc['PSAR']['lot'], cap_atr_mult=lc['PSAR']['cap_atr_mult'],
                                sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04, max_hold=20)),
        ("TSMOM", bt_tsmom, dict(lot=lc['TSMOM']['lot'], cap_atr_mult=lc['TSMOM']['cap_atr_mult'],
                                  sl_atr=4.5, tp_atr=6.0, max_hold=20)),
        ("SESS_BO", bt_sess_bo, dict(lot=lc['SESS_BO']['lot'], cap_atr_mult=lc['SESS_BO']['cap_atr_mult'],
                                      d1_ema20_filter=False)),
        ("DUAL_THRUST", bt_dual_thrust, dict(lot=lc['DUAL_THRUST']['lot'], cap_atr_mult=lc['DUAL_THRUST']['cap_atr_mult'],
                                              confirmed_bar=False)),
        ("CHANDELIER", bt_chandelier, dict(lot=lc['CHANDELIER']['lot'], cap_atr_mult=lc['CHANDELIER']['cap_atr_mult'],
                                            filter_type='ema100', atr_period_for_lines=22)),
    ]

    # Portfolio B: Live actual params, no extra filters
    portfolio_b = [
        ("L8_MAX", bt_l8_max, dict(lot=lc['L8_MAX']['lot'], cap_atr_mult=lc['L8_MAX']['cap_atr_mult'],
                                    max_hold=2, session_adx=False)),
        ("PSAR", bt_psar, dict(lot=lc['PSAR']['lot'], cap_atr_mult=lc['PSAR']['cap_atr_mult'],
                                sl_atr=4.0, tp_atr=6.0, trail_act=0.08, trail_dist=0.015, max_hold=15)),
        ("TSMOM", bt_tsmom, dict(lot=lc['TSMOM']['lot'], cap_atr_mult=lc['TSMOM']['cap_atr_mult'],
                                  sl_atr=6.0, tp_atr=8.0, max_hold=12)),
        ("SESS_BO", bt_sess_bo, dict(lot=lc['SESS_BO']['lot'], cap_atr_mult=lc['SESS_BO']['cap_atr_mult'],
                                      d1_ema20_filter=False)),
        ("DUAL_THRUST", bt_dual_thrust, dict(lot=lc['DUAL_THRUST']['lot'], cap_atr_mult=lc['DUAL_THRUST']['cap_atr_mult'],
                                              confirmed_bar=False)),
        ("CHANDELIER", bt_chandelier, dict(lot=lc['CHANDELIER']['lot'], cap_atr_mult=lc['CHANDELIER']['cap_atr_mult'],
                                            filter_type='ema100', atr_period_for_lines=22)),
    ]

    # Portfolio C: Live actual + all live-only filters
    portfolio_c = [
        ("L8_MAX", bt_l8_max, dict(lot=lc['L8_MAX']['lot'], cap_atr_mult=lc['L8_MAX']['cap_atr_mult'],
                                    max_hold=2, session_adx=True)),
        ("PSAR", bt_psar, dict(lot=lc['PSAR']['lot'], cap_atr_mult=lc['PSAR']['cap_atr_mult'],
                                sl_atr=4.0, tp_atr=6.0, trail_act=0.08, trail_dist=0.015, max_hold=15,
                                skip_hours={3, 7, 22})),
        ("TSMOM", bt_tsmom, dict(lot=lc['TSMOM']['lot'], cap_atr_mult=lc['TSMOM']['cap_atr_mult'],
                                  sl_atr=6.0, tp_atr=8.0, max_hold=12)),
        ("SESS_BO", bt_sess_bo, dict(lot=lc['SESS_BO']['lot'], cap_atr_mult=lc['SESS_BO']['cap_atr_mult'],
                                      d1_ema20_filter=True)),
        ("DUAL_THRUST", bt_dual_thrust, dict(lot=lc['DUAL_THRUST']['lot'], cap_atr_mult=lc['DUAL_THRUST']['cap_atr_mult'],
                                              confirmed_bar=True)),
        ("CHANDELIER", bt_chandelier, dict(lot=lc['CHANDELIER']['lot'], cap_atr_mult=lc['CHANDELIER']['cap_atr_mult'],
                                            filter_type='rsi3070', atr_period_for_lines=14)),
    ]

    for label, portfolio in [("A_Research_Baseline", portfolio_a),
                              ("B_Live_Params", portfolio_b),
                              ("C_Live_Params+Filters", portfolio_c)]:
        print(f"\n  Portfolio {label}:", flush=True)
        all_trades, per_strat = run_portfolio_test(h1_df, label, portfolio)

        for sn, st in per_strat.items():
            era_s = run_era_stats(st)
            fs = era_s['full']
            print(f"    {sn:<14} N={fs['n']:>5}  Sharpe={fs['sharpe']:>7.3f}  "
                  f"PnL=${fs['pnl']:>9,.0f}  WR={fs['wr']:>5.1f}%  MaxDD=${fs['max_dd']:>7,.0f}", flush=True)

        portfolio_era = run_era_stats(all_trades)
        print_era_table(f"Portfolio {label} COMBINED", portfolio_era)

    # ──────────────────────────────────────────────
    # Phase 3: Summary & recommendations
    # ──────────────────────────────────────────────
    print(f"\n\n{'=' * 90}", flush=True)
    print("  PHASE 3: Key Findings & Recommendations", flush=True)
    print(f"{'=' * 90}", flush=True)

    comparisons = [
        ("KC Session ADX", "KC_flat_ADX14", "KC_session_ADX_R178"),
        ("KC MH2 vs MH5", "KC_MH2", "KC_MH5"),
        ("KC MH2 vs MH20", "KC_MH2", "KC_MH20"),
        ("PSAR Skip Hours", "PSAR_no_skip", "PSAR_skip_3_7_22"),
        ("PSAR Old vs Live", "PSAR_old_params", "PSAR_live_params"),
        ("TSMOM Old vs Live", "TSMOM_old_params", "TSMOM_live_params"),
        ("SESS_BO D1 Filter", "SESSBO_no_d1", "SESSBO_d1_ema20"),
        ("DT Signal Timing", "DT_current_bar", "DT_confirmed_bar"),
        ("CH EMA100 vs RSI", "CH_ema100_atr22", "CH_rsi3070_atr22"),
        ("CH ATR14 vs ATR22", "CH_atr14_ema100", "CH_atr22_ema100"),
    ]

    print(f"\n  {'Comparison':<22} {'A_Sharpe':>9} {'B_Sharpe':>9} {'Delta':>7} {'A_N':>6} {'B_N':>6} {'Verdict':>10}", flush=True)
    print(f"  {'-'*22} {'-'*9} {'-'*9} {'-'*7} {'-'*6} {'-'*6} {'-'*10}", flush=True)

    for label, a_name, b_name in comparisons:
        a = all_results[a_name]['era_stats']['full']
        b = all_results[b_name]['era_stats']['full']
        delta = b['sharpe'] - a['sharpe']
        verdict = "B_BETTER" if delta > 0.1 else ("A_BETTER" if delta < -0.1 else "SIMILAR")
        print(f"  {label:<22} {a['sharpe']:>9.3f} {b['sharpe']:>9.3f} {delta:>+7.3f} {a['n']:>6} {b['n']:>6} {verdict:>10}", flush=True)

    # Save results
    elapsed = time.time() - t0
    print(f"\n  Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    save_data = {
        'individual_tests': {},
        'runtime_seconds': round(elapsed, 1),
    }
    for tn, data in all_results.items():
        save_data['individual_tests'][tn] = {
            'strategy': data['strategy'],
            'era_stats': data['era_stats'],
            'kwargs': data['kwargs'],
        }
    out_path = OUTPUT_DIR / "r181_results.json"
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Saved: {out_path}", flush=True)
    print(f"{'=' * 90}", flush=True)


if __name__ == "__main__":
    main()
