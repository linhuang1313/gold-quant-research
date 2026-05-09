#!/usr/bin/env python3
"""
R104 — Multi-Timeframe Confluence Filter
==========================================
Tests whether D1 trend filters improve H1 entry signals.

  Phase 1: Build D1 trend indicators from H1 data
  Phase 2: Filter test — only allow entries when D1 trend agrees
  Phase 3: ADX strength threshold grid search
  Phase 4: Portfolio integration + K-Fold validation
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import DataBundle, load_m15, load_h1_aligned, H1_CSV_PATH
from backtest.runner import run_variant, LIVE_PARITY_KWARGS

OUTPUT_DIR = Path("results/r104_mtf_filter")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.09, 'SESS_BO': 0.09}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']

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
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _mk(pos, exit_px, exit_time, reason, bar_i, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_px,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_i - pos['bar']}


def _run_exit_with_cap(pos, i, hi, lo, cl, spread, lot, pv, times,
                       sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap):
    atr = pos['atr']
    sl = atr * sl_atr
    tp = atr * tp_atr
    bars = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_now = (cl - pos['entry'] - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if lo <= pos['entry'] - sl:
            pnl = -sl * lot * pv
            return _mk(pos, pos['entry'] - sl, times[i], "SL", i, pnl)
        if hi >= pos['entry'] + tp:
            pnl = tp * lot * pv
            return _mk(pos, pos['entry'] + tp, times[i], "TP", i, pnl)
        extreme = pos.get('extreme', pos['entry'])
        extreme = max(extreme, hi)
        pos['extreme'] = extreme
        act_dist = atr * trail_act
        if extreme - pos['entry'] >= act_dist:
            trail_price = extreme - atr * trail_dist
            if cl <= trail_price:
                pnl = (trail_price - pos['entry'] - spread) * lot * pv
                return _mk(pos, trail_price, times[i], "Trail", i, pnl)
        if bars >= max_hold:
            pnl = (cl - pos['entry'] - spread) * lot * pv
            return _mk(pos, cl, times[i], "TimeExit", i, pnl)
    else:
        pnl_now = (pos['entry'] - cl - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if hi >= pos['entry'] + sl:
            pnl = -sl * lot * pv
            return _mk(pos, pos['entry'] + sl, times[i], "SL", i, pnl)
        if lo <= pos['entry'] - tp:
            pnl = tp * lot * pv
            return _mk(pos, pos['entry'] - tp, times[i], "TP", i, pnl)
        extreme = pos.get('extreme', pos['entry'])
        extreme = min(extreme, lo)
        pos['extreme'] = extreme
        act_dist = atr * trail_act
        if pos['entry'] - extreme >= act_dist:
            trail_price = extreme + atr * trail_dist
            if cl >= trail_price:
                pnl = (pos['entry'] - trail_price - spread) * lot * pv
                return _mk(pos, trail_price, times[i], "Trail", i, pnl)
        if bars >= max_hold:
            pnl = (pos['entry'] - cl - spread) * lot * pv
            return _mk(pos, cl, times[i], "TimeExit", i, pnl)
    return None


def add_psar(df, af_step=0.01, af_max=0.05):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    n = len(df)
    psar = np.empty(n); psar[:] = np.nan
    af = af_step; rising = True
    ep = h[0]; psar[0] = l[0]
    for i in range(1, n):
        prev = psar[i-1]
        if rising:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], l[i-1], l[max(0,i-2)])
            if l[i] < psar[i]:
                rising = False; psar[i] = ep; ep = l[i]; af = af_step
            else:
                if h[i] > ep: ep = h[i]; af = min(af + af_step, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], h[i-1], h[max(0,i-2)])
            if h[i] > psar[i]:
                rising = True; psar[i] = ep; ep = h[i]; af = af_step
            else:
                if l[i] < ep: ep = l[i]; af = min(af + af_step, af_max)
    df['PSAR'] = psar


# ═══════════════════════════════════════════════════════════════
# Strategy backtests (unfiltered)
# ═══════════════════════════════════════════════════════════════

def bt_psar(h1_df, spread, lot, maxloss_cap=0,
            sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04,
            max_hold=20, af_step=0.01, af_max=0.05, min_atr=0.1):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    add_psar(df, af_step, af_max)
    df = df.dropna(subset=['ATR', 'PSAR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; psar = df['PSAR'].values; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < min_atr: continue
        if psar[i-1] > c[i-1] and psar[i] < c[i]:
            pos = {'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif psar[i-1] < c[i-1] and psar[i] > c[i]:
            pos = {'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
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
            pos = {'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
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
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif c[i] < ll:
            pos = {'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
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
# Filtered strategy backtests (D1 trend filter)
# ═══════════════════════════════════════════════════════════════

def bt_psar_filtered(h1_df, spread, lot, d1_trend, maxloss_cap=0,
                     sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04,
                     max_hold=20, af_step=0.01, af_max=0.05, min_atr=0.1):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    add_psar(df, af_step, af_max)
    df = df.dropna(subset=['ATR', 'PSAR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; psar = df['PSAR'].values; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < min_atr: continue
        entry_date = times[i].date()
        trend_val = d1_trend.get(entry_date, 0)
        if psar[i-1] > c[i-1] and psar[i] < c[i]:
            if trend_val >= 0:
                pos = {'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif psar[i-1] < c[i-1] and psar[i] > c[i]:
            if trend_val <= 0:
                pos = {'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
    return trades


def bt_tsmom_filtered(h1_df, spread, lot, d1_trend, maxloss_cap=0,
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
        entry_date = times[i].date()
        trend_val = d1_trend.get(entry_date, 0)
        if score[i] > 0 and score[i-1] <= 0:
            if trend_val >= 0:
                pos = {'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            if trend_val <= 0:
                pos = {'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
    return trades


def bt_sess_bo_filtered(h1_df, spread, lot, d1_trend, maxloss_cap=0,
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
        trend_val = d1_trend.get(entry_date, 0)
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            if trend_val >= 0:
                pos = {'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif c[i] < ll:
            if trend_val <= 0:
                pos = {'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
    return trades


# ═══════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════

def trades_to_daily_series(trades):
    if not trades: return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


def sharpe(arr):
    if len(arr) < 10: return 0.0
    s = np.std(arr, ddof=1)
    return float(np.mean(arr) / s * np.sqrt(252)) if s > 0 else 0.0


def max_dd(arr):
    if len(arr) == 0: return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def win_rate(trades):
    if not trades: return 0.0
    return sum(1 for t in trades if t['pnl'] > 0) / len(trades) * 100


def build_portfolio_daily(unit_dailies, lots):
    all_dates = set()
    for ds in unit_dailies.values():
        all_dates.update(ds.index)
    all_dates = sorted(all_dates)
    idx = pd.DatetimeIndex(all_dates)
    portfolio = np.zeros(len(idx))
    for name in STRAT_ORDER:
        if name not in unit_dailies: continue
        ds = unit_dailies[name]
        multiplier = lots[name] / UNIT_LOT
        aligned = ds.reindex(idx, fill_value=0.0).values * multiplier
        portfolio += aligned
    return portfolio, idx


def metrics_from_trades(trades):
    daily = trades_to_daily_series(trades)
    pnl_total = sum(t['pnl'] for t in trades)
    return {
        'n_trades': len(trades),
        'sharpe': round(sharpe(daily.values), 3) if len(daily) > 0 else 0,
        'pnl': round(pnl_total, 2),
        'max_dd': round(max_dd(daily.values), 2) if len(daily) > 0 else 0,
        'wr': round(win_rate(trades), 1),
    }


# ═══════════════════════════════════════════════════════════════
# D1 Indicator Builders
# ═══════════════════════════════════════════════════════════════

def build_d1_ohlc(h1_df):
    """Resample H1 data to D1 OHLC."""
    d1 = h1_df.resample('D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    return d1


def compute_d1_sma50(d1_df):
    """D1 SMA(50) direction: +1 if close > SMA50, -1 if below."""
    sma = d1_df['Close'].rolling(50).mean()
    direction = pd.Series(0, index=d1_df.index)
    direction[d1_df['Close'] > sma] = 1
    direction[d1_df['Close'] < sma] = -1
    return direction.to_dict()


def compute_d1_ema20_slope(d1_df):
    """D1 EMA(20) slope: +1 if rising, -1 if falling."""
    ema = d1_df['Close'].ewm(span=20).mean()
    slope = ema.diff()
    direction = pd.Series(0, index=d1_df.index)
    direction[slope > 0] = 1
    direction[slope < 0] = -1
    return direction.to_dict()


def compute_d1_psar(d1_df):
    """D1 PSAR direction: +1 if close > PSAR, -1 if below."""
    df = d1_df.copy()
    add_psar(df, af_step=0.02, af_max=0.20)
    direction = pd.Series(0, index=df.index)
    direction[df['Close'] > df['PSAR']] = 1
    direction[df['Close'] < df['PSAR']] = -1
    return direction.to_dict()


def compute_d1_adx(d1_df, period=14):
    """Compute ADX(14) on D1 data. Returns Series indexed by date."""
    h = d1_df['High'].values
    l = d1_df['Low'].values
    c = d1_df['Close'].values
    n = len(d1_df)

    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        up = h[i] - h[i-1]
        down = l[i-1] - l[i]
        plus_dm[i] = up if (up > down and up > 0) else 0
        minus_dm[i] = down if (down > up and down > 0) else 0
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))

    atr = pd.Series(tr).rolling(period).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean().values / np.where(atr > 0, atr, 1)
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean().values / np.where(atr > 0, atr, 1)

    dx = np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1) * 100
    adx = pd.Series(dx).rolling(period).mean().values

    return pd.Series(adx, index=d1_df.index)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  R104 — Multi-Timeframe Confluence Filter")
    print("=" * 80)

    print("\n  Loading data...")
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    bundle = DataBundle.load_custom()
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0].date()} ~ {h1_df.index[-1].date()})")

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Build D1 Trend Indicators
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 1: Build D1 Trend Indicators from H1 Data")
    print("=" * 60)

    d1_df = build_d1_ohlc(h1_df)
    print(f"  D1 bars: {len(d1_df)} ({d1_df.index[0].date()} ~ {d1_df.index[-1].date()})")

    d1_sma50 = compute_d1_sma50(d1_df)
    d1_ema20 = compute_d1_ema20_slope(d1_df)
    d1_psar = compute_d1_psar(d1_df)
    d1_adx = compute_d1_adx(d1_df)

    filters = {
        'D1_SMA50': d1_sma50,
        'D1_EMA20': d1_ema20,
        'D1_PSAR': d1_psar,
    }

    # Convert index to date for lookup
    d1_sma50_dates = {k.date() if hasattr(k, 'date') else k: v for k, v in d1_sma50.items()}
    d1_ema20_dates = {k.date() if hasattr(k, 'date') else k: v for k, v in d1_ema20.items()}
    d1_psar_dates = {k.date() if hasattr(k, 'date') else k: v for k, v in d1_psar.items()}

    filters_by_date = {
        'D1_SMA50': d1_sma50_dates,
        'D1_EMA20': d1_ema20_dates,
        'D1_PSAR': d1_psar_dates,
    }

    # Sample output
    sample_dates = sorted(d1_sma50_dates.keys())[-5:]
    print(f"\n  Sample D1 indicators (last 5 days):")
    print(f"  {'Date':<12} {'SMA50':>6} {'EMA20':>6} {'PSAR':>6} {'ADX':>6}")
    for dt in sample_dates:
        adx_val = d1_adx.get(pd.Timestamp(dt), np.nan)
        if hasattr(adx_val, '__float__'):
            adx_str = f"{float(adx_val):.1f}"
        else:
            adx_str = "N/A"
        print(f"  {dt}   {d1_sma50_dates.get(dt, 0):>+3d}   "
              f"{d1_ema20_dates.get(dt, 0):>+3d}   "
              f"{d1_psar_dates.get(dt, 0):>+3d}   {adx_str:>6}")

    # Distribution
    for fname, fdata in filters_by_date.items():
        vals = list(fdata.values())
        up = sum(1 for v in vals if v > 0)
        dn = sum(1 for v in vals if v < 0)
        neutral = sum(1 for v in vals if v == 0)
        print(f"  {fname}: +1={up} ({up/len(vals)*100:.0f}%), "
              f"-1={dn} ({dn/len(vals)*100:.0f}%), 0={neutral} ({neutral/len(vals)*100:.0f}%)")

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Individual Filter Tests
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 2: Individual Filter Tests (H1 strategies)")
    print("=" * 60)

    h1_strategies = {
        'PSAR': (bt_psar, bt_psar_filtered, CAPS['PSAR']),
        'TSMOM': (bt_tsmom, bt_tsmom_filtered, CAPS['TSMOM']),
        'SESS_BO': (bt_sess_bo, bt_sess_bo_filtered, CAPS['SESS_BO']),
    }

    phase2_results = {}
    for strat_name, (bt_fn, bt_fn_filtered, cap) in h1_strategies.items():
        print(f"\n  --- {strat_name} ---")
        baseline = bt_fn(h1_df, SPREAD, UNIT_LOT, maxloss_cap=cap)
        bm = metrics_from_trades(baseline)
        print(f"  {'Unfiltered':<12}: {bm['n_trades']:5d} trades, Sharpe={bm['sharpe']:6.3f}, "
              f"PnL=${bm['pnl']:9.2f}, MaxDD=${bm['max_dd']:7.2f}, WR={bm['wr']:.1f}%")

        strat_results = {'baseline': bm}
        for filter_name, filter_data in filters_by_date.items():
            filtered = bt_fn_filtered(h1_df, SPREAD, UNIT_LOT, filter_data, maxloss_cap=cap)
            fm = metrics_from_trades(filtered)
            delta_sh = fm['sharpe'] - bm['sharpe']
            marker = " *" if delta_sh > 0.05 else ""
            print(f"  {filter_name:<12}: {fm['n_trades']:5d} trades, Sharpe={fm['sharpe']:6.3f}, "
                  f"PnL=${fm['pnl']:9.2f}, MaxDD=${fm['max_dd']:7.2f}, WR={fm['wr']:.1f}% "
                  f"(ΔSh={delta_sh:+.3f}){marker}")
            strat_results[filter_name] = fm
        phase2_results[strat_name] = strat_results

    # Determine best overall filter
    filter_scores = {f: 0.0 for f in filters_by_date.keys()}
    for strat_name, results in phase2_results.items():
        baseline_sh = results['baseline']['sharpe']
        for filter_name in filters_by_date.keys():
            delta = results[filter_name]['sharpe'] - baseline_sh
            filter_scores[filter_name] += delta

    best_filter_name = max(filter_scores, key=filter_scores.get)
    print(f"\n  Filter score totals (sum of Sharpe deltas across strategies):")
    for fname, score in sorted(filter_scores.items(), key=lambda x: -x[1]):
        marker = " <-- BEST" if fname == best_filter_name else ""
        print(f"    {fname}: {score:+.3f}{marker}")

    best_filter_data = filters_by_date[best_filter_name]

    # ═══════════════════════════════════════════════════════════
    # Phase 3: ADX Strength Threshold Grid Search
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"  Phase 3: ADX Strength Threshold (using {best_filter_name})")
    print("=" * 60)

    adx_by_date = {}
    for ts, val in d1_adx.items():
        dt = ts.date() if hasattr(ts, 'date') else ts
        adx_by_date[dt] = val

    adx_thresholds = [10, 15, 20, 25, 30, 35]
    phase3_results = {}

    for threshold in adx_thresholds:
        # Build combined filter: best_filter AND ADX > threshold
        combined_filter = {}
        for dt, trend_val in best_filter_data.items():
            adx_val = adx_by_date.get(dt, 0)
            if not np.isnan(adx_val) and adx_val > threshold:
                combined_filter[dt] = trend_val
            else:
                combined_filter[dt] = 0  # neutral = allow both directions

        total_sharpe = 0.0
        total_trades = 0
        total_pnl = 0.0
        strat_detail = {}
        for strat_name, (bt_fn, bt_fn_filtered, cap) in h1_strategies.items():
            filtered = bt_fn_filtered(h1_df, SPREAD, UNIT_LOT, combined_filter, maxloss_cap=cap)
            fm = metrics_from_trades(filtered)
            total_sharpe += fm['sharpe']
            total_trades += fm['n_trades']
            total_pnl += fm['pnl']
            strat_detail[strat_name] = fm

        avg_sharpe = total_sharpe / len(h1_strategies)
        phase3_results[threshold] = {
            'avg_sharpe': round(avg_sharpe, 3),
            'total_trades': total_trades,
            'total_pnl': round(total_pnl, 2),
            'detail': strat_detail,
        }
        print(f"  ADX>{threshold:2d}: avg_Sharpe={avg_sharpe:.3f}, "
              f"trades={total_trades}, PnL=${total_pnl:.0f}")

    best_adx_threshold = max(phase3_results, key=lambda k: phase3_results[k]['avg_sharpe'])
    print(f"\n  Best ADX threshold: {best_adx_threshold} "
          f"(avg_Sharpe={phase3_results[best_adx_threshold]['avg_sharpe']:.3f})")

    # Build final combined filter with best threshold
    final_filter = {}
    for dt, trend_val in best_filter_data.items():
        adx_val = adx_by_date.get(dt, 0)
        if not np.isnan(adx_val) and adx_val > best_adx_threshold:
            final_filter[dt] = trend_val
        else:
            final_filter[dt] = 0

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Portfolio Integration + K-Fold Validation
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 4: Portfolio Integration + K-Fold Validation")
    print("=" * 60)

    # Unfiltered portfolio
    print("\n  Running unfiltered portfolio (baseline)...")
    psar_base = bt_psar(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['PSAR'])
    tsmom_base = bt_tsmom(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['TSMOM'])
    sess_base = bt_sess_bo(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['SESS_BO'])
    l8_trades = bt_l8_max(bundle, SPREAD, UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])

    base_dailies = {
        'L8_MAX': trades_to_daily_series(l8_trades),
        'PSAR': trades_to_daily_series(psar_base),
        'TSMOM': trades_to_daily_series(tsmom_base),
        'SESS_BO': trades_to_daily_series(sess_base),
    }
    port_base, idx_base = build_portfolio_daily(base_dailies, R89_LOTS)
    sh_base = sharpe(port_base)
    dd_base = max_dd(port_base)
    pnl_base = float(port_base.sum())

    print(f"  Unfiltered portfolio: Sharpe={sh_base:.3f}, PnL=${pnl_base:.0f}, MaxDD=${dd_base:.0f}")

    # Filtered portfolio
    print(f"\n  Running filtered portfolio ({best_filter_name} + ADX>{best_adx_threshold})...")
    psar_filt = bt_psar_filtered(h1_df, SPREAD, UNIT_LOT, final_filter, maxloss_cap=CAPS['PSAR'])
    tsmom_filt = bt_tsmom_filtered(h1_df, SPREAD, UNIT_LOT, final_filter, maxloss_cap=CAPS['TSMOM'])
    sess_filt = bt_sess_bo_filtered(h1_df, SPREAD, UNIT_LOT, final_filter, maxloss_cap=CAPS['SESS_BO'])

    filt_dailies = {
        'L8_MAX': trades_to_daily_series(l8_trades),
        'PSAR': trades_to_daily_series(psar_filt),
        'TSMOM': trades_to_daily_series(tsmom_filt),
        'SESS_BO': trades_to_daily_series(sess_filt),
    }
    port_filt, idx_filt = build_portfolio_daily(filt_dailies, R89_LOTS)
    sh_filt = sharpe(port_filt)
    dd_filt = max_dd(port_filt)
    pnl_filt = float(port_filt.sum())

    print(f"  Filtered portfolio:   Sharpe={sh_filt:.3f}, PnL=${pnl_filt:.0f}, MaxDD=${dd_filt:.0f}")
    print(f"\n  Delta: Sharpe={sh_filt - sh_base:+.3f}, PnL=${pnl_filt - pnl_base:+.0f}, "
          f"MaxDD={dd_filt - dd_base:+.0f}")

    # Individual strategy comparison
    print(f"\n  Strategy-level comparison (filtered vs unfiltered):")
    print(f"  {'Strategy':<10} {'Base_Sh':>8} {'Filt_Sh':>8} {'ΔSh':>7} "
          f"{'Base_N':>7} {'Filt_N':>7}")
    strat_comparison = {}
    for name, (base_tr, filt_tr) in [
        ('PSAR', (psar_base, psar_filt)),
        ('TSMOM', (tsmom_base, tsmom_filt)),
        ('SESS_BO', (sess_base, sess_filt)),
        ('L8_MAX', (l8_trades, l8_trades)),
    ]:
        bm = metrics_from_trades(base_tr)
        fm = metrics_from_trades(filt_tr)
        delta = fm['sharpe'] - bm['sharpe']
        print(f"  {name:<10} {bm['sharpe']:>8.3f} {fm['sharpe']:>8.3f} {delta:>+7.3f} "
              f"{bm['n_trades']:>7d} {fm['n_trades']:>7d}")
        strat_comparison[name] = {
            'base': bm, 'filtered': fm, 'delta_sharpe': round(delta, 3)
        }

    # K-Fold Validation
    print(f"\n  K-Fold Validation (6 folds):")
    fold_sharpes_base = []
    fold_sharpes_filt = []

    for fname, start, end in FOLDS:
        fold_h1 = h1_df[(h1_df.index >= start) & (h1_df.index < end)]
        if len(fold_h1) < 100:
            fold_sharpes_base.append(0.0)
            fold_sharpes_filt.append(0.0)
            continue

        # Build D1 indicators for this fold
        fold_d1 = build_d1_ohlc(fold_h1)
        if len(fold_d1) < 60:
            fold_sharpes_base.append(0.0)
            fold_sharpes_filt.append(0.0)
            continue

        fold_filter_raw = {
            'D1_SMA50': compute_d1_sma50,
            'D1_EMA20': compute_d1_ema20_slope,
            'D1_PSAR': compute_d1_psar,
        }[best_filter_name](fold_d1)

        fold_filter_dates = {k.date() if hasattr(k, 'date') else k: v
                            for k, v in fold_filter_raw.items()}

        fold_adx = compute_d1_adx(fold_d1)
        fold_adx_dates = {ts.date() if hasattr(ts, 'date') else ts: val
                         for ts, val in fold_adx.items()}

        fold_final_filter = {}
        for dt, trend_val in fold_filter_dates.items():
            adx_val = fold_adx_dates.get(dt, 0)
            if not np.isnan(adx_val) and adx_val > best_adx_threshold:
                fold_final_filter[dt] = trend_val
            else:
                fold_final_filter[dt] = 0

        # Base fold
        f_psar = bt_psar(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=CAPS['PSAR'])
        f_tsmom = bt_tsmom(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=CAPS['TSMOM'])
        f_sess = bt_sess_bo(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=CAPS['SESS_BO'])
        f_l8 = bt_l8_max(bundle, SPREAD, UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])

        fd_base = {
            'L8_MAX': trades_to_daily_series(f_l8),
            'PSAR': trades_to_daily_series(f_psar),
            'TSMOM': trades_to_daily_series(f_tsmom),
            'SESS_BO': trades_to_daily_series(f_sess),
        }
        pb, _ = build_portfolio_daily(fd_base, R89_LOTS)
        fold_sharpes_base.append(sharpe(pb))

        # Filtered fold
        f_psar_f = bt_psar_filtered(fold_h1, SPREAD, UNIT_LOT, fold_final_filter, maxloss_cap=CAPS['PSAR'])
        f_tsmom_f = bt_tsmom_filtered(fold_h1, SPREAD, UNIT_LOT, fold_final_filter, maxloss_cap=CAPS['TSMOM'])
        f_sess_f = bt_sess_bo_filtered(fold_h1, SPREAD, UNIT_LOT, fold_final_filter, maxloss_cap=CAPS['SESS_BO'])

        fd_filt = {
            'L8_MAX': trades_to_daily_series(f_l8),
            'PSAR': trades_to_daily_series(f_psar_f),
            'TSMOM': trades_to_daily_series(f_tsmom_f),
            'SESS_BO': trades_to_daily_series(f_sess_f),
        }
        pf, _ = build_portfolio_daily(fd_filt, R89_LOTS)
        fold_sharpes_filt.append(sharpe(pf))

    print(f"  {'Fold':<8} {'Base':>8} {'Filtered':>10} {'Delta':>8}")
    for i, (fname, _, _) in enumerate(FOLDS):
        delta = fold_sharpes_filt[i] - fold_sharpes_base[i]
        print(f"  {fname:<8} {fold_sharpes_base[i]:>8.3f} {fold_sharpes_filt[i]:>10.3f} {delta:>+8.3f}")

    mean_base = np.mean(fold_sharpes_base)
    mean_filt = np.mean(fold_sharpes_filt)
    filt_wins = sum(1 for a, b in zip(fold_sharpes_filt, fold_sharpes_base) if a > b)
    base_positive = sum(1 for s in fold_sharpes_base if s > 0)
    filt_positive = sum(1 for s in fold_sharpes_filt if s > 0)

    print(f"\n  Mean Sharpe: base={mean_base:.3f}, filtered={mean_filt:.3f} "
          f"(delta={mean_filt - mean_base:+.3f})")
    print(f"  Filter wins: {filt_wins}/6 folds")
    print(f"  Positive folds: base={base_positive}/6, filtered={filt_positive}/6")

    # ═══════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    results = {
        'experiment': 'R104 Multi-Timeframe Confluence Filter',
        'elapsed_s': round(elapsed, 1),
        'best_filter': best_filter_name,
        'best_adx_threshold': best_adx_threshold,
        'phase2_filter_scores': {k: round(v, 3) for k, v in filter_scores.items()},
        'phase2_results': phase2_results,
        'phase3_adx_grid': {str(k): v for k, v in phase3_results.items()},
        'portfolio_baseline': {
            'sharpe': round(sh_base, 3), 'pnl': round(pnl_base, 2), 'max_dd': round(dd_base, 2),
        },
        'portfolio_filtered': {
            'sharpe': round(sh_filt, 3), 'pnl': round(pnl_filt, 2), 'max_dd': round(dd_filt, 2),
        },
        'strat_comparison': strat_comparison,
        'kfold': {
            'base_folds': [round(s, 3) for s in fold_sharpes_base],
            'filtered_folds': [round(s, 3) for s in fold_sharpes_filt],
            'base_mean': round(mean_base, 3),
            'filtered_mean': round(mean_filt, 3),
            'filter_wins': filt_wins,
            'base_positive_folds': base_positive,
            'filtered_positive_folds': filt_positive,
        },
        'recommendation': (
            f"ADOPT: {best_filter_name}+ADX>{best_adx_threshold} filter improves portfolio"
            if filt_wins >= 4 and mean_filt > mean_base
            else f"SKIP: MTF filter does not consistently improve ({filt_wins}/6 folds)"
        ),
    }

    out_file = OUTPUT_DIR / "r104_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"  R104 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Best filter: {best_filter_name} + ADX>{best_adx_threshold}")
    print(f"  Portfolio: base Sharpe={sh_base:.3f} -> filtered Sharpe={sh_filt:.3f} "
          f"(delta={sh_filt-sh_base:+.3f})")
    if filt_wins >= 4 and mean_filt > mean_base:
        print(f"  RECOMMENDATION: ADOPT filter ({filt_wins}/6 folds improved)")
    else:
        print(f"  RECOMMENDATION: SKIP — filter not consistently better ({filt_wins}/6)")
    print(f"{'='*80}")
    print(f"  Saved: {out_file}")


if __name__ == '__main__':
    main()
