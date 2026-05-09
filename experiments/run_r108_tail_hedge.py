#!/usr/bin/env python3
"""
R108 — Tail Risk Hedging
==========================
Tests portfolio-level hedging mechanisms to reduce extreme drawdowns.

  Phase 1: Build gold volatility proxy (ATR rank + return kurtosis)
  Phase 2: Test 4 hedge strategies during high-vol periods
  Phase 3: Activation threshold sweep (80th-95th percentile)
  Phase 4: K-Fold validation on best hedge mode
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

OUTPUT_DIR = Path("results/r108_tail_hedge")
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


# ═══════════════════════════════════════════════════════════════
# Strategy backtests (4 core)
# ═══════════════════════════════════════════════════════════════

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


def build_portfolio_daily(unit_dailies, lots):
    all_dates = set()
    for ds in unit_dailies.values():
        all_dates.update(ds.index)
    all_dates = sorted(all_dates)
    idx = pd.DatetimeIndex(all_dates)
    portfolio = np.zeros(len(idx))
    for name in STRAT_ORDER:
        if name not in unit_dailies:
            continue
        ds = unit_dailies[name]
        multiplier = lots[name] / UNIT_LOT
        aligned = ds.reindex(idx, fill_value=0.0).values * multiplier
        portfolio += aligned
    return portfolio


# ═══════════════════════════════════════════════════════════════
# Vol proxy helpers
# ═══════════════════════════════════════════════════════════════

def build_vol_proxy(h1_df):
    """Build composite volatility proxy from ATR rank + return kurtosis."""
    df = h1_df.copy()
    df['ATR14'] = compute_atr(df, period=14)
    df['ATR_rank'] = df['ATR14'].rolling(20, min_periods=10).apply(
        lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100, raw=False
    )
    df['ret'] = df['Close'].pct_change()
    df['ret_kurt'] = df['ret'].rolling(24, min_periods=12).apply(
        lambda x: x.kurtosis(), raw=False
    )
    kurt_rolling = df['ret_kurt'].rolling(20, min_periods=10).apply(
        lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100, raw=False
    )
    df['kurt_rank'] = kurt_rolling
    df['vol_proxy'] = 0.6 * df['ATR_rank'] + 0.4 * df['kurt_rank']
    df = df.dropna(subset=['vol_proxy'])

    daily_proxy = df['vol_proxy'].resample('D').max().dropna()
    return df['vol_proxy'], daily_proxy


def get_sma50(h1_df):
    """Compute SMA(50) on H1 close for trend detection."""
    return h1_df['Close'].rolling(50, min_periods=30).mean()


# ═══════════════════════════════════════════════════════════════
# Hedge application
# ═══════════════════════════════════════════════════════════════

def apply_hedge_mode_a(trades, daily_proxy, threshold):
    """Half-Size: scale PnL by 0.5 when vol_proxy > threshold on entry date."""
    hedged = []
    n_affected = 0
    for t in trades:
        entry_date = pd.Timestamp(t['entry_time']).normalize()
        proxy_val = daily_proxy.get(entry_date, None)
        if proxy_val is None:
            nearest = daily_proxy.index.searchsorted(entry_date)
            if nearest > 0:
                proxy_val = daily_proxy.iloc[nearest - 1]
        nt = dict(t)
        if proxy_val is not None and proxy_val > threshold:
            nt['pnl'] = t['pnl'] * 0.5
            n_affected += 1
        hedged.append(nt)
    return hedged, n_affected


def apply_hedge_mode_b(trades, daily_proxy, threshold, h1_df):
    """Counter-Trend Only: in high-vol, only allow counter-trend entries."""
    sma50 = get_sma50(h1_df)
    hedged = []
    n_affected = 0
    for t in trades:
        entry_time = pd.Timestamp(t['entry_time'])
        entry_date = entry_time.normalize()
        proxy_val = daily_proxy.get(entry_date, None)
        if proxy_val is None:
            nearest = daily_proxy.index.searchsorted(entry_date)
            if nearest > 0:
                proxy_val = daily_proxy.iloc[nearest - 1]
        if proxy_val is not None and proxy_val > threshold:
            sma_val = sma50.asof(entry_time)
            if pd.isna(sma_val):
                hedged.append(dict(t))
                continue
            close_at_entry = t['entry']
            uptrend = close_at_entry > sma_val
            is_counter = (t['dir'] == 'SELL' and uptrend) or (t['dir'] == 'BUY' and not uptrend)
            if not is_counter:
                n_affected += 1
                continue
        hedged.append(dict(t))
    return hedged, n_affected


def apply_hedge_mode_c(trades, daily_proxy, threshold, hourly_proxy):
    """Full Stop + Cooldown: skip entries when vol_proxy > threshold.
    Resume only after proxy drops below threshold for 4 consecutive hours."""
    below_count = {}
    proxy_sorted = hourly_proxy.sort_index()
    consecutive_below = 0
    cooldown_end = {}
    in_cooldown = False
    cooldown_resume_dates = set()

    for ts, val in proxy_sorted.items():
        d = ts.normalize()
        if val > threshold:
            in_cooldown = True
            consecutive_below = 0
        else:
            consecutive_below += 1
            if consecutive_below >= 4 and in_cooldown:
                in_cooldown = False
                cooldown_resume_dates.add(d)

        if in_cooldown:
            below_count[d] = True

    hedged = []
    n_affected = 0
    for t in trades:
        entry_date = pd.Timestamp(t['entry_time']).normalize()
        if below_count.get(entry_date, False):
            entry_time = pd.Timestamp(t['entry_time'])
            hourly_before = proxy_sorted[proxy_sorted.index <= entry_time]
            if len(hourly_before) > 0 and hourly_before.iloc[-1] > threshold:
                n_affected += 1
                continue
            recent_4 = hourly_before.tail(4)
            if len(recent_4) < 4 or (recent_4 > threshold).any():
                n_affected += 1
                continue
        hedged.append(dict(t))
    return hedged, n_affected


def apply_hedge_mode_d(trades, daily_proxy, threshold):
    """Tight Stops: in high-vol, approximate tighter SL by halving losing PnL."""
    hedged = []
    n_affected = 0
    for t in trades:
        entry_date = pd.Timestamp(t['entry_time']).normalize()
        proxy_val = daily_proxy.get(entry_date, None)
        if proxy_val is None:
            nearest = daily_proxy.index.searchsorted(entry_date)
            if nearest > 0:
                proxy_val = daily_proxy.iloc[nearest - 1]
        nt = dict(t)
        if proxy_val is not None and proxy_val > threshold:
            n_affected += 1
            if t['pnl'] < 0:
                nt['pnl'] = t['pnl'] * 0.5
        hedged.append(nt)
    return hedged, n_affected


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  R108 — Tail Risk Hedging")
    print("=" * 80)

    print("\n  Loading data...")
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    bundle = DataBundle.load_custom()
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0].date()} ~ {h1_df.index[-1].date()})")

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Build Gold Volatility Proxy
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 1: Build Gold Volatility Proxy")
    print("=" * 60)

    hourly_proxy, daily_proxy = build_vol_proxy(h1_df)
    dp_vals = daily_proxy.values

    print(f"\n  Vol proxy (daily, max of hourly):")
    print(f"    N days:  {len(dp_vals)}")
    print(f"    Mean:    {np.mean(dp_vals):.2f}")
    print(f"    Std:     {np.std(dp_vals):.2f}")
    for pct in [50, 80, 90, 95, 99]:
        print(f"    P{pct:02d}:     {np.percentile(dp_vals, pct):.2f}")

    threshold_90 = float(np.percentile(dp_vals, 90))
    high_vol_days = (dp_vals > threshold_90).sum()
    print(f"\n  90th percentile threshold: {threshold_90:.2f}")
    print(f"  Days above 90th pct: {high_vol_days} / {len(dp_vals)} "
          f"({high_vol_days/len(dp_vals)*100:.1f}%)")

    # ═══════════════════════════════════════════════════════════
    # Run base backtests for all strategies
    # ═══════════════════════════════════════════════════════════
    print("\n  Running base backtests (unit lot)...")
    base_trades = {}
    unit_dailies = {}
    for name in STRAT_ORDER:
        if name == 'L8_MAX':
            trades = bt_l8_max(bundle, SPREAD, UNIT_LOT, maxloss_cap=CAPS[name])
        elif name == 'PSAR':
            trades = bt_psar(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS[name])
        elif name == 'TSMOM':
            trades = bt_tsmom(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS[name])
        elif name == 'SESS_BO':
            trades = bt_sess_bo(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS[name])
        base_trades[name] = trades
        unit_dailies[name] = trades_to_daily_series(trades)
        print(f"    {name}: {len(trades)} trades")

    base_portfolio = build_portfolio_daily(unit_dailies, R89_LOTS)
    base_sharpe = sharpe(base_portfolio)
    base_maxdd = max_dd(base_portfolio)
    base_pnl = float(np.sum(base_portfolio))
    print(f"\n  Base portfolio (unhedged):")
    print(f"    Sharpe={base_sharpe:.3f}, MaxDD=${base_maxdd:.2f}, PnL=${base_pnl:.2f}")

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Hedge Strategies
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 2: Test 4 Hedge Strategies (90th pct threshold)")
    print("=" * 60)

    all_base_trades = []
    for name in STRAT_ORDER:
        lot = R89_LOTS[name]
        scale = lot / UNIT_LOT
        for t in base_trades[name]:
            st = dict(t)
            st['pnl'] = t['pnl'] * scale
            st['_strat'] = name
            all_base_trades.append(st)

    mode_results = {}

    # Mode A: Half-Size
    hedged_a, n_a = apply_hedge_mode_a(all_base_trades, daily_proxy, threshold_90)
    ds_a = trades_to_daily_series(hedged_a)
    arr_a = ds_a.values
    mode_results['A_HalfSize'] = {
        'sharpe': sharpe(arr_a), 'max_dd': max_dd(arr_a),
        'pnl': float(np.sum(arr_a)), 'n_affected': n_a,
        'n_trades': len(hedged_a),
    }

    # Mode B: Counter-Trend Only
    hedged_b, n_b = apply_hedge_mode_b(all_base_trades, daily_proxy, threshold_90, h1_df)
    ds_b = trades_to_daily_series(hedged_b)
    arr_b = ds_b.values
    mode_results['B_CounterTrend'] = {
        'sharpe': sharpe(arr_b), 'max_dd': max_dd(arr_b),
        'pnl': float(np.sum(arr_b)), 'n_affected': n_b,
        'n_trades': len(hedged_b),
    }

    # Mode C: Full Stop + Cooldown
    hedged_c, n_c = apply_hedge_mode_c(all_base_trades, daily_proxy, threshold_90, hourly_proxy)
    ds_c = trades_to_daily_series(hedged_c)
    arr_c = ds_c.values
    mode_results['C_FullStop'] = {
        'sharpe': sharpe(arr_c), 'max_dd': max_dd(arr_c),
        'pnl': float(np.sum(arr_c)), 'n_affected': n_c,
        'n_trades': len(hedged_c),
    }

    # Mode D: Tight Stops
    hedged_d, n_d = apply_hedge_mode_d(all_base_trades, daily_proxy, threshold_90)
    ds_d = trades_to_daily_series(hedged_d)
    arr_d = ds_d.values
    mode_results['D_TightStops'] = {
        'sharpe': sharpe(arr_d), 'max_dd': max_dd(arr_d),
        'pnl': float(np.sum(arr_d)), 'n_affected': n_d,
        'n_trades': len(hedged_d),
    }

    print(f"\n  {'Mode':<20s} {'Sharpe':>8s} {'MaxDD':>10s} {'PnL':>10s} {'Affected':>10s} {'Trades':>8s}")
    print(f"  {'─'*66}")
    print(f"  {'Unhedged':<20s} {base_sharpe:>8.3f} {base_maxdd:>10.2f} {base_pnl:>10.2f} {'—':>10s} {len(all_base_trades):>8d}")
    for mode_name, mr in mode_results.items():
        print(f"  {mode_name:<20s} {mr['sharpe']:>8.3f} {mr['max_dd']:>10.2f} "
              f"{mr['pnl']:>10.2f} {mr['n_affected']:>10d} {mr['n_trades']:>8d}")

    ranked_modes = sorted(mode_results.items(),
                          key=lambda x: (x[1]['sharpe'], -x[1]['max_dd']),
                          reverse=True)
    best_2_modes = [ranked_modes[0][0], ranked_modes[1][0]]
    print(f"\n  Best 2 modes for Phase 3: {best_2_modes[0]}, {best_2_modes[1]}")

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Threshold Sweep
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 3: Activation Threshold Sweep")
    print("=" * 60)

    pct_sweep = [75, 80, 85, 90, 95]
    sweep_results = {}
    mode_apply_fns = {
        'A_HalfSize': lambda trades, dp, th: apply_hedge_mode_a(trades, dp, th),
        'B_CounterTrend': lambda trades, dp, th: apply_hedge_mode_b(trades, dp, th, h1_df),
        'C_FullStop': lambda trades, dp, th: apply_hedge_mode_c(trades, dp, th, hourly_proxy),
        'D_TightStops': lambda trades, dp, th: apply_hedge_mode_d(trades, dp, th),
    }

    for mode_name in best_2_modes:
        sweep_results[mode_name] = {}
        print(f"\n  {mode_name}:")
        print(f"    {'Pctile':>8s} {'Thresh':>8s} {'Sharpe':>8s} {'MaxDD':>10s} {'PnL':>10s} {'Affected':>10s}")
        print(f"    {'─'*54}")
        best_sh = -999
        best_pct = 90
        for pct in pct_sweep:
            th = float(np.percentile(dp_vals, pct))
            hedged, n_aff = mode_apply_fns[mode_name](all_base_trades, daily_proxy, th)
            ds = trades_to_daily_series(hedged)
            arr = ds.values
            sh = sharpe(arr)
            dd = max_dd(arr)
            pnl = float(np.sum(arr))
            sweep_results[mode_name][pct] = {
                'threshold': th, 'sharpe': sh, 'max_dd': dd, 'pnl': pnl,
                'n_affected': n_aff, 'n_trades': len(hedged),
            }
            marker = ""
            if sh > best_sh:
                best_sh = sh
                best_pct = pct
                marker = " <-- best"
            print(f"    P{pct:>3d}     {th:>8.2f} {sh:>8.3f} {dd:>10.2f} {pnl:>10.2f} {n_aff:>10d}{marker}")
        sweep_results[mode_name]['_best_pct'] = best_pct
        sweep_results[mode_name]['_best_sharpe'] = best_sh

    overall_best_mode = max(best_2_modes,
                            key=lambda m: sweep_results[m]['_best_sharpe'])
    overall_best_pct = sweep_results[overall_best_mode]['_best_pct']
    overall_best_threshold = float(np.percentile(dp_vals, overall_best_pct))
    print(f"\n  Overall best: {overall_best_mode} @ P{overall_best_pct} "
          f"(threshold={overall_best_threshold:.2f}, "
          f"Sharpe={sweep_results[overall_best_mode]['_best_sharpe']:.3f})")

    # ═══════════════════════════════════════════════════════════
    # Phase 4: K-Fold Validation
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"  Phase 4: K-Fold Validation ({overall_best_mode} @ P{overall_best_pct})")
    print("=" * 60)

    kfold_results = []
    for fold_name, fold_start, fold_end in FOLDS:
        print(f"\n  {fold_name}: {fold_start} -> {fold_end}")
        fold_h1 = h1_df[(h1_df.index >= fold_start) & (h1_df.index < fold_end)]
        if len(fold_h1) < 200:
            kfold_results.append({
                'fold': fold_name, 'start': fold_start, 'end': fold_end,
                'unhedged_sharpe': 0.0, 'hedged_sharpe': 0.0,
                'unhedged_maxdd': 0.0, 'hedged_maxdd': 0.0,
                'unhedged_pnl': 0.0, 'hedged_pnl': 0.0,
                'note': 'insufficient data',
            })
            print(f"    Skipped (insufficient data)")
            continue

        train_h1 = h1_df[(h1_df.index < fold_start) | (h1_df.index >= fold_end)]
        if len(train_h1) < 500:
            train_h1 = h1_df[h1_df.index < fold_start]

        _, train_daily_proxy = build_vol_proxy(train_h1)
        train_dp_vals = train_daily_proxy.values
        if len(train_dp_vals) < 50:
            kfold_results.append({
                'fold': fold_name, 'start': fold_start, 'end': fold_end,
                'unhedged_sharpe': 0.0, 'hedged_sharpe': 0.0,
                'unhedged_maxdd': 0.0, 'hedged_maxdd': 0.0,
                'unhedged_pnl': 0.0, 'hedged_pnl': 0.0,
                'note': 'insufficient training data',
            })
            print(f"    Skipped (insufficient training data)")
            continue

        train_threshold = float(np.percentile(train_dp_vals, overall_best_pct))

        fold_hourly_proxy, fold_daily_proxy = build_vol_proxy(fold_h1)

        try:
            fold_bundle = DataBundle.load_default(start=fold_start, end=fold_end)
        except Exception:
            fold_bundle = bundle

        fold_trades = {}
        fold_unit_dailies = {}
        for name in STRAT_ORDER:
            if name == 'L8_MAX':
                trades = bt_l8_max(fold_bundle, SPREAD, UNIT_LOT, maxloss_cap=CAPS[name])
            elif name == 'PSAR':
                trades = bt_psar(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=CAPS[name])
            elif name == 'TSMOM':
                trades = bt_tsmom(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=CAPS[name])
            elif name == 'SESS_BO':
                trades = bt_sess_bo(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=CAPS[name])
            fold_trades[name] = trades
            fold_unit_dailies[name] = trades_to_daily_series(trades)

        unhedged_port = build_portfolio_daily(fold_unit_dailies, R89_LOTS)
        uh_sharpe = sharpe(unhedged_port)
        uh_maxdd = max_dd(unhedged_port)
        uh_pnl = float(np.sum(unhedged_port))

        fold_all_trades = []
        for name in STRAT_ORDER:
            lot = R89_LOTS[name]
            scale = lot / UNIT_LOT
            for t in fold_trades[name]:
                st = dict(t)
                st['pnl'] = t['pnl'] * scale
                st['_strat'] = name
                fold_all_trades.append(st)

        if overall_best_mode == 'A_HalfSize':
            hedged_trades, n_aff = apply_hedge_mode_a(fold_all_trades, fold_daily_proxy, train_threshold)
        elif overall_best_mode == 'B_CounterTrend':
            hedged_trades, n_aff = apply_hedge_mode_b(fold_all_trades, fold_daily_proxy, train_threshold, fold_h1)
        elif overall_best_mode == 'C_FullStop':
            hedged_trades, n_aff = apply_hedge_mode_c(fold_all_trades, fold_daily_proxy, train_threshold, fold_hourly_proxy)
        elif overall_best_mode == 'D_TightStops':
            hedged_trades, n_aff = apply_hedge_mode_d(fold_all_trades, fold_daily_proxy, train_threshold)
        else:
            hedged_trades, n_aff = fold_all_trades, 0

        ds_h = trades_to_daily_series(hedged_trades)
        arr_h = ds_h.values
        h_sharpe = sharpe(arr_h)
        h_maxdd = max_dd(arr_h)
        h_pnl = float(np.sum(arr_h))

        kfold_results.append({
            'fold': fold_name, 'start': fold_start, 'end': fold_end,
            'train_threshold': train_threshold,
            'unhedged_sharpe': round(uh_sharpe, 3),
            'hedged_sharpe': round(h_sharpe, 3),
            'unhedged_maxdd': round(uh_maxdd, 2),
            'hedged_maxdd': round(h_maxdd, 2),
            'unhedged_pnl': round(uh_pnl, 2),
            'hedged_pnl': round(h_pnl, 2),
            'n_affected': n_aff,
            'n_trades_unhedged': len(fold_all_trades),
            'n_trades_hedged': len(hedged_trades),
        })
        print(f"    Train threshold: {train_threshold:.2f}")
        print(f"    Unhedged: Sharpe={uh_sharpe:.3f}, MaxDD=${uh_maxdd:.2f}, PnL=${uh_pnl:.2f}")
        print(f"    Hedged:   Sharpe={h_sharpe:.3f}, MaxDD=${h_maxdd:.2f}, PnL=${h_pnl:.2f}")
        print(f"    Trades affected: {n_aff}")

    # K-Fold summary
    uh_sharpes = [f['unhedged_sharpe'] for f in kfold_results]
    h_sharpes = [f['hedged_sharpe'] for f in kfold_results]
    uh_dds = [f['unhedged_maxdd'] for f in kfold_results]
    h_dds = [f['hedged_maxdd'] for f in kfold_results]
    folds_improved_sharpe = sum(1 for u, h in zip(uh_sharpes, h_sharpes) if h > u)
    folds_reduced_dd = sum(1 for u, h in zip(uh_dds, h_dds) if h < u)

    print(f"\n  K-Fold Summary:")
    print(f"    Unhedged mean Sharpe: {np.mean(uh_sharpes):.3f} +/- {np.std(uh_sharpes):.3f}")
    print(f"    Hedged mean Sharpe:   {np.mean(h_sharpes):.3f} +/- {np.std(h_sharpes):.3f}")
    print(f"    Folds w/ improved Sharpe: {folds_improved_sharpe}/{len(FOLDS)}")
    print(f"    Folds w/ reduced MaxDD:   {folds_reduced_dd}/{len(FOLDS)}")
    print(f"    Unhedged mean MaxDD:  ${np.mean(uh_dds):.2f}")
    print(f"    Hedged mean MaxDD:    ${np.mean(h_dds):.2f}")

    # ═══════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    results = {
        'experiment': 'R108 Tail Risk Hedging',
        'elapsed_s': round(elapsed, 1),
        'phase1_vol_proxy': {
            'n_daily': len(dp_vals),
            'mean': round(float(np.mean(dp_vals)), 3),
            'std': round(float(np.std(dp_vals)), 3),
            'percentiles': {str(p): round(float(np.percentile(dp_vals, p)), 3)
                           for p in [50, 80, 90, 95, 99]},
        },
        'phase2_hedge_modes': {
            'base_sharpe': round(base_sharpe, 3),
            'base_maxdd': round(base_maxdd, 2),
            'base_pnl': round(base_pnl, 2),
            'modes': {k: {kk: round(vv, 3) if isinstance(vv, float) else vv
                          for kk, vv in v.items()}
                      for k, v in mode_results.items()},
        },
        'phase3_threshold_sweep': {
            mode_name: {
                str(pct): {kk: round(vv, 3) if isinstance(vv, float) else vv
                           for kk, vv in data.items()}
                for pct, data in mode_data.items()
                if not str(pct).startswith('_')
            }
            for mode_name, mode_data in sweep_results.items()
        },
        'phase3_best': {
            'mode': overall_best_mode,
            'percentile': overall_best_pct,
            'threshold': round(overall_best_threshold, 3),
        },
        'phase4_kfold': {
            'mode': overall_best_mode,
            'percentile': overall_best_pct,
            'folds': kfold_results,
            'summary': {
                'unhedged_mean_sharpe': round(float(np.mean(uh_sharpes)), 3),
                'hedged_mean_sharpe': round(float(np.mean(h_sharpes)), 3),
                'folds_improved_sharpe': folds_improved_sharpe,
                'folds_reduced_dd': folds_reduced_dd,
                'unhedged_mean_maxdd': round(float(np.mean(uh_dds)), 2),
                'hedged_mean_maxdd': round(float(np.mean(h_dds)), 2),
            },
        },
        'recommendation': (
            f"Use {overall_best_mode} @ P{overall_best_pct} (improves {folds_improved_sharpe}/6 folds)"
            if folds_improved_sharpe >= 4
            else f"Hedge {overall_best_mode} marginal ({folds_improved_sharpe}/6 folds improved), "
                 f"but reduces MaxDD in {folds_reduced_dd}/6 folds"
        ),
    }

    out_file = OUTPUT_DIR / "r108_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"  R108 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Best hedge: {overall_best_mode} @ P{overall_best_pct}")
    print(f"  Sharpe improved in {folds_improved_sharpe}/6 folds, "
          f"MaxDD reduced in {folds_reduced_dd}/6 folds")
    print(f"{'='*80}")
    print(f"  Saved: {out_file}")


if __name__ == '__main__':
    main()
