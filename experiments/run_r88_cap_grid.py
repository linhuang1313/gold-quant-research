#!/usr/bin/env python3
"""
R88 — Per-Strategy MaxLoss Cap Grid Search
============================================
Tests optimal Cap for each active strategy at its actual live lot size.

Live lot sizes (updated):
  L8_MAX (KC):  0.05    TSMOM:   0.04
  SESS_BO:      0.02    PSAR:    0.01

Test grid: Cap $5-$80 (step $5) + NoCap baseline
Dimensions: Full sample + Recent (2023-2026)
Validation: Top 3 Caps per strategy -> K-Fold 6-fold
"""
import sys, os, io, time, json
import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r88_cap_grid")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100

# Live lot sizes per strategy (user-confirmed)
LIVE_LOTS = {
    'L8_MAX':  0.05,
    'TSMOM':   0.04,
    'SESS_BO': 0.02,
    'PSAR':    0.01,
}

# Cap grid
CAP_VALUES = list(range(5, 85, 5))  # $5, $10, ..., $80

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


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


def add_kc(df, ema_period=20, atr_period=14, mult=1.5):
    df = df.copy()
    df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    df['ATR'] = tr.rolling(atr_period).mean()
    df['KC_upper'] = df['EMA'] + mult * df['ATR']
    df['KC_lower'] = df['EMA'] - mult * df['ATR']
    df['ADX'] = compute_adx(df, atr_period)
    return df


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
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0, 'cap_hits': 0, 'cap_pct': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    cap_hits = sum(1 for t in trades if t.get('reason') == 'MaxLossCap')
    return {
        'n': n,
        'sharpe': round(_sharpe(daily), 2),
        'pnl': round(sum(pnls), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'max_dd': round(_max_dd(daily), 2),
        'cap_hits': cap_hits,
        'cap_pct': round(cap_hits / n * 100, 1),
    }


# ═══════════════════════════════════════════════════════════════
# Exit logic with MaxLoss Cap support
# ═══════════════════════════════════════════════════════════════

def _run_exit_with_cap(pos, i, h, lo_v, c, spread, lot, pv, times,
                       sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold,
                       maxloss_cap=0):
    """Unified exit logic with optional MaxLoss Cap."""
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

    # MaxLoss Cap: check close-based floating PnL
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


# ═══════════════════════════════════════════════════════════════
# Strategy backtests with Cap parameter
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
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pdir[i-1] == -1 and pdir[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif pdir[i-1] == 1 and pdir[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom(h1_df, spread, lot, maxloss_cap=0,
             fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
             trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df['fma'] = df['Close'].rolling(fast).mean()
    df['sma'] = df['Close'].rolling(slow).mean()
    df = df.dropna(subset=['ATR', 'fma', 'sma'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; fm = df['fma'].values; sm = df['sma'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            # Reversal exit (TSMOM-specific)
            if pos['dir'] == 'BUY' and fm[i] < sm[i]:
                pnl = (c[i]-pos['entry']-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and fm[i] > sm[i]:
                pnl = (pos['entry']-c[i]-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if fm[i] > sm[i] and fm[i-1] <= sm[i-1]:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif fm[i] < sm[i] and fm[i-1] >= sm[i-1]:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
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
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_kc(df_prepared, spread, lot, maxloss_cap=0,
          adx_thresh=18, sl_atr=3.5, tp_atr=8.0,
          trail_act=0.28, trail_dist=0.06, max_hold=20):
    """KC (Keltner Channel) backtest with cap. Works for D1/H4/H1 timeframes."""
    df = df_prepared; trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    kc_up = df['KC_upper'].values; kc_lo = df['KC_lower'].values
    atr = df['ATR'].values; adx = df['ADX'].values
    times = df.index; n = len(df); last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, high[i], low[i], close[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(adx[i]) or adx[i] < adx_thresh: continue
        if close[i] > kc_up[i] and close[i-1] <= kc_up[i-1]:
            pos = {'dir': 'BUY', 'entry': close[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif close[i] < kc_lo[i] and close[i-1] >= kc_lo[i-1]:
            pos = {'dir': 'SELL', 'entry': close[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_l8_max(data_bundle, spread, lot, maxloss_cap=0):
    """L8_MAX via engine with MaxLoss Cap. data_bundle must be a prepared DataBundle."""
    from backtest.runner import run_variant, LIVE_PARITY_KWARGS
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
            'pnl': t.pnl, 'reason': t.exit_reason,
            'bars': 0,
        })
    return trades


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80)
    print("  R88 — Per-Strategy MaxLoss Cap Grid Search")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80, flush=True)

    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH, DataBundle
    print("\n  Loading data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})")

    # Prepare DataBundle for L8_MAX engine (needs KC indicators)
    print("  Preparing L8_MAX DataBundle (with indicators)...", flush=True)
    l8_bundle = DataBundle.load_custom()
    print(f"  L8 bundle ready.", flush=True)

    SPREAD = 0.30

    # Strategy configs: (name, bt_func, bt_kwargs, data)
    strat_configs = [
        ('PSAR',    bt_psar,    {},  h1_df),
        ('TSMOM',   bt_tsmom,   {},  h1_df),
        ('SESS_BO', bt_sess_bo, {},  h1_df),
    ]

    all_results = {}

    # ── Phase 1: Full-sample Cap Grid for H1 strategies + KC strategies ──
    print(f"\n{'='*80}")
    print(f"  Phase 1: Full-Sample Cap Grid (4 strategies x {len(CAP_VALUES)+1} caps)")
    print(f"{'='*80}\n", flush=True)

    for strat_name, bt_fn, bt_kw, data in strat_configs:
        lot = LIVE_LOTS[strat_name]
        print(f"\n  --- {strat_name} (lot={lot}) ---", flush=True)

        caps_to_test = [0] + CAP_VALUES  # 0 = NoCap
        results = []
        for cap in caps_to_test:
            trades = bt_fn(data, spread=SPREAD, lot=lot, maxloss_cap=cap, **bt_kw)
            stats = _compute_stats(trades)
            label = f"NoCap" if cap == 0 else f"Cap${cap}"
            # Price tolerance for context
            price_tol = cap / (lot * PV) if cap > 0 else float('inf')
            stats['label'] = label
            stats['cap'] = cap
            stats['lot'] = lot
            stats['price_tolerance'] = round(price_tol, 2) if cap > 0 else None
            results.append(stats)

        # Sort by Sharpe
        results.sort(key=lambda x: x['sharpe'], reverse=True)

        print(f"  {'Label':<10} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} "
              f"{'MaxDD':>10} {'CapHits':>8} {'Cap%':>7} {'PriceTol':>10}")
        print(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*12} {'-'*7} {'-'*10} {'-'*8} {'-'*7} {'-'*10}")
        for r in results:
            pt = f"${r['price_tolerance']:.1f}" if r['price_tolerance'] else "inf"
            print(f"  {r['label']:<10} {r['n']:>6} {r['sharpe']:>8.2f} {fmt(r['pnl']):>12} "
                  f"{r['wr']:>6.1f}% {fmt(r['max_dd']):>10} {r['cap_hits']:>8} "
                  f"{r['cap_pct']:>6.1f}% {pt:>10}", flush=True)

        best = results[0]
        print(f"\n  >>> Best: {best['label']} (Sharpe={best['sharpe']:.2f})", flush=True)
        all_results[strat_name] = {'full_sample': results}

    # ── Phase 1b: L8_MAX (uses engine) ──
    print(f"\n  --- L8_MAX (lot={LIVE_LOTS['L8_MAX']}, engine-based) ---", flush=True)
    lot = LIVE_LOTS['L8_MAX']
    l8_results = []
    for cap in [0] + CAP_VALUES:
        trades = bt_l8_max(l8_bundle, spread=SPREAD, lot=lot, maxloss_cap=cap)
        stats = _compute_stats(trades)
        label = f"NoCap" if cap == 0 else f"Cap${cap}"
        price_tol = cap / (lot * PV) if cap > 0 else float('inf')
        stats['label'] = label
        stats['cap'] = cap
        stats['lot'] = lot
        stats['price_tolerance'] = round(price_tol, 2) if cap > 0 else None
        # Count MaxLossCap from engine exit reason
        stats['cap_hits'] = sum(1 for t in trades if 'MaxLoss' in str(t.get('reason', '')))
        stats['cap_pct'] = round(stats['cap_hits'] / max(stats['n'], 1) * 100, 1)
        l8_results.append(stats)

    l8_results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"  {'Label':<10} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} "
          f"{'MaxDD':>10} {'CapHits':>8} {'Cap%':>7} {'PriceTol':>10}")
    print(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*12} {'-'*7} {'-'*10} {'-'*8} {'-'*7} {'-'*10}")
    for r in l8_results:
        pt = f"${r['price_tolerance']:.1f}" if r['price_tolerance'] else "inf"
        print(f"  {r['label']:<10} {r['n']:>6} {r['sharpe']:>8.2f} {fmt(r['pnl']):>12} "
              f"{r['wr']:>6.1f}% {fmt(r['max_dd']):>10} {r['cap_hits']:>8} "
              f"{r['cap_pct']:>6.1f}% {pt:>10}", flush=True)
    best = l8_results[0]
    print(f"\n  >>> Best: {best['label']} (Sharpe={best['sharpe']:.2f})", flush=True)
    all_results['L8_MAX'] = {'full_sample': l8_results}

    # ── Phase 2: Recent period (2023-2026) ──
    print(f"\n{'='*80}")
    print(f"  Phase 2: Recent Period (2023-2026)")
    print(f"{'='*80}\n", flush=True)

    h1_recent = h1_df["2023-01-01":"2026-05-01"]

    recent_configs = [
        ('PSAR',    bt_psar,    {},  h1_recent),
        ('TSMOM',   bt_tsmom,   {},  h1_recent),
        ('SESS_BO', bt_sess_bo, {},  h1_recent),
    ]

    for strat_name, bt_fn, bt_kw, data in recent_configs:
        lot = LIVE_LOTS[strat_name]
        # Use top 5 caps from full sample + NoCap
        full_top5 = [r['cap'] for r in all_results[strat_name]['full_sample'][:5]]
        if 0 not in full_top5:
            full_top5.append(0)
        caps_to_test = sorted(set(full_top5))

        results = []
        for cap in caps_to_test:
            trades = bt_fn(data, spread=SPREAD, lot=lot, maxloss_cap=cap, **bt_kw)
            stats = _compute_stats(trades)
            label = f"NoCap" if cap == 0 else f"Cap${cap}"
            stats['label'] = label; stats['cap'] = cap
            results.append(stats)

        results.sort(key=lambda x: x['sharpe'], reverse=True)
        print(f"  {strat_name} (recent): Best={results[0]['label']} Sharpe={results[0]['sharpe']:.2f} "
              f"PnL={fmt(results[0]['pnl'])}", flush=True)
        all_results[strat_name]['recent'] = results

    # L8_MAX recent
    l8_bundle_recent = l8_bundle.slice("2023-01-01", "2026-05-01")
    l8_recent_results = []
    full_top5 = [r['cap'] for r in all_results['L8_MAX']['full_sample'][:5]]
    if 0 not in full_top5:
        full_top5.append(0)
    for cap in sorted(set(full_top5)):
        trades = bt_l8_max(l8_bundle_recent, spread=SPREAD, lot=LIVE_LOTS['L8_MAX'], maxloss_cap=cap)
        stats = _compute_stats(trades)
        label = f"NoCap" if cap == 0 else f"Cap${cap}"
        stats['label'] = label; stats['cap'] = cap
        stats['cap_hits'] = sum(1 for t in trades if 'MaxLoss' in str(t.get('reason', '')))
        stats['cap_pct'] = round(stats['cap_hits'] / max(stats['n'], 1) * 100, 1)
        l8_recent_results.append(stats)
    l8_recent_results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"  L8_MAX (recent): Best={l8_recent_results[0]['label']} "
          f"Sharpe={l8_recent_results[0]['sharpe']:.2f} PnL={fmt(l8_recent_results[0]['pnl'])}", flush=True)
    all_results['L8_MAX']['recent'] = l8_recent_results

    # ── Phase 3: K-Fold for top 3 per strategy ──
    print(f"\n{'='*80}")
    print(f"  Phase 3: K-Fold 6-Fold Validation (top 3 per strategy)")
    print(f"{'='*80}\n", flush=True)

    for strat_name, bt_fn, bt_kw, data_full in strat_configs:
        lot = LIVE_LOTS[strat_name]
        top3_caps = [r['cap'] for r in all_results[strat_name]['full_sample'][:3]]
        print(f"\n  --- {strat_name} K-Fold: testing Caps {top3_caps} ---", flush=True)

        kfold_results = {}
        for cap in top3_caps:
            label = f"NoCap" if cap == 0 else f"Cap${cap}"
            fold_sharpes = []
            for fold_name, start, end in FOLDS:
                fold_data = h1_df[start:end]
                if len(fold_data) < 100:
                    fold_sharpes.append(0)
                    continue
                trades = bt_fn(fold_data, spread=SPREAD, lot=lot, maxloss_cap=cap, **bt_kw)
                daily = _trades_to_daily(trades)
                fold_sharpes.append(_sharpe(daily))

            positive = sum(1 for s in fold_sharpes if s > 0)
            mean_sh = np.mean(fold_sharpes)
            kfold_results[label] = {
                'cap': cap,
                'fold_sharpes': [round(s, 2) for s in fold_sharpes],
                'positive_folds': positive,
                'mean_sharpe': round(float(mean_sh), 2),
                'pass_4of6': positive >= 4,
            }
            status = "PASS" if positive >= 4 else "FAIL"
            print(f"    {label:<10}: {positive}/6 positive, mean={mean_sh:.2f}  [{status}]  "
                  f"folds={[round(s,1) for s in fold_sharpes]}", flush=True)

        all_results[strat_name]['kfold'] = kfold_results

    # L8_MAX K-Fold
    print(f"\n  --- L8_MAX K-Fold ---", flush=True)
    top3_caps = [r['cap'] for r in all_results['L8_MAX']['full_sample'][:3]]
    kfold_l8 = {}
    for cap in top3_caps:
        label = f"NoCap" if cap == 0 else f"Cap${cap}"
        fold_sharpes = []
        for fold_name, start, end in FOLDS:
            try:
                l8_fold = l8_bundle.slice(start, end)
            except Exception:
                fold_sharpes.append(0)
                continue
            trades = bt_l8_max(l8_fold, spread=SPREAD,
                               lot=LIVE_LOTS['L8_MAX'], maxloss_cap=cap)
            daily = _trades_to_daily(trades)
            fold_sharpes.append(_sharpe(daily))
        positive = sum(1 for s in fold_sharpes if s > 0)
        mean_sh = np.mean(fold_sharpes)
        kfold_l8[label] = {
            'cap': cap, 'fold_sharpes': [round(s, 2) for s in fold_sharpes],
            'positive_folds': positive, 'mean_sharpe': round(float(mean_sh), 2),
            'pass_4of6': positive >= 4,
        }
        status = "PASS" if positive >= 4 else "FAIL"
        print(f"    {label:<10}: {positive}/6 positive, mean={mean_sh:.2f}  [{status}]", flush=True)
    all_results['L8_MAX']['kfold'] = kfold_l8

    # ── Summary ──
    print(f"\n{'='*80}")
    print(f"  SUMMARY — Recommended Cap per Strategy")
    print(f"{'='*80}\n")

    summary = {}
    for strat_name in ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']:
        lot = LIVE_LOTS[strat_name]
        full_best = all_results[strat_name]['full_sample'][0]
        recent_best = all_results[strat_name].get('recent', [{}])[0]
        # Recommend: cap that is best in full AND passes K-Fold
        kfold = all_results[strat_name].get('kfold', {})
        recommended = None
        for r in all_results[strat_name]['full_sample'][:5]:
            label = r['label']
            if label in kfold and kfold[label].get('pass_4of6', False):
                recommended = r
                break

        cap_val = recommended['cap'] if recommended else full_best['cap']
        price_tol = cap_val / (lot * PV) if cap_val > 0 else float('inf')
        summary[strat_name] = {
            'lot': lot,
            'recommended_cap': cap_val,
            'price_tolerance': round(price_tol, 2) if cap_val > 0 else None,
            'full_best': full_best['label'],
            'full_best_sharpe': full_best['sharpe'],
            'recent_best': recent_best.get('label', 'N/A'),
            'kfold_pass': recommended is not None,
        }
        pt_str = f"${price_tol:.1f}" if cap_val > 0 else "N/A"
        print(f"  {strat_name:>10} (lot={lot}):  Cap=${cap_val if cap_val > 0 else 'None':<5}  "
              f"PriceTol={pt_str:<8}  FullSharpe={full_best['sharpe']:.2f}  "
              f"KFold={'PASS' if recommended else 'N/A'}", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  R88 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}")

    # Save
    output = {'summary': summary, 'details': {}}
    for k, v in all_results.items():
        output['details'][k] = v
    output['elapsed_s'] = round(elapsed, 1)
    output['live_lots'] = LIVE_LOTS

    with open(OUTPUT_DIR / "r88_cap_grid.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r88_cap_grid.json", flush=True)


if __name__ == "__main__":
    main()
