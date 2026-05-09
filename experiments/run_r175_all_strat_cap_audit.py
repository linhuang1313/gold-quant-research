#!/usr/bin/env python3
"""
R175 — All Live Strategy MaxLoss Cap Audit at Actual Lot Sizes
================================================================
For each live strategy, runs a Cap grid at the ACTUAL deployed lot size,
then K-Fold validates the top candidates.

Addresses the R89 design flaw: Cap was optimized at 0.01 lot then linearly
scaled, but Cap is a dollar-absolute threshold with nonlinear lot interaction.

Live config (2026-05-08):
  L8_MAX:      0.02 lot, Cap=$35
  PSAR:        0.09 lot, Cap=$60  (just fixed from $5)
  TSMOM:       0.15 lot, Cap=None (no cap)
  SESS_BO:     0.13 lot, Cap=$35  <-- likely too tight
  DUAL_THRUST: 0.04 lot, Cap=$35
  CHANDELIER:  0.08 lot, Cap=$35
"""
import sys, os, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r175_cap_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30

LIVE_STRATEGIES = {
    'PSAR':        {'lot': 0.09, 'current_cap': 60},
    'SESS_BO':     {'lot': 0.13, 'current_cap': 35},
    'DUAL_THRUST': {'lot': 0.04, 'current_cap': 35},
    'CHANDELIER':  {'lot': 0.08, 'current_cap': 35},
    'TSMOM':       {'lot': 0.15, 'current_cap': 0},
    'L8_MAX':      {'lot': 0.02, 'current_cap': 35},
}

FIXED_CAPS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100]

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-06-01"),
]


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


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


def _run_exit(pos, i, h, lo_v, c, spread, lot, pv, times,
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


# ═══════════════════════════════════════════════════════════════
# Strategy backtest functions (actual lot sizes, not unit lot)
# ═══════════════════════════════════════════════════════════════

def bt_psar(h1_df, spread, lot, maxloss_cap=0,
            sl_atr=4.0, tp_atr=6.0, trail_act=0.08, trail_dist=0.015, max_hold=15):
    df = add_psar(h1_df).dropna(subset=['PSAR_dir', 'ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    pdir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pdir[i-1] == -1 and pdir[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif pdir[i-1] == 1 and pdir[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom(h1_df, spread, lot, maxloss_cap=0,
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
        if c[i-fast] > 0: s += 0.5 * np.sign(c[i]/c[i-fast] - 1.0)
        if c[i-slow] > 0: s += 0.5 * np.sign(c[i]/c[i-slow] - 1.0)
        score[i] = s
    trades = []; pos = None; last_exit = -999
    for i in range(max_lb+1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
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
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
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


def bt_dual_thrust(h1_df, spread, lot, maxloss_cap=35,
                   n_bars=6, k=0.5, sl_atr=4.5, tp_atr=8.0,
                   trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    o = df['Open'].values; atr = df['ATR'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(n_bars, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = np.max(h[i-n_bars:i])
        lc = np.min(c[i-n_bars:i])
        hc = np.max(c[i-n_bars:i])
        ll = np.min(lo[i-n_bars:i])
        rng = max(hh - lc, hc - ll)
        buy_line = o[i] + k * rng
        sell_line = o[i] - k * rng
        if c[i] > buy_line:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < sell_line:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_chandelier(h1_df, spread, lot, maxloss_cap=35,
                  period=22, mult=3.0, ema_period=100,
                  sl_atr=4.5, tp_atr=8.0,
                  trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df, period=period)
    df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    df = df.dropna(subset=['ATR', 'EMA'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; ema = df['EMA'].values
    times = df.index; n = len(df)
    chandelier_long = np.full(n, np.nan)
    chandelier_short = np.full(n, np.nan)
    for i in range(period, n):
        hh = np.max(h[i-period+1:i+1])
        ll = np.min(lo[i-period+1:i+1])
        chandelier_long[i] = hh - mult * atr[i]
        chandelier_short[i] = ll + mult * atr[i]
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
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        flip_bull = direction[i] == 1 and direction[i-1] != 1 and c[i] > ema[i]
        flip_bear = direction[i] == -1 and direction[i-1] != -1 and c[i] < ema[i]
        if flip_bull:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif flip_bear:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


STRAT_BT = {
    'PSAR': bt_psar,
    'TSMOM': bt_tsmom,
    'SESS_BO': bt_sess_bo,
    'DUAL_THRUST': bt_dual_thrust,
    'CHANDELIER': bt_chandelier,
}


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
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0,
                'cap_hits': 0, 'cap_pct': 0, 'max_single_loss': 0}
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
        'max_single_loss': round(min(pnls), 3) if pnls else 0,
    }


def load_h1():
    import glob as _glob
    candidates = sorted(_glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
    if not candidates:
        raise FileNotFoundError("No H1 data found")
    csv_path = candidates[-1]
    print(f"  Loading H1: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    print(f"  {len(df)} bars ({df.index[0]} ~ {df.index[-1]})", flush=True)
    return df


def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("  R175 — All Live Strategy Cap Audit (Actual Lot Sizes)", flush=True)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 80, flush=True)

    h1_df = load_h1()
    mean_atr = compute_atr(h1_df).dropna().mean()
    print(f"  H1 mean ATR: ${mean_atr:.2f}\n", flush=True)

    all_results = {}

    for strat_name in ['PSAR', 'SESS_BO', 'DUAL_THRUST', 'CHANDELIER', 'TSMOM']:
        cfg = LIVE_STRATEGIES[strat_name]
        lot = cfg['lot']
        current_cap = cfg['current_cap']
        bt_fn = STRAT_BT[strat_name]

        print(f"\n{'='*80}", flush=True)
        print(f"  {strat_name}  (lot={lot}, current Cap=${current_cap})", flush=True)
        cur_tol = current_cap / (lot * PV) if current_cap > 0 else float('inf')
        cur_atr_pct = cur_tol / mean_atr * 100 if current_cap > 0 else float('inf')
        print(f"  Current tolerance: ${cur_tol:.1f}/oz = {cur_atr_pct:.0f}% ATR", flush=True)
        print(f"{'='*80}\n", flush=True)

        # Phase 1: Full Cap Grid
        print(f"  Phase 1: Fixed Cap Grid", flush=True)
        results = []
        for cap in FIXED_CAPS:
            trades = bt_fn(h1_df, spread=SPREAD, lot=lot, maxloss_cap=cap)
            stats = _compute_stats(trades)
            label = "NoCap" if cap == 0 else f"Cap${cap}"
            price_tol = cap / (lot * PV) if cap > 0 else float('inf')
            atr_ratio = price_tol / mean_atr if cap > 0 else float('inf')
            stats['label'] = label
            stats['cap'] = cap
            stats['lot'] = lot
            stats['price_tolerance'] = round(price_tol, 2) if cap > 0 else None
            stats['atr_pct'] = round(atr_ratio * 100, 1) if cap > 0 else None
            results.append(stats)

        print(f"\n  {'Label':<10} {'N':>5} {'Sharpe':>7} {'PnL':>11} {'WR':>6} "
              f"{'MaxDD':>9} {'Cap%':>6} {'$/oz':>6} {'ATR%':>6} {'MaxLoss':>9}", flush=True)
        print(f"  {'-'*10} {'-'*5} {'-'*7} {'-'*11} {'-'*6} {'-'*9} {'-'*6} {'-'*6} {'-'*6} {'-'*9}", flush=True)
        for r in results:
            pt = f"{r['price_tolerance']:.1f}" if r['price_tolerance'] else "inf"
            ar = f"{r['atr_pct']:.0f}%" if r['atr_pct'] else "inf"
            marker = " <-- current" if r['cap'] == current_cap else ""
            print(f"  {r['label']:<10} {r['n']:>5} {r['sharpe']:>7.2f} {fmt(r['pnl']):>11} "
                  f"{r['wr']:>5.1f}% {fmt(r['max_dd']):>9} {r['cap_pct']:>5.1f}% "
                  f"${pt:>5} {ar:>5} {fmt(r['max_single_loss']):>9}{marker}", flush=True)

        results_sorted = sorted(results, key=lambda x: x['sharpe'], reverse=True)
        top3_str = ', '.join(f"{r['label']}({r['sharpe']:.2f})" for r in results_sorted[:3])
        print(f"\n  Top 3: {top3_str}", flush=True)

        # Phase 2: K-Fold for top candidates + current + NoCap
        print(f"\n  Phase 2: K-Fold Validation", flush=True)
        kfold_caps = set()
        kfold_caps.add(0)
        kfold_caps.add(current_cap)
        for r in results_sorted[:5]:
            kfold_caps.add(r['cap'])

        kfold_results = {}
        for cap in sorted(kfold_caps):
            label = "NoCap" if cap == 0 else f"Cap${cap}"
            fold_sharpes = []
            for fold_name, start, end in FOLDS:
                fold_data = h1_df[start:end]
                if len(fold_data) < 200:
                    fold_sharpes.append(0)
                    continue
                trades = bt_fn(fold_data, spread=SPREAD, lot=lot, maxloss_cap=cap)
                stats = _compute_stats(trades)
                fold_sharpes.append(stats['sharpe'])

            positive = sum(1 for s in fold_sharpes if s > 0)
            mean_sh = float(np.mean(fold_sharpes))
            min_sh = float(min(fold_sharpes))
            status = "PASS" if positive >= 4 else "FAIL"
            marker = " <-- current" if cap == current_cap else ""

            kfold_results[label] = {
                'cap': cap, 'positive_folds': positive,
                'mean_sharpe': round(mean_sh, 2), 'min_sharpe': round(min_sh, 2),
                'pass': positive >= 4,
                'fold_sharpes': [round(s, 2) for s in fold_sharpes],
            }
            print(f"    {label:<10}: {positive}/6 pos, mean={mean_sh:.2f}, min={min_sh:.2f}  "
                  f"[{status}]  {[round(s,1) for s in fold_sharpes]}{marker}", flush=True)

        # Recommendation
        passing = [(l, d) for l, d in kfold_results.items() if d['pass'] and l != 'NoCap']
        if passing:
            best = max(passing, key=lambda x: x[1]['mean_sharpe'])
            label, data = best
            cap_val = data['cap']
            tol = cap_val / (lot * PV) if cap_val > 0 else float('inf')
            change_needed = cap_val != current_cap
            print(f"\n  RECOMMENDATION: {label}", flush=True)
            print(f"    Price tolerance: ${tol:.2f}/oz = {tol/mean_atr:.0f}% ATR", flush=True)
            print(f"    K-Fold: {data['positive_folds']}/6 pos, mean={data['mean_sharpe']:.2f}", flush=True)
            if change_needed:
                print(f"    *** CHANGE NEEDED: Cap ${current_cap} -> ${cap_val} ***", flush=True)
            else:
                print(f"    Current Cap=${current_cap} is optimal. No change needed.", flush=True)

        all_results[strat_name] = {
            'lot': lot,
            'current_cap': current_cap,
            'grid': results,
            'kfold': kfold_results,
        }

    # L8_MAX via backtest engine (separate handling)
    print(f"\n{'='*80}", flush=True)
    print(f"  L8_MAX  (lot=0.02, current Cap=$35)", flush=True)
    print(f"  Note: Uses backtest engine, separate from H1 strategies", flush=True)
    print(f"  Price tolerance: ${35/(0.02*PV):.1f}/oz = {35/(0.02*PV)/mean_atr:.0f}% ATR", flush=True)
    print(f"  L8_MAX Cap is very loose at 0.02 lot -- skipping grid, likely no change needed.", flush=True)
    print(f"{'='*80}\n", flush=True)

    # Summary
    print(f"\n{'='*80}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'='*80}\n", flush=True)
    print(f"  {'Strategy':<14} {'Lot':>5} {'Cur Cap':>8} {'$/oz':>6} {'ATR%':>6} {'Recommended':>12} {'Action':>12}", flush=True)
    print(f"  {'-'*14} {'-'*5} {'-'*8} {'-'*6} {'-'*6} {'-'*12} {'-'*12}", flush=True)

    for strat_name in ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO', 'DUAL_THRUST', 'CHANDELIER']:
        cfg = LIVE_STRATEGIES[strat_name]
        lot = cfg['lot']
        cur = cfg['current_cap']
        tol = cur / (lot * PV) if cur > 0 else float('inf')
        atr_pct = tol / mean_atr * 100 if cur > 0 else float('inf')
        tol_str = f"${tol:.1f}" if cur > 0 else "inf"
        atr_str = f"{atr_pct:.0f}%" if cur > 0 else "N/A"

        if strat_name in all_results:
            kf = all_results[strat_name]['kfold']
            passing = [(l, d) for l, d in kf.items() if d['pass'] and l != 'NoCap']
            if passing:
                best = max(passing, key=lambda x: x[1]['mean_sharpe'])
                rec_cap = best[1]['cap']
                rec_str = f"Cap${rec_cap}" if rec_cap > 0 else "NoCap"
                action = "CHANGE" if rec_cap != cur else "OK"
            else:
                rec_str = "N/A"
                action = "REVIEW"
        elif strat_name == 'L8_MAX':
            rec_str = "Cap$35"
            action = "OK (loose)"
        else:
            rec_str = "?"
            action = "?"

        print(f"  {strat_name:<14} {lot:>5.2f} {'$'+str(cur) if cur > 0 else 'None':>8} "
              f"{tol_str:>6} {atr_str:>6} {rec_str:>12} {action:>12}", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*80}", flush=True)
    print(f"  R175 complete in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"{'='*80}", flush=True)

    with open(OUTPUT_DIR / "r175_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r175_results.json", flush=True)


if __name__ == "__main__":
    main()
