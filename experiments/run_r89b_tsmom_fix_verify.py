#!/usr/bin/env python3
"""
R89-B — Quick verification that corrected TSMOM signal matches R53/R56/live EA.
Expected: ~975 trades (vs 121 with the old SMA bug).
"""
import sys, os, time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH

PV = 100
SPREAD = 0.30


def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


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


def bt_tsmom_score(h1_df, spread, lot, maxloss_cap=0,
                   fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
                   trail_act=0.14, trail_dist=0.025, max_hold=20):
    """Correct Score-based TSMOM (matches live EA)."""
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
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom_sma(h1_df, spread, lot, maxloss_cap=0,
                 fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
                 trail_act=0.14, trail_dist=0.025, max_hold=20):
    """Old buggy SMA-based TSMOM (for comparison)."""
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


def sharpe(arr):
    if len(arr) < 10 or np.std(arr, ddof=1) == 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))


def trades_to_daily(trades):
    if not trades:
        return np.array([0.0])
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    return np.array([daily[k] for k in sorted(daily.keys())])


def max_dd(arr):
    if len(arr) == 0:
        return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def main():
    print("=" * 70)
    print("  R89-B: TSMOM Signal Fix Verification")
    print("  Comparing Score method (live EA) vs SMA crossover (R89 bug)")
    print("=" * 70, flush=True)

    print("\n  Loading H1 data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0].date()} ~ {h1_df.index[-1].date()})")

    lot = 0.08
    lots_to_test = [0.04, 0.06, 0.08, 0.10]

    print(f"\n{'='*70}")
    print(f"  Test 1: Side-by-side comparison (lot={lot})")
    print(f"{'='*70}")

    # Score method (correct)
    t0 = time.time()
    trades_score = bt_tsmom_score(h1_df, SPREAD, lot, maxloss_cap=0)
    elapsed_score = time.time() - t0

    # SMA method (buggy)
    t0 = time.time()
    trades_sma = bt_tsmom_sma(h1_df, SPREAD, lot, maxloss_cap=0)
    elapsed_sma = time.time() - t0

    for label, trades, elapsed in [("Score (LIVE EA)", trades_score, elapsed_score),
                                    ("SMA (R89 BUG)", trades_sma, elapsed_sma)]:
        daily = trades_to_daily(trades)
        pnls = [t['pnl'] for t in trades]
        n = len(trades)
        wr = sum(1 for p in pnls if p > 0) / max(n, 1) * 100
        sh = sharpe(daily)
        dd = max_dd(daily)
        tot = sum(pnls)
        print(f"\n  {label}:")
        print(f"    Trades: {n} ({n/11:.1f}/year)")
        print(f"    Sharpe: {sh:.2f}")
        print(f"    PnL:    ${tot:.0f}")
        print(f"    WinRate: {wr:.1f}%")
        print(f"    MaxDD:  ${dd:.0f}")
        print(f"    Time:   {elapsed:.1f}s")

        # Exit reason breakdown
        reasons = {}
        for t in trades:
            reasons[t['reason']] = reasons.get(t['reason'], 0) + 1
        print(f"    Exits:  {dict(sorted(reasons.items(), key=lambda x: -x[1]))}")

    print(f"\n{'='*70}")
    print(f"  Test 2: Score method at multiple lot sizes (no Cap)")
    print(f"{'='*70}")
    print(f"\n  {'Lot':>6} {'Trades':>7} {'Sharpe':>7} {'PnL':>8} {'WR%':>6} {'MaxDD':>8}")
    print(f"  {'-'*6} {'-'*7} {'-'*7} {'-'*8} {'-'*6} {'-'*8}")
    for l in lots_to_test:
        trades = bt_tsmom_score(h1_df, SPREAD, l, maxloss_cap=0)
        daily = trades_to_daily(trades)
        pnls = [t['pnl'] for t in trades]
        n = len(trades)
        wr = sum(1 for p in pnls if p > 0) / max(n, 1) * 100
        sh = sharpe(daily)
        dd = max_dd(daily)
        tot = sum(pnls)
        print(f"  {l:>6.2f} {n:>7} {sh:>7.2f} {tot:>8.0f} {wr:>5.1f}% {dd:>8.0f}")

    print(f"\n{'='*70}")
    print(f"  Test 3: Score method with Cap grid")
    print(f"{'='*70}")
    caps_to_test = [0, 5, 10, 15, 20, 25, 30, 35, 50]
    print(f"\n  {'Cap':>5} {'Trades':>7} {'Sharpe':>7} {'PnL':>8} {'WR%':>6} {'MaxDD':>8}")
    print(f"  {'-'*5} {'-'*7} {'-'*7} {'-'*8} {'-'*6} {'-'*8}")
    for cap in caps_to_test:
        trades = bt_tsmom_score(h1_df, SPREAD, lot, maxloss_cap=cap)
        daily = trades_to_daily(trades)
        pnls = [t['pnl'] for t in trades]
        n = len(trades)
        wr = sum(1 for p in pnls if p > 0) / max(n, 1) * 100
        sh = sharpe(daily)
        dd = max_dd(daily)
        tot = sum(pnls)
        print(f"  ${cap:>4} {n:>7} {sh:>7.2f} {tot:>8.0f} {wr:>5.1f}% {dd:>8.0f}")

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
