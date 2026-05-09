#!/usr/bin/env python3
"""
R147 — S3/S4 Era Robustness: Performance across different start years
=====================================================================
Tests S3 Dual Thrust, S4 Chandelier, and S3+S4 combo starting from
every year (2015-2024) to check if performance depends on a specific era.

Also tests end-year truncation to check if recent data inflates results.
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import load_csv

OUTPUT_DIR = Path("results/r147_era_robustness")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; CAPS = 35

H1_CANDIDATES = [
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-27.csv"),
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv"),
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-03-25.csv"),
]

# ═══════════════════════════════════════════════════════════════
# Indicator & helper functions (from R146)
# ═══════════════════════════════════════════════════════════════

def calc_dual_thrust_range(df, n_bars=6):
    hh = df['High'].rolling(n_bars).max()
    lc = df['Close'].rolling(n_bars).min()
    hc = df['Close'].rolling(n_bars).max()
    ll = df['Low'].rolling(n_bars).min()
    return pd.concat([hh - lc, hc - ll], axis=1).max(axis=1)

def calc_chandelier(df, period=22, mult=3.0):
    atr = (df['High'] - df['Low']).rolling(14).mean()
    hh = df['High'].rolling(period).max()
    ll = df['Low'].rolling(period).min()
    out = pd.DataFrame(index=df.index)
    out['Chand_long'] = hh - mult * atr
    out['Chand_short'] = ll + mult * atr
    return out

def compute_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _mk(pos, exit_px, exit_time, reason, bar_i, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_px,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_i - pos['bar']}

def _run_exit(pos, i, hi, lo, cl, spread, lot, pv, times,
              sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap):
    atr = pos['atr']
    sl = atr * sl_atr; tp = atr * tp_atr; bars = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_now = (cl - pos['entry'] - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if lo <= pos['entry'] - sl:
            return _mk(pos, pos['entry'] - sl, times[i], "SL", i, -sl * lot * pv)
        if hi >= pos['entry'] + tp:
            return _mk(pos, pos['entry'] + tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry']); extreme = max(extreme, hi); pos['extreme'] = extreme
        if extreme - pos['entry'] >= atr * trail_act:
            trail_price = extreme - atr * trail_dist
            if cl <= trail_price:
                return _mk(pos, trail_price, times[i], "Trail", i, (trail_price - pos['entry'] - spread) * lot * pv)
        if bars >= max_hold:
            return _mk(pos, cl, times[i], "TimeExit", i, pnl_now)
    else:
        pnl_now = (pos['entry'] - cl - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if hi >= pos['entry'] + sl:
            return _mk(pos, pos['entry'] + sl, times[i], "SL", i, -sl * lot * pv)
        if lo <= pos['entry'] - tp:
            return _mk(pos, pos['entry'] - tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry']); extreme = min(extreme, lo); pos['extreme'] = extreme
        if pos['entry'] - extreme >= atr * trail_act:
            trail_price = extreme + atr * trail_dist
            if cl >= trail_price:
                return _mk(pos, trail_price, times[i], "Trail", i, (pos['entry'] - trail_price - spread) * lot * pv)
        if bars >= max_hold:
            return _mk(pos, cl, times[i], "TimeExit", i, pnl_now)
    return None


def bt_s3(h1_df, spread, lot, start=None, end=None):
    df = h1_df.copy()
    df['ATR14'] = compute_atr(df, 14)
    dt_range = calc_dual_thrust_range(df, 6)
    daily_open = df.groupby(df.index.date)['Open'].transform('first')
    sig = pd.Series(0, index=df.index)
    sig[df['Close'] > daily_open + 0.5 * dt_range] = 1
    sig[df['Close'] < daily_open - 0.5 * dt_range] = -1

    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr_arr = df['ATR14'].values; times = df.index; n = len(df)
    sig_arr = sig.values; dates = df.index.date

    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if start and str(dates[i]) < start: continue
        if end and str(dates[i]) > end: break
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               4.5, 8.0, 0.14, 0.025, 20, CAPS)
            if result: trades.append(result); pos = None; last_exit = i
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr_arr[i]) or atr_arr[i] < 0.1: continue
        if sig_arr[i] == 1 and sig_arr[i-1] != 1:
            pos = {'dir': 'BUY', 'entry': c[i] + spread/2, 'bar': i, 'time': times[i], 'atr': atr_arr[i]}
        elif sig_arr[i] == -1 and sig_arr[i-1] != -1:
            pos = {'dir': 'SELL', 'entry': c[i] - spread/2, 'bar': i, 'time': times[i], 'atr': atr_arr[i]}
    return trades


def bt_s4(h1_df, spread, lot, start=None, end=None):
    df = h1_df.copy()
    df['ATR14'] = compute_atr(df, 14)
    ch = calc_chandelier(df, 22, 3.0)
    ema100 = df['Close'].ewm(span=100).mean()
    above_long = df['Close'] > ch['Chand_long']
    flip_bull = above_long & (~above_long.shift(1).fillna(False))
    below_short = df['Close'] < ch['Chand_short']
    flip_bear = below_short & (~below_short.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index)
    sig[flip_bull & (df['Close'] > ema100)] = 1
    sig[flip_bear & (df['Close'] < ema100)] = -1

    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr_arr = df['ATR14'].values; times = df.index; n = len(df)
    sig_arr = sig.values; dates = df.index.date

    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if start and str(dates[i]) < start: continue
        if end and str(dates[i]) > end: break
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               4.5, 8.0, 0.14, 0.025, 20, CAPS)
            if result: trades.append(result); pos = None; last_exit = i
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr_arr[i]) or atr_arr[i] < 0.1: continue
        if sig_arr[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i] + spread/2, 'bar': i, 'time': times[i], 'atr': atr_arr[i]}
        elif sig_arr[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i] - spread/2, 'bar': i, 'time': times[i], 'atr': atr_arr[i]}
    return trades


def stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0.0, 'pnl': 0.0, 'wr': 0.0, 'max_dd': 0.0, 'pnl_per_trade': 0.0, 'years': 0.0}
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    arr = np.array([daily[k] for k in sorted(daily.keys())])
    eq = np.cumsum(arr)
    dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 1 else 0.0
    sh = float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252)) if len(arr) > 10 and np.std(arr, ddof=1) > 0 else 0.0
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    total = sum(pnls)
    dates_sorted = sorted(daily.keys())
    years = (dates_sorted[-1] - dates_sorted[0]).days / 365.25 if len(dates_sorted) > 1 else 1.0
    return {
        'n': n, 'sharpe': round(sh, 2), 'pnl': round(total, 1),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'max_dd': round(dd, 1), 'pnl_per_trade': round(total / n, 2),
        'years': round(years, 1),
    }


def main():
    t0 = time.time()
    csv_path = next((p for p in H1_CANDIDATES if p.exists()), H1_CANDIDATES[0])
    print(f"Loading H1: {csv_path}")
    h1 = load_csv(str(csv_path))
    print(f"  {len(h1)} bars: {h1.index[0]} -> {h1.index[-1]}\n")

    LOT = 0.04  # same as R146 optimal s3_lot

    # ══════════════════════════════════════════════════════════
    # Phase 1: Different start years, all run to end of data
    # ══════════════════════════════════════════════════════════
    print("=" * 80)
    print("  Phase 1: Performance by Start Year (run to end)")
    print("=" * 80)

    start_years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    phase1 = []

    hdr = f"  {'Start':>7s}  {'Strat':>5s}  {'N':>5s}  {'Sharpe':>7s}  {'PnL':>10s}  {'WR':>5s}  {'MaxDD':>8s}  {'$/trade':>8s}  {'Yrs':>4s}"
    print(hdr)
    print("  " + "-" * 75)

    for year in start_years:
        start = f"{year}-01-01"
        s3_t = bt_s3(h1, SPREAD, LOT, start=start)
        s4_t = bt_s4(h1, SPREAD, LOT, start=start)
        combo_t = s3_t + s4_t

        for name, trades in [("S3", s3_t), ("S4", s4_t), ("S3+S4", combo_t)]:
            st = stats(trades)
            phase1.append({'start': year, 'strategy': name, **st})
            print(f"  {year:>7d}  {name:>5s}  {st['n']:>5d}  {st['sharpe']:>7.2f}  "
                  f"${st['pnl']:>9.0f}  {st['wr']:>4.1f}%  ${st['max_dd']:>7.0f}  "
                  f"${st['pnl_per_trade']:>7.2f}  {st['years']:>4.1f}")
        print()

    # ══════════════════════════════════════════════════════════
    # Phase 2: Fixed 2-year windows (non-overlapping eras)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 2: Fixed 2-Year Windows (isolated eras)")
    print("=" * 80)

    windows = [
        ("2015-2016", "2015-01-01", "2016-12-31"),
        ("2017-2018", "2017-01-01", "2018-12-31"),
        ("2019-2020", "2019-01-01", "2020-12-31"),
        ("2021-2022", "2021-01-01", "2022-12-31"),
        ("2023-2024", "2023-01-01", "2024-12-31"),
        ("2025-2026", "2025-01-01", "2026-12-31"),
    ]

    phase2 = []
    hdr2 = f"  {'Window':>11s}  {'Strat':>5s}  {'N':>5s}  {'Sharpe':>7s}  {'PnL':>10s}  {'WR':>5s}  {'MaxDD':>8s}  {'$/trade':>8s}"
    print(hdr2)
    print("  " + "-" * 70)

    for label, ws, we in windows:
        s3_t = bt_s3(h1, SPREAD, LOT, start=ws, end=we)
        s4_t = bt_s4(h1, SPREAD, LOT, start=ws, end=we)
        combo_t = s3_t + s4_t

        for name, trades in [("S3", s3_t), ("S4", s4_t), ("S3+S4", combo_t)]:
            st = stats(trades)
            phase2.append({'window': label, 'strategy': name, **st})
            print(f"  {label:>11s}  {name:>5s}  {st['n']:>5d}  {st['sharpe']:>7.2f}  "
                  f"${st['pnl']:>9.0f}  {st['wr']:>4.1f}%  ${st['max_dd']:>7.0f}  "
                  f"${st['pnl_per_trade']:>7.2f}")
        print()

    # ══════════════════════════════════════════════════════════
    # Phase 3: Rolling 3-year windows (1-year step)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 3: Rolling 3-Year Windows (S3+S4 combo only)")
    print("=" * 80)

    phase3 = []
    hdr3 = f"  {'Window':>11s}  {'N':>5s}  {'Sharpe':>7s}  {'PnL':>10s}  {'WR':>5s}  {'MaxDD':>8s}  {'$/trade':>8s}  {'Ann.PnL':>9s}"
    print(hdr3)
    print("  " + "-" * 70)

    for y in range(2015, 2024):
        ws = f"{y}-01-01"
        we = f"{y+3}-01-01"
        s3_t = bt_s3(h1, SPREAD, LOT, start=ws, end=we)
        s4_t = bt_s4(h1, SPREAD, LOT, start=ws, end=we)
        combo_t = s3_t + s4_t
        st = stats(combo_t)
        ann_pnl = st['pnl'] / st['years'] if st['years'] > 0 else 0
        phase3.append({'window': f"{y}-{y+3}", **st, 'ann_pnl': round(ann_pnl, 0)})
        print(f"  {y}-{y+3:>4d}  {st['n']:>5d}  {st['sharpe']:>7.2f}  "
              f"${st['pnl']:>9.0f}  {st['wr']:>4.1f}%  ${st['max_dd']:>7.0f}  "
              f"${st['pnl_per_trade']:>7.2f}  ${ann_pnl:>8.0f}")

    # ══════════════════════════════════════════════════════════
    # Phase 4: Annualized $/trade consistency check
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 4: Summary — Is S3+S4 era-dependent?")
    print("=" * 80)

    combo_2yr = [r for r in phase2 if r['strategy'] == 'S3+S4']
    sharpes = [r['sharpe'] for r in combo_2yr if r['n'] > 0]
    ppt = [r['pnl_per_trade'] for r in combo_2yr if r['n'] > 0]
    print(f"\n  2-Year Window Sharpes: {sharpes}")
    print(f"  Min={min(sharpes):.2f}  Max={max(sharpes):.2f}  Std={np.std(sharpes):.2f}  CV={np.std(sharpes)/np.mean(sharpes)*100:.0f}%")
    print(f"  All positive? {'YES' if all(s > 0 for s in sharpes) else 'NO'}")
    print(f"\n  $/Trade by window: {ppt}")
    print(f"  Min=${min(ppt):.2f}  Max=${max(ppt):.2f}")

    worst = min(combo_2yr, key=lambda x: x['sharpe'])
    best = max(combo_2yr, key=lambda x: x['sharpe'])
    print(f"\n  Worst era: {worst['window']} (Sharpe {worst['sharpe']:.2f})")
    print(f"  Best  era: {best['window']} (Sharpe {best['sharpe']:.2f})")
    print(f"  Ratio best/worst: {best['sharpe']/worst['sharpe']:.1f}x" if worst['sharpe'] > 0 else "")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")

    results = {
        'experiment': 'R147 S3/S4 Era Robustness',
        'phase1_start_years': phase1,
        'phase2_2yr_windows': phase2,
        'phase3_rolling_3yr': phase3,
        'runtime_sec': round(elapsed, 1),
    }
    with open(OUTPUT_DIR / "r147_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_DIR / 'r147_results.json'}")


if __name__ == '__main__':
    main()
