#!/usr/bin/env python3
"""
R176c — SESS_BO D1 EMA20 Filter Review
========================================
Same methodology as R176b (TSMOM): compare D1 ON vs OFF for SESS_BO
at its current live parameters and lot size.

Config A: SESS_BO + D1 ON  (current live)
Config B: SESS_BO + D1 OFF
"""
import sys, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

PV = 100
SPREAD = 0.30
LOT = 0.13

SESS_BO_PARAMS = {
    'session_hour': 12,
    'lookback': 4,
    'sl_atr': 4.5,
    'tp_atr': 4.0,
    'trail_act': 0.14,
    'trail_dist': 0.025,
    'max_hold': 20,
    'cap': 60,
}

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-06-01"),
]


def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


def compute_d1_dir(h1_df):
    d1 = h1_df.resample('1D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    d1['EMA20'] = d1['Close'].ewm(span=20, adjust=False).mean()
    result = {}
    for idx, row in d1.iterrows():
        d = idx.date() if hasattr(idx, 'date') else idx
        if row['Close'] > row['EMA20']:
            result[d] = 1
        elif row['Close'] < row['EMA20']:
            result[d] = -1
        else:
            result[d] = 0
    return result


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


def _run_exit(pos, i, h, lo_v, c, spread, lot, pv, times,
              sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold, maxloss_cap):
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
        return _mk(pos, c, times[i], "Cap", i, -maxloss_cap)
    ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
    if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
        ts_p = h - td
        if lo_v <= ts_p:
            return _mk(pos, c, times[i], "Trail", i, (ts_p - pos['entry'] - spread) * lot * pv)
    elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
        ts_p = lo_v + td
        if h >= ts_p:
            return _mk(pos, c, times[i], "Trail", i, (pos['entry'] - ts_p - spread) * lot * pv)
    if i - pos['bar'] >= max_hold:
        return _mk(pos, c, times[i], "Timeout", i, pnl_c)
    return None


def bt_sess_bo(h1_df, d1_filter=False):
    p = SESS_BO_PARAMS
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour; times = df.index; n = len(df)
    dates = df.index.date

    d1_dir = compute_d1_dir(df) if d1_filter else None

    trades = []; pos = None; last_exit = -999
    for i in range(p['lookback'], n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], SPREAD, LOT, PV, times,
                               p['sl_atr'], p['tp_atr'], p['trail_act'], p['trail_dist'],
                               p['max_hold'], p['cap'])
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if hours[i] != p['session_hour']: continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i-j] for j in range(1, p['lookback']+1))
        ll = min(lo[i-j] for j in range(1, p['lookback']+1))

        direction = None
        if c[i] > hh:
            direction = 'BUY'
        elif c[i] < ll:
            direction = 'SELL'
        if direction is None:
            continue

        if d1_dir is not None:
            d = dates[i]
            sorted_dates = sorted([dd for dd in d1_dir.keys() if dd < d])
            if sorted_dates:
                d1_val = d1_dir[sorted_dates[-1]]
                if direction == 'BUY' and d1_val == -1:
                    continue
                elif direction == 'SELL' and d1_val == 1:
                    continue

        pos = {'dir': direction,
               'entry': c[i] + (SPREAD/2 if direction == 'BUY' else -SPREAD/2),
               'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def _trades_to_daily(trades):
    if not trades:
        return np.array([0.0])
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    return np.array([daily[k] for k in sorted(daily.keys())]) if daily else np.array([0.0])


def _sharpe(arr):
    if len(arr) < 10 or np.std(arr, ddof=1) == 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))


def _max_dd(arr):
    if len(arr) == 0: return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def _stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0,
                'n_buy': 0, 'n_sell': 0, 'buy_pnl': 0, 'sell_pnl': 0,
                'buy_wr': 0, 'sell_wr': 0, 'exits': {}}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    buys = [t for t in trades if t['dir'] == 'BUY']
    sells = [t for t in trades if t['dir'] == 'SELL']
    exits = {}
    for t in trades:
        exits[t['reason']] = exits.get(t['reason'], 0) + 1
    return {
        'n': n, 'sharpe': round(_sharpe(daily), 2),
        'pnl': round(sum(pnls), 0), 'wr': round(sum(1 for p in pnls if p > 0)/n*100, 1),
        'max_dd': round(_max_dd(daily), 0),
        'n_buy': len(buys), 'n_sell': len(sells),
        'buy_pnl': round(sum(t['pnl'] for t in buys), 0),
        'sell_pnl': round(sum(t['pnl'] for t in sells), 0),
        'buy_wr': round(sum(1 for t in buys if t['pnl']>0)/len(buys)*100, 1) if buys else 0,
        'sell_wr': round(sum(1 for t in sells if t['pnl']>0)/len(sells)*100, 1) if sells else 0,
        'exits': exits,
    }


def load_h1():
    import glob as _glob
    candidates = sorted(_glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
    csv_path = candidates[-1]
    print(f"  Loading: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    return df


def main():
    t0 = time.time()
    print("=" * 100, flush=True)
    print("  R176c — SESS_BO D1 EMA20 Filter Review", flush=True)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  Lot={LOT}, Params: SL={SESS_BO_PARAMS['sl_atr']}, TP={SESS_BO_PARAMS['tp_atr']}, "
          f"Trail={SESS_BO_PARAMS['trail_act']}/{SESS_BO_PARAMS['trail_dist']}, "
          f"MH={SESS_BO_PARAMS['max_hold']}, Cap=${SESS_BO_PARAMS['cap']}", flush=True)
    print("=" * 100, flush=True)

    h1_df = load_h1()
    print(f"  {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})\n", flush=True)

    configs = {
        'A_D1_ON':  True,
        'B_D1_OFF': False,
    }

    # Phase 1: Full sample
    print("Phase 1: Full Sample Comparison", flush=True)
    print("-" * 100, flush=True)
    print(f"\n  {'Config':<14} {'#':>5} {'Shrp':>6} {'PnL':>9} {'WR':>6} {'MaxDD':>7} "
          f"{'BUY#':>5} {'SELL#':>5} {'BUY$':>8} {'SELL$':>8} {'BuyWR':>6} {'SellWR':>7} Exits")

    all_stats = {}
    for name, d1_on in configs.items():
        trades = bt_sess_bo(h1_df, d1_filter=d1_on)
        s = _stats(trades)
        all_stats[name] = s
        exits_str = ', '.join(f"{k}:{v}" for k, v in sorted(s['exits'].items()))
        print(f"  {name:<14} {s['n']:>5} {s['sharpe']:>6.2f} ${s['pnl']:>8,} {s['wr']:>5.1f}% ${s['max_dd']:>6,} "
              f"{s['n_buy']:>5} {s['n_sell']:>5} ${s['buy_pnl']:>7,} ${s['sell_pnl']:>7,} "
              f"{s['buy_wr']:>5.1f}% {s['sell_wr']:>6.1f}% {exits_str}")

    a = all_stats['A_D1_ON']; b = all_stats['B_D1_OFF']
    delta_sharpe = b['sharpe'] - a['sharpe']
    delta_pct = (delta_sharpe / a['sharpe'] * 100) if a['sharpe'] != 0 else 0
    filtered_trades = b['n'] - a['n']
    print(f"\n  D1 impact: Sharpe {a['sharpe']:.2f} → {b['sharpe']:.2f} ({delta_sharpe:+.2f}, {delta_pct:+.1f}%)")
    print(f"  D1 filtered {filtered_trades} trades ({filtered_trades}/{b['n']}={filtered_trades/b['n']*100:.1f}%)")

    # Phase 2: Period breakdown
    periods = [
        ("2015-2017", "2015-01-01", "2017-01-01"),
        ("2017-2019", "2017-01-01", "2019-01-01"),
        ("2019-2021", "2019-01-01", "2021-01-01"),
        ("2021-2023", "2021-01-01", "2023-01-01"),
        ("2023-2025", "2023-01-01", "2025-01-01"),
        ("2025-2026", "2025-01-01", "2026-06-01"),
    ]

    print(f"\n\nPhase 2: Period Breakdown", flush=True)
    print("-" * 100, flush=True)
    print(f"\n  {'Period':<12} {'D1_ON Shrp':>10} {'D1_ON #':>8} {'D1_ON PnL':>10} "
          f"{'D1_OFF Shrp':>11} {'D1_OFF #':>9} {'D1_OFF PnL':>11} {'Delta':>8}")

    period_sharpes = {'A_D1_ON': [], 'B_D1_OFF': []}
    for pname, start, end in periods:
        sub = h1_df[start:end]
        if len(sub) < 500:
            print(f"  {pname:<12} {'N/A':>10}")
            period_sharpes['A_D1_ON'].append(0)
            period_sharpes['B_D1_OFF'].append(0)
            continue

        trades_on = bt_sess_bo(sub, d1_filter=True)
        s_on = _stats(trades_on)
        trades_off = bt_sess_bo(sub, d1_filter=False)
        s_off = _stats(trades_off)
        delta = s_off['sharpe'] - s_on['sharpe']
        period_sharpes['A_D1_ON'].append(s_on['sharpe'])
        period_sharpes['B_D1_OFF'].append(s_off['sharpe'])

        print(f"  {pname:<12} {s_on['sharpe']:>10.2f} {s_on['n']:>8} ${s_on['pnl']:>9,} "
              f"{s_off['sharpe']:>11.2f} {s_off['n']:>9} ${s_off['pnl']:>10,} {delta:>+8.2f}")

    print(f"\n  Period Sharpe consistency:")
    for name in ['A_D1_ON', 'B_D1_OFF']:
        vals = period_sharpes[name]
        pos = sum(1 for v in vals if v > 0)
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        min_v = min(vals)
        print(f"    {name:<14}: {pos}/6 positive, mean={mean_v:.2f}, std={std_v:.2f}, min={min_v:.2f}")

    # D1 ON wins / OFF wins count
    on_wins = sum(1 for i in range(len(periods)) if period_sharpes['A_D1_ON'][i] > period_sharpes['B_D1_OFF'][i])
    off_wins = len(periods) - on_wins
    print(f"    D1_ON wins {on_wins}/{len(periods)} periods, D1_OFF wins {off_wins}/{len(periods)} periods")

    # Phase 3: K-Fold
    print(f"\n\nPhase 3: K-Fold Validation", flush=True)
    print("-" * 100, flush=True)

    for name, d1_on in configs.items():
        fold_sharpes = []
        for fname, start, end in FOLDS:
            sub = h1_df[start:end]
            if len(sub) < 500:
                fold_sharpes.append(0)
                continue
            trades = bt_sess_bo(sub, d1_filter=d1_on)
            fold_sharpes.append(_sharpe(_trades_to_daily(trades)))

        pos = sum(1 for s in fold_sharpes if s > 0)
        mean_s = np.mean(fold_sharpes)
        std_s = np.std(fold_sharpes)
        min_s = min(fold_sharpes)
        status = "PASS" if pos >= 4 else "FAIL"
        print(f"  {name:<14}: {pos}/6 pos, mean={mean_s:.2f}, std={std_s:.2f}, min={min_s:.2f} "
              f"[{status}]  {[round(s,1) for s in fold_sharpes]}")

    # Phase 4: BUY vs SELL direction analysis
    print(f"\n\nPhase 4: Direction Analysis (D1 ON vs OFF)", flush=True)
    print("-" * 100, flush=True)

    for pname, start, end in periods:
        sub = h1_df[start:end]
        if len(sub) < 500: continue

        trades_on = bt_sess_bo(sub, d1_filter=True)
        trades_off = bt_sess_bo(sub, d1_filter=False)
        s_on = _stats(trades_on)
        s_off = _stats(trades_off)

        filtered_buys = s_off['n_buy'] - s_on['n_buy']
        filtered_sells = s_off['n_sell'] - s_on['n_sell']
        print(f"  {pname}: D1 filtered {filtered_buys} BUYs (WR OFF:{s_off['buy_wr']:.0f}% vs ON:{s_on['buy_wr']:.0f}%), "
              f"{filtered_sells} SELLs (WR OFF:{s_off['sell_wr']:.0f}% vs ON:{s_on['sell_wr']:.0f}%)")

    # Summary
    print(f"\n\n{'='*100}")
    print(f"  SUMMARY")
    print(f"{'='*100}\n")
    a = all_stats['A_D1_ON']; b = all_stats['B_D1_OFF']
    print(f"  D1 ON  (current live): Sharpe={a['sharpe']:.2f}, PnL=${a['pnl']:,}, #{a['n']}, MaxDD=${a['max_dd']:,}")
    print(f"  D1 OFF (proposed):     Sharpe={b['sharpe']:.2f}, PnL=${b['pnl']:,}, #{b['n']}, MaxDD=${b['max_dd']:,}")
    print(f"  Delta: Sharpe {delta_sharpe:+.2f} ({delta_pct:+.1f}%), PnL ${b['pnl']-a['pnl']:+,}")

    if delta_sharpe > 0:
        print(f"\n  >>> D1 OFF is better for SESS_BO (same conclusion as TSMOM)")
    elif delta_sharpe < -0.3:
        print(f"\n  >>> D1 ON is clearly better for SESS_BO — KEEP IT")
    else:
        print(f"\n  >>> D1 filter has minimal impact — default to keeping for safety")

    elapsed = time.time() - t0
    print(f"\n  R176c complete in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
