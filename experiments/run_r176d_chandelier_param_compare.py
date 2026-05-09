#!/usr/bin/env python3
"""
R176d — Chandelier Parameter Comparison: Live (R146) vs Paper (R45/R46)
========================================================================
Direct comparison at same lot size (0.08).

Config A (Live R146):   Period=22, SL=4.5, TP=8.0, Trail=0.14/0.025, EMA100=ON,  Cap=$25,  MH=20
Config B (Paper Old):   Period=10, SL=3.0, TP=8.0, Trail=0.28/0.06,  EMA100=OFF, Cap=$80*, MH=20
Config C (Live NoEMA):  Period=22, SL=4.5, TP=8.0, Trail=0.14/0.025, EMA100=OFF, Cap=$25,  MH=20
Config D (Paper+EMA):   Period=10, SL=3.0, TP=8.0, Trail=0.28/0.06,  EMA100=ON,  Cap=$80*, MH=20
Config E (Hybrid):      Period=22, SL=3.0, TP=8.0, Trail=0.28/0.06,  EMA100=ON,  Cap=$25,  MH=20

* Paper Cap=$80 at 0.01 lot → scaled to $640 at 0.08 lot (same $/oz tolerance).
  But for fair comparison, we test paper params with no Cap (as $640 is effectively none).
"""
import sys, os, time
import numpy as np
import pandas as pd
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

PV = 100
SPREAD = 0.30
LOT = 0.08

CONFIGS = {
    'A_Live_R146':    {'period': 22, 'sl': 4.5, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'ema': True,  'cap': 25,  'mh': 20},
    'B_Paper_Old':    {'period': 10, 'sl': 3.0, 'tp': 8.0, 'trail_act': 0.28, 'trail_dist': 0.06,  'ema': False, 'cap': 0,   'mh': 20},
    'C_Live_NoEMA':   {'period': 22, 'sl': 4.5, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'ema': False, 'cap': 25,  'mh': 20},
    'D_Paper_EMA':    {'period': 10, 'sl': 3.0, 'tp': 8.0, 'trail_act': 0.28, 'trail_dist': 0.06,  'ema': True,  'cap': 0,   'mh': 20},
    'E_Hybrid':       {'period': 22, 'sl': 3.0, 'tp': 8.0, 'trail_act': 0.28, 'trail_dist': 0.06,  'ema': True,  'cap': 25,  'mh': 20},
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


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


def _run_exit(pos, i, h, lo_v, c, spread, lot, pv, times, cfg):
    sl_atr = cfg['sl']; tp_atr = cfg['tp']
    trail_act_atr = cfg['trail_act']; trail_dist_atr = cfg['trail_dist']
    max_hold = cfg['mh']; maxloss_cap = cfg['cap']

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


def bt_chandelier(h1_df, cfg):
    period = cfg['period']
    mult = 3.0
    use_ema = cfg['ema']

    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    if use_ema:
        df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df = df.dropna(subset=['ATR'])

    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values
    ema = df['EMA100'].values if use_ema else None
    times = df.index; n = len(df)

    chand_long = np.full(n, np.nan)
    chand_short = np.full(n, np.nan)
    for i in range(period, n):
        hh = np.max(h[i-period+1:i+1])
        ll = np.min(lo[i-period+1:i+1])
        chand_long[i] = hh - mult * atr[i]
        chand_short[i] = ll + mult * atr[i]

    trades = []; pos = None; last_exit = -999
    min_start = max(period + 2, 105 if use_ema else period + 2)
    for i in range(min_start, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], SPREAD, LOT, PV, times, cfg)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(chand_long[i]) or np.isnan(chand_long[i-1]): continue
        if np.isnan(chand_short[i]) or np.isnan(chand_short[i-1]): continue

        flip_bull = c[i] > chand_long[i] and c[i-1] <= chand_long[i-1]
        flip_bear = c[i] < chand_short[i] and c[i-1] >= chand_short[i-1]

        if use_ema:
            if ema is None or np.isnan(ema[i]):
                continue
            if flip_bull and c[i] > ema[i]:
                pos = {'dir': 'BUY', 'entry': c[i]+SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
            elif flip_bear and c[i] < ema[i]:
                pos = {'dir': 'SELL', 'entry': c[i]-SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        else:
            if flip_bull:
                pos = {'dir': 'BUY', 'entry': c[i]+SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
            elif flip_bear:
                pos = {'dir': 'SELL', 'entry': c[i]-SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
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
                'buy_wr': 0, 'sell_wr': 0, 'exits': {}, 'avg_bars': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    buys = [t for t in trades if t['dir'] == 'BUY']
    sells = [t for t in trades if t['dir'] == 'SELL']
    exits = {}
    for t in trades:
        exits[t['reason']] = exits.get(t['reason'], 0) + 1
    avg_bars = np.mean([t['bars'] for t in trades])
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
        'avg_bars': round(avg_bars, 1),
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
    print("=" * 115, flush=True)
    print("  R176d — Chandelier Parameter Comparison: Live (R146) vs Paper (R45/R46)", flush=True)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Lot={LOT}", flush=True)
    print("=" * 115, flush=True)

    h1_df = load_h1()
    print(f"  {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})\n", flush=True)

    print("  Configs:", flush=True)
    for name, cfg in CONFIGS.items():
        print(f"    {name}: Period={cfg['period']}, SL={cfg['sl']}, TP={cfg['tp']}, "
              f"Trail={cfg['trail_act']}/{cfg['trail_dist']}, EMA100={'ON' if cfg['ema'] else 'OFF'}, "
              f"Cap={'$'+str(cfg['cap']) if cfg['cap'] else 'None'}, MH={cfg['mh']}")

    # Phase 1: Full sample
    print(f"\n\nPhase 1: Full Sample", flush=True)
    print("-" * 115, flush=True)
    print(f"  {'Config':<16} {'#':>5} {'Shrp':>6} {'PnL':>9} {'WR':>6} {'MaxDD':>7} {'AvgH':>5} "
          f"{'BUY#':>5} {'SELL#':>5} {'BUY$':>8} {'SELL$':>8} Exits")

    all_stats = {}
    for name, cfg in CONFIGS.items():
        trades = bt_chandelier(h1_df, cfg)
        s = _stats(trades)
        all_stats[name] = s
        exits_str = ', '.join(f"{k}:{v}" for k, v in sorted(s['exits'].items()))
        print(f"  {name:<16} {s['n']:>5} {s['sharpe']:>6.2f} ${s['pnl']:>8,} {s['wr']:>5.1f}% ${s['max_dd']:>6,} {s['avg_bars']:>5.1f} "
              f"{s['n_buy']:>5} {s['n_sell']:>5} ${s['buy_pnl']:>7,} ${s['sell_pnl']:>7,} {exits_str}")

    # Phase 2: Period breakdown
    periods = [
        ("2015-2017", "2015-01-01", "2017-01-01"),
        ("2017-2019", "2017-01-01", "2019-01-01"),
        ("2019-2021", "2019-01-01", "2021-01-01"),
        ("2021-2023", "2021-01-01", "2023-01-01"),
        ("2023-2025", "2023-01-01", "2025-01-01"),
        ("2025-2026", "2025-01-01", "2026-06-01"),
    ]

    print(f"\n\nPhase 2: Period Breakdown (Sharpe / #trades)", flush=True)
    print("-" * 115, flush=True)
    header = f"  {'Period':<12}"
    for name in CONFIGS:
        header += f" {name[:14]:>16}"
    print(header)
    print(f"  {'-'*12}" + f" {'-'*16}" * len(CONFIGS))

    period_stats = {name: [] for name in CONFIGS}
    for pname, start, end in periods:
        sub = h1_df[start:end]
        if len(sub) < 500:
            line = f"  {pname:<12}"
            for name in CONFIGS:
                line += f" {'N/A':>16}"
                period_stats[name].append(0)
            print(line)
            continue
        line = f"  {pname:<12}"
        for name, cfg in CONFIGS.items():
            trades = bt_chandelier(sub, cfg)
            s = _stats(trades)
            line += f" {s['sharpe']:>6.2f} ({s['n']:>4})"
            period_stats[name].append(s['sharpe'])
        print(line)

    print(f"\n  Consistency:")
    for name in CONFIGS:
        vals = period_stats[name]
        pos = sum(1 for v in vals if v > 0)
        mean_v = np.mean(vals); std_v = np.std(vals); min_v = min(vals)
        print(f"    {name:<16}: {pos}/6 positive, mean={mean_v:.2f}, std={std_v:.2f}, min={min_v:.2f}")

    # Phase 3: K-Fold
    print(f"\n\nPhase 3: K-Fold Validation", flush=True)
    print("-" * 115, flush=True)

    for name, cfg in CONFIGS.items():
        fold_sharpes = []
        for fname, start, end in FOLDS:
            sub = h1_df[start:end]
            if len(sub) < 500:
                fold_sharpes.append(0); continue
            trades = bt_chandelier(sub, cfg)
            fold_sharpes.append(_sharpe(_trades_to_daily(trades)))
        pos = sum(1 for s in fold_sharpes if s > 0)
        mean_s = np.mean(fold_sharpes); std_s = np.std(fold_sharpes); min_s = min(fold_sharpes)
        status = "PASS" if pos >= 4 else "FAIL"
        print(f"  {name:<16}: {pos}/6 pos, mean={mean_s:.2f}, std={std_s:.2f}, min={min_s:.2f} "
              f"[{status}]  {[round(s,1) for s in fold_sharpes]}")

    # Summary
    print(f"\n\n{'='*115}")
    print(f"  SUMMARY")
    print(f"{'='*115}\n")

    best = max(all_stats.items(), key=lambda x: x[1]['sharpe'])
    print(f"  Best full-sample Sharpe: {best[0]} ({best[1]['sharpe']:.2f})\n")

    a = all_stats['A_Live_R146']
    b = all_stats['B_Paper_Old']
    c_s = all_stats['C_Live_NoEMA']
    print(f"  A (Live R146) vs B (Paper Old): Sharpe {a['sharpe']:.2f} vs {b['sharpe']:.2f}")
    print(f"  A (Live R146) vs C (No EMA):    Sharpe {a['sharpe']:.2f} vs {c_s['sharpe']:.2f}")
    print(f"  EMA100 impact on Live params:    {c_s['sharpe']:.2f} → {a['sharpe']:.2f} ({a['sharpe']-c_s['sharpe']:+.2f})")

    if 'D_Paper_EMA' in all_stats:
        d = all_stats['D_Paper_EMA']
        print(f"  D (Paper + EMA): Sharpe {d['sharpe']:.2f}")
    if 'E_Hybrid' in all_stats:
        e = all_stats['E_Hybrid']
        print(f"  E (Hybrid):      Sharpe {e['sharpe']:.2f}")

    elapsed = time.time() - t0
    print(f"\n  R176d complete in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
