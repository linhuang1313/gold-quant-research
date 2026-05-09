#!/usr/bin/env python3
"""
R176b — TSMOM Parameter Comparison: Live (R166b) vs Paper (Old)
================================================================
Direct comparison of two TSMOM parameter sets on the same data.

Config A (Live R166b):  slow=720,  SL=6.0, TP=8.0,  Trail=0.14/0.025, MH=12, D1=ON,  Cap=$60
Config B (Paper Old):   slow=1440, SL=3.5, TP=12.0, Trail=0.28/0.06,  MH=50, D1=OFF, Cap=None
Config C (Live NoDI):   slow=720,  SL=6.0, TP=8.0,  Trail=0.14/0.025, MH=12, D1=OFF, Cap=$60
Config D (Paper+D1):    slow=1440, SL=3.5, TP=12.0, Trail=0.28/0.06,  MH=50, D1=ON,  Cap=None

Plus K-Fold validation for top configs.
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
LOT = 0.15

CONFIGS = {
    'A_Live_R166b':   {'slow': 720,  'sl': 6.0, 'tp': 8.0,  'trail_act': 0.14, 'trail_dist': 0.025, 'mh': 12, 'd1': True,  'cap': 60},
    'B_Paper_Old':    {'slow': 1440, 'sl': 3.5, 'tp': 12.0, 'trail_act': 0.28, 'trail_dist': 0.06,  'mh': 50, 'd1': False, 'cap': 0},
    'C_Live_NoD1':    {'slow': 720,  'sl': 6.0, 'tp': 8.0,  'trail_act': 0.14, 'trail_dist': 0.025, 'mh': 12, 'd1': False, 'cap': 60},
    'D_Paper_D1':     {'slow': 1440, 'sl': 3.5, 'tp': 12.0, 'trail_act': 0.28, 'trail_dist': 0.06,  'mh': 50, 'd1': True,  'cap': 0},
    'E_Hybrid_Best':  {'slow': 1440, 'sl': 6.0, 'tp': 8.0,  'trail_act': 0.14, 'trail_dist': 0.025, 'mh': 12, 'd1': False, 'cap': 60},
}

FAST = 480

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


def bt_tsmom(h1_df, cfg):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])

    slow = cfg['slow']
    d1_dir = compute_d1_dir(df) if cfg['d1'] else None

    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    dates = df.index.date

    max_lb = max(FAST, slow)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        if c[i-FAST] > 0: s += 0.5 * np.sign(c[i]/c[i-FAST] - 1.0)
        if c[i-slow] > 0: s += 0.5 * np.sign(c[i]/c[i-slow] - 1.0)
        score[i] = s

    trades = []; pos = None; last_exit = -999
    for i in range(max_lb+1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], SPREAD, LOT, PV, times, cfg)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and score[i] < 0:
                pnl = (c[i]-pos['entry']-SPREAD)*LOT*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and score[i] > 0:
                pnl = (pos['entry']-c[i]-SPREAD)*LOT*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(score[i]) or np.isnan(score[i-1]): continue

        direction = None
        if score[i] > 0 and score[i-1] <= 0:
            direction = 'BUY'
        elif score[i] < 0 and score[i-1] >= 0:
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

        pos = {'dir': direction, 'entry': c[i] + (SPREAD/2 if direction == 'BUY' else -SPREAD/2),
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
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0, 'n_buy': 0, 'n_sell': 0,
                'buy_pnl': 0, 'sell_pnl': 0, 'buy_wr': 0, 'sell_wr': 0, 'exits': {}}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    buys = [t for t in trades if t['dir'] == 'BUY']
    sells = [t for t in trades if t['dir'] == 'SELL']
    exits = {}
    for t in trades:
        r = t['reason']
        exits[r] = exits.get(r, 0) + 1
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
    print("=" * 110, flush=True)
    print("  R176b — TSMOM Parameter Comparison: Live vs Paper vs Hybrid", flush=True)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 110, flush=True)

    h1_df = load_h1()
    print(f"  {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})\n", flush=True)

    # Phase 1: Full sample comparison
    print("Phase 1: Full Sample Comparison", flush=True)
    print("=" * 110, flush=True)

    print(f"\n  Configs:")
    for name, cfg in CONFIGS.items():
        print(f"    {name}: slow={cfg['slow']}, SL={cfg['sl']}, TP={cfg['tp']}, "
              f"Trail={cfg['trail_act']}/{cfg['trail_dist']}, MH={cfg['mh']}, "
              f"D1={'ON' if cfg['d1'] else 'OFF'}, Cap={'$'+str(cfg['cap']) if cfg['cap'] else 'None'}")

    print(f"\n  {'Config':<18} {'#':>5} {'Shrp':>6} {'PnL':>9} {'WR':>6} {'MaxDD':>7} "
          f"{'BUY#':>5} {'SELL#':>5} {'BUY$':>8} {'SELL$':>8} {'Exits'}")
    print(f"  {'-'*18} {'-'*5} {'-'*6} {'-'*9} {'-'*6} {'-'*7} "
          f"{'-'*5} {'-'*5} {'-'*8} {'-'*8} {'-'*30}")

    all_stats = {}
    for name, cfg in CONFIGS.items():
        trades = bt_tsmom(h1_df, cfg)
        s = _stats(trades)
        all_stats[name] = s
        exits_str = ', '.join(f"{k}:{v}" for k, v in sorted(s['exits'].items()))
        print(f"  {name:<18} {s['n']:>5} {s['sharpe']:>6.2f} ${s['pnl']:>8,} {s['wr']:>5.1f}% ${s['max_dd']:>6,} "
              f"{s['n_buy']:>5} {s['n_sell']:>5} ${s['buy_pnl']:>7,} ${s['sell_pnl']:>7,} {exits_str}")

    # Phase 2: Period breakdown for top configs
    print(f"\n\nPhase 2: Period Breakdown (Sharpe)", flush=True)
    print("=" * 110, flush=True)

    periods = [
        ("2015-2017", "2015-01-01", "2017-01-01"),
        ("2017-2019", "2017-01-01", "2019-01-01"),
        ("2019-2021", "2019-01-01", "2021-01-01"),
        ("2021-2023", "2021-01-01", "2023-01-01"),
        ("2023-2025", "2023-01-01", "2025-01-01"),
        ("2025-2026", "2025-01-01", "2026-06-01"),
    ]

    header = f"  {'Period':<12}"
    for name in CONFIGS:
        header += f" {name[:14]:>14}"
    print(header)
    print(f"  {'-'*12}" + f" {'-'*14}" * len(CONFIGS))

    period_stats = {name: [] for name in CONFIGS}
    for pname, start, end in periods:
        sub = h1_df[start:end]
        if len(sub) < 800:
            line = f"  {pname:<12}"
            for name in CONFIGS:
                line += f" {'N/A':>14}"
                period_stats[name].append(0)
            print(line)
            continue
        line = f"  {pname:<12}"
        for name, cfg in CONFIGS.items():
            trades = bt_tsmom(sub, cfg)
            s = _stats(trades)
            line += f" {s['sharpe']:>14.2f}"
            period_stats[name].append(s['sharpe'])
        print(line)

    # Consistency
    print(f"\n  Positive folds:")
    for name in CONFIGS:
        pos = sum(1 for s in period_stats[name] if s > 0)
        mean_s = np.mean(period_stats[name])
        std_s = np.std(period_stats[name])
        min_s = min(period_stats[name])
        print(f"    {name:<18}: {pos}/6 positive, mean={mean_s:.2f}, std={std_s:.2f}, min={min_s:.2f}")

    # Phase 3: K-Fold
    print(f"\n\nPhase 3: K-Fold Validation", flush=True)
    print("=" * 110, flush=True)

    for name, cfg in CONFIGS.items():
        fold_sharpes = []
        for fname, start, end in FOLDS:
            sub = h1_df[start:end]
            if len(sub) < 500:
                fold_sharpes.append(0)
                continue
            trades = bt_tsmom(sub, cfg)
            fold_sharpes.append(_sharpe(_trades_to_daily(trades)))

        pos = sum(1 for s in fold_sharpes if s > 0)
        mean_s = np.mean(fold_sharpes)
        std_s = np.std(fold_sharpes)
        min_s = min(fold_sharpes)
        status = "PASS" if pos >= 4 else "FAIL"
        print(f"  {name:<18}: {pos}/6 pos, mean={mean_s:.2f}, std={std_s:.2f}, min={min_s:.2f} "
              f"[{status}]  {[round(s,1) for s in fold_sharpes]}")

    # Summary
    print(f"\n\n{'='*110}")
    print(f"  SUMMARY")
    print(f"{'='*110}\n")
    best = max(all_stats.items(), key=lambda x: x[1]['sharpe'])
    print(f"  Best full-sample Sharpe: {best[0]} ({best[1]['sharpe']:.2f})")
    print(f"\n  Key findings:")
    a = all_stats['A_Live_R166b']
    b = all_stats['B_Paper_Old']
    c_s = all_stats['C_Live_NoD1']
    print(f"    A (Live current) vs B (Paper old): Sharpe {a['sharpe']:.2f} vs {b['sharpe']:.2f}")
    print(f"    A (Live current) vs C (Live no D1): Sharpe {a['sharpe']:.2f} vs {c_s['sharpe']:.2f}")
    print(f"    D1 filter impact on Live params: {a['sharpe']:.2f} -> {c_s['sharpe']:.2f} ({c_s['sharpe']-a['sharpe']:+.2f})")
    if 'E_Hybrid_Best' in all_stats:
        e = all_stats['E_Hybrid_Best']
        print(f"    E (Hybrid: paper slow + live exit): Sharpe {e['sharpe']:.2f}")

    elapsed = time.time() - t0
    print(f"\n  R176b complete in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
