#!/usr/bin/env python3
"""
R176 — TSMOM D1 EMA20 Filter Review
=====================================
R123 validated D1 EMA20 filter on 11y data (K-Fold 5/5, Sharpe +22%).
But recent paper TSMOM (no filter) is 15/15 wins, all SELL.
Live TSMOM (with filter) has 0 trades — all SELL signals blocked.

Q: Is the D1 filter hurting recent performance?

Test:
  Phase 1: Full sample (2015-2026) — confirm R123 conclusion
  Phase 2: Recent period (2025-2026) — check if filter still helps
  Phase 3: Rolling window — detect if filter effectiveness decayed
  Phase 4: Breakdown by filtered direction — how much alpha is lost
"""
import sys, os, time, json
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

# Current live params (R166b)
FAST = 480
SLOW = 720
SL_ATR = 6.0
TP_ATR = 8.0
TRAIL_ACT = 0.14
TRAIL_DIST = 0.025
MAX_HOLD = 12
COOLDOWN = 2


def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


def compute_d1_ema20(h1_df):
    """Resample H1 to D1 and compute EMA20 direction. Returns dict date->int."""
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
              sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold,
              maxloss_cap=0):
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
    if i - pos['bar'] >= max_hold:
        return _mk(pos, c, times[i], "Timeout", i, pnl_c)
    return None


def bt_tsmom(h1_df, d1_filter=False, maxloss_cap=60):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])

    # D1 direction
    d1_dir = None
    if d1_filter:
        d1_dir = compute_d1_ema20(df)

    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    dates = df.index.date

    max_lb = max(FAST, SLOW)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        if c[i-FAST] > 0: s += 0.5 * np.sign(c[i]/c[i-FAST] - 1.0)
        if c[i-SLOW] > 0: s += 0.5 * np.sign(c[i]/c[i-SLOW] - 1.0)
        score[i] = s

    trades = []; filtered_trades = []
    pos = None; last_exit = -999
    for i in range(max_lb+1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], SPREAD, LOT, PV, times,
                               SL_ATR, TP_ATR, TRAIL_ACT, TRAIL_DIST, MAX_HOLD, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and score[i] < 0:
                pnl = (c[i]-pos['entry']-SPREAD)*LOT*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and score[i] > 0:
                pnl = (pos['entry']-c[i]-SPREAD)*LOT*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            continue
        if i-last_exit < COOLDOWN: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(score[i]) or np.isnan(score[i-1]): continue

        direction = None
        if score[i] > 0 and score[i-1] <= 0:
            direction = 'BUY'
        elif score[i] < 0 and score[i-1] >= 0:
            direction = 'SELL'

        if direction is None:
            continue

        # D1 filter check
        if d1_filter and d1_dir is not None:
            d = dates[i]
            sorted_dates = sorted([dd for dd in d1_dir.keys() if dd < d])
            if sorted_dates:
                prev_d = sorted_dates[-1]
                d1_val = d1_dir[prev_d]
                if direction == 'BUY' and d1_val == -1:
                    filtered_trades.append({'dir': direction, 'time': times[i], 'atr': atr[i], 'price': c[i], 'filter_reason': 'D1偏空拒多'})
                    continue
                elif direction == 'SELL' and d1_val == 1:
                    filtered_trades.append({'dir': direction, 'time': times[i], 'atr': atr[i], 'price': c[i], 'filter_reason': 'D1偏多拒空'})
                    continue

        pos = {'dir': direction, 'entry': c[i] + (SPREAD/2 if direction == 'BUY' else -SPREAD/2),
               'bar': i, 'time': times[i], 'atr': atr[i]}

    return trades, filtered_trades


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
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    buys = [t for t in trades if t['dir'] == 'BUY']
    sells = [t for t in trades if t['dir'] == 'SELL']
    return {
        'n': n,
        'sharpe': round(_sharpe(daily), 2),
        'pnl': round(sum(pnls), 0),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'max_dd': round(_max_dd(daily), 0),
        'n_buy': len(buys),
        'n_sell': len(sells),
        'buy_pnl': round(sum(t['pnl'] for t in buys), 0),
        'sell_pnl': round(sum(t['pnl'] for t in sells), 0),
        'buy_wr': round(sum(1 for t in buys if t['pnl'] > 0) / len(buys) * 100, 1) if buys else 0,
        'sell_wr': round(sum(1 for t in sells if t['pnl'] > 0) / len(sells) * 100, 1) if sells else 0,
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
    print("  R176 — TSMOM D1 EMA20 Filter Review", flush=True)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 100, flush=True)

    h1_df = load_h1()
    print(f"  {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})\n", flush=True)

    # Phase 1: Full sample
    print("Phase 1: Full Sample (2015-2026)", flush=True)
    print("-" * 80, flush=True)
    trades_no, _ = bt_tsmom(h1_df, d1_filter=False)
    trades_yes, filtered = bt_tsmom(h1_df, d1_filter=True)

    s_no = _stats(trades_no)
    s_yes = _stats(trades_yes)

    print(f"  {'Metric':<15} {'No Filter':>12} {'D1 Filter':>12} {'Delta':>10}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'Trades':<15} {s_no['n']:>12} {s_yes['n']:>12} {s_yes['n']-s_no['n']:>+10}")
    print(f"  {'Sharpe':<15} {s_no['sharpe']:>12.2f} {s_yes['sharpe']:>12.2f} {s_yes['sharpe']-s_no['sharpe']:>+10.2f}")
    print(f"  {'PnL':<15} {'$'+str(s_no['pnl']):>12} {'$'+str(s_yes['pnl']):>12} {'$'+str(s_yes['pnl']-s_no['pnl']):>10}")
    print(f"  {'WinRate':<15} {s_no['wr']:>11.1f}% {s_yes['wr']:>11.1f}% {s_yes['wr']-s_no['wr']:>+9.1f}%")
    print(f"  {'MaxDD':<15} {'$'+str(s_no['max_dd']):>12} {'$'+str(s_yes['max_dd']):>12}")
    print(f"  {'BUY trades':<15} {s_no['n_buy']:>12} {s_yes['n_buy']:>12}")
    print(f"  {'SELL trades':<15} {s_no['n_sell']:>12} {s_yes['n_sell']:>12}")
    print(f"  {'BUY PnL':<15} {'$'+str(s_no['buy_pnl']):>12} {'$'+str(s_yes['buy_pnl']):>12}")
    print(f"  {'SELL PnL':<15} {'$'+str(s_no['sell_pnl']):>12} {'$'+str(s_yes['sell_pnl']):>12}")
    print(f"  {'Filtered sigs':<15} {'':>12} {len(filtered):>12}")
    print(f"\n  Full sample: D1 Filter {'HELPS' if s_yes['sharpe'] > s_no['sharpe'] else 'HURTS'} "
          f"(Sharpe {s_no['sharpe']:.2f} -> {s_yes['sharpe']:.2f})\n", flush=True)

    # Phase 2: Period breakdown
    print("Phase 2: Period Breakdown", flush=True)
    print("-" * 80, flush=True)

    periods = [
        ("2015-2017", "2015-01-01", "2017-01-01"),
        ("2017-2019", "2017-01-01", "2019-01-01"),
        ("2019-2021", "2019-01-01", "2021-01-01"),
        ("2021-2023", "2021-01-01", "2023-01-01"),
        ("2023-2025", "2023-01-01", "2025-01-01"),
        ("2025-2026", "2025-01-01", "2026-06-01"),
        ("Last 6mo", "2025-11-01", "2026-06-01"),
        ("Last 3mo", "2026-02-01", "2026-06-01"),
    ]

    print(f"  {'Period':<12} {'NoFilt#':>8} {'NoFiltSh':>9} {'Filt#':>7} {'FiltSh':>8} {'Delta':>8} {'Filtered':>9} {'Winner':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*9} {'-'*7} {'-'*8} {'-'*8} {'-'*9} {'-'*10}")

    for name, start, end in periods:
        sub = h1_df[start:end]
        if len(sub) < 800:
            print(f"  {name:<12} (insufficient data)")
            continue
        t_no, _ = bt_tsmom(sub, d1_filter=False)
        t_yes, filt = bt_tsmom(sub, d1_filter=True)
        sn = _stats(t_no)
        sy = _stats(t_yes)
        delta = sy['sharpe'] - sn['sharpe']
        winner = "Filter" if delta > 0 else "NoFilter" if delta < 0 else "Tie"
        print(f"  {name:<12} {sn['n']:>8} {sn['sharpe']:>9.2f} {sy['n']:>7} {sy['sharpe']:>8.2f} "
              f"{delta:>+8.2f} {len(filt):>9} {winner:>10}")

    # Phase 3: What did the filter block recently?
    print(f"\nPhase 3: Filtered Signals Analysis (Last 6 months)", flush=True)
    print("-" * 80, flush=True)

    sub_6m = h1_df["2025-11-01":"2026-06-01"]
    _, filtered_6m = bt_tsmom(sub_6m, d1_filter=True)
    trades_no_6m, _ = bt_tsmom(sub_6m, d1_filter=False)

    # Find trades that would have happened without filter
    buy_filtered = [f for f in filtered_6m if f['dir'] == 'BUY']
    sell_filtered = [f for f in filtered_6m if f['dir'] == 'SELL']

    print(f"  Filtered BUY signals (D1偏空拒多): {len(buy_filtered)}")
    print(f"  Filtered SELL signals (D1偏多拒空): {len(sell_filtered)}")

    # Show the no-filter trades and mark which ones would be filtered
    no_filter_buys = [t for t in trades_no_6m if t['dir'] == 'BUY']
    no_filter_sells = [t for t in trades_no_6m if t['dir'] == 'SELL']
    filter_buys = [t for t in bt_tsmom(sub_6m, d1_filter=True)[0] if t['dir'] == 'BUY']
    filter_sells = [t for t in bt_tsmom(sub_6m, d1_filter=True)[0] if t['dir'] == 'SELL']

    print(f"\n  Last 6mo without filter: {len(no_filter_buys)} BUY (PnL ${sum(t['pnl'] for t in no_filter_buys):+.0f}), "
          f"{len(no_filter_sells)} SELL (PnL ${sum(t['pnl'] for t in no_filter_sells):+.0f})")
    print(f"  Last 6mo with filter:    {len(filter_buys)} BUY (PnL ${sum(t['pnl'] for t in filter_buys):+.0f}), "
          f"{len(filter_sells)} SELL (PnL ${sum(t['pnl'] for t in filter_sells):+.0f})")

    # Detailed last 3mo
    print(f"\nPhase 4: Last 3 Months Trade-by-Trade (No Filter)", flush=True)
    print("-" * 80, flush=True)
    sub_3m = h1_df["2026-02-01":"2026-06-01"]
    t_no_3m, _ = bt_tsmom(sub_3m, d1_filter=False)
    t_yes_3m, filt_3m = bt_tsmom(sub_3m, d1_filter=True)

    print(f"  Without filter: {len(t_no_3m)} trades, Sharpe={_stats(t_no_3m)['sharpe']:.2f}, PnL=${sum(t['pnl'] for t in t_no_3m):+.0f}")
    print(f"  With filter:    {len(t_yes_3m)} trades, Sharpe={_stats(t_yes_3m)['sharpe']:.2f}, PnL=${sum(t['pnl'] for t in t_yes_3m):+.0f}")
    print(f"  Filtered out:   {len(filt_3m)} signals")

    print(f"\n  {'Time':<22} {'Dir':<5} {'Entry':>9} {'Exit':>9} {'PnL':>9} {'Reason':<12} {'WouldFilter':>12}")
    print(f"  {'-'*22} {'-'*5} {'-'*9} {'-'*9} {'-'*9} {'-'*12} {'-'*12}")
    for t in t_no_3m:
        entry_date = pd.Timestamp(t['entry_time']).date()
        # Check if this would be filtered
        would_filter = False
        for f in filt_3m:
            if abs((pd.Timestamp(f['time']) - pd.Timestamp(t['entry_time'])).total_seconds()) < 7200:
                would_filter = True
                break
        # Also check if it exists in filtered trades
        in_filtered = any(t2 for t2 in t_yes_3m
                         if abs((pd.Timestamp(t2['entry_time']) - pd.Timestamp(t['entry_time'])).total_seconds()) < 7200)

        status = "KEPT" if in_filtered else "FILTERED"
        print(f"  {str(t['entry_time'])[:22]:<22} {t['dir']:<5} {t['entry']:>9.2f} {t['exit']:>9.2f} "
              f"{t['pnl']:>+9.2f} {t['reason']:<12} {status:>12}")

    elapsed = time.time() - t0
    print(f"\n{'='*100}")
    print(f"  R176 complete in {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
