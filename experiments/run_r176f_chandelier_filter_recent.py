#!/usr/bin/env python3
"""
R176f — Chandelier Filter: Recent Period Deep Dive
====================================================
Test top filters on recent months (3M, 6M, 12M) to see if the full-sample
winners hold up in the most recent market regime.
"""
import sys, os, time
import numpy as np
import pandas as pd
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

PV = 100; SPREAD = 0.30; LOT = 0.08
PERIOD = 22; MULT = 3.0; SL = 4.5; TP = 8.0
TRAIL_ACT = 0.14; TRAIL_DIST = 0.025; MH = 20; CAP = 25


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


def _run_exit(pos, i, h, lo_v, c, times):
    if pos['dir'] == 'BUY':
        pnl_h = (h - pos['entry'] - SPREAD) * LOT * PV
        pnl_l = (lo_v - pos['entry'] - SPREAD) * LOT * PV
        pnl_c = (c - pos['entry'] - SPREAD) * LOT * PV
    else:
        pnl_h = (pos['entry'] - lo_v - SPREAD) * LOT * PV
        pnl_l = (pos['entry'] - h - SPREAD) * LOT * PV
        pnl_c = (pos['entry'] - c - SPREAD) * LOT * PV
    tp_val = TP * pos['atr'] * LOT * PV
    sl_val = SL * pos['atr'] * LOT * PV
    if pnl_h >= tp_val:
        return _mk(pos, c, times[i], "TP", i, tp_val)
    if pnl_l <= -sl_val:
        return _mk(pos, c, times[i], "SL", i, -sl_val)
    if CAP > 0 and pnl_c < -CAP:
        return _mk(pos, c, times[i], "Cap", i, -CAP)
    ad = TRAIL_ACT * pos['atr']; td = TRAIL_DIST * pos['atr']
    if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
        ts_p = h - td
        if lo_v <= ts_p:
            return _mk(pos, c, times[i], "Trail", i, (ts_p - pos['entry'] - SPREAD) * LOT * PV)
    elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
        ts_p = lo_v + td
        if h >= ts_p:
            return _mk(pos, c, times[i], "Trail", i, (pos['entry'] - ts_p - SPREAD) * LOT * PV)
    if i - pos['bar'] >= MH:
        return _mk(pos, c, times[i], "Timeout", i, pnl_c)
    return None


def precompute_indicators(df):
    ind = {}
    ind['ATR'] = compute_atr(df)
    ind['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    ind['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = (-df['Low'].diff()).clip(lower=0)
    mask = plus_dm < minus_dm; plus_dm[mask] = 0
    mask2 = minus_dm < plus_dm; minus_dm[mask2] = 0
    atr14 = ind['ATR']
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di) * 100).replace([np.inf, -np.inf], 0)
    ind['ADX'] = dx.rolling(14).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    ind['RSI14'] = 100 - (100 / (1 + rs))

    ind['ATR_pctl'] = ind['ATR'].rolling(500).rank(pct=True)

    d1 = df.resample('1D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    d1_ema20 = d1['Close'].ewm(span=20, adjust=False).mean()
    d1_dir20 = {}
    for idx, row in d1.iterrows():
        d = idx.date()
        e20 = d1_ema20.loc[idx]
        d1_dir20[d] = 1 if row['Close'] > e20 else (-1 if row['Close'] < e20 else 0)
    ind['D1_EMA20_dir'] = d1_dir20

    atr_raw = (df['High'] - df['Low']).rolling(14).mean()
    hh = df['High'].rolling(PERIOD).max()
    ll = df['Low'].rolling(PERIOD).min()
    ind['chand_long'] = hh - MULT * atr_raw
    ind['chand_short'] = ll + MULT * atr_raw
    return ind


def _get_d1_val(d1_dict, date_val):
    sorted_dates = sorted([dd for dd in d1_dict.keys() if dd < date_val])
    return d1_dict[sorted_dates[-1]] if sorted_dates else 0


FILTERS = [
    'NoFilter', 'EMA100', 'D1_EMA20+ADX>20', 'RSI_30_70',
    'D1_EMA20', 'ATR>25pctl', 'ADX>20', 'EMA200',
]


def bt_chandelier_filtered(df, ind, filter_name):
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = ind['ATR'].values
    cl = ind['chand_long'].values; cs = ind['chand_short'].values
    times = df.index; dates = df.index.date; n = len(df)
    ema100 = ind['EMA100'].values
    ema200 = ind['EMA200'].values
    adx = ind['ADX'].values
    rsi = ind['RSI14'].values
    atr_pctl = ind['ATR_pctl'].values

    trades = []; pos = None; last_exit = -999
    start = max(PERIOD + 2, 500)
    for i in range(start, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], times)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(cl[i]) or np.isnan(cl[i-1]) or np.isnan(cs[i]) or np.isnan(cs[i-1]):
            continue
        flip_bull = c[i] > cl[i] and c[i-1] <= cl[i-1]
        flip_bear = c[i] < cs[i] and c[i-1] >= cs[i-1]
        if not flip_bull and not flip_bear: continue
        direction = 'BUY' if flip_bull else 'SELL'

        passed = True
        if filter_name == 'NoFilter':
            pass
        elif filter_name == 'EMA100':
            if direction == 'BUY' and c[i] <= ema100[i]: passed = False
            elif direction == 'SELL' and c[i] >= ema100[i]: passed = False
        elif filter_name == 'EMA200':
            if direction == 'BUY' and c[i] <= ema200[i]: passed = False
            elif direction == 'SELL' and c[i] >= ema200[i]: passed = False
        elif filter_name == 'D1_EMA20':
            d1v = _get_d1_val(ind['D1_EMA20_dir'], dates[i])
            if direction == 'BUY' and d1v == -1: passed = False
            elif direction == 'SELL' and d1v == 1: passed = False
        elif filter_name == 'ADX>20':
            if np.isnan(adx[i]) or adx[i] <= 20: passed = False
        elif filter_name == 'RSI_30_70':
            if np.isnan(rsi[i]): passed = False
            elif direction == 'BUY' and rsi[i] > 70: passed = False
            elif direction == 'SELL' and rsi[i] < 30: passed = False
        elif filter_name == 'ATR>25pctl':
            if np.isnan(atr_pctl[i]) or atr_pctl[i] < 0.25: passed = False
        elif filter_name == 'D1_EMA20+ADX>20':
            d1v = _get_d1_val(ind['D1_EMA20_dir'], dates[i])
            d1_ok = not (direction == 'BUY' and d1v == -1) and not (direction == 'SELL' and d1v == 1)
            adx_ok = not np.isnan(adx[i]) and adx[i] > 20
            if not (d1_ok and adx_ok): passed = False
        if not passed: continue
        pos = {'dir': direction,
               'entry': c[i] + (SPREAD/2 if direction == 'BUY' else -SPREAD/2),
               'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def _trades_to_daily(trades):
    if not trades: return np.array([0.0])
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    return np.array([daily[k] for k in sorted(daily.keys())]) if daily else np.array([0.0])


def _sharpe(arr):
    if len(arr) < 10 or np.std(arr, ddof=1) == 0: return 0.0
    return float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))


def _max_dd(arr):
    if len(arr) == 0: return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def _stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0,
                'buy_n': 0, 'sell_n': 0, 'buy_pnl': 0, 'sell_pnl': 0, 'exits': {}}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    buys = [t for t in trades if t['dir'] == 'BUY']
    sells = [t for t in trades if t['dir'] == 'SELL']
    exits = {}
    for t in trades: exits[t['reason']] = exits.get(t['reason'], 0) + 1
    return {
        'n': n, 'sharpe': round(_sharpe(daily), 2),
        'pnl': round(sum(pnls), 0), 'wr': round(sum(1 for p in pnls if p > 0)/n*100, 1),
        'max_dd': round(_max_dd(daily), 0),
        'buy_n': len(buys), 'sell_n': len(sells),
        'buy_pnl': round(sum(t['pnl'] for t in buys), 0),
        'sell_pnl': round(sum(t['pnl'] for t in sells), 0),
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
    print("=" * 130, flush=True)
    print("  R176f — Chandelier Filter: Recent Period Deep Dive", flush=True)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 130, flush=True)

    h1_df = load_h1()
    end_date = h1_df.index[-1]
    print(f"  {len(h1_df)} bars, ends at {end_date}\n", flush=True)

    # Need enough warmup data for indicators, so load from much earlier but only score recent trades
    windows = [
        ("Last 1M",  "2026-04-01", None),
        ("Last 2M",  "2026-03-01", None),
        ("Last 3M",  "2026-02-01", None),
        ("Last 6M",  "2025-11-01", None),
        ("Last 12M", "2025-05-01", None),
        ("2025-H1",  "2025-01-01", "2025-07-01"),
        ("2025-H2",  "2025-07-01", "2026-01-01"),
        ("2024-H2",  "2024-07-01", "2025-01-01"),
        ("2024-H1",  "2024-01-01", "2024-07-01"),
    ]

    # For each window, use full dataset for indicator warmup, filter trades by date range
    print(f"  Precomputing indicators on full dataset...", flush=True)
    ind_full = precompute_indicators(h1_df)

    # Phase 1: Detailed comparison for each recent window
    for wname, w_start, w_end in windows:
        print(f"\n{'='*130}")
        print(f"  {wname}: {w_start} ~ {w_end or 'now'}")
        print(f"{'='*130}")
        print(f"  {'Filter':<20} {'#':>5} {'Shrp':>6} {'PnL':>9} {'WR':>6} {'MaxDD':>7} "
              f"{'BUY#':>5} {'SELL#':>5} {'BUY$':>8} {'SELL$':>8} Exits")

        for fname in FILTERS:
            all_trades = bt_chandelier_filtered(h1_df, ind_full, fname)
            # Filter trades within the window
            filtered = []
            for t in all_trades:
                exit_dt = pd.Timestamp(t['exit_time'])
                if exit_dt < pd.Timestamp(w_start, tz='UTC'):
                    continue
                if w_end and exit_dt >= pd.Timestamp(w_end, tz='UTC'):
                    continue
                filtered.append(t)

            s = _stats(filtered)
            exits_str = ', '.join(f"{k}:{v}" for k, v in sorted(s['exits'].items()))
            print(f"  {fname:<20} {s['n']:>5} {s['sharpe']:>6.2f} ${s['pnl']:>8,} {s['wr']:>5.1f}% ${s['max_dd']:>6,} "
                  f"{s['buy_n']:>5} {s['sell_n']:>5} ${s['buy_pnl']:>7,} ${s['sell_pnl']:>7,} {exits_str}")

    # Phase 2: Monthly breakdown for top 4 filters (last 12 months)
    print(f"\n\n{'='*130}")
    print(f"  Monthly Sharpe Breakdown (Last 12M)")
    print(f"{'='*130}")

    top4 = ['NoFilter', 'D1_EMA20+ADX>20', 'RSI_30_70', 'EMA100']
    months = pd.date_range('2025-05-01', '2026-05-01', freq='MS')

    header = f"  {'Month':<10}"
    for fname in top4:
        header += f" {fname[:18]:>20}"
    print(header)
    print(f"  {'-'*10}" + f" {'-'*20}" * len(top4))

    for mi in range(len(months) - 1):
        m_start = months[mi]
        m_end = months[mi + 1]
        mname = m_start.strftime('%Y-%m')
        line = f"  {mname:<10}"
        for fname in top4:
            all_trades = bt_chandelier_filtered(h1_df, ind_full, fname)
            monthly_trades = [t for t in all_trades
                              if pd.Timestamp(t['exit_time']) >= m_start.tz_localize('UTC')
                              and pd.Timestamp(t['exit_time']) < m_end.tz_localize('UTC')]
            s = _stats(monthly_trades)
            line += f"  {s['sharpe']:>6.2f} ({s['n']:>3}) ${s['pnl']:>6,}"
        print(line)

    elapsed = time.time() - t0
    print(f"\n  R176f complete in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
