#!/usr/bin/env python3
"""
R176e — Chandelier Filter Shootout
====================================
Test various entry filters on Chandelier (Period=22, live params) to find
if any filter improves on the NoFilter baseline (C_Live_NoEMA, Sharpe 4.85).

Filters tested:
 0. NoFilter (baseline)
 1. EMA100 (current live)
 2. EMA50
 3. EMA200
 4. D1 EMA20 (like TSMOM/SESS_BO)
 5. ADX>20
 6. ADX>25
 7. ADX>30
 8. RSI14 range filter (30<RSI<70 = no extreme)
 9. ATR percentile filter (skip low-vol: ATR > 25th pctl)
10. Keltner Channel alignment (close above/below KC mid)
11. MACD histogram direction
12. Volume filter (volume > 20-bar MA)
13. D1 trend (close > D1 EMA50)
14. Multi-TF momentum (H4 close > H4 EMA20)
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
    """Precompute all indicators once for the full dataset."""
    ind = {}
    ind['ATR'] = compute_atr(df)
    ind['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    ind['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    ind['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # ADX
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = (-df['Low'].diff()).clip(lower=0)
    mask = plus_dm < minus_dm; plus_dm[mask] = 0
    mask2 = minus_dm < plus_dm; minus_dm[mask2] = 0
    atr14 = ind['ATR']
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di) * 100).replace([np.inf, -np.inf], 0)
    ind['ADX'] = dx.rolling(14).mean()

    # RSI14
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    ind['RSI14'] = 100 - (100 / (1 + rs))

    # ATR percentile (rolling 500-bar)
    ind['ATR_pctl'] = ind['ATR'].rolling(500).rank(pct=True)

    # Keltner Channel mid (EMA20 + ATR*1.5)
    ind['KC_mid'] = df['Close'].ewm(span=20, adjust=False).mean()

    # MACD histogram
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    ind['MACD_hist'] = macd_line - signal_line

    # D1 EMA20 direction
    d1 = df.resample('1D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    d1_ema20 = d1['Close'].ewm(span=20, adjust=False).mean()
    d1_ema50 = d1['Close'].ewm(span=50, adjust=False).mean()
    d1_dir20 = {}
    d1_dir50 = {}
    for idx, row in d1.iterrows():
        d = idx.date()
        e20 = d1_ema20.loc[idx]
        e50 = d1_ema50.loc[idx]
        d1_dir20[d] = 1 if row['Close'] > e20 else (-1 if row['Close'] < e20 else 0)
        d1_dir50[d] = 1 if row['Close'] > e50 else (-1 if row['Close'] < e50 else 0)
    ind['D1_EMA20_dir'] = d1_dir20
    ind['D1_EMA50_dir'] = d1_dir50

    # Chandelier lines
    hh = df['High'].rolling(PERIOD).max()
    ll = df['Low'].rolling(PERIOD).min()
    atr_raw = (df['High'] - df['Low']).rolling(14).mean()
    ind['chand_long'] = hh - MULT * atr_raw
    ind['chand_short'] = ll + MULT * atr_raw

    return ind


def _get_d1_val(d1_dict, date_val):
    sorted_dates = sorted([dd for dd in d1_dict.keys() if dd < date_val])
    if sorted_dates:
        return d1_dict[sorted_dates[-1]]
    return 0


def bt_chandelier_filtered(df, ind, filter_name):
    """Run Chandelier backtest with a specific filter."""
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = ind['ATR'].values
    cl = ind['chand_long'].values; cs = ind['chand_short'].values
    times = df.index; dates = df.index.date; n = len(df)

    ema50 = ind['EMA50'].values
    ema100 = ind['EMA100'].values
    ema200 = ind['EMA200'].values
    adx = ind['ADX'].values
    rsi = ind['RSI14'].values
    atr_pctl = ind['ATR_pctl'].values
    kc_mid = ind['KC_mid'].values
    macd_hist = ind['MACD_hist'].values

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

        if not flip_bull and not flip_bear:
            continue

        direction = 'BUY' if flip_bull else 'SELL'

        # Apply filter
        passed = True
        if filter_name == 'NoFilter':
            pass
        elif filter_name == 'EMA100':
            if direction == 'BUY' and c[i] <= ema100[i]: passed = False
            elif direction == 'SELL' and c[i] >= ema100[i]: passed = False
        elif filter_name == 'EMA50':
            if direction == 'BUY' and c[i] <= ema50[i]: passed = False
            elif direction == 'SELL' and c[i] >= ema50[i]: passed = False
        elif filter_name == 'EMA200':
            if direction == 'BUY' and c[i] <= ema200[i]: passed = False
            elif direction == 'SELL' and c[i] >= ema200[i]: passed = False
        elif filter_name == 'D1_EMA20':
            d1v = _get_d1_val(ind['D1_EMA20_dir'], dates[i])
            if direction == 'BUY' and d1v == -1: passed = False
            elif direction == 'SELL' and d1v == 1: passed = False
        elif filter_name == 'D1_EMA50':
            d1v = _get_d1_val(ind['D1_EMA50_dir'], dates[i])
            if direction == 'BUY' and d1v == -1: passed = False
            elif direction == 'SELL' and d1v == 1: passed = False
        elif filter_name == 'ADX>20':
            if np.isnan(adx[i]) or adx[i] <= 20: passed = False
        elif filter_name == 'ADX>25':
            if np.isnan(adx[i]) or adx[i] <= 25: passed = False
        elif filter_name == 'ADX>30':
            if np.isnan(adx[i]) or adx[i] <= 30: passed = False
        elif filter_name == 'RSI_30_70':
            if np.isnan(rsi[i]): passed = False
            elif direction == 'BUY' and rsi[i] > 70: passed = False
            elif direction == 'SELL' and rsi[i] < 30: passed = False
        elif filter_name == 'RSI_extreme':
            if np.isnan(rsi[i]): passed = False
            elif direction == 'BUY' and rsi[i] < 40: passed = False
            elif direction == 'SELL' and rsi[i] > 60: passed = False
        elif filter_name == 'ATR>25pctl':
            if np.isnan(atr_pctl[i]) or atr_pctl[i] < 0.25: passed = False
        elif filter_name == 'ATR>50pctl':
            if np.isnan(atr_pctl[i]) or atr_pctl[i] < 0.50: passed = False
        elif filter_name == 'KC_align':
            if direction == 'BUY' and c[i] <= kc_mid[i]: passed = False
            elif direction == 'SELL' and c[i] >= kc_mid[i]: passed = False
        elif filter_name == 'MACD_dir':
            if np.isnan(macd_hist[i]): passed = False
            elif direction == 'BUY' and macd_hist[i] <= 0: passed = False
            elif direction == 'SELL' and macd_hist[i] >= 0: passed = False
        elif filter_name == 'EMA50+ADX>20':
            ema_ok = (direction == 'BUY' and c[i] > ema50[i]) or (direction == 'SELL' and c[i] < ema50[i])
            adx_ok = not np.isnan(adx[i]) and adx[i] > 20
            if not (ema_ok and adx_ok): passed = False
        elif filter_name == 'D1_EMA20+ADX>20':
            d1v = _get_d1_val(ind['D1_EMA20_dir'], dates[i])
            d1_ok = not (direction == 'BUY' and d1v == -1) and not (direction == 'SELL' and d1v == 1)
            adx_ok = not np.isnan(adx[i]) and adx[i] > 20
            if not (d1_ok and adx_ok): passed = False

        if not passed:
            continue

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


FILTERS = [
    'NoFilter', 'EMA100', 'EMA50', 'EMA200',
    'D1_EMA20', 'D1_EMA50',
    'ADX>20', 'ADX>25', 'ADX>30',
    'RSI_30_70', 'RSI_extreme',
    'ATR>25pctl', 'ATR>50pctl',
    'KC_align', 'MACD_dir',
    'EMA50+ADX>20', 'D1_EMA20+ADX>20',
]


def main():
    t0 = time.time()
    print("=" * 120, flush=True)
    print("  R176e — Chandelier Filter Shootout (Period=22, Live exit params)", flush=True)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Lot={LOT}, SL={SL}, TP={TP}, "
          f"Trail={TRAIL_ACT}/{TRAIL_DIST}, MH={MH}, Cap=${CAP}", flush=True)
    print("=" * 120, flush=True)

    h1_df = load_h1()
    print(f"  {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})\n", flush=True)

    print("  Precomputing indicators...", flush=True)
    ind = precompute_indicators(h1_df)
    h1_df_clean = h1_df.copy()

    # Phase 1: Full sample screening
    print(f"\nPhase 1: Full Sample Screening", flush=True)
    print("-" * 120, flush=True)
    print(f"  {'Filter':<20} {'#':>5} {'Shrp':>6} {'PnL':>9} {'WR':>6} {'MaxDD':>7} {'Filt%':>6} {'vs Base':>8}")

    base_n = None
    results = []
    for fname in FILTERS:
        trades = bt_chandelier_filtered(h1_df_clean, ind, fname)
        daily = _trades_to_daily(trades)
        n = len(trades)
        sharpe = _sharpe(daily)
        pnl = sum(t['pnl'] for t in trades)
        wr = sum(1 for t in trades if t['pnl'] > 0) / n * 100 if n > 0 else 0
        mdd = _max_dd(daily)
        if base_n is None:
            base_n = n; base_sharpe = sharpe
        filt_pct = (1 - n / base_n) * 100 if base_n > 0 else 0
        delta = sharpe - base_sharpe

        results.append({'name': fname, 'n': n, 'sharpe': sharpe, 'pnl': pnl,
                        'wr': wr, 'mdd': mdd, 'filt_pct': filt_pct, 'delta': delta})
        print(f"  {fname:<20} {n:>5} {sharpe:>6.2f} ${pnl:>8,.0f} {wr:>5.1f}% ${mdd:>6,.0f} {filt_pct:>5.1f}% {delta:>+7.2f}")

    # Rank by Sharpe
    ranked = sorted(results, key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  Top 5 by Sharpe:")
    for i, r in enumerate(ranked[:5]):
        print(f"    #{i+1} {r['name']:<20} Sharpe={r['sharpe']:.2f} (Δ{r['delta']:+.2f}), "
              f"#{r['n']}, WR={r['wr']:.1f}%, MaxDD=${r['mdd']:,.0f}")

    # Phase 2: K-Fold for top candidates (Sharpe > baseline)
    top_filters = [r['name'] for r in ranked if r['delta'] > 0 or r['name'] == 'NoFilter'][:8]

    print(f"\n\nPhase 2: K-Fold Validation (top filters)", flush=True)
    print("-" * 120, flush=True)

    kfold_results = []
    for fname in top_filters:
        fold_sharpes = []
        for foldname, start, end in FOLDS:
            sub = h1_df_clean[start:end]
            if len(sub) < 600:
                fold_sharpes.append(0); continue
            sub_ind = precompute_indicators(sub)
            trades = bt_chandelier_filtered(sub, sub_ind, fname)
            fold_sharpes.append(_sharpe(_trades_to_daily(trades)))

        pos_folds = sum(1 for s in fold_sharpes if s > 0)
        mean_s = np.mean(fold_sharpes)
        std_s = np.std(fold_sharpes)
        min_s = min(fold_sharpes)
        status = "PASS" if pos_folds >= 4 else "FAIL"
        kfold_results.append({'name': fname, 'pos': pos_folds, 'mean': mean_s,
                              'std': std_s, 'min': min_s, 'folds': fold_sharpes})
        print(f"  {fname:<20}: {pos_folds}/6 pos, mean={mean_s:.2f}, std={std_s:.2f}, min={min_s:.2f} "
              f"[{status}]  {[round(s,1) for s in fold_sharpes]}")

    # Phase 3: Period breakdown for top 3
    top3 = [r['name'] for r in ranked[:3]]
    if 'NoFilter' not in top3:
        top3.append('NoFilter')

    print(f"\n\nPhase 3: Period Breakdown (top filters)", flush=True)
    print("-" * 120, flush=True)

    periods = [
        ("2015-2017", "2015-01-01", "2017-01-01"),
        ("2017-2019", "2017-01-01", "2019-01-01"),
        ("2019-2021", "2019-01-01", "2021-01-01"),
        ("2021-2023", "2021-01-01", "2023-01-01"),
        ("2023-2025", "2023-01-01", "2025-01-01"),
        ("2025-2026", "2025-01-01", "2026-06-01"),
    ]

    header = f"  {'Period':<12}"
    for fname in top3:
        header += f" {fname[:16]:>18}"
    print(header)

    for pname, start, end in periods:
        sub = h1_df_clean[start:end]
        if len(sub) < 600:
            line = f"  {pname:<12}"
            for _ in top3:
                line += f" {'N/A':>18}"
            print(line); continue
        sub_ind = precompute_indicators(sub)
        line = f"  {pname:<12}"
        for fname in top3:
            trades = bt_chandelier_filtered(sub, sub_ind, fname)
            daily = _trades_to_daily(trades)
            s = _sharpe(daily)
            line += f" {s:>6.2f} ({len(trades):>4})"
        print(line)

    # Summary
    print(f"\n\n{'='*120}")
    print(f"  SUMMARY")
    print(f"{'='*120}\n")
    best = ranked[0]
    print(f"  Baseline (NoFilter): Sharpe={base_sharpe:.2f}")
    print(f"  Best filter: {best['name']} (Sharpe={best['sharpe']:.2f}, Δ{best['delta']:+.2f})")

    better = [r for r in ranked if r['delta'] > 0.1]
    if better:
        print(f"\n  Filters with Sharpe > baseline + 0.10:")
        for r in better:
            print(f"    {r['name']:<20} Sharpe={r['sharpe']:.2f} (Δ{r['delta']:+.2f}), "
                  f"filtered {r['filt_pct']:.0f}% trades")
    else:
        print(f"\n  No filter improves Sharpe by > 0.10 over NoFilter baseline.")
        print(f"  Recommendation: Keep NoFilter or current EMA100.")

    elapsed = time.time() - t0
    print(f"\n  R176e complete in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
