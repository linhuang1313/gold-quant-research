#!/usr/bin/env python3
"""
Round 52 — Multi-Strategy Lot Portfolio Optimization
=====================================================
Step 1: Generate daily PnL series for each strategy (fixed lot=0.03):
    - L8_MAX  (M15 engine, existing full backtest)
    - D1 KC   (R51 best: E10/M2.5/ADX18/SL4.5/TP8.0/MH20/Trail0.2/0.05)
    - H4 KC   (R51 best: E15/M2.5/ADX10/SL4.5/TP6.0/MH50/Trail0.2/0.04)
    - PSAR    (R51 best: AF0.01/MX0.05/SL4.5/TP16.0/MH20/Trail0.2/0.04)

Step 2: Brute-force lot multiplier grid:
    each strategy: [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    → combined daily PnL → Sharpe, MaxDD, PnL

Step 3: K-Fold validation on top combos

USAGE (server)
--------------
    cd /root/gold-quant-research
    nohup python3 -u experiments/run_round52_lot_portfolio.py \
        > results/round52_results/stdout.txt 2>&1 &
"""
import sys, os, io, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

_script_dir = os.path.dirname(os.path.abspath(__file__))
for _candidate in [os.path.join(_script_dir, '..'), os.path.join(_script_dir, '..', '..'), os.getcwd()]:
    _candidate = os.path.abspath(_candidate)
    if os.path.isdir(os.path.join(_candidate, 'backtest')):
        sys.path.insert(0, _candidate)
        os.chdir(_candidate)
        break

OUTPUT_DIR = Path("results/round52_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPREAD = 0.50
BASE_LOT = 0.03

KFOLD_FOLDS = [
    ("F1_2015-2016", "2015-01-01", "2016-12-31"),
    ("F2_2017-2018", "2017-01-01", "2018-12-31"),
    ("F3_2019-2020", "2019-01-01", "2020-12-31"),
    ("F4_2021-2022", "2021-01-01", "2022-12-31"),
    ("F5_2023-2024", "2023-01-01", "2024-12-31"),
    ("F6_2025-2026", "2025-01-01", "2026-12-31"),
]


# ═══════════════════════════════════════════════════════════════
# Indicator helpers (from R51)
# ═══════════════════════════════════════════════════════════════

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

def add_psar(df, af_start=0.02, af_max=0.20):
    df = df.copy()
    n = len(df); psar = np.zeros(n); direction = np.ones(n)
    af = af_start; ep = df['High'].iloc[0]; psar[0] = df['Low'].iloc[0]
    for i in range(1, n):
        prev_psar = psar[i-1]
        if direction[i-1] == 1:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = min(psar[i], df['Low'].iloc[i-1], df['Low'].iloc[max(0,i-2)])
            if df['Low'].iloc[i] < psar[i]:
                direction[i] = -1; psar[i] = ep; ep = df['Low'].iloc[i]; af = af_start
            else:
                direction[i] = 1
                if df['High'].iloc[i] > ep:
                    ep = df['High'].iloc[i]; af = min(af + af_start, af_max)
        else:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = max(psar[i], df['High'].iloc[i-1], df['High'].iloc[max(0,i-2)])
            if df['High'].iloc[i] > psar[i]:
                direction[i] = 1; psar[i] = ep; ep = df['High'].iloc[i]; af = af_start
            else:
                direction[i] = -1
                if df['Low'].iloc[i] < ep:
                    ep = df['Low'].iloc[i]; af = min(af + af_start, af_max)
    df['PSAR'] = psar; df['PSAR_dir'] = direction
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    return df


# ═══════════════════════════════════════════════════════════════
# Simple backtest → trade list (returning raw trades for daily PnL)
# ═══════════════════════════════════════════════════════════════

def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}

def backtest_kc_trades(df_prepared, adx_thresh=18, sl_atr=3.5, tp_atr=8.0,
                       trail_act_atr=0.28, trail_dist_atr=0.06, max_hold=20,
                       spread=SPREAD, lot=BASE_LOT):
    df = df_prepared; trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    kc_up = df['KC_upper'].values; kc_lo = df['KC_lower'].values
    atr = df['ATR'].values; adx_arr = df['ADX'].values
    times = df.index; n = len(df); last_exit = -999

    for i in range(1, n):
        c = close[i]; h = high[i]; lo_v = low[i]
        cur_atr = atr[i]; cur_adx = adx_arr[i]
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (h - pos['entry'] - spread) * lot * 100
                pnl_l = (lo_v - pos['entry'] - spread) * lot * 100
                pnl_c = (c - pos['entry'] - spread) * lot * 100
            else:
                pnl_h = (pos['entry'] - lo_v - spread) * lot * 100
                pnl_l = (pos['entry'] - h - spread) * lot * 100
                pnl_c = (pos['entry'] - c - spread) * lot * 100
            tp_val = tp_atr * pos['atr'] * lot * 100
            sl_val = sl_atr * pos['atr'] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                trades.append(_mk(pos, c, times[i], "TP", i, tp_val)); exited = True
            elif pnl_l <= -sl_val:
                trades.append(_mk(pos, c, times[i], "SL", i, -sl_val)); exited = True
            else:
                act_dist = trail_act_atr * pos['atr']; trail_d = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= act_dist:
                    ts_p = h - trail_d
                    if lo_v <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos['entry'] - spread) * lot * 100)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= act_dist:
                    ts_p = lo_v + trail_d
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * 100)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if exited: pos = None; last_exit = i; continue
        if pos is not None: continue
        if i - last_exit < 2: continue
        if np.isnan(cur_adx) or cur_adx < adx_thresh: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        prev_c = close[i-1]
        if prev_c > kc_up[i-1]:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif prev_c < kc_lo[i-1]:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
    return trades


def backtest_psar_trades(df_prepared, sl_atr=3.5, tp_atr=8.0,
                         trail_act_atr=0.28, trail_dist_atr=0.06, max_hold=50,
                         spread=SPREAD, lot=BASE_LOT):
    df = df_prepared.dropna(subset=['PSAR_dir', 'ATR'])
    trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    psar_dir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df); last_exit = -999

    for i in range(1, n):
        c = close[i]; h = high[i]; lo_v = low[i]; cur_atr = atr[i]
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (h - pos['entry'] - spread) * lot * 100
                pnl_l = (lo_v - pos['entry'] - spread) * lot * 100
                pnl_c = (c - pos['entry'] - spread) * lot * 100
            else:
                pnl_h = (pos['entry'] - lo_v - spread) * lot * 100
                pnl_l = (pos['entry'] - h - spread) * lot * 100
                pnl_c = (pos['entry'] - c - spread) * lot * 100
            tp_val = tp_atr * pos['atr'] * lot * 100
            sl_val = sl_atr * pos['atr'] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                trades.append(_mk(pos, c, times[i], "TP", i, tp_val)); exited = True
            elif pnl_l <= -sl_val:
                trades.append(_mk(pos, c, times[i], "SL", i, -sl_val)); exited = True
            else:
                act_dist = trail_act_atr * pos['atr']; trail_d = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= act_dist:
                    ts_p = h - trail_d
                    if lo_v <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos['entry'] - spread) * lot * 100)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= act_dist:
                    ts_p = lo_v + trail_d
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * 100)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if exited: pos = None; last_exit = i; continue
        if pos is not None: continue
        if i - last_exit < 2: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        prev_d = psar_dir[i-1]; cur_d = psar_dir[i]
        if prev_d == -1 and cur_d == 1:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif prev_d == 1 and cur_d == -1:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
    return trades


def trades_to_daily_pnl(trades):
    """Convert trade list to daily PnL Series (date -> total pnl)."""
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    if not daily:
        return pd.Series(dtype=float)
    s = pd.Series(daily).sort_index()
    return s


def calc_portfolio_stats(daily_pnl_series, label=""):
    """Calculate stats from combined daily PnL series."""
    if daily_pnl_series.empty or daily_pnl_series.sum() == 0:
        return {'label': label, 'sharpe': 0, 'total_pnl': 0, 'max_dd': 0,
                'n_days': 0, 'avg_daily': 0, 'volatility': 0}
    arr = daily_pnl_series.values
    eq = np.cumsum(arr)
    dd = (np.maximum.accumulate(eq) - eq).max()
    sh = float(arr.mean() / arr.std() * np.sqrt(252)) if arr.std() > 0 else 0
    return {
        'label': label,
        'sharpe': round(sh, 3),
        'total_pnl': round(float(arr.sum()), 2),
        'max_dd': round(float(dd), 2),
        'n_days': len(arr),
        'avg_daily': round(float(arr.mean()), 2),
        'volatility': round(float(arr.std()), 2),
    }


def save_checkpoint(data, filename):
    path = OUTPUT_DIR / filename
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, default=str)
    print(f"  [Checkpoint] {path}", flush=True)

def load_checkpoint(filename):
    path = OUTPUT_DIR / filename
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════
# Step 1: Generate daily PnL for each strategy
# ═══════════════════════════════════════════════════════════════

def generate_daily_pnl_all(h1_df, m15_df):
    """Generate daily PnL for all 4 strategies. Returns dict of {name: pd.Series}."""

    print("\n  [Step 1] Generating daily PnL for each strategy...", flush=True)
    daily_pnls = {}

    # --- L8_MAX via engine ---
    print("    L8_MAX (M15 engine)...", flush=True)
    from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS
    from backtest.engine import BacktestEngine

    l8_max_kw = {
        **LIVE_PARITY_KWARGS,
        'keltner_adx_threshold': 14,
        'regime_config': {
            'low':    {'trail_act': 0.22, 'trail_dist': 0.04},
            'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
            'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
        },
        'min_entry_gap_hours': 1.0,
        'keltner_max_hold_m15': 20,
        'spread_cost': SPREAD,
        'initial_capital': 2000,
        'min_lot_size': BASE_LOT, 'max_lot_size': BASE_LOT,
    }
    data = DataBundle(m15_df, h1_df)
    result = run_variant(data, "L8_MAX", **l8_max_kw)
    l8_trades = result.get('_trades', [])

    # Apply H1 KC filter + Cap30 post-processing (as in R49)
    h1_kc = add_kc(h1_df.copy(), ema_period=15, atr_period=14, mult=2.0)
    h1_kc_dir = (h1_kc['Close'] > h1_kc['EMA']).astype(int) * 2 - 1

    filtered_trades = []
    for t in l8_trades:
        et = t.entry_time if hasattr(t, 'entry_time') else t.get('entry_time', '')
        entry_time = pd.Timestamp(et)
        if pd.isna(entry_time): continue
        h1_time = entry_time.floor('h')
        if h1_time in h1_kc_dir.index:
            h1_dir_val = h1_kc_dir.loc[h1_time]
            direction = t.direction if hasattr(t, 'direction') else t.get('direction', '')
            trade_dir = 1 if direction in ('BUY', 'LONG', 1) else -1
            if h1_dir_val != trade_dir:
                continue
        pnl = t.pnl if hasattr(t, 'pnl') else t.get('pnl', 0)
        cap = 30
        if pnl < -cap:
            pnl = -cap
        ext = t.exit_time if hasattr(t, 'exit_time') else t.get('exit_time', '')
        filtered_trades.append({'pnl': pnl, 'exit_time': ext})

    daily_pnls['L8_MAX'] = trades_to_daily_pnl(filtered_trades)
    print(f"      L8_MAX: {len(filtered_trades)} trades, ${sum(t['pnl'] for t in filtered_trades):,.0f} PnL",
          flush=True)

    # --- D1 KC (R51 best) ---
    print("    D1 KC...", flush=True)
    d1_df = h1_df.resample('D').agg({'Open':'first','High':'max','Low':'min',
                                      'Close':'last','Volume':'sum'}).dropna()
    d1_df = add_kc(d1_df, ema_period=10, atr_period=14, mult=2.5)
    d1_df = d1_df.dropna()
    d1_trades = backtest_kc_trades(d1_df, adx_thresh=18, sl_atr=4.5, tp_atr=8.0,
                                    trail_act_atr=0.20, trail_dist_atr=0.05, max_hold=20)
    daily_pnls['D1_KC'] = trades_to_daily_pnl(d1_trades)
    print(f"      D1_KC: {len(d1_trades)} trades, ${sum(t['pnl'] for t in d1_trades):,.0f} PnL", flush=True)

    # --- H4 KC (R51 best) ---
    print("    H4 KC...", flush=True)
    h4_df = h1_df.resample('4h').agg({'Open':'first','High':'max','Low':'min',
                                       'Close':'last','Volume':'sum'}).dropna()
    h4_df = add_kc(h4_df, ema_period=15, atr_period=14, mult=2.5)
    h4_df = h4_df.dropna()
    h4_trades = backtest_kc_trades(h4_df, adx_thresh=10, sl_atr=4.5, tp_atr=6.0,
                                    trail_act_atr=0.20, trail_dist_atr=0.04, max_hold=50)
    daily_pnls['H4_KC'] = trades_to_daily_pnl(h4_trades)
    print(f"      H4_KC: {len(h4_trades)} trades, ${sum(t['pnl'] for t in h4_trades):,.0f} PnL", flush=True)

    # --- PSAR (R51 best) ---
    print("    PSAR...", flush=True)
    h1_psar = add_psar(h1_df.copy(), af_start=0.01, af_max=0.05)
    psar_trades = backtest_psar_trades(h1_psar, sl_atr=4.5, tp_atr=16.0,
                                        trail_act_atr=0.20, trail_dist_atr=0.04, max_hold=20)
    daily_pnls['PSAR'] = trades_to_daily_pnl(psar_trades)
    print(f"      PSAR: {len(psar_trades)} trades, ${sum(t['pnl'] for t in psar_trades):,.0f} PnL", flush=True)

    # Align to common date range
    all_dates = set()
    for s in daily_pnls.values():
        all_dates.update(s.index.tolist())
    all_dates = sorted(all_dates)
    idx = pd.Index(all_dates)
    for name in daily_pnls:
        daily_pnls[name] = daily_pnls[name].reindex(idx, fill_value=0.0)

    print(f"\n  Daily PnL range: {all_dates[0]} -> {all_dates[-1]} ({len(all_dates)} days)", flush=True)

    # Print individual stats
    print(f"\n  {'Strategy':>10} {'Sharpe':>8} {'PnL':>12} {'MaxDD':>10} {'Days':>6}")
    for name, s in daily_pnls.items():
        st = calc_portfolio_stats(s, name)
        print(f"  {name:>10} {st['sharpe']:>8.2f} ${st['total_pnl']:>11,.0f} ${st['max_dd']:>9,.0f} {st['n_days']:>6}")

    return daily_pnls


# ═══════════════════════════════════════════════════════════════
# Step 2: Lot multiplier grid search
# ═══════════════════════════════════════════════════════════════

LOT_GRID = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]

def run_lot_grid(daily_pnls):
    print(f"\n{'='*80}")
    print(f"  [Step 2] Lot Portfolio Grid Search")
    print(f"{'='*80}")

    existing = load_checkpoint("lot_grid_results.json")
    if existing:
        print(f"  [Resume] Found {len(existing)} results", flush=True)
        return existing

    strat_names = ['L8_MAX', 'D1_KC', 'H4_KC', 'PSAR']
    base_daily = {name: daily_pnls[name].values for name in strat_names}
    base_lots = {name: BASE_LOT for name in strat_names}

    combos = list(product(LOT_GRID, LOT_GRID, LOT_GRID, LOT_GRID))
    combos = [c for c in combos if sum(c) > 0]

    total = len(combos)
    print(f"  Total combos: {total:,} (after removing all-zero)", flush=True)

    t0 = time.time()
    results = []

    for idx, (l_l8, l_d1, l_h4, l_psar) in enumerate(combos):
        combined = np.zeros_like(base_daily['L8_MAX'], dtype=float)
        if l_l8 > 0:
            combined += base_daily['L8_MAX'] * (l_l8 / BASE_LOT)
        if l_d1 > 0:
            combined += base_daily['D1_KC'] * (l_d1 / BASE_LOT)
        if l_h4 > 0:
            combined += base_daily['H4_KC'] * (l_h4 / BASE_LOT)
        if l_psar > 0:
            combined += base_daily['PSAR'] * (l_psar / BASE_LOT)

        eq = np.cumsum(combined)
        dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 0 else 0
        total_pnl = float(combined.sum())

        std = combined.std()
        sharpe = float(combined.mean() / std * np.sqrt(252)) if std > 0 else 0

        total_lot = l_l8 + l_d1 + l_h4 + l_psar
        label = f"L8={l_l8}_D1={l_d1}_H4={l_h4}_PS={l_psar}"

        results.append({
            'label': label,
            'l8': l_l8, 'd1': l_d1, 'h4': l_h4, 'psar': l_psar,
            'total_lot': round(total_lot, 2),
            'sharpe': round(sharpe, 3),
            'total_pnl': round(total_pnl, 2),
            'max_dd': round(dd, 2),
            'pnl_per_lot': round(total_pnl / total_lot, 2) if total_lot > 0 else 0,
        })

        if (idx + 1) % 1000 == 0 or (idx + 1) == total:
            elapsed = time.time() - t0
            print(f"    {idx+1}/{total} ({(idx+1)/total*100:.0f}%) | {elapsed:.1f}s", flush=True)

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    save_checkpoint(results, "lot_grid_results.json")

    # Print top 30
    print(f"\n  {'Rank':>4} {'Label':>35} {'Sharpe':>8} {'PnL':>12} {'MaxDD':>10} "
          f"{'TotLot':>8} {'PnL/Lot':>10}")
    for i, r in enumerate(results[:30], 1):
        print(f"  {i:>4} {r['label']:>35} {r['sharpe']:>8.3f} ${r['total_pnl']:>11,.0f} "
              f"${r['max_dd']:>9,.0f} {r['total_lot']:>8.2f} ${r['pnl_per_lot']:>9,.0f}")

    return results


# ═══════════════════════════════════════════════════════════════
# Step 3: K-Fold on top combos
# ═══════════════════════════════════════════════════════════════

def run_kfold_portfolio(h1_df, m15_df, results, top_n=30):
    print(f"\n{'='*80}")
    print(f"  [Step 3] K-Fold 6-Fold: Top {top_n} Portfolio Combos")
    print(f"{'='*80}")

    existing = load_checkpoint("kfold_portfolio.json")
    if existing:
        print(f"  [Resume] Found checkpoint", flush=True)
        return existing

    candidates = results[:top_n]
    kfold_results = []

    for ci, cand in enumerate(candidates):
        fold_sharpes = []
        for fname, start, end in KFOLD_FOLDS:
            fold_h1 = h1_df[start:end]
            fold_m15 = m15_df[start:end]
            if len(fold_h1) < 100:
                continue

            fold_pnls = generate_daily_pnl_fold(fold_h1, fold_m15, cand)
            combined = np.zeros(len(next(iter(fold_pnls.values()))))
            for name, arr in fold_pnls.items():
                lot = cand.get(name.lower().replace('_kc','').replace('l8_max','l8'), 0)
                if name == 'L8_MAX': lot = cand['l8']
                elif name == 'D1_KC': lot = cand['d1']
                elif name == 'H4_KC': lot = cand['h4']
                elif name == 'PSAR': lot = cand['psar']
                if lot > 0:
                    combined += arr * (lot / BASE_LOT)

            std = combined.std()
            sh = float(combined.mean() / std * np.sqrt(252)) if std > 0 else 0
            fold_sharpes.append(sh)

        passed = len(fold_sharpes) == 6 and all(s > 0 for s in fold_sharpes)
        kf_mean = np.mean(fold_sharpes) if fold_sharpes else 0
        kf_min = min(fold_sharpes) if fold_sharpes else 0

        kfold_results.append({
            'label': cand['label'],
            'full_sharpe': cand['sharpe'],
            'kfold_mean': round(kf_mean, 2),
            'kfold_min': round(kf_min, 2),
            'kfold_folds': [round(s, 2) for s in fold_sharpes],
            'passed': passed,
            **{k: cand[k] for k in ['l8','d1','h4','psar','total_lot','total_pnl','max_dd']},
        })

        p_str = "PASS" if passed else "FAIL"
        print(f"  [{ci+1}/{len(candidates)}] {cand['label']:>35} "
              f"Full={cand['sharpe']:.3f} KF={kf_mean:.2f} {p_str}", flush=True)

    save_checkpoint(kfold_results, "kfold_portfolio.json")
    return kfold_results


def generate_daily_pnl_fold(h1_df, m15_df, combo):
    """Generate daily PnL arrays for a K-Fold window."""
    daily = {}

    # L8_MAX
    from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS
    l8_max_kw = {
        **LIVE_PARITY_KWARGS,
        'keltner_adx_threshold': 14,
        'regime_config': {
            'low':    {'trail_act': 0.22, 'trail_dist': 0.04},
            'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
            'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
        },
        'min_entry_gap_hours': 1.0,
        'keltner_max_hold_m15': 20,
        'spread_cost': SPREAD,
        'initial_capital': 2000,
        'min_lot_size': BASE_LOT, 'max_lot_size': BASE_LOT,
    }
    data = DataBundle(m15_df, h1_df)
    result = run_variant(data, "L8_MAX", **l8_max_kw)
    l8_raw = result.get('_trades', [])
    l8_as_dicts = []
    for t in l8_raw:
        pnl = t.pnl if hasattr(t, 'pnl') else t.get('pnl', 0)
        ext = t.exit_time if hasattr(t, 'exit_time') else t.get('exit_time', '')
        l8_as_dicts.append({'pnl': pnl, 'exit_time': ext})
    daily['L8_MAX'] = trades_to_daily_pnl(l8_as_dicts)

    # D1 KC
    d1_df = h1_df.resample('D').agg({'Open':'first','High':'max','Low':'min',
                                      'Close':'last','Volume':'sum'}).dropna()
    d1_df = add_kc(d1_df, 10, 14, 2.5); d1_df = d1_df.dropna()
    daily['D1_KC'] = trades_to_daily_pnl(
        backtest_kc_trades(d1_df, adx_thresh=18, sl_atr=4.5, tp_atr=8.0,
                           trail_act_atr=0.20, trail_dist_atr=0.05, max_hold=20))

    # H4 KC
    h4_df = h1_df.resample('4h').agg({'Open':'first','High':'max','Low':'min',
                                       'Close':'last','Volume':'sum'}).dropna()
    h4_df = add_kc(h4_df, 15, 14, 2.5); h4_df = h4_df.dropna()
    daily['H4_KC'] = trades_to_daily_pnl(
        backtest_kc_trades(h4_df, adx_thresh=10, sl_atr=4.5, tp_atr=6.0,
                           trail_act_atr=0.20, trail_dist_atr=0.04, max_hold=50))

    # PSAR
    h1_psar = add_psar(h1_df.copy(), 0.01, 0.05)
    daily['PSAR'] = trades_to_daily_pnl(
        backtest_psar_trades(h1_psar, sl_atr=4.5, tp_atr=16.0,
                             trail_act_atr=0.20, trail_dist_atr=0.04, max_hold=20))

    # Align
    all_dates = set()
    for s in daily.values():
        all_dates.update(s.index.tolist())
    all_dates = sorted(all_dates)
    idx = pd.Index(all_dates)
    result_arrays = {}
    for name in daily:
        result_arrays[name] = daily[name].reindex(idx, fill_value=0.0).values
    return result_arrays


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 80)
    print("  R52: Multi-Strategy Lot Portfolio Optimization")
    print("=" * 80)

    from backtest.runner import DataBundle
    data = DataBundle.load_default()
    h1_df = data.h1_df.copy()
    m15_df = data.m15_df.copy()
    print(f"  H1: {len(h1_df)} bars | M15: {len(m15_df)} bars")

    # Step 1: daily PnL
    daily_pnls = generate_daily_pnl_all(h1_df, m15_df)

    # Step 2: lot grid
    lot_results = run_lot_grid(daily_pnls)

    # Step 3: K-Fold
    kfold_results = run_kfold_portfolio(h1_df, m15_df, lot_results, top_n=30)

    # Final summary
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  R52 COMPLETE — {elapsed/60:.1f} min")
    print(f"{'='*80}")

    n_passed = sum(1 for k in kfold_results if k.get('passed'))
    print(f"  K-Fold passed: {n_passed}/{len(kfold_results)}")
    print(f"\n  Top 10 K-Fold passed:")
    passed = [k for k in kfold_results if k.get('passed')]
    passed.sort(key=lambda x: x['kfold_mean'], reverse=True)
    for i, k in enumerate(passed[:10], 1):
        print(f"  {i:>3}. {k['label']:>35} Full={k['full_sharpe']:.3f} "
              f"KF={k['kfold_mean']:.2f} (min={k['kfold_min']:.2f}) "
              f"PnL=${k['total_pnl']:,.0f} DD=${k['max_dd']:,.0f}")

    print(f"\n  Results in: {OUTPUT_DIR}")
