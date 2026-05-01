#!/usr/bin/env python3
"""
R63 — Walk-Forward Nested Validation (Anti-Overfitting)
========================================================
True out-of-sample test: optimize lot allocation on TRAIN only,
freeze params, then evaluate on unseen TEST period.

3 Walk-Forward Windows:
  WF1: Train 2015-2020, Test 2021-2022
  WF2: Train 2017-2022, Test 2023-2024
  WF3: Train 2019-2024, Test 2025-2026
"""
import sys, os, io, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r63_walk_forward")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPREAD = 0.30
BASE_LOT = 0.03
STRAT_NAMES = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']
STRAT_KEYS  = ['l8', 'psar', 'ts', 'sb']
LOT_GRID = [0.00, 0.01, 0.02, 0.03]
MAX_TOTAL_LOT = 0.15

WF_WINDOWS = [
    {'name': 'WF1', 'train': ('2015-01-01','2020-12-31'), 'test': ('2021-01-01','2022-12-31')},
    {'name': 'WF2', 'train': ('2017-01-01','2022-12-31'), 'test': ('2023-01-01','2024-12-31')},
    {'name': 'WF3', 'train': ('2019-01-01','2024-12-31'), 'test': ('2025-01-01','2026-05-01')},
]

def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"

# ═══════════════════════════════════════════════════════════════
# Indicator & backtest helpers (from R61, self-contained)
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

def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()

def add_psar(df, af_start=0.02, af_max=0.20):
    df = df.copy()
    n = len(df); psar = np.zeros(n); direction = np.ones(n)
    af = af_start; ep = df['High'].iloc[0]; psar[0] = df['Low'].iloc[0]
    for i in range(1, n):
        prev_psar = psar[i-1]
        if direction[i-1] == 1:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = min(psar[i], df['Low'].iloc[i-1], df['Low'].iloc[max(0, i-2)])
            if df['Low'].iloc[i] < psar[i]:
                direction[i] = -1; psar[i] = ep; ep = df['Low'].iloc[i]; af = af_start
            else:
                direction[i] = 1
                if df['High'].iloc[i] > ep:
                    ep = df['High'].iloc[i]; af = min(af + af_start, af_max)
        else:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = max(psar[i], df['High'].iloc[i-1], df['High'].iloc[max(0, i-2)])
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

def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}

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

def backtest_tsmom_trades(h1_df, fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
                          trail_act_atr=0.14, trail_dist_atr=0.025, max_hold=20,
                          spread=SPREAD, lot=BASE_LOT):
    df = h1_df.copy()
    if 'ATR' not in df.columns:
        df['ATR'] = compute_atr(df)
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(close)
    weights = [(fast, 0.5), (slow, 0.5)]
    max_lb = max(lb for lb, _ in weights)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        for lb, w in weights:
            if i >= lb:
                s += w * np.sign(close[i] / close[i - lb] - 1.0)
        score[i] = s
    trades = []; pos = None; last_exit = -999
    for i in range(max_lb + 1, n):
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
                ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
                    ts_p = h - td
                    if lo_v <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos['entry'] - spread) * lot * 100)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
                    ts_p = lo_v + td
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * 100)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if not exited and not np.isnan(score[i]):
                if pos['dir'] == 'BUY' and score[i] < 0:
                    trades.append(_mk(pos, c, times[i], "Reversal", i, pnl_c)); exited = True
                elif pos['dir'] == 'SELL' and score[i] > 0:
                    trades.append(_mk(pos, c, times[i], "Reversal", i, pnl_c)); exited = True
            if exited: pos = None; last_exit = i; continue
        if pos is not None or i - last_exit < 2: continue
        if np.isnan(score[i]) or np.isnan(cur_atr) or cur_atr < 0.1: continue
        if score[i] > 0 and score[i - 1] <= 0:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif score[i] < 0 and score[i - 1] >= 0:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
    return trades

def backtest_session_trades(h1_df, session="peak_12_14", lookback_bars=3,
                            sl_atr=3.0, tp_atr=6.0, trail_act_atr=0.14,
                            trail_dist_atr=0.025, max_hold=20,
                            spread=SPREAD, lot=BASE_LOT):
    SESSION_DEFS = {"asian": (0,7), "london": (8,11), "ny_peak": (12,16),
                    "late": (17,23), "peak_12_14": (12,14)}
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour
    times = df.index; n = len(df); last_exit = -999
    sess_start, sess_end = SESSION_DEFS[session]
    for i in range(lookback_bars, n):
        c = close[i]; h = high[i]; lo_v = low[i]; cur_atr = atr[i]; cur_hour = hours[i]
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
        if cur_hour != sess_start: continue
        if i > 0 and hours[i-1] == sess_start: continue
        range_high = max(high[i - lookback_bars:i])
        range_low  = min(low[i - lookback_bars:i])
        if c > range_high:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif c < range_low:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
    return trades

def trades_to_daily_pnl(trades):
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    if not daily:
        return pd.Series(dtype=float)
    return pd.Series(daily).sort_index()

def _run_l8_max(m15_df, h1_df, lot=BASE_LOT, maxloss_cap=37, spread=SPREAD):
    from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS
    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': maxloss_cap,
          'spread_cost': spread, 'initial_capital': 2000,
          'min_lot_size': lot, 'max_lot_size': lot}
    data = DataBundle(m15_df, h1_df)
    result = run_variant(data, "L8_MAX", verbose=False, **kw)
    raw = result.get('_trades', [])
    trades = []
    for t in raw:
        pnl = t.pnl if hasattr(t, 'pnl') else t.get('pnl', 0)
        ext = t.exit_time if hasattr(t, 'exit_time') else t.get('exit_time', '')
        trades.append({'pnl': pnl, 'exit_time': ext})
    return trades

def generate_daily_pnls(h1_df, m15_df, spread=SPREAD):
    daily = {}
    l8_raw = _run_l8_max(m15_df, h1_df, lot=BASE_LOT, maxloss_cap=37, spread=spread)
    daily['L8_MAX'] = trades_to_daily_pnl(l8_raw)
    h1_psar = add_psar(h1_df.copy(), 0.01, 0.05)
    daily['PSAR'] = trades_to_daily_pnl(
        backtest_psar_trades(h1_psar, sl_atr=2.0, tp_atr=16.0,
                             trail_act_atr=0.20, trail_dist_atr=0.04, max_hold=80, spread=spread))
    daily['TSMOM'] = trades_to_daily_pnl(
        backtest_tsmom_trades(h1_df, fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
                              trail_act_atr=0.14, trail_dist_atr=0.025, max_hold=20, spread=spread))
    daily['SESS_BO'] = trades_to_daily_pnl(
        backtest_session_trades(h1_df, session="peak_12_14", lookback_bars=3,
                                 sl_atr=3.0, tp_atr=6.0, trail_act_atr=0.14,
                                 trail_dist_atr=0.025, max_hold=20, spread=spread))
    all_dates = set()
    for s in daily.values():
        all_dates.update(s.index.tolist())
    all_dates = sorted(all_dates)
    idx = pd.Index(all_dates)
    return {name: daily[name].reindex(idx, fill_value=0.0) for name in daily}

def lot_grid_search(daily_pnls):
    base = {name: daily_pnls[name].values for name in STRAT_NAMES}
    combos = list(product(*([LOT_GRID] * len(STRAT_NAMES))))
    combos = [c for c in combos if sum(c) > 0 and sum(c) <= MAX_TOTAL_LOT]
    best = None
    for lots in combos:
        combined = np.zeros_like(base['L8_MAX'], dtype=float)
        for name, lot_val in zip(STRAT_NAMES, lots):
            if lot_val > 0:
                combined += base[name] * (lot_val / BASE_LOT)
        std = combined.std()
        sharpe = float(combined.mean() / std * np.sqrt(252)) if std > 0 else 0
        if best is None or sharpe > best['sharpe']:
            best = {'sharpe': sharpe, 'lots': dict(zip(STRAT_KEYS, lots)),
                    'total_pnl': float(combined.sum()),
                    'total_lot': sum(lots)}
    return best

def portfolio_sharpe(daily_pnls, lots):
    base = {name: daily_pnls[name].values for name in STRAT_NAMES}
    combined = np.zeros_like(base['L8_MAX'], dtype=float)
    for name, key in zip(STRAT_NAMES, STRAT_KEYS):
        lot_val = lots.get(key, 0)
        if lot_val > 0:
            combined += base[name] * (lot_val / BASE_LOT)
    eq = np.cumsum(combined)
    dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 0 else 0
    std = combined.std()
    sharpe = float(combined.mean() / std * np.sqrt(252)) if std > 0 else 0
    return {'sharpe': round(sharpe, 4), 'total_pnl': round(float(combined.sum()), 2),
            'max_dd': round(dd, 2), 'n_days': len(combined)}


# ═══════════════════════════════════════════════════════════════
# Main Walk-Forward
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80)
    print("  R63: Walk-Forward Nested Validation")
    print("=" * 80)

    from backtest.runner import DataBundle
    data = DataBundle.load_default()
    h1_full = data.h1_df.copy()
    m15_full = data.m15_df.copy()
    print(f"  H1: {len(h1_full)} bars | M15: {len(m15_full)} bars", flush=True)

    results = []
    all_test_pnl = []

    for wf in WF_WINDOWS:
        wf_name = wf['name']
        train_s, train_e = wf['train']
        test_s, test_e = wf['test']
        print(f"\n{'='*60}")
        print(f"  {wf_name}: Train {train_s}~{train_e} | Test {test_s}~{test_e}")
        print(f"{'='*60}", flush=True)

        # Train phase
        print(f"  [TRAIN] Generating daily PnL...", flush=True)
        h1_train = h1_full[train_s:train_e]
        m15_train = m15_full[train_s:train_e]
        train_pnls = generate_daily_pnls(h1_train, m15_train)
        print(f"  [TRAIN] Lot grid search...", flush=True)
        best_train = lot_grid_search(train_pnls)
        train_stats = portfolio_sharpe(train_pnls, best_train['lots'])
        print(f"  [TRAIN] Best: {best_train['lots']} Sharpe={train_stats['sharpe']}", flush=True)

        # Test phase (FROZEN lots)
        print(f"  [TEST] Generating daily PnL...", flush=True)
        h1_test = h1_full[test_s:test_e]
        m15_test = m15_full[test_s:test_e]
        test_pnls = generate_daily_pnls(h1_test, m15_test)
        test_stats = portfolio_sharpe(test_pnls, best_train['lots'])
        print(f"  [TEST] Frozen lots -> Sharpe={test_stats['sharpe']}", flush=True)

        # Collect test PnL for combined OOS
        base_test = {name: test_pnls[name].values for name in STRAT_NAMES}
        combined_test = np.zeros_like(base_test['L8_MAX'], dtype=float)
        for name, key in zip(STRAT_NAMES, STRAT_KEYS):
            lot_val = best_train['lots'].get(key, 0)
            if lot_val > 0:
                combined_test += base_test[name] * (lot_val / BASE_LOT)
        all_test_pnl.extend(combined_test.tolist())

        decay = (train_stats['sharpe'] - test_stats['sharpe']) / train_stats['sharpe'] * 100 \
            if train_stats['sharpe'] > 0 else 0

        wf_result = {
            'window': wf_name,
            'train_period': f"{train_s} ~ {train_e}",
            'test_period': f"{test_s} ~ {test_e}",
            'train_sharpe': train_stats['sharpe'],
            'test_sharpe': test_stats['sharpe'],
            'sharpe_decay_pct': round(decay, 1),
            'chosen_lots': best_train['lots'],
            'total_lot': best_train['total_lot'],
            'test_pnl': test_stats['total_pnl'],
            'test_maxdd': test_stats['max_dd'],
            'test_days': test_stats['n_days'],
        }
        results.append(wf_result)
        print(f"  Sharpe Decay: {decay:.1f}%", flush=True)

    # Combined OOS
    all_arr = np.array(all_test_pnl)
    combined_sharpe = float(all_arr.mean() / all_arr.std() * np.sqrt(252)) if all_arr.std() > 0 else 0
    combined_pnl = float(all_arr.sum())
    eq = np.cumsum(all_arr)
    combined_dd = float((np.maximum.accumulate(eq) - eq).max())

    elapsed = time.time() - t0

    # Summary
    lines = [
        "R63 Walk-Forward Nested Validation — Summary",
        "=" * 70,
        f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)\n",
        f"{'Window':<6} {'Train Period':<22} {'Test Period':<22} {'Train Sh':>9} {'Test Sh':>9} {'Decay%':>7} {'Test PnL':>10} {'Test DD':>10} {'Lots':>30}",
        "-" * 130,
    ]
    for r in results:
        lots_str = " ".join(f"{k}={v}" for k,v in r['chosen_lots'].items() if v > 0)
        lines.append(f"{r['window']:<6} {r['train_period']:<22} {r['test_period']:<22} "
                      f"{r['train_sharpe']:>9.2f} {r['test_sharpe']:>9.2f} {r['sharpe_decay_pct']:>6.1f}% "
                      f"{fmt(r['test_pnl']):>10} {fmt(r['test_maxdd']):>10} {lots_str:>30}")

    lines.append(f"\n{'='*70}")
    lines.append(f"Combined OOS Sharpe (all test periods): {combined_sharpe:.2f}")
    lines.append(f"Combined OOS PnL:    {fmt(combined_pnl)}")
    lines.append(f"Combined OOS MaxDD:  {fmt(combined_dd)}")
    lines.append(f"Combined OOS Days:   {len(all_arr)}")

    avg_decay = np.mean([r['sharpe_decay_pct'] for r in results])
    lines.append(f"\nAverage Sharpe Decay: {avg_decay:.1f}%")

    if combined_sharpe > 3.0:
        lines.append("\nVERDICT: STRONG — OOS Sharpe > 3.0, strategy is robust")
    elif combined_sharpe > 1.5:
        lines.append("\nVERDICT: ACCEPTABLE — OOS Sharpe > 1.5, moderate confidence")
    elif combined_sharpe > 0:
        lines.append("\nVERDICT: WEAK — OOS Sharpe > 0 but < 1.5, significant overfitting risk")
    else:
        lines.append("\nVERDICT: FAIL — OOS Sharpe <= 0, strategy is likely overfit")

    summary = "\n".join(lines)
    print(f"\n{summary}", flush=True)

    with open(OUTPUT_DIR / "r63_summary.txt", 'w', encoding='utf-8') as f:
        f.write(summary)
    with open(OUTPUT_DIR / "r63_results.json", 'w', encoding='utf-8') as f:
        json.dump({'windows': results,
                   'combined_oos_sharpe': round(combined_sharpe, 4),
                   'combined_oos_pnl': round(combined_pnl, 2),
                   'combined_oos_maxdd': round(combined_dd, 2),
                   'avg_sharpe_decay_pct': round(avg_decay, 1),
                   'elapsed_s': round(elapsed, 1)}, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  R63 Complete — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
