#!/usr/bin/env python3
"""
R69 — R56 vs R61 Parameter Reconciliation
==========================================
Two paths that diverged PSAR & SESS_BO params need reconciliation.

EA (deployed) matches R56 exactly. R61 explored different params under Cap$37.

Path A: Re-run R61-style portfolio lot grid using R56 (EA) params + Cap$37
         -> Does R56 param set still produce good portfolios under Cap$37?
Path B: K-Fold + Walk-Forward validation of R61's new PSAR/SESS_BO params
         -> Are R61 params genuinely better, or overfit to full sample?

Parameter Diff:
  Strategy   Param      R56 (EA)   R61
  PSAR       SL_ATR     4.5        2.0
  PSAR       MaxHold    20         80
  SESS_BO    LB         4          3
  SESS_BO    SL_ATR     4.5        3.0
  SESS_BO    TP_ATR     4.0        6.0
"""
import sys, os, io, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r69_param_reconcile")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPREAD = 0.30
BASE_LOT = 0.03

PARAM_SETS = {
    'R56': {
        'PSAR':    {'sl_atr': 4.5, 'tp_atr': 16.0, 'trail_act_atr': 0.20, 'trail_dist_atr': 0.04, 'max_hold': 20},
        'SESS_BO': {'lookback_bars': 4, 'sl_atr': 4.5, 'tp_atr': 4.0, 'trail_act_atr': 0.14, 'trail_dist_atr': 0.025, 'max_hold': 20},
    },
    'R61': {
        'PSAR':    {'sl_atr': 2.0, 'tp_atr': 16.0, 'trail_act_atr': 0.20, 'trail_dist_atr': 0.04, 'max_hold': 80},
        'SESS_BO': {'lookback_bars': 3, 'sl_atr': 3.0, 'tp_atr': 6.0, 'trail_act_atr': 0.14, 'trail_dist_atr': 0.025, 'max_hold': 20},
    },
}

STRAT_NAMES = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']
STRAT_KEYS  = ['l8', 'psar', 'ts', 'sb']

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]

WF_WINDOWS = [
    {'name': 'WF1', 'train': ('2015-01-01', '2020-12-31'), 'test': ('2021-01-01', '2022-12-31')},
    {'name': 'WF2', 'train': ('2017-01-01', '2022-12-31'), 'test': ('2023-01-01', '2024-12-31')},
    {'name': 'WF3', 'train': ('2019-01-01', '2024-12-31'), 'test': ('2025-01-01', '2026-05-01')},
]

LOT_GRID = [0.00, 0.01, 0.02, 0.03]
MAX_TOTAL_LOT = 0.12

def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"

# ═══════════════════════════════════════════════════════════════
# Indicators (self-contained)
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# Backtests (parameterized)
# ═══════════════════════════════════════════════════════════════

def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


def backtest_psar_trades(df_prepared, sl_atr=4.5, tp_atr=16.0,
                         trail_act_atr=0.20, trail_dist_atr=0.04, max_hold=20,
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


def backtest_session_trades(h1_df, session="peak_12_14", lookback_bars=4,
                            sl_atr=4.5, tp_atr=4.0, trail_act_atr=0.14,
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


def calc_stats(daily_pnl_series, label=""):
    if daily_pnl_series.empty or daily_pnl_series.sum() == 0:
        return {'label': label, 'sharpe': 0, 'total_pnl': 0, 'max_dd': 0, 'n_days': 0}
    arr = daily_pnl_series.values
    eq = np.cumsum(arr)
    dd = (np.maximum.accumulate(eq) - eq).max()
    sh = float(arr.mean() / arr.std() * np.sqrt(252)) if arr.std() > 0 else 0
    return {'label': label, 'sharpe': round(sh, 3), 'total_pnl': round(float(arr.sum()), 2),
            'max_dd': round(float(dd), 2), 'n_days': len(arr)}


def save_text(filename, text):
    path = OUTPUT_DIR / filename
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"  [Saved] {path}", flush=True)


def save_json(data, filename):
    path = OUTPUT_DIR / filename
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, default=str, indent=2)
    print(f"  [Saved] {path}", flush=True)


# ═══════════════════════════════════════════════════════════════
# Core: generate daily PnL for a given param set label
# ═══════════════════════════════════════════════════════════════

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


def generate_daily_pnls(h1_df, m15_df, param_label='R56'):
    """Generate daily PnL for 4 strategies using specified param set."""
    p = PARAM_SETS[param_label]
    daily = {}

    l8_raw = _run_l8_max(m15_df, h1_df, lot=BASE_LOT, maxloss_cap=37, spread=SPREAD)
    daily['L8_MAX'] = trades_to_daily_pnl(l8_raw)

    h1_psar = add_psar(h1_df.copy(), 0.01, 0.05)
    daily['PSAR'] = trades_to_daily_pnl(
        backtest_psar_trades(h1_psar, spread=SPREAD, lot=BASE_LOT, **p['PSAR']))

    daily['TSMOM'] = trades_to_daily_pnl(
        backtest_tsmom_trades(h1_df, fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
                              trail_act_atr=0.14, trail_dist_atr=0.025, max_hold=20,
                              spread=SPREAD, lot=BASE_LOT))

    daily['SESS_BO'] = trades_to_daily_pnl(
        backtest_session_trades(h1_df, session="peak_12_14", spread=SPREAD,
                                lot=BASE_LOT, **p['SESS_BO']))

    all_dates = set()
    for s in daily.values():
        all_dates.update(s.index.tolist())
    all_dates = sorted(all_dates)
    idx = pd.Index(all_dates)
    return {name: daily[name].reindex(idx, fill_value=0.0) for name in daily}


# ═══════════════════════════════════════════════════════════════
# PATH A: R56 params + Cap$37 portfolio lot grid + K-Fold
# ═══════════════════════════════════════════════════════════════

def path_a_lot_grid(daily_pnls):
    """Brute-force lot grid on R56-param daily PnL."""
    base = {name: daily_pnls[name].values for name in STRAT_NAMES}
    combos = list(product(*([LOT_GRID] * len(STRAT_NAMES))))
    combos = [c for c in combos if sum(c) > 0 and sum(c) <= MAX_TOTAL_LOT]

    results = []
    for lots in combos:
        combined = np.zeros_like(base['L8_MAX'], dtype=float)
        for name, lot_val in zip(STRAT_NAMES, lots):
            if lot_val > 0:
                combined += base[name] * (lot_val / BASE_LOT)
        eq = np.cumsum(combined)
        dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 0 else 0
        total_pnl = float(combined.sum())
        std = combined.std()
        sharpe = float(combined.mean() / std * np.sqrt(252)) if std > 0 else 0
        total_lot = sum(lots)
        label = f"L8={lots[0]}_PS={lots[1]}_TS={lots[2]}_SB={lots[3]}"
        results.append({
            'label': label,
            'l8': lots[0], 'psar': lots[1], 'ts': lots[2], 'sb': lots[3],
            'total_lot': round(total_lot, 2),
            'sharpe': round(sharpe, 3), 'total_pnl': round(total_pnl, 2),
            'max_dd': round(dd, 2),
        })
    results.sort(key=lambda x: x['sharpe'], reverse=True)
    return results


def path_a_kfold(h1_df, m15_df, top_combos, param_label='R56', top_n=10):
    """K-Fold validation for top lot combos."""
    candidates = top_combos[:top_n]
    kfold_results = []

    for ci, cand in enumerate(candidates):
        fold_sharpes = []
        for fname, start, end in FOLDS:
            fold_h1 = h1_df[start:end]
            fold_m15 = m15_df[start:end]
            if len(fold_h1) < 100:
                continue
            fold_pnls = generate_daily_pnls(fold_h1, fold_m15, param_label)
            base = {name: fold_pnls[name].values for name in STRAT_NAMES}
            n_days = len(next(iter(base.values())))
            if n_days == 0:
                fold_sharpes.append(0); continue
            combined = np.zeros(n_days)
            for name, key in zip(STRAT_NAMES, STRAT_KEYS):
                lot = cand.get(key, 0)
                if lot > 0:
                    combined += base[name] * (lot / BASE_LOT)
            std = combined.std()
            sh = float(combined.mean() / std * np.sqrt(252)) if std > 0 else 0
            fold_sharpes.append(sh)

        passed = len(fold_sharpes) == len(FOLDS) and all(s > 0 for s in fold_sharpes)
        kf_mean = np.mean(fold_sharpes) if fold_sharpes else 0
        kf_min = min(fold_sharpes) if fold_sharpes else 0

        kfold_results.append({
            'label': cand['label'], 'full_sharpe': cand['sharpe'],
            'kfold_mean': round(float(kf_mean), 3), 'kfold_min': round(float(kf_min), 3),
            'kfold_folds': [round(s, 3) for s in fold_sharpes],
            'passed': passed,
            **{k: cand[k] for k in STRAT_KEYS + ['total_lot', 'total_pnl', 'max_dd']},
        })
        p_str = "PASS" if passed else "FAIL"
        print(f"    [{ci+1}/{len(candidates)}] {cand['label']:>40} "
              f"Full={cand['sharpe']:.3f} KF={kf_mean:.3f} {p_str}", flush=True)

    return kfold_results


# ═══════════════════════════════════════════════════════════════
# PATH B: Per-strategy head-to-head K-Fold + Walk-Forward
# ═══════════════════════════════════════════════════════════════

def path_b_per_strategy_kfold(h1_df, m15_df):
    """K-Fold PSAR & SESS_BO individually: R56 vs R61 params."""
    results = {}

    for strat in ['PSAR', 'SESS_BO']:
        results[strat] = {}
        for plabel in ['R56', 'R61']:
            p = PARAM_SETS[plabel][strat]
            fold_sharpes = []
            fold_pnls_total = []
            fold_trades = []
            for fname, start, end in FOLDS:
                fold_h1 = h1_df[start:end]
                if len(fold_h1) < 100:
                    continue

                if strat == 'PSAR':
                    h1_psar = add_psar(fold_h1.copy(), 0.01, 0.05)
                    trades = backtest_psar_trades(h1_psar, spread=SPREAD, lot=BASE_LOT, **p)
                else:
                    trades = backtest_session_trades(fold_h1, session="peak_12_14",
                                                     spread=SPREAD, lot=BASE_LOT, **p)

                daily = trades_to_daily_pnl(trades)
                st = calc_stats(daily, fname)
                fold_sharpes.append(st['sharpe'])
                fold_pnls_total.append(st['total_pnl'])
                fold_trades.append(len(trades))

            results[strat][plabel] = {
                'fold_sharpes': fold_sharpes,
                'fold_pnls': fold_pnls_total,
                'fold_trades': fold_trades,
                'mean_sharpe': round(float(np.mean(fold_sharpes)), 3) if fold_sharpes else 0,
                'min_sharpe': round(float(min(fold_sharpes)), 3) if fold_sharpes else 0,
                'all_positive': all(s > 0 for s in fold_sharpes),
                'total_pnl': round(float(sum(fold_pnls_total)), 2),
                'total_trades': sum(fold_trades),
                'params': p,
            }

    return results


def path_b_walk_forward(h1_df, m15_df):
    """Walk-forward for PSAR & SESS_BO: compare R56 vs R61 on frozen OOS."""
    results = {}

    for strat in ['PSAR', 'SESS_BO']:
        results[strat] = {}
        for plabel in ['R56', 'R61']:
            p = PARAM_SETS[plabel][strat]
            wf_results = []
            for wf in WF_WINDOWS:
                test_h1 = h1_df[wf['test'][0]:wf['test'][1]]
                if len(test_h1) < 50:
                    continue

                if strat == 'PSAR':
                    h1_psar = add_psar(test_h1.copy(), 0.01, 0.05)
                    trades = backtest_psar_trades(h1_psar, spread=SPREAD, lot=BASE_LOT, **p)
                else:
                    trades = backtest_session_trades(test_h1, session="peak_12_14",
                                                     spread=SPREAD, lot=BASE_LOT, **p)

                daily = trades_to_daily_pnl(trades)
                st = calc_stats(daily, wf['name'])
                wf_results.append({
                    'window': wf['name'],
                    'test_period': f"{wf['test'][0]} ~ {wf['test'][1]}",
                    **st,
                    'n_trades': len(trades),
                })

            results[strat][plabel] = wf_results

    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80)
    print("  R69: R56 vs R61 Parameter Reconciliation")
    print("=" * 80)

    from backtest.runner import DataBundle
    data = DataBundle.load_default()
    h1_df = data.h1_df.copy()
    m15_df = data.m15_df.copy()
    print(f"  H1: {len(h1_df)} bars | M15: {len(m15_df)} bars\n")

    all_results = {}

    # ─── PATH A: R56 params under Cap$37, lot grid + K-Fold ───
    print("=" * 80)
    print("  PATH A: R56 (EA) params + Cap$37 — Portfolio Lot Grid + K-Fold")
    print("=" * 80)

    print("  Generating full-sample daily PnL (R56 params)...", flush=True)
    r56_daily = generate_daily_pnls(h1_df, m15_df, 'R56')

    for name in STRAT_NAMES:
        st = calc_stats(r56_daily[name], name)
        print(f"    {name:>10}: Sharpe={st['sharpe']:.3f}  PnL={fmt(st['total_pnl'])}  "
              f"DD={fmt(st['max_dd'])}  Days={st['n_days']}", flush=True)

    print("\n  Lot grid search...", flush=True)
    r56_grid = path_a_lot_grid(r56_daily)
    print(f"  Total combos: {len(r56_grid)}")
    print(f"\n  Top 10 R56-param portfolios:")
    for i, r in enumerate(r56_grid[:10], 1):
        print(f"    {i:>3}. {r['label']:>40} Sharpe={r['sharpe']:.3f} "
              f"PnL={fmt(r['total_pnl'])} DD={fmt(r['max_dd'])}", flush=True)

    print("\n  K-Fold validation (R56 params)...", flush=True)
    r56_kfold = path_a_kfold(h1_df, m15_df, r56_grid, 'R56', top_n=10)
    all_results['path_a_r56'] = {'grid_top10': r56_grid[:10], 'kfold': r56_kfold}

    # Also run lot grid with R61 params for direct comparison
    print("\n  Generating full-sample daily PnL (R61 params)...", flush=True)
    r61_daily = generate_daily_pnls(h1_df, m15_df, 'R61')

    for name in STRAT_NAMES:
        st = calc_stats(r61_daily[name], name)
        print(f"    {name:>10}: Sharpe={st['sharpe']:.3f}  PnL={fmt(st['total_pnl'])}  "
              f"DD={fmt(st['max_dd'])}  Days={st['n_days']}", flush=True)

    print("\n  Lot grid search (R61 params)...", flush=True)
    r61_grid = path_a_lot_grid(r61_daily)
    print(f"\n  Top 10 R61-param portfolios:")
    for i, r in enumerate(r61_grid[:10], 1):
        print(f"    {i:>3}. {r['label']:>40} Sharpe={r['sharpe']:.3f} "
              f"PnL={fmt(r['total_pnl'])} DD={fmt(r['max_dd'])}", flush=True)

    print("\n  K-Fold validation (R61 params)...", flush=True)
    r61_kfold = path_a_kfold(h1_df, m15_df, r61_grid, 'R61', top_n=10)
    all_results['path_a_r61'] = {'grid_top10': r61_grid[:10], 'kfold': r61_kfold}

    # ─── PATH B: Per-strategy K-Fold + Walk-Forward ───
    print(f"\n{'='*80}")
    print("  PATH B: Per-Strategy K-Fold + Walk-Forward (PSAR & SESS_BO)")
    print(f"{'='*80}")

    print("\n  K-Fold per strategy...", flush=True)
    kfold_compare = path_b_per_strategy_kfold(h1_df, m15_df)
    all_results['path_b_kfold'] = kfold_compare

    for strat in ['PSAR', 'SESS_BO']:
        print(f"\n  {strat}:")
        for plabel in ['R56', 'R61']:
            r = kfold_compare[strat][plabel]
            folds_str = ", ".join(f"{s:.2f}" for s in r['fold_sharpes'])
            print(f"    {plabel}: Mean={r['mean_sharpe']:.3f} Min={r['min_sharpe']:.3f} "
                  f"AllPos={'Y' if r['all_positive'] else 'N'} "
                  f"PnL={fmt(r['total_pnl'])} Trades={r['total_trades']} [{folds_str}]", flush=True)

    print("\n  Walk-Forward per strategy...", flush=True)
    wf_compare = path_b_walk_forward(h1_df, m15_df)
    all_results['path_b_walk_forward'] = wf_compare

    for strat in ['PSAR', 'SESS_BO']:
        print(f"\n  {strat} Walk-Forward OOS:")
        for plabel in ['R56', 'R61']:
            wfs = wf_compare[strat][plabel]
            for w in wfs:
                print(f"    {plabel} {w['window']}: Sharpe={w['sharpe']:.3f} "
                      f"PnL={fmt(w['total_pnl'])} DD={fmt(w['max_dd'])} "
                      f"Trades={w['n_trades']} ({w['test_period']})", flush=True)

    # ─── FINAL SUMMARY ───
    elapsed = time.time() - t0
    lines = [
        "R69 Parameter Reconciliation — Final Summary",
        "=" * 80,
        f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)",
        f"Spread: ${SPREAD}  |  Base lot: {BASE_LOT}  |  MaxLoss Cap: $37",
        "",
        "Parameter Diff:",
        "  Strategy   Param      R56 (EA)   R61",
        "  PSAR       SL_ATR     4.5        2.0",
        "  PSAR       MaxHold    20         80",
        "  SESS_BO    LB         4          3",
        "  SESS_BO    SL_ATR     4.5        3.0",
        "  SESS_BO    TP_ATR     4.0        6.0",
        "",
        "═══ PATH A: Portfolio-Level Comparison ═══",
        "",
    ]

    r56_best_kf = [k for k in r56_kfold if k['passed']]
    r56_best_kf.sort(key=lambda x: x['kfold_mean'], reverse=True)
    r61_best_kf = [k for k in r61_kfold if k['passed']]
    r61_best_kf.sort(key=lambda x: x['kfold_mean'], reverse=True)

    lines.append(f"  R56 params: {len(r56_best_kf)}/{len(r56_kfold)} passed K-Fold")
    if r56_best_kf:
        b = r56_best_kf[0]
        lines.append(f"    Best: {b['label']}  Full={b['full_sharpe']:.3f}  KF={b['kfold_mean']:.3f}")
    lines.append(f"  R61 params: {len(r61_best_kf)}/{len(r61_kfold)} passed K-Fold")
    if r61_best_kf:
        b = r61_best_kf[0]
        lines.append(f"    Best: {b['label']}  Full={b['full_sharpe']:.3f}  KF={b['kfold_mean']:.3f}")

    lines.extend(["", "═══ PATH B: Per-Strategy Head-to-Head ═══", ""])

    for strat in ['PSAR', 'SESS_BO']:
        r56_r = kfold_compare[strat]['R56']
        r61_r = kfold_compare[strat]['R61']
        lines.append(f"  {strat} K-Fold:")
        lines.append(f"    R56: Mean Sharpe = {r56_r['mean_sharpe']:.3f}  "
                      f"Min = {r56_r['min_sharpe']:.3f}  AllPos = {'Y' if r56_r['all_positive'] else 'N'}  "
                      f"Trades = {r56_r['total_trades']}")
        lines.append(f"    R61: Mean Sharpe = {r61_r['mean_sharpe']:.3f}  "
                      f"Min = {r61_r['min_sharpe']:.3f}  AllPos = {'Y' if r61_r['all_positive'] else 'N'}  "
                      f"Trades = {r61_r['total_trades']}")

        winner = 'R61' if r61_r['mean_sharpe'] > r56_r['mean_sharpe'] else 'R56'
        margin = abs(r61_r['mean_sharpe'] - r56_r['mean_sharpe'])
        if margin < 0.2:
            lines.append(f"    -> Marginal difference ({margin:.3f}), keep R56 (EA) for stability")
        else:
            lines.append(f"    -> {winner} is meaningfully better by {margin:.3f} Sharpe")

        lines.append(f"\n  {strat} Walk-Forward OOS:")
        for plabel in ['R56', 'R61']:
            wfs = wf_compare[strat][plabel]
            avg_sh = np.mean([w['sharpe'] for w in wfs]) if wfs else 0
            win_strs = [f"{w['window']}={w['sharpe']:.2f}" for w in wfs]
            lines.append(f"    {plabel}: Avg OOS Sharpe = {avg_sh:.3f}  "
                              f"Windows = {', '.join(win_strs)}")
        lines.append("")

    # Verdict
    lines.extend(["═══ VERDICT ═══", ""])
    for strat in ['PSAR', 'SESS_BO']:
        r56_kf_mean = kfold_compare[strat]['R56']['mean_sharpe']
        r61_kf_mean = kfold_compare[strat]['R61']['mean_sharpe']
        r56_wf_avg = np.mean([w['sharpe'] for w in wf_compare[strat]['R56']]) if wf_compare[strat]['R56'] else 0
        r61_wf_avg = np.mean([w['sharpe'] for w in wf_compare[strat]['R61']]) if wf_compare[strat]['R61'] else 0

        r61_better_kf = r61_kf_mean > r56_kf_mean + 0.2
        r61_better_wf = r61_wf_avg > r56_wf_avg + 0.2

        if r61_better_kf and r61_better_wf:
            lines.append(f"  {strat}: R61 params are BETTER in both K-Fold and Walk-Forward. "
                          f"Consider updating EA.")
        elif r61_better_kf or r61_better_wf:
            lines.append(f"  {strat}: R61 params show MIXED results. Keep R56 (EA) as safer choice.")
        else:
            lines.append(f"  {strat}: R56 (EA) params are EQUAL or BETTER. No change needed.")

    summary = "\n".join(lines)
    print(f"\n{summary}", flush=True)
    save_text("r69_summary.txt", summary)
    save_json(all_results, "r69_results.json")

    print(f"\n{'='*80}")
    print(f"  R69 Complete — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
