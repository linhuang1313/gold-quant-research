#!/usr/bin/env python3
"""
R82 — Dual PBO Validation: Perturbation vs CSCV (Bailey et al. 2017)
=====================================================================
Re-run Stage 4 for all 4 strategies using the upgraded dual PBO framework:

  PBO-Perturb : random ±20% param perturbation → parameter stability
  PBO-CSCV    : systematic grid search + CSCV → selection bias (Bailey 2017)

Each strategy runs its full grid of backtests, builds a T×N daily PnL matrix,
and computes PBO via Combinatorially Symmetric Cross-Validation.

Strategies: PSAR, SESS_BO, TSMOM, MACD
Estimated runtime: ~5-8 minutes (grid backtests dominate).
"""
import sys, os, io, time, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r82_dual_pbo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPREAD = 0.30
REALISTIC_SPREAD = 0.88
LOT = 0.03
PV = 100


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

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


def _run_exit_logic(pos, i, h, lo_v, c, spread, lot, atr_entry,
                    sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold, times):
    held = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_h = (h - pos['entry'] - spread) * lot * PV
        pnl_l = (lo_v - pos['entry'] - spread) * lot * PV
        pnl_c = (c - pos['entry'] - spread) * lot * PV
    else:
        pnl_h = (pos['entry'] - lo_v - spread) * lot * PV
        pnl_l = (pos['entry'] - h - spread) * lot * PV
        pnl_c = (pos['entry'] - c - spread) * lot * PV
    tp_val = tp_atr * atr_entry * lot * PV
    sl_val = sl_atr * atr_entry * lot * PV
    if pnl_h >= tp_val:
        return _mk(pos, c, times[i], "TP", i, tp_val)
    if pnl_l <= -sl_val:
        return _mk(pos, c, times[i], "SL", i, -sl_val)
    ad = trail_act_atr * atr_entry; td = trail_dist_atr * atr_entry
    if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
        ts_p = h - td
        if lo_v <= ts_p:
            return _mk(pos, c, times[i], "Trail", i,
                        (ts_p - pos['entry'] - spread) * lot * PV)
    elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
        ts_p = lo_v + td
        if h >= ts_p:
            return _mk(pos, c, times[i], "Trail", i,
                        (pos['entry'] - ts_p - spread) * lot * PV)
    if held >= max_hold:
        return _mk(pos, c, times[i], "Timeout", i, pnl_c)
    return None


# ═══════════════════════════════════════════════════════════════
# PSAR
# ═══════════════════════════════════════════════════════════════

def add_psar(df, af_start=0.01, af_max=0.05):
    df = df.copy(); n = len(df)
    psar = np.zeros(n); direction = np.ones(n)
    af = af_start; ep = df['High'].iloc[0]; psar[0] = df['Low'].iloc[0]
    for i in range(1, n):
        prev = psar[i-1]
        if direction[i-1] == 1:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], df['Low'].iloc[i-1], df['Low'].iloc[max(0,i-2)])
            if df['Low'].iloc[i] < psar[i]:
                direction[i] = -1; psar[i] = ep; ep = df['Low'].iloc[i]; af = af_start
            else:
                direction[i] = 1
                if df['High'].iloc[i] > ep:
                    ep = df['High'].iloc[i]; af = min(af+af_start, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], df['High'].iloc[i-1], df['High'].iloc[max(0,i-2)])
            if df['High'].iloc[i] > psar[i]:
                direction[i] = 1; psar[i] = ep; ep = df['High'].iloc[i]; af = af_start
            else:
                direction[i] = -1
                if df['Low'].iloc[i] < ep:
                    ep = df['Low'].iloc[i]; af = min(af+af_start, af_max)
    df['PSAR_dir'] = direction
    df['ATR'] = compute_atr(df)
    return df


def backtest_psar(h1_df, spread=SPREAD, lot=LOT,
                  sl_atr=4.5, tp_atr=16.0, trail_act_atr=0.20,
                  trail_dist_atr=0.04, max_hold=20):
    df = add_psar(h1_df).dropna(subset=['PSAR_dir', 'ATR'])
    trades = []; pos = None
    c_arr = df['Close'].values; h_arr = df['High'].values; l_arr = df['Low'].values
    psar_dir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df); last_exit = -999
    for i in range(1, n):
        c = c_arr[i]; h = h_arr[i]; lo = l_arr[i]; cur_atr = atr[i]
        if pos is not None:
            result = _run_exit_logic(pos, i, h, lo, c, spread, lot, pos['atr'],
                                     sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold, times)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        if psar_dir[i-1] == -1 and psar_dir[i] == 1:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif psar_dir[i-1] == 1 and psar_dir[i] == -1:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
    return trades


def psar_perturb_fn(h1_df, spread, lot, rng):
    def p(base, pct=0.20): return base * (1 + rng.uniform(-pct, pct))
    return backtest_psar(h1_df, spread, lot, sl_atr=p(4.5), tp_atr=p(16.0),
                         trail_act_atr=p(0.20), trail_dist_atr=p(0.04),
                         max_hold=max(5, int(p(20))))


def psar_grid_fn(h1_df, spread, lot):
    from backtest.validator import _trades_to_daily, _sharpe
    results = {}
    for sl in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5]:
        for tp in [8.0, 12.0, 16.0, 20.0]:
            for mh in [15, 20, 30]:
                trades = backtest_psar(h1_df, spread, lot, sl_atr=sl, tp_atr=tp, max_hold=mh)
                daily = _trades_to_daily(trades)
                results[f"SL={sl}_TP={tp}_MH={mh}"] = _sharpe(daily)
    return results


def psar_grid_backtest_fn(h1_df, spread, lot):
    """Return {label: trades_list} for CSCV PBO."""
    results = {}
    for sl in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5]:
        for tp in [8.0, 12.0, 16.0, 20.0]:
            for mh in [15, 20, 30]:
                label = f"SL={sl}_TP={tp}_MH={mh}"
                results[label] = backtest_psar(h1_df, spread, lot,
                                               sl_atr=sl, tp_atr=tp, max_hold=mh)
    return results


# ═══════════════════════════════════════════════════════════════
# SESS_BO
# ═══════════════════════════════════════════════════════════════

def backtest_sess_bo(h1_df, spread=SPREAD, lot=LOT,
                     session="peak_12_14", lookback_bars=4,
                     sl_atr=4.5, tp_atr=4.0, trail_act_atr=0.14,
                     trail_dist_atr=0.025, max_hold=20):
    SESSION_DEFS = {"peak_12_14": (12,14)}
    df = h1_df.copy(); df['ATR'] = compute_atr(df); df = df.dropna(subset=['ATR'])
    trades = []; pos = None
    c_arr = df['Close'].values; h_arr = df['High'].values; l_arr = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour
    times = df.index; n = len(df); last_exit = -999
    sess_start, _ = SESSION_DEFS[session]
    for i in range(lookback_bars, n):
        c = c_arr[i]; h = h_arr[i]; lo = l_arr[i]; cur_atr = atr[i]; cur_hour = hours[i]
        if pos is not None:
            result = _run_exit_logic(pos, i, h, lo, c, spread, lot, pos['atr'],
                                     sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold, times)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        if cur_hour != sess_start: continue
        if i > 0 and hours[i-1] == sess_start: continue
        rh = max(h_arr[i-lookback_bars:i]); rl = min(l_arr[i-lookback_bars:i])
        if c > rh:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif c < rl:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
    return trades


def sess_perturb_fn(h1_df, spread, lot, rng):
    def p(base, pct=0.20): return base * (1 + rng.uniform(-pct, pct))
    return backtest_sess_bo(h1_df, spread, lot, sl_atr=p(4.5), tp_atr=p(4.0),
                            lookback_bars=max(2, int(p(4))),
                            trail_act_atr=p(0.14), trail_dist_atr=p(0.025),
                            max_hold=max(5, int(p(20))))


def sess_grid_fn(h1_df, spread, lot):
    from backtest.validator import _trades_to_daily, _sharpe
    results = {}
    for sl in [3.0, 3.5, 4.0, 4.5, 5.0]:
        for tp in [3.0, 3.5, 4.0, 5.0, 6.0]:
            for lb in [2, 3, 4, 5]:
                trades = backtest_sess_bo(h1_df, spread, lot, sl_atr=sl, tp_atr=tp, lookback_bars=lb)
                daily = _trades_to_daily(trades)
                results[f"SL={sl}_TP={tp}_LB={lb}"] = _sharpe(daily)
    return results


def sess_grid_backtest_fn(h1_df, spread, lot):
    """Return {label: trades_list} for CSCV PBO."""
    results = {}
    for sl in [3.0, 3.5, 4.0, 4.5, 5.0]:
        for tp in [3.0, 3.5, 4.0, 5.0, 6.0]:
            for lb in [2, 3, 4, 5]:
                label = f"SL={sl}_TP={tp}_LB={lb}"
                results[label] = backtest_sess_bo(h1_df, spread, lot,
                                                   sl_atr=sl, tp_atr=tp, lookback_bars=lb)
    return results


# ═══════════════════════════════════════════════════════════════
# TSMOM
# ═══════════════════════════════════════════════════════════════

def backtest_tsmom(h1_df, spread=SPREAD, lot=LOT,
                   fast_period=480, slow_period=720,
                   sl_atr=4.5, tp_atr=6.0, trail_act_atr=0.14,
                   trail_dist_atr=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df['fast_ma'] = df['Close'].rolling(fast_period).mean()
    df['slow_ma'] = df['Close'].rolling(slow_period).mean()
    df = df.dropna(subset=['ATR', 'fast_ma', 'slow_ma'])
    c_arr = df['Close'].values; h_arr = df['High'].values; l_arr = df['Low'].values
    atr = df['ATR'].values; fast = df['fast_ma'].values; slow = df['slow_ma'].values
    times = df.index; n = len(df); trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        c = c_arr[i]; h = h_arr[i]; lo = l_arr[i]; cur_atr = atr[i]
        if pos is not None:
            result = _run_exit_logic(pos, i, h, lo, c, spread, lot, pos['atr'],
                                     sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold, times)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            if (pos['dir'] == 'BUY' and fast[i] < slow[i]) or \
               (pos['dir'] == 'SELL' and fast[i] > slow[i]):
                if pos['dir'] == 'BUY':
                    pnl = (c - pos['entry'] - spread) * lot * PV
                else:
                    pnl = (pos['entry'] - c - spread) * lot * PV
                trades.append(_mk(pos, c, times[i], "Reversal", i, pnl))
                pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        if fast[i] > slow[i] and fast[i-1] <= slow[i-1]:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif fast[i] < slow[i] and fast[i-1] >= slow[i-1]:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
    return trades


def tsmom_perturb_fn(h1_df, spread, lot, rng):
    def p(base, pct=0.20): return base * (1 + rng.uniform(-pct, pct))
    return backtest_tsmom(h1_df, spread, lot,
                          fast_period=max(100, int(p(480))),
                          slow_period=max(200, int(p(720))),
                          sl_atr=p(4.5), tp_atr=p(6.0),
                          trail_act_atr=p(0.14), trail_dist_atr=p(0.025),
                          max_hold=max(5, int(p(20))))


def tsmom_grid_fn(h1_df, spread, lot):
    from backtest.validator import _trades_to_daily, _sharpe
    results = {}
    for f in [360, 480, 600]:
        for s in [600, 720, 960]:
            if s <= f: continue
            for sl in [3.0, 4.0, 4.5, 5.5]:
                for tp in [4.0, 6.0, 8.0]:
                    trades = backtest_tsmom(h1_df, spread, lot,
                                            fast_period=f, slow_period=s,
                                            sl_atr=sl, tp_atr=tp)
                    daily = _trades_to_daily(trades)
                    results[f"F={f}_S={s}_SL={sl}_TP={tp}"] = _sharpe(daily)
    return results


def tsmom_grid_backtest_fn(h1_df, spread, lot):
    """Return {label: trades_list} for CSCV PBO."""
    results = {}
    for f in [360, 480, 600]:
        for s in [600, 720, 960]:
            if s <= f: continue
            for sl in [3.0, 4.0, 4.5, 5.5]:
                for tp in [4.0, 6.0, 8.0]:
                    label = f"F={f}_S={s}_SL={sl}_TP={tp}"
                    results[label] = backtest_tsmom(h1_df, spread, lot,
                                                     fast_period=f, slow_period=s,
                                                     sl_atr=sl, tp_atr=tp)
    return results


# ═══════════════════════════════════════════════════════════════
# MACD
# ═══════════════════════════════════════════════════════════════

def _compute_adx(df):
    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    return dx.rolling(14).mean()


def backtest_macd(h1_df, spread=SPREAD, lot=LOT,
                  fast_period=12, slow_period=26, signal_period=9,
                  ema_trend=100, adx_threshold=20,
                  sl_atr=4.5, tp_atr=6.0, trail_act_atr=0.14,
                  trail_dist_atr=0.025, max_hold=20):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df['EMA_fast'] = df['Close'].ewm(span=fast_period, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    df['EMA_trend'] = df['Close'].ewm(span=ema_trend, adjust=False).mean()
    df['ADX'] = _compute_adx(df)
    df = df.dropna(subset=['ATR', 'Hist', 'EMA_trend', 'ADX'])
    c_arr = df['Close'].values; h_arr = df['High'].values; l_arr = df['Low'].values
    atr = df['ATR'].values; hist = df['Hist'].values; ema_t = df['EMA_trend'].values
    adx = df['ADX'].values; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        c = c_arr[i]; h = h_arr[i]; lo = l_arr[i]; cur_atr = atr[i]
        if pos is not None:
            result = _run_exit_logic(pos, i, h, lo, c, spread, lot, pos['atr'],
                                     sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold, times)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        if adx[i] < adx_threshold: continue
        if hist[i] > 0 and hist[i-1] <= 0 and c > ema_t[i]:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif hist[i] < 0 and hist[i-1] >= 0 and c < ema_t[i]:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
    return trades


def macd_perturb_fn(h1_df, spread, lot, rng):
    def p(base, pct=0.20): return base * (1 + rng.uniform(-pct, pct))
    return backtest_macd(h1_df, spread, lot,
                         fast_period=max(5, int(p(12))), slow_period=max(15, int(p(26))),
                         signal_period=max(3, int(p(9))), ema_trend=max(50, int(p(100))),
                         adx_threshold=p(20), sl_atr=p(4.5), tp_atr=p(6.0),
                         trail_act_atr=p(0.14), trail_dist_atr=p(0.025),
                         max_hold=max(5, int(p(20))))


def macd_grid_fn(h1_df, spread, lot):
    from backtest.validator import _trades_to_daily, _sharpe
    results = {}
    for sl in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5]:
        for tp in [4.0, 5.0, 6.0, 8.0]:
            for adx in [15, 20, 25]:
                trades = backtest_macd(h1_df, spread, lot, sl_atr=sl, tp_atr=tp, adx_threshold=adx)
                daily = _trades_to_daily(trades)
                results[f"SL={sl}_TP={tp}_ADX={adx}"] = _sharpe(daily)
    return results


def macd_grid_backtest_fn(h1_df, spread, lot):
    """Return {label: trades_list} for CSCV PBO."""
    results = {}
    for sl in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5]:
        for tp in [4.0, 5.0, 6.0, 8.0]:
            for adx in [15, 20, 25]:
                label = f"SL={sl}_TP={tp}_ADX={adx}"
                results[label] = backtest_macd(h1_df, spread, lot,
                                                sl_atr=sl, tp_atr=tp, adx_threshold=adx)
    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    from backtest.validator import StrategyValidator, ValidatorConfig

    t0 = time.time()
    print("=" * 72)
    print("  R82 — Dual PBO Validation")
    print("  Method A: Random Perturbation (parameter stability)")
    print("  Method B: CSCV Grid (Bailey et al. 2017, selection bias)")
    print("=" * 72, flush=True)

    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH
    print("\n  Loading H1 data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})\n")

    strategies = [
        {
            'name': 'PSAR_EA',
            'backtest_fn': lambda df, sp, lt: backtest_psar(df, sp, lt),
            'base_fn': lambda df, sp, lt: backtest_psar(df, sp, lt, sl_atr=3.0, tp_atr=6.0,
                                                         trail_act_atr=99.0, trail_dist_atr=99.0, max_hold=200),
            'perturb_fn': psar_perturb_fn,
            'grid_fn': psar_grid_fn,
            'grid_bt_fn': psar_grid_backtest_fn,
            'n_trials': 72,
            'grid_desc': '6 SL x 4 TP x 3 MH = 72 combos',
        },
        {
            'name': 'SESS_BO_EA',
            'backtest_fn': lambda df, sp, lt: backtest_sess_bo(df, sp, lt),
            'base_fn': lambda df, sp, lt: backtest_sess_bo(df, sp, lt, sl_atr=3.0, tp_atr=3.0,
                                                            trail_act_atr=99.0, trail_dist_atr=99.0, max_hold=50),
            'perturb_fn': sess_perturb_fn,
            'grid_fn': sess_grid_fn,
            'grid_bt_fn': sess_grid_backtest_fn,
            'n_trials': 100,
            'grid_desc': '5 SL x 5 TP x 4 LB = 100 combos',
        },
        {
            'name': 'TSMOM_EA',
            'backtest_fn': lambda df, sp, lt: backtest_tsmom(df, sp, lt),
            'base_fn': lambda df, sp, lt: backtest_tsmom(df, sp, lt, sl_atr=3.0, tp_atr=3.0,
                                                          trail_act_atr=99.0, trail_dist_atr=99.0, max_hold=50),
            'perturb_fn': tsmom_perturb_fn,
            'grid_fn': tsmom_grid_fn,
            'grid_bt_fn': tsmom_grid_backtest_fn,
            'n_trials': 72,
            'grid_desc': '3 F x 3 S x 4 SL x 3 TP (filtered) ~ 72 combos',
        },
        {
            'name': 'MACD_H1',
            'backtest_fn': lambda df, sp, lt: backtest_macd(df, sp, lt),
            'base_fn': lambda df, sp, lt: backtest_macd(df, sp, lt, sl_atr=3.0, tp_atr=3.0,
                                                         trail_act_atr=99.0, trail_dist_atr=99.0,
                                                         max_hold=50, adx_threshold=10),
            'perturb_fn': macd_perturb_fn,
            'grid_fn': macd_grid_fn,
            'grid_bt_fn': macd_grid_backtest_fn,
            'n_trials': 72,
            'grid_desc': '6 SL x 4 TP x 3 ADX = 72 combos',
        },
    ]

    all_results = {}
    for strat in strategies:
        print(f"\n{'='*72}")
        print(f"  {strat['name']}  |  Grid: {strat['grid_desc']}")
        print(f"{'='*72}", flush=True)

        config = ValidatorConfig(
            n_trials_tested=strat['n_trials'],
            realistic_spread=REALISTIC_SPREAD,
            purge_bars=30,
            n_param_perturb=200,
            n_bootstrap=5000,
            n_trade_removal=500,
            pbo_max_grid_combos=200,
        )

        out_dir = str(OUTPUT_DIR / strat['name'])
        validator = StrategyValidator(
            name=strat['name'],
            backtest_fn=strat['backtest_fn'],
            spread=SPREAD, lot=LOT,
            h1_df=h1_df,
            base_backtest_fn=strat['base_fn'],
            param_perturb_fn=strat['perturb_fn'],
            param_grid_fn=strat['grid_fn'],
            param_grid_backtest_fn=strat['grid_bt_fn'],
            config=config,
            output_dir=out_dir,
        )

        results = validator.run_all(stop_on_fail=False)
        passed = sum(1 for r in results.values() if r.passed)
        total = len(results)
        all_results[strat['name']] = {
            'passed': passed, 'total': total,
            'stages': {f"stage{s}": {'passed': r.passed, 'verdict': r.verdict}
                       for s, r in sorted(results.items())},
        }

    # ─── Summary ───
    elapsed = time.time() - t0
    print(f"\n\n{'='*72}")
    print(f"  R82 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*72}")
    print(f"\n  {'Strategy':<15} {'Pass':>6} {'PBO-Perturb':>14} {'PBO-CSCV':>14} {'Verdict':>10}")
    print(f"  {'-'*60}")

    for name, data in all_results.items():
        s4 = data['stages'].get('stage4', {})
        verdict_str = s4.get('verdict', '')
        pbo_p = pbo_c = 'N/A'
        import re
        m_perturb = re.search(r'PBO-Perturb=([\d.]+%)', verdict_str)
        m_cscv = re.search(r'PBO-CSCV=([\d.]+%)', verdict_str)
        if m_perturb: pbo_p = m_perturb.group(1)
        if m_cscv: pbo_c = m_cscv.group(1)
        status = f"{data['passed']}/{data['total']}"
        print(f"  {name:<15} {status:>6} {pbo_p:>14} {pbo_c:>14}")

    with open(OUTPUT_DIR / "r82_summary.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
