#!/usr/bin/env python3
"""
R146 — S3+S4 Independent Portfolio Validation
===============================================
R135 showed S3+S4 has Sharpe 5.78 with low correlation (0.10) to L8.
This experiment runs full 8-stage validation and optimal integration.

Phases:
  1. Full 8-stage validation for S3 standalone
  2. Full 8-stage validation for S4 standalone
  3. Portfolio correlation matrix (6 strategies)
  4. Lot optimization under MaxDD constraint
  5. Regime performance (trending vs ranging by ADX)
  6. Break-even spread analysis
  7. Marginal Sharpe contribution
  8. Capital allocation optimization
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS, load_csv
from backtest.validator import StrategyValidator, ValidatorConfig

OUTPUT_DIR = Path("results/r146_s3s4_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01
CAPS_S3S4 = 35

CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.08, 'SESS_BO': 0.08}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']

H1_CANDIDATES = [
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-27.csv"),
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv"),
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-03-25.csv"),
]

t0 = time.time()


# ═══════════════════════════════════════════════════════════════
# Inline indicator functions
# ═══════════════════════════════════════════════════════════════

def calc_dual_thrust_range(df, n_bars=6):
    hh = df['High'].rolling(n_bars).max()
    lc = df['Close'].rolling(n_bars).min()
    hc = df['Close'].rolling(n_bars).max()
    ll = df['Low'].rolling(n_bars).min()
    return pd.concat([hh - lc, hc - ll], axis=1).max(axis=1)


def calc_chandelier(df, period=22, mult=3.0):
    atr = (df['High'] - df['Low']).rolling(14).mean()
    hh = df['High'].rolling(period).max()
    ll = df['Low'].rolling(period).min()
    out = pd.DataFrame(index=df.index)
    out['Chand_long'] = hh - mult * atr
    out['Chand_short'] = ll + mult * atr
    return out


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _mk(pos, exit_px, exit_time, reason, bar_i, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_px,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_i - pos['bar']}


def _run_exit(pos, i, hi, lo, cl, spread, lot, pv, times,
              sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap):
    atr = pos['atr']
    sl = atr * sl_atr; tp = atr * tp_atr; bars = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_now = (cl - pos['entry'] - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if lo <= pos['entry'] - sl:
            return _mk(pos, pos['entry'] - sl, times[i], "SL", i, -sl * lot * pv)
        if hi >= pos['entry'] + tp:
            return _mk(pos, pos['entry'] + tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry']); extreme = max(extreme, hi); pos['extreme'] = extreme
        if extreme - pos['entry'] >= atr * trail_act:
            trail_price = extreme - atr * trail_dist
            if cl <= trail_price:
                return _mk(pos, trail_price, times[i], "Trail", i, (trail_price - pos['entry'] - spread) * lot * pv)
        if bars >= max_hold:
            return _mk(pos, cl, times[i], "TimeExit", i, pnl_now)
    else:
        pnl_now = (pos['entry'] - cl - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if hi >= pos['entry'] + sl:
            return _mk(pos, pos['entry'] + sl, times[i], "SL", i, -sl * lot * pv)
        if lo <= pos['entry'] - tp:
            return _mk(pos, pos['entry'] - tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry']); extreme = min(extreme, lo); pos['extreme'] = extreme
        if pos['entry'] - extreme >= atr * trail_act:
            trail_price = extreme + atr * trail_dist
            if cl >= trail_price:
                return _mk(pos, trail_price, times[i], "Trail", i, (pos['entry'] - trail_price - spread) * lot * pv)
        if bars >= max_hold:
            return _mk(pos, cl, times[i], "TimeExit", i, pnl_now)
    return None


def _trades_to_daily(trades):
    if not trades:
        return np.array([0.0])
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    if not daily:
        return np.array([0.0])
    return np.array([daily[k] for k in sorted(daily.keys())])


def _sharpe(arr):
    if len(arr) < 10 or np.std(arr, ddof=1) == 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))


def _max_dd(arr):
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 1 else 0.0


def _compute_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0, 'calmar': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    sh = _sharpe(daily)
    dd = _max_dd(daily)
    pnl = sum(pnls)
    return {
        'n': n, 'sharpe': round(sh, 3), 'pnl': round(pnl, 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1) if n else 0,
        'max_dd': round(dd, 2),
        'calmar': round(pnl / dd, 2) if dd > 0 else 9999,
    }


def _trades_to_daily_series(trades):
    if not trades:
        return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


# ═══════════════════════════════════════════════════════════════
# Strategy backtests: S3 Dual Thrust, S4 Chandelier
# ═══════════════════════════════════════════════════════════════

def bt_s3_dual_thrust(h1_df, spread, lot, n_bars=6, k_up=0.5, k_down=0.5,
                      sl_atr=4.5, tp_atr=8.0, trail_act=0.14, trail_dist=0.025,
                      max_hold=20, cap=CAPS_S3S4, start=None, end=None):
    df = h1_df.copy()
    df['ATR14'] = compute_atr(df, 14)
    dt_range = calc_dual_thrust_range(df, n_bars)
    daily_open = df.groupby(df.index.date)['Open'].transform('first')
    sig = pd.Series(0, index=df.index)
    sig[df['Close'] > daily_open + k_up * dt_range] = 1
    sig[df['Close'] < daily_open - k_down * dt_range] = -1

    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr_arr = df['ATR14'].values; times = df.index; n = len(df)
    sig_arr = sig.values; dates = df.index.date

    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if start and str(dates[i]) < start:
            continue
        if end and str(dates[i]) > end:
            break
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap)
            if result:
                trades.append(result); pos = None; last_exit = i
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr_arr[i]) or atr_arr[i] < 0.1:
            continue
        if sig_arr[i] == 1 and sig_arr[i-1] != 1:
            pos = {'dir': 'BUY', 'entry': c[i] + spread/2, 'bar': i, 'time': times[i], 'atr': atr_arr[i]}
        elif sig_arr[i] == -1 and sig_arr[i-1] != -1:
            pos = {'dir': 'SELL', 'entry': c[i] - spread/2, 'bar': i, 'time': times[i], 'atr': atr_arr[i]}
    return trades


def bt_s4_chandelier(h1_df, spread, lot, period=22, mult=3.0,
                     sl_atr=4.5, tp_atr=8.0, trail_act=0.14, trail_dist=0.025,
                     max_hold=20, cap=CAPS_S3S4, start=None, end=None):
    df = h1_df.copy()
    df['ATR14'] = compute_atr(df, 14)
    ch = calc_chandelier(df, period, mult)
    ema100 = df['Close'].ewm(span=100).mean()
    above_long = df['Close'] > ch['Chand_long']
    flip_bull = above_long & (~above_long.shift(1).fillna(False))
    below_short = df['Close'] < ch['Chand_short']
    flip_bear = below_short & (~below_short.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index)
    sig[flip_bull & (df['Close'] > ema100)] = 1
    sig[flip_bear & (df['Close'] < ema100)] = -1

    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr_arr = df['ATR14'].values; times = df.index; n = len(df)
    sig_arr = sig.values; dates = df.index.date

    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if start and str(dates[i]) < start:
            continue
        if end and str(dates[i]) > end:
            break
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap)
            if result:
                trades.append(result); pos = None; last_exit = i
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr_arr[i]) or atr_arr[i] < 0.1:
            continue
        if sig_arr[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i] + spread/2, 'bar': i, 'time': times[i], 'atr': atr_arr[i]}
        elif sig_arr[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i] - spread/2, 'bar': i, 'time': times[i], 'atr': atr_arr[i]}
    return trades


# ═══════════════════════════════════════════════════════════════
# Additional strategy backtests (from R139)
# ═══════════════════════════════════════════════════════════════

def add_psar(df, af_step=0.01, af_max=0.05):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    n = len(df)
    psar = np.empty(n); psar[:] = np.nan
    af = af_step; rising = True; ep = h[0]; psar[0] = l[0]
    for i in range(1, n):
        prev = psar[i-1]
        if rising:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], l[i-1], l[max(0, i-2)])
            if l[i] < psar[i]:
                rising = False; psar[i] = ep; ep = l[i]; af = af_step
            else:
                if h[i] > ep: ep = h[i]; af = min(af + af_step, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], h[i-1], h[max(0, i-2)])
            if h[i] > psar[i]:
                rising = True; psar[i] = ep; ep = h[i]; af = af_step
            else:
                if l[i] < ep: ep = l[i]; af = min(af + af_step, af_max)
    df['PSAR'] = psar


def bt_psar(h1_df, spread, lot, cap, params=None):
    if params is None:
        params = {'sl_atr': 4.5, 'tp_atr': 16.0, 'trail_act': 0.20, 'trail_dist': 0.04, 'max_hold': 20}
    df = h1_df.copy(); add_psar(df); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR', 'PSAR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; psar = df['PSAR'].values; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               params['sl_atr'], params['tp_atr'], params['trail_act'],
                               params['trail_dist'], params['max_hold'], cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if psar[i-1] > c[i-1] and psar[i] < c[i]:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif psar[i-1] < c[i-1] and psar[i] > c[i]:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom(h1_df, spread, lot, cap,
             fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
             trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    max_lb = max(fast, slow)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        if c[i-fast] > 0: s += 0.5 * np.sign(c[i]/c[i-fast] - 1.0)
        if c[i-slow] > 0: s += 0.5 * np.sign(c[i]/c[i-slow] - 1.0)
        score[i] = s
    trades = []; pos = None; last_exit = -999
    for i in range(max_lb+1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and score[i] < 0:
                pnl = (c[i]-pos['entry']-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and score[i] > 0:
                pnl = (pos['entry']-c[i]-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(score[i]) or np.isnan(score[i-1]): continue
        if score[i] > 0 and score[i-1] <= 0:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_sess_bo(h1_df, spread, lot, cap,
               session_hour=12, lookback=4, sl_atr=4.5, tp_atr=4.0,
               trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(lookback, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if hours[i] != session_hour: continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80, flush=True)
    print("  R146 — S3+S4 Independent Portfolio Validation", flush=True)
    print("=" * 80, flush=True)

    all_results = {'experiment': 'R146 S3+S4 Independent Portfolio Validation'}

    # ── Load Data ──
    h1_path = next((p for p in H1_CANDIDATES if p.exists()), H1_CANDIDATES[-1])
    print(f"\n  Loading H1: {h1_path}", flush=True)
    h1 = load_csv(str(h1_path))
    print(f"  H1 loaded: {len(h1)} bars ({h1.index[0]} → {h1.index[-1]})", flush=True)
    all_results['data'] = {'h1_bars': len(h1), 'h1_path': str(h1_path)}

    # ════════════════════════════════════════════════════════════════
    # Phase 1: Full 8-stage validation for S3 standalone
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 1: Full 8-Stage Validation — S3 Dual Thrust", flush=True)
    print("=" * 70, flush=True)

    def s3_backtest_fn(h1_df, spread, lot):
        return bt_s3_dual_thrust(h1_df, spread, lot, cap=CAPS_S3S4)

    s3_validator = StrategyValidator(
        name="S3_DualThrust",
        backtest_fn=s3_backtest_fn,
        spread=SPREAD,
        lot=0.05,
        config=ValidatorConfig(n_trials_tested=1),
        output_dir=str(OUTPUT_DIR / "S3_validator"),
        h1_df=h1,
    )
    s3_val_results = s3_validator.run_all(stop_on_fail=False)

    s3_summary = {}
    for stage, res in s3_val_results.items():
        s3_summary[f'stage{stage}'] = {
            'name': res.name, 'passed': res.passed, 'sharpe': res.sharpe,
            'verdict': res.verdict, 'elapsed_s': res.elapsed_s,
        }
    s3_passed = sum(1 for r in s3_val_results.values() if r.passed)
    s3_total = len(s3_val_results)
    print(f"\n  S3 Validation: {s3_passed}/{s3_total} stages passed", flush=True)
    all_results['phase1_s3_validation'] = s3_summary
    all_results['phase1_s3_pass_rate'] = f"{s3_passed}/{s3_total}"

    # ════════════════════════════════════════════════════════════════
    # Phase 2: Full 8-stage validation for S4 standalone
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: Full 8-Stage Validation — S4 Chandelier", flush=True)
    print("=" * 70, flush=True)

    def s4_backtest_fn(h1_df, spread, lot):
        return bt_s4_chandelier(h1_df, spread, lot, cap=CAPS_S3S4)

    s4_validator = StrategyValidator(
        name="S4_Chandelier",
        backtest_fn=s4_backtest_fn,
        spread=SPREAD,
        lot=0.09,
        config=ValidatorConfig(n_trials_tested=1),
        output_dir=str(OUTPUT_DIR / "S4_validator"),
        h1_df=h1,
    )
    s4_val_results = s4_validator.run_all(stop_on_fail=False)

    s4_summary = {}
    for stage, res in s4_val_results.items():
        s4_summary[f'stage{stage}'] = {
            'name': res.name, 'passed': res.passed, 'sharpe': res.sharpe,
            'verdict': res.verdict, 'elapsed_s': res.elapsed_s,
        }
    s4_passed = sum(1 for r in s4_val_results.values() if r.passed)
    s4_total = len(s4_val_results)
    print(f"\n  S4 Validation: {s4_passed}/{s4_total} stages passed", flush=True)
    all_results['phase2_s4_validation'] = s4_summary
    all_results['phase2_s4_pass_rate'] = f"{s4_passed}/{s4_total}"

    # ════════════════════════════════════════════════════════════════
    # Phase 3: Portfolio correlation matrix (6 strategies)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 3: Portfolio Correlation Matrix (6 Strategies)", flush=True)
    print("=" * 70, flush=True)

    print("  Running S3 Dual Thrust...", flush=True)
    s3_trades = bt_s3_dual_thrust(h1, SPREAD, UNIT_LOT)
    s3_daily = _trades_to_daily_series(s3_trades)
    print(f"    S3: {len(s3_trades)} trades", flush=True)

    print("  Running S4 Chandelier...", flush=True)
    s4_trades = bt_s4_chandelier(h1, SPREAD, UNIT_LOT)
    s4_daily = _trades_to_daily_series(s4_trades)
    print(f"    S4: {len(s4_trades)} trades", flush=True)

    print("  Running L8_MAX...", flush=True)
    l8_daily = pd.Series(dtype=float)
    try:
        bundle = DataBundle.load_default()
        kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': CAPS['L8_MAX'],
              'spread_cost': SPREAD, 'initial_capital': 2000,
              'min_lot_size': UNIT_LOT, 'max_lot_size': UNIT_LOT}
        l8_result = run_variant(bundle, "L8_MAX", verbose=False, **kw)
        l8_trades_raw = l8_result.get('_trades', [])
        l8_trades = [
            {'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
             'entry_time': t.entry_time, 'exit_time': t.exit_time,
             'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0}
            for t in l8_trades_raw
        ]
        l8_daily = _trades_to_daily_series(l8_trades)
        print(f"    L8_MAX: {len(l8_trades)} trades", flush=True)
    except Exception as e:
        print(f"    WARNING: L8_MAX failed: {e}", flush=True)
        l8_trades = []

    print("  Running PSAR...", flush=True)
    psar_trades = bt_psar(h1, SPREAD, UNIT_LOT, CAPS['PSAR'])
    psar_daily = _trades_to_daily_series(psar_trades)
    print(f"    PSAR: {len(psar_trades)} trades", flush=True)

    print("  Running TSMOM...", flush=True)
    tsmom_trades = bt_tsmom(h1, SPREAD, UNIT_LOT, CAPS['TSMOM'])
    tsmom_daily = _trades_to_daily_series(tsmom_trades)
    print(f"    TSMOM: {len(tsmom_trades)} trades", flush=True)

    print("  Running SESS_BO...", flush=True)
    sess_trades = bt_sess_bo(h1, SPREAD, UNIT_LOT, CAPS['SESS_BO'])
    sess_daily = _trades_to_daily_series(sess_trades)
    print(f"    SESS_BO: {len(sess_trades)} trades", flush=True)

    strat_names = ['S3', 'S4', 'L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']
    daily_series = [s3_daily, s4_daily, l8_daily, psar_daily, tsmom_daily, sess_daily]

    all_idx = pd.DatetimeIndex([])
    for ds in daily_series:
        if len(ds) > 0:
            all_idx = all_idx.union(ds.index)

    aligned = pd.DataFrame(index=all_idx)
    for name, ds in zip(strat_names, daily_series):
        aligned[name] = ds.reindex(all_idx, fill_value=0.0)

    corr_matrix = aligned.corr()
    print("\n  6x6 Daily PnL Correlation Matrix:", flush=True)
    print(f"  {'':>10s}", end="", flush=True)
    for sn in strat_names:
        print(f" {sn:>8s}", end="")
    print(flush=True)
    for sn in strat_names:
        print(f"  {sn:>10s}", end="", flush=True)
        for sn2 in strat_names:
            print(f" {corr_matrix.loc[sn, sn2]:>8.3f}", end="")
        print(flush=True)

    corr_dict = {}
    for sn in strat_names:
        corr_dict[sn] = {sn2: round(float(corr_matrix.loc[sn, sn2]), 4) for sn2 in strat_names}
    all_results['phase3_correlation'] = corr_dict

    per_strat_stats = {}
    for name, trades in zip(strat_names, [s3_trades, s4_trades, l8_trades, psar_trades, tsmom_trades, sess_trades]):
        per_strat_stats[name] = _compute_stats(trades)
    all_results['phase3_individual_stats'] = per_strat_stats

    # ════════════════════════════════════════════════════════════════
    # Phase 4: Lot optimization under MaxDD constraint
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 4: Lot Optimization (S3+S4, MaxDD < $200)", flush=True)
    print("=" * 70, flush=True)

    lot_values = [round(x * 0.01, 2) for x in range(1, 11)]
    lot_grid = []

    print(f"\n  {'S3_lot':>7s}  {'S4_lot':>7s}  {'n':>5s}  {'Sharpe':>7s}  "
          f"{'PnL':>10s}  {'WR':>6s}  {'MaxDD':>8s}", flush=True)
    print("  " + "-" * 60, flush=True)

    for s3_lot, s4_lot in product(lot_values, lot_values):
        s3_t = bt_s3_dual_thrust(h1, SPREAD, s3_lot)
        s4_t = bt_s4_chandelier(h1, SPREAD, s4_lot)
        combined = s3_t + s4_t
        st = _compute_stats(combined)

        if st['max_dd'] > 200:
            continue

        lot_grid.append({'s3_lot': s3_lot, 's4_lot': s4_lot, **st})

    lot_grid.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  Top 20 by Sharpe (MaxDD < $200):", flush=True)
    for i, g in enumerate(lot_grid[:20]):
        print(f"    #{i+1:2d}: S3={g['s3_lot']:.2f}, S4={g['s4_lot']:.2f} -> "
              f"Sharpe={g['sharpe']:.3f}, PnL=${g['pnl']:.0f}, "
              f"MaxDD=${g['max_dd']:.0f}, WR={g['wr']:.1f}%", flush=True)

    all_results['phase4_lot_grid_top20'] = lot_grid[:20]
    best_lot = lot_grid[0] if lot_grid else {'s3_lot': 0.05, 's4_lot': 0.09}
    all_results['phase4_best_lot'] = {'s3_lot': best_lot['s3_lot'], 's4_lot': best_lot['s4_lot']}
    print(f"\n  Best combo: S3={best_lot['s3_lot']:.2f}, S4={best_lot['s4_lot']:.2f}", flush=True)

    # ════════════════════════════════════════════════════════════════
    # Phase 5: Regime performance (ADX trending vs ranging)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 5: Regime Performance (ADX Trending vs Ranging)", flush=True)
    print("=" * 70, flush=True)

    df_adx = h1.copy()
    high_d = df_adx['High'] - df_adx['High'].shift(1)
    low_d = df_adx['Low'].shift(1) - df_adx['Low']
    plus_dm = pd.Series(np.where((high_d > low_d) & (high_d > 0), high_d, 0), index=df_adx.index)
    minus_dm = pd.Series(np.where((low_d > high_d) & (low_d > 0), low_d, 0), index=df_adx.index)
    atr14 = compute_atr(df_adx, 14)
    plus_di = 100 * plus_dm.rolling(14).mean() / atr14
    minus_di = 100 * minus_dm.rolling(14).mean() / atr14
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.rolling(14).mean()
    df_adx['ADX14'] = adx

    trending_mask = df_adx['ADX14'] > 25
    ranging_mask = df_adx['ADX14'] <= 25

    regime_results = {}
    for regime_name, mask in [('trending', trending_mask), ('ranging', ranging_mask)]:
        n_bars_regime = int(mask.sum())
        regime_dates = set(df_adx.index[mask].date)

        s3_regime = [t for t in s3_trades if pd.Timestamp(t['entry_time']).date() in regime_dates]
        s4_regime = [t for t in s4_trades if pd.Timestamp(t['entry_time']).date() in regime_dates]

        s3_st = _compute_stats(s3_regime)
        s4_st = _compute_stats(s4_regime)
        combined_st = _compute_stats(s3_regime + s4_regime)

        regime_results[regime_name] = {
            'n_bars': n_bars_regime, 'S3': s3_st, 'S4': s4_st, 'S3_S4': combined_st,
        }

        print(f"\n  {regime_name.upper()} regime ({n_bars_regime} bars):", flush=True)
        print(f"    S3: n={s3_st['n']}, Sharpe={s3_st['sharpe']:.3f}, PnL=${s3_st['pnl']:.0f}", flush=True)
        print(f"    S4: n={s4_st['n']}, Sharpe={s4_st['sharpe']:.3f}, PnL=${s4_st['pnl']:.0f}", flush=True)
        print(f"    S3+S4: n={combined_st['n']}, Sharpe={combined_st['sharpe']:.3f}, PnL=${combined_st['pnl']:.0f}", flush=True)

    all_results['phase5_regime'] = regime_results

    # ════════════════════════════════════════════════════════════════
    # Phase 6: Break-even spread analysis
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 6: Break-Even Spread Analysis", flush=True)
    print("=" * 70, flush=True)

    spread_levels = [0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.50]
    spread_results = []
    best_s3_lot = best_lot['s3_lot']
    best_s4_lot = best_lot['s4_lot']

    print(f"\n  Using best lots: S3={best_s3_lot:.2f}, S4={best_s4_lot:.2f}", flush=True)
    print(f"\n  {'Spread':>8s}  {'n':>5s}  {'Sharpe':>7s}  {'PnL':>10s}  {'WR':>6s}  {'MaxDD':>8s}", flush=True)
    print("  " + "-" * 50, flush=True)

    breakeven_spread = None
    for sp in spread_levels:
        s3_t = bt_s3_dual_thrust(h1, sp, best_s3_lot)
        s4_t = bt_s4_chandelier(h1, sp, best_s4_lot)
        combined = s3_t + s4_t
        st = _compute_stats(combined)
        spread_results.append({'spread': sp, **st})
        print(f"  ${sp:7.2f}  {st['n']:5d}  {st['sharpe']:7.3f}  ${st['pnl']:>9.0f}  "
              f"{st['wr']:5.1f}%  ${st['max_dd']:>7.0f}", flush=True)
        if breakeven_spread is None and st['sharpe'] < 1.0:
            breakeven_spread = sp

    if breakeven_spread is None:
        breakeven_spread = "> 1.50"
        print(f"\n  Sharpe stays above 1.0 even at spread=$1.50", flush=True)
    else:
        print(f"\n  Break-even spread (Sharpe < 1.0): ${breakeven_spread}", flush=True)

    all_results['phase6_spread'] = spread_results
    all_results['phase6_breakeven_spread'] = breakeven_spread

    # ════════════════════════════════════════════════════════════════
    # Phase 7: Marginal Sharpe contribution
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 7: Marginal Sharpe Contribution", flush=True)
    print("=" * 70, flush=True)

    main_daily = pd.Series(dtype=float)
    main_all = {}
    for sn in STRAT_ORDER:
        lot_mult = R89_LOTS[sn] / UNIT_LOT
        if sn == 'L8_MAX':
            trades_sn = l8_trades
        elif sn == 'PSAR':
            trades_sn = psar_trades
        elif sn == 'TSMOM':
            trades_sn = tsmom_trades
        elif sn == 'SESS_BO':
            trades_sn = sess_trades
        else:
            trades_sn = []
        for t in trades_sn:
            d = pd.Timestamp(t['exit_time']).date()
            main_all[d] = main_all.get(d, 0) + t['pnl'] * lot_mult

    main_dates = sorted(main_all.keys())
    main_daily = pd.Series([main_all[d] for d in main_dates], index=pd.DatetimeIndex(main_dates)) if main_dates else pd.Series(dtype=float)

    s3s4_all = {}
    s3_best = bt_s3_dual_thrust(h1, SPREAD, best_s3_lot)
    s4_best = bt_s4_chandelier(h1, SPREAD, best_s4_lot)
    for t in s3_best + s4_best:
        d = pd.Timestamp(t['exit_time']).date()
        s3s4_all[d] = s3s4_all.get(d, 0) + t['pnl']
    s3s4_dates = sorted(s3s4_all.keys())
    s3s4_daily = pd.Series([s3s4_all[d] for d in s3s4_dates], index=pd.DatetimeIndex(s3s4_dates)) if s3s4_dates else pd.Series(dtype=float)

    combined_idx = main_daily.index.union(s3s4_daily.index)
    main_aligned = main_daily.reindex(combined_idx, fill_value=0.0)
    s3s4_aligned = s3s4_daily.reindex(combined_idx, fill_value=0.0)
    combined_daily = main_aligned + s3s4_aligned

    main_sharpe = _sharpe(main_aligned.values)
    s3s4_sharpe = _sharpe(s3s4_aligned.values)
    combined_sharpe = _sharpe(combined_daily.values)
    marginal = combined_sharpe - main_sharpe

    print(f"  Main portfolio Sharpe:    {main_sharpe:.3f}", flush=True)
    print(f"  S3+S4 standalone Sharpe:  {s3s4_sharpe:.3f}", flush=True)
    print(f"  Combined Sharpe:          {combined_sharpe:.3f}", flush=True)
    print(f"  Marginal contribution:    {marginal:+.3f}", flush=True)
    print(f"  Correlation(main, S3S4):  {main_aligned.corr(s3s4_aligned):.4f}", flush=True)

    all_results['phase7_marginal'] = {
        'main_sharpe': round(main_sharpe, 3),
        's3s4_sharpe': round(s3s4_sharpe, 3),
        'combined_sharpe': round(combined_sharpe, 3),
        'marginal_contribution': round(marginal, 3),
        'correlation': round(float(main_aligned.corr(s3s4_aligned)), 4),
        'main_pnl': round(float(main_aligned.sum()), 2),
        's3s4_pnl': round(float(s3s4_aligned.sum()), 2),
        'combined_pnl': round(float(combined_daily.sum()), 2),
        'main_maxdd': round(_max_dd(main_aligned.values), 2),
        's3s4_maxdd': round(_max_dd(s3s4_aligned.values), 2),
        'combined_maxdd': round(_max_dd(combined_daily.values), 2),
    }

    # ════════════════════════════════════════════════════════════════
    # Phase 8: Capital allocation optimization
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 8: Capital Allocation Optimization", flush=True)
    print("=" * 70, flush=True)

    alloc_weights = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alloc_results = []

    print(f"\n  {'main_w':>7s}  {'s3s4_w':>7s}  {'Sharpe':>7s}  {'PnL':>10s}  "
          f"{'MaxDD':>8s}  {'Sharpe/DD':>9s}", flush=True)
    print("  " + "-" * 55, flush=True)

    for main_w in alloc_weights:
        s3s4_w = 1.0 - main_w
        weighted_daily = main_aligned * main_w + s3s4_aligned * s3s4_w
        sh = _sharpe(weighted_daily.values)
        pnl = float(weighted_daily.sum())
        dd = _max_dd(weighted_daily.values)
        sh_dd = sh / dd if dd > 0 else 9999

        alloc_results.append({
            'main_weight': main_w, 's3s4_weight': round(s3s4_w, 2),
            'sharpe': round(sh, 3), 'pnl': round(pnl, 2),
            'max_dd': round(dd, 2), 'sharpe_per_dd': round(sh_dd, 4),
        })

        print(f"  {main_w:7.1f}  {s3s4_w:7.1f}  {sh:7.3f}  ${pnl:>9.0f}  "
              f"${dd:>7.0f}  {sh_dd:>9.4f}", flush=True)

    best_by_sharpe = max(alloc_results, key=lambda x: x['sharpe'])
    best_by_sh_dd = max(alloc_results, key=lambda x: x['sharpe_per_dd'])
    print(f"\n  Best by Sharpe:    main={best_by_sharpe['main_weight']:.1f}, "
          f"s3s4={best_by_sharpe['s3s4_weight']:.1f} "
          f"(Sharpe={best_by_sharpe['sharpe']:.3f})", flush=True)
    print(f"  Best by Sharpe/DD: main={best_by_sh_dd['main_weight']:.1f}, "
          f"s3s4={best_by_sh_dd['s3s4_weight']:.1f} "
          f"(Sharpe/DD={best_by_sh_dd['sharpe_per_dd']:.4f})", flush=True)

    all_results['phase8_allocation'] = alloc_results
    all_results['phase8_best_by_sharpe'] = {
        'main_weight': best_by_sharpe['main_weight'],
        's3s4_weight': best_by_sharpe['s3s4_weight'],
    }
    all_results['phase8_best_by_sharpe_dd'] = {
        'main_weight': best_by_sh_dd['main_weight'],
        's3s4_weight': best_by_sh_dd['s3s4_weight'],
    }

    # ═══════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    all_results['runtime_sec'] = round(elapsed, 1)
    all_results['runtime_min'] = round(elapsed / 60, 1)

    out_file = OUTPUT_DIR / "r146_results.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)

    print(f"\n{'=' * 80}", flush=True)
    print(f"  R146 complete in {elapsed/60:.1f} min. Results → {out_file}", flush=True)
    print(f"{'=' * 80}", flush=True)


if __name__ == '__main__':
    main()
