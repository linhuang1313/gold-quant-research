#!/usr/bin/env python3
"""
R141 — Tail Risk Budget + Extreme Monte Carlo
================================================
Fit per-strategy PnL distributions (normal, t, empirical), generate
10,000 synthetic portfolio paths with Poisson jump process and spread noise,
then optimize lot allocation subject to CVaR99 constraint.

Phases:
  1. Load data, run all 4 strategies at unit lot
  2. Fit distributions (normal, t-distribution, empirical)
  3. Generate 10,000 synthetic paths with jumps + spread noise
  4. Compute risk metrics per path (Sharpe, MaxDD, CVaR95, CVaR99)
  5. Grid search: maximize E[Sharpe] subject to CVaR99 < $300
  6. Compare R89 allocation vs tail-constrained optimal
  7. Stress scenario: 2x COVID-level drawdown overlay
  8. Risk budget breakdown by strategy
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS

OUTPUT_DIR = Path("results/r141_tail_risk_budget")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01
CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.08, 'SESS_BO': 0.08}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']

N_PATHS = 10000
CVAR99_LIMIT = 300.0

t0 = time.time()


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


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


def _cvar(arr, alpha=0.05):
    if len(arr) < 20:
        return 0.0
    threshold = np.percentile(arr, alpha * 100)
    tail = arr[arr <= threshold]
    return float(np.mean(tail)) if len(tail) > 0 else 0.0


# ═══════════════════════════════════════════════════════════════
# Strategy backtests
# ═══════════════════════════════════════════════════════════════

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
# Distribution fitting + synthetic path generation
# ═══════════════════════════════════════════════════════════════

def fit_distributions(trade_pnls):
    """Fit normal and t-distribution to per-trade PnL array."""
    arr = np.array(trade_pnls)
    if len(arr) < 20:
        return {'normal': {'mu': 0, 'sigma': 1}, 't': {'df': 5, 'loc': 0, 'scale': 1},
                'empirical_mean': 0, 'empirical_std': 1, 'n': len(arr)}

    mu, sigma = float(np.mean(arr)), float(np.std(arr, ddof=1))
    sigma = max(sigma, 1e-6)

    try:
        t_params = sp_stats.t.fit(arr)
        t_df, t_loc, t_scale = t_params
    except Exception:
        t_df, t_loc, t_scale = 5.0, mu, sigma

    return {
        'normal': {'mu': round(mu, 4), 'sigma': round(sigma, 4)},
        't': {'df': round(float(t_df), 2), 'loc': round(float(t_loc), 4),
              'scale': round(float(t_scale), 4)},
        'empirical_mean': round(mu, 4),
        'empirical_std': round(sigma, 4),
        'n': len(arr),
    }


def generate_synthetic_paths(strat_fits, strat_n_trades, lots, n_paths, n_days, rng):
    """Generate synthetic portfolio daily PnL paths.

    Uses t-distribution for fat tails + Poisson jump process.
    """
    all_paths = np.zeros((n_paths, n_days))

    for sn in STRAT_ORDER:
        fit = strat_fits[sn]
        lot_mult = lots[sn] / UNIT_LOT
        avg_trades_per_day = strat_n_trades[sn] / max(n_days, 1)

        t_df = fit['t']['df']
        t_loc = fit['t']['loc']
        t_scale = fit['t']['scale']
        sigma = fit['empirical_std']

        for pi in range(n_paths):
            for di in range(n_days):
                n_trades_today = rng.poisson(max(avg_trades_per_day, 0.1))
                daily_pnl = 0.0

                for _ in range(n_trades_today):
                    raw_pnl = sp_stats.t.rvs(t_df, loc=t_loc, scale=t_scale, random_state=rng)
                    spread_noise = rng.uniform(0.20, 0.50)
                    spread_cost = (spread_noise - SPREAD) * lots[sn] * PV
                    raw_pnl -= spread_cost / (lot_mult * PV) if lot_mult > 0 else 0
                    daily_pnl += raw_pnl * lot_mult

                # Poisson jump process: lambda=0.02/day, jump = 5*sigma
                if rng.random() < 0.02:
                    jump_sign = 1 if rng.random() < 0.5 else -1
                    jump_size = 5.0 * sigma * lot_mult * jump_sign
                    daily_pnl += jump_size

                all_paths[pi, di] += daily_pnl

    return all_paths


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_h1():
    from backtest.runner import load_csv
    candidates = [
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-27.csv"),
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv"),
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-03-25.csv"),
    ]
    h1_path = next((p for p in candidates if p.exists()), candidates[-1])
    return load_csv(str(h1_path))


def run_all_strategies(h1_df, bundle):
    strat_trades = {}
    strat_trades['PSAR'] = bt_psar(h1_df, SPREAD, UNIT_LOT, CAPS['PSAR'])
    strat_trades['TSMOM'] = bt_tsmom(h1_df, SPREAD, UNIT_LOT, CAPS['TSMOM'])
    strat_trades['SESS_BO'] = bt_sess_bo(h1_df, SPREAD, UNIT_LOT, CAPS['SESS_BO'])

    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': CAPS['L8_MAX'],
          'spread_cost': SPREAD, 'initial_capital': 2000,
          'min_lot_size': UNIT_LOT, 'max_lot_size': UNIT_LOT}
    l8_result = run_variant(bundle, "L8_MAX", verbose=False, **kw)
    strat_trades['L8_MAX'] = [
        {'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
         'entry_time': t.entry_time, 'exit_time': t.exit_time,
         'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0}
        for t in l8_result.get('_trades', [])
    ]
    return strat_trades


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    results = {'experiment': 'R141 Tail Risk Budget + Extreme Monte Carlo'}

    print("=" * 80, flush=True)
    print("  R141 — Tail Risk Budget + Extreme Monte Carlo", flush=True)
    print("=" * 80, flush=True)

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Load data + run strategies at unit lot
    # ═══════════════════════════════════════════════════════════
    print("\n  Phase 1: Loading data + running strategies at unit lot...", flush=True)
    h1_df = load_h1()
    bundle = DataBundle.load_default()
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})", flush=True)

    strat_trades = run_all_strategies(h1_df, bundle)
    strat_pnls = {}
    for sn in STRAT_ORDER:
        pnls = np.array([t['pnl'] for t in strat_trades[sn]])
        strat_pnls[sn] = pnls
        print(f"    {sn:>10s}: {len(pnls)} trades, mean=${np.mean(pnls):.2f}, "
              f"std=${np.std(pnls):.2f}", flush=True)

    results['phase1'] = {sn: {'n': len(strat_pnls[sn]),
                              'mean': round(float(np.mean(strat_pnls[sn])), 2),
                              'std': round(float(np.std(strat_pnls[sn])), 2)}
                         for sn in STRAT_ORDER}

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Fit distributions
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: Distribution Fitting", flush=True)
    print("=" * 70, flush=True)

    strat_fits = {}
    for sn in STRAT_ORDER:
        fit = fit_distributions(strat_pnls[sn])
        strat_fits[sn] = fit
        print(f"\n  {sn}:", flush=True)
        print(f"    Normal:  mu={fit['normal']['mu']:.4f}, sigma={fit['normal']['sigma']:.4f}", flush=True)
        print(f"    t-dist:  df={fit['t']['df']:.2f}, loc={fit['t']['loc']:.4f}, "
              f"scale={fit['t']['scale']:.4f}", flush=True)
        print(f"    n={fit['n']} trades", flush=True)

        if fit['t']['df'] < 10:
            print(f"    -> Fat tails detected (df={fit['t']['df']:.1f} < 10)", flush=True)

    results['phase2_fits'] = strat_fits

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Generate 10,000 synthetic paths
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print(f"  Phase 3: Generating {N_PATHS} Synthetic Portfolio Paths", flush=True)
    print("=" * 70, flush=True)

    daily_arr = _trades_to_daily(
        [dict(t, pnl=t['pnl'] * R89_LOTS[sn] / UNIT_LOT)
         for sn in STRAT_ORDER for t in strat_trades[sn]])
    n_days = max(len(daily_arr), 252 * 10)

    strat_n_trades = {sn: len(strat_trades[sn]) for sn in STRAT_ORDER}

    rng = np.random.RandomState(42)
    t_gen = time.time()
    paths = generate_synthetic_paths(strat_fits, strat_n_trades, R89_LOTS, N_PATHS, n_days, rng)
    gen_elapsed = time.time() - t_gen
    print(f"    Generated {N_PATHS} x {n_days} days in {gen_elapsed:.1f}s", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Risk metrics per path
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 4: Risk Metrics Across Synthetic Paths", flush=True)
    print("=" * 70, flush=True)

    path_sharpes = np.zeros(N_PATHS)
    path_dds = np.zeros(N_PATHS)
    path_cvar95 = np.zeros(N_PATHS)
    path_cvar99 = np.zeros(N_PATHS)

    for pi in range(N_PATHS):
        daily = paths[pi]
        path_sharpes[pi] = _sharpe(daily)
        path_dds[pi] = _max_dd(daily)
        path_cvar95[pi] = _cvar(daily, 0.05)
        path_cvar99[pi] = _cvar(daily, 0.01)

    def _pct(arr):
        return {f'P{p}': round(float(np.percentile(arr, p)), 3)
                for p in [5, 25, 50, 75, 95]}

    risk_summary = {
        'sharpe': _pct(path_sharpes),
        'max_dd': _pct(path_dds),
        'cvar95': _pct(path_cvar95),
        'cvar99': _pct(path_cvar99),
    }

    print(f"\n  {'Metric':<10s} {'P5':>10s} {'P25':>10s} {'P50':>10s} {'P75':>10s} {'P95':>10s}", flush=True)
    print(f"  {'─'*55}", flush=True)
    for metric, pcts in risk_summary.items():
        vals = [pcts[f'P{p}'] for p in [5, 25, 50, 75, 95]]
        if metric in ('max_dd', 'cvar95', 'cvar99'):
            print(f"  {metric:<10s} ${vals[0]:>9,.1f} ${vals[1]:>9,.1f} ${vals[2]:>9,.1f} "
                  f"${vals[3]:>9,.1f} ${vals[4]:>9,.1f}", flush=True)
        else:
            print(f"  {metric:<10s} {vals[0]:>10.3f} {vals[1]:>10.3f} {vals[2]:>10.3f} "
                  f"{vals[3]:>10.3f} {vals[4]:>10.3f}", flush=True)

    prob_positive_sharpe = float(np.mean(path_sharpes > 0) * 100)
    prob_cvar99_ok = float(np.mean(np.abs(path_cvar99) < CVAR99_LIMIT) * 100)
    print(f"\n  P(Sharpe > 0):        {prob_positive_sharpe:.1f}%", flush=True)
    print(f"  P(|CVaR99| < ${CVAR99_LIMIT:.0f}): {prob_cvar99_ok:.1f}%", flush=True)

    results['phase4_risk'] = {
        'percentiles': risk_summary,
        'prob_positive_sharpe': round(prob_positive_sharpe, 1),
        'prob_cvar99_ok': round(prob_cvar99_ok, 1),
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 5: Lot optimization (grid search)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print(f"  Phase 5: Lot Optimization (CVaR99 < ${CVAR99_LIMIT:.0f})", flush=True)
    print("=" * 70, flush=True)

    lot_grid = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15]
    n_combos = len(lot_grid) ** 4
    print(f"    Grid: {len(lot_grid)} lot values x 4 strategies = {n_combos} combos", flush=True)
    print(f"    Using {min(N_PATHS, 1000)} paths for screening...", flush=True)

    t_opt = time.time()
    rng_opt = np.random.RandomState(123)
    n_screen_paths = min(N_PATHS, 1000)

    best_sharpe = -999
    best_lots = dict(R89_LOTS)
    best_cvar99 = 0
    candidates = []

    for l8 in [0.01, 0.02, 0.03, 0.04]:
        for psar in [0.03, 0.05, 0.07, 0.09, 0.12]:
            for tsmom in [0.03, 0.05, 0.08, 0.10]:
                for sessbo in [0.03, 0.05, 0.08, 0.10]:
                    test_lots = {'L8_MAX': l8, 'PSAR': psar, 'TSMOM': tsmom, 'SESS_BO': sessbo}

                    screen_paths = generate_synthetic_paths(
                        strat_fits, strat_n_trades, test_lots,
                        n_screen_paths, min(n_days, 2520), rng_opt)

                    sharpes = np.array([_sharpe(screen_paths[i]) for i in range(n_screen_paths)])
                    cvars = np.array([_cvar(screen_paths[i], 0.01) for i in range(n_screen_paths)])

                    mean_sharpe = float(np.mean(sharpes))
                    mean_cvar99 = float(np.mean(cvars))

                    if abs(mean_cvar99) < CVAR99_LIMIT:
                        candidates.append({
                            'lots': dict(test_lots),
                            'mean_sharpe': round(mean_sharpe, 3),
                            'mean_cvar99': round(mean_cvar99, 2),
                        })
                        if mean_sharpe > best_sharpe:
                            best_sharpe = mean_sharpe
                            best_lots = dict(test_lots)
                            best_cvar99 = mean_cvar99

    opt_elapsed = time.time() - t_opt
    candidates.sort(key=lambda x: x['mean_sharpe'], reverse=True)
    top10 = candidates[:10]

    print(f"\n    Optimization done in {opt_elapsed:.1f}s", flush=True)
    print(f"    {len(candidates)}/{len(lot_grid)**4} combos pass CVaR99 constraint", flush=True)

    print(f"\n  Top 10 Allocations:", flush=True)
    print(f"  {'#':>3} {'L8':>5} {'PSAR':>5} {'TSMOM':>5} {'SESS':>5} {'Sharpe':>7} {'CVaR99':>9}", flush=True)
    print(f"  {'─'*45}", flush=True)
    for i, c in enumerate(top10, 1):
        lo = c['lots']
        print(f"  {i:>3} {lo['L8_MAX']:>5.2f} {lo['PSAR']:>5.2f} {lo['TSMOM']:>5.2f} "
              f"{lo['SESS_BO']:>5.2f} {c['mean_sharpe']:>7.3f} ${c['mean_cvar99']:>8,.1f}", flush=True)

    results['phase5_optimization'] = {
        'n_feasible': len(candidates),
        'best_lots': best_lots,
        'best_sharpe': round(best_sharpe, 3),
        'best_cvar99': round(best_cvar99, 2),
        'top10': top10,
        'elapsed_s': round(opt_elapsed, 1),
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 6: R89 vs optimal comparison
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 6: R89 Current vs Tail-Constrained Optimal", flush=True)
    print("=" * 70, flush=True)

    r89_daily = _trades_to_daily(
        [dict(t, pnl=t['pnl'] * R89_LOTS[sn] / UNIT_LOT)
         for sn in STRAT_ORDER for t in strat_trades[sn]])
    opt_daily = _trades_to_daily(
        [dict(t, pnl=t['pnl'] * best_lots[sn] / UNIT_LOT)
         for sn in STRAT_ORDER for t in strat_trades[sn]])

    def _full_stats(daily):
        return {
            'sharpe': round(_sharpe(daily), 3),
            'pnl': round(float(np.sum(daily)), 2),
            'max_dd': round(_max_dd(daily), 2),
            'cvar95': round(_cvar(daily, 0.05), 2),
            'cvar99': round(_cvar(daily, 0.01), 2),
        }

    r89_stats = _full_stats(r89_daily)
    opt_stats = _full_stats(opt_daily)

    print(f"\n  {'Metric':<12s} {'R89':>12s} {'Optimal':>12s} {'Delta':>12s}", flush=True)
    print(f"  {'─'*48}", flush=True)

    print(f"  {'Lots':<12s}", flush=True)
    for sn in STRAT_ORDER:
        delta = best_lots[sn] - R89_LOTS[sn]
        print(f"    {sn:<10s} {R89_LOTS[sn]:>10.2f} {best_lots[sn]:>12.2f} {delta:>+12.2f}", flush=True)

    for metric in ['sharpe', 'pnl', 'max_dd', 'cvar95', 'cvar99']:
        rv = r89_stats[metric]; ov = opt_stats[metric]
        delta = ov - rv
        if metric in ('pnl', 'max_dd', 'cvar95', 'cvar99'):
            print(f"  {metric:<12s} ${rv:>11,.2f} ${ov:>11,.2f} ${delta:>+11,.2f}", flush=True)
        else:
            print(f"  {metric:<12s} {rv:>12.3f} {ov:>12.3f} {delta:>+12.3f}", flush=True)

    results['phase6_comparison'] = {'r89': r89_stats, 'optimal': opt_stats, 'optimal_lots': best_lots}

    # ═══════════════════════════════════════════════════════════
    # Phase 7: Stress scenario (2x COVID drawdown)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 7: Stress Scenario — 2x COVID Drawdown", flush=True)
    print("=" * 70, flush=True)

    covid_start = pd.Timestamp("2020-02-20")
    covid_end = pd.Timestamp("2020-04-15")
    h1_covid = h1_df[(h1_df.index >= covid_start) & (h1_df.index <= covid_end)]

    covid_strat = {}
    covid_strat['PSAR'] = bt_psar(h1_covid, SPREAD, UNIT_LOT, CAPS['PSAR'])
    covid_strat['TSMOM'] = bt_tsmom(h1_covid, SPREAD, UNIT_LOT, CAPS['TSMOM'])
    covid_strat['SESS_BO'] = bt_sess_bo(h1_covid, SPREAD, UNIT_LOT, CAPS['SESS_BO'])
    covid_strat['L8_MAX'] = []

    stress_configs = {
        'R89_1x_COVID': (R89_LOTS, 1.0),
        'R89_2x_COVID': (R89_LOTS, 2.0),
        'OPT_1x_COVID': (best_lots, 1.0),
        'OPT_2x_COVID': (best_lots, 2.0),
    }

    print(f"\n  {'Scenario':<20s} {'PnL':>10s} {'MaxDD':>10s} {'CVaR99':>10s}", flush=True)
    print(f"  {'─'*52}", flush=True)

    stress_results = {}
    for label, (lots, mult) in stress_configs.items():
        covid_trades = []
        for sn in STRAT_ORDER:
            for t in covid_strat.get(sn, []):
                tc = dict(t)
                tc['pnl'] = t['pnl'] * (lots[sn] / UNIT_LOT) * mult
                covid_trades.append(tc)

        daily = _trades_to_daily(covid_trades)
        pnl = float(np.sum(daily))
        dd = _max_dd(daily)
        cv99 = _cvar(daily, 0.01)

        stress_results[label] = {
            'pnl': round(pnl, 2), 'max_dd': round(dd, 2), 'cvar99': round(cv99, 2),
        }
        print(f"  {label:<20s} ${pnl:>9,.2f} ${dd:>9,.2f} ${cv99:>9,.2f}", flush=True)

    results['phase7_stress'] = stress_results

    # ═══════════════════════════════════════════════════════════
    # Phase 8: Risk budget breakdown by strategy
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 8: Risk Budget Breakdown by Strategy", flush=True)
    print("=" * 70, flush=True)

    risk_budget = {}
    total_var = 0
    strat_vars = {}

    for sn in STRAT_ORDER:
        pnls = strat_pnls[sn]
        lot_mult = best_lots[sn] / UNIT_LOT
        scaled_pnls = pnls * lot_mult
        daily = _trades_to_daily(
            [dict(t, pnl=t['pnl'] * lot_mult) for t in strat_trades[sn]])

        var = float(np.var(daily, ddof=1)) if len(daily) > 1 else 0
        strat_vars[sn] = var
        total_var += var

        mean_pnl = float(np.mean(scaled_pnls))
        cvar99 = _cvar(daily, 0.01)
        sharpe = _sharpe(daily)

        risk_budget[sn] = {
            'lot': best_lots[sn],
            'n_trades': len(pnls),
            'mean_trade_pnl': round(mean_pnl, 2),
            'daily_variance': round(var, 2),
            'cvar99': round(cvar99, 2),
            'sharpe': round(sharpe, 3),
        }

    for sn in STRAT_ORDER:
        pct = strat_vars[sn] / total_var * 100 if total_var > 0 else 25
        risk_budget[sn]['risk_pct'] = round(pct, 1)

    print(f"\n  {'Strategy':<10s} {'Lot':>5s} {'Trades':>7s} {'MeanPnL':>9s} {'CVaR99':>9s} "
          f"{'Sharpe':>7s} {'Risk%':>7s}", flush=True)
    print(f"  {'─'*58}", flush=True)
    for sn in STRAT_ORDER:
        rb = risk_budget[sn]
        print(f"  {sn:<10s} {rb['lot']:>5.2f} {rb['n_trades']:>7d} ${rb['mean_trade_pnl']:>8,.2f} "
              f"${rb['cvar99']:>8,.2f} {rb['sharpe']:>7.3f} {rb['risk_pct']:>6.1f}%", flush=True)

    total_lot = sum(best_lots[sn] for sn in STRAT_ORDER)
    print(f"\n  Total lot: {total_lot:.2f}", flush=True)
    print(f"  Leverage vs R89: {total_lot / sum(R89_LOTS.values()):.2f}x", flush=True)

    results['phase8_risk_budget'] = risk_budget

    # ═══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)

    print("\n" + "=" * 80, flush=True)
    print("  R141 FINAL SUMMARY", flush=True)
    print("=" * 80, flush=True)

    print(f"\n  R89 allocation:     Sharpe={r89_stats['sharpe']:.3f}, "
          f"CVaR99=${r89_stats['cvar99']:,.2f}", flush=True)
    print(f"  Optimal allocation: Sharpe={opt_stats['sharpe']:.3f}, "
          f"CVaR99=${opt_stats['cvar99']:,.2f}", flush=True)

    sharpe_improved = opt_stats['sharpe'] > r89_stats['sharpe']
    cvar_ok = abs(opt_stats['cvar99']) < CVAR99_LIMIT

    print(f"\n  Sharpe improved: {'YES' if sharpe_improved else 'NO'} "
          f"({opt_stats['sharpe'] - r89_stats['sharpe']:+.3f})", flush=True)
    print(f"  CVaR99 within budget: {'YES' if cvar_ok else 'NO'} "
          f"(${abs(opt_stats['cvar99']):,.0f} vs ${CVAR99_LIMIT:,.0f} limit)", flush=True)

    print(f"\n  Optimal lot allocation:", flush=True)
    for sn in STRAT_ORDER:
        delta = best_lots[sn] - R89_LOTS[sn]
        print(f"    {sn:>10s}: {best_lots[sn]:.2f} (was {R89_LOTS[sn]:.2f}, {delta:+.2f})", flush=True)

    print(f"\n  Risk budget: " + ", ".join(
        f"{sn}={risk_budget[sn]['risk_pct']:.0f}%" for sn in STRAT_ORDER), flush=True)

    stress_2x = stress_results.get('OPT_2x_COVID', {})
    print(f"\n  2x COVID stress (optimal): PnL=${stress_2x.get('pnl', 0):,.0f}, "
          f"MaxDD=${stress_2x.get('max_dd', 0):,.0f}", flush=True)

    verdict = "ADOPT OPTIMAL" if sharpe_improved and cvar_ok else "KEEP R89"
    results['verdict'] = verdict
    print(f"\n  Verdict: {verdict}", flush=True)

    out_file = OUTPUT_DIR / "r141_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"  Saved: {out_file}", flush=True)
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
