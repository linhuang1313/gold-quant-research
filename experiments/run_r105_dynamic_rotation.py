#!/usr/bin/env python3
"""
R105 — Dynamic Strategy Rotation
===================================
Tests dynamic lot-weight rotation based on recent strategy performance.

  Phase 1: Momentum weighting (allocate more to recent winners)
  Phase 2: Mean-reversion weighting (allocate more to recent losers)
  Phase 3: Combined momentum + mean-rev blend
  Phase 4: Regime-conditional rotation
  Phase 5: K-Fold validation + comparison vs fixed weights
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

from backtest.runner import DataBundle, load_m15, load_h1_aligned, H1_CSV_PATH
from backtest.runner import run_variant, LIVE_PARITY_KWARGS

OUTPUT_DIR = Path("results/r105_dynamic_rotation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01
CAPITAL = 5000

CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.09, 'SESS_BO': 0.09}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]

TOTAL_LOT_BUDGET = sum(R89_LOTS.values())  # 0.29

t0 = time.time()

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

def _run_exit_with_cap(pos, i, hi, lo, cl, spread, lot, pv, times,
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


# ═══════════════════════════════════════════════════════════════
# Strategy backtests
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
            psar[i] = min(psar[i], l[i-1], l[max(0,i-2)])
            if l[i] < psar[i]:
                rising = False; psar[i] = ep; ep = l[i]; af = af_step
            else:
                if h[i] > ep: ep = h[i]; af = min(af + af_step, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], h[i-1], h[max(0,i-2)])
            if h[i] > psar[i]:
                rising = True; psar[i] = ep; ep = h[i]; af = af_step
            else:
                if l[i] < ep: ep = l[i]; af = min(af + af_step, af_max)
    df['PSAR'] = psar

def bt_psar(h1_df, spread, lot, maxloss_cap=0,
            sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04, max_hold=20):
    df = h1_df.copy(); add_psar(df); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR', 'PSAR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; psar = df['PSAR'].values; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if psar[i-1] > c[i-1] and psar[i] < c[i]:
            pos = {'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif psar[i-1] < c[i-1] and psar[i] > c[i]:
            pos = {'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
    return trades

def bt_tsmom(h1_df, spread, lot, maxloss_cap=0,
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
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
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
            pos = {'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
    return trades

def bt_sess_bo(h1_df, spread, lot, maxloss_cap=0,
               session_hour=12, lookback=4, sl_atr=4.5, tp_atr=4.0,
               trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(lookback, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if hours[i] != session_hour: continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif c[i] < ll:
            pos = {'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
    return trades

def bt_l8_max(data_bundle, spread, lot, maxloss_cap=35):
    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': maxloss_cap,
           'spread_cost': spread, 'initial_capital': 2000,
           'min_lot_size': lot, 'max_lot_size': lot}
    result = run_variant(data_bundle, "L8_MAX", verbose=False, **kw)
    raw_trades = result.get('_trades', [])
    trades = []
    for t in raw_trades:
        trades.append({
            'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
            'entry_time': t.entry_time, 'exit_time': t.exit_time,
            'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0,
        })
    return trades


# ═══════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════

def trades_to_daily_series(trades):
    if not trades: return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))

def sharpe(arr):
    if len(arr) < 10: return 0.0
    s = np.std(arr, ddof=1)
    return float(np.mean(arr) / s * np.sqrt(252)) if s > 0 else 0.0

def max_dd(arr):
    if len(arr) == 0: return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())

def build_portfolio_daily(unit_dailies, lots):
    all_dates = set()
    for ds in unit_dailies.values():
        all_dates.update(ds.index)
    all_dates = sorted(all_dates)
    idx = pd.DatetimeIndex(all_dates)
    portfolio = np.zeros(len(idx))
    for name in STRAT_ORDER:
        if name not in unit_dailies: continue
        ds = unit_dailies[name]
        multiplier = lots[name] / UNIT_LOT
        aligned = ds.reindex(idx, fill_value=0.0).values * multiplier
        portfolio += aligned
    return portfolio

def portfolio_metrics(daily_arr):
    return {
        'sharpe': round(sharpe(daily_arr), 3),
        'pnl': round(float(np.sum(daily_arr)), 2),
        'max_dd': round(max_dd(daily_arr), 2),
    }


# ═══════════════════════════════════════════════════════════════
# Dynamic rotation helpers
# ═══════════════════════════════════════════════════════════════

def softmax(x):
    """Numerically stable softmax."""
    x = np.array(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def compute_rolling_sharpe_matrix(unit_dailies, lookback):
    """Build a DataFrame of rolling Sharpe ratios per strategy, indexed by date.
    Returns (dates, matrix) where matrix is shape (n_dates, n_strats)."""
    all_dates = set()
    for ds in unit_dailies.values():
        all_dates.update(ds.index)
    all_dates = sorted(all_dates)
    idx = pd.DatetimeIndex(all_dates)

    strat_daily = {}
    for name in STRAT_ORDER:
        if name in unit_dailies:
            strat_daily[name] = unit_dailies[name].reindex(idx, fill_value=0.0)
        else:
            strat_daily[name] = pd.Series(0.0, index=idx)

    n = len(idx)
    matrix = np.zeros((n, len(STRAT_ORDER)))
    for si, name in enumerate(STRAT_ORDER):
        vals = strat_daily[name].values
        for d in range(lookback, n):
            window = vals[d - lookback:d]
            s = np.std(window, ddof=1)
            if s > 0 and len(window) >= 10:
                matrix[d, si] = float(np.mean(window) / s * np.sqrt(252))
            else:
                matrix[d, si] = 0.0

    return idx, matrix


def build_dynamic_portfolio(unit_dailies, weight_matrix, date_index):
    """Build portfolio daily PnL using time-varying weights.
    weight_matrix: (n_dates, n_strats) — each row sums to 1, gives fraction of TOTAL_LOT_BUDGET.
    Returns daily PnL array."""
    n = len(date_index)
    strat_daily = {}
    for name in STRAT_ORDER:
        if name in unit_dailies:
            strat_daily[name] = unit_dailies[name].reindex(date_index, fill_value=0.0).values
        else:
            strat_daily[name] = np.zeros(n)

    portfolio = np.zeros(n)
    for d in range(n):
        for si, name in enumerate(STRAT_ORDER):
            lot = weight_matrix[d, si] * TOTAL_LOT_BUDGET
            multiplier = lot / UNIT_LOT
            portfolio[d] += strat_daily[name][d] * multiplier

    return portfolio


def compute_turnover(weight_matrix):
    """Average daily absolute change in weights — proxy for rebalancing cost."""
    if len(weight_matrix) < 2:
        return 0.0
    diffs = np.abs(np.diff(weight_matrix, axis=0))
    return float(np.mean(np.sum(diffs, axis=1)))


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  R105 — Dynamic Strategy Rotation")
    print("=" * 80)

    print("\n  Loading data...")
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    bundle = DataBundle.load_custom()
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0].date()} ~ {h1_df.index[-1].date()})")

    # ─── Run base strategies at unit lot ─────────────────────────
    print("\n  Running 4 strategies at unit lot (0.01)...")
    base_trades = {}
    base_trades['PSAR'] = bt_psar(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['PSAR'])
    base_trades['TSMOM'] = bt_tsmom(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['TSMOM'])
    base_trades['SESS_BO'] = bt_sess_bo(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['SESS_BO'])
    base_trades['L8_MAX'] = bt_l8_max(bundle, SPREAD, UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])

    unit_dailies = {}
    for name in STRAT_ORDER:
        unit_dailies[name] = trades_to_daily_series(base_trades[name])
        n_t = len(base_trades[name])
        pnl = sum(t['pnl'] for t in base_trades[name])
        print(f"    {name:10s}: {n_t:5d} trades, unit PnL=${pnl:,.2f}")

    # Fixed-lot baseline
    fixed_daily = build_portfolio_daily(unit_dailies, R89_LOTS)
    fixed_m = portfolio_metrics(fixed_daily)
    print(f"\n  Fixed R89 Baseline: Sharpe={fixed_m['sharpe']}, "
          f"PnL=${fixed_m['pnl']:,.2f}, MaxDD=${fixed_m['max_dd']:,.2f}")
    print(f"  Total lot budget: {TOTAL_LOT_BUDGET:.2f}")

    results = {'experiment': 'R105 Dynamic Strategy Rotation',
               'fixed_baseline': fixed_m,
               'total_lot_budget': TOTAL_LOT_BUDGET}

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Momentum Weighting
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 1: Momentum Weighting (allocate to recent winners)")
    print("=" * 60)

    lookbacks = [30, 60, 90]
    phase1 = {}
    best_p1_lb = None
    best_p1_sharpe = -999

    for lb in lookbacks:
        date_idx, sharpe_mat = compute_rolling_sharpe_matrix(unit_dailies, lb)
        n = len(date_idx)
        weights = np.zeros((n, len(STRAT_ORDER)))

        for d in range(n):
            if d < lb:
                weights[d] = 1.0 / len(STRAT_ORDER)
            else:
                weights[d] = softmax(sharpe_mat[d])

        port = build_dynamic_portfolio(unit_dailies, weights, date_idx)
        m = portfolio_metrics(port)
        turnover = compute_turnover(weights)
        delta_sh = m['sharpe'] - fixed_m['sharpe']
        phase1[f"mom_lb{lb}"] = {**m, 'turnover': round(turnover, 4)}

        print(f"    Lookback={lb:3d}d: Sharpe={m['sharpe']:6.3f} (Δ={delta_sh:+.3f}), "
              f"PnL=${m['pnl']:,.2f}, MaxDD=${m['max_dd']:,.2f}, Turnover={turnover:.4f}")

        if m['sharpe'] > best_p1_sharpe:
            best_p1_sharpe = m['sharpe']
            best_p1_lb = lb

    results['phase1_momentum'] = phase1
    print(f"  -> Best momentum lookback: {best_p1_lb}d (Sharpe={best_p1_sharpe:.3f})")

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Mean-Reversion Weighting
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 2: Mean-Reversion Weighting (allocate to recent losers)")
    print("=" * 60)

    phase2 = {}
    best_p2_lb = None
    best_p2_sharpe = -999

    for lb in lookbacks:
        date_idx, sharpe_mat = compute_rolling_sharpe_matrix(unit_dailies, lb)
        n = len(date_idx)
        weights = np.zeros((n, len(STRAT_ORDER)))

        for d in range(n):
            if d < lb:
                weights[d] = 1.0 / len(STRAT_ORDER)
            else:
                weights[d] = softmax(-sharpe_mat[d])

        port = build_dynamic_portfolio(unit_dailies, weights, date_idx)
        m = portfolio_metrics(port)
        turnover = compute_turnover(weights)
        delta_sh = m['sharpe'] - fixed_m['sharpe']
        phase2[f"meanrev_lb{lb}"] = {**m, 'turnover': round(turnover, 4)}

        print(f"    Lookback={lb:3d}d: Sharpe={m['sharpe']:6.3f} (Δ={delta_sh:+.3f}), "
              f"PnL=${m['pnl']:,.2f}, MaxDD=${m['max_dd']:,.2f}, Turnover={turnover:.4f}")

        if m['sharpe'] > best_p2_sharpe:
            best_p2_sharpe = m['sharpe']
            best_p2_lb = lb

    results['phase2_meanrev'] = phase2
    print(f"  -> Best mean-rev lookback: {best_p2_lb}d (Sharpe={best_p2_sharpe:.3f})")

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Combined Momentum + Mean-Rev Blend
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 3: Combined Blend (alpha * mom + (1-alpha) * meanrev)")
    print("=" * 60)

    # Use best lookbacks from phases 1 and 2
    mom_lb = best_p1_lb
    rev_lb = best_p2_lb
    print(f"    Using mom_lookback={mom_lb}, meanrev_lookback={rev_lb}")

    date_idx_mom, sharpe_mat_mom = compute_rolling_sharpe_matrix(unit_dailies, mom_lb)
    date_idx_rev, sharpe_mat_rev = compute_rolling_sharpe_matrix(unit_dailies, rev_lb)

    common_dates = sorted(set(date_idx_mom).intersection(set(date_idx_rev)))
    common_idx = pd.DatetimeIndex(common_dates)

    mom_lookup = {d: i for i, d in enumerate(date_idx_mom)}
    rev_lookup = {d: i for i, d in enumerate(date_idx_rev)}

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    phase3 = {}
    best_p3_alpha = None
    best_p3_sharpe = -999

    for alpha in alphas:
        n = len(common_idx)
        weights = np.zeros((n, len(STRAT_ORDER)))

        for d in range(n):
            dt = common_idx[d]
            mi = mom_lookup.get(dt)
            ri = rev_lookup.get(dt)
            if mi is None or ri is None or mi < mom_lb or ri < rev_lb:
                weights[d] = 1.0 / len(STRAT_ORDER)
            else:
                mom_w = softmax(sharpe_mat_mom[mi])
                rev_w = softmax(-sharpe_mat_rev[ri])
                weights[d] = alpha * mom_w + (1 - alpha) * rev_w

        strat_daily_aligned = {}
        for name in STRAT_ORDER:
            if name in unit_dailies:
                strat_daily_aligned[name] = unit_dailies[name].reindex(common_idx, fill_value=0.0).values
            else:
                strat_daily_aligned[name] = np.zeros(n)

        port = np.zeros(n)
        for d_i in range(n):
            for si, name in enumerate(STRAT_ORDER):
                lot = weights[d_i, si] * TOTAL_LOT_BUDGET
                multiplier = lot / UNIT_LOT
                port[d_i] += strat_daily_aligned[name][d_i] * multiplier

        m = portfolio_metrics(port)
        turnover = compute_turnover(weights)
        delta_sh = m['sharpe'] - fixed_m['sharpe']
        phase3[f"alpha_{alpha:.2f}"] = {**m, 'turnover': round(turnover, 4)}

        print(f"    α={alpha:.2f}: Sharpe={m['sharpe']:6.3f} (Δ={delta_sh:+.3f}), "
              f"PnL=${m['pnl']:,.2f}, MaxDD=${m['max_dd']:,.2f}, Turnover={turnover:.4f}")

        if m['sharpe'] > best_p3_sharpe:
            best_p3_sharpe = m['sharpe']
            best_p3_alpha = alpha

    results['phase3_blend'] = phase3
    print(f"  -> Best blend alpha: {best_p3_alpha:.2f} (Sharpe={best_p3_sharpe:.3f})")

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Regime-Conditional Rotation
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 4: Regime-Conditional Rotation")
    print("=" * 60)

    vol_proxy = None
    csv_path = Path("data/external/aligned_daily.csv")
    if csv_path.exists():
        try:
            ext_df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
            if 'VIX_Close' in ext_df.columns:
                vix = ext_df['VIX_Close'].dropna()
                if len(vix) >= 100:
                    vol_proxy = vix
                    print("  Using VIX as volatility proxy")
        except Exception:
            pass

    if vol_proxy is None:
        print("  VIX not available — using gold ATR percentile rank as regime proxy")
        daily_ohlc = pd.DataFrame({
            'High': h1_df['High'].resample('D').max(),
            'Low': h1_df['Low'].resample('D').min(),
            'Close': h1_df['Close'].resample('D').last()
        }).dropna()
        vol_proxy = compute_atr(daily_ohlc, period=14).dropna()

    vol_pctrank = vol_proxy.rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    ).dropna()

    thresholds = [60, 70, 80, 90]
    phase4 = {}
    best_p4_thresh = None
    best_p4_sharpe = -999

    for thresh in thresholds:
        high_pct = thresh / 100.0
        low_pct = 1.0 - high_pct

        all_dates_set = set()
        for ds in unit_dailies.values():
            all_dates_set.update(ds.index)
        all_dates_sorted = sorted(all_dates_set)
        date_idx_p4 = pd.DatetimeIndex(all_dates_sorted)
        n = len(date_idx_p4)

        date_idx_mom_p4, sharpe_mat_mom_p4 = compute_rolling_sharpe_matrix(unit_dailies, best_p1_lb)
        date_idx_rev_p4, sharpe_mat_rev_p4 = compute_rolling_sharpe_matrix(unit_dailies, best_p2_lb)

        mom_lookup_p4 = {d: i for i, d in enumerate(date_idx_mom_p4)}
        rev_lookup_p4 = {d: i for i, d in enumerate(date_idx_rev_p4)}

        weights = np.zeros((n, len(STRAT_ORDER)))

        for d in range(n):
            dt = date_idx_p4[d]
            vol_idx = vol_pctrank.index.searchsorted(dt)
            vol_idx = min(max(vol_idx - 1, 0), len(vol_pctrank) - 1)
            pct_val = vol_pctrank.iloc[vol_idx]

            mi = mom_lookup_p4.get(dt)
            ri = rev_lookup_p4.get(dt)

            if mi is None or ri is None or mi < best_p1_lb or ri < best_p2_lb:
                weights[d] = 1.0 / len(STRAT_ORDER)
            elif pct_val >= high_pct:
                weights[d] = softmax(-sharpe_mat_rev_p4[ri])
            elif pct_val <= low_pct:
                weights[d] = softmax(sharpe_mat_mom_p4[mi])
            else:
                mom_w = softmax(sharpe_mat_mom_p4[mi])
                rev_w = softmax(-sharpe_mat_rev_p4[ri])
                weights[d] = 0.5 * mom_w + 0.5 * rev_w

        port = build_dynamic_portfolio(unit_dailies, weights, date_idx_p4)
        m = portfolio_metrics(port)
        turnover = compute_turnover(weights)
        delta_sh = m['sharpe'] - fixed_m['sharpe']
        phase4[f"thresh_{thresh}"] = {**m, 'turnover': round(turnover, 4)}

        print(f"    Threshold={thresh}th pctile: Sharpe={m['sharpe']:6.3f} (Δ={delta_sh:+.3f}), "
              f"PnL=${m['pnl']:,.2f}, MaxDD=${m['max_dd']:,.2f}, Turnover={turnover:.4f}")

        if m['sharpe'] > best_p4_sharpe:
            best_p4_sharpe = m['sharpe']
            best_p4_thresh = thresh

    results['phase4_regime'] = phase4
    print(f"  -> Best threshold: {best_p4_thresh}th pctile (Sharpe={best_p4_sharpe:.3f})")

    # ═══════════════════════════════════════════════════════════
    # Phase 5: K-Fold Validation + Comparison vs Fixed Weights
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 5: K-Fold Validation vs Fixed Weights")
    print("=" * 60)

    # Determine best rotation method across phases
    candidates = [
        ('momentum', f'lb{best_p1_lb}', best_p1_sharpe, best_p1_lb),
        ('meanrev', f'lb{best_p2_lb}', best_p2_sharpe, best_p2_lb),
        ('blend', f'alpha{best_p3_alpha:.2f}', best_p3_sharpe, None),
        ('regime', f'thresh{best_p4_thresh}', best_p4_sharpe, None),
    ]
    candidates.sort(key=lambda x: x[2], reverse=True)
    winner_method = candidates[0][0]
    winner_label = candidates[0][1]
    winner_sharpe = candidates[0][2]

    print(f"\n  Phase ranking:")
    for rank, (method, label, sh, _) in enumerate(candidates):
        marker = " <-- WINNER" if rank == 0 else ""
        print(f"    #{rank+1}: {method:12s} ({label:15s}) Sharpe={sh:6.3f}{marker}")

    print(f"\n  K-Fold validation: {winner_method} ({winner_label}) vs fixed R89 lots")

    kfold_results = []

    for fname, start, end in FOLDS:
        print(f"\n  [{fname}] {start} ~ {end}")

        # Get fold-specific data
        fold_h1 = h1_df[(h1_df.index >= start) & (h1_df.index < end)]
        if len(fold_h1) < 100:
            kfold_results.append({
                'fold': fname, 'fixed_sharpe': 0.0, 'dynamic_sharpe': 0.0,
                'delta': 0.0, 'turnover': 0.0
            })
            print(f"    Skipped (too few bars: {len(fold_h1)})")
            continue

        # Run strategies on fold data
        fold_trades = {}
        fold_trades['PSAR'] = bt_psar(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=CAPS['PSAR'])
        fold_trades['TSMOM'] = bt_tsmom(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=CAPS['TSMOM'])
        fold_trades['SESS_BO'] = bt_sess_bo(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=CAPS['SESS_BO'])
        fold_trades['L8_MAX'] = bt_l8_max(bundle, SPREAD, UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])
        fold_trades['L8_MAX'] = [t for t in fold_trades['L8_MAX']
                                 if start <= str(pd.Timestamp(t['exit_time']).date()) < end]

        fold_unit_dailies = {n: trades_to_daily_series(fold_trades[n]) for n in STRAT_ORDER}

        # Fixed baseline
        fold_fixed = build_portfolio_daily(fold_unit_dailies, R89_LOTS)
        fixed_sh = sharpe(fold_fixed)

        # Calibrate rotation on training folds (all other folds)
        train_unit_dailies = {}
        for tname, tstart, tend in FOLDS:
            if tname == fname:
                continue
            for name in STRAT_ORDER:
                train_trades_s = [t for t in base_trades[name]
                                  if tstart <= str(pd.Timestamp(t['exit_time']).date()) < tend]
                if name not in train_unit_dailies:
                    train_unit_dailies[name] = trades_to_daily_series(train_trades_s)
                else:
                    new_ds = trades_to_daily_series(train_trades_s)
                    train_unit_dailies[name] = pd.concat([train_unit_dailies[name], new_ds])

        # Build dynamic weights for this fold
        if winner_method == 'momentum':
            lb = best_p1_lb
            date_idx_f, sharpe_mat_f = compute_rolling_sharpe_matrix(fold_unit_dailies, min(lb, 20))
            n_f = len(date_idx_f)
            w_f = np.zeros((n_f, len(STRAT_ORDER)))
            eff_lb = min(lb, 20)
            for d in range(n_f):
                if d < eff_lb:
                    # Warm-start from training data rolling sharpe
                    train_idx, train_smat = compute_rolling_sharpe_matrix(train_unit_dailies, lb)
                    if len(train_smat) > 0:
                        w_f[d] = softmax(train_smat[-1])
                    else:
                        w_f[d] = 1.0 / len(STRAT_ORDER)
                else:
                    w_f[d] = softmax(sharpe_mat_f[d])

        elif winner_method == 'meanrev':
            lb = best_p2_lb
            date_idx_f, sharpe_mat_f = compute_rolling_sharpe_matrix(fold_unit_dailies, min(lb, 20))
            n_f = len(date_idx_f)
            w_f = np.zeros((n_f, len(STRAT_ORDER)))
            eff_lb = min(lb, 20)
            for d in range(n_f):
                if d < eff_lb:
                    train_idx, train_smat = compute_rolling_sharpe_matrix(train_unit_dailies, lb)
                    if len(train_smat) > 0:
                        w_f[d] = softmax(-train_smat[-1])
                    else:
                        w_f[d] = 1.0 / len(STRAT_ORDER)
                else:
                    w_f[d] = softmax(-sharpe_mat_f[d])

        elif winner_method == 'blend':
            alpha = best_p3_alpha
            lb_m = best_p1_lb
            lb_r = best_p2_lb
            eff_lb = min(max(lb_m, lb_r), 20)
            date_idx_f_m, smat_m = compute_rolling_sharpe_matrix(fold_unit_dailies, min(lb_m, 20))
            date_idx_f_r, smat_r = compute_rolling_sharpe_matrix(fold_unit_dailies, min(lb_r, 20))

            common_f = sorted(set(date_idx_f_m).intersection(set(date_idx_f_r)))
            date_idx_f = pd.DatetimeIndex(common_f)
            n_f = len(date_idx_f)
            w_f = np.zeros((n_f, len(STRAT_ORDER)))

            m_lookup = {d: i for i, d in enumerate(date_idx_f_m)}
            r_lookup = {d: i for i, d in enumerate(date_idx_f_r)}

            for d in range(n_f):
                dt = date_idx_f[d]
                mi = m_lookup.get(dt)
                ri = r_lookup.get(dt)
                if mi is None or ri is None or mi < min(lb_m, 20) or ri < min(lb_r, 20):
                    w_f[d] = 1.0 / len(STRAT_ORDER)
                else:
                    mom_w = softmax(smat_m[mi])
                    rev_w = softmax(-smat_r[ri])
                    w_f[d] = alpha * mom_w + (1 - alpha) * rev_w

        else:  # regime
            thresh = best_p4_thresh
            high_pct = thresh / 100.0
            low_pct = 1.0 - high_pct
            lb_m = best_p1_lb
            lb_r = best_p2_lb
            date_idx_f_m, smat_m = compute_rolling_sharpe_matrix(fold_unit_dailies, min(lb_m, 20))
            date_idx_f_r, smat_r = compute_rolling_sharpe_matrix(fold_unit_dailies, min(lb_r, 20))

            common_f = sorted(set(date_idx_f_m).intersection(set(date_idx_f_r)))
            date_idx_f = pd.DatetimeIndex(common_f)
            n_f = len(date_idx_f)
            w_f = np.zeros((n_f, len(STRAT_ORDER)))

            m_lookup = {d: i for i, d in enumerate(date_idx_f_m)}
            r_lookup = {d: i for i, d in enumerate(date_idx_f_r)}

            for d in range(n_f):
                dt = date_idx_f[d]
                mi = m_lookup.get(dt)
                ri = r_lookup.get(dt)

                vol_i = vol_pctrank.index.searchsorted(dt)
                vol_i = min(max(vol_i - 1, 0), len(vol_pctrank) - 1)
                pct_val = vol_pctrank.iloc[vol_i]

                if mi is None or ri is None or mi < min(lb_m, 20) or ri < min(lb_r, 20):
                    w_f[d] = 1.0 / len(STRAT_ORDER)
                elif pct_val >= high_pct:
                    w_f[d] = softmax(-smat_r[ri])
                elif pct_val <= low_pct:
                    w_f[d] = softmax(smat_m[mi])
                else:
                    mom_w = softmax(smat_m[mi])
                    rev_w = softmax(-smat_r[ri])
                    w_f[d] = 0.5 * mom_w + 0.5 * rev_w

        # Build dynamic portfolio for fold
        if winner_method in ('blend', 'regime'):
            fold_ud_aligned = {}
            for name in STRAT_ORDER:
                if name in fold_unit_dailies:
                    fold_ud_aligned[name] = fold_unit_dailies[name].reindex(date_idx_f, fill_value=0.0).values
                else:
                    fold_ud_aligned[name] = np.zeros(n_f)
            fold_dynamic = np.zeros(n_f)
            for d_i in range(n_f):
                for si, name in enumerate(STRAT_ORDER):
                    lot = w_f[d_i, si] * TOTAL_LOT_BUDGET
                    multiplier = lot / UNIT_LOT
                    fold_dynamic[d_i] += fold_ud_aligned[name][d_i] * multiplier
        else:
            fold_dynamic = build_dynamic_portfolio(fold_unit_dailies, w_f, date_idx_f)

        dynamic_sh = sharpe(fold_dynamic)
        turnover_f = compute_turnover(w_f)
        delta = dynamic_sh - fixed_sh

        kfold_results.append({
            'fold': fname, 'fixed_sharpe': round(fixed_sh, 3),
            'dynamic_sharpe': round(dynamic_sh, 3),
            'delta': round(delta, 3), 'turnover': round(turnover_f, 4)
        })

        print(f"    Fixed Sharpe={fixed_sh:6.3f} | Dynamic Sharpe={dynamic_sh:6.3f} | "
              f"Δ={delta:+.3f} | Turnover={turnover_f:.4f}")

    # K-Fold summary
    print(f"\n  {'─'*65}")
    print(f"  {'Fold':<8} {'Fixed':>10} {'Dynamic':>10} {'Delta':>10} {'Turnover':>10}")
    print(f"  {'─'*65}")
    for r in kfold_results:
        print(f"  {r['fold']:<8} {r['fixed_sharpe']:10.3f} {r['dynamic_sharpe']:10.3f} "
              f"{r['delta']:+10.3f} {r['turnover']:10.4f}")

    fixed_sharpes = [r['fixed_sharpe'] for r in kfold_results]
    dynamic_sharpes = [r['dynamic_sharpe'] for r in kfold_results]
    turnovers = [r['turnover'] for r in kfold_results]
    mean_fixed = np.mean(fixed_sharpes)
    mean_dynamic = np.mean(dynamic_sharpes)
    mean_turnover = np.mean(turnovers)
    dynamic_wins = sum(1 for r in kfold_results if r['delta'] > 0)

    print(f"  {'─'*65}")
    print(f"  {'Mean':<8} {mean_fixed:10.3f} {mean_dynamic:10.3f} "
          f"{mean_dynamic - mean_fixed:+10.3f} {mean_turnover:10.4f}")
    print(f"\n  Dynamic wins: {dynamic_wins}/{len(FOLDS)} folds")

    kfold_pass = dynamic_wins >= 4

    results['phase5_kfold'] = {
        'winner_method': winner_method,
        'winner_label': winner_label,
        'folds': kfold_results,
        'fixed_mean_sharpe': round(mean_fixed, 3),
        'dynamic_mean_sharpe': round(mean_dynamic, 3),
        'mean_turnover': round(mean_turnover, 4),
        'dynamic_wins': dynamic_wins,
        'pass_4of6': kfold_pass,
    }

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    print("\n" + "=" * 80)
    print("  R105 SUMMARY — Dynamic Strategy Rotation")
    print("=" * 80)

    print(f"\n  Fixed R89 Baseline:  Sharpe={fixed_m['sharpe']:.3f}, "
          f"PnL=${fixed_m['pnl']:,.2f}, MaxDD=${fixed_m['max_dd']:,.2f}")

    print(f"\n  {'Method':<20} {'Sharpe':>8} {'Delta':>8} {'PnL':>12} {'MaxDD':>10}")
    print(f"  {'─'*60}")

    summary_rows = []
    best_p1_key = f"mom_lb{best_p1_lb}"
    best_p2_key = f"meanrev_lb{best_p2_lb}"
    best_p3_key = f"alpha_{best_p3_alpha:.2f}"
    best_p4_key = f"thresh_{best_p4_thresh}"

    summary_rows.append(('Momentum', best_p1_key, phase1[best_p1_key]))
    summary_rows.append(('Mean-Rev', best_p2_key, phase2[best_p2_key]))
    summary_rows.append(('Blend', best_p3_key, phase3[best_p3_key]))
    summary_rows.append(('Regime-Cond', best_p4_key, phase4[best_p4_key]))

    for label, key, m in summary_rows:
        d = m['sharpe'] - fixed_m['sharpe']
        t_val = m.get('turnover', 0)
        print(f"  {label:<20} {m['sharpe']:8.3f} {d:+8.3f} ${m['pnl']:>10,.2f} ${m['max_dd']:>8,.2f}")

    print(f"\n  K-Fold: Dynamic wins {dynamic_wins}/{len(FOLDS)} folds "
          f"(mean Sharpe: fixed={mean_fixed:.3f}, dynamic={mean_dynamic:.3f})")
    print(f"  Mean daily turnover: {mean_turnover:.4f}")

    if kfold_pass:
        print(f"\n  RECOMMENDATION: Use {winner_method} rotation ({winner_label}) — "
              f"wins {dynamic_wins}/{len(FOLDS)} folds, "
              f"mean Sharpe {mean_dynamic:.3f} vs fixed {mean_fixed:.3f}")
    else:
        print(f"\n  RECOMMENDATION: Keep fixed R89 lots — "
              f"dynamic rotation wins only {dynamic_wins}/{len(FOLDS)} folds")

    results['elapsed_s'] = round(elapsed, 1)
    results['recommendation'] = (
        f"Use {winner_method} rotation ({winner_label}), "
        f"wins {dynamic_wins}/{len(FOLDS)} folds"
        if kfold_pass
        else f"Keep fixed R89 lots, dynamic wins only {dynamic_wins}/{len(FOLDS)} folds"
    )

    out_file = OUTPUT_DIR / "r105_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Saved: {out_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
