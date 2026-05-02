#!/usr/bin/env python3
"""
R90-B — Factor-Enhanced Signal Filtering
==========================================
Phase B of the R90 external-data test plan.

Step 1: Factor-Signal Correlation Scan
  - Run each strategy on full H1 data at live lots/caps
  - Convert trades to daily PnL, correlate with ~30 factor columns from aligned_daily
  - Rank by |correlation|, select top 10

Step 2: Conditional Entry Filter Grid  (parallel)
  - 4 strategies x top-10 factors x 5 quantile thresholds
  - Backtest with entry only on allowed days, compare Sharpe vs unfiltered baseline

Step 3: Walk-Forward Validation
  - Top-5 filter combos -> 6-fold walk-forward
  - Recompute thresholds on train, apply on test

Output: results/r90_external_data/r90b_factor_filter/r90b_results.json
"""
import sys, os, io, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from scipy.stats import spearmanr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r90_external_data/r90b_factor_filter")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30

CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
LIVE_LOTS = {'L8_MAX': 0.05, 'TSMOM': 0.04, 'SESS_BO': 0.02, 'PSAR': 0.01}

STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']

QUANTILE_THRESHOLDS = [0.20, 0.35, 0.50, 0.65, 0.80]

FACTOR_COLS = [
    'VIX_Close', 'VIX_SMA20', 'VIX_Zscore',
    'DXY_Close', 'DXY_Mom5', 'DXY_Mom20', 'DXY_SMA50',
    'US10Y_Close', 'US10Y_Change5', 'US10Y_Change20',
    'YIELD_CURVE_10Y2Y',
    'REAL_YIELD_DFII10', 'REAL_YIELD_Change5', 'REAL_YIELD_Change20', 'REAL_YIELD_SMA20',
    'GLD_Vol_SMA20', 'GLD_Vol_Ratio',
    'COT_MM_Net', 'COT_MM_Net_Zscore', 'COT_MM_Net_Pct',
    'CREDIT_STRESS',
    'COPPER_GOLD_RATIO', 'CG_RATIO_Mom20',
    'CRUDE_Mom5', 'CRUDE_Mom20',
    'USDJPY_Mom5', 'USDCNH_Mom5', 'USDCNH_Mom20',
    'M2_YoY', 'M2_Mom3M',
    'SPX_Mom5', 'HYG_Mom5',
    'RISK_APPETITE', 'RISK_APPETITE_Z',
    'FED_FUNDS_DFF', 'GVZ_Close',
]

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]

N_WORKERS = 20


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


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


def add_psar(df, af_start=0.01, af_max=0.05):
    df = df.copy(); n = len(df)
    psar = np.zeros(n); direction = np.ones(n)
    af = af_start; ep = df['High'].iloc[0]; psar[0] = df['Low'].iloc[0]
    for i in range(1, n):
        prev = psar[i-1]
        if direction[i-1] == 1:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], df['Low'].iloc[i-1], df['Low'].iloc[max(0, i-2)])
            if df['Low'].iloc[i] < psar[i]:
                direction[i] = -1; psar[i] = ep; ep = df['Low'].iloc[i]; af = af_start
            else:
                direction[i] = 1
                if df['High'].iloc[i] > ep:
                    ep = df['High'].iloc[i]; af = min(af+af_start, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], df['High'].iloc[i-1], df['High'].iloc[max(0, i-2)])
            if df['High'].iloc[i] > psar[i]:
                direction[i] = 1; psar[i] = ep; ep = df['High'].iloc[i]; af = af_start
            else:
                direction[i] = -1
                if df['Low'].iloc[i] < ep:
                    ep = df['Low'].iloc[i]; af = min(af+af_start, af_max)
    df['PSAR_dir'] = direction; df['ATR'] = compute_atr(df)
    return df


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


def _run_exit_with_cap(pos, i, h, lo_v, c, spread, lot, pv, times,
                       sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold,
                       maxloss_cap=0):
    held = i - pos['bar']
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
    if held >= max_hold:
        return _mk(pos, c, times[i], "Timeout", i, pnl_c)
    return None


# ═══════════════════════════════════════════════════════════════
# Strategy backtests with optional allowed_dates filter
# ═══════════════════════════════════════════════════════════════

def bt_psar(h1_df, spread, lot, maxloss_cap=0, allowed_dates=None,
            sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04, max_hold=20):
    df = add_psar(h1_df).dropna(subset=['PSAR_dir', 'ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    pdir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if allowed_dates is not None and pd.Timestamp(times[i]).normalize().tz_localize(None) not in allowed_dates:
            continue
        if pdir[i-1] == -1 and pdir[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif pdir[i-1] == 1 and pdir[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom(h1_df, spread, lot, maxloss_cap=0, allowed_dates=None,
             fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
             trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df['fma'] = df['Close'].rolling(fast).mean()
    df['sma'] = df['Close'].rolling(slow).mean()
    df = df.dropna(subset=['ATR', 'fma', 'sma'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; fm = df['fma'].values; sm = df['sma'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and fm[i] < sm[i]:
                pnl = (c[i]-pos['entry']-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and fm[i] > sm[i]:
                pnl = (pos['entry']-c[i]-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if allowed_dates is not None and pd.Timestamp(times[i]).normalize().tz_localize(None) not in allowed_dates:
            continue
        if fm[i] > sm[i] and fm[i-1] <= sm[i-1]:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif fm[i] < sm[i] and fm[i-1] >= sm[i-1]:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_sess_bo(h1_df, spread, lot, maxloss_cap=0, allowed_dates=None,
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
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if allowed_dates is not None and pd.Timestamp(times[i]).normalize().tz_localize(None) not in allowed_dates:
            continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_l8_max(data_bundle, spread, lot, maxloss_cap=0, allowed_dates=None):
    from backtest.runner import run_variant, LIVE_PARITY_KWARGS
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
    if allowed_dates is not None:
        trades = [t for t in trades
                  if pd.Timestamp(t['entry_time']).normalize().tz_localize(None) in allowed_dates]
    return trades


# ═══════════════════════════════════════════════════════════════
# Stats helpers
# ═══════════════════════════════════════════════════════════════

def trades_to_daily_series(trades):
    if not trades:
        return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


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


def sharpe(arr):
    if len(arr) < 10:
        return 0.0
    s = np.std(arr, ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(arr) / s * np.sqrt(252))


def max_dd(arr):
    if len(arr) == 0:
        return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def compute_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0.0, 'pnl': 0.0, 'wr': 0.0, 'max_dd': 0.0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    return {
        'n': n,
        'sharpe': round(sharpe(daily), 3),
        'pnl': round(sum(pnls), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'max_dd': round(max_dd(daily), 2),
    }


# ═══════════════════════════════════════════════════════════════
# Factor data loading
# ═══════════════════════════════════════════════════════════════

def load_factor_data():
    fpath = Path("data/external/aligned_daily.csv")
    df = pd.read_csv(fpath, parse_dates=['Date'])
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    df = df.set_index('Date')
    available = [c for c in FACTOR_COLS if c in df.columns]
    missing = [c for c in FACTOR_COLS if c not in df.columns]
    if missing:
        print(f"  [WARN] Missing factor columns: {missing}", flush=True)
    return df[available]


# ═══════════════════════════════════════════════════════════════
# Step 1: Factor-Signal Correlation Scan
# ═══════════════════════════════════════════════════════════════

def compute_correlations(strat_daily_pnl, factor_df):
    """
    Compute predictive correlations: factor on day T-1 vs PnL on day T.
    Returns dict of {factor_col: {'pearson': r, 'spearman': r}} for valid pairs.
    """
    results = {}
    pnl_dates = strat_daily_pnl.index
    if len(pnl_dates) < 20:
        return results

    factor_shifted = factor_df.shift(1)

    for col in factor_df.columns:
        fvals = factor_shifted[col].reindex(pnl_dates)
        pvals = strat_daily_pnl.reindex(pnl_dates)
        mask = fvals.notna() & pvals.notna()
        if mask.sum() < 30:
            continue
        f_clean = fvals[mask].values
        p_clean = pvals[mask].values

        pearson_r = float(np.corrcoef(f_clean, p_clean)[0, 1])
        spearman_r = float(spearmanr(f_clean, p_clean).statistic)

        results[col] = {
            'pearson': round(pearson_r, 4),
            'spearman': round(spearman_r, 4),
            'abs_mean': round((abs(pearson_r) + abs(spearman_r)) / 2, 4),
            'n_obs': int(mask.sum()),
        }
    return results


# ═══════════════════════════════════════════════════════════════
# Step 2: Parallel grid worker
# ═══════════════════════════════════════════════════════════════

def _grid_worker(args):
    """
    Worker for a single (strategy, factor, quantile) combo.
    Receives serialised data to avoid pickling issues.
    """
    (strat_name, factor_col, q_val, q_label,
     h1_df_bytes, factor_series_bytes, factor_dates_bytes,
     lot, cap, base_sharpe, base_n, base_wr, base_dd) = args

    h1_df = pd.read_pickle(io.BytesIO(h1_df_bytes))
    factor_vals = np.frombuffer(factor_series_bytes, dtype=np.float64)
    factor_dates = pd.DatetimeIndex(np.frombuffer(factor_dates_bytes, dtype='datetime64[ns]'))

    fser = pd.Series(factor_vals, index=factor_dates)
    threshold = float(np.nanpercentile(fser.dropna().values, q_val * 100))

    above = fser[fser >= threshold]
    allowed = set(above.index)

    bt_fns = {'PSAR': bt_psar, 'TSMOM': bt_tsmom, 'SESS_BO': bt_sess_bo}
    bt_fn = bt_fns[strat_name]
    trades = bt_fn(h1_df, spread=SPREAD, lot=lot, maxloss_cap=cap, allowed_dates=allowed)
    stats = compute_stats(trades)

    sharpe_imp = ((stats['sharpe'] - base_sharpe) / abs(base_sharpe) * 100
                  if base_sharpe != 0 else 0.0)
    trade_reduction = ((base_n - stats['n']) / base_n * 100
                       if base_n > 0 else 0.0)
    dd_change = ((stats['max_dd'] - base_dd) / abs(base_dd) * 100
                 if base_dd != 0 else 0.0)
    wr_change = stats['wr'] - base_wr

    return {
        'strategy': strat_name,
        'factor': factor_col,
        'quantile': q_label,
        'q_value': q_val,
        'threshold': round(threshold, 4),
        'n_allowed_days': len(allowed),
        'filtered_sharpe': stats['sharpe'],
        'filtered_n': stats['n'],
        'filtered_pnl': stats['pnl'],
        'filtered_wr': stats['wr'],
        'filtered_dd': stats['max_dd'],
        'base_sharpe': round(base_sharpe, 3),
        'base_n': base_n,
        'sharpe_improvement_pct': round(sharpe_imp, 2),
        'trade_reduction_pct': round(trade_reduction, 1),
        'dd_change_pct': round(dd_change, 1),
        'wr_change': round(wr_change, 1),
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80)
    print("  R90-B — Factor-Enhanced Signal Filtering")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80, flush=True)

    # ── Load data ──
    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH, DataBundle

    print("\n  Loading H1 data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})", flush=True)

    print("  Loading L8_MAX DataBundle...", flush=True)
    l8_bundle = DataBundle.load_custom()
    print("  L8 bundle ready.", flush=True)

    print("  Loading factor data (aligned_daily)...", flush=True)
    factor_df = load_factor_data()
    print(f"  Factors: {len(factor_df)} days, {len(factor_df.columns)} columns", flush=True)
    print(f"  Factor date range: {factor_df.index.min()} ~ {factor_df.index.max()}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # STEP 1: Factor-Signal Correlation Scan
    # ══════════════════════════════════════════════════════════════
    t1 = time.time()
    print(f"\n{'='*80}")
    print(f"  Step 1: Factor-Signal Correlation Scan")
    print(f"{'='*80}\n", flush=True)

    base_trades = {}
    base_stats = {}
    strat_daily_series = {}

    h1_strats = {
        'PSAR':    (bt_psar, {}),
        'TSMOM':   (bt_tsmom, {}),
        'SESS_BO': (bt_sess_bo, {}),
    }

    for name, (fn, kw) in h1_strats.items():
        lot = LIVE_LOTS[name]
        cap = CAPS[name]
        trades = fn(h1_df, spread=SPREAD, lot=lot, maxloss_cap=cap, **kw)
        base_trades[name] = trades
        base_stats[name] = compute_stats(trades)
        strat_daily_series[name] = trades_to_daily_series(trades)
        print(f"    {name:>8}: {base_stats[name]['n']} trades, "
              f"Sharpe={base_stats[name]['sharpe']:.3f}, "
              f"PnL={fmt(base_stats[name]['pnl'])}", flush=True)

    cap = CAPS['L8_MAX']; lot = LIVE_LOTS['L8_MAX']
    trades = bt_l8_max(l8_bundle, spread=SPREAD, lot=lot, maxloss_cap=cap)
    base_trades['L8_MAX'] = trades
    base_stats['L8_MAX'] = compute_stats(trades)
    strat_daily_series['L8_MAX'] = trades_to_daily_series(trades)
    print(f"    {'L8_MAX':>8}: {base_stats['L8_MAX']['n']} trades, "
          f"Sharpe={base_stats['L8_MAX']['sharpe']:.3f}, "
          f"PnL={fmt(base_stats['L8_MAX']['pnl'])}", flush=True)

    # Portfolio daily PnL
    all_dates = set()
    for ds in strat_daily_series.values():
        all_dates.update(ds.index)
    all_dates = sorted(all_dates)
    port_idx = pd.DatetimeIndex(all_dates)
    port_daily = pd.Series(0.0, index=port_idx)
    for name, ds in strat_daily_series.items():
        port_daily = port_daily.add(ds.reindex(port_idx, fill_value=0.0), fill_value=0.0)

    print(f"\n  Computing correlations (factor T-1 vs PnL T)...", flush=True)

    correlation_matrix = {}
    for name in STRAT_ORDER:
        ds = strat_daily_series[name]
        corrs = compute_correlations(ds, factor_df)
        correlation_matrix[name] = corrs
        ranked = sorted(corrs.items(), key=lambda x: x[1]['abs_mean'], reverse=True)
        print(f"\n    {name} — Top 10 correlated factors:")
        print(f"    {'Factor':<25} {'Pearson':>8} {'Spearman':>9} {'|Mean|':>8} {'N':>6}")
        for fc, vals in ranked[:10]:
            print(f"    {fc:<25} {vals['pearson']:>8.4f} {vals['spearman']:>9.4f} "
                  f"{vals['abs_mean']:>8.4f} {vals['n_obs']:>6}")

    port_corrs = compute_correlations(port_daily, factor_df)
    correlation_matrix['PORTFOLIO'] = port_corrs
    ranked_port = sorted(port_corrs.items(), key=lambda x: x[1]['abs_mean'], reverse=True)
    print(f"\n    PORTFOLIO — Top 10 correlated factors:")
    print(f"    {'Factor':<25} {'Pearson':>8} {'Spearman':>9} {'|Mean|':>8} {'N':>6}")
    for fc, vals in ranked_port[:10]:
        print(f"    {fc:<25} {vals['pearson']:>8.4f} {vals['spearman']:>9.4f} "
              f"{vals['abs_mean']:>8.4f} {vals['n_obs']:>6}")

    # Select top-10 factors per strategy (union across all + portfolio)
    top_factors_per_strat = {}
    for name in STRAT_ORDER:
        corrs = correlation_matrix[name]
        ranked = sorted(corrs.items(), key=lambda x: x[1]['abs_mean'], reverse=True)
        top_factors_per_strat[name] = [fc for fc, _ in ranked[:10]]

    t1_elapsed = time.time() - t1
    print(f"\n  Step 1 complete: {t1_elapsed:.0f}s", flush=True)

    # ══════════════════════════════════════════════════════════════
    # STEP 2: Conditional Entry Filter Grid (parallel)
    # ══════════════════════════════════════════════════════════════
    t2 = time.time()
    print(f"\n{'='*80}")
    print(f"  Step 2: Conditional Entry Filter Grid")
    print(f"{'='*80}\n", flush=True)

    h1_strat_names = ['PSAR', 'TSMOM', 'SESS_BO']

    buf = io.BytesIO()
    h1_df.to_pickle(buf)
    h1_df_bytes = buf.getvalue()

    tasks = []
    q_labels = {0.20: 'Q20', 0.35: 'Q35', 0.50: 'Q50', 0.65: 'Q65', 0.80: 'Q80'}

    for strat_name in h1_strat_names:
        lot = LIVE_LOTS[strat_name]
        cap = CAPS[strat_name]
        bs = base_stats[strat_name]
        top10 = top_factors_per_strat[strat_name]

        for factor_col in top10:
            fser = factor_df[factor_col].dropna()
            if len(fser) < 50:
                continue

            factor_vals_buf = fser.values.astype(np.float64).tobytes()
            factor_dates_buf = fser.index.values.astype('datetime64[ns]').tobytes()

            for q_val in QUANTILE_THRESHOLDS:
                tasks.append((
                    strat_name, factor_col, q_val, q_labels[q_val],
                    h1_df_bytes, factor_vals_buf, factor_dates_buf,
                    lot, cap, bs['sharpe'], bs['n'], bs['wr'], bs['max_dd'],
                ))

    total_h1_tasks = len(tasks)
    print(f"  H1 strategy grid: {total_h1_tasks} combos "
          f"({len(h1_strat_names)} strats x up-to-10 factors x {len(QUANTILE_THRESHOLDS)} quantiles)", flush=True)
    print(f"  Running with {N_WORKERS} parallel workers...", flush=True)

    grid_results = []
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_grid_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            done += 1
            try:
                res = fut.result()
                grid_results.append(res)
            except Exception as e:
                task_info = futures[fut]
                print(f"    [ERR] {task_info[0]}/{task_info[1]}/{task_info[3]}: {e}", flush=True)
            if done % 50 == 0 or done == total_h1_tasks:
                print(f"    Progress: {done}/{total_h1_tasks}", flush=True)

    # L8_MAX: run sequentially (engine-based, can't easily pickle DataBundle)
    print(f"\n  Running L8_MAX filter grid (sequential)...", flush=True)
    l8_top10 = top_factors_per_strat.get('L8_MAX', [])
    l8_bs = base_stats['L8_MAX']
    l8_lot = LIVE_LOTS['L8_MAX']
    l8_cap = CAPS['L8_MAX']
    l8_count = 0

    for factor_col in l8_top10:
        fser = factor_df[factor_col].dropna()
        if len(fser) < 50:
            continue
        for q_val in QUANTILE_THRESHOLDS:
            q_label = q_labels[q_val]
            threshold = float(np.nanpercentile(fser.values, q_val * 100))
            above = fser[fser >= threshold]
            allowed = set(above.index)

            trades = bt_l8_max(l8_bundle, spread=SPREAD, lot=l8_lot,
                               maxloss_cap=l8_cap, allowed_dates=allowed)
            stats = compute_stats(trades)

            sharpe_imp = ((stats['sharpe'] - l8_bs['sharpe']) / abs(l8_bs['sharpe']) * 100
                          if l8_bs['sharpe'] != 0 else 0.0)
            trade_red = ((l8_bs['n'] - stats['n']) / l8_bs['n'] * 100
                         if l8_bs['n'] > 0 else 0.0)
            dd_change = ((stats['max_dd'] - l8_bs['max_dd']) / abs(l8_bs['max_dd']) * 100
                         if l8_bs['max_dd'] != 0 else 0.0)
            wr_change = stats['wr'] - l8_bs['wr']

            grid_results.append({
                'strategy': 'L8_MAX',
                'factor': factor_col,
                'quantile': q_label,
                'q_value': q_val,
                'threshold': round(threshold, 4),
                'n_allowed_days': len(allowed),
                'filtered_sharpe': stats['sharpe'],
                'filtered_n': stats['n'],
                'filtered_pnl': stats['pnl'],
                'filtered_wr': stats['wr'],
                'filtered_dd': stats['max_dd'],
                'base_sharpe': round(l8_bs['sharpe'], 3),
                'base_n': l8_bs['n'],
                'sharpe_improvement_pct': round(sharpe_imp, 2),
                'trade_reduction_pct': round(trade_red, 1),
                'dd_change_pct': round(dd_change, 1),
                'wr_change': round(wr_change, 1),
            })
            l8_count += 1
            if l8_count % 10 == 0:
                print(f"    L8_MAX progress: {l8_count}/{len(l8_top10)*len(QUANTILE_THRESHOLDS)}", flush=True)

    print(f"  L8_MAX grid complete: {l8_count} combos", flush=True)

    grid_results.sort(key=lambda x: x['sharpe_improvement_pct'], reverse=True)

    print(f"\n  Total grid results: {len(grid_results)}")
    print(f"\n  Top 20 filter combos by Sharpe improvement:")
    print(f"  {'Strat':<8} {'Factor':<22} {'Q':>4} {'FiltSh':>7} {'BaseSh':>7} "
          f"{'ShImp%':>7} {'FiltN':>6} {'TrdRed%':>8} {'WRchg':>6} {'DDchg%':>7}")
    print(f"  {'-'*8} {'-'*22} {'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*8} {'-'*6} {'-'*7}")
    for r in grid_results[:20]:
        print(f"  {r['strategy']:<8} {r['factor']:<22} {r['quantile']:>4} "
              f"{r['filtered_sharpe']:>7.3f} {r['base_sharpe']:>7.3f} "
              f"{r['sharpe_improvement_pct']:>+7.1f} {r['filtered_n']:>6} "
              f"{r['trade_reduction_pct']:>7.1f}% {r['wr_change']:>+5.1f} "
              f"{r['dd_change_pct']:>+6.1f}%", flush=True)

    t2_elapsed = time.time() - t2
    print(f"\n  Step 2 complete: {t2_elapsed:.0f}s ({t2_elapsed/60:.1f}min)", flush=True)

    # ══════════════════════════════════════════════════════════════
    # STEP 3: Walk-Forward Validation (top 5)
    # ══════════════════════════════════════════════════════════════
    t3 = time.time()
    print(f"\n{'='*80}")
    print(f"  Step 3: Walk-Forward Validation (top 5 filters)")
    print(f"{'='*80}\n", flush=True)

    positive_filters = [r for r in grid_results
                        if r['sharpe_improvement_pct'] > 0 and r['filtered_n'] >= 10]
    top5 = positive_filters[:5]

    if not top5:
        print("  [WARN] No positive-improvement filters found. Skipping WF validation.", flush=True)
        wf_results = {}
    else:
        print(f"  Validating {len(top5)} filter combos across {len(FOLDS)} folds:\n", flush=True)

        wf_results = {}
        for rank, combo in enumerate(top5):
            strat = combo['strategy']
            factor_col = combo['factor']
            q_val = combo['q_value']
            combo_label = f"{strat}_{factor_col}_{combo['quantile']}"

            print(f"  #{rank+1}: {combo_label}", flush=True)

            fold_sharpes_filtered = []
            fold_sharpes_base = []
            fold_details = []

            for fold_name, fold_start, fold_end in FOLDS:
                # Train = all other folds, test = this fold
                train_starts = []
                train_ends = []
                for fn, fs, fe in FOLDS:
                    if fn != fold_name:
                        train_starts.append(fs)
                        train_ends.append(fe)

                train_factor = pd.concat([
                    factor_df[factor_col][s:e] for s, e in zip(train_starts, train_ends)
                ]).dropna()

                if len(train_factor) < 30:
                    fold_sharpes_filtered.append(0.0)
                    fold_sharpes_base.append(0.0)
                    fold_details.append({'fold': fold_name, 'status': 'insufficient_train'})
                    continue

                threshold = float(np.nanpercentile(train_factor.values, q_val * 100))

                test_factor = factor_df[factor_col][fold_start:fold_end].dropna()
                allowed = set(test_factor[test_factor >= threshold].index)

                if strat in h1_strats:
                    h1_fold = h1_df[fold_start:fold_end]
                    if len(h1_fold) < 100:
                        fold_sharpes_filtered.append(0.0)
                        fold_sharpes_base.append(0.0)
                        fold_details.append({'fold': fold_name, 'status': 'insufficient_data'})
                        continue

                    fn, kw = h1_strats[strat]
                    lot = LIVE_LOTS[strat]
                    cap = CAPS[strat]

                    trades_base = fn(h1_fold, spread=SPREAD, lot=lot, maxloss_cap=cap, **kw)
                    trades_filt = fn(h1_fold, spread=SPREAD, lot=lot, maxloss_cap=cap,
                                     allowed_dates=allowed, **kw)
                else:
                    try:
                        l8_fold = l8_bundle.slice(fold_start, fold_end)
                    except Exception:
                        fold_sharpes_filtered.append(0.0)
                        fold_sharpes_base.append(0.0)
                        fold_details.append({'fold': fold_name, 'status': 'l8_slice_error'})
                        continue
                    trades_base = bt_l8_max(l8_fold, spread=SPREAD,
                                            lot=LIVE_LOTS['L8_MAX'],
                                            maxloss_cap=CAPS['L8_MAX'])
                    trades_filt = bt_l8_max(l8_fold, spread=SPREAD,
                                            lot=LIVE_LOTS['L8_MAX'],
                                            maxloss_cap=CAPS['L8_MAX'],
                                            allowed_dates=allowed)

                sh_base = sharpe(_trades_to_daily(trades_base))
                sh_filt = sharpe(_trades_to_daily(trades_filt))
                fold_sharpes_base.append(sh_base)
                fold_sharpes_filtered.append(sh_filt)
                fold_details.append({
                    'fold': fold_name,
                    'threshold': round(threshold, 4),
                    'n_allowed': len(allowed),
                    'base_sharpe': round(sh_base, 3),
                    'filtered_sharpe': round(sh_filt, 3),
                    'base_n': len(trades_base),
                    'filtered_n': len(trades_filt),
                    'improvement': round(sh_filt - sh_base, 3),
                })

            improvements = [f - b for f, b in zip(fold_sharpes_filtered, fold_sharpes_base)]
            positive_improvement_folds = sum(1 for imp in improvements if imp > 0)
            mean_improvement = float(np.mean(improvements))
            consistent = positive_improvement_folds >= 4

            wf_results[combo_label] = {
                'rank': rank + 1,
                'strategy': strat,
                'factor': factor_col,
                'quantile': combo['quantile'],
                'q_value': q_val,
                'full_sample_sharpe_imp': combo['sharpe_improvement_pct'],
                'fold_sharpes_base': [round(s, 3) for s in fold_sharpes_base],
                'fold_sharpes_filtered': [round(s, 3) for s in fold_sharpes_filtered],
                'fold_improvements': [round(imp, 3) for imp in improvements],
                'positive_improvement_folds': positive_improvement_folds,
                'mean_improvement': round(mean_improvement, 3),
                'consistent_4of6': consistent,
                'fold_details': fold_details,
            }

            status = "PASS" if consistent else "FAIL"
            print(f"       {positive_improvement_folds}/6 improved, "
                  f"mean_imp={mean_improvement:+.3f}  [{status}]")
            print(f"       base:     {[round(s,2) for s in fold_sharpes_base]}")
            print(f"       filtered: {[round(s,2) for s in fold_sharpes_filtered]}", flush=True)

    t3_elapsed = time.time() - t3
    print(f"\n  Step 3 complete: {t3_elapsed:.0f}s ({t3_elapsed/60:.1f}min)", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}\n")

    validated = [k for k, v in wf_results.items() if v['consistent_4of6']]
    print(f"  Grid combos tested: {len(grid_results)}")
    print(f"  Positive improvement: {len(positive_filters)}")
    print(f"  Walk-forward validated (4/6+): {len(validated)}")

    if validated:
        print(f"\n  Validated filters:")
        for label in validated:
            wf = wf_results[label]
            print(f"    {label}: full-sample Sharpe imp={wf['full_sample_sharpe_imp']:+.1f}%, "
                  f"WF mean imp={wf['mean_improvement']:+.3f}, "
                  f"{wf['positive_improvement_folds']}/6 folds improved")
    else:
        print(f"\n  No filters passed walk-forward validation.")
        print(f"  This suggests factor-based entry filtering does not reliably improve these strategies.")

    print(f"\n  Per-strategy base performance:")
    for name in STRAT_ORDER:
        bs = base_stats[name]
        print(f"    {name:>8}: Sharpe={bs['sharpe']:.3f}, N={bs['n']}, "
              f"PnL={fmt(bs['pnl'])}, WR={bs['wr']:.1f}%")

    print(f"\n{'='*80}")
    print(f"  R90-B COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"    Step 1 (correlations): {t1_elapsed:.0f}s")
    print(f"    Step 2 (grid search):  {t2_elapsed:.0f}s")
    print(f"    Step 3 (walk-forward): {t3_elapsed:.0f}s")
    print(f"{'='*80}", flush=True)

    # ── Save ──
    output = {
        'config': {
            'spread': SPREAD,
            'pv': PV,
            'caps': CAPS,
            'live_lots': LIVE_LOTS,
            'factor_cols_tested': list(factor_df.columns),
            'quantile_thresholds': QUANTILE_THRESHOLDS,
            'n_workers': N_WORKERS,
        },
        'base_stats': base_stats,
        'correlation_matrix': correlation_matrix,
        'top_factors_per_strategy': top_factors_per_strat,
        'grid_results': grid_results,
        'top_filters': positive_filters[:20],
        'walk_forward': wf_results,
        'validated_filters': validated,
        'timing': {
            'step1_s': round(t1_elapsed, 1),
            'step2_s': round(t2_elapsed, 1),
            'step3_s': round(t3_elapsed, 1),
            'total_s': round(elapsed, 1),
        },
    }

    outpath = OUTPUT_DIR / "r90b_results.json"
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {outpath}", flush=True)


if __name__ == "__main__":
    main()
