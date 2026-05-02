#!/usr/bin/env python3
"""
R90-E: Dynamic Regime-Conditional Portfolio Allocation
========================================================
Dynamically adjusts strategy lot sizes based on macro regime,
replacing static allocation from R89.
"""
import sys, os, io, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r90_external_data/r90e_portfolio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01
CAPITAL = 5000
MAX_DD_LIMIT = 1000

CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']

# R89 winner (static baseline)
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.08, 'SESS_BO': 0.08}

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


# ═══════════════════════════════════════════════════════════════
# Shared helpers (from R88/R89)
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
# Strategy backtests (from R88/R89)
# ═══════════════════════════════════════════════════════════════

def bt_psar(h1_df, spread, lot, maxloss_cap=0,
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
        if pdir[i-1] == -1 and pdir[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif pdir[i-1] == 1 and pdir[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom(h1_df, spread, lot, maxloss_cap=0,
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
        if fm[i] > sm[i] and fm[i-1] <= sm[i-1]:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif fm[i] < sm[i] and fm[i-1] >= sm[i-1]:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
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
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_l8_max(data_bundle, spread, lot, maxloss_cap=0):
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
    return trades


# ═══════════════════════════════════════════════════════════════
# Daily PnL helpers (from R89)
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


def cvar99(arr):
    if len(arr) < 20:
        return 0.0
    v = np.percentile(arr, 1)
    tail = arr[arr <= v]
    return float(tail.mean()) if len(tail) > 0 else float(v)


def build_portfolio_daily(unit_dailies, lots):
    all_dates = set()
    for ds in unit_dailies.values():
        all_dates.update(ds.index)
    all_dates = sorted(all_dates)
    idx = pd.DatetimeIndex(all_dates)
    portfolio = np.zeros(len(idx))
    for name in STRAT_ORDER:
        if name not in unit_dailies:
            continue
        ds = unit_dailies[name]
        multiplier = lots[name] / UNIT_LOT
        aligned = ds.reindex(idx, fill_value=0.0).values * multiplier
        portfolio += aligned
    return portfolio, idx


def dd_recovery_days(daily_arr):
    """Max number of days to recover from drawdown peak to new equity high."""
    if len(daily_arr) == 0:
        return 0
    eq = np.cumsum(daily_arr)
    peak = np.maximum.accumulate(eq)
    in_dd = eq < peak
    max_recovery = 0
    current_run = 0
    for v in in_dd:
        if v:
            current_run += 1
            max_recovery = max(max_recovery, current_run)
        else:
            current_run = 0
    return max_recovery


# ═══════════════════════════════════════════════════════════════
# Regime detection
# ═══════════════════════════════════════════════════════════════

def load_regime_labels():
    """Load regime labels from R90a output, or build fallback from aligned_daily."""
    regime_path = Path("results/r90_external_data/r90a_regime/regime_labels.csv")
    if regime_path.exists():
        print("  Loading regime labels from R90a output...", flush=True)
        df = pd.read_csv(regime_path, parse_dates=['Date'], index_col='Date')
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
        # Prefer rule_regime if available, else cluster_regime
        for col in ['rule_regime', 'cluster_regime', 'regime']:
            if col in df.columns:
                print(f"    Using column: {col}", flush=True)
                return df[col].dropna()
        # Take first non-Date column
        col = [c for c in df.columns if c != 'Date'][0]
        print(f"    Using column: {col}", flush=True)
        return df[col].dropna()

    print("  R90a regime labels not found — building fallback from aligned_daily...", flush=True)
    aligned_path = Path("data/external/aligned_daily.csv")
    if not aligned_path.exists():
        print("    [WARN] aligned_daily.csv not found, using uniform regime", flush=True)
        return None

    df = pd.read_csv(aligned_path, parse_dates=['Date'], index_col='Date')

    vix = df['VIX_Close'] if 'VIX_Close' in df.columns else None
    dxy = df['DXY_Close'] if 'DXY_Close' in df.columns else None

    if vix is None or dxy is None:
        print("    [WARN] VIX/DXY not available, using uniform regime", flush=True)
        return None

    # Rule-based regime: VIX > 25 = risk_off, DXY < 50th pctile & VIX < 18 = risk_on, else neutral
    vix_clean = vix.dropna()
    dxy_clean = dxy.dropna()
    common_idx = vix_clean.index.intersection(dxy_clean.index)
    vix_vals = vix_clean.loc[common_idx]
    dxy_vals = dxy_clean.loc[common_idx]

    dxy_median = dxy_vals.median()

    regime = pd.Series('neutral', index=common_idx)
    regime[vix_vals > 25] = 'risk_off'
    regime[(vix_vals < 18) & (dxy_vals < dxy_median)] = 'risk_on'

    counts = regime.value_counts()
    print(f"    Fallback regime distribution: {dict(counts)}", flush=True)
    return regime


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80)
    print("  R90-E: Dynamic Regime-Conditional Portfolio Allocation")
    print(f"  Capital: ${CAPITAL:,}  |  MaxDD limit: ${MAX_DD_LIMIT:,}")
    print(f"  Static baseline (R89): {R89_LOTS}")
    print("=" * 80, flush=True)

    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH, DataBundle

    # ── Step 1: Run all strategies at unit lot ──
    print(f"\n{'='*80}")
    print(f"  Step 1: Per-Strategy Backtest at unit lot ({UNIT_LOT})")
    print(f"{'='*80}\n", flush=True)

    print("  Loading data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})", flush=True)

    print("  Preparing L8_MAX DataBundle...", flush=True)
    l8_bundle = DataBundle.load_custom()
    print("  L8 bundle ready.\n", flush=True)

    unit_trades = {}
    unit_dailies = {}
    unit_stats = {}

    h1_strats = {
        'PSAR':    (bt_psar, {}),
        'TSMOM':   (bt_tsmom, {}),
        'SESS_BO': (bt_sess_bo, {}),
    }
    for name, (fn, kw) in h1_strats.items():
        cap = CAPS[name]
        trades = fn(h1_df, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=cap, **kw)
        unit_trades[name] = trades
        unit_dailies[name] = trades_to_daily_series(trades)
        pnls = [t['pnl'] for t in trades]
        daily_arr = unit_dailies[name].values
        unit_stats[name] = {
            'n_trades': len(trades),
            'pnl': round(sum(pnls), 2),
            'sharpe': round(sharpe(daily_arr), 2),
            'max_dd': round(max_dd(daily_arr), 2),
            'wr': round(sum(1 for p in pnls if p > 0) / max(len(pnls), 1) * 100, 1),
        }
        print(f"    {name:>8}: {len(trades)} trades, Sharpe={unit_stats[name]['sharpe']:.2f}, "
              f"PnL={fmt(unit_stats[name]['pnl'])}, MaxDD={fmt(unit_stats[name]['max_dd'])}", flush=True)

    cap = CAPS['L8_MAX']
    trades = bt_l8_max(l8_bundle, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=cap)
    unit_trades['L8_MAX'] = trades
    unit_dailies['L8_MAX'] = trades_to_daily_series(trades)
    pnls = [t['pnl'] for t in trades]
    daily_arr = unit_dailies['L8_MAX'].values
    unit_stats['L8_MAX'] = {
        'n_trades': len(trades),
        'pnl': round(sum(pnls), 2),
        'sharpe': round(sharpe(daily_arr), 2),
        'max_dd': round(max_dd(daily_arr), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / max(len(pnls), 1) * 100, 1),
    }
    print(f"    {'L8_MAX':>8}: {len(trades)} trades, Sharpe={unit_stats['L8_MAX']['sharpe']:.2f}, "
          f"PnL={fmt(unit_stats['L8_MAX']['pnl'])}, MaxDD={fmt(unit_stats['L8_MAX']['max_dd'])}", flush=True)

    print(f"\n  Step 1 complete. Unit-lot baselines ready.", flush=True)

    # ── Step 2: Per-Regime Strategy Performance Analysis ──
    print(f"\n{'='*80}")
    print(f"  Step 2: Per-Regime Strategy Performance Analysis")
    print(f"{'='*80}\n", flush=True)

    regime_labels = load_regime_labels()

    if regime_labels is None:
        # Uniform single regime — dynamic allocation degenerates to static
        print("  [WARN] No regime data available. Using single 'all' regime.", flush=True)
        all_dates = set()
        for ds in unit_dailies.values():
            all_dates.update(ds.index)
        all_dates = sorted(all_dates)
        regime_labels = pd.Series('all', index=pd.DatetimeIndex(all_dates))

    regime_names = sorted(regime_labels.unique())
    print(f"  Regimes found: {regime_names}", flush=True)

    # Build date->regime mapping (normalize to date only)
    date_to_regime = {}
    for dt, reg in regime_labels.items():
        date_to_regime[pd.Timestamp(dt).normalize()] = reg

    # Per-regime performance at unit lot
    regime_perf = {}
    print(f"\n  {'Regime':<12} {'Strategy':<10} {'Days':>6} {'Sharpe':>8} {'MeanPnL':>10} "
          f"{'Std':>10} {'MaxDD':>10} {'WinRate':>8}")
    print(f"  {'-'*12} {'-'*10} {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for regime in regime_names:
        regime_perf[regime] = {}
        regime_dates = set()
        for dt, reg in date_to_regime.items():
            if reg == regime:
                regime_dates.add(dt)

        for name in STRAT_ORDER:
            ds = unit_dailies[name]
            mask = ds.index.normalize().isin(regime_dates)
            regime_pnl = ds[mask].values

            n_days = len(regime_pnl)
            if n_days < 5:
                regime_perf[regime][name] = {
                    'n_days': n_days, 'sharpe': 0.0, 'mean_pnl': 0.0,
                    'std': 0.0, 'max_dd': 0.0, 'win_rate': 0.0,
                }
                continue

            sh = sharpe(regime_pnl)
            mean_p = float(np.mean(regime_pnl))
            std_p = float(np.std(regime_pnl, ddof=1)) if n_days > 1 else 0.0
            dd = max_dd(regime_pnl)
            wr = float(np.sum(regime_pnl > 0) / n_days * 100)

            regime_perf[regime][name] = {
                'n_days': n_days, 'sharpe': round(sh, 2),
                'mean_pnl': round(mean_p, 4), 'std': round(std_p, 4),
                'max_dd': round(dd, 2), 'win_rate': round(wr, 1),
            }
            print(f"  {regime:<12} {name:<10} {n_days:>6} {sh:>8.2f} {mean_p:>10.4f} "
                  f"{std_p:>10.4f} {fmt(dd):>10} {wr:>7.1f}%", flush=True)

    # ── Step 3: Regime-Conditional Lot Optimization ──
    print(f"\n{'='*80}")
    print(f"  Step 3: Regime-Conditional Lot Optimization")
    print(f"{'='*80}\n", flush=True)

    # Multiplier grid: 0.5x to 2.0x of R89 base lots, step 0.25x
    MULT_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    # Convert multipliers to actual lots (clamped to 0.01 minimum)
    def mult_to_lots(mults):
        return {name: max(0.01, round(R89_LOTS[name] * mults[name], 2))
                for name in STRAT_ORDER}

    # Step 3A: For each regime, find top-10 multiplier combos by regime-specific Sharpe
    print(f"  Step 3A: Per-regime top combos ({len(MULT_GRID)}^4 = {len(MULT_GRID)**4} per regime)")
    print(f"  Regimes: {regime_names}\n", flush=True)

    regime_top_combos = {}
    for regime in regime_names:
        regime_dates = set()
        for dt, reg in date_to_regime.items():
            if reg == regime:
                regime_dates.add(dt)

        # Build per-regime unit PnL arrays aligned to common index
        all_dates_sorted = sorted(set().union(*(set(unit_dailies[n].index) for n in STRAT_ORDER)))
        all_idx = pd.DatetimeIndex(all_dates_sorted)
        regime_mask = all_idx.normalize().isin(regime_dates)

        if regime_mask.sum() < 10:
            print(f"    {regime}: too few days ({regime_mask.sum()}), skipping optimization", flush=True)
            regime_top_combos[regime] = [{'mults': {n: 1.0 for n in STRAT_ORDER}, 'sharpe': 0.0}]
            continue

        # Pre-compute aligned unit arrays for this regime
        regime_unit_arrays = {}
        for name in STRAT_ORDER:
            ds = unit_dailies[name]
            aligned = ds.reindex(all_idx, fill_value=0.0).values
            regime_unit_arrays[name] = aligned[regime_mask]

        candidates = []
        for m_l8, m_psar, m_tsmom, m_sess in product(MULT_GRID, repeat=4):
            mults = {'L8_MAX': m_l8, 'PSAR': m_psar, 'TSMOM': m_tsmom, 'SESS_BO': m_sess}
            port = np.zeros(regime_mask.sum())
            for name in STRAT_ORDER:
                lot = max(0.01, round(R89_LOTS[name] * mults[name], 2))
                port += regime_unit_arrays[name] * (lot / UNIT_LOT)

            sh = sharpe(port)
            candidates.append({'mults': mults, 'sharpe': round(sh, 3)})

        candidates.sort(key=lambda x: x['sharpe'], reverse=True)
        regime_top_combos[regime] = candidates[:10]

        print(f"    {regime}: top Sharpe = {candidates[0]['sharpe']:.3f}  "
              f"mults = {candidates[0]['mults']}", flush=True)

    # Step 3B: Combine top regime combos, simulate full time series
    print(f"\n  Step 3B: Cross-regime combination search", flush=True)

    all_dates_sorted = sorted(set().union(*(set(unit_dailies[n].index) for n in STRAT_ORDER)))
    all_idx = pd.DatetimeIndex(all_dates_sorted)

    # Pre-compute day-to-regime mapping for the portfolio date index
    default_regime = regime_names[0] if regime_names else 'Neutral'
    day_regime = []
    for dt in all_idx:
        dt_norm = dt.normalize()
        day_regime.append(date_to_regime.get(dt_norm, default_regime))
    day_regime = np.array(day_regime)

    # Pre-compute aligned unit arrays
    unit_arrays = {}
    for name in STRAT_ORDER:
        ds = unit_dailies[name]
        unit_arrays[name] = ds.reindex(all_idx, fill_value=0.0).values

    n_days_total = len(all_idx)

    # Generate all cross-regime combos
    regime_combo_lists = [regime_top_combos.get(r, [{'mults': {n: 1.0 for n in STRAT_ORDER}}])
                          for r in regime_names]
    combo_indices = [range(len(cl)) for cl in regime_combo_lists]

    best_dynamic = None
    best_sharpe = -999
    n_combos_tested = 0
    n_feasible = 0

    for combo_idx in product(*combo_indices):
        n_combos_tested += 1
        # Build the regime-lot table for this combination
        regime_lot_table = {}
        for ri, regime in enumerate(regime_names):
            mults = regime_combo_lists[ri][combo_idx[ri]]['mults']
            regime_lot_table[regime] = mult_to_lots(mults)

        # Simulate full time series with regime-conditional lots
        port_daily = np.zeros(n_days_total)
        for name in STRAT_ORDER:
            for ri, regime in enumerate(regime_names):
                lot = regime_lot_table[regime][name]
                mask = day_regime == regime
                port_daily[mask] += unit_arrays[name][mask] * (lot / UNIT_LOT)

        dd = max_dd(port_daily)
        if dd > MAX_DD_LIMIT:
            continue

        n_feasible += 1
        sh = sharpe(port_daily)
        if sh > best_sharpe:
            best_sharpe = sh
            pnl_total = float(np.sum(port_daily))
            best_dynamic = {
                'regime_lot_table': regime_lot_table,
                'sharpe': round(sh, 3),
                'pnl': round(pnl_total, 2),
                'max_dd': round(dd, 2),
                'cvar99': round(cvar99(port_daily), 2),
                'port_daily': port_daily,
            }

    print(f"  Tested {n_combos_tested} cross-regime combos, {n_feasible} feasible (DD<=${MAX_DD_LIMIT})")
    if best_dynamic:
        print(f"  Best dynamic: Sharpe={best_dynamic['sharpe']:.3f}, "
              f"PnL={fmt(best_dynamic['pnl'])}, MaxDD={fmt(best_dynamic['max_dd'])}", flush=True)
        print(f"\n  Regime-Lot Table:")
        for regime in regime_names:
            lots = best_dynamic['regime_lot_table'][regime]
            lot_str = "  ".join(f"{n}={lots[n]:.2f}" for n in STRAT_ORDER)
            print(f"    {regime:<12}: {lot_str}", flush=True)
    else:
        print("  [WARN] No feasible dynamic combination found! Using static as fallback.", flush=True)

    # ── Step 4: Static vs Dynamic Comparison ──
    print(f"\n{'='*80}")
    print(f"  Step 4: Static vs Dynamic Comparison")
    print(f"{'='*80}\n", flush=True)

    # Static baseline (R89 lots every day)
    static_daily, static_idx = build_portfolio_daily(unit_dailies, R89_LOTS)
    n_years = len(static_daily) / 252

    static_metrics = {
        'sharpe': round(sharpe(static_daily), 3),
        'pnl': round(float(np.sum(static_daily)), 2),
        'max_dd': round(max_dd(static_daily), 2),
        'cvar99': round(cvar99(static_daily), 2),
        'annual_return_pct': round(float(np.sum(static_daily)) / n_years / CAPITAL * 100, 1),
        'calmar': 0.0,
        'dd_recovery_days': dd_recovery_days(static_daily),
    }
    static_dd = max_dd(static_daily)
    if static_dd > 0:
        static_metrics['calmar'] = round(float(np.sum(static_daily)) / n_years / static_dd, 2)

    if best_dynamic:
        dyn_daily = best_dynamic['port_daily']
        dyn_dd = best_dynamic['max_dd']
        dynamic_metrics = {
            'sharpe': best_dynamic['sharpe'],
            'pnl': best_dynamic['pnl'],
            'max_dd': best_dynamic['max_dd'],
            'cvar99': best_dynamic['cvar99'],
            'annual_return_pct': round(best_dynamic['pnl'] / n_years / CAPITAL * 100, 1),
            'calmar': round(best_dynamic['pnl'] / n_years / dyn_dd, 2) if dyn_dd > 0 else 0.0,
            'dd_recovery_days': dd_recovery_days(dyn_daily),
        }
    else:
        dynamic_metrics = dict(static_metrics)

    print(f"  {'Metric':<25} {'Static (R89)':>15} {'Dynamic':>15} {'Delta':>12}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*12}")

    comparison_rows = [
        ('Sharpe', 'sharpe', '.3f'),
        ('Total PnL ($)', 'pnl', ',.0f'),
        ('MaxDD ($)', 'max_dd', ',.0f'),
        ('CVaR 99 ($)', 'cvar99', ',.2f'),
        ('Annual Return (%)', 'annual_return_pct', '.1f'),
        ('Calmar Ratio', 'calmar', '.2f'),
        ('DD Recovery (days)', 'dd_recovery_days', 'd'),
    ]

    comparison_output = {}
    for label, key, fmtstr in comparison_rows:
        sv = static_metrics[key]
        dv = dynamic_metrics[key]
        delta = dv - sv
        delta_sign = '+' if delta >= 0 else ''

        sv_str = f"{sv:{fmtstr}}"
        dv_str = f"{dv:{fmtstr}}"
        delta_str = f"{delta_sign}{delta:{fmtstr}}"
        print(f"  {label:<25} {sv_str:>15} {dv_str:>15} {delta_str:>12}", flush=True)
        comparison_output[key] = {'static': sv, 'dynamic': dv, 'delta': round(delta, 4)}

    dynamic_better = dynamic_metrics['sharpe'] > static_metrics['sharpe']
    print(f"\n  Dynamic {'OUTPERFORMS' if dynamic_better else 'UNDERPERFORMS'} static by "
          f"{abs(dynamic_metrics['sharpe'] - static_metrics['sharpe']):.3f} Sharpe", flush=True)

    # ── Step 5: K-Fold Robustness ──
    print(f"\n{'='*80}")
    print(f"  Step 5: K-Fold Robustness (6-fold)")
    print(f"{'='*80}\n", flush=True)

    # Pre-compute fold unit dailies for H1 strategies
    fold_unit_dailies = {}
    for fold_name, start, end in FOLDS:
        fold_unit_dailies[fold_name] = {}
        h1_fold = h1_df[start:end]
        for name, (fn, kw) in h1_strats.items():
            cap = CAPS[name]
            trades = fn(h1_fold, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=cap, **kw)
            fold_unit_dailies[fold_name][name] = trades_to_daily_series(trades)

    for fold_name, start, end in FOLDS:
        try:
            l8_fold = l8_bundle.slice(start, end)
            trades = bt_l8_max(l8_fold, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])
            fold_unit_dailies[fold_name]['L8_MAX'] = trades_to_daily_series(trades)
        except Exception as e:
            print(f"    [WARN] L8_MAX fold {fold_name}: {e}", flush=True)
            fold_unit_dailies[fold_name]['L8_MAX'] = pd.Series(dtype=float)

    kfold_results = []
    dynamic_wins = 0

    print(f"  {'Fold':<8} {'Static Sharpe':>15} {'Dynamic Sharpe':>16} {'Winner':>10}")
    print(f"  {'-'*8} {'-'*15} {'-'*16} {'-'*10}")

    for fold_name, start, end in FOLDS:
        fud = fold_unit_dailies[fold_name]

        # Static Sharpe on this fold
        static_fold, _ = build_portfolio_daily(fud, R89_LOTS)
        static_sh = sharpe(static_fold)

        # Dynamic Sharpe on this fold: apply regime-conditional lots
        if best_dynamic:
            fold_dates = sorted(set().union(*(set(fud[n].index) for n in STRAT_ORDER if n in fud and len(fud[n]) > 0)))
            if not fold_dates:
                dynamic_sh = 0.0
            else:
                fold_idx = pd.DatetimeIndex(fold_dates)
                fold_daily = np.zeros(len(fold_idx))
                for name in STRAT_ORDER:
                    if name not in fud or len(fud[name]) == 0:
                        continue
                    ds = fud[name]
                    arr = ds.reindex(fold_idx, fill_value=0.0).values
                    for dt_i, dt in enumerate(fold_idx):
                        dt_norm = dt.normalize()
                        reg = date_to_regime.get(dt_norm, 'neutral')
                        # Use 'neutral' as fallback if regime not in table
                        if reg in best_dynamic['regime_lot_table']:
                            lot = best_dynamic['regime_lot_table'][reg][name]
                        elif 'neutral' in best_dynamic['regime_lot_table']:
                            lot = best_dynamic['regime_lot_table']['neutral'][name]
                        else:
                            lot = R89_LOTS[name]
                        fold_daily[dt_i] += arr[dt_i] * (lot / UNIT_LOT)
                dynamic_sh = sharpe(fold_daily)
        else:
            dynamic_sh = static_sh

        winner = "Dynamic" if dynamic_sh > static_sh else "Static"
        if dynamic_sh > static_sh:
            dynamic_wins += 1

        kfold_results.append({
            'fold': fold_name,
            'start': start,
            'end': end,
            'static_sharpe': round(static_sh, 3),
            'dynamic_sharpe': round(dynamic_sh, 3),
            'winner': winner,
        })
        print(f"  {fold_name:<8} {static_sh:>15.3f} {dynamic_sh:>16.3f} {winner:>10}", flush=True)

    pass_rate = dynamic_wins / len(FOLDS)
    print(f"\n  Dynamic wins: {dynamic_wins}/{len(FOLDS)} folds ({pass_rate*100:.0f}%)")
    kfold_pass = dynamic_wins >= 4
    print(f"  K-Fold pass (>=4/6): {'PASS' if kfold_pass else 'FAIL'}", flush=True)

    # ── Step 6: Final Recommendation ──
    print(f"\n{'='*80}")
    print(f"  Step 6: Final Recommendation")
    print(f"{'='*80}\n", flush=True)

    recommend_dynamic = dynamic_better and kfold_pass

    if recommend_dynamic:
        print("  RECOMMENDATION: Use DYNAMIC regime-conditional allocation")
        print(f"  Rationale: Dynamic outperforms static by "
              f"{dynamic_metrics['sharpe'] - static_metrics['sharpe']:.3f} Sharpe, "
              f"passes {dynamic_wins}/6 K-Fold tests\n")
    else:
        reasons = []
        if not dynamic_better:
            reasons.append("dynamic does not outperform static on full sample")
        if not kfold_pass:
            reasons.append(f"dynamic wins only {dynamic_wins}/6 folds (need >=4)")
        print(f"  RECOMMENDATION: Keep STATIC allocation (R89 lots)")
        print(f"  Rationale: {'; '.join(reasons)}\n")

    print(f"  Static lots (R89 baseline):")
    for name in STRAT_ORDER:
        print(f"    {name:<10}  {R89_LOTS[name]:.2f} lot")

    if best_dynamic:
        print(f"\n  Dynamic regime-lot table:")
        for regime in regime_names:
            lots = best_dynamic['regime_lot_table'][regime]
            lot_str = "  ".join(f"{n}={lots[n]:.2f}" for n in STRAT_ORDER)
            print(f"    {regime:<12}: {lot_str}")

    print(f"\n  Portfolio metrics comparison:")
    print(f"    {'':>25} {'Static':>12} {'Dynamic':>12}")
    print(f"    {'Sharpe':>25} {static_metrics['sharpe']:>12.3f} {dynamic_metrics['sharpe']:>12.3f}")
    print(f"    {'Total PnL':>25} {fmt(static_metrics['pnl']):>12} {fmt(dynamic_metrics['pnl']):>12}")
    print(f"    {'MaxDD':>25} {fmt(static_metrics['max_dd']):>12} {fmt(dynamic_metrics['max_dd']):>12}")
    print(f"    {'Calmar':>25} {static_metrics['calmar']:>12.2f} {dynamic_metrics['calmar']:>12.2f}")
    print(f"    {'Annual Return %':>25} {static_metrics['annual_return_pct']:>11.1f}% {dynamic_metrics['annual_return_pct']:>11.1f}%")

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  R90-E COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}", flush=True)

    # ── Save results ──
    regime_lot_table_serializable = {}
    if best_dynamic:
        for regime in regime_names:
            regime_lot_table_serializable[regime] = best_dynamic['regime_lot_table'][regime]

    output = {
        'config': {
            'capital': CAPITAL,
            'max_dd_limit': MAX_DD_LIMIT,
            'spread': SPREAD,
            'caps': CAPS,
            'r89_lots': R89_LOTS,
            'unit_lot': UNIT_LOT,
            'mult_grid': MULT_GRID,
        },
        'unit_stats': unit_stats,
        'regime_names': regime_names,
        'regime_performance': regime_perf,
        'regime_lot_table': regime_lot_table_serializable,
        'static_vs_dynamic': comparison_output,
        'static_metrics': static_metrics,
        'dynamic_metrics': {k: v for k, v in dynamic_metrics.items()},
        'kfold_results': kfold_results,
        'kfold_pass_rate': round(pass_rate, 2),
        'kfold_pass': kfold_pass,
        'recommend_dynamic': recommend_dynamic,
        'final_recommendation': 'dynamic' if recommend_dynamic else 'static',
        'elapsed_s': round(elapsed, 1),
    }

    with open(OUTPUT_DIR / "r90e_results.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r90e_results.json", flush=True)


if __name__ == "__main__":
    main()
