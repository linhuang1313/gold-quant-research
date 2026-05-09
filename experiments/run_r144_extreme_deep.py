#!/usr/bin/env python3
"""
R144 — Deep Validation of R139's "Rule B" (Skip Entry 12 Bars After Extreme)
=============================================================================
R139 found Rule B (skip entries 12 bars after extreme market detection) improved
Sharpe from 6.93 to 7.30 (+5.3%), K-Fold 5/5 PASS. This experiment validates
that Rule B is NOT overfit via:

Phases:
  1. Load data (reuse R139 patterns)
  2. Parameter sensitivity sweep (skip_bars 4-24)
  3. Regime definition sensitivity (CUSUM sigma x extreme_window grid)
  4. Walk-Forward OOS (6 expanding windows)
  5. Era analysis (3 eras)
  6. Monte Carlo (1000 bootstrap paths)
  7. 8-stage StrategyValidator on best config
  8. Integration with v2 production config (PSAR optimized)
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

from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS, load_csv
from backtest.validator import StrategyValidator, ValidatorConfig

OUTPUT_DIR = Path("results/r144_extreme_deep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01
CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.08, 'SESS_BO': 0.08}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']

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


def _compute_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0,
                'cvar95': 0, 'cvar99': 0}
    daily = _trades_to_daily(trades)
    pnls = np.array([t['pnl'] for t in trades])
    n = len(trades)
    return {
        'n': n, 'sharpe': round(_sharpe(daily), 3),
        'pnl': round(float(np.sum(pnls)), 2),
        'wr': round(float(np.sum(pnls > 0)) / n * 100, 1),
        'max_dd': round(_max_dd(daily), 2),
        'cvar95': round(_cvar(daily, 0.05), 2),
        'cvar99': round(_cvar(daily, 0.01), 2),
    }


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
# Extreme regime detector (parameterized)
# ═══════════════════════════════════════════════════════════════

def build_extreme_mask(h1_df, macro_df, cusum_sigma=3.0, extreme_window=24):
    """Build boolean mask: True = EXTREME mode for that H1 bar.

    Parameters
    ----------
    cusum_sigma : float
        ATR must exceed rolling mean + cusum_sigma * rolling std to trigger.
    extreme_window : int
        Number of bars to mark as extreme after each trigger.
    """
    n = len(h1_df)
    extreme = np.zeros(n, dtype=bool)
    atr = compute_atr(h1_df).values

    atr_clean = np.nan_to_num(atr, nan=0.0)
    atr_mean = pd.Series(atr_clean).rolling(60, min_periods=20).mean().values
    atr_std = pd.Series(atr_clean).rolling(60, min_periods=20).std().values
    atr_std = np.maximum(atr_std, 1e-6)
    cusum_trigger = atr_clean > (atr_mean + cusum_sigma * atr_std)

    for i in range(n):
        if cusum_trigger[i]:
            end_i = min(i + extreme_window, n)
            extreme[i:end_i] = True

    if macro_df is not None and 'vix_close' in macro_df.columns:
        vix = macro_df['vix_close'].dropna()
        vix_pct = vix.pct_change()
        h1_dates = h1_df.index.date if hasattr(h1_df.index, 'date') else pd.DatetimeIndex(h1_df.index).date
        for idx_val, v in vix.items():
            d = pd.Timestamp(idx_val).date()
            vix_change = vix_pct.get(idx_val, 0)
            if v > 30 or (not np.isnan(vix_change) and abs(vix_change) > 0.20):
                mask = np.array([dt == d for dt in h1_dates])
                bar_indices = np.where(mask)[0]
                for bi in bar_indices:
                    end_i = min(bi + extreme_window, n)
                    extreme[bi:end_i] = True

    return extreme


def apply_protection(trades, h1_df, extreme_mask, skip_bars=12):
    """Apply Rule B: skip new entries for skip_bars after EXTREME trigger."""
    if not trades:
        return []

    times_idx = h1_df.index
    n_bars = len(times_idx)

    def _find_bar(ts):
        ts = pd.Timestamp(ts)
        idx = times_idx.searchsorted(ts)
        return min(idx, n_bars - 1) if idx < n_bars else n_bars - 1

    protected = []
    for t in trades:
        entry_bar = _find_bar(t['entry_time'])
        skip_end = entry_bar
        for j in range(max(0, entry_bar - skip_bars), entry_bar):
            if j < n_bars and extreme_mask[j]:
                skip_end = j + skip_bars
                break
        if entry_bar < skip_end:
            continue
        protected.append(t)

    return protected


# ═══════════════════════════════════════════════════════════════
# Portfolio helpers
# ═══════════════════════════════════════════════════════════════

def load_h1():
    candidates = [
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-27.csv"),
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv"),
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-03-25.csv"),
    ]
    h1_path = next((p for p in candidates if p.exists()), candidates[-1])
    return load_csv(str(h1_path))


def load_macro():
    macro_path = Path("data/external/aligned_daily.csv")
    if macro_path.exists():
        df = pd.read_csv(macro_path, index_col=0, parse_dates=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    return None


def run_all_strategies(h1_df, bundle):
    """Run all 4 strategies at unit lot, return dict of trade lists."""
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


def run_strategies_on_slice(h1_slice):
    """Run 3 strategies (no L8_MAX) on a H1 slice. Used for fold/era analysis."""
    trades = {}
    trades['PSAR'] = bt_psar(h1_slice, SPREAD, UNIT_LOT, CAPS['PSAR'])
    trades['TSMOM'] = bt_tsmom(h1_slice, SPREAD, UNIT_LOT, CAPS['TSMOM'])
    trades['SESS_BO'] = bt_sess_bo(h1_slice, SPREAD, UNIT_LOT, CAPS['SESS_BO'])
    trades['L8_MAX'] = []
    return trades


def portfolio_daily(strat_trades, lots=None):
    if lots is None:
        lots = R89_LOTS
    all_daily = {}
    for sn in STRAT_ORDER:
        mult = lots[sn] / UNIT_LOT
        for t in strat_trades.get(sn, []):
            d = pd.Timestamp(t['exit_time']).date()
            all_daily[d] = all_daily.get(d, 0) + t['pnl'] * mult
    dates = sorted(all_daily.keys())
    return np.array([all_daily[d] for d in dates]) if dates else np.array([0.0])


def portfolio_daily_from_trades(all_trades, lot_mult=1.0):
    """Convert a flat list of trades to daily PnL array."""
    daily = {}
    for t in all_trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl'] * lot_mult
    dates = sorted(daily.keys())
    return np.array([daily[d] for d in dates]) if dates else np.array([0.0])


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    results = {'experiment': 'R144 Deep Validation of R139 Rule B'}

    print("=" * 80, flush=True)
    print("  R144 — Deep Validation of R139's Rule B (Skip Entry After Extreme)", flush=True)
    print("=" * 80, flush=True)

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Load data
    # ═══════════════════════════════════════════════════════════
    print("\n  Phase 1: Loading data...", flush=True)
    h1_df = load_h1()
    macro_df = load_macro()
    bundle = DataBundle.load_default()
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})", flush=True)
    if macro_df is not None:
        print(f"    Macro: {len(macro_df)} days, cols={list(macro_df.columns[:5])}", flush=True)
    else:
        print("    Macro: NOT FOUND (VIX detection disabled)", flush=True)

    print("\n    Running all 4 strategies (full period)...", flush=True)
    strat_trades = run_all_strategies(h1_df, bundle)
    for sn in STRAT_ORDER:
        print(f"      {sn:>10s}: {len(strat_trades[sn])} trades", flush=True)

    base_daily = portfolio_daily(strat_trades)
    base_sharpe = round(_sharpe(base_daily), 3)
    base_dd = round(_max_dd(base_daily), 2)
    base_pnl = round(float(np.sum(base_daily)), 2)
    print(f"    Base portfolio: Sharpe={base_sharpe}, PnL=${base_pnl:,.0f}, MaxDD=${base_dd:,.0f}", flush=True)

    results['phase1'] = {
        'h1_bars': len(h1_df), 'macro_available': macro_df is not None,
        'base_sharpe': base_sharpe, 'base_pnl': base_pnl, 'base_max_dd': base_dd,
        'trade_counts': {sn: len(strat_trades[sn]) for sn in STRAT_ORDER},
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Parameter sensitivity sweep (skip_bars)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: skip_bars Parameter Sensitivity", flush=True)
    print("=" * 70, flush=True)

    extreme_mask_default = build_extreme_mask(h1_df, macro_df, cusum_sigma=3.0, extreme_window=24)
    n_extreme = int(np.sum(extreme_mask_default))
    print(f"    Default extreme mask: {n_extreme}/{len(h1_df)} bars ({n_extreme/len(h1_df)*100:.1f}%)", flush=True)

    skip_bars_values = [4, 8, 12, 16, 20, 24]
    phase2_results = {}

    print(f"\n  {'skip_bars':>10s} {'Sharpe':>8s} {'Delta':>8s} {'MaxDD':>10s} {'CVaR95':>10s} {'N_trades':>10s}", flush=True)
    print(f"  {'─'*58}", flush=True)
    print(f"  {'BASE':>10s} {base_sharpe:>8.3f} {'---':>8s} ${base_dd:>9,.0f} "
          f"${round(_cvar(base_daily, 0.05), 2):>9,.0f} "
          f"{'---':>10s}", flush=True)

    for sb in skip_bars_values:
        prot_trades = {}
        for sn in STRAT_ORDER:
            prot_trades[sn] = apply_protection(strat_trades[sn], h1_df, extreme_mask_default, skip_bars=sb)
        daily = portfolio_daily(prot_trades)
        sh = round(_sharpe(daily), 3)
        dd = round(_max_dd(daily), 2)
        cv95 = round(_cvar(daily, 0.05), 2)
        n_total = sum(len(prot_trades[s]) for s in STRAT_ORDER)
        delta = round(sh - base_sharpe, 3)
        phase2_results[sb] = {'sharpe': sh, 'delta': delta, 'max_dd': dd, 'cvar95': cv95, 'n_trades': n_total}
        print(f"  {sb:>10d} {sh:>8.3f} {delta:>+8.3f} ${dd:>9,.0f} ${cv95:>9,.0f} {n_total:>10d}", flush=True)

    positive_count = sum(1 for v in phase2_results.values() if v['delta'] > 0)
    best_sb = max(phase2_results, key=lambda k: phase2_results[k]['sharpe'])
    print(f"\n    Positive delta: {positive_count}/{len(skip_bars_values)} configs", flush=True)
    print(f"    Best skip_bars: {best_sb} (Sharpe={phase2_results[best_sb]['sharpe']:.3f})", flush=True)
    if positive_count <= 1:
        print("    WARNING: Only 1 or 0 configs improve Sharpe — likely OVERFIT!", flush=True)
    elif positive_count >= 4:
        print("    GOOD: Broad improvement across multiple skip_bars — robust signal", flush=True)

    results['phase2_skip_bars'] = {str(k): v for k, v in phase2_results.items()}
    results['phase2_summary'] = {'positive_count': positive_count, 'best_skip_bars': best_sb}

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Regime definition sensitivity (CUSUM sigma x window grid)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 3: Regime Definition Sensitivity Grid", flush=True)
    print("=" * 70, flush=True)

    cusum_sigmas = [2.0, 2.5, 3.0, 3.5, 4.0]
    extreme_windows = [12, 18, 24, 36]
    phase3_results = {}

    col_label = 'sigma\\window'
    header = f"  {col_label:>12s}"
    for ew in extreme_windows:
        header += f" {'w='+str(ew):>10s}"
    print(header, flush=True)
    print(f"  {'─'*(12 + 11*len(extreme_windows))}", flush=True)

    for cs in cusum_sigmas:
        row = f"  {cs:>12.1f}"
        for ew in extreme_windows:
            mask = build_extreme_mask(h1_df, macro_df, cusum_sigma=cs, extreme_window=ew)
            prot_trades = {}
            for sn in STRAT_ORDER:
                prot_trades[sn] = apply_protection(strat_trades[sn], h1_df, mask, skip_bars=best_sb)
            daily = portfolio_daily(prot_trades)
            sh = round(_sharpe(daily), 3)
            key = f"{cs}_{ew}"
            phase3_results[key] = {
                'cusum_sigma': cs, 'extreme_window': ew,
                'sharpe': sh, 'delta': round(sh - base_sharpe, 3),
                'n_extreme_bars': int(np.sum(mask)),
            }
            row += f" {sh:>10.3f}"
        print(row, flush=True)

    positive_grid = sum(1 for v in phase3_results.values() if v['delta'] > 0)
    total_grid = len(phase3_results)
    best_grid_key = max(phase3_results, key=lambda k: phase3_results[k]['sharpe'])
    best_grid = phase3_results[best_grid_key]
    print(f"\n    Positive delta: {positive_grid}/{total_grid} combos ({positive_grid/total_grid*100:.0f}%)", flush=True)
    print(f"    Best combo: sigma={best_grid['cusum_sigma']}, window={best_grid['extreme_window']} "
          f"(Sharpe={best_grid['sharpe']:.3f}, delta={best_grid['delta']:+.3f})", flush=True)
    print(f"    Base Sharpe: {base_sharpe:.3f}", flush=True)

    results['phase3_grid'] = phase3_results
    results['phase3_summary'] = {
        'positive_count': positive_grid, 'total': total_grid,
        'best_sigma': best_grid['cusum_sigma'], 'best_window': best_grid['extreme_window'],
        'best_sharpe': best_grid['sharpe'],
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Walk-Forward OOS (6 expanding windows)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 4: Walk-Forward Out-of-Sample (6 Expanding Windows)", flush=True)
    print("=" * 70, flush=True)

    wf_windows = [
        {'name': 'WF1', 'train': ('2015-01-01', '2018-12-31'), 'test': ('2019-01-01', '2019-12-31')},
        {'name': 'WF2', 'train': ('2015-01-01', '2019-12-31'), 'test': ('2020-01-01', '2020-12-31')},
        {'name': 'WF3', 'train': ('2015-01-01', '2020-12-31'), 'test': ('2021-01-01', '2021-12-31')},
        {'name': 'WF4', 'train': ('2015-01-01', '2021-12-31'), 'test': ('2022-01-01', '2022-12-31')},
        {'name': 'WF5', 'train': ('2015-01-01', '2022-12-31'), 'test': ('2023-01-01', '2023-12-31')},
        {'name': 'WF6', 'train': ('2015-01-01', '2023-12-31'), 'test': ('2024-01-01', '2026-05-01')},
    ]

    phase4_results = {}
    print(f"\n  {'Window':>8s} {'Train':>24s} {'Test':>24s} {'Base_Sh':>8s} {'RuleB_Sh':>9s} {'Delta':>8s}", flush=True)
    print(f"  {'─'*83}", flush=True)

    for wf in wf_windows:
        train_start, train_end = wf['train']
        test_start, test_end = wf['test']

        h1_train = h1_df[(h1_df.index >= train_start) & (h1_df.index <= train_end)]
        h1_test = h1_df[(h1_df.index >= test_start) & (h1_df.index <= test_end)]

        if len(h1_train) < 200 or len(h1_test) < 100:
            print(f"  {wf['name']:>8s} SKIPPED (insufficient data)", flush=True)
            continue

        train_mask = build_extreme_mask(h1_train, macro_df, cusum_sigma=3.0, extreme_window=24)

        test_trades = run_strategies_on_slice(h1_test)
        test_base_daily = portfolio_daily(test_trades)
        test_base_sh = round(_sharpe(test_base_daily), 3)

        test_mask = build_extreme_mask(h1_test, macro_df, cusum_sigma=3.0, extreme_window=24)
        test_prot = {}
        for sn in STRAT_ORDER:
            test_prot[sn] = apply_protection(test_trades[sn], h1_test, test_mask, skip_bars=best_sb)
        test_prot_daily = portfolio_daily(test_prot)
        test_prot_sh = round(_sharpe(test_prot_daily), 3)

        delta = round(test_prot_sh - test_base_sh, 3)
        phase4_results[wf['name']] = {
            'train': wf['train'], 'test': wf['test'],
            'base_sharpe': test_base_sh, 'ruleb_sharpe': test_prot_sh, 'delta': delta,
            'n_test_bars': len(h1_test), 'n_extreme_test': int(np.sum(test_mask)),
        }
        print(f"  {wf['name']:>8s} {train_start}~{train_end:>10s} {test_start}~{test_end:>10s} "
              f"{test_base_sh:>8.3f} {test_prot_sh:>9.3f} {delta:>+8.3f}", flush=True)

    wf_deltas = [v['delta'] for v in phase4_results.values()]
    wf_positive = sum(1 for d in wf_deltas if d > 0)
    wf_mean_delta = round(float(np.mean(wf_deltas)), 3) if wf_deltas else 0
    print(f"\n    OOS positive: {wf_positive}/{len(wf_deltas)} windows", flush=True)
    print(f"    Mean OOS delta: {wf_mean_delta:+.3f}", flush=True)
    if wf_positive >= 4:
        print("    PASS: Rule B holds in majority of OOS windows", flush=True)
    else:
        print("    WARN: Rule B does NOT consistently improve OOS", flush=True)

    results['phase4_walkforward'] = phase4_results
    results['phase4_summary'] = {'positive_windows': wf_positive, 'total_windows': len(wf_deltas),
                                  'mean_delta': wf_mean_delta}

    # ═══════════════════════════════════════════════════════════
    # Phase 5: Era analysis (3 eras)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 5: Era Analysis (3 Eras)", flush=True)
    print("=" * 70, flush=True)

    eras = [
        ('Era1_2015-2018', '2015-01-01', '2018-12-31'),
        ('Era2_2019-2022', '2019-01-01', '2022-12-31'),
        ('Era3_2023-2026', '2023-01-01', '2026-05-01'),
    ]
    phase5_results = {}

    print(f"\n  {'Era':>20s} {'Base_Sh':>8s} {'RuleB_Sh':>9s} {'Delta':>8s} {'Base_DD':>10s} {'RuleB_DD':>10s}", flush=True)
    print(f"  {'─'*67}", flush=True)

    for era_name, era_start, era_end in eras:
        h1_era = h1_df[(h1_df.index >= era_start) & (h1_df.index <= era_end)]
        if len(h1_era) < 200:
            print(f"  {era_name:>20s} SKIPPED (insufficient data)", flush=True)
            continue

        era_trades = run_strategies_on_slice(h1_era)
        era_base_daily = portfolio_daily(era_trades)
        era_base_sh = round(_sharpe(era_base_daily), 3)
        era_base_dd = round(_max_dd(era_base_daily), 2)

        era_mask = build_extreme_mask(h1_era, macro_df, cusum_sigma=3.0, extreme_window=24)
        era_prot = {}
        for sn in STRAT_ORDER:
            era_prot[sn] = apply_protection(era_trades[sn], h1_era, era_mask, skip_bars=best_sb)
        era_prot_daily = portfolio_daily(era_prot)
        era_prot_sh = round(_sharpe(era_prot_daily), 3)
        era_prot_dd = round(_max_dd(era_prot_daily), 2)

        delta = round(era_prot_sh - era_base_sh, 3)
        phase5_results[era_name] = {
            'base_sharpe': era_base_sh, 'ruleb_sharpe': era_prot_sh, 'delta': delta,
            'base_max_dd': era_base_dd, 'ruleb_max_dd': era_prot_dd,
            'n_bars': len(h1_era), 'n_extreme': int(np.sum(era_mask)),
        }
        print(f"  {era_name:>20s} {era_base_sh:>8.3f} {era_prot_sh:>9.3f} {delta:>+8.3f} "
              f"${era_base_dd:>9,.0f} ${era_prot_dd:>9,.0f}", flush=True)

    era_deltas = [v['delta'] for v in phase5_results.values()]
    era_positive = sum(1 for d in era_deltas if d > 0)
    consistent = era_positive >= 2
    print(f"\n    Consistent across eras: {era_positive}/{len(era_deltas)} positive", flush=True)
    print(f"    Verdict: {'CONSISTENT' if consistent else 'INCONSISTENT'}", flush=True)

    results['phase5_era'] = phase5_results
    results['phase5_summary'] = {'positive_eras': era_positive, 'consistent': consistent}

    # ═══════════════════════════════════════════════════════════
    # Phase 6: Monte Carlo (1000 bootstrap paths)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 6: Monte Carlo Bootstrap (1000 paths)", flush=True)
    print("=" * 70, flush=True)

    extreme_mask_full = build_extreme_mask(h1_df, macro_df, cusum_sigma=3.0, extreme_window=24)

    all_base_trades = []
    all_prot_trades = []
    for sn in STRAT_ORDER:
        mult = R89_LOTS[sn] / UNIT_LOT
        for t in strat_trades[sn]:
            tc = dict(t); tc['pnl'] = t['pnl'] * mult
            all_base_trades.append(tc)
        prot = apply_protection(strat_trades[sn], h1_df, extreme_mask_full, skip_bars=best_sb)
        for t in prot:
            tc = dict(t); tc['pnl'] = t['pnl'] * mult
            all_prot_trades.append(tc)

    base_pnls_arr = np.array([t['pnl'] for t in all_base_trades])
    prot_pnls_arr = np.array([t['pnl'] for t in all_prot_trades])

    n_mc = 1000
    rng = np.random.default_rng(42)
    base_sharpes_mc = np.zeros(n_mc)
    prot_sharpes_mc = np.zeros(n_mc)

    print(f"    Base trades: {len(base_pnls_arr)}, Protected trades: {len(prot_pnls_arr)}", flush=True)
    print(f"    Running {n_mc} bootstrap iterations...", flush=True)

    for mc_i in range(n_mc):
        b_sample = rng.choice(base_pnls_arr, size=len(base_pnls_arr), replace=True)
        n_days_b = max(len(b_sample) // 3, 10)
        daily_chunks_b = np.array_split(b_sample, n_days_b)
        daily_b = np.array([chunk.sum() for chunk in daily_chunks_b])
        base_sharpes_mc[mc_i] = _sharpe(daily_b)

        p_sample = rng.choice(prot_pnls_arr, size=len(prot_pnls_arr), replace=True)
        n_days_p = max(len(p_sample) // 3, 10)
        daily_chunks_p = np.array_split(p_sample, n_days_p)
        daily_p = np.array([chunk.sum() for chunk in daily_chunks_p])
        prot_sharpes_mc[mc_i] = _sharpe(daily_p)

    base_mc_mean = round(float(np.mean(base_sharpes_mc)), 3)
    base_mc_p5 = round(float(np.percentile(base_sharpes_mc, 5)), 3)
    base_mc_p95 = round(float(np.percentile(base_sharpes_mc, 95)), 3)
    prot_mc_mean = round(float(np.mean(prot_sharpes_mc)), 3)
    prot_mc_p5 = round(float(np.percentile(prot_sharpes_mc, 5)), 3)
    prot_mc_p95 = round(float(np.percentile(prot_sharpes_mc, 95)), 3)

    prot_beats_base = float(np.mean(prot_sharpes_mc > base_sharpes_mc)) * 100

    print(f"\n    {'Metric':>20s} {'Base':>12s} {'Rule B':>12s}", flush=True)
    print(f"    {'─'*46}", flush=True)
    print(f"    {'MC Mean Sharpe':>20s} {base_mc_mean:>12.3f} {prot_mc_mean:>12.3f}", flush=True)
    print(f"    {'MC P5 Sharpe':>20s} {base_mc_p5:>12.3f} {prot_mc_p5:>12.3f}", flush=True)
    print(f"    {'MC P95 Sharpe':>20s} {base_mc_p95:>12.3f} {prot_mc_p95:>12.3f}", flush=True)
    print(f"    {'P(RuleB > Base)':>20s} {prot_beats_base:>11.1f}%", flush=True)

    mc_verdict = "PASS" if prot_beats_base > 55 else ("MARGINAL" if prot_beats_base > 45 else "FAIL")
    print(f"\n    MC Verdict: {mc_verdict} (Rule B beats Base in {prot_beats_base:.1f}% of paths)", flush=True)

    results['phase6_montecarlo'] = {
        'n_paths': n_mc,
        'base': {'mean': base_mc_mean, 'p5': base_mc_p5, 'p95': base_mc_p95},
        'ruleb': {'mean': prot_mc_mean, 'p5': prot_mc_p5, 'p95': prot_mc_p95},
        'prot_beats_base_pct': round(prot_beats_base, 1),
        'verdict': mc_verdict,
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 7: 8-stage StrategyValidator
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 7: 8-Stage StrategyValidator (Rule B Portfolio)", flush=True)
    print("=" * 70, flush=True)

    best_skip = best_sb

    def ruleb_backtest_fn(h1_df_in, spread, lot):
        """Combined 4-strategy portfolio with Rule B applied."""
        trades_out = []
        psar_trades = bt_psar(h1_df_in, spread, lot, CAPS['PSAR'])
        tsmom_trades = bt_tsmom(h1_df_in, spread, lot, CAPS['TSMOM'])
        sessbo_trades = bt_sess_bo(h1_df_in, spread, lot, CAPS['SESS_BO'])

        mask = build_extreme_mask(h1_df_in, macro_df, cusum_sigma=3.0, extreme_window=24)

        for sn, tr_list in [('PSAR', psar_trades), ('TSMOM', tsmom_trades), ('SESS_BO', sessbo_trades)]:
            protected = apply_protection(tr_list, h1_df_in, mask, skip_bars=best_skip)
            trades_out.extend(protected)

        return trades_out

    validator_config = ValidatorConfig(
        n_trials_tested=len(skip_bars_values) * len(cusum_sigmas) * len(extreme_windows),
        realistic_spread=0.88,
        n_bootstrap=1000,
    )

    validator = StrategyValidator(
        name="R144_RuleB_Portfolio",
        backtest_fn=ruleb_backtest_fn,
        spread=SPREAD,
        lot=UNIT_LOT,
        config=validator_config,
        h1_df=h1_df,
        output_dir=str(OUTPUT_DIR),
    )

    print("    Running validator...", flush=True)
    try:
        validator_report = validator.run_all()
        print(f"    Validator complete.", flush=True)

        phase7_results = {}
        if isinstance(validator_report, dict):
            for stage_key, stage_val in validator_report.items():
                if hasattr(stage_val, '__dict__'):
                    phase7_results[str(stage_key)] = {
                        'passed': getattr(stage_val, 'passed', None),
                        'sharpe': getattr(stage_val, 'sharpe', None),
                        'verdict': getattr(stage_val, 'verdict', ''),
                    }
                else:
                    phase7_results[str(stage_key)] = stage_val

            n_passed = sum(1 for v in phase7_results.values()
                          if isinstance(v, dict) and v.get('passed') is True)
            n_stages = len(phase7_results)
            print(f"    Stages passed: {n_passed}/{n_stages}", flush=True)
        elif isinstance(validator_report, list):
            for sr in validator_report:
                phase7_results[f"stage_{sr.stage}"] = {
                    'name': sr.name, 'passed': sr.passed,
                    'sharpe': sr.sharpe, 'verdict': sr.verdict,
                }
                status = "PASS" if sr.passed else "FAIL"
                print(f"      Stage {sr.stage}: {sr.name} — {status} (Sharpe={sr.sharpe:.3f})", flush=True)
            n_passed = sum(1 for sr in validator_report if sr.passed)
            n_stages = len(validator_report)
            print(f"    Stages passed: {n_passed}/{n_stages}", flush=True)
        else:
            phase7_results = {'raw': str(validator_report)}
            print(f"    Validator returned: {type(validator_report)}", flush=True)

        results['phase7_validator'] = phase7_results
    except Exception as e:
        print(f"    Validator ERROR: {e}", flush=True)
        results['phase7_validator'] = {'error': str(e)}

    # ═══════════════════════════════════════════════════════════
    # Phase 8: Integration with v2 production config (PSAR optimized)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 8: v2 Production Config (PSAR Optimized) + Rule B", flush=True)
    print("=" * 70, flush=True)

    v2_psar_params = {
        'sl_atr': 4.0, 'tp_atr': 6.0,
        'trail_act': 0.08, 'trail_dist': 0.015,
        'max_hold': 15,
    }

    print(f"    v2 PSAR params: {v2_psar_params}", flush=True)

    v2_psar_trades = bt_psar(h1_df, SPREAD, UNIT_LOT, CAPS['PSAR'], params=v2_psar_params)
    print(f"    v2 PSAR trades: {len(v2_psar_trades)}", flush=True)

    v2_strat_trades = dict(strat_trades)
    v2_strat_trades['PSAR'] = v2_psar_trades

    v2_base_daily = portfolio_daily(v2_strat_trades)
    v2_base_sharpe = round(_sharpe(v2_base_daily), 3)
    v2_base_dd = round(_max_dd(v2_base_daily), 2)
    v2_base_pnl = round(float(np.sum(v2_base_daily)), 2)

    v2_prot_trades = {}
    for sn in STRAT_ORDER:
        v2_prot_trades[sn] = apply_protection(v2_strat_trades[sn], h1_df, extreme_mask_full, skip_bars=best_sb)
    v2_prot_daily = portfolio_daily(v2_prot_trades)
    v2_prot_sharpe = round(_sharpe(v2_prot_daily), 3)
    v2_prot_dd = round(_max_dd(v2_prot_daily), 2)
    v2_prot_pnl = round(float(np.sum(v2_prot_daily)), 2)
    v2_prot_cvar95 = round(_cvar(v2_prot_daily, 0.05), 2)
    v2_prot_cvar99 = round(_cvar(v2_prot_daily, 0.01), 2)

    v2_delta = round(v2_prot_sharpe - v2_base_sharpe, 3)

    print(f"\n  {'Config':>25s} {'Sharpe':>8s} {'PnL':>12s} {'MaxDD':>10s}", flush=True)
    print(f"  {'─'*57}", flush=True)
    print(f"  {'v2_base':>25s} {v2_base_sharpe:>8.3f} ${v2_base_pnl:>11,.0f} ${v2_base_dd:>9,.0f}", flush=True)
    print(f"  {'v2 + Rule B':>25s} {v2_prot_sharpe:>8.3f} ${v2_prot_pnl:>11,.0f} ${v2_prot_dd:>9,.0f}", flush=True)
    print(f"  {'Delta':>25s} {v2_delta:>+8.3f}", flush=True)

    results['phase8_v2'] = {
        'v2_psar_params': v2_psar_params,
        'v2_base': {'sharpe': v2_base_sharpe, 'pnl': v2_base_pnl, 'max_dd': v2_base_dd},
        'v2_with_ruleb': {
            'sharpe': v2_prot_sharpe, 'pnl': v2_prot_pnl, 'max_dd': v2_prot_dd,
            'cvar95': v2_prot_cvar95, 'cvar99': v2_prot_cvar99,
        },
        'delta': v2_delta,
    }

    # ═══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)

    print("\n" + "=" * 80, flush=True)
    print("  R144 FINAL SUMMARY — Rule B Deep Validation", flush=True)
    print("=" * 80, flush=True)

    print(f"\n  Phase 2 — skip_bars sensitivity:", flush=True)
    print(f"    {positive_count}/{len(skip_bars_values)} skip_bars values improve Sharpe", flush=True)
    print(f"    Best skip_bars={best_sb} (Sharpe={phase2_results[best_sb]['sharpe']:.3f})", flush=True)

    print(f"\n  Phase 3 — Regime definition grid:", flush=True)
    print(f"    {positive_grid}/{total_grid} combos improve Sharpe ({positive_grid/total_grid*100:.0f}%)", flush=True)
    print(f"    Best: sigma={best_grid['cusum_sigma']}, window={best_grid['extreme_window']}", flush=True)

    print(f"\n  Phase 4 — Walk-Forward OOS:", flush=True)
    print(f"    {wf_positive}/{len(wf_deltas)} windows positive (mean delta={wf_mean_delta:+.3f})", flush=True)

    print(f"\n  Phase 5 — Era consistency:", flush=True)
    print(f"    {era_positive}/{len(era_deltas)} eras positive — {'CONSISTENT' if consistent else 'INCONSISTENT'}", flush=True)

    print(f"\n  Phase 6 — Monte Carlo:", flush=True)
    print(f"    Rule B beats Base in {prot_beats_base:.1f}% of paths — {mc_verdict}", flush=True)

    print(f"\n  Phase 8 — v2 integration:", flush=True)
    print(f"    v2+RuleB delta: {v2_delta:+.3f} Sharpe", flush=True)

    overfit_signals = 0
    robust_signals = 0

    if positive_count <= 1:
        overfit_signals += 1
    elif positive_count >= 4:
        robust_signals += 1

    if positive_grid / total_grid < 0.3:
        overfit_signals += 1
    elif positive_grid / total_grid >= 0.5:
        robust_signals += 1

    if wf_positive < 3:
        overfit_signals += 1
    elif wf_positive >= 4:
        robust_signals += 1

    if not consistent:
        overfit_signals += 1
    else:
        robust_signals += 1

    if prot_beats_base < 50:
        overfit_signals += 1
    elif prot_beats_base > 55:
        robust_signals += 1

    if v2_delta <= 0:
        overfit_signals += 1
    else:
        robust_signals += 1

    if overfit_signals >= 3:
        final_verdict = "OVERFIT — Rule B benefit is NOT robust"
    elif robust_signals >= 4:
        final_verdict = "ROBUST — Rule B provides genuine protection"
    else:
        final_verdict = "MARGINAL — Rule B has some value but not conclusive"

    print(f"\n  Overfit signals: {overfit_signals}/6", flush=True)
    print(f"  Robust signals: {robust_signals}/6", flush=True)
    print(f"\n  *** FINAL VERDICT: {final_verdict} ***", flush=True)

    results['final_verdict'] = {
        'overfit_signals': overfit_signals,
        'robust_signals': robust_signals,
        'verdict': final_verdict,
    }

    out_file = OUTPUT_DIR / "r144_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"  Saved: {out_file}", flush=True)
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
