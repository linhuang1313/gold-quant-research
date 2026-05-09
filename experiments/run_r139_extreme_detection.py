#!/usr/bin/env python3
"""
R139 — Extreme Market Detection + Adaptive Protection
=======================================================
Detect extreme market regimes (COVID, Russia-Ukraine, Liberation Day, etc.)
using CUSUM on ATR + VIX spike detection, then evaluate adaptive protection
rules (lot reduction, entry skip, tighter trailing, full shutdown).

Phases:
  1. Load H1 + macro data, identify historical extreme events
  2. Build regime-shift detector (CUSUM on ATR + VIX spikes)
  3. Define 4 adaptive protection rules
  4. Run all 4 base strategies with each protection rule
  5. Compare base vs protected (Sharpe, MaxDD, tail metrics)
  6. K-Fold 5-fold
  7. CVaR analysis (CVaR95/99 base vs protected)
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

OUTPUT_DIR = Path("results/r139_extreme_detection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01
CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.08, 'SESS_BO': 0.08}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']

EXTREME_EVENTS = {
    'COVID':       ('2020-02-20', '2020-04-15'),
    'RUS_UKR':     ('2022-02-24', '2022-04-01'),
    'LIBERATION':  ('2025-04-02', '2025-04-15'),
}

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


def _calmar(arr):
    dd = _max_dd(arr)
    if dd == 0 or len(arr) < 10:
        return 0.0
    n_years = max(len(arr) / 252, 0.5)
    return round(float(np.sum(arr)) / n_years / dd, 3)


def _cvar(arr, alpha=0.05):
    """Conditional Value at Risk (expected shortfall)."""
    if len(arr) < 20:
        return 0.0
    threshold = np.percentile(arr, alpha * 100)
    tail = arr[arr <= threshold]
    return float(np.mean(tail)) if len(tail) > 0 else 0.0


def _compute_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0, 'calmar': 0,
                'cvar95': 0, 'cvar99': 0}
    daily = _trades_to_daily(trades)
    pnls = np.array([t['pnl'] for t in trades])
    n = len(trades)
    return {
        'n': n, 'sharpe': round(_sharpe(daily), 3),
        'pnl': round(float(np.sum(pnls)), 2),
        'wr': round(float(np.sum(pnls > 0)) / n * 100, 1),
        'max_dd': round(_max_dd(daily), 2),
        'calmar': _calmar(daily),
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
# Extreme regime detector
# ═══════════════════════════════════════════════════════════════

def build_extreme_mask(h1_df, macro_df):
    """Build boolean mask: True = EXTREME mode for that H1 bar."""
    n = len(h1_df)
    extreme = np.zeros(n, dtype=bool)
    atr = compute_atr(h1_df).values

    # CUSUM on ATR: detect when ATR jumps 3sigma above its 60-bar rolling mean
    atr_clean = np.nan_to_num(atr, nan=0.0)
    atr_mean = pd.Series(atr_clean).rolling(60, min_periods=20).mean().values
    atr_std = pd.Series(atr_clean).rolling(60, min_periods=20).std().values
    atr_std = np.maximum(atr_std, 1e-6)
    cusum_trigger = atr_clean > (atr_mean + 3.0 * atr_std)

    for i in range(n):
        if cusum_trigger[i]:
            end_i = min(i + 24, n)
            extreme[i:end_i] = True

    # VIX spike detector from macro data
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
                    end_i = min(bi + 24, n)
                    extreme[bi:end_i] = True

    return extreme


def apply_protection(trades, h1_df, extreme_mask, rule, lot_base):
    """Apply protection rule to trades based on extreme mask.

    Rules:
      A: Reduce lot by 50% for bars in EXTREME window
      B: Skip new entries for 12 bars after EXTREME trigger, keep existing
      C: Tighten trailing stop (trail_dist *= 0.5) — approximated via PnL haircut
      D: Full shutdown for 24 bars after EXTREME trigger
    """
    if not trades:
        return []

    times_idx = h1_df.index
    n_bars = len(times_idx)

    extreme_entry_bars = set()
    if n_bars > 0:
        for i in range(n_bars):
            if extreme_mask[i]:
                extreme_entry_bars.add(i)

    def _find_bar(ts):
        ts = pd.Timestamp(ts)
        idx = times_idx.searchsorted(ts)
        return min(idx, n_bars - 1) if idx < n_bars else n_bars - 1

    protected = []
    for t in trades:
        entry_bar = _find_bar(t['entry_time'])
        is_extreme = entry_bar < n_bars and extreme_mask[entry_bar]

        if rule == 'A':
            if is_extreme:
                tc = dict(t)
                tc['pnl'] = t['pnl'] * 0.5
                protected.append(tc)
            else:
                protected.append(t)

        elif rule == 'B':
            skip_end = entry_bar
            for j in range(max(0, entry_bar - 12), entry_bar):
                if j < n_bars and extreme_mask[j]:
                    skip_end = j + 12
                    break
            if entry_bar < skip_end:
                continue
            protected.append(t)

        elif rule == 'C':
            if is_extreme:
                tc = dict(t)
                if t['pnl'] > 0:
                    tc['pnl'] = t['pnl'] * 0.85
                else:
                    tc['pnl'] = t['pnl'] * 0.7
                protected.append(tc)
            else:
                protected.append(t)

        elif rule == 'D':
            if is_extreme:
                continue
            any_extreme_nearby = False
            for j in range(max(0, entry_bar - 24), entry_bar):
                if j < n_bars and extreme_mask[j]:
                    any_extreme_nearby = True
                    break
            if any_extreme_nearby:
                continue
            protected.append(t)

    return protected


# ═══════════════════════════════════════════════════════════════
# Portfolio helpers
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


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    results = {'experiment': 'R139 Extreme Market Detection + Adaptive Protection'}

    print("=" * 80, flush=True)
    print("  R139 — Extreme Market Detection + Adaptive Protection", flush=True)
    print("=" * 80, flush=True)

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Load data + identify extreme events
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

    print("\n  Known extreme events:", flush=True)
    for name, (start, end) in EXTREME_EVENTS.items():
        h1_slice = h1_df[(h1_df.index >= start) & (h1_df.index <= end)]
        atr_vals = compute_atr(h1_slice).dropna()
        mean_atr = float(atr_vals.mean()) if len(atr_vals) > 0 else 0
        print(f"    {name:>14s}: {start} ~ {end}  ({len(h1_slice)} bars, mean ATR={mean_atr:.2f})", flush=True)

    results['phase1'] = {'h1_bars': len(h1_df), 'macro_available': macro_df is not None,
                         'events': {k: list(v) for k, v in EXTREME_EVENTS.items()}}

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Build regime-shift detector
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: Building Extreme Regime Detector", flush=True)
    print("=" * 70, flush=True)

    extreme_mask = build_extreme_mask(h1_df, macro_df)
    n_extreme = int(np.sum(extreme_mask))
    pct_extreme = n_extreme / len(h1_df) * 100
    print(f"    Extreme bars: {n_extreme}/{len(h1_df)} ({pct_extreme:.1f}%)", flush=True)

    for name, (start, end) in EXTREME_EVENTS.items():
        mask_slice = h1_df.index[(h1_df.index >= start) & (h1_df.index <= end)]
        idx_start = h1_df.index.searchsorted(mask_slice[0]) if len(mask_slice) > 0 else 0
        idx_end = h1_df.index.searchsorted(mask_slice[-1]) + 1 if len(mask_slice) > 0 else 0
        detected = int(np.sum(extreme_mask[idx_start:idx_end]))
        total = idx_end - idx_start
        print(f"    {name:>14s}: {detected}/{total} bars detected as extreme", flush=True)

    results['phase2'] = {'n_extreme_bars': n_extreme, 'pct_extreme': round(pct_extreme, 2)}

    # ═══════════════════════════════════════════════════════════
    # Phase 3-4: Run strategies + apply protection rules
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 3-4: Running Strategies + Protection Rules", flush=True)
    print("=" * 70, flush=True)

    strat_trades = run_all_strategies(h1_df, bundle)
    for sn in STRAT_ORDER:
        print(f"    {sn:>10s}: {len(strat_trades[sn])} trades", flush=True)

    rules = ['A', 'B', 'C', 'D']
    rule_names = {
        'A': 'Lot reduction 50%',
        'B': 'Skip entries 12 bars',
        'C': 'Tighter trailing',
        'D': 'Full shutdown 24 bars',
    }

    base_daily = portfolio_daily(strat_trades)
    base_stats = {
        'sharpe': round(_sharpe(base_daily), 3),
        'pnl': round(float(np.sum(base_daily)), 2),
        'max_dd': round(_max_dd(base_daily), 2),
        'calmar': _calmar(base_daily),
        'cvar95': round(_cvar(base_daily, 0.05), 2),
        'cvar99': round(_cvar(base_daily, 0.01), 2),
    }

    print(f"\n  Base portfolio: Sharpe={base_stats['sharpe']:.3f}, "
          f"PnL=${base_stats['pnl']:,.0f}, MaxDD=${base_stats['max_dd']:,.0f}", flush=True)

    rule_results = {}
    print(f"\n  {'Rule':<30s} {'Sharpe':>7} {'PnL':>10} {'MaxDD':>9} {'CVaR95':>9} {'CVaR99':>9}", flush=True)
    print(f"  {'─'*75}", flush=True)
    print(f"  {'BASE (no protection)':<30s} {base_stats['sharpe']:>7.3f} "
          f"${base_stats['pnl']:>9,.0f} ${base_stats['max_dd']:>8,.0f} "
          f"${base_stats['cvar95']:>8,.0f} ${base_stats['cvar99']:>8,.0f}", flush=True)

    for rule in rules:
        protected_trades = {}
        for sn in STRAT_ORDER:
            protected_trades[sn] = apply_protection(
                strat_trades[sn], h1_df, extreme_mask, rule, R89_LOTS[sn])

        daily = portfolio_daily(protected_trades)
        stats = {
            'sharpe': round(_sharpe(daily), 3),
            'pnl': round(float(np.sum(daily)), 2),
            'max_dd': round(_max_dd(daily), 2),
            'calmar': _calmar(daily),
            'cvar95': round(_cvar(daily, 0.05), 2),
            'cvar99': round(_cvar(daily, 0.01), 2),
            'n_trades': sum(len(protected_trades[s]) for s in STRAT_ORDER),
        }
        rule_results[rule] = stats
        label = f"Rule {rule}: {rule_names[rule]}"
        print(f"  {label:<30s} {stats['sharpe']:>7.3f} "
              f"${stats['pnl']:>9,.0f} ${stats['max_dd']:>8,.0f} "
              f"${stats['cvar95']:>8,.0f} ${stats['cvar99']:>8,.0f}", flush=True)

    results['phase3_4'] = {'base': base_stats, 'rules': rule_results}

    # ═══════════════════════════════════════════════════════════
    # Phase 5: Detailed comparison
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 5: Base vs Protected — Detailed Comparison", flush=True)
    print("=" * 70, flush=True)

    print(f"\n  {'Metric':<15s} {'Base':>10s}", end="", flush=True)
    for r in rules:
        print(f" {'Rule'+r:>10s}", end="")
    print(flush=True)
    print(f"  {'─'*15}" + " " + " ".join("─"*10 for _ in rules), flush=True)

    for metric in ['sharpe', 'pnl', 'max_dd', 'calmar', 'cvar95', 'cvar99']:
        bv = base_stats[metric]
        line = f"  {metric:<15s} {bv:>10.2f}"
        for r in rules:
            rv = rule_results[r][metric]
            line += f" {rv:>10.2f}"
        print(line, flush=True)

    best_rule = max(rules, key=lambda r: rule_results[r]['sharpe'])
    print(f"\n  Best by Sharpe: Rule {best_rule} ({rule_names[best_rule]})", flush=True)
    best_dd_rule = min(rules, key=lambda r: rule_results[r]['max_dd'])
    print(f"  Best by MaxDD: Rule {best_dd_rule} ({rule_names[best_dd_rule]})", flush=True)
    best_cvar = min(rules, key=lambda r: abs(rule_results[r]['cvar99']))
    print(f"  Best by CVaR99: Rule {best_cvar} ({rule_names[best_cvar]})", flush=True)

    results['phase5'] = {'best_sharpe_rule': best_rule, 'best_dd_rule': best_dd_rule,
                         'best_cvar_rule': best_cvar}

    # ═══════════════════════════════════════════════════════════
    # Phase 6: K-Fold 5-fold
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 6: K-Fold 5-Fold Validation", flush=True)
    print("=" * 70, flush=True)

    fold_starts = pd.date_range("2015-01-01", "2026-05-01", periods=6)
    kfold_results = {'base': []}
    for r in rules:
        kfold_results[f'rule_{r}'] = []

    for fi in range(5):
        fs = str(fold_starts[fi].date())
        fe = str(fold_starts[fi + 1].date())
        h1_fold = h1_df[(h1_df.index >= fs) & (h1_df.index < fe)]
        if len(h1_fold) < 200:
            continue

        fold_mask = build_extreme_mask(h1_fold, macro_df)
        fold_trades = {}
        fold_trades['PSAR'] = bt_psar(h1_fold, SPREAD, UNIT_LOT, CAPS['PSAR'])
        fold_trades['TSMOM'] = bt_tsmom(h1_fold, SPREAD, UNIT_LOT, CAPS['TSMOM'])
        fold_trades['SESS_BO'] = bt_sess_bo(h1_fold, SPREAD, UNIT_LOT, CAPS['SESS_BO'])
        fold_trades['L8_MAX'] = []

        fold_base = portfolio_daily(fold_trades)
        kfold_results['base'].append(round(_sharpe(fold_base), 3))

        for r in rules:
            prot = {}
            for sn in ['PSAR', 'TSMOM', 'SESS_BO']:
                prot[sn] = apply_protection(fold_trades[sn], h1_fold, fold_mask, r, R89_LOTS[sn])
            prot['L8_MAX'] = []
            fold_daily = portfolio_daily(prot)
            kfold_results[f'rule_{r}'].append(round(_sharpe(fold_daily), 3))

    print(f"\n  {'Config':<15s} {'Fold Sharpes':>35s} {'Mean':>6} {'Min':>6}", flush=True)
    print(f"  {'─'*65}", flush=True)
    for key, sharpes in kfold_results.items():
        if not sharpes:
            continue
        fold_str = ', '.join(f'{s:.2f}' for s in sharpes)
        m = np.mean(sharpes); mn = min(sharpes)
        print(f"  {key:<15s} [{fold_str:>33s}] {m:>6.2f} {mn:>6.2f}", flush=True)

    results['phase6_kfold'] = kfold_results

    # ═══════════════════════════════════════════════════════════
    # Phase 7: CVaR analysis
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 7: CVaR Analysis — Base vs Protected", flush=True)
    print("=" * 70, flush=True)

    cvar_table = {'base': {'cvar95': base_stats['cvar95'], 'cvar99': base_stats['cvar99']}}
    for r in rules:
        cvar_table[f'rule_{r}'] = {
            'cvar95': rule_results[r]['cvar95'],
            'cvar99': rule_results[r]['cvar99'],
        }

    print(f"\n  {'Config':<15s} {'CVaR95':>10s} {'CVaR99':>10s} {'Delta95':>10s} {'Delta99':>10s}", flush=True)
    print(f"  {'─'*55}", flush=True)
    print(f"  {'base':<15s} ${base_stats['cvar95']:>9,.2f} ${base_stats['cvar99']:>9,.2f} "
          f"{'---':>10s} {'---':>10s}", flush=True)
    for r in rules:
        d95 = rule_results[r]['cvar95'] - base_stats['cvar95']
        d99 = rule_results[r]['cvar99'] - base_stats['cvar99']
        print(f"  {'rule_'+r:<15s} ${rule_results[r]['cvar95']:>9,.2f} "
              f"${rule_results[r]['cvar99']:>9,.2f} "
              f"${d95:>+9,.2f} ${d99:>+9,.2f}", flush=True)

    results['phase7_cvar'] = cvar_table

    # ═══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)

    print("\n" + "=" * 80, flush=True)
    print("  R139 FINAL SUMMARY", flush=True)
    print("=" * 80, flush=True)

    print(f"\n  Extreme regime detection: {n_extreme} bars ({pct_extreme:.1f}% of data)", flush=True)
    print(f"  Best protection by Sharpe: Rule {best_rule} ({rule_names[best_rule]})", flush=True)
    print(f"  Best protection by MaxDD:  Rule {best_dd_rule} ({rule_names[best_dd_rule]})", flush=True)
    print(f"  Best protection by CVaR99: Rule {best_cvar} ({rule_names[best_cvar]})", flush=True)

    sharpe_improved = any(rule_results[r]['sharpe'] > base_stats['sharpe'] for r in rules)
    dd_improved = any(rule_results[r]['max_dd'] < base_stats['max_dd'] for r in rules)
    print(f"\n  Sharpe improved by any rule: {'YES' if sharpe_improved else 'NO'}", flush=True)
    print(f"  MaxDD improved by any rule:  {'YES' if dd_improved else 'NO'}", flush=True)

    recommendation = best_rule if sharpe_improved and dd_improved else 'NONE'
    results['recommendation'] = recommendation
    print(f"\n  Recommendation: {'Rule ' + recommendation if recommendation != 'NONE' else 'No protection needed'}", flush=True)

    out_file = OUTPUT_DIR / "r139_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"  Saved: {out_file}", flush=True)
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
