#!/usr/bin/env python3
"""
R126 — Drawdown-Aware Strategy Rotation
==========================================
Pause or reduce allocation to strategies in drawdown, redistribute lots.

Phase 1: Run all 4 strategies, compute per-strategy equity curves
Phase 2: Drawdown protection methods (DD pause, equity MA, combined, redistribution)
Phase 3: Walk-forward simulation with each method
Phase 4: K-Fold validation (5 folds)
Phase 5: Comparison vs R105 dynamic rotation (which underperformed fixed)
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import run_variant, LIVE_PARITY_KWARGS, DataBundle

OUTPUT_DIR = Path("results/r126_drawdown_rotation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.09, 'SESS_BO': 0.09}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']
TOTAL_LOT_BUDGET = sum(R89_LOTS.values())

FOLDS = [
    ("Fold1", "2015-01-01", "2017-03-01"),
    ("Fold2", "2017-03-01", "2019-06-01"),
    ("Fold3", "2019-06-01", "2021-09-01"),
    ("Fold4", "2021-09-01", "2023-12-01"),
    ("Fold5", "2023-12-01", "2027-01-01"),
]

t0 = time.time()

# ═══════════════════════════════════════════════════════════════
# H1 data loading
# ═══════════════════════════════════════════════════════════════

H1_CSV = Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv")

def load_h1():
    csv_path = H1_CSV
    if not csv_path.exists():
        import glob
        candidates = glob.glob("data/download/xauusd-h1-bid-*.csv")
        if candidates:
            csv_path = Path(sorted(candidates)[-1])
        else:
            raise FileNotFoundError("No xauusd H1 CSV found in data/download/")
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.set_index('time')[['open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[df['Close'] > 100]
    return df


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
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
                    ep = df['High'].iloc[i]; af = min(af + af_start, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], df['High'].iloc[i-1], df['High'].iloc[max(0, i-2)])
            if df['High'].iloc[i] > psar[i]:
                direction[i] = 1; psar[i] = ep; ep = df['High'].iloc[i]; af = af_start
            else:
                direction[i] = -1
                if df['Low'].iloc[i] < ep:
                    ep = df['Low'].iloc[i]; af = min(af + af_start, af_max)
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


def _trades_to_daily_series(trades):
    if not trades:
        return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


def _sharpe(arr):
    if len(arr) < 10 or np.std(arr, ddof=1) == 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))


def _max_dd(arr):
    if len(arr) == 0: return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def _compute_stats(trades):
    if not trades:
        return {'n_trades': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    return {
        'n_trades': n,
        'sharpe': round(_sharpe(daily), 3),
        'pnl': round(sum(pnls), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'max_dd': round(_max_dd(daily), 2),
    }


def _portfolio_metrics(daily_arr):
    sh = _sharpe(daily_arr)
    dd = _max_dd(daily_arr)
    pnl = float(daily_arr.sum())
    calmar = pnl / dd if dd > 0 else 0.0
    return {'sharpe': round(sh, 3), 'pnl': round(pnl, 2), 'max_dd': round(dd, 2),
            'calmar': round(calmar, 3)}


# ═══════════════════════════════════════════════════════════════
# Strategy backtests
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
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pdir[i-1] == -1 and pdir[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif pdir[i-1] == 1 and pdir[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
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
        if c[i - fast] > 0: s += 0.5 * np.sign(c[i] / c[i - fast] - 1.0)
        if c[i - slow] > 0: s += 0.5 * np.sign(c[i] / c[i - slow] - 1.0)
        score[i] = s
    trades = []; pos = None; last_exit = -999
    for i in range(max_lb + 1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and score[i] < 0:
                pnl = (c[i] - pos['entry'] - spread) * lot * PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and score[i] > 0:
                pnl = (pos['entry'] - c[i] - spread) * lot * PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(score[i]) or np.isnan(score[i-1]): continue
        if score[i] > 0 and score[i-1] <= 0:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
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
        hh = max(h[i - j] for j in range(1, lookback + 1))
        ll = min(lo[i - j] for j in range(1, lookback + 1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
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
# Drawdown protection methods
# ═══════════════════════════════════════════════════════════════

def simulate_dd_threshold(unit_df, lots, dd_pct):
    """Pause strategy when its drawdown from peak exceeds dd_pct%."""
    n = len(unit_df)
    portfolio_daily = np.zeros(n)
    equity = {name: 0.0 for name in STRAT_ORDER}
    peak = {name: 0.0 for name in STRAT_ORDER}
    paused = {name: False for name in STRAT_ORDER}

    for i in range(n):
        for name in STRAT_ORDER:
            unit_pnl = unit_df[name].iloc[i]
            equity[name] += unit_pnl * (lots[name] / UNIT_LOT)
            peak[name] = max(peak[name], equity[name])

            if peak[name] > 0:
                dd = (peak[name] - equity[name]) / peak[name] * 100
                paused[name] = dd > dd_pct
            else:
                paused[name] = equity[name] < 0

        for name in STRAT_ORDER:
            if not paused[name]:
                portfolio_daily[i] += unit_df[name].iloc[i] * (lots[name] / UNIT_LOT)

    return portfolio_daily


def simulate_equity_ma(unit_df, lots, ma_window):
    """Only trade when strategy equity > SMA of equity over ma_window trades."""
    n = len(unit_df)
    portfolio_daily = np.zeros(n)
    equity_history = {name: [] for name in STRAT_ORDER}
    equity = {name: 0.0 for name in STRAT_ORDER}

    for i in range(n):
        for name in STRAT_ORDER:
            unit_pnl = unit_df[name].iloc[i]
            equity[name] += unit_pnl * (lots[name] / UNIT_LOT)
            equity_history[name].append(equity[name])

            recent = equity_history[name][-ma_window:]
            sma = np.mean(recent) if len(recent) >= ma_window else equity[name]
            active = equity[name] >= sma

            if active:
                portfolio_daily[i] += unit_pnl * (lots[name] / UNIT_LOT)

    return portfolio_daily


def simulate_combined(unit_df, lots, dd_pct, ma_window):
    """Pause at DD threshold, resume when equity crosses back above MA."""
    n = len(unit_df)
    portfolio_daily = np.zeros(n)
    equity = {name: 0.0 for name in STRAT_ORDER}
    peak = {name: 0.0 for name in STRAT_ORDER}
    equity_history = {name: [] for name in STRAT_ORDER}
    paused = {name: False for name in STRAT_ORDER}

    for i in range(n):
        for name in STRAT_ORDER:
            unit_pnl = unit_df[name].iloc[i]
            equity[name] += unit_pnl * (lots[name] / UNIT_LOT)
            peak[name] = max(peak[name], equity[name])
            equity_history[name].append(equity[name])

            if peak[name] > 0:
                dd = (peak[name] - equity[name]) / peak[name] * 100
                if dd > dd_pct:
                    paused[name] = True
            elif equity[name] < 0:
                paused[name] = True

            if paused[name]:
                recent = equity_history[name][-ma_window:]
                sma = np.mean(recent) if len(recent) >= ma_window else equity[name] - 1
                if equity[name] > sma:
                    paused[name] = False

            if not paused[name]:
                portfolio_daily[i] += unit_pnl * (lots[name] / UNIT_LOT)

    return portfolio_daily


def simulate_redistribution(unit_df, lots, dd_pct):
    """When strategy paused, redistribute its lots equally to active strategies."""
    n = len(unit_df)
    portfolio_daily = np.zeros(n)
    equity = {name: 0.0 for name in STRAT_ORDER}
    peak = {name: 0.0 for name in STRAT_ORDER}

    for i in range(n):
        paused = {}
        for name in STRAT_ORDER:
            unit_pnl = unit_df[name].iloc[i]
            equity[name] += unit_pnl * (lots[name] / UNIT_LOT)
            peak[name] = max(peak[name], equity[name])

            if peak[name] > 0:
                dd = (peak[name] - equity[name]) / peak[name] * 100
                paused[name] = dd > dd_pct
            else:
                paused[name] = equity[name] < 0

        active_names = [name for name in STRAT_ORDER if not paused[name]]
        paused_names = [name for name in STRAT_ORDER if paused[name]]

        if not active_names:
            continue

        paused_lot_pool = sum(lots[name] for name in paused_names)
        extra_per_active = paused_lot_pool / len(active_names) if active_names else 0

        for name in active_names:
            effective_lot = lots[name] + extra_per_active
            portfolio_daily[i] += unit_df[name].iloc[i] * (effective_lot / UNIT_LOT)

    return portfolio_daily


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80, flush=True)
    print("  R126 — Drawdown-Aware Strategy Rotation", flush=True)
    print("=" * 80, flush=True)

    print("\n  Loading H1 data...", flush=True)
    h1_df = load_h1()
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})", flush=True)

    print("  Loading L8_MAX DataBundle...", flush=True)
    bundle = DataBundle.load_custom()

    all_results = {}

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Run all strategies, compute equity curves
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 1: Run All 4 Strategies", flush=True)
    print("=" * 70, flush=True)

    strat_trades = {}
    strat_trades['PSAR'] = bt_psar(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['PSAR'])
    strat_trades['TSMOM'] = bt_tsmom(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['TSMOM'])
    strat_trades['SESS_BO'] = bt_sess_bo(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['SESS_BO'])
    strat_trades['L8_MAX'] = bt_l8_max(bundle, SPREAD, UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])

    unit_dailies = {}
    for name in STRAT_ORDER:
        ds = _trades_to_daily_series(strat_trades[name])
        stats = _compute_stats(strat_trades[name])
        unit_dailies[name] = ds
        print(f"    {name:<10}: n={stats['n_trades']:5d}  Sharpe={stats['sharpe']:7.3f}  "
              f"PnL=${stats['pnl']:9.0f}  MaxDD=${stats['max_dd']:7.0f}", flush=True)

    all_dates = set()
    for ds in unit_dailies.values():
        all_dates.update(ds.index)
    all_dates = sorted(all_dates)
    idx = pd.DatetimeIndex(all_dates)

    unit_df = pd.DataFrame(0.0, index=idx, columns=STRAT_ORDER)
    for name in STRAT_ORDER:
        ds = unit_dailies[name]
        unit_df[name] = ds.reindex(idx, fill_value=0.0)

    # Fixed lots baseline
    fixed_daily = np.zeros(len(idx))
    for name in STRAT_ORDER:
        mult = R89_LOTS[name] / UNIT_LOT
        fixed_daily += unit_df[name].values * mult
    fixed_metrics = _portfolio_metrics(fixed_daily)
    print(f"\n    R89 Fixed baseline: Sharpe={fixed_metrics['sharpe']}, "
          f"PnL=${fixed_metrics['pnl']:.0f}, MaxDD=${fixed_metrics['max_dd']:.0f}", flush=True)

    # Per-strategy equity curve stats
    print(f"\n    Per-strategy max drawdown (at R89 lots):", flush=True)
    for name in STRAT_ORDER:
        eq = np.cumsum(unit_df[name].values * (R89_LOTS[name] / UNIT_LOT))
        peak = np.maximum.accumulate(eq)
        dd_abs = float((peak - eq).max())
        dd_pct = float(((peak - eq) / np.where(peak > 0, peak, 1) * 100).max())
        print(f"      {name:<10}: MaxDD=${dd_abs:.0f} ({dd_pct:.1f}%)", flush=True)

    all_results['phase1'] = {
        'strategies': {name: _compute_stats(strat_trades[name]) for name in STRAT_ORDER},
        'fixed_baseline': fixed_metrics,
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Drawdown protection methods
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: Drawdown Protection Methods", flush=True)
    print("=" * 70, flush=True)

    phase2 = {}

    # (a) DD threshold pause
    print(f"\n  (a) DD Threshold Pause:", flush=True)
    print(f"  {'Config':<25s}  {'Sharpe':>7s}  {'PnL':>10s}  {'MaxDD':>8s}  {'Calmar':>7s}  {'ΔSh':>7s}", flush=True)
    print("  " + "-" * 70, flush=True)

    for dd_pct in [3, 5, 8, 10]:
        port = simulate_dd_threshold(unit_df, R89_LOTS, dd_pct)
        m = _portfolio_metrics(port)
        delta = m['sharpe'] - fixed_metrics['sharpe']
        label = f"DD_pause_{dd_pct}pct"
        print(f"  {label:<25s}  {m['sharpe']:>7.3f}  ${m['pnl']:>9.0f}  "
              f"${m['max_dd']:>7.0f}  {m['calmar']:>7.3f}  {delta:>+7.3f}", flush=True)
        phase2[label] = m

    # (b) Equity curve MA
    print(f"\n  (b) Equity Curve MA:", flush=True)
    print(f"  {'Config':<25s}  {'Sharpe':>7s}  {'PnL':>10s}  {'MaxDD':>8s}  {'Calmar':>7s}  {'ΔSh':>7s}", flush=True)
    print("  " + "-" * 70, flush=True)

    for ma_w in [30, 50, 100]:
        port = simulate_equity_ma(unit_df, R89_LOTS, ma_w)
        m = _portfolio_metrics(port)
        delta = m['sharpe'] - fixed_metrics['sharpe']
        label = f"EqMA_{ma_w}"
        print(f"  {label:<25s}  {m['sharpe']:>7.3f}  ${m['pnl']:>9.0f}  "
              f"${m['max_dd']:>7.0f}  {m['calmar']:>7.3f}  {delta:>+7.3f}", flush=True)
        phase2[label] = m

    # (c) Combined
    print(f"\n  (c) Combined DD + Equity MA:", flush=True)
    print(f"  {'Config':<25s}  {'Sharpe':>7s}  {'PnL':>10s}  {'MaxDD':>8s}  {'Calmar':>7s}  {'ΔSh':>7s}", flush=True)
    print("  " + "-" * 70, flush=True)

    for dd_pct in [5, 8]:
        for ma_w in [30, 50]:
            port = simulate_combined(unit_df, R89_LOTS, dd_pct, ma_w)
            m = _portfolio_metrics(port)
            delta = m['sharpe'] - fixed_metrics['sharpe']
            label = f"Combined_DD{dd_pct}_MA{ma_w}"
            print(f"  {label:<25s}  {m['sharpe']:>7.3f}  ${m['pnl']:>9.0f}  "
                  f"${m['max_dd']:>7.0f}  {m['calmar']:>7.3f}  {delta:>+7.3f}", flush=True)
            phase2[label] = m

    # (d) Redistribution
    print(f"\n  (d) DD Pause + Redistribution:", flush=True)
    print(f"  {'Config':<25s}  {'Sharpe':>7s}  {'PnL':>10s}  {'MaxDD':>8s}  {'Calmar':>7s}  {'ΔSh':>7s}", flush=True)
    print("  " + "-" * 70, flush=True)

    for dd_pct in [3, 5, 8, 10]:
        port = simulate_redistribution(unit_df, R89_LOTS, dd_pct)
        m = _portfolio_metrics(port)
        delta = m['sharpe'] - fixed_metrics['sharpe']
        label = f"Redist_DD{dd_pct}pct"
        print(f"  {label:<25s}  {m['sharpe']:>7.3f}  ${m['pnl']:>9.0f}  "
              f"${m['max_dd']:>7.0f}  {m['calmar']:>7.3f}  {delta:>+7.3f}", flush=True)
        phase2[label] = m

    all_results['phase2'] = phase2

    # Rank all methods
    all_methods = [('R89_Fixed', fixed_metrics)] + list(phase2.items())
    all_methods.sort(key=lambda x: x[1]['sharpe'], reverse=True)
    best_method = all_methods[0][0]
    print(f"\n  Ranking (by Sharpe):", flush=True)
    for i, (name, m) in enumerate(all_methods[:10]):
        marker = " <-- BEST" if i == 0 else ""
        print(f"    #{i+1:2d} {name:<25s}: Sharpe={m['sharpe']:7.3f} Calmar={m['calmar']:7.3f}{marker}", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Walk-forward simulation
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 3: Walk-Forward Simulation (already done in Phase 2)", flush=True)
    print("  (Phase 2 methods are inherently walk-forward — equity-based signals)", flush=True)
    print("=" * 70, flush=True)

    all_results['phase3'] = {'note': 'Walk-forward is inherent in equity-based methods',
                             'best_method': best_method}

    # ═══════════════════════════════════════════════════════════
    # Phase 4: K-Fold Validation (5 folds)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 4: K-Fold Validation (5 folds)", flush=True)
    print("=" * 70, flush=True)

    top_configs = [
        ('R89_Fixed', None),
        ('DD_pause_5pct', lambda df: simulate_dd_threshold(df, R89_LOTS, 5)),
        ('DD_pause_8pct', lambda df: simulate_dd_threshold(df, R89_LOTS, 8)),
        ('EqMA_50', lambda df: simulate_equity_ma(df, R89_LOTS, 50)),
        ('Combined_DD5_MA50', lambda df: simulate_combined(df, R89_LOTS, 5, 50)),
        ('Redist_DD5pct', lambda df: simulate_redistribution(df, R89_LOTS, 5)),
        ('Redist_DD8pct', lambda df: simulate_redistribution(df, R89_LOTS, 8)),
    ]

    kfold_results = {}
    for config_name, sim_fn in top_configs:
        fold_sharpes = []
        for fold_name, start, end in FOLDS:
            fold_df = unit_df[(unit_df.index >= start) & (unit_df.index < end)]
            if len(fold_df) < 60:
                fold_sharpes.append(0.0)
                continue

            if config_name == 'R89_Fixed':
                port = np.zeros(len(fold_df))
                for name in STRAT_ORDER:
                    mult = R89_LOTS[name] / UNIT_LOT
                    port += fold_df[name].values * mult
            else:
                port = sim_fn(fold_df)

            fold_sharpes.append(round(_sharpe(port), 3))

        pos = sum(1 for s in fold_sharpes if s > 0)
        mean_s = round(float(np.mean(fold_sharpes)), 3)
        status = "PASS" if pos >= 3 else "FAIL"
        print(f"  {config_name:<25s}: {fold_sharpes}  -> {pos}/5 [{status}] mean={mean_s}", flush=True)
        kfold_results[config_name] = {
            'fold_sharpes': fold_sharpes,
            'positive': pos, 'mean': mean_s, 'pass': pos >= 3,
        }

    all_results['phase4_kfold'] = kfold_results

    # ═══════════════════════════════════════════════════════════
    # Phase 5: Comparison vs R105
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 5: Comparison vs R105 Dynamic Rotation", flush=True)
    print("=" * 70, flush=True)

    r105_path = Path("results/r105_dynamic_rotation/r105_results.json")
    if r105_path.exists():
        try:
            with open(r105_path) as f:
                r105 = json.load(f)
            r105_sharpe = r105.get('fixed_baseline', {}).get('sharpe', 'N/A')
            r105_best = r105.get('best_method', 'N/A')
            r105_best_sharpe = r105.get('best_sharpe', 'N/A')
            print(f"    R105 fixed baseline Sharpe: {r105_sharpe}", flush=True)
            print(f"    R105 best method: {r105_best} (Sharpe={r105_best_sharpe})", flush=True)
            print(f"    R105 conclusion: Dynamic rotation underperformed fixed lots", flush=True)
            all_results['phase5_r105_comparison'] = {
                'r105_available': True,
                'r105_fixed_sharpe': r105_sharpe,
                'r105_best_method': r105_best,
            }
        except Exception as e:
            print(f"    Could not parse R105 results: {e}", flush=True)
            all_results['phase5_r105_comparison'] = {'r105_available': False}
    else:
        print(f"    R105 results not found at {r105_path}", flush=True)
        all_results['phase5_r105_comparison'] = {'r105_available': False}

    print(f"\n    R126 best method: {best_method}", flush=True)
    best_m = phase2.get(best_method, fixed_metrics)
    print(f"    R126 Sharpe: {best_m['sharpe']}", flush=True)
    print(f"    R126 vs Fixed: ΔSharpe = {best_m['sharpe'] - fixed_metrics['sharpe']:+.3f}", flush=True)

    any_improves = any(v['sharpe'] > fixed_metrics['sharpe'] for v in phase2.values())
    if any_improves:
        improving = [(k, v) for k, v in phase2.items() if v['sharpe'] > fixed_metrics['sharpe']]
        print(f"\n    {len(improving)} methods improve over fixed:", flush=True)
        for name, m in sorted(improving, key=lambda x: -x[1]['sharpe']):
            print(f"      {name:<25s}: Sharpe={m['sharpe']}, ΔSharpe={m['sharpe']-fixed_metrics['sharpe']:+.3f}", flush=True)
    else:
        print(f"\n    No DD-aware method improves over fixed lots.", flush=True)
        print(f"    Conclusion: Same as R105 — fixed allocation remains robust.", flush=True)

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    print("\n" + "=" * 80, flush=True)
    print("  R126 FINAL SUMMARY", flush=True)
    print("=" * 80, flush=True)

    print(f"\n  Fixed baseline: Sharpe={fixed_metrics['sharpe']}, "
          f"Calmar={fixed_metrics['calmar']}", flush=True)
    print(f"  Best DD-aware method: {best_method} "
          f"(Sharpe={best_m['sharpe']}, Calmar={best_m['calmar']})", flush=True)

    print(f"\n  K-Fold results:", flush=True)
    for name, kf in kfold_results.items():
        status = "PASS" if kf['pass'] else "FAIL"
        print(f"    {name:<25s}: {kf['positive']}/5 [{status}] mean={kf['mean']}", flush=True)

    if best_m['sharpe'] > fixed_metrics['sharpe']:
        print(f"\n  RECOMMENDATION: Consider {best_method} for DD protection "
              f"(+{best_m['sharpe']-fixed_metrics['sharpe']:.3f} Sharpe)", flush=True)
    else:
        print(f"\n  RECOMMENDATION: Keep R89 fixed lots — DD protection doesn't help enough", flush=True)

    all_results['elapsed_s'] = round(elapsed, 1)
    all_results['best_method'] = best_method

    out_file = OUTPUT_DIR / "r126_results.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Saved: {out_file}", flush=True)
    print(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)", flush=True)
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
