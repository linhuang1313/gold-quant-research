#!/usr/bin/env python3
"""
R100 — Loss-Streak Circuit Breaker
====================================
Tests portfolio-level protection mechanisms that pause or reduce exposure
after consecutive losses.

Variants:
  Baseline: No pause (all trades taken)
  V1: After 3 consecutive losses, pause 1 hour
  V2: After 3 consecutive losses, pause 2 hours
  V3: After 5 consecutive losses, pause 4 hours
  V4: After 3 consecutive losses, reduce lot multiplier to 0.5x for next 3 trades
  V5: If cumulative daily PnL < -$100, stop for rest of that day

Steps:
  1. Run all 4 strategies, get trade lists with timestamps
  2. Merge into single chronological portfolio trade stream (sorted by exit_time)
  3. Scale PnL by R89_LOTS ratios
  4. Simulate each circuit breaker variant
  5. Compute stats: n_trades_taken, n_skipped, Sharpe, MaxDD, PnL, WR
  6. K-Fold validation on variants that beat baseline
  7. Save results/r100_circuit_breaker/r100_results.json
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r100_circuit_breaker")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.09, 'SESS_BO': 0.09}

CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]


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
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
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


def bt_l8_max(data_bundle, spread, lot, maxloss_cap=35):
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
# Stats helpers
# ═══════════════════════════════════════════════════════════════

def trades_to_daily_series(trades):
    if not trades:
        return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        ts = pd.Timestamp(t['exit_time'])
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        d = ts.normalize()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


def sharpe(daily_pnl):
    if len(daily_pnl) < 10:
        return 0.0
    arr = np.array(daily_pnl) if not isinstance(daily_pnl, np.ndarray) else daily_pnl
    if arr.std() == 0:
        return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(252))


def max_dd(daily_pnl):
    arr = np.array(daily_pnl) if not isinstance(daily_pnl, np.ndarray) else daily_pnl
    if len(arr) == 0:
        return 0.0
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(dd.max()) if len(dd) > 0 else 0.0


# ═══════════════════════════════════════════════════════════════
# Portfolio trade stream
# ═══════════════════════════════════════════════════════════════

def _normalize_ts(ts):
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    return t


def merge_portfolio_trades(strat_trades_dict):
    """Merge all strategy trades into one chronological stream, scaled by R89_LOTS."""
    all_trades = []
    for strat_name, trades in strat_trades_dict.items():
        lot_multiplier = R89_LOTS[strat_name] / UNIT_LOT
        for t in trades:
            all_trades.append({
                'strategy': strat_name,
                'dir': t['dir'],
                'entry': t['entry'],
                'exit': t['exit'],
                'entry_time': _normalize_ts(t['entry_time']),
                'exit_time': _normalize_ts(t['exit_time']),
                'pnl': t['pnl'] * lot_multiplier,
                'pnl_unit': t['pnl'],
                'reason': t['reason'],
            })
    all_trades.sort(key=lambda x: x['exit_time'])
    return all_trades


# ═══════════════════════════════════════════════════════════════
# Circuit breaker simulation
# ═══════════════════════════════════════════════════════════════

def simulate_circuit_breaker(portfolio_trades, variant):
    """
    Walk through trades chronologically and apply the circuit breaker rule.

    Returns (taken_trades, n_skipped).
    """
    taken = []
    skipped = 0
    consec_losses = 0
    pause_until = None
    reduced_lot_remaining = 0
    daily_pnl_accum = {}

    for trade in portfolio_trades:
        entry_t = trade['entry_time']
        exit_t = trade['exit_time']
        trade_date = entry_t.normalize()

        # --- Baseline: take all trades ---
        if variant == 'Baseline':
            taken.append(trade)
            continue

        # --- V5: daily PnL stop ---
        if variant == 'V5':
            day_key = trade_date
            cum_day_pnl = daily_pnl_accum.get(day_key, 0.0)
            if cum_day_pnl < -100:
                skipped += 1
                continue
            taken.append(trade)
            daily_pnl_accum[day_key] = cum_day_pnl + trade['pnl']
            continue

        # --- V1/V2/V3: time-based pause ---
        if variant in ('V1', 'V2', 'V3'):
            if variant == 'V1':
                streak_thresh, pause_hours = 3, 1
            elif variant == 'V2':
                streak_thresh, pause_hours = 3, 2
            else:
                streak_thresh, pause_hours = 5, 4

            if pause_until is not None and entry_t < pause_until:
                skipped += 1
                continue

            if pause_until is not None and entry_t >= pause_until:
                pause_until = None
                consec_losses = 0

            taken.append(trade)
            if trade['pnl'] < 0:
                consec_losses += 1
                if consec_losses >= streak_thresh:
                    pause_until = exit_t + timedelta(hours=pause_hours)
            else:
                consec_losses = 0
            continue

        # --- V4: reduced lot after streak ---
        if variant == 'V4':
            if reduced_lot_remaining > 0:
                adjusted_trade = dict(trade)
                adjusted_trade['pnl'] = trade['pnl'] * 0.5
                taken.append(adjusted_trade)
                reduced_lot_remaining -= 1
                if adjusted_trade['pnl'] < 0:
                    consec_losses += 1
                    if consec_losses >= 3:
                        reduced_lot_remaining = 3
                else:
                    consec_losses = 0
                continue

            taken.append(trade)
            if trade['pnl'] < 0:
                consec_losses += 1
                if consec_losses >= 3:
                    reduced_lot_remaining = 3
            else:
                consec_losses = 0
            continue

    return taken, skipped


def compute_variant_stats(taken_trades, n_skipped):
    n_taken = len(taken_trades)
    total_pnl = sum(t['pnl'] for t in taken_trades) if taken_trades else 0
    wins = sum(1 for t in taken_trades if t['pnl'] > 0)
    wr = wins / n_taken * 100 if n_taken > 0 else 0
    ds = trades_to_daily_series(taken_trades)
    sh = sharpe(ds)
    dd = max_dd(ds)
    return {
        'n_trades_taken': n_taken,
        'n_skipped': n_skipped,
        'pnl': round(total_pnl, 2),
        'sharpe': round(sh, 3),
        'max_dd': round(dd, 2),
        'wr': round(wr, 1),
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("  R100 — Loss-Streak Circuit Breaker", flush=True)
    print("=" * 80, flush=True)

    from backtest.runner import DataBundle, load_m15, load_h1_aligned, H1_CSV_PATH

    # ══════════════════════════════════════════════════════════════
    # Step 1: Run all 4 strategies
    # ══════════════════════════════════════════════════════════════
    print("\n  Loading data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    l8_bundle = DataBundle.load_custom()
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("  Step 1: Run all strategies at unit lot", flush=True)
    print("=" * 70, flush=True)

    strat_trades = {}

    print("    Running L8_MAX...", flush=True)
    strat_trades['L8_MAX'] = bt_l8_max(l8_bundle, spread=SPREAD, lot=UNIT_LOT,
                                        maxloss_cap=CAPS['L8_MAX'])
    print(f"      L8_MAX: {len(strat_trades['L8_MAX'])} trades", flush=True)

    print("    Running PSAR...", flush=True)
    strat_trades['PSAR'] = bt_psar(h1_df, spread=SPREAD, lot=UNIT_LOT,
                                    maxloss_cap=CAPS['PSAR'])
    print(f"      PSAR: {len(strat_trades['PSAR'])} trades", flush=True)

    print("    Running TSMOM...", flush=True)
    strat_trades['TSMOM'] = bt_tsmom(h1_df, spread=SPREAD, lot=UNIT_LOT,
                                      maxloss_cap=CAPS['TSMOM'])
    print(f"      TSMOM: {len(strat_trades['TSMOM'])} trades", flush=True)

    print("    Running SESS_BO...", flush=True)
    strat_trades['SESS_BO'] = bt_sess_bo(h1_df, spread=SPREAD, lot=UNIT_LOT,
                                          maxloss_cap=CAPS['SESS_BO'])
    print(f"      SESS_BO: {len(strat_trades['SESS_BO'])} trades", flush=True)

    total_raw = sum(len(v) for v in strat_trades.values())
    print(f"\n    Total raw trades: {total_raw}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Step 2-3: Merge and scale by R89_LOTS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Step 2-3: Merge portfolio stream & scale by R89_LOTS", flush=True)
    print("=" * 70, flush=True)

    portfolio_trades = merge_portfolio_trades(strat_trades)
    print(f"    Merged portfolio: {len(portfolio_trades)} trades", flush=True)
    print(f"    R89 lots: {R89_LOTS}", flush=True)
    print(f"    Date range: {portfolio_trades[0]['exit_time'].date()} ~ "
          f"{portfolio_trades[-1]['exit_time'].date()}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Step 4-5: Simulate circuit breaker variants
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Step 4-5: Circuit Breaker Simulation", flush=True)
    print("=" * 70, flush=True)

    variants = ['Baseline', 'V1', 'V2', 'V3', 'V4', 'V5']
    variant_desc = {
        'Baseline': 'No pause (all trades taken)',
        'V1': '3 consec losses -> pause 1h',
        'V2': '3 consec losses -> pause 2h',
        'V3': '5 consec losses -> pause 4h',
        'V4': '3 consec losses -> 0.5x lot for 3 trades',
        'V5': 'Daily PnL < -$100 -> stop rest of day',
    }

    results = {}
    for v in variants:
        taken, n_skip = simulate_circuit_breaker(portfolio_trades, v)
        stats = compute_variant_stats(taken, n_skip)
        stats['description'] = variant_desc[v]
        results[v] = stats
        print(f"\n    {v} ({variant_desc[v]}):", flush=True)
        print(f"      Taken={stats['n_trades_taken']}, Skipped={stats['n_skipped']}, "
              f"Sharpe={stats['sharpe']:.3f}, MaxDD=${stats['max_dd']:.1f}, "
              f"PnL=${stats['pnl']:.1f}, WR={stats['wr']:.1f}%", flush=True)

    # Summary table
    print("\n" + "-" * 90, flush=True)
    print(f"  {'Variant':<12} {'Desc':<40} {'Taken':>6} {'Skip':>5} {'Sharpe':>7} "
          f"{'MaxDD':>8} {'PnL':>10} {'WR':>5}", flush=True)
    print("-" * 90, flush=True)
    for v in variants:
        r = results[v]
        print(f"  {v:<12} {variant_desc[v]:<40} {r['n_trades_taken']:>6} {r['n_skipped']:>5} "
              f"{r['sharpe']:>7.3f} {r['max_dd']:>8.1f} {r['pnl']:>10.1f} {r['wr']:>5.1f}%",
              flush=True)
    print("-" * 90, flush=True)

    # ══════════════════════════════════════════════════════════════
    # Step 6: K-Fold Validation on variants that beat baseline
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Step 6: K-Fold Validation", flush=True)
    print("=" * 70, flush=True)

    baseline_sharpe = results['Baseline']['sharpe']
    candidates = [v for v in variants if v != 'Baseline' and results[v]['sharpe'] > baseline_sharpe]
    candidates.append('Baseline')
    print(f"\n    Candidates (beat baseline Sharpe={baseline_sharpe:.3f}): "
          f"{[v for v in candidates if v != 'Baseline']}", flush=True)

    kfold_results = {}
    for v in candidates:
        fold_sharpes = []
        print(f"\n    K-Fold: {v}", flush=True)

        for fold_name, start, end in FOLDS:
            fs = pd.Timestamp(start)
            fe = pd.Timestamp(end)

            fold_strat_trades = {}

            h1_fold = h1_df[(h1_df.index >= start) & (h1_df.index < end)]
            if len(h1_fold) < 100:
                fold_sharpes.append(0.0)
                continue

            fold_strat_trades['PSAR'] = bt_psar(h1_fold, SPREAD, UNIT_LOT, CAPS['PSAR'])
            fold_strat_trades['TSMOM'] = bt_tsmom(h1_fold, SPREAD, UNIT_LOT, CAPS['TSMOM'])
            fold_strat_trades['SESS_BO'] = bt_sess_bo(h1_fold, SPREAD, UNIT_LOT, CAPS['SESS_BO'])

            try:
                l8_fold = l8_bundle.slice(start, end)
                fold_strat_trades['L8_MAX'] = bt_l8_max(l8_fold, SPREAD, UNIT_LOT, CAPS['L8_MAX'])
            except Exception:
                fold_strat_trades['L8_MAX'] = [
                    t for t in strat_trades['L8_MAX']
                    if fs <= _normalize_ts(t['exit_time']) < fe
                ]

            fold_portfolio = merge_portfolio_trades(fold_strat_trades)
            if not fold_portfolio:
                fold_sharpes.append(0.0)
                continue

            taken, _ = simulate_circuit_breaker(fold_portfolio, v)
            ds = trades_to_daily_series(taken)
            fold_sharpes.append(sharpe(ds))

        positive = sum(1 for s in fold_sharpes if s > 0)
        kfold_results[v] = {
            'fold_sharpes': [round(s, 2) for s in fold_sharpes],
            'positive_folds': positive,
            'mean_sharpe': round(float(np.mean(fold_sharpes)), 3),
            'pass_4of6': positive >= 4,
        }
        status = "PASS" if positive >= 4 else "FAIL"
        print(f"      {positive}/6 positive, mean={np.mean(fold_sharpes):.3f}, "
              f"folds={[f'{s:.2f}' for s in fold_sharpes]} [{status}]", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Save results
    # ══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    best_variant = 'Baseline'
    best_sharpe = baseline_sharpe
    for v, kf in kfold_results.items():
        if v == 'Baseline':
            continue
        if kf['pass_4of6'] and results[v]['sharpe'] > best_sharpe:
            best_variant = v
            best_sharpe = results[v]['sharpe']

    if best_variant == 'Baseline':
        recommendation = "Keep Baseline (no circuit breaker) - no variant passes K-Fold with higher Sharpe"
    else:
        recommendation = (f"Adopt {best_variant} ({variant_desc[best_variant]}): "
                          f"Sharpe {results[best_variant]['sharpe']:.3f} vs "
                          f"Baseline {baseline_sharpe:.3f}, "
                          f"K-Fold {kfold_results[best_variant]['positive_folds']}/6 positive")

    print(f"\n{'='*80}", flush=True)
    print(f"  R100 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"  Recommendation: {recommendation}", flush=True)
    print(f"{'='*80}", flush=True)

    output = {
        'experiment': 'R100 Loss-Streak Circuit Breaker',
        'config': {
            'pv': PV, 'spread': SPREAD, 'unit_lot': UNIT_LOT,
            'r89_lots': R89_LOTS, 'caps': CAPS,
        },
        'variant_descriptions': variant_desc,
        'full_sample': results,
        'kfold': kfold_results,
        'recommendation': recommendation,
        'best_variant': best_variant,
        'elapsed_s': round(elapsed, 1),
    }

    with open(OUTPUT_DIR / "r100_results.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r100_results.json", flush=True)


if __name__ == "__main__":
    main()
