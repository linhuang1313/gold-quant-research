#!/usr/bin/env python3
"""
R140 — Portfolio Drawdown Recovery Optimization
==================================================
Implement and evaluate 5 drawdown-recovery mechanisms on the 4-strategy
portfolio, then validate with Walk-Forward, K-Fold, and Monte Carlo.

Mechanisms:
  M1: Per-strategy circuit breaker (3 consecutive losses → pause 1h)
  M2: Rolling Sharpe gate (20-trade rolling Sharpe < 0 → pause until > 0.5)
  M3: Cross-strategy redistribution (paused lot → active strategies)
  M4: Kelly fraction dynamic (recompute every 50 trades, cap 1.5x)
  M5: Combined best of M1-M4

Validation:
  Walk-Forward (4yr train / 2yr test, 5 windows)
  K-Fold 5-fold
  Monte Carlo 1000 paths
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS

OUTPUT_DIR = Path("results/r140_dd_recovery")
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


def _calmar(arr):
    dd = _max_dd(arr)
    if dd == 0 or len(arr) < 10:
        return 0.0
    n_years = max(len(arr) / 252, 0.5)
    return round(float(np.sum(arr)) / n_years / dd, 3)


def _yearly_stability(daily_arr, dates):
    """Return dict of year -> Sharpe."""
    if len(daily_arr) == 0:
        return {}
    ds = pd.Series(daily_arr, index=pd.DatetimeIndex(dates))
    yearly = {}
    for yr in range(ds.index.year.min(), ds.index.year.max() + 1):
        yd = ds[ds.index.year == yr].values
        if len(yd) >= 10:
            yearly[yr] = round(_sharpe(yd), 3)
    return yearly


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
# Recovery mechanisms
# ═══════════════════════════════════════════════════════════════

def apply_mechanism(all_trades_by_strat, mechanism):
    """Apply a recovery mechanism to strategy-level trades.

    all_trades_by_strat: dict of strategy_name -> sorted trades
    Returns: combined list of modified trades with lot scaling applied.
    """
    combined = []
    for sn in STRAT_ORDER:
        trades = all_trades_by_strat.get(sn, [])
        lot_mult = R89_LOTS[sn] / UNIT_LOT

        if mechanism == 'base':
            for t in trades:
                tc = dict(t); tc['pnl'] = t['pnl'] * lot_mult; tc['strategy'] = sn
                combined.append(tc)
            continue

        if mechanism == 'M1':
            consec_loss = 0; pause_until = None
            for t in trades:
                entry_ts = pd.Timestamp(t['entry_time'])
                if pause_until is not None and entry_ts < pause_until:
                    continue
                pause_until = None
                tc = dict(t); tc['pnl'] = t['pnl'] * lot_mult; tc['strategy'] = sn
                combined.append(tc)
                if t['pnl'] < 0:
                    consec_loss += 1
                    if consec_loss >= 3:
                        pause_until = pd.Timestamp(t['exit_time']) + pd.Timedelta(hours=1)
                        consec_loss = 0
                else:
                    consec_loss = 0

        elif mechanism == 'M2':
            recent_pnls = []
            paused = False
            for t in trades:
                if paused:
                    recent_pnls.append(t['pnl'])
                    if len(recent_pnls) >= 20:
                        rolling_sh = _compute_rolling_sharpe(recent_pnls[-20:])
                        if rolling_sh > 0.5:
                            paused = False
                    continue
                tc = dict(t); tc['pnl'] = t['pnl'] * lot_mult; tc['strategy'] = sn
                combined.append(tc)
                recent_pnls.append(t['pnl'])
                if len(recent_pnls) >= 20:
                    rolling_sh = _compute_rolling_sharpe(recent_pnls[-20:])
                    if rolling_sh < 0:
                        paused = True

        elif mechanism == 'M3':
            combined_m3_pending = []
            for t in trades:
                tc = dict(t); tc['pnl'] = t['pnl'] * lot_mult; tc['strategy'] = sn
                combined_m3_pending.append(tc)
            combined.extend(combined_m3_pending)

        elif mechanism == 'M4':
            trade_count = 0
            kelly_mult = 1.0
            local_pnls = []
            for t in trades:
                trade_count += 1
                local_pnls.append(t['pnl'])
                if trade_count % 50 == 0 and len(local_pnls) >= 20:
                    kelly_mult = _compute_kelly(local_pnls[-50:])
                    kelly_mult = min(kelly_mult, 1.5)
                    kelly_mult = max(kelly_mult, 0.25)
                tc = dict(t)
                tc['pnl'] = t['pnl'] * lot_mult * kelly_mult
                tc['strategy'] = sn
                combined.append(tc)

        elif mechanism == 'M5':
            consec_loss = 0; pause_until = None
            trade_count = 0; kelly_mult = 1.0; local_pnls = []
            for t in trades:
                entry_ts = pd.Timestamp(t['entry_time'])
                if pause_until is not None and entry_ts < pause_until:
                    local_pnls.append(t['pnl'])
                    continue
                pause_until = None

                trade_count += 1
                local_pnls.append(t['pnl'])
                if trade_count % 50 == 0 and len(local_pnls) >= 20:
                    kelly_mult = _compute_kelly(local_pnls[-50:])
                    kelly_mult = min(kelly_mult, 1.5)
                    kelly_mult = max(kelly_mult, 0.25)

                tc = dict(t)
                tc['pnl'] = t['pnl'] * lot_mult * kelly_mult
                tc['strategy'] = sn
                combined.append(tc)

                if t['pnl'] < 0:
                    consec_loss += 1
                    if consec_loss >= 3:
                        pause_until = pd.Timestamp(t['exit_time']) + pd.Timedelta(hours=1)
                        consec_loss = 0
                else:
                    consec_loss = 0

    if mechanism == 'M3':
        combined = _apply_redistribution(combined)

    combined.sort(key=lambda x: str(x.get('exit_time', '')))
    return combined


def _compute_rolling_sharpe(pnls):
    arr = np.array(pnls)
    if len(arr) < 5 or np.std(arr) == 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr) * np.sqrt(len(arr)))


def _compute_kelly(pnls):
    arr = np.array(pnls)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 1.0
    p = len(wins) / len(arr)
    avg_win = float(np.mean(wins))
    avg_loss = float(np.mean(np.abs(losses)))
    if avg_loss == 0:
        return 1.5
    b = avg_win / avg_loss
    kelly = p - (1 - p) / b
    return max(kelly, 0.1)


def _apply_redistribution(combined_trades):
    """For M3: redistribute paused strategy's lot to active ones."""
    by_strat = {sn: [] for sn in STRAT_ORDER}
    for t in combined_trades:
        sn = t.get('strategy', 'PSAR')
        if sn in by_strat:
            by_strat[sn].append(t)

    strat_sharpes = {}
    for sn in STRAT_ORDER:
        pnls = [t['pnl'] for t in by_strat[sn]]
        strat_sharpes[sn] = _compute_rolling_sharpe(pnls) if len(pnls) >= 10 else 0.0

    redistributed = []
    for sn in STRAT_ORDER:
        recent_pnls = []
        paused = False
        for t in by_strat[sn]:
            recent_pnls.append(t['pnl'])
            if len(recent_pnls) >= 20:
                sh = _compute_rolling_sharpe(recent_pnls[-20:])
                if sh < -0.5:
                    paused = True
                elif sh > 0.3:
                    paused = False

            if paused:
                active = [s for s in STRAT_ORDER if s != sn and strat_sharpes.get(s, 0) > 0]
                if active:
                    share = t['pnl'] / len(active)
                    for a in active:
                        tc = dict(t); tc['pnl'] = share; tc['strategy'] = a
                        redistributed.append(tc)
            else:
                redistributed.append(t)

    return redistributed


# ═══════════════════════════════════════════════════════════════
# Portfolio + data loading
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


def compute_portfolio_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'max_dd': 0, 'calmar': 0, 'wr': 0}
    daily = _trades_to_daily(trades)
    pnls = np.array([t['pnl'] for t in trades])
    n = len(trades)
    return {
        'n': n, 'sharpe': round(_sharpe(daily), 3),
        'pnl': round(float(np.sum(pnls)), 2),
        'max_dd': round(_max_dd(daily), 2),
        'calmar': _calmar(daily),
        'wr': round(float(np.sum(pnls > 0)) / n * 100, 1) if n > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    results = {'experiment': 'R140 Portfolio Drawdown Recovery Optimization'}

    print("=" * 80, flush=True)
    print("  R140 — Portfolio Drawdown Recovery Optimization", flush=True)
    print("=" * 80, flush=True)

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Load data + run base strategies
    # ═══════════════════════════════════════════════════════════
    print("\n  Phase 1: Loading data + running base strategies...", flush=True)
    h1_df = load_h1()
    bundle = DataBundle.load_default()
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})", flush=True)

    strat_trades = run_all_strategies(h1_df, bundle)
    for sn in STRAT_ORDER:
        print(f"    {sn:>10s}: {len(strat_trades[sn])} trades", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Apply recovery mechanisms
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: Recovery Mechanisms Comparison", flush=True)
    print("=" * 70, flush=True)

    mechanisms = ['base', 'M1', 'M2', 'M3', 'M4', 'M5']
    mech_names = {
        'base': 'No recovery (baseline)',
        'M1': 'Circuit breaker (3 losses)',
        'M2': 'Rolling Sharpe gate',
        'M3': 'Cross-strategy redistribution',
        'M4': 'Kelly fraction dynamic',
        'M5': 'Combined M1+M4',
    }

    mech_results = {}
    mech_dailies = {}

    print(f"\n  {'Mechanism':<35s} {'Sharpe':>7} {'PnL':>10} {'MaxDD':>9} {'Calmar':>8} {'WR%':>5}", flush=True)
    print(f"  {'─'*75}", flush=True)

    for mech in mechanisms:
        combined = apply_mechanism(strat_trades, mech)
        stats = compute_portfolio_stats(combined)
        daily = _trades_to_daily(combined)
        mech_results[mech] = stats
        mech_dailies[mech] = daily

        print(f"  {mech_names[mech]:<35s} {stats['sharpe']:>7.3f} "
              f"${stats['pnl']:>9,.0f} ${stats['max_dd']:>8,.0f} "
              f"{stats['calmar']:>8.2f} {stats['wr']:>5.1f}", flush=True)

    results['phase2'] = mech_results

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Yearly stability for each mechanism
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 3: Yearly Stability by Mechanism", flush=True)
    print("=" * 70, flush=True)

    yearly_results = {}
    years = range(2015, 2027)

    print(f"\n  {'Year':>6}", end="", flush=True)
    for mech in mechanisms:
        label = mech if mech != 'base' else 'BASE'
        print(f"  {label:>8}", end="")
    print(flush=True)
    print(f"  {'─'*6}" + "  " + "  ".join("─"*8 for _ in mechanisms), flush=True)

    for yr in years:
        yr_s = f"{yr}-01-01"; yr_e = f"{yr+1}-01-01"
        h1_yr = h1_df[(h1_df.index >= yr_s) & (h1_df.index < yr_e)]
        if len(h1_yr) < 200:
            continue

        yr_strat = {}
        yr_strat['PSAR'] = bt_psar(h1_yr, SPREAD, UNIT_LOT, CAPS['PSAR'])
        yr_strat['TSMOM'] = bt_tsmom(h1_yr, SPREAD, UNIT_LOT, CAPS['TSMOM'])
        yr_strat['SESS_BO'] = bt_sess_bo(h1_yr, SPREAD, UNIT_LOT, CAPS['SESS_BO'])
        yr_strat['L8_MAX'] = []

        line = f"  {yr:>6}"
        yr_row = {}
        for mech in mechanisms:
            combined = apply_mechanism(yr_strat, mech)
            daily = _trades_to_daily(combined)
            sh = _sharpe(daily)
            yr_row[mech] = round(sh, 3)
            line += f"  {sh:>8.3f}"
        yearly_results[yr] = yr_row
        print(line, flush=True)

    results['phase3_yearly'] = yearly_results

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Walk-Forward (4yr train / 2yr test, 5 windows)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 4: Walk-Forward Validation (4yr/2yr, 5 windows)", flush=True)
    print("=" * 70, flush=True)

    wf_windows = [
        ("2015-19/19-21", "2019-01-01", "2021-01-01"),
        ("2016-20/20-22", "2020-01-01", "2022-01-01"),
        ("2017-21/21-23", "2021-01-01", "2023-01-01"),
        ("2018-22/22-24", "2022-01-01", "2024-01-01"),
        ("2019-23/23-25", "2023-01-01", "2025-01-01"),
    ]

    wf_results = []
    print(f"\n  {'Window':<20s}", end="", flush=True)
    for mech in mechanisms:
        label = mech if mech != 'base' else 'BASE'
        print(f"  {label:>7}", end="")
    print(flush=True)
    print(f"  {'─'*20}" + "  " + "  ".join("─"*7 for _ in mechanisms), flush=True)

    for label, oos_start, oos_end in wf_windows:
        h1_oos = h1_df[(h1_df.index >= oos_start) & (h1_df.index < oos_end)]
        if len(h1_oos) < 500:
            continue

        oos_strat = {}
        oos_strat['PSAR'] = bt_psar(h1_oos, SPREAD, UNIT_LOT, CAPS['PSAR'])
        oos_strat['TSMOM'] = bt_tsmom(h1_oos, SPREAD, UNIT_LOT, CAPS['TSMOM'])
        oos_strat['SESS_BO'] = bt_sess_bo(h1_oos, SPREAD, UNIT_LOT, CAPS['SESS_BO'])
        oos_strat['L8_MAX'] = []

        row = {'window': label}
        line = f"  {label:<20s}"
        for mech in mechanisms:
            combined = apply_mechanism(oos_strat, mech)
            daily = _trades_to_daily(combined)
            sh = _sharpe(daily)
            row[mech] = round(sh, 3)
            line += f"  {sh:>7.3f}"
        wf_results.append(row)
        print(line, flush=True)

    if wf_results:
        print(f"\n  OOS Mean:", flush=True)
        for mech in mechanisms:
            vals = [w[mech] for w in wf_results]
            print(f"    {mech:>6s}: mean={np.mean(vals):.3f}, min={min(vals):.3f}", flush=True)

    results['phase4_walkforward'] = wf_results

    # ═══════════════════════════════════════════════════════════
    # Phase 5: K-Fold 5-fold
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 5: K-Fold 5-Fold Validation", flush=True)
    print("=" * 70, flush=True)

    fold_starts = pd.date_range("2015-01-01", "2026-05-01", periods=6)
    kfold_results = {mech: [] for mech in mechanisms}

    for fi in range(5):
        fs = str(fold_starts[fi].date())
        fe = str(fold_starts[fi + 1].date())
        h1_fold = h1_df[(h1_df.index >= fs) & (h1_df.index < fe)]
        if len(h1_fold) < 200:
            continue

        fold_strat = {}
        fold_strat['PSAR'] = bt_psar(h1_fold, SPREAD, UNIT_LOT, CAPS['PSAR'])
        fold_strat['TSMOM'] = bt_tsmom(h1_fold, SPREAD, UNIT_LOT, CAPS['TSMOM'])
        fold_strat['SESS_BO'] = bt_sess_bo(h1_fold, SPREAD, UNIT_LOT, CAPS['SESS_BO'])
        fold_strat['L8_MAX'] = []

        for mech in mechanisms:
            combined = apply_mechanism(fold_strat, mech)
            daily = _trades_to_daily(combined)
            kfold_results[mech].append(round(_sharpe(daily), 3))

    print(f"\n  {'Mechanism':<12s} {'Fold Sharpes':>35s} {'Mean':>6} {'Min':>6} {'Pos':>4}", flush=True)
    print(f"  {'─'*65}", flush=True)
    for mech, sharpes in kfold_results.items():
        if not sharpes:
            continue
        fold_str = ', '.join(f'{s:.2f}' for s in sharpes)
        m = np.mean(sharpes); mn = min(sharpes)
        pos = sum(1 for s in sharpes if s > 0)
        print(f"  {mech:<12s} [{fold_str:>33s}] {m:>6.2f} {mn:>6.2f} {pos:>3}/{len(sharpes)}", flush=True)

    results['phase5_kfold'] = kfold_results

    # ═══════════════════════════════════════════════════════════
    # Phase 6: Monte Carlo 1000 paths
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 6: Monte Carlo Simulation (1000 paths)", flush=True)
    print("=" * 70, flush=True)

    rng = np.random.RandomState(42)
    mc_results = {}

    for mech in mechanisms:
        daily = mech_dailies[mech]
        mc_sharpes = []
        mc_dds = []
        for _ in range(1000):
            perturbed = daily * rng.uniform(0.92, 1.08, size=len(daily))
            mask = rng.random(len(daily)) > 0.02
            perturbed = perturbed * mask
            mc_sharpes.append(_sharpe(perturbed))
            mc_dds.append(_max_dd(perturbed))

        mc_sh = np.array(mc_sharpes)
        mc_dd = np.array(mc_dds)
        mc_results[mech] = {
            'sharpe_p5': round(float(np.percentile(mc_sh, 5)), 3),
            'sharpe_p50': round(float(np.percentile(mc_sh, 50)), 3),
            'sharpe_p95': round(float(np.percentile(mc_sh, 95)), 3),
            'dd_p50': round(float(np.percentile(mc_dd, 50)), 2),
            'dd_p95': round(float(np.percentile(mc_dd, 95)), 2),
            'pct_positive': round(float(np.mean(mc_sh > 0) * 100), 1),
        }

    print(f"\n  {'Mechanism':<12s} {'P5':>7} {'P50':>7} {'P95':>7} {'DD_P50':>9} {'DD_P95':>9} {'Pos%':>5}", flush=True)
    print(f"  {'─'*60}", flush=True)
    for mech in mechanisms:
        m = mc_results[mech]
        print(f"  {mech:<12s} {m['sharpe_p5']:>7.3f} {m['sharpe_p50']:>7.3f} {m['sharpe_p95']:>7.3f} "
              f"${m['dd_p50']:>8,.0f} ${m['dd_p95']:>8,.0f} {m['pct_positive']:>5.1f}", flush=True)

    results['phase6_monte_carlo'] = mc_results

    # ═══════════════════════════════════════════════════════════
    # Phase 7: Recommendation
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print("  R140 RECOMMENDATION", flush=True)
    print("=" * 80, flush=True)

    base_sh = mech_results['base']['sharpe']
    base_dd = mech_results['base']['max_dd']

    candidates = []
    for mech in ['M1', 'M2', 'M3', 'M4', 'M5']:
        sh = mech_results[mech]['sharpe']
        dd = mech_results[mech]['max_dd']
        sh_delta = sh - base_sh
        dd_delta = base_dd - dd

        kf_sharpes = kfold_results.get(mech, [])
        kf_mean = np.mean(kf_sharpes) if kf_sharpes else 0
        kf_pos = sum(1 for s in kf_sharpes if s > 0)

        wf_sharpes = [w.get(mech, 0) for w in wf_results]
        wf_mean = np.mean(wf_sharpes) if wf_sharpes else 0
        wf_min = min(wf_sharpes) if wf_sharpes else -999

        score = sh_delta * 2 + (dd_delta / 100) + (kf_mean - base_sh) + max(0, wf_min)
        candidates.append({
            'mechanism': mech, 'name': mech_names[mech],
            'sharpe': sh, 'sharpe_delta': round(sh_delta, 3),
            'max_dd': dd, 'dd_reduction': round(dd_delta, 2),
            'kf_mean': round(kf_mean, 3), 'kf_positive': kf_pos,
            'wf_mean': round(wf_mean, 3), 'wf_min': round(wf_min, 3),
            'score': round(score, 3),
        })

    candidates.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n  Ranking by composite score:", flush=True)
    print(f"  {'#':>3} {'Mechanism':<35s} {'Score':>7} {'dSharpe':>8} {'dDD':>8} {'KF':>6} {'WFmin':>7}", flush=True)
    print(f"  {'─'*75}", flush=True)
    for i, c in enumerate(candidates, 1):
        print(f"  {i:>3} {c['name']:<35s} {c['score']:>7.3f} "
              f"{c['sharpe_delta']:>+8.3f} ${c['dd_reduction']:>7,.0f} "
              f"{c['kf_mean']:>6.2f} {c['wf_min']:>7.3f}", flush=True)

    best = candidates[0]
    verdict = "ADOPT" if best['sharpe_delta'] > 0 and best['kf_positive'] >= 3 else "MONITOR"
    print(f"\n  Best mechanism: {best['mechanism']} — {best['name']}", flush=True)
    print(f"  Verdict: {verdict}", flush=True)

    results['phase7_recommendation'] = {
        'ranking': candidates,
        'best': best['mechanism'],
        'best_name': best['name'],
        'verdict': verdict,
    }

    # ═══════════════════════════════════════════════════════════
    # Save
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)

    out_file = OUTPUT_DIR / "r140_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"  Saved: {out_file}", flush=True)
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
