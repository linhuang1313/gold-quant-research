#!/usr/bin/env python3
"""
R142 — Execution Shortfall Deep Analysis
==========================================
Analyzes execution quality by simulating realistic execution conditions:

  Phase 1: Load H1 data, run all 4 strategies at base lot → trade list
  Phase 2: Simulate execution imperfections (latency, slippage, missed/partial fills)
  Phase 3: Compare Sharpe/PnL/WR vs ideal for each imperfection model
  Phase 4: Session-based execution quality (per-session spread model)
  Phase 5: Realistic execution cost model
  Phase 6: Re-validate portfolio with realistic costs vs fixed $0.30
  Phase 7: Break-even analysis (spread at which Sharpe < 3.0)
  Phase 8: Report — execution budget per strategy, recommended windows
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
from backtest.runner import load_csv, load_m15, load_h1_aligned, H1_CSV_PATH

OUTPUT_DIR = Path("results/r142_execution_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.08, 'SESS_BO': 0.08}
CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']

PSAR_DEFAULT = {'sl_atr': 4.5, 'tp_atr': 16.0, 'trail_act': 0.20, 'trail_dist': 0.04, 'max_hold': 20}

SESSION_SPREADS = {
    'Asia':   0.40,   # UTC 0-8
    'London': 0.25,   # UTC 8-14
    'NY':     0.25,   # UTC 14-20
    'Late':   0.45,   # UTC 20-24
}

N_SEEDS = 200

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
            'pnl': pnl, 'reason': reason, 'bars': bar_i - pos['bar'],
            'atr': pos.get('atr', 0), 'bar_idx': pos['bar']}


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
        return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


def _sharpe(arr):
    if len(arr) < 10:
        return 0.0
    s = np.std(arr, ddof=1)
    return float(np.mean(arr) / s * np.sqrt(252)) if s > 0 else 0.0


def _max_dd(arr):
    if len(arr) == 0:
        return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def _compute_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0.0, 'pnl': 0.0, 'max_dd': 0.0, 'wr': 0.0}
    pnls = [t['pnl'] for t in trades]
    daily = _trades_to_daily(trades)
    daily_arr = daily.values if len(daily) > 0 else np.array([])
    wins = sum(1 for p in pnls if p > 0)
    return {
        'n': len(trades),
        'sharpe': round(_sharpe(daily_arr), 3),
        'pnl': round(sum(pnls), 2),
        'max_dd': round(_max_dd(daily_arr), 2),
        'wr': round(wins / len(trades) * 100, 1),
    }


def get_session(hour):
    if 0 <= hour < 8:
        return 'Asia'
    elif 8 <= hour < 14:
        return 'London'
    elif 14 <= hour < 20:
        return 'NY'
    else:
        return 'Late'


# ═══════════════════════════════════════════════════════════════
# Strategy backtests (returning enriched trade dicts)
# ═══════════════════════════════════════════════════════════════

def bt_psar(h1_df, spread, lot, maxloss_cap=0, params=None):
    if params is None:
        params = PSAR_DEFAULT
    df = h1_df.copy(); add_psar(df); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR', 'PSAR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; psar = df['PSAR'].values; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               params['sl_atr'], params['tp_atr'], params['trail_act'],
                               params['trail_dist'], params['max_hold'], maxloss_cap)
            if result:
                result['strategy'] = 'PSAR'
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if psar[i-1] > c[i-1] and psar[i] < c[i]:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif psar[i-1] < c[i-1] and psar[i] > c[i]:
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
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                result['strategy'] = 'TSMOM'
                trades.append(result); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and score[i] < 0:
                pnl = (c[i]-pos['entry']-spread)*lot*PV
                t = _mk(pos, c[i], times[i], "Reversal", i, pnl)
                t['strategy'] = 'TSMOM'
                trades.append(t); pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and score[i] > 0:
                pnl = (pos['entry']-c[i]-spread)*lot*PV
                t = _mk(pos, c[i], times[i], "Reversal", i, pnl)
                t['strategy'] = 'TSMOM'
                trades.append(t); pos = None; last_exit = i; continue
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
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                result['strategy'] = 'SESS_BO'
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


def bt_l8_max(data_bundle, spread, lot, maxloss_cap=35):
    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': maxloss_cap,
          'spread_cost': spread, 'initial_capital': 2000,
          'min_lot_size': lot, 'max_lot_size': lot}
    result = run_variant(data_bundle, "L8_MAX", verbose=False, **kw)
    trades = []
    for t in result.get('_trades', []):
        trades.append({
            'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
            'entry_time': t.entry_time, 'exit_time': t.exit_time,
            'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0,
            'atr': 0, 'bar_idx': 0, 'strategy': 'L8_MAX',
        })
    return trades


# ═══════════════════════════════════════════════════════════════
# Portfolio assembly
# ═══════════════════════════════════════════════════════════════

def build_portfolio_daily(strat_trades, lots):
    all_daily = {}
    for sn in STRAT_ORDER:
        if sn not in strat_trades:
            continue
        lot = lots[sn]; multiplier = lot / UNIT_LOT
        for t in strat_trades[sn]:
            d = pd.Timestamp(t['exit_time']).date()
            all_daily[d] = all_daily.get(d, 0) + t['pnl'] * multiplier
    dates = sorted(all_daily.keys())
    return np.array([all_daily[d] for d in dates]), dates


def portfolio_stats(strat_trades, lots):
    daily, dates = build_portfolio_daily(strat_trades, lots)
    return {
        'sharpe': round(_sharpe(daily), 3),
        'pnl': round(float(np.sum(daily)), 2),
        'max_dd': round(_max_dd(daily), 2),
        'n_days': len(daily),
    }


# ═══════════════════════════════════════════════════════════════
# Execution imperfection simulators
# ═══════════════════════════════════════════════════════════════

def apply_latency(trades, h1_df, rng):
    """Shift entry price by 1-3 bars forward (delayed execution)."""
    modified = []
    close_vals = h1_df['Close'].values
    times = h1_df.index
    n_bars = len(close_vals)
    for t in trades:
        t2 = dict(t)
        r = rng.rand()
        delay = 1 if r < 0.70 else (2 if r < 0.90 else 3)
        bar_i = t.get('bar_idx', 0)
        new_bar = min(bar_i + delay, n_bars - 1)
        if new_bar >= n_bars:
            modified.append(t2)
            continue
        old_entry = t2['entry']
        new_entry = close_vals[new_bar]
        if t2['dir'] == 'BUY':
            new_entry += SPREAD / 2
        else:
            new_entry -= SPREAD / 2
        price_diff = new_entry - old_entry
        if t2['dir'] == 'BUY':
            t2['pnl'] -= price_diff * UNIT_LOT * PV
        else:
            t2['pnl'] += price_diff * UNIT_LOT * PV
        t2['entry'] = new_entry
        modified.append(t2)
    return modified


def apply_slippage(trades, rng):
    """Random slippage: entry price worsened by 0-0.3 * ATR_ratio."""
    modified = []
    for t in trades:
        t2 = dict(t)
        atr = t.get('atr', 1.0)
        atr_ratio = max(atr / 10.0, 0.01) if atr > 0 else 0.01
        slip = rng.uniform(0, 0.3) * atr_ratio
        if t2['dir'] == 'BUY':
            t2['entry'] += slip
            t2['pnl'] -= slip * UNIT_LOT * PV
        else:
            t2['entry'] -= slip
            t2['pnl'] -= slip * UNIT_LOT * PV
        modified.append(t2)
    return modified


def apply_missed_fills(trades, rng, miss_rate=0.02):
    """Randomly drop a fraction of trades (missed fills)."""
    return [t for t in trades if rng.rand() >= miss_rate]


def apply_partial_fills(trades, rng, partial_rate=0.05, fill_fraction=0.50):
    """Reduce lot on a fraction of trades (partial fills)."""
    modified = []
    for t in trades:
        t2 = dict(t)
        if rng.rand() < partial_rate:
            t2['pnl'] *= fill_fraction
        modified.append(t2)
    return modified


def apply_session_spread(trades):
    """Recompute PnL using session-aware spread instead of fixed $0.30."""
    modified = []
    for t in trades:
        t2 = dict(t)
        entry_hour = pd.Timestamp(t['entry_time']).hour
        session = get_session(entry_hour)
        new_spread = SESSION_SPREADS[session]
        spread_delta = new_spread - SPREAD
        t2['pnl'] -= spread_delta * UNIT_LOT * PV
        t2['session'] = session
        t2['session_spread'] = new_spread
        modified.append(t2)
    return modified


# ═══════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════

def load_h1():
    candidates = [
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-27.csv"),
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv"),
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-03-25.csv"),
    ]
    h1_path = next((p for p in candidates if p.exists()), candidates[-1])
    return load_csv(str(h1_path))


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    results = {'experiment': 'R142 Execution Shortfall Deep Analysis'}

    print("=" * 80, flush=True)
    print("  R142 — Execution Shortfall Deep Analysis", flush=True)
    print("=" * 80, flush=True)

    # ─── Phase 1: Load data & run ideal baseline ──────────────
    print("\n" + "=" * 60, flush=True)
    print("  Phase 1: Load Data & Run Ideal Baseline", flush=True)
    print("=" * 60, flush=True)

    h1_df = load_h1()
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0].date()} ~ {h1_df.index[-1].date()})", flush=True)

    print("  Loading DataBundle for L8_MAX...", flush=True)
    try:
        bundle = DataBundle.load_custom()
        have_l8 = True
    except Exception as e:
        print(f"  WARN: DataBundle load failed: {e}", flush=True)
        bundle = None; have_l8 = False

    print("\n  Running ideal baseline (unit lot, $0.30 spread)...", flush=True)
    ideal_trades = {}
    ideal_trades['PSAR'] = bt_psar(h1_df, SPREAD, UNIT_LOT, CAPS['PSAR'])
    ideal_trades['TSMOM'] = bt_tsmom(h1_df, SPREAD, UNIT_LOT, CAPS['TSMOM'])
    ideal_trades['SESS_BO'] = bt_sess_bo(h1_df, SPREAD, UNIT_LOT, CAPS['SESS_BO'])
    ideal_trades['L8_MAX'] = bt_l8_max(bundle, SPREAD, UNIT_LOT, CAPS['L8_MAX']) if have_l8 else []

    all_ideal = []
    for sn in STRAT_ORDER:
        print(f"    {sn}: {len(ideal_trades[sn])} trades", flush=True)
        all_ideal.extend(ideal_trades[sn])

    ideal_stats = portfolio_stats(ideal_trades, R89_LOTS)
    print(f"\n  Ideal portfolio: Sharpe={ideal_stats['sharpe']:.3f}, "
          f"PnL=${ideal_stats['pnl']:,.2f}, MaxDD=${ideal_stats['max_dd']:,.2f}", flush=True)

    results['phase1'] = {
        'trades_per_strat': {s: len(ideal_trades[s]) for s in STRAT_ORDER},
        'total_trades': len(all_ideal),
        'ideal_stats': ideal_stats,
    }

    # ─── Phase 2+3: Execution imperfection simulations ────────
    print("\n" + "=" * 60, flush=True)
    print("  Phase 2-3: Execution Imperfection Models", flush=True)
    print("=" * 60, flush=True)

    imperfection_models = {
        'latency':      lambda tr, rng: apply_latency(tr, h1_df, rng),
        'slippage':     lambda tr, rng: apply_slippage(tr, rng),
        'missed_fills': lambda tr, rng: apply_missed_fills(tr, rng, 0.02),
        'partial_fills': lambda tr, rng: apply_partial_fills(tr, rng, 0.05, 0.50),
        'latency+slippage': lambda tr, rng: apply_slippage(apply_latency(tr, h1_df, rng), rng),
        'all_combined': lambda tr, rng: apply_partial_fills(
            apply_missed_fills(apply_slippage(apply_latency(tr, h1_df, rng), rng), rng, 0.02),
            rng, 0.05, 0.50),
    }

    print(f"\n  Running {N_SEEDS} Monte Carlo seeds per model...", flush=True)
    print(f"\n  {'Model':<22s} {'Sharpe':>7} {'dSharpe':>8} {'PnL':>12} {'dPnL%':>7} "
          f"{'WR':>6} {'dWR':>6}", flush=True)
    print(f"  {'─'*72}", flush=True)

    phase23 = {}
    ideal_wr = sum(1 for t in all_ideal if t['pnl'] > 0) / max(len(all_ideal), 1) * 100

    for model_name, model_fn in imperfection_models.items():
        mc_sharpes = []; mc_pnls = []; mc_wrs = []

        for seed in range(N_SEEDS):
            rng = np.random.RandomState(seed)
            mod_strat_trades = {}
            for sn in STRAT_ORDER:
                mod_strat_trades[sn] = model_fn(list(ideal_trades[sn]), rng)

            stats = portfolio_stats(mod_strat_trades, R89_LOTS)
            mc_sharpes.append(stats['sharpe'])
            mc_pnls.append(stats['pnl'])

            all_mod = []
            for sn in STRAT_ORDER:
                all_mod.extend(mod_strat_trades[sn])
            wr = sum(1 for t in all_mod if t['pnl'] > 0) / max(len(all_mod), 1) * 100
            mc_wrs.append(wr)

        mean_sh = np.mean(mc_sharpes)
        mean_pnl = np.mean(mc_pnls)
        mean_wr = np.mean(mc_wrs)
        d_sharpe = mean_sh - ideal_stats['sharpe']
        d_pnl_pct = (mean_pnl - ideal_stats['pnl']) / max(abs(ideal_stats['pnl']), 1) * 100
        d_wr = mean_wr - ideal_wr

        phase23[model_name] = {
            'sharpe_mean': round(mean_sh, 3),
            'sharpe_std': round(float(np.std(mc_sharpes)), 3),
            'sharpe_delta': round(d_sharpe, 3),
            'pnl_mean': round(mean_pnl, 2),
            'pnl_delta_pct': round(d_pnl_pct, 1),
            'wr_mean': round(mean_wr, 1),
            'wr_delta': round(d_wr, 1),
            'sharpe_p5': round(float(np.percentile(mc_sharpes, 5)), 3),
            'sharpe_p95': round(float(np.percentile(mc_sharpes, 95)), 3),
        }

        print(f"  {model_name:<22s} {mean_sh:>7.3f} {d_sharpe:>+8.3f} ${mean_pnl:>11,.0f} "
              f"{d_pnl_pct:>+6.1f}% {mean_wr:>5.1f}% {d_wr:>+5.1f}%", flush=True)

    results['phase23_imperfections'] = phase23

    # ─── Phase 4: Session-based execution quality ─────────────
    print("\n" + "=" * 60, flush=True)
    print("  Phase 4: Session-Based Execution Quality", flush=True)
    print("=" * 60, flush=True)

    session_trades = {s: [] for s in SESSION_SPREADS}
    for t in all_ideal:
        hour = pd.Timestamp(t['entry_time']).hour
        sess = get_session(hour)
        session_trades[sess].append(t)

    print(f"\n  {'Session':<10s} {'Hours':>10s} {'Trades':>7} {'Spread':>8} {'AvgPnL':>10} {'WR':>6}", flush=True)
    print(f"  {'─'*55}", flush=True)

    session_analysis = {}
    for sess, hours_label in [('Asia', '0-8'), ('London', '8-14'), ('NY', '14-20'), ('Late', '20-24')]:
        trades = session_trades[sess]
        n = len(trades)
        if n == 0:
            session_analysis[sess] = {'n': 0, 'spread': SESSION_SPREADS[sess]}
            print(f"  {sess:<10s} {hours_label:>10s}       0 ${SESSION_SPREADS[sess]:.2f}        N/A    N/A", flush=True)
            continue
        pnls = [t['pnl'] for t in trades]
        avg_pnl = np.mean(pnls)
        wr = sum(1 for p in pnls if p > 0) / n * 100
        session_analysis[sess] = {
            'n': n, 'spread': SESSION_SPREADS[sess],
            'avg_pnl': round(avg_pnl, 2), 'total_pnl': round(sum(pnls), 2),
            'wr': round(wr, 1),
        }
        print(f"  {sess:<10s} {hours_label:>10s} {n:>7d} ${SESSION_SPREADS[sess]:.2f} "
              f"${avg_pnl:>9.2f} {wr:>5.1f}%", flush=True)

    sess_mod_trades = {}
    for sn in STRAT_ORDER:
        sess_mod_trades[sn] = apply_session_spread(ideal_trades[sn])
    sess_stats = portfolio_stats(sess_mod_trades, R89_LOTS)
    print(f"\n  Session-spread portfolio: Sharpe={sess_stats['sharpe']:.3f}, "
          f"PnL=${sess_stats['pnl']:,.2f}", flush=True)
    print(f"  vs Ideal:                Sharpe={ideal_stats['sharpe']:.3f}, "
          f"PnL=${ideal_stats['pnl']:,.2f}", flush=True)

    session_analysis['portfolio_with_session_spread'] = sess_stats
    results['phase4_session'] = session_analysis

    # Per-strategy session breakdown
    print(f"\n  Per-strategy session distribution:", flush=True)
    strat_session_dist = {}
    for sn in STRAT_ORDER:
        dist = {}
        for t in ideal_trades[sn]:
            sess = get_session(pd.Timestamp(t['entry_time']).hour)
            dist[sess] = dist.get(sess, 0) + 1
        strat_session_dist[sn] = dist
        total = sum(dist.values())
        parts = ', '.join(f"{s}={dist.get(s,0)} ({dist.get(s,0)/max(total,1)*100:.0f}%)"
                          for s in SESSION_SPREADS)
        print(f"    {sn:<10s}: {parts}", flush=True)
    results['phase4_strat_sessions'] = strat_session_dist

    # ─── Phase 5: Realistic execution cost model ──────────────
    print("\n" + "=" * 60, flush=True)
    print("  Phase 5: Realistic Execution Cost Model", flush=True)
    print("=" * 60, flush=True)

    SLIPPAGE_RANGE = (0.05, 0.15)
    LATENCY_COST = 0.05

    print(f"\n  Cost components:", flush=True)
    print(f"    Base spread:    $0.30 (fixed assumption)", flush=True)
    print(f"    Slippage:       ${SLIPPAGE_RANGE[0]:.2f}-${SLIPPAGE_RANGE[1]:.2f}", flush=True)
    print(f"    Latency cost:   ${LATENCY_COST:.2f}", flush=True)
    print(f"    Session adj:    Asia +$0.10, London -$0.05, NY -$0.05, Late +$0.15", flush=True)

    session_adj = {'Asia': 0.10, 'London': -0.05, 'NY': -0.05, 'Late': 0.15}
    cost_model = {}

    print(f"\n  {'Session':<10s} {'Base':>6} {'SessAdj':>8} {'Slip':>6} {'Latency':>8} {'Total':>7}", flush=True)
    print(f"  {'─'*50}", flush=True)
    for sess in SESSION_SPREADS:
        base = 0.30
        adj = session_adj[sess]
        slip_mid = np.mean(SLIPPAGE_RANGE)
        total = base + adj + slip_mid + LATENCY_COST
        cost_model[sess] = {
            'base': base, 'session_adj': adj,
            'slippage': round(slip_mid, 2), 'latency': LATENCY_COST,
            'total_effective': round(total, 2),
        }
        print(f"  {sess:<10s} ${base:.2f}  {adj:>+7.2f} ${slip_mid:.2f}   ${LATENCY_COST:.2f}  "
              f"${total:.2f}", flush=True)

    results['phase5_cost_model'] = cost_model

    # ─── Phase 6: Re-validate with realistic costs ────────────
    print("\n" + "=" * 60, flush=True)
    print("  Phase 6: Portfolio with Realistic Costs vs Fixed $0.30", flush=True)
    print("=" * 60, flush=True)

    realistic_trades = {}
    for sn in STRAT_ORDER:
        mod = []
        for t in ideal_trades[sn]:
            t2 = dict(t)
            entry_hour = pd.Timestamp(t['entry_time']).hour
            sess = get_session(entry_hour)
            eff_spread = cost_model[sess]['total_effective']
            spread_delta = eff_spread - SPREAD
            t2['pnl'] -= spread_delta * UNIT_LOT * PV
            mod.append(t2)
        realistic_trades[sn] = mod

    realistic_stats = portfolio_stats(realistic_trades, R89_LOTS)

    spreads_to_test = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]
    spread_sweep = {}
    for test_spread in spreads_to_test:
        mod_trades = {}
        for sn in STRAT_ORDER:
            mod = []
            for t in ideal_trades[sn]:
                t2 = dict(t)
                spread_delta = test_spread - SPREAD
                t2['pnl'] -= spread_delta * UNIT_LOT * PV
                mod.append(t2)
            mod_trades[sn] = mod
        stats = portfolio_stats(mod_trades, R89_LOTS)
        spread_sweep[f'${test_spread:.2f}'] = stats

    print(f"\n  {'Scenario':<30s} {'Sharpe':>7} {'PnL':>12} {'MaxDD':>9}", flush=True)
    print(f"  {'─'*60}", flush=True)
    print(f"  {'Ideal ($0.30 fixed)':<30s} {ideal_stats['sharpe']:>7.3f} "
          f"${ideal_stats['pnl']:>11,.2f} ${ideal_stats['max_dd']:>8,.2f}", flush=True)
    print(f"  {'Realistic cost model':<30s} {realistic_stats['sharpe']:>7.3f} "
          f"${realistic_stats['pnl']:>11,.2f} ${realistic_stats['max_dd']:>8,.2f}", flush=True)

    print(f"\n  Uniform spread sweep:", flush=True)
    print(f"  {'Spread':<10s} {'Sharpe':>7} {'PnL':>12} {'MaxDD':>9}", flush=True)
    print(f"  {'─'*40}", flush=True)
    for sp_label, stats in spread_sweep.items():
        print(f"  {sp_label:<10s} {stats['sharpe']:>7.3f} ${stats['pnl']:>11,.2f} "
              f"${stats['max_dd']:>8,.2f}", flush=True)

    results['phase6'] = {
        'ideal': ideal_stats,
        'realistic': realistic_stats,
        'spread_sweep': spread_sweep,
    }

    # ─── Phase 7: Break-even analysis ─────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("  Phase 7: Break-Even Analysis (Sharpe < 3.0 threshold)", flush=True)
    print("=" * 60, flush=True)

    TARGET_SHARPE = 3.0
    fine_spreads = np.arange(0.20, 2.01, 0.05)
    breakeven_data = []

    for test_spread in fine_spreads:
        mod_trades = {}
        for sn in STRAT_ORDER:
            mod = []
            for t in ideal_trades[sn]:
                t2 = dict(t)
                spread_delta = test_spread - SPREAD
                t2['pnl'] -= spread_delta * UNIT_LOT * PV
                mod.append(t2)
            mod_trades[sn] = mod
        stats = portfolio_stats(mod_trades, R89_LOTS)
        breakeven_data.append({'spread': round(test_spread, 2), **stats})

    breakeven_spread = None
    for i, row in enumerate(breakeven_data):
        if row['sharpe'] < TARGET_SHARPE:
            if i > 0:
                prev = breakeven_data[i-1]
                frac = (prev['sharpe'] - TARGET_SHARPE) / max(prev['sharpe'] - row['sharpe'], 0.001)
                breakeven_spread = round(prev['spread'] + frac * 0.05, 3)
            else:
                breakeven_spread = row['spread']
            break

    if breakeven_spread is None:
        breakeven_spread = fine_spreads[-1]
        print(f"  Portfolio Sharpe stays above {TARGET_SHARPE} even at ${fine_spreads[-1]:.2f} spread", flush=True)
    else:
        print(f"  Break-even spread (Sharpe < {TARGET_SHARPE}): ${breakeven_spread:.3f}", flush=True)
        margin = breakeven_spread - SPREAD
        print(f"  Safety margin vs $0.30 base: ${margin:.3f} ({margin/SPREAD*100:.0f}%)", flush=True)

    strat_breakevens = {}
    for sn in STRAT_ORDER:
        if not ideal_trades[sn]:
            strat_breakevens[sn] = None
            continue
        for test_spread in fine_spreads:
            mod = []
            for t in ideal_trades[sn]:
                t2 = dict(t)
                spread_delta = test_spread - SPREAD
                lot = R89_LOTS[sn]; multiplier = lot / UNIT_LOT
                t2['pnl'] -= spread_delta * UNIT_LOT * PV
                mod.append(t2)
            daily = _trades_to_daily(mod)
            if len(daily) > 0:
                daily_arr = daily.values * (R89_LOTS[sn] / UNIT_LOT)
                sh = _sharpe(daily_arr)
            else:
                sh = 0
            if sh < TARGET_SHARPE:
                strat_breakevens[sn] = round(test_spread, 2)
                break
        else:
            strat_breakevens[sn] = round(fine_spreads[-1], 2)

    print(f"\n  Per-strategy break-even spreads (Sharpe < {TARGET_SHARPE}):", flush=True)
    for sn in STRAT_ORDER:
        be = strat_breakevens[sn]
        if be is not None:
            print(f"    {sn:<10s}: ${be:.2f}", flush=True)
        else:
            print(f"    {sn:<10s}: N/A (no trades)", flush=True)

    results['phase7_breakeven'] = {
        'target_sharpe': TARGET_SHARPE,
        'portfolio_breakeven_spread': breakeven_spread,
        'safety_margin': round(breakeven_spread - SPREAD, 3) if breakeven_spread else None,
        'per_strategy_breakeven': strat_breakevens,
        'spread_curve': breakeven_data,
    }

    # ─── Phase 8: Execution budget & recommended windows ──────
    print("\n" + "=" * 60, flush=True)
    print("  Phase 8: Execution Budget & Recommendations", flush=True)
    print("=" * 60, flush=True)

    budget = {}
    for sn in STRAT_ORDER:
        if not ideal_trades[sn]:
            budget[sn] = {'status': 'no trades'}
            continue
        sess_dist = strat_session_dist.get(sn, {})
        total_trades = sum(sess_dist.values())
        primary_session = max(sess_dist, key=sess_dist.get) if sess_dist else 'Unknown'
        primary_pct = sess_dist.get(primary_session, 0) / max(total_trades, 1) * 100

        best_sessions = sorted(sess_dist.items(), key=lambda x: -x[1])
        recommended = [s for s, _ in best_sessions
                       if cost_model.get(s, {}).get('total_effective', 99) <= 0.50]
        if not recommended:
            recommended = [best_sessions[0][0]] if best_sessions else ['London']

        pnls = [t['pnl'] for t in ideal_trades[sn]]
        avg_trade_pnl = np.mean(pnls)
        be_spread = strat_breakevens.get(sn, 0)

        budget[sn] = {
            'total_trades': total_trades,
            'primary_session': primary_session,
            'primary_session_pct': round(primary_pct, 1),
            'recommended_windows': recommended,
            'avg_trade_pnl': round(avg_trade_pnl, 2),
            'breakeven_spread': be_spread,
            'execution_budget': round(be_spread - SPREAD, 2) if be_spread else None,
        }

    print(f"\n  {'Strategy':<10s} {'Trades':>7} {'Primary':>10} {'Recom':>15s} "
          f"{'AvgPnL':>8} {'Budget':>8}", flush=True)
    print(f"  {'─'*65}", flush=True)
    for sn in STRAT_ORDER:
        b = budget[sn]
        if b.get('status') == 'no trades':
            print(f"  {sn:<10s}      N/A", flush=True)
            continue
        recom = ','.join(b['recommended_windows'])
        exec_budget = b.get('execution_budget')
        budget_str = f"${exec_budget:.2f}" if exec_budget else "N/A"
        print(f"  {sn:<10s} {b['total_trades']:>7d} {b['primary_session']:>10s} "
              f"{recom:>15s} ${b['avg_trade_pnl']:>7.2f} {budget_str:>8s}", flush=True)

    results['phase8_budget'] = budget

    # Best/worst execution hours
    hourly_pnl = {}
    for t in all_ideal:
        h = pd.Timestamp(t['entry_time']).hour
        hourly_pnl.setdefault(h, []).append(t['pnl'])

    hourly_stats = {}
    for h in sorted(hourly_pnl.keys()):
        pnls = hourly_pnl[h]
        hourly_stats[h] = {
            'n': len(pnls),
            'avg_pnl': round(np.mean(pnls), 2),
            'total_pnl': round(sum(pnls), 2),
            'wr': round(sum(1 for p in pnls if p > 0) / max(len(pnls), 1) * 100, 1),
        }

    print(f"\n  Hourly PnL distribution (top/bottom 5):", flush=True)
    sorted_hours = sorted(hourly_stats.items(), key=lambda x: -x[1]['total_pnl'])
    print(f"  Best:  ", end='', flush=True)
    for h, s in sorted_hours[:5]:
        print(f"  {h:02d}h=${s['total_pnl']:,.0f}({s['n']})", end='', flush=True)
    print(flush=True)
    print(f"  Worst: ", end='', flush=True)
    for h, s in sorted_hours[-5:]:
        print(f"  {h:02d}h=${s['total_pnl']:,.0f}({s['n']})", end='', flush=True)
    print(flush=True)

    results['phase8_hourly'] = hourly_stats

    # Final verdict
    realistic_sharpe = realistic_stats['sharpe']
    print(f"\n  ─── Final Verdict ───", flush=True)
    print(f"  Ideal Sharpe:       {ideal_stats['sharpe']:.3f}", flush=True)
    print(f"  Realistic Sharpe:   {realistic_sharpe:.3f}", flush=True)
    print(f"  Sharpe degradation: {ideal_stats['sharpe'] - realistic_sharpe:.3f}", flush=True)
    print(f"  Break-even spread:  ${breakeven_spread:.3f}", flush=True)
    if realistic_sharpe >= TARGET_SHARPE:
        print(f"  STATUS: PASS — realistic execution still meets Sharpe >= {TARGET_SHARPE}", flush=True)
    else:
        print(f"  STATUS: WARN — realistic execution Sharpe below {TARGET_SHARPE}", flush=True)

    results['verdict'] = {
        'ideal_sharpe': ideal_stats['sharpe'],
        'realistic_sharpe': realistic_stats['sharpe'],
        'degradation': round(ideal_stats['sharpe'] - realistic_sharpe, 3),
        'breakeven_spread': breakeven_spread,
        'passes_target': realistic_sharpe >= TARGET_SHARPE,
    }

    # ─── Save results ─────────────────────────────────────────
    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)
    print(f"\n  Total elapsed: {elapsed:.0f}s", flush=True)

    out_path = OUTPUT_DIR / "r142_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {out_path}", flush=True)


if __name__ == '__main__':
    main()
