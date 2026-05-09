#!/usr/bin/env python3
"""
R106 — Full Production Simulation
====================================
Monte Carlo simulation of real-world trading conditions.
1000 runs with execution noise, capital growth, and disconnections.

  Phase 1: Execution model (slippage, spread variation, delay)
  Phase 2: Signal conflicts + position limits
  Phase 3: Capital growth with compounding
  Phase 4: Disconnection model (missed signals, early exits)
  Phase 5: Statistics — P5/P25/P50/P75/P95 distribution
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import DataBundle, load_m15, load_h1_aligned, H1_CSV_PATH
from backtest.runner import run_variant, LIVE_PARITY_KWARGS

OUTPUT_DIR = Path("results/r106_production_sim")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01
CAPITAL = 5000
MAX_POSITIONS = 4

CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.09, 'SESS_BO': 0.09}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']

N_SIMULATIONS = 1000

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


# ═══════════════════════════════════════════════════════════════
# Monte Carlo helpers
# ═══════════════════════════════════════════════════════════════

def _enrich_trades(all_trades, h1_df):
    """Add strategy name, lot, and ATR at entry to each trade for MC processing."""
    atr_series = compute_atr(h1_df)
    for t in all_trades:
        entry_ts = pd.Timestamp(t['entry_time'])
        idx = atr_series.index.searchsorted(entry_ts)
        idx = min(max(idx - 1, 0), len(atr_series) - 1)
        t['atr_at_entry'] = float(atr_series.iloc[idx]) if not np.isnan(atr_series.iloc[idx]) else 5.0


def _is_weekend_held(entry_time, exit_time):
    """Check if a trade spans a weekend (Friday -> Monday)."""
    et = pd.Timestamp(entry_time)
    xt = pd.Timestamp(exit_time)
    entry_dow = et.dayofweek
    exit_dow = xt.dayofweek
    if exit_dow < entry_dow and (xt - et).days >= 2:
        return True
    if (xt - et).days >= 3:
        return True
    return False


def _compute_monthly_equity(trades):
    """Group trades by month and return cumulative equity at each month-end."""
    if not trades:
        return {}
    monthly = {}
    for t in trades:
        month_key = pd.Timestamp(t['exit_time']).to_period('M')
        monthly.setdefault(month_key, 0.0)
        monthly[month_key] += t['pnl']
    return monthly


def _worst_month(trades):
    """Return the worst single-month PnL."""
    monthly = _compute_monthly_equity(trades)
    if not monthly:
        return 0.0
    return min(monthly.values())


def run_single_mc(base_trades, run_idx, params):
    """
    Run one Monte Carlo simulation.

    params dict keys:
      base_spread, spread_mean_log, spread_spike_prob, spread_spike_val,
      max_slippage, signal_drop_rate, early_exit_rate, early_exit_pnl_frac,
      max_positions, capital, r89_lots, pv, weekend_gap_atr_frac
    """
    rng = np.random.RandomState(run_idx)

    base_spread = params['base_spread']
    spread_mean = params.get('spread_mean_log', 0.35)
    spike_prob = params.get('spread_spike_prob', 0.05)
    spike_val = params.get('spread_spike_val', 0.80)
    max_slip = params.get('max_slippage', 0.05)
    drop_rate = params.get('signal_drop_rate', 0.02)
    early_exit_rate = params.get('early_exit_rate', 0.01)
    early_exit_frac = params.get('early_exit_pnl_frac', 0.50)
    max_pos = params.get('max_positions', MAX_POSITIONS)
    capital = params.get('capital', CAPITAL)
    r89_lots = params.get('r89_lots', R89_LOTS)
    pv = params.get('pv', PV)
    gap_frac = params.get('weekend_gap_atr_frac', 0.5)

    equity = capital
    sim_trades = []
    open_positions = []
    n_dropped = 0
    n_early_exit = 0
    n_pos_limited = 0

    month_equity = {}
    current_month = None

    for t in base_trades:
        strat = t['strategy']
        base_lot = r89_lots[strat]

        # --- Signal dropping ---
        if rng.rand() < drop_rate:
            n_dropped += 1
            continue

        # --- Position limit ---
        entry_ts = pd.Timestamp(t['entry_time'])
        exit_ts = pd.Timestamp(t['exit_time'])

        still_open = [p for p in open_positions if pd.Timestamp(p['exit_time']) > entry_ts]
        open_positions = still_open
        if len(open_positions) >= max_pos:
            n_pos_limited += 1
            continue

        # --- Capital compounding: lot scaling ---
        lot_scale = min(equity / capital, 3.0)
        lot = base_lot * lot_scale

        # --- Spread variation ---
        if rng.rand() < spike_prob:
            actual_spread = spike_val
        else:
            log_mean = np.log(spread_mean) - 0.5 * 0.3**2
            actual_spread = rng.lognormal(mean=log_mean, sigma=0.3)
            actual_spread = max(actual_spread, 0.10)
            actual_spread = min(actual_spread, 1.50)

        spread_cost_delta = (actual_spread - base_spread) * lot * pv

        # --- Slippage ---
        slippage = rng.uniform(0, max_slip)
        slippage_cost = slippage * lot * pv

        # --- Base PnL (scale from unit lot to actual lot) ---
        unit_pnl = t['pnl']
        lot_multiplier = lot / UNIT_LOT
        raw_pnl = unit_pnl * lot_multiplier

        # --- Early exit ---
        if rng.rand() < early_exit_rate:
            raw_pnl = raw_pnl * early_exit_frac
            n_early_exit += 1

        # --- Weekend gap noise ---
        if _is_weekend_held(t['entry_time'], t['exit_time']):
            atr_val = t.get('atr_at_entry', 5.0)
            gap_noise = rng.uniform(-gap_frac, gap_frac) * atr_val * lot * pv
            raw_pnl += gap_noise

        # --- Final PnL after execution costs ---
        final_pnl = raw_pnl - spread_cost_delta - slippage_cost

        equity += final_pnl

        trade_month = exit_ts.to_period('M')
        if trade_month not in month_equity:
            month_equity[trade_month] = 0.0
        month_equity[trade_month] += final_pnl

        sim_trades.append({
            'strategy': strat,
            'pnl': final_pnl,
            'entry_time': t['entry_time'],
            'exit_time': t['exit_time'],
        })
        open_positions.append(t)

    # --- Compute run metrics ---
    daily = trades_to_daily_series(sim_trades)
    daily_arr = daily.values if len(daily) > 0 else np.array([])

    total_pnl = equity - capital
    n_years = max(len(daily_arr) / 252, 0.5)
    annual_return = total_pnl / n_years
    worst_m = min(month_equity.values()) if month_equity else 0.0

    return {
        'final_equity': round(equity, 2),
        'total_pnl': round(total_pnl, 2),
        'sharpe': round(sharpe(daily_arr), 3),
        'max_dd': round(max_dd(daily_arr), 2),
        'n_trades': len(sim_trades),
        'annual_return': round(annual_return, 2),
        'worst_month': round(worst_m, 2),
        'n_dropped': n_dropped,
        'n_early_exit': n_early_exit,
        'n_pos_limited': n_pos_limited,
    }


def compute_percentiles(mc_results, key):
    vals = [r[key] for r in mc_results]
    return {
        'P5': round(float(np.percentile(vals, 5)), 2),
        'P25': round(float(np.percentile(vals, 25)), 2),
        'P50': round(float(np.percentile(vals, 50)), 2),
        'P75': round(float(np.percentile(vals, 75)), 2),
        'P95': round(float(np.percentile(vals, 95)), 2),
        'mean': round(float(np.mean(vals)), 2),
        'std': round(float(np.std(vals)), 2),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  R106 — Full Production Simulation")
    print("  Monte Carlo: 1000 runs with execution noise, compounding, disconnections")
    print("=" * 80)

    # ─── Load data ────────────────────────────────────────────
    print("\n  Loading data...")
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    bundle = DataBundle.load_custom()
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0].date()} ~ {h1_df.index[-1].date()})")

    # ═══════════════════════════════════════════════════════════
    # Step 1: Base Trade Generation (deterministic)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Step 1: Base Trade Generation (unit lot)")
    print("=" * 60)

    base_trades = {}
    base_trades['PSAR'] = bt_psar(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['PSAR'])
    base_trades['TSMOM'] = bt_tsmom(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['TSMOM'])
    base_trades['SESS_BO'] = bt_sess_bo(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['SESS_BO'])
    base_trades['L8_MAX'] = bt_l8_max(bundle, SPREAD, UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])

    all_trades = []
    for strat in STRAT_ORDER:
        for t in base_trades[strat]:
            tc = dict(t)
            tc['strategy'] = strat
            all_trades.append(tc)

    all_trades.sort(key=lambda x: pd.Timestamp(x['entry_time']))
    _enrich_trades(all_trades, h1_df)

    total_base_trades = len(all_trades)
    for strat in STRAT_ORDER:
        n_t = len(base_trades[strat])
        pnl = sum(t['pnl'] for t in base_trades[strat])
        print(f"    {strat:10s}: {n_t:5d} trades, unit PnL=${pnl:,.2f}")
    print(f"    {'TOTAL':10s}: {total_base_trades:5d} trades (merged timeline)")

    # Deterministic baseline (no noise)
    unit_dailies = {name: trades_to_daily_series(base_trades[name]) for name in STRAT_ORDER}
    fixed_daily = build_portfolio_daily(unit_dailies, R89_LOTS)
    base_sharpe = sharpe(fixed_daily)
    base_pnl = float(np.sum(fixed_daily))
    base_mdd = max_dd(fixed_daily)
    print(f"\n  Deterministic Baseline: Sharpe={base_sharpe:.3f}, "
          f"PnL=${base_pnl:,.2f}, MaxDD=${base_mdd:,.2f}")

    results = {
        'experiment': 'R106 Full Production Simulation',
        'n_base_trades': total_base_trades,
        'deterministic_baseline': {
            'sharpe': round(base_sharpe, 3),
            'pnl': round(base_pnl, 2),
            'max_dd': round(base_mdd, 2),
        }
    }

    # ═══════════════════════════════════════════════════════════
    # Step 2: Monte Carlo Simulation (1000 runs)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"  Step 2: Monte Carlo Simulation ({N_SIMULATIONS} runs)")
    print("=" * 60)

    mc_params = {
        'base_spread': SPREAD,
        'spread_mean_log': 0.35,
        'spread_spike_prob': 0.05,
        'spread_spike_val': 0.80,
        'max_slippage': 0.05,
        'signal_drop_rate': 0.02,
        'early_exit_rate': 0.01,
        'early_exit_pnl_frac': 0.50,
        'max_positions': MAX_POSITIONS,
        'capital': CAPITAL,
        'r89_lots': R89_LOTS,
        'pv': PV,
        'weekend_gap_atr_frac': 0.5,
    }

    mc_results = []
    t_mc_start = time.time()
    for run_i in range(N_SIMULATIONS):
        res = run_single_mc(all_trades, run_i, mc_params)
        mc_results.append(res)
        if (run_i + 1) % 100 == 0:
            elapsed_mc = time.time() - t_mc_start
            avg_time = elapsed_mc / (run_i + 1)
            eta = avg_time * (N_SIMULATIONS - run_i - 1)
            med_pnl = np.median([r['total_pnl'] for r in mc_results])
            print(f"    Run {run_i+1:5d}/{N_SIMULATIONS}: "
                  f"median PnL=${med_pnl:,.0f}, "
                  f"elapsed={elapsed_mc:.0f}s, ETA={eta:.0f}s")

    mc_elapsed = time.time() - t_mc_start
    print(f"\n  MC complete: {mc_elapsed:.1f}s ({mc_elapsed/N_SIMULATIONS*1000:.1f}ms/run)")

    # ═══════════════════════════════════════════════════════════
    # Step 3: Statistics
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Step 3: Percentile Statistics")
    print("=" * 60)

    metrics_to_report = ['final_equity', 'sharpe', 'max_dd', 'annual_return']
    percentile_stats = {}
    for metric in metrics_to_report:
        percentile_stats[metric] = compute_percentiles(mc_results, metric)

    print(f"\n  {'Metric':<18} {'P5':>10} {'P25':>10} {'P50':>10} {'P75':>10} {'P95':>10} {'Mean':>10}")
    print(f"  {'-'*78}")
    for metric in metrics_to_report:
        p = percentile_stats[metric]
        label = metric.replace('_', ' ').title()
        print(f"  {label:<18} {p['P5']:>10,.2f} {p['P25']:>10,.2f} {p['P50']:>10,.2f} "
              f"{p['P75']:>10,.2f} {p['P95']:>10,.2f} {p['mean']:>10,.2f}")

    # Probability metrics
    pnl_values = [r['total_pnl'] for r in mc_results]
    dd_values = [r['max_dd'] for r in mc_results]
    n_trades_values = [r['n_trades'] for r in mc_results]

    prob_positive = sum(1 for p in pnl_values if p > 0) / len(pnl_values) * 100
    prob_dd_1000 = sum(1 for d in dd_values if d > 1000) / len(dd_values) * 100
    prob_dd_2000 = sum(1 for d in dd_values if d > 2000) / len(dd_values) * 100

    print(f"\n  Probability metrics:")
    print(f"    P(positive return)   = {prob_positive:6.1f}%")
    print(f"    P(drawdown > $1000)  = {prob_dd_1000:6.1f}%")
    print(f"    P(drawdown > $2000)  = {prob_dd_2000:6.1f}%")
    print(f"    Avg trades/run       = {np.mean(n_trades_values):.0f}")

    # Expected vs worst-case
    p5_equity = percentile_stats['final_equity']['P5']
    p50_equity = percentile_stats['final_equity']['P50']
    p95_equity = percentile_stats['final_equity']['P95']
    worst_run = min(mc_results, key=lambda r: r['total_pnl'])
    best_run = max(mc_results, key=lambda r: r['total_pnl'])

    print(f"\n  Expected vs Worst-Case:")
    print(f"    Best run:     equity=${best_run['final_equity']:>10,.2f}  "
          f"(PnL=${best_run['total_pnl']:>10,.2f}, Sharpe={best_run['sharpe']:.3f})")
    print(f"    P95 (good):   equity=${p95_equity:>10,.2f}")
    print(f"    P50 (median): equity=${p50_equity:>10,.2f}")
    print(f"    P5 (bad):     equity=${p5_equity:>10,.2f}")
    print(f"    Worst run:    equity=${worst_run['final_equity']:>10,.2f}  "
          f"(PnL=${worst_run['total_pnl']:>10,.2f}, Sharpe={worst_run['sharpe']:.3f})")

    # Noise breakdown
    avg_dropped = np.mean([r['n_dropped'] for r in mc_results])
    avg_early = np.mean([r['n_early_exit'] for r in mc_results])
    avg_pos_lim = np.mean([r['n_pos_limited'] for r in mc_results])
    print(f"\n  Noise breakdown (averages per run):")
    print(f"    Signals dropped:      {avg_dropped:.1f}")
    print(f"    Early exits:          {avg_early:.1f}")
    print(f"    Position-limited:     {avg_pos_lim:.1f}")

    results['mc_main'] = {
        'n_simulations': N_SIMULATIONS,
        'params': {k: v for k, v in mc_params.items() if k != 'r89_lots'},
        'r89_lots': {k: v for k, v in R89_LOTS.items()},
        'percentiles': percentile_stats,
        'prob_positive_pct': round(prob_positive, 1),
        'prob_dd_gt_1000_pct': round(prob_dd_1000, 1),
        'prob_dd_gt_2000_pct': round(prob_dd_2000, 1),
        'avg_trades_per_run': round(float(np.mean(n_trades_values)), 0),
        'best_run': best_run,
        'worst_run': worst_run,
        'avg_dropped': round(avg_dropped, 1),
        'avg_early_exit': round(avg_early, 1),
        'avg_pos_limited': round(avg_pos_lim, 1),
        'elapsed_s': round(mc_elapsed, 1),
    }

    # ═══════════════════════════════════════════════════════════
    # Step 4: Sensitivity Analysis
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Step 4: Sensitivity Analysis")
    print("=" * 60)

    N_SENSITIVITY = 100

    # --- 4a: Doubled slippage ---
    print(f"\n  4a) Doubled slippage (0-0.10), {N_SENSITIVITY} runs...")
    params_2x_slip = dict(mc_params)
    params_2x_slip['max_slippage'] = 0.10

    sens_2x_slip = []
    for run_i in range(N_SENSITIVITY):
        res = run_single_mc(all_trades, 10000 + run_i, params_2x_slip)
        sens_2x_slip.append(res)

    slip_pctiles = compute_percentiles(sens_2x_slip, 'total_pnl')
    slip_sharpe_pctiles = compute_percentiles(sens_2x_slip, 'sharpe')
    base_p50_pnl = percentile_stats['final_equity']['P50'] - CAPITAL
    slip_p50_pnl = slip_pctiles['P50']
    pnl_impact = slip_p50_pnl - (percentile_stats['final_equity']['P50'] - CAPITAL)

    print(f"    Baseline median PnL: ${percentile_stats['final_equity']['P50'] - CAPITAL:>10,.2f}")
    print(f"    2x-slip  median PnL: ${slip_p50_pnl:>10,.2f}  (delta=${pnl_impact:>+,.2f})")
    print(f"    Baseline median Sharpe: {percentile_stats['sharpe']['P50']:.3f}")
    print(f"    2x-slip  median Sharpe: {slip_sharpe_pctiles['P50']:.3f}")

    # --- 4b: Higher signal drop rate ---
    print(f"\n  4b) 5% signal drop rate, {N_SENSITIVITY} runs...")
    params_5pct_drop = dict(mc_params)
    params_5pct_drop['signal_drop_rate'] = 0.05

    sens_5pct_drop = []
    for run_i in range(N_SENSITIVITY):
        res = run_single_mc(all_trades, 20000 + run_i, params_5pct_drop)
        sens_5pct_drop.append(res)

    drop_pctiles = compute_percentiles(sens_5pct_drop, 'total_pnl')
    drop_sharpe_pctiles = compute_percentiles(sens_5pct_drop, 'sharpe')
    drop_p50_pnl = drop_pctiles['P50']
    drop_pnl_impact = drop_p50_pnl - (percentile_stats['final_equity']['P50'] - CAPITAL)

    print(f"    Baseline median PnL: ${percentile_stats['final_equity']['P50'] - CAPITAL:>10,.2f}")
    print(f"    5%-drop  median PnL: ${drop_p50_pnl:>10,.2f}  (delta=${drop_pnl_impact:>+,.2f})")
    print(f"    Baseline median Sharpe: {percentile_stats['sharpe']['P50']:.3f}")
    print(f"    5%-drop  median Sharpe: {drop_sharpe_pctiles['P50']:.3f}")

    # Distribution comparison
    print(f"\n  Distribution comparison:")
    print(f"  {'Scenario':<22} {'P5 PnL':>10} {'P50 PnL':>10} {'P95 PnL':>10} {'P50 Sharpe':>12}")
    print(f"  {'-'*66}")
    base_pnl_pctiles = compute_percentiles(mc_results, 'total_pnl')
    print(f"  {'Baseline (2% drop)':<22} "
          f"${base_pnl_pctiles['P5']:>9,.0f} "
          f"${base_pnl_pctiles['P50']:>9,.0f} "
          f"${base_pnl_pctiles['P95']:>9,.0f} "
          f"{percentile_stats['sharpe']['P50']:>12.3f}")
    print(f"  {'2x Slippage':<22} "
          f"${slip_pctiles['P5']:>9,.0f} "
          f"${slip_pctiles['P50']:>9,.0f} "
          f"${slip_pctiles['P95']:>9,.0f} "
          f"{slip_sharpe_pctiles['P50']:>12.3f}")
    print(f"  {'5% Signal Drop':<22} "
          f"${drop_pctiles['P5']:>9,.0f} "
          f"${drop_pctiles['P50']:>9,.0f} "
          f"${drop_pctiles['P95']:>9,.0f} "
          f"{drop_sharpe_pctiles['P50']:>12.3f}")

    results['sensitivity'] = {
        'doubled_slippage': {
            'n_runs': N_SENSITIVITY,
            'max_slippage': 0.10,
            'pnl_percentiles': slip_pctiles,
            'sharpe_percentiles': slip_sharpe_pctiles,
        },
        'high_signal_drop': {
            'n_runs': N_SENSITIVITY,
            'signal_drop_rate': 0.05,
            'pnl_percentiles': drop_pctiles,
            'sharpe_percentiles': drop_sharpe_pctiles,
        },
    }

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print("  R106 SUMMARY — Full Production Simulation")
    print("=" * 80)

    print(f"\n  Deterministic baseline:  Sharpe={base_sharpe:.3f}, PnL=${base_pnl:,.2f}")
    print(f"  MC runs:                 {N_SIMULATIONS}")
    print(f"  Sensitivity runs:        {N_SENSITIVITY * 2}")

    print(f"\n  Production reality check:")
    print(f"    Median final equity:  ${p50_equity:,.2f} (starting ${CAPITAL:,})")
    print(f"    P5-P95 equity range:  ${p5_equity:,.2f} — ${p95_equity:,.2f}")
    print(f"    P(positive return):   {prob_positive:.1f}%")
    print(f"    P(DD > $1000):        {prob_dd_1000:.1f}%")
    print(f"    Median Sharpe:        {percentile_stats['sharpe']['P50']:.3f}")
    print(f"    Median MaxDD:         ${percentile_stats['max_dd']['P50']:,.2f}")

    degradation_slip = slip_sharpe_pctiles['P50'] - percentile_stats['sharpe']['P50']
    degradation_drop = drop_sharpe_pctiles['P50'] - percentile_stats['sharpe']['P50']
    print(f"\n  Sensitivity impact on Sharpe:")
    print(f"    2x slippage:          {degradation_slip:+.3f}")
    print(f"    5% signal drop:       {degradation_drop:+.3f}")

    robust = prob_positive >= 80 and percentile_stats['sharpe']['P5'] > 0
    fragile = prob_positive < 60 or percentile_stats['sharpe']['P25'] < 0

    if robust:
        verdict = "ROBUST — portfolio survives realistic execution conditions"
    elif fragile:
        verdict = "FRAGILE — high sensitivity to execution noise"
    else:
        verdict = "MODERATE — acceptable but monitor execution quality"

    print(f"\n  VERDICT: {verdict}")

    results['verdict'] = verdict
    results['elapsed_s'] = round(elapsed, 1)

    out_file = OUTPUT_DIR / "r106_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Saved: {out_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
