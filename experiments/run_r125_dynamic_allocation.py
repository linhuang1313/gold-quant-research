#!/usr/bin/env python3
"""
R125 — Dynamic Lot Allocation Methods
=======================================
Compare dynamic allocation vs R89 fixed lots across 4 strategies.

Phase 1: Run all 4 strategies at unit lot, collect daily PnL
Phase 2: Implement allocation methods (InvVol, RiskParity, RollSharpe, Kelly)
Phase 3: Walk-forward portfolio simulation (6-month lookback, monthly rebalance)
Phase 4: Monte Carlo (500 paths)
Phase 5: K-Fold validation (5 folds)
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

OUTPUT_DIR = Path("results/r125_dynamic_allocation")
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
# Allocation methods
# ═══════════════════════════════════════════════════════════════

def alloc_inverse_vol(daily_df, lookback=20):
    """Inverse volatility: lots = 1 / rolling_20d_vol, normalized."""
    vols = daily_df.rolling(lookback).std()
    inv = 1.0 / vols.replace(0, np.nan)
    inv = inv.fillna(0)
    row_sum = inv.sum(axis=1).replace(0, 1)
    weights = inv.div(row_sum, axis=0)
    return weights * TOTAL_LOT_BUDGET


def alloc_risk_parity(daily_df, lookback=60):
    """Risk parity: equalize risk contribution using rolling covariance."""
    n_strats = daily_df.shape[1]
    weights = pd.DataFrame(0.0, index=daily_df.index, columns=daily_df.columns)

    for i in range(lookback, len(daily_df)):
        window = daily_df.iloc[i - lookback:i]
        cov = window.cov().values
        vols = window.std().values

        if np.any(vols == 0) or np.any(np.isnan(vols)):
            weights.iloc[i] = TOTAL_LOT_BUDGET / n_strats
            continue

        inv_vol = 1.0 / vols
        w = inv_vol / inv_vol.sum()

        for _ in range(20):
            sigma_p = np.sqrt(w @ cov @ w) if (w @ cov @ w) > 0 else 1e-6
            mc = cov @ w
            rc = w * mc / sigma_p
            target = sigma_p / n_strats
            w = w * (target / np.where(rc > 0, rc, 1e-6))
            w = np.maximum(w, 0)
            w_sum = w.sum()
            if w_sum > 0:
                w = w / w_sum

        weights.iloc[i] = w * TOTAL_LOT_BUDGET

    return weights


def alloc_rolling_sharpe(daily_df, lookback=60):
    """Lots proportional to rolling 60d Sharpe (set to 0 if negative)."""
    weights = pd.DataFrame(0.0, index=daily_df.index, columns=daily_df.columns)

    for i in range(lookback, len(daily_df)):
        window = daily_df.iloc[i - lookback:i]
        sharpes = {}
        for col in daily_df.columns:
            arr = window[col].values
            s = np.std(arr, ddof=1)
            sh = float(np.mean(arr) / s * np.sqrt(252)) if s > 0 and len(arr) > 10 else 0.0
            sharpes[col] = max(sh, 0.0)

        total_sh = sum(sharpes.values())
        if total_sh > 0:
            for col in daily_df.columns:
                weights.loc[weights.index[i], col] = (sharpes[col] / total_sh) * TOTAL_LOT_BUDGET
        else:
            for col in daily_df.columns:
                weights.loc[weights.index[i], col] = TOTAL_LOT_BUDGET / len(daily_df.columns)

    return weights


def alloc_kelly(daily_df, lookback=60):
    """1/4 Kelly: f* = p/a - q/b where p=win_rate, q=1-p, a=avg_loss, b=avg_win."""
    weights = pd.DataFrame(0.0, index=daily_df.index, columns=daily_df.columns)

    for i in range(lookback, len(daily_df)):
        window = daily_df.iloc[i - lookback:i]
        raw_f = {}
        for col in daily_df.columns:
            arr = window[col].dropna().values
            if len(arr) < 10:
                raw_f[col] = 0.0
                continue
            wins = arr[arr > 0]
            losses = arr[arr < 0]
            if len(wins) == 0 or len(losses) == 0:
                raw_f[col] = 0.0
                continue
            p = len(wins) / len(arr)
            q = 1 - p
            b = float(np.mean(wins))
            a = float(np.mean(np.abs(losses)))
            if a == 0 or b == 0:
                raw_f[col] = 0.0
                continue
            f_star = p / a - q / b
            raw_f[col] = max(f_star * 0.25, 0.0)

        total_f = sum(raw_f.values())
        if total_f > 0:
            for col in daily_df.columns:
                weights.loc[weights.index[i], col] = (raw_f[col] / total_f) * TOTAL_LOT_BUDGET
        else:
            for col in daily_df.columns:
                weights.loc[weights.index[i], col] = TOTAL_LOT_BUDGET / len(daily_df.columns)

    return weights


# ═══════════════════════════════════════════════════════════════
# Walk-forward portfolio simulation
# ═══════════════════════════════════════════════════════════════

def walk_forward_portfolio(unit_dailies_df, alloc_func, lookback_months=6,
                           rebalance='M', alloc_lookback=None):
    """
    Rolling lookback_months to compute weights, rebalance monthly.
    unit_dailies_df: DataFrame with columns = strategy names, index = dates, values = unit-lot daily PnL.
    """
    if alloc_lookback is None:
        alloc_lookback = lookback_months * 21

    all_weights = alloc_func(unit_dailies_df, lookback=alloc_lookback)

    rebalance_dates = unit_dailies_df.resample(rebalance).last().index
    active_weights = pd.Series({col: TOTAL_LOT_BUDGET / len(STRAT_ORDER) for col in STRAT_ORDER})

    portfolio_daily = pd.Series(0.0, index=unit_dailies_df.index)

    for i, dt in enumerate(unit_dailies_df.index):
        if dt in rebalance_dates and dt in all_weights.index:
            w = all_weights.loc[dt]
            if w.sum() > 0:
                active_weights = w

        for col in unit_dailies_df.columns:
            lot_multiplier = active_weights.get(col, 0) / UNIT_LOT
            portfolio_daily[dt] += unit_dailies_df.loc[dt, col] * lot_multiplier

    return portfolio_daily.values


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80, flush=True)
    print("  R125 — Dynamic Lot Allocation Methods", flush=True)
    print("=" * 80, flush=True)

    print("\n  Loading H1 data...", flush=True)
    h1_df = load_h1()
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})", flush=True)

    print("  Loading L8_MAX DataBundle...", flush=True)
    bundle = DataBundle.load_custom()

    all_results = {}

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Run all strategies at unit lot
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 1: Run All 4 Strategies at Unit Lot", flush=True)
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
              f"PnL=${stats['pnl']:9.0f}  WR={stats['wr']:.1f}%", flush=True)

    all_dates = set()
    for ds in unit_dailies.values():
        all_dates.update(ds.index)
    all_dates = sorted(all_dates)
    idx = pd.DatetimeIndex(all_dates)

    unit_df = pd.DataFrame(0.0, index=idx, columns=STRAT_ORDER)
    for name in STRAT_ORDER:
        ds = unit_dailies[name]
        unit_df[name] = ds.reindex(idx, fill_value=0.0)

    print(f"\n    Combined daily PnL matrix: {unit_df.shape[0]} days × {unit_df.shape[1]} strategies", flush=True)

    # Fixed lots baseline
    fixed_daily = np.zeros(len(idx))
    for name in STRAT_ORDER:
        mult = R89_LOTS[name] / UNIT_LOT
        fixed_daily += unit_df[name].values * mult
    fixed_metrics = _portfolio_metrics(fixed_daily)
    print(f"\n    R89 Fixed lots baseline: Sharpe={fixed_metrics['sharpe']}, "
          f"PnL=${fixed_metrics['pnl']:.0f}, MaxDD=${fixed_metrics['max_dd']:.0f}, "
          f"Calmar={fixed_metrics['calmar']}", flush=True)

    all_results['phase1'] = {
        'strategies': {name: _compute_stats(strat_trades[name]) for name in STRAT_ORDER},
        'fixed_baseline': fixed_metrics,
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Implement allocation methods
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: Allocation Methods", flush=True)
    print("=" * 70, flush=True)

    alloc_methods = {
        'InvVol_20d': lambda df, lookback=20: alloc_inverse_vol(df, lookback=lookback),
        'RiskParity_60d': lambda df, lookback=60: alloc_risk_parity(df, lookback=lookback),
        'RollSharpe_60d': lambda df, lookback=60: alloc_rolling_sharpe(df, lookback=lookback),
        'Kelly_1_4_60d': lambda df, lookback=60: alloc_kelly(df, lookback=lookback),
    }

    alloc_lookbacks = {
        'InvVol_20d': 20,
        'RiskParity_60d': 60,
        'RollSharpe_60d': 60,
        'Kelly_1_4_60d': 60,
    }

    for method_name, alloc_func in alloc_methods.items():
        weights = alloc_func(unit_df, lookback=alloc_lookbacks[method_name])
        tail = weights.tail(5)
        print(f"\n  {method_name} — last 5 daily weights:", flush=True)
        for i_row in range(len(tail)):
            row = tail.iloc[i_row]
            dt = tail.index[i_row]
            print(f"    {dt.date()}: " + "  ".join(f"{name}={row[name]:.3f}" for name in STRAT_ORDER), flush=True)

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Walk-forward portfolio simulation
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 3: Walk-Forward Portfolio Simulation", flush=True)
    print("  (6-month lookback, monthly rebalance)", flush=True)
    print("=" * 70, flush=True)

    phase3 = {}
    print(f"\n  {'Method':<20s}  {'Sharpe':>7s}  {'PnL':>10s}  {'MaxDD':>8s}  {'Calmar':>7s}", flush=True)
    print("  " + "-" * 60, flush=True)

    # Fixed baseline
    print(f"  {'R89_Fixed':<20s}  {fixed_metrics['sharpe']:>7.3f}  ${fixed_metrics['pnl']:>9.0f}  "
          f"${fixed_metrics['max_dd']:>7.0f}  {fixed_metrics['calmar']:>7.3f}", flush=True)
    phase3['R89_Fixed'] = fixed_metrics

    for method_name, alloc_func in alloc_methods.items():
        port_daily = walk_forward_portfolio(
            unit_df, alloc_func,
            lookback_months=6, rebalance='M',
            alloc_lookback=alloc_lookbacks[method_name])
        m = _portfolio_metrics(port_daily)
        delta = m['sharpe'] - fixed_metrics['sharpe']
        marker = " ***" if delta > 0.1 else " *" if delta > 0.05 else ""
        print(f"  {method_name:<20s}  {m['sharpe']:>7.3f}  ${m['pnl']:>9.0f}  "
              f"${m['max_dd']:>7.0f}  {m['calmar']:>7.3f}  (ΔSh={delta:+.3f}){marker}", flush=True)
        phase3[method_name] = m

    all_results['phase3_walkforward'] = phase3

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Monte Carlo (500 paths)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 4: Monte Carlo Simulation (500 paths)", flush=True)
    print("=" * 70, flush=True)

    rng = np.random.RandomState(42)
    N_MC = 500

    mc_results = {}
    methods_to_mc = list(alloc_methods.keys()) + ['R89_Fixed']

    for method_name in methods_to_mc:
        mc_sharpes = []
        for path_i in range(N_MC):
            perturbed_df = unit_df.copy()
            spread_noise = rng.uniform(-0.5, 0.5, size=perturbed_df.shape)
            pnl_scale = rng.uniform(0.9, 1.1, size=perturbed_df.shape)
            perturbed_df = perturbed_df * pnl_scale + spread_noise * UNIT_LOT * PV * 0.01

            if method_name == 'R89_Fixed':
                port = np.zeros(len(perturbed_df))
                for name in STRAT_ORDER:
                    mult = R89_LOTS[name] / UNIT_LOT
                    port += perturbed_df[name].values * mult
            else:
                alloc_func = alloc_methods[method_name]
                port = walk_forward_portfolio(
                    perturbed_df, alloc_func,
                    lookback_months=6, rebalance='M',
                    alloc_lookback=alloc_lookbacks[method_name])

            mc_sharpes.append(_sharpe(port))

        mc_arr = np.array(mc_sharpes)
        mc_results[method_name] = {
            'mean': round(float(np.mean(mc_arr)), 3),
            'median': round(float(np.median(mc_arr)), 3),
            'std': round(float(np.std(mc_arr)), 3),
            'p5': round(float(np.percentile(mc_arr, 5)), 3),
            'p25': round(float(np.percentile(mc_arr, 25)), 3),
            'p75': round(float(np.percentile(mc_arr, 75)), 3),
            'p95': round(float(np.percentile(mc_arr, 95)), 3),
            'pct_positive': round(float((mc_arr > 0).mean() * 100), 1),
        }

        print(f"  {method_name:<20s}: mean={mc_results[method_name]['mean']:.3f}  "
              f"median={mc_results[method_name]['median']:.3f}  "
              f"[p5={mc_results[method_name]['p5']:.3f}, p95={mc_results[method_name]['p95']:.3f}]  "
              f"positive={mc_results[method_name]['pct_positive']:.0f}%", flush=True)

    all_results['phase4_monte_carlo'] = mc_results

    # ═══════════════════════════════════════════════════════════
    # Phase 5: K-Fold Validation (5 folds)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 5: K-Fold Validation (5 folds)", flush=True)
    print("=" * 70, flush=True)

    kfold_results = {}

    for method_name in ['R89_Fixed'] + list(alloc_methods.keys()):
        fold_sharpes = []

        for fold_name, start, end in FOLDS:
            fold_df = unit_df[(unit_df.index >= start) & (unit_df.index < end)]
            if len(fold_df) < 60:
                fold_sharpes.append(0.0)
                continue

            if method_name == 'R89_Fixed':
                port = np.zeros(len(fold_df))
                for name in STRAT_ORDER:
                    mult = R89_LOTS[name] / UNIT_LOT
                    port += fold_df[name].values * mult
            else:
                alloc_func = alloc_methods[method_name]
                port = walk_forward_portfolio(
                    fold_df, alloc_func,
                    lookback_months=6, rebalance='M',
                    alloc_lookback=alloc_lookbacks[method_name])

            fold_sharpes.append(round(_sharpe(port), 3))

        pos = sum(1 for s in fold_sharpes if s > 0)
        mean_s = round(float(np.mean(fold_sharpes)), 3)
        status = "PASS" if pos >= 3 else "FAIL"
        print(f"  {method_name:<20s}: {fold_sharpes}  -> {pos}/5 [{status}] mean={mean_s}", flush=True)
        kfold_results[method_name] = {
            'fold_sharpes': fold_sharpes,
            'positive': pos, 'mean': mean_s, 'pass': pos >= 3,
        }

    all_results['phase5_kfold'] = kfold_results

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    print("\n" + "=" * 80, flush=True)
    print("  R125 FINAL SUMMARY", flush=True)
    print("=" * 80, flush=True)

    print(f"\n  Walk-Forward Portfolio Sharpe:", flush=True)
    sorted_methods = sorted(phase3.items(), key=lambda x: x[1]['sharpe'], reverse=True)
    for method_name, m in sorted_methods:
        marker = " <-- BEST" if method_name == sorted_methods[0][0] else ""
        print(f"    {method_name:<20s}: Sharpe={m['sharpe']:7.3f}, Calmar={m['calmar']:7.3f}{marker}", flush=True)

    best_dynamic = max((k, v) for k, v in phase3.items() if k != 'R89_Fixed')
    delta = best_dynamic[1]['sharpe'] - fixed_metrics['sharpe']
    if delta > 0:
        print(f"\n  Best dynamic method: {best_dynamic[0]} (ΔSharpe vs fixed = {delta:+.3f})", flush=True)
    else:
        print(f"\n  Fixed lots still best (all dynamic methods underperform by Sharpe)", flush=True)

    print(f"\n  K-Fold summary:", flush=True)
    for method_name, kf in kfold_results.items():
        status = "PASS" if kf['pass'] else "FAIL"
        print(f"    {method_name:<20s}: {kf['positive']}/5 [{status}] mean={kf['mean']}", flush=True)

    all_results['elapsed_s'] = round(elapsed, 1)

    out_file = OUTPUT_DIR / "r125_results.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Saved: {out_file}", flush=True)
    print(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)", flush=True)
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
