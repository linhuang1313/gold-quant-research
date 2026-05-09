#!/usr/bin/env python3
"""
R101 — Mega Experiment: Monte Carlo + Walk-Forward + Exhaustive Combos
=======================================================================
~100-hour experiment in 3 parts with full checkpointing.

Part A (~20h): 10,000-iteration Monte Carlo stress test on current portfolio.
Part B (~50h): Walk-forward parameter optimization with rolling windows.
Part C (~30h): Exhaustive 2^7 strategy combination search with lot allocation.

Checkpoints saved every hour / every 1000 MC iters / after each WF window /
after each combo batch — safe to interrupt and resume at any point.
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import product
from copy import deepcopy

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS, load_m15, load_h1_aligned, H1_CSV_PATH

OUTPUT_DIR = Path("results/r101_mega_experiment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.09, 'SESS_BO': 0.09}

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]


# ─── Checkpoint helpers ──────────────────────────────────────────────
def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_checkpoint(cp):
    tmp = CHECKPOINT_FILE.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(cp, f, indent=2, default=str)
    tmp.replace(CHECKPOINT_FILE)


# ─── Core helpers ────────────────────────────────────────────────────
def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


def add_psar(df, af_start=0.01, af_max=0.05):
    df = df.copy()
    n = len(df)
    psar = np.zeros(n)
    direction = np.ones(n)
    af = af_start
    ep = df['High'].iloc[0]
    psar[0] = df['Low'].iloc[0]
    for i in range(1, n):
        prev = psar[i - 1]
        if direction[i - 1] == 1:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], df['Low'].iloc[i - 1], df['Low'].iloc[max(0, i - 2)])
            if df['Low'].iloc[i] < psar[i]:
                direction[i] = -1
                psar[i] = ep
                ep = df['Low'].iloc[i]
                af = af_start
            else:
                direction[i] = 1
                if df['High'].iloc[i] > ep:
                    ep = df['High'].iloc[i]
                    af = min(af + af_start, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], df['High'].iloc[i - 1], df['High'].iloc[max(0, i - 2)])
            if df['High'].iloc[i] > psar[i]:
                direction[i] = 1
                psar[i] = ep
                ep = df['High'].iloc[i]
                af = af_start
            else:
                direction[i] = -1
                if df['Low'].iloc[i] < ep:
                    ep = df['Low'].iloc[i]
                    af = min(af + af_start, af_max)
    df['PSAR_dir'] = direction
    df['ATR'] = compute_atr(df)
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
    ad = trail_act_atr * pos['atr']
    td = trail_dist_atr * pos['atr']
    if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
        ts_p = h - td
        if lo_v <= ts_p:
            return _mk(pos, c, times[i], "Trail", i,
                       (ts_p - pos['entry'] - spread) * lot * pv)
    elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
        ts_p = lo_v + td
        if h >= ts_p:
            return _mk(pos, c, times[i], "Trail", i,
                       (pos['entry'] - ts_p - spread) * lot * pv)
    if held >= max_hold:
        return _mk(pos, c, times[i], "Timeout", i, pnl_c)
    return None


# ─── Portfolio stats ─────────────────────────────────────────────────
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


def worst_month(daily_series):
    if daily_series.empty:
        return 0.0
    monthly = daily_series.resample('M').sum()
    return float(monthly.min()) if len(monthly) > 0 else 0.0


def annual_return(daily_series):
    if daily_series.empty:
        return 0.0
    years = (daily_series.index[-1] - daily_series.index[0]).days / 365.25
    if years < 0.1:
        return 0.0
    return float(daily_series.sum() / years)


# ─── Strategy backtests (4 core) ────────────────────────────────────
def bt_psar(h1_df, spread, lot, maxloss_cap=0,
            sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04, max_hold=20):
    df = add_psar(h1_df).dropna(subset=['PSAR_dir', 'ATR'])
    c = df['Close'].values
    h = df['High'].values
    lo = df['Low'].values
    pdir = df['PSAR_dir'].values
    atr = df['ATR'].values
    times = df.index
    n = len(df)
    trades = []
    pos = None
    last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result)
                pos = None
                last_exit = i
                continue
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
        if pdir[i - 1] == -1 and pdir[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif pdir[i - 1] == 1 and pdir[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom(h1_df, spread, lot, maxloss_cap=0,
             fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
             trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values
    h = df['High'].values
    lo = df['Low'].values
    atr = df['ATR'].values
    times = df.index
    n = len(df)
    max_lb = max(fast, slow)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        if c[i - fast] > 0:
            s += 0.5 * np.sign(c[i] / c[i - fast] - 1.0)
        if c[i - slow] > 0:
            s += 0.5 * np.sign(c[i] / c[i - slow] - 1.0)
        score[i] = s
    trades = []
    pos = None
    last_exit = -999
    for i in range(max_lb + 1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result)
                pos = None
                last_exit = i
                continue
            if pos['dir'] == 'BUY' and score[i] < 0:
                pnl = (c[i] - pos['entry'] - spread) * lot * PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl))
                pos = None
                last_exit = i
                continue
            elif pos['dir'] == 'SELL' and score[i] > 0:
                pnl = (pos['entry'] - c[i] - spread) * lot * PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl))
                pos = None
                last_exit = i
                continue
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
        if np.isnan(score[i]) or np.isnan(score[i - 1]):
            continue
        if score[i] > 0 and score[i - 1] <= 0:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i - 1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_sess_bo(h1_df, spread, lot, maxloss_cap=0,
               session_hour=12, lookback=4, sl_atr=4.5, tp_atr=4.0,
               trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values
    h = df['High'].values
    lo = df['Low'].values
    atr = df['ATR'].values
    hours = df.index.hour
    times = df.index
    n = len(df)
    trades = []
    pos = None
    last_exit = -999
    for i in range(lookback, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result)
                pos = None
                last_exit = i
                continue
            continue
        if hours[i] != session_hour:
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
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


# ─── New candidate strategies (Part C) ──────────────────────────────
def bt_donchian50(h1_df, spread, lot, maxloss_cap=0,
                  channel=50, sl_atr=3.0, tp_atr=4.0,
                  trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df['HH'] = df['High'].rolling(channel).max()
    df['LL'] = df['Low'].rolling(channel).min()
    df = df.dropna(subset=['ATR', 'HH', 'LL'])
    c = df['Close'].values
    h = df['High'].values
    lo = df['Low'].values
    atr = df['ATR'].values
    hh = df['HH'].values
    ll = df['LL'].values
    times = df.index
    n = len(df)
    trades = []
    pos = None
    last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result)
                pos = None
                last_exit = i
                continue
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
        if c[i] > hh[i - 1]:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll[i - 1]:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_overnight_gap(h1_df, spread, lot, maxloss_cap=0,
                     gap_atr_mult=0.5, sl_atr=2.0, tp_atr=3.0,
                     trail_act=0.14, trail_dist=0.025, max_hold=12):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values
    h = df['High'].values
    lo = df['Low'].values
    o = df['Open'].values
    atr = df['ATR'].values
    hours = df.index.hour
    times = df.index
    n = len(df)
    trades = []
    pos = None
    last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result)
                pos = None
                last_exit = i
                continue
            continue
        if hours[i] not in (0, 22):
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
        gap = o[i] - c[i - 1]
        threshold = gap_atr_mult * atr[i]
        if abs(gap) < threshold:
            continue
        if gap > 0:
            pos = {'dir': 'SELL', 'entry': o[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        else:
            pos = {'dir': 'BUY', 'entry': o[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_bollinger_squeeze(h1_df, spread, lot, maxloss_cap=0,
                         bb_period=20, bb_std=2.0, squeeze_pct=20,
                         sl_atr=2.0, tp_atr=3.0,
                         trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    sma = df['Close'].rolling(bb_period).mean()
    std = df['Close'].rolling(bb_period).std()
    df['BB_upper'] = sma + bb_std * std
    df['BB_lower'] = sma - bb_std * std
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / sma
    df['BB_width_pctrank'] = df['BB_width'].rolling(200, min_periods=50).apply(
        lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100, raw=False
    )
    df = df.dropna(subset=['ATR', 'BB_upper', 'BB_lower', 'BB_width_pctrank'])
    c = df['Close'].values
    h = df['High'].values
    lo = df['Low'].values
    atr = df['ATR'].values
    bb_up = df['BB_upper'].values
    bb_lo = df['BB_lower'].values
    width_rank = df['BB_width_pctrank'].values
    times = df.index
    n = len(df)
    trades = []
    pos = None
    last_exit = -999
    in_squeeze = False
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result)
                pos = None
                last_exit = i
                in_squeeze = False
                continue
            continue
        if width_rank[i] < squeeze_pct:
            in_squeeze = True
        if not in_squeeze:
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
        if c[i] > bb_up[i]:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
            in_squeeze = False
        elif c[i] < bb_lo[i]:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
            in_squeeze = False
    return trades


# ─── Run all strategies to get base trades ───────────────────────────
ALL_STRATEGY_NAMES = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO', 'DONCH50', 'OVGAP', 'BBSQZ']

def run_strategy(name, data_bundle, h1_df, spread, lot, cap):
    if name == 'L8_MAX':
        return bt_l8_max(data_bundle, spread, lot, maxloss_cap=cap)
    elif name == 'PSAR':
        return bt_psar(h1_df, spread, lot, maxloss_cap=cap)
    elif name == 'TSMOM':
        return bt_tsmom(h1_df, spread, lot, maxloss_cap=cap)
    elif name == 'SESS_BO':
        return bt_sess_bo(h1_df, spread, lot, maxloss_cap=cap)
    elif name == 'DONCH50':
        return bt_donchian50(h1_df, spread, lot, maxloss_cap=cap)
    elif name == 'OVGAP':
        return bt_overnight_gap(h1_df, spread, lot, maxloss_cap=cap)
    elif name == 'BBSQZ':
        return bt_bollinger_squeeze(h1_df, spread, lot, maxloss_cap=cap)
    else:
        raise ValueError(f"Unknown strategy: {name}")


def run_strategy_with_params(name, data_bundle, h1_df, spread, lot, cap, params):
    """Run strategy with non-default parameters for grid search."""
    p = dict(params)
    effective_cap = p.pop('maxloss_cap', cap)
    if name == 'PSAR':
        return bt_psar(h1_df, spread, lot, maxloss_cap=effective_cap, **p)
    elif name == 'TSMOM':
        return bt_tsmom(h1_df, spread, lot, maxloss_cap=effective_cap, **p)
    elif name == 'SESS_BO':
        return bt_sess_bo(h1_df, spread, lot, maxloss_cap=effective_cap, **p)
    elif name == 'L8_MAX':
        return bt_l8_max(data_bundle, spread, lot, maxloss_cap=effective_cap)
    else:
        raise ValueError(f"No param grid for strategy: {name}")


# ─── Helper: combine multiple strategy trades into portfolio series ──
def combine_portfolio(trade_lists):
    all_trades = []
    for tl in trade_lists:
        all_trades.extend(tl)
    return trades_to_daily_series(all_trades), all_trades


# ─── PART A: Monte Carlo Stress Test ────────────────────────────────
MC_ITERATIONS = 10000
MC_CHECKPOINT_INTERVAL = 1000


def run_part_a(checkpoint):
    print("=" * 80)
    print("PART A: Monte Carlo Stress Test (10,000 iterations)")
    print("=" * 80)
    t0 = time.time()

    data_bundle = DataBundle.load_default()
    h1_df = data_bundle.h1_df

    core_strats = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']
    core_caps = {k: CAPS[k] for k in core_strats}

    print("\n[A] Running base backtests for current portfolio...")
    base_trades = {}
    for name in core_strats:
        lot = R89_LOTS[name]
        cap = core_caps[name]
        trades = run_strategy(name, data_bundle, h1_df, SPREAD, lot, cap)
        base_trades[name] = trades
        print(f"  {name}: {len(trades)} trades, lot={lot}")

    base_daily, base_all = combine_portfolio(list(base_trades.values()))
    base_arr = base_daily.values
    print(f"\n[A] Base portfolio: Sharpe={sharpe(base_arr):.2f}, "
          f"MaxDD=${max_dd(base_arr):.0f}, Total=${sum(t['pnl'] for t in base_all):.0f}")

    completed_iters = checkpoint.get('part_a_iters', 0)
    mc_results = checkpoint.get('part_a_results', {
        'sharpes': [], 'max_dds': [], 'total_pnls': [], 'worst_months': [],
        'annual_returns': [],
    })

    rng = np.random.default_rng(seed=42)

    print(f"\n[A] Starting MC from iteration {completed_iters}...")
    for iteration in range(completed_iters, MC_ITERATIONS):
        if iteration % 500 == 0 and iteration > completed_iters:
            elapsed = time.time() - t0
            rate = iteration / elapsed if elapsed > 0 else 0
            eta = (MC_ITERATIONS - iteration) / rate / 3600 if rate > 0 else 0
            print(f"  MC iter {iteration}/{MC_ITERATIONS} "
                  f"({elapsed / 3600:.1f}h elapsed, ~{eta:.1f}h remaining)")

        modified_trades = []
        for name in core_strats:
            strat_trades = base_trades[name]
            if not strat_trades:
                continue
            nt = len(strat_trades)

            # 1) Bootstrap: resample with replacement
            indices = rng.integers(0, nt, size=nt)
            resampled = [deepcopy(strat_trades[idx]) for idx in indices]

            # 2) Random drop: remove 20% of trades
            keep_mask = rng.random(len(resampled)) > 0.20
            resampled = [t for t, keep in zip(resampled, keep_mask) if keep]

            # 3) Slippage injection: N(0, 0.15) noise to PnL
            for t in resampled:
                t['pnl'] += rng.normal(0, 0.15)

            modified_trades.extend(resampled)

        daily = trades_to_daily_series(modified_trades)
        arr = daily.values
        mc_results['sharpes'].append(sharpe(arr))
        mc_results['max_dds'].append(max_dd(arr))
        mc_results['total_pnls'].append(float(sum(t['pnl'] for t in modified_trades)))
        mc_results['worst_months'].append(worst_month(daily))
        mc_results['annual_returns'].append(annual_return(daily))

        if (iteration + 1) % MC_CHECKPOINT_INTERVAL == 0:
            checkpoint['part_a_iters'] = iteration + 1
            checkpoint['part_a_results'] = mc_results
            save_checkpoint(checkpoint)
            print(f"  [Checkpoint] Saved at iteration {iteration + 1}")

    sharpes = np.array(mc_results['sharpes'])
    max_dds = np.array(mc_results['max_dds'])
    total_pnls = np.array(mc_results['total_pnls'])
    worst_ms = np.array(mc_results['worst_months'])
    ann_rets = np.array(mc_results['annual_returns'])

    pctiles = [5, 25, 50, 75, 95]
    summary = {
        'base_sharpe': sharpe(base_arr),
        'base_max_dd': max_dd(base_arr),
        'base_total_pnl': float(sum(t['pnl'] for t in base_all)),
        'mc_iterations': MC_ITERATIONS,
        'sharpe_95ci': [float(np.percentile(sharpes, 2.5)), float(np.percentile(sharpes, 97.5))],
        'max_dd_95ci': [float(np.percentile(max_dds, 2.5)), float(np.percentile(max_dds, 97.5))],
        'annual_return_95ci': [float(np.percentile(ann_rets, 2.5)), float(np.percentile(ann_rets, 97.5))],
        'prob_dd_over_1000': float(np.mean(max_dds > 1000)),
        'prob_worst_month_under_neg500': float(np.mean(worst_ms < -500)),
        'sharpe_percentiles': {str(p): float(np.percentile(sharpes, p)) for p in pctiles},
        'max_dd_percentiles': {str(p): float(np.percentile(max_dds, p)) for p in pctiles},
        'total_pnl_percentiles': {str(p): float(np.percentile(total_pnls, p)) for p in pctiles},
        'worst_month_percentiles': {str(p): float(np.percentile(worst_ms, p)) for p in pctiles},
        'annual_return_percentiles': {str(p): float(np.percentile(ann_rets, p)) for p in pctiles},
        'runtime_hours': (time.time() - t0) / 3600,
    }

    with open(OUTPUT_DIR / "part_a_monte_carlo.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'─' * 60}")
    print("PART A RESULTS:")
    print(f"  Sharpe 95% CI: [{summary['sharpe_95ci'][0]:.2f}, {summary['sharpe_95ci'][1]:.2f}]")
    print(f"  MaxDD 95% CI:  [${summary['max_dd_95ci'][0]:.0f}, ${summary['max_dd_95ci'][1]:.0f}]")
    print(f"  Annual Return 95% CI: [${summary['annual_return_95ci'][0]:.0f}, ${summary['annual_return_95ci'][1]:.0f}]")
    print(f"  P(MaxDD > $1000): {summary['prob_dd_over_1000']:.1%}")
    print(f"  P(WorstMonth < -$500): {summary['prob_worst_month_under_neg500']:.1%}")
    print(f"  Sharpe percentiles: {summary['sharpe_percentiles']}")
    print(f"  Runtime: {summary['runtime_hours']:.1f}h")
    print(f"{'─' * 60}")

    checkpoint['part_a_complete'] = True
    checkpoint['part_a_summary'] = summary
    save_checkpoint(checkpoint)


# ─── PART B: Walk-Forward Parameter Optimization ────────────────────
WF_TRAIN_YEARS = 2.0
WF_TEST_MONTHS = 6
WF_STEP_MONTHS = 6

PARAM_GRIDS = {
    'PSAR': {
        'sl_atr': [3.5, 4.0, 4.5, 5.0, 5.5],
        'tp_atr': [12, 14, 16, 18],
        'maxloss_cap': [3, 5, 8, 10],
    },
    'TSMOM': {
        'fast': [400, 480, 560],
        'slow': [640, 720, 800],
        'sl_atr': [3.5, 4.5, 5.5],
        'tp_atr': [5, 6, 7],
    },
    'SESS_BO': {
        'lookback': [3, 4, 5],
        'sl_atr': [3.5, 4.5, 5.5],
        'tp_atr': [3, 4, 5],
        'session_hour': [12, 13],
    },
    'L8_MAX': {
        'maxloss_cap': [25, 30, 35, 40, 50],
    },
}

R89_DEFAULTS = {
    'PSAR': {'sl_atr': 4.5, 'tp_atr': 16.0},
    'TSMOM': {'fast': 480, 'slow': 720, 'sl_atr': 4.5, 'tp_atr': 6.0},
    'SESS_BO': {'lookback': 4, 'sl_atr': 4.5, 'tp_atr': 4.0, 'session_hour': 12},
    'L8_MAX': {'maxloss_cap': 35},
}


def generate_wf_windows(start='2015-01-01', end='2026-05-01',
                         train_years=2.0, test_months=6, step_months=6):
    """Generate rolling walk-forward windows."""
    windows = []
    train_start = pd.Timestamp(start)
    data_end = pd.Timestamp(end)
    while True:
        train_end = train_start + pd.DateOffset(years=int(train_years),
                                                months=int((train_years % 1) * 12))
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_end > data_end:
            break
        windows.append({
            'train_start': str(train_start.date()),
            'train_end': str(train_end.date()),
            'test_start': str(test_start.date()),
            'test_end': str(test_end.date()),
        })
        train_start += pd.DateOffset(months=step_months)
    return windows


def grid_search_strategy(name, h1_df, data_bundle, spread, lot, cap, param_grid):
    """Run full grid for a strategy, return list of (params, sharpe)."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    results = []
    for combo in product(*values):
        params = dict(zip(keys, combo))
        try:
            trades = run_strategy_with_params(name, data_bundle, h1_df, spread, lot, cap, params)
            daily = trades_to_daily_series(trades)
            sh = sharpe(daily.values)
            results.append((params, sh, len(trades)))
        except Exception as e:
            results.append((params, 0.0, 0))
    return results


def run_part_b(checkpoint):
    print("\n" + "=" * 80)
    print("PART B: Walk-Forward Parameter Optimization")
    print("=" * 80)
    t0 = time.time()

    windows = generate_wf_windows()
    print(f"\n[B] Generated {len(windows)} walk-forward windows")
    for i, w in enumerate(windows):
        print(f"  W{i:02d}: train {w['train_start']}->{w['train_end']}, "
              f"test {w['test_start']}->{w['test_end']}")

    completed_windows = checkpoint.get('part_b_completed_windows', 0)
    wf_results = checkpoint.get('part_b_results', [])

    strats_to_optimize = ['PSAR', 'TSMOM', 'SESS_BO', 'L8_MAX']

    print(f"\n[B] Starting from window {completed_windows}...")
    for w_idx in range(completed_windows, len(windows)):
        w = windows[w_idx]
        w_t0 = time.time()
        print(f"\n{'─' * 60}")
        print(f"[B] Window {w_idx}/{len(windows)}: "
              f"train {w['train_start']}->{w['train_end']}, "
              f"test {w['test_start']}->{w['test_end']}")

        train_bundle = DataBundle.load_default(start=w['train_start'], end=w['train_end'])
        test_bundle = DataBundle.load_default(start=w['test_start'], end=w['test_end'])
        train_h1 = train_bundle.h1_df
        test_h1 = test_bundle.h1_df

        window_result = {
            'window_idx': w_idx,
            'window': w,
            'strategies': {},
        }

        for name in strats_to_optimize:
            lot = R89_LOTS[name]
            cap = CAPS[name]
            grid = PARAM_GRIDS[name]
            defaults = R89_DEFAULTS[name]

            n_combos = 1
            for v in grid.values():
                n_combos *= len(v)
            print(f"  [{name}] Grid search: {n_combos} combos...")

            grid_results = grid_search_strategy(name, train_h1, train_bundle,
                                                SPREAD, lot, cap, grid)
            grid_results.sort(key=lambda x: x[1], reverse=True)
            best_params, best_train_sharpe, best_n = grid_results[0]
            print(f"  [{name}] Best train params: {best_params} -> Sharpe={best_train_sharpe:.2f}")

            # OOS with best params
            best_oos_trades = run_strategy_with_params(
                name, test_bundle, test_h1, SPREAD, lot, cap, best_params)
            best_oos_daily = trades_to_daily_series(best_oos_trades)
            best_oos_sharpe = sharpe(best_oos_daily.values)

            # OOS with fixed (R89 default) params
            fixed_oos_trades = run_strategy_with_params(
                name, test_bundle, test_h1, SPREAD, lot, cap, defaults)
            fixed_oos_daily = trades_to_daily_series(fixed_oos_trades)
            fixed_oos_sharpe = sharpe(fixed_oos_daily.values)

            window_result['strategies'][name] = {
                'best_params': {k: (float(v) if isinstance(v, (int, float)) else v)
                                for k, v in best_params.items()},
                'best_train_sharpe': best_train_sharpe,
                'best_oos_sharpe': best_oos_sharpe,
                'best_oos_trades': len(best_oos_trades),
                'fixed_oos_sharpe': fixed_oos_sharpe,
                'fixed_oos_trades': len(fixed_oos_trades),
                'top3_params': [
                    {'params': {k: float(v) if isinstance(v, (int, float)) else v
                                for k, v in r[0].items()},
                     'sharpe': r[1]}
                    for r in grid_results[:3]
                ],
            }
            print(f"  [{name}] OOS: adaptive={best_oos_sharpe:.2f}, "
                  f"fixed={fixed_oos_sharpe:.2f}")

        window_result['runtime_min'] = (time.time() - w_t0) / 60
        wf_results.append(window_result)

        checkpoint['part_b_completed_windows'] = w_idx + 1
        checkpoint['part_b_results'] = wf_results
        save_checkpoint(checkpoint)
        print(f"  [Checkpoint] Window {w_idx} saved ({window_result['runtime_min']:.1f}min)")

    # Summary
    summary = {
        'n_windows': len(windows),
        'per_strategy': {},
    }
    for name in strats_to_optimize:
        adaptive_sharpes = [w['strategies'][name]['best_oos_sharpe'] for w in wf_results
                           if name in w.get('strategies', {})]
        fixed_sharpes = [w['strategies'][name]['fixed_oos_sharpe'] for w in wf_results
                        if name in w.get('strategies', {})]

        all_best_params = [w['strategies'][name]['best_params'] for w in wf_results
                          if name in w.get('strategies', {})]
        param_keys = list(PARAM_GRIDS[name].keys())
        param_stability = {}
        for pk in param_keys:
            vals = [p[pk] for p in all_best_params if pk in p]
            param_stability[pk] = {
                'values': vals,
                'std': float(np.std(vals)) if vals else 0,
                'unique_count': len(set(str(v) for v in vals)),
            }

        summary['per_strategy'][name] = {
            'adaptive_oos_sharpe_mean': float(np.mean(adaptive_sharpes)) if adaptive_sharpes else 0,
            'adaptive_oos_sharpe_std': float(np.std(adaptive_sharpes)) if adaptive_sharpes else 0,
            'fixed_oos_sharpe_mean': float(np.mean(fixed_sharpes)) if fixed_sharpes else 0,
            'fixed_oos_sharpe_std': float(np.std(fixed_sharpes)) if fixed_sharpes else 0,
            'adaptive_wins': sum(1 for a, f in zip(adaptive_sharpes, fixed_sharpes) if a > f),
            'total_windows': len(adaptive_sharpes),
            'param_stability': param_stability,
        }
        print(f"\n  {name}: Adaptive OOS Sharpe = "
              f"{summary['per_strategy'][name]['adaptive_oos_sharpe_mean']:.2f} "
              f"+/- {summary['per_strategy'][name]['adaptive_oos_sharpe_std']:.2f}")
        print(f"  {name}: Fixed OOS Sharpe = "
              f"{summary['per_strategy'][name]['fixed_oos_sharpe_mean']:.2f} "
              f"+/- {summary['per_strategy'][name]['fixed_oos_sharpe_std']:.2f}")
        print(f"  {name}: Adaptive wins {summary['per_strategy'][name]['adaptive_wins']}/"
              f"{summary['per_strategy'][name]['total_windows']} windows")

    summary['runtime_hours'] = (time.time() - t0) / 3600

    with open(OUTPUT_DIR / "part_b_walk_forward.json", 'w') as f:
        json.dump(summary, f, indent=2)
    with open(OUTPUT_DIR / "part_b_window_details.json", 'w') as f:
        json.dump(wf_results, f, indent=2, default=str)

    print(f"\n[B] Total runtime: {summary['runtime_hours']:.1f}h")

    checkpoint['part_b_complete'] = True
    checkpoint['part_b_summary'] = summary
    save_checkpoint(checkpoint)


# ─── PART C: Exhaustive Strategy Combination ────────────────────────
COMBO_BATCH_SIZE = 16
LOT_RANGE = [round(x * 0.01, 2) for x in range(1, 11)]


def run_part_c(checkpoint):
    print("\n" + "=" * 80)
    print("PART C: Exhaustive Strategy Combination (2^7 = 128 combos)")
    print("=" * 80)
    t0 = time.time()

    data_bundle = DataBundle.load_default()
    h1_df = data_bundle.h1_df

    all_names = ALL_STRATEGY_NAMES
    all_caps = {**CAPS, 'DONCH50': 0, 'OVGAP': 0, 'BBSQZ': 0}
    default_lots = {**R89_LOTS, 'DONCH50': UNIT_LOT, 'OVGAP': UNIT_LOT, 'BBSQZ': UNIT_LOT}

    print("\n[C] Running base backtests for all 7 strategies...")
    base_trades = {}
    strat_sharpes = {}
    for name in all_names:
        lot = UNIT_LOT
        cap = all_caps[name]
        trades = run_strategy(name, data_bundle, h1_df, SPREAD, lot, cap)
        base_trades[name] = trades
        daily = trades_to_daily_series(trades)
        sh = sharpe(daily.values)
        strat_sharpes[name] = sh
        print(f"  {name}: {len(trades)} trades, Sharpe={sh:.2f} (unit lot)")

    # Generate all 2^7 on/off combinations (skip all-off)
    n_strats = len(all_names)
    all_combos = []
    for mask in range(1, 2 ** n_strats):
        active = [all_names[j] for j in range(n_strats) if mask & (1 << j)]
        all_combos.append(active)
    print(f"\n[C] Total combinations: {len(all_combos)} (excluding all-off)")

    completed_combos = checkpoint.get('part_c_completed_combos', 0)
    combo_results = checkpoint.get('part_c_results', [])

    print(f"[C] Starting from combo {completed_combos}...")
    for batch_start in range(completed_combos, len(all_combos), COMBO_BATCH_SIZE):
        batch_end = min(batch_start + COMBO_BATCH_SIZE, len(all_combos))
        batch = all_combos[batch_start:batch_end]

        for c_idx, active_strats in enumerate(batch, start=batch_start):
            combo_key = "+".join(active_strats)

            # Allocate lots proportional to individual Sharpe, then scale to MaxDD <= $1000
            active_sharpes = {s: max(strat_sharpes[s], 0.01) for s in active_strats}
            total_sharpe = sum(active_sharpes.values())
            raw_lots = {s: active_sharpes[s] / total_sharpe * 0.5 for s in active_strats}

            # Build portfolio with proportional lots
            combo_trades = []
            for s in active_strats:
                lot = max(0.01, round(raw_lots[s], 2))
                scaled_trades = []
                for t in base_trades[s]:
                    st = deepcopy(t)
                    st['pnl'] = t['pnl'] * (lot / UNIT_LOT)
                    scaled_trades.append(st)
                combo_trades.extend(scaled_trades)

            daily = trades_to_daily_series(combo_trades)
            arr = daily.values
            dd = max_dd(arr)

            # Scale down if MaxDD > $1000
            if dd > 1000 and dd > 0:
                scale = 1000 / dd * 0.95
                for t in combo_trades:
                    t['pnl'] *= scale
                raw_lots = {s: max(0.01, round(raw_lots[s] * scale, 2)) for s in active_strats}
                daily = trades_to_daily_series(combo_trades)
                arr = daily.values
                dd = max_dd(arr)

            sh = sharpe(arr)
            total_pnl = float(sum(t['pnl'] for t in combo_trades))
            cv = cvar99(arr)
            wm = worst_month(daily)

            combo_results.append({
                'combo_idx': c_idx,
                'active': active_strats,
                'combo_key': combo_key,
                'lots': {s: round(raw_lots.get(s, 0.01), 2) for s in active_strats},
                'sharpe': sh,
                'max_dd': dd,
                'total_pnl': total_pnl,
                'cvar99': cv,
                'worst_month': wm,
                'n_trades': len(combo_trades),
            })

        checkpoint['part_c_completed_combos'] = batch_end
        checkpoint['part_c_results'] = combo_results
        save_checkpoint(checkpoint)
        elapsed = time.time() - t0
        print(f"  [Checkpoint] Combos {batch_start}-{batch_end} done "
              f"({elapsed / 3600:.1f}h elapsed)")

    # Sort by Sharpe, take top 20 for K-Fold validation
    combo_results.sort(key=lambda x: x['sharpe'], reverse=True)
    top20 = combo_results[:20]

    print(f"\n[C] Top 20 combos by Sharpe:")
    for i, cr in enumerate(top20):
        print(f"  #{i + 1}: {cr['combo_key']} | Sharpe={cr['sharpe']:.2f} | "
              f"MaxDD=${cr['max_dd']:.0f} | PnL=${cr['total_pnl']:.0f}")

    # K-Fold validation on top 20
    print(f"\n[C] Running K-Fold validation on top 20 combos...")
    kfold_results = checkpoint.get('part_c_kfold_results', [])
    kfold_done = checkpoint.get('part_c_kfold_done', 0)

    for rank, combo in enumerate(top20[kfold_done:], start=kfold_done):
        active_strats = combo['active']
        lots = combo['lots']
        fold_sharpes = []

        for fold_name, fold_start, fold_end in FOLDS:
            try:
                fold_bundle = DataBundle.load_default(start=fold_start, end=fold_end)
                fold_h1 = fold_bundle.h1_df
            except Exception:
                fold_sharpes.append(0.0)
                continue

            fold_trades = []
            for s in active_strats:
                lot = lots.get(s, UNIT_LOT)
                cap = all_caps.get(s, 0)
                trades = run_strategy(s, fold_bundle, fold_h1, SPREAD, lot, cap)
                fold_trades.extend(trades)

            fold_daily = trades_to_daily_series(fold_trades)
            fold_sharpes.append(sharpe(fold_daily.values))

        kfold_results.append({
            'rank': rank + 1,
            'combo_key': combo['combo_key'],
            'active': active_strats,
            'lots': lots,
            'full_sharpe': combo['sharpe'],
            'fold_sharpes': fold_sharpes,
            'mean_fold_sharpe': float(np.mean(fold_sharpes)),
            'std_fold_sharpe': float(np.std(fold_sharpes)),
            'min_fold_sharpe': float(np.min(fold_sharpes)),
        })

        checkpoint['part_c_kfold_done'] = rank + 1
        checkpoint['part_c_kfold_results'] = kfold_results
        save_checkpoint(checkpoint)
        print(f"  K-Fold #{rank + 1}: {combo['combo_key']} -> "
              f"mean={kfold_results[-1]['mean_fold_sharpe']:.2f} "
              f"+/- {kfold_results[-1]['std_fold_sharpe']:.2f}")

    # Compare best found vs current 4-strategy
    current_4 = None
    for cr in combo_results:
        if set(cr['active']) == {'L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO'}:
            current_4 = cr
            break

    kfold_results.sort(key=lambda x: x['mean_fold_sharpe'], reverse=True)
    best_combo = kfold_results[0] if kfold_results else None

    summary = {
        'total_combos_tested': len(combo_results),
        'top20_sharpes': [cr['sharpe'] for cr in top20],
        'top20_combos': [cr['combo_key'] for cr in top20],
        'kfold_ranking': [
            {'rank': kf['rank'], 'combo': kf['combo_key'],
             'mean_sharpe': kf['mean_fold_sharpe'],
             'std_sharpe': kf['std_fold_sharpe'],
             'min_sharpe': kf['min_fold_sharpe']}
            for kf in kfold_results
        ],
        'best_combo': {
            'combo': best_combo['combo_key'] if best_combo else 'N/A',
            'lots': best_combo['lots'] if best_combo else {},
            'mean_fold_sharpe': best_combo['mean_fold_sharpe'] if best_combo else 0,
        },
        'current_4_strategy': {
            'sharpe': current_4['sharpe'] if current_4 else 0,
            'max_dd': current_4['max_dd'] if current_4 else 0,
        },
        'runtime_hours': (time.time() - t0) / 3600,
    }

    with open(OUTPUT_DIR / "part_c_combinations.json", 'w') as f:
        json.dump(summary, f, indent=2)
    with open(OUTPUT_DIR / "part_c_all_combos.json", 'w') as f:
        json.dump(combo_results[:50], f, indent=2, default=str)
    with open(OUTPUT_DIR / "part_c_kfold.json", 'w') as f:
        json.dump(kfold_results, f, indent=2, default=str)

    print(f"\n{'─' * 60}")
    print("PART C RESULTS:")
    if best_combo:
        print(f"  Best combo: {best_combo['combo_key']}")
        print(f"  Mean fold Sharpe: {best_combo['mean_fold_sharpe']:.2f}")
    if current_4:
        print(f"  Current 4-strat Sharpe: {current_4['sharpe']:.2f}")
    print(f"  Runtime: {summary['runtime_hours']:.1f}h")
    print(f"{'─' * 60}")

    checkpoint['part_c_complete'] = True
    checkpoint['part_c_summary'] = summary
    save_checkpoint(checkpoint)


# ─── Final summary ──────────────────────────────────────────────────
def save_final_results(checkpoint):
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    final = {
        'experiment': 'R101 Mega Experiment',
        'completed_at': datetime.now().isoformat(),
        'part_a': checkpoint.get('part_a_summary', {}),
        'part_b': checkpoint.get('part_b_summary', {}),
        'part_c': checkpoint.get('part_c_summary', {}),
    }

    # Print key findings
    pa = final.get('part_a', {})
    if pa:
        print(f"\n  Part A (Monte Carlo {pa.get('mc_iterations', 0)} iters):")
        ci = pa.get('sharpe_95ci', [0, 0])
        print(f"    Sharpe 95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
        print(f"    P(MaxDD > $1000): {pa.get('prob_dd_over_1000', 0):.1%}")
        print(f"    P(Worst month < -$500): {pa.get('prob_worst_month_under_neg500', 0):.1%}")

    pb = final.get('part_b', {})
    if pb:
        print(f"\n  Part B (Walk-Forward):")
        for name, data in pb.get('per_strategy', {}).items():
            print(f"    {name}: Adaptive={data.get('adaptive_oos_sharpe_mean', 0):.2f}, "
                  f"Fixed={data.get('fixed_oos_sharpe_mean', 0):.2f}, "
                  f"Wins={data.get('adaptive_wins', 0)}/{data.get('total_windows', 0)}")

    pc = final.get('part_c', {})
    if pc:
        print(f"\n  Part C (Combinations):")
        bc = pc.get('best_combo', {})
        print(f"    Best combo: {bc.get('combo', 'N/A')}")
        print(f"    Mean fold Sharpe: {bc.get('mean_fold_sharpe', 0):.2f}")
        c4 = pc.get('current_4_strategy', {})
        print(f"    Current 4-strat Sharpe: {c4.get('sharpe', 0):.2f}")

    total_hours = sum(
        final.get(p, {}).get('runtime_hours', 0) for p in ['part_a', 'part_b', 'part_c']
    )
    final['total_runtime_hours'] = total_hours
    print(f"\n  Total runtime: {total_hours:.1f}h")

    with open(OUTPUT_DIR / "r101_final_summary.json", 'w') as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_DIR}")


# ─── Entry point ─────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("R101 MEGA EXPERIMENT")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Checkpoint: {CHECKPOINT_FILE}")
    print("=" * 80)

    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"Loaded checkpoint with keys: {list(checkpoint.keys())}")

    if checkpoint.get('part_a_complete'):
        print("\nPart A already complete, skipping...")
    else:
        run_part_a(checkpoint)

    if checkpoint.get('part_b_complete'):
        print("\nPart B already complete, skipping...")
    else:
        run_part_b(checkpoint)

    if checkpoint.get('part_c_complete'):
        print("\nPart C already complete, skipping...")
    else:
        run_part_c(checkpoint)

    save_final_results(checkpoint)
    print(f"\nR101 COMPLETE at {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
