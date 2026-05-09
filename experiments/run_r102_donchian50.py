#!/usr/bin/env python3
"""
R102 — Donchian50 Strategy Independent Backtest
=================================================
R101 Part C showed Donchian50 appears in multiple top combos.
This experiment does a thorough standalone evaluation:

  Phase 1: Baseline backtest with default params (channel=50)
  Phase 2: Parameter grid search (channel, SL, TP, trail, max_hold)
  Phase 3: K-Fold validation on top 5 param sets
  Phase 4: Portfolio integration test (current 4-strat + Donchian50)
  Phase 5: Lot optimization for 5-strategy portfolio
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import DataBundle, load_m15, load_h1_aligned, H1_CSV_PATH
from backtest.runner import run_variant, LIVE_PARITY_KWARGS

OUTPUT_DIR = Path("results/r102_donchian50")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35, 'DONCH50': 0}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.09, 'SESS_BO': 0.09}

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]

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
    sl = atr * sl_atr
    tp = atr * tp_atr
    bars = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_now = (cl - pos['entry'] - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if lo <= pos['entry'] - sl:
            pnl = -sl * lot * pv
            return _mk(pos, pos['entry'] - sl, times[i], "SL", i, pnl)
        if hi >= pos['entry'] + tp:
            pnl = tp * lot * pv
            return _mk(pos, pos['entry'] + tp, times[i], "TP", i, pnl)
        extreme = pos.get('extreme', pos['entry'])
        extreme = max(extreme, hi)
        pos['extreme'] = extreme
        act_dist = atr * trail_act
        if extreme - pos['entry'] >= act_dist:
            trail_price = extreme - atr * trail_dist
            if cl <= trail_price:
                pnl = (trail_price - pos['entry'] - spread) * lot * pv
                return _mk(pos, trail_price, times[i], "Trail", i, pnl)
        if bars >= max_hold:
            pnl = (cl - pos['entry'] - spread) * lot * pv
            return _mk(pos, cl, times[i], "TimeExit", i, pnl)
    else:
        pnl_now = (pos['entry'] - cl - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if hi >= pos['entry'] + sl:
            pnl = -sl * lot * pv
            return _mk(pos, pos['entry'] + sl, times[i], "SL", i, pnl)
        if lo <= pos['entry'] - tp:
            pnl = tp * lot * pv
            return _mk(pos, pos['entry'] - tp, times[i], "TP", i, pnl)
        extreme = pos.get('extreme', pos['entry'])
        extreme = min(extreme, lo)
        pos['extreme'] = extreme
        act_dist = atr * trail_act
        if pos['entry'] - extreme >= act_dist:
            trail_price = extreme + atr * trail_dist
            if cl >= trail_price:
                pnl = (pos['entry'] - trail_price - spread) * lot * pv
                return _mk(pos, trail_price, times[i], "Trail", i, pnl)
        if bars >= max_hold:
            pnl = (pos['entry'] - cl - spread) * lot * pv
            return _mk(pos, cl, times[i], "TimeExit", i, pnl)
    return None


# ═══════════════════════════════════════════════════════════════
# Donchian50 backtest
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# Other strategy backtests (for portfolio integration)
# ═══════════════════════════════════════════════════════════════

def add_psar(df, af_step=0.01, af_max=0.05):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    n = len(df)
    psar = np.empty(n); psar[:] = np.nan
    af = af_step; rising = True
    ep = h[0]; psar[0] = l[0]
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
            sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04,
            max_hold=20, af_step=0.01, af_max=0.05, min_atr=0.1):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    add_psar(df, af_step, af_max)
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
        if np.isnan(atr[i]) or atr[i] < min_atr: continue
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
# Metrics
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


def win_rate(trades):
    if not trades: return 0.0
    return sum(1 for t in trades if t['pnl'] > 0) / len(trades) * 100


def avg_pnl(trades):
    if not trades: return 0.0
    return sum(t['pnl'] for t in trades) / len(trades)


def metrics(trades):
    daily = trades_to_daily_series(trades)
    return {
        'n_trades': len(trades),
        'sharpe': round(sharpe(daily.values), 3) if len(daily) > 0 else 0,
        'pnl': round(sum(t['pnl'] for t in trades), 2),
        'max_dd': round(max_dd(daily.values), 2) if len(daily) > 0 else 0,
        'wr': round(win_rate(trades), 1),
        'avg_pnl': round(avg_pnl(trades), 3),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  R102 — Donchian50 Strategy Independent Backtest")
    print("=" * 80)

    print("\n  Loading data...")
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    bundle = DataBundle.load_custom()
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0].date()} ~ {h1_df.index[-1].date()})")

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Baseline backtest
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 1: Baseline Donchian50 (channel=50, SL=3xATR, TP=4xATR)")
    print("=" * 60)

    trades_base = bt_donchian50(h1_df, SPREAD, UNIT_LOT)
    m = metrics(trades_base)
    print(f"  Results: {m['n_trades']} trades, Sharpe={m['sharpe']}, PnL=${m['pnl']}, "
          f"MaxDD=${m['max_dd']}, WR={m['wr']}%, AvgPnL=${m['avg_pnl']}")

    # Year-by-year breakdown
    print("\n  Year-by-year:")
    for year in range(2015, 2027):
        year_trades = [t for t in trades_base
                       if pd.Timestamp(t['exit_time']).year == year]
        if year_trades:
            ym = metrics(year_trades)
            print(f"    {year}: {ym['n_trades']:4d} trades, Sharpe={ym['sharpe']:6.2f}, "
                  f"PnL=${ym['pnl']:8.2f}, WR={ym['wr']:.0f}%")

    # Exit reason distribution
    reasons = {}
    for t in trades_base:
        r = t.get('reason', 'Unknown')
        reasons[r] = reasons.get(r, 0) + 1
    print("\n  Exit reasons:")
    for r, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {r}: {cnt} ({cnt/len(trades_base)*100:.1f}%)")

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Parameter grid search
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 2: Parameter Grid Search")
    print("=" * 60)

    param_grid = {
        'channel': [30, 40, 50, 60, 80],
        'sl_atr': [2.0, 3.0, 4.0, 5.0],
        'tp_atr': [3.0, 4.0, 5.0, 6.0, 8.0],
        'max_hold': [10, 15, 20, 30],
    }
    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    print(f"  Total combinations: {total_combos}")

    grid_results = []
    tested = 0
    for ch, sl, tp, mh in product(param_grid['channel'], param_grid['sl_atr'],
                                   param_grid['tp_atr'], param_grid['max_hold']):
        trades = bt_donchian50(h1_df, SPREAD, UNIT_LOT, channel=ch,
                               sl_atr=sl, tp_atr=tp, max_hold=mh)
        m = metrics(trades)
        grid_results.append({
            'params': {'channel': ch, 'sl_atr': sl, 'tp_atr': tp, 'max_hold': mh},
            **m
        })
        tested += 1
        if tested % 50 == 0:
            print(f"    {tested}/{total_combos}...")

    grid_results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  Top 10 parameter sets:")
    for i, g in enumerate(grid_results[:10]):
        p = g['params']
        print(f"    #{i+1}: ch={p['channel']}, SL={p['sl_atr']}x, TP={p['tp_atr']}x, "
              f"MH={p['max_hold']} -> Sharpe={g['sharpe']}, PnL=${g['pnl']}, "
              f"Trades={g['n_trades']}, WR={g['wr']}%")

    # Cap sweep for top params
    print("\n  MaxLoss Cap sweep (top params):")
    best_params = grid_results[0]['params']
    cap_results = []
    for cap in [0, 5, 10, 15, 20, 25, 30, 35, 50]:
        trades = bt_donchian50(h1_df, SPREAD, UNIT_LOT, maxloss_cap=cap, **best_params)
        m = metrics(trades)
        cap_results.append({'cap': cap, **m})
        print(f"    Cap=${cap}: Sharpe={m['sharpe']}, PnL=${m['pnl']}, MaxDD=${m['max_dd']}")

    # ═══════════════════════════════════════════════════════════
    # Phase 3: K-Fold validation on top 5
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 3: K-Fold Validation (6 folds)")
    print("=" * 60)

    kfold_results = {}
    for rank, g in enumerate(grid_results[:5]):
        p = g['params']
        label = f"#{rank+1} ch={p['channel']}/SL={p['sl_atr']}/TP={p['tp_atr']}/MH={p['max_hold']}"
        fold_sharpes = []
        for fname, start, end in FOLDS:
            fold_h1 = h1_df[(h1_df.index >= start) & (h1_df.index < end)]
            if len(fold_h1) < 100:
                fold_sharpes.append(0.0)
                continue
            trades = bt_donchian50(fold_h1, SPREAD, UNIT_LOT, **p)
            m = metrics(trades)
            fold_sharpes.append(m['sharpe'])
        positive = sum(1 for s in fold_sharpes if s > 0)
        mean_sh = np.mean(fold_sharpes)
        passed = positive >= 4
        kfold_results[label] = {
            'params': p,
            'fold_sharpes': [round(s, 3) for s in fold_sharpes],
            'positive_folds': positive,
            'mean_sharpe': round(mean_sh, 3),
            'pass_4of6': passed,
        }
        status = "PASS" if passed else "FAIL"
        print(f"  {label}: folds={[f'{s:.2f}' for s in fold_sharpes]} -> "
              f"{positive}/6 [{status}] mean={mean_sh:.3f}")

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Portfolio integration
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 4: Portfolio Integration (4-strat + Donchian50)")
    print("=" * 60)

    best_validated = None
    for label, info in kfold_results.items():
        if info['pass_4of6']:
            if best_validated is None or info['mean_sharpe'] > best_validated[1]['mean_sharpe']:
                best_validated = (label, info)

    if best_validated is None:
        print("  No Donchian50 config passed K-Fold. Using baseline params.")
        donch_params = {'channel': 50, 'sl_atr': 3.0, 'tp_atr': 4.0, 'max_hold': 20}
    else:
        donch_params = best_validated[1]['params']
        print(f"  Best validated: {best_validated[0]}")

    # Run all 5 strategies
    print("  Running all strategies...")
    psar_trades = bt_psar(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['PSAR'])
    tsmom_trades = bt_tsmom(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['TSMOM'])
    sess_trades = bt_sess_bo(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS['SESS_BO'])
    l8_trades = bt_l8_max(bundle, SPREAD, UNIT_LOT, maxloss_cap=CAPS['L8_MAX'])
    donch_trades = bt_donchian50(h1_df, SPREAD, UNIT_LOT, **donch_params)

    strat_trades = {
        'PSAR': psar_trades, 'TSMOM': tsmom_trades,
        'SESS_BO': sess_trades, 'L8_MAX': l8_trades,
        'DONCH50': donch_trades,
    }

    print("\n  Individual strategy performance (unit lot):")
    for name, trades in strat_trades.items():
        m = metrics(trades)
        print(f"    {name:10s}: {m['n_trades']:5d} trades, Sharpe={m['sharpe']:6.3f}, "
              f"PnL=${m['pnl']:10.2f}, MaxDD=${m['max_dd']:7.2f}, WR={m['wr']:.1f}%")

    # Current 4-strategy portfolio
    def build_portfolio(trade_dict, lots):
        all_daily = {}
        for name, trades in trade_dict.items():
            lot = lots.get(name, UNIT_LOT)
            scale = lot / UNIT_LOT
            for t in trades:
                d = pd.Timestamp(t['exit_time']).date()
                all_daily[d] = all_daily.get(d, 0) + t['pnl'] * scale
        dates = sorted(all_daily.keys())
        return pd.Series([all_daily[d] for d in dates], index=pd.DatetimeIndex(dates))

    current_4 = {k: v for k, v in strat_trades.items() if k != 'DONCH50'}
    port_4 = build_portfolio(current_4, R89_LOTS)
    sh_4 = sharpe(port_4.values)
    dd_4 = max_dd(port_4.values)
    pnl_4 = port_4.sum()

    print(f"\n  Current 4-strategy portfolio (R89 lots):")
    print(f"    Sharpe={sh_4:.3f}, PnL=${pnl_4:.2f}, MaxDD=${dd_4:.2f}")

    # ═══════════════════════════════════════════════════════════
    # Phase 5: Lot optimization for 5-strategy portfolio
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 5: 5-Strategy Lot Optimization")
    print("=" * 60)

    donch_lot_candidates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    lot_results = []
    for dl in donch_lot_candidates:
        lots_5 = {**R89_LOTS, 'DONCH50': dl}
        port_5 = build_portfolio(strat_trades, lots_5)
        sh = sharpe(port_5.values)
        dd = max_dd(port_5.values)
        pnl = port_5.sum()
        lot_results.append({
            'donch_lot': dl, 'sharpe': round(sh, 3),
            'max_dd': round(dd, 2), 'pnl': round(pnl, 2),
        })
        marker = " <-- best" if sh == max(r['sharpe'] for r in lot_results) else ""
        print(f"    DONCH50={dl:.2f}: Sharpe={sh:.3f}, PnL=${pnl:.0f}, MaxDD=${dd:.0f}{marker}")

    best_lot = max(lot_results, key=lambda x: x['sharpe'])
    print(f"\n  Best Donchian50 lot: {best_lot['donch_lot']}")
    print(f"  5-strat Sharpe={best_lot['sharpe']} vs 4-strat Sharpe={sh_4:.3f} "
          f"(delta={best_lot['sharpe']-sh_4:+.3f})")

    # K-Fold on best 5-strategy portfolio
    print("\n  K-Fold on 5-strategy portfolio:")
    best_dl = best_lot['donch_lot']
    lots_5_best = {**R89_LOTS, 'DONCH50': best_dl}
    fold_sharpes_4 = []
    fold_sharpes_5 = []
    for fname, start, end in FOLDS:
        fold_h1 = h1_df[(h1_df.index >= start) & (h1_df.index < end)]
        if len(fold_h1) < 100:
            fold_sharpes_4.append(0.0)
            fold_sharpes_5.append(0.0)
            continue
        # 4-strat fold
        ft = {
            'PSAR': bt_psar(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=CAPS['PSAR']),
            'TSMOM': bt_tsmom(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=CAPS['TSMOM']),
            'SESS_BO': bt_sess_bo(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=CAPS['SESS_BO']),
            'L8_MAX': bt_l8_max(bundle, SPREAD, UNIT_LOT, maxloss_cap=CAPS['L8_MAX']),
        }
        p4 = build_portfolio(ft, R89_LOTS)
        fold_sharpes_4.append(sharpe(p4.values))
        # 5-strat fold
        ft['DONCH50'] = bt_donchian50(fold_h1, SPREAD, UNIT_LOT, **donch_params)
        p5 = build_portfolio(ft, lots_5_best)
        fold_sharpes_5.append(sharpe(p5.values))

    print(f"    4-strat folds: {[f'{s:.2f}' for s in fold_sharpes_4]} mean={np.mean(fold_sharpes_4):.3f}")
    print(f"    5-strat folds: {[f'{s:.2f}' for s in fold_sharpes_5]} mean={np.mean(fold_sharpes_5):.3f}")
    folds_5_better = sum(1 for a, b in zip(fold_sharpes_5, fold_sharpes_4) if a > b)
    print(f"    5-strat wins: {folds_5_better}/6 folds")

    # ═══════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    results = {
        'experiment': 'R102 Donchian50 Independent Backtest',
        'elapsed_s': round(elapsed, 1),
        'baseline': metrics(trades_base),
        'grid_top10': grid_results[:10],
        'cap_sweep': cap_results,
        'kfold': kfold_results,
        'best_validated_params': donch_params,
        'portfolio_4strat': {'sharpe': round(sh_4, 3), 'pnl': round(pnl_4, 2), 'max_dd': round(dd_4, 2)},
        'lot_optimization': lot_results,
        'best_5strat': best_lot,
        'portfolio_kfold': {
            '4strat_folds': [round(s, 3) for s in fold_sharpes_4],
            '5strat_folds': [round(s, 3) for s in fold_sharpes_5],
            '5strat_wins': folds_5_better,
            '4strat_mean': round(np.mean(fold_sharpes_4), 3),
            '5strat_mean': round(np.mean(fold_sharpes_5), 3),
        },
        'recommendation': (
            f"Add DONCH50 at {best_dl} lots" if folds_5_better >= 4
            else "Keep current 4-strategy portfolio"
        ),
    }

    out_file = OUTPUT_DIR / "r102_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"  R102 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    if folds_5_better >= 4:
        print(f"  RECOMMENDATION: Add Donchian50 at {best_dl} lots (5-strat wins {folds_5_better}/6 folds)")
    else:
        print(f"  RECOMMENDATION: Keep current 4-strategy portfolio (5-strat wins only {folds_5_better}/6)")
    print(f"{'='*80}")
    print(f"  Saved: {out_file}")


if __name__ == '__main__':
    main()
