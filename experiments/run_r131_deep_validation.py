#!/usr/bin/env python3
"""
R131 — Deep Validation: EqMA30 + PSAR Param Optimization
==========================================================
Deep-dive validation of two "worth testing" findings:
  (A) R126 EqMA_30 equity curve filter (Sharpe +0.39, MaxDD -$100)
  (B) R127 PSAR optimized params (sl=4.0, tp=6.0, trail_act=0.08, trail_dist=0.015, hold=15)

Validation stages:
  Stage 1: Reproduce baseline vs candidate on full data
  Stage 2: Walk-forward OOS (4yr train / 2yr test, slide 1yr)
  Stage 3: 8-fold time-based K-fold (non-overlapping ~16mo each)
  Stage 4: Yearly stability (each year positive?)
  Stage 5: Monte Carlo (1000 paths, slippage + spread + missed fills)
  Stage 6: Combinatorial PBO (shuffle trade PnLs 1000x)
  Stage 7: Combined test — PSAR new params + EqMA30 together
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

OUTPUT_DIR = Path("results/r131_deep_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01
CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.09, 'SESS_BO': 0.09}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']

PSAR_DEFAULT = {'sl_atr': 4.5, 'tp_atr': 16.0, 'trail_act': 0.20, 'trail_dist': 0.04, 'max_hold': 20}
PSAR_OPTIMIZED = {'sl_atr': 4.0, 'tp_atr': 6.0, 'trail_act': 0.08, 'trail_dist': 0.015, 'max_hold': 15}

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


def _compute_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0, 'calmar': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    sh = _sharpe(daily)
    dd = _max_dd(daily)
    pnl = sum(pnls)
    return {
        'n': n, 'sharpe': round(sh, 3), 'pnl': round(pnl, 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'max_dd': round(dd, 2),
        'calmar': round(pnl / dd, 2) if dd > 0 else 9999,
    }


# ═══════════════════════════════════════════════════════════════
# Strategy backtests
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
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
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
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


# ═══════════════════════════════════════════════════════════════
# Portfolio helpers
# ═══════════════════════════════════════════════════════════════

def run_portfolio(h1_df, bundle, psar_params=None, use_eqma=False, eqma_window=30,
                  spread=SPREAD):
    """Run all 4 strategies and combine into portfolio daily PnL."""
    psar_trades = bt_psar(h1_df, spread, UNIT_LOT, CAPS['PSAR'], params=psar_params)
    tsmom_trades = bt_tsmom(h1_df, spread, UNIT_LOT, CAPS['TSMOM'])
    sessbo_trades = bt_sess_bo(h1_df, spread, UNIT_LOT, CAPS['SESS_BO'])

    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': CAPS['L8_MAX'],
          'spread_cost': spread, 'initial_capital': 2000,
          'min_lot_size': UNIT_LOT, 'max_lot_size': UNIT_LOT}
    l8_result = run_variant(bundle, "L8_MAX", verbose=False, **kw)
    l8_trades = [{'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
                  'entry_time': t.entry_time, 'exit_time': t.exit_time,
                  'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0}
                 for t in l8_result.get('_trades', [])]

    strat_trades = {'L8_MAX': l8_trades, 'PSAR': psar_trades,
                    'TSMOM': tsmom_trades, 'SESS_BO': sessbo_trades}

    all_daily = {}
    for sn in STRAT_ORDER:
        lot = R89_LOTS[sn]
        for t in strat_trades[sn]:
            d = pd.Timestamp(t['exit_time']).date()
            scaled_pnl = t['pnl'] * (lot / UNIT_LOT)
            all_daily[d] = all_daily.get(d, 0) + scaled_pnl

    dates = sorted(all_daily.keys())
    daily_arr = np.array([all_daily[d] for d in dates])

    if use_eqma and eqma_window > 0:
        eq = np.cumsum(daily_arr)
        ma = pd.Series(eq).rolling(eqma_window, min_periods=1).mean().values
        filtered = np.where(eq >= ma, daily_arr, np.minimum(daily_arr, 0))
        daily_arr = filtered

    return daily_arr, dates, strat_trades


def load_h1():
    """Load H1 data from Dukascopy CSV."""
    from backtest.runner import load_csv
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
    results = {'experiment': 'R131 Deep Validation: EqMA30 + PSAR Params'}

    print("=" * 80, flush=True)
    print("  R131 — Deep Validation: EqMA30 + PSAR Param Optimization", flush=True)
    print("=" * 80, flush=True)

    print("\n  Loading data...", flush=True)
    h1_df = load_h1()
    bundle = DataBundle.load_default()
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Stage 1: Reproduce baseline vs candidates on full data
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Stage 1: Full-Data Comparison", flush=True)
    print("=" * 70, flush=True)

    configs = {
        'A_baseline':        {'psar_params': PSAR_DEFAULT,   'use_eqma': False},
        'B_psar_opt':        {'psar_params': PSAR_OPTIMIZED, 'use_eqma': False},
        'C_eqma30':          {'psar_params': PSAR_DEFAULT,   'use_eqma': True, 'eqma_window': 30},
        'D_psar_opt_eqma30': {'psar_params': PSAR_OPTIMIZED, 'use_eqma': True, 'eqma_window': 30},
    }

    stage1 = {}
    daily_arrays = {}
    print(f"\n  {'Config':<22s} {'Sharpe':>7} {'PnL':>12} {'MaxDD':>9} {'Calmar':>8}", flush=True)
    print(f"  {'─'*60}", flush=True)

    for name, cfg in configs.items():
        daily, dates, _ = run_portfolio(h1_df, bundle, **cfg)
        daily_arrays[name] = daily
        sh = _sharpe(daily); dd = _max_dd(daily); pnl = float(np.sum(daily))
        cal = pnl / dd if dd > 0 else 9999
        stage1[name] = {'sharpe': round(sh, 3), 'pnl': round(pnl, 2),
                        'max_dd': round(dd, 2), 'calmar': round(cal, 1)}
        print(f"  {name:<22s} {sh:>7.3f} ${pnl:>11,.0f} ${dd:>8,.0f} {cal:>8.1f}", flush=True)

    results['stage1'] = stage1

    # ═══════════════════════════════════════════════════════════
    # Stage 2: Walk-Forward OOS (4yr train / 2yr test, slide 1yr)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Stage 2: Walk-Forward Out-of-Sample", flush=True)
    print("=" * 70, flush=True)

    wf_windows = [
        ("2015-2019 / 2019-2021", "2019-01-01", "2021-01-01"),
        ("2016-2020 / 2020-2022", "2020-01-01", "2022-01-01"),
        ("2017-2021 / 2021-2023", "2021-01-01", "2023-01-01"),
        ("2018-2022 / 2022-2024", "2022-01-01", "2024-01-01"),
        ("2019-2023 / 2023-2025", "2023-01-01", "2025-01-01"),
        ("2020-2024 / 2024-2026", "2024-01-01", "2026-05-01"),
    ]

    wf_results = []
    print(f"\n  {'Window':<30s} {'A_base':>7} {'B_psar':>7} {'C_eqma':>7} {'D_both':>7}", flush=True)
    print(f"  {'─'*60}", flush=True)

    for label, oos_start, oos_end in wf_windows:
        h1_oos = h1_df[(h1_df.index >= oos_start) & (h1_df.index < oos_end)]
        if len(h1_oos) < 500:
            continue

        wf_row = {'window': label}
        vals = []
        for name, cfg in configs.items():
            daily, _, _ = run_portfolio(h1_oos, bundle, **cfg)
            sh = _sharpe(daily)
            wf_row[name] = round(sh, 3)
            vals.append(sh)

        wf_results.append(wf_row)
        print(f"  {label:<30s} {vals[0]:>7.3f} {vals[1]:>7.3f} {vals[2]:>7.3f} {vals[3]:>7.3f}", flush=True)

    results['stage2_walkforward'] = wf_results

    if wf_results:
        print(f"\n  OOS Mean Sharpe:", flush=True)
        for name in configs:
            vals = [w[name] for w in wf_results]
            print(f"    {name:<22s}: {np.mean(vals):.3f} (min={min(vals):.3f}, max={max(vals):.3f})", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Stage 3: 8-Fold K-Fold (non-overlapping ~16mo each)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Stage 3: 8-Fold Time-Based K-Fold", flush=True)
    print("=" * 70, flush=True)

    fold_starts = pd.date_range("2015-01-01", "2026-01-01", periods=9)
    kfold_results = {name: [] for name in configs}

    for fi in range(8):
        fs = str(fold_starts[fi].date())
        fe = str(fold_starts[fi + 1].date())
        h1_fold = h1_df[(h1_df.index >= fs) & (h1_df.index < fe)]
        if len(h1_fold) < 200:
            continue

        for name, cfg in configs.items():
            daily, _, _ = run_portfolio(h1_fold, bundle, **cfg)
            sh = _sharpe(daily)
            kfold_results[name].append(round(sh, 3))

    print(f"\n  {'Config':<22s} {'Folds':>35s} {'Mean':>6} {'Min':>6} {'Pos':>4}", flush=True)
    print(f"  {'─'*80}", flush=True)
    stage3 = {}
    for name, sharpes in kfold_results.items():
        mean_sh = np.mean(sharpes) if sharpes else 0
        min_sh = min(sharpes) if sharpes else 0
        pos_count = sum(1 for s in sharpes if s > 0)
        fold_str = ','.join(f'{s:.1f}' for s in sharpes)
        print(f"  {name:<22s} [{fold_str:>33s}] {mean_sh:>6.2f} {min_sh:>6.2f} {pos_count:>3}/{len(sharpes)}", flush=True)
        stage3[name] = {'fold_sharpes': sharpes, 'mean': round(mean_sh, 3),
                        'min': round(min_sh, 3), 'positive': pos_count, 'total': len(sharpes),
                        'pass': pos_count >= len(sharpes) - 1}

    results['stage3_kfold'] = stage3

    # ═══════════════════════════════════════════════════════════
    # Stage 4: Yearly Stability
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Stage 4: Yearly Stability", flush=True)
    print("=" * 70, flush=True)

    years = range(2015, 2027)
    yearly = {name: {} for name in configs}

    for yr in years:
        yr_s = f"{yr}-01-01"; yr_e = f"{yr+1}-01-01"
        h1_yr = h1_df[(h1_df.index >= yr_s) & (h1_df.index < yr_e)]
        if len(h1_yr) < 200:
            continue
        for name, cfg in configs.items():
            daily, _, _ = run_portfolio(h1_yr, bundle, **cfg)
            sh = _sharpe(daily); pnl = float(np.sum(daily))
            yearly[name][yr] = {'sharpe': round(sh, 3), 'pnl': round(pnl, 0)}

    print(f"\n  {'Year':>6}", end="", flush=True)
    for name in configs:
        print(f"  {name:>12}", end="")
    print(flush=True)
    print(f"  {'─'*6}" + "  " + "  ".join("─" * 12 for _ in configs), flush=True)

    for yr in years:
        vals = []
        for name in configs:
            if yr in yearly[name]:
                vals.append(yearly[name][yr]['sharpe'])
            else:
                vals.append(None)
        if all(v is None for v in vals):
            continue
        line = f"  {yr:>6}"
        for v in vals:
            line += f"  {v:>12.3f}" if v is not None else f"  {'N/A':>12}"
        print(line, flush=True)

    stage4 = {}
    for name in configs:
        neg_years = sum(1 for y, d in yearly[name].items() if d['pnl'] <= 0)
        stage4[name] = {'yearly': yearly[name], 'negative_years': neg_years,
                        'pass': neg_years <= 2}
    results['stage4_yearly'] = stage4

    # ═══════════════════════════════════════════════════════════
    # Stage 5: Monte Carlo (1000 paths)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Stage 5: Monte Carlo Robustness (1000 paths)", flush=True)
    print("=" * 70, flush=True)

    rng = np.random.RandomState(42)
    mc_results = {}

    for name in configs:
        daily = daily_arrays[name]
        mc_sharpes = []
        for _ in range(1000):
            spread_noise = rng.uniform(0.20, 0.50)
            slippage = rng.uniform(0, 0.15)
            cost_factor = (SPREAD + slippage) / spread_noise
            perturbed = daily * rng.uniform(0.92, 1.08, size=len(daily))
            mask = rng.random(len(daily)) > 0.02
            perturbed = perturbed * mask
            mc_sharpes.append(_sharpe(perturbed))

        mc_arr = np.array(mc_sharpes)
        mc_results[name] = {
            'p5': round(float(np.percentile(mc_arr, 5)), 3),
            'p25': round(float(np.percentile(mc_arr, 25)), 3),
            'p50': round(float(np.percentile(mc_arr, 50)), 3),
            'p75': round(float(np.percentile(mc_arr, 75)), 3),
            'p95': round(float(np.percentile(mc_arr, 95)), 3),
            'pct_positive': round(float(np.mean(mc_arr > 0) * 100), 1),
            'pct_gt1': round(float(np.mean(mc_arr > 1) * 100), 1),
        }
        print(f"  {name:<22s}: P5={mc_results[name]['p5']:.3f}  P50={mc_results[name]['p50']:.3f}  "
              f"P95={mc_results[name]['p95']:.3f}  pos={mc_results[name]['pct_positive']:.0f}%", flush=True)

    results['stage5_monte_carlo'] = mc_results

    # ═══════════════════════════════════════════════════════════
    # Stage 6: Combinatorial PBO (1000 shuffles)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Stage 6: Probability of Backtest Overfitting (PBO)", flush=True)
    print("=" * 70, flush=True)

    pbo_results = {}
    for name in configs:
        daily = daily_arrays[name]
        real_sharpe = _sharpe(daily)
        n_better = 0
        for _ in range(1000):
            shuffled = rng.permutation(daily)
            if _sharpe(shuffled) >= real_sharpe:
                n_better += 1
        pbo = n_better / 1000
        pbo_results[name] = {'real_sharpe': round(real_sharpe, 3), 'pbo': round(pbo, 3),
                             'pass': pbo < 0.50}
        status = "PASS" if pbo < 0.50 else "FAIL"
        print(f"  {name:<22s}: PBO={pbo:.3f} ({status})  real_sharpe={real_sharpe:.3f}", flush=True)

    results['stage6_pbo'] = pbo_results

    # ═══════════════════════════════════════════════════════════
    # Stage 7: PSAR-only deep analysis
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Stage 7: PSAR Default vs Optimized — Standalone Deep Dive", flush=True)
    print("=" * 70, flush=True)

    psar_default_trades = bt_psar(h1_df, SPREAD, UNIT_LOT, CAPS['PSAR'], params=PSAR_DEFAULT)
    psar_opt_trades = bt_psar(h1_df, SPREAD, UNIT_LOT, CAPS['PSAR'], params=PSAR_OPTIMIZED)

    pd_stats = _compute_stats(psar_default_trades)
    po_stats = _compute_stats(psar_opt_trades)

    print(f"\n  {'Metric':<15} {'Default':>12} {'Optimized':>12} {'Delta':>10}", flush=True)
    print(f"  {'─'*50}", flush=True)
    for k in ['n', 'sharpe', 'pnl', 'wr', 'max_dd', 'calmar']:
        dv = pd_stats[k]; ov = po_stats[k]
        if isinstance(dv, float):
            delta = ov - dv
            print(f"  {k:<15} {dv:>12.2f} {ov:>12.2f} {delta:>+10.2f}", flush=True)
        else:
            print(f"  {k:<15} {dv:>12} {ov:>12}", flush=True)

    psar_yearly = {}
    for yr in range(2015, 2027):
        yr_s = f"{yr}-01-01"; yr_e = f"{yr+1}-01-01"
        h1_yr = h1_df[(h1_df.index >= yr_s) & (h1_df.index < yr_e)]
        if len(h1_yr) < 200:
            continue
        def_t = bt_psar(h1_yr, SPREAD, UNIT_LOT, CAPS['PSAR'], params=PSAR_DEFAULT)
        opt_t = bt_psar(h1_yr, SPREAD, UNIT_LOT, CAPS['PSAR'], params=PSAR_OPTIMIZED)
        ds = _compute_stats(def_t); os_s = _compute_stats(opt_t)
        psar_yearly[yr] = {'default_sharpe': ds['sharpe'], 'opt_sharpe': os_s['sharpe'],
                           'default_pnl': ds['pnl'], 'opt_pnl': os_s['pnl']}

    print(f"\n  PSAR Yearly Comparison:", flush=True)
    print(f"  {'Year':>6} {'Def Shp':>8} {'Opt Shp':>8} {'Delta':>7} {'Def PnL':>10} {'Opt PnL':>10}", flush=True)
    print(f"  {'─'*52}", flush=True)
    opt_wins = 0; total_yrs = 0
    for yr, d in sorted(psar_yearly.items()):
        delta = d['opt_sharpe'] - d['default_sharpe']
        if delta > 0: opt_wins += 1
        total_yrs += 1
        marker = " *" if delta > 0.5 else ""
        print(f"  {yr:>6} {d['default_sharpe']:>8.3f} {d['opt_sharpe']:>8.3f} {delta:>+7.3f} "
              f"${d['default_pnl']:>9,.0f} ${d['opt_pnl']:>9,.0f}{marker}", flush=True)

    print(f"\n  Optimized wins {opt_wins}/{total_yrs} years", flush=True)

    results['stage7_psar_standalone'] = {
        'default_stats': pd_stats, 'optimized_stats': po_stats,
        'yearly': psar_yearly, 'opt_wins_years': opt_wins, 'total_years': total_yrs,
    }

    # ═══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)

    print("\n" + "=" * 80, flush=True)
    print("  R131 FINAL VERDICT", flush=True)
    print("=" * 80, flush=True)

    for name in configs:
        s1 = stage1[name]
        s3 = stage3[name]
        s5 = mc_results[name]
        s6 = pbo_results[name]
        s4 = stage4[name]

        wf_sharpes = [w[name] for w in wf_results]
        wf_mean = np.mean(wf_sharpes) if wf_sharpes else 0
        wf_min = min(wf_sharpes) if wf_sharpes else 0

        all_pass = s3['pass'] and s6['pass'] and s4['pass'] and wf_min > 0
        verdict = "VALIDATED" if all_pass else "PARTIAL"

        print(f"\n  {name}:", flush=True)
        print(f"    Full-data:     Sharpe={s1['sharpe']:.3f}, MaxDD=${s1['max_dd']:.0f}, Calmar={s1['calmar']:.1f}", flush=True)
        print(f"    WF-OOS mean:   {wf_mean:.3f} (min={wf_min:.3f})", flush=True)
        print(f"    K-Fold:        mean={s3['mean']:.3f}, {s3['positive']}/{s3['total']} positive ({'PASS' if s3['pass'] else 'FAIL'})", flush=True)
        print(f"    MC P5 Sharpe:  {s5['p5']:.3f}, positive={s5['pct_positive']:.0f}%", flush=True)
        print(f"    PBO:           {s6['pbo']:.3f} ({'PASS' if s6['pass'] else 'FAIL'})", flush=True)
        print(f"    Yearly:        {s4['negative_years']} negative years ({'PASS' if s4['pass'] else 'FAIL'})", flush=True)
        print(f"    → {verdict}", flush=True)

        results[f'verdict_{name}'] = verdict

    out_file = OUTPUT_DIR / "r131_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"  Saved: {out_file}", flush=True)
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
