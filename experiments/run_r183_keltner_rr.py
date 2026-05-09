#!/usr/bin/env python3
"""
R183 — Keltner Risk-Reward Optimization
=========================================
Core problem: SL=3.5xATR but trailing captures only 0.06-0.14xATR -> R=0.12.
System survives on 81% WR but safety margin is thin.

Phase 1: Per-axis grid scan (SL, trail activation, trail distance, max_hold)
Phase 2: Interaction grid (top 2 from each axis combined)
Phase 3: Robustness validation (top 3 configs: 6-Fold CV + 1000x MC)
Phase 4: Per-trade anatomy (exit reason breakdown, R-multiple, safety margin)
"""
import sys, os, time, json, itertools
import numpy as np
import pandas as pd
from pathlib import Path
import glob as _glob

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r183_keltner_rr")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
LOT = 0.02
CAP_ATR_MULT = 4.0

ERA_SEGMENTS = {
    'full':      None,
    'hike':      [("2015-12-01", "2019-01-01"), ("2022-03-01", "2023-08-01")],
    'cut':       [("2019-07-01", "2022-03-01"), ("2024-09-01", "2026-06-01")],
    'recent_3y': [("2023-06-01", "2026-06-01")],
}

FOLDS = [
    ("F1_2015_2017", "2015-01-01", "2017-01-01"),
    ("F2_2017_2019", "2017-01-01", "2019-01-01"),
    ("F3_2019_2021", "2019-01-01", "2021-01-01"),
    ("F4_2021_2023", "2021-01-01", "2023-01-01"),
    ("F5_2023_2025", "2023-01-01", "2025-01-01"),
    ("F6_2025_2026", "2025-01-01", "2026-06-01"),
]

MC_ITERATIONS = 1000
MC_SEED = 42

BASELINE = dict(sl_atr=3.5, tp_atr=8.0, trail_act=0.14, trail_dist=0.025, max_hold=2)


# ═══════════════════════════════════════════════════════════════
# Shared helpers (from R181)
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


def compute_adx(df, period=14):
    high = df['High']; low = df['Low']; close = df['Close']
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.DataFrame({'hl': high - low, 'hc': (high - close.shift(1)).abs(),
                        'lc': (low - close.shift(1)).abs()}).max(axis=1)
    atr_s = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_s)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_s)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.rolling(period).mean()


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar'],
            'atr': pos['atr']}


def _run_exit(pos, i, h, lo_v, c, spread, lot, pv, times,
              sl_atr, tp_atr, trail_act_atr, trail_dist_atr,
              max_hold, cap_atr_mult=0):
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
    if cap_atr_mult > 0 and cap_atr_mult < 900:
        cap_dollar = lot * pv * cap_atr_mult * pos['atr']
        if pnl_c < -cap_dollar:
            return _mk(pos, c, times[i], "Cap", i, -cap_dollar)
    if pos['dir'] == 'BUY' and h - pos['entry'] >= trail_act_atr * pos['atr']:
        ts_p = h - trail_dist_atr * pos['atr']
        if lo_v <= ts_p:
            return _mk(pos, c, times[i], "Trail", i, (ts_p - pos['entry'] - spread) * lot * pv)
    elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= trail_act_atr * pos['atr']:
        ts_p = lo_v + trail_dist_atr * pos['atr']
        if h >= ts_p:
            return _mk(pos, c, times[i], "Trail", i, (pos['entry'] - ts_p - spread) * lot * pv)
    if held >= max_hold:
        return _mk(pos, c, times[i], "Timeout", i, pnl_c)
    return None


def bt_keltner(h1_df, sl_atr=3.5, tp_atr=8.0,
               trail_act=0.14, trail_dist=0.025, max_hold=2,
               adx_th=14, ema_period=25, kc_mult=1.2):
    """Keltner backtest with configurable SL/trail/hold params."""
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df['ADX'] = compute_adx(df)
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['KC_mid'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    df['KC_upper'] = df['KC_mid'] + kc_mult * df['ATR']
    df['KC_lower'] = df['KC_mid'] - kc_mult * df['ATR']
    df = df.dropna(subset=['ATR', 'ADX', 'EMA100', 'KC_upper'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; adx = df['ADX'].values
    ema100 = df['EMA100'].values
    kc_u = df['KC_upper'].values; kc_l = df['KC_lower'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], SPREAD, LOT, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, CAP_ATR_MULT)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1 or np.isnan(adx[i]): continue
        if adx[i] < adx_th: continue
        if c[i] > kc_u[i] and c[i] > ema100[i]:
            pos = {'dir': 'BUY', 'entry': c[i] + SPREAD / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < kc_l[i] and c[i] < ema100[i]:
            pos = {'dir': 'SELL', 'entry': c[i] - SPREAD / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


# ═══════════════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════════════

def _trades_to_daily(trades):
    if not trades: return pd.Series(dtype=float)
    df = pd.DataFrame(trades)
    df['date'] = pd.to_datetime(df['entry_time']).dt.date
    return df.groupby('date')['pnl'].sum()


def _sharpe(daily):
    if len(daily) < 10: return 0.0
    return float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0


def _max_dd(daily):
    if len(daily) == 0: return 0.0
    eq = daily.cumsum()
    return float((np.maximum.accumulate(eq) - eq).max())


def compute_full_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0,
                'r_mult': 0, 'avg_win': 0, 'avg_loss': 0, 'breakeven_wr': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    r_mult = avg_win / avg_loss if avg_loss > 0 else float('inf')
    breakeven = 1.0 / (1.0 + r_mult) if r_mult > 0 else 1.0
    return {
        'n': n,
        'sharpe': round(_sharpe(daily), 3),
        'pnl': round(sum(pnls), 2),
        'wr': round(len(wins) / n * 100, 1),
        'max_dd': round(_max_dd(daily), 2),
        'r_mult': round(r_mult, 3),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'breakeven_wr': round(breakeven * 100, 1),
    }


def filter_trades_by_era(trades, era_name):
    if era_name == 'full' or ERA_SEGMENTS[era_name] is None:
        return trades
    periods = ERA_SEGMENTS[era_name]
    filtered = []
    for t in trades:
        entry = pd.Timestamp(t['entry_time'])
        for start, end in periods:
            if pd.Timestamp(start) <= entry < pd.Timestamp(end):
                filtered.append(t)
                break
    return filtered


def load_h1():
    candidates = sorted(_glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
    if not candidates:
        raise FileNotFoundError("No H1 data found in data/download/")
    csv_path = candidates[-1]
    print(f"  Loading H1: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                        'close': 'Close', 'volume': 'Volume'}, inplace=True)
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    print(f"  {len(df)} bars ({df.index[0]} ~ {df.index[-1]})", flush=True)
    return df


# ═══════════════════════════════════════════════════════════════
# Phase 1: Per-axis grid scan
# ═══════════════════════════════════════════════════════════════

def phase1_grid_scan(h1_df):
    print(f"\n{'=' * 100}", flush=True)
    print(f"  PHASE 1: Per-Axis Grid Scan", flush=True)
    print(f"{'=' * 100}", flush=True)

    results = {}

    header = (f"  {'Config':<30} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} "
              f"{'MaxDD':>8} {'R_mult':>7} {'AvgWin':>8} {'AvgLoss':>8} {'BEwr%':>6}")

    def run_and_print(label, **kwargs):
        params = {**BASELINE, **kwargs}
        trades = bt_keltner(h1_df, **params)
        s = compute_full_stats(trades)
        pnl_str = f"${s['pnl']:>9,.0f}" if s['pnl'] >= 0 else f"-${abs(s['pnl']):>8,.0f}"
        print(f"  {label:<30} {s['n']:>6} {s['sharpe']:>7.3f} {pnl_str} {s['wr']:>5.1f}% "
              f"${s['max_dd']:>7,.0f} {s['r_mult']:>7.3f} ${s['avg_win']:>7.2f} "
              f"${s['avg_loss']:>7.2f} {s['breakeven_wr']:>5.1f}%", flush=True)
        results[label] = {'params': params, 'stats': s, 'trades': trades}
        return s

    # --- Axis 1: SL ---
    print(f"\n  --- Axis 1: Stop Loss ---", flush=True)
    print(header, flush=True)
    for sl in [1.5, 2.0, 2.5, 3.0, 3.5]:
        run_and_print(f"SL_{sl}", sl_atr=sl)

    # --- Axis 2: Trail Activation ---
    print(f"\n  --- Axis 2: Trailing Activation ---", flush=True)
    print(header, flush=True)
    for act in [0.14, 0.30, 0.50, 0.75, 1.0, 1.5, 2.0]:
        dist = round(act / 5.6, 4)
        run_and_print(f"Act_{act}_D_{dist:.3f}", trail_act=act, trail_dist=dist)

    # --- Axis 3: Trail Distance ---
    print(f"\n  --- Axis 3: Trailing Distance (act=0.14 fixed) ---", flush=True)
    print(header, flush=True)
    for dist in [0.025, 0.05, 0.10, 0.15, 0.20, 0.30]:
        run_and_print(f"Dist_{dist}", trail_dist=dist)

    # --- Axis 4: Max Hold ---
    print(f"\n  --- Axis 4: Max Hold ---", flush=True)
    print(header, flush=True)
    for mh in [2, 3, 5, 8, 12, 20]:
        run_and_print(f"MH_{mh}", max_hold=mh)

    return results


def pick_top_n(results, prefix, n=2):
    """Pick top N configs by Sharpe from a prefix group."""
    candidates = [(k, v) for k, v in results.items() if k.startswith(prefix)]
    candidates.sort(key=lambda x: x[1]['stats']['sharpe'], reverse=True)
    return candidates[:n]


# ═══════════════════════════════════════════════════════════════
# Phase 2: Interaction grid
# ═══════════════════════════════════════════════════════════════

def phase2_interaction(h1_df, p1_results):
    print(f"\n{'=' * 100}", flush=True)
    print(f"  PHASE 2: Interaction Grid (Top 2 from each axis)", flush=True)
    print(f"{'=' * 100}", flush=True)

    top_sl = pick_top_n(p1_results, "SL_")
    top_act = pick_top_n(p1_results, "Act_")
    top_dist = pick_top_n(p1_results, "Dist_")
    top_mh = pick_top_n(p1_results, "MH_")

    print(f"  Top SL:   {[k for k,_ in top_sl]}", flush=True)
    print(f"  Top Act:  {[k for k,_ in top_act]}", flush=True)
    print(f"  Top Dist: {[k for k,_ in top_dist]}", flush=True)
    print(f"  Top MH:   {[k for k,_ in top_mh]}", flush=True)

    sl_vals = [v['params']['sl_atr'] for _, v in top_sl]
    act_vals = [v['params']['trail_act'] for _, v in top_act]
    dist_vals = [v['params']['trail_dist'] for _, v in top_dist]
    mh_vals = [v['params']['max_hold'] for _, v in top_mh]

    header = (f"  {'Config':<45} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} "
              f"{'MaxDD':>8} {'R_mult':>7} {'AvgWin':>8} {'AvgLoss':>8} {'BEwr%':>6}")
    print(f"\n{header}", flush=True)

    results = {}
    for sl, act, dist, mh in itertools.product(sl_vals, act_vals, dist_vals, mh_vals):
        label = f"SL{sl}_A{act}_D{dist}_MH{mh}"
        params = dict(sl_atr=sl, tp_atr=8.0, trail_act=act, trail_dist=dist, max_hold=mh)
        trades = bt_keltner(h1_df, **params)
        s = compute_full_stats(trades)
        pnl_str = f"${s['pnl']:>9,.0f}" if s['pnl'] >= 0 else f"-${abs(s['pnl']):>8,.0f}"
        print(f"  {label:<45} {s['n']:>6} {s['sharpe']:>7.3f} {pnl_str} {s['wr']:>5.1f}% "
              f"${s['max_dd']:>7,.0f} {s['r_mult']:>7.3f} ${s['avg_win']:>7.2f} "
              f"${s['avg_loss']:>7.2f} {s['breakeven_wr']:>5.1f}%", flush=True)
        results[label] = {'params': params, 'stats': s, 'trades': trades}

    # Also test baseline for comparison
    label = "BASELINE_CURRENT"
    trades = bt_keltner(h1_df, **BASELINE)
    s = compute_full_stats(trades)
    results[label] = {'params': BASELINE.copy(), 'stats': s, 'trades': trades}
    pnl_str = f"${s['pnl']:>9,.0f}" if s['pnl'] >= 0 else f"-${abs(s['pnl']):>8,.0f}"
    print(f"\n  {label:<45} {s['n']:>6} {s['sharpe']:>7.3f} {pnl_str} {s['wr']:>5.1f}% "
          f"${s['max_dd']:>7,.0f} {s['r_mult']:>7.3f} ${s['avg_win']:>7.2f} "
          f"${s['avg_loss']:>7.2f} {s['breakeven_wr']:>5.1f}%", flush=True)

    return results


# ═══════════════════════════════════════════════════════════════
# Phase 3: Robustness validation
# ═══════════════════════════════════════════════════════════════

def phase3_robustness(h1_df, p2_results):
    print(f"\n{'=' * 100}", flush=True)
    print(f"  PHASE 3: Robustness Validation (Top 3 vs Baseline)", flush=True)
    print(f"{'=' * 100}", flush=True)

    # Pick top 3 by Sharpe (excluding baseline)
    candidates = [(k, v) for k, v in p2_results.items() if k != "BASELINE_CURRENT"]
    candidates.sort(key=lambda x: x[1]['stats']['sharpe'], reverse=True)
    top3 = candidates[:3]
    baseline = p2_results["BASELINE_CURRENT"]

    print(f"  Top 3 candidates:", flush=True)
    for k, v in top3:
        print(f"    {k}: Sharpe={v['stats']['sharpe']:.3f}, R={v['stats']['r_mult']:.3f}", flush=True)

    robustness_results = {}

    for cand_name, cand_data in top3:
        print(f"\n  {'='*80}", flush=True)
        print(f"  Testing: {cand_name}", flush=True)
        print(f"  {'='*80}", flush=True)

        cand_params = cand_data['params']

        # --- K-Fold ---
        print(f"\n  K-Fold (6 folds):", flush=True)
        print(f"  {'Fold':<16} {'Base_Sh':>8} {'Cand_Sh':>8} {'Delta':>7} {'Base_R':>7} {'Cand_R':>7} {'Winner':>8}", flush=True)
        print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*8}", flush=True)

        fold_data = []
        cand_wins = 0
        for fold_name, start, end in FOLDS:
            fold_df = h1_df.loc[start:end].copy()
            if len(fold_df) < 200: continue
            base_trades = bt_keltner(fold_df, **BASELINE)
            cand_trades = bt_keltner(fold_df, **cand_params)
            bs = compute_full_stats(base_trades)
            cs = compute_full_stats(cand_trades)
            delta = cs['sharpe'] - bs['sharpe']
            winner = "CAND" if delta > 0 else "BASE"
            if delta > 0: cand_wins += 1
            fold_data.append({'fold': fold_name, 'base': bs, 'cand': cs, 'delta': delta})
            print(f"  {fold_name:<16} {bs['sharpe']:>8.3f} {cs['sharpe']:>8.3f} {delta:>+7.3f} "
                  f"{bs['r_mult']:>7.3f} {cs['r_mult']:>7.3f} {winner:>8}", flush=True)

        kf_total = len(fold_data)
        kf_pass = cand_wins >= max(1, int(kf_total * 2 / 3))
        kf_verdict = f"PASS ({cand_wins}/{kf_total})" if kf_pass else f"FAIL ({cand_wins}/{kf_total})"
        print(f"  K-Fold: {kf_verdict}", flush=True)

        # --- Monte Carlo ---
        print(f"\n  Monte Carlo ({MC_ITERATIONS}x):", flush=True)
        base_trades = bt_keltner(h1_df, **BASELINE)
        cand_trades = bt_keltner(h1_df, **cand_params)
        base_pnls = np.array([t['pnl'] for t in base_trades])
        cand_pnls = np.array([t['pnl'] for t in cand_trades])

        rng = np.random.RandomState(MC_SEED)
        delta_sharpes = np.zeros(MC_ITERATIONS)
        for it in range(MC_ITERATIONS):
            bs = rng.choice(base_pnls, size=len(base_pnls), replace=True)
            cs = rng.choice(cand_pnls, size=len(cand_pnls), replace=True)
            bs_std = bs.std(); cs_std = cs.std()
            bs_sh = bs.mean() / bs_std * np.sqrt(252) if bs_std > 0 else 0
            cs_sh = cs.mean() / cs_std * np.sqrt(252) if cs_std > 0 else 0
            delta_sharpes[it] = cs_sh - bs_sh

        prob = float(np.mean(delta_sharpes > 0))
        ci_lo = float(np.percentile(delta_sharpes, 2.5))
        ci_hi = float(np.percentile(delta_sharpes, 97.5))
        mc_pass = prob >= 0.70 and ci_lo > 0
        mc_verdict = "PASS" if mc_pass else ("MARGINAL" if prob >= 0.70 else "FAIL")

        print(f"  P(cand > base): {prob:.1%}", flush=True)
        print(f"  Delta 95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]", flush=True)
        print(f"  MC: {mc_verdict}", flush=True)

        # --- Era test ---
        era_wins = 0
        print(f"\n  Era comparison:", flush=True)
        for era in ['full', 'hike', 'cut', 'recent_3y']:
            bt = filter_trades_by_era(base_trades, era)
            ct = filter_trades_by_era(cand_trades, era)
            bs_e = compute_full_stats(bt)
            cs_e = compute_full_stats(ct)
            d = cs_e['sharpe'] - bs_e['sharpe']
            if d > 0: era_wins += 1
            print(f"    {era:<12} Base={bs_e['sharpe']:.3f} Cand={cs_e['sharpe']:.3f} "
                  f"Delta={d:+.3f} R_base={bs_e['r_mult']:.3f} R_cand={cs_e['r_mult']:.3f}", flush=True)

        tests_passed = sum([kf_pass, mc_pass])
        overall = "GO" if tests_passed == 2 else ("CAUTION" if tests_passed == 1 else "NO-GO")

        robustness_results[cand_name] = {
            'params': cand_params,
            'kfold_verdict': kf_verdict, 'kfold_pass': kf_pass,
            'mc_verdict': mc_verdict, 'mc_pass': mc_pass,
            'mc_prob': round(prob, 3), 'mc_ci': [round(ci_lo, 3), round(ci_hi, 3)],
            'era_wins': era_wins,
            'overall': overall,
        }

        print(f"\n  >>> {cand_name}: {overall} (K-Fold={kf_verdict}, MC={mc_verdict}, Eras={era_wins}/4)", flush=True)

    return robustness_results


# ═══════════════════════════════════════════════════════════════
# Phase 4: Per-trade anatomy
# ═══════════════════════════════════════════════════════════════

def phase4_anatomy(h1_df, p2_results, robustness_results):
    print(f"\n{'=' * 100}", flush=True)
    print(f"  PHASE 4: Per-Trade Anatomy Report", flush=True)
    print(f"{'=' * 100}", flush=True)

    # Get best candidate (by overall verdict, then Sharpe)
    go_configs = [(k, v) for k, v in robustness_results.items() if v['overall'] == 'GO']
    if not go_configs:
        caution_configs = [(k, v) for k, v in robustness_results.items() if v['overall'] == 'CAUTION']
        candidates = caution_configs if caution_configs else list(robustness_results.items())
    else:
        candidates = go_configs

    if not candidates:
        print("  No candidates passed robustness. Showing baseline anatomy only.", flush=True)
        configs_to_analyze = [("BASELINE_CURRENT", BASELINE)]
    else:
        best_name = candidates[0][0]
        best_params = candidates[0][1]['params']
        configs_to_analyze = [
            ("BASELINE_CURRENT", BASELINE),
            (best_name, best_params),
        ]

    for label, params in configs_to_analyze:
        trades = bt_keltner(h1_df, **params)
        if not trades:
            print(f"\n  {label}: No trades", flush=True)
            continue

        print(f"\n  {'='*80}", flush=True)
        print(f"  {label} (SL={params['sl_atr']}, Act={params['trail_act']}, "
              f"Dist={params['trail_dist']}, MH={params['max_hold']})", flush=True)
        print(f"  {'='*80}", flush=True)

        # Exit reason breakdown
        reasons = {}
        for t in trades:
            r = t['reason']
            if r not in reasons:
                reasons[r] = {'pnls': [], 'bars': []}
            reasons[r]['pnls'].append(t['pnl'])
            reasons[r]['bars'].append(t['bars'])

        print(f"\n  {'Reason':<12} {'Count':>6} {'Share':>6} {'WR%':>6} {'AvgPnL':>9} "
              f"{'TotalPnL':>10} {'AvgBars':>8}", flush=True)
        print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*9} {'-'*10} {'-'*8}", flush=True)

        total_n = len(trades)
        for reason in sorted(reasons.keys()):
            d = reasons[reason]
            n = len(d['pnls'])
            wr = sum(1 for p in d['pnls'] if p > 0) / n * 100
            avg_pnl = np.mean(d['pnls'])
            total_pnl = sum(d['pnls'])
            avg_bars = np.mean(d['bars'])
            share = n / total_n * 100
            print(f"  {reason:<12} {n:>6} {share:>5.1f}% {wr:>5.1f}% ${avg_pnl:>8.2f} "
                  f"${total_pnl:>9.0f} {avg_bars:>8.1f}", flush=True)

        # Overall stats
        s = compute_full_stats(trades)
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        print(f"\n  Summary:", flush=True)
        print(f"    Trades: {s['n']}", flush=True)
        print(f"    WR: {s['wr']:.1f}% (actual) vs {s['breakeven_wr']:.1f}% (breakeven)", flush=True)
        print(f"    Safety margin: {s['wr'] - s['breakeven_wr']:.1f} pp", flush=True)
        print(f"    R-multiple: {s['r_mult']:.3f} (avg_win=${s['avg_win']:.2f} / avg_loss=${s['avg_loss']:.2f})", flush=True)
        print(f"    Sharpe: {s['sharpe']:.3f}", flush=True)
        print(f"    PnL: ${s['pnl']:,.0f}", flush=True)
        print(f"    MaxDD: ${s['max_dd']:,.0f}", flush=True)

        # Bars held distribution
        bars_arr = np.array([t['bars'] for t in trades])
        print(f"\n  Bars Held Distribution:", flush=True)
        for bv in sorted(set(bars_arr)):
            cnt = np.sum(bars_arr == bv)
            if cnt > 10:
                pct = cnt / len(bars_arr) * 100
                avg_p = np.mean([t['pnl'] for t in trades if t['bars'] == bv])
                print(f"    {bv:>3} bars: {cnt:>6} ({pct:>5.1f}%) avg_pnl=${avg_p:.2f}", flush=True)

    return configs_to_analyze


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 100, flush=True)
    print("  R183 -- Keltner Risk-Reward Optimization", flush=True)
    print("=" * 100, flush=True)

    h1_df = load_h1()

    # Phase 1
    p1 = phase1_grid_scan(h1_df)

    # Phase 2
    p2 = phase2_interaction(h1_df, p1)

    # Phase 3
    p3 = phase3_robustness(h1_df, p2)

    # Phase 4
    p4 = phase4_anatomy(h1_df, p2, p3)

    # ──────────────────────────────────────────────
    # Final Summary
    # ──────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n\n{'=' * 100}", flush=True)
    print(f"  FINAL SUMMARY", flush=True)
    print(f"{'=' * 100}", flush=True)

    print(f"\n  {'Config':<45} {'KFold':>16} {'MC':>12} {'Overall':>8}", flush=True)
    print(f"  {'-'*45} {'-'*16} {'-'*12} {'-'*8}", flush=True)
    for name, data in p3.items():
        print(f"  {name:<45} {data['kfold_verdict']:>16} {data['mc_verdict']:>12} {data['overall']:>8}", flush=True)

    print(f"\n  Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    # Save (strip trades for JSON size)
    save_data = {
        'phase1': {k: {'params': v['params'], 'stats': v['stats']} for k, v in p1.items()},
        'phase2': {k: {'params': v['params'], 'stats': v['stats']} for k, v in p2.items()},
        'phase3': p3,
        'runtime_seconds': round(elapsed, 1),
    }
    out_path = OUTPUT_DIR / "r183_results.json"
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Saved: {out_path}", flush=True)
    print(f"{'=' * 100}", flush=True)


if __name__ == "__main__":
    main()
