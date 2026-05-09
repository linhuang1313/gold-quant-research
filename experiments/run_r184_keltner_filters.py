#!/usr/bin/env python3
"""
R184 — Keltner Entry Filter Impact on R:R
===========================================
Test how each entry filter changes Keltner's trade quality (R-multiple, WR, safety margin).

Filters under test:
  1. ADX threshold: 0 (no ADX), 10, 14 (baseline), 18, 22, 25
  2. Session filter: all hours, Asia only, London only, NY only, London+NY
  3. KCBW bandwidth expansion filter: OFF (baseline), ON
  4. EMA direction: EMA100 (baseline), EMA200, no EMA
  5. ML-style quality filter proxy: simulate via ADX+ATR rank combined
  6. KC multiplier: 1.0, 1.2 (baseline), 1.5, 2.0

All tests use baseline exit params (SL=3.5, trail_act=0.14, trail_dist=0.025, MH=2).
Output: per-config R-multiple, Sharpe, WR%, safety margin, era breakdown.
Top candidates get 6-Fold + MC robustness.
"""
import sys, os, time, json
import numpy as np
import pandas as pd
from pathlib import Path
import glob as _glob

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r184_keltner_filters")
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

SL_ATR = 3.5
TP_ATR = 8.0
TRAIL_ACT = 0.14
TRAIL_DIST = 0.025
MAX_HOLD = 2

SESSION_MAP = {
    'asia':      list(range(0, 8)),
    'london':    list(range(8, 13)),
    'ny':        list(range(13, 18)),
    'evening':   list(range(18, 24)),
    'london_ny': list(range(8, 18)),
}


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


def bt_keltner_filtered(h1_df, adx_th=14, ema_type='ema100', kc_mult=1.2,
                         session_hours=None, kcbw_filter=False):
    """Keltner backtest with configurable entry filters and fixed exit params."""
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df['ADX'] = compute_adx(df)
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['KC_mid'] = df['Close'].ewm(span=25, adjust=False).mean()
    df['KC_upper'] = df['KC_mid'] + kc_mult * df['ATR']
    df['KC_lower'] = df['KC_mid'] - kc_mult * df['ATR']
    if kcbw_filter:
        df['KCBW'] = (df['KC_upper'] - df['KC_lower']) / df['KC_mid']
        df['KCBW5'] = df['KCBW'].rolling(5).mean()
    df = df.dropna(subset=['ATR', 'ADX', 'EMA100', 'KC_upper'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; adx = df['ADX'].values
    ema100 = df['EMA100'].values; ema200 = df['EMA200'].values
    kc_u = df['KC_upper'].values; kc_l = df['KC_lower'].values
    hours = df.index.hour
    kcbw = df['KCBW'].values if kcbw_filter else None
    kcbw5 = df['KCBW5'].values if kcbw_filter else None
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999

    for i in range(1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], SPREAD, LOT, PV, times,
                               SL_ATR, TP_ATR, TRAIL_ACT, TRAIL_DIST, MAX_HOLD, CAP_ATR_MULT)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue

        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue

        # ADX filter
        if adx_th > 0:
            if np.isnan(adx[i]) or adx[i] < adx_th: continue

        # Session filter
        if session_hours is not None:
            if hours[i] not in session_hours: continue

        # KCBW expansion filter
        if kcbw_filter and kcbw is not None and kcbw5 is not None:
            if np.isnan(kcbw[i]) or np.isnan(kcbw5[i]): continue
            if kcbw[i] <= kcbw5[i]: continue

        # EMA direction filter
        if ema_type == 'ema100':
            buy_ok = c[i] > ema100[i]
            sell_ok = c[i] < ema100[i]
        elif ema_type == 'ema200':
            if np.isnan(ema200[i]): continue
            buy_ok = c[i] > ema200[i]
            sell_ok = c[i] < ema200[i]
        elif ema_type == 'none':
            buy_ok = True
            sell_ok = True
        else:
            buy_ok = c[i] > ema100[i]
            sell_ok = c[i] < ema100[i]

        if c[i] > kc_u[i] and buy_ok:
            pos = {'dir': 'BUY', 'entry': c[i] + SPREAD / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < kc_l[i] and sell_ok:
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
                'r_mult': 0, 'avg_win': 0, 'avg_loss': 0, 'breakeven_wr': 0, 'safety_margin': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    r_mult = avg_win / avg_loss if avg_loss > 0 else float('inf')
    breakeven = 1.0 / (1.0 + r_mult) if r_mult > 0 else 1.0
    wr = len(wins) / n * 100
    return {
        'n': n,
        'sharpe': round(_sharpe(daily), 3),
        'pnl': round(sum(pnls), 2),
        'wr': round(wr, 1),
        'max_dd': round(_max_dd(daily), 2),
        'r_mult': round(r_mult, 3),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'breakeven_wr': round(breakeven * 100, 1),
        'safety_margin': round(wr - breakeven * 100, 1),
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


def print_row(label, s):
    pnl_str = f"${s['pnl']:>9,.0f}" if s['pnl'] >= 0 else f"-${abs(s['pnl']):>8,.0f}"
    print(f"  {label:<35} {s['n']:>6} {s['sharpe']:>7.3f} {pnl_str} {s['wr']:>5.1f}% "
          f"${s['max_dd']:>7,.0f} {s['r_mult']:>7.3f} ${s['avg_win']:>7.2f} "
          f"${s['avg_loss']:>7.2f} {s['safety_margin']:>6.1f}pp", flush=True)


HEADER = (f"  {'Config':<35} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} "
          f"{'MaxDD':>8} {'R_mult':>7} {'AvgWin':>8} {'AvgLoss':>8} {'Margin':>8}")


# ═══════════════════════════════════════════════════════════════
# Phase 1: Filter grid scan
# ═══════════════════════════════════════════════════════════════

def phase1_filter_scan(h1_df):
    print(f"\n{'=' * 110}", flush=True)
    print(f"  PHASE 1: Entry Filter Grid Scan", flush=True)
    print(f"{'=' * 110}", flush=True)

    results = {}

    def run_test(label, **kwargs):
        trades = bt_keltner_filtered(h1_df, **kwargs)
        s = compute_full_stats(trades)
        print_row(label, s)
        results[label] = {'kwargs': kwargs, 'stats': s, 'trades': trades}
        return s

    # --- 1. ADX Threshold ---
    print(f"\n  --- Filter 1: ADX Threshold ---", flush=True)
    print(HEADER, flush=True)
    for adx in [0, 8, 10, 12, 14, 16, 18, 20, 22, 25]:
        run_test(f"ADX_{adx}", adx_th=adx)

    # --- 2. Session ---
    print(f"\n  --- Filter 2: Session Hours ---", flush=True)
    print(HEADER, flush=True)
    run_test("Session_ALL", session_hours=None)
    for sess_name, sess_hours in SESSION_MAP.items():
        run_test(f"Session_{sess_name}", session_hours=set(sess_hours))

    # --- 3. KCBW ---
    print(f"\n  --- Filter 3: KCBW Bandwidth Expansion ---", flush=True)
    print(HEADER, flush=True)
    run_test("KCBW_OFF", kcbw_filter=False)
    run_test("KCBW_ON", kcbw_filter=True)

    # --- 4. EMA Direction ---
    print(f"\n  --- Filter 4: EMA Direction ---", flush=True)
    print(HEADER, flush=True)
    run_test("EMA_100", ema_type='ema100')
    run_test("EMA_200", ema_type='ema200')
    run_test("EMA_NONE", ema_type='none')

    # --- 5. KC Multiplier ---
    print(f"\n  --- Filter 5: KC Band Multiplier ---", flush=True)
    print(HEADER, flush=True)
    for kc_m in [0.8, 1.0, 1.2, 1.5, 2.0, 2.5]:
        run_test(f"KC_mult_{kc_m}", kc_mult=kc_m)

    # --- 6. Combined filters ---
    print(f"\n  --- Filter 6: Interesting Combinations ---", flush=True)
    print(HEADER, flush=True)
    run_test("ADX18_LondonNY", adx_th=18, session_hours=set(SESSION_MAP['london_ny']))
    run_test("ADX20_LondonNY", adx_th=20, session_hours=set(SESSION_MAP['london_ny']))
    run_test("ADX14_LondonNY", adx_th=14, session_hours=set(SESSION_MAP['london_ny']))
    run_test("ADX18_KCBW", adx_th=18, kcbw_filter=True)
    run_test("ADX14_KCBW", adx_th=14, kcbw_filter=True)
    run_test("ADX20_EMA200", adx_th=20, ema_type='ema200')
    run_test("ADX18_London_KCBW", adx_th=18, session_hours=set(SESSION_MAP['london']), kcbw_filter=True)
    run_test("ADX14_KC1.5", adx_th=14, kc_mult=1.5)
    run_test("ADX18_KC1.5", adx_th=18, kc_mult=1.5)
    run_test("ADX20_KC1.5_LondonNY", adx_th=20, kc_mult=1.5, session_hours=set(SESSION_MAP['london_ny']))

    return results


# ═══════════════════════════════════════════════════════════════
# Phase 2: Robustness for top candidates
# ═══════════════════════════════════════════════════════════════

def phase2_robustness(h1_df, p1_results):
    print(f"\n{'=' * 110}", flush=True)
    print(f"  PHASE 2: Robustness Validation (Top 5 by Sharpe, excluding baseline)", flush=True)
    print(f"{'=' * 110}", flush=True)

    baseline_name = "ADX_14"
    baseline_kwargs = p1_results[baseline_name]['kwargs']
    baseline_stats = p1_results[baseline_name]['stats']

    candidates = [(k, v) for k, v in p1_results.items()
                  if k != baseline_name and v['stats']['n'] >= 500]
    candidates.sort(key=lambda x: x[1]['stats']['sharpe'], reverse=True)
    top5 = candidates[:5]

    print(f"\n  Baseline: {baseline_name} Sharpe={baseline_stats['sharpe']:.3f} "
          f"R={baseline_stats['r_mult']:.3f} Margin={baseline_stats['safety_margin']:.1f}pp", flush=True)
    print(f"\n  Top 5 candidates:", flush=True)
    for k, v in top5:
        s = v['stats']
        print(f"    {k}: Sharpe={s['sharpe']:.3f} R={s['r_mult']:.3f} "
              f"N={s['n']} Margin={s['safety_margin']:.1f}pp", flush=True)

    robustness_results = {}

    for cand_name, cand_data in top5:
        cand_kwargs = cand_data['kwargs']

        print(f"\n  {'='*80}", flush=True)
        print(f"  {cand_name} vs {baseline_name}", flush=True)
        print(f"  {'='*80}", flush=True)

        # K-Fold
        print(f"  {'Fold':<16} {'Base_Sh':>8} {'Cand_Sh':>8} {'Delta':>7} {'Base_R':>7} {'Cand_R':>7}", flush=True)
        cand_wins = 0
        fold_data = []
        for fold_name, start, end in FOLDS:
            fold_df = h1_df.loc[start:end].copy()
            if len(fold_df) < 200: continue
            bt = bt_keltner_filtered(fold_df, **baseline_kwargs)
            ct = bt_keltner_filtered(fold_df, **cand_kwargs)
            bs = compute_full_stats(bt)
            cs = compute_full_stats(ct)
            delta = cs['sharpe'] - bs['sharpe']
            if delta > 0: cand_wins += 1
            fold_data.append({'fold': fold_name, 'delta': delta})
            w = "+" if delta > 0 else "-"
            print(f"  {fold_name:<16} {bs['sharpe']:>8.3f} {cs['sharpe']:>8.3f} {delta:>+7.3f} "
                  f"{bs['r_mult']:>7.3f} {cs['r_mult']:>7.3f} {w}", flush=True)

        kf_total = len(fold_data)
        kf_pass = cand_wins >= max(1, int(kf_total * 2 / 3))

        # Monte Carlo
        base_trades = bt_keltner_filtered(h1_df, **baseline_kwargs)
        cand_trades = bt_keltner_filtered(h1_df, **cand_kwargs)
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

        # Era test
        era_wins = 0
        for era in ['full', 'hike', 'cut', 'recent_3y']:
            bt_e = filter_trades_by_era(base_trades, era)
            ct_e = filter_trades_by_era(cand_trades, era)
            if compute_full_stats(ct_e)['sharpe'] > compute_full_stats(bt_e)['sharpe']:
                era_wins += 1

        tests_passed = sum([kf_pass, mc_pass])
        overall = "GO" if tests_passed == 2 else ("CAUTION" if tests_passed == 1 else "NO-GO")

        kf_str = f"PASS ({cand_wins}/{kf_total})" if kf_pass else f"FAIL ({cand_wins}/{kf_total})"
        mc_str = "PASS" if mc_pass else ("MARGINAL" if prob >= 0.70 else "FAIL")

        print(f"  K-Fold: {kf_str} | MC: P={prob:.1%} CI=[{ci_lo:+.3f},{ci_hi:+.3f}] {mc_str} | Eras: {era_wins}/4 | {overall}", flush=True)

        robustness_results[cand_name] = {
            'kwargs': cand_kwargs,
            'stats': cand_data['stats'],
            'kfold': kf_str, 'kfold_pass': kf_pass,
            'mc': mc_str, 'mc_pass': mc_pass, 'mc_prob': round(prob, 3),
            'mc_ci': [round(ci_lo, 3), round(ci_hi, 3)],
            'era_wins': era_wins, 'overall': overall,
        }

    return robustness_results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 110, flush=True)
    print("  R184 -- Keltner Entry Filter Impact on R:R", flush=True)
    print("=" * 110, flush=True)

    h1_df = load_h1()

    p1 = phase1_filter_scan(h1_df)
    p2 = phase2_robustness(h1_df, p1)

    # Final Summary
    elapsed = time.time() - t0
    print(f"\n\n{'=' * 110}", flush=True)
    print(f"  FINAL SUMMARY", flush=True)
    print(f"{'=' * 110}", flush=True)

    print(f"\n  {'Config':<35} {'Sharpe':>7} {'R_mult':>7} {'Margin':>8} {'KFold':>16} {'MC':>12} {'Overall':>8}", flush=True)
    print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*8} {'-'*16} {'-'*12} {'-'*8}", flush=True)
    for name, data in p2.items():
        s = data['stats']
        print(f"  {name:<35} {s['sharpe']:>7.3f} {s['r_mult']:>7.3f} {s['safety_margin']:>6.1f}pp "
              f"{data['kfold']:>16} {data['mc']:>12} {data['overall']:>8}", flush=True)

    # Baseline for reference
    bl = p1['ADX_14']['stats']
    print(f"\n  Baseline (ADX_14):  Sharpe={bl['sharpe']:.3f}  R={bl['r_mult']:.3f}  "
          f"Margin={bl['safety_margin']:.1f}pp  N={bl['n']}", flush=True)

    print(f"\n  Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    save_data = {
        'phase1': {k: {'kwargs': v['kwargs'], 'stats': v['stats']} for k, v in p1.items()},
        'phase2': p2,
        'runtime_seconds': round(elapsed, 1),
    }
    out_path = OUTPUT_DIR / "r184_results.json"
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Saved: {out_path}", flush=True)
    print(f"{'=' * 110}", flush=True)


if __name__ == "__main__":
    main()
