#!/usr/bin/env python3
"""
R159 — Rule B + Circuit Breaker Joint Deployment Test
=====================================================
MaxHold=8 is already deployed. This experiment tests the combined effect of
adding Rule B (R144 extreme protection) and Circuit Breaker V1 (R100 loss-
streak pause) on top of the current production baseline.

Phases:
  1. Full-sample 4-variant comparison (Baseline / RuleB / CB / RuleB+CB)
  2. Interaction analysis (marginal contributions, synergy check)
  3. K-Fold validation (6 folds) on Baseline vs RuleB+CB
  4. Production-level lot test with R89_LOTS
  5. Sensitivity check (skip_bars grid x pause_hours grid)
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

from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS

OUTPUT_DIR = Path("results/r159_joint_deploy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01
CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.09, 'SESS_BO': 0.09}

CUSUM_SIGMA = 3.0
EXTREME_WINDOW = 12
SKIP_BARS = 8
CB_STREAK = 3
CB_PAUSE_HOURS = 1

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
# Strategy backtests (from R100, unified)
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
    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': maxloss_cap,
          'spread_cost': spread, 'initial_capital': 2000,
          'keltner_max_hold_m15': 8,
          'min_lot_size': lot, 'max_lot_size': lot}
    result = run_variant(data_bundle, "L8_MAX", verbose=False, **kw)
    raw_trades = result.get('_trades', [])
    return [
        {'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
         'entry_time': t.entry_time, 'exit_time': t.exit_time,
         'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0}
        for t in raw_trades
    ]


# ═══════════════════════════════════════════════════════════════
# Stats helpers
# ═══════════════════════════════════════════════════════════════

def _normalize_ts(ts):
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    return t


def trades_to_daily(trades):
    if not trades:
        return np.array([0.0])
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).normalize()
        if d.tzinfo is not None:
            d = d.tz_localize(None)
        daily[d] = daily.get(d, 0) + t['pnl']
    if not daily:
        return np.array([0.0])
    return np.array([daily[k] for k in sorted(daily.keys())])


def sharpe(arr):
    if len(arr) < 10 or np.std(arr, ddof=1) == 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))


def max_dd(arr):
    if len(arr) < 2:
        return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def compute_stats(trades, label=""):
    n = len(trades)
    if n == 0:
        return {'label': label, 'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0,
                'max_dd': 0, 'avg_pnl': 0}
    pnls = np.array([t['pnl'] for t in trades])
    daily = trades_to_daily(trades)
    return {
        'label': label,
        'n': n,
        'sharpe': round(sharpe(daily), 3),
        'pnl': round(float(pnls.sum()), 2),
        'wr': round(float((pnls > 0).sum()) / n * 100, 1),
        'max_dd': round(max_dd(daily), 2),
        'avg_pnl': round(float(pnls.mean()), 4),
    }


# ═══════════════════════════════════════════════════════════════
# Portfolio merge (from R100)
# ═══════════════════════════════════════════════════════════════

def merge_portfolio_trades(strat_trades_dict, lot_scale=None):
    """Merge all strategy trades into chronological stream, optionally scaled."""
    if lot_scale is None:
        lot_scale = R89_LOTS
    all_trades = []
    for strat_name, trades in strat_trades_dict.items():
        mult = lot_scale.get(strat_name, UNIT_LOT) / UNIT_LOT
        for t in trades:
            all_trades.append({
                'strategy': strat_name,
                'dir': t['dir'], 'entry': t['entry'], 'exit': t['exit'],
                'entry_time': _normalize_ts(t['entry_time']),
                'exit_time': _normalize_ts(t['exit_time']),
                'pnl': t['pnl'] * mult,
                'pnl_unit': t['pnl'],
                'reason': t['reason'],
            })
    all_trades.sort(key=lambda x: x['exit_time'])
    return all_trades


# ═══════════════════════════════════════════════════════════════
# Rule B: Extreme protection (from R144)
# ═══════════════════════════════════════════════════════════════

def build_extreme_mask(h1_df, cusum_sigma=3.0, extreme_window=12):
    n = len(h1_df)
    extreme = np.zeros(n, dtype=bool)
    atr = compute_atr(h1_df).values
    atr_clean = np.nan_to_num(atr, nan=0.0)
    atr_mean = pd.Series(atr_clean).rolling(60, min_periods=20).mean().values
    atr_std = pd.Series(atr_clean).rolling(60, min_periods=20).std().values
    atr_std = np.maximum(atr_std, 1e-6)
    cusum_trigger = atr_clean > (atr_mean + cusum_sigma * atr_std)
    for i in range(n):
        if cusum_trigger[i]:
            end_i = min(i + extreme_window, n)
            extreme[i:end_i] = True
    return extreme


def apply_rule_b(trades, h1_df, extreme_mask, skip_bars=8):
    if not trades:
        return [], 0
    times_idx = h1_df.index
    n_bars = len(times_idx)
    idx_is_tz_aware = times_idx.tz is not None

    def _find_bar(ts):
        ts = pd.Timestamp(ts)
        if idx_is_tz_aware and ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        elif not idx_is_tz_aware and ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        idx = times_idx.searchsorted(ts)
        return min(idx, n_bars - 1) if idx < n_bars else n_bars - 1

    protected = []; skipped = 0
    for t in trades:
        entry_bar = _find_bar(t['entry_time'])
        skip_end = entry_bar
        for j in range(max(0, entry_bar - skip_bars), entry_bar):
            if j < n_bars and extreme_mask[j]:
                skip_end = j + skip_bars
                break
        if entry_bar < skip_end:
            skipped += 1
            continue
        protected.append(t)
    return protected, skipped


# ═══════════════════════════════════════════════════════════════
# Circuit Breaker V1 (from R100)
# ═══════════════════════════════════════════════════════════════

def apply_circuit_breaker(portfolio_trades, streak_thresh=3, pause_hours=1):
    taken = []; skipped = 0
    consec_losses = 0; pause_until = None
    for trade in portfolio_trades:
        entry_t = trade['entry_time']
        exit_t = trade['exit_time']
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
    return taken, skipped


# ═══════════════════════════════════════════════════════════════
# Run all strategies
# ═══════════════════════════════════════════════════════════════

def run_all_strategies(h1_df, l8_bundle):
    strat_trades = {}
    strat_trades['L8_MAX'] = bt_l8_max(l8_bundle, SPREAD, UNIT_LOT, CAPS['L8_MAX'])
    strat_trades['PSAR'] = bt_psar(h1_df, SPREAD, UNIT_LOT, CAPS['PSAR'])
    strat_trades['TSMOM'] = bt_tsmom(h1_df, SPREAD, UNIT_LOT, CAPS['TSMOM'])
    strat_trades['SESS_BO'] = bt_sess_bo(h1_df, SPREAD, UNIT_LOT, CAPS['SESS_BO'])
    return strat_trades


def apply_variant(portfolio_trades, h1_df, extreme_mask,
                  use_rule_b=False, use_cb=False,
                  skip_bars=SKIP_BARS, cb_streak=CB_STREAK, cb_pause=CB_PAUSE_HOURS):
    """Apply Rule B and/or Circuit Breaker to portfolio trades."""
    trades = list(portfolio_trades)
    rb_skipped = 0; cb_skipped = 0

    if use_rule_b:
        trades, rb_skipped = apply_rule_b(trades, h1_df, extreme_mask, skip_bars)

    if use_cb:
        trades, cb_skipped = apply_circuit_breaker(trades, cb_streak, cb_pause)

    return trades, rb_skipped, cb_skipped


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH

    print("=" * 80)
    print("  R159 — Rule B + Circuit Breaker Joint Deployment Test")
    print("  Baseline: MaxHold=8 (already deployed)")
    print("=" * 80, flush=True)

    # ─── Load data ───
    print("\n  Loading data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    l8_bundle = DataBundle.load_custom()
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})")
    print(f"  M15: {len(m15_raw)} bars", flush=True)

    # ─── Run strategies once ───
    print("\n" + "=" * 70)
    print("  Running all 4 strategies (L8_MAX with MaxHold=8)...")
    print("=" * 70, flush=True)
    strat_trades = run_all_strategies(h1_df, l8_bundle)
    for s, tr in strat_trades.items():
        print(f"    {s}: {len(tr)} trades", flush=True)

    portfolio_all = merge_portfolio_trades(strat_trades)
    print(f"\n    Merged portfolio: {len(portfolio_all)} trades")
    print(f"    Range: {portfolio_all[0]['exit_time'].date()} ~ "
          f"{portfolio_all[-1]['exit_time'].date()}", flush=True)

    # ─── Build extreme mask ───
    extreme_mask = build_extreme_mask(h1_df, CUSUM_SIGMA, EXTREME_WINDOW)
    n_extreme = int(extreme_mask.sum())
    print(f"\n    Extreme mask: {n_extreme}/{len(h1_df)} bars "
          f"({n_extreme/len(h1_df)*100:.1f}%)", flush=True)

    # ═════════════════════════════════════════════════════════════
    # Phase 1: Full-sample 4-variant comparison
    # ═════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Phase 1: Full-Sample 4-Variant Comparison")
    print("=" * 70, flush=True)

    VARIANTS = {
        'A_Baseline':  {'rule_b': False, 'cb': False},
        'B_RuleB':     {'rule_b': True,  'cb': False},
        'C_CB':        {'rule_b': False, 'cb': True},
        'D_RuleB_CB':  {'rule_b': True,  'cb': True},
    }

    phase1 = {}
    for vname, cfg in VARIANTS.items():
        trades, rb_skip, cb_skip = apply_variant(
            portfolio_all, h1_df, extreme_mask,
            use_rule_b=cfg['rule_b'], use_cb=cfg['cb'])
        st = compute_stats(trades, vname)
        st['rb_skipped'] = rb_skip
        st['cb_skipped'] = cb_skip
        phase1[vname] = st
        print(f"\n    {vname}:")
        print(f"      N={st['n']}, Sharpe={st['sharpe']:.3f}, PnL=${st['pnl']:.1f}, "
              f"MaxDD=${st['max_dd']:.1f}, WR={st['wr']:.1f}%")
        print(f"      RuleB skipped={rb_skip}, CB skipped={cb_skip}", flush=True)

    print(f"\n    {'Variant':<16} {'N':>6} {'Sharpe':>7} {'PnL':>10} "
          f"{'MaxDD':>8} {'WR':>6} {'RB_skip':>8} {'CB_skip':>8}")
    print("    " + "-" * 75)
    for v, st in phase1.items():
        print(f"    {v:<16} {st['n']:>6} {st['sharpe']:>7.3f} {st['pnl']:>10.1f} "
              f"{st['max_dd']:>8.1f} {st['wr']:>5.1f}% {st['rb_skipped']:>8} "
              f"{st['cb_skipped']:>8}", flush=True)

    # ═════════════════════════════════════════════════════════════
    # Phase 2: Interaction analysis
    # ═════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Phase 2: Interaction Analysis")
    print("=" * 70, flush=True)

    sa = phase1['A_Baseline']['sharpe']
    sb = phase1['B_RuleB']['sharpe']
    sc = phase1['C_CB']['sharpe']
    sd = phase1['D_RuleB_CB']['sharpe']

    delta_rb_standalone = sb - sa
    delta_cb_standalone = sc - sa
    delta_rb_marginal = sd - sc
    delta_cb_marginal = sd - sb
    synergy = sd - sa - delta_rb_standalone - delta_cb_standalone

    phase2 = {
        'delta_rb_standalone': round(delta_rb_standalone, 4),
        'delta_cb_standalone': round(delta_cb_standalone, 4),
        'delta_rb_marginal': round(delta_rb_marginal, 4),
        'delta_cb_marginal': round(delta_cb_marginal, 4),
        'synergy': round(synergy, 4),
        'combined_better_than_best_single': sd >= max(sb, sc),
        'negative_interaction': sd < max(sb, sc),
    }

    print(f"    RuleB standalone:  {delta_rb_standalone:+.4f}")
    print(f"    CB standalone:     {delta_cb_standalone:+.4f}")
    print(f"    RuleB marginal (on top of CB): {delta_rb_marginal:+.4f}")
    print(f"    CB marginal (on top of RuleB): {delta_cb_marginal:+.4f}")
    print(f"    Synergy:           {synergy:+.4f}")
    print(f"    Combined >= best single? {phase2['combined_better_than_best_single']}")
    if phase2['negative_interaction']:
        print("    *** WARNING: NEGATIVE INTERACTION ***", flush=True)
    else:
        print("    No negative interaction detected.", flush=True)

    # ═════════════════════════════════════════════════════════════
    # Phase 3: K-Fold validation
    # ═════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Phase 3: K-Fold Validation (6 folds)")
    print("=" * 70, flush=True)

    kfold_results = {}
    for vname in ['A_Baseline', 'D_RuleB_CB']:
        cfg = VARIANTS[vname]
        fold_sharpes = []
        print(f"\n    K-Fold: {vname}", flush=True)

        for fold_name, start, end in FOLDS:
            fs = pd.Timestamp(start); fe = pd.Timestamp(end)
            h1_fold = h1_df[(h1_df.index >= start) & (h1_df.index < end)]
            if len(h1_fold) < 100:
                fold_sharpes.append(0.0)
                print(f"      {fold_name}: skipped (too few bars)", flush=True)
                continue

            fold_strat = {}
            fold_strat['PSAR'] = bt_psar(h1_fold, SPREAD, UNIT_LOT, CAPS['PSAR'])
            fold_strat['TSMOM'] = bt_tsmom(h1_fold, SPREAD, UNIT_LOT, CAPS['TSMOM'])
            fold_strat['SESS_BO'] = bt_sess_bo(h1_fold, SPREAD, UNIT_LOT, CAPS['SESS_BO'])
            try:
                l8_fold = l8_bundle.slice(start, end)
                fold_strat['L8_MAX'] = bt_l8_max(l8_fold, SPREAD, UNIT_LOT, CAPS['L8_MAX'])
            except Exception:
                fold_strat['L8_MAX'] = [
                    t for t in strat_trades['L8_MAX']
                    if fs <= _normalize_ts(t['exit_time']) < fe
                ]

            fold_portfolio = merge_portfolio_trades(fold_strat)
            if not fold_portfolio:
                fold_sharpes.append(0.0)
                continue

            if cfg['rule_b'] or cfg['cb']:
                fold_mask = build_extreme_mask(h1_fold, CUSUM_SIGMA, EXTREME_WINDOW)
                fold_trades, _, _ = apply_variant(
                    fold_portfolio, h1_fold, fold_mask,
                    use_rule_b=cfg['rule_b'], use_cb=cfg['cb'])
            else:
                fold_trades = fold_portfolio

            daily = trades_to_daily(fold_trades)
            s = sharpe(daily)
            fold_sharpes.append(s)
            print(f"      {fold_name}: Sharpe={s:.3f}, N={len(fold_trades)}", flush=True)

        pos = sum(1 for s in fold_sharpes if s > 0)
        kfold_results[vname] = {
            'fold_sharpes': [round(s, 3) for s in fold_sharpes],
            'positive_folds': pos,
            'mean_sharpe': round(float(np.mean(fold_sharpes)), 3),
            'pass_4of6': pos >= 4,
        }
        status = "PASS" if pos >= 4 else "FAIL"
        print(f"    {vname}: {pos}/6 positive, mean={np.mean(fold_sharpes):.3f} [{status}]",
              flush=True)

    d_wins = sum(1 for a_s, d_s in zip(
        kfold_results['A_Baseline']['fold_sharpes'],
        kfold_results['D_RuleB_CB']['fold_sharpes']) if d_s > a_s)
    print(f"\n    D wins over A in {d_wins}/6 folds", flush=True)
    kfold_results['d_wins_over_a'] = d_wins

    # ═════════════════════════════════════════════════════════════
    # Phase 4: Production-level lot test
    # ═════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Phase 4: Production-Level Lot Test")
    print("=" * 70, flush=True)

    prod_portfolio = merge_portfolio_trades(strat_trades, lot_scale=R89_LOTS)
    phase4 = {}
    for vname, cfg in VARIANTS.items():
        trades, rb_skip, cb_skip = apply_variant(
            prod_portfolio, h1_df, extreme_mask,
            use_rule_b=cfg['rule_b'], use_cb=cfg['cb'])
        st = compute_stats(trades, f"PROD_{vname}")
        st['rb_skipped'] = rb_skip; st['cb_skipped'] = cb_skip
        phase4[vname] = st
        print(f"    {vname}: Sharpe={st['sharpe']:.3f}, PnL=${st['pnl']:.1f}, "
              f"MaxDD=${st['max_dd']:.1f}", flush=True)

    print(f"\n    Production improvement (D vs A):")
    print(f"      Sharpe: {phase4['A_Baseline']['sharpe']:.3f} -> "
          f"{phase4['D_RuleB_CB']['sharpe']:.3f} "
          f"({phase4['D_RuleB_CB']['sharpe'] - phase4['A_Baseline']['sharpe']:+.3f})")
    print(f"      PnL: ${phase4['A_Baseline']['pnl']:.1f} -> "
          f"${phase4['D_RuleB_CB']['pnl']:.1f}")
    print(f"      MaxDD: ${phase4['A_Baseline']['max_dd']:.1f} -> "
          f"${phase4['D_RuleB_CB']['max_dd']:.1f}", flush=True)

    # ═════════════════════════════════════════════════════════════
    # Phase 5: Sensitivity check
    # ═════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Phase 5: Sensitivity Check")
    print("=" * 70, flush=True)

    skip_bars_grid = [4, 6, 8, 10, 12]
    pause_hours_grid = [0.5, 1.0, 1.5, 2.0]

    sensitivity = []
    print(f"\n    {'skip_bars':>10} {'pause_h':>8} {'Sharpe':>8} {'PnL':>10} "
          f"{'MaxDD':>8} {'N':>6}")
    print("    " + "-" * 60)

    for sb in skip_bars_grid:
        mask_sb = build_extreme_mask(h1_df, CUSUM_SIGMA, EXTREME_WINDOW)
        for ph in pause_hours_grid:
            trades, rb_skip, cb_skip = apply_variant(
                portfolio_all, h1_df, mask_sb,
                use_rule_b=True, use_cb=True,
                skip_bars=sb, cb_pause=ph)
            st = compute_stats(trades, f"sb{sb}_ph{ph}")
            row = {'skip_bars': sb, 'pause_hours': ph,
                   'sharpe': st['sharpe'], 'pnl': st['pnl'],
                   'max_dd': st['max_dd'], 'n': st['n'],
                   'rb_skipped': rb_skip, 'cb_skipped': cb_skip}
            sensitivity.append(row)
            print(f"    {sb:>10} {ph:>8.1f} {st['sharpe']:>8.3f} "
                  f"{st['pnl']:>10.1f} {st['max_dd']:>8.1f} {st['n']:>6}", flush=True)

    best_sens = max(sensitivity, key=lambda x: x['sharpe'])
    print(f"\n    Best sensitivity: skip_bars={best_sens['skip_bars']}, "
          f"pause_hours={best_sens['pause_hours']}, Sharpe={best_sens['sharpe']:.3f}",
          flush=True)

    # ═════════════════════════════════════════════════════════════
    # Save results
    # ═════════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    d_sharpe = phase1['D_RuleB_CB']['sharpe']
    a_sharpe = phase1['A_Baseline']['sharpe']
    d_kfold_pass = kfold_results['D_RuleB_CB']['pass_4of6']
    no_neg = not phase2['negative_interaction']

    if d_sharpe >= a_sharpe and d_kfold_pass and no_neg:
        verdict = "DEPLOY — RuleB + CB joint deployment recommended"
    elif d_sharpe >= a_sharpe and d_kfold_pass:
        verdict = "CAUTIOUS — marginal improvement with some interaction concerns"
    else:
        verdict = "HOLD — insufficient evidence for joint deployment"

    print(f"\n{'='*80}")
    print(f"  R159 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Verdict: {verdict}")
    print(f"{'='*80}", flush=True)

    output = {
        'experiment': 'R159 Rule B + Circuit Breaker Joint Deployment Test',
        'baseline': 'MaxHold=8 (already deployed)',
        'config': {
            'pv': PV, 'spread': SPREAD, 'unit_lot': UNIT_LOT,
            'r89_lots': R89_LOTS, 'caps': CAPS,
            'cusum_sigma': CUSUM_SIGMA, 'extreme_window': EXTREME_WINDOW,
            'skip_bars': SKIP_BARS, 'cb_streak': CB_STREAK,
            'cb_pause_hours': CB_PAUSE_HOURS,
        },
        'phase1_full_sample': phase1,
        'phase2_interaction': phase2,
        'phase3_kfold': kfold_results,
        'phase4_production': phase4,
        'phase5_sensitivity': sensitivity,
        'best_sensitivity': best_sens,
        'verdict': verdict,
        'elapsed_s': round(elapsed, 1),
    }

    with open(OUTPUT_DIR / "r159_results.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r159_results.json", flush=True)


if __name__ == "__main__":
    main()
