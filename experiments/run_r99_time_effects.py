#!/usr/bin/env python3
"""
R99 — Time-of-Day & Day-of-Week Effect Analysis
==================================================
Analyzes whether specific entry hours or weekdays systematically
produce better/worse results across all 4 strategies.

Steps:
  1. Run all 4 strategies (L8_MAX, PSAR, TSMOM, SESS_BO) at unit lot
  2. Group trades by entry hour (0-23) and weekday (Mon-Fri)
  3. Z-test each group's avg PnL vs overall mean
  4. Identify worst hours/days, simulate filtering them out
  5. K-Fold validation on filters that improve Sharpe by >5%
  6. Save results/r99_time_effects/r99_results.json
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r99_time_effects")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]

DAY_NAMES = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}


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


def _compute_stats(trades):
    if not trades:
        return {'n_trades': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    return {
        'n_trades': n,
        'sharpe': round(_sharpe(daily), 3),
        'pnl': round(sum(pnls), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
    }


# ═══════════════════════════════════════════════════════════════
# Strategy Backtests
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
    from backtest.runner import run_variant, LIVE_PARITY_KWARGS
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
            'entry_bar': 0,
            'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0,
        })
    return trades


# ═══════════════════════════════════════════════════════════════
# Time effect analysis
# ═══════════════════════════════════════════════════════════════

def extract_time_info(trades):
    """Add entry_hour and entry_weekday to each trade dict."""
    for t in trades:
        ts = pd.Timestamp(t['entry_time'])
        t['entry_hour'] = ts.hour
        t['entry_weekday'] = ts.dayofweek
    return trades


def z_test_group(group_pnls, overall_mean, overall_std, overall_n):
    """Two-sided Z-test: is group mean significantly different from overall mean?"""
    n = len(group_pnls)
    if n < 5 or overall_std == 0:
        return 0.0, 1.0
    group_mean = np.mean(group_pnls)
    se = overall_std / np.sqrt(n)
    z = (group_mean - overall_mean) / se
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p)


def analyze_by_group(trades, group_key, group_labels=None):
    """Analyze trades grouped by a key (hour or weekday)."""
    pnls = np.array([t['pnl'] for t in trades])
    overall_mean = float(np.mean(pnls))
    overall_std = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 0
    overall_n = len(pnls)

    groups = {}
    for t in trades:
        g = t[group_key]
        groups.setdefault(g, []).append(t['pnl'])

    results = {}
    for g in sorted(groups.keys()):
        g_pnls = groups[g]
        z, p = z_test_group(g_pnls, overall_mean, overall_std, overall_n)
        n = len(g_pnls)
        label = group_labels.get(g, str(g)) if group_labels else str(g)
        results[str(g)] = {
            'label': label,
            'count': n,
            'avg_pnl': round(float(np.mean(g_pnls)), 4),
            'total_pnl': round(float(np.sum(g_pnls)), 2),
            'win_rate': round(sum(1 for p in g_pnls if p > 0) / n * 100, 1) if n > 0 else 0,
            'z_score': round(z, 3),
            'p_value': round(p, 4),
            'significant': p < 0.05,
        }
    return results


def find_worst_hours(hour_analysis, min_trades=50):
    """Return worst 3 hours (lowest avg_pnl with n>min_trades)."""
    candidates = [(int(h), v) for h, v in hour_analysis.items()
                  if v['count'] >= min_trades]
    candidates.sort(key=lambda x: x[1]['avg_pnl'])
    return [h for h, _ in candidates[:3]]


def find_worst_day(day_analysis):
    """Return worst day if significant, else None."""
    for d, v in day_analysis.items():
        if v['significant'] and v['avg_pnl'] < 0:
            return int(d)
    candidates = [(int(d), v) for d, v in day_analysis.items()]
    candidates.sort(key=lambda x: x[1]['avg_pnl'])
    if candidates:
        worst = candidates[0]
        if worst[1]['avg_pnl'] < 0:
            return worst[0]
    return None


def filter_trades(trades, skip_hours=None, skip_days=None):
    """Filter out trades entered during specified hours/days."""
    filtered = []
    for t in trades:
        if skip_hours and t['entry_hour'] in skip_hours:
            continue
        if skip_days and t['entry_weekday'] in skip_days:
            continue
        filtered.append(t)
    return filtered


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("  R99 — Time-of-Day & Day-of-Week Effect Analysis", flush=True)
    print("=" * 80, flush=True)

    # ── Step 1: Load data & run all strategies ──
    print("\n  Loading data...", flush=True)
    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH, DataBundle

    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0].date()} ~ {h1_df.index[-1].date()})")

    print("  Loading L8_MAX DataBundle...", flush=True)
    bundle = DataBundle.load_custom()

    print("\n  Running all 4 strategies at unit lot...", flush=True)
    all_trades = {}

    print("    L8_MAX...", end=" ", flush=True)
    all_trades['L8_MAX'] = extract_time_info(bt_l8_max(bundle, SPREAD, UNIT_LOT, CAPS['L8_MAX']))
    print(f"{len(all_trades['L8_MAX'])} trades")

    print("    PSAR...", end=" ", flush=True)
    all_trades['PSAR'] = extract_time_info(bt_psar(h1_df, SPREAD, UNIT_LOT, CAPS['PSAR']))
    print(f"{len(all_trades['PSAR'])} trades")

    print("    TSMOM...", end=" ", flush=True)
    all_trades['TSMOM'] = extract_time_info(bt_tsmom(h1_df, SPREAD, UNIT_LOT, CAPS['TSMOM']))
    print(f"{len(all_trades['TSMOM'])} trades")

    print("    SESS_BO...", end=" ", flush=True)
    all_trades['SESS_BO'] = extract_time_info(bt_sess_bo(h1_df, SPREAD, UNIT_LOT, CAPS['SESS_BO']))
    print(f"{len(all_trades['SESS_BO'])} trades")

    results = {}

    # ── Step 2-3: Hourly & weekday analysis per strategy ──
    for strat_name in ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']:
        trades = all_trades[strat_name]
        if len(trades) < 50:
            print(f"\n  {strat_name}: SKIP (only {len(trades)} trades)")
            results[strat_name] = {'skip': True, 'reason': f'too_few_trades_{len(trades)}'}
            continue

        base_stats = _compute_stats(trades)
        print(f"\n{'='*70}", flush=True)
        print(f"  {strat_name}: {base_stats['n_trades']} trades, "
              f"Sharpe={base_stats['sharpe']:.3f}, WR={base_stats['wr']:.1f}%", flush=True)
        print(f"{'='*70}", flush=True)

        # Hour analysis
        print(f"\n    Hour-of-Day Analysis:", flush=True)
        hour_analysis = analyze_by_group(trades, 'entry_hour')
        print(f"    {'Hour':>6}  {'Count':>6}  {'AvgPnL':>8}  {'WR%':>6}  {'Z':>7}  {'p':>7}  Sig")
        print(f"    {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*7}  ---")
        for h in sorted(hour_analysis.keys(), key=int):
            v = hour_analysis[h]
            sig = " ***" if v['significant'] else ""
            print(f"    {h:>6}  {v['count']:>6}  {v['avg_pnl']:>8.3f}  {v['win_rate']:>5.1f}%  "
                  f"{v['z_score']:>7.2f}  {v['p_value']:>7.4f}{sig}")

        # Day analysis
        print(f"\n    Day-of-Week Analysis:", flush=True)
        day_analysis = analyze_by_group(trades, 'entry_weekday', DAY_NAMES)
        print(f"    {'Day':>6}  {'Count':>6}  {'AvgPnL':>8}  {'WR%':>6}  {'Z':>7}  {'p':>7}  Sig")
        print(f"    {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*7}  ---")
        for d in sorted(day_analysis.keys(), key=int):
            v = day_analysis[d]
            sig = " ***" if v['significant'] else ""
            print(f"    {v['label']:>6}  {v['count']:>6}  {v['avg_pnl']:>8.3f}  {v['win_rate']:>5.1f}%  "
                  f"{v['z_score']:>7.2f}  {v['p_value']:>7.4f}{sig}")

        # Combined filter test
        worst_hours = find_worst_hours(hour_analysis, min_trades=50)
        worst_day = find_worst_day(day_analysis)

        skip_hours = worst_hours if worst_hours else []
        skip_days = [worst_day] if worst_day is not None else []

        filtered = filter_trades(trades, skip_hours=skip_hours, skip_days=skip_days)
        filt_stats = _compute_stats(filtered)

        sharpe_change_pct = ((filt_stats['sharpe'] - base_stats['sharpe']) / base_stats['sharpe'] * 100
                             if base_stats['sharpe'] != 0 else 0)

        print(f"\n    Combined filter: skip hours={skip_hours}, skip days={[DAY_NAMES.get(d, d) for d in skip_days]}")
        print(f"      Before: {base_stats['n_trades']} trades, Sharpe={base_stats['sharpe']:.3f}")
        print(f"      After:  {filt_stats['n_trades']} trades, Sharpe={filt_stats['sharpe']:.3f} ({sharpe_change_pct:+.1f}%)")

        strat_result = {
            'baseline': base_stats,
            'hour_analysis': hour_analysis,
            'day_analysis': day_analysis,
            'worst_hours': worst_hours,
            'worst_day': worst_day,
            'filtered_stats': filt_stats,
            'sharpe_change_pct': round(sharpe_change_pct, 1),
            'filter_improves_5pct': sharpe_change_pct > 5,
        }

        # K-Fold if filter improves Sharpe by >5%
        if sharpe_change_pct > 5:
            print(f"\n    K-Fold validation (filter improves by {sharpe_change_pct:.1f}%):", flush=True)
            fold_sharpes_base = []
            fold_sharpes_filt = []

            for fold_name, fold_start, fold_end in FOLDS:
                h1_fold = h1_df[(h1_df.index >= fold_start) & (h1_df.index < fold_end)]
                if len(h1_fold) < 100:
                    fold_sharpes_base.append(0.0)
                    fold_sharpes_filt.append(0.0)
                    continue

                if strat_name == 'L8_MAX':
                    fold_bundle = bundle.slice(fold_start, fold_end) if hasattr(bundle, 'slice') else bundle
                    try:
                        fold_trades = extract_time_info(bt_l8_max(fold_bundle, SPREAD, UNIT_LOT, CAPS['L8_MAX']))
                    except Exception:
                        fold_sharpes_base.append(0.0)
                        fold_sharpes_filt.append(0.0)
                        continue
                elif strat_name == 'PSAR':
                    fold_trades = extract_time_info(bt_psar(h1_fold, SPREAD, UNIT_LOT, CAPS['PSAR']))
                elif strat_name == 'TSMOM':
                    fold_trades = extract_time_info(bt_tsmom(h1_fold, SPREAD, UNIT_LOT, CAPS['TSMOM']))
                elif strat_name == 'SESS_BO':
                    fold_trades = extract_time_info(bt_sess_bo(h1_fold, SPREAD, UNIT_LOT, CAPS['SESS_BO']))
                else:
                    continue

                daily_base = _trades_to_daily(fold_trades)
                fold_sharpes_base.append(_sharpe(daily_base))

                fold_filtered = filter_trades(fold_trades, skip_hours=skip_hours, skip_days=skip_days)
                daily_filt = _trades_to_daily(fold_filtered)
                fold_sharpes_filt.append(_sharpe(daily_filt))

            filt_better = sum(1 for b, f in zip(fold_sharpes_base, fold_sharpes_filt) if f > b)
            filt_positive = sum(1 for s in fold_sharpes_filt if s > 0)

            kfold = {
                'base_fold_sharpes': [round(s, 3) for s in fold_sharpes_base],
                'filtered_fold_sharpes': [round(s, 3) for s in fold_sharpes_filt],
                'folds_filter_better': filt_better,
                'folds_positive': filt_positive,
                'pass_4of6': filt_positive >= 4,
                'mean_base': round(float(np.mean(fold_sharpes_base)), 3),
                'mean_filt': round(float(np.mean(fold_sharpes_filt)), 3),
            }
            strat_result['kfold'] = kfold

            print(f"      Base folds:     {[f'{s:.3f}' for s in fold_sharpes_base]}")
            print(f"      Filtered folds: {[f'{s:.3f}' for s in fold_sharpes_filt]}")
            print(f"      Filter better in {filt_better}/6 folds, positive in {filt_positive}/6")
        else:
            print(f"\n    K-Fold: SKIP (Sharpe change {sharpe_change_pct:+.1f}% < 5%)")

        results[strat_name] = strat_result

    # ── Summary ──
    elapsed = time.time() - t0
    print(f"\n\n{'='*80}", flush=True)
    print(f"  R99 SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)

    print(f"\n  {'Strategy':<10} {'Trades':>7} {'BaseShp':>8} {'FiltShp':>8} {'Change':>8} {'Worst Hours':<20} {'Worst Day':<10}")
    print(f"  {'-'*10} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*20} {'-'*10}")
    for strat in ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']:
        r = results.get(strat, {})
        if r.get('skip'):
            print(f"  {strat:<10} {'SKIP':>7}")
            continue
        bl = r['baseline']
        fl = r['filtered_stats']
        wh = str(r['worst_hours'])
        wd = DAY_NAMES.get(r['worst_day'], 'None') if r['worst_day'] is not None else 'None'
        chg = r['sharpe_change_pct']
        print(f"  {strat:<10} {bl['n_trades']:>7} {bl['sharpe']:>8.3f} {fl['sharpe']:>8.3f} {chg:>+7.1f}% {wh:<20} {wd:<10}")

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"{'='*80}", flush=True)

    output = {
        'experiment': 'R99 Time-of-Day & Day-of-Week Effect Analysis',
        'elapsed_s': round(elapsed, 1),
        'strategies': results,
    }
    with open(OUTPUT_DIR / "r99_results.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r99_results.json", flush=True)


if __name__ == "__main__":
    main()
