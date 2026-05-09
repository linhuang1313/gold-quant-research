#!/usr/bin/env python3
"""
R134 — Overnight Gap + Session Momentum Deep Research
======================================================
Phases:
  1. Load H1 data, compute overnight gaps (day open vs prev day close)
  2. Statistical analysis: gap size distribution by day of week, direction win rate
  3. Session momentum: Asia (0-8 UTC), London (8-14), NY (14-22) first-2h momentum
  4. Standalone strategy: gap > threshold*ATR14 AND session momentum confirms
  5. Parameter sweep: gap_threshold, confirmation_hours, SL/TP grid
  6. K-Fold 5-fold validation
  7. Walk-Forward (4yr train / 2yr test, 5 windows)
  8. Correlation with existing strategies

Estimated ~10h runtime.
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

from backtest.runner import load_csv, DataBundle, run_variant, LIVE_PARITY_KWARGS

OUTPUT_DIR = Path("results/r134_overnight_session")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01

H1_CANDIDATES = [
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-27.csv"),
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv"),
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-03-25.csv"),
]

FOLDS = [
    ("Fold1", "2015-01-01", "2017-03-01"),
    ("Fold2", "2017-03-01", "2019-06-01"),
    ("Fold3", "2019-06-01", "2021-09-01"),
    ("Fold4", "2021-09-01", "2023-12-01"),
    ("Fold5", "2023-12-01", "2027-01-01"),
]

WF_WINDOWS = [
    ("WF1", "2015-01-01", "2019-01-01", "2019-01-01", "2021-01-01"),
    ("WF2", "2016-01-01", "2020-01-01", "2020-01-01", "2022-01-01"),
    ("WF3", "2017-01-01", "2021-01-01", "2021-01-01", "2023-01-01"),
    ("WF4", "2018-01-01", "2022-01-01", "2022-01-01", "2024-01-01"),
    ("WF5", "2019-01-01", "2023-01-01", "2023-01-01", "2025-01-01"),
]

SESSIONS = {
    'Asia':   (0, 8),
    'London': (8, 14),
    'NY':     (14, 22),
}

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
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1) if n else 0,
        'max_dd': round(dd, 2),
        'calmar': round(pnl / dd, 2) if dd > 0 else 9999,
    }


# ═══════════════════════════════════════════════════════════════
# Data preparation: overnight gaps + session momentum
# ═══════════════════════════════════════════════════════════════

def prepare_gap_data(df):
    """Compute daily-level gap and session momentum features from H1 bars."""
    df = df.copy()
    df['ATR14'] = compute_atr(df, 14)
    df['date'] = df.index.date
    df['hour'] = df.index.hour

    daily_open = df.groupby('date')['Open'].first()
    daily_close = df.groupby('date')['Close'].last()
    daily_high = df.groupby('date')['High'].max()
    daily_low = df.groupby('date')['Low'].min()
    daily_atr = df.groupby('date')['ATR14'].last()

    daily = pd.DataFrame({
        'open': daily_open, 'close': daily_close,
        'high': daily_high, 'low': daily_low, 'atr': daily_atr,
    })
    daily['prev_close'] = daily['close'].shift(1)
    daily['gap'] = daily['open'] - daily['prev_close']
    daily['gap_pct_atr'] = daily['gap'] / daily['atr'].replace(0, np.nan)
    daily['dow'] = pd.DatetimeIndex(daily.index).dayofweek
    daily = daily.dropna(subset=['gap', 'atr'])

    sess_mom = {}
    for name, (start_h, end_h) in SESSIONS.items():
        mask = (df['hour'] >= start_h) & (df['hour'] < end_h)
        sess = df[mask].copy()
        first_2h_mask = (df['hour'] >= start_h) & (df['hour'] < start_h + 2)
        first_2h = df[first_2h_mask].copy()
        sess_close = first_2h.groupby(first_2h.index.date)['Close'].last()
        sess_open = first_2h.groupby(first_2h.index.date)['Open'].first()
        mom = sess_close - sess_open
        sess_mom[f'{name}_mom'] = mom

    for key, val in sess_mom.items():
        daily[key] = val

    return daily, df


def backtest_gap_session(h1_df, daily, gap_threshold, confirm_hours,
                         sl_mult, tp_mult, max_hold, trail_act, trail_dist,
                         lot=UNIT_LOT, spread=SPREAD, cap=35,
                         start=None, end=None):
    """Backtest: enter in gap direction when gap > threshold*ATR and session momentum confirms."""
    df = h1_df.copy()
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr_arr = df['ATR14'].values if 'ATR14' in df.columns else compute_atr(df).values
    times = df.index; hours = df.index.hour; n = len(df)
    dates = df.index.date

    daily_dict = daily.to_dict('index')
    trades = []; pos = None; last_exit = -999

    for i in range(1, n):
        if start and str(dates[i]) < start:
            continue
        if end and str(dates[i]) > end:
            break

        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_mult, tp_mult, trail_act, trail_dist, max_hold, cap)
            if result:
                trades.append(result); pos = None; last_exit = i
            continue

        if i - last_exit < 2:
            continue
        if np.isnan(atr_arr[i]) or atr_arr[i] < 0.1:
            continue

        cur_date = dates[i]
        if cur_date not in daily_dict:
            continue
        d = daily_dict[cur_date]

        gap = d.get('gap', 0)
        atr_val = d.get('atr', 0)
        if atr_val <= 0:
            continue

        gap_ratio = abs(gap) / atr_val
        if gap_ratio < gap_threshold:
            continue

        sess_name = None
        hr = hours[i]
        for sn, (sh, eh) in SESSIONS.items():
            if sh <= hr < eh:
                sess_name = sn
                break
        if sess_name is None:
            continue

        if hr < SESSIONS[sess_name][0] + confirm_hours:
            continue

        mom_key = f'{sess_name}_mom'
        mom = d.get(mom_key, 0)
        if pd.isna(mom):
            continue

        gap_dir = 1 if gap > 0 else -1
        mom_dir = 1 if mom > 0 else -1

        if gap_dir != mom_dir:
            continue

        if gap_dir == 1:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i,
                   'time': times[i], 'atr': atr_arr[i]}
        else:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i,
                   'time': times[i], 'atr': atr_arr[i]}

    return trades


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80, flush=True)
    print("  R134 — Overnight Gap + Session Momentum Deep Research", flush=True)
    print("=" * 80, flush=True)

    # ── Load Data ──
    csv_path = next((p for p in H1_CANDIDATES if p.exists()), H1_CANDIDATES[0])
    print(f"\n  Loading H1: {csv_path}", flush=True)
    h1_raw = load_csv(str(csv_path))
    h1_raw['ATR14'] = compute_atr(h1_raw, 14)
    h1_raw = h1_raw.dropna(subset=['ATR14'])
    print(f"  H1 loaded: {len(h1_raw)} bars ({h1_raw.index[0]} → {h1_raw.index[-1]})", flush=True)

    daily, h1 = prepare_gap_data(h1_raw)
    print(f"  Daily data: {len(daily)} days", flush=True)

    all_results = {
        'experiment': 'R134 Overnight Gap + Session Momentum',
        'data_bars': len(h1_raw), 'daily_days': len(daily),
    }

    # ════════════════════════════════════════════════════════════════
    # Phase 1-2: Statistical analysis of overnight gaps
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 1-2: Overnight Gap Statistics", flush=True)
    print("=" * 70, flush=True)

    gap_valid = daily.dropna(subset=['gap', 'gap_pct_atr'])
    print(f"\n  Gaps: {len(gap_valid)} days with valid gap data", flush=True)
    print(f"  Gap mean: ${gap_valid['gap'].mean():.2f}, std: ${gap_valid['gap'].std():.2f}", flush=True)
    print(f"  Gap/ATR mean: {gap_valid['gap_pct_atr'].mean():.3f}, std: {gap_valid['gap_pct_atr'].std():.3f}", flush=True)

    dow_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
    print(f"\n  Gap by day of week:", flush=True)
    print(f"  {'Day':<6s}  {'Count':>6s}  {'MeanGap':>9s}  {'StdGap':>9s}  {'Mean|Gap|':>10s}  {'Gap>0%':>7s}", flush=True)
    print("  " + "-" * 55, flush=True)

    phase1_stats = {}
    for dow in range(5):
        subset = gap_valid[gap_valid['dow'] == dow]
        if len(subset) < 10:
            continue
        mg = subset['gap'].mean()
        sg = subset['gap'].std()
        mag = subset['gap'].abs().mean()
        pos_pct = (subset['gap'] > 0).mean() * 100
        day_name = dow_names.get(dow, str(dow))
        print(f"  {day_name:<6s}  {len(subset):6d}  ${mg:>8.2f}  ${sg:>8.2f}  ${mag:>9.2f}  {pos_pct:6.1f}%", flush=True)
        phase1_stats[day_name] = {
            'count': len(subset), 'mean_gap': round(mg, 3),
            'std_gap': round(sg, 3), 'mean_abs_gap': round(mag, 3),
            'pct_positive': round(pos_pct, 1),
        }

    gap_up = gap_valid[gap_valid['gap'] > 0]
    gap_dn = gap_valid[gap_valid['gap'] < 0]
    up_win = (gap_up['close'] > gap_up['open']).mean() * 100
    dn_win = (gap_dn['close'] < gap_dn['open']).mean() * 100
    print(f"\n  Gap-up days: {len(gap_up)}, close > open (follow-through): {up_win:.1f}%", flush=True)
    print(f"  Gap-down days: {len(gap_dn)}, close < open (follow-through): {dn_win:.1f}%", flush=True)

    all_results['phase1_gap_stats'] = phase1_stats
    all_results['phase1_gap_winrate'] = {
        'gap_up_follow': round(up_win, 1), 'gap_down_follow': round(dn_win, 1),
    }

    # ════════════════════════════════════════════════════════════════
    # Phase 3: Session momentum analysis
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 3: Session Momentum (first 2h)", flush=True)
    print("=" * 70, flush=True)

    phase3_stats = {}
    for sess_name in SESSIONS:
        key = f'{sess_name}_mom'
        if key not in daily.columns:
            continue
        vals = daily[key].dropna()
        if len(vals) < 50:
            continue
        mn = vals.mean(); sd = vals.std()
        pos_pct = (vals > 0).mean() * 100
        print(f"  {sess_name:<8s}: n={len(vals)}, mean=${mn:.3f}, std=${sd:.3f}, pos%={pos_pct:.1f}%", flush=True)
        phase3_stats[sess_name] = {
            'n': len(vals), 'mean': round(mn, 4), 'std': round(sd, 4),
            'pct_positive': round(pos_pct, 1),
        }

    gap_mom_corr = {}
    for sess_name in SESSIONS:
        key = f'{sess_name}_mom'
        if key in daily.columns:
            valid = daily[['gap', key]].dropna()
            if len(valid) > 30:
                c_val = valid['gap'].corr(valid[key])
                gap_mom_corr[sess_name] = round(c_val, 4)
                print(f"  Corr(gap, {sess_name}_mom): {c_val:.4f}", flush=True)

    all_results['phase3_session_momentum'] = phase3_stats
    all_results['phase3_gap_session_corr'] = gap_mom_corr

    # ════════════════════════════════════════════════════════════════
    # Phase 4-5: Strategy backtest + parameter sweep
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 4-5: Parameter Sweep", flush=True)
    print("=" * 70, flush=True)

    gap_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    confirm_hours_list = [1, 2, 3]
    sl_mults = [2.0, 3.0, 4.0]
    tp_mults = [4.0, 6.0, 8.0]
    max_hold_val = 20
    trail_act_val = 0.14
    trail_dist_val = 0.025

    grid_results = []
    total = len(gap_thresholds) * len(confirm_hours_list) * len(sl_mults) * len(tp_mults)
    print(f"  Grid size: {total} combos", flush=True)
    print(f"\n  {'Config':<42s}  {'n':>5s}  {'Sharpe':>7s}  {'PnL':>10s}  {'WR':>6s}  {'MaxDD':>8s}", flush=True)
    print("  " + "-" * 85, flush=True)

    done = 0
    for gt, ch, sl_m, tp_m in product(gap_thresholds, confirm_hours_list, sl_mults, tp_mults):
        trades = backtest_gap_session(
            h1, daily, gap_threshold=gt, confirm_hours=ch,
            sl_mult=sl_m, tp_mult=tp_m, max_hold=max_hold_val,
            trail_act=trail_act_val, trail_dist=trail_dist_val,
        )
        st = _compute_stats(trades)
        label = f"gap={gt}/ch={ch}/sl={sl_m}/tp={tp_m}"
        grid_results.append({'gap_threshold': gt, 'confirm_hours': ch,
                             'sl_mult': sl_m, 'tp_mult': tp_m, **st})
        if st['sharpe'] > 0.3 and st['n'] >= 20:
            print(f"  {label:<42s}  {st['n']:5d}  {st['sharpe']:7.3f}  ${st['pnl']:>9.0f}  "
                  f"{st['wr']:5.1f}%  ${st['max_dd']:>7.0f}", flush=True)
        done += 1
        if done % 50 == 0:
            print(f"  ... {done}/{total} done ({time.time()-t0:.0f}s)", flush=True)

    grid_results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  Top 10 by Sharpe:", flush=True)
    for i, g in enumerate(grid_results[:10]):
        print(f"    #{i+1}: gap={g['gap_threshold']}/ch={g['confirm_hours']}/"
              f"sl={g['sl_mult']}/tp={g['tp_mult']} -> Sharpe={g['sharpe']:.3f}, "
              f"n={g['n']}, PnL=${g['pnl']:.0f}, WR={g['wr']:.1f}%", flush=True)

    all_results['phase5_param_sweep'] = grid_results[:30]

    if not grid_results or grid_results[0]['n'] < 5:
        print("\n  WARNING: No viable parameter set found. Skipping phases 6-8.", flush=True)
        best = {'gap_threshold': 0.5, 'confirm_hours': 2, 'sl_mult': 3.0, 'tp_mult': 6.0}
    else:
        best = grid_results[0]

    print(f"\n  Best params: gap={best['gap_threshold']}, ch={best['confirm_hours']}, "
          f"sl={best['sl_mult']}, tp={best['tp_mult']}", flush=True)

    # ════════════════════════════════════════════════════════════════
    # Phase 6: K-Fold 5-fold validation
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 6: K-Fold 5-Fold Validation", flush=True)
    print("=" * 70, flush=True)

    kfold_results = []
    print(f"\n  {'Fold':<8s}  {'n':>5s}  {'Sharpe':>7s}  {'PnL':>10s}  {'WR':>6s}  {'MaxDD':>8s}", flush=True)
    print("  " + "-" * 50, flush=True)

    for fold_name, fold_start, fold_end in FOLDS:
        trades = backtest_gap_session(
            h1, daily, gap_threshold=best['gap_threshold'],
            confirm_hours=best['confirm_hours'],
            sl_mult=best['sl_mult'], tp_mult=best['tp_mult'],
            max_hold=max_hold_val, trail_act=trail_act_val,
            trail_dist=trail_dist_val, start=fold_start, end=fold_end,
        )
        st = _compute_stats(trades)
        kfold_results.append({'fold': fold_name, **st})
        print(f"  {fold_name:<8s}  {st['n']:5d}  {st['sharpe']:7.3f}  ${st['pnl']:>9.0f}  "
              f"{st['wr']:5.1f}%  ${st['max_dd']:>7.0f}", flush=True)

    pos_folds = sum(1 for f in kfold_results if f['sharpe'] > 0)
    print(f"\n  Positive folds: {pos_folds}/{len(kfold_results)}", flush=True)
    avg_sharpe = np.mean([f['sharpe'] for f in kfold_results])
    print(f"  Average fold Sharpe: {avg_sharpe:.3f}", flush=True)

    all_results['phase6_kfold'] = kfold_results
    all_results['phase6_summary'] = {
        'positive_folds': pos_folds, 'total_folds': len(kfold_results),
        'avg_sharpe': round(avg_sharpe, 3),
    }

    # ════════════════════════════════════════════════════════════════
    # Phase 7: Walk-Forward (4yr train / 2yr test, 5 windows)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 7: Walk-Forward Validation (4yr train / 2yr test)", flush=True)
    print("=" * 70, flush=True)

    wf_results = []
    print(f"\n  {'Window':<6s}  {'Train Sharpe':>12s}  {'Test Sharpe':>12s}  {'Test n':>6s}  "
          f"{'Test PnL':>10s}  {'Test WR':>7s}", flush=True)
    print("  " + "-" * 65, flush=True)

    for wf_name, tr_start, tr_end, te_start, te_end in WF_WINDOWS:
        train_grid = []
        for gt, ch in product([0.3, 0.5, 0.7, 1.0], [1, 2, 3]):
            tr_trades = backtest_gap_session(
                h1, daily, gap_threshold=gt, confirm_hours=ch,
                sl_mult=best['sl_mult'], tp_mult=best['tp_mult'],
                max_hold=max_hold_val, trail_act=trail_act_val,
                trail_dist=trail_dist_val, start=tr_start, end=tr_end,
            )
            st = _compute_stats(tr_trades)
            train_grid.append({'gap_threshold': gt, 'confirm_hours': ch, **st})

        train_grid.sort(key=lambda x: x['sharpe'], reverse=True)
        wf_best = train_grid[0] if train_grid else best

        te_trades = backtest_gap_session(
            h1, daily, gap_threshold=wf_best['gap_threshold'],
            confirm_hours=wf_best['confirm_hours'],
            sl_mult=best['sl_mult'], tp_mult=best['tp_mult'],
            max_hold=max_hold_val, trail_act=trail_act_val,
            trail_dist=trail_dist_val, start=te_start, end=te_end,
        )
        te_st = _compute_stats(te_trades)

        wf_results.append({
            'window': wf_name, 'train_sharpe': wf_best.get('sharpe', 0),
            'train_params': {'gap_threshold': wf_best['gap_threshold'],
                             'confirm_hours': wf_best['confirm_hours']},
            'test': te_st,
        })
        print(f"  {wf_name:<6s}  {wf_best.get('sharpe', 0):12.3f}  {te_st['sharpe']:12.3f}  "
              f"{te_st['n']:6d}  ${te_st['pnl']:>9.0f}  {te_st['wr']:6.1f}%", flush=True)

    wf_oos_sharpes = [w['test']['sharpe'] for w in wf_results]
    wf_pos = sum(1 for s in wf_oos_sharpes if s > 0)
    print(f"\n  OOS positive: {wf_pos}/{len(wf_results)}, avg OOS Sharpe: {np.mean(wf_oos_sharpes):.3f}", flush=True)

    all_results['phase7_walk_forward'] = wf_results

    # ════════════════════════════════════════════════════════════════
    # Phase 8: Correlation with existing strategies
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 8: Correlation with Existing Strategies", flush=True)
    print("=" * 70, flush=True)

    r134_trades = backtest_gap_session(
        h1, daily, gap_threshold=best['gap_threshold'],
        confirm_hours=best['confirm_hours'],
        sl_mult=best['sl_mult'], tp_mult=best['tp_mult'],
        max_hold=max_hold_val, trail_act=trail_act_val,
        trail_dist=trail_dist_val,
    )
    r134_daily = _trades_to_daily(r134_trades)

    try:
        bundle = DataBundle.load_default()
        existing_strats = {}
        for strat_name in ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']:
            try:
                result = run_variant(bundle, strat_name, verbose=False, **LIVE_PARITY_KWARGS)
                raw_trades = result.get('_trades', [])
                strat_trades = [{'pnl': t.pnl, 'exit_time': t.exit_time} for t in raw_trades]
                existing_strats[strat_name] = _trades_to_daily(strat_trades)
            except Exception as e:
                print(f"    WARNING: {strat_name} failed: {e}", flush=True)

        corr_results = {}
        r134_series = pd.Series(r134_daily)
        for name, other_daily in existing_strats.items():
            min_len = min(len(r134_daily), len(other_daily))
            if min_len < 20:
                continue
            c_val = np.corrcoef(r134_daily[:min_len], other_daily[:min_len])[0, 1]
            corr_results[name] = round(c_val, 4)
            print(f"  Corr(R134, {name}): {c_val:.4f}", flush=True)

        all_results['phase8_correlation'] = corr_results
    except Exception as e:
        print(f"  WARNING: Could not load existing strategies: {e}", flush=True)
        all_results['phase8_correlation'] = {'error': str(e)}

    # ═══════════════════════════════════════════════════════════════
    # Save
    # ═══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    all_results['runtime_sec'] = round(elapsed, 1)
    out_file = OUTPUT_DIR / "r134_results.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*80}", flush=True)
    print(f"  R134 complete in {elapsed/60:.1f} min. Results → {out_file}", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == '__main__':
    main()
