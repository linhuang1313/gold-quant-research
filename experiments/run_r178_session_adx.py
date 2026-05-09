#!/usr/bin/env python3
"""
R178 — Session-Adaptive ADX Threshold for Keltner Channel
===========================================================
Tests whether adapting the ADX entry threshold by trading session improves
Sharpe for XAUUSD H1. London/NY open = lower threshold (reliable trends),
Asia session = higher threshold (less trend reliability).

Phases:
  1. Per-hour profiling with flat ADX=14
  2. Session-adaptive ADX grid search
  3. Top-10 K-Fold 6-fold validation
  4. Lot boosting by session for best ADX combo
"""
import sys, os, time, json, warnings, itertools, glob
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r178_session_adx")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01

SESSION_GROUPS = {
    'asia':    set(range(22, 24)) | set(range(0, 7)),
    'london':  set(range(7, 12)),
    'ny':      set(range(12, 17)),
    'overlap': set(range(13, 16)),
    'evening': set(range(17, 22)),
}

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]


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


def _max_dd(arr):
    if len(arr) == 0:
        return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def _compute_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0.0, 'pnl': 0.0, 'wr': 0.0, 'max_dd': 0.0, 'avg_pnl': 0.0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    return {
        'n': n,
        'sharpe': round(_sharpe(daily), 3),
        'pnl': round(sum(pnls), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'max_dd': round(_max_dd(daily), 2),
        'avg_pnl': round(float(np.mean(pnls)), 3),
    }


def load_h1():
    candidates = sorted(glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
    csv_path = candidates[-1]
    print(f"  Loading: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.set_index('time')[['open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return df


def prepare_kc(df, ema_period=25, atr_period=14, kc_mult=1.2):
    out = df.copy()
    out['ATR'] = compute_atr(out, atr_period)
    out['ADX'] = compute_adx(out, atr_period)
    out['EMA100'] = out['Close'].ewm(span=100, adjust=False).mean()
    ema_kc = out['Close'].ewm(span=ema_period, adjust=False).mean()
    out['KC_upper'] = ema_kc + kc_mult * out['ATR']
    out['KC_lower'] = ema_kc - kc_mult * out['ATR']
    return out


def hour_to_session(h):
    for name, hours in SESSION_GROUPS.items():
        if name == 'overlap':
            continue
        if h in hours:
            return name
    return 'unknown'


def bt_keltner_session_adx(df_prepared, spread, lot, hour_adx_map,
                           sl_atr=3.5, tp_atr=8.0, trail_act=0.14,
                           trail_dist=0.025, max_hold=20, maxloss_cap=35,
                           lot_map=None):
    c = df_prepared['Close'].values; h = df_prepared['High'].values
    lo = df_prepared['Low'].values
    atr = df_prepared['ATR'].values; adx = df_prepared['ADX'].values
    kc_up = df_prepared['KC_upper'].values; kc_lo = df_prepared['KC_lower'].values
    ema100 = df_prepared['EMA100'].values
    times = df_prepared.index; hours = df_prepared.index.hour
    n = len(df_prepared)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            use_lot = pos.get('lot', lot)
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, use_lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
        if np.isnan(adx[i]):
            continue
        hr = hours[i]
        adx_th = hour_adx_map.get(hr, 14)
        if adx[i] < adx_th:
            continue
        direction = None
        if c[i] > kc_up[i] and c[i] > ema100[i]:
            direction = 'BUY'
        elif c[i] < kc_lo[i] and c[i] < ema100[i]:
            direction = 'SELL'
        if direction is None:
            continue
        entry_lot = lot
        if lot_map is not None:
            entry_lot = lot_map.get(hr, lot)
        pos = {'dir': direction, 'entry': c[i] + (spread / 2 if direction == 'BUY' else -spread / 2),
               'bar': i, 'time': times[i], 'atr': atr[i], 'lot': entry_lot}
    return trades


def build_hour_adx_map(asia_adx, london_adx, ny_adx, evening_adx):
    m = {}
    for hr in SESSION_GROUPS['asia']:
        m[hr] = asia_adx
    for hr in SESSION_GROUPS['london']:
        m[hr] = london_adx
    for hr in SESSION_GROUPS['ny']:
        m[hr] = ny_adx
    for hr in SESSION_GROUPS['evening']:
        m[hr] = evening_adx
    return m


def main():
    t0 = time.time()
    print("=" * 100, flush=True)
    print("  R178 — Session-Adaptive ADX Threshold for Keltner Channel", flush=True)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 100, flush=True)

    h1_df = load_h1()
    print(f"  {len(h1_df)} bars, {h1_df.index[0]} ~ {h1_df.index[-1]}", flush=True)

    print("\n  Preparing indicators...", flush=True)
    df_kc = prepare_kc(h1_df)

    results = {}

    # ══════════════════════════════════════════════════════════
    # Phase 1: Per-hour profiling with flat ADX=14
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("  Phase 1: Per-Hour Profiling (flat ADX=14)", flush=True)
    print(f"{'='*100}", flush=True)

    flat_map = {hr: 14 for hr in range(24)}
    all_trades = bt_keltner_session_adx(df_kc, SPREAD, UNIT_LOT, flat_map)
    print(f"\n  Total trades (ADX=14 flat): {len(all_trades)}", flush=True)

    hour_stats = {}
    print(f"\n  {'Hour':>4} {'Sess':>7} {'#':>5} {'WR%':>6} {'AvgPnL':>9} {'TotalPnL':>10} {'Sharpe':>7}", flush=True)
    print(f"  {'-'*4} {'-'*7} {'-'*5} {'-'*6} {'-'*9} {'-'*10} {'-'*7}", flush=True)

    for hr in range(24):
        hr_trades = [t for t in all_trades if pd.Timestamp(t['entry_time']).hour == hr]
        s = _compute_stats(hr_trades)
        sess = hour_to_session(hr)
        hour_stats[str(hr)] = {**s, 'session': sess}
        if s['n'] > 0:
            print(f"  {hr:>4} {sess:>7} {s['n']:>5} {s['wr']:>5.1f}% {s['avg_pnl']:>9.3f} "
                  f"${s['pnl']:>9,.0f} {s['sharpe']:>7.2f}", flush=True)
        else:
            print(f"  {hr:>4} {sess:>7}     0      -         -          -       -", flush=True)

    for sess_name, sess_hours in SESSION_GROUPS.items():
        if sess_name == 'overlap':
            continue
        sess_trades = [t for t in all_trades if pd.Timestamp(t['entry_time']).hour in sess_hours]
        s = _compute_stats(sess_trades)
        print(f"\n  Session {sess_name:>8}: n={s['n']:>4}, WR={s['wr']:.1f}%, "
              f"Sharpe={s['sharpe']:.2f}, PnL=${s['pnl']:,.0f}", flush=True)

    results['phase1_hour_stats'] = hour_stats

    # ══════════════════════════════════════════════════════════
    # Phase 2: Session-Adaptive ADX Grid Search
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("  Phase 2: Session-Adaptive ADX Grid Search", flush=True)
    print(f"{'='*100}", flush=True)

    asia_grid = [16, 18, 20, 22, 25, 999]
    london_grid = [10, 12, 14, 16]
    ny_grid = [10, 12, 14, 16]
    evening_grid = [14, 16, 18, 20, 999]

    combos = list(itertools.product(asia_grid, london_grid, ny_grid, evening_grid))
    print(f"\n  Grid: {len(combos)} combinations", flush=True)
    print(f"  Asia: {asia_grid}", flush=True)
    print(f"  London: {london_grid}", flush=True)
    print(f"  NY: {ny_grid}", flush=True)
    print(f"  Evening: {evening_grid}", flush=True)

    grid_results = []
    for idx, (a_adx, l_adx, n_adx, e_adx) in enumerate(combos):
        if (idx + 1) % 100 == 0:
            print(f"    {idx+1}/{len(combos)}...", flush=True)
        hour_map = build_hour_adx_map(a_adx, l_adx, n_adx, e_adx)
        trades = bt_keltner_session_adx(df_kc, SPREAD, UNIT_LOT, hour_map)
        s = _compute_stats(trades)
        grid_results.append({
            'asia_adx': a_adx, 'london_adx': l_adx, 'ny_adx': n_adx, 'evening_adx': e_adx,
            **s,
        })

    grid_results.sort(key=lambda x: x['sharpe'], reverse=True)

    baseline_stats = _compute_stats(all_trades)
    print(f"\n  Baseline (flat ADX=14): Sharpe={baseline_stats['sharpe']:.3f}, "
          f"n={baseline_stats['n']}, PnL=${baseline_stats['pnl']:,.0f}", flush=True)

    print(f"\n  Top 20 Combinations:", flush=True)
    print(f"  {'Rank':>4} {'Asia':>5} {'Lon':>5} {'NY':>5} {'Eve':>5} "
          f"{'#':>5} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'MaxDD':>8}", flush=True)
    print(f"  {'-'*4} {'-'*5} {'-'*5} {'-'*5} {'-'*5} "
          f"{'-'*5} {'-'*7} {'-'*10} {'-'*6} {'-'*8}", flush=True)

    for rank, r in enumerate(grid_results[:20], 1):
        a_str = 'OFF' if r['asia_adx'] == 999 else str(r['asia_adx'])
        e_str = 'OFF' if r['evening_adx'] == 999 else str(r['evening_adx'])
        print(f"  {rank:>4} {a_str:>5} {r['london_adx']:>5} {r['ny_adx']:>5} {e_str:>5} "
              f"{r['n']:>5} {r['sharpe']:>7.3f} ${r['pnl']:>9,.0f} {r['wr']:>5.1f}% "
              f"${r['max_dd']:>7,.0f}", flush=True)

    results['phase2_top20'] = grid_results[:20]
    results['phase2_baseline'] = baseline_stats

    # ══════════════════════════════════════════════════════════
    # Phase 3: Top-10 K-Fold 6-Fold Validation
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("  Phase 3: K-Fold 6-Fold Validation (Top 10)", flush=True)
    print(f"{'='*100}", flush=True)

    kfold_results = []
    for rank, combo in enumerate(grid_results[:10], 1):
        hour_map = build_hour_adx_map(combo['asia_adx'], combo['london_adx'],
                                       combo['ny_adx'], combo['evening_adx'])
        fold_sharpes = []
        fold_details = []
        for fold_name, f_start, f_end in FOLDS:
            fold_df = df_kc[(df_kc.index >= f_start) & (df_kc.index < f_end)]
            if len(fold_df) < 200:
                fold_sharpes.append(0.0)
                fold_details.append({'fold': fold_name, 'n': 0, 'sharpe': 0.0})
                continue
            fold_trades = bt_keltner_session_adx(fold_df, SPREAD, UNIT_LOT, hour_map)
            fs = _compute_stats(fold_trades)
            fold_sharpes.append(fs['sharpe'])
            fold_details.append({'fold': fold_name, **fs})

        positive = sum(1 for s in fold_sharpes if s > 0)
        mean_sh = float(np.mean(fold_sharpes))
        min_sh = float(min(fold_sharpes))

        a_str = 'OFF' if combo['asia_adx'] == 999 else str(combo['asia_adx'])
        e_str = 'OFF' if combo['evening_adx'] == 999 else str(combo['evening_adx'])
        status = "PASS" if positive >= 4 else "FAIL"
        print(f"  #{rank:>2} A={a_str:>3} L={combo['london_adx']:>2} N={combo['ny_adx']:>2} E={e_str:>3} | "
              f"mean={mean_sh:.3f} min={min_sh:.3f} pos={positive}/6 [{status}] "
              f"folds={[round(s, 2) for s in fold_sharpes]}", flush=True)

        kfold_results.append({
            'rank': rank,
            'asia_adx': combo['asia_adx'], 'london_adx': combo['london_adx'],
            'ny_adx': combo['ny_adx'], 'evening_adx': combo['evening_adx'],
            'full_sharpe': combo['sharpe'], 'full_n': combo['n'],
            'fold_sharpes': [round(s, 3) for s in fold_sharpes],
            'positive_folds': positive, 'mean_fold_sharpe': round(mean_sh, 3),
            'min_fold_sharpe': round(min_sh, 3),
            'pass_4of6': positive >= 4,
            'fold_details': fold_details,
        })

    baseline_folds = []
    for fold_name, f_start, f_end in FOLDS:
        fold_df = df_kc[(df_kc.index >= f_start) & (df_kc.index < f_end)]
        if len(fold_df) < 200:
            baseline_folds.append(0.0)
            continue
        fold_trades = bt_keltner_session_adx(fold_df, SPREAD, UNIT_LOT, flat_map)
        fs = _compute_stats(fold_trades)
        baseline_folds.append(fs['sharpe'])

    print(f"\n  Baseline (flat ADX=14) folds: {[round(s, 2) for s in baseline_folds]}", flush=True)
    print(f"  Baseline mean={np.mean(baseline_folds):.3f}, pos={sum(1 for s in baseline_folds if s > 0)}/6", flush=True)

    results['phase3_kfold'] = kfold_results
    results['phase3_baseline_folds'] = [round(s, 3) for s in baseline_folds]

    # ══════════════════════════════════════════════════════════
    # Phase 4: Lot Boosting by Session
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("  Phase 4: Lot Boosting by Session (best ADX combo)", flush=True)
    print(f"{'='*100}", flush=True)

    best_passing = None
    for kr in kfold_results:
        if kr['pass_4of6']:
            best_passing = kr
            break
    if best_passing is None:
        best_passing = kfold_results[0]
        print(f"  WARNING: no combo passed 4/6, using rank #1 anyway", flush=True)

    best_hour_map = build_hour_adx_map(best_passing['asia_adx'], best_passing['london_adx'],
                                        best_passing['ny_adx'], best_passing['evening_adx'])

    a_str = 'OFF' if best_passing['asia_adx'] == 999 else str(best_passing['asia_adx'])
    e_str = 'OFF' if best_passing['evening_adx'] == 999 else str(best_passing['evening_adx'])
    print(f"\n  Best ADX combo: Asia={a_str}, London={best_passing['london_adx']}, "
          f"NY={best_passing['ny_adx']}, Evening={e_str}", flush=True)
    print(f"  Full-sample Sharpe={best_passing['full_sharpe']:.3f}, n={best_passing['full_n']}", flush=True)

    lot_configs = [
        ("Flat_1.0x", None),
        ("OvLap_1.5x", {hr: UNIT_LOT * (1.5 if hr in SESSION_GROUPS['overlap'] else 1.0) for hr in range(24)}),
        ("London_1.25x", {hr: UNIT_LOT * (1.25 if hr in SESSION_GROUPS['london'] else 1.0) for hr in range(24)}),
        ("Lon1.25_OvLap1.5", {hr: UNIT_LOT * (1.5 if hr in SESSION_GROUPS['overlap']
                              else 1.25 if hr in SESSION_GROUPS['london']
                              else 1.0) for hr in range(24)}),
        ("Lon1.25_NY1.25", {hr: UNIT_LOT * (1.25 if hr in (SESSION_GROUPS['london'] | SESSION_GROUPS['ny'])
                           else 1.0) for hr in range(24)}),
        ("Lon1.5_NY1.25", {hr: UNIT_LOT * (1.5 if hr in SESSION_GROUPS['london']
                          else 1.25 if hr in SESSION_GROUPS['ny']
                          else 1.0) for hr in range(24)}),
    ]

    lot_results = []
    print(f"\n  {'Config':<20} {'#':>5} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'MaxDD':>8}", flush=True)
    print(f"  {'-'*20} {'-'*5} {'-'*7} {'-'*10} {'-'*6} {'-'*8}", flush=True)

    for config_name, lot_map in lot_configs:
        trades = bt_keltner_session_adx(df_kc, SPREAD, UNIT_LOT, best_hour_map, lot_map=lot_map)
        s = _compute_stats(trades)
        lot_results.append({'config': config_name, **s})
        print(f"  {config_name:<20} {s['n']:>5} {s['sharpe']:>7.3f} ${s['pnl']:>9,.0f} "
              f"{s['wr']:>5.1f}% ${s['max_dd']:>7,.0f}", flush=True)

    best_lot = max(lot_results, key=lambda x: x['sharpe'])
    print(f"\n  Best lot config: {best_lot['config']} (Sharpe={best_lot['sharpe']:.3f})", flush=True)

    if best_lot['config'] != 'Flat_1.0x':
        print(f"\n  K-Fold for best lot config ({best_lot['config']}):", flush=True)
        bl_config_name = best_lot['config']
        bl_lot_map = [lm for cn, lm in lot_configs if cn == bl_config_name][0]
        bl_fold_sharpes = []
        for fold_name, f_start, f_end in FOLDS:
            fold_df = df_kc[(df_kc.index >= f_start) & (df_kc.index < f_end)]
            if len(fold_df) < 200:
                bl_fold_sharpes.append(0.0)
                continue
            fold_trades = bt_keltner_session_adx(fold_df, SPREAD, UNIT_LOT, best_hour_map, lot_map=bl_lot_map)
            fs = _compute_stats(fold_trades)
            bl_fold_sharpes.append(fs['sharpe'])
        bl_pos = sum(1 for s in bl_fold_sharpes if s > 0)
        print(f"  Folds: {[round(s, 2) for s in bl_fold_sharpes]}", flush=True)
        print(f"  Mean={np.mean(bl_fold_sharpes):.3f}, pos={bl_pos}/6", flush=True)
        best_lot['fold_sharpes'] = [round(s, 3) for s in bl_fold_sharpes]
        best_lot['pass_4of6'] = bl_pos >= 4

    results['phase4_lot_boost'] = lot_results
    results['phase4_best_lot'] = best_lot

    # ══════════════════════════════════════════════════════════
    # Summary & Recommendation
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("  R178 RECOMMENDATION", flush=True)
    print(f"{'='*100}", flush=True)

    rec = {
        'best_adx_combo': {
            'asia_adx': best_passing['asia_adx'],
            'london_adx': best_passing['london_adx'],
            'ny_adx': best_passing['ny_adx'],
            'evening_adx': best_passing['evening_adx'],
        },
        'full_sharpe': best_passing['full_sharpe'],
        'mean_fold_sharpe': best_passing['mean_fold_sharpe'],
        'pass_4of6': best_passing['pass_4of6'],
        'baseline_sharpe': baseline_stats['sharpe'],
        'improvement_pct': round((best_passing['full_sharpe'] - baseline_stats['sharpe'])
                                  / max(abs(baseline_stats['sharpe']), 0.001) * 100, 1),
        'best_lot_config': best_lot['config'],
        'best_lot_sharpe': best_lot['sharpe'],
    }
    results['recommendation'] = rec

    print(f"\n  Baseline (flat ADX=14): Sharpe={baseline_stats['sharpe']:.3f}, n={baseline_stats['n']}", flush=True)
    print(f"  Best session-ADX:       Sharpe={best_passing['full_sharpe']:.3f}, "
          f"n={best_passing['full_n']}", flush=True)
    print(f"  Improvement:            {rec['improvement_pct']:+.1f}%", flush=True)
    print(f"  K-Fold:                 {'PASS' if best_passing['pass_4of6'] else 'FAIL'} "
          f"({best_passing['positive_folds']}/6 positive)", flush=True)
    print(f"  Config:                 Asia={a_str}, London={best_passing['london_adx']}, "
          f"NY={best_passing['ny_adx']}, Evening={e_str}", flush=True)
    print(f"  Best lot boost:         {best_lot['config']} (Sharpe={best_lot['sharpe']:.3f})", flush=True)

    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)

    print(f"\n  R178 complete in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"{'='*100}", flush=True)

    out_path = OUTPUT_DIR / "r178_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
