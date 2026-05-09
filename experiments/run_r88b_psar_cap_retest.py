#!/usr/bin/env python3
"""
R88b — PSAR MaxLoss Cap Re-evaluation (Actual Live Lot 0.03)
==============================================================
Background: Expert review flagged PSAR Cap=$5 as too tight.
  - At lot=0.03, $5 Cap = $1.67/oz price tolerance
  - H1 ATR ≈ $21, so normal noise of $10-15 triggers Cap constantly
  - Original R88 tested at lot=0.01 (old config), not 0.03 (current live)

This experiment:
  Phase 1: Fixed Cap grid $5-$60 at actual live lot=0.03 (+ NoCap baseline)
           Report price tolerance and Cap trigger frequency
  Phase 2: ATR-based dynamic Cap (0.5×ATR to 3.0×ATR, step 0.25)
           Cap = N × ATR × lot × PV, adapts to volatility
  Phase 3: K-Fold 6-fold for top candidates from Phase 1+2
  Phase 4: Comparison table — old Cap=$5 vs new recommendation
"""
import sys, os, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r88b_psar_cap_retest")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
PSAR_LOT = 0.03  # actual live lot

FIXED_CAPS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60]
ATR_MULTS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-06-01"),
]


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


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
    df['PSAR_dir'] = direction
    df['ATR'] = compute_atr(df)
    return df


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {
        'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
        'entry_time': pos['time'], 'exit_time': exit_time,
        'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar'],
    }


def _run_exit(pos, i, h, lo_v, c, spread, lot, pv, times,
              sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold,
              maxloss_cap=0, maxloss_cap_atr_mult=0):
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

    cap_limit = maxloss_cap
    if maxloss_cap_atr_mult > 0:
        cap_limit = maxloss_cap_atr_mult * pos['atr'] * lot * pv
    if cap_limit > 0 and pnl_c < -cap_limit:
        return _mk(pos, c, times[i], "MaxLossCap", i, -cap_limit)

    ad = trail_act_atr * pos['atr']
    td = trail_dist_atr * pos['atr']
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


def bt_psar(h1_df, spread, lot, maxloss_cap=0, maxloss_cap_atr_mult=0,
            sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04, max_hold=20):
    df = add_psar(h1_df).dropna(subset=['PSAR_dir', 'ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    pdir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold,
                               maxloss_cap, maxloss_cap_atr_mult)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
        if pdir[i-1] == -1 and pdir[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i] + spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif pdir[i-1] == 1 and pdir[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i] - spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


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
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0,
                'cap_hits': 0, 'cap_pct': 0, 'avg_pnl': 0, 'avg_loss': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    cap_hits = sum(1 for t in trades if t.get('reason') == 'MaxLossCap')
    losses = [p for p in pnls if p < 0]
    return {
        'n': n,
        'sharpe': round(_sharpe(daily), 2),
        'pnl': round(sum(pnls), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'max_dd': round(_max_dd(daily), 2),
        'cap_hits': cap_hits,
        'cap_pct': round(cap_hits / n * 100, 1),
        'avg_pnl': round(np.mean(pnls), 3),
        'avg_loss': round(np.mean(losses), 3) if losses else 0,
    }


def load_h1():
    import glob
    candidates = sorted(glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
    if not candidates:
        raise FileNotFoundError("No H1 data found")
    csv_path = candidates[-1]
    print(f"  Loading H1: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    print(f"  {len(df)} bars ({df.index[0]} ~ {df.index[-1]})", flush=True)
    return df


def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("  R88b — PSAR MaxLoss Cap Re-evaluation (Live Lot 0.03)", flush=True)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  Background: Expert review flagged Cap=$5 as too tight at 0.03 lot", flush=True)
    print("=" * 80, flush=True)

    h1_df = load_h1()
    mean_atr = compute_atr(h1_df).dropna().mean()
    print(f"  H1 mean ATR: ${mean_atr:.2f}", flush=True)
    print(f"  At lot={PSAR_LOT}: $5 Cap = ${5/(PSAR_LOT*PV):.2f}/oz tolerance "
          f"(ATR=${mean_atr:.1f}, ratio={5/(PSAR_LOT*PV)/mean_atr:.1%})", flush=True)

    all_results = {}

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Fixed Cap Grid
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print(f"  Phase 1: Fixed Cap Grid (lot={PSAR_LOT})", flush=True)
    print(f"{'='*80}\n", flush=True)

    fixed_results = []
    for cap in FIXED_CAPS:
        trades = bt_psar(h1_df, spread=SPREAD, lot=PSAR_LOT, maxloss_cap=cap)
        stats = _compute_stats(trades)
        label = "NoCap" if cap == 0 else f"Cap${cap}"
        price_tol = cap / (PSAR_LOT * PV) if cap > 0 else float('inf')
        atr_ratio = price_tol / mean_atr if cap > 0 else float('inf')
        stats['label'] = label
        stats['cap'] = cap
        stats['price_tolerance'] = round(price_tol, 2) if cap > 0 else None
        stats['atr_ratio'] = round(atr_ratio, 2) if cap > 0 else None
        fixed_results.append(stats)

    print(f"  {'Label':<10} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} "
          f"{'MaxDD':>10} {'CapHit':>7} {'Cap%':>6} {'PrcTol':>8} {'ATR%':>7}", flush=True)
    print(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*12} {'-'*7} {'-'*10} {'-'*7} {'-'*6} {'-'*8} {'-'*7}", flush=True)
    for r in fixed_results:
        pt = f"${r['price_tolerance']:.1f}" if r['price_tolerance'] else "inf"
        ar = f"{r['atr_ratio']:.0%}" if r['atr_ratio'] else "inf"
        print(f"  {r['label']:<10} {r['n']:>6} {r['sharpe']:>8.2f} {fmt(r['pnl']):>12} "
              f"{r['wr']:>6.1f}% {fmt(r['max_dd']):>10} {r['cap_hits']:>7} "
              f"{r['cap_pct']:>5.1f}% {pt:>8} {ar:>7}", flush=True)

    fixed_results_sorted = sorted(fixed_results, key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  Top 3 by Sharpe:", flush=True)
    for i, r in enumerate(fixed_results_sorted[:3]):
        print(f"    #{i+1}: {r['label']} Sharpe={r['sharpe']:.2f} PnL={fmt(r['pnl'])} "
              f"CapHits={r['cap_hits']} ({r['cap_pct']:.1f}%)", flush=True)

    all_results['phase1_fixed'] = fixed_results

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: ATR-Based Dynamic Cap
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print(f"  Phase 2: ATR-Based Dynamic Cap (0.5x-3.0x ATR)", flush=True)
    print(f"  Formula: Cap = N × ATR × lot × PV (adapts per-trade)", flush=True)
    print(f"{'='*80}\n", flush=True)

    atr_results = []
    for mult in ATR_MULTS:
        trades = bt_psar(h1_df, spread=SPREAD, lot=PSAR_LOT, maxloss_cap_atr_mult=mult)
        stats = _compute_stats(trades)
        avg_cap = mult * mean_atr * PSAR_LOT * PV
        label = f"ATR×{mult:.2f}"
        stats['label'] = label
        stats['atr_mult'] = mult
        stats['avg_cap_usd'] = round(avg_cap, 2)
        stats['avg_price_tol'] = round(mult * mean_atr, 2)
        atr_results.append(stats)

    print(f"  {'Label':<12} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'WR':>7} "
          f"{'MaxDD':>10} {'CapHit':>7} {'Cap%':>6} {'AvgCap$':>8} {'PrcTol':>8}", flush=True)
    print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*12} {'-'*7} {'-'*10} {'-'*7} {'-'*6} {'-'*8} {'-'*8}", flush=True)
    for r in atr_results:
        print(f"  {r['label']:<12} {r['n']:>6} {r['sharpe']:>8.2f} {fmt(r['pnl']):>12} "
              f"{r['wr']:>6.1f}% {fmt(r['max_dd']):>10} {r['cap_hits']:>7} "
              f"{r['cap_pct']:>5.1f}% ${r['avg_cap_usd']:>6.1f} ${r['avg_price_tol']:>6.1f}", flush=True)

    atr_results_sorted = sorted(atr_results, key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  Top 3 ATR-based by Sharpe:", flush=True)
    for i, r in enumerate(atr_results_sorted[:3]):
        print(f"    #{i+1}: {r['label']} Sharpe={r['sharpe']:.2f} PnL={fmt(r['pnl'])} "
              f"AvgCap=${r['avg_cap_usd']:.1f} CapHits={r['cap_hits']}", flush=True)

    all_results['phase2_atr'] = atr_results

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: K-Fold for top candidates
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print(f"  Phase 3: K-Fold 6-Fold Validation", flush=True)
    print(f"{'='*80}\n", flush=True)

    candidates = []
    candidates.append(('NoCap_baseline', 0, 0))
    candidates.append(('Cap$5_old', 5, 0))
    for r in fixed_results_sorted[:3]:
        if r['cap'] not in [0, 5]:
            candidates.append((r['label'], r['cap'], 0))
    for r in atr_results_sorted[:3]:
        candidates.append((r['label'], 0, r['atr_mult']))

    seen = set()
    unique_candidates = []
    for label, cap, atr_m in candidates:
        key = (cap, atr_m)
        if key not in seen:
            seen.add(key)
            unique_candidates.append((label, cap, atr_m))

    kfold_results = {}
    for label, cap, atr_m in unique_candidates:
        fold_sharpes = []
        fold_details = []
        for fold_name, start, end in FOLDS:
            fold_data = h1_df[start:end]
            if len(fold_data) < 200:
                fold_sharpes.append(0)
                fold_details.append({'fold': fold_name, 'sharpe': 0, 'n': 0})
                continue
            trades = bt_psar(fold_data, spread=SPREAD, lot=PSAR_LOT,
                             maxloss_cap=cap, maxloss_cap_atr_mult=atr_m)
            stats = _compute_stats(trades)
            fold_sharpes.append(stats['sharpe'])
            fold_details.append({'fold': fold_name, 'sharpe': stats['sharpe'],
                                 'n': stats['n'], 'cap_hits': stats['cap_hits']})

        positive = sum(1 for s in fold_sharpes if s > 0)
        mean_sh = float(np.mean(fold_sharpes))
        min_sh = float(min(fold_sharpes))
        status = "PASS" if positive >= 4 else "FAIL"

        kfold_results[label] = {
            'cap': cap, 'atr_mult': atr_m,
            'fold_sharpes': [round(s, 2) for s in fold_sharpes],
            'positive_folds': positive,
            'mean_sharpe': round(mean_sh, 2),
            'min_sharpe': round(min_sh, 2),
            'pass': positive >= 4,
            'fold_details': fold_details,
        }

        print(f"  {label:<15}: {positive}/6 pos, mean={mean_sh:.2f}, min={min_sh:.2f}  "
              f"[{status}]  folds={[round(s, 1) for s in fold_sharpes]}", flush=True)

    all_results['phase3_kfold'] = kfold_results

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Comparison & Recommendation
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print(f"  Phase 4: Comparison — Old vs New Cap Recommendation", flush=True)
    print(f"{'='*80}\n", flush=True)

    old_cap5 = next((r for r in fixed_results if r['cap'] == 5), None)
    nocap = next((r for r in fixed_results if r['cap'] == 0), None)

    print(f"  Context:", flush=True)
    print(f"    Live lot: {PSAR_LOT}", flush=True)
    print(f"    H1 mean ATR: ${mean_atr:.2f}", flush=True)
    print(f"    Old Cap=$5: price tolerance = ${5/(PSAR_LOT*PV):.2f}/oz "
          f"= {5/(PSAR_LOT*PV)/mean_atr:.0%} of ATR", flush=True)
    if old_cap5:
        print(f"    Old Cap=$5: Sharpe={old_cap5['sharpe']:.2f}, "
              f"Cap triggered {old_cap5['cap_hits']} times ({old_cap5['cap_pct']:.1f}%)", flush=True)

    best_fixed = fixed_results_sorted[0]
    best_atr = atr_results_sorted[0]

    print(f"\n  Best fixed Cap: {best_fixed['label']} "
          f"(Sharpe={best_fixed['sharpe']:.2f}, price_tol=${best_fixed.get('price_tolerance','N/A')})", flush=True)
    print(f"  Best ATR Cap: {best_atr['label']} "
          f"(Sharpe={best_atr['sharpe']:.2f}, avg_cap=${best_atr['avg_cap_usd']:.1f})", flush=True)

    print(f"\n  K-Fold comparison:", flush=True)
    print(f"  {'Config':<15} {'KF Mean':>8} {'KF Min':>8} {'Pass':>6} {'Improvement':>12}", flush=True)
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*6} {'-'*12}", flush=True)

    baseline_kf = kfold_results.get('NoCap_baseline', {})
    baseline_mean = baseline_kf.get('mean_sharpe', 0)
    for label, kf in kfold_results.items():
        delta = kf['mean_sharpe'] - baseline_mean
        delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
        pass_str = "PASS" if kf['pass'] else "FAIL"
        print(f"  {label:<15} {kf['mean_sharpe']:>8.2f} {kf['min_sharpe']:>8.2f} "
              f"{pass_str:>6} {delta_str:>12}", flush=True)

    passing = [(l, kf) for l, kf in kfold_results.items()
               if kf['pass'] and l != 'NoCap_baseline']
    if passing:
        best_passing = max(passing, key=lambda x: x[1]['mean_sharpe'])
        rec_label, rec_kf = best_passing
        print(f"\n  RECOMMENDATION: {rec_label}", flush=True)
        print(f"    K-Fold mean Sharpe: {rec_kf['mean_sharpe']:.2f}", flush=True)
        print(f"    K-Fold min Sharpe: {rec_kf['min_sharpe']:.2f}", flush=True)
        if rec_kf['atr_mult'] > 0:
            avg_cap = rec_kf['atr_mult'] * mean_atr * PSAR_LOT * PV
            print(f"    ATR multiplier: {rec_kf['atr_mult']}× → avg Cap=${avg_cap:.1f} "
                  f"(${rec_kf['atr_mult']*mean_atr:.1f}/oz tolerance)", flush=True)
        else:
            pt = rec_kf['cap'] / (PSAR_LOT * PV)
            print(f"    Fixed Cap: ${rec_kf['cap']} → ${pt:.1f}/oz tolerance "
                  f"({pt/mean_atr:.0%} of ATR)", flush=True)
    else:
        print(f"\n  RECOMMENDATION: NoCap (no Cap configuration passes K-Fold better than baseline)", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*80}", flush=True)
    print(f"  R88b complete in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"  Results: {OUTPUT_DIR}/", flush=True)
    print(f"{'='*80}", flush=True)

    all_results['metadata'] = {
        'lot': PSAR_LOT, 'spread': SPREAD, 'mean_atr': round(mean_atr, 2),
        'elapsed_s': round(elapsed, 1),
    }
    with open(OUTPUT_DIR / "r88b_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r88b_results.json", flush=True)


if __name__ == "__main__":
    main()
