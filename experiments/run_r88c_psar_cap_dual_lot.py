#!/usr/bin/env python3
"""
R88c — PSAR MaxLoss Cap: Dual Lot Size Comparison
===================================================
Runs Cap grid for BOTH lot sizes:
  - 0.03 (EA deployed)
  - 0.09 (R89/R150 recommended)

For each lot size, tests:
  Phase 1: Fixed Cap grid $5-$80
  Phase 2: K-Fold 6-fold for top candidates
  Phase 3: Side-by-side comparison with price tolerance in $/oz and %ATR
"""
import sys, os, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r88c_psar_cap_dual")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30

LOT_CONFIGS = {
    'EA_deployed': 0.03,
    'R89_optimized': 0.09,
}

FIXED_CAPS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80]

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
            return _mk(pos, c, times[i], "Trail", i, (ts_p - pos['entry'] - spread) * lot * pv)
    elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
        ts_p = lo_v + td
        if h >= ts_p:
            return _mk(pos, c, times[i], "Trail", i, (pos['entry'] - ts_p - spread) * lot * pv)

    if held >= max_hold:
        return _mk(pos, c, times[i], "Timeout", i, pnl_c)
    return None


def bt_psar(h1_df, spread, lot, maxloss_cap=0,
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
                               maxloss_cap)
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
                'cap_hits': 0, 'cap_pct': 0, 'avg_pnl': 0, 'avg_loss': 0,
                'max_single_loss': 0}
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
        'max_single_loss': round(min(pnls), 3),
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
    print("  R88c — PSAR Cap: Dual Lot Size (0.03 EA vs 0.09 R89)", flush=True)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 80, flush=True)

    h1_df = load_h1()
    mean_atr = compute_atr(h1_df).dropna().mean()
    print(f"  H1 mean ATR: ${mean_atr:.2f}", flush=True)

    all_results = {}

    for config_name, lot in LOT_CONFIGS.items():
        print(f"\n{'='*80}", flush=True)
        print(f"  Config: {config_name} (lot={lot})", flush=True)
        print(f"  Cap=$5 => ${5/(lot*PV):.2f}/oz = {5/(lot*PV)/mean_atr:.0%} ATR", flush=True)
        print(f"{'='*80}\n", flush=True)

        # Phase 1: Full grid
        print(f"  Phase 1: Fixed Cap Grid", flush=True)
        results = []
        for cap in FIXED_CAPS:
            trades = bt_psar(h1_df, spread=SPREAD, lot=lot, maxloss_cap=cap)
            stats = _compute_stats(trades)
            label = "NoCap" if cap == 0 else f"Cap${cap}"
            price_tol = cap / (lot * PV) if cap > 0 else float('inf')
            atr_ratio = price_tol / mean_atr if cap > 0 else float('inf')
            stats['label'] = label
            stats['cap'] = cap
            stats['lot'] = lot
            stats['price_tolerance'] = round(price_tol, 2) if cap > 0 else None
            stats['atr_ratio'] = round(atr_ratio, 3) if cap > 0 else None
            stats['atr_pct'] = round(atr_ratio * 100, 1) if cap > 0 else None
            results.append(stats)

        print(f"\n  {'Label':<10} {'N':>5} {'Sharpe':>7} {'PnL':>11} {'WR':>6} "
              f"{'MaxDD':>9} {'Cap%':>6} {'$/oz':>6} {'ATR%':>6} {'MaxLoss':>9}", flush=True)
        print(f"  {'-'*10} {'-'*5} {'-'*7} {'-'*11} {'-'*6} {'-'*9} {'-'*6} {'-'*6} {'-'*6} {'-'*9}", flush=True)
        for r in results:
            pt = f"{r['price_tolerance']:.1f}" if r['price_tolerance'] else "inf"
            ar = f"{r['atr_pct']:.0f}%" if r['atr_pct'] else "inf"
            print(f"  {r['label']:<10} {r['n']:>5} {r['sharpe']:>7.2f} {fmt(r['pnl']):>11} "
                  f"{r['wr']:>5.1f}% {fmt(r['max_dd']):>9} {r['cap_pct']:>5.1f}% "
                  f"${pt:>5} {ar:>5} {fmt(r['max_single_loss']):>9}", flush=True)

        results_sorted = sorted(results, key=lambda x: x['sharpe'], reverse=True)
        top3_str = ', '.join(f"{r['label']}({r['sharpe']:.2f})" for r in results_sorted[:3])
        print(f"\n  Top 3: {top3_str}", flush=True)

        # Phase 2: K-Fold for top 5 + NoCap + Cap$5
        print(f"\n  Phase 2: K-Fold Validation", flush=True)
        kfold_caps = set()
        kfold_caps.add(0)  # NoCap
        kfold_caps.add(5)  # old baseline
        for r in results_sorted[:5]:
            kfold_caps.add(r['cap'])

        kfold_results = {}
        for cap in sorted(kfold_caps):
            label = "NoCap" if cap == 0 else f"Cap${cap}"
            fold_sharpes = []
            for fold_name, start, end in FOLDS:
                fold_data = h1_df[start:end]
                if len(fold_data) < 200:
                    fold_sharpes.append(0)
                    continue
                trades = bt_psar(fold_data, spread=SPREAD, lot=lot, maxloss_cap=cap)
                stats = _compute_stats(trades)
                fold_sharpes.append(stats['sharpe'])

            positive = sum(1 for s in fold_sharpes if s > 0)
            mean_sh = float(np.mean(fold_sharpes))
            min_sh = float(min(fold_sharpes))
            status = "PASS" if positive >= 4 else "FAIL"

            kfold_results[label] = {
                'cap': cap, 'positive_folds': positive,
                'mean_sharpe': round(mean_sh, 2), 'min_sharpe': round(min_sh, 2),
                'pass': positive >= 4,
                'fold_sharpes': [round(s, 2) for s in fold_sharpes],
            }
            print(f"    {label:<10}: {positive}/6 pos, mean={mean_sh:.2f}, min={min_sh:.2f}  "
                  f"[{status}]  {[round(s,1) for s in fold_sharpes]}", flush=True)

        all_results[config_name] = {
            'lot': lot,
            'grid': results,
            'grid_sorted': [r['label'] for r in results_sorted],
            'kfold': kfold_results,
        }

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: Cross-comparison
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print(f"  Phase 3: Cross-Lot Comparison", flush=True)
    print(f"  Same dollar Cap => different price tolerance per lot size", flush=True)
    print(f"{'='*80}\n", flush=True)

    print(f"  {'Cap':>6} | {'--- 0.03 lot (EA) ---':^35} | {'--- 0.09 lot (R89) ---':^35}", flush=True)
    print(f"  {'':>6} | {'$/oz':>6} {'ATR%':>6} {'Sharpe':>7} {'Cap%':>6} {'WR':>6} | "
          f"{'$/oz':>6} {'ATR%':>6} {'Sharpe':>7} {'Cap%':>6} {'WR':>6}", flush=True)
    print(f"  {'-'*6}-+-{'-'*35}-+-{'-'*35}", flush=True)

    for cap in FIXED_CAPS:
        row = f"  {'NoCap' if cap == 0 else f'${cap}':>6} |"
        for cfg_name in ['EA_deployed', 'R89_optimized']:
            lot = LOT_CONFIGS[cfg_name]
            r = next((x for x in all_results[cfg_name]['grid'] if x['cap'] == cap), None)
            if r:
                pt = f"{r['price_tolerance']:.1f}" if r['price_tolerance'] else "inf"
                ar = f"{r['atr_pct']:.0f}%" if r['atr_pct'] else "inf"
                row += f" ${pt:>5} {ar:>5} {r['sharpe']:>7.2f} {r['cap_pct']:>5.1f}% {r['wr']:>5.1f}% |"
        print(row, flush=True)

    # Equivalent tolerance comparison
    print(f"\n  Equivalent Price Tolerance Mapping:", flush=True)
    print(f"  (Which Cap$ at 0.09 lot gives same $/oz tolerance as Cap$X at 0.03 lot?)", flush=True)
    print(f"  {'0.03 lot':>10} {'$/oz':>7} {'=> 0.09 lot':>12} {'$/oz':>7}", flush=True)
    print(f"  {'-'*10} {'-'*7} {'-'*12} {'-'*7}", flush=True)
    for cap_03 in [5, 10, 15, 20, 25, 30]:
        tol_03 = cap_03 / (0.03 * PV)
        cap_09_equiv = tol_03 * 0.09 * PV
        print(f"  Cap${cap_03:>4}  ${tol_03:>5.1f}  => Cap${cap_09_equiv:>5.0f}   ${cap_09_equiv/(0.09*PV):>5.1f}", flush=True)

    # Recommendation per lot
    print(f"\n  RECOMMENDATIONS:", flush=True)
    for cfg_name, lot in LOT_CONFIGS.items():
        kf = all_results[cfg_name]['kfold']
        passing = [(l, d) for l, d in kf.items() if d['pass'] and l != 'NoCap']
        if passing:
            best = max(passing, key=lambda x: x[1]['mean_sharpe'])
            label, data = best
            cap = data['cap']
            tol = cap / (lot * PV)
            print(f"\n  {cfg_name} (lot={lot}):", flush=True)
            print(f"    Recommended: {label}", flush=True)
            print(f"    Price tolerance: ${tol:.2f}/oz = {tol/mean_atr:.0%} ATR", flush=True)
            print(f"    K-Fold: {data['positive_folds']}/6 pos, mean={data['mean_sharpe']:.2f}", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*80}", flush=True)
    print(f"  R88c complete in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"{'='*80}", flush=True)

    all_results['metadata'] = {
        'mean_atr': round(mean_atr, 2), 'spread': SPREAD,
        'elapsed_s': round(elapsed, 1),
    }
    with open(OUTPUT_DIR / "r88c_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r88c_results.json", flush=True)


if __name__ == "__main__":
    main()
