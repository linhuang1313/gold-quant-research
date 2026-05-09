#!/usr/bin/env python3
"""
R175b — Fine-Grained Cap Grid for TSMOM / SESS_BO / DUAL_THRUST
=================================================================
R175 found Cap=$5 "optimal" for all three, but 50%+ trigger rate is
essentially replacing SL with Cap — too aggressive.

This script runs a $2 step grid from $0 to $80 and breaks out:
  - Cap trigger rate
  - Win rate (only non-Cap trades)
  - Sharpe with and without Cap
  - "Sweet spot" = highest Sharpe where Cap trigger rate < 20%

Also computes 0.01-lot equivalent for cross-comparison with R89 history.
"""
import sys, os, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r175b_fine_cap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30

STRATEGIES = {
    'TSMOM':       {'lot': 0.15, 'current_cap': 0},
    'SESS_BO':     {'lot': 0.13, 'current_cap': 35},
    'DUAL_THRUST': {'lot': 0.04, 'current_cap': 35},
}

FINE_CAPS = [0] + list(range(3, 42, 3)) + [50, 60, 80]

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-06-01"),
]


def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


def _run_exit(pos, i, h, lo_v, c, spread, lot, pv, times,
              sl_atr, tp_atr, trail_act_atr, trail_dist_atr, max_hold,
              maxloss_cap=0):
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
    if held := (i - pos['bar']):
        pass
    if i - pos['bar'] >= max_hold:
        return _mk(pos, c, times[i], "Timeout", i, pnl_c)
    return None


def bt_tsmom(h1_df, spread, lot, maxloss_cap=0,
             fast=480, slow=720, sl_atr=6.0, tp_atr=8.0,
             trail_act=0.14, trail_dist=0.025, max_hold=12):
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
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_dual_thrust(h1_df, spread, lot, maxloss_cap=0,
                   n_bars=6, k=0.5, sl_atr=4.5, tp_atr=8.0,
                   trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    o = df['Open'].values; atr = df['ATR'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(n_bars, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = np.max(h[i-n_bars:i])
        lc = np.min(c[i-n_bars:i])
        hc = np.max(c[i-n_bars:i])
        ll = np.min(lo[i-n_bars:i])
        rng = max(hh - lc, hc - ll)
        buy_line = o[i] + k * rng
        sell_line = o[i] - k * rng
        if c[i] > buy_line:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < sell_line:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


STRAT_BT = {
    'TSMOM': bt_tsmom,
    'SESS_BO': bt_sess_bo,
    'DUAL_THRUST': bt_dual_thrust,
}


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
    if len(arr) == 0: return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def _detailed_stats(trades, lot, mean_atr, cap_val):
    if not trades:
        return None
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    cap_trades = [t for t in trades if t.get('reason') == 'MaxLossCap']
    sl_trades = [t for t in trades if t.get('reason') == 'SL']
    tp_trades = [t for t in trades if t.get('reason') == 'TP']
    trail_trades = [t for t in trades if t.get('reason') == 'Trail']
    timeout_trades = [t for t in trades if t.get('reason') == 'Timeout']
    reversal_trades = [t for t in trades if t.get('reason') == 'Reversal']

    non_cap = [t for t in trades if t.get('reason') != 'MaxLossCap']
    non_cap_wr = sum(1 for t in non_cap if t['pnl'] > 0) / len(non_cap) * 100 if non_cap else 0

    price_tol = cap_val / (lot * PV) if cap_val > 0 else float('inf')
    atr_pct = price_tol / mean_atr * 100 if cap_val > 0 else float('inf')

    avg_cap_loss = np.mean([t['pnl'] for t in cap_trades]) if cap_trades else 0
    avg_sl_loss = np.mean([t['pnl'] for t in sl_trades]) if sl_trades else 0

    cap_would_win = 0
    if cap_trades:
        nocap_pnls_for_cap_trades = []
        for t in cap_trades:
            sl_loss = -t.get('atr', mean_atr) * 4.5 * lot * PV
            nocap_pnls_for_cap_trades.append(sl_loss)

    return {
        'cap': cap_val,
        'n': n,
        'sharpe': round(_sharpe(daily), 2),
        'pnl': round(sum(pnls), 0),
        'max_dd': round(_max_dd(daily), 0),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'non_cap_wr': round(non_cap_wr, 1),
        'cap_n': len(cap_trades),
        'cap_pct': round(len(cap_trades) / n * 100, 1),
        'sl_n': len(sl_trades),
        'tp_n': len(tp_trades),
        'trail_n': len(trail_trades),
        'timeout_n': len(timeout_trades),
        'reversal_n': len(reversal_trades),
        'price_tol': round(price_tol, 2) if cap_val > 0 else None,
        'atr_pct': round(atr_pct, 0) if cap_val > 0 else None,
        'avg_cap_loss': round(avg_cap_loss, 1),
        'avg_sl_loss': round(avg_sl_loss, 1),
        'avg_win': round(np.mean([p for p in pnls if p > 0]), 1) if any(p > 0 for p in pnls) else 0,
        'avg_loss': round(np.mean([p for p in pnls if p < 0]), 1) if any(p < 0 for p in pnls) else 0,
    }


def load_h1():
    import glob as _glob
    candidates = sorted(_glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
    if not candidates:
        raise FileNotFoundError("No H1 data found")
    csv_path = candidates[-1]
    print(f"  Loading: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    return df


def main():
    t0 = time.time()
    print("=" * 90, flush=True)
    print("  R175b — Fine Cap Grid: TSMOM / SESS_BO / DUAL_THRUST", flush=True)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  Cap grid: {FINE_CAPS}", flush=True)
    print("=" * 90, flush=True)

    h1_df = load_h1()
    mean_atr = compute_atr(h1_df).dropna().mean()
    print(f"  {len(h1_df)} bars, mean ATR: ${mean_atr:.2f}\n", flush=True)

    all_results = {}

    for strat_name in ['TSMOM', 'SESS_BO', 'DUAL_THRUST']:
        cfg = STRATEGIES[strat_name]
        lot = cfg['lot']
        current_cap = cfg['current_cap']
        bt_fn = STRAT_BT[strat_name]

        print(f"\n{'='*90}", flush=True)
        print(f"  {strat_name}  (lot={lot}, current Cap=${current_cap})", flush=True)
        print(f"  SL at 4.5xATR = ${4.5 * mean_atr * lot * PV:.1f} (price tol ${4.5*mean_atr:.1f}/oz)", flush=True)
        if strat_name == 'TSMOM':
            print(f"  SL at 6.0xATR = ${6.0 * mean_atr * lot * PV:.1f} (price tol ${6.0*mean_atr:.1f}/oz)", flush=True)
        print(f"{'='*90}\n", flush=True)

        # Phase 1: Fine grid
        print(f"  Phase 1: Fine Cap Grid ($3 steps)", flush=True)
        results = []
        for cap in FINE_CAPS:
            trades = bt_fn(h1_df, spread=SPREAD, lot=lot, maxloss_cap=cap)
            stats = _detailed_stats(trades, lot, mean_atr, cap)
            if stats:
                results.append(stats)

        # Print header
        print(f"\n  {'Cap':>5} {'N':>5} {'Shrp':>5} {'PnL':>8} {'WR':>5} {'ncWR':>5} "
              f"{'Cap%':>5} {'SL#':>4} {'TP#':>4} {'Trl#':>4} {'TO#':>4} "
              f"{'$/oz':>6} {'ATR%':>5} {'AvgW':>6} {'AvgL':>7} {'MaxDD':>7}", flush=True)
        print(f"  {'-'*5} {'-'*5} {'-'*5} {'-'*8} {'-'*5} {'-'*5} "
              f"{'-'*5} {'-'*4} {'-'*4} {'-'*4} {'-'*4} "
              f"{'-'*6} {'-'*5} {'-'*6} {'-'*7} {'-'*7}", flush=True)

        for r in results:
            cap = r['cap']
            pt = f"${r['price_tol']:.1f}" if r['price_tol'] else "inf"
            ar = f"{r['atr_pct']:.0f}%" if r['atr_pct'] else "inf"
            marker = " <<<" if cap == current_cap else ""
            print(f"  ${cap:>4} {r['n']:>5} {r['sharpe']:>5.2f} "
                  f"${r['pnl']:>7,.0f} {r['wr']:>4.1f}% {r['non_cap_wr']:>4.1f}% "
                  f"{r['cap_pct']:>4.1f}% {r['sl_n']:>4} {r['tp_n']:>4} {r['trail_n']:>4} {r['timeout_n']:>4} "
                  f"{pt:>6} {ar:>5} ${r['avg_win']:>5.1f} -${abs(r['avg_loss']):>5.1f} "
                  f"${r['max_dd']:>6,.0f}{marker}", flush=True)

        all_results[strat_name] = results

        # Find sweet spots
        print(f"\n  Sweet Spot Analysis:", flush=True)

        # Best Sharpe with Cap% < 20%
        filtered_20 = [r for r in results if r['cap'] > 0 and r['cap_pct'] < 20]
        if filtered_20:
            best_20 = max(filtered_20, key=lambda x: x['sharpe'])
            print(f"    Best Sharpe with Cap% < 20%: Cap=${best_20['cap']} "
                  f"(Sharpe={best_20['sharpe']:.2f}, Cap%={best_20['cap_pct']:.1f}%, "
                  f"$/oz=${best_20['price_tol']:.1f}, WR={best_20['wr']:.1f}%)", flush=True)

        # Best Sharpe with Cap% < 10%
        filtered_10 = [r for r in results if r['cap'] > 0 and r['cap_pct'] < 10]
        if filtered_10:
            best_10 = max(filtered_10, key=lambda x: x['sharpe'])
            print(f"    Best Sharpe with Cap% < 10%: Cap=${best_10['cap']} "
                  f"(Sharpe={best_10['sharpe']:.2f}, Cap%={best_10['cap_pct']:.1f}%, "
                  f"$/oz=${best_10['price_tol']:.1f}, WR={best_10['wr']:.1f}%)", flush=True)

        # Best overall Sharpe with cap
        best_all = max([r for r in results if r['cap'] > 0], key=lambda x: x['sharpe'])
        print(f"    Best overall Sharpe (with cap): Cap=${best_all['cap']} "
              f"(Sharpe={best_all['sharpe']:.2f}, Cap%={best_all['cap_pct']:.1f}%)", flush=True)

        # NoCap baseline
        nocap = [r for r in results if r['cap'] == 0][0]
        print(f"    NoCap baseline: Sharpe={nocap['sharpe']:.2f}, WR={nocap['wr']:.1f}%, MaxDD=${nocap['max_dd']:,.0f}", flush=True)

        # 0.01-lot equivalent analysis
        print(f"\n  0.01-lot Equivalent Cap (for R89 comparison):", flush=True)
        print(f"    lot={lot} → 0.01 multiplier = {lot/0.01:.0f}x", flush=True)
        for cap in [5, 10, 15, 20, 25, 30, 35]:
            equiv_001 = cap / (lot / 0.01)
            print(f"    Cap${cap} at {lot} lot = Cap${equiv_001:.1f} at 0.01 lot", flush=True)

        # Phase 2: K-Fold for short-list
        print(f"\n  Phase 2: K-Fold Validation (Cap% < 25% + current + NoCap)", flush=True)
        kfold_caps = set()
        kfold_caps.add(0)
        kfold_caps.add(current_cap)
        candidates_for_kfold = [r for r in results if r['cap'] > 0 and r['cap_pct'] < 25]
        for r in sorted(candidates_for_kfold, key=lambda x: x['sharpe'], reverse=True)[:6]:
            kfold_caps.add(r['cap'])
        # Also add the best from each zone
        for r in sorted(filtered_20, key=lambda x: x['sharpe'], reverse=True)[:2]:
            kfold_caps.add(r['cap'])

        for cap in sorted(kfold_caps):
            label = "NoCap" if cap == 0 else f"Cap${cap}"
            fold_sharpes = []
            fold_wrs = []
            for fold_name, start, end in FOLDS:
                fold_data = h1_df[start:end]
                if len(fold_data) < 200:
                    fold_sharpes.append(0); fold_wrs.append(0); continue
                trades = bt_fn(fold_data, spread=SPREAD, lot=lot, maxloss_cap=cap)
                if not trades:
                    fold_sharpes.append(0); fold_wrs.append(0); continue
                daily = _trades_to_daily(trades)
                fold_sharpes.append(_sharpe(daily))
                fold_wrs.append(sum(1 for t in trades if t['pnl'] > 0) / len(trades) * 100)

            positive = sum(1 for s in fold_sharpes if s > 0)
            mean_sh = float(np.mean(fold_sharpes))
            min_sh = float(min(fold_sharpes))
            std_sh = float(np.std(fold_sharpes))
            status = "PASS" if positive >= 4 else "FAIL"
            marker = " <<<" if cap == current_cap else ""

            r_match = [r for r in results if r['cap'] == cap]
            cap_pct_str = f"{r_match[0]['cap_pct']:.0f}%" if r_match and r_match[0].get('cap_pct') is not None else "N/A"

            print(f"    {label:<10}: {positive}/6 pos, mean={mean_sh:.2f}, min={min_sh:.2f}, "
                  f"std={std_sh:.2f}, Cap%={cap_pct_str} [{status}] "
                  f"{[round(s,1) for s in fold_sharpes]}{marker}", flush=True)

    # Final summary
    print(f"\n\n{'='*90}", flush=True)
    print(f"  FINAL RECOMMENDATIONS", flush=True)
    print(f"{'='*90}\n", flush=True)
    print(f"  Key principle: Cap should be a SAFETY NET (catch catastrophic outliers),", flush=True)
    print(f"  not a pseudo-SL (triggering on 30%+ of trades).", flush=True)
    print(f"  Target: Cap trigger rate 5-15%, price tolerance > 100% ATR.\n", flush=True)

    for strat_name in ['TSMOM', 'SESS_BO', 'DUAL_THRUST']:
        cfg = STRATEGIES[strat_name]
        lot = cfg['lot']
        results = all_results[strat_name]
        sweet = [r for r in results if r['cap'] > 0 and 5 <= r['cap_pct'] <= 15]
        if sweet:
            best = max(sweet, key=lambda x: x['sharpe'])
            print(f"  {strat_name} (lot={lot}): SUGGEST Cap=${best['cap']} "
                  f"(Sharpe={best['sharpe']:.2f}, Cap%={best['cap_pct']:.1f}%, "
                  f"WR={best['wr']:.1f}%, $/oz=${best['price_tol']:.1f}, "
                  f"{best['atr_pct']:.0f}% ATR)", flush=True)
        else:
            mild = [r for r in results if r['cap'] > 0 and r['cap_pct'] < 20]
            if mild:
                best = max(mild, key=lambda x: x['sharpe'])
                print(f"  {strat_name} (lot={lot}): SUGGEST Cap=${best['cap']} "
                      f"(Sharpe={best['sharpe']:.2f}, Cap%={best['cap_pct']:.1f}%)", flush=True)
            else:
                print(f"  {strat_name}: All caps > 20% trigger rate, consider NoCap", flush=True)

    elapsed = time.time() - t0
    print(f"\n  R175b complete in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    with open(OUTPUT_DIR / "r175b_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r175b_results.json", flush=True)


if __name__ == "__main__":
    main()
