#!/usr/bin/env python3
"""
R155 — Keltner (L8_MAX) Holding Period + Session Analysis
==========================================================
Based on LIVE trade data (91 trades, 03-25 ~ 05-06):
  - Asia+London (00-08 UTC): ~85% WR, best time
  - US session (13-21 UTC): large losses concentrated here
  - Most profit trades are short-hold via Trailing
  - MaxLoss/timeout losses dominate long-held trades

This experiment:
  Phase 1: Baseline — extract all trades with bars_held + entry_hour
  Phase 2: PnL by bars_held bucket
  Phase 3: PnL by UTC entry hour (validate live observation)
  Phase 4: Exit reason breakdown (trailing vs maxloss vs timeout)
  Phase 5: MaxHold M15 sweep (engine-based)
  Phase 6: Session filter sweep (block bad hours)
  Phase 7: Combined MaxHold + Session filter
  Phase 8: K-Fold validation on best config
  Phase 9: Final recommendation (lot=0.05 production comparison)
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r155_keltner_hold_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

t0 = time.time()

PV = 100


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def sharpe(arr):
    if len(arr) < 10: return 0.0
    s = np.std(arr, ddof=1)
    if s == 0: return 0.0
    return float(np.mean(arr) / s * np.sqrt(252))


def max_dd(arr):
    if len(arr) == 0: return 0.0
    eq = np.cumsum(arr)
    peak = np.maximum.accumulate(eq)
    return float((peak - eq).max())


def trades_to_daily(trades):
    if not trades: return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


def run_l8(bundle, spread=0.30, lot=0.01, cap=35,
           keltner_max_hold_m15=0, h1_allowed_sessions=None):
    from backtest.runner import run_variant, LIVE_PARITY_KWARGS
    kw = dict(LIVE_PARITY_KWARGS)
    kw['maxloss_cap'] = cap
    kw['spread_cost'] = spread
    kw['initial_capital'] = 2000
    kw['min_lot_size'] = lot
    kw['max_lot_size'] = lot
    if keltner_max_hold_m15 > 0:
        kw['keltner_max_hold_m15'] = keltner_max_hold_m15
    if h1_allowed_sessions is not None:
        kw['h1_allowed_sessions'] = h1_allowed_sessions
    result = run_variant(bundle, "L8_MAX", verbose=False, **kw)
    raw = result.get('_trades', [])
    trades = []
    for t in raw:
        entry_ts = pd.Timestamp(t.entry_time)
        exit_ts = pd.Timestamp(t.exit_time)
        entry_utc = entry_ts.tz_localize('UTC') if entry_ts.tzinfo is None else entry_ts.tz_convert('UTC')
        trades.append({
            'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
            'entry_time': t.entry_time, 'exit_time': t.exit_time,
            'entry_hour_utc': entry_utc.hour,
            'pnl': t.pnl, 'reason': t.exit_reason, 'bars_held': t.bars_held,
        })
    return trades


def print_stats(trades, label=""):
    if not trades:
        print(f"  {label}: NO TRADES", flush=True)
        return {}
    pnls = [t['pnl'] for t in trades]
    ds = trades_to_daily(trades)
    n = len(trades)
    wr = sum(1 for p in pnls if p > 0) / n * 100
    sh = sharpe(ds.values)
    dd = max_dd(ds.values)
    pnl = sum(pnls)
    avg = np.mean(pnls)
    bars = np.mean([t['bars_held'] for t in trades])
    print(f"  {label}: n={n:>5}, Sharpe={sh:>5.2f}, PnL={fmt(pnl)}, "
          f"MaxDD={fmt(dd)}, WR={wr:.1f}%, Avg={avg:.2f}, AvgBars={bars:.1f}", flush=True)
    return {'n': n, 'sharpe': round(sh, 2), 'pnl': round(pnl, 2),
            'max_dd': round(dd, 2), 'wr': round(wr, 1), 'avg_pnl': round(avg, 2),
            'avg_bars': round(bars, 1)}


def main():
    results = {}

    print("=" * 80, flush=True)
    print("  R155 — Keltner (L8_MAX) Holding Period + Session Analysis", flush=True)
    print(f"  Started: {datetime.now()}", flush=True)
    print("  Based on 91 live trades (03-25 ~ 05-06) pattern analysis", flush=True)
    print("=" * 80, flush=True)

    from backtest.runner import DataBundle
    print("\n  Loading DataBundle...", flush=True)
    bundle = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    print("  Bundle ready.\n", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Baseline
    # ═══════════════════════════════════════════════════════════════
    print(f"{'='*80}", flush=True)
    print("  Phase 1: Baseline L8_MAX (Cap=$35, default MaxHold)", flush=True)
    print(f"{'='*80}\n", flush=True)

    trades = run_l8(bundle, cap=35)
    n = len(trades)
    base_stats = print_stats(trades, "Baseline")
    results['phase1_baseline'] = base_stats

    bars_list = [t['bars_held'] for t in trades]
    print(f"\n  Bars held: min={min(bars_list)}, max={max(bars_list)}, "
          f"mean={np.mean(bars_list):.1f}, median={np.median(bars_list):.0f}", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: PnL by bars_held bucket
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print("  Phase 2: PnL Breakdown by Holding Duration (M15 bars)", flush=True)
    print(f"{'='*80}\n", flush=True)

    buckets = [
        (1, 4, "1-4 bars (0-1h)"),
        (5, 8, "5-8 bars (1-2h)"),
        (9, 12, "9-12 bars (2-3h)"),
        (13, 16, "13-16 bars (3-4h)"),
        (17, 24, "17-24 bars (4-6h)"),
        (25, 32, "25-32 bars (6-8h)"),
        (33, 48, "33-48 bars (8-12h)"),
        (49, 64, "49-64 bars (12-16h)"),
        (65, 96, "65-96 bars (16-24h)"),
        (97, 128, "97-128 bars (24-32h)"),
        (129, 192, "129-192 bars (32-48h)"),
        (193, 9999, "193+ bars (48h+)"),
    ]

    hdr = f"  {'Bucket':<25} {'N':>6} {'WinRate':>8} {'AvgPnL':>10} {'TotalPnL':>12} {'AvgBars':>8} {'%ofAll':>7}"
    sep = f"  {'-'*25} {'-'*6} {'-'*8} {'-'*10} {'-'*12} {'-'*8} {'-'*7}"
    print(hdr, flush=True)
    print(sep, flush=True)

    bucket_data = []
    for lo, hi, label in buckets:
        bt = [t for t in trades if lo <= t['bars_held'] <= hi]
        if not bt:
            continue
        bp = [t['pnl'] for t in bt]
        wr = sum(1 for p in bp if p > 0) / len(bp) * 100
        avg = np.mean(bp)
        total = sum(bp)
        avg_bars = np.mean([t['bars_held'] for t in bt])
        pct = len(bt) / n * 100
        print(f"  {label:<25} {len(bt):>6} {wr:>7.1f}% {avg:>10.2f} {fmt(total):>12} {avg_bars:>8.1f} {pct:>6.1f}%", flush=True)
        bucket_data.append({
            'label': label, 'lo': lo, 'hi': hi, 'n': len(bt),
            'wr': round(wr, 1), 'avg_pnl': round(avg, 2),
            'total_pnl': round(total, 2), 'avg_bars': round(avg_bars, 1),
        })

    results['phase2_buckets'] = bucket_data

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: PnL by UTC entry hour — validate live observation
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print("  Phase 3: PnL by UTC Entry Hour (Live finding: Asia 00-08 best)", flush=True)
    print(f"{'='*80}\n", flush=True)

    hour_data = defaultdict(list)
    for t in trades:
        hour_data[t['entry_hour_utc']].append(t)

    print(f"  {'Hour':>6} {'N':>6} {'WinRate':>8} {'AvgPnL':>10} {'TotalPnL':>12} {'AvgBars':>8} {'BigLoss':>10}", flush=True)
    print(f"  {'-'*6} {'-'*6} {'-'*8} {'-'*10} {'-'*12} {'-'*8} {'-'*10}", flush=True)

    hour_results = []
    for h in range(24):
        ht = hour_data.get(h, [])
        if not ht:
            continue
        hp = [t['pnl'] for t in ht]
        wr = sum(1 for p in hp if p > 0) / len(hp) * 100
        avg = np.mean(hp)
        total = sum(hp)
        avg_bars = np.mean([t['bars_held'] for t in ht])
        big_loss = min(hp)
        print(f"  {h:>4}h {len(ht):>6} {wr:>7.1f}% {avg:>10.2f} {fmt(total):>12} {avg_bars:>8.1f} {big_loss:>+10.2f}", flush=True)
        hour_results.append({
            'hour': h, 'n': len(ht), 'wr': round(wr, 1),
            'avg_pnl': round(avg, 2), 'total_pnl': round(total, 2),
            'avg_bars': round(avg_bars, 1), 'worst': round(big_loss, 2),
        })

    # Session summary
    sessions = [
        ("Asia (00-07 UTC)", list(range(0, 8))),
        ("London (08-12 UTC)", list(range(8, 13))),
        ("US (13-20 UTC)", list(range(13, 21))),
        ("Late (21-23 UTC)", list(range(21, 24))),
    ]

    print(f"\n  Session Summary:", flush=True)
    session_data = []
    for sname, hours in sessions:
        st = [t for t in trades if t['entry_hour_utc'] in hours]
        if not st:
            continue
        sp = [t['pnl'] for t in st]
        wr = sum(1 for p in sp if p > 0) / len(sp) * 100
        total = sum(sp)
        avg = np.mean(sp)
        ds_s = trades_to_daily(st)
        sh = sharpe(ds_s.values)
        print(f"    {sname:<25}: n={len(st):>5}, WR={wr:.1f}%, PnL={fmt(total)}, "
              f"Avg={avg:.2f}, Sharpe={sh:.2f}", flush=True)
        session_data.append({
            'session': sname, 'hours': hours, 'n': len(st), 'wr': round(wr, 1),
            'total_pnl': round(total, 2), 'avg_pnl': round(avg, 2), 'sharpe': round(sh, 2),
        })

    results['phase3_hours'] = hour_results
    results['phase3_sessions'] = session_data

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Exit reason breakdown
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print("  Phase 4: Exit Reason Analysis", flush=True)
    print(f"{'='*80}\n", flush=True)

    reason_data = defaultdict(list)
    for t in trades:
        reason_data[t['reason']].append(t)

    print(f"  {'Reason':<20} {'N':>6} {'WinRate':>8} {'AvgPnL':>10} {'TotalPnL':>12} {'AvgBars':>8}", flush=True)
    print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*10} {'-'*12} {'-'*8}", flush=True)

    exit_data = []
    for reason in sorted(reason_data.keys()):
        rt = reason_data[reason]
        rp = [t['pnl'] for t in rt]
        wr = sum(1 for p in rp if p > 0) / len(rp) * 100
        total = sum(rp)
        avg = np.mean(rp)
        avg_bars = np.mean([t['bars_held'] for t in rt])
        print(f"  {reason:<20} {len(rt):>6} {wr:>7.1f}% {avg:>10.2f} {fmt(total):>12} {avg_bars:>8.1f}", flush=True)
        exit_data.append({
            'reason': reason, 'n': len(rt), 'wr': round(wr, 1),
            'avg_pnl': round(avg, 2), 'total_pnl': round(total, 2),
            'avg_bars': round(avg_bars, 1),
        })

    results['phase4_exit_reasons'] = exit_data

    # Cross: exit reason by session
    print(f"\n  Exit reason by entry session:", flush=True)
    for sname, hours in sessions:
        st = [t for t in trades if t['entry_hour_utc'] in hours]
        if not st:
            continue
        reason_counts = defaultdict(int)
        reason_pnl = defaultdict(float)
        for t in st:
            reason_counts[t['reason']] += 1
            reason_pnl[t['reason']] += t['pnl']
        parts = []
        for r in sorted(reason_counts.keys()):
            parts.append(f"{r}={reason_counts[r]}({reason_pnl[r]:+.0f})")
        print(f"    {sname:<25}: {', '.join(parts)}", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Phase 5: MaxHold M15 Sweep (engine-based)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print("  Phase 5: MaxHold M15 Sweep (Engine-based)", flush=True)
    print(f"{'='*80}\n", flush=True)

    hold_grid = [4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 0]
    hold_results_list = []

    for mh in hold_grid:
        label = f"MaxHold={mh:>3}" if mh > 0 else "Default    "
        t_list = run_l8(bundle, cap=35, keltner_max_hold_m15=mh)
        s = print_stats(t_list, label)
        s['max_hold_m15'] = mh
        hold_results_list.append(s)

    results['phase5_hold_sweep'] = hold_results_list

    best_hold = max(hold_results_list, key=lambda x: x.get('sharpe', 0))
    best_mh = best_hold['max_hold_m15']
    print(f"\n  >>> Best MaxHold: {best_mh} M15 bars ({best_mh*15/60:.0f}h) — "
          f"Sharpe={best_hold['sharpe']:.2f}", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Phase 6: Session filter sweep
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print("  Phase 6: Session Filter Sweep (block bad UTC hours)", flush=True)
    print(f"{'='*80}\n", flush=True)

    session_configs = [
        ("All hours (baseline)", None),
        ("Asia only (00-07)", list(range(0, 8))),
        ("Asia+London (00-12)", list(range(0, 13))),
        ("Asia+London+earlyUS (00-16)", list(range(0, 17))),
        ("Block US close (00-12,21-23)", list(range(0, 13)) + list(range(21, 24))),
        ("Block late US (no 17-20)", [h for h in range(24) if h not in [17, 18, 19, 20]]),
        ("Block 13-20 US core", [h for h in range(24) if h not in range(13, 21)]),
    ]

    session_results = []
    for label, hours in session_configs:
        t_list = run_l8(bundle, cap=35, h1_allowed_sessions=hours)
        s = print_stats(t_list, label)
        s['label'] = label
        s['hours'] = hours
        session_results.append(s)

    results['phase6_session_sweep'] = session_results

    best_session = max(session_results, key=lambda x: x.get('sharpe', 0))
    print(f"\n  >>> Best session: {best_session['label']} — Sharpe={best_session['sharpe']:.2f}", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Phase 7: Combined MaxHold + Session filter
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print("  Phase 7: Combined MaxHold + Session Optimization", flush=True)
    print(f"{'='*80}\n", flush=True)

    combo_configs = []
    top_holds = [mh for mh in [best_mh, 16, 24, 32, 48] if mh > 0]
    top_holds = sorted(set(top_holds))

    top_sessions = [
        ("All", None),
        ("Asia+London 00-12", list(range(0, 13))),
        ("No US core 13-20", [h for h in range(24) if h not in range(13, 21)]),
        ("Asia+London+earlyUS 00-16", list(range(0, 17))),
    ]

    combo_results = []
    print(f"  {'MaxHold':>10} {'Session':<30} {'N':>6} {'Sharpe':>7} {'PnL':>12} {'MaxDD':>10} {'WR':>7}", flush=True)
    print(f"  {'-'*10} {'-'*30} {'-'*6} {'-'*7} {'-'*12} {'-'*10} {'-'*7}", flush=True)

    for mh in top_holds + [0]:
        for slabel, shours in top_sessions:
            t_list = run_l8(bundle, cap=35, keltner_max_hold_m15=mh,
                            h1_allowed_sessions=shours)
            tp = [t['pnl'] for t in t_list]
            ds_c = trades_to_daily(t_list)
            n_c = len(t_list)
            sh = sharpe(ds_c.values)
            pnl_c = sum(tp)
            dd_c = max_dd(ds_c.values)
            wr_c = sum(1 for p in tp if p > 0) / max(n_c, 1) * 100
            mh_label = f"MH={mh}" if mh > 0 else "Default"
            print(f"  {mh_label:>10} {slabel:<30} {n_c:>6} {sh:>7.2f} {fmt(pnl_c):>12} {fmt(dd_c):>10} {wr_c:>6.1f}%", flush=True)
            combo_results.append({
                'max_hold': mh, 'session': slabel, 'hours': shours,
                'n': n_c, 'sharpe': round(sh, 2), 'pnl': round(pnl_c, 2),
                'max_dd': round(dd_c, 2), 'wr': round(wr_c, 1),
            })

    results['phase7_combos'] = combo_results

    best_combo = max(combo_results, key=lambda x: x['sharpe'])
    print(f"\n  >>> Best combo: MaxHold={best_combo['max_hold']}, "
          f"Session={best_combo['session']} — Sharpe={best_combo['sharpe']:.2f}", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Phase 8: K-Fold validation
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print("  Phase 8: K-Fold Validation", flush=True)
    print(f"{'='*80}\n", flush=True)

    FOLDS = [
        ("Fold1", "2015-01-01", "2017-03-01"),
        ("Fold2", "2017-03-01", "2019-06-01"),
        ("Fold3", "2019-06-01", "2021-09-01"),
        ("Fold4", "2021-09-01", "2023-12-01"),
        ("Fold5", "2023-12-01", "2027-01-01"),
    ]

    configs_to_validate = [
        ("Default", 0, None),
        (f"MaxHold={best_mh}", best_mh, None),
        (f"BestCombo(MH={best_combo['max_hold']},{best_combo['session']})",
         best_combo['max_hold'], best_combo['hours']),
    ]

    if best_combo['max_hold'] == best_mh and best_combo['hours'] is None:
        configs_to_validate = configs_to_validate[:2]

    kfold_results = {}
    for cfg_label, mh, hours in configs_to_validate:
        fold_sharpes = []
        for fold_name, start, end in FOLDS:
            try:
                fb = bundle.slice(start, end)
            except Exception:
                fold_sharpes.append(0.0)
                continue
            t_f = run_l8(fb, cap=35, keltner_max_hold_m15=mh,
                         h1_allowed_sessions=hours)
            ds_f = trades_to_daily(t_f)
            fold_sharpes.append(sharpe(ds_f.values))

        pos = sum(1 for s in fold_sharpes if s > 0)
        mean_sh = np.mean(fold_sharpes)
        print(f"  {cfg_label:<50}: folds=[{', '.join(f'{s:.2f}' for s in fold_sharpes)}], "
              f"pos={pos}/5, mean={mean_sh:.2f}", flush=True)

        kfold_results[cfg_label] = {
            'folds': [round(s, 2) for s in fold_sharpes],
            'positive_folds': pos,
            'mean_sharpe': round(float(mean_sh), 2),
        }

    results['phase8_kfold'] = kfold_results

    # ═══════════════════════════════════════════════════════════════
    # Phase 9: Final recommendation (lot=0.05 production)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}", flush=True)
    print("  Phase 9: Final Recommendation (lot=0.05 production)", flush=True)
    print(f"{'='*80}\n", flush=True)

    final_configs = [
        ("Current production (default)", 0, None),
        (f"Optimized MaxHold={best_mh}", best_mh, None),
        (f"BestCombo(MH={best_combo['max_hold']},{best_combo['session']})",
         best_combo['max_hold'], best_combo['hours']),
    ]

    final_data = []
    for label, mh, hours in final_configs:
        t_list = run_l8(bundle, cap=35, lot=0.05, keltner_max_hold_m15=mh,
                        h1_allowed_sessions=hours)
        tp = [t['pnl'] for t in t_list]
        ds_f = trades_to_daily(t_list)
        n_f = len(t_list)
        sh = sharpe(ds_f.values)
        pnl_f = sum(tp)
        dd_f = max_dd(ds_f.values)
        wr = sum(1 for p in tp if p > 0) / max(n_f, 1) * 100
        avg = np.mean(tp) if tp else 0
        bars = np.mean([t['bars_held'] for t in t_list]) if t_list else 0

        print(f"  {label}:", flush=True)
        print(f"    n={n_f}, Sharpe={sh:.2f}, PnL={fmt(pnl_f)}, MaxDD={fmt(dd_f)}", flush=True)
        print(f"    WR={wr:.1f}%, AvgPnL={avg:.2f}, AvgBars={bars:.1f}", flush=True)

        final_data.append({
            'label': label, 'max_hold': mh, 'sessions': hours,
            'n': n_f, 'sharpe': round(sh, 2), 'pnl': round(pnl_f, 2),
            'max_dd': round(dd_f, 2), 'wr': round(wr, 1), 'avg_pnl': round(avg, 2),
        })

    results['phase9_final'] = final_data

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n{'='*80}", flush=True)
    print(f"  R155 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"{'='*80}", flush=True)

    results['elapsed_s'] = round(elapsed, 1)
    with open(OUTPUT_DIR / "r155_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_DIR}/r155_results.json", flush=True)


if __name__ == "__main__":
    main()
