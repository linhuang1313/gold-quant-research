#!/usr/bin/env python3
"""
R157 — Session-based Dynamic Lot Sizing for Keltner (L8_MAX)
==============================================================
R155 showed clear session quality differences:
  London (08-12 UTC): Sharpe 4.00, WR 87.5%
  US (13-20 UTC):     Sharpe 4.36, WR 84.3%  (but live slippage worst here)
  Asia (00-07 UTC):   Sharpe 2.57, WR 81.1%
  Late (21-23 UTC):   Sharpe 2.38, WR 79.2%

Idea: allocate larger lots to high-quality sessions, smaller lots to weak
sessions, while keeping weighted-average lot = baseline (risk-neutral).

All tests use MaxHold=8 (R155) + current trailing (R156 confirmed optimal).

Phases:
  1: Baseline (fixed lot=0.01 unit, then 0.05 production)
  2: 2-tier lots (good=0.07, bad=0.03)
  3: 3-tier lots (best=0.08, mid=0.05, worst=0.02)
  4: Grid sweep — sweep good_lot / bad_lot combos, weighted avg = 0.05
  5: 5-Fold K-Fold validation on top 3 configs
  6: Production comparison at 0.05 base lot
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r157_session_lot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

t0 = time.time()

ASIA_HOURS  = list(range(0, 8))     # 00-07 UTC
LONDON_HOURS = list(range(8, 13))   # 08-12 UTC
EARLY_US     = list(range(13, 17))  # 13-16 UTC
LATE_US      = list(range(17, 21))  # 17-20 UTC
LATE_HOURS   = list(range(21, 24))  # 21-23 UTC


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
    return float((np.maximum.accumulate(eq) - eq).max())


def trades_to_daily(trades):
    if not trades: return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


def build_session_lot_map(asia, london, early_us, late_us, late):
    """Build {utc_hour: lot_size} dict from per-session lot sizes."""
    m = {}
    for h in ASIA_HOURS:   m[h] = asia
    for h in LONDON_HOURS: m[h] = london
    for h in EARLY_US:     m[h] = early_us
    for h in LATE_US:      m[h] = late_us
    for h in LATE_HOURS:   m[h] = late
    return m


def weighted_avg_lot(slm):
    """Approximate weighted-average lot (assuming uniform entry distribution)."""
    return sum(slm.values()) / len(slm)


def run_l8(bundle, spread=0.30, lot=0.01, cap=35,
           keltner_max_hold_m15=8, session_lot_map=None):
    from backtest.runner import run_variant, LIVE_PARITY_KWARGS
    kw = dict(LIVE_PARITY_KWARGS)
    kw['maxloss_cap'] = cap
    kw['spread_cost'] = spread
    kw['initial_capital'] = 2000
    kw['min_lot_size'] = lot
    kw['max_lot_size'] = lot
    kw['keltner_max_hold_m15'] = keltner_max_hold_m15
    if session_lot_map is not None:
        kw['session_lot_map'] = session_lot_map
        all_lots = list(session_lot_map.values())
        kw['min_lot_size'] = min(all_lots)
        kw['max_lot_size'] = max(all_lots)
    result = run_variant(bundle, "L8_MAX", verbose=False, **kw)
    raw = result.get('_trades', [])
    trades = []
    for t in raw:
        entry_ts = pd.Timestamp(t.entry_time)
        entry_utc = entry_ts.tz_localize('UTC') if entry_ts.tzinfo is None else entry_ts.tz_convert('UTC')
        trades.append({
            'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
            'entry_time': t.entry_time, 'exit_time': t.exit_time,
            'entry_hour_utc': entry_utc.hour,
            'pnl': t.pnl, 'reason': t.exit_reason, 'bars_held': t.bars_held,
            'lots': t.lots,
        })
    return trades


def stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'max_dd': 0, 'wr': 0,
                'avg_pnl': 0, 'avg_bars': 0, 'avg_lot': 0}
    pnls = [t['pnl'] for t in trades]
    ds = trades_to_daily(trades)
    n = len(trades)
    lots_list = [t.get('lots', 0.01) for t in trades]
    trailing = [t for t in trades if 'Trailing' in str(t.get('reason', ''))]
    trail_pnls = [t['pnl'] for t in trailing] if trailing else [0]
    return {
        'n': n, 'sharpe': round(sharpe(ds.values), 2),
        'pnl': round(sum(pnls), 2), 'max_dd': round(max_dd(ds.values), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'avg_pnl': round(np.mean(pnls), 2),
        'avg_bars': round(np.mean([t['bars_held'] for t in trades]), 1),
        'avg_lot': round(np.mean(lots_list), 4),
        'trail_n': len(trailing),
        'trail_avg': round(np.mean(trail_pnls), 2),
        'trail_wr': round(sum(1 for p in trail_pnls if p > 0) / max(len(trailing), 1) * 100, 1),
    }


def session_breakdown(trades):
    """PnL breakdown by session for analysis."""
    sessions = {
        'Asia (00-07)': ASIA_HOURS,
        'London (08-12)': LONDON_HOURS,
        'EarlyUS (13-16)': EARLY_US,
        'LateUS (17-20)': LATE_US,
        'Late (21-23)': LATE_HOURS,
    }
    result = {}
    for name, hours in sessions.items():
        sess_trades = [t for t in trades if t['entry_hour_utc'] in hours]
        if sess_trades:
            pnls = [t['pnl'] for t in sess_trades]
            lots = [t.get('lots', 0.01) for t in sess_trades]
            result[name] = {
                'n': len(sess_trades),
                'pnl': round(sum(pnls), 2),
                'avg_pnl': round(np.mean(pnls), 2),
                'wr': round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
                'avg_lot': round(np.mean(lots), 4),
            }
    return result


def print_row(label, s):
    print(f"  {label:<50} n={s['n']:>5} Sh={s['sharpe']:>5.2f} PnL={fmt(s['pnl'])} "
          f"DD={fmt(s['max_dd'])} WR={s['wr']:.0f}% AvgPnL={s['avg_pnl']:.2f} "
          f"AvgLot={s['avg_lot']:.4f}", flush=True)


def main():
    results = {}

    print("=" * 100, flush=True)
    print("  R157 — Session-based Dynamic Lot Sizing (Keltner)", flush=True)
    print(f"  Started: {datetime.now()}", flush=True)
    print("=" * 100, flush=True)

    from backtest.runner import DataBundle
    print("\n  Loading DataBundle...", flush=True)
    bundle = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    print("  Bundle ready.\n", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Baseline (fixed lot)
    # ═══════════════════════════════════════════════════════════════
    print(f"{'='*100}", flush=True)
    print("  Phase 1: Baseline (MaxHold=8, fixed lot=0.01)", flush=True)
    print(f"{'='*100}\n", flush=True)

    trades_base = run_l8(bundle, lot=0.01)
    base = stats(trades_base)
    print_row("Baseline (fixed lot=0.01)", base)
    results['phase1_baseline'] = base

    base_sess = session_breakdown(trades_base)
    print("\n  Session breakdown (baseline):", flush=True)
    for name, sb in base_sess.items():
        print(f"    {name:<25} n={sb['n']:>5}  PnL={fmt(sb['pnl'])}  "
              f"WR={sb['wr']:.0f}%  AvgPnL={sb['avg_pnl']:.2f}  Lot={sb['avg_lot']:.4f}", flush=True)
    results['phase1_session_breakdown'] = base_sess

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: 2-tier lots (good vs bad sessions)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("  Phase 2: 2-Tier Session Lots", flush=True)
    print("  Good sessions (London 08-12 + EarlyUS 13-16) = higher lot", flush=True)
    print("  Bad sessions (Asia 00-07 + LateUS 17-20 + Late 21-23) = lower lot", flush=True)
    print(f"{'='*100}\n", flush=True)

    tier2_configs = [
        ("good=0.013 bad=0.008", 0.013, 0.008),
        ("good=0.015 bad=0.007", 0.015, 0.007),
        ("good=0.017 bad=0.005", 0.017, 0.005),
        ("good=0.020 bad=0.003", 0.020, 0.003),
    ]
    tier2_results = []
    for label, good_lot, bad_lot in tier2_configs:
        slm = build_session_lot_map(
            asia=bad_lot, london=good_lot, early_us=good_lot,
            late_us=bad_lot, late=bad_lot
        )
        wavg = weighted_avg_lot(slm)
        trades_t2 = run_l8(bundle, session_lot_map=slm)
        s = stats(trades_t2)
        s['good_lot'] = good_lot
        s['bad_lot'] = bad_lot
        s['weighted_avg_lot'] = round(wavg, 4)
        print_row(f"{label} (wavg={wavg:.4f})", s)
        tier2_results.append(s)

        sess = session_breakdown(trades_t2)
        s['session_breakdown'] = sess

    results['phase2_tier2'] = tier2_results

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: 3-tier lots (best / mid / worst)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("  Phase 3: 3-Tier Session Lots", flush=True)
    print("  Best = London (08-12)", flush=True)
    print("  Mid  = EarlyUS (13-16)", flush=True)
    print("  Worst = Asia (00-07) + LateUS (17-20) + Late (21-23)", flush=True)
    print(f"{'='*100}\n", flush=True)

    tier3_configs = [
        ("best=0.015 mid=0.010 worst=0.007", 0.015, 0.010, 0.007),
        ("best=0.018 mid=0.010 worst=0.005", 0.018, 0.010, 0.005),
        ("best=0.020 mid=0.010 worst=0.004", 0.020, 0.010, 0.004),
        ("best=0.025 mid=0.010 worst=0.003", 0.025, 0.010, 0.003),
        ("best=0.020 mid=0.015 worst=0.003", 0.020, 0.015, 0.003),
    ]
    tier3_results = []
    for label, best_lot, mid_lot, worst_lot in tier3_configs:
        slm = build_session_lot_map(
            asia=worst_lot, london=best_lot, early_us=mid_lot,
            late_us=worst_lot, late=worst_lot
        )
        wavg = weighted_avg_lot(slm)
        trades_t3 = run_l8(bundle, session_lot_map=slm)
        s = stats(trades_t3)
        s['best_lot'] = best_lot
        s['mid_lot'] = mid_lot
        s['worst_lot'] = worst_lot
        s['weighted_avg_lot'] = round(wavg, 4)
        print_row(f"{label} (wavg={wavg:.4f})", s)
        tier3_results.append(s)

        sess = session_breakdown(trades_t3)
        s['session_breakdown'] = sess

    results['phase3_tier3'] = tier3_results

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: 5-session grid sweep
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("  Phase 4: 5-Session Grid Sweep", flush=True)
    print("  Sweep lot for each session independently", flush=True)
    print("  Constraint: weighted avg lot ~= 0.01 (unit lot)", flush=True)
    print(f"{'='*100}\n", flush=True)

    # Session hour counts for weighting
    n_hours = {
        'asia': len(ASIA_HOURS),       # 8
        'london': len(LONDON_HOURS),   # 5
        'early_us': len(EARLY_US),     # 4
        'late_us': len(LATE_US),       # 4
        'late': len(LATE_HOURS),       # 3
    }
    total_hours = sum(n_hours.values())  # 24

    lot_choices = [0.003, 0.005, 0.007, 0.010, 0.013, 0.015, 0.018, 0.020, 0.025]

    # To keep grid manageable, sweep 2 axes:
    #   axis1: London lot (best session)
    #   axis2: Asia/LateUS/Late lot (worst sessions)
    #   EarlyUS = derived to keep weighted avg = 0.01
    grid_results = []
    for london_lot in [0.012, 0.015, 0.018, 0.020, 0.025, 0.030]:
        for bad_lot in [0.003, 0.005, 0.007, 0.008, 0.010]:
            # Solve for early_us lot to maintain wavg=0.01
            # wavg = (8*bad + 5*london + 4*early_us + 4*bad + 3*bad) / 24 = 0.01
            # 15*bad + 5*london + 4*early_us = 0.24
            # early_us = (0.24 - 15*bad - 5*london) / 4
            early_us_lot = (0.24 - 15 * bad_lot - 5 * london_lot) / 4
            if early_us_lot < 0.002 or early_us_lot > 0.035:
                continue

            early_us_lot = round(early_us_lot, 3)
            slm = build_session_lot_map(
                asia=bad_lot, london=london_lot, early_us=early_us_lot,
                late_us=bad_lot, late=bad_lot
            )
            wavg = weighted_avg_lot(slm)

            trades_g = run_l8(bundle, session_lot_map=slm)
            s = stats(trades_g)
            s['london_lot'] = london_lot
            s['bad_lot'] = bad_lot
            s['early_us_lot'] = early_us_lot
            s['weighted_avg_lot'] = round(wavg, 4)
            print_row(f"Ldn={london_lot:.3f} Bad={bad_lot:.3f} EUS={early_us_lot:.3f} (wavg={wavg:.4f})", s)
            grid_results.append(s)

    grid_results.sort(key=lambda x: x['sharpe'], reverse=True)
    results['phase4_grid'] = grid_results

    print(f"\n  Grid: {len(grid_results)} combos tested", flush=True)
    print("  Top 5 by Sharpe:", flush=True)
    for i, s in enumerate(grid_results[:5]):
        print(f"    #{i+1} Ldn={s['london_lot']:.3f} Bad={s['bad_lot']:.3f} "
              f"EUS={s['early_us_lot']:.3f} -> Sh={s['sharpe']:.2f} PnL={fmt(s['pnl'])} "
              f"DD={fmt(s['max_dd'])}", flush=True)

    # Top 5 by PnL
    grid_by_pnl = sorted(grid_results, key=lambda x: x['pnl'], reverse=True)
    print("\n  Top 5 by PnL:", flush=True)
    for i, s in enumerate(grid_by_pnl[:5]):
        print(f"    #{i+1} Ldn={s['london_lot']:.3f} Bad={s['bad_lot']:.3f} "
              f"EUS={s['early_us_lot']:.3f} -> Sh={s['sharpe']:.2f} PnL={fmt(s['pnl'])} "
              f"DD={fmt(s['max_dd'])}", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Phase 5: K-Fold validation on top configs
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("  Phase 5: 5-Fold K-Fold Validation", flush=True)
    print(f"{'='*100}\n", flush=True)

    m15_dates = bundle.m15_df.index
    start_date = str(m15_dates[0].date())
    end_date = str(m15_dates[-1].date())
    total_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days

    fold_configs = []
    n_folds = 5
    fold_days = total_days // n_folds
    for i in range(n_folds):
        fs = pd.Timestamp(start_date) + pd.Timedelta(days=i * fold_days)
        fe = fs + pd.Timedelta(days=fold_days) if i < n_folds - 1 else pd.Timestamp(end_date)
        fold_configs.append((str(fs.date()), str(fe.date())))

    # Configs to validate:
    kfold_candidates = [
        ("Baseline (fixed lot=0.01)", None),
    ]
    # Best Sharpe from grid
    if grid_results:
        best_sh = grid_results[0]
        kfold_candidates.append((
            f"BestSharpe (Ldn={best_sh['london_lot']:.3f} Bad={best_sh['bad_lot']:.3f})",
            build_session_lot_map(
                asia=best_sh['bad_lot'], london=best_sh['london_lot'],
                early_us=best_sh['early_us_lot'],
                late_us=best_sh['bad_lot'], late=best_sh['bad_lot']
            )
        ))
    # Best PnL from grid
    if grid_by_pnl and grid_by_pnl[0] != grid_results[0]:
        best_pnl = grid_by_pnl[0]
        kfold_candidates.append((
            f"BestPnL (Ldn={best_pnl['london_lot']:.3f} Bad={best_pnl['bad_lot']:.3f})",
            build_session_lot_map(
                asia=best_pnl['bad_lot'], london=best_pnl['london_lot'],
                early_us=best_pnl['early_us_lot'],
                late_us=best_pnl['bad_lot'], late=best_pnl['bad_lot']
            )
        ))
    # Best 2-tier from Phase 2
    if tier2_results:
        best_t2 = max(tier2_results, key=lambda x: x['sharpe'])
        kfold_candidates.append((
            f"Best2Tier (good={best_t2['good_lot']:.3f} bad={best_t2['bad_lot']:.3f})",
            build_session_lot_map(
                asia=best_t2['bad_lot'], london=best_t2['good_lot'],
                early_us=best_t2['good_lot'],
                late_us=best_t2['bad_lot'], late=best_t2['bad_lot']
            )
        ))

    kfold_results = {}
    for label, slm in kfold_candidates:
        print(f"  {label}:", flush=True)
        fold_sharpes = []
        fold_pnls = []
        for fi, (fs, fe) in enumerate(fold_configs):
            fold_bundle = bundle.slice(fs, fe)
            if slm is None:
                t_fold = run_l8(fold_bundle, lot=0.01)
            else:
                t_fold = run_l8(fold_bundle, session_lot_map=slm)
            s = stats(t_fold)
            fold_sharpes.append(s['sharpe'])
            fold_pnls.append(s['pnl'])
            print(f"    Fold {fi+1} ({fs} ~ {fe}): Sh={s['sharpe']:.2f} PnL={fmt(s['pnl'])}", flush=True)

        pos_folds = sum(1 for sh in fold_sharpes if sh > 0)
        mean_sh = round(np.mean(fold_sharpes), 2)
        print(f"    -> Mean Sharpe = {mean_sh:.2f}, Positive folds = {pos_folds}/5\n", flush=True)

        kfold_results[label] = {
            'folds_sharpe': fold_sharpes,
            'folds_pnl': fold_pnls,
            'positive_folds': pos_folds,
            'mean_sharpe': mean_sh,
        }
    results['phase5_kfold'] = kfold_results

    # ═══════════════════════════════════════════════════════════════
    # Phase 6: Production comparison (lot=0.05 base)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("  Phase 6: Production Comparison (base lot=0.05)", flush=True)
    print(f"{'='*100}\n", flush=True)

    prod_results = []

    # Baseline fixed
    trades_prod_base = run_l8(bundle, lot=0.05)
    s = stats(trades_prod_base)
    s['label'] = "Fixed lot=0.05 (current production)"
    print_row(s['label'], s)
    prod_results.append(s)

    sess = session_breakdown(trades_prod_base)
    print("    Session breakdown:", flush=True)
    for name, sb in sess.items():
        print(f"      {name:<25} n={sb['n']:>5}  PnL={fmt(sb['pnl'])}  "
              f"WR={sb['wr']:.0f}%  AvgPnL={sb['avg_pnl']:.2f}", flush=True)

    # Best configs scaled to 0.05 base
    if grid_results:
        best_sh = grid_results[0]
        scale = 0.05 / 0.01  # 5x
        slm = build_session_lot_map(
            asia=round(best_sh['bad_lot'] * scale, 2),
            london=round(best_sh['london_lot'] * scale, 2),
            early_us=round(best_sh['early_us_lot'] * scale, 2),
            late_us=round(best_sh['bad_lot'] * scale, 2),
            late=round(best_sh['bad_lot'] * scale, 2),
        )
        trades_prod_best = run_l8(bundle, session_lot_map=slm)
        s = stats(trades_prod_best)
        s['label'] = (f"BestSharpe scaled (Ldn={best_sh['london_lot']*scale:.2f} "
                      f"Bad={best_sh['bad_lot']*scale:.2f} EUS={best_sh['early_us_lot']*scale:.2f})")
        s['session_lot_map'] = {str(k): v for k, v in slm.items()}
        print_row(s['label'], s)
        prod_results.append(s)

        sess = session_breakdown(trades_prod_best)
        print("    Session breakdown:", flush=True)
        for name, sb in sess.items():
            print(f"      {name:<25} n={sb['n']:>5}  PnL={fmt(sb['pnl'])}  "
                  f"WR={sb['wr']:.0f}%  AvgPnL={sb['avg_pnl']:.2f}  Lot={sb['avg_lot']:.2f}", flush=True)

    if grid_by_pnl and grid_by_pnl[0] != grid_results[0]:
        best_pnl = grid_by_pnl[0]
        scale = 0.05 / 0.01
        slm = build_session_lot_map(
            asia=round(best_pnl['bad_lot'] * scale, 2),
            london=round(best_pnl['london_lot'] * scale, 2),
            early_us=round(best_pnl['early_us_lot'] * scale, 2),
            late_us=round(best_pnl['bad_lot'] * scale, 2),
            late=round(best_pnl['bad_lot'] * scale, 2),
        )
        trades_prod_pnl = run_l8(bundle, session_lot_map=slm)
        s = stats(trades_prod_pnl)
        s['label'] = (f"BestPnL scaled (Ldn={best_pnl['london_lot']*scale:.2f} "
                      f"Bad={best_pnl['bad_lot']*scale:.2f} EUS={best_pnl['early_us_lot']*scale:.2f})")
        s['session_lot_map'] = {str(k): v for k, v in slm.items()}
        print_row(s['label'], s)
        prod_results.append(s)

    results['phase6_production'] = prod_results

    # ═══════════════════════════════════════════════════════════════
    # Save
    # ═══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)

    out_file = OUTPUT_DIR / "r157_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*100}", flush=True)
    print(f"  R157 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"{'='*100}\n", flush=True)
    print(f"  Saved: {out_file}\n", flush=True)


if __name__ == "__main__":
    main()
