"""
R39: L8 Strategy Validation (Same Method as R38)
==================================================
用 R38 同样的方法对 L8 三个变体做完整验证:

Phase 1: L8a/L8b/L8c 全样本 + K-Fold (corrected Sharpe)
Phase 2: KCBW Lookback 参数敏感度 (L8c best vs L7 baseline)
Phase 3: Spread 衰减曲线
Phase 4: L7 vs L8 对比总结
"""
import sys, os, time, copy, multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.runner import DataBundle, run_variant, run_kfold, LIVE_PARITY_KWARGS
from backtest.stats import calc_stats, aggregate_daily_pnl
from backtest.engine import TradeRecord
import research_config as config

OUT_DIR = Path("results/round39_l8_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = min(mp.cpu_count(), 8)

# L7 baseline (MH=8, current production)
L7_MH8 = {
    **LIVE_PARITY_KWARGS,
    'time_adaptive_trail': {'start': 2, 'decay': 0.75, 'floor': 0.003},
    'min_entry_gap_hours': 1.0,
    'keltner_max_hold_m15': 8,
}

# L8 variants — only high regime trail differs from L7
L8A_MH8 = {
    **L7_MH8,
    'regime_config': {
        'low':    {'trail_act': 0.40, 'trail_dist': 0.10},
        'normal': {'trail_act': 0.28, 'trail_dist': 0.06},
        'high':   {'trail_act': 0.12, 'trail_dist': 0.01},
    },
}

L8B_MH8 = {
    **L7_MH8,
    'regime_config': {
        'low':    {'trail_act': 0.40, 'trail_dist': 0.10},
        'normal': {'trail_act': 0.28, 'trail_dist': 0.06},
        'high':   {'trail_act': 0.08, 'trail_dist': 0.01},
    },
}

L8C_MH8 = {
    **L7_MH8,
    'regime_config': {
        'low':    {'trail_act': 0.40, 'trail_dist': 0.10},
        'normal': {'trail_act': 0.28, 'trail_dist': 0.06},
        'high':   {'trail_act': 0.06, 'trail_dist': 0.005},
    },
}

STRATEGIES = {
    'L7_Baseline': L7_MH8,
    'L8a_high12_01': L8A_MH8,
    'L8b_high08_01': L8B_MH8,
    'L8c_high06_005': L8C_MH8,
}


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            try: f.write(data)
            except: pass
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def corrected_sharpe(trades, start_date=None, end_date=None):
    if not trades:
        return 0.0
    trade_daily = {}
    for t in trades:
        d = pd.Timestamp(t.exit_time).date()
        trade_daily[d] = trade_daily.get(d, 0) + t.pnl
    if start_date is None:
        start_date = min(trade_daily.keys())
    if end_date is None:
        end_date = max(trade_daily.keys())
    all_dates = pd.bdate_range(start_date, end_date)
    full_daily = [trade_daily.get(d.date(), 0.0) for d in all_dates]
    arr = np.array(full_daily)
    if len(arr) < 2 or np.std(arr, ddof=1) <= 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))


def apply_max_loss_cap(trades, cap_usd):
    capped = []
    for t in trades:
        if t.pnl < -cap_usd:
            capped.append(TradeRecord(
                strategy=t.strategy, direction=t.direction,
                entry_price=t.entry_price, exit_price=t.exit_price,
                entry_time=t.entry_time, exit_time=t.exit_time,
                lots=t.lots, pnl=-cap_usd, exit_reason=t.exit_reason,
                bars_held=t.bars_held,
            ))
        else:
            capped.append(t)
    return capped


def _run_one(args):
    label, base_kwargs, spread = args
    data = DataBundle.load_default()
    kw = {**base_kwargs, 'spread_cost': spread}
    r = run_variant(data, label, verbose=False, **kw)
    csh = corrected_sharpe(r['_trades'])
    return {
        'label': label, 'spread': spread,
        'n': r['n'], 'total_pnl': r['total_pnl'], 'win_rate': r['win_rate'],
        'avg_win': r['avg_win'], 'avg_loss': r['avg_loss'], 'rr': r['rr'],
        'max_dd': r['max_dd'], 'corr_sharpe': csh, 'orig_sharpe': r['sharpe'],
    }


def _run_kfold_one(args):
    label, base_kwargs, spread, cap = args
    data = DataBundle.load_default()
    kw = {**base_kwargs, 'spread_cost': spread}
    folds = run_kfold(data, kw, n_folds=6)
    results = []
    for f in folds:
        trades_f = f.get('_trades', [])
        if cap < 999 and trades_f:
            trades_f = apply_max_loss_cap(trades_f, cap)
        csh = corrected_sharpe(trades_f)
        pnl = sum(t.pnl for t in trades_f) if trades_f else f['total_pnl']
        wr = f['win_rate']
        results.append({
            'fold': f['label'], 'n': f['n'], 'orig_sharpe': f['sharpe'],
            'corr_sharpe': csh, 'pnl': pnl, 'win_rate': wr,
        })
    return {'label': label, 'folds': results}


def print_kfold(kr):
    print(f"\n  [{kr['label']}]")
    print(f"  {'Fold':<8} {'N':>6} {'OrigSh':>8} {'CorrSh':>8} {'PnL':>10} {'WR%':>6}")
    sharpes = []
    for f in kr['folds']:
        sharpes.append(f['corr_sharpe'])
        print(f"  {f['fold']:<8} {f['n']:>6} {f['orig_sharpe']:>8.2f} {f['corr_sharpe']:>8.2f} "
              f"${f['pnl']:>9.0f} {f['win_rate']:>5.1f}%")
    all_pos = all(s > 0 for s in sharpes)
    pos_count = sum(1 for s in sharpes if s > 0)
    print(f"  Mean={np.mean(sharpes):.2f}, Std={np.std(sharpes):.2f}, "
          f"Positive={pos_count}/6, PASS={'YES' if all_pos else 'NO'}")
    return sharpes


# ═════════════════════════════════════════════════════════════
# Phase 1: L8 variants full-sample + K-Fold
# ═════════════════════════════════════════════════════════════

def phase_1():
    print("\n" + "=" * 90)
    print("  PHASE 1: L7 vs L8a/L8b/L8c — Full Sample + K-Fold (Corrected Sharpe)")
    print(f"  Workers: {MAX_WORKERS}")
    print("=" * 90)

    # 1A: Full sample at multiple spreads
    print("\n  --- 1A: Full Sample ---")
    tasks = []
    for sp in [0.30, 0.50]:
        for sname, skw in STRATEGIES.items():
            tasks.append((f"{sname}_sp{sp}", skw, sp))

    t0 = time.time()
    with mp.Pool(MAX_WORKERS) as pool:
        results = pool.map(_run_one, tasks)
    print(f"  Full sample done in {time.time()-t0:.0f}s")

    for sp in [0.30, 0.50]:
        print(f"\n  === Spread = ${sp:.2f} ===")
        print(f"  {'Strategy':<20} {'N':>6} {'PnL':>10} {'OrigSh':>8} {'CorrSh':>8} "
              f"{'WR%':>6} {'AvgW':>8} {'AvgL':>8} {'RR':>5} {'MaxDD':>8}")
        print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*8} {'-'*8} "
              f"{'-'*6} {'-'*8} {'-'*8} {'-'*5} {'-'*8}")
        for r in results:
            if r['spread'] != sp: continue
            short = r['label'].replace(f'_sp{sp}', '')
            print(f"  {short:<20} {r['n']:>6} ${r['total_pnl']:>9.0f} {r['orig_sharpe']:>8.2f} "
                  f"{r['corr_sharpe']:>8.2f} {r['win_rate']:>5.1f}% "
                  f"${r['avg_win']:>7.2f} ${r['avg_loss']:>7.2f} {r['rr']:>4.2f} "
                  f"${r['max_dd']:>7.0f}")

    # 1B: K-Fold all variants (spread=0.50)
    print("\n  --- 1B: K-Fold 6-Fold (spread=$0.50) ---")
    kfold_tasks = []
    for sname, skw in STRATEGIES.items():
        kfold_tasks.append((f"KF_{sname}", skw, 0.50, 999))

    t0 = time.time()
    with mp.Pool(MAX_WORKERS) as pool:
        kf_results = pool.map(_run_kfold_one, kfold_tasks)
    print(f"  K-Fold done in {time.time()-t0:.0f}s")

    all_kf = {}
    for kr in kf_results:
        sharpes = print_kfold(kr)
        all_kf[kr['label']] = sharpes

    # 1C: K-Fold with KCBW5 + Cap30 (best combo from R38)
    print("\n  --- 1C: K-Fold with KCBW5 + Cap30 (spread=$0.50) ---")
    kfold_combo_tasks = []
    for sname, skw in STRATEGIES.items():
        combo_kw = {**skw, 'kc_bw_filter_bars': 5}
        kfold_combo_tasks.append((f"KF_{sname}+KCBW5+Cap30", combo_kw, 0.50, 30))

    t0 = time.time()
    with mp.Pool(MAX_WORKERS) as pool:
        kf_combo_results = pool.map(_run_kfold_one, kfold_combo_tasks)
    print(f"  K-Fold combo done in {time.time()-t0:.0f}s")

    for kr in kf_combo_results:
        sharpes = print_kfold(kr)
        all_kf[kr['label']] = sharpes

    # Summary table
    print("\n  --- K-Fold Summary ---")
    print(f"  {'Config':<35} {'Mean':>6} {'Std':>6} {'Min':>6} {'P/6':>4} {'PASS':>5}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*4} {'-'*5}")
    for label, sh in all_kf.items():
        pos = sum(1 for s in sh if s > 0)
        passed = 'YES' if pos == 6 else 'NO'
        print(f"  {label:<35} {np.mean(sh):>6.2f} {np.std(sh):>6.2f} "
              f"{min(sh):>6.2f} {pos:>3}/6 {passed:>5}")


# ═════════════════════════════════════════════════════════════
# Phase 2: KCBW Lookback sensitivity for L8c
# ═════════════════════════════════════════════════════════════

def phase_2():
    print("\n" + "=" * 90)
    print("  PHASE 2: KCBW Lookback Sensitivity — L8c vs L7")
    print("=" * 90)

    tasks = []
    for lb in [3, 5, 7, 10]:
        for sname, skw in [('L7', L7_MH8), ('L8c', L8C_MH8)]:
            extra = {**skw, 'kc_bw_filter_bars': lb}
            tasks.append((f"{sname}_KCBW{lb}", extra, 0.50))
    # Baseline (no KCBW)
    for sname, skw in [('L7', L7_MH8), ('L8c', L8C_MH8)]:
        tasks.append((f"{sname}_NoKCBW", skw, 0.50))

    t0 = time.time()
    with mp.Pool(MAX_WORKERS) as pool:
        results = pool.map(_run_one, tasks)
    print(f"  Done in {time.time()-t0:.0f}s")

    for sname in ['L7', 'L8c']:
        print(f"\n  --- {sname} ---")
        print(f"  {'Variant':<20} {'N':>6} {'PnL':>10} {'CorrSh':>8} {'WR%':>6} {'MaxDD':>8}")
        print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*8} {'-'*6} {'-'*8}")
        for r in sorted(results, key=lambda x: x['label']):
            if not r['label'].startswith(sname): continue
            print(f"  {r['label']:<20} {r['n']:>6} ${r['total_pnl']:>9.0f} "
                  f"{r['corr_sharpe']:>8.2f} {r['win_rate']:>5.1f}% ${r['max_dd']:>7.0f}")


# ═════════════════════════════════════════════════════════════
# Phase 3: Spread decay curve — L7 vs L8c
# ═════════════════════════════════════════════════════════════

def phase_3():
    print("\n" + "=" * 90)
    print("  PHASE 3: Spread Decay Curve — L7 vs L8c (+KCBW5)")
    print("=" * 90)

    spreads = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00, 1.50, 2.00]

    configs = [
        ("L7_Base",       L7_MH8),
        ("L7_KCBW5",      {**L7_MH8, 'kc_bw_filter_bars': 5}),
        ("L8c_Base",      L8C_MH8),
        ("L8c_KCBW5",     {**L8C_MH8, 'kc_bw_filter_bars': 5}),
    ]

    tasks = []
    for sp in spreads:
        for cname, ckw in configs:
            tasks.append((f"{cname}_sp{sp}", ckw, sp))

    t0 = time.time()
    with mp.Pool(MAX_WORKERS) as pool:
        all_results = pool.map(_run_one, tasks)
    print(f"  Done in {time.time()-t0:.0f}s")

    for cname, _ in configs:
        print(f"\n  --- {cname} ---")
        print(f"  {'Spread':>8} {'N':>6} {'PnL':>10} {'CorrSh':>8} {'WR%':>6} {'MaxDD':>8}")
        print(f"  {'-'*8} {'-'*6} {'-'*10} {'-'*8} {'-'*6} {'-'*8}")
        for r in all_results:
            if not r['label'].startswith(cname + "_sp"): continue
            sp_str = r['label'].split("_sp")[1]
            print(f"  ${float(sp_str):>6.2f} {r['n']:>6} ${r['total_pnl']:>9.0f} "
                  f"{r['corr_sharpe']:>8.2f} {r['win_rate']:>5.1f}% ${r['max_dd']:>7.0f}")

    # Break-even comparison
    print("\n  --- Break-Even Spread ---")
    for cname, _ in configs:
        prev_sharpe = None
        for r in sorted(all_results, key=lambda x: x['spread']):
            if not r['label'].startswith(cname + "_sp"): continue
            if r['corr_sharpe'] <= 0:
                sp_str = r['label'].split("_sp")[1]
                print(f"  {cname}: Sharpe <= 0 at spread >= ${float(sp_str):.2f}")
                break
            prev_sharpe = r['corr_sharpe']


# ═════════════════════════════════════════════════════════════
# Phase 4: MaxLoss Cap analysis for L8c
# ═════════════════════════════════════════════════════════════

def phase_4():
    print("\n" + "=" * 90)
    print("  PHASE 4: MaxLoss Cap Sensitivity — L8c (spread=$0.50)")
    print("=" * 90)

    data = DataBundle.load_default()

    for sname, skw in [('L7', L7_MH8), ('L8c', L8C_MH8)]:
        print(f"\n  --- {sname} ---")
        r = run_variant(data, f"{sname}_base", verbose=False, **skw, spread_cost=0.50)
        trades = r['_trades']
        base_csh = corrected_sharpe(trades)

        caps = [20, 25, 30, 35, 40, 50, 60, 80, 100, 999]
        print(f"  {'Cap':>8} {'N':>6} {'Capped':>7} {'PnL':>10} {'CorrSh':>8} {'dSh':>6} {'WR%':>6}")
        print(f"  {'-'*8} {'-'*6} {'-'*7} {'-'*10} {'-'*8} {'-'*6} {'-'*6}")

        for cap in caps:
            if cap >= 999:
                capped_trades = trades
                cnt = 0
            else:
                capped_trades = apply_max_loss_cap(trades, cap)
                cnt = sum(1 for t, ct in zip(trades, capped_trades) if t.pnl != ct.pnl)
            total_pnl = sum(t.pnl for t in capped_trades)
            wins = sum(1 for t in capped_trades if t.pnl > 0)
            wr = wins / len(capped_trades) * 100 if capped_trades else 0
            csh = corrected_sharpe(capped_trades)
            dsh = csh - base_csh
            label = "NoCap" if cap >= 999 else f"${cap}"
            print(f"  {label:>8} {len(capped_trades):>6} {cnt:>7} ${total_pnl:>9.0f} "
                  f"{csh:>8.2f} {dsh:>+5.2f} {wr:>5.1f}%")

    # Loss distribution
    print("\n  --- Loss Distribution (L8c, spread=$0.50) ---")
    data = DataBundle.load_default()
    r = run_variant(data, "L8c_dist", verbose=False, **L8C_MH8, spread_cost=0.50)
    losses = sorted([t.pnl for t in r['_trades'] if t.pnl < 0])
    if losses:
        arr = np.array(losses)
        pcts = [10, 25, 50, 75, 90, 95, 99]
        print(f"  Total losses: {len(losses)}")
        print(f"  Mean: ${np.mean(arr):.2f}, Std: ${np.std(arr):.2f}")
        for p in pcts:
            val = np.percentile(arr, p)
            print(f"  P{p:>2}: ${val:.2f}")
        print(f"  Worst 10:")
        for pnl in losses[:10]:
            print(f"    ${pnl:.2f}")


# ═════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════

def main():
    out_path = OUT_DIR / "R39_L8_validation_output.txt"
    f_out = open(out_path, 'w', encoding='utf-8')
    tee = Tee(sys.stdout, f_out)
    sys.stdout = tee

    print("=" * 90)
    print("  R39: L8 STRATEGY VALIDATION")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  CPU cores: {mp.cpu_count()}, Workers: {MAX_WORKERS}")
    print("=" * 90)
    print("\n  L8 variants differ from L7 only in high-volatility regime trail:")
    print("  L7:  high trail_act=0.12, trail_dist=0.02")
    print("  L8a: high trail_act=0.12, trail_dist=0.01")
    print("  L8b: high trail_act=0.08, trail_dist=0.01")
    print("  L8c: high trail_act=0.06, trail_dist=0.005")
    print("  All use MH=8 (R25 structural improvement)")

    t0 = time.time()

    phase_1()
    print(f"\n  [Checkpoint] Phase 1 done, elapsed: {(time.time()-t0)/60:.1f} min")

    phase_2()
    print(f"\n  [Checkpoint] Phase 2 done, elapsed: {(time.time()-t0)/60:.1f} min")

    phase_3()
    print(f"\n  [Checkpoint] Phase 3 done, elapsed: {(time.time()-t0)/60:.1f} min")

    phase_4()
    print(f"\n  [Checkpoint] Phase 4 done, elapsed: {(time.time()-t0)/60:.1f} min")

    total = time.time() - t0
    print(f"\n\n{'=' * 90}")
    print(f"  R39 L8 VALIDATION COMPLETE")
    print(f"  Total runtime: {total/60:.1f} minutes ({total/3600:.1f} hours)")
    print(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Results: {out_path}")
    print(f"{'=' * 90}")

    sys.stdout = sys.__stdout__
    f_out.close()
    print(f"Done. Output: {out_path}")


if __name__ == "__main__":
    main()
