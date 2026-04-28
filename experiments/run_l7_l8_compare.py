"""
R41: L7 vs L8_BASELINE vs L8_hybrid_full — Unified Comparison
================================================================
统一口径对比三个策略:
- L7: 当前实盘 (ADX18, trail 0.28/0.06, TATrail ON, MH=8)
- L8_BASELINE: ADX14, trail 0.14/0.025, TATrail OFF, MH=20
- L8_hybrid_full: ADX14, trail 0.14/0.025, TATrail ON, MH=8
- 额外: L8c_R39 (我的版本: ADX18, high trail 0.06/0.005, TATrail ON, MH=8)

Phase 1: Full sample + K-Fold (corrected Sharpe, spread=$0.50)
Phase 2: Spread 衰减曲线
Phase 3: Cap 分析
Phase 4: KCBW5 组合测试
Phase 5: 最优组合 K-Fold 验证
"""
import sys, os, time, multiprocessing as mp
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.runner import DataBundle, run_variant, run_kfold, LIVE_PARITY_KWARGS
from backtest.engine import TradeRecord

OUT_DIR = Path("results/round41_compare")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = min(mp.cpu_count(), 8)

# L7: current production
L7 = {
    **LIVE_PARITY_KWARGS,
    'time_adaptive_trail': {'start': 2, 'decay': 0.75, 'floor': 0.003},
    'min_entry_gap_hours': 1.0,
    'keltner_max_hold_m15': 8,
}

# L8_BASELINE: ADX14, tighter normal trail, no TATrail, MH=20
L8_BASE = {
    **LIVE_PARITY_KWARGS,
    'keltner_adx_threshold': 14,
    'regime_config': {
        'low':    {'trail_act': 0.22, 'trail_dist': 0.04},
        'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
        'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
    },
    'min_entry_gap_hours': 1.0,
    'keltner_max_hold_m15': 20,
}

# L8_hybrid_full: L8_BASELINE + TATrail + MH=8
L8_HYBRID = {
    **LIVE_PARITY_KWARGS,
    'keltner_adx_threshold': 14,
    'regime_config': {
        'low':    {'trail_act': 0.22, 'trail_dist': 0.04},
        'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
        'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
    },
    'time_adaptive_trail': {'start': 2, 'decay': 0.75, 'floor': 0.003},
    'min_entry_gap_hours': 1.0,
    'keltner_max_hold_m15': 8,
}

# L8c_R39: ADX18, only high trail tighter, TATrail ON, MH=8
L8C_R39 = {
    **LIVE_PARITY_KWARGS,
    'regime_config': {
        'low':    {'trail_act': 0.40, 'trail_dist': 0.10},
        'normal': {'trail_act': 0.28, 'trail_dist': 0.06},
        'high':   {'trail_act': 0.06, 'trail_dist': 0.005},
    },
    'time_adaptive_trail': {'start': 2, 'decay': 0.75, 'floor': 0.003},
    'min_entry_gap_hours': 1.0,
    'keltner_max_hold_m15': 8,
}

STRATEGIES = {
    'L7':             L7,
    'L8_BASE':        L8_BASE,
    'L8_HYBRID':      L8_HYBRID,
    'L8c_R39':        L8C_R39,
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


def corrected_sharpe(trades):
    if not trades:
        return 0.0
    trade_daily = {}
    for t in trades:
        d = pd.Timestamp(t.exit_time).date()
        trade_daily[d] = trade_daily.get(d, 0) + t.pnl
    if not trade_daily:
        return 0.0
    start_date = min(trade_daily.keys())
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
        '_trades': r['_trades'],
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
        results.append({
            'fold': f['label'], 'n': f['n'], 'orig_sharpe': f['sharpe'],
            'corr_sharpe': csh, 'pnl': pnl, 'win_rate': f['win_rate'],
        })
    return {'label': label, 'folds': results}


def print_kfold(kr):
    print(f"\n  [{kr['label']}]")
    print(f"  {'Fold':<8} {'N':>6} {'OrigSh':>8} {'CorrSh':>8} {'PnL':>10} {'WR%':>6}")
    sharpes = []
    orig_sharpes = []
    for f in kr['folds']:
        sharpes.append(f['corr_sharpe'])
        orig_sharpes.append(f['orig_sharpe'])
        print(f"  {f['fold']:<8} {f['n']:>6} {f['orig_sharpe']:>8.2f} {f['corr_sharpe']:>8.2f} "
              f"${f['pnl']:>9.0f} {f['win_rate']:>5.1f}%")
    pos = sum(1 for s in sharpes if s > 0)
    print(f"  CorrSh: Mean={np.mean(sharpes):.2f}, Std={np.std(sharpes):.2f}, "
          f"Min={min(sharpes):.2f}, Positive={pos}/6, PASS={'YES' if pos==6 else 'NO'}")
    print(f"  OrigSh: Mean={np.mean(orig_sharpes):.2f}, Min={min(orig_sharpes):.2f}")
    return sharpes, orig_sharpes


# ═════════════════════════════════════════════════════════════
# Phase 1: Full Sample + K-Fold
# ═════════════════════════════════════════════════════════════

def phase_1():
    print("\n" + "=" * 90)
    print("  PHASE 1: Full Sample + K-Fold (4 strategies)")
    print(f"  Workers: {MAX_WORKERS}")
    print("=" * 90)

    # Full sample
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
        print(f"  {'Strategy':<15} {'N':>6} {'PnL':>10} {'OrigSh':>8} {'CorrSh':>8} "
              f"{'WR%':>6} {'AvgW':>8} {'AvgL':>8} {'RR':>5} {'MaxDD':>8}")
        print(f"  {'-'*15} {'-'*6} {'-'*10} {'-'*8} {'-'*8} "
              f"{'-'*6} {'-'*8} {'-'*8} {'-'*5} {'-'*8}")
        for r in results:
            if r['spread'] != sp: continue
            short = r['label'].replace(f'_sp{sp}', '')
            print(f"  {short:<15} {r['n']:>6} ${r['total_pnl']:>9.0f} {r['orig_sharpe']:>8.2f} "
                  f"{r['corr_sharpe']:>8.2f} {r['win_rate']:>5.1f}% "
                  f"${r['avg_win']:>7.2f} ${r['avg_loss']:>7.2f} {r['rr']:>4.2f} "
                  f"${r['max_dd']:>7.0f}")

    # K-Fold (spread=0.50, no cap)
    print("\n  --- K-Fold 6-Fold (spread=$0.50, no cap) ---")
    kf_tasks = [(f"KF_{sname}", skw, 0.50, 999) for sname, skw in STRATEGIES.items()]
    t0 = time.time()
    with mp.Pool(MAX_WORKERS) as pool:
        kf_results = pool.map(_run_kfold_one, kf_tasks)
    print(f"  K-Fold done in {time.time()-t0:.0f}s")

    all_kf = {}
    for kr in kf_results:
        sharpes, orig_sharpes = print_kfold(kr)
        all_kf[kr['label']] = (sharpes, orig_sharpes)

    # Summary
    print(f"\n  --- K-Fold Summary (no cap) ---")
    print(f"  {'Config':<20} {'CorrMean':>8} {'CorrMin':>8} {'OrigMean':>8} {'OrigMin':>8} {'P/6':>4}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*4}")
    for label, (cs, os_) in all_kf.items():
        pos = sum(1 for s in cs if s > 0)
        print(f"  {label:<20} {np.mean(cs):>8.2f} {min(cs):>8.2f} "
              f"{np.mean(os_):>8.2f} {min(os_):>8.2f} {pos:>3}/6")


# ═════════════════════════════════════════════════════════════
# Phase 2: Spread Decay
# ═════════════════════════════════════════════════════════════

def phase_2():
    print("\n" + "=" * 90)
    print("  PHASE 2: Spread Decay Curve")
    print("=" * 90)

    spreads = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00, 1.50, 2.00]
    tasks = []
    for sp in spreads:
        for sname, skw in STRATEGIES.items():
            tasks.append((f"{sname}_sp{sp}", skw, sp))

    t0 = time.time()
    with mp.Pool(MAX_WORKERS) as pool:
        results = pool.map(_run_one, tasks)
    print(f"  Done in {time.time()-t0:.0f}s")

    for sname in STRATEGIES:
        print(f"\n  --- {sname} ---")
        print(f"  {'Spread':>8} {'N':>6} {'PnL':>10} {'CorrSh':>8} {'WR%':>6} {'MaxDD':>8}")
        print(f"  {'-'*8} {'-'*6} {'-'*10} {'-'*8} {'-'*6} {'-'*8}")
        for r in sorted(results, key=lambda x: x['spread']):
            if not r['label'].startswith(sname + "_sp"): continue
            sp_str = r['label'].split("_sp")[1]
            print(f"  ${float(sp_str):>6.2f} {r['n']:>6} ${r['total_pnl']:>9.0f} "
                  f"{r['corr_sharpe']:>8.2f} {r['win_rate']:>5.1f}% ${r['max_dd']:>7.0f}")

    # Side-by-side at key spreads
    print(f"\n  --- Side-by-Side (CorrSh) ---")
    print(f"  {'Spread':>8} {'L7':>8} {'L8_BASE':>8} {'L8_HYBRID':>10} {'L8c_R39':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    for sp in spreads:
        vals = {}
        for r in results:
            if r['spread'] == sp:
                sname = r['label'].replace(f"_sp{sp}", "")
                vals[sname] = r['corr_sharpe']
        print(f"  ${sp:>6.2f} {vals.get('L7',0):>8.2f} {vals.get('L8_BASE',0):>8.2f} "
              f"{vals.get('L8_HYBRID',0):>10.2f} {vals.get('L8c_R39',0):>8.2f}")


# ═════════════════════════════════════════════════════════════
# Phase 3: Cap Analysis
# ═════════════════════════════════════════════════════════════

def phase_3():
    print("\n" + "=" * 90)
    print("  PHASE 3: MaxLoss Cap Analysis (spread=$0.50)")
    print("=" * 90)

    data = DataBundle.load_default()
    caps = [20, 25, 30, 35, 40, 50, 60, 999]

    for sname, skw in STRATEGIES.items():
        r = run_variant(data, f"{sname}_cap", verbose=False, **skw, spread_cost=0.50)
        trades = r['_trades']
        base_csh = corrected_sharpe(trades)

        print(f"\n  --- {sname} ---")
        print(f"  {'Cap':>8} {'N':>6} {'Capped':>7} {'PnL':>10} {'CorrSh':>8} {'dSh':>6}")
        print(f"  {'-'*8} {'-'*6} {'-'*7} {'-'*10} {'-'*8} {'-'*6}")
        for cap in caps:
            if cap >= 999:
                capped_trades = trades
                cnt = 0
            else:
                capped_trades = apply_max_loss_cap(trades, cap)
                cnt = sum(1 for t, ct in zip(trades, capped_trades) if t.pnl != ct.pnl)
            pnl = sum(t.pnl for t in capped_trades)
            csh = corrected_sharpe(capped_trades)
            label = "NoCap" if cap >= 999 else f"${cap}"
            print(f"  {label:>8} {len(capped_trades):>6} {cnt:>7} ${pnl:>9.0f} "
                  f"{csh:>8.2f} {csh-base_csh:>+5.2f}")


# ═════════════════════════════════════════════════════════════
# Phase 4: + KCBW5 Combo
# ═════════════════════════════════════════════════════════════

def phase_4():
    print("\n" + "=" * 90)
    print("  PHASE 4: + KCBW5 Full Sample (spread=$0.50)")
    print("=" * 90)

    tasks = []
    for sname, skw in STRATEGIES.items():
        combo = {**skw, 'kc_bw_filter_bars': 5}
        tasks.append((f"{sname}+KCBW5", combo, 0.50))
        tasks.append((f"{sname}", skw, 0.50))

    t0 = time.time()
    with mp.Pool(MAX_WORKERS) as pool:
        results = pool.map(_run_one, tasks)
    print(f"  Done in {time.time()-t0:.0f}s")

    print(f"\n  {'Strategy':<20} {'N':>6} {'PnL':>10} {'CorrSh':>8} {'WR%':>6} {'MaxDD':>8}")
    print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*8} {'-'*6} {'-'*8}")
    for r in sorted(results, key=lambda x: x['label']):
        print(f"  {r['label']:<20} {r['n']:>6} ${r['total_pnl']:>9.0f} "
              f"{r['corr_sharpe']:>8.2f} {r['win_rate']:>5.1f}% ${r['max_dd']:>7.0f}")


# ═════════════════════════════════════════════════════════════
# Phase 5: Best Combos K-Fold
# ═════════════════════════════════════════════════════════════

def phase_5():
    print("\n" + "=" * 90)
    print("  PHASE 5: Best Combos K-Fold (spread=$0.50)")
    print("=" * 90)

    combos = [
        ("KF_L7",                   L7,        999),
        ("KF_L7+KCBW5+Cap30",       {**L7, 'kc_bw_filter_bars': 5}, 30),
        ("KF_L8_BASE",              L8_BASE,   999),
        ("KF_L8_BASE+KCBW5+Cap30",  {**L8_BASE, 'kc_bw_filter_bars': 5}, 30),
        ("KF_L8_HYBRID",            L8_HYBRID, 999),
        ("KF_L8_HYBRID+KCBW5+Cap30",{**L8_HYBRID, 'kc_bw_filter_bars': 5}, 30),
        ("KF_L8c_R39",              L8C_R39,   999),
        ("KF_L8c_R39+KCBW5+Cap30",  {**L8C_R39, 'kc_bw_filter_bars': 5}, 30),
    ]

    tasks = [(c[0], c[1], 0.50, c[2]) for c in combos]

    t0 = time.time()
    with mp.Pool(MAX_WORKERS) as pool:
        results = pool.map(_run_kfold_one, tasks)
    print(f"  Done in {time.time()-t0:.0f}s")

    for kr in results:
        print_kfold(kr)

    # Final summary
    print(f"\n  {'='*90}")
    print(f"  FINAL K-FOLD COMPARISON")
    print(f"  {'='*90}")
    print(f"  {'Config':<30} {'CorrMean':>8} {'CorrStd':>8} {'CorrMin':>8} "
          f"{'OrigMean':>8} {'P/6':>4} {'PASS':>5}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*4} {'-'*5}")
    for kr in results:
        cs = [f['corr_sharpe'] for f in kr['folds']]
        os_ = [f['orig_sharpe'] for f in kr['folds']]
        pos = sum(1 for s in cs if s > 0)
        print(f"  {kr['label']:<30} {np.mean(cs):>8.2f} {np.std(cs):>8.2f} "
              f"{min(cs):>8.2f} {np.mean(os_):>8.2f} {pos:>3}/6 "
              f"{'YES' if pos==6 else 'NO':>5}")


# ═════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════

def main():
    out_path = OUT_DIR / "R41_compare_output.txt"
    f_out = open(out_path, 'w', encoding='utf-8')
    tee = Tee(sys.stdout, f_out)
    sys.stdout = tee

    print("=" * 90)
    print("  R41: L7 vs L8_BASELINE vs L8_hybrid_full vs L8c_R39")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  CPU cores: {mp.cpu_count()}, Workers: {MAX_WORKERS}")
    print("=" * 90)
    print("\n  Strategy definitions:")
    print("  L7:         ADX=18, trail 0.28/0.06, TATrail ON, MH=8")
    print("  L8_BASE:    ADX=14, trail 0.14/0.025, TATrail OFF, MH=20")
    print("  L8_HYBRID:  ADX=14, trail 0.14/0.025, TATrail ON, MH=8")
    print("  L8c_R39:    ADX=18, high trail 0.06/0.005, TATrail ON, MH=8")

    t0 = time.time()

    phase_1()
    print(f"\n  [Checkpoint] Phase 1 done, elapsed: {(time.time()-t0)/60:.1f} min")

    phase_2()
    print(f"\n  [Checkpoint] Phase 2 done, elapsed: {(time.time()-t0)/60:.1f} min")

    phase_3()
    print(f"\n  [Checkpoint] Phase 3 done, elapsed: {(time.time()-t0)/60:.1f} min")

    phase_4()
    print(f"\n  [Checkpoint] Phase 4 done, elapsed: {(time.time()-t0)/60:.1f} min")

    phase_5()
    print(f"\n  [Checkpoint] Phase 5 done, elapsed: {(time.time()-t0)/60:.1f} min")

    total = time.time() - t0
    print(f"\n\n{'=' * 90}")
    print(f"  R41 COMPARISON COMPLETE")
    print(f"  Total runtime: {total/60:.1f} minutes ({total/3600:.1f} hours)")
    print(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 90}")

    sys.stdout = sys.__stdout__
    f_out.close()
    print(f"Done. Output: {out_path}")


if __name__ == "__main__":
    main()
