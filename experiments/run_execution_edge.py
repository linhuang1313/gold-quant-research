"""
Execution Edge: 实盘执行优化测试 (Parallel Version)
=====================================================
Phase A: 单笔亏损硬封顶 (MaxLoss cap)
Phase B: 高 spread 时段过滤 (Session filter)
Phase C: Regime detector (低波动期减仓/暂停)
Phase D: 降频 (提高信号质量阈值) — PARALLEL
Phase E: 综合最优组合 + K-Fold — PARALLEL
"""
import sys, os, time, multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from functools import partial

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.runner import DataBundle, run_variant, run_kfold, LIVE_PARITY_KWARGS
from backtest.stats import calc_stats, aggregate_daily_pnl
from backtest.engine import TradeRecord
import research_config as config

OUT_DIR = Path("results/execution_edge")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = min(mp.cpu_count(), 8)

L7_MH8 = {
    **LIVE_PARITY_KWARGS,
    'time_adaptive_trail': {'start': 2, 'decay': 0.75, 'floor': 0.003},
    'min_entry_gap_hours': 1.0,
    'keltner_max_hold_m15': 8,
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
        return 0.0, 0, 0
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
        return 0.0, len(trade_daily), len(arr)
    return float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252)), len(trade_daily), len(arr)


def sharpe_from_trades(trades):
    if not trades:
        return 0.0
    csh, _, _ = corrected_sharpe(trades)
    return csh


def apply_max_loss_cap(trades, cap_usd):
    capped = []
    cap_count = 0
    saved = 0.0
    for t in trades:
        if t.pnl < -cap_usd:
            diff = t.pnl - (-cap_usd)
            saved += abs(diff)
            cap_count += 1
            capped.append(TradeRecord(
                strategy=t.strategy, direction=t.direction,
                entry_price=t.entry_price, exit_price=t.exit_price,
                entry_time=t.entry_time, exit_time=t.exit_time,
                lots=t.lots, pnl=-cap_usd, exit_reason=t.exit_reason,
                bars_held=t.bars_held,
            ))
        else:
            capped.append(t)
    return capped, cap_count, saved


# ═══════════════════════════════════════════════════════════════
# Worker function for multiprocessing
# ═══════════════════════════════════════════════════════════════

def _run_one(args):
    """Worker: load data, run variant, return summary dict."""
    label, extra_kwargs, spread = args
    data = DataBundle.load_default()
    kw = {**L7_MH8, 'spread_cost': spread, **extra_kwargs}
    r = run_variant(data, label, verbose=False, **kw)
    csh, ntd, ntot = corrected_sharpe(r['_trades'])
    return {
        'label': label, 'spread': spread,
        'n': r['n'], 'total_pnl': r['total_pnl'], 'win_rate': r['win_rate'],
        'avg_win': r['avg_win'], 'avg_loss': r['avg_loss'], 'rr': r['rr'],
        'max_dd': r['max_dd'], 'corr_sharpe': csh,
        'orig_sharpe': r['sharpe'],
    }


def _run_one_kfold(args):
    """Worker: load data, run kfold, return fold results."""
    label, extra_kwargs, spread, cap = args
    data = DataBundle.load_default()
    kw = {**L7_MH8, 'spread_cost': spread, **extra_kwargs}
    folds = run_kfold(data, kw, n_folds=6)
    results = []
    for f in folds:
        trades_f = f.get('_trades', [])
        if cap < 999 and trades_f:
            trades_f, _, _ = apply_max_loss_cap(trades_f, cap)
        csh_f = sharpe_from_trades(trades_f)
        pnl_f = sum(t.pnl for t in trades_f) if trades_f else f['total_pnl']
        results.append({
            'fold': f['label'], 'n': f['n'], 'orig_sharpe': f['sharpe'],
            'corr_sharpe': csh_f, 'pnl': pnl_f, 'win_rate': f['win_rate'],
        })
    return {'label': label, 'folds': results}


# ═══════════════════════════════════════════════════════════════
# Phase A: Max Loss Cap (post-processing, no extra engine runs)
# ═══════════════════════════════════════════════════════════════

def phase_A(base_results):
    print("\n" + "=" * 90)
    print("  PHASE A: Single-Trade Max Loss Cap")
    print("=" * 90)

    caps = [20, 30, 40, 50, 60, 80, 100, 150, 999]

    for sp, (r, trades) in base_results.items():
        print(f"\n  === Spread = ${sp:.2f} ===")
        base_csh = sharpe_from_trades(trades)
        print(f"  {'MaxLoss':>8} {'N':>6} {'Capped':>7} {'Saved':>8} {'PnL':>10} {'CorrSh':>8} {'dSh':>6} {'WR%':>6}")
        print(f"  {'-'*8} {'-'*6} {'-'*7} {'-'*8} {'-'*10} {'-'*8} {'-'*6} {'-'*6}")

        for cap in caps:
            capped, cnt, saved = apply_max_loss_cap(trades, cap)
            total_pnl = sum(t.pnl for t in capped)
            wins = sum(1 for t in capped if t.pnl > 0)
            wr = wins / len(capped) * 100 if capped else 0
            csh = sharpe_from_trades(capped)
            dsh = csh - base_csh
            label = "NoCap" if cap >= 999 else f"${cap}"
            print(f"  {label:>8} {len(capped):>6} {cnt:>7} ${saved:>7.0f} ${total_pnl:>9.0f} "
                  f"{csh:>8.2f} {dsh:>+5.2f} {wr:>5.1f}%")


# ═══════════════════════════════════════════════════════════════
# Phase B: Session Filter (post-processing)
# ═══════════════════════════════════════════════════════════════

def phase_B(base_results):
    print("\n" + "=" * 90)
    print("  PHASE B: High-Spread Session Filter")
    print("=" * 90)

    for sp, (r, trades) in base_results.items():
        print(f"\n  === Spread = ${sp:.2f} ===")
        base_csh = sharpe_from_trades(trades)

        hourly = {}
        for t in trades:
            h = pd.Timestamp(t.entry_time).hour
            hourly.setdefault(h, []).append(t.pnl)

        print(f"\n  --- B1: Hourly PnL Breakdown ---")
        print(f"  {'Hour':>5} {'N':>6} {'PnL':>9} {'AvgPnL':>8} {'WR%':>6}")
        worst_hours = []
        for h in range(24):
            pnls = hourly.get(h, [])
            if not pnls: continue
            n = len(pnls); pnl = sum(pnls); wr = sum(1 for p in pnls if p > 0) / n * 100
            avg = pnl / n
            if avg < 0: worst_hours.append(h)
            mark = " ***" if avg < -1.0 else ""
            print(f"  {h:>5} {n:>6} ${pnl:>8.0f} ${avg:>7.2f} {wr:>5.1f}%{mark}")

        filter_sets = {
            'No filter':    [],
            'Skip 21-02':   [21, 22, 23, 0, 1, 2],
            'Skip 22-03':   [22, 23, 0, 1, 2, 3],
            'Skip 21-05':   [21, 22, 23, 0, 1, 2, 3, 4, 5],
            'Skip worst':   worst_hours,
            'London+NY':    [h for h in range(24) if h not in range(7, 21)],
        }

        print(f"\n  --- B2: Session Filter Impact ---")
        print(f"  {'Filter':<20} {'N':>6} {'PnL':>10} {'CorrSh':>8} {'dSh':>6} {'WR%':>6}")
        print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*8} {'-'*6} {'-'*6}")

        for fname, skip in filter_sets.items():
            if not skip:
                print(f"  {fname:<20} {len(trades):>6} ${r['total_pnl']:>9.0f} "
                      f"{base_csh:>8.2f} {'+0.00':>6} {r['win_rate']:>5.1f}%")
                continue
            kept = [t for t in trades if pd.Timestamp(t.entry_time).hour not in skip]
            if not kept: continue
            pnl = sum(t.pnl for t in kept)
            wins = sum(1 for t in kept if t.pnl > 0)
            wr = wins / len(kept) * 100
            csh = sharpe_from_trades(kept)
            print(f"  {fname:<20} {len(kept):>6} ${pnl:>9.0f} {csh:>8.2f} {csh - base_csh:>+5.2f} {wr:>5.1f}%")


# ═══════════════════════════════════════════════════════════════
# Phase C: Regime Detector (post-processing)
# ═══════════════════════════════════════════════════════════════

def phase_C(base_results, h1_df):
    print("\n" + "=" * 90)
    print("  PHASE C: Regime Detector (Low-Volatility Filter)")
    print("=" * 90)

    h1_atr = h1_df['ATR'] if 'ATR' in h1_df.columns else (h1_df['High'] - h1_df['Low']).rolling(14).mean()
    d1_atr = h1_atr.resample('1D').mean().dropna()
    d1_atr_pct = d1_atr.rolling(60).rank(pct=True) * 100
    if d1_atr_pct.index.tz is not None:
        d1_atr_pct.index = d1_atr_pct.index.tz_localize(None)

    h1_adx = h1_df['ADX'] if 'ADX' in h1_df.columns else None
    if h1_adx is not None and hasattr(h1_adx.index, 'tz') and h1_adx.index.tz is not None:
        h1_adx = h1_adx.copy()
        h1_adx.index = h1_adx.index.tz_localize(None)

    def get_atr_pct(entry_time):
        et = pd.Timestamp(entry_time)
        if et.tzinfo is not None: et = et.tz_localize(None)
        d = et.normalize()
        idx = d1_atr_pct.index.searchsorted(d, side='right') - 1
        if 0 <= idx < len(d1_atr_pct):
            val = d1_atr_pct.iloc[idx]
            return val if not pd.isna(val) else 50
        return 50

    for sp, (r, trades) in base_results.items():
        print(f"\n  === Spread = ${sp:.2f} ===")
        base_csh = sharpe_from_trades(trades)

        print(f"\n  --- C1: D1 ATR Percentile Regime ---")
        print(f"  {'ATR<X':>8} {'N_skip':>7} {'N_keep':>7} {'PnL_skip':>10} {'PnL_keep':>10} "
              f"{'CorrSh':>8} {'dSh':>6} {'WR%':>6}")
        print(f"  {'-'*8} {'-'*7} {'-'*7} {'-'*10} {'-'*10} {'-'*8} {'-'*6} {'-'*6}")

        for thr in [10, 15, 20, 25, 30, 40]:
            kept, skipped = [], []
            for t in trades:
                (skipped if get_atr_pct(t.entry_time) < thr else kept).append(t)
            pnl_s = sum(t.pnl for t in skipped)
            pnl_k = sum(t.pnl for t in kept)
            csh = sharpe_from_trades(kept) if kept else 0
            wr = sum(1 for t in kept if t.pnl > 0) / len(kept) * 100 if kept else 0
            print(f"  {'<'+str(thr)+'%':>8} {len(skipped):>7} {len(kept):>7} ${pnl_s:>9.0f} ${pnl_k:>9.0f} "
                  f"{csh:>8.2f} {csh-base_csh:>+5.2f} {wr:>5.1f}%")

        if h1_adx is not None:
            print(f"\n  --- C2: H1 ADX Regime ---")
            print(f"  {'ADX<X':>8} {'N_skip':>7} {'N_keep':>7} {'PnL_skip':>10} {'PnL_keep':>10} "
                  f"{'CorrSh':>8} {'dSh':>6}")
            print(f"  {'-'*8} {'-'*7} {'-'*7} {'-'*10} {'-'*10} {'-'*8} {'-'*6}")
            for adx_thr in [15, 18, 20, 22, 25, 30]:
                kept, skipped = [], []
                for t in trades:
                    et = pd.Timestamp(t.entry_time)
                    if et.tzinfo is not None: et = et.tz_localize(None)
                    idx = h1_adx.index.searchsorted(et, side='right') - 1
                    adx_val = float(h1_adx.iloc[idx]) if 0 <= idx < len(h1_adx) else 25
                    (skipped if adx_val < adx_thr else kept).append(t)
                pnl_s = sum(t.pnl for t in skipped)
                pnl_k = sum(t.pnl for t in kept)
                csh = sharpe_from_trades(kept) if kept else 0
                print(f"  {'<'+str(adx_thr):>8} {len(skipped):>7} {len(kept):>7} ${pnl_s:>9.0f} ${pnl_k:>9.0f} "
                      f"{csh:>8.2f} {csh-base_csh:>+5.2f}")


# ═══════════════════════════════════════════════════════════════
# Phase D: Frequency Reduction — PARALLEL
# ═══════════════════════════════════════════════════════════════

def phase_D():
    print("\n" + "=" * 90)
    print("  PHASE D: Frequency Reduction (Stricter Signal Quality) — PARALLEL")
    print(f"  Workers: {MAX_WORKERS}")
    print("=" * 90)

    variants = [
        ("D0_Baseline",        {}),
        ("D1_ADX20",           {'keltner_adx_threshold': 20}),
        ("D2_ADX22",           {'keltner_adx_threshold': 22}),
        ("D3_ADX25",           {'keltner_adx_threshold': 25}),
        ("D4_Gap2h",           {'min_entry_gap_hours': 2.0}),
        ("D5_Gap3h",           {'min_entry_gap_hours': 3.0}),
        ("D6_Gap4h",           {'min_entry_gap_hours': 4.0}),
        ("D7_ADX22+Gap2h",     {'keltner_adx_threshold': 22, 'min_entry_gap_hours': 2.0}),
        ("D8_ADX25+Gap3h",     {'keltner_adx_threshold': 25, 'min_entry_gap_hours': 3.0}),
        ("D9_KCBW3",           {'kc_bw_filter_bars': 3}),
        ("D10_KCBW5",          {'kc_bw_filter_bars': 5}),
        ("D11_ADX22+KCBW3",    {'keltner_adx_threshold': 22, 'kc_bw_filter_bars': 3}),
    ]

    tasks = []
    for sp in [0.30, 0.50]:
        for vname, extra in variants:
            tasks.append((f"{vname}_sp{sp}", extra, sp))

    print(f"  Dispatching {len(tasks)} tasks...")
    t0 = time.time()
    with mp.Pool(MAX_WORKERS) as pool:
        results = pool.map(_run_one, tasks)
    print(f"  Phase D done in {time.time()-t0:.0f}s")

    for sp in [0.30, 0.50]:
        print(f"\n  === Spread = ${sp:.2f} ===")
        print(f"  {'Variant':<25} {'N':>6} {'PnL':>10} {'CorrSh':>8} {'WR%':>6} {'AvgPnL':>8} {'RR':>5} {'MaxDD':>8}")
        print(f"  {'-'*25} {'-'*6} {'-'*10} {'-'*8} {'-'*6} {'-'*8} {'-'*5} {'-'*8}")
        for r in results:
            if r['spread'] != sp: continue
            short = r['label'].replace(f'_sp{sp}', '')
            avg = r['total_pnl'] / r['n'] if r['n'] > 0 else 0
            print(f"  {short:<25} {r['n']:>6} ${r['total_pnl']:>9.0f} {r['corr_sharpe']:>8.2f} "
                  f"{r['win_rate']:>5.1f}% ${avg:>7.2f} {r['rr']:>4.2f} ${r['max_dd']:>7.0f}")


# ═══════════════════════════════════════════════════════════════
# Phase E: Best Combo + K-Fold — PARALLEL
# ═══════════════════════════════════════════════════════════════

def phase_E():
    print("\n" + "=" * 90)
    print("  PHASE E: Best Combo + K-Fold — PARALLEL")
    print(f"  Workers: {MAX_WORKERS}")
    print("=" * 90)

    combos = [
        ("E0_Baseline",              {},                                                          999),
        ("E1_Cap60",                 {},                                                           60),
        ("E2_Cap80",                 {},                                                           80),
        ("E3_ADX22+Gap2h",           {'keltner_adx_threshold': 22, 'min_entry_gap_hours': 2.0},   999),
        ("E4_ADX22+Gap2h+Cap60",     {'keltner_adx_threshold': 22, 'min_entry_gap_hours': 2.0},    60),
        ("E5_ADX22+Gap2h+Cap80",     {'keltner_adx_threshold': 22, 'min_entry_gap_hours': 2.0},    80),
        ("E6_ADX25+Gap3h+Cap60",     {'keltner_adx_threshold': 25, 'min_entry_gap_hours': 3.0},    60),
    ]

    # Full-sample runs (parallel)
    full_tasks = []
    for sp in [0.30, 0.50]:
        for cname, extra, cap in combos:
            full_tasks.append((f"{cname}_sp{sp}", extra, sp))

    print(f"  Dispatching {len(full_tasks)} full-sample tasks...")
    t0 = time.time()
    with mp.Pool(MAX_WORKERS) as pool:
        full_results = pool.map(_run_one, full_tasks)
    print(f"  Full-sample done in {time.time()-t0:.0f}s")

    combo_cap = {c[0]: c[2] for c in combos}

    for sp in [0.30, 0.50]:
        print(f"\n  === Spread = ${sp:.2f} ===")
        print(f"  {'Combo':<28} {'N':>6} {'PnL':>10} {'CorrSh':>8} {'WR%':>6} {'AvgPnL':>8}")
        print(f"  {'-'*28} {'-'*6} {'-'*10} {'-'*8} {'-'*6} {'-'*8}")
        for r in full_results:
            if r['spread'] != sp: continue
            short = r['label'].replace(f'_sp{sp}', '')
            cap = combo_cap.get(short, 999)
            avg = r['total_pnl'] / r['n'] if r['n'] > 0 else 0
            note = f" (cap=${cap})" if cap < 999 else ""
            print(f"  {short:<28} {r['n']:>6} ${r['total_pnl']:>9.0f} {r['corr_sharpe']:>8.2f} "
                  f"{r['win_rate']:>5.1f}% ${avg:>7.2f}{note}")

    # K-Fold validation (parallel, spread=0.50 only)
    kfold_combos = [
        ("KF_Baseline",           {},                                                          999),
        ("KF_Cap80",              {},                                                           80),
        ("KF_ADX22+Gap2h+Cap80",  {'keltner_adx_threshold': 22, 'min_entry_gap_hours': 2.0},    80),
    ]
    kfold_tasks = [(c[0], c[1], 0.50, c[2]) for c in kfold_combos]

    print(f"\n  Dispatching {len(kfold_tasks)} K-Fold tasks (6 folds each)...")
    t0 = time.time()
    with mp.Pool(min(MAX_WORKERS, len(kfold_tasks))) as pool:
        kfold_results = pool.map(_run_one_kfold, kfold_tasks)
    print(f"  K-Fold done in {time.time()-t0:.0f}s")

    for kr in kfold_results:
        print(f"\n  [{kr['label']}] (spread=$0.50)")
        print(f"  {'Fold':<8} {'N':>6} {'OrigSh':>8} {'CorrSh':>8} {'PnL':>10} {'WR%':>6}")
        sharpes = []
        for f in kr['folds']:
            sharpes.append(f['corr_sharpe'])
            print(f"  {f['fold']:<8} {f['n']:>6} {f['orig_sharpe']:>8.2f} {f['corr_sharpe']:>8.2f} "
                  f"${f['pnl']:>9.0f} {f['win_rate']:>5.1f}%")
        all_pos = all(s > 0 for s in sharpes)
        print(f"  Mean={np.mean(sharpes):.2f}, Std={np.std(sharpes):.2f}, "
              f"AllPositive={all_pos}, PASS={'YES' if all_pos else 'NO'}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    out_path = OUT_DIR / "execution_edge_output.txt"
    f_out = open(out_path, 'w', encoding='utf-8')
    tee = Tee(sys.stdout, f_out)
    sys.stdout = tee

    print("=" * 90)
    print("  EXECUTION EDGE: Practical Optimization (Parallel)")
    print(f"  Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  CPU cores: {mp.cpu_count()}, Workers: {MAX_WORKERS}")
    print("=" * 90)

    t0 = time.time()

    # Load data once for Phase A/B/C (post-processing phases)
    data = DataBundle.load_default()
    print(f"\n  Data: M15={len(data.m15_df)}, H1={len(data.h1_df)}")

    # Run baseline for each spread (only 2 engine runs needed for A/B/C)
    base_results = {}
    for sp in [0.30, 0.50]:
        print(f"\n  Running baseline at spread=${sp:.2f}...")
        r = run_variant(data, f"Baseline_sp{sp}", verbose=True, **L7_MH8, spread_cost=sp)
        base_results[sp] = (r, r['_trades'])

    # Phase A/B/C: all post-processing, very fast
    phase_A(base_results)
    phase_B(base_results)
    phase_C(base_results, data.h1_df)

    # Phase D: parallel engine runs
    phase_D()

    # Phase E: parallel engine runs + K-Fold
    phase_E()

    elapsed = time.time() - t0
    print(f"\n\n{'=' * 90}")
    print(f"  EXECUTION EDGE COMPLETE")
    print(f"  Total runtime: {elapsed/60:.1f} minutes")
    print(f"  Results: {out_path}")
    print(f"{'=' * 90}")

    sys.stdout = sys.__stdout__
    f_out.close()
    print(f"Done. Output: {out_path}")


if __name__ == "__main__":
    main()
