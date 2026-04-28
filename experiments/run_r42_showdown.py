"""
R42: L8_BASE vs L8_HYBRID — 精简版对决 (实盘选型)
=================================================
Phase 1 已完成 (见之前结果)，直接从 Phase 2 开始。
所有子进程共享主进程加载的数据，避免重复加载。
"""
import sys, os, time, multiprocessing as mp
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.runner import (DataBundle, run_variant, LIVE_PARITY_KWARGS,
                              _worker_run_variant)
from backtest.engine import TradeRecord

OUT_DIR = Path("results/round42_showdown")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = min(mp.cpu_count(), 8)

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

STRATEGIES = {'L8_BASE': L8_BASE, 'L8_HYBRID': L8_HYBRID}
SPREAD = 0.50


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


def filter_kcbw5(trades, h1_df, kc_ema=25, kc_mult=1.2, lookback=5):
    if h1_df is None or len(h1_df) == 0:
        return trades
    atr = h1_df['ATR'] if 'ATR' in h1_df.columns else (h1_df['High'] - h1_df['Low']).rolling(14).mean()
    ema = h1_df['Close'].ewm(span=kc_ema).mean()
    bw = (kc_mult * atr * 2.0) / ema.replace(0, np.nan)
    bw_min = bw.rolling(lookback).min()
    expanding = bw > bw_min.shift(1)

    filtered = []
    for t in trades:
        et = pd.Timestamp(t.entry_time)
        if et.tzinfo is None:
            et = et.tz_localize('UTC')
        h1_ts = h1_df.index[h1_df.index <= et]
        if len(h1_ts) == 0:
            continue
        nearest = h1_ts[-1]
        if nearest in expanding.index and expanding.loc[nearest]:
            filtered.append(t)
    return filtered


def main():
    t_start = time.time()
    log_path = OUT_DIR / "R42_showdown_output.txt"
    log_f = open(log_path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, log_f)

    print("=" * 70)
    print("  R42: L8_BASE vs L8_HYBRID -- SHOWDOWN (R41 delta)")
    print("  Skipping: K-Fold, Spread Decay, Cap (already in R41)")
    print("  New: Walk-Forward, Yearly, Param Cliff, PnL Dist, Recent")
    print("=" * 70)
    print(f"\n  L8_BASE:   ADX=14, trail 0.14/0.025, TATrail OFF, MH=20")
    print(f"  L8_HYBRID: ADX=14, trail 0.14/0.025, TATrail ON,  MH=8")
    print(f"  Workers: {MAX_WORKERS}")
    print(f"  Start: {datetime.now()}")

    # Load data ONCE
    print(f"\n  Loading data (once)...")
    data = DataBundle.load_default()
    h1_df = data.h1_df
    m15_df = data.m15_df
    print(f"  Data loaded: M15={len(m15_df)}, H1={len(h1_df)}")

    # ================================================================
    # PHASE 1: Full Sample (collect trades)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 1: Full Sample (spread=${SPREAD})")
    print(f"{'='*70}")

    all_trades = {}
    for name, kwargs in STRATEGIES.items():
        kw = {**kwargs, 'spread_cost': SPREAD}
        r = run_variant(data, name, verbose=True, **kw)
        trades = r['_trades']
        csh = corrected_sharpe(trades)
        all_trades[name] = trades

        t_kcbw = filter_kcbw5(trades, h1_df)
        t_kcbw_cap = apply_max_loss_cap(t_kcbw, 30)
        csh_combo = corrected_sharpe(t_kcbw_cap)

        print(f"    {name}:            CorrSh={csh:.2f}, N={r['n']}, PnL=${r['total_pnl']:.0f}, "
              f"WR={r['win_rate']:.1f}%, MaxDD=${r['max_dd']:.0f}")
        print(f"    {name}+KCBW5+Cap30: CorrSh={csh_combo:.2f}, N={len(t_kcbw_cap)}, "
              f"PnL=${sum(t.pnl for t in t_kcbw_cap):.0f}")

    print(f"\n  [Checkpoint] Phase 1 done, elapsed: {(time.time()-t_start)/60:.1f} min")

    # ================================================================
    # PHASE 2: Walk-Forward (shared data, parallel)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Walk-Forward (18-month rolling windows)")
    print(f"{'='*70}")

    wf_starts = pd.date_range("2016-01-01", "2025-01-01", freq="6MS")
    wf_tasks = []
    for name, kwargs in STRATEGIES.items():
        for ws in wf_starts:
            we = ws + pd.DateOffset(months=18)
            start_str = ws.strftime("%Y-%m-%d")
            end_str = we.strftime("%Y-%m-%d")
            sliced = data.slice(start_str, end_str)
            if len(sliced.m15_df) < 500:
                continue
            kw = {**kwargs, 'spread_cost': SPREAD}
            wf_tasks.append((sliced.m15_df, sliced.h1_df,
                            f"{name}_{start_str}", dict(kw)))

    print(f"  Running {len(wf_tasks)} walk-forward windows ({MAX_WORKERS} workers)...")
    wf_results = [None] * len(wf_tasks)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {pool.submit(_worker_run_variant, t): i for i, t in enumerate(wf_tasks)}
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                stats = future.result()
                wf_results[idx] = stats
                print(f"    [{idx+1}/{len(wf_tasks)}] {stats['label']}: "
                      f"Sh={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}, "
                      f"{stats['elapsed_s']}s", flush=True)
            except Exception as e:
                print(f"    [{idx+1}/{len(wf_tasks)}] ERROR: {e}", flush=True)

    wf_by_strat = defaultdict(list)
    for i, r in enumerate(wf_results):
        if r is None:
            continue
        task_label = wf_tasks[i][2]
        csh = corrected_sharpe(r.get('_trades', []) if '_trades' in r else [])
        # re-derive start/end from label
        parts = task_label.rsplit('_', 1)
        strat_name = parts[0]
        start_str = parts[1] if len(parts) > 1 else ""
        for sn in STRATEGIES:
            if task_label.startswith(sn):
                wf_by_strat[sn].append({
                    'label': task_label, 'start': start_str,
                    'n': r['n'], 'total_pnl': r['total_pnl'],
                    'corr_sharpe': corrected_sharpe([]) if '_trades' not in r else 0,
                    'orig_sharpe': r['sharpe'],
                    'win_rate': r['win_rate'], 'max_dd': r.get('max_dd', 0),
                })
                break

    # Since _worker_run_variant doesn't return _trades, use orig_sharpe
    for name in STRATEGIES:
        results = wf_by_strat.get(name, [])
        if not results:
            print(f"\n  {name}: No walk-forward results")
            continue
        print(f"\n  --- {name} Walk-Forward ---")
        print(f"  {'Window':<25} {'N':>5} {'PnL':>8} {'Sharpe':>8} {'WR%':>6} {'MaxDD':>7}")
        sharpes = []
        for r in sorted(results, key=lambda x: x['start']):
            sh = r['orig_sharpe']
            print(f"  {r['label']:<25} {r['n']:>5} ${r['total_pnl']:>7.0f} "
                  f"{sh:>8.2f} {r['win_rate']:>5.1f}% ${r['max_dd']:>6.0f}")
            sharpes.append(sh)
        pos = sum(1 for s in sharpes if s > 0)
        mean_sh = np.mean(sharpes) if sharpes else 0
        min_sh = min(sharpes) if sharpes else 0
        max_sh = max(sharpes) if sharpes else 0
        print(f"  WF Summary: {pos}/{len(sharpes)} positive, Mean={mean_sh:.2f}, "
              f"Min={min_sh:.2f}, Max={max_sh:.2f}")

    print(f"\n  [Checkpoint] Phase 2 done, elapsed: {(time.time()-t_start)/60:.1f} min")

    # ================================================================
    # PHASE 3: Yearly Breakdown (from Phase 1 trades)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 3: Yearly Breakdown (2015-2026)")
    print(f"{'='*70}")

    years = list(range(2015, 2027))
    for name in STRATEGIES:
        trades = all_trades[name]
        t_kcbw_cap = apply_max_loss_cap(filter_kcbw5(trades, h1_df), 30)

        for suffix, t_list in [(name, trades), (f"{name}+KCBW5+Cap30", t_kcbw_cap)]:
            print(f"\n  --- {suffix} ---")
            print(f"  {'Year':<6} {'N':>5} {'PnL':>9} {'CorrSh':>8} {'WR%':>6} {'AvgW':>7} {'AvgL':>7} {'MaxDD':>8}")
            for yr in years:
                yr_trades = [t for t in t_list if pd.Timestamp(t.exit_time).year == yr]
                if not yr_trades:
                    continue
                pnl = sum(t.pnl for t in yr_trades)
                csh = corrected_sharpe(yr_trades)
                wins = [t.pnl for t in yr_trades if t.pnl > 0]
                losses = [t.pnl for t in yr_trades if t.pnl <= 0]
                wr = len(wins) / len(yr_trades) * 100
                avg_w = np.mean(wins) if wins else 0
                avg_l = np.mean(losses) if losses else 0
                cum = np.cumsum([t.pnl for t in yr_trades])
                peak = np.maximum.accumulate(cum)
                dd = peak - cum
                maxdd = dd.max() if len(dd) > 0 else 0
                print(f"  {yr:<6} {len(yr_trades):>5} ${pnl:>8.0f} {csh:>8.2f} {wr:>5.1f}% "
                      f"${avg_w:>6.2f} ${avg_l:>6.2f} ${maxdd:>7.0f}")

    print(f"\n  [Checkpoint] Phase 3 done, elapsed: {(time.time()-t_start)/60:.1f} min")

    # ================================================================
    # PHASE 4: Parameter Cliff (shared data, parallel)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 4: Parameter Cliff (perturbation test)")
    print(f"{'='*70}")

    cliff_tasks = []
    for name, kwargs in STRATEGIES.items():
        # baseline
        kw_base = {**kwargs, 'spread_cost': SPREAD}
        cliff_tasks.append((m15_df, h1_df, f"{name}_BASE", dict(kw_base)))

        for adx_val in [12, 14, 16, 18, 20]:
            kw_mod = {**kwargs, 'keltner_adx_threshold': adx_val, 'spread_cost': SPREAD}
            cliff_tasks.append((m15_df, h1_df, f"{name}_ADX{adx_val}", dict(kw_mod)))

        rc = kwargs.get('regime_config', {}).get('normal', {})
        base_act = rc.get('trail_act', 0.14)
        base_dist = rc.get('trail_dist', 0.025)
        for pct in [-0.30, -0.20, 0.20, 0.30]:
            new_act = round(base_act * (1 + pct), 4)
            new_dist = round(base_dist * (1 + pct), 4)
            rc_new = {
                'low': kwargs['regime_config']['low'],
                'normal': {'trail_act': new_act, 'trail_dist': new_dist},
                'high': kwargs['regime_config']['high'],
            }
            kw_mod = {**kwargs, 'regime_config': rc_new, 'spread_cost': SPREAD}
            cliff_tasks.append((m15_df, h1_df, f"{name}_Trail{pct:+.0%}", dict(kw_mod)))

        base_mh = kwargs.get('keltner_max_hold_m15', 20)
        for mh_val in [4, 6, 8, 12, 16, 20, 25, 30]:
            if mh_val == base_mh:
                continue
            kw_mod = {**kwargs, 'keltner_max_hold_m15': mh_val, 'spread_cost': SPREAD}
            cliff_tasks.append((m15_df, h1_df, f"{name}_MH{mh_val}", dict(kw_mod)))

    print(f"  Running {len(cliff_tasks)} cliff variants ({MAX_WORKERS} workers)...")
    cliff_results = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {pool.submit(_worker_run_variant, t): t[2] for t in cliff_tasks}
        done = 0
        for future in as_completed(future_map):
            label = future_map[future]
            done += 1
            try:
                stats = future.result()
                cliff_results[label] = stats
                if done % 5 == 0 or done == len(cliff_tasks):
                    print(f"    [{done}/{len(cliff_tasks)}] {label}: "
                          f"Sh={stats['sharpe']:.2f}, {stats['elapsed_s']}s", flush=True)
            except Exception as e:
                print(f"    [{done}/{len(cliff_tasks)}] {label} ERROR: {e}", flush=True)

    for name in STRATEGIES:
        base_label = f"{name}_BASE"
        base_sh = cliff_results.get(base_label, {}).get('sharpe', 0)

        print(f"\n  --- {name} Parameter Cliff ---")
        print(f"  {'Config':<30} {'N':>6} {'PnL':>10} {'Sharpe':>8} {'dSh':>6} {'WR%':>6}")
        relevant = {k: v for k, v in cliff_results.items() if k.startswith(name)}
        for label in sorted(relevant.keys()):
            r = relevant[label]
            dsh = r['sharpe'] - base_sh
            flag = " <-- CLIFF" if abs(dsh) > 1.5 else (" *" if abs(dsh) > 0.8 else "")
            print(f"  {label:<30} {r['n']:>6} ${r['total_pnl']:>9.0f} "
                  f"{r['sharpe']:>8.2f} {dsh:>+5.2f} {r['win_rate']:>5.1f}%{flag}")

        dsh_list = [abs(v['sharpe'] - base_sh) for k, v in relevant.items() if k != base_label]
        cliff_pct = sum(1 for d in dsh_list if d > 1.5) / max(len(dsh_list), 1) * 100
        print(f"  Cliff risk: {cliff_pct:.0f}% of perturbations cause >1.5 Sharpe drop")

    print(f"\n  [Checkpoint] Phase 4 done, elapsed: {(time.time()-t_start)/60:.1f} min")

    # ================================================================
    # PHASE 5: PnL Distribution & Tail Risk
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 5: PnL Distribution & Tail Risk")
    print(f"{'='*70}")

    for name in STRATEGIES:
        trades = all_trades[name]
        pnls = [t.pnl for t in trades]
        arr = np.array(pnls)

        print(f"\n  --- {name} ---")
        print(f"  Total trades: {len(arr)}")
        print(f"  Mean PnL:     ${np.mean(arr):.2f}")
        print(f"  Median PnL:   ${np.median(arr):.2f}")
        print(f"  Std PnL:      ${np.std(arr):.2f}")
        print(f"  Skewness:     {pd.Series(arr).skew():.3f}")
        print(f"  Kurtosis:     {pd.Series(arr).kurtosis():.3f}")

        pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        vals = np.percentile(arr, pcts)
        print(f"\n  Percentile distribution:")
        for p, v in zip(pcts, vals):
            print(f"    P{p:>2}: ${v:>8.2f}")

        worst = sorted(trades, key=lambda t: t.pnl)[:10]
        print(f"\n  10 Worst trades:")
        for i, t in enumerate(worst, 1):
            print(f"    {i}. ${t.pnl:.2f}  ({pd.Timestamp(t.entry_time).strftime('%Y-%m-%d %H:%M')} "
                  f"{t.direction} {t.exit_reason})")

        best = sorted(trades, key=lambda t: t.pnl, reverse=True)[:10]
        print(f"\n  10 Best trades:")
        for i, t in enumerate(best, 1):
            print(f"    {i}. ${t.pnl:.2f}  ({pd.Timestamp(t.entry_time).strftime('%Y-%m-%d %H:%M')} "
                  f"{t.direction} {t.exit_reason})")

        max_consec_loss = 0; max_cl_pnl = 0
        curr = 0; curr_pnl = 0
        for p in pnls:
            if p <= 0:
                curr += 1; curr_pnl += p
                if curr > max_consec_loss:
                    max_consec_loss = curr; max_cl_pnl = curr_pnl
            else:
                curr = 0; curr_pnl = 0
        print(f"\n  Max consecutive losses: {max_consec_loss} trades (total ${max_cl_pnl:.2f})")

        max_cw = 0; curr = 0
        for p in pnls:
            if p > 0:
                curr += 1; max_cw = max(max_cw, curr)
            else:
                curr = 0
        print(f"  Max consecutive wins:   {max_cw} trades")

        sorted_pnls = sorted(pnls, reverse=True)
        top10_n = max(1, len(sorted_pnls) // 10)
        top10_pnl = sum(sorted_pnls[:top10_n])
        total_pnl = sum(pnls)
        print(f"\n  Top 10% trades ({top10_n}): ${top10_pnl:.0f} "
              f"({top10_pnl/max(total_pnl,1)*100:.1f}% of total PnL)")

        exit_reasons = defaultdict(list)
        for t in trades:
            exit_reasons[t.exit_reason].append(t.pnl)
        print(f"\n  Exit reason breakdown:")
        print(f"  {'Reason':<20} {'N':>6} {'PnL':>10} {'AvgPnL':>8} {'WR%':>6}")
        for reason in sorted(exit_reasons.keys()):
            ps = exit_reasons[reason]
            n = len(ps)
            total = sum(ps)
            avg = np.mean(ps)
            wr = sum(1 for p in ps if p > 0) / n * 100
            print(f"  {reason:<20} {n:>6} ${total:>9.0f} ${avg:>7.2f} {wr:>5.1f}%")

    print(f"\n  [Checkpoint] Phase 5 done, elapsed: {(time.time()-t_start)/60:.1f} min")

    # ================================================================
    # PHASE 6: Recent Market (2024-2026) Monthly
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 6: Recent Market (2024-01 to 2026-04) Monthly")
    print(f"{'='*70}")

    for name in STRATEGIES:
        trades = all_trades[name]
        recent_trades = [t for t in trades if pd.Timestamp(t.entry_time).year >= 2024]
        csh = corrected_sharpe(recent_trades)

        t_kcbw = filter_kcbw5(recent_trades, h1_df)
        t_kcbw_cap = apply_max_loss_cap(t_kcbw, 30)
        csh_combo = corrected_sharpe(t_kcbw_cap)
        pnl_bare = sum(t.pnl for t in recent_trades)
        pnl_combo = sum(t.pnl for t in t_kcbw_cap)

        print(f"\n  --- {name} (2024-2026) ---")
        print(f"    Bare:        CorrSh={csh:.2f}, N={len(recent_trades)}, PnL=${pnl_bare:.0f}")
        print(f"    +KCBW5+Cap30: CorrSh={csh_combo:.2f}, N={len(t_kcbw_cap)}, PnL=${pnl_combo:.0f}")

        print(f"\n    Monthly (bare):")
        print(f"    {'Month':<8} {'N':>4} {'PnL':>8} {'WR%':>6}")
        monthly = defaultdict(list)
        for t in recent_trades:
            m = pd.Timestamp(t.exit_time).to_period('M')
            monthly[m].append(t.pnl)
        for m in sorted(monthly.keys()):
            ps = monthly[m]
            n = len(ps); pnl = sum(ps)
            wr = sum(1 for p in ps if p > 0) / n * 100
            print(f"    {str(m):<8} {n:>4} ${pnl:>7.0f} {wr:>5.1f}%")

        print(f"\n    Monthly (+KCBW5+Cap30):")
        print(f"    {'Month':<8} {'N':>4} {'PnL':>8} {'WR%':>6}")
        monthly2 = defaultdict(list)
        for t in t_kcbw_cap:
            m = pd.Timestamp(t.exit_time).to_period('M')
            monthly2[m].append(t.pnl)
        for m in sorted(monthly2.keys()):
            ps = monthly2[m]
            n = len(ps); pnl = sum(ps)
            wr = sum(1 for p in ps if p > 0) / n * 100
            print(f"    {str(m):<8} {n:>4} ${pnl:>7.0f} {wr:>5.1f}%")

    print(f"\n  [Checkpoint] Phase 6 done, elapsed: {(time.time()-t_start)/60:.1f} min")

    # ================================================================
    # PHASE 7: Final Verdict
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 7: FINAL VERDICT (R41 + R42)")
    print(f"{'='*70}")

    print(f"\n  === R41 Existing Results ===")
    print(f"  K-Fold (bare):       L8_BASE 6/6 PASS (Min=0.50)  vs  L8_HYBRID 5/6 FAIL (Min=-0.51)")
    print(f"  K-Fold +KCBW5+Cap:   L8_BASE 6/6 (Min=1.14)      vs  L8_HYBRID 6/6 (Min=0.89)")

    print(f"\n  === R42 New Results ===")
    verdicts = []

    # WF
    a_wf = wf_by_strat.get('L8_BASE', [])
    b_wf = wf_by_strat.get('L8_HYBRID', [])
    a_sharpes = [r['orig_sharpe'] for r in a_wf]
    b_sharpes = [r['orig_sharpe'] for r in b_wf]
    a_pos = sum(1 for s in a_sharpes if s > 0)
    b_pos = sum(1 for s in b_sharpes if s > 0)

    print(f"\n  {'Metric':<40} {'L8_BASE':>10} {'L8_HYBRID':>10} {'Winner':>8}")
    print(f"  {'-'*70}")

    w = "BASE" if a_pos > b_pos else ("HYBRID" if b_pos > a_pos else "TIE")
    verdicts.append(w)
    print(f"  {'WF positive windows':<40} {a_pos}/{len(a_sharpes):>7} {b_pos}/{len(b_sharpes):>7} {w:>8}")

    a_mean = np.mean(a_sharpes) if a_sharpes else 0
    b_mean = np.mean(b_sharpes) if b_sharpes else 0
    w = "BASE" if a_mean > b_mean else ("HYBRID" if b_mean > a_mean else "TIE")
    verdicts.append(w)
    print(f"  {'WF mean Sharpe':<40} {a_mean:>10.2f} {b_mean:>10.2f} {w:>8}")

    a_min = min(a_sharpes) if a_sharpes else -99
    b_min = min(b_sharpes) if b_sharpes else -99
    w = "BASE" if a_min > b_min else ("HYBRID" if b_min > a_min else "TIE")
    verdicts.append(w)
    print(f"  {'WF min Sharpe':<40} {a_min:>10.2f} {b_min:>10.2f} {w:>8}")

    # Cliff
    for name in STRATEGIES:
        base_label = f"{name}_BASE"
        base_sh = cliff_results.get(base_label, {}).get('sharpe', 0)
        relevant = {k: v for k, v in cliff_results.items() if k.startswith(name) and k != base_label}
        dsh_list = [abs(v['sharpe'] - base_sh) for v in relevant.values()]
        cliff_pct = sum(1 for d in dsh_list if d > 1.5) / max(len(dsh_list), 1) * 100
        if name == 'L8_BASE':
            a_cliff = cliff_pct
        else:
            b_cliff = cliff_pct

    w = "BASE" if a_cliff <= b_cliff else "HYBRID"
    verdicts.append(w)
    print(f"  {'Param cliff risk % (lower=better)':<40} {a_cliff:>9.0f}% {b_cliff:>9.0f}% {w:>8}")

    # Consec loss
    for name in STRATEGIES:
        trades = all_trades[name]
        max_cl = 0; curr = 0
        for t in trades:
            if t.pnl <= 0:
                curr += 1; max_cl = max(max_cl, curr)
            else:
                curr = 0
        if name == 'L8_BASE':
            a_cl = max_cl
        else:
            b_cl = max_cl

    w = "BASE" if a_cl <= b_cl else "HYBRID"
    verdicts.append(w)
    print(f"  {'Max consec losses (lower=better)':<40} {a_cl:>10} {b_cl:>10} {w:>8}")

    # Recent PnL
    for name in STRATEGIES:
        recent = [t for t in all_trades[name] if pd.Timestamp(t.entry_time).year >= 2024]
        rpnl = sum(t.pnl for t in recent)
        if name == 'L8_BASE':
            a_rpnl = rpnl
        else:
            b_rpnl = rpnl

    w = "BASE" if a_rpnl > b_rpnl else ("HYBRID" if b_rpnl > a_rpnl else "TIE")
    verdicts.append(w)
    print(f"  {'Recent (2024+) PnL':<40} ${a_rpnl:>9.0f} ${b_rpnl:>9.0f} {w:>8}")

    base_wins = verdicts.count("BASE")
    hybrid_wins = verdicts.count("HYBRID")
    ties = verdicts.count("TIE")

    print(f"\n  R42 SCORE: L8_BASE {base_wins} vs L8_HYBRID {hybrid_wins} (Ties: {ties})")
    print(f"\n  Combined with R41:")
    print(f"  - R41 K-Fold bare: BASE 6/6 PASS, HYBRID 5/6 FAIL -> +1 BASE")
    print(f"  - R41 K-Fold combo: BASE min=1.14 > HYBRID min=0.89 -> +1 BASE")
    print(f"  - R42 score: BASE {base_wins} vs HYBRID {hybrid_wins}")

    total_base = base_wins + 2
    total_hybrid = hybrid_wins
    print(f"\n  TOTAL: L8_BASE {total_base} vs L8_HYBRID {total_hybrid}")

    if total_base >= total_hybrid:
        print(f"\n  >>> RECOMMENDATION: L8_BASE + KCBW5 + Cap30 <<<")
    else:
        print(f"\n  >>> RECOMMENDATION: L8_HYBRID + KCBW5 + Cap30 <<<")

    elapsed_total = (time.time() - t_start) / 60
    print(f"\n{'='*70}")
    print(f"  Total runtime: {elapsed_total:.1f} minutes ({elapsed_total/60:.1f} hours)")
    print(f"{'='*70}")

    log_f.close()
    sys.stdout = sys.__stdout__
    print(f"\nR42 complete! Output: {log_path}")
    print(f"Runtime: {elapsed_total:.1f} min")


if __name__ == '__main__':
    main()
