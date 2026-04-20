#!/usr/bin/env python3
"""
EXP-CHOPPY-OC: Choppy Gate Opportunity Cost Analysis
=====================================================
Research question: When IntradayTrendMeter blocks entries as "choppy",
how many of those blocked signals would have been profitable?

Specifically addresses the scenario observed on 2026-04-17:
  - Gold made a large move (>$40 in 2 hours)
  - trend_score stayed at 0.24-0.38 all day (choppy, threshold=0.50)
  - Keltner signals fired but were blocked by choppy filter
  - Paper trades (unfiltered) captured the move

Approach:
  1. Run baseline (choppy ON, current 0.50) vs no-filter (choppy OFF)
  2. Identify the "delta trades" — trades that only exist when filter is OFF
  3. Analyze those delta trades: win rate, avg PnL, MFE, exit reasons
  4. Classify delta trades by context: was the signal at a "regime transition"
     (choppy -> trending) or during sustained choppy?
  5. Check if a lower threshold (e.g. 0.40, 0.45) captures the best delta
     trades while still filtering the worst ones
  6. K-Fold validation for any promising threshold
  7. Year-by-year stability

This is a HYPOTHESIS — no code changes until data confirms.
"""
import io
import sys
import os
import time
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional

try:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
except Exception:
    pass

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_this_dir, '..')
if not os.path.isdir(os.path.join(_project_root, 'backtest')):
    _project_root = os.path.join(_this_dir, '..', '..')
sys.path.insert(0, os.path.abspath(_project_root))
os.chdir(os.path.abspath(_project_root))

import numpy as np
import pandas as pd
from backtest import DataBundle, run_variant
from backtest.runner import LIVE_PARITY_KWARGS, calc_stats
from backtest.engine import TradeRecord

OUTPUT_FILE = "exp_choppy_opportunity_cost_output.txt"
SPREAD = 0.30

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-10"),
]

BASE = {**LIVE_PARITY_KWARGS, "keltner_max_hold_m15": 20, "spread_cost": SPREAD}


class TeeOutput:
    def __init__(self, fp):
        self.file = open(fp, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, d):
        self.stdout.write(d)
        self.file.write(d)
        self.file.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()


tee = TeeOutput(OUTPUT_FILE)
sys.stdout = tee


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def _ts(x):
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t


def trade_key(t: TradeRecord) -> str:
    """Unique key for a trade based on entry time + direction + strategy."""
    return f"{t.entry_time}|{t.direction}|{t.strategy}"


def find_delta_trades(
    trades_filtered: List[TradeRecord],
    trades_unfiltered: List[TradeRecord],
) -> List[TradeRecord]:
    """Find trades that only exist in unfiltered (i.e., were blocked by filter)."""
    filtered_keys = {trade_key(t) for t in trades_filtered}
    return [t for t in trades_unfiltered if trade_key(t) not in filtered_keys]


def analyze_trades(trades: List[TradeRecord], label: str) -> Dict[str, Any]:
    """Compute detailed stats for a list of trades."""
    if not trades:
        return {"label": label, "n": 0}

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    exit_reasons = defaultdict(int)
    for t in trades:
        exit_reasons[t.exit_reason] += 1

    return {
        "label": label,
        "n": len(trades),
        "total_pnl": sum(pnls),
        "avg_pnl": np.mean(pnls),
        "median_pnl": np.median(pnls),
        "win_rate": 100.0 * len(wins) / len(pnls),
        "avg_win": np.mean(wins) if wins else 0,
        "avg_loss": np.mean(losses) if losses else 0,
        "avg_bars_held": np.mean([t.bars_held for t in trades]),
        "exit_reasons": dict(exit_reasons),
        "by_strategy": _by_strategy(trades),
    }


def _by_strategy(trades: List[TradeRecord]) -> Dict[str, Dict]:
    strats = defaultdict(list)
    for t in trades:
        strats[t.strategy].append(t)
    result = {}
    for s, ts in sorted(strats.items()):
        pnls = [t.pnl for t in ts]
        result[s] = {
            "n": len(ts),
            "pnl": sum(pnls),
            "wr": 100.0 * sum(1 for p in pnls if p > 0) / len(pnls),
            "avg": np.mean(pnls),
        }
    return result


def classify_delta_by_hour(delta_trades: List[TradeRecord]) -> None:
    """Analyze delta trades by entry hour (UTC) to find time patterns."""
    by_hour = defaultdict(list)
    for t in delta_trades:
        h = _ts(t.entry_time).hour
        by_hour[h].append(t.pnl)

    print(f"\n    {'Hour':>6s}  {'N':>5s}  {'WR%':>6s}  {'Avg$':>8s}  {'Total$':>10s}")
    print("    " + "-" * 40)
    for h in sorted(by_hour.keys()):
        pnls = by_hour[h]
        n = len(pnls)
        wr = 100.0 * sum(1 for p in pnls if p > 0) / n
        avg = np.mean(pnls)
        total = sum(pnls)
        print(f"    {h:>4d}h  {n:>5d}  {wr:>5.1f}%  ${avg:>7.2f}  {fmt(total)}")


def classify_delta_by_year(delta_trades: List[TradeRecord]) -> None:
    """Year-by-year stability of delta trades."""
    by_year = defaultdict(list)
    for t in delta_trades:
        y = _ts(t.entry_time).year
        by_year[y].append(t)

    print(f"\n    {'Year':>6s}  {'N':>5s}  {'WR%':>6s}  {'Avg$':>8s}  {'Total$':>10s}  {'Sharpe':>7s}")
    print("    " + "-" * 50)
    for y in sorted(by_year.keys()):
        ts = by_year[y]
        pnls = [t.pnl for t in ts]
        n = len(pnls)
        wr = 100.0 * sum(1 for p in pnls if p > 0) / n
        avg = np.mean(pnls)
        total = sum(pnls)
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0
        print(f"    {y:>6d}  {n:>5d}  {wr:>5.1f}%  ${avg:>7.2f}  {fmt(total)}  {sharpe:>7.2f}")


def classify_delta_by_exit(delta_trades: List[TradeRecord]) -> None:
    """Exit reason distribution for delta trades vs typical."""
    reasons = defaultdict(list)
    for t in delta_trades:
        reasons[t.exit_reason].append(t.pnl)

    print(f"\n    {'Exit Reason':>15s}  {'N':>5s}  {'%':>6s}  {'WR%':>6s}  {'Avg$':>8s}  {'Total$':>10s}")
    print("    " + "-" * 55)
    total_n = len(delta_trades)
    for reason in sorted(reasons.keys(), key=lambda r: -len(reasons[r])):
        pnls = reasons[reason]
        n = len(pnls)
        pct = 100.0 * n / total_n
        wr = 100.0 * sum(1 for p in pnls if p > 0) / n
        avg = np.mean(pnls)
        total = sum(pnls)
        print(f"    {reason:>15s}  {n:>5d}  {pct:>5.1f}%  {wr:>5.1f}%  ${avg:>7.2f}  {fmt(total)}")


def main():
    print("\n" + "=" * 80)
    print("  EXP-CHOPPY-OC: CHOPPY GATE OPPORTUNITY COST ANALYSIS")
    print("  Research: What do we lose by blocking trades in 'choppy' regime?")
    print("=" * 80)
    print(f"  Started: {datetime.now()}")

    t0 = time.time()
    data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

    # ═══════════════════════════════════════════════════════════
    # PART 1: Baseline vs No-Filter — overall comparison
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  PART 1: BASELINE (choppy=0.50) vs NO FILTER (adaptive OFF)")
    print("=" * 80)

    s_base = run_variant(data, "Baseline_choppy0.50", verbose=False, **BASE)
    s_off = run_variant(data, "NoFilter_adaptive_OFF", verbose=False,
                        **{**BASE, "intraday_adaptive": False})

    print(f"\n  {'Config':<30s}  {'N':>5s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'$/t':>8s}  {'MaxDD':>11s}  {'skip_chp':>8s}")
    print("  " + "-" * 95)
    for label, s in [("Baseline (choppy=0.50)", s_base), ("No Filter (OFF)", s_off)]:
        n = s['n']
        avg = s['total_pnl'] / n if n > 0 else 0
        sc = s.get('skipped_choppy', 0)
        print(f"  {label:<30s}  {n:>5d}  {s['sharpe']:>7.2f}  "
              f"{fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  "
              f"${avg:>7.2f}  {fmt(s['max_dd'])}  {sc:>8d}")

    base_trades = s_base.get('_trades', [])
    off_trades = s_off.get('_trades', [])

    # ═══════════════════════════════════════════════════════════
    # PART 2: Identify & analyze delta trades
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  PART 2: DELTA TRADES (blocked by choppy filter)")
    print("=" * 80)

    delta = find_delta_trades(base_trades, off_trades)
    also_in_base = [t for t in off_trades if trade_key(t) in {trade_key(bt) for bt in base_trades}]

    print(f"\n  Baseline trades: {len(base_trades)}")
    print(f"  No-filter trades: {len(off_trades)}")
    print(f"  Delta trades (blocked by choppy): {len(delta)}")
    print(f"  Shared trades: {len(also_in_base)}")

    if not delta:
        print("  [!] No delta trades found — choppy filter may not be active or has no effect")
        return

    da = analyze_trades(delta, "Delta (blocked)")
    print(f"\n  Delta trade summary:")
    print(f"    N={da['n']}  WR={da['win_rate']:.1f}%  avg_pnl=${da['avg_pnl']:.2f}  "
          f"total_pnl={fmt(da['total_pnl'])}  avg_bars={da['avg_bars_held']:.1f}")
    print(f"    avg_win=${da['avg_win']:.2f}  avg_loss=${da['avg_loss']:.2f}")

    print(f"\n  By strategy:")
    for s, info in da['by_strategy'].items():
        print(f"    {s}: N={info['n']}  WR={info['wr']:.1f}%  avg=${info['avg']:.2f}  total={fmt(info['pnl'])}")

    # 2a. Exit reason distribution
    print("\n  --- 2A: EXIT REASONS FOR DELTA TRADES ---")
    classify_delta_by_exit(delta)

    # 2b. Hour distribution
    print("\n  --- 2B: DELTA TRADES BY ENTRY HOUR (UTC) ---")
    classify_delta_by_hour(delta)

    # 2c. Year-by-year stability
    print("\n  --- 2C: DELTA TRADES BY YEAR ---")
    classify_delta_by_year(delta)

    # ═══════════════════════════════════════════════════════════
    # PART 3: MFE analysis — were blocked trades near big moves?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  PART 3: DELTA TRADE QUALITY — MFE DISTRIBUTION")
    print("=" * 80)

    # Note: TradeRecord doesn't have MFE by default in the engine,
    # but we can look at PnL + exit_reason as proxies
    delta_wins = [t for t in delta if t.pnl > 0]
    delta_losses = [t for t in delta if t.pnl <= 0]

    if delta_wins:
        win_pnls = [t.pnl for t in delta_wins]
        print(f"\n  Winning delta trades: {len(delta_wins)}")
        print(f"    Avg win: ${np.mean(win_pnls):.2f}")
        print(f"    Median win: ${np.median(win_pnls):.2f}")
        print(f"    Max win: ${max(win_pnls):.2f}")
        print(f"    Top 10 wins: {[f'${p:.2f}' for p in sorted(win_pnls, reverse=True)[:10]]}")

        # Big wins (> $10)
        big_wins = [t for t in delta_wins if t.pnl > 10]
        print(f"\n    Big wins (>$10): {len(big_wins)} trades")
        for t in sorted(big_wins, key=lambda x: -x.pnl)[:10]:
            print(f"      {_ts(t.entry_time).strftime('%Y-%m-%d %H:%M')} "
                  f"{t.direction} {t.strategy} PnL=${t.pnl:.2f} "
                  f"bars={t.bars_held} exit={t.exit_reason}")

    if delta_losses:
        loss_pnls = [t.pnl for t in delta_losses]
        print(f"\n  Losing delta trades: {len(delta_losses)}")
        print(f"    Avg loss: ${np.mean(loss_pnls):.2f}")
        print(f"    Median loss: ${np.median(loss_pnls):.2f}")
        print(f"    Worst 10: {[f'${p:.2f}' for p in sorted(loss_pnls)[:10]]}")

    # ═══════════════════════════════════════════════════════════
    # PART 4: Threshold sweep — find optimal choppy threshold
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  PART 4: CHOPPY THRESHOLD SWEEP")
    print("  (0.50 = current, lower = more permissive)")
    print("=" * 80)

    THRESHOLDS = [0.30, 0.35, 0.40, 0.42, 0.45, 0.47, 0.50, 0.55]
    sweep_results = {}

    print(f"\n  {'Choppy_th':>9s}  {'N':>5s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'$/t':>8s}  {'MaxDD':>11s}  {'vs Base':>8s}")
    print("  " + "-" * 75)

    for th in THRESHOLDS:
        kwargs = {**BASE, "choppy_threshold": th}
        s = run_variant(data, f"choppy_{th}", verbose=False, **kwargs)
        n = s['n']
        avg = s['total_pnl'] / n if n > 0 else 0
        delta_s = s['sharpe'] - s_base['sharpe']
        marker = " <-- current" if th == 0.50 else ""
        print(f"  {th:>9.2f}  {n:>5d}  {s['sharpe']:>7.2f}  "
              f"{fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  "
              f"${avg:>7.2f}  {fmt(s['max_dd'])}  {delta_s:>+7.2f}{marker}")
        sweep_results[th] = s

    # ═══════════════════════════════════════════════════════════
    # PART 5: Double spread stress test
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  PART 5: DOUBLE SPREAD STRESS TEST ($0.50)")
    print("=" * 80)

    print(f"\n  {'Choppy_th':>9s}  {'N':>5s}  {'Sharpe':>7s}  {'PnL':>11s}  {'WR%':>6s}  {'$/t':>8s}")
    print("  " + "-" * 55)

    for th in [0.40, 0.42, 0.45, 0.50]:
        kwargs = {**LIVE_PARITY_KWARGS, "keltner_max_hold_m15": 20,
                  "spread_cost": 0.50, "choppy_threshold": th}
        s = run_variant(data, f"choppy_{th}_sp50", verbose=False, **kwargs)
        n = s['n']
        avg = s['total_pnl'] / n if n > 0 else 0
        marker = " <-- current" if th == 0.50 else ""
        print(f"  {th:>9.2f}  {n:>5d}  {s['sharpe']:>7.2f}  "
              f"{fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  "
              f"${avg:>7.2f}{marker}")

    # ═══════════════════════════════════════════════════════════
    # PART 6: K-Fold for best non-current threshold
    # ═══════════════════════════════════════════════════════════
    ranked = sorted(
        [(th, s) for th, s in sweep_results.items() if th != 0.50],
        key=lambda x: -x[1]['sharpe']
    )
    best_th, best_s = ranked[0]

    print("\n" + "=" * 80)
    print(f"  PART 6: K-FOLD VALIDATION — best={best_th} vs current=0.50")
    print("=" * 80)

    if best_s['sharpe'] <= s_base['sharpe']:
        print(f"\n  Current threshold (0.50) already optimal (Sharpe {s_base['sharpe']:.2f})")
        print(f"  Best alternative: {best_th} (Sharpe {best_s['sharpe']:.2f}) — NOT better")
    else:
        print(f"\n  Testing: choppy={best_th} (Sharpe {best_s['sharpe']:.2f}) vs 0.50 ({s_base['sharpe']:.2f})")

        # K-Fold at both $0.30 and $0.50 spread
        for sp_label, sp_val in [("$0.30", 0.30), ("$0.50", 0.50)]:
            wins = 0
            print(f"\n  --- K-Fold @ spread {sp_label} ---")
            print(f"  {'Fold':<8s}  {'Base(0.50)':>10s}  {'Test':>10s}  {'Delta':>7s}  {'Win?':>5s}")
            print("  " + "-" * 45)

            for fold_name, start, end in FOLDS:
                fold_data = data.slice(start, end)
                if len(fold_data.m15_df) < 1000:
                    continue
                sb = run_variant(fold_data, f"KF_B_{fold_name}", verbose=False,
                                 **{**LIVE_PARITY_KWARGS, "keltner_max_hold_m15": 20,
                                    "spread_cost": sp_val})
                st = run_variant(fold_data, f"KF_T_{fold_name}", verbose=False,
                                 **{**LIVE_PARITY_KWARGS, "keltner_max_hold_m15": 20,
                                    "spread_cost": sp_val, "choppy_threshold": best_th})
                d = st['sharpe'] - sb['sharpe']
                won = d > 0
                if won:
                    wins += 1
                print(f"  {fold_name:<8s}  {sb['sharpe']:>10.2f}  {st['sharpe']:>10.2f}  "
                      f"{d:>+6.2f}  {'V' if won else 'X':>5s}")
            result = "PASS" if wins >= 5 else "FAIL"
            print(f"  Result: {wins}/6 {result}")

    # Also test top 2-3 other promising thresholds
    print("\n  --- Additional threshold K-Folds (sp=$0.30) ---")
    for th, s in ranked[1:3]:
        if s['sharpe'] <= s_base['sharpe']:
            continue
        wins = 0
        for fold_name, start, end in FOLDS:
            fold_data = data.slice(start, end)
            if len(fold_data.m15_df) < 1000:
                continue
            sb = run_variant(fold_data, f"KF2_B_{fold_name}", verbose=False, **BASE)
            st = run_variant(fold_data, f"KF2_T_{fold_name}", verbose=False,
                             **{**BASE, "choppy_threshold": th})
            d = st['sharpe'] - sb['sharpe']
            if d > 0:
                wins += 1
        result = "PASS" if wins >= 5 else "FAIL"
        print(f"    choppy={th}: {wins}/6 {result}  (full Sharpe={s['sharpe']:.2f})")

    # ═══════════════════════════════════════════════════════════
    # PART 7: Year-by-year stability for promising thresholds
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  PART 7: YEAR-BY-YEAR COMPARISON — current vs best threshold")
    print("=" * 80)

    for th_test in [best_th, 0.50]:
        kwargs = {**BASE, "choppy_threshold": th_test}
        s = run_variant(data, f"yearly_{th_test}", verbose=False, **kwargs)
        trades = s.get('_trades', [])
        if not trades:
            continue

        print(f"\n  choppy={th_test}:")
        by_year = defaultdict(list)
        for t in trades:
            y = _ts(t.entry_time).year
            by_year[y].append(t)

        print(f"    {'Year':>6s}  {'N':>5s}  {'WR%':>6s}  {'PnL':>10s}  {'Avg$':>8s}")
        print("    " + "-" * 40)
        for y in sorted(by_year.keys()):
            ts = by_year[y]
            pnls = [t.pnl for t in ts]
            print(f"    {y:>6d}  {len(ts):>5d}  "
                  f"{100*sum(1 for p in pnls if p>0)/len(pnls):>5.1f}%  "
                  f"{fmt(sum(pnls))}  ${np.mean(pnls):>7.2f}")

    # ═══════════════════════════════════════════════════════════
    # CONCLUSION
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  CONCLUSION TEMPLATE")
    print("=" * 80)
    print("""
  Fill in after reviewing results:

  1. Choppy filter opportunity cost?
     - Delta trades total PnL: $???
     - Delta trades avg PnL: $???  (positive = filter is too aggressive)
     - Delta trades win rate: ???%

  2. Is there a better threshold?
     - Best threshold: ???
     - Sharpe improvement over 0.50: +???
     - K-Fold result: ???/6

  3. Delta trade characteristics:
     - Mostly trailing/TP wins, or mostly timeout/SL losses?
     - Concentrated in specific hours? (late-day breakouts?)
     - Year-by-year stable, or driven by 1-2 outlier years?

  4. Recommendation:
     [ ] Keep current threshold (0.50) — filter cost is justified
     [ ] Lower threshold to ??? — captures good signals without too much noise
     [ ] Needs further research on ???
    """)

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed/60:.1f} minutes")
    print(f"  Completed: {datetime.now()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
