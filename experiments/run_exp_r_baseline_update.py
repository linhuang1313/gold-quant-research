"""
Experiment R: Baseline robustness on updated data (incl. April 2026 tariff crash).
Remote-friendly UTF-8 stdout, LIVE_PARITY, 6-fold equal time K-Fold on M15.
"""
import sys
import io
import time
import traceback

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

import pandas as pd

import research_config as config
from backtest.runner import (
    DataBundle,
    LIVE_PARITY_KWARGS,
    calc_stats,
    run_variant,
)

PREV_BASELINE_SHARPE_030 = 2.29
PREV_MAXHOLD20_SHARPE_030 = 2.62


def _utc_str(ts) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.strftime("%Y-%m-%d %H:%M:%S") + "+00:00"


def equity_from_trades(trades):
    if not trades:
        return [float(config.CAPITAL)]
    ordered = sorted(trades, key=lambda x: x.exit_time)
    eq = [float(config.CAPITAL)]
    for t in ordered:
        eq.append(eq[-1] + t.pnl)
    return eq


def six_equal_time_folds(bundle: DataBundle):
    """Split M15 timeline into 6 non-overlapping equal-length chunks (by bar count)."""
    m15 = bundle.m15_df
    n = len(m15)
    folds = []
    for i in range(6):
        lo = i * n // 6
        hi = (i + 1) * n // 6
        if lo >= hi:
            continue
        start_ts = _utc_str(m15.index[lo])
        if hi >= n:
            end_ts = _utc_str(m15.index[-1] + pd.Timedelta(hours=2))
        else:
            end_ts = _utc_str(m15.index[hi])
        folds.append((f"Fold{i + 1}", start_ts, end_ts, lo, hi))
    return folds


def yearly_sharpe_pnl(trades):
    rows = []
    if not trades:
        return rows
    by_year = {}
    for t in trades:
        y = pd.Timestamp(t.exit_time).year
        by_year.setdefault(y, []).append(t)
    for y in sorted(by_year.keys()):
        st = by_year[y]
        s = calc_stats(st, equity_from_trades(st))
        rows.append(
            {
                "year": y,
                "n": s["n"],
                "sharpe": s["sharpe"],
                "total_pnl": s["total_pnl"],
                "win_rate": s["win_rate"],
                "max_dd": s["max_dd"],
            }
        )
    return rows


def run_phase(name, fn):
    print("\n" + "=" * 88)
    print(f"  PHASE: {name}")
    print("=" * 88)
    t0 = time.time()
    try:
        out = fn()
        print(f"  (phase ok in {time.time() - t0:.1f}s)")
        return out
    except Exception as e:
        print(f"  PHASE FAILED: {e}")
        traceback.print_exc()
        return None


def main():
    t_script = time.time()
    print("\n" + "=" * 88)
    print("  run_exp_r_baseline_update.py — Baseline robustness (updated sample)")
    print("=" * 88)
    print(
        "  Estimated runtime: ~8–25 min depending on CPU (full sample × ~10+ engine runs + 6 folds)."
    )

    summary_rows = []

    bundle = None
    res = run_phase("Load DataBundle.load_default()", lambda: DataBundle.load_default(start="2015-01-01"))
    bundle = res
    if bundle is None:
        print("Cannot continue without data.")
        return

    # --- Phase 1: baseline spreads ---
    def p1():
        nonlocal summary_rows
        for spread, label in [(0.0, "LIVE $0.00"), (0.30, "LIVE $0.30"), (0.50, "LIVE $0.50")]:
            kw = {**LIVE_PARITY_KWARGS, "spread_cost": spread}
            st = run_variant(bundle, label, verbose=True, **kw)
            delta = st["sharpe"] - PREV_BASELINE_SHARPE_030 if spread == 0.30 else None
            print(
                f"    vs prev baseline @ $0.30 Sharpe={PREV_BASELINE_SHARPE_030:.2f} → "
                f"delta={delta:+.2f}" if delta is not None else ""
            )
            summary_rows.append(
                {
                    "phase": "1 Baseline",
                    "label": label,
                    "sharpe": st["sharpe"],
                    "pnl": st["total_pnl"],
                    "n": st["n"],
                    "wr": st["win_rate"],
                    "max_dd": st["max_dd"],
                }
            )
        return True

    run_phase("1) Baseline full sample: $0.00 / $0.30 / $0.50", p1)

    # --- Phase 2: MaxHold 20 ---
    def p2():
        nonlocal summary_rows
        kw = {**LIVE_PARITY_KWARGS, "spread_cost": 0.30, "keltner_max_hold_m15": 20}
        st = run_variant(bundle, "MaxHold=20 @ $0.30", verbose=True, **kw)
        d = st["sharpe"] - PREV_MAXHOLD20_SHARPE_030
        print(f"    vs prev MaxHold=20 Sharpe={PREV_MAXHOLD20_SHARPE_030:.2f} → delta={d:+.2f}")
        summary_rows.append(
            {
                "phase": "2 MaxHold20",
                "label": "keltner_max_hold_m15=20",
                "sharpe": st["sharpe"],
                "pnl": st["total_pnl"],
                "n": st["n"],
                "wr": st["win_rate"],
                "max_dd": st["max_dd"],
            }
        )
        return st

    run_phase("2) MaxHold=20 @ $0.30 (full sample)", p2)

    # --- Phase 3: 6-fold equal time, baseline $0.30 ---
    fold_results = []

    def p3():
        nonlocal fold_results
        base_kw = {**LIVE_PARITY_KWARGS, "spread_cost": 0.30}
        full = run_variant(bundle, "FULL ref $0.30", verbose=False, **base_kw)
        ref_sharpe = full["sharpe"]
        print(f"  Full-sample Sharpe (reference): {ref_sharpe:.3f}")
        folds = six_equal_time_folds(bundle)
        for fname, start_s, end_s, lo, hi in folds:
            sub = bundle.slice(start_s, end_s)
            if len(sub.m15_df) < 800:
                print(f"  {fname}: skip (M15 bars={len(sub.m15_df)})")
                continue
            st = run_variant(sub, fname, verbose=False, **base_kw)
            d = st["sharpe"] - ref_sharpe
            print(
                f"  {fname} [{start_s[:10]}..{end_s[:10]}] M15={len(sub.m15_df):5d} "
                f"Sharpe={st['sharpe']:.3f}  Δvs_full={d:+.3f}  PnL=${st['total_pnl']:.0f}  N={st['n']}"
            )
            fold_results.append({**st, "delta_vs_full": d, "fold_name": fname})
        return fold_results

    run_phase("3) K-Fold (6 equal M15 chunks) LIVE_PARITY @ $0.30", p3)

    # --- Phase 4: April 2026 ---
    def p4():
        kw = {**LIVE_PARITY_KWARGS, "spread_cost": 0.30}
        st = run_variant(bundle, "internal", verbose=False, **kw)
        trades = st["_trades"]
        apr = [
            t
            for t in trades
            if pd.Timestamp(t.exit_time).year == 2026 and pd.Timestamp(t.exit_time).month == 4
        ]
        if not apr:
            print("  No trades with exit in 2026-04.")
            return
        s = calc_stats(apr, equity_from_trades(apr))
        print(
            f"  Trades (exit in 2026-04): N={s['n']}  Sharpe={s['sharpe']:.3f}  "
            f"PnL=${s['total_pnl']:.0f}  WR={s['win_rate']:.1f}%  MaxDD=${s['max_dd']:.0f}"
        )
        summary_rows.append(
            {
                "phase": "4 Apr2026",
                "label": "exit in 2026-04",
                "sharpe": s["sharpe"],
                "pnl": s["total_pnl"],
                "n": s["n"],
                "wr": s["win_rate"],
                "max_dd": s["max_dd"],
            }
        )

    run_phase("4) Focus: trades exiting in April 2026 (tariff crash window)", p4)

    # --- Phase 5: year-by-year ---
    def p5():
        kw = {**LIVE_PARITY_KWARGS, "spread_cost": 0.30}
        st = run_variant(bundle, "internal", verbose=False, **kw)
        trades = st["_trades"]
        print(f"  {'Year':<6} {'N':>5} {'Sharpe':>8} {'PnL':>12} {'WR%':>7} {'MaxDD':>10}")
        print("  " + "-" * 52)
        for row in yearly_sharpe_pnl(trades):
            print(
                f"  {row['year']:<6} {row['n']:>5} {row['sharpe']:>8.3f} "
                f"${row['total_pnl']:>10.0f} {row['win_rate']:>6.1f}% ${row['max_dd']:>8.0f}"
            )

    run_phase("5) Year-by-year Sharpe & PnL (@ $0.30)", p5)

    # --- Summary table ---
    print("\n" + "=" * 88)
    print("  SUMMARY (key runs)")
    print("=" * 88)
    hdr = f"  {'Phase':<14} {'Label':<22} {'Sharpe':>8} {'PnL':>10} {'N':>5} {'WR%':>7} {'MaxDD':>8}"
    print(hdr)
    print("  " + "-" * 86)
    for r in summary_rows:
        print(
            f"  {r['phase']:<14} {r['label'][:22]:<22} {r['sharpe']:>8.3f} "
            f"${r['pnl']:>8.0f} {r['n']:>5} {r['wr']:>6.1f}% {r['max_dd']:>8.0f}"
        )
    if fold_results:
        print("\n  K-Fold Sharpe list:", ", ".join(f"{x['fold_name']}={x['sharpe']:.2f}" for x in fold_results))

    print(f"\n  Total script time: {time.time() - t_script:.1f}s\n")


if __name__ == "__main__":
    main()
