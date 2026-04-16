"""
Experiment S: Full-sample + K-Fold under historical (Dukascopy) spread vs fixed spreads.
"""
import sys
import io
import time
import traceback

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

import research_config as config
from backtest.runner import (
    DataBundle,
    LIVE_PARITY_KWARGS,
    calc_stats,
    load_spread_series,
    run_variant,
)

SPREAD_MAX = 3.0


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
        folds.append((f"Fold{i + 1}", start_ts, end_ts))
    return folds


def slice_spread_to_m15(spread_series: pd.Series, m15_df: pd.DataFrame) -> pd.Series:
    """Restrict spread series to M15 window (ms index, same as engine)."""
    if spread_series is None or len(spread_series) == 0:
        return spread_series
    t0 = int(pd.Timestamp(m15_df.index[0]).timestamp() * 1000)
    t1 = int(pd.Timestamp(m15_df.index[-1]).timestamp() * 1000)
    s = spread_series[(spread_series.index >= t0) & (spread_series.index <= t1)].copy()
    s = s.sort_index()
    if len(s) == 0:
        return spread_series
    return s


def historical_spread_at_exit(bar_time, spread_series: pd.Series, spread_base: float) -> float:
    """Mirror BacktestEngine._calc_dynamic_spread for historical model."""
    if spread_series is None or len(spread_series) == 0:
        return spread_base
    ts_ms = int(pd.Timestamp(bar_time).timestamp() * 1000)
    idx = spread_series.index.searchsorted(ts_ms, side="right") - 1
    if 0 <= idx < len(spread_series):
        return min(float(spread_series.iloc[idx]), SPREAD_MAX)
    return spread_base


def spread_dollars_charged(trade, spread_series: pd.Series, spread_base: float) -> float:
    spr = historical_spread_at_exit(trade.exit_time, spread_series, spread_base)
    return round(spr * trade.lots * config.POINT_VALUE_PER_LOT, 2)


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
    print("  run_exp_s_historical_spread.py — Historical spread model")
    print("=" * 88)
    print(
        "  Estimated runtime: ~10–35 min (4 full runs + 6 fold×4 modes + analysis; "
        "historical path slightly heavier)."
    )

    summary_rows = []
    spread_full = load_spread_series()
    hist_kw_base = {
        **LIVE_PARITY_KWARGS,
        "spread_model": "historical",
        "spread_series": spread_full,
        "spread_base": 0.30,
    }

    bundle = run_phase("Load DataBundle.load_default()", lambda: DataBundle.load_default(start="2015-01-01"))
    if bundle is None:
        print("Cannot continue without data.")
        return

    spread_full_slice = slice_spread_to_m15(spread_full, bundle.m15_df) if spread_full is not None else None

    # --- Phase 1: full sample 4-way ---
    def p1():
        nonlocal summary_rows
        modes = [
            ("Fixed $0.00", {**LIVE_PARITY_KWARGS, "spread_cost": 0.0}),
            ("Fixed $0.30", {**LIVE_PARITY_KWARGS, "spread_cost": 0.30}),
            ("Historical", dict(hist_kw_base)),
            ("Fixed $0.50", {**LIVE_PARITY_KWARGS, "spread_cost": 0.50}),
        ]
        if spread_full is None:
            print("  WARNING: spread file missing — Historical row will match fallback behavior.")
        results = {}
        for label, kw in modes:
            st = run_variant(bundle, label, verbose=True, **kw)
            results[label] = st
            summary_rows.append(
                {
                    "phase": "1 Full",
                    "mode": label,
                    "sharpe": st["sharpe"],
                    "pnl": st["total_pnl"],
                    "n": st["n"],
                    "wr": st["win_rate"],
                    "max_dd": st["max_dd"],
                }
            )
        print(
            f"\n  {'Mode':<16} {'Sharpe':>8} {'PnL':>12} {'N':>6} {'WR%':>7} {'MaxDD':>10}"
        )
        print("  " + "-" * 62)
        for label in ["Fixed $0.00", "Fixed $0.30", "Historical", "Fixed $0.50"]:
            st = results[label]
            print(
                f"  {label:<16} {st['sharpe']:>8.3f} ${st['total_pnl']:>10.0f} "
                f"{st['n']:>6} {st['win_rate']:>6.1f}% ${st['max_dd']:>8.0f}"
            )
        return results

    full_results = run_phase(
        "1) Full sample: Fixed $0.00 / $0.30 / Historical / Fixed $0.50", p1
    )

    # --- Phase 2: K-Fold historical ---
    fold_stats = []

    def p2():
        nonlocal fold_stats
        folds = six_equal_time_folds(bundle)
        for fname, start_s, end_s in folds:
            sub = bundle.slice(start_s, end_s)
            if len(sub.m15_df) < 800:
                print(f"  {fname}: skip (M15={len(sub.m15_df)})")
                continue
            sp_sub = slice_spread_to_m15(spread_full, sub.m15_df) if spread_full is not None else None
            kw = {**hist_kw_base, "spread_series": sp_sub}
            st = run_variant(sub, f"{fname} HistSpread", verbose=False, **kw)
            sp_len = len(sp_sub) if sp_sub is not None else 0
            print(
                f"  {fname} M15={len(sub.m15_df):5d} spread_pts={sp_len:6d} "
                f"Sharpe={st['sharpe']:.3f} PnL=${st['total_pnl']:.0f} N={st['n']}"
            )
            fold_stats.append(st)
        return fold_stats

    run_phase("2) K-Fold (6 equal chunks) — Historical spread (sliced series per fold)", p2)

    # --- Phase 3: spread distribution at exit (full sample historical) ---
    def p3():
        if spread_full_slice is None or len(spread_full_slice) == 0:
            print("  No spread series — skip distribution.")
            return
        kw = dict(hist_kw_base)
        kw["spread_series"] = spread_full_slice
        st = run_variant(bundle, "hist_for_dist", verbose=False, **kw)
        trades = st["_trades"]
        charged = [spread_dollars_charged(t, spread_full_slice, 0.30) for t in trades]
        spr_units = [
            historical_spread_at_exit(t.exit_time, spread_full_slice, 0.30) for t in trades
        ]
        arr = np.array(spr_units, dtype=float)
        print(f"  Trades: {len(trades)}")
        print(
            f"  Spread (engine units at exit): mean={arr.mean():.4f} median={np.median(arr):.4f} "
            f"std={arr.std():.4f} min={arr.min():.4f} max={arr.max():.4f}"
        )
        pct = [5, 25, 50, 75, 95]
        qs = np.percentile(arr, pct)
        print("  Percentiles: " + ", ".join(f"p{p}={qs[i]:.4f}" for i, p in enumerate(pct)))
        cd = np.array(charged, dtype=float)
        print(
            f"  $ charged (spread×lots×PV): mean=${cd.mean():.2f} median=${np.median(cd):.2f} "
            f"sum=${cd.sum():.2f}"
        )

    run_phase("3) Historical spread at each trade exit — distribution (full sample)", p3)

    # --- Phase 4: session (exit hour UTC) ---
    def p4():
        if spread_full_slice is None or len(spread_full_slice) == 0:
            print("  No spread series — skip session table.")
            return
        kw = dict(hist_kw_base)
        kw["spread_series"] = spread_full_slice
        st = run_variant(bundle, "hist_session", verbose=False, **kw)
        trades = st["_trades"]
        buckets = {}
        for t in trades:
            h = pd.Timestamp(t.exit_time).hour
            buckets.setdefault(h, []).append(t)
        print(f"  {'UTC_h':>5} {'N':>5} {'avg_spread':>12} {'avg_pnl':>10}")
        print("  " + "-" * 38)
        for h in range(24):
            if h not in buckets:
                continue
            bt = buckets[h]
            avg_spr = np.mean(
                [historical_spread_at_exit(x.exit_time, spread_full_slice, 0.30) for x in bt]
            )
            avg_pnl = np.mean([x.pnl for x in bt])
            print(f"  {h:>5} {len(bt):>5} {avg_spr:>12.4f} {avg_pnl:>10.2f}")

    run_phase("4) Session analysis — by exit hour (UTC)", p4)

    # --- Phase 5: year PnL fixed vs historical ---
    def p5():
        st_fix = run_variant(bundle, "yfix", verbose=False, **{**LIVE_PARITY_KWARGS, "spread_cost": 0.30})
        st_hist = run_variant(
            bundle,
            "yhist",
            verbose=False,
            **{**hist_kw_base, "spread_series": spread_full_slice},
        )
        yf = st_fix.get("year_pnl") or {}
        yh = st_hist.get("year_pnl") or {}
        years = sorted(set(yf.keys()) | set(yh.keys()))
        print(f"  {'Year':<6} {'PnL_fix030':>14} {'PnL_hist':>14} {'Delta':>12}")
        print("  " + "-" * 50)
        for y in years:
            a = yf.get(y, 0.0)
            b = yh.get(y, 0.0)
            print(f"  {y:<6} ${a:>12.0f} ${b:>12.0f} ${b - a:>10.0f}")

    run_phase("5) Year-by-year PnL: Fixed $0.30 vs Historical", p5)

    # --- Summary ---
    print("\n" + "=" * 88)
    print("  SUMMARY TABLE")
    print("=" * 88)
    hdr = f"  {'Phase':<8} {'Mode':<18} {'Sharpe':>8} {'PnL':>12} {'N':>6} {'WR%':>7} {'MaxDD':>10}"
    print(hdr)
    print("  " + "-" * 78)
    for r in summary_rows:
        print(
            f"  {r['phase']:<8} {r['mode'][:18]:<18} {r['sharpe']:>8.3f} "
            f"${r['pnl']:>10.0f} {r['n']:>6} {r['wr']:>6.1f}% ${r['max_dd']:>8.0f}"
        )
    if fold_stats:
        shs = [x["sharpe"] for x in fold_stats]
        print(
            f"\n  K-Fold Historical: Sharpe mean={np.mean(shs):.3f} std={np.std(shs):.3f} "
            f"min={np.min(shs):.3f} max={np.max(shs):.3f}"
        )

    print(f"\n  Total script time: {time.time() - t_script:.1f}s\n")


if __name__ == "__main__":
    main()
