#!/usr/bin/env python3
"""
EXP-V: Breakout strength (Keltner) vs trade quality & tiered lot simulation (post-hoc).
Baseline: LIVE_PARITY_KWARGS + spread_cost=0.30.
"""
from __future__ import annotations

import io
import sys
import time
from dataclasses import replace
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

try:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
except Exception:
    pass

sys.path.insert(0, ".")

import research_config as config  # noqa: E402
from backtest.engine import BacktestEngine, TradeRecord  # noqa: E402
from backtest.runner import (  # noqa: E402
    DataBundle,
    LIVE_PARITY_KWARGS,
    calc_stats,
    run_kfold,
    run_variant,
)


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t


def h1_closed_bar_index(entry_time, h1_df: pd.DataFrame) -> int:
    """Last fully closed H1 bar at entry (engine closed_only alignment)."""
    et = _ts(entry_time)
    idx = int(h1_df.index.searchsorted(et, side="right")) - 2
    return idx


def keltner_breakout_strength(trade: TradeRecord, h1_df: pd.DataFrame) -> float:
    idx = h1_closed_bar_index(trade.entry_time, h1_df)
    if idx < 0 or idx >= len(h1_df):
        return float("nan")
    row = h1_df.iloc[idx]
    atr = float(row.get("ATR", np.nan))
    if not np.isfinite(atr) or atr <= 0:
        return float("nan")
    ku = float(row.get("KC_upper", np.nan))
    kl = float(row.get("KC_lower", np.nan))
    if not np.isfinite(ku) or not np.isfinite(kl):
        return float("nan")
    ep = float(trade.entry_price)
    if trade.direction == "BUY":
        return (ep - ku) / atr
    if trade.direction == "SELL":
        return (kl - ep) / atr
    return float("nan")


def build_equity_from_trades(trades: List[TradeRecord]) -> List[float]:
    ordered = sorted(trades, key=lambda t: t.exit_time)
    ec = [float(config.CAPITAL)]
    cum = 0.0
    for t in ordered:
        cum += t.pnl
        ec.append(float(config.CAPITAL + cum))
    return ec


def apply_keltner_lot_tiers(
    trades: List[TradeRecord],
    h1_df: pd.DataFrame,
    forward: bool = True,
) -> List[TradeRecord]:
    """
    forward=True: weak(33%) *0.5, mid *1.0, strong *1.5
    forward=False (inverse): weak *1.5, mid *1.0, strong *0.5
    """
    kel = [t for t in trades if t.strategy == "keltner"]
    strengths = np.array([keltner_breakout_strength(t, h1_df) for t in kel], dtype=float)
    valid = np.isfinite(strengths)
    if valid.sum() < 6:
        return list(trades)

    s_valid = strengths[valid]
    try:
        bins = pd.qcut(s_valid, q=3, labels=False, duplicates="drop")
    except Exception:
        return list(trades)

    # Map each keltner trade to tertile (0=low, …)
    tertile_by_id: Dict[int, int] = {}
    vi = 0
    for i, t in enumerate(kel):
        if not np.isfinite(strengths[i]):
            continue
        tertile_by_id[id(t)] = int(bins[vi])
        vi += 1

    if forward:
        mult_of_tert = {0: 0.5, 1: 1.0, 2: 1.5}
    else:
        mult_of_tert = {0: 1.5, 1: 1.0, 2: 0.5}

    out: List[TradeRecord] = []
    for t in trades:
        if t.strategy != "keltner":
            out.append(t)
            continue
        tid = id(t)
        if tid not in tertile_by_id:
            out.append(t)
            continue
        ter = tertile_by_id[tid]
        m = mult_of_tert.get(ter, 1.0)
        new_lots = round(t.lots * m, 4)
        new_pnl = round(t.pnl * m, 2)
        out.append(
            replace(
                t,
                lots=new_lots,
                pnl=new_pnl,
            )
        )
    return out


def print_quartile_table(df: pd.DataFrame, title: str) -> None:
    print(f"\n{'='*80}\n  {title}\n{'='*80}")
    if df.empty:
        print("  (empty)")
        return
    print(df.to_string(index=False))


def main() -> None:
    print("\n" + "=" * 80)
    print("  EXP-V: BREAKOUT STRENGTH & TIERED SIZING (Keltner, post-hoc)")
    print("=" * 80)

    t0 = time.time()
    try:
        bundle = DataBundle.load_default()
    except Exception as e:
        print(f"[FATAL] Data load failed: {e}")
        return

    base_kw = {**LIVE_PARITY_KWARGS, "spread_cost": 0.30}
    try:
        baseline = run_variant(bundle, "LIVE_PARITY sp$0.30", **base_kw)
    except Exception as e:
        print(f"[FATAL] Baseline backtest failed: {e}")
        return

    trades: List[TradeRecord] = baseline.get("_trades") or []
    kel = [t for t in trades if t.strategy == "keltner"]
    print(f"\n  Total trades: {len(trades)} | Keltner: {len(kel)}")

    strengths = np.array([keltner_breakout_strength(t, bundle.h1_df) for t in kel])
    valid_mask = np.isfinite(strengths)
    s_ok = strengths[valid_mask]
    kel_ok = [t for t, v in zip(kel, valid_mask) if v]

    if len(s_ok) < 8:
        print("[WARN] Too few Keltner trades with valid breakout_strength; aborting analysis.")
        return

    # --- Quartiles ---
    try:
        q_labels = pd.qcut(s_ok, q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
    except Exception as e:
        print(f"[WARN] Quartile split failed: {e}")
        q_labels = None

    rows = []
    if q_labels is not None:
        for q in q_labels.categories:
            m = q_labels == q
            sub = [kel_ok[i] for i in range(len(kel_ok)) if bool(m[i])]
            if not sub:
                continue
            pnls = [t.pnl for t in sub]
            wr = 100.0 * sum(1 for p in pnls if p > 0) / len(pnls)
            bh = [t.bars_held for t in sub]
            rows.append(
                {
                    "quartile": str(q),
                    "N": len(sub),
                    "avg_pnl": np.mean(pnls),
                    "win_rate_pct": wr,
                    "avg_bars_held": np.mean(bh),
                }
            )
    print_quartile_table(pd.DataFrame(rows), "Breakout strength quartiles vs PnL (Keltner)")

    # Spearman rank IC
    ic = float("nan")
    p_ic = float("nan")
    try:
        r = stats.spearmanr(s_ok, [t.pnl for t in kel_ok])
        ic = float(r.correlation) if r.correlation is not None else float("nan")
        p_ic = float(r.pvalue) if r.pvalue is not None else float("nan")
    except Exception as e:
        print(f"[WARN] Spearman IC failed: {e}")

    print(f"\n  Spearman rank IC (breakout_strength vs PnL): {ic:.4f}  (p={p_ic:.4g})")

    # --- Tiered Sharpe ---
    ec0 = baseline.get("_equity_curve") or build_equity_from_trades(trades)
    st0 = calc_stats(trades, ec0)

    fwd_tr = apply_keltner_lot_tiers(trades, bundle.h1_df, forward=True)
    inv_tr = apply_keltner_lot_tiers(trades, bundle.h1_df, forward=False)
    st_fwd = calc_stats(fwd_tr, build_equity_from_trades(fwd_tr))
    st_inv = calc_stats(inv_tr, build_equity_from_trades(inv_tr))

    cmp_df = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "N": st0["n"],
                "Sharpe": st0["sharpe"],
                "total_pnl": st0["total_pnl"],
            },
            {
                "variant": "tier_fwd_weak0.5_strong1.5",
                "N": st_fwd["n"],
                "Sharpe": st_fwd["sharpe"],
                "total_pnl": st_fwd["total_pnl"],
            },
            {
                "variant": "tier_INV_control",
                "N": st_inv["n"],
                "Sharpe": st_inv["sharpe"],
                "total_pnl": st_inv["total_pnl"],
            },
        ]
    )
    print_quartile_table(cmp_df, "Tiered lot simulation (Keltner only scaled) vs baseline")

    # ADX vs breakout_strength (H1 bar at entry)
    adx_list = []
    for t in kel_ok:
        idx = h1_closed_bar_index(t.entry_time, bundle.h1_df)
        if 0 <= idx < len(bundle.h1_df):
            adx_list.append(float(bundle.h1_df.iloc[idx].get("ADX", np.nan)))
        else:
            adx_list.append(np.nan)
    adx_arr = np.array(adx_list, dtype=float)
    mask2 = np.isfinite(adx_arr) & np.isfinite(s_ok)
    rho = float("nan")
    try:
        if mask2.sum() > 5:
            rho = float(np.corrcoef(s_ok[mask2], adx_arr[mask2])[0, 1])
    except Exception:
        pass
    print(f"\n  Pearson corr(ADX@H1_entry, breakout_strength): {rho:.4f}  (n={int(mask2.sum())})")

    # --- K-Fold if forward tier improves Sharpe ---
    improved = st_fwd["sharpe"] > st0["sharpe"]
    print(
        f"\n  Tiered forward Sharpe {'>' if improved else '<='} baseline "
        f"({st_fwd['sharpe']:.3f} vs {st0['sharpe']:.3f}) → "
        f"{'RUN' if improved else 'SKIP'} 6-fold K-Fold"
    )

    if improved:
        try:
            fold_stats = run_kfold(
                bundle, base_kw, n_folds=6, label_prefix="EXP-V-baseline-"
            )
        except Exception as e:
            print(f"[WARN] K-Fold baseline failed: {e}")
            fold_stats = []

        fold_rows = []
        for fs in fold_stats:
            tr_f = fs.get("_trades") or []
            if not tr_f:
                continue
            b_sh = fs.get("sharpe", 0.0)
            fwd_f = apply_keltner_lot_tiers(tr_f, bundle.h1_df, forward=True)
            inv_f = apply_keltner_lot_tiers(tr_f, bundle.h1_df, forward=False)
            sh_fwd = calc_stats(fwd_f, build_equity_from_trades(fwd_f))["sharpe"]
            sh_inv = calc_stats(inv_f, build_equity_from_trades(inv_f))["sharpe"]
            fold_rows.append(
                {
                    "fold": fs.get("fold", ""),
                    "baseline_Sharpe": b_sh,
                    "tier_fwd_Sharpe": sh_fwd,
                    "tier_inv_Sharpe": sh_inv,
                    "dSharpe_fwd": sh_fwd - b_sh,
                }
            )
        print_quartile_table(pd.DataFrame(fold_rows), "6-fold time-series: tiered vs baseline (per fold)")

        if fold_rows:
            mean_b = np.mean([r["baseline_Sharpe"] for r in fold_rows])
            mean_f = np.mean([r["tier_fwd_Sharpe"] for r in fold_rows])
            print(f"\n  Mean Sharpe across folds: baseline={mean_b:.3f}  tier_fwd={mean_f:.3f}")

    # --- Summary ---
    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print("  SUMMARY (EXP-V)")
    print("=" * 80)
    print(
        f"  Keltner N={len(kel)}, valid breakout N={len(s_ok)}, "
        f"Spearman IC={ic:.4f}, ADX–strength ρ={rho:.4f}"
    )
    print(
        f"  Baseline Sharpe={st0['sharpe']:.3f} | "
        f"Tier fwd={st_fwd['sharpe']:.3f} | Tier inv={st_inv['sharpe']:.3f}"
    )
    print(f"  Elapsed {elapsed:.1f}s")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
