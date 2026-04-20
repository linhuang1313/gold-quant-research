#!/usr/bin/env python3
"""
EXP-CE-V2: Consecutive Entry — Validation Backtest
===================================================
Based on EXP-CE findings, test two actionable ideas:
  A) Sequence-late position sizing: reduce lot size for pos3+ in Keltner sequences
  B) RSI2 auxiliary filter: reduce lot size when RSI2 is not extreme enough at entry
  C) Combined A+B

Method:
  1. Run baseline backtest → collect all trades
  2. Build Keltner consecutive sequences (same as EXP-CE)
  3. For each variant, adjust PnL of affected trades by a lot_ratio factor
  4. Rebuild equity curve and compute stats for comparison
"""
from __future__ import annotations
import io
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
except Exception:
    pass

sys.path.insert(0, ".")
from backtest.engine import BacktestEngine, TradeRecord
from backtest.runner import DataBundle, LIVE_PARITY_KWARGS, run_variant

GAP_HOURS = 24
SPREAD_COST = 0.30


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t


def h1_closed_bar_index(entry_time, h1_df: pd.DataFrame) -> int:
    et = _ts(entry_time)
    return int(h1_df.index.searchsorted(et, side="right")) - 2


def build_keltner_sequences(
    trades: List[TradeRecord], gap_hours: float = GAP_HOURS
) -> Tuple[List[List[int]], Dict[int, Tuple[int, int, int]]]:
    """
    Build Keltner consecutive same-direction sequences.
    Returns:
      - sequences: list of lists of trade indices (into the trades list)
      - trade_info: dict mapping trade_index -> (seq_idx, pos_in_seq_1based, seq_len)
    """
    kc_indices = [i for i, t in enumerate(trades) if t.strategy == "keltner"]
    if not kc_indices:
        return [], {}

    sequences: List[List[int]] = []
    cur_seq: List[int] = [kc_indices[0]]

    for idx in kc_indices[1:]:
        prev_idx = cur_seq[-1]
        prev_t = trades[prev_idx]
        cur_t = trades[idx]
        gap = (_ts(cur_t.entry_time) - _ts(prev_t.entry_time)).total_seconds() / 3600
        if cur_t.direction == prev_t.direction and gap <= gap_hours:
            cur_seq.append(idx)
        else:
            sequences.append(cur_seq)
            cur_seq = [idx]
    sequences.append(cur_seq)

    trade_info: Dict[int, Tuple[int, int, int]] = {}
    for seq_idx, seq in enumerate(sequences):
        seq_len = len(seq)
        for pos_0, trade_idx in enumerate(seq):
            trade_info[trade_idx] = (seq_idx, pos_0 + 1, seq_len)

    return sequences, trade_info


def get_rsi2_at_entry(trade: TradeRecord, h1_df: pd.DataFrame) -> float:
    hi = h1_closed_bar_index(trade.entry_time, h1_df)
    if 0 <= hi < len(h1_df):
        return float(h1_df.iloc[hi].get("RSI2", np.nan))
    return np.nan


def apply_variant(
    trades: List[TradeRecord],
    trade_info: Dict[int, Tuple[int, int, int]],
    h1_df: pd.DataFrame,
    variant: str,
    params: Dict[str, Any],
) -> List[TradeRecord]:
    """
    Return a modified copy of trades with PnL adjusted based on variant logic.
    Only Keltner trades in sequences are affected.
    """
    new_trades = deepcopy(trades)

    for i, t in enumerate(new_trades):
        if i not in trade_info:
            continue

        seq_idx, pos, seq_len = trade_info[i]
        lot_mult = 1.0

        if variant == "seq_reduce":
            if pos >= params.get("reduce_from_pos", 3):
                lot_mult = params.get("lot_mult", 0.8)

        elif variant == "rsi2_filter":
            rsi2 = get_rsi2_at_entry(t, h1_df)
            threshold = params.get("rsi2_threshold", 75)
            if np.isfinite(rsi2) and pos >= 2:
                if t.direction == "BUY" and rsi2 > threshold:
                    lot_mult = params.get("lot_mult", 0.5)
                elif t.direction == "SELL" and rsi2 < (100 - threshold):
                    lot_mult = params.get("lot_mult", 0.5)

        elif variant == "combined":
            seq_mult = 1.0
            if pos >= params.get("reduce_from_pos", 3):
                seq_mult = params.get("seq_lot_mult", 0.8)

            rsi_mult = 1.0
            rsi2 = get_rsi2_at_entry(t, h1_df)
            threshold = params.get("rsi2_threshold", 75)
            if np.isfinite(rsi2) and pos >= 2:
                if t.direction == "BUY" and rsi2 > threshold:
                    rsi_mult = params.get("rsi_lot_mult", 0.6)
                elif t.direction == "SELL" and rsi2 < (100 - threshold):
                    rsi_mult = params.get("rsi_lot_mult", 0.6)

            lot_mult = seq_mult * rsi_mult

        elif variant == "seq_skip":
            if pos >= params.get("skip_from_pos", 4):
                lot_mult = 0.0

        if lot_mult != 1.0:
            t.pnl = t.pnl * lot_mult
            t.lots = round(t.lots * lot_mult, 2)

    return new_trades


def compute_stats(trades: List[TradeRecord], initial_capital: float = 10000.0) -> Dict:
    """Compute key stats from a trade list."""
    active_trades = [t for t in trades if t.lots > 0]
    if not active_trades:
        return {"n": 0, "total_pnl": 0, "sharpe": 0, "win_rate": 0,
                "max_dd": 0, "max_dd_pct": 0, "avg_pnl": 0, "rr": 0}

    pnls = [t.pnl for t in active_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    n = len(pnls)
    win_rate = len(wins) / n * 100
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    rr = avg_win / avg_loss if avg_loss > 0 else 0
    avg_pnl = total_pnl / n

    daily = defaultdict(float)
    for t in active_trades:
        day = pd.Timestamp(t.exit_time).strftime("%Y-%m-%d")
        daily[day] += t.pnl
    daily_vals = list(daily.values())
    if len(daily_vals) > 1 and np.std(daily_vals, ddof=1) > 0:
        sharpe = np.mean(daily_vals) / np.std(daily_vals, ddof=1) * np.sqrt(252)
    else:
        sharpe = 0.0

    eq = [initial_capital]
    for p in pnls:
        eq.append(eq[-1] + p)
    eq = np.array(eq)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_dd = abs(dd.min())
    max_dd_pct = max_dd / peak[np.argmin(dd)] * 100 if peak[np.argmin(dd)] > 0 else 0

    kc_trades = [t for t in active_trades if t.strategy == "keltner"]
    kc_pnl = sum(t.pnl for t in kc_trades)
    kc_n = len(kc_trades)

    return {
        "n": n, "total_pnl": total_pnl, "sharpe": sharpe,
        "win_rate": win_rate, "max_dd": max_dd, "max_dd_pct": max_dd_pct,
        "avg_pnl": avg_pnl, "rr": rr,
        "kc_n": kc_n, "kc_pnl": kc_pnl,
    }


def year_stats(trades: List[TradeRecord]) -> Dict[int, Dict]:
    by_year: Dict[int, List[TradeRecord]] = defaultdict(list)
    for t in trades:
        yr = pd.Timestamp(t.exit_time).year
        by_year[yr].append(t)
    result = {}
    for yr in sorted(by_year):
        result[yr] = compute_stats(by_year[yr])
    return result


def print_section(title: str):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print("=" * 80)


def print_comparison(label: str, stats: Dict, baseline: Dict):
    delta_pnl = stats["total_pnl"] - baseline["total_pnl"]
    delta_dd = stats["max_dd"] - baseline["max_dd"]
    delta_sharpe = stats["sharpe"] - baseline["sharpe"]
    print(f"  {label}")
    print(f"    N={stats['n']}  PnL=${stats['total_pnl']:.0f} ({delta_pnl:+.0f})"
          f"  Sharpe={stats['sharpe']:.3f} ({delta_sharpe:+.3f})"
          f"  WR={stats['win_rate']:.1f}%"
          f"  MaxDD=${stats['max_dd']:.0f} ({delta_dd:+.0f})"
          f"  RR={stats['rr']:.2f}"
          f"  KC: N={stats['kc_n']} PnL=${stats['kc_pnl']:.0f}")


def main():
    print("\n" + "=" * 80)
    print("  EXP-CE-V2: CONSECUTIVE ENTRY VALIDATION BACKTEST")
    print("  Testing: Sequence-late reduction, RSI2 filter, Combined")
    print("=" * 80)
    t0 = time.time()

    bundle = DataBundle.load_default()
    base_kw = {**LIVE_PARITY_KWARGS, "spread_cost": SPREAD_COST}
    baseline_result = run_variant(bundle, "BASELINE", **base_kw)
    all_trades: List[TradeRecord] = baseline_result.get("_trades") or []
    print(f"\n  Baseline trades: {len(all_trades)}")

    sequences, trade_info = build_keltner_sequences(all_trades)
    kc_in_seq = sum(1 for v in trade_info.values() if v[2] >= 2)
    kc_pos3p = sum(1 for v in trade_info.values() if v[1] >= 3)
    print(f"  Keltner sequences: {len(sequences)}")
    print(f"  KC trades in multi-trade sequences: {kc_in_seq}")
    print(f"  KC trades at pos3+: {kc_pos3p}")

    baseline_stats = compute_stats(all_trades)

    # ── VARIANT A: Sequence-late position reduction ──
    print_section("VARIANT A: SEQUENCE-LATE LOT REDUCTION")
    print(f"  Baseline: N={baseline_stats['n']}  PnL=${baseline_stats['total_pnl']:.0f}"
          f"  Sharpe={baseline_stats['sharpe']:.3f}  WR={baseline_stats['win_rate']:.1f}%"
          f"  MaxDD=${baseline_stats['max_dd']:.0f}"
          f"  KC: N={baseline_stats['kc_n']} PnL=${baseline_stats['kc_pnl']:.0f}")

    a_variants = [
        ("A1: pos3+ x0.80", {"reduce_from_pos": 3, "lot_mult": 0.80}),
        ("A2: pos3+ x0.60", {"reduce_from_pos": 3, "lot_mult": 0.60}),
        ("A3: pos3+ x0.50", {"reduce_from_pos": 3, "lot_mult": 0.50}),
        ("A4: pos4+ x0.80", {"reduce_from_pos": 4, "lot_mult": 0.80}),
        ("A5: pos4+ x0.50", {"reduce_from_pos": 4, "lot_mult": 0.50}),
        ("A6: pos2+ x0.80", {"reduce_from_pos": 2, "lot_mult": 0.80}),
    ]
    for label, params in a_variants:
        modified = apply_variant(all_trades, trade_info, bundle.h1_df, "seq_reduce", params)
        stats = compute_stats(modified)
        print_comparison(label, stats, baseline_stats)

    # ── VARIANT B: RSI2 auxiliary filter ──
    print_section("VARIANT B: RSI2 AUXILIARY FILTER ON SEQUENCE TRADES")

    b_variants = [
        ("B1: RSI2>75 pos2+ x0.50", {"rsi2_threshold": 75, "lot_mult": 0.50}),
        ("B2: RSI2>80 pos2+ x0.50", {"rsi2_threshold": 80, "lot_mult": 0.50}),
        ("B3: RSI2>70 pos2+ x0.50", {"rsi2_threshold": 70, "lot_mult": 0.50}),
        ("B4: RSI2>75 pos2+ x0.30", {"rsi2_threshold": 75, "lot_mult": 0.30}),
        ("B5: RSI2>80 pos2+ x0.00 (skip)", {"rsi2_threshold": 80, "lot_mult": 0.00}),
    ]
    for label, params in b_variants:
        modified = apply_variant(all_trades, trade_info, bundle.h1_df, "rsi2_filter", params)
        stats = compute_stats(modified)
        print_comparison(label, stats, baseline_stats)

    # ── VARIANT C: Combined ──
    print_section("VARIANT C: COMBINED (sequence reduce + RSI2 filter)")

    c_variants = [
        ("C1: pos3+ x0.80 + RSI2>75 x0.60", {
            "reduce_from_pos": 3, "seq_lot_mult": 0.80,
            "rsi2_threshold": 75, "rsi_lot_mult": 0.60,
        }),
        ("C2: pos3+ x0.60 + RSI2>80 x0.50", {
            "reduce_from_pos": 3, "seq_lot_mult": 0.60,
            "rsi2_threshold": 80, "rsi_lot_mult": 0.50,
        }),
        ("C3: pos3+ x0.80 + RSI2>70 x0.50", {
            "reduce_from_pos": 3, "seq_lot_mult": 0.80,
            "rsi2_threshold": 70, "rsi_lot_mult": 0.50,
        }),
        ("C4: pos4+ x0.80 + RSI2>75 x0.50", {
            "reduce_from_pos": 4, "seq_lot_mult": 0.80,
            "rsi2_threshold": 75, "rsi_lot_mult": 0.50,
        }),
    ]
    for label, params in c_variants:
        modified = apply_variant(all_trades, trade_info, bundle.h1_df, "combined", params)
        stats = compute_stats(modified)
        print_comparison(label, stats, baseline_stats)

    # ── VARIANT D: Hard skip (for reference) ──
    print_section("VARIANT D: HARD SKIP (reference only — NOT recommended)")
    d_variants = [
        ("D1: skip pos4+", {"skip_from_pos": 4}),
        ("D2: skip pos5+", {"skip_from_pos": 5}),
        ("D3: skip pos3+", {"skip_from_pos": 3}),
    ]
    for label, params in d_variants:
        modified = apply_variant(all_trades, trade_info, bundle.h1_df, "seq_skip", params)
        stats = compute_stats(modified)
        print_comparison(label, stats, baseline_stats)

    # ── YEAR-BY-YEAR for top candidates ──
    print_section("YEAR-BY-YEAR COMPARISON: BASELINE vs TOP CANDIDATES")

    best_candidates = [
        ("Baseline", all_trades),
        ("A2: pos3+ x0.60",
         apply_variant(all_trades, trade_info, bundle.h1_df, "seq_reduce",
                       {"reduce_from_pos": 3, "lot_mult": 0.60})),
        ("B1: RSI2>75 pos2+ x0.50",
         apply_variant(all_trades, trade_info, bundle.h1_df, "rsi2_filter",
                       {"rsi2_threshold": 75, "lot_mult": 0.50})),
        ("C1: combined",
         apply_variant(all_trades, trade_info, bundle.h1_df, "combined",
                       {"reduce_from_pos": 3, "seq_lot_mult": 0.80,
                        "rsi2_threshold": 75, "rsi_lot_mult": 0.60})),
    ]

    headers = [f"{'Year':>6}"]
    for label, _ in best_candidates:
        headers.append(f"  {label[:20]:>20}")
    print("  " + " | ".join(headers))
    print("  " + "-" * (len(" | ".join(headers)) + 2))

    all_years = set()
    yr_data = {}
    for label, trades_list in best_candidates:
        ys = year_stats(trades_list)
        yr_data[label] = ys
        all_years |= set(ys.keys())

    for yr in sorted(all_years):
        parts = [f"{yr:>6}"]
        for label, _ in best_candidates:
            ys = yr_data[label]
            if yr in ys:
                s = ys[yr]
                parts.append(f"  ${s['total_pnl']:>7.0f} Sh={s['sharpe']:.2f}")
            else:
                parts.append(f"  {'N/A':>20}")
        print("  " + " | ".join(parts))

    # ── Affected trade count summary ──
    print_section("AFFECTED TRADE SUMMARY")
    for label, params in [("seq_reduce pos3+ x0.60", {"reduce_from_pos": 3, "lot_mult": 0.60}),
                          ("rsi2_filter 75 x0.50", {"rsi2_threshold": 75, "lot_mult": 0.50})]:
        variant_type = label.split()[0]
        modified = apply_variant(all_trades, trade_info, bundle.h1_df, variant_type, params)
        affected = sum(1 for i, (o, m) in enumerate(zip(all_trades, modified)) if abs(o.pnl - m.pnl) > 0.001)
        saved = sum(o.pnl - m.pnl for o, m in zip(all_trades, modified) if o.pnl < 0 and abs(o.pnl - m.pnl) > 0.001)
        lost = sum(o.pnl - m.pnl for o, m in zip(all_trades, modified) if o.pnl > 0 and abs(o.pnl - m.pnl) > 0.001)
        print(f"  {label}: affected={affected} trades")
        print(f"    Loss reduction (saved): ${-saved:.0f}")
        print(f"    Profit reduction (lost): ${lost:.0f}")
        print(f"    Net impact: ${-(saved+lost):.0f}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
