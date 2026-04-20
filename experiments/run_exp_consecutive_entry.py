#!/usr/bin/env python3
"""
EXP-CE: Consecutive Entry Decay Analysis
=========================================
Hypothesis: When Keltner strategy opens >=N trades in the same direction
within a short window, later trades in the sequence have lower win-rate
and worse average PnL (trend exhaustion effect).
Observed in live trading (2 samples only — this experiment quantifies
the effect across the full backtest history).
"""
from __future__ import annotations
import io
import sys
import time
from collections import defaultdict
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
from backtest.engine import BacktestEngine, TradeRecord  # noqa: E402
from backtest.runner import DataBundle, LIVE_PARITY_KWARGS, calc_stats, run_variant  # noqa: E402
# ── Configuration ─────────────────────────────────────────────
GAP_HOURS = 24          # max hours between consecutive same-dir entries to count as a "sequence"
MIN_SEQ_LEN = 2         # minimum sequence length to analyze
SPREAD_COST = 0.30      # match live spread assumption
# ──────────────────────────────────────────────────────────────
def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t
def h1_closed_bar_index(entry_time, h1_df: pd.DataFrame) -> int:
    et = _ts(entry_time)
    return int(h1_df.index.searchsorted(et, side="right")) - 2
def build_consecutive_sequences(
    trades: List[TradeRecord],
    strategy_filter: Optional[str] = None,
    gap_hours: float = GAP_HOURS,
) -> List[List[TradeRecord]]:
    """
    Group trades into consecutive same-direction sequences.
    A new sequence starts when:
      - direction changes, OR
      - gap between current entry and previous entry > gap_hours
    """
    filtered = [t for t in trades if strategy_filter is None or t.strategy == strategy_filter]
    filtered.sort(key=lambda t: t.entry_time)
    if not filtered:
        return []
    sequences: List[List[TradeRecord]] = []
    current_seq: List[TradeRecord] = [filtered[0]]
    for t in filtered[1:]:
        prev = current_seq[-1]
        hours_gap = (_ts(t.entry_time) - _ts(prev.entry_time)).total_seconds() / 3600
        if t.direction == prev.direction and hours_gap <= gap_hours:
            current_seq.append(t)
        else:
            sequences.append(current_seq)
            current_seq = [t]
    sequences.append(current_seq)
    return sequences
def extract_entry_features(
    trade: TradeRecord, h1_df: pd.DataFrame, m15_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Extract key features at entry time for context."""
    et = _ts(trade.entry_time)
    hi = h1_closed_bar_index(trade.entry_time, h1_df)
    out: Dict[str, Any] = {}
    if 0 <= hi < len(h1_df):
        hr = h1_df.iloc[hi]
        out["h1_RSI2"] = float(hr.get("RSI2", np.nan))
        out["h1_RSI14"] = float(hr.get("RSI14", np.nan))
        out["h1_ADX"] = float(hr.get("ADX", np.nan))
        out["h1_ATR"] = float(hr.get("ATR", np.nan))
        out["h1_atr_percentile"] = float(hr.get("atr_percentile", np.nan))
        ku = float(hr.get("KC_upper", np.nan))
        kl = float(hr.get("KC_lower", np.nan))
        c = float(hr["Close"])
        if np.isfinite(ku) and np.isfinite(kl) and abs(ku - kl) > 1e-9:
            out["kc_position"] = (c - kl) / (ku - kl)
        else:
            out["kc_position"] = np.nan
    else:
        for k in ["h1_RSI2", "h1_RSI14", "h1_ADX", "h1_ATR", "h1_atr_percentile", "kc_position"]:
            out[k] = np.nan
    out["entry_hour_utc"] = int(et.hour)
    return out
def analyze_position_in_sequence(
    sequences: List[List[TradeRecord]],
    h1_df: pd.DataFrame,
    m15_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each trade, record its position within the sequence (1st, 2nd, 3rd, ...)
    along with PnL and entry features.
    """
    rows: List[Dict[str, Any]] = []
    for seq in sequences:
        seq_len = len(seq)
        for pos_idx, trade in enumerate(seq):
            row: Dict[str, Any] = {
                "seq_len": seq_len,
                "pos_in_seq": pos_idx + 1,   # 1-based
                "is_last_in_seq": pos_idx == seq_len - 1,
                "direction": trade.direction,
                "strategy": trade.strategy,
                "entry_time": _ts(trade.entry_time),
                "exit_time": _ts(trade.exit_time),
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "pnl": trade.pnl,
                "exit_reason": trade.exit_reason,
                "bars_held": trade.bars_held,
                "is_win": trade.pnl > 0,
            }
            row.update(extract_entry_features(trade, h1_df, m15_df))
            rows.append(row)
    return pd.DataFrame(rows)
def print_section(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print("=" * 80)
def main() -> None:
    print("\n" + "=" * 80)
    print("  EXP-CE: CONSECUTIVE ENTRY DECAY ANALYSIS")
    print("  Hypothesis: Later trades in same-direction sequences perform worse")
    print("=" * 80)
    t0 = time.time()
    # ── Load data & run baseline ──
    try:
        bundle = DataBundle.load_default()
    except Exception as e:
        print(f"[FATAL] Data load failed: {e}")
        return
    base_kw = {**LIVE_PARITY_KWARGS, "spread_cost": SPREAD_COST}
    try:
        baseline = run_variant(bundle, "LIVE_PARITY sp$0.30", **base_kw)
    except Exception as e:
        print(f"[FATAL] Baseline backtest failed: {e}")
        return
    all_trades: List[TradeRecord] = baseline.get("_trades") or []
    st = calc_stats(all_trades, baseline.get("_equity_curve") or [])
    print(f"\n  Baseline: N={st['n']}  Sharpe={st['sharpe']:.3f}  WR={st['win_rate']:.1f}%  PnL=${st['total_pnl']:.0f}")
    # ── Part 1: Keltner-only analysis (primary hypothesis) ──
    print_section("PART 1: KELTNER CONSECUTIVE SAME-DIRECTION SEQUENCES")
    kc_sequences = build_consecutive_sequences(all_trades, strategy_filter="keltner", gap_hours=GAP_HOURS)
    kc_df = analyze_position_in_sequence(kc_sequences, bundle.h1_df, bundle.m15_df)
    print(f"\n  Total Keltner trades: {len(kc_df)}")
    print(f"  Total sequences: {len(kc_sequences)}")
    print(f"  Sequence length distribution:")
    seq_lens = pd.Series([len(s) for s in kc_sequences])
    for length, count in seq_lens.value_counts().sort_index().items():
        pct = 100.0 * count / len(kc_sequences)
        print(f"    len={length}: {count} sequences ({pct:.1f}%)")
    # ── 1A: Win-rate & avg PnL by position in sequence ──
    print_section("1A: PERFORMANCE BY POSITION IN SEQUENCE (all seq lengths)")
    for pos in sorted(kc_df["pos_in_seq"].unique()):
        subset = kc_df[kc_df["pos_in_seq"] == pos]
        n = len(subset)
        wr = 100.0 * subset["is_win"].mean()
        avg_pnl = subset["pnl"].mean()
        med_pnl = subset["pnl"].median()
        total = subset["pnl"].sum()
        print(f"    Position #{pos}: N={n}  WR={wr:.1f}%  avg_pnl=${avg_pnl:.2f}  median=${med_pnl:.2f}  total=${total:.0f}")
    # ── 1B: Performance of "last trade in sequence" vs others ──
    print_section("1B: LAST TRADE IN SEQUENCE vs OTHERS (seq_len >= 2)")
    multi = kc_df[kc_df["seq_len"] >= MIN_SEQ_LEN].copy()
    if len(multi) > 0:
        last = multi[multi["is_last_in_seq"]]
        not_last = multi[~multi["is_last_in_seq"]]
        print(f"    Last trades:     N={len(last)}  WR={100*last['is_win'].mean():.1f}%  avg_pnl=${last['pnl'].mean():.2f}")
        print(f"    Non-last trades: N={len(not_last)}  WR={100*not_last['is_win'].mean():.1f}%  avg_pnl=${not_last['pnl'].mean():.2f}")
        # statistical test
        if len(last) >= 5 and len(not_last) >= 5:
            from scipy import stats as sp_stats
            tt = sp_stats.ttest_ind(last["pnl"].values, not_last["pnl"].values, equal_var=False)
            mw = sp_stats.mannwhitneyu(last["pnl"].values, not_last["pnl"].values, alternative="two-sided")
            print(f"    t-test p={tt.pvalue:.4f}  Mann-Whitney p={mw.pvalue:.4f}")
    # ── 1C: Focus on sequences of length >= 3 ──
    print_section("1C: SEQUENCES OF LENGTH >= 3 (detailed)")
    long_seqs = [s for s in kc_sequences if len(s) >= 3]
    print(f"    Found {len(long_seqs)} sequences with length >= 3")
    if long_seqs:
        long_df = kc_df[kc_df["seq_len"] >= 3].copy()
        for pos in sorted(long_df["pos_in_seq"].unique()):
            sub = long_df[long_df["pos_in_seq"] == pos]
            wr = 100.0 * sub["is_win"].mean()
            avg = sub["pnl"].mean()
            print(f"      Pos #{pos}: N={sub.shape[0]}  WR={wr:.1f}%  avg_pnl=${avg:.2f}")
        # Compare pos >= 3 vs pos <= 2
        early = long_df[long_df["pos_in_seq"] <= 2]
        late = long_df[long_df["pos_in_seq"] >= 3]
        if len(early) > 0 and len(late) > 0:
            print(f"\n    Early (pos 1-2): N={len(early)}  WR={100*early['is_win'].mean():.1f}%  avg=${early['pnl'].mean():.2f}")
            print(f"    Late  (pos 3+):  N={len(late)}  WR={100*late['is_win'].mean():.1f}%  avg=${late['pnl'].mean():.2f}")
    # ── 1D: RSI2 extreme at entry for late positions ──
    print_section("1D: RSI2 AT ENTRY — EARLY vs LATE POSITIONS")
    if len(kc_df) > 0:
        kc_with_rsi = kc_df[kc_df["h1_RSI2"].notna()].copy()
        for pos in sorted(kc_with_rsi["pos_in_seq"].unique()):
            sub = kc_with_rsi[kc_with_rsi["pos_in_seq"] == pos]
            buy_sub = sub[sub["direction"] == "BUY"]
            sell_sub = sub[sub["direction"] == "SELL"]
            print(f"    Pos #{pos}:")
            if len(buy_sub) > 0:
                print(f"      BUY  RSI2: mean={buy_sub['h1_RSI2'].mean():.1f}  median={buy_sub['h1_RSI2'].median():.1f}  >80: {(buy_sub['h1_RSI2']>80).sum()}/{len(buy_sub)}")
            if len(sell_sub) > 0:
                print(f"      SELL RSI2: mean={sell_sub['h1_RSI2'].mean():.1f}  median={sell_sub['h1_RSI2'].median():.1f}  <20: {(sell_sub['h1_RSI2']<20).sum()}/{len(sell_sub)}")
    # ── 1E: Exit reason distribution by position ──
    print_section("1E: EXIT REASON BY POSITION IN SEQUENCE")
    if len(kc_df) > 0:
        for pos in sorted(kc_df["pos_in_seq"].unique()):
            sub = kc_df[kc_df["pos_in_seq"] == pos]
            if len(sub) == 0:
                continue
            vc = sub["exit_reason"].value_counts()
            top3 = vc.head(3)
            reasons_str = ", ".join(f"{r}:{c}" for r, c in top3.items())
            print(f"    Pos #{pos} (N={len(sub)}): {reasons_str}")
    # ── Part 2: All strategies combined ──
    print_section("PART 2: ALL STRATEGIES — SAME-DIRECTION SEQUENCES")
    all_sequences = build_consecutive_sequences(all_trades, strategy_filter=None, gap_hours=GAP_HOURS)
    all_df = analyze_position_in_sequence(all_sequences, bundle.h1_df, bundle.m15_df)
    print(f"  Total trades: {len(all_df)}  Total sequences: {len(all_sequences)}")
    for pos in sorted(all_df["pos_in_seq"].unique()):
        sub = all_df[all_df["pos_in_seq"] == pos]
        wr = 100.0 * sub["is_win"].mean()
        avg = sub["pnl"].mean()
        print(f"    Pos #{pos}: N={len(sub)}  WR={wr:.1f}%  avg_pnl=${avg:.2f}")
    # ── Part 3: Sensitivity to GAP_HOURS ──
    print_section("PART 3: SENSITIVITY — VARYING GAP_HOURS (Keltner only)")
    for gap in [8, 12, 24, 48]:
        seqs = build_consecutive_sequences(all_trades, strategy_filter="keltner", gap_hours=gap)
        df = analyze_position_in_sequence(seqs, bundle.h1_df, bundle.m15_df)
        pos1 = df[df["pos_in_seq"] == 1]
        pos3p = df[df["pos_in_seq"] >= 3]
        n3p = len(pos3p)
        wr3p = 100.0 * pos3p["is_win"].mean() if n3p > 0 else float("nan")
        avg3p = pos3p["pnl"].mean() if n3p > 0 else float("nan")
        print(
            f"    gap={gap:2d}h: seqs={len(seqs):4d}  "
            f"pos1: N={len(pos1)} WR={100*pos1['is_win'].mean():.1f}%  |  "
            f"pos3+: N={n3p} WR={wr3p:.1f}% avg=${avg3p:.2f}"
        )
    # ── Part 4: Year-by-year stability ──
    print_section("PART 4: YEAR-BY-YEAR STABILITY (Keltner, pos>=3 vs pos==1)")
    if "entry_time" in kc_df.columns and len(kc_df) > 0:
        kc_df["year"] = kc_df["entry_time"].dt.year
        for yr in sorted(kc_df["year"].unique()):
            yr_df = kc_df[kc_df["year"] == yr]
            p1 = yr_df[yr_df["pos_in_seq"] == 1]
            p3 = yr_df[yr_df["pos_in_seq"] >= 3]
            if len(p1) == 0:
                continue
            wr1 = 100.0 * p1["is_win"].mean()
            avg1 = p1["pnl"].mean()
            line = f"    {yr}: pos1 N={len(p1):3d} WR={wr1:.1f}% avg=${avg1:.2f}"
            if len(p3) > 0:
                wr3 = 100.0 * p3["is_win"].mean()
                avg3 = p3["pnl"].mean()
                line += f"  |  pos3+ N={len(p3):3d} WR={wr3:.1f}% avg=${avg3:.2f}"
            else:
                line += "  |  pos3+ N=  0"
            print(line)
    # ── Summary ──
    print_section("CONCLUSION TEMPLATE")
    print("""
    Fill in after reviewing results:
    1. Decay effect exists?      YES / NO / MARGINAL
    2. Statistical significance? p = ???
    3. Sample size adequate?     pos3+ N = ???  (need >= 30 for confidence)
    4. Year-over-year stable?    YES / NO
    5. RSI2 extreme confirms?    YES / NO
    If YES to 1-4:
      -> Design a "consecutive entry counter" filter
      -> Backtest variants: skip after N=3 / N=4 / reduce lot size
      -> Paper trade validation before live deployment
    If NO or MARGINAL:
      -> Hypothesis rejected, no filter needed
      -> Record in constraints.md as rejected direction
    """)
    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    print("=" * 80 + "\n")
if __name__ == "__main__":
    main()
