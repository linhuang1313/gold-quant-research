#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_exp_x_eurusd_stress.py
==========================
EUR/USD Keltner-style stress test (post-hoc M15 pip simulation).

Full BacktestEngine uses gold-centric POINT_VALUE / lot sizing; this script avoids
instantiating EUR trades in the engine and instead simulates:
  signal at bar i close -> enter next bar Open -> exit Open after MAX_HOLD M15 bars,
  with round-trip spread cost expressed in pips.

Requires bid CSVs under data/download/ (see EURUSD_*_CANDIDATES below).
If missing, exits with a clear message (no npx download on remote).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from scipy.stats import pearsonr

from backtest.engine import BacktestEngine
from backtest.runner import (
    LIVE_PARITY_KWARGS,
    M15_CSV_PATH,
    H1_CSV_PATH,
    add_atr_percentile,
    load_csv,
    load_h1_aligned,
    prepare_indicators_custom,
)
from indicators import prepare_indicators

# ═══════════════════════════════════════════════════════════════
# EUR/USD data paths (newer file first, then legacy 2026-03-25 names)
# ═══════════════════════════════════════════════════════════════

EURUSD_M15_CANDIDATES = [
    Path("data/download/eurusd-m15-bid-2015-01-01-2026-04-10.csv"),
    Path("data/download/eurusd-m15-bid-2015-01-01-2026-03-25.csv"),
]
EURUSD_H1_CANDIDATES = [
    Path("data/download/eurusd-h1-bid-2015-01-01-2026-04-10.csv"),
    Path("data/download/eurusd-h1-bid-2015-01-01-2026-03-25.csv"),
]

# Live-style reference (engine not used for EUR PnL; documents parity intent)
EURUSD_KWARGS = {
    **LIVE_PARITY_KWARGS,
    "keltner_max_hold_m15": 20,
    "spread_cost": 0.00018,
}

PIP = 0.0001  # EURUSD standard pip
LOOKBACK = 150
MAX_HOLD = 20
ADX_MIN = float(LIVE_PARITY_KWARGS.get("keltner_adx_threshold", 18))
KC_SWEEP = [1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
SPREAD_PIPS_SWEEP = [1.0, 1.5, 1.8, 2.5, 3.0]

# Same windows as backtest.runner.run_kfold (6-fold time-series)
KFOLD_WINDOWS: List[Tuple[str, str, str]] = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-01"),
]


def resolve_first_existing(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.is_file():
            return p
    return None


def check_eurusd_data_or_exit() -> Tuple[Path, Path]:
    m15 = resolve_first_existing(EURUSD_M15_CANDIDATES)
    h1 = resolve_first_existing(EURUSD_H1_CANDIDATES)
    if m15 is None or h1 is None:
        print("\n" + "=" * 72)
        print("EUR/USD DATA NOT FOUND")
        print("=" * 72)
        print("This experiment needs bid CSV files under data/download/, for example:")
        for p in EURUSD_M15_CANDIDATES + EURUSD_H1_CANDIDATES:
            print(f"  - {p}")
        print("\nPlace the files on the server or download out-of-band; this script does not")
        print("invoke npx/dukascopy (remote env may not have Node).")
        print("=" * 72 + "\n")
        sys.exit(2)
    return m15, h1


def load_eurusd_frames(m15_path: Path, h1_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\nLoading EUR/USD M15: {m15_path}")
    m15_raw = load_csv(str(m15_path))
    print(f"  bars={len(m15_raw)}  {m15_raw.index[0]} -> {m15_raw.index[-1]}")
    print(f"Loading EUR/USD H1: {h1_path}")
    h1_raw = load_h1_aligned(h1_path, m15_raw.index[0])
    print(f"  bars={len(h1_raw)}  {h1_raw.index[0]} -> {h1_raw.index[-1]}")
    return m15_raw, h1_raw


def prepare_m15_kc(m15_raw: pd.DataFrame, kc_mult: float) -> pd.DataFrame:
    return prepare_indicators_custom(m15_raw.copy(), kc_mult=kc_mult)


def bar_signal(row: pd.Series) -> Optional[str]:
    """Return 'BUY', 'SELL', or None — mirrors check_keltner_signal thresholds on one row."""
    if any(
        pd.isna(row.get(c))
        for c in ("Close", "KC_upper", "KC_lower", "EMA100", "ADX")
    ):
        return None
    close = float(row["Close"])
    kc_u = float(row["KC_upper"])
    kc_l = float(row["KC_lower"])
    ema100 = float(row["EMA100"])
    adx = float(row["ADX"])
    if adx < ADX_MIN:
        return None
    if close > kc_u and close > ema100:
        return "BUY"
    if close < kc_l and close < ema100:
        return "SELL"
    return None


def simulate_trades(m15: pd.DataFrame, max_hold: int = MAX_HOLD) -> List[Dict]:
    """Non-overlapping trades; entry next bar Open after signal bar close."""
    n = len(m15)
    trades: List[Dict] = []
    i = LOOKBACK
    while i < n - 1 - max_hold:
        sig = bar_signal(m15.iloc[i])
        if sig is None:
            i += 1
            continue
        ent = i + 1
        ext = ent + max_hold
        if ext >= n:
            break
        o_ent = float(m15.iloc[ent]["Open"])
        o_ext = float(m15.iloc[ext]["Open"])
        if sig == "BUY":
            pips_gross = (o_ext - o_ent) / PIP
        else:
            pips_gross = (o_ent - o_ext) / PIP
        trades.append(
            {
                "signal_idx": i,
                "entry_idx": ent,
                "exit_idx": ext,
                "entry_time": m15.index[ent],
                "direction": sig,
                "pips_gross": pips_gross,
            }
        )
        i = ext
    return trades


def net_pips(trades: List[Dict], spread_pips: float) -> float:
    if not trades:
        return 0.0
    return float(sum(t["pips_gross"] for t in trades)) - len(trades) * spread_pips


def fold_for_timestamp(ts: pd.Timestamp) -> Optional[str]:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    for name, s, e in KFOLD_WINDOWS:
        if pd.Timestamp(s, tz="UTC") <= t < pd.Timestamp(e, tz="UTC"):
            return name
    return None


def summarize_kfold(trades: List[Dict], spread_pips: float) -> None:
    print("\n" + "-" * 72)
    print(f"6-fold time-series summary (spread = {spread_pips} pips RT per trade)")
    print("-" * 72)
    by_fold: Dict[str, List[Dict]] = {f[0]: [] for f in KFOLD_WINDOWS}
    for t in trades:
        fold = fold_for_timestamp(t["entry_time"])
        if fold:
            by_fold[fold].append(t)
    for name, _, _ in KFOLD_WINDOWS:
        sub = by_fold[name]
        np_ = net_pips(sub, spread_pips)
        print(f"  {name}: n={len(sub):5d}  net_pips={np_:+10.1f}  mean/trade={np_ / len(sub) if sub else 0:+.2f}")


def yearly_pnl(trades: List[Dict], spread_pips: float) -> None:
    print("\n" + "-" * 72)
    print(f"Year-by-year (spread = {spread_pips} pips RT)")
    print("-" * 72)
    by_year: Dict[int, List[Dict]] = {}
    for t in trades:
        y = int(pd.Timestamp(t["entry_time"]).year)
        by_year.setdefault(y, []).append(t)
    for y in sorted(by_year):
        sub = by_year[y]
        np_ = net_pips(sub, spread_pips)
        wins = sum(1 for x in sub if x["pips_gross"] > spread_pips)
        print(
            f"  {y}: n={len(sub):4d}  net_pips={np_:+9.1f}  "
            f"win%={100 * wins / len(sub) if sub else 0:.1f}"
        )


def collect_long_signal_dates(m15: pd.DataFrame) -> Set:
    dates: Set = set()
    for i in range(LOOKBACK, len(m15) - 1):
        sig = bar_signal(m15.iloc[i])
        if sig == "BUY":
            dates.add(pd.Timestamp(m15.index[i]).tz_convert("UTC").date())
    return dates


def gold_correlation_daily(
    eu_m15: pd.DataFrame, gold_m15: pd.DataFrame, kc_eur: float, kc_gold: float = 1.2
) -> Optional[Tuple[float, float]]:
    eu_df = prepare_m15_kc(eu_m15, kc_eur)
    gold_df = (
        prepare_indicators(gold_m15.copy())
        if abs(kc_gold - 1.2) < 1e-9
        else prepare_indicators_custom(gold_m15.copy(), kc_mult=kc_gold)
    )
    eu_dates = collect_long_signal_dates(eu_df)
    gold_dates = collect_long_signal_dates(gold_df)
    d0 = max(eu_m15.index[0].date(), gold_m15.index[0].date())
    d1 = min(eu_m15.index[-1].date(), gold_m15.index[-1].date())
    days = pd.date_range(d0, d1, freq="D")
    if len(days) < 30:
        return None
    x = np.array([1 if d.date() in eu_dates else 0 for d in days], dtype=float)
    y = np.array([1 if d.date() in gold_dates else 0 for d in days], dtype=float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return None
    r, p = pearsonr(x, y)
    return float(r), float(p)


def main() -> None:
    print("=" * 72)
    print("EUR/USD KELTNER STRESS (post-hoc M15 pip simulation)")
    print("=" * 72)
    print(f"BacktestEngine available: {BacktestEngine.__name__} (gold-centric PnL; not used for EUR)")
    print(f"LIVE_PARITY_KWARGS keltner_adx_threshold={LIVE_PARITY_KWARGS.get('keltner_adx_threshold')}")
    print(f"EURUSD_KWARGS (reference): max_hold={EURUSD_KWARGS['keltner_max_hold_m15']}, "
          f"spread_cost(price)={EURUSD_KWARGS['spread_cost']}")
    print(f"Simulation: ADX>={ADX_MIN}, entry=next Open, hold={MAX_HOLD} M15 bars, pip={PIP}")
    print("K-fold windows match backtest.runner.run_kfold (6-fold time-series).")

    m15_path, h1_path = check_eurusd_data_or_exit()
    m15_raw, h1_raw = load_eurusd_frames(m15_path, h1_path)

    print("\nPreparing EUR/USD indicators (H1 + atr_percentile for pipeline parity)...")
    h1_eu = add_atr_percentile(prepare_indicators(h1_raw.copy()))
    print(f"  H1 prepared: {len(h1_eu)} bars (unused in simple sim; matches gold load path)")

    print("\n" + "=" * 72)
    print("KC MULTIPLIER SWEEP (spread = 1.8 pips RT)")
    print("=" * 72)
    best_kc: Optional[float] = None
    best_net = -float("inf")
    sweep_rows: List[Tuple[float, int, float]] = []

    for kc in KC_SWEEP:
        m15 = prepare_m15_kc(m15_raw, kc)
        tr = simulate_trades(m15)
        np18 = net_pips(tr, 1.8)
        sweep_rows.append((kc, len(tr), np18))
        print(f"  KC_mult={kc:.1f}  trades={len(tr):5d}  net_pips@1.8spd={np18:+10.1f}")
        if np18 > best_net:
            best_net = np18
            best_kc = kc

    assert best_kc is not None
    m15_best = prepare_m15_kc(m15_raw, best_kc)
    trades_best = simulate_trades(m15_best)

    print("\n" + "=" * 72)
    print("SPREAD STRESS (best KC_mult from sweep)")
    print("=" * 72)
    print(f"  Best KC_mult = {best_kc} (net pips @ 1.8 spd = {best_net:+.1f})")
    for sp in SPREAD_PIPS_SWEEP:
        np_ = net_pips(trades_best, sp)
        print(f"  spread {sp:.1f} pips RT: net_pips={np_:+10.1f}  (n={len(trades_best)})")

    summarize_kfold(trades_best, 1.8)
    yearly_pnl(trades_best, 1.8)

    # Gold correlation
    print("\n" + "=" * 72)
    print("DAILY LONG-SIGNAL CORRELATION vs XAUUSD (M15, same ADX/KC logic)")
    print("=" * 72)
    if not M15_CSV_PATH.exists() or not H1_CSV_PATH.exists():
        print("  Gold M15/H1 CSV not found — skip correlation.")
        print(f"    Tried M15={M15_CSV_PATH}  H1={H1_CSV_PATH}")
    else:
        print(f"  Loading gold M15 from {M15_CSV_PATH}")
        gold_m15 = load_csv(str(M15_CSV_PATH))
        corr_p9 = gold_correlation_daily(m15_raw, gold_m15, kc_eur=2.0, kc_gold=1.2)
        corr_best = gold_correlation_daily(m15_raw, gold_m15, kc_eur=float(best_kc), kc_gold=1.2)
        if corr_p9:
            print(f"  EUR KC=2.0 (paper-style) vs Gold KC=1.2: r={corr_p9[0]:+.3f}  p={corr_p9[1]:.2e}")
        else:
            print("  EUR KC=2.0 vs Gold: insufficient overlap or constant series — skip.")
        if corr_best and abs(float(best_kc) - 2.0) > 1e-6:
            print(f"  EUR KC={best_kc} vs Gold KC=1.2: r={corr_best[0]:+.3f}  p={corr_best[1]:.2e}")
        elif corr_best and abs(float(best_kc) - 2.0) <= 1e-6:
            print("  (best KC equals 2.0 — same row as P9-style correlation.)")

    # Summary
    rec_spread = 1.8
    net_at_rec = net_pips(trades_best, rec_spread)
    mean_trade = net_at_rec / len(trades_best) if trades_best else 0.0
    wins = sum(1 for t in trades_best if t["pips_gross"] > rec_spread)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Best KC_mult (by net pips @ 1.8 spd): {best_kc}")
    print(f"  Recommended spread assumption (baseline): {rec_spread} pips RT "
          f"(net_pips={net_at_rec:+.1f}, n={len(trades_best)})")
    print(f"  Mean net pips / trade @ {rec_spread} spd: {mean_trade:+.3f}")
    print(f"  Win rate (gross > {rec_spread} pips): {100 * wins / len(trades_best) if trades_best else 0:.1f}%")

    print("\n  Viability (post-hoc, not engine-validated):")
    if not trades_best:
        print("    No trades — not viable under these rules.")
    elif net_at_rec <= 0:
        print(f"    Net pips <= 0 @ {rec_spread} spd — poor edge before execution/slippage.")
    elif mean_trade < 0.5:
        print("    Small positive edge; fragile to slippage and model mismatch vs live H1 Keltner.")
    else:
        print("    Positive net pips in this toy hold-20-bar test; validate with full engine if point-value extended.")

    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
