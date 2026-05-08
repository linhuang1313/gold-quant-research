#!/usr/bin/env python3
"""
Rule B ATR% Validation: compare absolute ATR 3σ vs ATR/Close 3σ trigger rates.
Confirms the ATR% version doesn't become too loose (should still catch real extremes).
"""
import sys, os, io
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

LOOKBACK = 60
SIGMA = 3.0


def main():
    from backtest.runner import load_h1_aligned, H1_CSV_PATH, load_m15
    m15 = load_m15()
    h1 = load_h1_aligned(H1_CSV_PATH, m15.index[0])

    atr = (h1['High'] - h1['Low']).rolling(14).mean()
    close = h1['Close']
    atr_pct = atr / close

    abs_triggers = []
    pct_triggers = []

    for i in range(LOOKBACK + 1, len(h1)):
        # Absolute ATR
        hist_abs = atr.iloc[i - LOOKBACK - 1:i - 1].values
        cur_abs = float(atr.iloc[i])
        mean_abs = hist_abs.mean()
        std_abs = hist_abs.std(ddof=0)
        if std_abs > 1e-6 and cur_abs > mean_abs + SIGMA * std_abs:
            abs_triggers.append(h1.index[i])

        # ATR%
        hist_pct = atr_pct.iloc[i - LOOKBACK - 1:i - 1].values
        cur_pct = float(atr_pct.iloc[i])
        mean_pct = hist_pct.mean()
        std_pct = hist_pct.std(ddof=0)
        if std_pct > 1e-6 and cur_pct > mean_pct + SIGMA * std_pct:
            pct_triggers.append(h1.index[i])

    total_bars = len(h1) - LOOKBACK - 1
    print(f"\n{'='*60}")
    print(f"  Rule B Trigger Rate Comparison")
    print(f"  Data: {h1.index[0].date()} → {h1.index[-1].date()}")
    print(f"  Total H1 bars: {total_bars:,}")
    print(f"{'='*60}")
    print(f"\n  Absolute ATR 3σ:")
    print(f"    Triggers: {len(abs_triggers)} ({len(abs_triggers)/total_bars*100:.2f}%)")
    print(f"\n  ATR% (ATR/Close) 3σ:")
    print(f"    Triggers: {len(pct_triggers)} ({len(pct_triggers)/total_bars*100:.2f}%)")

    both = set(abs_triggers) & set(pct_triggers)
    abs_only = set(abs_triggers) - set(pct_triggers)
    pct_only = set(pct_triggers) - set(abs_triggers)
    print(f"\n  Overlap: {len(both)} bars triggered by both")
    print(f"  Abs-only (ATR% would MISS): {len(abs_only)}")
    print(f"  Pct-only (ATR% catches extra): {len(pct_only)}")

    # Yearly breakdown
    print(f"\n  Yearly breakdown:")
    print(f"  {'Year':>6}  {'Abs':>6}  {'Pct':>6}  {'Overlap':>8}")
    all_years = sorted(set(t.year for t in abs_triggers + pct_triggers))
    for year in all_years:
        a = sum(1 for t in abs_triggers if t.year == year)
        p = sum(1 for t in pct_triggers if t.year == year)
        o = sum(1 for t in both if t.year == year)
        print(f"  {year:>6}  {a:>6}  {p:>6}  {o:>8}")

    # Recent period (2025-2026): ATR% should trigger LESS due to high gold price
    recent_abs = [t for t in abs_triggers if t.year >= 2025]
    recent_pct = [t for t in pct_triggers if t.year >= 2025]
    print(f"\n  Recent (2025+):")
    print(f"    Abs: {len(recent_abs)} triggers")
    print(f"    Pct: {len(recent_pct)} triggers")
    print(f"    Reduction: {len(recent_abs) - len(recent_pct)} fewer triggers with ATR%")

    verdict = "SAFE" if len(pct_triggers) <= len(abs_triggers) * 1.2 else "REVIEW"
    print(f"\n  Verdict: {verdict}")
    print(f"  (ATR% should have similar or fewer triggers than absolute ATR)")


if __name__ == '__main__':
    main()
