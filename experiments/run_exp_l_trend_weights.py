#!/usr/bin/env python3
"""
EXP-L: IntradayTrendMeter Sub-Factor Weight Optimization
==========================================================
Current fixed weights: ADX(0.30) + KC_break(0.25) + EMA_align(0.25) + Trend_intensity(0.20)
These were NEVER swept. The TrendMeter is the core of Adaptive gating.
Test: weight variations + individual factor ablation + threshold interaction.
"""
import sys, os, time, gc
from datetime import datetime
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from backtest import DataBundle, run_variant
from backtest.runner import LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine

OUTPUT_FILE = "exp_l_trend_weights_output.txt"

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-01"),
]

BASE = {**LIVE_PARITY_KWARGS, "keltner_max_hold_m15": 20}


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

print("=" * 80)
print("EXP-L: INTRADAY TREND METER WEIGHT OPTIMIZATION")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

# Current weights in _calc_realtime_score:
# 0.30 * adx_score + 0.25 * kc_score + 0.25 * ema_score + 0.20 * ti
# We monkey-patch this to test different weights.

_WEIGHTS = (0.30, 0.25, 0.25, 0.20)  # (adx, kc, ema, ti)

_original_calc = BacktestEngine.__dict__['_calc_realtime_score']

def _make_patched_score(w_adx, w_kc, w_ema, w_ti):
    def _patched(today_bars):
        if len(today_bars) < 2:
            return 0.5
        latest = today_bars.iloc[-1]
        
        adx = float(latest.get('ADX', 20))
        if np.isnan(adx):
            adx = 20
        adx_score = min(adx / 40.0, 1.0)
        
        kc_upper = today_bars.get('KC_upper')
        kc_lower = today_bars.get('KC_lower')
        if kc_upper is not None and kc_lower is not None:
            breaks = ((today_bars['Close'] > kc_upper) | (today_bars['Close'] < kc_lower)).sum()
            kc_score = min(float(breaks) / len(today_bars), 1.0)
        else:
            kc_score = 0.0
        
        ema9 = today_bars.get('EMA9')
        ema21 = today_bars.get('EMA21')
        ema100 = today_bars.get('EMA100')
        if ema9 is not None and ema21 is not None and ema100 is not None:
            bullish = (ema9 > ema21) & (ema21 > ema100)
            bearish = (ema9 < ema21) & (ema21 < ema100)
            aligned = (bullish | bearish).sum()
            ema_score = float(aligned) / len(today_bars)
        else:
            ema_score = 0.0
        
        day_open = float(today_bars.iloc[0]['Open'])
        day_close = float(latest['Close'])
        day_high = float(today_bars['High'].max())
        day_low = float(today_bars['Low'].min())
        day_range = day_high - day_low
        ti = abs(day_close - day_open) / day_range if day_range > 0.01 else 0.0
        
        return round(w_adx * adx_score + w_kc * kc_score + w_ema * ema_score + w_ti * ti, 3)
    
    return staticmethod(_patched)

def run_with_weights(data, label, w_adx, w_kc, w_ema, w_ti, spread=0.30):
    """Run backtest with custom TrendMeter weights."""
    patched = _make_patched_score(w_adx, w_kc, w_ema, w_ti)
    BacktestEngine._calc_realtime_score = patched
    s = run_variant(data, label, verbose=False, **BASE, spread_cost=spread)
    BacktestEngine._calc_realtime_score = _original_calc
    return s

# ── Part 1: Factor Ablation ──
print("\n--- Part 1: Factor Ablation (zero out each factor) ---")
ablations = [
    ("Current (0.30/0.25/0.25/0.20)", 0.30, 0.25, 0.25, 0.20),
    ("No ADX (0/0.35/0.35/0.30)", 0.00, 0.35, 0.35, 0.30),
    ("No KC_break (0.40/0/0.33/0.27)", 0.40, 0.00, 0.33, 0.27),
    ("No EMA_align (0.40/0.33/0/0.27)", 0.40, 0.33, 0.00, 0.27),
    ("No Trend_int (0.38/0.31/0.31/0)", 0.38, 0.31, 0.31, 0.00),
    ("ADX only (1/0/0/0)", 1.00, 0.00, 0.00, 0.00),
    ("KC only (0/1/0/0)", 0.00, 1.00, 0.00, 0.00),
]

print(f"\n{'Weights':<38s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'WR%':>5s}  {'$/t':>7s}  {'MaxDD':>11s}")
print("-" * 90)

for name, wa, wk, we, wt in ablations:
    s = run_with_weights(data, f"L_{name[:10]}", wa, wk, we, wt)
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    print(f"  {name:<36s}  {n:>5d}  {s['sharpe']:>6.2f}  "
          f"{fmt(s['total_pnl'])}  {s['win_rate']:>5.1f}%  "
          f"${avg:>6.2f}  {fmt(s['max_dd'])}")

# ── Part 2: Weight Grid ──
print("\n--- Part 2: Weight Grid (normalized to sum=1) ---")

WEIGHT_SETS = [
    (0.30, 0.25, 0.25, 0.20),  # current
    (0.40, 0.20, 0.20, 0.20),  # more ADX
    (0.20, 0.30, 0.30, 0.20),  # more KC+EMA
    (0.25, 0.25, 0.25, 0.25),  # equal
    (0.35, 0.30, 0.20, 0.15),  # ADX+KC heavy
    (0.20, 0.20, 0.40, 0.20),  # EMA heavy
    (0.25, 0.35, 0.25, 0.15),  # KC heavy
    (0.30, 0.20, 0.30, 0.20),  # ADX+EMA
    (0.20, 0.30, 0.20, 0.30),  # KC+TI
    (0.40, 0.30, 0.20, 0.10),  # ADX+KC dominant
    (0.15, 0.25, 0.35, 0.25),  # EMA dominant
    (0.35, 0.15, 0.30, 0.20),  # ADX+EMA, low KC
]

print(f"\n{'ADX':>4s} {'KC':>4s} {'EMA':>4s} {'TI':>4s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'$/t':>7s}  {'MaxDD':>11s}")
print("-" * 65)

weight_results = {}
for wa, wk, we, wt in WEIGHT_SETS:
    s = run_with_weights(data, f"L_{wa}{wk}{we}{wt}", wa, wk, we, wt)
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    marker = " <--" if (wa, wk, we, wt) == (0.30, 0.25, 0.25, 0.20) else ""
    print(f"  {wa:.2f} {wk:.2f} {we:.2f} {wt:.2f}  {n:>5d}  {s['sharpe']:>6.2f}  "
          f"{fmt(s['total_pnl'])}  ${avg:>6.2f}  {fmt(s['max_dd'])}{marker}")
    weight_results[(wa, wk, we, wt)] = s

# ── Part 3: K-Fold for best weight set ──
ranked = sorted(weight_results.items(), key=lambda x: -x[1]['sharpe'])
best_w = ranked[0][0]
current_w = (0.30, 0.25, 0.25, 0.20)

if best_w != current_w and ranked[0][1]['sharpe'] > weight_results[current_w]['sharpe'] + 0.05:
    print(f"\n--- Part 3: K-Fold for {best_w} vs Current @ $0.30 ---")
    wins = 0
    for fold_name, start, end in FOLDS:
        fold_data = data.slice(start, end)
        if len(fold_data.m15_df) < 1000:
            continue
        sb = run_with_weights(fold_data, f"L_B_{fold_name}", *current_w)
        st = run_with_weights(fold_data, f"L_T_{fold_name}", *best_w)
        delta = st['sharpe'] - sb['sharpe']
        won = delta > 0
        if won: wins += 1
        print(f"    {fold_name}: Current={sb['sharpe']:>6.2f}  Test={st['sharpe']:>6.2f}  delta={delta:>+.2f} {'V' if won else 'X'}")
    print(f"    Result: {wins}/6 {'PASS' if wins >= 5 else 'FAIL'}")
else:
    print(f"\n  Current weights are optimal or difference < 0.05 Sharpe.")

# ── Part 4: Threshold × Weight interaction ──
print("\n--- Part 4: Choppy Threshold Sensitivity (with current weights) ---")
THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45]

print(f"\n{'choppy_th':>9s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'$/t':>7s}")
print("-" * 50)

# Restore original
BacktestEngine._calc_realtime_score = _original_calc

for th in THRESHOLDS:
    kwargs = {**BASE, "choppy_threshold": th}
    s = run_variant(data, f"L_th{th}", verbose=False, **kwargs, spread_cost=0.30)
    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    marker = " <-- current" if th == 0.35 else ""
    print(f"  {th:>7.2f}  {n:>5d}  {s['sharpe']:>6.2f}  {fmt(s['total_pnl'])}  ${avg:>6.2f}{marker}")

elapsed = time.time() - t_total
print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
print(f"Completed: {datetime.now()}")
sys.stdout = tee.stdout
tee.close()
print(f"Results saved to {OUTPUT_FILE}")
