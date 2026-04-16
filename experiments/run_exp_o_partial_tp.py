#!/usr/bin/env python3
"""
EXP-O: Partial Profit Taking (Post-hoc Simulation)
=====================================================
EXP37 on old engine showed partial TP was "mildly positive" but never deployed.
Re-test on fixed engine: simulate taking 50% profit at various ATR levels,
then letting the remaining 50% ride with trailing.
Since engine doesn't support partial exits, we do post-hoc replay on trade PnL paths.
"""
import sys, os, time, gc
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from backtest import DataBundle, run_variant
from backtest.runner import LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine

OUTPUT_FILE = "exp_o_partial_tp_output.txt"
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
print("EXP-O: PARTIAL PROFIT TAKING ANALYSIS")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

# Run baseline with verbose to get trade objects
print("\n--- Loading baseline trades ---")
s_base = run_variant(data, "O_baseline", verbose=True, **BASE, spread_cost=0.30)
trades = s_base.get('_trades', [])
print(f"Baseline: N={s_base['n']}, Sharpe={s_base['sharpe']:.2f}, PnL=${s_base['total_pnl']:,.0f}")

# Analyze MFE (Maximum Favorable Excursion) for each trade
print("\n--- Part 1: MFE/MAE Analysis ---")
kc_trades = [t for t in trades if t.strategy == 'keltner']
print(f"Keltner trades: {len(kc_trades)}")

# For each trade, calculate MFE using M15 bar data
mfe_list = []
mae_list = []
pnl_list = []
atr_at_entry = []

for t in kc_trades:
    entry_time = pd.Timestamp(t.entry_time)
    exit_time = pd.Timestamp(t.exit_time)
    entry_price = t.entry_price
    direction = t.direction
    
    # Get M15 bars during trade
    mask = (data.m15_df.index >= entry_time) & (data.m15_df.index <= exit_time)
    trade_bars = data.m15_df[mask]
    
    if len(trade_bars) < 1:
        continue
    
    # Get ATR at entry from H1
    h1_mask = data.h1_df.index <= entry_time
    if h1_mask.any():
        h1_idx = h1_mask.sum() - 1
        entry_atr = float(data.h1_df.iloc[h1_idx].get('ATR', 10))
    else:
        entry_atr = 10
    
    if direction == 'BUY':
        mfe = float(trade_bars['High'].max()) - entry_price
        mae = entry_price - float(trade_bars['Low'].min())
    else:
        mfe = entry_price - float(trade_bars['Low'].min())
        mae = float(trade_bars['High'].max()) - entry_price
    
    mfe_atr = mfe / entry_atr if entry_atr > 0 else 0
    mae_atr = mae / entry_atr if entry_atr > 0 else 0
    
    mfe_list.append(mfe_atr)
    mae_list.append(mae_atr)
    pnl_list.append(t.pnl)
    atr_at_entry.append(entry_atr)

mfe_arr = np.array(mfe_list)
mae_arr = np.array(mae_list)
pnl_arr = np.array(pnl_list)
atr_arr = np.array(atr_at_entry)

print(f"\n  MFE (ATR): mean={np.mean(mfe_arr):.2f}, median={np.median(mfe_arr):.2f}, "
      f"P25={np.percentile(mfe_arr, 25):.2f}, P75={np.percentile(mfe_arr, 75):.2f}")
print(f"  MAE (ATR): mean={np.mean(mae_arr):.2f}, median={np.median(mae_arr):.2f}")

# MFE distribution
print("\n  MFE Distribution (ATR multiples):")
for threshold in [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00]:
    pct = (mfe_arr >= threshold).sum() / len(mfe_arr) * 100
    print(f"    MFE >= {threshold:.2f} ATR: {pct:.1f}% of trades")

# ── Part 2: Partial TP Simulation ──
print("\n\n--- Part 2: Partial TP Simulation ---")
print("Model: take X% profit at Y×ATR, let rest ride to actual exit")
print("Partial PnL = (take_pct * Y*ATR*lot) + ((1-take_pct) * actual_pnl)")

TAKE_PCTS = [0.25, 0.50, 0.75]
TAKE_ATR_LEVELS = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]

print(f"\n{'Take%':>5s} {'ATR_lv':>6s}  {'Hit%':>5s}  {'PnL':>11s}  {'vs_Base':>8s}  {'$/t':>7s}  {'WR%':>5s}")
print("-" * 60)

base_pnl = sum(pnl_arr)
for take_pct in TAKE_PCTS:
    for take_atr in TAKE_ATR_LEVELS:
        # For each trade: did MFE reach take_atr?
        hit_mask = mfe_arr >= take_atr
        hit_pct = hit_mask.sum() / len(mfe_arr) * 100
        
        # Calculate new PnL
        new_pnls = []
        for i in range(len(pnl_arr)):
            if hit_mask[i]:
                # Partial was triggered
                partial_profit = take_pct * take_atr * atr_arr[i] * 0.01  # 0.01 lot
                remaining_pnl = (1 - take_pct) * pnl_arr[i]
                new_pnls.append(partial_profit + remaining_pnl)
            else:
                # MFE never reached take level, all at original PnL
                new_pnls.append(pnl_arr[i])
        
        total_new = sum(new_pnls)
        avg_new = total_new / len(new_pnls)
        wr_new = sum(1 for p in new_pnls if p > 0) / len(new_pnls) * 100
        vs_base = total_new - base_pnl
        
        print(f"  {take_pct*100:>3.0f}%  {take_atr:>5.2f}  {hit_pct:>5.1f}%  "
              f"{fmt(total_new)}  {'+' if vs_base >= 0 else ''}{vs_base:>7.0f}  "
              f"${avg_new:>6.2f}  {wr_new:>5.1f}%")

# ── Part 3: "Float-back" analysis ──
print("\n\n--- Part 3: Float-Back Analysis ---")
print("How many winning trades gave back >50% of their MFE?")

float_back = []
for i in range(len(pnl_arr)):
    if mfe_arr[i] > 0:
        realized_pct = pnl_arr[i] / (mfe_arr[i] * atr_arr[i] * 0.01) if mfe_arr[i] * atr_arr[i] > 0 else 0
        float_back.append(realized_pct)
    else:
        float_back.append(0)

float_back = np.array(float_back)
winners = pnl_arr > 0
losers = pnl_arr <= 0

if winners.sum() > 0:
    win_fb = float_back[winners]
    print(f"\n  Winners (N={winners.sum()}):")
    print(f"    Avg MFE capture: {np.mean(win_fb)*100:.1f}%")
    print(f"    Median MFE capture: {np.median(win_fb)*100:.1f}%")
    gave_back_50 = (win_fb < 0.50).sum()
    print(f"    Gave back >50% of MFE: {gave_back_50} ({gave_back_50/len(win_fb)*100:.1f}%)")

if losers.sum() > 0:
    lose_fb = float_back[losers]
    lose_mfe = mfe_arr[losers]
    print(f"\n  Losers (N={losers.sum()}):")
    print(f"    Avg MFE before loss: {np.mean(lose_mfe):.2f} ATR")
    could_have_won = (lose_mfe >= 0.50).sum()
    print(f"    Had MFE >= 0.50 ATR before losing: {could_have_won} ({could_have_won/losers.sum()*100:.1f}%)")
    big_mfe = (lose_mfe >= 1.00).sum()
    print(f"    Had MFE >= 1.00 ATR before losing: {big_mfe} ({big_mfe/losers.sum()*100:.1f}%)")

# ── Part 4: Optimal partial TP level ──
print("\n--- Part 4: Optimal Partial TP (50% take) ---")
best_atr = 0
best_improvement = -999
for take_atr in np.arange(0.20, 3.01, 0.10):
    hit_mask = mfe_arr >= take_atr
    new_pnls = []
    for i in range(len(pnl_arr)):
        if hit_mask[i]:
            partial = 0.50 * take_atr * atr_arr[i] * 0.01
            remaining = 0.50 * pnl_arr[i]
            new_pnls.append(partial + remaining)
        else:
            new_pnls.append(pnl_arr[i])
    improvement = sum(new_pnls) - base_pnl
    if improvement > best_improvement:
        best_improvement = improvement
        best_atr = take_atr

print(f"  Best partial TP level: {best_atr:.1f} ATR (50% take)")
print(f"  Improvement over baseline: ${best_improvement:,.0f}")
print(f"  (Post-hoc analysis — needs proper engine implementation for real validation)")

elapsed = time.time() - t_total
print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
print(f"Completed: {datetime.now()}")
sys.stdout = tee.stdout
tee.close()
print(f"Results saved to {OUTPUT_FILE}")
