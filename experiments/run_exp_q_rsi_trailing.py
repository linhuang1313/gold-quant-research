#!/usr/bin/env python3
"""
EXP-Q: RSI2 Factor as Dynamic Trailing Adjuster
==================================================
RSI2 IC=-0.031 (strongest factor), currently only used for M15 entry.
Hypothesis: when RSI2 hits extreme during an open keltner trade,
tighten trailing stop (the reversal is coming).
Post-hoc analysis: match trade PnL paths with concurrent RSI2 values.
"""
import sys, os, time, gc
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from backtest import DataBundle, run_variant
from backtest.runner import LIVE_PARITY_KWARGS

OUTPUT_FILE = "exp_q_rsi_trailing_output.txt"
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
print("EXP-Q: RSI2 FACTOR AS DYNAMIC TRAILING ADJUSTER")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

# Ensure RSI2 is available on H1
if 'RSI2' not in data.h1_df.columns:
    delta = data.h1_df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(2).mean()
    avg_loss = loss.rolling(2).mean()
    rs = avg_gain / avg_loss
    data.h1_df['RSI2'] = 100 - (100 / (1 + rs))

if 'RSI2' not in data.m15_df.columns:
    delta = data.m15_df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(2).mean()
    avg_loss = loss.rolling(2).mean()
    rs = avg_gain / avg_loss
    data.m15_df['RSI2'] = 100 - (100 / (1 + rs))

print("\n--- Part 1: RSI2 During Open Keltner Trades ---")
s_base = run_variant(data, "Q_baseline", verbose=True, **BASE, spread_cost=0.30)
trades = s_base.get('_trades', [])
kc_trades = [t for t in trades if t.strategy == 'keltner']
print(f"Baseline: N={s_base['n']}, Sharpe={s_base['sharpe']:.2f}")
print(f"Keltner trades: {len(kc_trades)}")

# For each KC trade, find min/max RSI2 during the trade
trade_rsi_data = []
for t in kc_trades:
    entry_time = pd.Timestamp(t.entry_time)
    exit_time = pd.Timestamp(t.exit_time)
    direction = t.direction
    
    # Get H1 RSI2 during trade
    h1_mask = (data.h1_df.index >= entry_time) & (data.h1_df.index <= exit_time)
    h1_trade = data.h1_df[h1_mask]
    
    if len(h1_trade) < 1 or 'RSI2' not in h1_trade.columns:
        continue
    
    rsi2_values = h1_trade['RSI2'].dropna()
    if len(rsi2_values) < 1:
        continue
    
    max_rsi2 = float(rsi2_values.max())
    min_rsi2 = float(rsi2_values.min())
    entry_rsi2 = float(rsi2_values.iloc[0]) if len(rsi2_values) > 0 else 50
    
    # "Extreme against" = signal that reversal is coming
    if direction == 'BUY':
        extreme_against = max_rsi2 >= 90  # overbought during long
        extreme_with = min_rsi2 <= 10     # oversold during long (unusual)
    else:
        extreme_against = min_rsi2 <= 10  # oversold during short
        extreme_with = max_rsi2 >= 90     # overbought during short (unusual)
    
    trade_rsi_data.append({
        'pnl': t.pnl,
        'direction': direction,
        'max_rsi2': max_rsi2,
        'min_rsi2': min_rsi2,
        'entry_rsi2': entry_rsi2,
        'extreme_against': extreme_against,
        'extreme_with': extreme_with,
        'bars_held': (exit_time - entry_time).total_seconds() / 900,  # M15 bars
    })

df_rsi = pd.DataFrame(trade_rsi_data)
print(f"\nTrades with RSI2 data: {len(df_rsi)}")

if len(df_rsi) > 0:
    # ── Part 2: Performance split by RSI2 extreme ──
    print("\n--- Part 2: Performance Split by In-Trade RSI2 Extreme ---")
    
    for label, mask_col in [("Extreme against (reversal signal)", 'extreme_against'),
                            ("Extreme with (continuation)", 'extreme_with')]:
        yes = df_rsi[df_rsi[mask_col]]
        no = df_rsi[~df_rsi[mask_col]]
        
        print(f"\n  {label}:")
        for sub_label, sub_df in [("YES", yes), ("NO", no)]:
            if len(sub_df) == 0:
                continue
            n = len(sub_df)
            pnl = sub_df['pnl'].sum()
            avg = pnl / n
            wr = (sub_df['pnl'] > 0).sum() / n * 100
            print(f"    {sub_label}: N={n}, PnL=${pnl:,.0f}, $/t=${avg:.2f}, WR={wr:.1f}%")
    
    # ── Part 3: Entry RSI2 as quality filter ──
    print("\n--- Part 3: Entry RSI2 as Quality Filter ---")
    print("Does RSI2 at entry predict trade outcome?")
    
    for threshold_name, buy_ok, sell_ok in [
        ("RSI2<80 for BUY, >20 for SELL", lambda r: r < 80, lambda r: r > 20),
        ("RSI2<70 for BUY, >30 for SELL", lambda r: r < 70, lambda r: r > 30),
        ("RSI2<60 for BUY, >40 for SELL", lambda r: r < 60, lambda r: r > 40),
        ("RSI2 30-70 (neutral zone)", lambda r: 30 <= r <= 70, lambda r: 30 <= r <= 70),
    ]:
        filtered = df_rsi[
            ((df_rsi['direction'] == 'BUY') & df_rsi['entry_rsi2'].apply(buy_ok)) |
            ((df_rsi['direction'] == 'SELL') & df_rsi['entry_rsi2'].apply(sell_ok))
        ]
        if len(filtered) == 0:
            continue
        n = len(filtered)
        pnl = filtered['pnl'].sum()
        avg = pnl / n
        wr = (filtered['pnl'] > 0).sum() / n * 100
        skipped = len(df_rsi) - n
        print(f"  {threshold_name}: N={n} (skip {skipped}), PnL=${pnl:,.0f}, $/t=${avg:.2f}, WR={wr:.1f}%")
    
    # ── Part 4: RSI2 quintile analysis ──
    print("\n--- Part 4: Entry RSI2 Quintile Performance ---")
    df_rsi['rsi2_quintile'] = pd.qcut(df_rsi['entry_rsi2'], 5, labels=['Q1_low', 'Q2', 'Q3', 'Q4', 'Q5_high'], duplicates='drop')
    
    print(f"\n  {'Quintile':<10s}  {'N':>5s}  {'PnL':>10s}  {'$/t':>7s}  {'WR%':>5s}  {'RSI2_range':>15s}")
    print("  " + "-" * 55)
    
    for q in ['Q1_low', 'Q2', 'Q3', 'Q4', 'Q5_high']:
        qdf = df_rsi[df_rsi['rsi2_quintile'] == q]
        if len(qdf) == 0:
            continue
        n = len(qdf)
        pnl = qdf['pnl'].sum()
        avg = pnl / n
        wr = (qdf['pnl'] > 0).sum() / n * 100
        rsi_lo = qdf['entry_rsi2'].min()
        rsi_hi = qdf['entry_rsi2'].max()
        print(f"  {q:<10s}  {n:>5d}  ${pnl:>9,.0f}  ${avg:>6.2f}  {wr:>5.1f}%  {rsi_lo:.0f}-{rsi_hi:.0f}")
    
    # ── Part 5: RSI2 momentum during trade ──
    print("\n--- Part 5: RSI2 Direction During Trade ---")
    # Did RSI2 move toward extreme (momentum continuation) or away?
    
    for direction in ['BUY', 'SELL']:
        dir_df = df_rsi[df_rsi['direction'] == direction]
        if len(dir_df) == 0:
            continue
        
        # For BUY: RSI2 going up = continuation, going down = potential reversal
        if direction == 'BUY':
            rising = dir_df[dir_df['max_rsi2'] > dir_df['entry_rsi2'] + 10]
            falling = dir_df[dir_df['min_rsi2'] < dir_df['entry_rsi2'] - 10]
        else:
            rising = dir_df[dir_df['max_rsi2'] > dir_df['entry_rsi2'] + 10]
            falling = dir_df[dir_df['min_rsi2'] < dir_df['entry_rsi2'] - 10]
        
        print(f"\n  {direction} trades ({len(dir_df)}):")
        for label, sub in [("RSI2 rose >10pts", rising), ("RSI2 fell >10pts", falling)]:
            if len(sub) == 0:
                continue
            n = len(sub)
            pnl = sub['pnl'].sum()
            avg = pnl / n
            wr = (sub['pnl'] > 0).sum() / n * 100
            print(f"    {label}: N={n}, PnL=${pnl:,.0f}, $/t=${avg:.2f}, WR={wr:.1f}%")

    # ── Part 6: H1 RSI14 analysis (broader momentum) ──
    print("\n\n--- Part 6: H1 RSI14 at Entry ---")
    rsi14_data = []
    for t in kc_trades:
        entry_time = pd.Timestamp(t.entry_time)
        h1_mask = data.h1_df.index <= entry_time
        if h1_mask.any():
            h1_idx = h1_mask.sum() - 1
            rsi14 = float(data.h1_df.iloc[h1_idx].get('RSI14', 50))
            rsi14_data.append({'pnl': t.pnl, 'rsi14': rsi14, 'direction': t.direction})
    
    if rsi14_data:
        df14 = pd.DataFrame(rsi14_data)
        # Overbought/oversold at entry
        for label, mask in [("RSI14 > 70 (overbought)", df14['rsi14'] > 70),
                            ("RSI14 < 30 (oversold)", df14['rsi14'] < 30),
                            ("RSI14 30-70 (neutral)", (df14['rsi14'] >= 30) & (df14['rsi14'] <= 70))]:
            sub = df14[mask]
            if len(sub) == 0:
                continue
            n = len(sub)
            pnl = sub['pnl'].sum()
            avg = pnl / n
            wr = (sub['pnl'] > 0).sum() / n * 100
            print(f"  {label}: N={n}, PnL=${pnl:,.0f}, $/t=${avg:.2f}, WR={wr:.1f}%")

elapsed = time.time() - t_total
print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
print(f"Completed: {datetime.now()}")
sys.stdout = tee.stdout
tee.close()
print(f"Results saved to {OUTPUT_FILE}")
