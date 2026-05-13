#!/usr/bin/env python3
"""
R198c — Replay live Keltner trades with ATR Pctl Floor = 30 vs 35
For each live trade, compute the ATR percentile at entry time.
Show which trades would be filtered by pctl=35 but not pctl=30.
"""
import sys, os, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import glob as _glob

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

# Load H1 data
candidates = sorted(_glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
df = pd.read_csv(candidates[-1])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
df = df.set_index('timestamp')
df.index = df.index.tz_localize(None)
df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
h1 = df[['Open', 'High', 'Low', 'Close']].copy()

# Compute ATR and percentile
tr = pd.DataFrame({
    'hl': h1['High'] - h1['Low'],
    'hc': (h1['High'] - h1['Close'].shift(1)).abs(),
    'lc': (h1['Low'] - h1['Close'].shift(1)).abs()
}).max(axis=1)
h1['ATR'] = tr.rolling(14).mean()

lb = 300
atr_v = h1['ATR'].values
pctl = np.full(len(atr_v), np.nan)
for i in range(lb, len(atr_v)):
    w = atr_v[i-lb:i]
    valid = w[~np.isnan(w)]
    if len(valid) >= 30:
        pctl[i] = np.sum(valid <= atr_v[i]) / len(valid) * 100
h1['ATR_pctl'] = pctl

print(f"Loaded {len(h1)} H1 bars: {h1.index[0]} to {h1.index[-1]}")

# Load live trades
trade_paths = [
    Path(r"c:\Users\hlin2\gold-quant-trading\data\gold_trade_log.json"),
    Path("/root/gold-quant-trading/data/gold_trade_log.json"),
    Path("data/gold_trade_log.json"),
]
trade_log = None
for p in trade_paths:
    if p.exists():
        with open(p) as f:
            trade_log = json.load(f)
        break

if trade_log is None:
    print("ERROR: Cannot find gold_trade_log.json")
    sys.exit(1)

# Filter Keltner CLOSE trades in the last month
cutoff = '2026-04-09'
kelt_closes = [t for t in trade_log if t.get('action') == 'CLOSE' and t.get('strategy') == 'keltner' and t.get('time', '') >= cutoff]

# For each CLOSE, find the matching OPEN
kelt_opens = [t for t in trade_log if t.get('action') == 'OPEN' and t.get('strategy') == 'keltner']

# Match opens to closes by looking for the OPEN just before each CLOSE
# Use ticket number if available, otherwise match by time proximity
close_with_open = []
for ct in kelt_closes:
    close_time = pd.Timestamp(ct['time'])
    ticket = ct.get('ticket')
    
    # Find the corresponding OPEN
    best_open = None
    if ticket:
        for ot in kelt_opens:
            if ot.get('ticket') == ticket:
                best_open = ot
                break
    
    if best_open is None:
        open_price = ct.get('open_price')
        if open_price:
            for ot in reversed(kelt_opens):
                ot_time = pd.Timestamp(ot['time'])
                if ot_time < close_time and abs(ot.get('price', 0) - open_price) < 1.0:
                    best_open = ot
                    break
    
    entry_time = pd.Timestamp(best_open['time']) if best_open else close_time - pd.Timedelta(hours=1)
    close_with_open.append({
        'entry_time': entry_time,
        'close_time': close_time,
        'open_price': ct.get('open_price', 0),
        'close_price': ct.get('close_price', 0),
        'profit': ct.get('profit', 0),
        'lots': ct.get('lots', 0),
        'reason': str(ct.get('close_reason', ct.get('reason', '?')))[:40],
    })

# For each trade, find the ATR percentile at entry time
print(f"\n{'='*120}")
print(f"  R198c: Live Keltner Trade Replay — ATR Pctl Floor 30 vs 35")
print(f"  {len(close_with_open)} trades since {cutoff}")
print(f"{'='*120}")

filtered_30_35 = []  # trades that pctl=35 would filter but pctl=30 would not
all_with_pctl = []

for t in close_with_open:
    entry = t['entry_time']
    # Find nearest H1 bar
    idx = h1.index.searchsorted(entry)
    if idx >= len(h1):
        idx = len(h1) - 1
    if idx > 0 and abs((h1.index[idx] - entry).total_seconds()) > abs((h1.index[idx-1] - entry).total_seconds()):
        idx = idx - 1
    
    atr_val = h1['ATR'].iloc[idx]
    pctl_val = h1['ATR_pctl'].iloc[idx]
    
    t['atr'] = round(float(atr_val), 2) if not np.isnan(atr_val) else None
    t['atr_pctl'] = round(float(pctl_val), 1) if not np.isnan(pctl_val) else None
    all_with_pctl.append(t)
    
    if t['atr_pctl'] is not None and 30 <= t['atr_pctl'] < 35:
        filtered_30_35.append(t)

# Sort by entry time
all_with_pctl.sort(key=lambda x: x['entry_time'])

print(f"\n  --- All trades with ATR percentile ---")
print(f"  {'Entry Time':<20} {'Price':>8} {'PnL':>8} {'ATR':>6} {'Pctl':>6} {'30?':>4} {'35?':>4} {'Reason'}")
print(f"  {'-'*110}")

total_pnl_30 = 0
total_pnl_35 = 0
n_30 = 0
n_35 = 0

for t in all_with_pctl:
    pctl_v = t['atr_pctl']
    pnl = t['profit']
    
    pass_30 = pctl_v is None or pctl_v >= 30
    pass_35 = pctl_v is None or pctl_v >= 35
    
    if pass_30:
        total_pnl_30 += pnl
        n_30 += 1
    if pass_35:
        total_pnl_35 += pnl
        n_35 += 1
    
    marker_30 = "OK" if pass_30 else "SKIP"
    marker_35 = "OK" if pass_35 else "SKIP"
    in_range = " <-- FILTERED by 35" if pass_30 and not pass_35 else ""
    
    et = str(t['entry_time'])[:16]
    print(f"  {et:<20} {t['open_price']:>8.1f} {pnl:>8.2f} {t['atr'] or 0:>6.1f} {pctl_v or 0:>6.1f} {marker_30:>4} {marker_35:>4}{in_range}")

print(f"\n{'='*120}")
print(f"  SUMMARY")
print(f"{'='*120}")
print(f"  Total trades:                    {len(all_with_pctl)}")
print(f"  Trades passing pctl>=30:         {n_30}  (PnL: ${total_pnl_30:.2f})")
print(f"  Trades passing pctl>=35:         {n_35}  (PnL: ${total_pnl_35:.2f})")
print(f"  Trades filtered by 35 but not 30: {n_30 - n_35}")
print(f"  PnL of filtered trades:          ${total_pnl_30 - total_pnl_35:.2f}")
print(f"")
if n_30 > 0 and n_35 > 0:
    print(f"  Avg PnL per trade (pctl>=30):    ${total_pnl_30/n_30:.2f}")
    print(f"  Avg PnL per trade (pctl>=35):    ${total_pnl_35/n_35:.2f}")

# Detail on filtered trades
if filtered_30_35:
    print(f"\n  --- Trades that pctl=35 would have REMOVED ---")
    filt_pnl = sum(t['profit'] for t in filtered_30_35)
    for t in filtered_30_35:
        et = str(t['entry_time'])[:16]
        print(f"    {et}  price={t['open_price']:.1f}  pnl=${t['profit']:>7.2f}  ATR={t['atr']}  pctl={t['atr_pctl']}")
    print(f"  Total PnL of removed trades: ${filt_pnl:.2f}")
    wins = sum(1 for t in filtered_30_35 if t['profit'] > 0)
    losses = len(filtered_30_35) - wins
    print(f"  Wins: {wins}, Losses: {losses}")
print(f"\n{'='*120}")
