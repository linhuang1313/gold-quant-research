#!/usr/bin/env python3
"""
EXP-E + EXP-F (parallel split from batch_postfix)
===================================================
Run these two post-hoc analysis experiments independently.
If batch_postfix hasn't reached them yet, we get results hours earlier.
"""
import sys, os, time, gc
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from backtest import DataBundle, run_variant
from backtest.runner import LIVE_PARITY_KWARGS

OUTPUT_FILE = "exp_ef_parallel_output.txt"
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
print("EXP-E + EXP-F (PARALLEL SPLIT)")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# EXP-E: KC SQUEEZE → EXPANSION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("EXP-E: KC SQUEEZE -> EXPANSION CONFIDENCE SCORING")
print("=" * 80)


def add_bollinger(df, period=20, std_mult=2.0):
    df = df.copy()
    df['BB_mid'] = df['Close'].rolling(period).mean()
    df['BB_std'] = df['Close'].rolling(period).std()
    df['BB_upper'] = df['BB_mid'] + std_mult * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - std_mult * df['BB_std']
    df['squeeze'] = (df['BB_lower'] > df['KC_lower']) & (df['BB_upper'] < df['KC_upper'])
    squeeze_count = 0
    squeeze_bars_list = []
    for sq in df['squeeze']:
        squeeze_count = squeeze_count + 1 if sq else 0
        squeeze_bars_list.append(squeeze_count)
    df['squeeze_bars'] = squeeze_bars_list
    df['squeeze_release'] = (~df['squeeze']) & (df['squeeze'].shift(1) == True)
    return df


data_sq = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
data_sq.h1_df = add_bollinger(data_sq.h1_df)
data_sq.m15_df = add_bollinger(data_sq.m15_df)

h1_squeezes = data_sq.h1_df['squeeze'].sum()
h1_releases = data_sq.h1_df['squeeze_release'].sum()
print(f"\n  H1: {h1_squeezes} squeeze bars, {h1_releases} squeeze releases")
print(f"  H1: squeeze rate = {h1_squeezes / len(data_sq.h1_df) * 100:.1f}%")

s_base = run_variant(data_sq, "E_baseline", verbose=True, **BASE, spread_cost=0.30)
trades = s_base.get('_trades', [])

squeeze_trades = []
nonsqueeze_trades = []
for t in trades:
    if t.strategy != 'keltner':
        continue
    entry_time = t.entry_time
    h1_mask = data_sq.h1_df.index <= pd.Timestamp(entry_time)
    if h1_mask.any():
        h1_idx = h1_mask.sum() - 1
        if 0 <= h1_idx < len(data_sq.h1_df):
            start_idx = max(0, h1_idx - 5)
            recent_squeeze = data_sq.h1_df.iloc[start_idx:h1_idx + 1]['squeeze_release'].any()
            if recent_squeeze:
                squeeze_trades.append(t)
            else:
                nonsqueeze_trades.append(t)

for label, tlist in [("Post-squeeze (5 bars)", squeeze_trades), ("Non-squeeze", nonsqueeze_trades)]:
    if not tlist:
        print(f"\n  {label}: N=0")
        continue
    n = len(tlist)
    pnl = sum(t.pnl for t in tlist)
    avg = pnl / n
    wr = sum(1 for t in tlist if t.pnl > 0) / n * 100
    print(f"  {label}: N={n}, PnL=${pnl:.0f}, $/t=${avg:.2f}, WR={wr:.1f}%")

# Also test tighter windows
for window in [3, 5, 8, 12]:
    sq_t = []
    for t in trades:
        if t.strategy != 'keltner':
            continue
        h1_mask = data_sq.h1_df.index <= pd.Timestamp(t.entry_time)
        if h1_mask.any():
            h1_idx = h1_mask.sum() - 1
            if 0 <= h1_idx < len(data_sq.h1_df):
                start_idx = max(0, h1_idx - window)
                recent = data_sq.h1_df.iloc[start_idx:h1_idx + 1]['squeeze_release'].any()
                if recent:
                    sq_t.append(t)
    if sq_t:
        n = len(sq_t)
        pnl = sum(t.pnl for t in sq_t)
        avg = pnl / n
        wr = sum(1 for t in sq_t if t.pnl > 0) / n * 100
        print(f"  Window={window}: N={n}, PnL=${pnl:.0f}, $/t=${avg:.2f}, WR={wr:.1f}%")

del data_sq
gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# EXP-F: MULTI-KC ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("EXP-F: MULTI-KC ENSEMBLE — Signal Overlap Scoring")
print("=" * 80)

KC_CONFIGS = [
    (20, 1.5, "KC20_15"),
    (25, 1.2, "KC25_12"),
    (30, 1.0, "KC30_10"),
]

kc_trade_sets = {}
for ema, mult, name in KC_CONFIGS:
    d = DataBundle.load_custom(kc_ema=ema, kc_mult=mult)
    s = run_variant(d, f"F_{name}", verbose=False, **BASE, spread_cost=0.30)
    kc_trades = [t for t in s.get('_trades', []) if t.strategy == 'keltner']
    entry_keys = set()
    for t in kc_trades:
        key = (pd.Timestamp(t.entry_time).floor('h'), t.direction)
        entry_keys.add(key)
    kc_trade_sets[name] = {'stats': s, 'trades': kc_trades, 'entry_keys': entry_keys}

    n = s['n']
    avg = s['total_pnl'] / n if n > 0 else 0
    kc_n = len(kc_trades)
    kc_pnl = sum(t.pnl for t in kc_trades)
    kc_avg = kc_pnl / kc_n if kc_n > 0 else 0
    print(f"\n  {name}: Total N={n}, Sharpe={s['sharpe']:.2f}, "
          f"KC trades={kc_n}, KC $/t=${kc_avg:.2f}")
    del d
    gc.collect()

primary = kc_trade_sets["KC25_12"]
overlap_3 = []
overlap_2 = []
single = []

for t in primary['trades']:
    key = (pd.Timestamp(t.entry_time).floor('h'), t.direction)
    count = sum(1 for name in kc_trade_sets if key in kc_trade_sets[name]['entry_keys'])
    if count >= 3:
        overlap_3.append(t)
    elif count >= 2:
        overlap_2.append(t)
    else:
        single.append(t)

for label, tlist in [("3-config overlap", overlap_3), ("2-config overlap", overlap_2), ("Single config", single)]:
    if not tlist:
        print(f"\n  {label}: N=0")
        continue
    n = len(tlist)
    pnl = sum(t.pnl for t in tlist)
    avg = pnl / n
    wr = sum(1 for t in tlist if t.pnl > 0) / n * 100
    print(f"\n  {label}: N={n}, PnL=${pnl:.0f}, $/t=${avg:.2f}, WR={wr:.1f}%")


elapsed = time.time() - t_total
print(f"\nTotal runtime: {elapsed / 60:.1f} minutes")
print(f"Completed: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"Results saved to {OUTPUT_FILE}")
