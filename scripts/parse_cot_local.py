#!/usr/bin/env python3
"""Parse locally downloaded CFTC COT files and extract Gold futures data."""
import sys, warnings
import pandas as pd
import numpy as np
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

ROOT = Path(".")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

legacy_files = [
    "FUT86_16.txt",
    "annual.txt",
]

print("=" * 70)
print("  Parsing CFTC COT Local Files -> Gold Weekly Data")
print("=" * 70)

all_dfs = []
for f in legacy_files:
    fp = ROOT / f
    if not fp.exists():
        print(f"  SKIP: {f} not found")
        continue
    df = pd.read_csv(fp, low_memory=False)
    print(f"  {f}: {df.shape[0]} rows, {df.shape[1]} cols")
    all_dfs.append(df)

if not all_dfs:
    print("ERROR: No files found"); sys.exit(1)

raw = pd.concat(all_dfs, ignore_index=True)
print(f"\n  Combined: {raw.shape[0]} rows")

name_col = 'Market and Exchange Names'
gold = raw[raw[name_col].str.contains('GOLD', na=False, case=False)]
print(f"  Gold rows: {len(gold)}")
print(f"  Unique names: {gold[name_col].unique()}")

gold_fut = gold[gold[name_col].str.contains('COMEX', na=False, case=False)]
if len(gold_fut) == 0:
    gold_fut = gold
print(f"  COMEX Gold rows: {len(gold_fut)}")

date_col = 'As of Date in Form YYYY-MM-DD'
cot = pd.DataFrame()
cot['date'] = pd.to_datetime(gold_fut[date_col])
cot['noncomm_long'] = pd.to_numeric(gold_fut['Noncommercial Positions-Long (All)'], errors='coerce')
cot['noncomm_short'] = pd.to_numeric(gold_fut['Noncommercial Positions-Short (All)'], errors='coerce')
cot['noncomm_spread'] = pd.to_numeric(gold_fut['Noncommercial Positions-Spreading (All)'], errors='coerce')
cot['comm_long'] = pd.to_numeric(gold_fut['Commercial Positions-Long (All)'], errors='coerce')
cot['comm_short'] = pd.to_numeric(gold_fut['Commercial Positions-Short (All)'], errors='coerce')
cot['open_interest'] = pd.to_numeric(gold_fut['Open Interest (All)'], errors='coerce')

cot = cot.set_index('date').sort_index()
cot = cot[~cot.index.duplicated(keep='last')]

cot['net_spec'] = cot['noncomm_long'] - cot['noncomm_short']
cot['net_comm'] = cot['comm_long'] - cot['comm_short']
cot['net_pct_oi'] = cot['net_spec'] / cot['open_interest'] * 100

rm52 = cot['net_spec'].rolling(52).mean()
rs52 = cot['net_spec'].rolling(52).std()
cot['net_spec_z'] = (cot['net_spec'] - rm52) / rs52

cot['net_change'] = cot['net_spec'].diff()
cot['net_change_z'] = (cot['net_change'] - cot['net_change'].rolling(26).mean()) / \
                       cot['net_change'].rolling(26).std()

out = DATA_DIR / "cot_gold_weekly.csv"
cot.to_csv(out)
print(f"\n  Saved: {out}")
print(f"  Period: {cot.index[0].date()} ~ {cot.index[-1].date()}")
print(f"  Rows: {len(cot)} weekly reports")

print(f"\n  Net Speculative Position:")
print(f"    Mean:    {cot['net_spec'].mean():>12,.0f} contracts")
print(f"    Std:     {cot['net_spec'].std():>12,.0f}")
print(f"    Min:     {cot['net_spec'].min():>12,.0f}  ({cot['net_spec'].idxmin().date()})")
print(f"    Max:     {cot['net_spec'].max():>12,.0f}  ({cot['net_spec'].idxmax().date()})")
print(f"    Current: {cot['net_spec'].iloc[-1]:>12,.0f}")

z = cot['net_spec_z'].dropna()
print(f"\n  Z-Score (52-week):")
print(f"    Range: [{z.min():.2f}, {z.max():.2f}]")
print(f"    Current: {z.iloc[-1]:.2f}")
print(f"    >1.5 (very bullish): {(z > 1.5).sum()} weeks ({(z > 1.5).mean()*100:.1f}%)")
print(f"    <-1.5 (very bearish): {(z < -1.5).sum()} weeks ({(z < -1.5).mean()*100:.1f}%)")

print("\n  DONE")
