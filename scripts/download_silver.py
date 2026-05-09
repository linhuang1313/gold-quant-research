#!/usr/bin/env python3
"""Download XAGUSD + XAUUSD data for Gold/Silver pair trading research."""
import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("  Downloading Silver & Gold data for pair trading research")
print("=" * 60)

# H1 Silver (max 730 days on yfinance)
print("\n1. XAGUSD H1 (Silver Futures)...")
silver_h1 = yf.download("SI=F", period="730d", interval="1h", progress=True)
if isinstance(silver_h1.columns, pd.MultiIndex):
    silver_h1.columns = silver_h1.columns.get_level_values(0)
silver_h1.to_csv(DATA_DIR / "xagusd_h1_yf.csv")
print(f"   {len(silver_h1)} bars: {silver_h1.index[0]} ~ {silver_h1.index[-1]}")

# Daily Silver (max history)
print("\n2. XAGUSD Daily (max history)...")
silver_d = yf.download("SI=F", period="max", interval="1d", progress=True)
if isinstance(silver_d.columns, pd.MultiIndex):
    silver_d.columns = silver_d.columns.get_level_values(0)
silver_d.to_csv(DATA_DIR / "xagusd_daily_yf.csv")
print(f"   {len(silver_d)} bars: {silver_d.index[0]} ~ {silver_d.index[-1]}")

# Daily Gold (max history)
print("\n3. XAUUSD Daily (max history)...")
gold_d = yf.download("GC=F", period="max", interval="1d", progress=True)
if isinstance(gold_d.columns, pd.MultiIndex):
    gold_d.columns = gold_d.columns.get_level_values(0)
gold_d.to_csv(DATA_DIR / "xauusd_daily_yf.csv")
print(f"   {len(gold_d)} bars: {gold_d.index[0]} ~ {gold_d.index[-1]}")

# H1 Gold (for H1-level pair trading)
print("\n4. XAUUSD H1 (Gold Futures, 730d)...")
gold_h1 = yf.download("GC=F", period="730d", interval="1h", progress=True)
if isinstance(gold_h1.columns, pd.MultiIndex):
    gold_h1.columns = gold_h1.columns.get_level_values(0)
gold_h1.to_csv(DATA_DIR / "xauusd_h1_yf.csv")
print(f"   {len(gold_h1)} bars: {gold_h1.index[0]} ~ {gold_h1.index[-1]}")

# Gold/Silver Ratio analysis
print("\n" + "=" * 60)
print("  Gold/Silver Ratio Analysis")
print("=" * 60)

merged = pd.DataFrame({
    "gold": gold_d["Close"],
    "silver": silver_d["Close"]
}).dropna()
merged["ratio"] = merged["gold"] / merged["silver"]

r = merged["ratio"]
print(f"  Data points:  {len(r)}")
print(f"  Current:      {r.iloc[-1]:.1f}")
print(f"  Mean:         {r.mean():.1f}")
print(f"  Std:          {r.std():.1f}")
print(f"  Min:          {r.min():.1f} ({r.idxmin().date()})")
print(f"  Max:          {r.max():.1f} ({r.idxmax().date()})")
print(f"  25th pctile:  {r.quantile(0.25):.1f}")
print(f"  75th pctile:  {r.quantile(0.75):.1f}")

# Rolling stats
r20 = r.rolling(20).mean()
r_std20 = r.rolling(20).std()
z_score = (r - r20) / r_std20
print(f"\n  Current Z-score (20d): {z_score.iloc[-1]:.2f}")
print(f"  Z > 2 count:  {(z_score > 2).sum()} ({(z_score > 2).sum()/len(z_score)*100:.1f}%)")
print(f"  Z < -2 count: {(z_score < -2).sum()} ({(z_score < -2).sum()/len(z_score)*100:.1f}%)")

# Year-by-year ratio range
print(f"\n  Year-by-year ratio range:")
for year in range(2015, 2027):
    yr = r[r.index.year == year]
    if len(yr) > 0:
        print(f"    {year}: mean={yr.mean():.1f}, min={yr.min():.1f}, max={yr.max():.1f}, "
              f"range={yr.max()-yr.min():.1f}")

merged.to_csv(DATA_DIR / "gold_silver_ratio.csv")
print(f"\n  Saved: {DATA_DIR / 'gold_silver_ratio.csv'}")
print("\nDone!")
