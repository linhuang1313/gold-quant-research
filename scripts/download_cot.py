#!/usr/bin/env python3
"""
Download CFTC Commitment of Traders (COT) data for Gold Futures.
Contract: 088691 (COMEX Gold)

Sources tried in order:
  1. cot_reports library (CFTC bulk download)
  2. Direct CFTC historical CSV
  3. Quandl/Nasdaq Data Link
"""
import sys, os, warnings
import pandas as pd
import numpy as np
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
OUT_FILE = DATA_DIR / "cot_gold_weekly.csv"

GOLD_CONTRACT = "088691"
GOLD_NAME_KEYWORDS = ["GOLD", "088691"]


def try_cot_reports_lib():
    """Method 1: cot_reports library."""
    print("Method 1: cot_reports library...")
    try:
        import cot_reports as cot
        df = cot.cot_all_reports()
        print(f"  Raw columns: {list(df.columns[:10])}...")
        print(f"  Raw rows: {len(df)}")

        id_col = None
        for col in df.columns:
            if 'cftc' in col.lower() and 'code' in col.lower():
                id_col = col; break
            if 'contract' in col.lower() and 'code' in col.lower():
                id_col = col; break
            if 'market' in col.lower() and 'code' in col.lower():
                id_col = col; break

        if id_col is None:
            for col in df.columns:
                sample = df[col].astype(str).head(20)
                if sample.str.contains(GOLD_CONTRACT).any():
                    id_col = col; break

        if id_col:
            print(f"  ID column: {id_col}")
            gold = df[df[id_col].astype(str).str.contains(GOLD_CONTRACT)]
            if len(gold) == 0:
                for col in df.columns:
                    if 'name' in col.lower() or 'market' in col.lower():
                        gold = df[df[col].astype(str).str.upper().str.contains('GOLD')]
                        if len(gold) > 0:
                            print(f"  Found via name column: {col}")
                            break
        else:
            print("  No ID column found, searching by name...")
            for col in df.columns:
                if 'name' in col.lower() or 'market' in col.lower():
                    gold = df[df[col].astype(str).str.upper().str.contains('GOLD')]
                    if len(gold) > 0:
                        print(f"  Found via: {col}")
                        break
            else:
                print("  FAILED: Cannot find gold contract")
                return None

        if len(gold) == 0:
            print("  FAILED: No gold rows found")
            return None

        print(f"  Gold rows: {len(gold)}")
        print(f"  Columns available: {list(gold.columns)}")
        return gold

    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def try_cftc_direct():
    """Method 2: Direct CFTC bulk CSV download."""
    print("\nMethod 2: Direct CFTC historical CSV...")
    try:
        import zipfile, io, urllib.request

        all_data = []
        for year in range(2006, 2027):
            url = f"https://www.cftc.gov/files/dea/history/deacmxsf{year}.zip"
            print(f"  Downloading {year}...", end=" ")
            try:
                resp = urllib.request.urlopen(url, timeout=15)
                z = zipfile.ZipFile(io.BytesIO(resp.read()))
                for name in z.namelist():
                    if name.endswith('.csv') or name.endswith('.txt'):
                        chunk = pd.read_csv(z.open(name), low_memory=False)
                        all_data.append(chunk)
                        print(f"OK ({len(chunk)} rows)")
                        break
            except Exception as e:
                print(f"skip ({e})")
                continue

        if not all_data:
            print("  FAILED: No data downloaded")
            return None

        df = pd.concat(all_data, ignore_index=True)
        print(f"  Total: {len(df)} rows, {len(df.columns)} columns")

        id_col = None
        for col in df.columns:
            if 'cftc' in col.lower() and 'code' in col.lower():
                id_col = col; break

        if id_col is None:
            for col in df.columns:
                sample = df[col].astype(str).head(100)
                if sample.str.contains(GOLD_CONTRACT).any():
                    id_col = col; break

        if id_col:
            gold = df[df[id_col].astype(str).str.contains(GOLD_CONTRACT)]
        else:
            for col in df.columns:
                if 'name' in col.lower() or 'market' in col.lower():
                    gold = df[df[col].astype(str).str.upper().str.contains('GOLD')]
                    if len(gold) > 0: break
            else:
                print("  FAILED: Cannot find gold")
                return None

        print(f"  Gold rows: {len(gold)}")
        return gold

    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def try_quandl():
    """Method 3: Nasdaq Data Link (Quandl)."""
    print("\nMethod 3: Nasdaq Data Link...")
    try:
        urls = [
            "https://data.nasdaq.com/api/v3/datasets/CFTC/088691_FO_ALL/data.csv",
            "https://data.nasdaq.com/api/v3/datasets/CFTC/088691_FO_L_ALL/data.csv",
            "https://data.nasdaq.com/api/v3/datasets/CFTC/GC_FO_ALL/data.csv",
        ]
        for url in urls:
            try:
                print(f"  Trying {url.split('/')[-2]}...", end=" ")
                df = pd.read_csv(url, parse_dates=['Date'])
                if len(df) > 50:
                    print(f"OK ({len(df)} rows)")
                    df = df.set_index('Date').sort_index()
                    return df
                print(f"only {len(df)} rows")
            except Exception as e:
                print(f"fail ({e})")
    except Exception as e:
        print(f"  FAILED: {e}")
    return None


def standardize_cot(raw):
    """Extract key COT columns and compute signals."""
    if raw is None or len(raw) == 0:
        return None

    cols = raw.columns.tolist()
    print(f"\n  Standardizing... ({len(raw)} rows, {len(cols)} cols)")

    date_col = None
    for c in cols:
        if 'date' in c.lower() or 'report' in c.lower():
            if raw[c].dtype == 'datetime64[ns]' or 'date' in c.lower():
                date_col = c; break

    long_col = short_col = oi_col = None
    spread_col = None

    col_mapping = {
        'long': ['noncommercial.*long', 'non.commercial.*long', 'noncomm.*positions.*long',
                 'Non Commercial Long', 'Noncommercial Long'],
        'short': ['noncommercial.*short', 'non.commercial.*short', 'noncomm.*positions.*short',
                  'Non Commercial Short', 'Noncommercial Short'],
        'oi': ['open.*interest', 'oi.*all', 'Open Interest'],
        'spread': ['spread', 'noncommercial.*spread'],
    }

    import re
    for key, patterns in col_mapping.items():
        for pat in patterns:
            for c in cols:
                if re.search(pat, c, re.IGNORECASE):
                    if key == 'long' and long_col is None: long_col = c
                    elif key == 'short' and short_col is None: short_col = c
                    elif key == 'oi' and oi_col is None: oi_col = c
                    elif key == 'spread' and spread_col is None: spread_col = c
                    break

    print(f"  Date: {date_col}")
    print(f"  Long: {long_col}")
    print(f"  Short: {short_col}")
    print(f"  OI: {oi_col}")

    if long_col is None or short_col is None:
        print("  WARNING: Missing long/short columns. Available columns:")
        for c in cols:
            if any(kw in c.lower() for kw in ['long', 'short', 'commercial', 'open', 'interest', 'noncomm']):
                print(f"    {c}: {raw[c].dtype}")
        return None

    result = pd.DataFrame()
    if date_col:
        result.index = pd.to_datetime(raw[date_col])
    else:
        result.index = raw.index

    result['noncomm_long'] = pd.to_numeric(raw[long_col], errors='coerce')
    result['noncomm_short'] = pd.to_numeric(raw[short_col], errors='coerce')
    if oi_col: result['open_interest'] = pd.to_numeric(raw[oi_col], errors='coerce')
    if spread_col: result['noncomm_spread'] = pd.to_numeric(raw[spread_col], errors='coerce')

    result['net_spec'] = result['noncomm_long'] - result['noncomm_short']
    rm = result['net_spec'].rolling(52).mean()
    rs = result['net_spec'].rolling(52).std()
    result['net_spec_z'] = (result['net_spec'] - rm) / rs

    if oi_col:
        result['net_pct_oi'] = result['net_spec'] / result['open_interest'] * 100

    result['net_change'] = result['net_spec'].diff()
    result['net_change_z'] = (result['net_change'] - result['net_change'].rolling(52).mean()) / \
                              result['net_change'].rolling(52).std()

    result = result.sort_index().dropna(subset=['noncomm_long', 'noncomm_short'])
    return result


def main():
    print("=" * 70)
    print("  CFTC COT Gold Data Downloader")
    print("=" * 70)

    raw = try_cot_reports_lib()
    if raw is None or len(raw) == 0:
        raw = try_cftc_direct()
    if raw is None or len(raw) == 0:
        raw = try_quandl()

    if raw is None or len(raw) == 0:
        print("\n  ALL METHODS FAILED")
        print("  Manual download: https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm")
        print("  Look for 'Futures and Options Combined Reports' -> 'Commodities - Chicago Mercantile Exchange'")
        sys.exit(1)

    cot = standardize_cot(raw)
    if cot is None:
        print("\n  Standardization failed")
        sys.exit(1)

    cot.to_csv(OUT_FILE)
    print(f"\n  Saved: {OUT_FILE}")
    print(f"  Period: {cot.index[0].date()} ~ {cot.index[-1].date()}")
    print(f"  Rows: {len(cot)}")
    print(f"  Columns: {list(cot.columns)}")

    print(f"\n  Net Speculative Position stats:")
    print(f"    Mean: {cot['net_spec'].mean():,.0f} contracts")
    print(f"    Std:  {cot['net_spec'].std():,.0f}")
    print(f"    Min:  {cot['net_spec'].min():,.0f}")
    print(f"    Max:  {cot['net_spec'].max():,.0f}")

    if 'net_spec_z' in cot.columns:
        z = cot['net_spec_z'].dropna()
        print(f"  Z-Score range: [{z.min():.2f}, {z.max():.2f}]")
        print(f"  Current Z: {z.iloc[-1]:.2f}")

    print("\n  DONE")


if __name__ == '__main__':
    main()
