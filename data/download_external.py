"""
R20 Phase 1: Download and align external data sources for signal quality scoring.

Data sources:
  - VIX (CBOE daily, 1990+)
  - DXY (yfinance, 2003+)
  - US 10Y yield (yfinance, 2003+)
  - GLD ETF volume (yfinance, 2004+)
  - COT Managed Money positioning (CFTC weekly, 2006+)

All outputs saved to data/external/ as CSV with DatetimeIndex.
"""

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

OUT_DIR = os.path.join(os.path.dirname(__file__), "external")
os.makedirs(OUT_DIR, exist_ok=True)

START = "2006-01-01"
END = datetime.now().strftime("%Y-%m-%d")


def download_vix():
    """VIX from CBOE official CSV."""
    print("[VIX] Downloading from CBOE...")
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    try:
        df = pd.read_csv(url, parse_dates=["DATE"], index_col="DATE")
        df.index.name = "Date"
        df = df.rename(columns={"OPEN": "VIX_Open", "HIGH": "VIX_High",
                                "LOW": "VIX_Low", "CLOSE": "VIX_Close"})
        df = df.loc[START:]
        path = os.path.join(OUT_DIR, "vix_daily.csv")
        df.to_csv(path)
        print(f"  Saved {len(df)} rows to {path}")
        print(f"  Range: {df.index[0]} ~ {df.index[-1]}")
        return df
    except Exception as e:
        print(f"  [FAIL] VIX download: {e}")
        # Fallback to yfinance
        print("  Trying yfinance ^VIX fallback...")
        df = yf.download("^VIX", start=START, end=END, progress=False)
        if len(df) > 0:
            df = df.rename(columns={"Open": "VIX_Open", "High": "VIX_High",
                                    "Low": "VIX_Low", "Close": "VIX_Close"})
            df = df[["VIX_Open", "VIX_High", "VIX_Low", "VIX_Close"]]
            path = os.path.join(OUT_DIR, "vix_daily.csv")
            df.to_csv(path)
            print(f"  Saved {len(df)} rows to {path}")
            return df
        return None


def download_yfinance(ticker, prefix, filename):
    """Generic yfinance daily downloader."""
    print(f"[{prefix}] Downloading {ticker} from yfinance...")
    df = yf.download(ticker, start=START, end=END, progress=False)
    if df is None or len(df) == 0:
        print(f"  [FAIL] No data for {ticker}")
        return None

    # yfinance may return MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    rename_map = {}
    for col in df.columns:
        rename_map[col] = f"{prefix}_{col}"
    df = df.rename(columns=rename_map)
    df.index.name = "Date"

    path = os.path.join(OUT_DIR, filename)
    df.to_csv(path)
    print(f"  Saved {len(df)} rows to {path}")
    print(f"  Range: {df.index[0]} ~ {df.index[-1]}")
    return df


def download_cot():
    """COT Managed Money gold positioning from CFTC."""
    print("[COT] Downloading from CFTC...")
    try:
        from cot_reports import cot_all
        df = cot_all(cot_report_type="disaggregated_fut")

        # Detect column naming convention (spaces vs underscores)
        market_col = None
        date_col = None
        for c in df.columns:
            if "market" in c.lower() and "exchange" in c.lower():
                market_col = c
            if "report_date" in c.lower() or "yyyy-mm-dd" in c.lower():
                date_col = c
        if market_col is None:
            market_col = df.columns[0]
        if date_col is None:
            date_col = df.columns[2]

        print(f"  Using market col: {market_col}")
        print(f"  Using date col: {date_col}")

        gold_mask = df[market_col].str.contains("GOLD", case=False, na=False)
        gold = df[gold_mask].copy()

        if len(gold) == 0:
            print("  [FAIL] No gold rows found in COT data")
            return None

        gold["Date"] = pd.to_datetime(gold[date_col])
        gold = gold.set_index("Date").sort_index()

        # Find Managed Money long/short columns
        long_col = short_col = None
        for col in gold.columns:
            if "M_Money_Positions_Long_All" in col:
                long_col = col
            elif "M_Money_Positions_Short_All" in col:
                short_col = col

        if long_col is None or short_col is None:
            # Broader search
            for col in gold.columns:
                cl = col.lower()
                if "m_money" in cl and "long" in cl and "all" in cl and "spread" not in cl:
                    long_col = long_col or col
                if "m_money" in cl and "short" in cl and "all" in cl and "spread" not in cl:
                    short_col = short_col or col

        if long_col is None or short_col is None:
            money_cols = [c for c in gold.columns if "money" in c.lower() or "M_Money" in c]
            print(f"  [WARN] Cannot find MM cols. Candidates: {money_cols[:15]}")
            return None

        print(f"  MM Long col: {long_col}")
        print(f"  MM Short col: {short_col}")

        result = pd.DataFrame({
            "COT_MM_Long": pd.to_numeric(gold[long_col], errors="coerce"),
            "COT_MM_Short": pd.to_numeric(gold[short_col], errors="coerce"),
        }, index=gold.index)
        result["COT_MM_Net"] = result["COT_MM_Long"] - result["COT_MM_Short"]
        result = result[~result.index.duplicated(keep="last")]

        path = os.path.join(OUT_DIR, "cot_gold_weekly.csv")
        result.to_csv(path)
        print(f"  Saved {len(result)} rows to {path}")
        print(f"  Range: {result.index[0]} ~ {result.index[-1]}")
        return result

    except Exception as e:
        print(f"  [FAIL] COT download: {e}")
        import traceback
        traceback.print_exc()
        return None


def build_aligned_daily():
    """Merge all external data into a single daily DataFrame."""
    print("\n[ALIGN] Building aligned daily dataset...")

    frames = {}
    for fname, key in [("vix_daily.csv", "VIX"),
                        ("dxy_daily.csv", "DXY"),
                        ("us10y_daily.csv", "US10Y"),
                        ("gld_daily.csv", "GLD"),
                        ("cot_gold_weekly.csv", "COT")]:
        path = os.path.join(OUT_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
            frames[key] = df
            print(f"  Loaded {key}: {len(df)} rows")
        else:
            print(f"  [SKIP] {fname} not found")

    if not frames:
        print("  [FAIL] No data to align")
        return None

    # Build daily index from 2006 to now
    date_range = pd.bdate_range(start=START, end=END)
    aligned = pd.DataFrame(index=date_range)
    aligned.index.name = "Date"

    for key, df in frames.items():
        # Forward fill weekly/missing data to daily
        aligned = aligned.join(df, how="left")

    # Forward-fill COT (weekly → daily) and other gaps
    aligned = aligned.ffill()

    # Compute derived features
    if "VIX_Close" in aligned.columns:
        aligned["VIX_SMA20"] = aligned["VIX_Close"].rolling(20).mean()
        aligned["VIX_Zscore"] = (
            (aligned["VIX_Close"] - aligned["VIX_Close"].rolling(60).mean())
            / aligned["VIX_Close"].rolling(60).std()
        )

    if "DXY_Close" in aligned.columns:
        aligned["DXY_Mom5"] = aligned["DXY_Close"].pct_change(5)
        aligned["DXY_Mom20"] = aligned["DXY_Close"].pct_change(20)

    if "US10Y_Close" in aligned.columns:
        aligned["US10Y_Change5"] = aligned["US10Y_Close"].diff(5)

    if "GLD_Volume" in aligned.columns:
        aligned["GLD_Vol_SMA20"] = aligned["GLD_Volume"].rolling(20).mean()
        aligned["GLD_Vol_Ratio"] = aligned["GLD_Volume"] / aligned["GLD_Vol_SMA20"]

    if "COT_MM_Net" in aligned.columns:
        aligned["COT_MM_Net_Zscore"] = (
            (aligned["COT_MM_Net"] - aligned["COT_MM_Net"].rolling(52).mean())
            / aligned["COT_MM_Net"].rolling(52).std()
        )
        aligned["COT_MM_Net_Pct"] = aligned["COT_MM_Net"].rolling(104).rank(pct=True)

    # Drop rows before first valid data
    aligned = aligned.dropna(how="all")

    path = os.path.join(OUT_DIR, "aligned_daily.csv")
    aligned.to_csv(path)
    print(f"\n  Aligned dataset: {len(aligned)} rows, {len(aligned.columns)} columns")
    print(f"  Range: {aligned.index[0]} ~ {aligned.index[-1]}")
    print(f"  Columns: {list(aligned.columns)}")
    print(f"  Saved to {path}")

    # Summary stats
    print("\n  Non-null counts:")
    for col in aligned.columns:
        nn = aligned[col].notna().sum()
        print(f"    {col}: {nn} ({nn/len(aligned)*100:.0f}%)")

    return aligned


if __name__ == "__main__":
    print("=" * 70)
    print("R20 Phase 1: External Data Download")
    print("=" * 70)
    print(f"Date range: {START} ~ {END}\n")

    download_vix()
    download_yfinance("DX-Y.NYB", "DXY", "dxy_daily.csv")
    download_yfinance("^TNX", "US10Y", "us10y_daily.csv")
    download_yfinance("GLD", "GLD", "gld_daily.csv")
    download_cot()

    print("\n" + "=" * 70)
    build_aligned_daily()
    print("\n" + "=" * 70)
    print("Phase 1 complete.")
