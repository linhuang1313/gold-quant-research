#!/usr/bin/env python3
"""Download XAUUSD M30 Bid data from Dukascopy (2015-01-01 -> today)."""
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import dukascopy_python as dk
from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD

ROOT = Path(__file__).resolve().parent.parent
DL_DIR = ROOT / "data" / "download"
DL_DIR.mkdir(parents=True, exist_ok=True)

TODAY = datetime.utcnow().date()
TODAY_STR = TODAY.strftime("%Y-%m-%d")

START = datetime(2015, 1, 1)
END = datetime(TODAY.year, TODAY.month, TODAY.day)
OUT_FILE = DL_DIR / f"xauusd-m30-bid-2015-01-01-{TODAY_STR}.csv"


def to_ms_timestamp(dt_index):
    return dt_index.astype('int64') // 10**6


def fetch_chunked(start, end, chunk_months=6):
    all_dfs = []
    chunk_start = start
    chunk_num = 0

    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=chunk_months * 30), end)
        chunk_num += 1
        print(f"  Chunk {chunk_num}: {chunk_start.strftime('%Y-%m-%d')} -> "
              f"{chunk_end.strftime('%Y-%m-%d')} ...", end="", flush=True)

        retries = 0
        while retries < 3:
            try:
                df = dk.fetch(
                    INSTRUMENT_FX_METALS_XAU_USD,
                    dk.INTERVAL_MIN_30,
                    dk.OFFER_SIDE_BID,
                    chunk_start, chunk_end,
                    max_retries=5
                )
                break
            except Exception as e:
                retries += 1
                print(f" RETRY {retries}/3 ({e})", end="", flush=True)
                time.sleep(5 * retries)
        else:
            print(f" FAILED after 3 retries, skipping")
            chunk_start = chunk_end
            continue

        if df is not None and len(df) > 0:
            all_dfs.append(df)
            print(f" {len(df)} bars", flush=True)
        else:
            print(f" 0 bars", flush=True)

        chunk_start = chunk_end
        time.sleep(1)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs)
    combined = combined[~combined.index.duplicated(keep='first')]
    return combined.sort_index()


def main():
    if OUT_FILE.exists():
        existing = pd.read_csv(OUT_FILE)
        print(f"M30 file already exists: {OUT_FILE.name} ({len(existing)} rows)")
        print("Delete it to re-download.")
        return

    print(f"Downloading XAUUSD M30 BID: {START.date()} -> {END.date()}")
    t0 = time.time()

    df = fetch_chunked(START, END)

    if df.empty:
        print("ERROR: No data fetched!")
        return

    out = pd.DataFrame({
        'timestamp': to_ms_timestamp(df.index),
        'open': df['open'],
        'high': df['high'],
        'low': df['low'],
        'close': df['close'],
        'volume': df['volume'],
    })
    out.to_csv(OUT_FILE, index=False)

    elapsed = time.time() - t0
    print(f"\nSaved: {OUT_FILE.name}")
    print(f"  Rows: {len(out)}")
    print(f"  Size: {OUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
