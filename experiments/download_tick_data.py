"""
Download XAUUSD tick data from Dukascopy for R34 analysis.
Downloads only 2025-2026 as a sample (full history would be too large).

Dukascopy stores tick data in hourly binary files (.bi5 = LZMA compressed).
Each tick: timestamp(ms), askprice(int*1e5 offset), bidprice(int*1e5 offset), askvol(float), bidvol(float)
"""

import sys, os, time, struct, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import urllib.request
import urllib.error

try:
    import lzma
except ImportError:
    print("lzma not available, trying backports.lzma...")
    from backports import lzma

OUT_DIR = Path("data/tick")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://datafeed.dukascopy.com/datafeed/XAUUSD"

def download_hour(dt):
    """Download tick data for one hour."""
    url = f"{BASE_URL}/{dt.year}/{dt.month-1:02d}/{dt.day:02d}/{dt.hour:02d}h_ticks.bi5"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=30)
        data = resp.read()
        if len(data) < 10:
            return []
        decompressed = lzma.decompress(data)
        n_ticks = len(decompressed) // 20
        ticks = []
        for i in range(n_ticks):
            row = struct.unpack('>IIIff', decompressed[i*20:(i+1)*20])
            ts_ms = row[0]
            ask = row[1] / 1e5
            bid = row[2] / 1e5
            ask_vol = row[3]
            bid_vol = row[4]
            tick_time = dt + timedelta(milliseconds=ts_ms)
            ticks.append((tick_time, ask, bid, ask_vol, bid_vol))
        return ticks
    except urllib.error.HTTPError:
        return []
    except Exception as e:
        return []


def download_day(date):
    """Download all ticks for one day (24 hours)."""
    all_ticks = []
    for hour in range(24):
        dt = datetime(date.year, date.month, date.day, hour)
        ticks = download_hour(dt)
        all_ticks.extend(ticks)
    return all_ticks


def main():
    print(f"# Tick Data Download for R34")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Download sample: Jan 2025 - Mar 2026 (about 15 months)
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2026, 4, 1)

    current = start_date
    all_data = []
    day_count = 0
    total_ticks = 0

    while current < end_date:
        if current.weekday() >= 5:  # skip weekends
            current += timedelta(days=1)
            continue

        ticks = download_day(current)
        if ticks:
            all_data.extend(ticks)
            total_ticks += len(ticks)
            day_count += 1

        if day_count % 20 == 0 and day_count > 0:
            print(f"  {current.date()}: {day_count} days, {total_ticks:,} ticks so far")

        current += timedelta(days=1)
        time.sleep(0.1)  # be gentle with the server

    print(f"\n  Total: {day_count} days, {total_ticks:,} ticks")

    if all_data:
        df = pd.DataFrame(all_data, columns=['timestamp', 'ask', 'bid', 'ask_vol', 'bid_vol'])
        df['spread'] = df['ask'] - df['bid']
        df['mid'] = (df['ask'] + df['bid']) / 2

        out_path = OUT_DIR / "xauusd_ticks_2025_2026.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved to {out_path} ({out_path.stat().st_size/1024/1024:.1f} MB)")

        # Quick summary stats
        print(f"\n  Spread stats: mean=${df['spread'].mean():.4f}, "
              f"median=${df['spread'].median():.4f}, "
              f"P95=${df['spread'].quantile(0.95):.4f}")
        print(f"  Ticks per day: {total_ticks/max(day_count,1):.0f}")
    else:
        print("  No data downloaded!")


if __name__ == "__main__":
    main()
