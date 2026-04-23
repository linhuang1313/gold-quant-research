"""
Download XAUUSD tick data locally (Dukascopy).
Sample: 2026-01 to 2026-03 (3 months).
"""

import urllib.request, lzma, struct, time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

OUT_DIR = Path("data/tick")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCALE = 1000  # XAUUSD uses 1e3 for Dukascopy

def download_hour(dt):
    url = ("https://datafeed.dukascopy.com/datafeed/XAUUSD/"
           "%d/%02d/%02d/%02dh_ticks.bi5" % (dt.year, dt.month-1, dt.day, dt.hour))
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=30)
        data = resp.read()
        if len(data) < 20:
            return []
        dec = lzma.decompress(data)
        n = len(dec) // 20
        ticks = []
        for i in range(n):
            row = struct.unpack('>IIIff', dec[i*20:(i+1)*20])
            tick_time = dt + timedelta(milliseconds=row[0])
            ask = row[1] / SCALE
            bid = row[2] / SCALE
            ask_vol = row[3]
            bid_vol = row[4]
            ticks.append((tick_time, ask, bid, ask_vol, bid_vol))
        return ticks
    except:
        return []


def main():
    start = datetime(2026, 1, 1)
    end = datetime(2026, 4, 1)

    print("Downloading XAUUSD tick data: %s to %s" % (start.date(), end.date()))
    print("(3 months sample for R34 analysis)\n")

    current = start
    all_data = []
    day_count = 0
    total_ticks = 0

    while current < end:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        day_ticks = 0
        for hour in range(24):
            dt = datetime(current.year, current.month, current.day, hour)
            ticks = download_hour(dt)
            all_data.extend(ticks)
            day_ticks += len(ticks)
            time.sleep(0.05)

        day_count += 1
        total_ticks += day_ticks

        if day_count % 5 == 0:
            print("  %s: day %d, %d ticks today, %s total" % (
                current.date(), day_count, day_ticks, "{:,}".format(total_ticks)))

        current += timedelta(days=1)

    print("\nTotal: %d days, %s ticks" % (day_count, "{:,}".format(total_ticks)))

    if all_data:
        df = pd.DataFrame(all_data, columns=['timestamp', 'ask', 'bid', 'ask_vol', 'bid_vol'])
        df['spread'] = df['ask'] - df['bid']
        df['mid'] = (df['ask'] + df['bid']) / 2

        out_path = OUT_DIR / "xauusd_ticks_2026q1.csv"
        df.to_csv(out_path, index=False)
        size_mb = out_path.stat().st_size / 1024 / 1024
        print("Saved to %s (%.1f MB)" % (out_path, size_mb))
        print("Spread: mean=$%.4f, median=$%.4f" % (df['spread'].mean(), df['spread'].median()))


if __name__ == "__main__":
    main()
