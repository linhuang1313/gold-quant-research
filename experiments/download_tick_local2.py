"""
Download XAUUSD tick data locally - memory efficient version.
Writes to CSV incrementally.
"""

import urllib.request, lzma, struct, time, csv
from pathlib import Path
from datetime import datetime, timedelta

OUT_DIR = Path("data/tick")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCALE = 1000

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
            ticks.append((str(tick_time), ask, bid, row[3], row[4], ask - bid, (ask + bid) / 2))
        return ticks
    except:
        return []


def main():
    start = datetime(2026, 1, 1)
    end = datetime(2026, 4, 1)

    out_path = OUT_DIR / "xauusd_ticks_2026q1.csv"
    print("Downloading XAUUSD tick data: %s to %s" % (start.date(), end.date()))

    current = start
    total_ticks = 0
    day_count = 0

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'ask', 'bid', 'ask_vol', 'bid_vol', 'spread', 'mid'])

        while current < end:
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            day_ticks = 0
            for hour in range(24):
                dt = datetime(current.year, current.month, current.day, hour)
                ticks = download_hour(dt)
                if ticks:
                    writer.writerows(ticks)
                    day_ticks += len(ticks)
                time.sleep(0.05)

            day_count += 1
            total_ticks += day_ticks

            if day_count % 5 == 0:
                print("  %s: day %d, %s total ticks" % (
                    current.date(), day_count, "{:,}".format(total_ticks)))
                f.flush()

            current += timedelta(days=1)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print("\nDone! %d days, %s ticks, %.1f MB" % (day_count, "{:,}".format(total_ticks), size_mb))


if __name__ == "__main__":
    main()
