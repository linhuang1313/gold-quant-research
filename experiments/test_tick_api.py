"""Quick test of Dukascopy tick API."""
import urllib.request, lzma, struct
from datetime import datetime, timedelta

url = "https://datafeed.dukascopy.com/datafeed/XAUUSD/2025/00/02/10h_ticks.bi5"
print("Testing:", url)
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req, timeout=30)
    data = resp.read()
    print("Response:", len(data), "bytes")
    if len(data) > 10:
        dec = lzma.decompress(data)
        n = len(dec) // 20
        print("Ticks:", n)
        if n > 0:
            for i in range(min(3, n)):
                row = struct.unpack('>IIIff', dec[i*20:(i+1)*20])
                print("  raw:", row)
                # Try different scale factors for XAUUSD
                for scale in [1, 10, 100, 1000, 1e4, 1e5]:
                    ask = row[1] / scale
                    bid = row[2] / scale
                    if 1000 < ask < 5000:
                        print("  MATCH scale=%s: ask=%.3f, bid=%.3f, spread=%.4f" % (scale, ask, bid, ask-bid))
    else:
        print("Empty data (market closed?)")
except Exception as e:
    print("Error:", e)

# Try another date/time
url2 = "https://datafeed.dukascopy.com/datafeed/XAUUSD/2026/02/03/14h_ticks.bi5"
print("\nTesting:", url2)
try:
    req = urllib.request.Request(url2, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req, timeout=30)
    data = resp.read()
    print("Response:", len(data), "bytes")
    if len(data) > 10:
        dec = lzma.decompress(data)
        n = len(dec) // 20
        print("Ticks:", n)
        if n > 0:
            row = struct.unpack('>IIIff', dec[:20])
            print("  raw:", row)
            for scale in [1, 10, 100, 1000, 1e4, 1e5]:
                ask = row[1] / scale
                bid = row[2] / scale
                if 1000 < ask < 5000:
                    print("  MATCH scale=%s: ask=%.3f, bid=%.3f" % (scale, ask, bid))
except Exception as e:
    print("Error:", e)
