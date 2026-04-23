"""Fix and retry tick download with better debugging."""
import paramiko, sys, io, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
for attempt in range(3):
    try:
        ssh.connect('connect.bjb1.seetacloud.com', port=45411, username='root',
                    password='5zQ8khQzttDN', timeout=120, banner_timeout=120)
        break
    except Exception as e:
        print(f"  SSH attempt {attempt+1} failed: {e}")
        time.sleep(5)

PY = "/root/miniconda3/bin/python"
BASE = "/root/gold-quant-research"

def run(cmd, timeout=60):
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    out = o.read().decode(errors='replace').strip()
    err = e.read().decode(errors='replace').strip()
    return out, err

# Kill existing tick download
run('pkill -f download_tick 2>/dev/null')
time.sleep(2)

# Quick test: try to download just 1 hour of tick data
test_code = '''
import urllib.request, lzma, struct
from datetime import datetime, timedelta

url = "https://datafeed.dukascopy.com/datafeed/XAUUSD/2025/00/02/00h_ticks.bi5"
print(f"Testing: {url}")
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req, timeout=30)
    data = resp.read()
    print(f"Response: {len(data)} bytes")
    if len(data) > 10:
        dec = lzma.decompress(data)
        n = len(dec) // 20
        print(f"Ticks: {n}")
        if n > 0:
            row = struct.unpack('>IIIff', dec[:20])
            print(f"First tick: ts={row[0]}, ask_raw={row[1]}, bid_raw={row[2]}")
            # XAUUSD price offset
            ask = row[1] / 1e3  # try 1e3 for gold (not 1e5)
            bid = row[2] / 1e3
            print(f"  ask={ask}, bid={bid} (1e3 scale)")
            ask2 = row[1] / 1e5
            bid2 = row[2] / 1e5
            print(f"  ask={ask2}, bid={bid2} (1e5 scale)")
            ask3 = row[1] / 100
            bid3 = row[2] / 100
            print(f"  ask={ask3}, bid={bid3} (1e2 scale)")
    else:
        print("Empty data")
except Exception as e:
    print(f"Error: {e}")

# Also try a more recent date
url2 = "https://datafeed.dukascopy.com/datafeed/XAUUSD/2026/02/01/10h_ticks.bi5"
print(f"\\nTesting: {url2}")
try:
    req = urllib.request.Request(url2, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req, timeout=30)
    data = resp.read()
    print(f"Response: {len(data)} bytes")
    if len(data) > 10:
        dec = lzma.decompress(data)
        n = len(dec) // 20
        print(f"Ticks: {n}")
except Exception as e:
    print(f"Error: {e}")
'''

print("[TEST] Testing Dukascopy tick API...")
out, err = run(f'{PY} -c "{test_code}"', timeout=60)
print(out)
if err: print(f"STDERR: {err}")

ssh.close()
