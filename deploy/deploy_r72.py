"""Deploy R72 multi-asset generalization + multi-split OOS to server.

Steps:
1. Upload experiment + framework files
2. Ensure XAGUSD H1 data exists (download via Dukascopy if needed)
3. Launch R72 script
"""
import paramiko, os, time

HOST = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASS = '3sCdENtzYfse'
LOCAL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = [
    'experiments/run_r72_generalization.py',
    'backtest/validator.py',
    'backtest/stats.py',
    'backtest/__init__.py',
    'backtest/runner.py',
    'scripts/download_dukascopy.py',
]

def connect(retries=5):
    for a in range(retries):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=90, banner_timeout=180)
            c.get_transport().set_keepalive(30)
            return c
        except Exception as e:
            print(f"  Connect {a+1}/{retries}: {e}")
            time.sleep(10*(a+1))
    raise RuntimeError("Cannot connect")

c = connect()
print("Connected. Uploading files...")

sftp = c.open_sftp()
for f in FILES:
    local = os.path.join(LOCAL_BASE, f)
    if not os.path.exists(local):
        print(f"  SKIP: {f}"); continue
    remote_dir = os.path.dirname(f"/root/gold-quant-research/{f}")
    try:
        sftp.stat(remote_dir)
    except FileNotFoundError:
        c.exec_command(f"mkdir -p {remote_dir}", timeout=15)
        time.sleep(1)
    sftp.put(local, f"/root/gold-quant-research/{f}")
    print(f"  OK: {f}")
sftp.close()

_, out, _ = c.exec_command(
    "mkdir -p /root/gold-quant-research/results/r72_generalization "
    "/root/gold-quant-research/data/download",
    timeout=15
)
out.read()
print("Dirs created.")

# Check if XAGUSD data already exists
print("\nChecking for XAGUSD H1 data...")
_, out, _ = c.exec_command(
    "ls -la /root/gold-quant-research/data/download/xagusd-h1-bid-*.csv 2>/dev/null",
    timeout=15
)
existing = out.read().decode('utf-8', errors='replace').strip()
if existing:
    print(f"  XAGUSD data found:\n{existing}")
else:
    print("  No XAGUSD data found. Downloading via Dukascopy...")
    print("  This may take 2-3 minutes...", flush=True)

    # Install dukascopy if needed
    _, out, err = c.exec_command(
        "pip install dukascopy-python 2>&1 | tail -5", timeout=120
    )
    print(f"  pip: {out.read().decode('utf-8', errors='replace').strip()}")

    # Download XAGUSD H1 only (bid + ask) for the experiment
    _, out, _ = c.exec_command(
        "cd /root/gold-quant-research && "
        "python3 -c \""
        "import dukascopy_python as dk; "
        "from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAG_USD; "
        "from datetime import datetime; "
        "import pandas as pd, time; "
        "start=datetime(2015,1,1); end=datetime(2026,5,1); "
        "print('Downloading XAGUSD H1 bid...'); "
        "all_dfs=[]; "
        "chunk_start=start; "
        "from datetime import timedelta; "
        "while chunk_start < end: "
        "    chunk_end=min(chunk_start+timedelta(days=180),end); "
        "    try: "
        "        df=dk.fetch(INSTRUMENT_FX_METALS_XAG_USD,dk.INTERVAL_HOUR_1,dk.OFFER_SIDE_BID,chunk_start,chunk_end,max_retries=5); "
        "        all_dfs.append(df) if df is not None and len(df)>0 else None; "
        "        print(f'  {chunk_start.date()} -> {chunk_end.date()}: {len(df) if df is not None else 0} bars'); "
        "    except Exception as e: print(f'  ERROR: {e}'); "
        "    chunk_start=chunk_end; time.sleep(1); "
        "combined=pd.concat(all_dfs).sort_index(); "
        "combined=combined[~combined.index.duplicated(keep='first')]; "
        "out=pd.DataFrame({'timestamp':(combined.index.astype('int64')//10**6),'open':combined['open'],'high':combined['high'],'low':combined['low'],'close':combined['close'],'volume':combined['volume']}); "
        "out.to_csv('data/download/xagusd-h1-bid-2015-01-01-2026-05-01.csv',index=False); "
        "print(f'Saved {len(out)} bars')\"",
        timeout=600
    )
    dl_result = out.read().decode('utf-8', errors='replace').strip()
    print(f"  Download result:\n{dl_result}")

# Launch R72
print("\nLaunching R72...", flush=True)
c.exec_command(
    "cd /root/gold-quant-research && "
    "nohup python3 -u experiments/run_r72_generalization.py "
    "> results/r72_stdout.txt 2>&1 &",
    timeout=10
)
time.sleep(5)

_, out, _ = c.exec_command("ps aux | grep r72 | grep -v grep", timeout=15)
try:
    proc = out.read().decode('utf-8', errors='replace').strip()
    if proc:
        print(f"LAUNCHED OK: {proc[:120]}")
    else:
        print("WARNING: process not found")
except:
    print("Process check timeout, but launch command was sent")

c.close()
print("\nDeploy complete!")
