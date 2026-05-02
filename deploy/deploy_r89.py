"""Deploy R89 Lot Optimizer to server."""
import paramiko, os, time, sys

sys.stdout.reconfigure(encoding='utf-8')

HOST = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASS = '3sCdENtzYfse'
REMOTE_BASE = '/root/gold-quant-research'
LOCAL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES_TO_SYNC = [
    'experiments/run_r89_lot_optimizer.py',
    'backtest/engine.py',
    'backtest/runner.py',
    'backtest/stats.py',
    'backtest/__init__.py',
    'indicators.py',
    'research_config.py',
]

def connect(retries=8):
    for a in range(retries):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(HOST, port=PORT, username=USER, password=PASS,
                      timeout=120, banner_timeout=180, auth_timeout=120)
            c.get_transport().set_keepalive(15)
            return c
        except Exception as e:
            print(f"  Connect attempt {a+1}/{retries}: {e}", flush=True)
            time.sleep(15 * (a + 1))
    raise RuntimeError("Cannot connect after retries")

def ssh(c, cmd, timeout=30):
    _, out, _ = c.exec_command(cmd, timeout=timeout)
    return out.read().decode('utf-8', errors='replace').strip()

# Step 1: Upload files
print("=== Step 1: Upload files ===", flush=True)
for rel_path in FILES_TO_SYNC:
    local = os.path.join(LOCAL_BASE, rel_path)
    remote = f'{REMOTE_BASE}/{rel_path}'
    if not os.path.exists(local):
        print(f"  SKIP: {rel_path}", flush=True)
        continue
    for attempt in range(5):
        try:
            c = connect()
            sftp = c.open_sftp()
            sftp.put(local, remote)
            sftp.close()
            c.close()
            print(f"  OK: {rel_path} ({os.path.getsize(local)/1024:.1f} KB)", flush=True)
            break
        except Exception as e:
            print(f"  Retry {attempt+1} for {rel_path}: {e}", flush=True)
            time.sleep(15)
    time.sleep(2)

# Step 2: Create output dir, kill old, launch
print("\n=== Step 2: Launch R89 ===", flush=True)
time.sleep(3)
c = connect()
ssh(c, f"mkdir -p {REMOTE_BASE}/results/r89_lot_optimizer")
ssh(c, "kill $(pgrep -f r89_lot_optimizer) 2>/dev/null")
c.close()

time.sleep(3)
c = connect()
launch_cmd = (
    f"cd {REMOTE_BASE} && "
    "nohup python3 experiments/run_r89_lot_optimizer.py "
    "> results/r89_stdout.txt 2>&1 &"
)
print(f"  Launching...", flush=True)
chan = c.get_transport().open_session()
chan.exec_command(launch_cmd)
time.sleep(3)
chan.close()
c.close()

# Verify
time.sleep(10)
c = connect()
proc = ssh(c, "ps aux | grep r89_lot_optimizer | grep -v grep")
if proc:
    print(f"  Process running: YES", flush=True)
else:
    print(f"  Process running: NO (may have finished already)", flush=True)
out = ssh(c, f"head -30 {REMOTE_BASE}/results/r89_stdout.txt 2>/dev/null")
print(f"  Initial output:\n{out}", flush=True)
c.close()
print("\nDone!", flush=True)
