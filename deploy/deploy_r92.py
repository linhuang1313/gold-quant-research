"""Deploy R92 ML Exit Robustness Suite to server."""
import paramiko, os, time, sys

sys.stdout.reconfigure(encoding='utf-8')

HOST = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASS = '3sCdENtzYfse'
REMOTE_BASE = '/root/gold-quant-research'
LOCAL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES_TO_SYNC = [
    'experiments/run_r92_ml_exit_robustness.py',
    'backtest/engine.py',
    'backtest/runner.py',
    'backtest/stats.py',
    'backtest/__init__.py',
    'indicators.py',
    'research_config.py',
    'data/external/aligned_daily.csv',
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


def ssh(c, cmd, timeout=60):
    _, out, err = c.exec_command(cmd, timeout=timeout)
    return out.read().decode('utf-8', errors='replace').strip()


print("=" * 70, flush=True)
print("  R92 Deployment — ML Exit Robustness Verification Suite", flush=True)
print("=" * 70, flush=True)

# Step 1: Upload files
print("\n=== Step 1: Upload files ===", flush=True)
for rel_path in FILES_TO_SYNC:
    local = os.path.join(LOCAL_BASE, rel_path)
    remote = f'{REMOTE_BASE}/{rel_path}'
    if not os.path.exists(local):
        print(f"  SKIP (not found): {rel_path}", flush=True)
        continue
    size_kb = os.path.getsize(local) / 1024
    for attempt in range(5):
        try:
            c = connect()
            remote_dir = os.path.dirname(remote).replace('\\', '/')
            ssh(c, f"mkdir -p {remote_dir}")
            sftp = c.open_sftp()
            sftp.put(local, remote)
            sftp.close()
            c.close()
            print(f"  OK: {rel_path} ({size_kb:.1f} KB)", flush=True)
            break
        except Exception as e:
            print(f"  Retry {attempt+1} for {rel_path}: {e}", flush=True)
            time.sleep(15)
    time.sleep(1)

# Step 2: Check dependencies
print("\n=== Step 2: Check dependencies ===", flush=True)
c = connect()
xgb_ver = ssh(c, "python3 -c \"import xgboost; print('XGBoost:', xgboost.__version__)\" 2>/dev/null")
print(f"  {xgb_ver}", flush=True)
sklearn_ver = ssh(c, "python3 -c \"import sklearn; print('scikit-learn:', sklearn.__version__)\" 2>/dev/null")
print(f"  {sklearn_ver}", flush=True)
gpu_info = ssh(c, "python3 -c \"import torch; print('CUDA:', torch.cuda.is_available())\" 2>/dev/null")
print(f"  {gpu_info}", flush=True)
c.close()

# Step 3: Create output dir, kill old processes, launch
print("\n=== Step 3: Launch R92 ===", flush=True)
time.sleep(3)
c = connect()
ssh(c, f"mkdir -p {REMOTE_BASE}/results/r92_robustness")
ssh(c, "kill $(pgrep -f run_r92) 2>/dev/null")
c.close()

time.sleep(3)
c = connect()
launch_cmd = (
    f"cd {REMOTE_BASE} && "
    "nohup python3 -u experiments/run_r92_ml_exit_robustness.py "
    "> results/r92_robustness/r92_stdout.txt 2>&1 &"
)
print(f"  Launching R92...", flush=True)
chan = c.get_transport().open_session()
chan.exec_command(launch_cmd)
time.sleep(5)
chan.close()
c.close()

# Step 4: Verify process started
time.sleep(15)
c = connect()
proc = ssh(c, "ps aux | grep run_r92 | grep -v grep")
if proc:
    print(f"\n  Process running: YES", flush=True)
    for line in proc.split('\n')[:3]:
        print(f"    {line.strip()}", flush=True)
else:
    print(f"\n  Process running: NO (check logs)", flush=True)

out = ssh(c, f"head -30 {REMOTE_BASE}/results/r92_robustness/r92_stdout.txt 2>/dev/null")
print(f"\n  Initial output:\n{out}", flush=True)
c.close()

print(f"\n{'='*70}", flush=True)
print("  R92 Deployed! Expected runtime: 10-15 minutes", flush=True)
print("  Monitor: ssh -p 41109 root@connect.westd.seetacloud.com", flush=True)
print("  Check:   tail -f /root/gold-quant-research/results/r92_robustness/r92_stdout.txt", flush=True)
print(f"{'='*70}", flush=True)
