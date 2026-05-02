"""Deploy R90 Full External Data Test to server."""
import paramiko, os, time, sys

sys.stdout.reconfigure(encoding='utf-8')

HOST = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASS = '3sCdENtzYfse'
REMOTE_BASE = '/root/gold-quant-research'
LOCAL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES_TO_SYNC = [
    # R90 scripts
    'experiments/run_r90_full.py',
    'experiments/run_r90a_regime.py',
    'experiments/run_r90b_factor_filter.py',
    'experiments/run_r90c_ml_direction.py',
    'experiments/run_r90d_ml_exit.py',
    'experiments/run_r90e_portfolio.py',
    # Core backtest engine
    'backtest/engine.py',
    'backtest/runner.py',
    'backtest/stats.py',
    'backtest/__init__.py',
    'indicators.py',
    'research_config.py',
    # External data
    'data/download_external.py',
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
print("  R90 Deployment", flush=True)
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
            # Ensure remote directory exists
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

# Step 2: Install dependencies
print("\n=== Step 2: Install dependencies ===", flush=True)
c = connect()
deps = "hmmlearn shap"
print(f"  Installing: {deps}", flush=True)
result = ssh(c, f"pip install {deps} -q 2>&1 | tail -5", timeout=120)
print(f"  {result}", flush=True)

# Check GPU availability
print("\n  Checking GPU...", flush=True)
gpu_info = ssh(c, "python3 -c \"import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')\" 2>/dev/null")
print(f"  {gpu_info}", flush=True)

xgb_gpu = ssh(c, "python3 -c \"import xgboost; print('XGBoost:', xgboost.__version__)\" 2>/dev/null")
print(f"  {xgb_gpu}", flush=True)

lgb_gpu = ssh(c, "python3 -c \"import lightgbm; print('LightGBM:', lightgbm.__version__)\" 2>/dev/null")
print(f"  {lgb_gpu}", flush=True)
c.close()

# Step 3: Create output dirs, kill old processes, launch
print("\n=== Step 3: Launch R90 ===", flush=True)
time.sleep(3)
c = connect()
ssh(c, f"mkdir -p {REMOTE_BASE}/results/r90_external_data/r90a_regime")
ssh(c, f"mkdir -p {REMOTE_BASE}/results/r90_external_data/r90b_factor_filter")
ssh(c, f"mkdir -p {REMOTE_BASE}/results/r90_external_data/r90c_ml_direction")
ssh(c, f"mkdir -p {REMOTE_BASE}/results/r90_external_data/r90d_ml_exit")
ssh(c, f"mkdir -p {REMOTE_BASE}/results/r90_external_data/r90e_portfolio")
ssh(c, "kill $(pgrep -f run_r90) 2>/dev/null")
c.close()

time.sleep(3)
c = connect()
launch_cmd = (
    f"cd {REMOTE_BASE} && "
    "nohup python3 -u experiments/run_r90_full.py "
    "> results/r90_external_data/r90_stdout.txt 2>&1 &"
)
print(f"  Launching R90...", flush=True)
chan = c.get_transport().open_session()
chan.exec_command(launch_cmd)
time.sleep(5)
chan.close()
c.close()

# Step 4: Verify
time.sleep(15)
c = connect()
proc = ssh(c, "ps aux | grep run_r90 | grep -v grep")
if proc:
    print(f"\n  Process running: YES", flush=True)
    for line in proc.split('\n')[:3]:
        print(f"    {line.strip()}", flush=True)
else:
    print(f"\n  Process running: NO (check logs)", flush=True)

out = ssh(c, f"head -30 {REMOTE_BASE}/results/r90_external_data/r90_stdout.txt 2>/dev/null")
print(f"\n  Initial output:\n{out}", flush=True)
c.close()

print(f"\n{'='*70}", flush=True)
print("  R90 Deployed! Expected runtime: 32-48 hours", flush=True)
print("  Monitor with: python deploy\\_r90_check.py", flush=True)
print(f"{'='*70}", flush=True)
