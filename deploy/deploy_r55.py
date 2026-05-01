"""Deploy R55 three-direction parallel brute-force to Westd server."""
import paramiko, os, time

HOST = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASS = '3sCdENtzYfse'

LOCAL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES_TO_SYNC = [
    'experiments/run_round55a_mean_reversion.py',
    'experiments/run_round55b_weekly_long.py',
    'experiments/run_round55c_ml_full.py',
    'backtest/engine.py',
    'backtest/runner.py',
    'backtest/fast_screen.py',
    'backtest/stats.py',
    'backtest/__init__.py',
    'indicators.py',
    'research_config.py',
]

def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=60, banner_timeout=120)
    return c

def ssh(c, cmd):
    _, out, _ = c.exec_command(cmd, timeout=60)
    return out.read().decode('utf-8', errors='replace').strip()

# --- Upload files ---
print("=== Syncing files ===")
for rel_path in FILES_TO_SYNC:
    local = os.path.join(LOCAL_BASE, rel_path)
    remote = f"/root/gold-quant-research/{rel_path}"
    if not os.path.exists(local):
        print(f"  SKIP: {rel_path}"); continue
    for attempt in range(5):
        try:
            time.sleep(2)
            c = connect()
            sftp = c.open_sftp()
            sftp.put(local, remote)
            sftp.close(); c.close()
            print(f"  OK: {rel_path} ({os.path.getsize(local)/1024:.1f} KB)")
            break
        except Exception as e:
            print(f"  Retry {attempt+1}: {rel_path} ({e})")
            time.sleep(10)

# --- Install ML deps ---
print("\n=== Installing ML dependencies ===")
time.sleep(3)
c = connect()
out = ssh(c, 'pip install xgboost lightgbm scikit-learn 2>&1 | tail -5')
print(out)
c.close()

# --- Create dirs and start ---
print("\n=== Starting R55 (3 scripts in parallel) ===")
time.sleep(3)
c = connect()
ssh(c, 'mkdir -p /root/gold-quant-research/results/round55a_results')
ssh(c, 'mkdir -p /root/gold-quant-research/results/round55b_results')
ssh(c, 'mkdir -p /root/gold-quant-research/results/round55c_results')

# Kill any existing R55 processes
ssh(c, 'pkill -f round55a 2>/dev/null; pkill -f round55b 2>/dev/null; pkill -f round55c 2>/dev/null; sleep 2')

# Start A (mean reversion)
c.exec_command(
    'cd /root/gold-quant-research && '
    'nohup python3 -u experiments/run_round55a_mean_reversion.py '
    '> results/round55a_results/stdout.txt 2>&1 &'
)
print("  R55A (mean reversion) started")

time.sleep(3)

# Start B (weekly/long)
c.exec_command(
    'cd /root/gold-quant-research && '
    'nohup python3 -u experiments/run_round55b_weekly_long.py '
    '> results/round55b_results/stdout.txt 2>&1 &'
)
print("  R55B (weekly/long) started")

time.sleep(3)

# Start C (ML)
c.exec_command(
    'cd /root/gold-quant-research && '
    'nohup python3 -u experiments/run_round55c_ml_full.py '
    '> results/round55c_results/stdout.txt 2>&1 &'
)
print("  R55C (ML full) started")

time.sleep(8)

# Verify
proc = ssh(c, 'ps aux | grep round55 | grep python | grep -v grep')
print(f"\n=== Running processes ===\n{proc or '!! NONE RUNNING !!'}")

c.close()
print("\nDeploy complete!")
