"""Upload and launch Round 5 experiments on remote server."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko, time

HOST = "connect.westb.seetacloud.com"
PORT = 25821
USER = "root"
PASS = "r1zlTZQUb+E4"
PYTHON = "/root/miniconda3/envs/3.10/bin/python"
REMOTE_DIR = "/root/gold-quant-trading"

def run_cmd(ssh, cmd, timeout=120):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace').strip()
    err = stderr.read().decode('utf-8', errors='replace').strip()
    return out, err

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print("Connecting to AutoDL...")
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
print("Connected!")

# 1. Git pull (with proxy for network access)
print("\n=== Git pull ===")
try:
    out, err = run_cmd(ssh, f"cd {REMOTE_DIR} && source /etc/network_turbo 2>/dev/null; git pull --rebase 2>&1 || echo 'git pull skipped'", timeout=30)
    print(out[:500] if out else "(no output)")
except Exception as e:
    print(f"  Git pull failed (non-critical): {e}")

# 2. Check data files exist
print("\n=== Check data files ===")
out, _ = run_cmd(ssh, f"ls -lh {REMOTE_DIR}/data/download/xauusd-m15-bid*.csv {REMOTE_DIR}/data/download/xauusd-h1-bid*.csv 2>/dev/null")
print(out)

# 3. Upload run_round5.py
print("\n=== Upload run_round5.py ===")
sftp = ssh.open_sftp()
local_path = "scripts/experiments/run_round5.py"
remote_path = f"{REMOTE_DIR}/run_round5.py"
sftp.put(local_path, remote_path)
print(f"  Uploaded {local_path} -> {remote_path}")

# 4. Create output dir
run_cmd(ssh, f"mkdir -p {REMOTE_DIR}/round5_results")

# 5. Check CPU cores
print("\n=== Server info ===")
out, _ = run_cmd(ssh, "nproc && free -h | head -2 && df -h / | tail -1")
print(out)

# 6. Kill any previous round5 processes
print("\n=== Check for existing processes ===")
out, _ = run_cmd(ssh, f"ps aux | grep run_round5 | grep -v grep")
if out:
    print(f"  Found existing: {out}")
    run_cmd(ssh, f"pkill -f run_round5")
    print("  Killed existing processes")
    time.sleep(2)
else:
    print("  No existing round5 processes")

# 7. Launch
print("\n=== Launching Round 5 ===")
cmd = (f"cd {REMOTE_DIR} && nohup {PYTHON} -u run_round5.py "
       f"> round5_stdout.txt 2>&1 &")
run_cmd(ssh, cmd)
time.sleep(3)

# 8. Verify running
out, _ = run_cmd(ssh, f"ps aux | grep run_round5 | grep -v grep")
if out:
    pid = out.split()[1]
    print(f"  Running! PID={pid}")
else:
    print("  WARNING: Process not found!")
    out, _ = run_cmd(ssh, f"tail -20 {REMOTE_DIR}/round5_stdout.txt")
    print(f"  stdout: {out}")

# 9. Show initial output
time.sleep(5)
out, _ = run_cmd(ssh, f"tail -30 {REMOTE_DIR}/round5_stdout.txt")
print(f"\n=== Initial output ===\n{out}")

sftp.close()
ssh.close()
print("\n=== Done! Monitor with _check_round5.py ===")
