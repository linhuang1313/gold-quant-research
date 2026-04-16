"""Upload Round 4 and set up full chain: R2 -> R3 -> R4."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko, time

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
PYTHON = "/root/miniconda3/envs/3.10/bin/python"
REMOTE_DIR = "/root/gold-quant-trading"

def run_cmd(ssh, cmd, timeout=60):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace').strip()
    err = stderr.read().decode('utf-8', errors='replace').strip()
    return out, err

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

# 1. Upload Round 4 script
print("Uploading run_round4.py...")
sftp = ssh.open_sftp()
sftp.put("run_round4.py", f"{REMOTE_DIR}/run_round4.py")
sftp.close()
print("  Uploaded.")

# 2. Create output dir
run_cmd(ssh, f"mkdir -p {REMOTE_DIR}/round4_results")

# 3. Kill old chainer and create new one that handles R2 -> R3 -> R4
print("Killing old chainer...")
run_cmd(ssh, "pkill -f 'chain_r2_r3' 2>/dev/null; sleep 1")

chain_script = f"""#!/bin/bash
echo "[$(date)] Full chainer started (R2 -> R3 -> R4)"

# Wait for R2
while pgrep -f "run_round2.py" > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date)] Round 2 finished."

# Check if R3 is already running (old chainer may have started it)
if pgrep -f "run_round3.py" > /dev/null 2>&1; then
    echo "[$(date)] Round 3 already running, waiting..."
    while pgrep -f "run_round3.py" > /dev/null 2>&1; do
        sleep 30
    done
else
    echo "[$(date)] Starting Round 3..."
    sleep 5
    cd {REMOTE_DIR}
    {PYTHON} -u run_round3.py > round3_stdout.txt 2>&1
fi
echo "[$(date)] Round 3 finished."

# Start R4
echo "[$(date)] Starting Round 4..."
sleep 5
cd {REMOTE_DIR}
{PYTHON} -u run_round4.py > round4_stdout.txt 2>&1
echo "[$(date)] Round 4 completed!"
echo "[$(date)] ALL ROUNDS COMPLETE."
"""

print("Creating full chain script...")
out, err = run_cmd(ssh, f"cat > {REMOTE_DIR}/_chain_full.sh << 'CHAINEOF'\n{chain_script}\nCHAINEOF")
run_cmd(ssh, f"chmod +x {REMOTE_DIR}/_chain_full.sh")

# 4. Launch
print("Launching full chainer (R2 -> R3 -> R4)...")
cmd = f"cd {REMOTE_DIR} && nohup bash _chain_full.sh > chain_full_log.txt 2>&1 &"
stdin, stdout, stderr = ssh.exec_command(cmd)
time.sleep(3)

# 5. Verify
out, _ = run_cmd(ssh, "ps aux | grep chain_full | grep -v grep")
if out:
    print(f"Full chainer running:\n  {out}")
else:
    print("WARNING: chainer not found!")

# Status
out, _ = run_cmd(ssh, "ps aux | grep -E 'run_round|chain' | grep -v grep")
print(f"\nAll processes:\n{out}")

ssh.close()
print("\nDone. Execution flow: R2 -> R3 -> R4 (all automatic)")
print("Monitor: python _check_round2.py / _check_round3.py")
print("Round 4 monitor: check chain_full_log.txt or round4_results/")
