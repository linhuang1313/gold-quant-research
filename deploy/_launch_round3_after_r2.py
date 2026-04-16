"""Upload Round 3 script and set up auto-chaining after Round 2 completes."""
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

# 1. Upload Round 3 script
print("Uploading run_round3.py...")
sftp = ssh.open_sftp()
sftp.put("run_round3.py", f"{REMOTE_DIR}/run_round3.py")
sftp.close()
print("  Uploaded.")

# 2. Create output dir
run_cmd(ssh, f"mkdir -p {REMOTE_DIR}/round3_results")

# 3. Create a chaining script on the server that waits for R2 then starts R3
chain_script = f"""#!/bin/bash
# Auto-chain: wait for Round 2 to finish, then start Round 3
echo "[$(date)] Chainer started, waiting for run_round2.py to finish..."

while true; do
    if ! pgrep -f "run_round2.py" > /dev/null 2>&1; then
        echo "[$(date)] Round 2 finished! Starting Round 3..."
        sleep 5
        cd {REMOTE_DIR}
        {PYTHON} -u run_round3.py > round3_stdout.txt 2>&1
        echo "[$(date)] Round 3 completed!"
        exit 0
    fi
    sleep 30
done
"""

print("Creating chain script on server...")
out, err = run_cmd(ssh, f"cat > {REMOTE_DIR}/_chain_r2_r3.sh << 'CHAINEOF'\n{chain_script}\nCHAINEOF")
run_cmd(ssh, f"chmod +x {REMOTE_DIR}/_chain_r2_r3.sh")

# 4. Launch the chainer in background
print("Launching chainer (waits for R2, then auto-starts R3)...")
cmd = f"cd {REMOTE_DIR} && nohup bash _chain_r2_r3.sh > chain_log.txt 2>&1 &"
stdin, stdout, stderr = ssh.exec_command(cmd)
time.sleep(3)

# 5. Verify
out, _ = run_cmd(ssh, "ps aux | grep chain_r2_r3 | grep -v grep")
if out:
    print(f"Chainer running:\n  {out}")
else:
    print("WARNING: chainer not found!")

# Check R2 status
out, _ = run_cmd(ssh, "ps aux | grep run_round2 | grep -v grep")
if out:
    print(f"\nRound 2 still running (chainer will wait):\n  {out}")
else:
    print("\nRound 2 already finished — Round 3 should start immediately!")

ssh.close()
print("\nDone. Round 3 will auto-start after Round 2 completes.")
print("Monitor R2: python _check_round2.py")
print("Monitor R3: python _check_round3.py (after R3 starts)")
