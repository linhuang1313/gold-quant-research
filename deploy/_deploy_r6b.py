#!/usr/bin/env python3
"""Deploy R6B on Server B: git pull + start R6B without killing existing R7."""
import paramiko
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SERVER_B = {"host": "connect.westb.seetacloud.com", "port": 25821, "user": "root", "password": "r1zlTZQUb+E4"}

def run_cmd(client, cmd, timeout=300):
    print(f"\n>>> {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    if out.strip():
        for line in out.strip().split('\n'):
            print(f"  {line}")
    if err.strip():
        for line in err.strip().split('\n'):
            print(f"  ERR: {line}")
    return out

print("=" * 60)
print("Deploying R6B on Server B (48 cores)")
print("=" * 60)

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(SERVER_B["host"], port=SERVER_B["port"], username=SERVER_B["user"], password=SERVER_B["password"], timeout=30)
print("Connected!")

# Step 1: Git pull with proxy
run_cmd(client, "cd /root/gold-quant-trading && export https_proxy=http://127.0.0.1:7890 && export http_proxy=http://127.0.0.1:7890 && git stash && git pull origin main 2>&1")

# Step 2: Verify the code version
run_cmd(client, "cd /root/gold-quant-trading && git log --oneline -3")

# Step 3: Check if R6B script exists and has the fix
run_cmd(client, "grep -n 'col_header' /root/gold-quant-trading/scripts/experiments/run_round6b.py | head -5")

# Step 4: Ensure round6_results directory exists
run_cmd(client, "mkdir -p /root/gold-quant-trading/round6_results")

# Step 5: Check existing python processes (don't kill R7)
run_cmd(client, "ps aux | grep 'python.*round' | grep -v grep")

# Step 6: Start R6B in background with 16 workers (leave remaining cores for R7)
run_cmd(client, "cd /root/gold-quant-trading && nohup python -u scripts/experiments/run_round6b.py > round6_results/round6b_stdout.txt 2>&1 &")

# Step 7: Verify it started
import time
time.sleep(3)
run_cmd(client, "ps aux | grep 'python.*round' | grep -v grep")

# Step 8: Check first few lines of output
time.sleep(5)
run_cmd(client, "head -30 /root/gold-quant-trading/round6_results/round6b_stdout.txt 2>/dev/null || echo 'NO OUTPUT YET'")

client.close()
print("\nDone! R6B should be running on Server B.")
